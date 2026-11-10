"""Native MLX MedSAM2 sessions with ``SAM2VideoPredictorNPZ`` prompt semantics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

if TYPE_CHECKING:
    from mlx.core import ThreadLocalStream, array
import numpy as np
from medmlx_core.errors import AssetNotReadyError

from medsam2_mlx._runtime import import_mlx
from medsam2_mlx._storage import native_memory
from medsam2_mlx._types import (
    ArrayInput,
    FloatArray,
    FrameOutput,
    HostArray,
    ImageFeatures,
    MemoryTensor,
    OutputBanks,
    PackedBFloat16,
    PointInputs,
    SessionState,
    Tensor,
)
from medsam2_mlx.mlx_weights import load_model
from medsam2_mlx.request import MedSAM2RuntimeConfig
from medsam2_mlx.sam2_base import MedSAM2Model
from medsam2_mlx.video_predictor import SAM2VideoPredictor, fill_small_holes

__all__ = ["MedSAM2MLXPredictor", "PreparedVideoFrames", "build_predictor", "fill_small_holes"]

# Upstream's fill_hole_area for the NPZ predictor (pixels of the low-resolution mask).
FILL_HOLE_AREA = 8


class PreparedVideoFrames(Protocol):
    """Lazy, host-resident source of normalized frames; execution requires Apple Silicon Metal.

    ``frames[i]`` is a finite FP32 array of shape (3, 512, 512): the i-th slice resized
    to the model input and ImageNet-normalized. Frames are read on demand, one at a time.
    """

    def __len__(self) -> int:
        """Return the number of available preprocessed frames."""
        ...
    def __getitem__(self, index: int, /) -> HostArray:
        """Return one host-resident normalized frame by index."""
        ...


class MedSAM2MLXPredictor(SAM2VideoPredictor):
    """NPZ predictor session with input validation, host outputs, and optional offloading.

    Returned mask logits are host ``np.ndarray`` (objects, 1, height, width) in FP32 at
    the original slice size given to ``init_state``. Foreground is logit > 0, the
    threshold ``medmlx.mask_result`` applies. Set
    ``inference_state["keep_image_features"] = True`` to cache every visited frame's
    backbone output instead of only the latest one (faster re-visits, more memory).
    On a one-frame session, a point correction after that frame has been tracked can
    produce nonfinite pointer positions; the predictor raises ``ValueError`` when
    its per-frame output check detects nonfinite values.
    """

    _prefetch_stream: ThreadLocalStream | None = None

    def init_state(
        self,
        images: HostArray | PreparedVideoFrames,
        height: int,
        width: int,
        *,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        eager_first_frame: bool = False,
    ) -> SessionState:
        """Start a session over preprocessed frames, encoding the first requested frame.

        Args:
            images: FP32 array of shape (D, 3, 512, 512), finite, one preprocessed
                frame per slice (D = slice axis, channels first), or a
                :class:`PreparedVideoFrames` source. A lazy source needs
                ``offload_video_to_cpu=True``.
            height: Original slice height in pixels. Output masks have this height.
            width: Original slice width in pixels. Output masks have this width, and
                pixel prompts are normalized by it.
            offload_video_to_cpu: Keep frames in host memory and move each to the
                device when it is read.
            offload_state_to_cpu: Keep masks and spatial memories in host memory.
            async_loading_frames: Accepted for upstream signature compatibility; it
                is stored in the state and has no effect.
            eager_first_frame: Validate and encode frame zero during initialization.
                The default defers lazy-frame validation until the frame is accessed.

        Raises:
            ValueError: An array stack is empty, non-finite, not FP32, or the wrong
                shape; a lazy source is empty or not offloaded. Lazy frames are
                checked for shape, dtype and finiteness when accessed.
            MissingDependencyError: Not on macOS arm64, or Metal is unavailable.
        """
        mx = import_mlx()
        size = self.image_size
        if isinstance(images, np.ndarray):
            images = cast(HostArray, images)
            if images.ndim != 4 or images.shape[1:] != (3, size, size):
                raise ValueError(f"MedSAM2 MLX images must have shape (D, 3, {size}, {size})")
            if not len(images) or images.dtype != np.float32 or not np.isfinite(images).all():
                raise ValueError("MedSAM2 MLX requires a nonempty finite FP32 image stack")
        elif not len(images) or not offload_video_to_cpu:
            raise ValueError("Lazy MedSAM2 frames require a nonempty host-resident frame source")
        stored = images if offload_video_to_cpu else mx.array(cast(HostArray, images))
        state = super().initialize_session(
            stored,
            height,
            width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            eager_first_frame=eager_first_frame,
        )
        state["keep_image_features"] = False
        return state

    def _frame(self, inference_state: SessionState, frame_idx: int) -> array:
        """Return frame ``frame_idx`` as a (1, 3, S, S) array; host frames must be finite FP32."""
        mx = import_mlx()
        frame = inference_state["images"][frame_idx]
        if tuple(frame.shape) != (3, self.image_size, self.image_size):
            raise ValueError(
                f"MedSAM2 frames must have shape (3, {self.image_size}, {self.image_size})"
            )
        if isinstance(frame, np.ndarray) and (
            cast(HostArray, frame).dtype != np.float32
            or not np.isfinite(cast(HostArray, frame)).all()
        ):
            raise ValueError("MedSAM2 frames must be finite FP32 arrays")
        return mx.array(cast(Tensor, frame))[None]

    def _get_image_feature(
        self, inference_state: SessionState, frame_idx: int, batch_size: int
    ) -> ImageFeatures:
        """Cache requested frame features and broadcast them for the object batch.

        Without ``keep_image_features`` the cache holds this frame and the frame
        prefetched for the next propagation step, if any.
        """
        cache = inference_state["cached_features"]
        if frame_idx not in cache:
            mx = import_mlx()
            backbone = self._compute_image_feature(inference_state, frame_idx)
            mx.eval(backbone["backbone_fpn"], backbone["vision_pos_enc"])
            cache[frame_idx] = backbone
        if not inference_state.get("keep_image_features"):
            keep = {frame_idx, inference_state.get("prefetched_frame_idx")}
            inference_state["cached_features"] = {
                index: value for index, value in cache.items() if index in keep
            }
        return super()._get_image_feature(inference_state, frame_idx, batch_size)

    def _prefetch_image_feature(self, inference_state: SessionState, frame_idx: int) -> None:
        """Start encoding the next propagation frame on a separate GPU stream.

        Image features depend only on the frame, so the encoder for frame t+1 can run
        while frame t is tracked and its masks are post-processed on the host. MLX
        orders the default stream's later reads of these features after this stream's
        work. The frame is read, and a host frame validated, one step early.
        """
        if frame_idx in inference_state["cached_features"]:
            return
        mx = import_mlx()
        if self._prefetch_stream is None:
            # Streams are bound to the creating thread; a thread-local stream lets a
            # predictor be driven from different threads.
            self._prefetch_stream = mx.new_thread_local_stream(mx.gpu)
        with mx.stream(self._prefetch_stream):
            backbone = self._compute_image_feature(inference_state, frame_idx)
            mx.async_eval(backbone["backbone_fpn"], backbone["vision_pos_enc"])
        inference_state["cached_features"][frame_idx] = backbone
        inference_state["prefetched_frame_idx"] = frame_idx

    def _overlap_next_frame(
        self, inference_state: SessionState, pending: list[array], frame_idx: int
    ) -> None:
        """Submit this frame's tracking, then prepare and encode the next frame.

        Submitting first lets the GPU track this frame while the host reads and
        normalizes the next one and builds its encoder graph.
        """
        mx = import_mlx()
        mx.async_eval(*pending)
        self._prefetch_image_feature(inference_state, frame_idx)

    def _store_tensor(self, inference_state: SessionState, value: Tensor) -> Tensor:
        """Offload state tensors to host memory when the session requests it."""
        if inference_state["offload_state_to_cpu"] and not isinstance(value, np.ndarray):
            mx = import_mlx()
            # BF16 memory is exactly representable in FP32 host arrays.
            return cast(HostArray, np.asarray(value.astype(mx.float32)))
        return value

    def _store_memory(self, inference_state: SessionState, value: array) -> MemoryTensor:
        """Offload spatial memory as tagged, owned BF16 bits without an FP32 expansion."""
        if inference_state["offload_state_to_cpu"]:
            mx = import_mlx()
            return PackedBFloat16(np.array(value.view(mx.uint16), dtype=np.uint16))
        return value

    def _run_single_frame_inference(
        self,
        inference_state: SessionState,
        output_dict: OutputBanks,
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: PointInputs | None,
        mask_inputs: array | None,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: array | None = None,
        prefetch_frame_idx: int | None = None,
    ) -> tuple[FrameOutput, array]:
        """Evaluate a tracked frame and reject nonfinite masks, pointers or memory."""
        mx = import_mlx()
        compact, pred_masks = super()._run_single_frame_inference(
            inference_state,
            output_dict,
            frame_idx,
            batch_size,
            is_init_cond_frame,
            point_inputs,
            mask_inputs,
            reverse,
            run_mem_encoder,
            prev_sam_mask_logits,
            prefetch_frame_idx,
        )
        arrays = [pred_masks, compact["obj_ptr"], compact["object_score_logits"]]
        if compact["maskmem_features"] is not None:
            arrays.append(native_memory(compact["maskmem_features"]))
        # MLX is lazy: evaluate here so each frame's work finishes before the next one
        # and non-finite output is caught at the frame that produced it. One combined
        # flag keeps this to a single host synchronization per frame.
        finite = mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in arrays]))
        mx.eval(*arrays, finite)
        if not finite.item():
            raise ValueError("MedSAM2 MLX produced nonfinite output")
        return compact, pred_masks

    @staticmethod
    def _host(masks: Tensor) -> FloatArray:
        """Evaluate mask logits and copy them to the host as FP32."""
        mx = import_mlx()
        masks = mx.array(masks).astype(mx.float32)
        mx.eval(masks)
        return cast(FloatArray, np.asarray(masks))

    def add_new_points_or_box(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        points: ArrayInput | None = None,
        labels: ArrayInput | None = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: ArrayInput | None = None,
    ) -> tuple[int, list[int], FloatArray]:
        """Add point or box prompts and return original-size FP32 masks on the host."""
        index, ids, masks = super().add_new_points_or_box(
            inference_state,
            frame_idx,
            obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            normalize_coords=normalize_coords,
            box=box,
        )
        return index, ids, self._host(masks)

    def add_new_mask(
        self, inference_state: SessionState, frame_idx: int, obj_id: int, mask: ArrayInput
    ) -> tuple[int, list[int], FloatArray]:
        """Add a mask prompt and return original-size FP32 masks on the host."""
        index, ids, masks = super().add_new_mask(inference_state, frame_idx, obj_id, mask)
        return index, ids, self._host(masks)

    def propagate_in_video(
        self,
        inference_state: SessionState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], FloatArray]]:
        """Yield tracked frames and original-size FP32 host masks in order."""
        for index, ids, masks in super().propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            yield index, ids, self._host(masks)

    @overload
    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[True] = True,
    ) -> tuple[int, list[int], FloatArray]:
        """Clear one object frame prompt and optionally return updated host masks."""
        ...

    @overload
    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[False],
    ) -> None:
        """Clear one object frame prompt and optionally return updated host masks."""
        ...

    @overload
    def clear_all_prompts_in_frame(
        self, inference_state: SessionState, frame_idx: int, obj_id: int, need_output: bool
    ) -> tuple[int, list[int], FloatArray] | None:
        """Clear one object frame prompt and optionally return updated host masks."""
        ...

    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: bool = True,
    ) -> tuple[int, list[int], FloatArray] | None:
        """Clear one object frame prompt and optionally return updated host masks."""
        result = super().clear_all_prompts_in_frame(
            inference_state, frame_idx, obj_id, need_output=need_output
        )
        if result is None:
            return None
        index, ids, masks = result
        return index, ids, self._host(masks)

    def remove_object(
        self,
        inference_state: SessionState,
        obj_id: int,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[int], list[tuple[int, FloatArray]]]:
        """Remove an object and return remaining IDs and requested updated host masks."""
        ids, frames = super().remove_object(
            inference_state, obj_id, strict=strict, need_output=need_output
        )
        return ids, [(index, self._host(masks)) for index, masks in frames]


def build_predictor(
    runtime: MedSAM2RuntimeConfig,
    *,
    weights: str | Path | Mapping[str, ArrayInput] | None = None,
    model: MedSAM2Model | None = None,
    retain_history: bool = True,
) -> MedSAM2MLXPredictor:
    """Build a predictor from converted weights or an already constructed MLX model.

    A given ``model`` is used as is. Otherwise ``weights`` (a ``.safetensors``
    path or state dict) takes precedence over ``runtime.checkpoint``. With
    ``runtime.apply_postprocessing`` the predictor fills holes of up to
    ``FILL_HOLE_AREA`` (8) pixels in the 128 x 128 mask logits.
    ``retain_history=False`` bounds automatic history for one-way batch propagation;
    reset before changing direction or revisiting an earlier frame.

    Raises:
        AssetNotReadyError: No weights, checkpoint or model was supplied.
        InvalidInputError: The weights do not match the graph (from ``load_model``).
    """
    if model is None:
        checkpoint = runtime.checkpoint if weights is None else weights
        if checkpoint is None:
            raise AssetNotReadyError(
                "MedSAM2 requires explicit converted weights or a constructed model",
                reason="missing weights",
                hint="Pass weights=... or set MedSAM2RuntimeConfig(checkpoint=...).",
            )
        model = load_model(
            checkpoint,
            apply_postprocessing=runtime.apply_postprocessing,
        )
    return MedSAM2MLXPredictor(
        model,
        fill_hole_area=FILL_HOLE_AREA if runtime.apply_postprocessing else 0,
        retain_history=retain_history,
    )
