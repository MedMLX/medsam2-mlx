# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/sam2_video_predictor_npz.py, sam2/utils/misc.py.
"""Interactive video (volume) session of ``SAM2VideoPredictorNPZ``, in MLX.

The predictor takes preprocessed frames, as the NPZ predictor does. Options that
the NPZ builder leaves at their defaults are hard-wired: no non-overlap constraints,
no clearing of memory around corrections, and corrections are not promoted to
conditioning frames. Postprocessing (hole filling, dynamic multimask fallback, and
binarized click masks in memory) is set by ``fill_hole_area`` and the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from medsam2_mlx._types import FloatArray, HostArray

if TYPE_CHECKING:
    from mlx.core import array

    from medsam2_mlx.mlx_predictor import PreparedVideoFrames

import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from importlib import import_module
from typing import Literal, cast, overload

import numpy as np

from medsam2_mlx._ops import interpolate_bilinear
from medsam2_mlx._runtime import mx
from medsam2_mlx._storage import native_memory
from medsam2_mlx._types import (
    ArrayInput,
    ConnectedComponents,
    ConsolidatedOutput,
    FrameOutput,
    FrameStorage,
    ImageFeatures,
    MemoryTensor,
    OutputBanks,
    PointInputs,
    SessionState,
    Tensor,
    VideoResolutionOutput,
)
from medsam2_mlx.sam2_base import NO_OBJ_SCORE, MedSAM2Model


def fill_small_holes(scores: FloatArray, max_area: int = 8) -> FloatArray:
    """Fill 8-connected background components of at most ``max_area`` pixels with 0.1.

    Host equivalent of upstream ``fill_holes_in_mask_scores`` (whose connected
    components kernel is unavailable on macOS) for (..., H, W) score arrays. Background
    is score <= 0; filled pixels become a small positive logit, i.e. foreground.
    Small components touching the image boundary are filled too.
    Returns a copy. Applied by the predictor to the 128 x 128 low-resolution logits.
    """
    ndimage = cast(ConnectedComponents, import_module("scipy.ndimage"))
    result = scores.copy()
    for index in np.ndindex(scores.shape[:-2]):
        labels = ndimage.label(scores[index] <= 0, structure=np.ones((3, 3), dtype=np.uint8))[0]
        areas = np.bincount(labels.ravel())
        small = (labels > 0) & (areas[labels] <= max_area)
        result[index][small] = np.float32(0.1)
    return result


def concat_points(old: PointInputs | None, points: array, labels: array) -> PointInputs:
    """Append new prompt points (B, N, 2) and labels (B, N) after the earlier ones, if any."""
    if old is None:
        return {"point_coords": points, "point_labels": labels}
    return {
        "point_coords": mx.concatenate([mx.array(old["point_coords"]), points], axis=1),
        "point_labels": mx.concatenate([mx.array(old["point_labels"]), labels], axis=1),
    }


class SAM2VideoPredictor:
    """Session logic of ``SAM2VideoPredictorNPZ`` around an MLX ``MedSAM2Model``.

    A session ("inference state") holds one stack of preprocessed frames and any
    number of tracked objects, each with its own prompts and per-frame outputs.
    A prompt on a frame not yet tracked produces a "conditioning" output; a correction
    on an already tracked frame initially enters the "non-conditioning" bank. During
    preflight an existing conditioning output takes precedence for that frame. Outputs
    live in banks keyed by frame index
    (``cond_frame_outputs`` and ``non_cond_frame_outputs``): mask logits at 128 x 128,
    an object pointer, an object score, and the BF16 spatial memory. Masks returned
    to the caller are bilinearly resized to the original slice size and are logits
    (foreground > 0); an object the model scores as absent gets the constant -1024.
    """

    def __init__(
        self, model: MedSAM2Model, *, fill_hole_area: int = 0, retain_history: bool = True
    ) -> None:
        """Attach a model, retaining history for revisits unless monotonic tracking is selected."""
        self.model = model
        self.image_size = model.image_size
        self.fill_hole_area = fill_hole_area
        self.retain_history = retain_history

    # --- storage hooks ---------------------------------------------------------------

    def _store_tensor(self, inference_state: SessionState, value: Tensor) -> Tensor:
        """Move a state tensor to its storage location (identity on device)."""
        return value

    def _store_memory(self, inference_state: SessionState, value: array) -> MemoryTensor:
        """Store the BF16 spatial memory, retaining native storage in the base predictor."""
        return value

    def _frame(self, inference_state: SessionState, frame_idx: int) -> array:
        """Load a frame as a batched FP32 MLX array."""
        return mx.array(inference_state["images"][frame_idx]).astype(mx.float32)[None]

    # --- session --------------------------------------------------------------------

    def initialize_session(
        self,
        images: array | HostArray | PreparedVideoFrames,
        video_height: int,
        video_width: int,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        eager_first_frame: bool = False,
    ) -> SessionState:
        """Create a session, encoding frames on first access by default.

        ``images`` are preprocessed frames (D, 3, 512, 512), ``video_height`` and
        ``video_width`` the original slice size in pixels. ``async_loading_frames`` is
        stored but unused. ``eager_first_frame`` primes frame zero for callers that
        require its validation and encoding during initialization.
        """
        inference_state: SessionState = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "async_loading_frames": async_loading_frames,
            "video_height": video_height,
            "video_width": video_width,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }
        if eager_first_frame:
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _obj_id_to_idx(self, inference_state: SessionState, obj_id: int) -> int:
        """Resolve or register an object index before tracking begins."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is not None:
            return int(obj_idx)
        if inference_state["tracking_has_started"]:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return obj_idx

    def _get_obj_num(self, inference_state: SessionState) -> int:
        """Return the number of registered session objects."""
        return len(inference_state["obj_idx_to_id"])

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
    ) -> tuple[int, list[int], Tensor]:
        """Prompt one object on one frame with a box and/or clicks and return its masks.

        Args:
            inference_state: Session from ``initialize_session``.
            frame_idx: Zero-based frame (slice) index.
            obj_id: Object label. New ids can only be added before propagation starts.
            points: Clicks as (x, y), shape (N, 2) or (1, N, 2), paired with ``labels``.
            labels: (N,) or (1, N) ints: 1 foreground, 0 background.
            clear_old_points: Drop this frame's earlier points first (required with
                ``box``).
            normalize_coords: ``True``: ``points`` and ``box`` are pixels of the original
                slice (``video_width`` x ``video_height``) and are divided by that size.
                ``False``: they are already fractions in [0, 1]. Both are then scaled
                to the 512-pixel model input.
            box: ``[x_min, y_min, x_max, y_max]`` in the same coordinates as ``points``.

        Returns:
            ``(frame_idx, obj_ids, masks)``: all object ids and their mask logits for this
            frame, shape (num_objects, 1, video_height, video_width). The frame is
            treated as a conditioning frame unless it was already tracked.

        Raises:
            ValueError: Points without labels or the reverse, neither points nor box, or a
                box with ``clear_old_points=False``.
            RuntimeError: ``obj_id`` is new but tracking has already started.

        A box added after tracking has started emits a ``UserWarning`` because it may not
        refine the existing track well.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")
        points_np = (
            np.zeros((0, 2), dtype=np.float32)
            if points is None
            else np.asarray(points, dtype=np.float32)
        )
        labels_np = (
            np.zeros((0,), dtype=np.int32) if labels is None else np.asarray(labels, dtype=np.int32)
        )
        if points_np.ndim == 2:
            points_np = points_np[None]
        if labels_np.ndim == 1:
            labels_np = labels_np[None]
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "You are adding a box after tracking starts. SAM 2 may not always be "
                    "able to incorporate a box prompt for *refinement*. If you intend to "
                    "use box prompt as an *initial* input before tracking, please call "
                    "'reset_state' on the inference state to restart from scratch.",
                    category=UserWarning,
                    stacklevel=2,
                )
            # A box is two labelled corner points: 2 = top-left, 3 = bottom-right.
            box_coords = np.asarray(box, dtype=np.float32).reshape(1, 2, 2)
            box_labels = np.array([[2, 3]], dtype=np.int32)
            points_np = np.concatenate([box_coords, points_np], axis=1)
            labels_np = np.concatenate([box_labels, labels_np], axis=1)
        points_mx = mx.array(points_np)
        if normalize_coords:
            size = mx.array(
                [inference_state["video_width"], inference_state["video_height"]],
                dtype=mx.float32,
            )
            points_mx = points_mx / size
        # Fractions of the slice become pixels of the 512 x 512 model input.
        points_mx = points_mx * self.image_size
        labels_mx = mx.array(labels_np)
        old = point_inputs_per_frame.get(frame_idx) if not clear_old_points else None
        point_inputs = concat_points(old, points_mx, labels_mx)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and cast(Tensor | None, prev_out["pred_masks"]) is not None:
            # Refining a frame feeds its previous logits back as a mask prompt, clipped
            # to [-32, 32] as upstream does.
            prev_sam_mask_logits = mx.clip(mx.array(prev_out["pred_masks"]), -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state,
            obj_output_dict,
            frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        consolidated = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], video_res_masks

    def add_new_mask(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        mask: ArrayInput,
    ) -> tuple[int, list[int], Tensor]:
        """Prompt one object on one frame with a mask and return the frame's masks.

        ``mask`` is a 2D array on the frame's grid; nonzero is foreground. If it is not
        512 x 512 it is resized bilinearly with antialiasing and thresholded at 0.5. The
        mask replaces any point prompt on that frame and becomes the frame's mask logits
        directly (about +10 inside and -10 outside); the decoder runs only to produce the
        object pointer. Returns ``(frame_idx, obj_ids, masks)`` as
        ``add_new_points_or_box`` does.

        Raises:
            ValueError: ``mask`` is not two dimensional.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        mask_np = np.asarray(mask).astype(bool)
        if mask_np.ndim != 2:
            raise ValueError("MedSAM2 mask prompt must be two dimensional")
        mask_inputs = mx.array(mask_np.astype(np.float32))[None, None]
        if mask_np.shape != (self.image_size, self.image_size):
            mask_inputs = interpolate_bilinear(
                mask_inputs, (self.image_size, self.image_size), antialias=True
            )
            mask_inputs = (mask_inputs >= 0.5).astype(mx.float32)
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        current_out, _ = self._run_single_frame_inference(
            inference_state,
            obj_output_dict,
            frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        consolidated = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], video_res_masks

    def _get_orig_video_res_output(
        self, inference_state: SessionState, any_res_masks: Tensor
    ) -> array:
        """Resize mask logits to the original video frame dimensions."""
        size = (inference_state["video_height"], inference_state["video_width"])
        return interpolate_bilinear(mx.array(any_res_masks), size)

    @overload
    def _consolidate_temp_output_across_obj(
        self,
        inference_state: SessionState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: Literal[True],
    ) -> VideoResolutionOutput:
        """Combine per-object frame outputs at the requested mask resolution."""
        ...

    @overload
    def _consolidate_temp_output_across_obj(
        self,
        inference_state: SessionState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: Literal[False] = False,
    ) -> FrameOutput:
        """Combine per-object frame outputs at the requested mask resolution."""
        ...

    def _consolidate_temp_output_across_obj(
        self,
        inference_state: SessionState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: bool = False,
    ) -> FrameOutput | VideoResolutionOutput:
        """Stack every object's output for one frame into a single batch.

        Uses the object's pending (temporary) output for ``frame_idx`` if it has one, else
        its stored output. Masks are resized to the video size when
        ``consolidate_at_video_res`` (for returning to the caller) and otherwise kept at
        128 x 128 (for storage and memory encoding). With ``run_mem_encoder`` the stacked
        low-resolution masks are also encoded into spatial memory.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            if run_mem_encoder:
                raise ValueError("memory encoder cannot run at video resolution")
            size = (inference_state["video_height"], inference_state["video_width"])
            mask_key = "pred_masks_video_res"
        else:
            size = (self.image_size // 4, self.image_size // 4)
            mask_key = "pred_masks"
        hidden = self.model.hidden_dim
        masks: list[array] = []
        pointers: list[array] = []
        scores: list[array] = []
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            out = inference_state["temp_output_dict_per_obj"][obj_idx][storage_key].get(frame_idx)
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            if out is None:
                # No output for this object on this frame: an "absent" mask and a placeholder
                # pointer, but an object score of +10 (treated as present), matching
                # upstream's fill.
                masks.append(mx.full((1, 1, *size), NO_OBJ_SCORE, dtype=mx.float32))
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(inference_state, frame_idx)
                    pointers.append(empty_mask_ptr)
                else:
                    pointers.append(mx.full((1, hidden), NO_OBJ_SCORE, dtype=mx.float32))
                scores.append(mx.full((1, 1), 10.0, dtype=mx.float32))
                continue
            masks.append(interpolate_bilinear(mx.array(out["pred_masks"]), size))
            pointers.append(mx.array(out["obj_ptr"]))
            scores.append(mx.array(out["object_score_logits"]))
        consolidated = cast(
            ConsolidatedOutput,
            {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                mask_key: self._store_tensor(inference_state, mx.concatenate(masks, axis=0)),
                "obj_ptr": mx.concatenate(pointers, axis=0),
                "object_score_logits": mx.concatenate(scores, axis=0),
            },
        )
        if run_mem_encoder:
            high_res_masks = interpolate_bilinear(
                mx.array(cast(FrameOutput, consolidated)["pred_masks"]),
                (self.image_size, self.image_size),
            )
            features, position = self._run_memory_encoder(
                inference_state,
                frame_idx,
                batch_size,
                high_res_masks,
                consolidated["object_score_logits"],
                is_mask_from_pts=True,
            )
            consolidated["maskmem_features"] = features
            consolidated["maskmem_pos_enc"] = position
        return cast(FrameOutput | VideoResolutionOutput, consolidated)

    def _get_empty_mask_ptr(self, inference_state: SessionState, frame_idx: int) -> array:
        """Compute the object pointer produced by an empty frame mask."""
        mask_inputs = mx.zeros((1, 1, self.image_size, self.image_size), dtype=mx.float32)
        backbone = self._get_image_feature(inference_state, frame_idx, 1)
        current = self.model.track_step(
            frame_idx,
            True,
            backbone,
            None,
            mask_inputs,
            {},
            inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
        )
        return current["obj_ptr"]

    def propagate_in_video_preflight(self, inference_state: SessionState) -> None:
        """Commit pending prompt outputs before tracking: encode their memories and bank them.

        Sets tracking as started before validating the banks, so new object ids are
        rejected even if validation raises. ``reset_state`` clears that state.
        Raises ``RuntimeError`` if the banks and the prompted frames disagree.
        """
        inference_state["tracking_has_started"] = True
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in (False, True):
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds: set[int] = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                output_dict[storage_key][frame_idx] = consolidated
                self._add_output_per_object(inference_state, frame_idx, consolidated, storage_key)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            if frame_idx not in output_dict["cond_frame_outputs"]:
                raise RuntimeError("Consolidated conditioning frame has no output")
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
        all_consolidated = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames: set[int] = set()
        for per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames.update(per_frame.keys())
        for per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames.update(per_frame.keys())
        if all_consolidated != input_frames:
            raise RuntimeError("Consolidated frames differ from the frames with prompts")

    def propagate_in_video(
        self,
        inference_state: SessionState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Tensor]]:
        """Track all objects from the prompted frames, yielding masks frame by frame.

        Args:
            inference_state: Session with at least one prompt.
            start_frame_idx: First frame to yield; defaults to the earliest conditioning frame.
            max_frame_num_to_track: How many frames beyond the start to visit (all of the
                video when ``None``).
            reverse: Track toward lower frame indices instead of higher ones.

        Yields:
            ``(frame_idx, obj_ids, masks)`` per frame, masks as logits of shape
            (num_objects, 1, video_height, video_width). Prompted frames are yielded from
            their stored outputs; other frames are predicted from the memory bank (the
            conditioning frames, up to ``num_maskmem - 1`` previous tracked frames in
            tracking order (six by default), and object pointers), then stored as
            non-conditioning frames. Reverse tracking
            from frame 0 yields nothing, as upstream.

        Raises:
            RuntimeError: No conditioning output exists, or prompt and output banks
                disagree. Preflight has already marked tracking as started; call
                ``reset_state`` to add new object ids after this failure.
        """
        self.propagate_in_video_preflight(inference_state)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if not output_dict["cond_frame_outputs"]:
            raise RuntimeError("No points are provided; please add points first")
        if start_frame_idx is None:
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        cursor = inference_state.get("history_cursor")
        if not self.retain_history and cursor is not None:
            previous, direction = cursor
            revisits = start_frame_idx > previous if reverse else start_frame_idx < previous
            if reverse != direction or revisits:
                raise ValueError(
                    "Monotonic tracking cannot revisit history; reset the session first"
                )
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            order = (
                range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else range(0)
            )
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            order = range(start_frame_idx, end_frame_idx + 1)
        for position, frame_idx in enumerate(order):
            # The next frame's image features are requested early when it will need
            # inference; a predictor may start computing them while this frame runs.
            next_frame_idx = order[position + 1] if position + 1 < len(order) else None
            if next_frame_idx is not None and any(
                next_frame_idx in frames for frames in consolidated_frame_inds.values()
            ):
                next_frame_idx = None
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if next_frame_idx is not None:
                    self._prefetch_image_feature(inference_state, next_frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if next_frame_idx is not None:
                    self._prefetch_image_feature(inference_state, next_frame_idx)
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state,
                    output_dict,
                    frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                    prefetch_frame_idx=next_frame_idx,
                )
                output_dict[storage_key][frame_idx] = current_out
            self._add_output_per_object(inference_state, frame_idx, current_out, storage_key)
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            if not self.retain_history:
                self._prune_history(inference_state, frame_idx, reverse)
            yield frame_idx, obj_ids, self._get_orig_video_res_output(inference_state, pred_masks)

    def _prune_history(self, state: SessionState, frame_idx: int, reverse: bool) -> None:
        """Keep every prompt/correction and only the next frame's spatial and pointer lookbacks."""
        spatial_keep = self.model.num_maskmem - 1
        pointer_keep = self.model.max_obj_ptrs_in_encoder - 1
        protected = state["consolidated_frame_inds"]["non_cond_frame_outputs"]
        banks = [state["output_dict"], *state["output_dict_per_obj"].values()]
        for bank in banks:
            outputs = bank["non_cond_frame_outputs"]
            for index in list(outputs):
                age = index - frame_idx if reverse else frame_idx - index
                if index in protected or age < 0:
                    continue
                if age >= max(spatial_keep, pointer_keep):
                    del outputs[index]
                elif age >= spatial_keep:
                    # Pointer-only history does not need the much larger spatial memory.
                    outputs[index]["maskmem_features"] = None
                    outputs[index]["maskmem_pos_enc"] = None
        for index in list(state["frames_already_tracked"]):
            age = index - frame_idx if reverse else frame_idx - index
            if (
                age >= max(spatial_keep, pointer_keep)
                and index not in protected
                and index not in state["output_dict"]["cond_frame_outputs"]
            ):
                del state["frames_already_tracked"][index]
        state["history_cursor"] = (frame_idx, reverse)

    def _add_output_per_object(
        self,
        inference_state: SessionState,
        frame_idx: int,
        current_out: FrameOutput,
        storage_key: FrameStorage,
    ) -> None:
        """Split one batched frame output into per-object storage."""
        maskmem_features = current_out["maskmem_features"]
        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            sl = slice(obj_idx, obj_idx + 1)
            obj_output_dict[storage_key][frame_idx] = {
                "maskmem_features": None if maskmem_features is None else maskmem_features[sl],
                "maskmem_pos_enc": None if maskmem_pos_enc is None else maskmem_pos_enc[sl],
                "pred_masks": current_out["pred_masks"][sl],
                "obj_ptr": current_out["obj_ptr"][sl],
                "object_score_logits": current_out["object_score_logits"][sl],
            }

    @overload
    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[True] = True,
    ) -> tuple[int, list[int], Tensor]:
        """Clear an object frame prompt and optionally return the updated output."""
        ...

    @overload
    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[False],
    ) -> None:
        """Clear an object frame prompt and optionally return the updated output."""
        ...

    @overload
    def clear_all_prompts_in_frame(
        self, inference_state: SessionState, frame_idx: int, obj_id: int, need_output: bool
    ) -> tuple[int, list[int], Tensor] | None:
        """Clear an object frame prompt and optionally return the updated output."""
        ...

    def clear_all_prompts_in_frame(
        self,
        inference_state: SessionState,
        frame_idx: int,
        obj_id: int,
        need_output: bool = True,
    ) -> tuple[int, list[int], Tensor] | None:
        """Remove one object's prompts on a frame, undoing its conditioning if none remain.

        If no object has a prompt left on ``frame_idx``, the frame is demoted from
        conditioning to ordinary tracked output. If no conditioning frame remains,
        tracking results and all objects' prompts are cleared. An unknown ``obj_id``
        creates an object before tracking starts and raises after tracking starts.
        With ``need_output`` returns
        ``(frame_idx, obj_ids, masks)`` as ``add_new_points_or_box`` does, otherwise ``None``.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = any(
            frame_idx in inference_state["point_inputs_per_obj"][index]
            or frame_idx in inference_state["mask_inputs_per_obj"][index]
            for index in range(batch_size)
        )
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for index in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][index]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if not output_dict["cond_frame_outputs"]:
                self._reset_tracking_results(inference_state)
        if not need_output:
            return None
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], video_res_masks

    def reset_state(self, inference_state: SessionState) -> None:
        """Forget all objects, prompts and tracked outputs; the loaded frames are kept."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state: SessionState) -> None:
        """Clear prompts and tracked outputs while retaining registered objects."""
        inference_state.pop("history_cursor", None)
        for value in inference_state["point_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["mask_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        for value in inference_state["temp_output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _compute_image_feature(
        self, inference_state: SessionState, frame_idx: int
    ) -> ImageFeatures:
        """Encode one session frame into backbone features."""
        return self.model.forward_image(self._frame(inference_state, frame_idx))

    def _prefetch_image_feature(self, inference_state: SessionState, frame_idx: int) -> None:
        """Hint that propagation will track ``frame_idx`` next; the base class ignores it."""

    def _overlap_next_frame(
        self, inference_state: SessionState, pending: list[array], frame_idx: int
    ) -> None:
        """Hint that ``pending`` outputs are built and ``frame_idx`` is tracked next.

        Called before the host first reads this frame's outputs; the base class ignores it.
        """

    def _get_image_feature(
        self, inference_state: SessionState, frame_idx: int, batch_size: int
    ) -> ImageFeatures:
        """The frame's backbone output, broadcast to ``batch_size`` objects."""
        backbone = inference_state["cached_features"].get(frame_idx)
        if backbone is None:
            backbone = self._compute_image_feature(inference_state, frame_idx)
            inference_state["cached_features"] = {frame_idx: backbone}
        return {
            "vision_features": mx.broadcast_to(
                backbone["vision_features"], (batch_size, *backbone["vision_features"].shape[1:])
            ),
            "backbone_fpn": [
                mx.broadcast_to(x, (batch_size, *x.shape[1:])) for x in backbone["backbone_fpn"]
            ],
            "vision_pos_enc": [
                mx.broadcast_to(x, (batch_size, *x.shape[1:])) for x in backbone["vision_pos_enc"]
            ],
        }

    def _fill_holes(self, pred_masks: array) -> array:
        """Fill configured small holes in mask logits and return them on MLX."""
        return mx.array(
            fill_small_holes(cast(FloatArray, np.asarray(pred_masks)), self.fill_hole_area)
        )

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
        """Track one frame and store its compact masks, pointers and optional memory.

        ``prefetch_frame_idx`` names the frame propagation tracks next, if it needs
        inference; it is passed to ``_overlap_next_frame`` once this frame's graph is built.
        """
        backbone = self._get_image_feature(inference_state, frame_idx, batch_size)
        if point_inputs is not None and mask_inputs is not None:
            raise ValueError("Points and a mask cannot prompt the same frame together")
        current = self.model.track_step(
            frame_idx,
            is_init_cond_frame,
            backbone,
            point_inputs,
            mask_inputs,
            output_dict,
            inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        maskmem_features = current["maskmem_features"]
        if maskmem_features is not None:
            # Upstream stores spatial memory in BF16, also for FP32 inference.
            maskmem_features = maskmem_features.astype(mx.bfloat16)
        if prefetch_frame_idx is not None:
            pending = [current["pred_masks"], current["obj_ptr"], current["object_score_logits"]]
            if maskmem_features is not None:
                pending.append(maskmem_features)
            self._overlap_next_frame(inference_state, pending, prefetch_frame_idx)
        stored_memory = (
            None
            if maskmem_features is None
            else self._store_memory(inference_state, maskmem_features)
        )
        pred_masks = current["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks = self._fill_holes(pred_masks)
        compact: FrameOutput = {
            "maskmem_features": stored_memory,
            "maskmem_pos_enc": self._get_maskmem_pos_enc(
                inference_state, current["maskmem_pos_enc"]
            ),
            "pred_masks": self._store_tensor(inference_state, pred_masks),
            "obj_ptr": current["obj_ptr"],
            "object_score_logits": current["object_score_logits"],
        }
        return compact, pred_masks

    def _run_memory_encoder(
        self,
        inference_state: SessionState,
        frame_idx: int,
        batch_size: int,
        high_res_masks: array,
        object_score_logits: array,
        is_mask_from_pts: bool,
    ) -> tuple[MemoryTensor, Tensor | None]:
        """Encode frame masks, round memory to BF16 and use the configured storage."""
        backbone = self._get_image_feature(inference_state, frame_idx, batch_size)
        features, position = self.model.encode_new_memory(
            backbone["backbone_fpn"][-1], high_res_masks, object_score_logits, is_mask_from_pts
        )
        stored_features = self._store_memory(inference_state, features.astype(mx.bfloat16))
        return stored_features, self._get_maskmem_pos_enc(inference_state, position)

    def _get_maskmem_pos_enc(
        self, inference_state: SessionState, position: Tensor | None
    ) -> Tensor | None:
        """Cache the frame- and object-independent memory position encoding.

        The encoding depends only on the feature grid, so one copy is computed (from the
        first object) and broadcast to the batch.
        """
        if position is None:
            return None
        constants = inference_state["constants"]
        if "maskmem_pos_enc" not in constants:
            constants["maskmem_pos_enc"] = self._store_tensor(inference_state, position[0:1])
        cached = mx.array(constants["maskmem_pos_enc"])
        return mx.broadcast_to(cached, (position.shape[0], *cached.shape[1:]))

    def remove_object(
        self,
        inference_state: SessionState,
        obj_id: int,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[int], Sequence[tuple[int, Tensor]]]:
        """Remove an object and re-index the rest, as upstream's ``remove_object``.

        Returns the remaining object ids and, with ``need_output``, ``(frame_idx, masks)``
        for each frame the removed object had prompts on. An unknown ``obj_id`` is
        ignored unless ``strict`` (then ``RuntimeError``). Removing the only object
        resets the session and returns no per-frame outputs.
        """
        old_idx = inference_state["obj_id_to_idx"].get(obj_id)
        updated_frames: list[tuple[int, Tensor]] = []
        if old_idx is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames
        input_frames = set(inference_state["point_inputs_per_obj"][old_idx])
        input_frames.update(inference_state["mask_inputs_per_obj"][old_idx])
        for frame_idx in input_frames:
            self.clear_all_prompts_in_frame(inference_state, frame_idx, obj_id, need_output=False)
        old_ids = inference_state["obj_ids"]
        old_inds = list(range(len(old_ids)))
        remain = [index for index in old_inds if index != old_idx]
        new_ids = [old_ids[index] for index in remain]
        old_to_new = {old: new for new, old in enumerate(remain)}
        inference_state["obj_id_to_idx"] = OrderedDict(
            zip(new_ids, range(len(new_ids)), strict=True)
        )
        inference_state["obj_idx_to_id"] = OrderedDict(enumerate(new_ids))
        inference_state["obj_ids"] = new_ids

        def map_keys[T](container: dict[int, T]) -> None:
            """Reindex remaining per-object entries after removing one object."""
            moved = [(old_to_new[k], container.pop(k)) for k in old_inds if k in old_to_new]
            container.pop(old_idx, None)
            container.update(moved)

        map_keys(inference_state["point_inputs_per_obj"])
        map_keys(inference_state["mask_inputs_per_obj"])
        map_keys(inference_state["output_dict_per_obj"])
        map_keys(inference_state["temp_output_dict_per_obj"])
        keep = mx.array(remain)
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for frame_idx, out in inference_state["output_dict"][storage_key].items():
                features = out["maskmem_features"]
                if features is not None:
                    out["maskmem_features"] = self._store_memory(
                        inference_state, native_memory(features)[keep]
                    )
                position = out["maskmem_pos_enc"]
                if position is not None:
                    out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(
                        inference_state, mx.array(position)[keep]
                    )
                out["pred_masks"] = self._store_tensor(
                    inference_state, mx.array(out["pred_masks"])[keep]
                )
                out["obj_ptr"] = out["obj_ptr"][keep]
                out["object_score_logits"] = out["object_score_logits"][keep]
                self._add_output_per_object(inference_state, frame_idx, out, storage_key)
        if need_output:
            temp = inference_state["temp_output_dict_per_obj"]
            for frame_idx in input_frames:
                is_cond = any(frame_idx in d["cond_frame_outputs"] for d in temp.values())
                consolidated = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                updated_frames.append(
                    (
                        frame_idx,
                        self._get_orig_video_res_output(
                            inference_state, consolidated["pred_masks_video_res"]
                        ),
                    )
                )
        return inference_state["obj_ids"], updated_frames
