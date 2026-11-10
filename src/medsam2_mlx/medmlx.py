"""MedMLX declaration and runner for upstream MedSAM2 prompted segmentation.

Frames follow the pinned CT example's bicubic RGB resize and ImageNet normalization.
Prompt and propagation calls follow SAM2VideoPredictorNPZ; each propagation direction
starts from an independently seeded state. NIfTI outputs retain the source grid;
NPZ and frame-folder inputs use a synthetic grid.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MedmlxError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.prompts import medsam2_prompt_payload
from medmlx_core.weights import weights_dir, weights_revision

from medsam2_mlx._types import (
    BoolArray,
    HostArray,
    NiftiConstructor,
    NiftiImage,
    NiftiRuntime,
    NumericArray,
    RunnerPredictor,
    RunnerResult,
    SessionState,
)
from medsam2_mlx.request import MedSAM2Prompt, MedSAM2RuntimeConfig

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

MODEL = {
    "id": "medsam2",
    "label": "MedSAM2",
    "task": "segmentation",
    "run": "medsam2_mlx.medmlx:run",
    # Measured MLX peak for one box tracked through 21 slices of a 512x512x147 CT;
    # larger inputs need more.
    "peak_memory_gb": 1.3,
    "summary": (
        "Box-, point- or mask-prompted segmentation of NIfTI slices or volumes, "
        "RECIST NPZ stacks or JPEG frame folders, with optional forward and/or "
        "reverse propagation."
    ),
}

MASK_NAME = "medsam2_mask.nii.gz"
OVERLAY_NAME = "medsam2_overlay.png"
PROPAGATION_MODES = ("none", "forward", "reverse", "both")
_WEIGHTS_HINT = (
    "Omit weights_path to download MedMLX/medsam2-ct-lesion-mlx, or pass a MedSAM2 "
    "safetensors checkpoint from https://huggingface.co/MedMLX/medsam2-mlx."
)


# --- NIfTI input ---------------------------------------------------------------------


def _nibabel() -> NiftiRuntime:
    """Import nibabel lazily so it is only required when NIfTI input is read."""

    try:
        import nibabel
    except ImportError as error:
        raise MissingDependencyError(
            "Reading NIfTI volumes requires nibabel",
            hint="Install nibabel (a medsam2-mlx dependency) in this environment.",
        ) from error
    return cast(NiftiRuntime, nibabel)


def _is_nifti(path: Path) -> bool:
    """Recognize .nii or .nii.gz input path suffixes."""
    return path.name.endswith(".nii.gz") or path.suffix.lower() == ".nii"


def read_nifti(path: Path) -> tuple[NiftiImage, NumericArray]:
    """Load a NIfTI image with SimpleITK-compatible scaling for finite real voxels.

    Unlike SimpleITK's replacement of non-finite stored floats with zero, this
    rejects them. It applies ``scl_slope``/``scl_inter`` in double precision to values
    cast to float32 (float64 for float64 files) and returns that type; unscaled data
    keep their stored type. The array is transposed to SimpleITK's ``(z, y, x)`` order
    (a 2D image becomes ``(y, x)``). The voxel grid is not reoriented.

    Raises:
        InvalidInputError: The file is unreadable or contains complex or non-finite voxels.
        MissingDependencyError: nibabel is not installed.
    """

    nib = _nibabel()
    try:
        image = nib.load(path)
        proxy = image.dataobj
        raw = np.asanyarray(proxy.get_unscaled())
        slope, intercept = float(proxy.slope), float(proxy.inter)
    except (OSError, EOFError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise InvalidInputError(f"Unreadable MedSAM2 NIfTI {path}: {error}") from error
    if np.iscomplexobj(raw):
        raise InvalidInputError(f"MedSAM2 NIfTI voxels must be real: {path}")
    if not np.isfinite(raw).all():
        raise InvalidInputError(f"MedSAM2 NIfTI voxels must be finite: {path}")
    if slope == 1.0 and intercept == 0.0:
        values = raw
    else:
        float_type = np.float64 if raw.dtype == np.float64 else np.float32
        with np.errstate(over="ignore", invalid="ignore"):
            values = (raw.astype(float_type).astype(np.float64) * slope + intercept).astype(
                float_type
            )
        if not np.isfinite(values).all():
            raise InvalidInputError(f"MedSAM2 NIfTI scaling produced non-finite voxels: {path}")
    return image, values.T


def load_image_stack(
    path: Path, *, hu_window: tuple[float, float] | None = None
) -> tuple[NiftiImage, HostArray]:
    """Load a NIfTI volume or slice as its image and a uint8 ``(D, H, W)`` stack.

    ``D`` runs along the third voxel axis (slices), ``H`` along the second and ``W`` along
    the first. See :func:`ensure_uint8_stack` for the intensity scaling and ``hu_window``.
    """

    if not path.is_file():
        raise InvalidInputError(f"Missing MedSAM2 input: {path}")
    if not _is_nifti(path):
        raise InvalidInputError(f"MedSAM2 input must be a NIfTI (.nii/.nii.gz) volume: {path}")
    image, values = read_nifti(path)
    return image, ensure_uint8_stack(values, hu_window=hu_window)


_FRAME_SUFFIXES = (".jpg", ".jpeg")


def load_npz_stack(path: Path) -> tuple[HostArray, tuple[float, float, float] | None]:
    """Load the pinned RECIST NPZ contract's ``imgs`` stack, in ``[0, 256)``.

    ``medsam2_infer_CT_lesion_npz_recist.py`` reads ``imgs`` (D, H, W) and asserts
    ``np.max(img_3D_ori) < 256`` before using it directly (divided by 255 for
    ImageNet normalization, not rescaled from its own min/max like a NIfTI volume).
    This loader does not min-max rescale as ``ensure_uint8_stack`` does: it requires
    finite values in [0, 256), rounds them to the nearest integer, clips to [0, 255]
    and casts to uint8. The optional ``spacing`` key supplies its first three values
    in the (x, y, z) order the upstream script passes to SimpleITK's ``SetSpacing``.
    """

    if not path.is_file():
        raise InvalidInputError(f"Missing MedSAM2 input: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "imgs" not in data.files:
                raise InvalidInputError(f"MedSAM2 NPZ input {path} is missing an 'imgs' array")
            stack = np.asarray(data["imgs"])
            spacing = (
                tuple(float(value) for value in np.asarray(data["spacing"]).reshape(-1)[:3])
                if "spacing" in data.files
                else None
            )
    except (OSError, EOFError, ValueError) as error:
        raise InvalidInputError(f"Unreadable MedSAM2 NPZ {path}: {error}") from error
    if spacing is not None and len(spacing) != 3:
        raise InvalidInputError(f"MedSAM2 NPZ 'spacing' must have 3 values: {path}")
    if stack.ndim == 2:
        stack = stack[None, :, :]
    elif stack.ndim != 3:
        raise InvalidInputError("MedSAM2 NPZ 'imgs' must be a 2D grayscale image or D,H,W stack")
    if not stack.size:
        raise InvalidInputError("MedSAM2 NPZ 'imgs' must be nonempty")
    if not np.isfinite(stack).all():
        raise InvalidInputError("MedSAM2 NPZ 'imgs' must contain only finite values")
    if float(np.min(stack)) < 0.0 or float(np.max(stack)) >= 256.0:
        raise InvalidInputError(
            "MedSAM2 NPZ 'imgs' must already be in [0, 256), as upstream's RECIST NPZ "
            f"contract requires: {path}"
        )
    return np.clip(np.round(stack), 0, 255).astype(np.uint8), spacing


def load_frame_folder(path: Path) -> HostArray:
    """Load a sorted ``.jpg``/``.jpeg`` directory as a grayscale uint8 ``(D, H, W)`` stack.

    Upstream's interactive demo (``app.py``) extracts a video to a sorted JPEG
    frame folder with ffmpeg and loads it with ``predictor.init_state(video_path=...)``,
    whose frame loader (``sam2/utils/misc.py``) converts each frame to RGB. This port
    instead decodes each frame to one grayscale channel and reuses the grayscale-to-RGB
    preprocessing (``prepare_medsam2_images``). That matches upstream's per-channel
    ImageNet normalization only for frames whose three channels are identical, as in
    single-channel ultrasound/echo content (the ``MedSAM2_US_Heart`` use case); color
    frames are reduced to luminance and so differ from upstream. Direct ``.mp4``
    decoding is out of scope: upstream's own video-file loader requires the extra
    ``decord`` dependency, and upstream's own documented fallback for a video file is
    this same ffmpeg-to-JPEG-folder path.
    """

    if not path.is_dir():
        raise InvalidInputError(f"Missing MedSAM2 input: {path}")
    from PIL import Image

    frame_paths = sorted(
        entry
        for entry in path.iterdir()
        if entry.is_file() and entry.suffix.lower() in _FRAME_SUFFIXES
    )
    if not frame_paths:
        raise InvalidInputError(f"MedSAM2 frame folder has no .jpg/.jpeg frames: {path}")
    frames: list[HostArray] = []
    shape: tuple[int, int] | None = None
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as handle:
                frame = np.asarray(handle.convert("L"), dtype=np.uint8)
        except OSError as error:
            raise InvalidInputError(f"Unreadable MedSAM2 frame {frame_path}: {error}") from error
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise InvalidInputError(
                f"MedSAM2 frame folder has mismatched frame sizes {shape} and {frame.shape}: {path}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _placeholder_source(
    stack: HostArray, *, spacing: tuple[float, float, float] | None
) -> NiftiImage:
    """A synthetic NIfTI image letting the NIfTI output path serve non-NIfTI input.

    Voxel sizes come from the NPZ ``spacing`` (x, y, z) when given; a frame folder
    has no physical spacing, so it gets the identity affine. This only lets the
    existing NIfTI-shaped output writer and ``source.shape``/``source.affine``
    bookkeeping run unchanged; nothing is read back from these synthetic voxels.
    """

    nib = _nibabel()
    affine = np.eye(4, dtype=np.float64) if spacing is None else np.diag([*spacing, 1.0])
    return nib.Nifti1Image(stack.T, affine)


def ensure_uint8_stack(
    images: NumericArray, *, hu_window: tuple[float, float] | None = None
) -> HostArray:
    """Normalize a grayscale slice or volume to a uint8 ``(D, H, W)`` stack.

    Without ``hu_window``, every voxel sets the min-max scale (this port's default,
    for inputs with no per-case window). With ``hu_window`` (lower, upper), values
    are clipped to it first, then min-max scaled from the clipped data's own range:
    the pinned CT example's preprocessing (``medsam2_infer_3D_CT.py``) for one
    lesion, using that dataset's per-case DICOM window instead of a fixed one.
    ``hu_window`` is in the volume's own intensity units (Hounsfield units for CT).

    The result is truncated, not rounded, to uint8 after scaling to 0-255, and a
    constant stack becomes all zeros. A 2D input gets a leading slice axis.

    Raises:
        InvalidInputError: The input is not 2D or 3D, is empty, is complex or non-finite,
            or its intensity range cannot be normalized without overflow.
    """

    array = np.asarray(images)
    if array.ndim == 2:
        stack = array[None, :, :]
    elif array.ndim == 3:
        stack = array
    else:
        raise InvalidInputError("MedSAM2 images must be a 2D grayscale image or D,H,W stack")
    stack = np.asarray(stack)
    if not stack.size:
        raise InvalidInputError("MedSAM2 image stack must be nonempty")
    if np.iscomplexobj(stack):
        raise InvalidInputError("MedSAM2 image stack must contain real values")
    if not np.isfinite(stack).all():
        raise InvalidInputError("MedSAM2 image stack must contain only finite values")
    if hu_window is not None:
        lower, upper = hu_window
        stack = np.clip(stack, lower, upper)
    low = float(np.min(stack))
    high = float(np.max(stack))
    if high <= low:
        return np.zeros(stack.shape, dtype=np.uint8)
    span = high - low
    if not np.isfinite(span):
        raise InvalidInputError("MedSAM2 intensity range is too large to normalize")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (stack - low) / span
    if not np.isfinite(normalized).all():
        raise InvalidInputError("MedSAM2 intensity normalization overflowed")
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


# --- Upstream CT preprocessing -------------------------------------------------------


@cache
def _normalization_table() -> HostArray:
    # Retain upstream's float64 arithmetic with float32 mean/std, then round once.
    """Precompute ImageNet-normalized float32 values for each uint8 intensity."""
    pixels = np.arange(256, dtype=np.float64)[None, :] / 255.0
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, None]
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, None]
    return ((pixels - mean) / std).astype(np.float32)


def prepare_medsam2_images(stack: HostArray) -> HostArray:
    """Preprocess a uint8 ``(D, H, W)`` stack into ``(D, 3, 512, 512)`` float32 frames.

    Each grayscale slice is resized to 512 x 512 with PIL bicubic (if not already that
    size), then replicated to three channels and ImageNet-normalized
    (``(x / 255 - mean) / std``) by table lookup. This is the pinned CT example's
    preprocessing, cast to FP32 for the native MLX graph.
    """
    prepared = np.empty((len(stack), 3, 512, 512), dtype=np.float32)
    for index, frame in enumerate(stack):
        _prepare_frame(frame, out=prepared[index])
    return prepared


def _prepare_frame(frame: HostArray, *, out: HostArray) -> None:
    """Write one uint8 ``(H, W)`` slice into ``out`` as in ``prepare_medsam2_images``."""
    from PIL import Image

    image = Image.fromarray(frame.astype(np.uint8))
    if image.size != (512, 512):
        image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    # Uint8 indices are already in range; clip avoids take's exception buffer.
    np.take(_normalization_table(), np.asarray(image), axis=1, out=out, mode="clip")


class _PreparedStack:
    """Lazy ``PreparedVideoFrames`` over a uint8 ``(D, H, W)`` stack.

    Each access prepares one slice exactly as ``prepare_medsam2_images`` does, so a
    session never holds the whole FP32 (D, 3, 512, 512) stack (4 bytes x 3 channels
    per 512 x 512 voxel) and the predictor's next-frame prefetch overlaps this host
    work with GPU tracking.
    """

    def __init__(self, stack: HostArray) -> None:
        """Wrap the uint8 slice stack without copying it."""
        self._stack = stack

    def __len__(self) -> int:
        """Return the number of slices."""
        return len(self._stack)

    def __getitem__(self, index: int, /) -> HostArray:
        """Return slice ``index`` as a finite FP32 (3, 512, 512) model input."""
        prepared = np.empty((3, 512, 512), dtype=np.float32)
        _prepare_frame(self._stack[index], out=prepared)
        return prepared


# --- Prompts and sessions ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaskResult:
    """Boolean ``(num_objects, height, width)`` masks for one frame."""

    frame_idx: int
    obj_ids: tuple[int, ...]
    masks: BoolArray


def mask_result(frame_idx: int, obj_ids: Sequence[int], mask_logits: NumericArray) -> MaskResult:
    """Threshold predictor logits at 0 into boolean masks.

    A four-dimensional input uses its first channel; a two-dimensional input gains
    an object axis. Object count and array shape are not checked here.
    """

    masks = np.asarray(mask_logits) > 0.0
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return MaskResult(
        frame_idx=int(frame_idx),
        obj_ids=tuple(int(value) for value in obj_ids),
        masks=masks.astype(bool),
    )


@dataclass(slots=True)
class Session[StateT]:
    """A predictor's inference state over one uint8 ``(D, H, W)`` stack."""

    predictor: RunnerPredictor[StateT]
    inference_state: StateT
    image_stack: HostArray

    @property
    def num_frames(self) -> int:
        """Return the number of slices in the session stack."""
        return int(self.image_stack.shape[0])

    @property
    def height(self) -> int:
        """Return the height of an original session slice."""
        return int(self.image_stack.shape[1])

    @property
    def width(self) -> int:
        """Return the width of an original session slice."""
        return int(self.image_stack.shape[2])

    def add_prompt(self, prompt: MedSAM2Prompt) -> MaskResult:
        """Send a prompt to the predictor (mask prompts alone, otherwise box and/or points)."""
        if prompt.mask is not None:
            frame_idx, obj_ids, mask_logits = self.predictor.add_new_mask(
                self.inference_state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
                mask=prompt.mask,
            )
        else:
            frame_idx, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
                points=np.asarray(prompt.points_xy, dtype=np.float32) if prompt.points_xy else None,
                labels=np.asarray(prompt.point_labels, dtype=np.int32)
                if prompt.point_labels
                else None,
                box=np.asarray(prompt.box_xyxy, dtype=np.float32)
                if prompt.box_xyxy is not None
                else None,
                clear_old_points=prompt.clear_old_points,
                normalize_coords=prompt.normalize_coords,
            )
        return mask_result(frame_idx, obj_ids, cast(NumericArray, mask_logits))

    def propagate(
        self,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        stop_after_empty: int | None = None,
    ) -> list[MaskResult]:
        """Propagate from the prompted frames and return every yielded frame's masks.

        With ``stop_after_empty``, tracking stops once that many consecutive yielded
        frames have no foreground for any object; those empty frames are returned and
        later frames are never computed.
        """
        return list(
            _iter_propagation(
                self,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
                stop_after_empty=stop_after_empty,
            )
        )


def _iter_propagation[StateT](
    session: Session[StateT],
    *,
    start_frame_idx: int | None,
    max_frame_num_to_track: int | None,
    reverse: bool,
    stop_after_empty: int | None,
) -> Iterator[MaskResult]:
    """Yield completed host masks without retaining a direction's results.

    Network execution and temporal state stay with the predictor. Thresholding,
    empty-frame counting and label assembly remain NumPy CPU postprocessing.
    """
    empty_run = 0
    frames = session.predictor.propagate_in_video(
        session.inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    )
    for frame_idx, obj_ids, mask_logits in frames:
        result = mask_result(frame_idx, obj_ids, cast(NumericArray, mask_logits))
        empty_run = 0 if result.masks.any() else empty_run + 1
        yield result
        if stop_after_empty is not None and empty_run >= stop_after_empty:
            break


def create_session[StateT](
    stack: HostArray, predictor: RunnerPredictor[StateT], runtime: MedSAM2RuntimeConfig
) -> Session[StateT]:
    """Initialize the upstream-compatible predictor state over lazily preprocessed frames.

    Frames are prepared on access from the uint8 stack, so the FP32 frame stack is never
    materialized on the host or device and ``runtime.offload_video_to_cpu`` does not
    apply; ``runtime.offload_state_to_cpu`` still selects where state tensors live.
    """

    inference_state = predictor.init_state(
        _PreparedStack(stack),
        int(stack.shape[1]),
        int(stack.shape[2]),
        offload_video_to_cpu=True,
        offload_state_to_cpu=runtime.offload_state_to_cpu,
    )
    return Session(predictor=predictor, inference_state=inference_state, image_stack=stack)


# --- Prompted run --------------------------------------------------------------------


def run_prompted_session[StateT](
    session: Session[StateT],
    *,
    prompt_tuple: tuple[MedSAM2Prompt, ...],
    propagation: Literal["none", "forward", "reverse", "both"],
    max_frame_num_to_track: int | None,
    slice_range: tuple[int, int] | None,
    stop_after_empty: int | None = None,
) -> HostArray:
    """Seed the prompts, propagate, and merge results into one uint16 label volume.

    Propagation starts at the first prompt's frame; the public runner requires a shared
    prompt frame when propagation is enabled. With ``"none"``, prompts may target
    different frames. With ``"both"`` the session is reset and re-seeded between the
    two directions, so the reverse pass does not see forward-pass memories.
    ``stop_after_empty`` ends each direction after that many consecutive empty frames.
    """

    mask_volume = np.zeros(session.image_stack.shape, dtype=np.uint16)
    for prompt in prompt_tuple:
        merge_result(mask_volume, session.add_prompt(prompt), slice_range=slice_range)
    start_frame_idx = prompt_tuple[0].frame_idx
    if propagation in {"forward", "both"}:
        for result in _iter_propagation(
            session,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=direction_limit(
                max_frame_num_to_track, slice_range, start_frame_idx, reverse=False
            ),
            reverse=False,
            stop_after_empty=stop_after_empty,
        ):
            merge_result(mask_volume, result, slice_range=slice_range)
    if propagation in {"reverse", "both"}:
        if propagation == "both":
            session.predictor.reset_state(session.inference_state)
            for prompt in prompt_tuple:
                session.add_prompt(prompt)
        for result in _iter_propagation(
            session,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=direction_limit(
                max_frame_num_to_track, slice_range, start_frame_idx, reverse=True
            ),
            reverse=True,
            stop_after_empty=stop_after_empty,
        ):
            merge_result(mask_volume, result, slice_range=slice_range)
    return mask_volume


def merge_result(
    mask_volume: HostArray, result: MaskResult, *, slice_range: tuple[int, int] | None = None
) -> None:
    """Stamp each object's mask into the uint16 ``(D, H, W)`` label volume.

    Voxels in a mask take the object's id; others are left unchanged, so later objects
    overwrite earlier ones where masks overlap. Results outside ``slice_range`` are
    skipped. Raises ``ModelExecutionError`` for a frame index outside the volume.
    """

    if result.frame_idx < 0 or result.frame_idx >= mask_volume.shape[0]:
        raise ModelExecutionError(f"MedSAM2 result frame out of range: {result.frame_idx}")
    if slice_range is not None and not (slice_range[0] <= result.frame_idx <= slice_range[1]):
        return
    for obj_id, mask in zip(result.obj_ids, result.masks, strict=True):
        mask_volume[result.frame_idx][mask] = int(obj_id)


def validate_slice_range(
    slice_range: tuple[int, int] | None, prompts: tuple[MedSAM2Prompt, ...], depth: int
) -> None:
    """Require prompt frames within the volume and an ordered ``slice_range`` containing them.

    Raises ``InvalidInputError`` otherwise. ``depth`` is the number of slices.
    """
    if any(prompt.frame_idx < 0 or prompt.frame_idx >= depth for prompt in prompts):
        raise InvalidInputError(f"MedSAM2 prompt frame must be within [0, {depth - 1}]")
    if slice_range is None:
        return
    if len(slice_range) != 2:
        raise InvalidInputError("MedSAM2 slice_range must contain exactly (start, end)")
    start, end = slice_range
    if any(type(value) is not int for value in slice_range):
        raise InvalidInputError("MedSAM2 slice_range values must be integer indices")
    if start < 0 or end >= depth or start > end:
        raise InvalidInputError(f"MedSAM2 slice_range must be within [0, {depth - 1}] and ordered")
    if not prompts:
        raise InvalidInputError("MedSAM2 slice_range requires a prompt")
    if any(prompt.frame_idx < start or prompt.frame_idx > end for prompt in prompts):
        raise InvalidInputError("MedSAM2 slice_range must include every prompt frame")


def direction_limit(
    requested: int | None,
    slice_range: tuple[int, int] | None,
    start_frame_idx: int,
    *,
    reverse: bool,
) -> int | None:
    """Frames to track from ``start_frame_idx`` so propagation stops at the ``slice_range`` end.

    Combines the requested limit with the distance to the range edge in the tracking
    direction. ``None`` means unbounded.
    """
    if slice_range is None:
        return requested
    bounded = start_frame_idx - slice_range[0] if reverse else slice_range[1] - start_frame_idx
    return bounded if requested is None else min(requested, bounded)


def load_source_aligned_mask(
    mask_path: Path, input_path: Path, source: NiftiImage, shape: tuple[int, ...], frame_idx: int
) -> HostArray:
    """Load a binary NIfTI mask on the input grid whose voxels lie on the seed slice.

    The scaled mask must be finite, contain only 0 and 1, match the source shape
    and affine exactly, and have at least one foreground voxel on the seed slice.
    """

    if not mask_path.is_file():
        raise InvalidInputError(f"Missing MedSAM2 input: {mask_path}")
    if not _is_nifti(mask_path):
        raise InvalidInputError(
            "MedSAM2 mask_path must be a source-aligned NIfTI (.nii/.nii.gz) mask"
        )
    if not _is_nifti(input_path):
        raise InvalidInputError("NIfTI mask prompts require a NIfTI input volume")
    nib = _nibabel()
    try:
        raw_mask = np.asanyarray(nib.load(mask_path).dataobj)
    except (OSError, EOFError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise InvalidInputError(f"Unreadable MedSAM2 mask {mask_path}: {error}") from error
    if not np.isfinite(raw_mask).all():
        raise InvalidInputError("MedSAM2 mask must contain only finite values")
    mask_image, mask = read_nifti(mask_path)
    if mask.ndim == 2 and len(source.shape) == 2:
        # Match the singleton slice axis added to a 2D input by ensure_uint8_stack.
        mask = mask[None]
    if mask.shape != shape:
        raise InvalidInputError(
            f"MedSAM2 mask shape {mask.shape} does not match source shape {shape}"
        )
    if not np.array_equal(mask_image.affine, source.affine):
        raise InvalidInputError("MedSAM2 mask affine/grid does not match the source NIfTI volume")
    if not np.isfinite(mask).all():
        raise InvalidInputError("MedSAM2 mask must contain only finite values")
    if not np.all(np.isin(mask, (0, 1))):
        raise InvalidInputError("MedSAM2 mask must be binary with values 0 and 1")
    if np.any(mask[:frame_idx]) or np.any(mask[frame_idx + 1 :]):
        raise InvalidInputError("MedSAM2 mask must contain nonzero voxels only on the seed slice")
    if not np.any(mask[frame_idx]):
        raise InvalidInputError("MedSAM2 mask seed slice must be nonempty")
    return mask[frame_idx].astype(bool, copy=True)


def _pil_from_uint8(image: HostArray) -> PILImage:
    """Convert the supplied array to an 8-bit grayscale PIL image."""
    from PIL import Image

    return Image.fromarray(image.astype(np.uint8)).convert("L")


def save_overlay(
    image: PILImage, mask: BoolArray, prompts: Sequence[MedSAM2Prompt], path: Path
) -> None:
    """Save a yellow mask overlay with the CT example's color and opacity.

    Point markers are drawn for every prompt that carries points, across every
    prompted object, in the overlay's own pixel frame. A prompt with
    ``normalize_coords=True`` already carries pixel coordinates of this slice
    (see :class:`MedSAM2Prompt`); one with ``normalize_coords=False`` carries
    fractions of the slice size in ``[0, 1]`` and is scaled by ``image``'s size
    before drawing.
    """

    from PIL import Image

    base = np.asarray(image.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.asarray([251.0, 252.0, 30.0], dtype=np.float64) * 0.5
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if any(prompt.points_xy for prompt in prompts):
        from PIL import ImageDraw

        width, height = output.size
        draw = ImageDraw.Draw(output)
        for prompt in prompts:
            scale_x, scale_y = (
                (1.0, 1.0) if prompt.normalize_coords else (float(width), float(height))
            )
            for point, label in zip(prompt.points_xy, prompt.point_labels, strict=True):
                x, y = point[0] * scale_x, point[1] * scale_y
                color = (80, 255, 120) if label > 0 else (255, 220, 60)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(0, 0, 0))
    output.save(path)


# --- Runner ----------------------------------------------------------------------------


def _require_distinct_outputs(outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
    """Keep source images, prompt masks and weights intact when replacing outputs."""

    if any(
        output.resolve() == source.resolve()
        or (output.exists() and source.exists() and output.samefile(source))
        for output in outputs
        for source in inputs
    ):
        raise InvalidInputError("MedSAM2 outputs must differ from input images, masks and weights")


def _admit_weights(path: Path) -> Path:
    """Require an existing converted ``.safetensors`` file, else raise ``AssetNotReadyError``."""

    if not path.is_file():
        raise AssetNotReadyError(
            f"Missing MedSAM2 MLX weights: {path}", reason=f"missing {path}", hint=_WEIGHTS_HINT
        )
    if path.suffix.lower() != ".safetensors":
        raise AssetNotReadyError(
            f"MedSAM2 weights_path must be converted MLX weights (.safetensors): {path}",
            reason=f"unconverted {path}",
            hint=_WEIGHTS_HINT,
        )
    return path


def _build_predictor(runtime: MedSAM2RuntimeConfig, weights: Path) -> RunnerPredictor[SessionState]:
    """Load converted weights into the ``sam2.1_hiera_t512`` MLX predictor.

    Caught input, filesystem, runtime and value errors are reported as incompatible
    weights; that wrapper does not distinguish their underlying causes.
    """

    from medsam2_mlx.mlx_predictor import build_predictor

    try:
        return build_predictor(runtime, weights=weights)
    except (InvalidInputError, OSError, RuntimeError, ValueError) as error:
        raise AssetNotReadyError(
            f"MedSAM2 MLX weights {weights} do not match the pinned sam2.1_hiera_t512 "
            f"graph: {error}",
            reason=f"incompatible {weights}",
            hint=_WEIGHTS_HINT,
        ) from error


def _require_bool(name: str, value: object) -> bool:
    """Require a boolean value for the named request option."""
    if not isinstance(value, bool):
        raise InvalidInputError(f"MedSAM2 {name} must be true or false, got {value!r}")
    return value


def _require_int(name: str, value: object) -> int:
    """Require an integer value for the named request option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(f"MedSAM2 {name} must be an integer, got {value!r}")
    return value


def _box(value: object) -> tuple[float, float, float, float]:
    """Parse an ``(x_min, y_min, x_max, y_max)`` box of four finite numbers (bools rejected).

    Ordering and bounds are not checked here.
    """

    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(cast(Sequence[object], value)) != 4
    ):
        raise InvalidInputError("MedSAM2 box prompt must be XYXY with 4 values")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float)
        for item in cast(Sequence[object], value)
    ):
        raise InvalidInputError("MedSAM2 box prompt values must be numbers")
    box = cast(
        tuple[float, float, float, float],
        tuple(float(cast(float, item)) for item in cast(Sequence[object], value)),
    )
    if not all(np.isfinite(box)):
        raise InvalidInputError("MedSAM2 box prompt values must be finite")
    return box


def _slice_range(value: object) -> tuple[int, int] | None:
    """Convert a sequence to a tuple; length and values are checked by ``validate_slice_range``."""

    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError("MedSAM2 slice_range must contain exactly (start, end)")
    return cast(tuple[int, int], tuple(cast(Sequence[object], value)))


def _hu_window(value: object) -> tuple[float, float] | None:
    """Parse ``(lower, upper)`` finite intensity bounds with ``lower < upper``, or ``None``."""

    if value is None:
        return None
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(cast(Sequence[object], value)) != 2
    ):
        raise InvalidInputError("MedSAM2 hu_window must contain exactly (lower, upper)")
    items = cast(Sequence[object], value)
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in items):
        raise InvalidInputError("MedSAM2 hu_window values must be numbers")
    lower, upper = (float(cast(float, item)) for item in items)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise InvalidInputError("MedSAM2 hu_window values must be finite")
    if lower >= upper:
        raise InvalidInputError("MedSAM2 hu_window must have lower < upper")
    return (lower, upper)


def _points(value: object) -> tuple[tuple[float, float], ...]:
    """Parse ``[x, y]`` click positions (column, row) as finite floats."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError("MedSAM2 points_xy must be a sequence of [x, y] pairs")
    points: list[tuple[float, float]] = []
    for item in cast(Sequence[object], value):
        if (
            isinstance(item, str | bytes)
            or not isinstance(item, Sequence)
            or len(cast(Sequence[object], item)) != 2
        ):
            raise InvalidInputError("MedSAM2 each point in points_xy must have 2 values")
        pair = cast(Sequence[object], item)
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in pair):
            raise InvalidInputError("MedSAM2 point coordinates must be numbers")
        x, y = (float(cast(float, value)) for value in pair)
        if not (np.isfinite(x) and np.isfinite(y)):
            raise InvalidInputError("MedSAM2 point coordinates must be finite")
        points.append((x, y))
    return tuple(points)


def _point_labels(value: object) -> tuple[int, ...]:
    """Parse integer click labels; the 0/1 range is enforced by ``MedSAM2Prompt``."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError("MedSAM2 point_labels must be a sequence of integers")
    labels: list[int] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise InvalidInputError("MedSAM2 point_labels values must be integers")
        labels.append(item)
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class _PromptRequest:
    """One parsed object prompt, before its optional seed mask is loaded."""

    frame_idx: int
    obj_id: int
    box: tuple[float, float, float, float] | None
    points_xy: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]
    mask_path: Path | None
    clear_old_points: bool
    normalize_coords: bool


def _require_prompts_sequence(value: object) -> Sequence[Mapping[str, object]]:
    """Require a non-empty sequence of mappings (one per object prompt)."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError("MedSAM2 prompts must be a sequence of prompt mappings")
    entries = cast(Sequence[object], value)
    if not entries:
        raise InvalidInputError("MedSAM2 prompts must not be empty")
    parsed: list[Mapping[str, object]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise InvalidInputError("MedSAM2 each prompts entry must be a mapping")
        parsed.append(cast(Mapping[str, object], entry))
    return parsed


def _parse_prompt_entry(entry: Mapping[str, object], *, default_frame_idx: int) -> _PromptRequest:
    """Validate one ``prompts`` mapping; ``obj_id`` is required and ``frame_idx`` defaults.

    ``mask_path`` is resolved to an absolute path and excludes ``box_xyxy``/``points_xy``.
    """

    frame_idx = _require_int("frame_idx", entry.get("frame_idx", default_frame_idx))
    if "obj_id" not in entry:
        raise InvalidInputError("MedSAM2 each prompts entry requires obj_id")
    obj_id = _require_int("obj_id", entry["obj_id"])
    raw_box = entry.get("box_xyxy")
    box = None if raw_box is None else _box(raw_box)
    raw_points = entry.get("points_xy")
    points_xy = () if raw_points is None else _points(raw_points)
    raw_labels = entry.get("point_labels")
    point_labels = () if raw_labels is None else _point_labels(raw_labels)
    raw_mask_path = entry.get("mask_path")
    if raw_mask_path is not None and not isinstance(raw_mask_path, str | Path):
        raise InvalidInputError("MedSAM2 prompts entry mask_path must be a path")
    mask_path = None if raw_mask_path is None else Path(raw_mask_path).expanduser().resolve()
    if mask_path is not None and (box is not None or points_xy):
        raise InvalidInputError(
            "MedSAM2 prompts entry mask_path cannot combine with box_xyxy/points_xy"
        )
    clear_old_points = _require_bool("clear_old_points", entry.get("clear_old_points", True))
    normalize_coords = _require_bool("normalize_coords", entry.get("normalize_coords", True))
    return _PromptRequest(
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=box,
        points_xy=points_xy,
        point_labels=point_labels,
        mask_path=mask_path,
        clear_old_points=clear_old_points,
        normalize_coords=normalize_coords,
    )


def run(
    *,
    image_path: str | Path,
    weights_path: str | Path | None = None,
    output_dir: str | Path,
    box_xyxy: Sequence[float] | None = None,
    points_xy: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[int] | None = None,
    mask_path: str | Path | None = None,
    frame_idx: int = 0,
    obj_id: int = 1,
    prompts: Sequence[Mapping[str, object]] | None = None,
    hu_window: Sequence[float] | None = None,
    propagation: Literal["none", "forward", "reverse", "both"] = "none",
    slice_range: Sequence[int] | None = None,
    stop_after_empty: int | None = 8,
    apply_postprocessing: bool = True,
    overwrite: bool = False,
) -> RunnerResult:
    """Segment one or more prompted objects in a volume or frame stack, optionally propagated.

    ``image_path`` is a NIfTI volume, an ``.npz`` RECIST stack, or a directory of
    ``.jpg``/``.jpeg`` frames (see ``image_path`` below for each format's contract).
    Every frame is bicubic-resized to 512x512 and ImageNet-normalized before the
    pinned predictor runs. For a NIfTI input, the label mask keeps its shape and
    affine; for ``.npz``/frame-folder input, which carry no NIfTI grid, it instead
    gets an identity affine (or one built from the NPZ's own ``spacing``, if given).

    There are two mutually exclusive ways to prompt. For one object, give
    ``box_xyxy`` and/or ``points_xy``/``point_labels`` (combinable, as upstream's
    interactive predictor allows), or ``mask_path`` alone, with ``frame_idx`` and
    ``obj_id`` naming its slice and label. For multiple objects, give ``prompts``
    instead: a sequence of mappings, each with a required ``obj_id`` and any of
    ``frame_idx`` (default 0), ``box_xyxy``, ``points_xy``, ``point_labels``,
    ``mask_path``, ``clear_old_points`` (default ``True``) and ``normalize_coords``
    (default ``True``), mirroring the upstream predictor's ``add_new_mask``/
    ``add_new_points_or_box`` calls for each tracked object. Every object is
    seeded before propagation runs; when propagation is enabled they must share a frame.

    Args:
        image_path: One of three input kinds. A NIfTI (``.nii``/``.nii.gz``) CT
            volume or a single 2D slice; frames are slices along its third voxel
            axis, and pixel ``x``/``y`` are its first and second voxel axes, with
            the volume min-max scaled to uint8 (optionally clipped first, see
            ``hu_window``). An ``.npz`` with an ``imgs`` (D, H, W) stack or (H, W) slice already
            in ``[0, 256)`` and an optional ``spacing`` (x, y, z), adapted from
            upstream's RECIST NPZ contract. Values are rounded, clipped to
            ``[0, 255]`` and cast to uint8 without min-max scaling. Or a directory of sorted
            ``.jpg``/``.jpeg`` frames (grayscale, one channel), the pinned
            interactive demo's non-NIfTI video path after its own ffmpeg frame
            extraction; direct ``.mp4`` decoding is out of scope here, since
            upstream's own video-file loader needs the extra ``decord``
            dependency and documents this same frame-folder path as its fallback.
            ``hu_window`` and NIfTI ``mask_path`` prompts apply only to a NIfTI
            input.
        weights_path: Optional model folder; omitted downloads ``MedMLX/medsam2-ct-lesion-mlx``.
            The folder contains ``MedSAM2_CTLesion.safetensors``. Explicit native
            checkpoint-file paths are also accepted for the pinned
            ``sam2.1_hiera_t512`` graph. Other published checkpoints are at
            https://huggingface.co/MedMLX/medsam2-mlx.
        output_dir: Directory for the outputs; created if missing.
        box_xyxy: Box prompt ``[x_min, y_min, x_max, y_max]`` in pixels of slice
            ``frame_idx`` as stored (before the 512 x 512 resize). May combine with
            ``points_xy``; not with ``mask_path``.
        points_xy: Point prompts, each an ``[x, y]`` pixel pair on slice
            ``frame_idx`` in the same coordinates, paired with ``point_labels``. May
            combine with ``box_xyxy``; not with ``mask_path``.
        point_labels: One upstream SAM label per entry of ``points_xy``: ``1``
            for foreground, ``0`` for background.
        mask_path: Binary NIfTI mask on the input's grid (same shape and exactly equal affine)
            whose nonzero voxels all lie on slice ``frame_idx``. Exclusive with
            ``box_xyxy``/``points_xy``.
        frame_idx: Zero-based slice index of the single-object prompt.
        obj_id: Positive label written for the single-object prompt. Output labels are
            uint16, so IDs must fit in 1..65535.
        prompts: Multiple objects' prompts; see above. Exclusive with
            ``box_xyxy``, ``points_xy``, ``point_labels``, ``mask_path``, and
            with passing a non-default ``frame_idx``/``obj_id``.
        hu_window: Optional ``(lower, upper)`` intensity clip, in the volume's own units
            (Hounsfield units for CT), applied before the volume is min-max scaled to
            uint8. It mirrors the pinned CT example's per-lesion DICOM window
            (``medsam2_infer_3D_CT.py``). Omit it to scale from the volume's own
            range; this port never chooses a window itself.
        propagation: ``"none"`` (only the prompted slices), ``"forward"`` (toward
            higher slice indices), ``"reverse"``, or ``"both"``.
        slice_range: Inclusive ``[start, end]`` slice indices; propagation stops at
            its ends and slices outside it stay empty. It must contain every
            prompt's frame.
        stop_after_empty: Positive count, 8 by default. Each propagation direction
            stops after this many consecutive slices in which no object has foreground,
            and later slices in that direction stay empty. This departs from upstream,
            which tracks to the end of the volume; in testing, the output dropped this
            way lay outside the prompted structure, where the tracker had moved to a
            different one. An object that genuinely reappears after the gap is missed.
            ``None`` tracks the whole volume or ``slice_range`` as upstream does.
        apply_postprocessing: Upstream mask stability fallback, binarized click masks
            in memory, and filling of background components up to 8 pixels.
        overwrite: Replace existing outputs once the new ones are written; a failed run
            keeps them. Input images, masks and weights are protected.

    Returns:
        ``outputs`` maps ``mask`` to ``medsam2_mask.nii.gz`` (uint16 labels 0 and
        each ``obj_id``) and, for single-slice inputs, ``overlay`` to
        ``medsam2_overlay.png``. ``details`` holds the frame count and size,
        labelled voxel count, slices with labels, object ids, and every prompt.

    Raises:
        InvalidInputError: Missing or unreadable inputs, no prompt or conflicting
            prompt styles, malformed prompts or slice ranges, prompts that
            propagate from different frames, duplicate ``obj_id`` values, or
            existing outputs without ``overwrite``.
        AssetNotReadyError: Missing, unconverted, or mismatched weights.
        MissingDependencyError: MLX or nibabel is unavailable.
        ModelExecutionError: MLX raised a runtime or value error during the run.

    Box ordering and coordinate bounds are caller responsibilities: the runner checks
    coordinate shapes and finiteness but does not clip coordinates to the image.
    When object masks overlap, the later object's ID overwrites the earlier one.
    """

    apply_postprocessing = _require_bool("apply_postprocessing", apply_postprocessing)
    overwrite = _require_bool("overwrite", overwrite)
    if propagation not in PROPAGATION_MODES:
        raise InvalidInputError(
            f"MedSAM2 propagation must be one of {', '.join(PROPAGATION_MODES)}, "
            f"got {propagation!r}"
        )
    requested_range = _slice_range(slice_range)
    if stop_after_empty is not None and _require_int("stop_after_empty", stop_after_empty) < 1:
        raise InvalidInputError(
            f"MedSAM2 stop_after_empty must be a positive integer, got {stop_after_empty!r}"
        )
    window = _hu_window(hu_window)

    source_path = Path(image_path).expanduser().resolve()
    if not source_path.exists():
        raise InvalidInputError(f"Missing MedSAM2 input: {source_path}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"MedSAM2 output_dir is not a directory: {output}")
    mask_output = output / MASK_NAME
    overlay_output = output / OVERLAY_NAME
    _require_distinct_outputs((mask_output, overlay_output), (source_path,))
    existing = [
        path.as_posix()
        for path in (mask_output, overlay_output)
        if path.exists() or path.is_symlink()
    ]
    if existing and not overwrite:
        raise InvalidInputError(
            f"MedSAM2 outputs already exist: {', '.join(existing)}; "
            "pass overwrite=True to replace them"
        )

    if source_path.is_dir():
        if window is not None:
            raise InvalidInputError(
                "MedSAM2 hu_window applies only to NIfTI input, not a frame folder"
            )
        _require_distinct_outputs(
            (mask_output, overlay_output),
            tuple(
                entry
                for entry in source_path.iterdir()
                if entry.is_file() and entry.suffix.lower() in _FRAME_SUFFIXES
            ),
        )
        stack = load_frame_folder(source_path)
        source = _placeholder_source(stack, spacing=None)
    elif source_path.suffix.lower() == ".npz":
        if window is not None:
            raise InvalidInputError(
                "MedSAM2 hu_window does not apply to NPZ input already scaled to [0, 255]"
            )
        stack, spacing = load_npz_stack(source_path)
        source = _placeholder_source(stack, spacing=spacing)
    elif _is_nifti(source_path):
        source, stack = load_image_stack(source_path, hu_window=window)
    else:
        raise InvalidInputError(
            "MedSAM2 input must be a NIfTI (.nii/.nii.gz) volume, an .npz RECIST "
            f"stack, or a directory of .jpg/.jpeg frames: {source_path}"
        )

    if prompts is not None:
        if (
            box_xyxy is not None
            or points_xy is not None
            or point_labels is not None
            or mask_path is not None
            or frame_idx != 0
            or obj_id != 1
        ):
            raise InvalidInputError(
                "MedSAM2 prompts replaces box_xyxy/points_xy/point_labels/mask_path/"
                "frame_idx/obj_id; use only one prompt style"
            )
        requests = [
            _parse_prompt_entry(entry, default_frame_idx=0)
            for entry in _require_prompts_sequence(prompts)
        ]
    else:
        if mask_path is not None and (box_xyxy is not None or points_xy is not None):
            raise InvalidInputError("MedSAM2 mask_path cannot combine with box_xyxy/points_xy")
        if mask_path is None and box_xyxy is None and points_xy is None:
            raise InvalidInputError(
                "MedSAM2 requires one of box_xyxy, points_xy and point_labels, or mask_path"
            )
        requests = [
            _PromptRequest(
                frame_idx=_require_int("frame_idx", frame_idx),
                obj_id=_require_int("obj_id", obj_id),
                box=None if box_xyxy is None else _box(box_xyxy),
                points_xy=() if points_xy is None else _points(points_xy),
                point_labels=() if point_labels is None else _point_labels(point_labels),
                mask_path=None if mask_path is None else Path(mask_path).expanduser().resolve(),
                clear_old_points=True,
                normalize_coords=True,
            )
        ]

    _require_distinct_outputs(
        (mask_output, overlay_output),
        tuple(request.mask_path for request in requests if request.mask_path is not None),
    )
    object_ids = [request.obj_id for request in requests]
    if any(not 1 <= obj_id <= 65535 for obj_id in object_ids):
        raise InvalidInputError("MedSAM2 output obj_id values must fit in uint16 (1..65535)")
    if len(set(object_ids)) != len(object_ids):
        raise InvalidInputError("MedSAM2 prompts obj_id values must be unique")
    if propagation != "none" and len({request.frame_idx for request in requests}) > 1:
        raise InvalidInputError(
            "MedSAM2 propagation requires every prompt to share the same frame_idx"
        )

    prompt_list: list[MedSAM2Prompt] = []
    single_mask_path: Path | None = None
    for request in requests:
        if request.frame_idx < 0 or request.frame_idx >= stack.shape[0]:
            raise InvalidInputError(
                f"MedSAM2 prompt frame must be within [0, {stack.shape[0] - 1}]"
            )
        mask = None
        if request.mask_path is not None:
            mask = load_source_aligned_mask(
                request.mask_path, source_path, source, stack.shape, request.frame_idx
            )
            if len(requests) == 1:
                single_mask_path = request.mask_path
        prompt_list.append(
            MedSAM2Prompt(
                box_xyxy=request.box,
                points_xy=request.points_xy,
                point_labels=request.point_labels,
                mask=mask,
                frame_idx=request.frame_idx,
                obj_id=request.obj_id,
                clear_old_points=request.clear_old_points,
                normalize_coords=request.normalize_coords,
            )
        )
    prompt_tuple = tuple(prompt_list)
    validate_slice_range(requested_range, prompt_tuple, stack.shape[0])

    # Every argument is checked; only now may the weights download.
    # Keep the .safetensors name of Hub-cache symlinks for admission and loading;
    # their content-addressed targets have no extension. Alias checks still resolve paths.
    bundle_dir = weights_dir("medsam2") if weights_path is None else Path(weights_path).expanduser()
    # Preserve explicit checkpoint-file callers and named cache symlinks.
    checkpoint = bundle_dir / "MedSAM2_CTLesion.safetensors" if bundle_dir.is_dir() else bundle_dir
    weights = _admit_weights(checkpoint.absolute())
    _require_distinct_outputs((mask_output, overlay_output), (weights,))
    runtime = MedSAM2RuntimeConfig(checkpoint=weights, apply_postprocessing=apply_postprocessing)
    predictor = _build_predictor(runtime, weights)
    try:
        session = create_session(stack, predictor, runtime)
        mask_volume = run_prompted_session(
            session,
            prompt_tuple=prompt_tuple,
            propagation=propagation,
            max_frame_num_to_track=None,
            slice_range=requested_range,
            stop_after_empty=stop_after_empty,
        )
    except MedmlxError:
        raise
    # MLX reports device and graph failures as RuntimeError or ValueError; other
    # exception types are defects and propagate unchanged.
    except (RuntimeError, ValueError) as error:
        raise ModelExecutionError(f"medsam2 failed during execution: {error}") from error

    nib = _nibabel()
    output.mkdir(parents=True, exist_ok=True)
    labels = mask_volume.reshape(source.shape[::-1]).T
    image = cast(NiftiConstructor, source.__class__)(labels, None, header=source.header)
    image.set_data_dtype(np.uint16)
    # Keep staged outputs private because the image writers follow links when opening
    # paths. Their filenames retain the suffixes required by the writers.
    with TemporaryDirectory(prefix=".medsam2-", dir=output) as staging_dir:
        staged = {mask_output: Path(staging_dir) / MASK_NAME}
        if session.num_frames == 1:
            staged[overlay_output] = Path(staging_dir) / OVERLAY_NAME
        nib.save(image, staged[mask_output])
        if overlay_output in staged:
            save_overlay(
                _pil_from_uint8(stack[0]), mask_volume[0] > 0, prompt_tuple, staged[overlay_output]
            )
        for target, temporary in staged.items():
            os.replace(temporary, target)
    outputs = {"mask": mask_output.as_posix()}
    if overlay_output in staged:
        outputs["overlay"] = overlay_output.as_posix()
    labelled = np.flatnonzero(mask_volume.reshape(mask_volume.shape[0], -1).any(axis=1))
    return {
        "outputs": outputs,
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "num_frames": session.num_frames,
            "height": session.height,
            "width": session.width,
            "mask_voxels": int(np.count_nonzero(mask_volume)),
            "frames_with_mask": [int(index) for index in labelled],
            "object_ids": sorted({item.obj_id for item in prompt_tuple}),
            "prompts": [medsam2_prompt_payload(item) for item in prompt_tuple],
            "mask_path": None if single_mask_path is None else single_mask_path.as_posix(),
            "hu_window": None if window is None else list(window),
            "propagation": propagation,
            "slice_range": None if requested_range is None else list(requested_range),
            "stop_after_empty": stop_after_empty,
            "postprocessing_applied": apply_postprocessing,
            "coordinate_space": "image_voxel",
            "output_shape": [int(value) for value in labels.shape],
            "output_dtype": "uint16",
            "output_units": "label_index",
            "backend": "mlx",
        },
    }
