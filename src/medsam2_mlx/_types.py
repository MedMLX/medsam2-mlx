"""Consumed MLX and runner interfaces; imports and native identities stay lazy."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, NotRequired, Protocol, TypedDict, overload

import numpy as np
import numpy.typing as npt
from medmlx_core.outputs import Artifact
from medmlx_core.typing import MetalKernel

if TYPE_CHECKING:
    from mlx.core import (
        Device,
        Dtype,
        StreamContext,
        StreamOrDevice,
        ThreadLocalStream,
        array,
    )

    from medsam2_mlx.mlx_predictor import PreparedVideoFrames

type Scalar = int | float | bool | complex
type ArrayInput = Scalar | array | npt.NDArray[np.generic] | Sequence[ArrayInput]
type FloatArray = npt.NDArray[np.float32]
type HostArray = npt.NDArray[np.generic]


class ArrayConstructor(Protocol):
    def __call__(self, value: ArrayInput, dtype: Dtype | None = None) -> array:
        """Create an MLX array from the given values and optional dtype."""
        ...


class RandomRuntime(Protocol):
    @staticmethod
    def uniform(
        low: Scalar | array = 0,
        high: Scalar | array = 1,
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Generate uniformly distributed MLX values with the requested shape."""
        ...
    @staticmethod
    def normal(
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        loc: Scalar | array | None = None,
        scale: Scalar | array | None = None,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Generate normally distributed MLX values with the requested shape."""
        ...


class MetalRuntime(Protocol):
    @staticmethod
    def is_available() -> bool:
        """Report whether the Metal runtime is available."""
        ...


class FastRuntime(Protocol):
    @staticmethod
    def metal_kernel(
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        *,
        compile_options: Mapping[str, str] | None = None,
    ) -> MetalKernel:
        """Construct a named MLX Metal kernel with the supplied source and options."""
        ...

    @staticmethod
    def layer_norm(
        x: array,
        weight: array | None,
        bias: array | None,
        eps: float,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply last-axis layer normalization with optional scale and bias."""
        ...
    @staticmethod
    def scaled_dot_product_attention(
        q: array,
        k: array,
        v: array,
        *,
        scale: float,
        mask: str | array | None = None,
        sinks: array | None = None,
        force_fused: bool = False,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply scaled attention to query, key and value tensors."""
        ...


class CoreRuntime(Protocol):
    __version__: str

    @staticmethod
    def set_default_device(device: Device) -> None:
        """Select the process-wide default MLX device."""
        ...

    gpu: Device
    array: ArrayConstructor
    float16: Dtype
    float32: Dtype
    bfloat16: Dtype
    int32: Dtype
    uint16: Dtype
    uint32: Dtype
    random: RandomRuntime
    metal: MetalRuntime
    fast: FastRuntime

    @staticmethod
    def save_safetensors(
        file: str | Path, arrays: dict[str, array], metadata: dict[str, str] | None = None
    ) -> None:
        """Write named arrays and optional metadata to safetensors."""
        ...

    @staticmethod
    def eval(*args: object) -> None:
        """Evaluate the supplied lazy MLX values."""
        ...

    @staticmethod
    def async_eval(*args: object) -> None:
        """Start evaluating the supplied lazy MLX values without waiting."""
        ...

    @staticmethod
    def new_thread_local_stream(device: Device) -> ThreadLocalStream:
        """Create a stream on ``device`` that is distinct in every thread."""
        ...

    @staticmethod
    def stream(s: ThreadLocalStream) -> StreamContext:
        """Make ``s`` the default stream inside a ``with`` block."""
        ...

    @staticmethod
    @overload
    def load(
        file: str | Path | BinaryIO,
        /,
        format: Literal["safetensors"],
        return_metadata: Literal[False] = False,
        *,
        stream: StreamOrDevice = None,
    ) -> dict[str, array]:
        """Load arrays or array metadata from an MLX-supported file."""
        ...

    @staticmethod
    @overload
    def load(
        file: str | Path | BinaryIO,
        /,
        format: str | None = None,
        return_metadata: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array | dict[str, array] | tuple[dict[str, array], dict[str, str]]:
        """Load arrays or array metadata from an MLX-supported file."""
        ...

    @staticmethod
    def reshape(a: array, /, shape: Sequence[int], *, stream: StreamOrDevice = None) -> array:
        """Reshape an array without changing the number of elements."""
        ...

    @staticmethod
    def square(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise square of an array."""
        ...

    @staticmethod
    def full(
        shape: int | Sequence[int],
        vals: Scalar | array,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create an array filled with the supplied value."""
        ...

    @staticmethod
    def zeros(
        shape: int | Sequence[int], dtype: Dtype | None = ..., *, stream: StreamOrDevice = None
    ) -> array:
        """Create an array filled with zeros."""
        ...

    @staticmethod
    def ones(
        shape: int | Sequence[int], dtype: Dtype | None = ..., *, stream: StreamOrDevice = None
    ) -> array:
        """Create an array filled with ones."""
        ...

    @staticmethod
    def all(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reduce boolean truth across the selected axes."""
        ...

    @staticmethod
    def maximum(
        a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise maximum of the operands."""
        ...

    @staticmethod
    def isfinite(a: array, stream: StreamOrDevice = None) -> array:
        """Mark array elements that are neither infinite nor NaN."""
        ...

    @staticmethod
    def transpose(
        a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Reorder the axes of an array."""
        ...

    @staticmethod
    def mean(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Average array elements across the selected axes."""
        ...

    @staticmethod
    def argmax(
        a: array,
        /,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return indices of maximum values along an axis."""
        ...

    @staticmethod
    def concatenate(
        arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along an existing axis."""
        ...

    @staticmethod
    def stack(
        arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along a new axis."""
        ...

    @staticmethod
    def where(
        condition: Scalar | array,
        x: Scalar | array,
        y: Scalar | array,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Choose elements from two operands using a condition."""
        ...

    @staticmethod
    def sin(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise sine."""
        ...

    @staticmethod
    def cos(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise cosine."""
        ...

    @staticmethod
    def sigmoid(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise logistic sigmoid."""
        ...

    @staticmethod
    @overload
    def arange(
        start: int | float,
        stop: int | float | None,
        step: int | float | None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create a regularly spaced sequence with the requested dtype."""
        ...

    @staticmethod
    @overload
    def arange(
        stop: int | float,
        step: int | float | None = None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create a regularly spaced sequence with the requested dtype."""
        ...

    @staticmethod
    def any(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reduce boolean truth across the selected axes."""
        ...

    @staticmethod
    def sum(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Sum array elements across the selected axes."""
        ...

    @staticmethod
    def max(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reduce maximum values across the selected axes."""
        ...

    @staticmethod
    def repeat(
        array: array, repeats: int, axis: int | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Repeat elements along an axis or a flattened array."""
        ...

    @staticmethod
    def clip(
        a: array,
        /,
        a_min: Scalar | array | None,
        a_max: Scalar | array | None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Clamp elements to optional lower and upper bounds."""
        ...

    @staticmethod
    def pad(
        a: array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        constant_values: Scalar | array = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Pad array axes using the requested boundary mode."""
        ...

    @staticmethod
    def tile(a: array, reps: int | Sequence[int], /, *, stream: StreamOrDevice = None) -> array:
        """Repeat the array along its axes."""
        ...

    @staticmethod
    def sqrt(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute elementwise square roots."""
        ...

    @staticmethod
    def broadcast_to(
        a: Scalar | array, /, shape: Sequence[int], *, stream: StreamOrDevice = None
    ) -> array:
        """Broadcast an array or scalar to the requested shape."""
        ...

    @staticmethod
    def conv2d(
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply MLX 2D convolution using the supplied stride, padding and groups."""
        ...


type Tensor = array | HostArray


@dataclass(frozen=True, slots=True)
class PackedBFloat16:
    """Tagged host BF16 bits; ordinary uint16 tensors are never interpreted as BF16."""

    bits: npt.NDArray[np.uint16]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the stored memory's original shape."""
        return self.bits.shape

    def __getitem__(self, key: slice) -> PackedBFloat16:
        """Borrow one object's leading-axis memory slice without expanding its dtype."""
        return PackedBFloat16(self.bits[key])


type MemoryTensor = Tensor | PackedBFloat16
type FrameStorage = Literal["cond_frame_outputs", "non_cond_frame_outputs"]


class PointInputs(TypedDict):
    """Prompt points for the SAM heads.

    ``point_coords`` is (B, N, 2) as (x, y) pixels of the 512 x 512 model input and
    ``point_labels`` is (B, N) ints (see ``PromptEncoder``).
    """

    point_coords: array
    point_labels: array


class ImageFeatures(TypedDict):
    """Image-encoder output, NCHW: the 32 x 32 features, FPN levels, and position encodings."""

    vision_features: array
    backbone_fpn: list[array]
    vision_pos_enc: list[array]


class MemoryEncoding(TypedDict):
    vision_features: array
    vision_pos_enc: list[array]


class MemoryEntry(TypedDict):
    maskmem_features: MemoryTensor | None
    maskmem_pos_enc: Tensor | None


class FrameOutput(MemoryEntry):
    """One tracked frame's stored output for a batch of objects.

    ``pred_masks`` are (B, 1, 128, 128) logits, ``obj_ptr`` (B, 256), and
    ``object_score_logits`` (B, 1). ``maskmem_features`` (BF16 values) and
    ``maskmem_pos_enc`` are ``None`` until a memory is encoded for the frame.
    """

    pred_masks: Tensor
    obj_ptr: array
    object_score_logits: array


class TrackOutput(TypedDict):
    point_inputs: PointInputs | None
    mask_inputs: array | None
    pred_masks: array
    pred_masks_high_res: array
    obj_ptr: array
    object_score_logits: array
    maskmem_features: array | None
    maskmem_pos_enc: array | None


class ConsolidatedOutput(MemoryEntry):
    obj_ptr: array
    object_score_logits: array
    pred_masks: NotRequired[Tensor]
    pred_masks_video_res: NotRequired[Tensor]


type OutputBanks = dict[FrameStorage, dict[int, FrameOutput]]


class FrameTracking(TypedDict):
    reverse: bool


class SessionState(TypedDict):
    """Mutable inference state of one predictor session, mirroring upstream's dict.

    ``images`` are the preprocessed (3, 512, 512) frames and ``video_height`` /
    ``video_width`` the original slice size in pixels. Objects are indexed densely
    (``obj_idx``) in the order their ``obj_id`` was registered. The ``*_per_obj`` and
    ``output_dict`` entries hold prompts and per-frame outputs; ``temp_output_dict_per_obj``
    holds outputs of prompts not yet committed by propagation.
    """

    images: array | HostArray | PreparedVideoFrames
    num_frames: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    async_loading_frames: bool
    video_height: int
    video_width: int
    point_inputs_per_obj: dict[int, dict[int, PointInputs]]
    mask_inputs_per_obj: dict[int, dict[int, array]]
    cached_features: dict[int, ImageFeatures]
    constants: dict[str, Tensor]
    obj_id_to_idx: OrderedDict[int, int]
    obj_idx_to_id: OrderedDict[int, int]
    obj_ids: list[int]
    output_dict: OutputBanks
    output_dict_per_obj: dict[int, OutputBanks]
    temp_output_dict_per_obj: dict[int, OutputBanks]
    consolidated_frame_inds: dict[FrameStorage, set[int]]
    tracking_has_started: bool
    frames_already_tracked: dict[int, FrameTracking]
    keep_image_features: NotRequired[bool]
    history_cursor: NotRequired[tuple[int, bool]]


class ConnectedComponents(Protocol):
    def label(
        self, input: HostArray, structure: HostArray | None = None, output: None = None
    ) -> tuple[npt.NDArray[np.signedinteger], int]:
        """Label connected components and return their integer labels and count."""
        ...


type NumericArray = npt.NDArray[np.number]
type BoolArray = npt.NDArray[np.bool_]
type BoxCoordinates = tuple[float, float, float, float]
type Propagation = Literal["none", "forward", "reverse", "both"]


class RunnerPredictor[StateT](Protocol):
    """Predictor surface the runner uses; implemented by ``MedSAM2MLXPredictor``."""

    def init_state(
        self,
        images: HostArray | PreparedVideoFrames,
        height: int,
        width: int,
        *,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> StateT:
        """Prepare a predictor state for frames and their original dimensions."""
        ...
    def add_new_points_or_box(
        self,
        inference_state: StateT,
        frame_idx: int,
        obj_id: int,
        points: HostArray | None = None,
        labels: HostArray | None = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: HostArray | None = None,
    ) -> tuple[int, list[int], HostArray]:
        """Add a point or box prompt and return the affected frame masks."""
        ...
    def add_new_mask(
        self, inference_state: StateT, frame_idx: int, obj_id: int, mask: HostArray
    ) -> tuple[int, list[int], HostArray]:
        """Add a mask prompt and return the affected frame masks."""
        ...
    def propagate_in_video(
        self,
        inference_state: StateT,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], HostArray]]:
        """Yield frame masks in the requested propagation direction."""
        ...
    def reset_state(self, inference_state: StateT) -> None:
        """Clear the predictor prompts and tracked state."""
        ...


class NiftiProxy(Protocol):
    slope: float
    inter: float

    def get_unscaled(self) -> NumericArray:
        """Return stored voxel values before NIfTI slope and intercept."""
        ...
    def __array__(self, dtype: object = None) -> NumericArray:
        """Expose the proxy as an array, applying its normal scaling."""
        ...


class NiftiHeader(Protocol):
    def set_slope_inter(self, slope: float | None, inter: float | None = None) -> None:
        """Set the NIfTI data scaling slope and intercept."""
        ...


class NiftiImage(Protocol):
    dataobj: NiftiProxy
    affine: NumericArray
    header: NiftiHeader
    shape: tuple[int, ...]

    def set_data_dtype(self, datatype: object) -> None:
        """Set the stored NIfTI voxel dtype."""
        ...
    def get_data_dtype(self) -> np.dtype[np.generic]:
        """Return the stored NIfTI voxel dtype."""
        ...
    def set_qform(
        self, affine: HostArray | None, code: int | str | None = None, strip_shears: bool = True
    ) -> None:
        """Set the qform affine and optional code, with shear handling."""
        ...
    def set_sform(self, affine: HostArray | None, code: int | str | None = None) -> None:
        """Set the sform affine and optional code."""
        ...


class NiftiConstructor(Protocol):
    def __call__(
        self, dataobj: HostArray, affine: HostArray | None, header: NiftiHeader | None = None
    ) -> NiftiImage:
        """Construct a NIfTI image from data, affine and optional header."""
        ...


class FileBasedImages(Protocol):
    ImageFileError: type[Exception]


class NiftiRuntime(Protocol):
    Nifti1Image: NiftiConstructor
    filebasedimages: FileBasedImages

    def load(self, filename: str | Path) -> NiftiImage:
        """Load a NIfTI image from the given path."""
        ...
    def save(self, img: NiftiImage, filename: str | Path) -> None:
        """Write a NIfTI image to the given path."""
        ...


class RunnerArguments(TypedDict, total=False):
    image_path: str | Path
    weights_path: str | Path
    output_dir: str | Path
    box_xyxy_pixels: Sequence[float] | None
    points_xy_pixels: Sequence[Sequence[float]] | None
    point_labels: Sequence[int] | None
    mask_path: str | Path | None
    frame_index: int
    obj_id: int
    prompts: Sequence[Mapping[str, object]] | None
    hu_window: Sequence[float] | None
    propagation: Propagation
    slice_range: Sequence[int] | None
    stop_after_empty: int | None
    apply_postprocessing: bool
    overwrite: bool


class RunnerDetails(TypedDict):
    """JSON-serializable metadata of a runner call.

    ``height`` and ``width`` are the size in pixels of one slice of the (D, H, W) stack;
    for a NIfTI volume, ``width`` is the first voxel axis and ``height`` the second.
    ``output_shape`` is the saved label array shape: ``(width, height, num_frames)``
    for a 3D source or ``(width, height)`` for a 2D NIfTI source. ``frames_with_mask``
    lists the slice indices (along the stack's first axis) that contain a label.
    """

    weights_revision: NotRequired[str | None]
    num_frames: int
    height: int
    width: int
    mask_voxels: int
    frames_with_mask: list[int]
    object_ids: list[int]
    prompts: list[dict[str, object]]
    mask_path: str | None
    hu_window: list[float] | None
    propagation: Propagation
    slice_range: list[int] | None
    stop_after_empty: int | None
    postprocessing_applied: bool
    coordinate_space: Literal["image_voxel"]
    output_shape: list[int]
    output_dtype: Literal["uint16"]
    output_units: Literal["label_index"]
    backend: Literal["mlx"]


class RunnerResult(TypedDict):
    outputs: dict[str, str]
    artifacts: dict[str, Artifact]
    details: RunnerDetails


class VideoResolutionOutput(MemoryEntry):
    pred_masks_video_res: Tensor
    obj_ptr: array
    object_score_logits: array


class HeadArguments(TypedDict, total=False):
    point_inputs: PointInputs | None
    mask_inputs: array | None
    multimask_output: bool


class PromptedRunArguments(TypedDict):
    image_path: str | Path
    propagation: Propagation
    frame_index: NotRequired[int]
    obj_id: NotRequired[int]
    box_xyxy_pixels: NotRequired[Sequence[float] | None]
    points_xy_pixels: NotRequired[Sequence[Sequence[float]] | None]
    point_labels: NotRequired[Sequence[int] | None]
    mask_path: NotRequired[str | Path | None]
    prompts: NotRequired[Sequence[Mapping[str, object]] | None]
    hu_window: NotRequired[Sequence[float] | None]
    slice_range: NotRequired[Sequence[int] | None]
    stop_after_empty: NotRequired[int | None]
