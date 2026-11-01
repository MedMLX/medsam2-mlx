"""Consumed APIs of the pinned MedSAM2 332f30d CPU reference.

These declarations describe the admitted upstream modules, never substitute for
or import the candidate network. Constructor interfaces cover consumed keyword arguments;
Tensor and Generator refer to the installed Torch identities only for checking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, Self, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from torch import Generator, Tensor, dtype

type HostArray = npt.NDArray[np.generic]
type NumericArray = npt.NDArray[np.number]
type FloatArray = npt.NDArray[np.float32]
type NumericInput = int | float | bool | HostArray | Sequence[NumericInput]
type SourceOutput = tuple[int, list[int], Tensor]
type Storage = Literal["cond_frame_outputs", "non_cond_frame_outputs"]


class SourceImageFeatures(TypedDict):
    """Image-encoder output consumed by the upstream predictor.

    ``vision_features`` is the lowest-resolution fused level; ``backbone_fpn``
    and ``vision_pos_enc`` list the FPN levels from highest to lowest resolution.
    """

    vision_features: Tensor
    backbone_fpn: list[Tensor]
    vision_pos_enc: list[Tensor]


class SourcePointInputs(TypedDict):
    """Point prompts passed to the upstream SAM heads.

    ``point_coords`` is (B, N, 2) in model-input pixels as (x, y);
    ``point_labels`` is (B, N) with foreground, background, box-corner and
    padding label values.
    """

    point_coords: Tensor
    point_labels: Tensor


class SourceFrameOutput(TypedDict):
    """Stored per-object output for one tracked frame.

    ``pred_masks`` are low-resolution logits; the memory entries stay ``None``
    until the frame's memory is encoded.
    """

    maskmem_features: Tensor | None
    maskmem_pos_enc: list[Tensor] | None
    pred_masks: Tensor
    obj_ptr: Tensor
    object_score_logits: Tensor


class SourceState(TypedDict):
    """Upstream predictor session state for prompt and propagation calls.

    ``output_dict`` holds per-object frame outputs under conditioning and
    non-conditioning frame keys.
    """

    obj_ids: list[int]
    tracking_has_started: bool
    output_dict: dict[Storage, dict[int, SourceFrameOutput]]


class DecoderExtras(TypedDict, total=False):
    """Optional mask-decoder settings for dynamic multimask selection."""

    dynamic_multimask_via_stability: bool
    dynamic_multimask_stability_delta: float
    dynamic_multimask_stability_thresh: float


class PointArguments(TypedDict, total=False):
    """Keyword arguments for the upstream point-or-box prompt call.

    Coordinates are original-frame pixels unless ``normalize_coords`` is false,
    which takes them in [0, 1]; a box is ``(x0, y0, x1, y1)``.
    """

    points: HostArray | None
    labels: HostArray | None
    box: HostArray | None
    normalize_coords: bool
    clear_old_points: bool


class TrackingArguments(TypedDict, total=False):
    """Keyword arguments for the upstream propagation call."""

    start_frame_idx: int | None
    max_frame_num_to_track: int | None
    reverse: bool


class FunctionalRuntime(Protocol):
    """Subset of ``torch.nn.functional`` used by the reference recorders."""

    def interpolate(
        self,
        input: Tensor,
        size: int | Sequence[int] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> Tensor:
        """Resize a Torch tensor using the selected interpolation options."""
        ...


class TorchNN(Protocol):
    """Torch ``nn`` namespace subset used by the reference recorders."""

    functional: FunctionalRuntime


class TorchRuntime(Protocol):
    """Torch module surface used to build and run the CPU reference."""

    __version__: str
    float16: dtype
    float32: dtype
    bfloat16: dtype
    int32: dtype
    nn: TorchNN
    Generator: Callable[[], Generator]

    def manual_seed(self, seed: int) -> Generator:
        """Set the Torch random seed and return its generator."""
        ...
    def from_numpy(self, nd_array: HostArray) -> Tensor:
        """Create a Torch tensor from NumPy data."""
        ...
    def tensor(self, data: NumericInput, *, dtype: dtype | None = None) -> Tensor:
        """Create a Torch tensor from numeric input."""
        ...
    def randn(self, size: Sequence[int], *, generator: Generator | None = None) -> Tensor:
        """Create normally distributed Torch values using an optional generator."""
        ...
    def full(self, size: Sequence[int], fill_value: int | float) -> Tensor:
        """Create a Torch tensor filled with the requested scalar."""
        ...
    def set_num_threads(self, num_threads: int) -> None:
        """Set the Torch CPU thread count for the reference run."""
        ...
    def inference_mode(self, mode: bool = True) -> AbstractContextManager[None]:
        """Enter or disable Torch inference mode."""
        ...
    def load(
        self, f: str | Path, map_location: str | None = None, *, weights_only: bool | None = None
    ) -> object:
        """Load a Torch checkpoint with the requested device and weights policy."""
        ...


class ModuleLifecycle(Protocol):
    """Torch module lifecycle shared by upstream reference components."""

    def state_dict(self) -> Mapping[str, Tensor]:
        """Return the module state tensors by name."""
        ...
    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> object:
        """Load module state tensors with optional strict key matching."""
        ...
    def eval(self) -> Self:
        """Set the source module to evaluation mode."""
        ...


class SourceMemoryAttention(ModuleLifecycle, Protocol):
    """Upstream memory-attention module consumed by the reference predictor."""

    def __call__(
        self,
        curr: Tensor | list[Tensor],
        memory: Tensor,
        curr_pos: Tensor | list[Tensor],
        memory_pos: Tensor,
        num_obj_ptr_tokens: int = 0,
    ) -> Tensor:
        """Condition current frame features on spatial and object memory."""
        ...


class SourceRoPE(ModuleLifecycle, Protocol):
    """Upstream rotary-attention module used by the memory and transformer stacks."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        """Apply source rotary attention while excluding trailing pointer keys."""
        ...


class SourceBackbonePreparation(Protocol):
    """Callable upstream backbone-preparation method returning flattened features and sizes."""

    def __call__(
        self, backbone_out: SourceImageFeatures
    ) -> tuple[SourceImageFeatures, list[Tensor], list[Tensor], list[tuple[int, int]]]:
        """Prepare image backbone features and their positional encodings."""
        ...


class SourceSamHeads(Protocol):
    """Callable upstream SAM-head method returning mask, pointer and score tensors."""

    def __call__(
        self,
        backbone_features: Tensor,
        point_inputs: SourcePointInputs | None = None,
        mask_inputs: Tensor | None = None,
        high_res_features: list[Tensor] | None = None,
        multimask_output: bool = False,
    ) -> tuple[Tensor, ...]:
        """Run source SAM heads for point or mask prompts."""
        ...


class SourceMaskOutput(Protocol):
    """Callable upstream mask-as-output method returning the SAM-head tuple."""

    def __call__(
        self, backbone_features: Tensor, high_res_features: list[Tensor] | None, mask_inputs: Tensor
    ) -> tuple[Tensor, ...]:
        """Produce the source mask-input output tuple."""
        ...


class SourceMemoryEncoding(Protocol):
    """Callable upstream memory-encoding method for one frame."""

    def __call__(
        self,
        current_vision_feats: list[Tensor],
        feat_sizes: list[tuple[int, int]],
        pred_masks_high_res: Tensor,
        object_score_logits: Tensor,
        is_mask_from_pts: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode source frame features and masks into memory."""
        ...


class SourcePredictor(ModuleLifecycle, Protocol):
    """Upstream video predictor session API used to record reference outputs."""

    sam_mask_decoder: ModuleLifecycle
    memory_attention: SourceMemoryAttention
    binarize_mask_from_pts_for_mem_enc: bool

    def forward_image(self, img_batch: Tensor) -> SourceImageFeatures:
        """Compute source image encoder features for a batch."""
        ...
    def init_state(
        self,
        images: Tensor,
        video_height: int,
        video_width: int,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> SourceState:
        """Initialize a source predictor session for a frame stack."""
        ...
    def add_new_points_or_box(
        self,
        inference_state: SourceState,
        frame_idx: int,
        obj_id: int,
        points: HostArray | None = None,
        labels: HostArray | None = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: HostArray | None = None,
    ) -> SourceOutput:
        """Add a source point or box prompt and return frame masks."""
        ...
    def add_new_mask(
        self, inference_state: SourceState, frame_idx: int, obj_id: int, mask: HostArray
    ) -> SourceOutput:
        """Add a source mask prompt and return frame masks."""
        ...
    def propagate_in_video(
        self,
        inference_state: SourceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[SourceOutput]:
        """Yield source masks across a sequence of tracked frames."""
        ...
    def clear_all_prompts_in_frame(
        self,
        inference_state: SourceState,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[True] = True,
    ) -> SourceOutput:
        """Clear a source object frame prompt and return its preview."""
        ...
    def reset_state(self, inference_state: SourceState) -> None:
        """Reset the source predictor session."""
        ...


class SourceBuilder(Protocol):
    """Build an upstream predictor from its configuration and optional checkpoint."""

    def __call__(
        self,
        config_file: str,
        ckpt_path: str | None = None,
        *,
        device: str = "cuda",
        mode: str = "eval",
        apply_postprocessing: bool = True,
    ) -> SourcePredictor:
        """Build an upstream predictor from its configuration and optional checkpoint."""
        ...


class ResizeGrayscale(Protocol):
    """Callable upstream CT helper that resizes a grayscale stack to RGB frames."""

    def __call__(self, array: HostArray, image_size: int) -> NumericArray:
        """Resize a grayscale source image to the model image size."""
        ...


class HieraFactory(Protocol):
    """Callable factory for the upstream Hiera image backbone."""

    def __call__(
        self,
        *,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: tuple[int, ...] = (2, 3, 16, 3),
        global_att_blocks: tuple[int, ...] = (12, 16, 20),
        window_spec: tuple[int, ...] = (8, 4, 14, 7),
        window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14),
    ) -> ModuleLifecycle:
        """Construct the upstream Hiera image backbone."""
        ...


class SineFactory(Protocol):
    """Callable factory for the upstream sine positional encoding."""

    def __call__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ) -> ModuleLifecycle:
        """Construct the upstream sine positional encoding."""
        ...


class NeckFactory(Protocol):
    """Callable factory for the upstream feature-pyramid neck."""

    def __call__(
        self,
        *,
        position_encoding: ModuleLifecycle,
        d_model: int,
        backbone_channel_list: list[int],
        fpn_top_down_levels: list[int] | None = None,
        fpn_interp_model: str = "bilinear",
    ) -> ModuleLifecycle:
        """Construct the upstream feature pyramid neck."""
        ...


class EncoderFactory(Protocol):
    """Callable factory for the upstream image encoder."""

    def __call__(
        self, *, trunk: ModuleLifecycle, neck: ModuleLifecycle, scalp: int = 0
    ) -> ModuleLifecycle:
        """Construct the upstream image encoder."""
        ...


class RoPEFactory(Protocol):
    """Callable factory for upstream rotary attention."""

    def __call__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: Sequence[int] = (32, 32),
    ) -> SourceRoPE:
        """Construct upstream rotary attention."""
        ...


class TwoWayFactory(Protocol):
    """Callable factory for the upstream two-way transformer."""

    def __call__(
        self, *, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int
    ) -> ModuleLifecycle:
        """Construct the upstream two-way SAM transformer."""
        ...


class MemoryLayerFactory(Protocol):
    """Callable factory for one upstream memory-attention layer."""

    def __call__(
        self,
        *,
        activation: str,
        cross_attention: ModuleLifecycle,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: ModuleLifecycle,
    ) -> ModuleLifecycle:
        """Construct one upstream memory-attention layer."""
        ...


class MemoryAttentionFactory(Protocol):
    """Callable factory for the upstream memory-attention stack."""

    def __call__(
        self,
        *,
        d_model: int,
        pos_enc_at_input: bool,
        layer: ModuleLifecycle,
        num_layers: int,
        batch_first: bool = True,
    ) -> SourceMemoryAttention:
        """Construct the upstream memory-attention stack."""
        ...


class DownsamplerFactory(Protocol):
    """Callable factory for the upstream mask downsampler."""

    def __call__(
        self,
        *,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
    ) -> ModuleLifecycle:
        """Construct the upstream mask downsampler."""
        ...


class CXFactory(Protocol):
    """Callable factory for an upstream ConvNeXt fusion block."""

    def __call__(
        self,
        *,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ) -> ModuleLifecycle:
        """Construct an upstream ConvNeXt fusion block."""
        ...


class FuserFactory(Protocol):
    """Callable factory for the upstream memory fuser stack."""

    def __call__(
        self,
        *,
        layer: ModuleLifecycle,
        num_layers: int,
        dim: int | None = None,
        input_projection: bool = False,
    ) -> ModuleLifecycle:
        """Construct the upstream memory fusion stack."""
        ...


class MemoryEncoderFactory(Protocol):
    """Callable factory for the upstream mask and image memory encoder."""

    def __call__(
        self,
        *,
        out_dim: int,
        mask_downsampler: ModuleLifecycle,
        fuser: ModuleLifecycle,
        position_encoding: ModuleLifecycle,
        in_dim: int = 256,
    ) -> ModuleLifecycle:
        """Construct the upstream mask and image memory encoder."""
        ...


class MaskDecoderFactory(Protocol):
    """Callable factory for the upstream prompt-conditioned mask decoder."""

    def __call__(
        self,
        *,
        transformer_dim: int,
        transformer: ModuleLifecycle,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> ModuleLifecycle:
        """Construct the upstream prompt-conditioned mask decoder."""
        ...


class PredictorFactory(Protocol):
    """Callable factory for the upstream video predictor and its model components."""

    def __call__(
        self,
        *,
        image_encoder: ModuleLifecycle,
        memory_attention: ModuleLifecycle,
        memory_encoder: ModuleLifecycle,
        fill_hole_area: int = 0,
        num_maskmem: int = 7,
        image_size: int = 512,
        sigmoid_scale_for_mem_enc: float = 1.0,
        sigmoid_bias_for_mem_enc: float = 0.0,
        binarize_mask_from_pts_for_mem_enc: bool = False,
        use_mask_input_as_output_without_sam: bool = False,
        directly_add_no_mem_embed: bool = False,
        no_obj_embed_spatial: bool = False,
        use_high_res_features_in_sam: bool = False,
        multimask_output_in_sam: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        use_obj_ptrs_in_encoder: bool = False,
        add_tpos_enc_to_obj_ptrs: bool = True,
        proj_tpos_enc_in_obj_ptrs: bool = False,
        use_signed_tpos_enc_to_obj_ptrs: bool = False,
        only_obj_ptrs_in_the_past_for_eval: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        multimask_output_for_tracking: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        use_mlp_for_obj_ptr_proj: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        sam_mask_decoder_extra_args: DecoderExtras | None = None,
    ) -> SourcePredictor:
        """Construct the upstream video predictor with its model components."""
        ...


class FileModule(Protocol):
    """Imported upstream module with the source path used to verify the checkout."""

    __file__: str


class HieraModule(FileModule, Protocol):
    """Imported upstream Hiera module exposing its backbone factory."""

    Hiera: HieraFactory


class EncoderModule(FileModule, Protocol):
    """Imported upstream image-encoder module namespace."""

    FpnNeck: NeckFactory
    ImageEncoder: EncoderFactory


class PositionModule(FileModule, Protocol):
    """Imported upstream positional-encoding module namespace."""

    PositionEmbeddingSine: SineFactory


class MemoryAttentionModule(FileModule, Protocol):
    """Imported upstream memory-attention module namespace."""

    MemoryAttention: MemoryAttentionFactory
    MemoryAttentionLayer: MemoryLayerFactory


class MemoryEncoderModule(FileModule, Protocol):
    """Imported upstream memory-encoder module namespace."""

    MemoryEncoder: MemoryEncoderFactory
    MaskDownSampler: DownsamplerFactory
    Fuser: FuserFactory
    CXBlock: CXFactory


class TransformerModule(FileModule, Protocol):
    """Imported upstream transformer module namespace."""

    RoPEAttention: RoPEFactory
    TwoWayTransformer: TwoWayFactory


class DecoderModule(FileModule, Protocol):
    """Imported upstream mask-decoder module namespace."""

    MaskDecoder: MaskDecoderFactory


class PredictorModule(FileModule, Protocol):
    """Imported upstream video-predictor module namespace."""

    SAM2VideoPredictorNPZ: PredictorFactory


class Components(Protocol):
    """Callable connected-components kernel returning labels and areas for a mask tensor."""

    def __call__(self, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return connected component labels and areas for a mask tensor."""
        ...


class MiscModule(FileModule, Protocol):
    """Imported upstream miscellaneous module exposing the connected-components kernel."""

    get_connected_components: Components


class SourceModules(TypedDict):
    """Imported upstream modules keyed by the roles consumed by the recorders."""

    hieradet: HieraModule
    image_encoder: EncoderModule
    position_encoding: PositionModule
    memory_attention: MemoryAttentionModule
    memory_encoder: MemoryEncoderModule
    transformer: TransformerModule
    mask_decoder: DecoderModule
    predictor: PredictorModule
    misc: MiscModule


class SourceHeadArguments(TypedDict, total=False):
    """Optional keyword arguments for the upstream SAM-head call."""

    point_inputs: SourcePointInputs | None
    mask_inputs: Tensor | None
    multimask_output: bool


class PromptPredictor[StateT, MaskT](Protocol):
    """Prompt API shared by the upstream predictor and fixture adapters."""

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
    ) -> tuple[int, list[int], MaskT]:
        """Add a point or box prompt and return frame outputs."""
        ...
    def add_new_mask(
        self, inference_state: StateT, frame_idx: int, obj_id: int, mask: HostArray
    ) -> tuple[int, list[int], MaskT]:
        """Add a mask prompt and return frame outputs."""
        ...
    def clear_all_prompts_in_frame(
        self,
        inference_state: StateT,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[True] = True,
    ) -> tuple[int, list[int], MaskT]:
        """Clear a prompt and return the updated frame output."""
        ...


class PromptOperation(Protocol):
    """Prompt operation applied to a predictor session and its state."""

    def __call__[StateT, MaskT](
        self, predictor: PromptPredictor[StateT, MaskT], state: StateT, /
    ) -> tuple[int, list[int], MaskT]:
        """Apply one prompt operation to the supplied predictor session."""
        ...


class MetricRow(TypedDict):
    """One comparison's shape, finiteness and numerical error metrics.

    Mask comparisons add IoU, foreground count, mask agreement and sentinel
    error; other comparisons add relative L2 error.
    """

    shape: list[int]
    finite: bool
    max_abs_error: float
    relative_error: float
    sentinel_max_abs_error: NotRequired[float]
    iou: NotRequired[float]
    foreground_pixels: NotRequired[int]
    mask_agreement: NotRequired[float]
    relative_l2_error: NotRequired[float]


class WorkMetric(TypedDict, total=False):
    """Mutable metric fields assembled by ``metrics`` before target evaluation."""

    shape: list[int]
    finite: bool
    max_abs_error: float
    relative_error: float
    sentinel_max_abs_error: float
    iou: float
    foreground_pixels: int
    mask_agreement: float
    relative_l2_error: float


class RecordRow(MetricRow):
    """Qualification row with its comparison identity and target result."""

    category: str
    name: str
    target_met: bool


class MaskRecordRow(TypedDict):
    """Mask metric fields read from a qualification row."""

    iou: float
    mask_agreement: float
    sentinel_max_abs_error: float


class FeatureRecordRow(TypedDict):
    """Non-mask metric field read from a qualification row."""

    relative_l2_error: float


class MemoryStep(TypedDict):
    """Fraction of stored spatial-memory values that differ at one frame."""

    case: str
    frame: int
    differing_fraction: float


class SummaryRow(TypedDict):
    """Aggregated metrics and acceptance result for one comparison category."""

    comparisons: int
    max_relative_error: float
    max_abs_error: float
    target_met: bool
    min_iou: NotRequired[float]
    min_mask_agreement: NotRequired[float]
    max_relative_l2_error: NotRequired[float]


# NumPy 2.5 reserves allow_pickle while other named NPZ entries are arrays.
if TYPE_CHECKING:
    from typing_extensions import TypedDict as OpenTypedDict

    class NPZPayload(OpenTypedDict, extra_items=HostArray):
        """NPZ archive entries as host arrays plus NumPy's reserved pickle flag."""

        allow_pickle: NotRequired[bool]


class QualificationReceipt(TypedDict):
    """Versioned JSON receipt for one checkpoint qualification run."""

    created: str
    package_commit: str
    package_worktree_dirty: bool
    upstream: dict[str, str]
    checkpoint: dict[str, str | int]
    reference: dict[str, str]
    mlx: dict[str, str | dict[str, str]]
    input: dict[str, str | list[int]]
    targets: dict[str, float | str | dict[str, float]]
    summary: dict[str, SummaryRow]
    diagnostic_bf16_aligned: SummaryRow
    bf16_memory_steps: dict[str, float | list[MemoryStep]]
    rows: list[RecordRow]
    seconds: float
    limits: list[str]
    passed: NotRequired[bool]
