"""Consumed APIs of the pinned MedSAM2 332f30d CPU reference.

These declarations describe the admitted upstream modules, never substitute for
or import the candidate network. Constructor interfaces cover consumed keyword arguments;
Tensor and Generator refer to the installed Torch identities only for checking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from torch import Generator, Tensor, dtype

type HostArray = npt.NDArray[np.generic]
type FloatArray = npt.NDArray[np.float32]
type NumericInput = int | float | bool | HostArray | Sequence[NumericInput]


class FunctionalRuntime(Protocol):
    """Subset of ``torch.nn.functional`` used by the reference tests."""

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


class SourcePredictor(ModuleLifecycle, Protocol):
    """Upstream video predictor whose seeded state dict the parity tests load."""

    sam_mask_decoder: ModuleLifecycle


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
        sam_mask_decoder_extra_args: Mapping[str, float | bool] | None = None,
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
