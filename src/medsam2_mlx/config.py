# Values transcribed from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d)
# sam2/configs/sam2.1_hiera_t512.yaml and the SAM2Base/Hiera constructor defaults.
# Licensed under the Apache License, Version 2.0.
"""Network configuration for the MedSAM2 (SAM 2.1 Hiera-T, 512 px) graph."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HieraConfig:
    """Hiera-T trunk: stage depths (1, 2, 7, 2), channels doubling per stage from 96.

    ``window_spec`` gives the attention window size per stage, except in the global
    attention blocks listed in ``global_att_blocks``.
    """

    embed_dim: int = 96
    num_heads: int = 1
    stages: tuple[int, ...] = (1, 2, 7, 2)
    global_att_blocks: tuple[int, ...] = (5, 7, 9)
    window_pos_embed_bkg_spatial_size: tuple[int, int] = (7, 7)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)
    q_pool: int = 3
    dim_mul: float = 2.0
    head_mul: float = 2.0


@dataclass(frozen=True)
class ImageEncoderConfig:
    """FPN neck; ``scalp`` drops that many deepest levels after fusion."""

    hiera: HieraConfig = field(default_factory=HieraConfig)
    d_model: int = 256
    backbone_channel_list: tuple[int, ...] = (768, 384, 192, 96)
    fpn_top_down_levels: tuple[int, ...] = (2, 3)
    scalp: int = 1


@dataclass(frozen=True)
class MemoryAttentionConfig:
    """Memory attention layers.

    ``feat_sizes`` mirrors the upstream YAML but is not read: the rotary position grid
    is taken from the length of the query sequence.
    """

    num_layers: int = 4
    dim_feedforward: int = 2048
    rope_theta: float = 10000.0
    feat_sizes: tuple[int, int] = (32, 32)


@dataclass(frozen=True)
class MemoryEncoderConfig:
    """Memory encoder: a mask downsampler with total stride 16, then a ConvNeXt fuser."""

    mask_kernel_size: int = 3
    mask_stride: int = 2
    mask_padding: int = 1
    mask_total_stride: int = 16
    fuser_kernel_size: int = 7
    fuser_padding: int = 3
    fuser_num_layers: int = 2


@dataclass(frozen=True)
class Sam2Config:
    """SAM2Base arguments of ``sam2.1_hiera_t512.yaml`` (eval mode).

    ``image_size`` is the square model input in pixels and ``backbone_stride`` the
    pixel stride of the top feature map, so the image embedding is 32 x 32.
    ``mem_dim`` is the memory channel count. The memory bank includes conditioning
    frames and up to ``num_maskmem - 1`` previous tracked frames. Mask logits are
    sigmoid-transformed, or binarized for click masks when that option is enabled,
    then mapped through ``mask * sigmoid_scale_for_mem_enc + sigmoid_bias_for_mem_enc``.
    ``max_obj_ptrs_in_encoder`` limits the tracked-frame pointer lookback; eligible
    conditioning-frame pointers are included separately.
    """

    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    memory_attention: MemoryAttentionConfig = field(default_factory=MemoryAttentionConfig)
    memory_encoder: MemoryEncoderConfig = field(default_factory=MemoryEncoderConfig)
    mem_dim: int = 64
    num_maskmem: int = 7
    image_size: int = 512
    backbone_stride: int = 16
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    use_mask_input_as_output_without_sam: bool = True
    directly_add_no_mem_embed: bool = True
    multimask_output_in_sam: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    multimask_output_for_tracking: bool = True
    use_multimask_token_for_obj_ptr: bool = True
    iou_prediction_use_sigmoid: bool = True
    max_obj_ptrs_in_encoder: int = 16
    # SAM head sizes hard-coded in SAM2Base._build_sam_heads.
    mask_in_chans: int = 16
    decoder_num_heads: int = 8
    decoder_mlp_dim: int = 2048
    iou_head_hidden_dim: int = 256
    # MaskDecoder defaults for dynamic multimask fallback.
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98

    @property
    def hidden_dim(self) -> int:
        """Channel count of the SAM heads and memory attention (the FPN ``d_model``)."""
        return self.image_encoder.d_model


SAM2_1_HIERA_T512 = Sam2Config()
"""The configuration of every MedSAM2 checkpoint built on ``sam2.1_hiera_t512.yaml``."""
