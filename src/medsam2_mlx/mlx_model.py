"""MedSAM2 512-pixel MLX graph and its memory helpers.

The network is ported from MedSAM2/SAM 2 (Apache-2.0, bowang-lab/MedSAM2 @ 332f30d);
see ``medsam2_mlx.sam2_base``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from medsam2_mlx._types import ArrayInput, ImageFeatures, MemoryEntry, OutputBanks

if TYPE_CHECKING:
    from mlx.core import array

from medsam2_mlx._ops import interpolate_bilinear
from medsam2_mlx._runtime import mx as _mx
from medsam2_mlx.config import SAM2_1_HIERA_T512, Sam2Config
from medsam2_mlx.sam2_base import MedSAM2Model, flatten_hw

__all__ = ["MedSAM2Model", "condition_memory", "create_model", "encode_memory", "resize_mask"]


def create_model(
    *,
    apply_postprocessing: bool = True,
    config: Sam2Config = SAM2_1_HIERA_T512,
) -> MedSAM2Model:
    """Build the randomly initialized FP32 graph.

    ``apply_postprocessing`` selects the NPZ builder's postprocessing model options:
    dynamic multimask fallback and binarized click masks in memory.
    """
    return MedSAM2Model(
        config,
        binarize_mask_from_pts_for_mem_enc=apply_postprocessing,
        dynamic_multimask_via_stability=apply_postprocessing,
    )


def resize_mask(mask: ArrayInput, size: tuple[int, int], *, antialias: bool = False) -> array:
    """Resize NCHW ``mask`` to ``size`` (height, width) like ``F.interpolate``.

    Bilinear with ``align_corners=False``; ``antialias=True`` widens the filter when
    downsampling, as ``antialias=True`` does in PyTorch.
    """
    return interpolate_bilinear(
        _mx_array(mask), cast(tuple[int, int], tuple(size)), antialias=antialias
    )


def condition_memory(
    model: MedSAM2Model,
    encoded: ImageFeatures,
    outputs: OutputBanks,
    *,
    frame_idx: int,
    num_frames: int,
    reverse: bool,
) -> array:
    """Condition a tracked frame's features on the memory bank (SAM2Base, eval mode).

    ``encoded`` holds ``vision_features`` (B, C, H, W) and ``vision_pos_enc``;
    ``outputs`` holds ``cond_frame_outputs`` and ``non_cond_frame_outputs`` with
    ``maskmem_features``, ``maskmem_pos_enc``, and ``obj_ptr`` per frame.
    ``frame_idx`` is the current frame, ``num_frames`` the video length, and
    ``reverse`` selects tracking toward lower frame indices. Returns the memory-conditioned
    features, (B, C, H, W).
    """
    features = _mx_array(encoded["vision_features"])
    position = _mx_array(encoded["vision_pos_enc"][-1])
    return model.prepare_memory_conditioned_features(
        frame_idx,
        False,
        flatten_hw(features),
        flatten_hw(position),
        (features.shape[-2], features.shape[-1]),
        outputs,
        num_frames,
        reverse,
    )


def encode_memory(
    model: MedSAM2Model,
    encoded: ImageFeatures,
    low: ArrayInput,
    scores: ArrayInput,
    *,
    binarize: bool,
) -> MemoryEntry:
    """Encode low-resolution mask logits into BF16 spatial memory, as upstream stores it.

    ``low`` are mask logits (B, 1, h, w), upsampled bilinearly to the model input size
    before encoding. ``scores`` are object-score logits (B, 1). With ``binarize`` the
    mask is thresholded at logit 0 instead of passed through a sigmoid. The features
    are cast to BF16 and the position encoding is returned unchanged.
    """
    high = interpolate_bilinear(
        _mx_array(low).astype(_mx.float32), (model.image_size, model.image_size)
    )
    features, position = model.encode_mask_memory(
        _mx_array(encoded["vision_features"]), high, _mx_array(scores), binarize=binarize
    )
    return {"maskmem_features": features.astype(_mx.bfloat16), "maskmem_pos_enc": position}


def _mx_array(value: ArrayInput) -> array:
    """Reuse an MLX array or convert the input values to one."""
    return value if isinstance(value, cast("type[array]", _mx.array)) else _mx.array(value)
