# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source file: sam2/modeling/sam/mask_decoder.py.
"""SAM 2 mask decoder with high-resolution skips and object-score head, in MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

from medsam2_mlx._ops import (
    MLP,
    Activation,
    Conv2d,
    ConvTranspose2x2,
    Embedding,
    LayerNorm2d,
    gelu,
)
from medsam2_mlx._runtime import mx, nn
from medsam2_mlx.transformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    """SAM 2 mask decoder: four mask tokens, an IoU head and an object-score head.

    With the default graph, image embeddings are (B, 256, 32, 32); masks are decoded
    at 128 x 128 using the two high-resolution skip features from
    ``MedSAM2Model.forward_image``.
    ``dynamic_multimask_via_stability`` enables upstream's fallback: when a single-mask
    output is requested, keep mask token 0 if its logits are stable (the area above
    +delta over the area above -delta is at least the threshold) and otherwise the
    best-IoU multimask. The single-mask path still returns pointer token 0 even
    when the fallback selects a multimask.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        iou_prediction_use_sigmoid: bool = True,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        use_multimask_token_for_obj_ptr: bool = True,
    ) -> None:
        """Build mask, IoU and object-score tokens and the configured decoder heads."""
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = Embedding(self.num_mask_tokens, transformer_dim)
        self.obj_score_token = Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.output_upscaling = [
            ConvTranspose2x2(transformer_dim, transformer_dim // 4),
            LayerNorm2d(transformer_dim // 4),
            Activation(gelu),
            ConvTranspose2x2(transformer_dim // 4, transformer_dim // 8),
            Activation(gelu),
        ]
        self.conv_s0 = Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)
        self.conv_s1 = Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1)
        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def __call__(
        self,
        image_embeddings: array,
        image_pe: array,
        sparse_prompt_embeddings: array,
        dense_prompt_embeddings: array,
        multimask_output: bool,
        high_res_features: list[array],
    ) -> tuple[array, array, array, array]:
        """Predict masks from prompt embeddings.

        Shapes below describe the default graph.

        Args:
            image_embeddings: (B, 256, 32, 32) image features.
            image_pe: (1, 256, 32, 32) dense position encoding.
            sparse_prompt_embeddings: (B, N, 256) point/box tokens.
            dense_prompt_embeddings: (B, 256, 32, 32) mask (or no-mask) embedding.
            multimask_output: Return the three multimask candidates instead of one.
            high_res_features: The two skip features, (B, 32, 128, 128) then (B, 64, 64, 64).

        Returns:
            ``(masks, iou_pred, sam_tokens_out, object_score_logits)``: mask logits
            (B, 3 or 1, 128, 128), predicted IoU (B, 3 or 1), the output tokens used for
            the object pointer (B, 3 or 1, 256) and the object score (B, 1). The
            multimask token count follows ``num_multimask_outputs`` (default 3);
            pointer tokens remain singular if ``use_multimask_token_for_obj_ptr`` is false.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_features,
        )
        if multimask_output:
            masks, iou_pred = masks[:, 1:], iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks, iou_pred = masks[:, 0:1], iou_pred[:, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: array,
        image_pe: array,
        sparse_prompt_embeddings: array,
        dense_prompt_embeddings: array,
        high_res_features: list[array],
    ) -> tuple[array, array, array, array]:
        """Run the two-way transformer and upscale to mask logits, one per mask token.

        Token order is object score, IoU, then mask tokens. Returns all mask-token masks
        (B, num_mask_tokens, 128, 128), their IoU predictions, the mask-token outputs and
        the object score logits; ``forward`` selects among them.
        """
        output_tokens = mx.concatenate(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], axis=0
        )
        batch = sparse_prompt_embeddings.shape[0]
        output_tokens = mx.broadcast_to(output_tokens[None], (batch, *output_tokens.shape))
        tokens = mx.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)
        if image_embeddings.shape[0] != batch:
            raise ValueError("Image and prompt batch sizes differ")
        src = image_embeddings + dense_prompt_embeddings
        pos_src = mx.broadcast_to(image_pe, (batch, *image_pe.shape[1:]))
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : 2 + self.num_mask_tokens, :]
        src = mx.reshape(mx.transpose(src, (0, 2, 1)), (b, c, h, w))
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(src) + feat_s1))
        upscaled = act2(dc2(upscaled) + feat_s0)
        hyper_in = mx.stack(
            [mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)],
            axis=1,
        )
        b, c, h, w = upscaled.shape
        masks = mx.reshape(hyper_in @ mx.reshape(upscaled, (b, c, h * w)), (b, -1, h, w))
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: array) -> array:
        """Return area above +delta over area above -delta; 1 if the latter area is zero."""
        mask_logits = mx.reshape(mask_logits, (*mask_logits.shape[:-2], -1))
        delta = self.dynamic_multimask_stability_delta
        area_i = mx.sum(mask_logits > delta, axis=-1).astype(mx.float32)
        area_u = mx.sum(mask_logits > -delta, axis=-1).astype(mx.float32)
        return mx.where(area_u > 0, area_i / mx.maximum(area_u, 1), 1.0)

    def _dynamic_multimask_via_stability(
        self, all_mask_logits: array, all_iou_scores: array
    ) -> tuple[array, array]:
        """Pick token 0 if stable, else the best-IoU multimask token (upstream's fallback).

        Inputs hold all mask tokens (token 0 is the single-mask output); returns one mask.
        """
        multimask_logits = all_mask_logits[:, 1:]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best = mx.argmax(multimask_iou_scores, axis=-1)
        batch = mx.arange(multimask_iou_scores.shape[0])
        best_multimask_logits = multimask_logits[batch, best][:, None]
        best_multimask_iou_scores = multimask_iou_scores[batch, best][:, None]
        singlemask_logits = all_mask_logits[:, 0:1]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability = self._get_stability_scores(singlemask_logits)
        is_stable = stability >= self.dynamic_multimask_stability_thresh
        mask_logits_out = mx.where(
            is_stable[..., None, None], singlemask_logits, best_multimask_logits
        )
        iou_scores_out = mx.where(is_stable, singlemask_iou_scores, best_multimask_iou_scores)
        return mask_logits_out, iou_scores_out
