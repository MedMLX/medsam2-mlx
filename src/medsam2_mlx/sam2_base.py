# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/modeling/sam2_base.py, sam2/modeling/sam2_utils.py.
"""SAM2Base in eval mode for ``sam2.1_hiera_t512``: SAM heads, memory, and pointers.

The options that the MedSAM2 configuration fixes are hard-wired here: high-resolution
decoder features, object-score prediction with a fixed no-object pointer, object
pointers with projected signed temporal encodings (past pointers only), a spatial
no-object embedding, memory stride 1, and no limit on conditioning frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

from typing import cast

from medsam2_mlx._ops import MLP, Conv2d, get_1d_sine_pe, interpolate_bilinear
from medsam2_mlx._runtime import mx, nn
from medsam2_mlx._types import (
    FrameOutput,
    ImageFeatures,
    OutputBanks,
    PointInputs,
    Tensor,
    TrackOutput,
)
from medsam2_mlx.config import SAM2_1_HIERA_T512, Sam2Config
from medsam2_mlx.image_encoder import ImageEncoder
from medsam2_mlx.mask_decoder import MaskDecoder
from medsam2_mlx.memory_attention import MemoryAttention
from medsam2_mlx.memory_encoder import MemoryEncoder
from medsam2_mlx.prompt_encoder import PromptEncoder
from medsam2_mlx.transformer import TwoWayTransformer

# Mask logit given to an object the model scores as absent (upstream's NO_OBJ_SCORE).
NO_OBJ_SCORE = -1024.0


def flatten_hw(x: array) -> array:
    """(B, C, H, W) to sequence-first (H * W, B, C)."""
    b, c, h, w = x.shape
    return mx.transpose(mx.reshape(x, (b, c, h * w)), (2, 0, 1))


def unflatten_hw(x: array, size: tuple[int, int]) -> array:
    """Sequence-first (H * W, B, C) to (B, C, H, W)."""
    _, b, c = x.shape
    return mx.reshape(mx.transpose(x, (1, 2, 0)), (b, c, *size))


class MedSAM2Model(nn.Module):
    """MLX MedSAM2 network. Parameter names match the MedSAM2 PyTorch state dict.

    Spatial tensors use PyTorch's NCHW layout; token sequences are (length, batch, C)
    where noted. The model input is a 512 x 512 ImageNet-normalized RGB frame. Mask
    logits are produced at 128 x 128 (``image_size // 4``) and, where needed, bilinearly
    upsampled to 512 x 512. A mask is foreground where its logit is > 0.

    ``binarize_mask_from_pts_for_mem_enc`` and ``dynamic_multimask_via_stability`` are the
    two upstream postprocessing options (see ``mlx_model.create_model``); both default
    to off here.
    """

    def __init__(
        self,
        config: Sam2Config = SAM2_1_HIERA_T512,
        *,
        binarize_mask_from_pts_for_mem_enc: bool = False,
        dynamic_multimask_via_stability: bool = False,
    ) -> None:
        """Build the configured image, memory and prompt-driven segmentation graph."""
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        mem_dim = config.mem_dim
        self.hidden_dim = hidden
        self.mem_dim = mem_dim
        self.image_size = config.image_size
        self.num_maskmem = config.num_maskmem
        self.max_obj_ptrs_in_encoder = config.max_obj_ptrs_in_encoder
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.image_encoder = ImageEncoder(config.image_encoder)
        self.mask_downsample = Conv2d(1, 1, kernel_size=4, stride=4)
        self.memory_attention = MemoryAttention(
            hidden,
            mem_dim,
            config.memory_attention.num_layers,
            config.memory_attention.dim_feedforward,
            config.memory_attention.rope_theta,
        )
        self.memory_encoder = MemoryEncoder(hidden, mem_dim, config.memory_encoder)
        self.maskmem_tpos_enc = mx.zeros((config.num_maskmem, 1, 1, mem_dim))
        self.no_mem_embed = mx.zeros((1, 1, hidden))
        self.no_mem_pos_enc = mx.zeros((1, 1, hidden))
        self.no_obj_ptr = mx.zeros((1, hidden))
        self.no_obj_embed_spatial = mx.zeros((1, mem_dim))
        embedding = config.image_size // config.backbone_stride
        self.sam_image_embedding_size = embedding
        self.sam_prompt_encoder = PromptEncoder(
            hidden,
            (embedding, embedding),
            (config.image_size, config.image_size),
            config.mask_in_chans,
        )
        self.sam_mask_decoder = MaskDecoder(
            transformer_dim=hidden,
            transformer=TwoWayTransformer(
                2, hidden, config.decoder_num_heads, config.decoder_mlp_dim
            ),
            iou_head_hidden_dim=config.iou_head_hidden_dim,
            iou_prediction_use_sigmoid=config.iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=config.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=config.dynamic_multimask_stability_thresh,
            use_multimask_token_for_obj_ptr=config.use_multimask_token_for_obj_ptr,
        )
        self.obj_ptr_proj = MLP(hidden, hidden, hidden, 3)
        self.obj_ptr_tpos_proj = nn.Linear(hidden, mem_dim)

    # --- image features -------------------------------------------------------------

    def forward_image(self, img_batch: array) -> ImageFeatures:
        """Encode (B, 3, 512, 512) frames; precompute the decoder's high-resolution skips.

        ``backbone_fpn`` holds three levels: the 128 x 128 level projected to 32 channels
        and the 64 x 64 level projected to 64 channels (the mask decoder's skip inputs),
        and the 32 x 32, 256-channel level that is also ``vision_features``.
        """
        out = self.image_encoder(img_batch)
        fpn = list(out["backbone_fpn"])
        fpn[0] = self.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.sam_mask_decoder.conv_s1(fpn[1])
        out["backbone_fpn"] = fpn
        return out

    # --- SAM heads ------------------------------------------------------------------

    def forward_sam_heads(
        self,
        backbone_features: array,
        point_inputs: PointInputs | None = None,
        mask_inputs: array | None = None,
        high_res_features: list[array] | None = None,
        multimask_output: bool = False,
    ) -> tuple[array, ...]:
        """Run the prompt encoder and mask decoder on (B, 256, 32, 32) image features.

        ``point_inputs`` hold ``point_coords`` (B, N, 2) in pixels of the 512 x 512 model
        input, (x, y), and ``point_labels`` (B, N) with 1 foreground, 0 background,
        2/3 box corners and -1 padding. ``mask_inputs`` is a dense mask prompt of logits
        (B, 1, h, w), resized to 128 x 128 if needed. Without points, a placeholder
        with label -1 is supplied; the prompt encoder appends another padding point.

        Returns:
            ``(low_res_multimasks, high_res_multimasks, ious, low_res_masks,
            high_res_masks, obj_ptr, object_score_logits)``. The multimasks are
            (B, K, 128, 128) and (B, K, 512, 512) logits with K = 3 when
            ``multimask_output`` else 1; ``ious`` is (B, K). With ``multimask_output`` the
            single best mask by predicted IoU is selected into ``low_res_masks``
            (B, 1, 128, 128) and ``high_res_masks`` (B, 1, 512, 512). ``obj_ptr`` is
            (B, 256) and ``object_score_logits`` (B, 1). Where the object score is <= 0
            the mask logits are set to ``NO_OBJ_SCORE`` and the pointer is the learned
            no-object pointer.
        """
        batch = backbone_features.shape[0]
        if point_inputs is not None:
            coords = point_inputs["point_coords"]
            labels = point_inputs["point_labels"]
        else:
            coords = mx.zeros((batch, 1, 2))
            labels = -mx.ones((batch, 1), dtype=mx.int32)
        if mask_inputs is not None:
            if tuple(mask_inputs.shape[-2:]) != self.sam_prompt_encoder.mask_input_size:
                mask_prompt = interpolate_bilinear(
                    mask_inputs.astype(mx.float32),
                    self.sam_prompt_encoder.mask_input_size,
                    antialias=True,
                )
            else:
                mask_prompt = mask_inputs
        else:
            mask_prompt = None
        sparse, dense = self.sam_prompt_encoder((coords, labels), mask_prompt)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            backbone_features,
            self.sam_prompt_encoder.get_dense_pe(),
            sparse,
            dense,
            multimask_output,
            high_res_features or [],
        )
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = mx.where(
            is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE
        ).astype(mx.float32)
        high_res_multimasks = interpolate_bilinear(
            low_res_multimasks, (self.image_size, self.image_size)
        )
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best = mx.argmax(ious, axis=-1)
            rows = mx.arange(batch)
            low_res_masks = low_res_multimasks[rows, best][:, None]
            high_res_masks = high_res_multimasks[rows, best][:, None]
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = sam_output_tokens[rows, best]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        appearing = is_obj_appearing.astype(mx.float32)
        obj_ptr = appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - appearing) * self.no_obj_ptr
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def use_mask_as_output(
        self,
        backbone_features: array,
        high_res_features: list[array] | None,
        mask_inputs: array,
    ) -> tuple[array, ...]:
        """Use a given mask as the frame's prediction instead of decoding one.

        ``mask_inputs`` (B, 1, 512, 512) in {0, 1} becomes logits ``20 * mask - 10`` (so
        +10 inside and -10 outside), downsampled with antialiasing to 128 x 128 for
        ``low_res_masks``. The decoder still runs, with the mask reduced to 128 x 128 by the
        learned ``mask_downsample`` convolution as its dense prompt, but only to produce
        ``obj_ptr``. The object counts as present when the mask has any foreground pixel.
        Returns the same 7-tuple as ``forward_sam_heads``.
        """
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.astype(mx.float32)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = interpolate_bilinear(
            high_res_masks,
            (high_res_masks.shape[-2] // 4, high_res_masks.shape[-1] // 4),
            antialias=True,
        )
        ious = mx.ones((mask_inputs.shape[0], 1))
        obj_ptr = self.forward_sam_heads(
            backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
        )[5]
        flat = mx.reshape(mask_inputs_float, (mask_inputs.shape[0], -1))
        appearing = mx.any(flat > 0.0, axis=1)[..., None].astype(mx.float32)
        object_score_logits = out_scale * appearing + out_bias
        obj_ptr = appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - appearing) * self.no_obj_ptr
        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    # --- memory ---------------------------------------------------------------------

    def _obj_ptr_pos(self, distances: list[int], batch: int, max_distance: int) -> array:
        """Projected 1D sine encodings of signed pointer distances, (N, B, mem_dim)."""
        positions = mx.array(distances, dtype=mx.float32) / max_distance
        encoding = self.obj_ptr_tpos_proj(get_1d_sine_pe(positions, dim=self.hidden_dim))
        return mx.broadcast_to(encoding[:, None], (len(distances), batch, self.mem_dim))

    def prepare_memory_conditioned_features(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: array,
        current_vision_pos_embeds: array,
        feat_size: tuple[int, int],
        output_dict: OutputBanks,
        num_frames: int,
        track_in_reverse: bool = False,
    ) -> array:
        """Fuse the (HW, B, C) features of the current frame with the memory bank.

        The memory consists of the spatial memories of every conditioning frame, those of
        the previous ``num_maskmem - 1`` tracked frames in tracking direction (``reverse``
        flips it), and object pointers of conditioning frames (at or before ``frame_idx``
        in tracking direction) and of up to ``min(num_frames, max_obj_ptrs_in_encoder) - 1``
        previous frames. Pointers are split into ``hidden_dim // mem_dim`` tokens
        and carry tracking-direction-relative temporal position encodings. For an
        initial conditioning frame (``is_init_cond_frame``), no memory is used and
        only ``no_mem_embed`` is added.
        For a tracked frame with an eligible object pointer,
        ``min(num_frames, max_obj_ptrs_in_encoder) == 1`` makes the pointer-position
        divisor zero, producing nonfinite values as in the pinned upstream formula.
        Returns (B, C, H, W) features.

        Raises:
            RuntimeError: Called for a tracked frame with no conditioning-frame output.
        """
        batch = current_vision_feats.shape[1]
        if is_init_cond_frame:
            return unflatten_hw(current_vision_feats + self.no_mem_embed, feat_size)
        cond_outputs = output_dict["cond_frame_outputs"]
        if not cond_outputs:
            raise RuntimeError("MedSAM2 propagation requires a conditioning frame")
        memories: list[array] = []
        positions: list[array] = []
        t_pos_and_prevs: list[tuple[int, FrameOutput | None]] = [
            (0, out) for out in cond_outputs.values()
        ]
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            prev = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
            t_pos_and_prevs.append((t_pos, output_dict["non_cond_frame_outputs"].get(prev)))
        for t_pos, prev_out in t_pos_and_prevs:
            if prev_out is None:
                continue
            features = mx.array(cast(Tensor, prev_out["maskmem_features"]))
            memories.append(flatten_hw(features).astype(mx.float32))
            position = flatten_hw(mx.array(cast(Tensor, prev_out["maskmem_pos_enc"])))
            positions.append(position + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])
        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        sign = -1 if track_in_reverse else 1
        pointers: list[tuple[int, array]] = [
            ((frame_idx - t) * sign, out["obj_ptr"])
            for t, out in cond_outputs.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        ]
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if t < 0 or t >= num_frames:
                break
            out = output_dict["non_cond_frame_outputs"].get(t)
            if out is not None:
                pointers.append((t_diff, out["obj_ptr"]))
        num_obj_ptr_tokens = 0
        if pointers:
            distances, vectors = zip(*pointers, strict=True)
            obj_ptrs = mx.stack([mx.array(v) for v in vectors], axis=0)
            obj_pos = self._obj_ptr_pos(list(distances), batch, max_obj_ptrs - 1)
            split = self.hidden_dim // self.mem_dim
            if split > 1:
                obj_ptrs = mx.reshape(obj_ptrs, (-1, batch, split, self.mem_dim))
                obj_ptrs = mx.reshape(
                    mx.transpose(obj_ptrs, (0, 2, 1, 3)), (-1, batch, self.mem_dim)
                )
                obj_pos = mx.repeat(obj_pos, split, axis=0)
            memories.append(obj_ptrs)
            positions.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = mx.concatenate(memories, axis=0)
        memory_pos = mx.concatenate(positions, axis=0)
        conditioned = self.memory_attention(
            current_vision_feats,
            memory,
            current_vision_pos_embeds,
            memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return unflatten_hw(conditioned, feat_size)

    def encode_new_memory(
        self,
        pix_feat: array,
        pred_masks_high_res: array,
        object_score_logits: array,
        is_mask_from_pts: bool,
    ) -> tuple[array, array]:
        """Spatial memory (B, mem_dim, 32, 32) in FP32 and its position encoding.

        The mask logits are binarized (> 0) when both ``is_mask_from_pts`` and the model's
        ``binarize_mask_from_pts_for_mem_enc`` option are set; otherwise a sigmoid is used.
        """
        return self.encode_mask_memory(
            pix_feat,
            pred_masks_high_res,
            object_score_logits,
            binarize=self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts,
        )

    def encode_mask_memory(
        self,
        pix_feat: array,
        pred_masks_high_res: array,
        object_score_logits: array,
        *,
        binarize: bool,
    ) -> tuple[array, array]:
        """Encode mask logits (B, 1, 512, 512) and image features into spatial memory.

        The mask (binarized at logit 0 or passed through a sigmoid) is mapped to
        ``scale * m + bias`` (20 and -10) before the memory encoder, which is told to skip
        its own sigmoid. Objects scored absent get the learned ``no_obj_embed_spatial``
        added to their features.
        """
        if binarize:
            mask_for_mem = (pred_masks_high_res > 0).astype(mx.float32)
        else:
            mask_for_mem = mx.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * self.config.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.config.sigmoid_bias_for_mem_enc
        out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        features = out["vision_features"]
        absent = 1 - (object_score_logits > 0).astype(mx.float32)
        features = features + absent[..., None, None] * self.no_obj_embed_spatial[..., None, None]
        return features, out["vision_pos_enc"][0]

    # --- tracking -------------------------------------------------------------------

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs: PointInputs | None) -> bool:
        """Whether to decode three candidate masks: only for 0 or 1 prompt points.

        A box counts as two points, so box prompts always get a single mask.
        """
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        return bool(
            self.config.multimask_output_in_sam
            and (is_init_cond_frame or self.config.multimask_output_for_tracking)
            and self.config.multimask_min_pt_num <= num_pts <= self.config.multimask_max_pt_num
        )

    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        backbone: ImageFeatures,
        point_inputs: PointInputs | None,
        mask_inputs: array | None,
        output_dict: OutputBanks,
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: array | None = None,
    ) -> TrackOutput:
        """Predict one frame for a batch of objects.

        ``backbone`` is the (batch-expanded) ``forward_image`` output. A mask prompt
        (``mask_inputs``) is used directly as the prediction when
        ``use_mask_input_as_output_without_sam`` is enabled; otherwise the features are
        conditioned on the memory bank and decoded with the prompts, using
        ``prev_sam_mask_logits`` as a dense mask prompt when refining a frame. Returns
        ``pred_masks`` (B, 1, 128, 128), ``pred_masks_high_res`` (B, 1, 512, 512),
        ``obj_ptr``, ``object_score_logits``, and spatial memory and its position encoding
        when ``run_mem_encoder`` and ``num_maskmem > 0`` (otherwise ``None``).
        """
        fpn = backbone["backbone_fpn"]
        high_res_features = list(fpn[:-1])
        pix_feat = fpn[-1]
        feat_size = (pix_feat.shape[-2], pix_feat.shape[-1])
        current = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if mask_inputs is not None and self.config.use_mask_input_as_output_without_sam:
            sam_outputs = self.use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat_with_mem = self.prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                flatten_hw(pix_feat),
                flatten_hw(backbone["vision_pos_enc"][-1]),
                feat_size,
                output_dict,
                num_frames,
                track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                mask_inputs = prev_sam_mask_logits
            sam_outputs = self.forward_sam_heads(
                pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
            )
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs
        current["pred_masks"] = low_res_masks
        current["pred_masks_high_res"] = high_res_masks
        current["obj_ptr"] = obj_ptr
        current["object_score_logits"] = object_score_logits
        if run_mem_encoder and self.num_maskmem > 0:
            features, position = self.encode_new_memory(
                pix_feat, high_res_masks, object_score_logits, point_inputs is not None
            )
            current["maskmem_features"] = features
            current["maskmem_pos_enc"] = position
        else:
            current["maskmem_features"] = None
            current["maskmem_pos_enc"] = None
        return cast(TrackOutput, current)
