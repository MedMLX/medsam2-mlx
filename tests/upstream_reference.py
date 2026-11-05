"""Pinned upstream source admission and the seeded reference graph for parity tests.

Nothing here records fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

from tests.reference_cases import MODEL_SEED, SMALL, UPSTREAM_REVISION
from tests.reference_types import (
    SourceModules,
    SourcePredictor,
    TorchRuntime,
)


def import_upstream(source: Path) -> SourceModules:
    """Import the consumed upstream modules from the verified checkout at ``source``.

    ``source`` must be a resolved path. Raises ``ValueError`` if a module resolves outside it.
    """
    sys.dont_write_bytecode = True
    verify_upstream(source)
    sys.path.insert(0, str(source))
    names = {
        "hieradet": "sam2.modeling.backbones.hieradet",
        "image_encoder": "sam2.modeling.backbones.image_encoder",
        "position_encoding": "sam2.modeling.position_encoding",
        "memory_attention": "sam2.modeling.memory_attention",
        "memory_encoder": "sam2.modeling.memory_encoder",
        "transformer": "sam2.modeling.sam.transformer",
        "mask_decoder": "sam2.modeling.sam.mask_decoder",
        "predictor": "sam2.sam2_video_predictor_npz",
    }
    modules = {key: import_module(name) for key, name in names.items()}
    for module in modules.values():
        if not Path(cast(str, module.__file__)).resolve().is_relative_to(source):
            raise ValueError(f"Upstream module resolved outside pinned source: {module.__file__}")
    return cast(SourceModules, modules)


def verify_upstream(source: Path) -> None:
    """Admit a pinned Git checkout or a snapshot whose manifest names the revision."""
    manifest = source / "source_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        if data["repository"] != "bowang-lab/MedSAM2":
            raise ValueError("Invalid upstream source manifest")
        revision = data["revision"]
    else:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected MedSAM2 at {UPSTREAM_REVISION}, found {revision!r}")


def build_reference(up: SourceModules) -> SourcePredictor:
    """SAM2VideoPredictorNPZ with the SMALL graph, mirroring sam2.1_hiera_t512.yaml."""
    d = SMALL["d_model"]
    sine = up["position_encoding"].PositionEmbeddingSine
    trunk = up["hieradet"].Hiera(
        embed_dim=SMALL["embed_dim"],
        num_heads=SMALL["num_heads"],
        stages=SMALL["stages"],
        global_att_blocks=SMALL["global_att_blocks"],
        window_spec=SMALL["window_spec"],
        window_pos_embed_bkg_spatial_size=SMALL["window_pos_embed_bkg_spatial_size"],
    )
    neck = up["image_encoder"].FpnNeck(
        position_encoding=sine(num_pos_feats=d, normalize=True, temperature=10000),
        d_model=d,
        backbone_channel_list=list(SMALL["backbone_channel_list"]),
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    image_encoder = up["image_encoder"].ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    rope = up["transformer"].RoPEAttention
    feat = SMALL["image_size"] // 16
    layer = up["memory_attention"].MemoryAttentionLayer(
        activation="relu",
        cross_attention=rope(
            rope_theta=10000.0,
            feat_sizes=[feat, feat],
            rope_k_repeat=True,
            embedding_dim=d,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=SMALL["mem_dim"],
        ),
        d_model=d,
        dim_feedforward=SMALL["dim_feedforward"],
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=rope(
            rope_theta=10000.0,
            feat_sizes=[feat, feat],
            embedding_dim=d,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
        ),
    )
    memory_attention = up["memory_attention"].MemoryAttention(
        d_model=d,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=SMALL["memory_attention_layers"],
    )
    me = up["memory_encoder"]
    memory_encoder = me.MemoryEncoder(
        out_dim=SMALL["mem_dim"],
        position_encoding=sine(num_pos_feats=SMALL["mem_dim"], normalize=True, temperature=10000),
        mask_downsampler=me.MaskDownSampler(embed_dim=d, kernel_size=3, stride=2, padding=1),
        fuser=me.Fuser(
            layer=me.CXBlock(
                dim=d, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True
            ),
            num_layers=2,
        ),
        in_dim=d,
    )
    model = up["predictor"].SAM2VideoPredictorNPZ(
        fill_hole_area=0,
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=SMALL["num_maskmem"],
        image_size=SMALL["image_size"],
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        max_obj_ptrs_in_encoder=SMALL["max_obj_ptrs_in_encoder"],
        sam_mask_decoder_extra_args={},
    )
    # SAM2Base hard-codes a 2048-wide decoder MLP and a 256-wide IoU head. Keep the
    # fixture small with the same MaskDecoder class and options at reduced widths.
    model.sam_mask_decoder = up["mask_decoder"].MaskDecoder(
        num_multimask_outputs=3,
        transformer=up["transformer"].TwoWayTransformer(
            depth=2, embedding_dim=d, mlp_dim=SMALL["decoder_mlp_dim"], num_heads=8
        ),
        transformer_dim=d,
        iou_head_depth=3,
        iou_head_hidden_dim=SMALL["iou_head_hidden_dim"],
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
    )
    return model


def seeded_state(up: SourceModules, torch: TorchRuntime) -> dict[str, Tensor]:
    """Seeded weights, rounded to FP16-representable FP32 so the fixture stores FP16."""
    torch.manual_seed(MODEL_SEED)
    model = build_reference(up)
    generator = torch.Generator().manual_seed(MODEL_SEED)
    state: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        # Default initializers leave many parameters at zeros or ones; perturb them all
        # so that every tensor and layout is observable in the outputs.
        noise = torch.randn(value.shape, generator=generator) * 0.05
        value = value + noise if value.abs().max() == 0 or value.ndim == 1 else value
        state[name] = value.to(torch.float16).to(torch.float32)
    # Give random-weight mask logits a useful range (about +-5) so that sign-based
    # steps (binarized memory, stability fallback, hole filling) are well separated.
    for name in list(state):
        if name.startswith("sam_mask_decoder.output_hypernetworks_mlps.") and ".layers.2." in name:
            state[name] = (state[name] * 60).to(torch.float16).to(torch.float32)
    # Keep random-weight masks visible instead of the absent-object sentinel.
    state["sam_mask_decoder.pred_obj_score_head.layers.2.bias"] = torch.full((1,), 2.0)
    return state
