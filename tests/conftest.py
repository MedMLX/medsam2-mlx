"""Small seeded MedSAM2 graph and its PyTorch reference recording."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from medsam2_mlx._types import ArrayInput, NumericArray

if TYPE_CHECKING:
    from medsam2_mlx.config import Sam2Config
    from medsam2_mlx.sam2_base import MedSAM2Model

import numpy as np
import pytest

from medsam2_mlx._types import HostArray
from tests.reference_cases import (
    REFERENCE_PLATFORM,
    SMALL,
    UPSTREAM_REVISION,
)
from tests.reference_types import TorchRuntime

FIXTURE = Path(__file__).parent / "fixtures" / "reference_small.npz"


def small_config() -> Sam2Config:
    """Build the reduced graph configuration used by seeded reference fixtures."""
    from medsam2_mlx.config import (
        HieraConfig,
        ImageEncoderConfig,
        MemoryAttentionConfig,
        Sam2Config,
    )

    feat = SMALL["image_size"] // 16
    return Sam2Config(
        image_encoder=ImageEncoderConfig(
            hiera=HieraConfig(
                embed_dim=SMALL["embed_dim"],
                num_heads=SMALL["num_heads"],
                stages=SMALL["stages"],
                global_att_blocks=SMALL["global_att_blocks"],
                window_spec=SMALL["window_spec"],
                window_pos_embed_bkg_spatial_size=SMALL["window_pos_embed_bkg_spatial_size"],
            ),
            d_model=SMALL["d_model"],
            backbone_channel_list=SMALL["backbone_channel_list"],
        ),
        memory_attention=MemoryAttentionConfig(
            num_layers=SMALL["memory_attention_layers"],
            dim_feedforward=SMALL["dim_feedforward"],
            feat_sizes=(feat, feat),
        ),
        mem_dim=SMALL["mem_dim"],
        num_maskmem=SMALL["num_maskmem"],
        image_size=SMALL["image_size"],
        max_obj_ptrs_in_encoder=SMALL["max_obj_ptrs_in_encoder"],
        decoder_mlp_dim=SMALL["decoder_mlp_dim"],
        iou_head_hidden_dim=SMALL["iou_head_hidden_dim"],
    )


@pytest.fixture(scope="session")
def reference() -> dict[str, HostArray]:
    """Load the saved small-graph source reference arrays."""
    with np.load(FIXTURE, allow_pickle=False) as data:
        assert data["recorded_platform"].item() == REFERENCE_PLATFORM
        assert data["meta_upstream_revision"].item() == UPSTREAM_REVISION
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="session")
def state_dict(reference: dict[str, HostArray]) -> dict[str, HostArray]:
    """Regenerate the pinned source graph's seeded parameters for parity tests."""
    source = os.environ.get("MEDSAM2_SOURCE_ROOT")
    if not source:
        pytest.skip("Set MEDSAM2_SOURCE_ROOT to the pinned upstream MedSAM2 checkout")
    from tests.upstream_reference import import_upstream, seeded_state

    torch = cast(TorchRuntime, import_module("torch"))
    expected_version = str(reference["meta_torch_version"].item())
    assert str(torch.__version__) == expected_version, (
        f"Reference parameters require Torch {expected_version}; found {torch.__version__}"
    )
    state = seeded_state(import_upstream(Path(source).resolve()), torch)
    return {name: cast(HostArray, value.numpy()) for name, value in state.items()}


def load_small_model(
    state_dict: dict[str, HostArray], *, apply_postprocessing: bool
) -> MedSAM2Model:
    """Build and load the reduced model from a named tensor mapping."""
    from medsam2_mlx.mlx_weights import load_model

    config = small_config()
    return load_model(state_dict, apply_postprocessing=apply_postprocessing, config=config)


def relative_error(actual: ArrayInput, expected: HostArray) -> float:
    """max |actual - expected| / max |expected|."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = cast(NumericArray, expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    return float(np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-12))
