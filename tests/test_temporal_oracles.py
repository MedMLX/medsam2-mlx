"""Real-checkpoint temporal RoPE parity against pinned MedSAM2 Torch attention.

Set MEDSAM2_SOURCE_ROOT to bowang-lab/MedSAM2 at 332f30d and
MEDSAM2_CHECKPOINT to its released CTLesion Torch checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

    from medsam2_mlx.transformer import RoPEAttention
import numpy as np
import pytest

from medsam2_mlx._types import CoreRuntime
from tests.reference_types import FloatArray, SourceRoPE, TorchRuntime
from tests.upstream_reference import import_upstream

_RTOL = 1e-4
_ATOL = 1e-4


@pytest.fixture(scope="module")
def rope_models() -> tuple[TorchRuntime, CoreRuntime, SourceRoPE, RoPEAttention]:
    """Use actual CTLesion projection weights and the full-size MLX graph."""
    source_value = os.environ.get("MEDSAM2_SOURCE_ROOT")
    checkpoint_value = os.environ.get("MEDSAM2_CHECKPOINT")
    if source_value is None or checkpoint_value is None:
        pytest.skip("set MEDSAM2_SOURCE_ROOT and MEDSAM2_CHECKPOINT for real temporal RoPE parity")
    source = Path(source_value).resolve()
    checkpoint = Path(checkpoint_value)
    assert source.is_dir(), f"MedSAM2 source is missing: {source}"
    assert checkpoint.is_file(), f"MedSAM2 checkpoint is missing: {checkpoint}"
    from medsam2_mlx._runtime import import_mlx

    mx = import_mlx()
    torch = cast(TorchRuntime, pytest.importorskip("torch"))
    from medsam2_mlx.mlx_model import create_model

    oracle_module = import_upstream(source)["transformer"]
    assert Path(oracle_module.__file__).resolve().is_relative_to(source)
    expected = oracle_module.RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[4, 4],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        kv_in_dim=64,
        rope_k_repeat=True,
        dropout=0.0,
    ).eval()
    state = cast(
        "dict[str, dict[str, Tensor]]",
        torch.load(checkpoint, map_location="cpu", weights_only=True),
    )["model"]
    prefix = "memory_attention.layers.0.cross_attn_image."
    projections = {
        name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)
    }
    expected.load_state_dict(projections, strict=True)
    model = create_model(apply_postprocessing=False)
    actual = model.memory_attention.layers[0].cross_attn_image
    actual.load_weights(
        [(name, mx.array(value.numpy())) for name, value in projections.items()],
        strict=True,
    )
    mx.eval(actual.parameters())
    return torch, mx, expected, actual


@pytest.mark.parametrize("memory_grids", [1, 2, 3, 7])
@pytest.mark.parametrize("pointer_tokens", [0, 4, 12])
def test_temporal_rope_matches_pinned_source(
    rope_models: tuple[TorchRuntime, CoreRuntime, SourceRoPE, RoPEAttention],
    memory_grids: int,
    pointer_tokens: int,
) -> None:
    """Rotate spatial memory grids while preserving appended object pointers."""
    torch, mx, expected, actual = rope_models
    rng = np.random.default_rng(715 + pointer_tokens)
    query = rng.standard_normal((1, 16, 256)).astype(np.float32)
    key = rng.standard_normal((1, 16 * memory_grids + pointer_tokens, 64)).astype(np.float32)
    value = rng.standard_normal(key.shape).astype(np.float32)
    with torch.inference_mode():
        oracle = expected(
            torch.from_numpy(query),
            torch.from_numpy(key),
            torch.from_numpy(value),
            num_k_exclude_rope=pointer_tokens,
        ).numpy()
    result = actual(
        mx.array(query), mx.array(key), mx.array(value), num_k_exclude_rope=pointer_tokens
    )
    mx.eval(result)
    observed = cast(FloatArray, np.asarray(result))
    assert observed.shape == oracle.shape == (1, 16, 256)
    assert observed.dtype == oracle.dtype == np.float32
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, oracle, rtol=_RTOL, atol=_ATOL)
