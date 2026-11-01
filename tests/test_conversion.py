"""Strict MedSAM2 weight validation: names, shapes, dtype and finiteness."""

from __future__ import annotations

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from medsam2_mlx._types import HostArray
from medsam2_mlx.mlx_weights import load_model
from tests.conftest import small_config


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "float16", "nonfinite"])
def test_loading_rejects_mismatched_or_invalid_tensors(
    state_dict: dict[str, HostArray], mutation: str
) -> None:
    """Reject missing, extra, misshaped, non-FP32 and nonfinite tensors."""
    state = dict(state_dict)
    if mutation == "missing":
        del state["no_obj_ptr"]
    elif mutation == "unexpected":
        state["extra.weight"] = np.zeros((1,), dtype=np.float32)
    elif mutation == "shape":
        state["mask_downsample.weight"] = np.zeros((1, 1, 2, 2), dtype=np.float32)
    elif mutation == "float16":
        state["no_obj_ptr"] = state["no_obj_ptr"].astype(np.float16)
    else:
        state["no_obj_ptr"] = np.full_like(state["no_obj_ptr"], np.nan)
    with pytest.raises(InvalidInputError):
        load_model(state, config=small_config())
