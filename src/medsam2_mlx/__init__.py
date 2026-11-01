"""Native MLX implementation of MedSAM2."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from medsam2_mlx.request import (
    MedSAM2Prompt,
    MedSAM2RuntimeConfig,
)

if TYPE_CHECKING:
    from medsam2_mlx.mlx_model import create_model
    from medsam2_mlx.mlx_predictor import PreparedVideoFrames, build_predictor
    from medsam2_mlx.mlx_weights import load_model

__version__ = "0.2.8"

__all__ = [
    "MedSAM2Prompt",
    "MedSAM2RuntimeConfig",
    "PreparedVideoFrames",
    "build_predictor",
    "create_model",
    "load_model",
]

# MLX loads with the model modules, so importing the medmlx declaration stays light.
_LAZY = {
    "PreparedVideoFrames": "medsam2_mlx.mlx_predictor",
    "build_predictor": "medsam2_mlx.mlx_predictor",
    "create_model": "medsam2_mlx.mlx_model",
    "load_model": "medsam2_mlx.mlx_weights",
}


def __getattr__(name: str) -> object:
    """Import a model-dependent public name on first access and cache it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
