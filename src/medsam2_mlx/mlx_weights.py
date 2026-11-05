"""Load caller-owned converted MedSAM2 weights without asset staging or artifact records.

Converted format: a flat mapping from each MedSAM2 PyTorch ``state_dict`` key to an
FP32 array with the PyTorch shape and memory layout (linear ``(out, in)``,
convolution ``(out, in / groups, kh, kw)``, transposed convolution
``(in, out, kh, kw)``). ``sam2.1_hiera_t512`` checkpoints have 471 tensors.
Callers usually store the mapping with ``mx.save_safetensors``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError

from medsam2_mlx._runtime import import_mlx
from medsam2_mlx._types import ArrayInput
from medsam2_mlx.config import SAM2_1_HIERA_T512, Sam2Config
from medsam2_mlx.mlx_model import create_model
from medsam2_mlx.sam2_base import MedSAM2Model

__all__ = ["load_model"]


def _check_parameter_names_and_shapes(
    shapes: Mapping[str, tuple[int, ...]], model: MedSAM2Model
) -> None:
    """Require exactly the model's parameter names, each with the model's shape."""
    from mlx.utils import tree_flatten

    expected = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    missing = sorted(set(expected) - set(shapes))
    unexpected = sorted(set(shapes) - set(expected))
    if missing or unexpected:
        raise InvalidInputError(
            "MedSAM2 weights do not match the MLX graph parameters: "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}. "
            "Use a published MedSAM2 safetensors checkpoint."
        )
    wrong = sorted(name for name, shape in expected.items() if tuple(shapes[name]) != shape)
    if wrong:
        details = ", ".join(f"{name}: {tuple(shapes[name])} != {expected[name]}" for name in wrong)
        raise InvalidInputError(f"MedSAM2 weights have wrong shapes: {details}")


def load_model(
    checkpoint: str | Path | Mapping[str, ArrayInput],
    *,
    apply_postprocessing: bool = True,
    config: Sam2Config = SAM2_1_HIERA_T512,
) -> MedSAM2Model:
    """Load FP32 MLX weights from a converted checkpoint path or a converted state dict.

    A path is read as safetensors. The names must be
    exactly the graph's parameter names with matching shapes, and every value must be
    FP32 and finite.

    Raises:
        InvalidInputError: Weights are not FP32, contain nonfinite values, or do not match
            the graph's names and shapes.
        MissingDependencyError: Not on macOS arm64, or Metal is unavailable.
    """
    if isinstance(checkpoint, str | Path) and Path(checkpoint).suffix != ".safetensors":
        raise InvalidInputError("Native MedSAM2 weights must use the .safetensors suffix")
    mx = import_mlx()
    model = create_model(apply_postprocessing=apply_postprocessing, config=config)
    weights = (
        cast(Mapping[str, ArrayInput], mx.load(str(checkpoint), format="safetensors"))
        if isinstance(checkpoint, str | Path)
        else checkpoint
    )
    weights = {name: mx.array(array) for name, array in weights.items()}
    if any(array.dtype != mx.float32 for array in weights.values()):
        raise InvalidInputError("MedSAM2 MLX weights must be FP32")
    _check_parameter_names_and_shapes(
        {name: tuple(array.shape) for name, array in weights.items()}, model
    )
    # One combined flag reads every tensor with a single host synchronization.
    finite = mx.all(mx.stack([mx.all(mx.isfinite(array)) for array in weights.values()]))
    if not finite.item():
        raise InvalidInputError("MedSAM2 MLX weights contain nonfinite values")
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
