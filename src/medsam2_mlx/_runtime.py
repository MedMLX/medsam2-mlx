"""Apple Silicon and Metal admission for the native graph; imports remain lazy."""

from __future__ import annotations

import platform
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, cast

from medsam2_mlx._types import CoreRuntime

if TYPE_CHECKING:
    from mlx import nn as nn

    mx: CoreRuntime

from medmlx_core.errors import MissingDependencyError
from medmlx_core.runtime import import_mlx as _import_mlx

_METAL_HINT = "Run medsam2-mlx on macOS arm64 with MLX installed and Metal available."


def import_mlx() -> CoreRuntime:
    """Return the MLX core module after admitting macOS arm64 with an available Metal GPU.

    Raises:
        MissingDependencyError: The host is not macOS arm64, MLX cannot be imported, or
            Metal is unavailable. There is no CPU fallback.
    """
    host = (platform.system(), platform.machine())
    if host != ("Darwin", "arm64"):
        raise MissingDependencyError(
            f"MedSAM2 requires macOS arm64 with Metal; this host is {' '.join(host)}",
            extra="mlx",
            hint=_METAL_HINT,
        )
    try:
        mx = cast(CoreRuntime, _import_mlx())
    except MissingDependencyError as error:
        raise MissingDependencyError(str(error), extra="mlx", hint=_METAL_HINT) from error
    if not mx.metal.is_available():
        raise MissingDependencyError(
            "MedSAM2 requires Metal; no Metal device is available",
            extra="mlx",
            hint=_METAL_HINT,
        )
    return mx


def __getattr__(name: str) -> CoreRuntime | ModuleType:
    """Resolve ``mx`` and ``nn`` lazily so importing the package does not load MLX."""
    if name not in {"mx", "nn"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mx = import_mlx()
    value = mx if name == "mx" else import_module("mlx.nn")
    globals()[name] = value
    return value
