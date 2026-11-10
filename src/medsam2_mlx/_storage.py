"""Lossless native restoration of tagged host spatial-memory payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from medsam2_mlx._runtime import import_mlx
from medsam2_mlx._types import MemoryTensor, PackedBFloat16

if TYPE_CHECKING:
    from mlx.core import array


def native_memory(value: MemoryTensor) -> array:
    """Restore BF16 bits or import an ordinary native/host memory tensor."""
    mx = import_mlx()
    if isinstance(value, PackedBFloat16):
        return mx.array(value.bits).view(mx.bfloat16)
    return mx.array(value)
