"""Repository-used MLX 0.32.4 ``mlx/nn/layers/base.py`` APIs, from the installed source.

Module is a partial interface: the runtime class subclasses dict, but only the
members this repository uses are declared.
"""

from typing import Self

from mlx.core import array

type ParameterTree = array | dict[str, ParameterTree] | list[ParameterTree]

class Module:
    def __init__(self) -> None: ...
    def parameters(self) -> dict[str, ParameterTree]: ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self: ...
