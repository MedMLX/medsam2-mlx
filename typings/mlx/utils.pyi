"""Repository-used MLX 0.32.4 ``mlx/utils.py`` APIs, from the installed source.

``tree_flatten`` is declared only for parameter trees without ``destination``,
where the runtime returns a list of dotted names and array leaves.
"""

from mlx.core import array
from mlx.nn.layers.base import ParameterTree

def tree_flatten(tree: ParameterTree, prefix: str = "") -> list[tuple[str, array]]: ...
