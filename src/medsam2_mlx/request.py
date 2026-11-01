"""Typed requests and prompts for MedSAM2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from medmlx_core.errors import InvalidInputError

from medsam2_mlx._types import HostArray


@dataclass(frozen=True, slots=True)
class MedSAM2Prompt:
    """Prompt for one object on one slice (frame) of the native MedSAM2 predictor.

    Coordinates are ``(x, y)`` = (column, row) of the slice. With
    ``normalize_coords=True`` they are pixels of the original slice, whose size is the
    ``height``/``width`` given to ``init_state``; with ``False`` they are fractions of the
    slice size in [0, 1]. Either way the predictor rescales them to the 512-pixel model
    input. Bounds and box ordering are caller responsibilities; construction does not
    validate them.

    Attributes:
        box_xyxy: ``(x_min, y_min, x_max, y_max)`` box.
        points_xy: Click positions, paired one-to-one with ``point_labels``.
        point_labels: 1 for a foreground click and 0 for a background click.
        mask: 2D seed mask for the slice (nonzero is foreground), resized to the
            512-pixel input when its shape differs. When set, ``Session.add_prompt``
            sends only the mask to the predictor.
        frame_idx: Zero-based slice index; must be >= 0.
        obj_id: Positive object label; must be >= 1.
        clear_old_points: Replace earlier points on this slice instead of adding to
            them. The predictor rejects a box when this is ``False``.
        normalize_coords: Whether coordinates are pixels (``True``) or fractions (``False``).
    """

    box_xyxy: tuple[float, float, float, float] | None = None
    points_xy: tuple[tuple[float, float], ...] = ()
    point_labels: tuple[int, ...] = ()
    mask: HostArray | None = None
    frame_idx: int = 0
    obj_id: int = 1
    clear_old_points: bool = True
    normalize_coords: bool = True

    def __post_init__(self) -> None:
        """Check indices and that point coordinates and 0/1 labels are paired."""
        if self.frame_idx < 0:
            raise InvalidInputError("MedSAM2 prompt frame_idx must be >= 0")
        if self.obj_id < 1:
            raise InvalidInputError("MedSAM2 prompt obj_id must be >= 1")
        if bool(self.points_xy) != bool(self.point_labels):
            raise InvalidInputError("MedSAM2 point coordinates and labels must be paired")
        if len(self.points_xy) != len(self.point_labels):
            raise InvalidInputError("MedSAM2 point coordinate/label counts differ")
        unsupported = sorted(set(self.point_labels) - {0, 1})
        if unsupported:
            raise InvalidInputError(
                "MedSAM2 point labels must be 0 or 1, got: "
                + ", ".join(str(value) for value in unsupported)
            )
        if self.box_xyxy is None and not self.points_xy and self.mask is None:
            raise InvalidInputError("MedSAM2 prompt requires a box, points, or mask")
        if self.box_xyxy is not None and len(self.box_xyxy) != 4:
            raise InvalidInputError("MedSAM2 box prompt must be XYXY with 4 values")
        if self.mask is not None and self.mask.ndim != 2:
            raise InvalidInputError("MedSAM2 mask prompt must be a 2D mask")

    @property
    def mode(self) -> Literal["mask", "box_plus_points", "box", "points"]:
        """Return the prompt kind inferred from which fields are set (a mask wins)."""

        if self.mask is not None:
            return "mask"
        if self.box_xyxy is not None and self.points_xy:
            return "box_plus_points"
        if self.box_xyxy is not None:
            return "box"
        return "points"


@dataclass(frozen=True, slots=True, kw_only=True)
class MedSAM2RuntimeConfig:
    """Native MedSAM2 predictor configuration.

    ``checkpoint`` is a MedSAM2 ``.safetensors`` file.
    ``apply_postprocessing`` enables upstream's dynamic multimask fallback, binarized
    click masks in memory, and filling of holes up to 8 pixels. The two offload flags
    keep the frame stack and the per-frame masks/memories in host memory instead of on
    the device; computation still runs on Metal.
    """

    checkpoint: Path | None = None
    apply_postprocessing: bool = True
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False


__all__ = [
    "MedSAM2Prompt",
    "MedSAM2RuntimeConfig",
]
