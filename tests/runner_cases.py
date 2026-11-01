"""Synthetic volumes and prompts shared by the runner recorder and tests.

``write_seeded_weights`` imports MLX only to save arrays; parameter shapes come
from the upstream recording.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medsam2_mlx._types import (
        BoxCoordinates,
        HostArray,
        NiftiRuntime,
        NumericArray,
        Propagation,
    )

from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np


class ExtraObjectCase(TypedDict):
    """Additional object prompt applied on a runner case's seed frame."""

    obj_id: int
    box: tuple[float, float, float, float]


class RunnerCase(TypedDict):
    """Runner scenario with its volume, seed frame, prompt payload and propagation direction."""

    volume: str
    frame: int
    propagation: Propagation
    box: NotRequired[BoxCoordinates]
    mask: NotRequired[tuple[slice, slice]]
    slice_range: NotRequired[tuple[int, int]]
    obj_id: NotRequired[int]
    points_xy: NotRequired[tuple[tuple[float, float], ...]]
    point_labels: NotRequired[tuple[int, ...]]
    extra_objects: NotRequired[tuple[ExtraObjectCase, ...]]


class BoxCase(TypedDict):
    """Runner case narrowed to a required box prompt."""

    box: tuple[float, float, float, float]


WEIGHT_SEED = 2026


def _affine(seed: int) -> HostArray:
    """Generate a seeded synthetic spatial affine for runner cases."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-0.4, 0.4)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(rng.uniform(0.5, 3.0, 3))
    affine[0] *= -1
    affine[:3, 3] = rng.uniform(-150, 150, 3)
    return affine


def _texture(shape: tuple[int, ...], seed: int) -> NumericArray:
    """Generate seeded smooth texture with noise for synthetic images."""
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.linspace(0, 3, size) for size in shape], indexing="ij")
    smooth = np.sin(grid[0] * 2.1) * np.cos(grid[1] * 1.7)
    if len(shape) == 3:
        smooth = smooth + 0.3 * np.sin(grid[2] * 2.5)
    return smooth + rng.normal(0, 0.15, shape)


# name: (seed, shape in NIfTI voxel order, dtype, (slope, intercept) or None, qform only)
VOLUMES: dict[str, tuple[int, tuple[int, ...], str, tuple[float, float] | None, bool]] = {
    "ct3": (12, (30, 40, 3), "int16", (1.0, -1024.0), False),
    "ct5_tenth": (13, (24, 20, 5), "int16", (0.1, -3.3), False),
    # Seeded noise whose uint8 stack depends on ITK's float32 rescaling (3 voxels).
    "tenth_noise": (17, (30, 32, 4), "int16", (0.1, -3.3), False),
    "float32": (15, (16, 18, 4), "float32", None, False),
    "uint8": (18, (20, 22, 3), "uint8", None, False),
    "f64_scaled": (14, (14, 12, 2), "float64", (0.3, 1.1), False),
    "slice2d": (16, (26, 34), "float32", None, False),
    "z1": (19, (28, 24, 1), "int16", None, True),
    "constant": (11, (10, 12, 2), "int16", None, False),
}


def write_volume(directory: Path, name: str) -> Path:
    """Write one named synthetic NIfTI volume and return its path."""

    import nibabel as nib

    nib = cast("NiftiRuntime", nib)

    seed, shape, dtype, scaling, qform_only = VOLUMES[name]
    texture = _texture(shape, seed)
    if name == "constant":
        data = np.full(shape, 7, dtype=np.int16)
    elif name == "tenth_noise":
        data = np.random.default_rng(5).integers(0, 4000, shape).astype(np.int16)
    elif dtype == "uint8":
        data = np.clip((texture + 1.5) * 80, 0, 255).astype(np.uint8)
    elif dtype == "int16":
        data = np.clip((texture + 1.5) * 1200, 0, 4000).astype(np.int16)
    else:
        data = (texture * 300 - 40).astype(dtype)
    image = nib.Nifti1Image(data, _affine(seed))
    if scaling is not None:
        image.header.set_slope_inter(*scaling)
    if qform_only:
        image.set_qform(image.affine, code=1)
        image.set_sform(None, code=0)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.nii.gz"
    nib.save(image, path)
    return path


def write_mask(
    directory: Path,
    like: Path,
    name: str,
    *,
    frame_idx: int,
    region: tuple[slice, slice],
) -> Path:
    """Write a mask NIfTI on ``like``'s grid with ``region`` (y, x) set on ``frame_idx``."""

    import nibabel as nib

    nib = cast("NiftiRuntime", nib)

    source = nib.load(like)
    zyx = source.shape[::-1]
    mask_zyx = np.zeros(zyx, dtype=np.uint8)
    if len(zyx) == 2:
        mask_zyx[region] = 1
    else:
        mask_zyx[frame_idx][region] = 1
    image = nib.Nifti1Image(mask_zyx.T, None, header=source.header)
    image.set_data_dtype("uint8")
    image.header.set_slope_inter(1.0, 0.0)
    path = directory / f"{name}.nii.gz"
    nib.save(image, path)
    return path


# Prompt and propagation cases run through the full-size seeded network.
FAKE_RUNS: dict[str, RunnerCase] = {
    "both": {
        "volume": "ct5_tenth",
        "box": (3.0, 4.5, 14.0, 17.0),
        "frame": 2,
        "propagation": "both",
    },
    "both_range": {
        "volume": "ct5_tenth",
        "box": (3.0, 4.5, 14.0, 17.0),
        "frame": 2,
        "propagation": "both",
        "slice_range": (1, 3),
    },
    "forward_range": {
        "volume": "ct5_tenth",
        "box": (0.0, 0.0, 9.0, 6.0),
        "frame": 0,
        "propagation": "forward",
        "slice_range": (0, 2),
    },
    "reverse": {
        "volume": "ct5_tenth",
        "box": (10.0, 2.0, 22.5, 11.0),
        "frame": 4,
        "propagation": "reverse",
    },
    "mask_forward": {
        "volume": "ct5_tenth",
        "mask": (np.s_[3:9], np.s_[5:12]),
        "frame": 1,
        "propagation": "forward",
        "obj_id": 3,
    },
    "slice2d": {
        "volume": "slice2d",
        "box": (2.0, 3.0, 20.0, 15.0),
        "frame": 0,
        "propagation": "none",
    },
    "z1_mask": {
        "volume": "z1",
        "mask": (np.s_[4:15], np.s_[6:20]),
        "frame": 0,
        "propagation": "both",
    },
    "float_none": {
        "volume": "float32",
        "box": (1.0, 2.0, 12.0, 14.0),
        "frame": 1,
        "propagation": "none",
        "obj_id": 7,
    },
    "points_none": {
        "volume": "slice2d",
        "points_xy": ((10.0, 12.0), (18.0, 20.0)),
        "point_labels": (1, 1),
        "frame": 0,
        "propagation": "none",
    },
    "two_objects_both": {
        "volume": "ct5_tenth",
        "box": (3.0, 4.5, 14.0, 17.0),
        "frame": 2,
        "propagation": "both",
        "extra_objects": ({"obj_id": 5, "box": (1.0, 1.0, 8.0, 8.0)},),
    },
}
# Runs with seeded full-size weights through the MLX network.
NETWORK_RUNS: dict[str, RunnerCase] = {
    "ct3_box_both": {
        "volume": "ct3",
        "box": (5.0, 6.0, 20.0, 30.0),
        "frame": 1,
        "propagation": "both",
    },
    "ct3_mask_forward": {
        "volume": "ct3",
        "mask": (np.s_[8:25], np.s_[4:18]),
        "frame": 0,
        "propagation": "forward",
        "obj_id": 2,
    },
    "slice2d_box": {
        "volume": "slice2d",
        "box": (2.0, 3.0, 20.0, 15.0),
        "frame": 0,
        "propagation": "none",
    },
    "slice2d_points": {
        "volume": "slice2d",
        "points_xy": ((9.0, 11.0), (19.0, 21.0)),
        "point_labels": (1, 1),
        "frame": 0,
        "propagation": "none",
    },
    "ct3_two_objects_forward": {
        "volume": "ct3",
        "box": (5.0, 6.0, 20.0, 30.0),
        "frame": 1,
        "propagation": "forward",
        "extra_objects": ({"obj_id": 9, "box": (2.0, 2.0, 10.0, 10.0)},),
    },
}
def prepare_inputs(directory: Path, run: RunnerCase) -> tuple[Path, Path | None]:
    """Write a run's volume and, for mask prompts, its seed mask."""

    volume = write_volume(directory, run["volume"])
    if "mask" not in run:
        return volume, None
    mask = write_mask(
        directory, volume, f"{run['volume']}_seed", frame_idx=run["frame"], region=run["mask"]
    )
    return volume, mask


def seeded_arrays(
    shapes: dict[str, tuple[int, ...]], seed: int = WEIGHT_SEED
) -> dict[str, HostArray]:
    """Deterministic FP32 weights in the recorded upstream parameter order."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, HostArray] = {}
    for name, shape in shapes.items():
        if len(shape) >= 2:
            bound = 1.0 / np.sqrt(np.prod(shape[1:]))
            array = rng.uniform(-bound, bound, shape)
        elif name.endswith("weight"):
            array = 1.0 + 0.05 * rng.standard_normal(shape)
        else:
            array = 0.02 * rng.standard_normal(shape)
        if name == "sam_mask_decoder.pred_obj_score_head.layers.2.bias":
            array = np.full(shape, 4.0)
        arrays[name] = array.astype(np.float32)
    return arrays


def write_seeded_weights(path: Path, seed: int = WEIGHT_SEED) -> Path:
    """Save the recording's seeded full-size parameters as MLX safetensors."""
    import json

    from medsam2_mlx._runtime import import_mlx

    mx = import_mlx()

    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/runner_reference.npz"
    with np.load(fixture, allow_pickle=False) as data:
        shapes = {name: tuple(shape) for name, shape in json.loads(data["parameter_shapes"].item())}
    arrays = seeded_arrays(shapes, seed)
    mx.save_safetensors(str(path), {name: mx.array(value) for name, value in arrays.items()})
    return path
