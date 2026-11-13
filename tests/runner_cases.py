"""Real-image crops and prompts shared by the runner recorder and tests.

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
        Propagation,
    )

import os
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


WEIGHT_SEED = 2026


def reference_path(name: str) -> Path:
    """Find regenerated reference arrays outside the repository."""
    default = Path(__file__).resolve().parents[1] / "work" / "real-reference"
    directory = os.environ.get("MEDMLX_TEST_REFERENCES", str(default))
    return Path(directory) / name


# Case dimensions are in NIfTI voxel order; optional scaling tests NIfTI decoding.
VOLUMES: dict[str, tuple[tuple[int, ...], str, tuple[float, float] | None, bool]] = {
    "ct3": ((30, 40, 3), "int16", (1.0, -1024.0), False),
    "ct5_tenth": ((24, 20, 5), "int16", (0.1, -3.3), False),
    "ct_tenth": ((30, 32, 4), "int16", (0.1, -3.3), False),
    "float32": ((16, 18, 4), "float32", None, False),
    "uint8": ((20, 22, 3), "uint8", None, False),
    "f64_scaled": ((14, 12, 2), "float64", (0.3, 1.1), False),
    "slice2d": ((26, 34), "float32", None, False),
    "z1": ((28, 24, 1), "int16", None, True),
    "constant": ((10, 12, 2), "int16", None, False),
}


def write_volume(directory: Path, name: str) -> Path:
    """Write a real CT or chest X-ray crop in the case's NIfTI representation."""
    import os

    import nibabel as nib_runtime
    from PIL import Image

    nib = cast("NiftiRuntime", nib_runtime)
    shape, dtype, scaling, qform_only = VOLUMES[name]
    if name == "slice2d":
        default = Path.home() / ".cache" / "medmlx" / "test-data" / "chest_xray.jpg"
        source = os.environ.get("MEDMLX_TEST_XRAY", str(default))
        with Image.open(source) as image:
            data = np.asarray(image.convert("L"))[200:200 + shape[1], 200:200 + shape[0]].T
        affine = np.eye(4)
    else:
        default = Path.home() / ".cache" / "medmlx" / "test-data" / "hcc_063_ct.nii.gz"
        source = os.environ.get("MEDMLX_TEST_CT", str(default))
        image = nib.load(source)
        start = (0, 0, 40) if name == "constant" else (150, 250, 40)
        data = np.asanyarray(image.dataobj)[
            start[0]:start[0] + shape[0], start[1]:start[1] + shape[1],
            start[2]:start[2] + shape[2],
        ]
        affine = np.asarray(image.affine, dtype=np.float64).copy()
        affine[:3, 3] += affine[:3, :3] @ np.asarray(start)
    if data.shape != shape:
        raise ValueError(f"{name} crop has shape {data.shape}, expected {shape}")
    if name == "constant" and np.ptp(data) != 0:
        raise ValueError("The constant case requires the HCC_063 air crop")
    if dtype == "uint8":
        data = np.clip((data.astype(np.float32) + 200) * 255 / 500, 0, 255)
    elif scaling is not None:
        data = (data.astype(np.float64) - scaling[1]) / scaling[0]
        if dtype == "int16":
            data = np.rint(data)
    image = nib.Nifti1Image(data.astype(dtype), affine)
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

    fixture = reference_path("runner_reference.npz")
    with np.load(fixture, allow_pickle=False) as data:
        shapes = {name: tuple(shape) for name, shape in json.loads(data["parameter_shapes"].item())}
    arrays = seeded_arrays(shapes, seed)
    mx.save_safetensors(str(path), {name: mx.array(value) for name, value in arrays.items()})
    return path
