"""Chest X-ray crops for image-operation tests."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image


def xray_planes(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    """Resize whole-image and quadrant crops to the requested trailing spatial shape."""
    default = Path.home() / ".cache/medmlx/test-data/chest_xray.jpg"
    path = Path(os.environ.get("MEDMLX_TEST_XRAY", str(default)))
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        boxes = (
            (0, 0, width, height),
            (0, 0, width // 2, height // 2),
            (width // 2, 0, width, height // 2),
            (0, height // 2, width // 2, height),
            (width // 2, height // 2, width, height),
        )
        size = (shape[-1], shape[-2])
        planes = [
            np.asarray(image.crop(box).resize(size), dtype=np.float32) / np.float32(255)
            for box in boxes
        ]
    count = int(np.prod(shape[:-2]))
    return np.stack([planes[index % len(planes)] for index in range(count)]).reshape(shape)
