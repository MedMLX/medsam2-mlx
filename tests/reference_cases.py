"""Small seeded MedSAM2 configuration and session scenarios shared by fixtures and tests.

This module imports neither Torch nor ``medsam2_mlx``. The scenario driver takes any
predictor with the ``SAM2VideoPredictorNPZ`` session API and returns host arrays, so
the recorder runs it on the PyTorch reference and the tests run it on the MLX port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medsam2_mlx._types import FloatArray, HostArray

from collections.abc import Callable, Iterator
from typing import Literal, Protocol, TypedDict, cast

import numpy as np

UPSTREAM_REVISION = "332f30d420f1d1b08e2a79b3ae6a602458808383"
REFERENCE_PLATFORM = "darwin-arm64"
MODEL_SEED = 2207


# A reduced sam2.1_hiera_t512 graph: 64 px images, 4x4 top-level features, Hiera
# windows that need padding (3 on a 4x4 grid), query pooling, and a global block.
class SmallConfiguration(TypedDict):
    """Dimensions of the reduced SAM 2.1 Hiera-T512 reference graph."""

    image_size: int
    embed_dim: int
    num_heads: int
    stages: tuple[int, int, int, int]
    global_att_blocks: tuple[int, ...]
    window_spec: tuple[int, int, int, int]
    window_pos_embed_bkg_spatial_size: tuple[int, int]
    d_model: int
    backbone_channel_list: tuple[int, ...]
    mem_dim: int
    memory_attention_layers: int
    dim_feedforward: int
    decoder_mlp_dim: int
    iou_head_hidden_dim: int
    num_maskmem: int
    max_obj_ptrs_in_encoder: int


SMALL: SmallConfiguration = {
    "image_size": 64,
    "embed_dim": 8,
    "num_heads": 1,
    "stages": (1, 2, 3, 1),
    "global_att_blocks": (5,),
    "window_spec": (4, 2, 3, 2),
    "window_pos_embed_bkg_spatial_size": (3, 3),
    "d_model": 16,
    "backbone_channel_list": (64, 32, 16, 8),
    "mem_dim": 8,
    "memory_attention_layers": 2,
    "dim_feedforward": 24,
    "decoder_mlp_dim": 24,
    "iou_head_hidden_dim": 16,
    "num_maskmem": 7,
    "max_obj_ptrs_in_encoder": 16,
}

VIDEO_FRAMES = 6
VIDEO_HEIGHT = 17
VIDEO_WIDTH = 23


def video_frames() -> FloatArray:
    """Prepare six consecutive HCC-TACE CT slices for the reduced graph."""
    import os
    from pathlib import Path

    import nibabel as nib_runtime
    from PIL import Image

    from medsam2_mlx._types import NiftiRuntime

    default = Path.home() / ".cache" / "medmlx" / "test-data" / "hcc_063_ct.nii.gz"
    path = os.environ.get("MEDMLX_TEST_CT", str(default))
    image = cast(NiftiRuntime, nib_runtime).load(path)
    volume = np.asarray(image.dataobj, dtype=np.float32)
    crop = volume[150:150 + VIDEO_WIDTH, 250:250 + VIDEO_HEIGHT, 40:40 + VIDEO_FRAMES]
    stack = np.transpose(crop, (2, 1, 0))
    if stack.shape != (VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH):
        raise ValueError(f"CT crop has unexpected shape {stack.shape}")
    if stack.max() == stack.min():
        raise ValueError("CT crop has no intensity variation")
    scale = (stack - stack.min()) / (stack.max() - stack.min())
    size = SMALL["image_size"]
    resized = np.stack([
        np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR))
        for frame in scale
    ]).astype(np.float32)
    return np.repeat(resized[:, None, :, :], 3, axis=1)


class ScenarioState(TypedDict):
    """Session fields inspected after the scenario resets the predictor."""

    obj_ids: list[int]
    tracking_has_started: bool


class ScenarioPredictor[StateT, MaskT](Protocol):
    """Prompt and propagation surface exercised by the shared scenario."""

    def add_new_points_or_box(
        self,
        inference_state: StateT,
        frame_idx: int,
        obj_id: int,
        points: HostArray | None = None,
        labels: HostArray | None = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: HostArray | None = None,
    ) -> tuple[int, list[int], MaskT]:
        """Add a point or box prompt to a scenario session."""
        ...
    def add_new_mask(
        self, inference_state: StateT, frame_idx: int, obj_id: int, mask: HostArray
    ) -> tuple[int, list[int], MaskT]:
        """Add a mask prompt to a scenario session."""
        ...
    def propagate_in_video(
        self,
        inference_state: StateT,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], MaskT]]:
        """Yield tracked frame outputs for the scenario."""
        ...
    def clear_all_prompts_in_frame(
        self,
        inference_state: StateT,
        frame_idx: int,
        obj_id: int,
        need_output: Literal[True] = True,
    ) -> tuple[int, list[int], MaskT]:
        """Clear an object frame prompt and return its preview."""
        ...
    def reset_state(self, inference_state: StateT) -> None:
        """Reset the scenario session."""
        ...


def session_scenario[StateT, MaskT](
    predictor: ScenarioPredictor[StateT, MaskT],
    init_state: Callable[[FloatArray, int, int], StateT],
    host: Callable[[MaskT], HostArray],
) -> dict[str, HostArray]:
    """Box and point prompts on different frames (so each object is missing from the
    other's conditioning frame), a normalized-coordinate correction, two-object
    propagation in both directions, a mask correction on a tracked frame, clearing a
    conditioning prompt, and a reset session."""
    state = init_state(video_frames(), VIDEO_HEIGHT, VIDEO_WIDTH)
    result: dict[str, HostArray] = {}

    def record(name: str, output: tuple[int, list[int], MaskT] | None) -> None:
        """Record a named scenario output for later comparison."""
        index, ids, masks = cast(tuple[int, list[int], MaskT], output)
        result[f"{name}_index"] = np.array(index)
        result[f"{name}_ids"] = np.array(list(ids))
        result[f"{name}_masks"] = host(masks)

    record(
        "box",
        predictor.add_new_points_or_box(
            state, 1, 7, box=np.array([3, 2, 19, 13], dtype=np.float32)
        ),
    )
    record(
        "point",
        predictor.add_new_points_or_box(
            state,
            3,
            9,
            points=np.array([[10, 8]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        ),
    )
    record(
        "correction",
        predictor.add_new_points_or_box(
            state,
            3,
            9,
            points=np.array([[0.35, 0.25], [0.8, 0.7]], dtype=np.float32),
            labels=np.array([1, 0], dtype=np.int32),
            clear_old_points=False,
            normalize_coords=False,
        ),
    )
    for direction, reverse in (("forward", False), ("reverse", True)):
        for output in predictor.propagate_in_video(state, start_frame_idx=1, reverse=reverse):
            record(f"{direction}_{output[0]}", output)
    yy, xx = np.mgrid[:VIDEO_HEIGHT, :VIDEO_WIDTH]
    record("mask", predictor.add_new_mask(state, 2, 7, (xx - 11) ** 2 + (yy - 8) ** 2 < 20))
    for output in predictor.propagate_in_video(state, start_frame_idx=2):
        record(f"mask_track_{output[0]}", output)
    record("clear", predictor.clear_all_prompts_in_frame(state, 3, 9))
    predictor.reset_state(state)
    result["reset"] = np.array(
        [
            len(cast(ScenarioState, state)["obj_ids"]),
            cast(ScenarioState, state)["tracking_has_started"],
        ],
        dtype=np.int32,
    )
    record(
        "after_reset",
        predictor.add_new_points_or_box(
            state, 4, 3, box=np.array([5, 4, 15, 12], dtype=np.float32)
        ),
    )
    for output in predictor.propagate_in_video(state, reverse=True, max_frame_num_to_track=2):
        record(f"after_reset_reverse_{output[0]}", output)
    return result
