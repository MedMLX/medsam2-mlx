"""Seeded small-graph equivalence with the PyTorch MedSAM2 reference recording.

Every expected array was recorded from upstream PyTorch MedSAM2 @ 332f30d; module
inputs come from the same recording, so each module is compared in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

    from medsam2_mlx.sam2_base import MedSAM2Model

import numpy as np
import pytest

from medsam2_mlx import mlx_model
from medsam2_mlx._ops import interpolate_bicubic
from medsam2_mlx._runtime import mx
from medsam2_mlx._types import HeadArguments, HostArray, PackedBFloat16, SessionState
from medsam2_mlx.mlx_predictor import build_predictor
from medsam2_mlx.request import MedSAM2RuntimeConfig
from tests.conftest import load_small_model, relative_error
from tests.reference_cases import session_scenario

# Module outputs agree to float32 rounding (about 1e-6); this bound leaves margin.
MODULE_RTOL = 1e-4


@pytest.fixture(scope="module")
def model(state_dict: dict[str, HostArray]) -> MedSAM2Model:
    """Load the seeded small model for source-reference comparisons."""
    return load_small_model(state_dict, apply_postprocessing=False)


@pytest.mark.parametrize(
    ("name", "size", "antialias"),
    [
        ("up", (19, 13), False),
        ("down", (3, 4), False),
        ("down_aa", (3, 4), True),
        ("up_aa", (19, 13), True),
    ],
)
def test_bilinear_resize_matches_torch(
    reference: dict[str, HostArray], name: str, size: tuple[int, int], antialias: bool
) -> None:
    """Compare bilinear resizing with the recorded source outputs."""
    actual = mlx_model.resize_mask(reference["interp/input"], size, antialias=antialias)
    assert relative_error(actual, reference[f"interp/bilinear_{name}"]) < 1e-6


def test_bicubic_resize_matches_torch(reference: dict[str, HostArray]) -> None:
    """Compare bicubic resizing with the recorded source output."""
    actual = interpolate_bicubic(mx.array(reference["interp/input"]), (16, 16))
    assert relative_error(actual, reference["interp/bicubic"]) < 1e-6


def test_image_encoder_matches_reference(
    model: MedSAM2Model, reference: dict[str, HostArray]
) -> None:
    """Compare each image feature and positional encoding with the source recording."""
    out = model.forward_image(mx.array(reference["image/input"]))
    for index in range(3):
        for key in ("backbone_fpn", "vision_pos_enc"):
            expected = reference[f"image/{key}_{index}"]
            assert relative_error(out[key][index], expected) < MODULE_RTOL, (key, index)


def _decoder_inputs(reference: dict[str, HostArray]) -> tuple[array, list[array]]:
    """Extract image and high-resolution features from the source fixture."""
    pix = mx.array(reference["image/backbone_fpn_2"])
    high_res = [
        mx.array(reference["image/backbone_fpn_0"]),
        mx.array(reference["image/backbone_fpn_1"]),
    ]
    return pix, high_res


@pytest.mark.parametrize("case", ["points_multi", "points_single", "mask"])
def test_prompt_encoder_and_mask_decoder_match_reference(
    model: MedSAM2Model, reference: dict[str, HostArray], case: str
) -> None:
    """Compare prompt and mask decoder outputs with source fixtures."""
    pix, high_res = _decoder_inputs(reference)
    coords = mx.array(reference["heads/point_coords"])
    labels = mx.array(reference["heads/point_labels"])
    cases: dict[str, HeadArguments] = {
        "points_multi": {
            "point_inputs": {"point_coords": coords[:, :1], "point_labels": labels[:, :1]},
            "multimask_output": True,
        },
        "points_single": {"point_inputs": {"point_coords": coords, "point_labels": labels}},
        "mask": {"mask_inputs": mx.array(reference["heads/mask_inputs"])},
    }
    kwargs = cases[case]
    outputs = model.forward_sam_heads(pix, high_res_features=high_res, **kwargs)
    actual = dict(
        zip(
            ("low_res_multimasks", "ious", "low_res_masks", "obj_ptr", "object_score_logits"),
            (outputs[0], outputs[2], outputs[3], outputs[5], outputs[6]),
            strict=True,
        )
    )
    for key, value in actual.items():
        assert relative_error(value, reference[f"heads/{case}/{key}"]) < MODULE_RTOL, key


def test_mask_prompt_as_output_matches_reference(
    model: MedSAM2Model, reference: dict[str, HostArray]
) -> None:
    """Compare the direct mask-prompt output path with the source recording."""
    pix, high_res = _decoder_inputs(reference)
    mask = (mx.array(reference["heads/mask_inputs"]) > 0).astype(mx.float32)
    outputs = model.use_mask_as_output(pix, high_res, mask)
    expected = reference["heads/mask_as_output/low_res_masks"]
    assert relative_error(outputs[0], expected) < MODULE_RTOL
    expected = reference["heads/mask_as_output/obj_ptr"]
    assert relative_error(outputs[5], expected) < MODULE_RTOL


def test_memory_attention_matches_reference(
    model: MedSAM2Model, reference: dict[str, HostArray]
) -> None:
    """Compare conditioned memory attention with the source recording."""
    actual = model.memory_attention(
        mx.array(reference["memattn/curr"]),
        mx.array(reference["memattn/memory"]),
        mx.array(reference["memattn/curr_pos"]),
        mx.array(reference["memattn/memory_pos"]),
        num_obj_ptr_tokens=4,
    )
    assert relative_error(actual, reference["memattn/output"]) < MODULE_RTOL


@pytest.mark.parametrize("binarize", [False, True])
def test_memory_encoder_matches_reference(
    model: MedSAM2Model, reference: dict[str, HostArray], binarize: bool
) -> None:
    """Compare memory encoding with and without mask binarization to source outputs."""
    features, position = model.encode_mask_memory(
        mx.array(reference["image/backbone_fpn_2"]),
        mx.array(reference["memenc/pred_masks_high_res"]),
        mx.array(reference["memenc/object_score_logits"]),
        binarize=binarize,
    )
    assert relative_error(features, reference[f"memenc/features_{binarize}"]) < MODULE_RTOL
    assert relative_error(position, reference[f"memenc/pos_{binarize}"]) < MODULE_RTOL


def _compare_masks(actual: HostArray, expected: HostArray) -> tuple[float, float]:
    """Relative error over real logits, and sign agreement over all pixels.

    The -1024 absent-object sentinel is compared absolutely and excluded from the scale.
    """
    observed = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    assert observed.shape == reference.shape
    sentinel = reference <= -1000
    assert np.all(np.abs(observed[sentinel] - reference[sentinel]) < 1e-2)
    real = ~sentinel
    rel = 0.0
    if real.any():
        delta = float(np.max(np.abs(observed[real] - reference[real])))
        rel = delta / float(np.max(np.abs(reference[real])))
    return rel, np.count_nonzero((observed > 0) == (reference > 0)) / observed.size


# Observed: about 3e-6 on Metal. Propagated frames read spatial memory that
# upstream stores in BF16, where a float32 rounding difference can move a value by one
# BF16 step; this small graph does not amplify such steps beyond the module bound.
SESSION_RTOL = 1e-4


@pytest.mark.parametrize("offload", [False, True], ids=["device", "offloaded"])
@pytest.mark.parametrize("postprocessing", [False, True], ids=["raw", "post"])
def test_session_matches_reference(
    state_dict: dict[str, HostArray],
    reference: dict[str, HostArray],
    postprocessing: bool,
    offload: bool,
) -> None:
    """Compare session prompt and propagation results with recorded source cases."""
    model = load_small_model(state_dict, apply_postprocessing=postprocessing)
    predictor = build_predictor(
        MedSAM2RuntimeConfig(apply_postprocessing=postprocessing), model=model
    )

    def init_state(images: HostArray, height: int, width: int) -> SessionState:
        """Initialize a reference-sized session for one test scenario."""
        return predictor.init_state(
            images,
            height,
            width,
            offload_video_to_cpu=offload,
            offload_state_to_cpu=offload,
        )

    actual = session_scenario(predictor, init_state, np.asarray)
    prefix = "post/" if postprocessing else "raw/"
    expected_keys = {key.removeprefix(prefix) for key in reference if key.startswith(prefix)}
    assert set(actual) == expected_keys
    for key, value in actual.items():
        expected = reference[prefix + key]
        if not key.endswith("_masks"):
            assert np.array_equal(value, expected), key
            continue
        rel, agreement = _compare_masks(value, expected)
        assert rel < SESSION_RTOL, (key, rel)
        assert agreement == 1.0, (key, agreement)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
@pytest.mark.parametrize("offload", [False, True], ids=["device", "host"])
def test_monotonic_history_preserves_tracking_and_corrections(
    model: MedSAM2Model, reverse: bool, offload: bool
) -> None:
    """Match retained history beyond both lookbacks while keeping prompts and corrections."""
    from tests.reference_cases import video_frames

    images = np.tile(video_frames(), (6, 1, 1, 1))
    start = len(images) - 1 if reverse else 0
    direction = -1 if reverse else 1
    mask = np.zeros((17, 23), dtype=np.bool_)
    mask[4:12, 6:16] = True
    results: list[list[HostArray]] = []
    for retain in (True, False):
        predictor = build_predictor(MedSAM2RuntimeConfig(), model=model, retain_history=retain)
        state = predictor.init_state(
            images, 17, 23, offload_video_to_cpu=True, offload_state_to_cpu=offload
        )
        for obj_id in (1, 2):
            predictor.add_new_mask(state, start, obj_id, mask)
        # A second conditioning frame must survive even after it leaves the rings.
        predictor.add_new_mask(state, start + direction * 3, 1, mask)
        outputs: list[HostArray] = []
        for _, _, masks in predictor.propagate_in_video(
            state, start_frame_idx=start, max_frame_num_to_track=12, reverse=reverse
        ):
            outputs.append(masks.copy())
        index = start + direction * 12
        predictor.add_new_mask(state, index, 1, ~mask)
        for _, _, masks in predictor.propagate_in_video(
            state, start_frame_idx=index + direction, reverse=reverse
        ):
            outputs.append(masks.copy())
        results.append(outputs)
        if offload:
            memory = state["output_dict"]["cond_frame_outputs"][start]["maskmem_features"]
            assert isinstance(memory, PackedBFloat16)
            assert memory.bits.dtype == np.uint16
            assert memory.bits.nbytes == memory.bits.size * 2
        if not retain:
            protected = state["consolidated_frame_inds"]["non_cond_frame_outputs"]
            for banks in [state["output_dict"], *state["output_dict_per_obj"].values()]:
                automatic = {
                    key: value
                    for key, value in banks["non_cond_frame_outputs"].items()
                    if key not in protected
                }
                assert len(automatic) <= model.max_obj_ptrs_in_encoder - 1
                assert sum(v["maskmem_features"] is not None for v in automatic.values()) <= (
                    model.num_maskmem - 1
                )
                assert start in banks["cond_frame_outputs"]
                assert start + direction * 3 in banks["cond_frame_outputs"]
                assert protected <= banks["non_cond_frame_outputs"].keys()
            with pytest.raises(ValueError, match="reset"):
                next(predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse))
            predictor.reset_state(state)
            predictor.add_new_mask(state, start, 1, mask)
            assert (
                next(predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse))[0]
                == start
            )
    assert len(results[0]) == len(results[1]) == len(images)
    for retained, bounded in zip(*results, strict=True):
        np.testing.assert_array_equal(retained, bounded)


def test_lazy_session_starts_at_prompt_and_defers_bad_frame(model: MedSAM2Model) -> None:
    """Allow a valid later prompt without reading an unrelated malformed frame zero."""
    from tests.reference_cases import video_frames

    class Frames:
        def __init__(self) -> None:
            self.values = video_frames()
            self.visited: list[int] = []

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> HostArray:
            self.visited.append(index)
            return np.zeros((3, 1, 1), dtype=np.float32) if index == 0 else self.values[index]

    predictor = build_predictor(MedSAM2RuntimeConfig(), model=model)
    frames = Frames()
    state = predictor.init_state(frames, 17, 23, offload_video_to_cpu=True)
    assert frames.visited == []
    mask = np.zeros((17, 23), dtype=np.bool_)
    mask[4:12, 6:16] = True
    index, ids, masks = predictor.add_new_mask(state, 4, 1, mask)
    assert index == 4 and ids == [1] and np.isfinite(masks).all()
    assert frames.visited == [4]
    with pytest.raises(ValueError, match="shape"):
        predictor.add_new_mask(state, 0, 1, mask)
    with pytest.raises(ValueError, match="shape"):
        predictor.init_state(Frames(), 17, 23, offload_video_to_cpu=True, eager_first_frame=True)
