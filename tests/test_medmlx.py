"""MedMLX runner: upstream-reference parity, weights admission and argument rejection.

References were recorded from the pinned upstream Torch implementation; the recorder
is not part of this repository.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import nibabel as _nib
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from PIL import Image

from medsam2_mlx import medmlx
from medsam2_mlx._types import (
    HostArray,
    NiftiRuntime,
    NumericArray,
    PromptedRunArguments,
    RunnerArguments,
    RunnerPredictor,
    RunnerResult,
)
from tests.real_images import xray_planes
from tests.reference_cases import REFERENCE_PLATFORM, UPSTREAM_REVISION
from tests.runner_cases import (
    FAKE_RUNS,
    NETWORK_RUNS,
    VOLUMES,
    RunnerCase,
    prepare_inputs,
    reference_path,
    write_seeded_weights,
    write_volume,
)

nib: NiftiRuntime = cast(NiftiRuntime, _nib)

@pytest.fixture(scope="module")
def reference() -> dict[str, HostArray]:
    """Load runner fixtures and verify their recorded source revision and platform."""
    with np.load(reference_path("runner_reference.npz"), allow_pickle=False) as data:
        assert data["recorded_platform"].item() == REFERENCE_PLATFORM
        assert data["upstream_revision"].item() == UPSTREAM_REVISION
        assert data["ct_source"].item() == "HCC-TACE-Seg HCC_063 contrast CT"
        assert data["xray_source"].item() == "Wikimedia Commons normal PA chest radiograph"
        return {key: data[key] for key in data.files}


def _arguments(run: RunnerCase, volume: Path, mask: Path | None) -> PromptedRunArguments:
    """Translate one runner case into run arguments; several objects go through ``prompts``."""
    prompt: dict[str, object] = {"frame_idx": run["frame"], "obj_id": run.get("obj_id", 1)}
    if "box" in run:
        prompt["box_xyxy"] = list(run["box"])
    if "points_xy" in run:
        prompt["points_xy"] = [list(point) for point in run["points_xy"]]
        prompt["point_labels"] = list(run.get("point_labels", ()))
    if mask is not None:
        prompt["mask_path"] = str(mask)
    arguments: dict[str, object] = {"image_path": str(volume), "propagation": run["propagation"]}
    if "slice_range" in run:
        arguments["slice_range"] = list(run["slice_range"])
    extra = [
        {"frame_idx": run["frame"], "obj_id": item["obj_id"], "box_xyxy": list(item["box"])}
        for item in run.get("extra_objects", ())
    ]
    if extra:
        arguments["prompts"] = [prompt, *extra]
    else:
        arguments.update(prompt)
    return cast(PromptedRunArguments, arguments)


def _check_outputs(
    result: RunnerResult, volume: Path, output_dir: Path, expected: dict[str, HostArray]
) -> None:
    """Compare the written label volume and overlay with the recorded upstream case."""
    outputs = result["outputs"]
    assert set(outputs) == ({"mask", "overlay"} if expected["masks"].shape[0] == 1 else {"mask"})
    for path in outputs.values():
        assert Path(path).parent == output_dir.resolve()
    source = nib.load(volume)
    labels = nib.load(outputs["mask"])
    assert labels.shape == source.shape
    assert np.array_equal(labels.affine, source.affine)
    assert labels.get_data_dtype() == np.uint16
    data = np.asanyarray(labels.dataobj)
    assert data.dtype == np.uint16
    # The upstream label volume is in SimpleITK (z, y, x) order.
    assert np.array_equal(data.T.reshape(expected["masks"].shape), expected["masks"])
    # The CT example's mask overlay; point prompts also draw markers, so skip those.
    if "overlay" in outputs and not any(p["points_xy"] for p in result["details"]["prompts"]):
        _, stack = medmlx.load_image_stack(volume)
        base = np.repeat(stack[0, :, :, None], 3, axis=2).astype(np.float32)
        selected = cast(NumericArray, expected["masks"])[0] > 0
        base[selected] = 0.5 * base[selected] + 0.5 * np.array([251, 252, 30])
        with Image.open(outputs["overlay"]) as overlay:
            assert np.array_equal(np.asarray(overlay), np.clip(base, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("name", list(VOLUMES))
def test_nifti_stacks_match_upstream_ct_reader(
    tmp_path: Path, reference: dict[str, HostArray], name: str
) -> None:
    """Compare NIfTI stack loading with the recorded source reader."""
    _, stack = medmlx.load_image_stack(write_volume(tmp_path, name))
    expected = reference[f"stack/{name}"]
    assert stack.dtype == expected.dtype == np.uint8
    assert np.array_equal(stack, expected)


def test_preprocessing_matches_upstream_ct(reference: dict[str, HostArray]) -> None:
    """Compare CT frame preprocessing with the recorded source result."""
    prepared = medmlx.prepare_medsam2_images(reference["stack/slice2d"])
    assert prepared.dtype == np.float32 and prepared.shape == (1, 3, 512, 512)
    assert np.array_equal(prepared, reference["prepared/slice2d"])


@pytest.fixture(scope="module")
def seeded_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write seeded small-network weights for runner comparisons."""
    return write_seeded_weights(tmp_path_factory.mktemp("weights") / "medsam2.safetensors")


@pytest.mark.parametrize("name", list(FAKE_RUNS) + list(NETWORK_RUNS))
@pytest.mark.parametrize("postprocessing", [False, True], ids=["raw", "post"])
def test_run_with_seeded_network_matches_upstream(
    tmp_path: Path,
    seeded_weights: Path,
    name: str,
    postprocessing: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare seeded runner results with recorded upstream cases."""
    fixture = reference_path("runner_run.npz")
    prefix = f"{'post' if postprocessing else 'raw'}/{name}/"
    with np.load(fixture, allow_pickle=False) as data:
        assert data["recorded_platform"].item() == REFERENCE_PLATFORM
        assert data["upstream_revision"].item() == UPSTREAM_REVISION
        assert data["ct_source"].item() == "HCC-TACE-Seg HCC_063 contrast CT"
        assert data["xray_source"].item() == "Wikimedia Commons normal PA chest radiograph"
        expected = {
            key.removeprefix(prefix): data[key] for key in data.files if key.startswith(prefix)
        }
    run = {**FAKE_RUNS, **NETWORK_RUNS}[name]
    volume, mask = prepare_inputs(tmp_path / "inputs", run)
    output_dir = tmp_path / "out"
    original = medmlx.mask_result
    observed: list[tuple[int, HostArray]] = []

    def capture(index: int, ids: Sequence[int], logits: HostArray) -> medmlx.MaskResult:
        """Record native predictor construction for the seeded runner."""
        observed.append((index, np.asarray(logits).copy()))
        return original(index, ids, logits)

    monkeypatch.setattr(medmlx, "mask_result", capture)
    result = medmlx.run(
        **_arguments(run, volume, mask),
        weights_path=seeded_weights,
        output_dir=output_dir,
        apply_postprocessing=postprocessing,
    )
    _check_outputs(result, volume, output_dir, expected)
    assert np.array_equal([index for index, _ in observed], expected["frame_order"])
    for (_, masks), scores in zip(observed, expected["logits"], strict=True):
        assert masks.dtype == np.bool_
        assert masks.itemsize == 1
        np.testing.assert_array_equal(masks[0], scores > 0)


def test_publication_failure_restores_prior_mask_and_overlay(
    tmp_path: Path, seeded_weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FAKE_RUNS["slice2d"]
    volume, mask = prepare_inputs(tmp_path / "inputs", run)
    output = tmp_path / "out"
    output.mkdir()
    old = {
        output / medmlx.MASK_NAME: b"old mask",
        output / medmlx.OVERLAY_NAME: b"old overlay",
    }
    for path, contents in old.items():
        path.write_bytes(contents)
    replace = os.replace
    failed = False

    def fail_overlay(source: Path, destination: Path) -> None:
        nonlocal failed
        if destination == output / medmlx.OVERLAY_NAME and not failed:
            failed = True
            raise OSError("injected publication failure")
        replace(source, destination)

    monkeypatch.setattr(medmlx.os, "replace", fail_overlay)
    with pytest.raises(OSError, match="injected publication failure"):
        medmlx.run(
            **_arguments(run, volume, mask),
            weights_path=seeded_weights, output_dir=output, overwrite=True,
        )
    for path, contents in old.items():
        assert path.read_bytes() == contents
    assert not list(output.glob(".medsam2-*"))


def test_multiframe_run_removes_prior_overlay(
    tmp_path: Path, seeded_weights: Path
) -> None:
    run = FAKE_RUNS["both"]
    volume, mask = prepare_inputs(tmp_path / "inputs", run)
    output = tmp_path / "out"
    output.mkdir()
    overlay = output / medmlx.OVERLAY_NAME
    overlay.write_bytes(b"overlay from a prior single-frame run")

    result = medmlx.run(
        **_arguments(run, volume, mask),
        weights_path=seeded_weights, output_dir=output, overwrite=True,
    )

    assert set(result["outputs"]) == {"mask"}
    assert not overlay.exists()


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param("MedSAM2_CTLesion.pt", id="unconverted"),
        pytest.param("wrong.safetensors", id="wrong-keys"),
        pytest.param("corrupt.safetensors", id="corrupt"),
    ],
)
def test_mismatched_weights_raise_asset_not_ready(tmp_path: Path, weights: str) -> None:
    """Reject unconverted, corrupt or wrong-inventory weights as unavailable assets."""
    (tmp_path / "MedSAM2_CTLesion.pt").write_bytes(b"torch checkpoint")
    if weights == "wrong.safetensors":
        from medsam2_mlx._runtime import import_mlx

        mx = import_mlx()
        mx.save_safetensors(str(tmp_path / weights), {"x": mx.zeros((2,))})
    (tmp_path / "corrupt.safetensors").write_bytes(b"\x00" * 16)
    with pytest.raises(AssetNotReadyError):
        medmlx.run(
            image_path=write_volume(tmp_path, "slice2d"),
            weights_path=tmp_path / weights,
            output_dir=tmp_path / "out",
            box_xyxy=[2, 3, 20, 15],
        )


@pytest.mark.parametrize(
    "problem",
    [
        "bad-suffix",
        "out-of-range",
        "non-bool",
        "zero-stop",
        "existing-output",
        "dangling-output",
        "object-id-overflow",
        "prompt-object-id-overflow",
    ],
)
def test_invalid_arguments_are_rejected_before_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each category of bad argument fails before the weights are requested."""

    def download(_model_id: str) -> Path:
        """Fail if the runner asks for weights before checking its arguments."""
        pytest.fail("weights were requested before the arguments were checked")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    output = tmp_path / "out"
    arguments: dict[str, object] = {
        "image_path": write_volume(tmp_path, "slice2d"),
        "output_dir": output,
        "box_xyxy": [2, 3, 20, 15],
    }
    if problem == "bad-suffix":
        arguments["image_path"] = tmp_path / "slice.png"
        (tmp_path / "slice.png").write_text("not an image", encoding="utf-8")
    elif problem == "out-of-range":
        arguments["frame_idx"] = 1
    elif problem == "non-bool":
        arguments["overwrite"] = "yes"
    elif problem == "zero-stop":
        arguments["stop_after_empty"] = 0
    elif problem == "object-id-overflow":
        arguments["obj_id"] = 65536
    elif problem == "prompt-object-id-overflow":
        arguments.pop("box_xyxy")
        arguments["prompts"] = [{"obj_id": 65536, "box_xyxy": [2, 3, 20, 15]}]
    else:
        output.mkdir()
        mask_path = output / medmlx.MASK_NAME
        if problem == "dangling-output":
            mask_path.symlink_to(tmp_path / "missing.nii.gz")
        else:
            mask_path.write_bytes(b"previous")
    with pytest.raises(InvalidInputError):
        medmlx.run(**cast(RunnerArguments, arguments))


class _ScriptedPredictor:
    """Yields one-object masks whose foreground follows ``occupied``; counts frames produced."""

    def __init__(self, occupied: Sequence[bool]) -> None:
        """Store the per-frame foreground pattern."""
        self.occupied = occupied
        self.produced = 0

    def propagate_in_video(
        self,
        inference_state: None,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], HostArray]]:
        """Yield ``(frame, ids, logits)`` lazily, as the predictor's generator does."""
        for index, present in enumerate(self.occupied):
            self.produced += 1
            yield index, [1], np.full((1, 1, 2, 2), 1.0 if present else -1.0, np.float32)


@pytest.mark.parametrize(("stop", "produced"), [(None, 8), (2, 6), (3, 8)])
def test_stop_after_empty_ends_tracking_after_consecutive_empty_frames(
    stop: int | None, produced: int
) -> None:
    """Tracking stops after ``stop`` consecutive empty frames and computes no later frames."""
    predictor = _ScriptedPredictor([True, True, False, True, False, False, True, True])
    session = medmlx.Session(
        predictor=cast("RunnerPredictor[None]", predictor),
        inference_state=None,
        image_stack=(xray_planes((8, 2, 2)) * np.float32(255)).astype(np.uint8),
    )
    results = session.propagate(stop_after_empty=stop)
    assert predictor.produced == produced
    assert [result.frame_idx for result in results] == list(range(produced))
