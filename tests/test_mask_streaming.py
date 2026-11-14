"""Keep runner masks bounded while preserving ordering, stopping and session reuse."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
import pytest

from medsam2_mlx import medmlx
from medsam2_mlx._types import BoolArray, HostArray, Propagation, RunnerPredictor
from medsam2_mlx.request import MedSAM2Prompt


class RecordedPredictor:
    """Supply deterministic model outputs at the predictor boundary, without a GPU."""

    def __init__(self, occupied: Sequence[bool], shape: tuple[int, int] = (2, 3)) -> None:
        self.occupied = occupied
        self.shape = shape
        self.ids: list[int] = []
        self.produced: list[tuple[bool, int]] = []
        self.resets = 0

    def _frame(self, index: int, reverse: bool) -> tuple[int, list[int], HostArray]:
        logits = np.full((len(self.ids), 1, *self.shape), -1.0, dtype=np.float32)
        if self.occupied[index]:
            logits[0, 0, 0, :2] = 1.0
            if len(self.ids) > 1:
                logits[1, 0, 0, :1] = 1.0 if reverse else -1.0
                logits[1, 0, 0, 1:3] = -1.0 if reverse else 1.0
        return index, self.ids.copy(), logits

    def add_new_points_or_box(
        self,
        inference_state: None,
        frame_idx: int,
        obj_id: int,
        points: HostArray | None = None,
        labels: HostArray | None = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: HostArray | None = None,
    ) -> tuple[int, list[int], HostArray]:
        if obj_id not in self.ids:
            self.ids.append(obj_id)
        return self._frame(frame_idx, False)

    def propagate_in_video(
        self,
        inference_state: None,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], HostArray]]:
        start = 0 if start_frame_idx is None else start_frame_idx
        frames = range(start, -1, -1) if reverse else range(start, len(self.occupied))
        for offset, index in enumerate(frames):
            if max_frame_num_to_track is not None and offset > max_frame_num_to_track:
                break
            self.produced.append((reverse, index))
            yield self._frame(index, reverse)

    def reset_state(self, inference_state: None) -> None:
        self.ids.clear()
        self.resets += 1


def make_session(predictor: RecordedPredictor) -> medmlx.Session[None]:
    """Bind the three predictor methods exercised by an already initialized Session."""
    return medmlx.Session(
        cast(RunnerPredictor[None], predictor),
        None,
        np.zeros((len(predictor.occupied), *predictor.shape), dtype=np.uint8),
    )


def test_runner_does_not_retain_a_volume_of_completed_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = RecordedPredictor([True] * 64)
    session = make_session(predictor)
    original = medmlx.mask_result
    references: list[weakref.ReferenceType[BoolArray]] = []
    live_counts: list[int] = []

    def observe(index: int, ids: Sequence[int], logits: HostArray) -> medmlx.MaskResult:
        # Instrument ownership only; the real conversion produces every result unchanged.
        result = original(index, ids, logits)
        references.append(weakref.ref(result.masks))
        live_counts.append(sum(reference() is not None for reference in references))
        return result

    monkeypatch.setattr(medmlx, "mask_result", observe)
    labels = medmlx.run_prompted_session(
        session,
        prompt_tuple=(MedSAM2Prompt(frame_idx=32, obj_id=7, box_xyxy=(0, 0, 2, 1)),),
        propagation="both",
        max_frame_num_to_track=None,
        slice_range=None,
    )
    expected = np.zeros((64, 2, 3), dtype=np.uint16)
    expected[:, 0, :2] = 7
    np.testing.assert_array_equal(labels, expected)
    assert max(live_counts) <= 2, "Completed frame masks must be released as propagation advances"
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("mode", ["none", "forward", "reverse", "both"])
def test_runner_preserves_labels_stopping_and_direction_reset(mode: Propagation) -> None:
    predictor = RecordedPredictor([True, False, False, True, True, True, False, False, True])
    session = make_session(predictor)
    prompts = tuple(MedSAM2Prompt(frame_idx=4, obj_id=i, box_xyxy=(0, 0, 2, 1)) for i in (7, 3))
    labels = medmlx.run_prompted_session(
        session,
        prompt_tuple=prompts,
        propagation=mode,
        max_frame_num_to_track=None,
        slice_range=None,
        stop_after_empty=2,
    )
    expected = np.zeros((9, 2, 3), dtype=np.uint16)
    expected[4, 0] = [7, 3, 3]
    produced: list[tuple[bool, int]] = []
    if mode in {"forward", "both"}:
        expected[5, 0] = [7, 3, 3]
        produced += [(False, 4), (False, 5), (False, 6), (False, 7)]
    if mode in {"reverse", "both"}:
        expected[3, 0] = [3, 7, 0]
        expected[4, 0] = [3, 7, 3]  # Reverse stamps only foreground; seed background is retained.
        produced += [(True, 4), (True, 3), (True, 2), (True, 1)]
    np.testing.assert_array_equal(labels, expected)
    assert predictor.produced == produced
    assert predictor.resets == (1 if mode == "both" else 0)
    # The public list-returning API and subsequent corrections still use the same state.
    session.add_prompt(MedSAM2Prompt(frame_idx=3, obj_id=7, box_xyxy=(0, 0, 2, 1)))
    corrected = session.propagate(start_frame_idx=3, max_frame_num_to_track=1)
    assert isinstance(corrected, list)
    assert [result.frame_idx for result in corrected] == [3, 4]
    assert [result.obj_ids for result in corrected] == [(7, 3), (7, 3)]


def test_streamed_directions_respect_slice_and_frame_limits() -> None:
    predictor = RecordedPredictor([True] * 9)
    labels = medmlx.run_prompted_session(
        make_session(predictor),
        prompt_tuple=(MedSAM2Prompt(frame_idx=4, obj_id=7, box_xyxy=(0, 0, 2, 1)),),
        propagation="both",
        max_frame_num_to_track=2,
        slice_range=(3, 7),
    )
    expected = np.zeros((9, 2, 3), dtype=np.uint16)
    expected[3:7, 0, :2] = 7
    np.testing.assert_array_equal(labels, expected)
    assert predictor.produced == [(False, 4), (False, 5), (False, 6), (True, 4), (True, 3)]
