"""Record the reduced MedSAM2 source graph on HCC-TACE image crops."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from tests.reference_cases import ScenarioPredictor, session_scenario, video_frames
from tests.reference_types import TorchRuntime
from tests.upstream_reference import build_reference, import_upstream, seeded_state

type HostArray = npt.NDArray[np.generic]
type SourceState = dict[str, object]
type HeadOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
TORCH = cast(TorchRuntime, torch)


class ArchiveWriter(Protocol):
    def __call__(self, path: Path, **arrays: HostArray) -> None: ...


class HeadCall(Protocol):
    def __call__(
        self, backbone_features: Tensor, point_inputs: dict[str, Tensor] | None = None,
        mask_inputs: Tensor | None = None, high_res_features: list[Tensor] | None = None,
        multimask_output: bool = False,
    ) -> HeadOutput: ...


class DirectCall(Protocol):
    def __call__(
        self, backbone_features: Tensor, high_res_features: list[Tensor], mask_inputs: Tensor,
    ) -> HeadOutput: ...


class MemoryCall(Protocol):
    def __call__(
        self, current_vision_feats: list[Tensor], feat_sizes: list[tuple[int, int]],
        pred_masks_high_res: Tensor, object_score_logits: Tensor, is_mask_from_pts: bool,
    ) -> tuple[Tensor, list[Tensor]]: ...


class Decoder(Protocol):
    """The upstream decoder options configured by the video builder."""

    dynamic_multimask_via_stability: bool
    dynamic_multimask_stability_delta: float
    dynamic_multimask_stability_thresh: float


class Predictor(ScenarioPredictor[SourceState, Tensor], Protocol):
    """Upstream predictor methods used to record the reduced graph."""

    sam_mask_decoder: Decoder
    fill_hole_area: int
    binarize_mask_from_pts_for_mem_enc: bool

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> object: ...
    def eval(self) -> object: ...
    def init_state(self, images: Tensor, video_height: int, video_width: int) -> SourceState: ...
    def forward_image(self, image: Tensor) -> dict[str, list[Tensor]]: ...
    def _forward_sam_heads(
        self, backbone_features: Tensor, point_inputs: dict[str, Tensor] | None = None,
        mask_inputs: Tensor | None = None, high_res_features: list[Tensor] | None = None,
        multimask_output: bool = False,
    ) -> HeadOutput: ...
    def _use_mask_as_output(
        self, backbone_features: Tensor, high_res_features: list[Tensor], mask_inputs: Tensor,
    ) -> HeadOutput: ...
    def _encode_new_memory(
        self, current_vision_feats: list[Tensor], feat_sizes: list[tuple[int, int]],
        pred_masks_high_res: Tensor, object_score_logits: Tensor, is_mask_from_pts: bool,
    ) -> tuple[Tensor, list[Tensor]]: ...


class CvRuntime(Protocol):
    CC_STAT_AREA: int

    def connectedComponentsWithStats(
        self, image: npt.NDArray[np.uint8], *, connectivity: int,
    ) -> tuple[int, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]: ...


class MiscRuntime(Protocol):
    get_connected_components: object


def install_cpu_components() -> None:
    """Replace the upstream CUDA component extension with OpenCV on CPU."""
    cv2 = cast(CvRuntime, import_module("cv2"))
    misc = cast(MiscRuntime, import_module("sam2.utils.misc"))

    def components(mask: Tensor) -> tuple[Tensor, Tensor]:
        labels = np.zeros(mask.shape, dtype=np.int32)
        areas = np.zeros(mask.shape, dtype=np.int32)
        for index in range(mask.shape[0]):
            pixels = mask[index, 0].cpu().numpy().astype(np.uint8)
            _, component, stats, _ = cv2.connectedComponentsWithStats(pixels, connectivity=8)
            labels[index, 0] = component
            areas[index, 0] = stats[component, cv2.CC_STAT_AREA]
        return TORCH.from_numpy(labels), TORCH.from_numpy(areas)

    misc.get_connected_components = components


def record(source: Path, output: Path) -> None:
    """Run source modules and sessions, retaining tensor-only fixed inputs."""
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise ValueError("Record MedSAM2 references on macOS arm64")
    torch.set_num_threads(4)
    upstream = import_upstream(source.resolve())
    install_cpu_components()
    model = cast(Predictor, build_reference(upstream))
    model.load_state_dict(seeded_state(upstream, TORCH), strict=True)
    model.eval()
    numeric = Path(__file__).parent / "fixtures" / "reference_numeric.npz"
    with np.load(numeric, allow_pickle=False) as base:
        result: dict[str, HostArray] = {key: base[key] for key in base.files}
    if str(result["meta_torch_version"].item()) != str(torch.__version__):
        raise ValueError("Installed Torch does not match the retained reference tensors")
    result["ct_source"] = np.array("HCC-TACE-Seg HCC_063 contrast CT")
    frames = video_frames()

    def tensor(key: str) -> Tensor:
        return TORCH.from_numpy(result[key])

    with torch.inference_mode():
        interpolation = TORCH.from_numpy(frames[:2, :1, :7, :9].transpose(1, 0, 2, 3).copy())
        result["interp/input"] = interpolation.numpy()
        for name, size, antialias in (
            ("up", (19, 13), False), ("down", (3, 4), False),
            ("down_aa", (3, 4), True), ("up_aa", (19, 13), True),
        ):
            value = torch.nn.functional.interpolate(
                interpolation, size=size, mode="bilinear", align_corners=False,
                antialias=antialias,
            )
            result[f"interp/bilinear_{name}"] = value.numpy()
        result["interp/bicubic"] = torch.nn.functional.interpolate(
            interpolation, size=(16, 16), mode="bicubic", align_corners=False,
        ).numpy()
        pixels = TORCH.from_numpy(frames[:2].copy())
        result["image/input"] = pixels.numpy()
        image = model.forward_image(pixels)
        for key in ("backbone_fpn", "vision_pos_enc"):
            for index, value in enumerate(image[key]):
                result[f"image/{key}_{index}"] = value.numpy()
        pix, high = image["backbone_fpn"][2], image["backbone_fpn"][:2]
        coordinates, labels, masks = (
            tensor("heads/point_coords"), tensor("heads/point_labels"), tensor("heads/mask_inputs")
        )
        methods = {"heads": "_forward_sam_heads", "direct": "_use_mask_as_output",
                   "memory": "_encode_new_memory"}
        heads = cast(HeadCall, getattr(model, methods["heads"]))
        outputs = {
            "points_multi": heads(
                pix, point_inputs={"point_coords": coordinates[:, :1],
                                   "point_labels": labels[:, :1]},
                high_res_features=high, multimask_output=True,
            ),
            "points_single": heads(
                pix, point_inputs={"point_coords": coordinates, "point_labels": labels},
                high_res_features=high,
            ),
            "mask": heads(pix, mask_inputs=masks, high_res_features=high),
        }
        for case, values in outputs.items():
            for key, index in (("low_res_multimasks", 0), ("ious", 2), ("low_res_masks", 3),
                               ("obj_ptr", 5), ("object_score_logits", 6)):
                result[f"heads/{case}/{key}"] = values[index].numpy()
        direct_call = cast(DirectCall, getattr(model, methods["direct"]))
        direct = direct_call(pix, high, (masks > 0).float())
        result["heads/mask_as_output/low_res_masks"] = direct[0].numpy()
        result["heads/mask_as_output/obj_ptr"] = direct[5].numpy()
        model.binarize_mask_from_pts_for_mem_enc = True
        for flag in (False, True):
            features, position = cast(MemoryCall, getattr(model, methods["memory"]))(
                [pix.flatten(2).permute(2, 0, 1)], [(4, 4)],
                tensor("memenc/pred_masks_high_res"), tensor("memenc/object_score_logits"), flag,
            )
            result[f"memenc/features_{flag}"] = features.numpy()
            result[f"memenc/pos_{flag}"] = position[0].numpy()
        for post in (False, True):
            model.fill_hole_area = 8 if post else 0
            model.binarize_mask_from_pts_for_mem_enc = post
            model.sam_mask_decoder.dynamic_multimask_via_stability = post
            model.sam_mask_decoder.dynamic_multimask_stability_delta = 0.05
            model.sam_mask_decoder.dynamic_multimask_stability_thresh = 0.98

            def initialize(images: npt.NDArray[np.float32], height: int, width: int) -> SourceState:
                return model.init_state(TORCH.from_numpy(images), height, width)

            def host(value: Tensor) -> HostArray:
                return value.cpu().numpy()

            session = session_scenario(model, initialize, host)
            result.update({f"{'post' if post else 'raw'}/{key}": value
                           for key, value in session.items()})
    output.mkdir(parents=True, exist_ok=True)
    cast(ArchiveWriter, np.savez_compressed)(output / "reference_small.npz", **result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medsam2-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    record(arguments.medsam2_source, arguments.output_dir)


if __name__ == "__main__":
    main()
