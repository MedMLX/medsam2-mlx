<div align="center">

# medsam2-mlx

Segment lesions or organs from a box, point or mask prompt, then track them through a volume with MedSAM2.

[![PyPI](https://img.shields.io/pypi/v/medsam2-mlx)](https://pypi.org/project/medsam2-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/medsam2-mlx)](https://pypi.org/project/medsam2-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/medsam2-mlx/blob/v0.3.0/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/medsam2-ct-lesion-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/MedMLX/medsam2-ct-lesion-mlx) · [MedMLX](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/medsam2-mlx/main/docs/showcase.png" alt="A lung nodule mask from one box prompt, propagated 5 mm above and below the prompted slice" width="100%">

<sub>One box prompt on a [LUNA16](https://zenodo.org/records/3723295) chest CT, CC BY 4.0, propagated through the nodule.</sub>

</div>

| Input | Output |
| :--- | :--- |
| Scan + slice-level prompt | Tracked masks on the input grid |

| Model | Weights download | Peak MLX memory | Measured on |
| :--- | ---: | ---: | :--- |
| `medsam2` | 156 MB | 1.3 GB | One box prompt tracked through 21 slices of a 512×512×147 chest CT |

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `medsam2` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe medsam2`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "medsam2-mlx>=0.3.0"
```

## Run

```python
from medsam2_mlx.medmlx import run

result = run(
    image_path="ct.nii.gz", output_dir="out",
    box_xyxy=[120, 140, 180, 200], frame_idx=40, propagation="both",
)
print(result["outputs"])
```

Weights download from `MedMLX/medsam2-ct-lesion-mlx` on first use; `medmlx setup medsam2`
downloads them ahead of time. For local weights, pass a folder containing
`MedSAM2_CTLesion.safetensors`, or a checkpoint file, as `weights_path`. The published
[weight repository](https://huggingface.co/MedMLX/medsam2-mlx) also contains the other
four MedSAM2 checkpoints.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | NIfTI CT volume or slice, `.npz` RECIST stack, or folder of `.jpg`/`.jpeg` frames |
| `output_dir` | Directory for the outputs; created if missing |
| `weights_path` | Optional folder containing `MedSAM2_CTLesion.safetensors`, or a `.safetensors` file |
| `box_xyxy` | `[x_min, y_min, x_max, y_max]` on `frame_idx`; combinable with `points_xy` |
| `points_xy`, `point_labels` | `[[x, y], ...]` on `frame_idx`, each labelled `1` foreground or `0` background |
| `mask_path` | Binary seed NIfTI with the input's shape and affine, instead of a box or points |
| `frame_idx`, `obj_id` | Zero-based prompt slice (default `0`) and positive object label (default `1`) |
| `prompts` | Prompts for several objects, replacing the single-object arguments |
| `hu_window` | Optional `(lower, upper)` clip before min-max scaling to uint8 |
| `propagation` | `"none"` (default), `"forward"`, `"reverse"` or `"both"` |
| `slice_range` | Optional inclusive `[start, end]` that contains every prompt's slice |
| `stop_after_empty` | `8` by default: stop each propagation direction after that many consecutive slices with no foreground; `None` tracks the whole volume as upstream does |
| `apply_postprocessing` | `True` by default: upstream stability fallback, binarized click memory and hole filling |
| `overwrite` | `False` by default; existing outputs are never replaced |

**Prompts**

- For NIfTI, `frame_idx` uses the third voxel axis, and box and point coordinates use the
  first two stored voxel axes. Coordinates are checked for shape and finiteness but not
  clipped.
- Each `prompts` entry is a mapping with a required `obj_id` and any of `frame_idx`,
  `box_xyxy`, `points_xy`, `point_labels`, `mask_path`, `clear_old_points` and
  `normalize_coords`. IDs must be unique and fit in uint16.
- With propagation, every prompt must be on the same frame. `"both"` resets and reseeds
  each object before the reverse pass, as the upstream CT examples do.

**Preprocessing**

| Input | Preprocessing |
| :--- | :--- |
| NIfTI volume | Optionally clipped to `hu_window` and min-max scaled to uint8; every frame is then bicubic-resized to 512×512 and ImageNet-normalized |
| `.npz` | Follows the upstream RECIST script: `imgs` in `[0, 256)` is rounded and clipped to uint8, never min-max scaled, with an optional `spacing` |
| Frame folder | Follows the upstream demo's video path; frames are read as grayscale, so color frames differ from upstream |

`.npz` stacks and frame folders have no NIfTI grid, so `hu_window` and `mask_path` are rejected for them.

**Outputs**

- `medsam2_mask.nii.gz`: uint16 labels with the input's shape, affine and header, plus
  `medsam2_overlay.png` for single-slice inputs.
- Where objects overlap, the later object wins. Masks are not reduced to their largest
  3D connected component, which upstream's CT script does for each lesion.
- `run` returns `{"outputs": {...}, "details": {...}}`, with run metadata in `details`.
- Invalid inputs raise `InvalidInputError`; missing or mismatched weights raise
  `AssetNotReadyError`; inference failures raise `ModelExecutionError`.

### Python API

```python
from pathlib import Path

import numpy as np
from medsam2_mlx import MedSAM2RuntimeConfig, build_predictor

config = MedSAM2RuntimeConfig(checkpoint=Path("MedSAM2_CTLesion.safetensors"))
predictor = build_predictor(config)
# images: finite FP32 (frames, 3, 512, 512); height, width: original slice size
state = predictor.init_state(images, height, width, offload_video_to_cpu=True)
predictor.add_new_points_or_box(state, 0, 1, box=np.array([x0, y0, x1, y1], np.float32))
for frame_idx, ids, logits in predictor.propagate_in_video(state):
    masks = logits > 0
```

- The predictor follows upstream `SAM2VideoPredictorNPZ`: box, point and mask prompts,
  corrections on tracked frames, multiple objects, forward and reverse propagation,
  `clear_all_prompts_in_frame`, `remove_object` and `reset_state`.
- Logits are host FP32 `(objects, 1, H, W)`; coordinates are original pixels unless
  `normalize_coords=False`.
- Weights must hold the 471 FP32 tensors of `sam2.1_hiera_t512`, which all five released
  MedSAM2 checkpoints share.
- Inference requires Apple Silicon with Metal.

## License

[Apache-2.0](https://github.com/MedMLX/medsam2-mlx/blob/v0.3.0/LICENSE). See [third-party notices](https://github.com/MedMLX/medsam2-mlx/blob/v0.3.0/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms; see the [model card](https://huggingface.co/MedMLX/medsam2-mlx).
