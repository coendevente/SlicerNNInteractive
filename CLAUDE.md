# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
必须用中文和我交流。这个仓库是用来对这个插件进行二次开发，以满足我们自己的需要。
dev_docs目录是用来放计划文档跟随进度的，你会和codex协同工作，请合理使用说明文档交流。AGENTS.md是codex的记忆文件。
## Overview

`SlicerNNInteractive` brings [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) (deep-learning-based interactive 3D image segmentation) into [3D Slicer](https://www.slicer.org/). It has two independently deployed halves that talk over HTTP:

- **Server** (`server/`) — a FastAPI app wrapping an `nnInteractiveInferenceSession`. Needs an NVIDIA GPU (~10GB VRAM recommended). Published to PyPI as `nninteractive-slicer-server` and to Docker Hub as `coendevente/nninteractive-slicer-server`.
- **Client** (`slicer_plugin/`) — a scripted 3D Slicer extension (the `nnInteractive` module under the Segmentation category). The client may run on the same machine as the server.

## Commands

### Server

```bash
cd server
python -m build                                          # build the wheel/sdist (what CI does)
uv run nninteractive-slicer-server --host 0.0.0.0 --port 1527   # run locally
docker build -t nninteractive_slicer_server .            # build container (see build.sh)
docker run -p 1527:1527 --gpus all --rm -d nninteractive_slicer_server   # run container (see run.sh)
```

Set `NNI_DEVICE=cpu` (or `cuda:0`) to override device auto-detection. On first run the server downloads model weights from Hugging Face (`nnInteractive/nnInteractive`, `nnInteractive_v1.0`) into `.nninteractive_weights/`.

### Client extension

There is no standalone build. The extension is loaded into 3D Slicer via the Extension Wizard pointed at the `slicer_plugin/` folder, or installed from the Extensions Manager. `cmake -S . -B build` only matters when packaging as an official Slicer extension.

### Tests

The test suite (`slicer_plugin/SlicerNNInteractive/Testing/Python/SlicerNNInteractiveSegmentationTest.py`) runs **inside Slicer**, not via `pytest`. It requires a running server.

1. Start the server and configure its URL in the module's `Configuration` tab (or set `SLICER_NNI_TEST_SERVER_URL`).
2. Enable Developer Mode in Slicer.
3. Open `Self Tests`, pick `SlicerNNInteractive`, click `Reload and Test`.

It downloads the `MRBrainTumor2` sample volume, replays each prompt sequence, and compares output against frozen reference NIfTI masks in `Testing/Data/` using a Dice threshold of 0.99. To regenerate references, set `SLICER_NNI_GENERATE_TEST_MASK=1`, run once, inspect the masks, then rerun without the variable.

## Conventions

- **ASCII only in `.py` files** — CI (`check-utf8.yml`) fails on any non-ASCII character in Python source.
- **Version consistency** — the version in `server/pyproject.toml` must match the release git tag (`vX.Y.Z`) before tagging; CI warns on mismatch. Releases trigger on `v*.*.*` tags pushed to `main`.
- `server/requirements.txt` (loose ranges, used by Docker/`uv`) and `server/pyproject.toml` `dependencies` (pinned, used by the PyPI package) list the same packages — keep them in sync when bumping.

## Architecture

### Client/server protocol

Each interaction type maps to a FastAPI endpoint in `server/.../main.py`:

- `/upload_image`, `/upload_segment` — push the current volume / current segment labelmap to the server.
- `/add_point_interaction`, `/add_bbox_interaction` — JSON body (voxel coords + `positive_click`).
- `/add_lasso_interaction`, `/add_scribble_interaction` — multipart upload of a gzipped `.npy` 3D mask.

All segmentation results are returned as gzip-compressed bit-packed binary masks (`segmentation_binary` / `unpack_binary_segmentation`); the client reshapes them back to the volume shape. The server keeps a single global `PROMPT_MANAGER` holding one image + one inference session — it is **stateful and single-session**, so concurrent clients would clobber each other.

### Coordinate ordering

A recurring source of bugs: numpy volumes are ordered `(z, y, x)`, while voxel coordinates exchanged with the model are `(x, y, z)`. The client reverses coordinate lists (`xyz[::-1]`) before sending point/bbox prompts. Respect this when touching coordinate logic on either side.

### Client sync model (`slicer_plugin/.../SlicerNNInteractive.py`)

The whole client is one large `SlicerNNInteractiveWidget`. Key mechanism: the `@ensure_synched` decorator wraps every prompt method (`point_prompt`, `bbox_prompt`, `lasso_or_scribble_prompt`). Before running a prompt it diffs the current volume and selected segment against `self.previous_states` and re-uploads whichever changed. This is how the server stays in sync without the client tracking server state explicitly. If the server reports "No image uploaded", `request_to_server` transparently re-uploads and retries.

Prompts are driven by Slicer Markups nodes registered in `self.prompt_types` (point = Fiducial, bbox = ROI, lasso = ClosedCurve), each with an `on_placed` observer that fires the corresponding prompt. The **scribble** prompt is different: it uses a hidden background `qMRMLSegmentEditorWidget` with the Paint effect, and sends the *diff* of consecutive strokes.

All segmentation always applies to the currently selected segment in the embedded Segment Editor. The hidden node named `"ScribbleSegmentNode (do not touch)"` is internal scaffolding and is deliberately excluded from `get_segmentation_node()`.

The test class is imported into the module file at the bottom and exposed as `SlicerNNInteractiveTest` so Slicer's `Reload and Test` picks it up.
