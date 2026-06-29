![alt text](img/header_image.png)

# `SlicerNNInteractive`: nnInteractive meets 3D Slicer

This repository makes [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) available in [3D Slicer](https://www.slicer.org/). nnInteractive is a deep learning-based framework for interactive segmentation of 3D images, allowing for fast voxel-wise segmentation using prompts like points, scribbles, bounding boxes, and lasso. You can read more about nnInteractive in the [ArXiv paper](https://arxiv.org/abs/2503.08373), or in the original [GitHub repository](https://github.com/MIC-DKFZ/nnInteractive). 3D slicer is a free and open source medical image viewer, and can be downloaded [here](https://download.slicer.org/).

[![arXiv](https://img.shields.io/badge/arXiv-2504.07991-b31b1b.svg)](https://arxiv.org/abs/2504.07991)

![](img/segmentation_result.jpg)

## Video tutorial

https://github.com/user-attachments/assets/c9f9ee0a-f74d-4907-aa21-484dcfd10948

## Table of contents

- [Compute modes](#compute-modes)
- [Installation](#installation)
  - [Install the extension in 3D Slicer](#install-the-extension-in-3d-slicer)
  - [First run: choose Local or Remote](#first-run-choose-local-or-remote)
  - [Running the official server (Remote mode)](#running-the-official-server-remote-mode)
- [Usage](#usage)
  - [Editing an existing segment](#editing-an-existing-segment)
  - [Keyboard shortcuts](#keyboard-shortcuts)
- [Common issues](#common-issues)
- [Testing](#testing)
- [Contributing](#contributing)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Compute modes

This extension targets **nnInteractive v2** and drives the official
[`nnInteractive`](https://github.com/MIC-DKFZ/nnInteractive) inference code directly — it no
longer ships its own server. Upstream is now split into two pip packages that share the
`nnInteractive` namespace:

- **`nninteractive-client`** — a lightweight, **torch-free** remote client (`numpy` / `httpx` /
  `blosc2` only). This is all Slicer needs in Remote mode.
- **`nnInteractive`** — the **full** local-inference + server stack (PyTorch, nnU-Net, …). It
  depends on `nninteractive-client`, so a full install includes the client too.

Accordingly, you can run inference in two ways:

- **Local mode** — inference runs **in-process inside Slicer's Python** via the full
  `nnInteractive` package. No separate server is needed, but Slicer's Python must have the full
  nnInteractive + PyTorch stack, so you need a machine with a (preferably NVIDIA) GPU. 10 GB of
  VRAM is recommended; small objects work with <6 GB. CPU is supported but slow.
- **Remote mode** — Slicer is a **lightweight client** (just `nninteractive-client`, no PyTorch)
  that talks to an `nninteractive-server` running on a GPU machine (which may be the same
  computer).

You choose the mode the first time you open the module (and can change it later in the
`Configuration` tab). The extension installs only the Python packages that the selected mode
needs — **Remote mode stays PyTorch-free**. If only `nninteractive-client` is present, Local mode
stays disabled until you install the full `nnInteractive` package and restart Slicer.

## Installation

### Install the extension in 3D Slicer

1. [Download and install the latest version of **3D Slicer**](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer).
2. [Install the **nnInteractive** extension](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html#install-extensions) from the Extensions Manager.
   (For development without the Extensions Manager, see [Development](#development).)

### First run: choose Local or Remote

The first time you open the `nnInteractive` module, a dialog asks whether you want **Local GPU
compute** or a **Remote server**. Based on your choice the extension installs the matching Python
packages into Slicer's Python:

- **Remote** → installs the torch-free **`nninteractive-client`** package (plus the `httpx` /
  `blosc2` wire stack). **No PyTorch is installed.** Then enter your server URL (and API key, if
  any) in the `Configuration` tab, and click `Initialize` at the top of the `nnInteractive Prompts`
  tab.
- **Local** → installs the full **`nnInteractive`** package (nnU-Net) **and PyTorch**. Pick a
  model in the `Configuration` tab — the dropdown is populated from the available-models manifest —
  and on the first segmentation its weights are downloaded automatically from
  [Hugging Face](https://huggingface.co/MIC-DKFZ/nnInteractive). For a reliable, CUDA-matched
  PyTorch in Slicer, install the **PyTorch** extension (SlicerPyTorch) from the Extensions Manager
  first; otherwise the extension falls back to `pip install torch`.

You can switch modes later in the `Configuration` tab — switching to Local triggers the heavy
install on demand.

### Running the official server (Remote mode)

On the GPU machine, install the full package — the server is part of it, there is no separate
`[server]` extra — and start it. The server downloads the model by name on first use:

```bash
pip install nnInteractive

nninteractive-server \
    --model nnInteractive_v1.0 \
    --host 0.0.0.0 --port 1527 \
    --api-key "$(openssl rand -hex 32)"
```

List or pre-download models with `nninteractive-available-models` and `nninteractive-download-model`.
Useful server flags: `--device cuda:0`, `--no-torch-compile`, `--max-sessions N`,
`--idle-timeout-seconds`. See the official
[SERVER_CLIENT.md](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md) for full
details (authentication, SSH-tunnel setups, multi-user deployment).

#### …or run the server in Docker

If you'd rather not install anything on the GPU box, the server is also published as a Docker image
with the model **baked in** (a GPU host with the NVIDIA Container Toolkit is required):

```bash
docker run --gpus all -p 1527:1527 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:latest
```

A `lite` tag is also available if you'd rather mount your own checkpoint folder at `/model`. See
the upstream [DOCKER.md](https://github.com/MIC-DKFZ/nnInteractive/blob/master/nnInteractive/inference/server/DOCKER.md)
for both flavours and configuration.

In Slicer's `Configuration` tab, set the server URL — e.g. `http://remote_host_name:1527`, or
`http://localhost:1527` if the server runs on the same machine — and the API key, then click
`Initialize` at the top of the `nnInteractive Prompts` tab.

## Usage

Once you have completed the installation above, you can use `SlicerNNInteractive` as follows:

1. If you haven't done so already, load in your image (e.g., through dragging your image file into Slicer).

2. Click `Initialize` at the top of the `nnInteractive Prompts` tab. This is mandatory before any prompt can be placed: it loads the model (Local) or connects to the server (Remote) and uploads the current image, so your first prompt is fast. It can take a moment (the local model may run a `torch.compile` warmup; a remote session has to upload the image). The interaction tools stay disabled until initialization finishes, and the button shows `Uninitialize` once a session is live. Clicking it again — or changing the server, API key, model or any local setting — uninitializes, so you'll need to re-initialize. Only the prompt types the loaded model supports are enabled.

3. Click one of the Interaction Tool buttons from the Interactive Prompts tab (point, bounding box, scribble, or lasso) and place your prompt in the image. This should result in a segmentation.

4. Click `Show 3D` button in the segment editor section (below the prompts section) to see the segmentation results in 3D.

5. If needed, you can correct the generated segmentation with positive and negative prompts (between which you can toggle using the Positive/Negative buttons). You can undo the last interaction with `Ctrl+Z`.

	a) Alternatively, you can reset the current segment using the "Reset segment button".

6. You can add a new segment by clicking the "Next segment" button, or clicking the "+ Add" button in the Segment Editor. You can always go back to previous segments by selecting it in the Segment Editor.

### Editing an existing segment
You can edit an existing segmentation (generated using this plugin, or obtained otherwise, such as through loading in a segmentation file), by selecting the segment in the Segment Editor. Prompts are always applied to the selected segment.

### Keyboard shortcuts
Each button in the Interactive Prompts tab has a keyboard shortcut, indicated by the underlined letter. The last interaction can be undone with `Ctrl+Z`.

## Common issues

- When the remote server restarts or a session times out, the extension surfaces a "session expired" message — click `Initialize` at the top of the `nnInteractive Prompts` tab to reconnect; your current segmentation is preserved and re-seeded automatically.

## Testing

`SlicerNNInteractiveSegmentationTest` is a set of regression tests that verifies the output of `SlicerNNInteractive`. For every interaction type, it processes a set of test cases through the extension and compares the resulting segmentations against reference segmentations. All tests use the publicly available `MRBrainTumor2` volume from the `Sample Data` extension. The tests run against a **local** nnInteractive session by default (no server required); set `SLICER_NNI_TEST_SERVER_URL` to test against a running server instead.

How to run the test from Slicer:
1. Make sure `Developer Mode` is enabled in Slicer (`Edit > Application Settings > Developer`, check `Enable developer mode`).
2. Launch Slicer and (optionally) load the `SlicerNNInteractive` module via the Extension Wizard. For local testing, make sure Local mode has been set up (PyTorch + `nnInteractive[local]` installed).
3. Open the `Self Tests` module, pick `SlicerNNInteractive`, and click `Reload and Test`. A "All SlicerNNInteractive segmentation tests passed" message will appear in the Python Console if everything matches the stored references.

Reference outputs are stored at `slicer_plugin/SlicerNNInteractive/Testing/Data/` (compressed NIfTI files). You normally do not need to regenerate these. If you do (e.g. after an intentional behavior change), set `SLICER_NNI_GENERATE_TEST_MASK=1` before launching Slicer, run the test once, manually review the newly written masks, then rerun without the variable so the test compares against the frozen references.

## Contributing
Read more on how to contribute to this repository [here](CONTRIBUTING.md), while taking into account the [code of conduct](CODE_OF_CONDUCT.md).

## Development

For development, `SlicerNNInteractive` can be installed directly from github, without the Extensions Manager of 3D Slicer.

1. `git clone git@github.com:coendevente/SlicerNNInteractive.git` (or download the current project as a `.zip` file from GitHub).
2. Open 3D Slicer and click the Module dropdown menu in the top left of the 3D Slicer window:
	![Slicer dropdown menu](img/dropdown.png)
3. Go to `Developer Tools` > `Extension Wizard`.
4. Click `Select Extension`.
5. Locate the `SlicerNNInteractive` folder you obtained in Step 1, and select the `slicer_plugin` folder.
6. Go to the Module dropdown menu again and go to `Segmentation` > `SlicerNNInteractive`. This should result in the following view:
  ![First view of the Slicer extension](img/plugin_first_sight.png)
	a) If you would like to have `SlicerNNInteractive` available in the top menu (as in the image above), go to `Edit` > `Application Settings` > `Modules` and drag `SlicerNNInteractive` from the `Modules:` list to the `Favorite Modules:` list.

## Citation

When using `SlicerNNInteractive`, please cite:

1. The original `nnInteractive` paper:

	> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
	> *: equal contribution

	[![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

2. The `SlicerNNInteractive` paper:
	> de Vente, C., Venkadesh, K.V., van Ginneken, B., Sánchez, C.I. (2025). nnInteractiveSlicer: A 3D Slicer extension for nnInteractive. https://arxiv.org/abs/2504.07991

	[![arXiv](https://img.shields.io/badge/arXiv-2504.07991-b31b1b.svg)](https://arxiv.org/abs/2504.07991)

## License
This repository is available under a Apache-2.0 license (see [here](LICENSE)).

> [!IMPORTANT]
> The model weights that are downloaded when running nnInteractive are available under a `Creative Commons Attribution Non Commercial Share Alike 4.0` license, as described in the original nnInteractive repository [here](https://github.com/MIC-DKFZ/nnInteractive/tree/master?tab=readme-ov-file#license).

## Acknowledgements

This extension brings [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) into 3D Slicer.
nnInteractive is developed at the German Cancer Research Center (DKFZ) and
[Helmholtz Imaging](https://www.helmholtz-imaging.de/), who also contribute to and help maintaining this Slicer extension.

<p align="left">
  <img src="img/DKFZ_Logo.png" alt="German Cancer Research Center (DKFZ)" height="55">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="img/HI_Logo.png" alt="Helmholtz Imaging" height="55">
</p>
