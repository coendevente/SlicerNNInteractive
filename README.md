![alt text](img/header_image.png)

# `SlicerNNInteractive`: nnInteractive meets 3D Slicer

This repository makes [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) available in [3D Slicer](https://www.slicer.org/). nnInteractive is a deep learning-based framework for interactive segmentation of 3D images, allowing for fast voxel-wise segmentation using prompts like points, scribbles, bounding boxes, and lasso. You can read more about nnInteractive in the [ArXiv paper](https://arxiv.org/abs/2503.08373), or in the original [GitHub repository](https://github.com/MIC-DKFZ/nnInteractive). 3D slicer is a free and open source medical image viewer, and can be downloaded [here](https://download.slicer.org/).

[![arXiv](https://img.shields.io/badge/arXiv-2504.07991-b31b1b.svg)](https://arxiv.org/abs/2504.07991)

![](img/segmentation_result.jpg)

## Video tutorial

https://github.com/user-attachments/assets/c9f9ee0a-f74d-4907-aa21-484dcfd10948

## Table of contents

- [Installation](#installation)
  - [Installation in 3D Slicer](#installation-in-3d-slicer)
  - [External server setup (macOS and advanced use)](external-server-setup.md)
- [Usage](#usage)
  - [Editing an existing segment
](#editing-an-existing-segment)
  - [Keyboard shortcuts](#keyboard-shortcuts)
- [Common issues](#common-issues)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

Hardware requirements: `nnInteractive` computations require a Windows or Linux computer with an NVIDIA GPU. 10GB of VRAM is recommended. Small objects should work with <6GB. nnInteractive supports Python 3.10+> (source: [The nnInteractive README](https://github.com/MIC-DKFZ/nnInteractive?tab=readme-ov-file#prerequisites). macOS users can run Slicer with the nnInteractive extension on their computer but they need access to a Linux or Windows computer and [set up an external server application](external-server-setup.md).

1. [Download and install latest version of **3D Slicer**](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer)
2. [Install **NNInteractive** extension](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html#install-extensions)
3. On computers that do not have a suitable GPU for running nnInteractive (such as macOS computers): [set up an external server application](external-server-setup.md) on a suitable computer.

## Usage

Once you have completed the installation above, you can use `NNInteractive` as follows:

1. If you haven't done so already, load in your image (e.g., by dragging your image file into Slicer application window).

2. Click one of the Interaction Tool buttons from the Interactive Prompts tab (point, bounding box, scribble, or lasso) and place your prompt in the image. This should result in a segmentation.

3. Click `Show 3D` button in the segment editor section (below the prompts section) to see the segmentation results in 3D.

4. If needed, you can correct the generated segmentation with positive and negative prompts (between which you can toggle using the Positive/Negative buttons).

	a) Alternatively, you can reset the current segment using the "Reset segment button".

5. You can add a new segment by clicking the "Next segment" button, or clicking the "+ Add" button in the Segment Editor. You can always go back to previous segments by selecting it in the Segment Editor.

### Editing an existing segment
You can edit an existing segmentation (generated using this plugin, or obtained otherwise, such as through loading in a segmentation file), by selecting the segment in the Segment Editor. Prompts are always applied to the selected segment.

### Keyboard shortcuts
Each button in the Interactive Prompts tab has a keyboard shortcut, indicated by the underlined letter.

## Common issues

- The computation server may fail silently. Reloading the plugin or restarting Slicer often helps. If this does not solve the problem then stop the server process (a python executable with significant memory usage) or restart the computer.

## Developers

### Testing

`SlicerNNInteractiveSegmentationTest` is a set of regression tests that verifies the output of `SlicerNNInteractive`. For every interaction type, it processes a set of test cases through the extension – which requires a running server – and compares the resulting segmentations against reference segementations. All tests use the publicly available `MRBrainTumor2` volume from the `Sample Data` extension.

How to run the test from Slicer:
1. Start the nnInteractive server and note its URL/port.
2. Launch Slicer, (optionally) load the `SlicerNNInteractive` module via the Extension Wizard, and configure the module with the server URL (under the `Configuration` tab).
3. Make sure `Developer Mode` is enabled in Slicer. You can verify this by going to `Edit > Application Settings > Developer`, and making sure `Enable developer mode:` is checked.
4. Open the `Self Tests` module, pick `SlicerNNInteractive`, and click `Reload and Test` (or use the toolbar’s `Reload and Test` button in the module itself). Slicer will re-import the module, execute the scripted prompts, and a "All SlicerNNInteractive segmentation tests passed" message will be in the Python Console if everything matches the stored references.

Reference outputs are stored at `slicer_plugin/SlicerNNInteractive/Testing/Data/` (compressed NIfTI files). When running these tests, you do not have to regenerate these. If, for any reason you would still like to do so, set `SLICER_NNI_GENERATE_TEST_MASK=1` before launching Slicer (or uncomment the line `self.generate_mode = True` in `SlicerNNInteractiveSegmentationTest.setUp`), run the test once, manually review the newly written masks, then rerun without the variable so the test compares against the frozen references.

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
> The weights that are being downloaded when running the `SlicerNNInteractive` server are available under a `Creative Commons Attribution Non Commercial Share Alike 4.0` license, as described in the original nnInteractive respository [here](https://github.com/MIC-DKFZ/nnInteractive/tree/master?tab=readme-ov-file#license).
