---
title: 'nnInteractiveSlicer: A 3D Slicer extension for nnInteractive'
tags:
  - Python
  - Slicer
  - nnInteractive
  - Efficient annotation
authors:
  - name: Coen de Vente
    orcid: 0000-0001-5908-8367
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Kiran Vaidhya Venkadesh
    equal-contrib: false
    affiliation: 3
    orcid: 0000-0002-4846-9049
  - name: Bram van Ginneken
    affiliation: 3
    orcid: 0000-0003-2028-8972
  - name: Clara I. Sánchez
    affiliation: 3
    affiliation: "1, 2"
    orcid: 0000-0001-9787-8319
affiliations:
 - name: Quantitative Healthcare Analysis (qurAI) Group, Informatics Institute, University of Amsterdam, Amsterdam, The Netherlands
   index: 1
 - name: Amsterdam UMC location University of Amsterdam, Biomedical Engineering and Physics, Amsterdam, The Netherlands
   index: 2
 - name: Diagnostic Image Analysis Group (DIAG), Department of Radiology and Nuclear Medicine, Radboud UMC, Nijmegen, The Netherlands
   index: 3
date: 7 April 2025
bibliography: paper.bib

---

# Summary
`nnInteractiveSlicer` integrates `nnInteractive` [@isensee2025nninteractive], a state-of-the-art promptable deep learning-based framework for 3D image segmentation, into the widely used `3D Slicer` platform. Our extension implements a client-server architecture that decouples computationally intensive model inference from the client-side interface. Therefore, `nnInteractiveSlicer` eliminates heavy hardware constraints on the client-side and enables better operating system compatibility than existing plugins for `nnInteractive`. Running both the client and server-side on a single machine is also possible, offering flexibility across different deployment scenarios. The extension provides an intuitive user interface with all interaction types available in the original framework (point, bounding box, scribble, and lasso prompts), while including a comprehensive set of keyboard shortcuts for efficient workflow.

# Statement of Need

Segmentation is a cornerstone of medical image analysis. Recently, `nnInteractive` [@isensee2025nninteractive], a deep learning-based framework allowing for fast, promptable segmentation of 3D medical images was released and was shown to substantially outperform existing approaches, such as SAM2 [@kirillov2023segment], SegVol [@du2024segvol], and SAM-Med-3D [@wang2023sam]. Alongside the `nnInteractive` model, plugins in the medical image viewers MITK [@MITK_Team_MITK_2024] and Napari [@Sofroniew2025-ty] were published. However, the original authors did not make an extension available for `3D Slicer`, a widely used viewer and processing environment in medical imaging research. Furthermore, these existing plugins require substantial computational resources on the machine of the image viewer itself (an NVIDIA GPU with at least 10 GB of VRAM is recommended), as these plugins do not facilitate the deployment of the backend on a separate server. Moreover, `nnInteractive` only runs on Windows and Linux, so the image viewer cannot be run on MacOS machines.

`nnInteractiveSlicer` decouples the computationally intensive `nnInteractive` inference by allowing users to configure a remote server (e.g., a node of a GPU cluster), while running the client on a machine with lower computational capabilities. This approach not only broadens platform compatibility, but also addresses the resource constraints of existing plugins, making `nnInteractive` more widely available and potentially accelerating research related to promptable segmentation.

# Overview of `nnInteractiveSlicer`

## nnInteractive

While foundation models such as SAM [@ravi2024sam] and SAM2 [@kirillov2023segment] have shown promising interactive segmentation performance in 2D natural images, their lack of volumetric awareness and the domain shift from natural to medical data resulted in limited utility in 3D medical imaging contexts. `nnInteractive` addresses these issues through an nnUNet-based architecture [@isensee2021nnu] with residual encoders [@isensee2024nnu] that supports diverse interation types: point, bounding box, scribble, and lasso prompts. Trained on over 120 diverse volumetric datasets across multiple modalities (CT, MRI, PET, 3D microscopy), the framework demonstrated high accuracy and versatility. Our implementation extends this capability to `3D Slicer`.

## Availability and Installation
`nnInteractiveSlicer` is available through multiple channels. The server-side is available through Docker Hub (`docker pull coendevente/nninteractive-slicer-server:latest`), Pip (`pip install nninteractive-slicer-server`), and GitHub ([https://github.com/coendevente/nninteractive-slicer](https://github.com/coendevente/nninteractive-slicer)). The client-side is currently only available through our [GitHub repository](https://github.com/coendevente/nninteractive-slicer). In future versions of this extension, we plan to include it in the official 3D Slicer Extensions Manager.

## Client-server Setup
`nnInteractiveSlicer` uses a client-server setup, which decouples the computationally intensive model inference from the `3D Slicer` client. The server-side and client-side communicate through FastAPI endpoints. The client maintains synchronization between the image and input mask in `3D Slicer`. An overview of the API is shown in \autoref{fig:api_overview}.

![API overview.\label{fig:api_overview}](img/nni_api.pdf){width="80%"}

## User Interface

The user interface of `nnInteractiveSlicer` largely follows the `nnInteractive` Napari and MITK plugins. A screenshot of the user interface, including segmentation results, is shown in \autoref{fig:screenshot}. A video showcasing the functionalities of the extension is available [here](https://www.youtube.com/watch?v=mW_fUT1-IWM).

![Screenshot of the `nnInteractiveSlicer` extension.\label{fig:screenshot}](img/screenshot.png)

The sidebar of the user interface consists of a menu with the tabs _nnInteractive Prompts_ and _Configuration_, and the _Segment Editor_. The _Configuration_ tab allows the user to change the Server URL. This URL is saved in `3D Slicer`'s settings, which will be remembered in future sessions. The _nnInteractive Prompts_ menu consists of the following sections:

- **Segment buttons:** The _Reset segment_ button removes all prompts from the current segment and deletes the current segmentation on the server and client-side. The _Next segment_ button creates a new empty segment in the _Segment Editor_.

- **Prompt Type:** These _Positive_ and _Negative_ buttons manage whether the provided prompt will be interpreted as a positive or negative prompt, respectively.

- **Interaction Tools:** The four buttons in this section activate or deactivate the interaction tools. When a prompt type is activated, the user can place the prompt in the image. When a prompt has been placed, the client synchronizes the image and the segment to the server if needed, and sends the prompt to the server. The server subsequently processes the prompt and sends the updated segmentation back. When a prompt has been placed and processed, a new prompt of the same type can be placed immediately.

Each button in the _nnInteractive Prompts_ menu has an associated keyboard shortcut, which is indicted using the underlined letters within the button text.

If a segment is selected in the _Segment Editor_, prompts will always be applied to that segment. Every time a user has switched segments, the associated segmentation is uploaded to server and used as input mask to the `nnInteractive` model. When no segment is selected, a new segment is created automatically.

# References
