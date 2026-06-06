cmake_minimum_required(VERSION 3.12.1)

project(NNInteractive)

# Mirror extension metadata here so Slicer's Extension Wizard can open either
# the repository root or the slicer_plugin directory as the extension source.
set(EXTENSION_HOMEPAGE "https://github.com/coendevente/SlicerNNInteractive")
set(EXTENSION_CONTRIBUTORS "Coen de Vente (University of Amsterdam)")
set(EXTENSION_DESCRIPTION "Deep learning-based framework for interactive segmentation of 3D images. The extension is available under an Apache-2.0 license, but the weights that are being downloaded when running the SlicerNNInteractive server are available under a Creative Commons Attribution Non Commercial Share Alike 4.0 license, as described in the original nnInteractive respository.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/slicer_plugin/SlicerNNInteractive/Resources/Icons/SlicerNNInteractive.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/img/segmentation_result.jpg https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/img/plugin_first_sight.png")
set(EXTENSION_DEPENDS "NA")

add_subdirectory(slicer_plugin)
