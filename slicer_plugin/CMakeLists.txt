cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NNInteractive)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/coendevente/SlicerNNInteractive")
set(EXTENSION_CONTRIBUTORS "Coen de Vente (University of Amsterdam)")
set(EXTENSION_DESCRIPTION "Deep learning-based framework for interactive segmentation of 3D images. The extension is available under an Apache-2.0 license, but the weights that are being downloaded when running the SlicerNNInteractive server are available under a Creative Commons Attribution Non Commercial Share Alike 4.0 license, as described in the original nnInteractive respository.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/slicer_plugin/SlicerNNInteractive/Resources/Icons/SlicerNNInteractive.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/img/segmentation_result.jpg https://raw.githubusercontent.com/coendevente/SlicerNNInteractive/main/img/plugin_first_sight.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerNNInteractive)
## NEXT_MODULE

if(BUILD_TESTING)
  add_subdirectory(SlicerNNInteractive/Testing)
endif()

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
