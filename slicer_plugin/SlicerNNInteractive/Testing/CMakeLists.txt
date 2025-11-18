if(NOT BUILD_TESTING)
  return()
endif()

set(SCRIPT_PATHS
  Python/SlicerNNInteractiveSegmentationTest.py
)

foreach(script ${SCRIPT_PATHS})
  slicer_add_python_unittest(SCRIPT ${script})
endforeach()
