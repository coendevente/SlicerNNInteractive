set(MODULE_NAME "SlicerNNInteractive")

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/prompts/dark/bbox_icon.svg
  Resources/Icons/prompts/dark/lasso_icon.svg
  Resources/Icons/prompts/dark/point_icon.svg
  Resources/Icons/prompts/dark/scribble_icon.svg
  Resources/Icons/prompts/light/bbox_icon.svg
  Resources/Icons/prompts/light/lasso_icon.svg
  Resources/Icons/prompts/light/point_icon.svg
  Resources/Icons/prompts/light/scribble_icon.svg
  Resources/UI/${MODULE_NAME}.ui
  Testing/Python/SlicerNNInteractiveSegmentationTest.py
  Testing/Data/MRBrainTumor2_lasso_lasso_tumor.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_brain.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_brain_bbox.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_full_brain.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_full_brain_with_negative.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_lasso_tumor.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_left_eye.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_right_eye.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_scribble_brain.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_scribble_tumor.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_tumor.nii.gz
  Testing/Data/MRBrainTumor2_point_prompt_tumor_bbox.nii.gz
  Testing/Data/MRBrainTumor2_scribble_scribble_brain.nii.gz
  Testing/Data/MRBrainTumor2_scribble_scribble_tumor.nii.gz
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
