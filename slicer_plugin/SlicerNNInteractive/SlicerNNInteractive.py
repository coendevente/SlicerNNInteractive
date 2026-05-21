import io
import gzip
import requests
import copy
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from vtkmodules.util.numpy_support import vtk_to_numpy

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox


###############################################################################
# Decorators and utility functions
###############################################################################


DEBUG_MODE = False


def debug_print(*args):
    if DEBUG_MODE:
        print(*args)


def ensure_synched(func):
    """
    Decorator that ensures the image and segment are synced before calling
    the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        failed_to_sync = False

        if self.image_changed():
            debug_print(
                "Image changed (or not previously set). Calling upload_segment_to_server()"
            )
            result = self.upload_image_to_server()

            failed_to_sync = result is None

        if not failed_to_sync and self.selected_segment_changed():
            debug_print(
                "Segment changed (or not previously set). Calling upload_segment_to_server()"
            )
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()

            failed_to_sync = result is None
        else:
            debug_print("Segment did not change!")

        if not failed_to_sync:
            return func(self, *args, **kwargs)

    return inner


###############################################################################
# SlicerNNInteractive
###############################################################################


class SlicerNNInteractive(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("nnInteractive")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente", "Kiran Vaidhya Venkadesh", "Bram van Ginneken", "Clara I. Sanchez"]
        self.parent.helpText = """
            This is an 3D Slicer extension for using nnInteractive.

            Read more about this plugin here: https://github.com/coendevente/SlicerNNInteractive.
            """
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""


###############################################################################
# SlicerNNInteractiveWidget
###############################################################################


class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    # Every historical name a magic wand seed node has had. Used to sweep
    # orphans on Clear Seeds / setup so reloads do not leak stale fiducials.
    _WAND_SEED_NODE_NAMES = (
        "SelectionOpWandSeeds",          # current
        "SelectionOpWandSeedsPositive",  # multi-seed v1 (positive)
        "SelectionOpWandSeedsNegative",  # multi-seed v1 (negative)
        "SelectionOpWandSeed",           # original single-point
    )

    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        # Operation ROI used as an alternative operand for Selection Operations.
        self._sel_op_roi_node = None
        # Sphere/ellipsoid visualization for the operation ROI.
        self._sel_op_roi_preview_node = None
        self._sel_op_roi_preview_transform_node = None
        # Magic wand seeds: a multi-point Fiducial node feeding nnInteractive's
        # point prompt.
        self._sel_op_wand_seed_node = None
        # Live preview of the magic wand region (hidden segmentation node).
        self._sel_op_wand_preview_segment_node = None
        self._wand_preview_segment_id = None
        # Selection Operations-private undo stack: list of (segment_id, mask_uint8).
        self._sel_op_undo_stack = []
        self._sel_op_undo_stack_limit = 10

    def setup(self):
        """
        Overridden setup method. Initializes UI and setups up prompts.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.install_dependencies()

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SlicerNNInteractive.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"
        self.wand_preview_segment_node_name = "MagicWandPreviewSegmentNode (do not touch)"

        # Set up editor widget
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        # Use the same segmentation parameter node as the Segment Editor core module
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
                "button_icon_filename": "lasso_icon.svg",
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

        # (numpy_axis, center) describing the slice plane of the last lasso
        # prompt. Set only by lasso_points_to_mask and consumed (reset to None)
        # by show_segmentation, so only lasso results get slice-range clipped.
        self._last_lasso_slice = None

        # Sweep any orphaned magic wand seed nodes left behind by earlier
        # versions / earlier reloads so the scene is clean on module load.
        self._destroy_wand_seed()

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to an empty string if not set)
        savedServer = slicer.util.settingsValue("SlicerNNInteractive/server", "")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)
        self.ui.pbTestServer.clicked.connect(self.test_server_connection)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.ui.pbInteractionLassoCancel.setVisible(False)
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.ui.pbInteractionLassoCancel.clicked.connect(self.on_lasso_cancel_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)

        # Selection operations (boolean editing) and manual server sync
        self.ui.pbSyncToServer.clicked.connect(self.on_sync_to_server_clicked)
        self.ui.pbApplySelectionOp.clicked.connect(self.on_apply_selection_op_clicked)
        self.ui.cbOperandSource.currentIndexChanged.connect(self._on_operand_source_changed)
        self.ui.cbRoiShape.currentIndexChanged.connect(self._on_roi_shape_changed)
        self.ui.pbPlaceRoi.clicked.connect(self.on_place_roi_clicked)
        self.ui.pbClearRoi.clicked.connect(self.on_clear_roi_clicked)
        self.ui.pbPlaceWandSeed.clicked.connect(self.on_place_wand_seed_clicked)
        self.ui.pbClearWandSeed.clicked.connect(self.on_clear_wand_seed_clicked)
        self.ui.pbPreviewWand.clicked.connect(self.on_preview_wand_clicked)
        self.ui.pbClearPreviewWand.clicked.connect(self.on_clear_preview_wand_clicked)
        self.ui.pbUndoSelectionOp.clicked.connect(self.on_undo_selection_op_clicked)
        self.ui.sldSegmentOpacity.valueChanged.connect(self._on_segment_opacity_changed)
        self.ui.cbEnableLassoClip.toggled.connect(self._on_lasso_clip_enabled_changed)
        self.ui.sbLassoClipN.valueChanged.connect(self._on_lasso_clip_n_changed)
        # Load persisted lasso-clip prefs into the widgets (block signals so
        # setting the value does not immediately re-save it).
        blocked = self.ui.cbEnableLassoClip.blockSignals(True)
        self.ui.cbEnableLassoClip.setChecked(self._get_lasso_clip_enabled())
        self.ui.cbEnableLassoClip.blockSignals(blocked)
        blocked = self.ui.sbLassoClipN.blockSignals(True)
        self.ui.sbLassoClipN.setValue(self._get_lasso_clip_n())
        self.ui.sbLassoClipN.blockSignals(blocked)
        self.populate_operand_selector()
        self._install_selection_op_observers()
        # Initialize operand-row visibility and Apply enable state for the
        # default source.
        self._on_operand_source_changed(self.ui.cbOperandSource.currentIndex)
        self._sync_opacity_slider_from_segment()

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "Shift+L": self.submit_lasso_if_present,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}

        for shortcut_key, shortcut_event in shortcuts.items():
            debug_print(f"Added shortcut for {shortcut_key}: {shortcut_event}")
            shortcut = qt.QShortcut(
                qt.QKeySequence(shortcut_key), slicer.util.mainWindow()
            )
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut

    def remove_shortcut_items(self):
        """
        Called at cleanup to remove all the shortcuts we attached.
        """
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None

    def install_dependencies(self):
        """
        Checks for (and installs if needed) python packages needed by the module.
        """
        dependencies = {
            "requests_toolbelt": "requests_toolbelt",
            "skimage": "scikit-image",
            "matplotlib": "matplotlib",
        }

        for dependency in dependencies:
            if self.check_dependency_installed(dependency, dependencies[dependency]):
                continue
            self.run_with_progress_bar(
                self.pip_install_wrapper,
                (dependencies[dependency],),
                "Installing dependencies: %s" % dependency,
            )

    def check_dependency_installed(self, import_name, module_name_and_version):
        """
        Checks if a package is installed with the correct version.
        """
        if "==" in module_name_and_version:
            module_name, module_version = module_name_and_version.split("==")
        else:
            module_name = module_name_and_version
            module_version = None

        spec = importlib.util.find_spec(import_name)
        if spec is None:
            # Not installed
            return False

        if module_version is not None:
            import importlib.metadata as metadata
            try:
                version = metadata.version(module_name)
                if version != module_version:
                    # Version mismatch
                    return False
            except metadata.PackageNotFoundError:
                debug_print(f"Could not determine version for {module_name}.")

        return True

    def pip_install_wrapper(self, command, event):
        """
        Installs pip packages.
        """
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        """
        Runs a function in a background thread, while showing a progress bar in the UI
        as a pop up window.
        """
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        self.removeObservers()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(
                prompt_type["name"]
            ):
                debug_print("Skipping", prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )

            self.prev_caller = None

            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    prompt_type["on_placed_function"],
                )

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)) 
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}"))
            prompt_type["button"].setIcon(icon)

        if (
            not skip_if_exists
            or slicer.mrmlScene.GetFirstNodeByName(self.scribble_segment_node_name)
            is None
        ):
            self.setup_scribble_prompt()

            self.ui.pbInteractionScribble.setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )
            self.all_prompt_buttons["scribble"] = self.ui.pbInteractionScribble

        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def setup_scribble_prompt(self):
        """
        Creates a hidden "Segment Editor" for the scribble prompt.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        # Create a background (headless) segment editor
        self.scribble_editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        # Create a separate SegmentEditorNode
        self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)

        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self.get_volume_node()
        )
        self.scribble_segment_node.SetName(self.scribble_segment_node_name)

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "bg", "bg", [0.0, 0.0, 1.0]
        )
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "fg", "fg", [0.0, 0.0, 1.0]
        )
        dn = self.scribble_segment_node.GetDisplayNode()

        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)

        self._prev_scribble_mask = None
            
        light_dark_mode = self.is_ui_dark_or_light_mode()
        icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/scribble_icon.svg"))
        self.ui.pbInteractionScribble.setIcon(icon)

    def is_ui_dark_or_light_mode(self):
        # Returns whether the current appearance of the UI is dark mode (will return "dark")
        # or light mode (will return "light")
        current_style = slicer.app.settings().value("Styles/Style")

        if current_style == "Dark Slicer":
            return "dark"
        elif current_style == "Light Slicer":
            return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            window_color = app_palette.color(QPalette.Active, QPalette.Window)
            lightness = window_color.lightness()
            dark_mode_threshold = 128

            if lightness < dark_mode_threshold:
                return "dark"
            else:
                return "light"
        return "light"

    def remove_prompt_nodes(self):
        """
        Removes all the Markups/Fiducials prompts.
        """

        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])

        self.ui.pbInteractionLassoCancel.setVisible(False)

        _remove(self.scribble_segment_node_name)

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                if prompt_type["name"] == "LassoPrompt" and (self.ui.pbInteractionLasso.isChecked()):
                    self.submit_lasso_if_present()
                prompt_type["button"].setChecked(False)
            elif interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
            self.ui.pbInteractionScribble.setChecked(False)

    def remove_all_but_last_prompt(self):
        """
        Removes all but the most recently placed markup points
        (helpful when segment change was detected).
        """
        last_modified_node = None
        all_nodes = []

        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)

                    all_nodes.append(node)
                    if (
                        last_modified_node is None
                        or node.GetMTime() > last_modified_node.GetMTime()
                    ):
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()

            if node == last_modified_node:
                if node.GetName() == "LassoPrompt":
                    continue
                n -= 1

            for i in range(n):
                node.RemoveNthControlPoint(0)

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            if prompt_name == "lasso":
                self.submit_lasso_if_present()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node.
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 0.0, 1.0)  # Green color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    def display_node_markup_lasso(self, display_node):
        """
        Handles the appearance of the lasso display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetGlyphScale(1)
        display_node.SetLineThickness(0.3)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        xyz = self.xyz_from_caller(caller)

        volume_node = self.get_volume_node()
        if volume_node:
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        """
        Uploads point prompt to the server.
        """
        url = f"{self.server}/add_point_interaction"

        seg_response = self.request_to_server(
            url, json={"voxel_coord": xyz[::-1], "positive_click": positive_click}
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        debug_print("unpacked_segmentation.sum():", unpacked_segmentation.sum())
        debug_print(seg_response)
        debug_print(f"{positive_click} point prompt triggered! {xyz}")

        self.show_segmentation(unpacked_segmentation)

    #
    #  -- Bounding Box
    #
    def on_bbox_placed(self, caller, event):
        """
        Every time a control point is placed/moved for the bounding box ROI node.
        Once two corners are placed, we send the bounding box to the server.
        """
        xyz = self.xyz_from_caller(caller)

        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self.get_volume_node()
            if volume_node:
                outer_point_two = self.prev_bbox_xyz

                outer_point_one = [
                    xyz[0] * 2 - outer_point_two[0],
                    xyz[1] * 2 - outer_point_two[1],
                    xyz[2] * 2 - outer_point_two[2],
                ]

                self.bbox_prompt(
                    outer_point_one=outer_point_one,
                    outer_point_two=outer_point_two,
                    positive_click=self.is_positive,
                )

                def _next():
                    self.setup_prompts()
                    # Start placing a new box
                    self.ui.pbInteractionBBox.click()

                qt.QTimer.singleShot(0, _next)

            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = caller

    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """
        Uploads BBox prompt to the server.
        """
        url = f"{self.server}/add_bbox_interaction"

        seg_response = self.request_to_server(
            url,
            json={
                "outer_point_one": outer_point_one[::-1],
                "outer_point_two": outer_point_two[::-1],
                "positive_click": positive_click,
            },
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        self.show_segmentation(unpacked_segmentation)

    #
    #  -- Lasso
    #
    def on_lasso_placed(self, caller, event):
        """
        Called whenever a new point is added to the lasso.
        """
        pointsDefined = self.prompt_types["lasso"]["node"].GetNumberOfControlPoints() > 0
        self.ui.pbInteractionLassoCancel.setVisible(pointsDefined)

    def on_lasso_cancel_clicked(self):
        """
        Called when the user clicks the cancel button for the lasso.
        """
        self.prompt_types["lasso"]["node"].RemoveAllControlPoints()
        self.ui.pbInteractionLassoCancel.setVisible(False)

    def submit_lasso_if_present(self):
        """
        Submits the currently open lasso. We gather all the control points,
        rasterize them into a mask, and send the mask to the server.
        """
        caller = self.prompt_types["lasso"]["node"]
        xyzs = self.xyz_from_caller(caller, point_type="curve_point")

        if len(xyzs) < 3:
            return

        # The lasso prompt only supports points on a single slice plane.
        # If on_interaction_node_modified auto-submits a lasso whose control
        # points span multiple slices, lasso_points_to_mask raises -- swallow
        # the error, clear the lasso, and tell the user.
        try:
            mask = self.lasso_points_to_mask(xyzs)
        except ValueError:
            slicer.util.showStatusMessage(
                "Lasso points must lie on a single slice plane; cleared.",
                4000,
            )
            caller.RemoveAllControlPoints()
            self.ui.pbInteractionLassoCancel.setVisible(False)
            return

        volume_node = self.get_volume_node()
        if volume_node:
            self.lasso_or_scribble_prompt(
                mask=mask, positive_click=self.is_positive, tp="lasso"
            )

            def _next():
                self.setup_prompts()
                # Start placing a new lasso
                self.ui.pbInteractionLasso.click()

            qt.QTimer.singleShot(0, _next)

    #
    #  -- Scribble
    #
    def on_scribble_clicked(self, checked=False):
        """
        Activates/deactivates the hidden Segment Editor's Paint effect on the
        scribble segment (bg or fg, depending on prompt type).
        """
        self.setup_prompts(skip_if_exists=True)

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName(
                    ""
                )  # Clears the active effect

            # Optionally clear or reset the segmentation node
            if hasattr(self, "_scribble_labelmap_callback_tag"):
                tag = self._scribble_labelmap_callback_tag.get("tag", None)
                if tag:
                    self.scribble_segment_node.RemoveObserver(tag)
                del self._scribble_labelmap_callback_tag

            return

        segment_id = "fg" if self.is_positive else "bg"

        # Set segmentation and segment
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        self.scribble_editor_node.SetSelectedSegmentID(segment_id)

        # Set reference volume
        volume_node = self.get_volume_node()
        self.scribble_editor_widget.setSourceVolumeNode(volume_node)

        # Activate paint effect
        self.scribble_editor_widget.setActiveEffectByName("Paint")
        self.scribble_editor_widget.updateWidgetFromMRML()

        paint_effect = self.scribble_editor_widget.activeEffect()
        if paint_effect:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segment_node.AddObserver(
                    vtk.vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        debug_print(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    #
    #  -- Lasso/scribble
    #
    @ensure_synched
    def lasso_or_scribble_prompt(self, mask, positive_click=False, tp="lasso"):
        """
        Uploads lasso or scribble prompt to the server.
        """
        if np.sum(mask) == 0:
            return
        
        url = f"{self.server}/add_{tp}_interaction"
        try:
            buffer = io.BytesIO()
            np.save(buffer, mask)
            compressed_data = gzip.compress(buffer.getvalue())

            from requests_toolbelt import MultipartEncoder

            fields = {
                "file": ("volume.npy.gz", compressed_data, "application/octet-stream"),
                "positive_click": str(
                    positive_click
                ),  # Make sure to send it as a string.
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = self.request_to_server(
                url,
                data=encoder,
                headers={
                    "Content-Type": encoder.content_type,
                    "Content-Encoding": "gzip",
                },
            )

            if seg_response.status_code == 200:
                unpacked_segmentation = self.unpack_binary_segmentation(
                    seg_response.content, decompress=False
                )
                self.show_segmentation(unpacked_segmentation)
            else:
                debug_print(
                    f"lasso_or_scribble_prompt upload failed with status code: {seg_response.status_code}"
                )
        except Exception as e:
            debug_print(f"Error in lasso_or_scribble_prompt: {e}")

    def on_scribble_finished(self, caller, event):
        """
        Called when the user completes a scribble stroke in the Paint effect.
        We calculate the diff in the drawn region and send it to the server.
        """
        debug_print("Scribble stroke finished - labelmap modified!")

        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.scribble_segment_node, label_name, self.get_volume_node()
        )

        if (
            hasattr(self, "_prev_scribble_mask")
            and self._prev_scribble_mask is not None
        ):
            prev_scribble_mask = self._prev_scribble_mask
        else:
            prev_scribble_mask = mask * 0

        diff_mask = mask - prev_scribble_mask
        self._prev_scribble_mask = mask

        self.lasso_or_scribble_prompt(
            mask=diff_mask, positive_click=self.is_positive, tp="scribble"
        )

        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
        # After creating a new segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        debug_print("doing make_new_segment")
        segmentation_node = self.get_segmentation_node()

        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [
                int(seg.split("_")[-1])
                for seg in segment_ids
                if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()
            ]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Make sure the right node is selected
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Apply the user's persisted opacity preference to the new segment so
        # the slider value survives across sessions / new segments.
        display_node = segmentation_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentOpacity(
                new_segment_id, self._get_preferred_segment_opacity()
            )

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            debug_print(f"Clearing segment: {selected_segment_id}")
            self.show_segmentation(
                np.zeros(self.get_image_data().shape, dtype=np.uint8)
            )
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            debug_print("No segment selected to clear.")

    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        """
        t0 = time.time()
        segmentation_mask = self._apply_lasso_slice_clip(segmentation_mask)
        self._last_lasso_slice = None  # consume; non-lasso paths must not clip
        self.previous_states["segment_data"] = segmentation_mask

        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())

        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            self.ui.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentation_mask,
                segmentationNode,
                selectedSegmentID,
                self.get_volume_node(),
            )
            if was_3d_shown:
                segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segment as being edited (can be useful for selective saving of only modified segments)
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        if segmentation_mask.sum() > 0:
            # If we do this when segmentation_mask.sum() == 0, sometimes Slicer will throw "bogus" OOM errors
            # (see https://github.com/coendevente/SlicerNNInteractive/issues/38)
            segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        
        del segmentation_mask

        debug_print(f"show_segmentation took {time.time() - t0}")

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one. Internal scaffolding nodes
        (scribble, magic wand preview) are excluded from this lookup.
        """
        internal_names = {
            self.scribble_segment_node_name,
            self.wand_preview_segment_node_name,
        }

        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            if segmentation_node.GetName() not in internal_names:
                return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            if segmentation_node.GetName() in internal_names:
                segmentation_node = None
                continue

        # Create new segmentation node if none suitable found
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Set segmentation node in widget
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())

        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        """
        Retrieve the currently selected segmentation node & segment ID.
        If none, create one.
        """
        debug_print("doing get_selected_segmentation_node_and_segment_id")
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        """
        Returns the ID of the segment currently selected in the segment editor.
        """
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def get_segment_data(self):
        """
        Gets the labelmap array (binary) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, self.get_volume_node()
        )
        seg_data_bool = mask.astype(bool)

        return seg_data_bool

    def selected_segment_changed(self):
        """
        Checks if the current segment mask has changed from our `self.previous_states`.
        """
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.array_equal(
            old_segment_data.astype(bool), segment_data.astype(bool)
        )

        debug_print(f"segment_data.sum(): {segment_data.sum()}")

        if old_segment_data is not None:
            debug_print(f"old_segment_data.sum(): {old_segment_data.sum()}")
        else:
            debug_print("old_segment_data is None")

        debug_print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    # -- Per-segment display opacity (right-side panel slider) --

    def _current_segmentation_display_node(self):
        seg_node = self.get_segmentation_node()
        if seg_node is None:
            return None
        return seg_node.GetDisplayNode()

    def _sync_opacity_slider_from_segment(self):
        """Push the active segment's current opacity onto the slider UI."""
        display_node = self._current_segmentation_display_node()
        seg_id = self.get_current_segment_id()
        enabled = display_node is not None and bool(seg_id)
        self.ui.sldSegmentOpacity.setEnabled(enabled)
        if not enabled:
            self.ui.lblSegOpacityValue.setText("--")
            return
        # vtkMRMLSegmentationDisplayNode exposes SetSegmentOpacity (master) but
        # not a matching GetSegmentOpacity; read back via the 3D dimension,
        # which SetSegmentOpacity also writes to.
        value = display_node.GetSegmentOpacity3D(seg_id)
        pct = int(round(float(value) * 100))
        blocked = self.ui.sldSegmentOpacity.blockSignals(True)
        try:
            self.ui.sldSegmentOpacity.setValue(pct)
        finally:
            self.ui.sldSegmentOpacity.blockSignals(blocked)
        self.ui.lblSegOpacityValue.setText(f"{pct} %")

    def _on_segment_opacity_changed(self, value):
        """Slider drag -- push opacity to the current segment and persist as
        a user preference applied to future newly-created segments."""
        self.ui.lblSegOpacityValue.setText(f"{int(value)} %")
        fraction = float(value) / 100.0
        self._save_preferred_segment_opacity(fraction)
        display_node = self._current_segmentation_display_node()
        seg_id = self.get_current_segment_id()
        if display_node is None or not seg_id:
            return
        display_node.SetSegmentOpacity(seg_id, fraction)

    def _get_preferred_segment_opacity(self):
        """Read the persisted preferred segment opacity (0..1). Default 1.0."""
        settings = qt.QSettings()
        try:
            v = float(settings.value("SlicerNNInteractive/segment_opacity", 1.0))
        except (TypeError, ValueError):
            v = 1.0
        return max(0.0, min(1.0, v))

    def _save_preferred_segment_opacity(self, value):
        """Persist the slider's current value as a user preference."""
        qt.QSettings().setValue(
            "SlicerNNInteractive/segment_opacity", float(value)
        )

    # -- Lasso slice-range clipping (keep only the lasso slice +/- N) --

    def _get_lasso_clip_enabled(self):
        """Read whether lasso slice-range clipping is enabled. Default False."""
        v = qt.QSettings().value("SlicerNNInteractive/lasso_clip_enabled", False)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    def _get_lasso_clip_n(self):
        """Read the persisted +/- N slices for lasso clipping. Default 0."""
        try:
            return max(
                0, int(qt.QSettings().value("SlicerNNInteractive/lasso_clip_n", 0))
            )
        except (TypeError, ValueError):
            return 0

    def _on_lasso_clip_enabled_changed(self, checked):
        """Persist the lasso-clip enable checkbox."""
        qt.QSettings().setValue(
            "SlicerNNInteractive/lasso_clip_enabled", bool(checked)
        )

    def _on_lasso_clip_n_changed(self, value):
        """Persist the lasso-clip +/- N slices spin box."""
        qt.QSettings().setValue("SlicerNNInteractive/lasso_clip_n", int(value))

    def _apply_lasso_slice_clip(self, mask):
        """If enabled and the last prompt was a lasso, keep only the slices in
        [center-N, center+N] along the lasso plane axis and zero out the rest.
        mask is (z, y, x) uint8. Returns a new array, or the original on no-op.
        """
        if not self._get_lasso_clip_enabled():
            return mask
        info = self._last_lasso_slice
        if info is None:
            return mask
        axis, center = info
        if axis not in (0, 1, 2):
            return mask
        n = self._get_lasso_clip_n()
        dim = mask.shape[axis]
        lo = max(0, center - n)
        hi = min(dim, center + n + 1)  # exclusive upper bound
        if lo >= hi:
            return mask
        clipped = np.zeros_like(mask)
        idx = [slice(None)] * 3
        idx[axis] = slice(lo, hi)
        idx = tuple(idx)
        clipped[idx] = mask[idx]
        return clipped

    ###############################################################################
    # Selection operations (boolean editing)
    ###############################################################################

    def get_operand_segment_ids(self):
        """
        Returns a list of (segment_id, segment_name) for every segment in the
        active segmentation node except the current target segment.
        """
        result = []
        segmentation_node = self.get_segmentation_node()
        if segmentation_node is None:
            return result

        target_id = self.get_current_segment_id()
        segmentation = segmentation_node.GetSegmentation()
        for segment_id in segmentation.GetSegmentIDs():
            if segment_id == target_id:
                continue
            segment = segmentation.GetSegment(segment_id)
            name = segment.GetName() if segment else segment_id
            result.append((segment_id, name))

        return result

    def populate_operand_selector(self):
        """
        Refreshes the operand segment combo box. The stable segment ID is stored
        as item data so renames do not break the current selection.
        """
        combo = self.ui.cbSelectionOperand
        previous_id = combo.currentData if combo.count > 0 else None

        combo.blockSignals(True)
        combo.clear()
        for segment_id, name in self.get_operand_segment_ids():
            combo.addItem(name, segment_id)

        if previous_id is not None:
            for i in range(combo.count):
                if combo.itemData(i) == previous_id:
                    combo.setCurrentIndex(i)
                    break
        combo.blockSignals(False)

        self.ui.cbSelectionOperand.setEnabled(combo.count > 0)
        self._refresh_apply_enabled()

    def segment_id_to_mask(self, segment_id):
        """
        Returns the binary (bool) mask of an arbitrary segment, sampled on the
        current volume geometry so it is shape-aligned with the target segment.
        """
        segmentation_node = self.get_segmentation_node()
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, self.get_volume_node()
        )
        if mask is None:
            return np.zeros(self.get_image_data().shape, dtype=bool)
        return mask.astype(bool)

    @staticmethod
    def compute_boolean_mask(target_mask, operand_mask, operation):
        """
        Pure set-algebra helper. `operation` is 0=Add, 1=Subtract, 2=Intersect.
        Returns a uint8 mask. Raises ValueError on shape mismatch or bad operation.
        """
        target_mask = target_mask.astype(bool)
        operand_mask = operand_mask.astype(bool)

        if target_mask.shape != operand_mask.shape:
            raise ValueError(
                "Target and operand masks have different shapes: "
                f"{target_mask.shape} vs {operand_mask.shape}."
            )

        if operation == 0:  # Add: S OR M
            result_mask = target_mask | operand_mask
        elif operation == 1:  # Subtract: S AND NOT M
            result_mask = target_mask & ~operand_mask
        elif operation == 2:  # Intersect: S AND M
            result_mask = target_mask & operand_mask
        else:
            raise ValueError(f"Unknown operation index: {operation}")

        return result_mask.astype(np.uint8)

    def apply_boolean_operation(self, operand_mask, operation):
        """
        Computes a boolean set operation between the current segment and the
        operand mask. Does not write back or upload.
        """
        return self.compute_boolean_mask(
            self.get_segment_data(), operand_mask, operation
        )

    def on_apply_selection_op_clicked(self, checked=False):
        """
        Applies the selected boolean operation to the current segment, writes the
        result back, and syncs it to the server. The operand can be either another
        segment or a 3D-positioned ROI box, depending on cbOperandSource.
        """
        self.populate_operand_selector()

        target_id = self.get_current_segment_id()
        if not target_id:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Please select a target segment first.",
            )
            return

        # Operand source order: 0=ROI box, 1=Magic wand, 2=Segment.
        source = self.ui.cbOperandSource.currentIndex
        if source == 0:
            if not self._is_selection_roi_valid():
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Click 'Place / Show ROI' to position an ROI before applying.",
                )
                return
            operand_mask = self.roi_node_to_mask(self._sel_op_roi_node)
        elif source == 1:
            if not self._is_selection_wand_seed_valid():
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Click 'Add Seed' and place a seed before applying.",
                )
                return
            operand_mask = self._compute_magic_wand_mask()
            if operand_mask is None:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Magic wand failed (no positive seed, server unreachable, "
                    "or seeds out of volume).",
                )
                return
        else:
            operand_id = self.ui.cbSelectionOperand.currentData
            if not operand_id:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "No operand segment is available. Add another segment first.",
                )
                return

            if operand_id == target_id:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "The operand segment must be different from the target segment.",
                )
                return

            operand_mask = self.segment_id_to_mask(operand_id)

        if operand_mask.sum() == 0:
            slicer.util.showStatusMessage(
                "Operand is empty; the operation may have no effect.", 3000
            )

        operation = self.ui.cbSelectionOperation.currentIndex
        try:
            result_mask = self.apply_boolean_operation(operand_mask, operation)
        except ValueError as e:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Selection Operations",
                f"Could not apply the operation:\n\n{e}",
            )
            return

        # Snapshot the pre-Apply target so our own Undo can restore it
        # reliably -- the embedded Segment Editor's history stack is not
        # always populated for these programmatic edits.
        pre_state = self.get_segment_data().astype(np.uint8).copy()
        self._record_selection_op_undo(target_id, pre_state)

        self.show_segmentation(result_mask)
        self.setup_prompts()
        # The wand preview reflected the about-to-apply mask -- clear it now
        # that the operation has landed on the actual target segment.
        self._clear_wand_preview_segment()

        sync_result = self.upload_segment_to_server()
        if sync_result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The operation was applied locally, but syncing to the server "
                "failed. You can retry with the 'Sync to server' button.",
            )
        else:
            slicer.util.showStatusMessage(
                "Selection operation applied and synced to server.", 3000
            )

    def on_sync_to_server_clicked(self, checked=False):
        """
        Pushes the current segment's mask to the server. Useful after editing the
        segment with native Segment Editor effects.
        """
        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Sync to server",
                "Failed to sync the current segment to the server. Please check "
                "the server connection in the 'Configuration' tab.",
            )
            return

        # Keep previous_states in sync so the next prompt's @ensure_synched does
        # not re-upload the identical mask.
        self.previous_states["segment_data"] = self.get_segment_data()
        slicer.util.showStatusMessage("Current segment synced to server.", 3000)

    def _install_selection_op_observers(self):
        """
        Observes segmentation and segment-editor changes so the operand selector
        stays up to date. Registration is idempotent.
        """
        if not self.hasObserver(
            self.segment_editor_node,
            vtk.vtkCommand.ModifiedEvent,
            self.on_segment_editor_node_modified,
        ):
            self.addObserver(
                self.segment_editor_node,
                vtk.vtkCommand.ModifiedEvent,
                self.on_segment_editor_node_modified,
            )

        self._observe_active_segmentation()

    def _observe_active_segmentation(self):
        """
        (Re)attaches observers on the active segmentation so adding, removing or
        renaming segments refreshes the operand selector.
        """
        segmentation_node = self.get_segmentation_node()
        if segmentation_node is None:
            return

        segmentation = segmentation_node.GetSegmentation()
        previous = getattr(self, "_observed_segmentation", None)
        if previous is segmentation:
            return

        events = (
            slicer.vtkSegmentation.SegmentAdded,
            slicer.vtkSegmentation.SegmentRemoved,
            slicer.vtkSegmentation.SegmentModified,
        )
        if previous is not None:
            for event in events:
                if self.hasObserver(previous, event, self.on_segmentation_modified):
                    self.removeObserver(
                        previous, event, self.on_segmentation_modified
                    )

        for event in events:
            self.addObserver(segmentation, event, self.on_segmentation_modified)

        self._observed_segmentation = segmentation

    def on_segmentation_modified(self, caller, event):
        """Refresh the operand selector when segments are added/removed/renamed."""
        self.populate_operand_selector()
        self._sync_opacity_slider_from_segment()

    def on_segment_editor_node_modified(self, caller, event):
        """Refresh observers and operand list when the node/segment selection changes."""
        self._observe_active_segmentation()
        self.populate_operand_selector()
        self._sync_opacity_slider_from_segment()

    # -- ROI operand --

    @staticmethod
    def _aabb_to_voxel_box(bounds_ras, ras_to_ijk_fn, shape):
        """
        Given a world AABB (xmin, xmax, ymin, ymax, zmin, zmax) in RAS, a function
        mapping an RAS point to (i, j, k) integer voxel coords, and the target
        volume shape (z, y, x), return a bool mask with the corresponding
        voxel-space box filled. Coordinates are clamped to the volume; if the
        resulting box is empty the mask is all-False.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds_ras
        corners_ras = [
            (xmin, ymin, zmin), (xmax, ymin, zmin),
            (xmin, ymax, zmin), (xmax, ymax, zmin),
            (xmin, ymin, zmax), (xmax, ymin, zmax),
            (xmin, ymax, zmax), (xmax, ymax, zmax),
        ]
        ijk = np.array([ras_to_ijk_fn(list(c)) for c in corners_ras])

        # ijk columns are (i, j, k) == (x, y, z); shape is (z, y, x).
        i_min, i_max = int(ijk[:, 0].min()), int(ijk[:, 0].max())
        j_min, j_max = int(ijk[:, 1].min()), int(ijk[:, 1].max())
        k_min, k_max = int(ijk[:, 2].min()), int(ijk[:, 2].max())

        i_min = max(0, i_min)
        i_max = min(shape[2] - 1, i_max)
        j_min = max(0, j_min)
        j_max = min(shape[1] - 1, j_max)
        k_min = max(0, k_min)
        k_max = min(shape[0] - 1, k_max)

        mask = np.zeros(shape, dtype=bool)
        if i_max >= i_min and j_max >= j_min and k_max >= k_min:
            mask[k_min:k_max + 1, j_min:j_max + 1, i_min:i_max + 1] = True
        return mask

    def roi_node_to_mask(self, roi_node):
        """
        Rasterize a vtkMRMLMarkupsROINode into a bool numpy mask by testing
        each candidate voxel inside the ROI's local (OBB) frame. This handles
        obliquely-acquired volumes and rotated ROIs correctly (an earlier
        world-AABB approach over-included voxels heavily for oblique scans).
        The candidate IJK box is first restricted by the ROI's world AABB
        (clamped to the volume) so iteration stays bounded.
        """
        # --- Production geometry fetch ---
        radius = [0.0, 0.0, 0.0]
        roi_node.GetRadiusXYZ(radius)
        rx, ry, rz = radius
        volume = self.get_volume_node()
        shape = self.get_image_data().shape
        mask = np.zeros(shape, dtype=bool)
        if min(rx, ry, rz) <= 0.0 or volume is None:
            return mask

        world_bounds = [0.0] * 6
        roi_node.GetRASBounds(world_bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = world_bounds
        corners_ras = [
            (xmin, ymin, zmin), (xmax, ymin, zmin),
            (xmin, ymax, zmin), (xmax, ymax, zmin),
            (xmin, ymin, zmax), (xmax, ymin, zmax),
            (xmin, ymax, zmax), (xmax, ymax, zmax),
        ]
        corners_ijk = [self.ras_to_xyz(list(c)) for c in corners_ras]
        ijk_arr = np.array(corners_ijk)
        i_lo_raw = int(ijk_arr[:, 0].min())
        i_hi_raw = int(ijk_arr[:, 0].max())
        j_lo_raw = int(ijk_arr[:, 1].min())
        j_hi_raw = int(ijk_arr[:, 1].max())
        k_lo_raw = int(ijk_arr[:, 2].min())
        k_hi_raw = int(ijk_arr[:, 2].max())
        i_lo_c = max(0, i_lo_raw)
        i_hi_c = min(shape[2] - 1, i_hi_raw)
        j_lo_c = max(0, j_lo_raw)
        j_hi_c = min(shape[1] - 1, j_hi_raw)
        k_lo_c = max(0, k_lo_raw)
        k_hi_c = min(shape[0] - 1, k_hi_raw)

        # vtkMRMLMarkupsROINode.GetObjectToWorldMatrix() is a 0-arg accessor
        # in this Slicer's PythonQt binding, returning the vtkMatrix4x4 directly
        # -- not the out-parameter style used by vtkMRMLScalarVolumeNode below.
        object_to_world_vtk = roi_node.GetObjectToWorldMatrix()
        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijk_to_ras_vtk)

        def _m_to_np(m):
            return np.array(
                [[m.GetElement(r, c) for c in range(4)] for r in range(4)]
            )

        ijk_to_object = (
            np.linalg.inv(_m_to_np(object_to_world_vtk))
            @ _m_to_np(ijk_to_ras_vtk)
        )

        # --- TEMP DIAGNOSTICS (remove once verified) ---
        center = [0.0, 0.0, 0.0]
        roi_node.GetCenter(center)
        local_bounds = [0.0] * 6
        roi_node.GetBounds(local_bounds)
        print("[SelectionOps] roi_node_to_mask diagnostics:")
        print(f"  ROI: name={roi_node.GetName()} id={roi_node.GetID()}")
        print(f"  ROI center (RAS): {center}")
        print(f"  ROI radius:       {radius}")
        print(f"  GetBounds (local? old code): {local_bounds}")
        print(f"  GetRASBounds (world, used): {world_bounds}")
        print(
            f"  Volume: name={volume.GetName()}"
            f" shape(z,y,x)={shape}"
        )
        print(f"  Volume spacing:    {tuple(volume.GetSpacing())}")
        print(f"  Volume origin:     {tuple(volume.GetOrigin())}")
        vrb = [0.0] * 6
        volume.GetRASBounds(vrb)
        print(f"  Volume RAS bounds: {vrb}")
        print("  Volume IJK->RAS matrix:")
        for row in range(4):
            print(
                "    [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                    ijk_to_ras_vtk.GetElement(row, 0),
                    ijk_to_ras_vtk.GetElement(row, 1),
                    ijk_to_ras_vtk.GetElement(row, 2),
                    ijk_to_ras_vtk.GetElement(row, 3),
                )
            )
        print(f"  ROI center IJK:    {self.ras_to_xyz(list(center))}")
        print(f"  8 RAS corners: {corners_ras}")
        print(f"  8 IJK corners: {corners_ijk}")
        print(
            f"  Raw IJK box:     i=[{i_lo_raw},{i_hi_raw}] "
            f"j=[{j_lo_raw},{j_hi_raw}] k=[{k_lo_raw},{k_hi_raw}]"
        )
        print(
            f"  Clamped IJK box: i=[{i_lo_c},{i_hi_c}] "
            f"j=[{j_lo_c},{j_hi_c}] k=[{k_lo_c},{k_hi_c}]"
        )
        seg_node = self.get_segmentation_node()
        print(
            f"  Segmentation node: name={seg_node.GetName() if seg_node else None}"
            f" id={seg_node.GetID() if seg_node else None}"
        )
        # --- END TEMP DIAGNOSTICS ---

        if i_hi_c < i_lo_c or j_hi_c < j_lo_c or k_hi_c < k_lo_c:
            print("  Resulting mask voxel count: 0")
            return mask

        # Exact OBB containment in ROI-local space. Vectorized over the
        # candidate IJK box.
        ii, jj, kk = np.meshgrid(
            np.arange(i_lo_c, i_hi_c + 1, dtype=np.float64),
            np.arange(j_lo_c, j_hi_c + 1, dtype=np.float64),
            np.arange(k_lo_c, k_hi_c + 1, dtype=np.float64),
            indexing="ij",
        )
        M = ijk_to_object
        x = M[0, 0] * ii + M[0, 1] * jj + M[0, 2] * kk + M[0, 3]
        y = M[1, 0] * ii + M[1, 1] * jj + M[1, 2] * kk + M[1, 3]
        z = M[2, 0] * ii + M[2, 1] * jj + M[2, 2] * kk + M[2, 3]

        shape_idx = self.ui.cbRoiShape.currentIndex  # 0=Box, 1=Sphere, 2=Ellipsoid
        if shape_idx == 1:
            # Sphere: inscribed in the (possibly non-cube) ROI box.
            r = min(rx, ry, rz)
            inside = (x * x + y * y + z * z) <= (r * r)
        elif shape_idx == 2:
            # Ellipsoid aligned with the ROI axes.
            inside = (
                (x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2
            ) <= 1.0
        else:
            # Box (oriented bounding box, default).
            inside = (np.abs(x) <= rx) & (np.abs(y) <= ry) & (np.abs(z) <= rz)
        if inside.any():
            mask[
                kk[inside].astype(np.int64),
                jj[inside].astype(np.int64),
                ii[inside].astype(np.int64),
            ] = True

        print(f"  Resulting mask voxel count: {int(mask.sum())}")
        return mask

    def _is_selection_roi_valid(self):
        """True iff the operation ROI node still exists in the MRML scene."""
        node = self._sel_op_roi_node
        return node is not None and slicer.mrmlScene.IsNodePresent(node)

    def _configure_selection_roi_display(self, display_node):
        """Style the operation ROI distinctly from the bbox prompt."""
        display_node.SetHandlesInteractive(True)
        display_node.SetFillOpacity(0.1)
        display_node.SetOutlineOpacity(0.8)
        # Orange, to stand apart from the blue bbox prompt.
        color = (1.0, 0.55, 0.1)
        display_node.SetSelectedColor(*color)
        display_node.SetColor(*color)
        display_node.SetActiveColor(*color)
        display_node.SetSliceProjectionColor(*color)
        display_node.SetGlyphScale(0)
        display_node.SetTextScale(0)

    def _initialize_selection_roi_geometry(self, node):
        """Place a freshly created ROI at the volume center with half-extent radii."""
        volume_node = self.get_volume_node()
        if volume_node is None:
            return
        ras_bounds = [0.0] * 6
        volume_node.GetRASBounds(ras_bounds)
        center = [
            0.5 * (ras_bounds[0] + ras_bounds[1]),
            0.5 * (ras_bounds[2] + ras_bounds[3]),
            0.5 * (ras_bounds[4] + ras_bounds[5]),
        ]
        radius = [
            0.25 * max(1.0, ras_bounds[1] - ras_bounds[0]),
            0.25 * max(1.0, ras_bounds[3] - ras_bounds[2]),
            0.25 * max(1.0, ras_bounds[5] - ras_bounds[4]),
        ]
        node.SetCenter(center)
        node.SetRadiusXYZ(radius)

    def _get_or_create_selection_roi(self):
        """
        Ensure a vtkMRMLMarkupsROINode named "SelectionOpROI" exists with
        interactive handles. Returns the node.
        """
        name = "SelectionOpROI"
        node = self._sel_op_roi_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLMarkupsROINode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
                node.SetName(name)
                self._initialize_selection_roi_geometry(node)
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            if display_node is not None:
                self._configure_selection_roi_display(display_node)
            self._sel_op_roi_node = node

        node.SetDisplayVisibility(True)

        # Ensure shape preview tracks this ROI's pose and size.
        if not self.hasObserver(
            node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
        ):
            self.addObserver(
                node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
            )
        self._get_or_create_selection_roi_preview()
        self._update_selection_roi_preview()
        return node

    def _destroy_selection_roi(self):
        """Remove the operation ROI node (and its preview) from the scene."""
        node = self._sel_op_roi_node
        if node is not None:
            if self.hasObserver(
                node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
            ):
                self.removeObserver(
                    node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
                )
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._sel_op_roi_node = None
        self._destroy_selection_roi_preview()

    # -- ROI shape preview (sphere / ellipsoid visualization) --

    def _get_or_create_selection_roi_preview(self):
        """
        Create (or recover) the hidden Model + LinearTransform nodes that
        visualize the actual sphere/ellipsoid acted upon by the boolean
        operation. The model carries a unit sphere mesh; the transform places
        and scales it to match the current ROI + cbRoiShape.
        """
        model_name = "SelectionOpROIPreview"
        transform_name = "SelectionOpROIPreviewTransform"

        transform_node = self._sel_op_roi_preview_transform_node
        if transform_node is None or not slicer.mrmlScene.IsNodePresent(transform_node):
            existing = slicer.mrmlScene.GetFirstNodeByName(transform_name)
            if existing is not None and existing.IsA("vtkMRMLLinearTransformNode"):
                transform_node = existing
            else:
                transform_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode"
                )
                transform_node.SetName(transform_name)
            transform_node.HideFromEditorsOn()
            self._sel_op_roi_preview_transform_node = transform_node

        model_node = self._sel_op_roi_preview_node
        if model_node is None or not slicer.mrmlScene.IsNodePresent(model_node):
            existing = slicer.mrmlScene.GetFirstNodeByName(model_name)
            if existing is not None and existing.IsA("vtkMRMLModelNode"):
                model_node = existing
            else:
                model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                model_node.SetName(model_name)

            # Unit sphere; transform handles all scale/rotation/translation.
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetThetaResolution(24)
            sphere.SetPhiResolution(24)
            sphere.Update()
            model_node.SetAndObservePolyData(sphere.GetOutput())

            model_node.HideFromEditorsOn()
            model_node.CreateDefaultDisplayNodes()
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                color = (1.0, 0.55, 0.1)
                display_node.SetColor(*color)
                display_node.SetEdgeColor(*color)
                display_node.SetOpacity(0.25)
                display_node.SetSliceIntersectionVisibility(True)
                display_node.SetSliceIntersectionThickness(2)
                display_node.SetVisibility2D(True)
                display_node.SetVisibility(True)
            self._sel_op_roi_preview_node = model_node

        # (Re)attach model to transform.
        model_node.SetAndObserveTransformNodeID(transform_node.GetID())
        return model_node, transform_node

    def _update_selection_roi_preview(self):
        """
        Recompute the preview transform (and visibility) from the current ROI
        + cbRoiShape selection. Safe to call even when the ROI is absent.
        """
        if not self._is_selection_roi_valid():
            self._set_preview_visible(False)
            return

        shape_idx = self.ui.cbRoiShape.currentIndex  # 0=Box, 1=Sphere, 2=Ellipsoid
        if shape_idx == 0:
            self._set_preview_visible(False)
            return

        radius = [0.0, 0.0, 0.0]
        self._sel_op_roi_node.GetRadiusXYZ(radius)
        rx, ry, rz = radius
        if min(rx, ry, rz) <= 0.0:
            self._set_preview_visible(False)
            return

        if shape_idx == 1:
            r = min(rx, ry, rz)
            sx = sy = sz = r
        else:
            sx, sy, sz = rx, ry, rz

        # World = ObjectToWorld @ Scale.
        o2w_vtk = self._sel_op_roi_node.GetObjectToWorldMatrix()
        scaled = vtk.vtkMatrix4x4()
        scaled.DeepCopy(o2w_vtk)
        # Multiply each column of the 3x3 part by its scale.
        scale = (sx, sy, sz)
        for col in range(3):
            for row in range(3):
                scaled.SetElement(
                    row, col, o2w_vtk.GetElement(row, col) * scale[col]
                )

        model_node, transform_node = self._get_or_create_selection_roi_preview()
        transform_node.SetMatrixTransformToParent(scaled)
        self._set_preview_visible(True)

    def _set_preview_visible(self, visible):
        """Toggle visibility of the preview model node (no-op if absent)."""
        model_node = self._sel_op_roi_preview_node
        if model_node is None or not slicer.mrmlScene.IsNodePresent(model_node):
            return
        display_node = model_node.GetDisplayNode()
        if display_node is None:
            return
        display_node.SetVisibility(bool(visible))
        display_node.SetVisibility2D(bool(visible))

    def _destroy_selection_roi_preview(self):
        """Remove the preview model + transform nodes from the scene."""
        for attr in ("_sel_op_roi_preview_node", "_sel_op_roi_preview_transform_node"):
            node = getattr(self, attr, None)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, attr, None)

    def _on_selection_roi_modified(self, caller, event):
        """ROI moved/resized/rotated -- keep preview transform in sync."""
        self._update_selection_roi_preview()

    def on_place_roi_clicked(self, checked=False):
        """Create or show the operation ROI in the 3D view."""
        self._get_or_create_selection_roi()
        self._refresh_apply_enabled()
        slicer.util.showStatusMessage(
            "Drag the ROI handles in the 3D view, then click Apply Operation.",
            5000,
        )

    def on_clear_roi_clicked(self, checked=False):
        """Remove the operation ROI."""
        self._destroy_selection_roi()
        self._refresh_apply_enabled()

    def _on_roi_shape_changed(self, index):
        """Status-bar hint clarifying what each ROI shape means, plus preview refresh."""
        if index == 1:
            slicer.util.showStatusMessage(
                "Sphere mode uses the inscribed sphere "
                "(radius = min of the ROI's three half-extents).",
                5000,
            )
        elif index == 2:
            slicer.util.showStatusMessage(
                "Ellipsoid mode uses the ellipsoid aligned with the ROI axes.",
                5000,
            )
        self._update_selection_roi_preview()

    def _refresh_apply_enabled(self):
        """Enable Apply only when the current operand source has a usable operand."""
        # Source order: 0=ROI box, 1=Magic wand, 2=Segment.
        source = self.ui.cbOperandSource.currentIndex
        if source == 0:
            enabled = self._is_selection_roi_valid()
        elif source == 1:
            enabled = self._is_selection_wand_seed_valid()
        else:
            enabled = self.ui.cbSelectionOperand.count > 0
        self.ui.pbApplySelectionOp.setEnabled(enabled)

    def _on_operand_source_changed(self, index):
        """
        Toggle the three operand rows and clean up after the modes we are
        leaving so a ROI / Magic wand preview never lingers across switches.
        Source order: 0=ROI box, 1=Magic wand, 2=Segment.
        """
        self.ui.operandRoiContainer.setVisible(index == 0)
        self.ui.operandMagicWandContainer.setVisible(index == 1)
        self.ui.operandSegmentContainer.setVisible(index == 2)
        # Leaving ROI -> destroy the ROI box (and its preview).
        if index != 0:
            self._destroy_selection_roi()
        # Leaving Magic wand -> destroy seeds and the mask preview.
        if index != 1:
            self._destroy_wand_seed()
            self._clear_wand_preview_segment()
        self._refresh_apply_enabled()

    # -- Magic wand seed lifecycle and flood fill --

    def _configure_wand_seed_display(self, display_node):
        """Distinct green so the wand seed is not confused with bbox/ROI."""
        color = (0.2, 0.85, 0.4)
        display_node.SetColor(*color)
        display_node.SetSelectedColor(*color)
        display_node.SetActiveColor(*color)
        display_node.SetGlyphScale(0.9)
        display_node.SetTextScale(0)
        display_node.SetSliceProjection(True)
        display_node.SetSliceProjectionColor(*color)

    def _get_or_create_wand_seed(self):
        """
        Ensure a vtkMRMLMarkupsFiducialNode exists for magic wand seeds. The
        node holds an unlimited list of seeds (each click adds another).
        Returns the node.
        """
        name = "SelectionOpWandSeeds"
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLMarkupsFiducialNode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode"
                )
                node.SetName(name)
            node.SetMaximumNumberOfControlPoints(-1)
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            if display_node is not None:
                self._configure_wand_seed_display(display_node)
            if not self.hasObserver(
                node,
                slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                self._on_wand_seed_placed,
            ):
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    self._on_wand_seed_placed,
                )
            self._sel_op_wand_seed_node = node

        node.SetDisplayVisibility(True)
        return node

    def _destroy_wand_seed(self):
        """
        Remove the current wand seed node AND any historically-named orphans
        from the scene, detach the placement observer, and bail out of Place
        mode if we were the active placer.
        """
        # 1) Detach observer on the tracked node before removal so the placement
        #    callback never fires on a half-dead node.
        tracked = self._sel_op_wand_seed_node
        if tracked is not None and self.hasObserver(
            tracked,
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self._on_wand_seed_placed,
        ):
            self.removeObserver(
                tracked,
                slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                self._on_wand_seed_placed,
            )

        # 2) Sweep every historical wand seed name out of the scene. Use a
        #    while-loop in case multiple nodes share the same name.
        for name in self._WAND_SEED_NODE_NAMES:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            while existing is not None:
                if existing.IsA("vtkMRMLMarkupsFiducialNode"):
                    slicer.mrmlScene.RemoveNode(existing)
                else:
                    break
                existing = slicer.mrmlScene.GetFirstNodeByName(name)
        self._sel_op_wand_seed_node = None

        # 3) If we left interaction mode in Place for a fiducial, bail out so
        #    the cursor stops behaving like "about to drop a point".
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        selection_node = slicer.app.applicationLogic().GetSelectionNode()
        if (
            interaction_node.GetCurrentInteractionMode()
            == slicer.vtkMRMLInteractionNode.Place
            and selection_node.GetActivePlaceNodeClassName()
            == "vtkMRMLMarkupsFiducialNode"
        ):
            interaction_node.SetCurrentInteractionMode(
                interaction_node.ViewTransform
            )

    def _is_selection_wand_seed_valid(self):
        """True iff at least one wand seed has been placed."""
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            return False
        return node.GetNumberOfControlPoints() >= 1

    def _collect_wand_seeds(self):
        """
        Collect placed wand seeds, converting RAS positions to (k, j, i) voxel
        coords. Returns a list of ((k, j, i), True) tuples (all seeds are
        positive prompts) or [].
        """
        arr = self.get_image_data()
        if arr is None:
            return []
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            return []
        out = []
        for idx in range(node.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(idx, ras)
            ijk = self.ras_to_xyz(list(ras))
            ii, jj, kk = int(ijk[0]), int(ijk[1]), int(ijk[2])
            if (
                0 <= ii < arr.shape[2]
                and 0 <= jj < arr.shape[1]
                and 0 <= kk < arr.shape[0]
            ):
                out.append(((kk, jj, ii), True))
        return out

    def _postprocess_wand_mask(self, mask):
        """Apply Grow/Shrink post-processing to the AI mask."""
        if mask is None or not mask.any():
            return mask
        n_iter = int(self.ui.sbGrowShrinkWand.value)
        if n_iter == 0:
            return mask.astype(bool)
        try:
            from scipy import ndimage
        except Exception:
            return mask.astype(bool)
        if n_iter > 0:
            mask = ndimage.binary_dilation(mask, iterations=n_iter)
        else:
            mask = ndimage.binary_erosion(mask, iterations=-n_iter)
        return mask.astype(bool)

    def _compute_magic_wand_mask(self, seed_node=None):
        """
        Ask the nnInteractive server for an AI segmentation built from one or
        more positive (and optional negative) seed points. The call cycle:
          1. back up the current target segment,
          2. POST an empty mask to /upload_segment (resets server interactions),
          3. POST /add_point_interaction once per seed (positive first, then
             negative); the last response holds the cumulative mask,
          4. POST the original target back to /upload_segment (restores state),
          5. apply local post-processing (Keep largest, Grow/Shrink).
        The `seed_node` argument is ignored -- seeds come from the two internal
        nodes -- and kept only for backward call sites.
        Returns a bool numpy mask aligned to the volume, or None on failure.
        """
        volume = self.get_volume_node()
        arr = self.get_image_data()
        if volume is None or arr is None or not self.server:
            return None

        seeds = self._collect_wand_seeds()
        if not seeds:
            return None

        pre_target = self.get_segment_data().astype(np.uint8).copy()
        shape = arr.shape

        # 1) Reset server interactions by uploading an empty mask.
        empty = np.zeros(shape, dtype=np.uint8)
        reset_resp = self.request_to_server(
            f"{self.server}/upload_segment",
            files=self.mask_to_np_upload_file(empty),
            headers={"Content-Encoding": "gzip"},
        )
        if reset_resp is None:
            return None

        seg_mask = None
        try:
            # 2) Send each seed in order; the LAST response holds the cumulative
            #    mask. voxel_coord uses (z, y, x) order (== ras_to_xyz()[::-1]).
            last_response = None
            for (kk, jj, ii), is_pos in seeds:
                last_response = self.request_to_server(
                    f"{self.server}/add_point_interaction",
                    json={
                        "voxel_coord": [kk, jj, ii],
                        "positive_click": bool(is_pos),
                    },
                )
                if last_response is None:
                    break
            if last_response is not None:
                seg_mask = self.unpack_binary_segmentation(
                    last_response.content, decompress=False
                ).astype(bool)
                seg_mask = self._postprocess_wand_mask(seg_mask)
        finally:
            # 3) Restore the user's target segment on the server so subsequent
            #    nnInteractive prompts continue from where they left off.
            restore_resp = self.request_to_server(
                f"{self.server}/upload_segment",
                files=self.mask_to_np_upload_file(pre_target),
                headers={"Content-Encoding": "gzip"},
            )
            if restore_resp is None:
                slicer.util.showStatusMessage(
                    "Magic wand could not restore server state; "
                    "the next prompt may resync automatically.",
                    4000,
                )

        return seg_mask

    def _enter_place_mode_for_wand(self, node, status_msg):
        selection_node = slicer.app.applicationLogic().GetSelectionNode()
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        selection_node.SetReferenceActivePlaceNodeClassName(
            "vtkMRMLMarkupsFiducialNode"
        )
        selection_node.SetActivePlaceNodeID(node.GetID())
        interaction_node.SetPlaceModePersistence(0)
        interaction_node.SetCurrentInteractionMode(interaction_node.Place)
        slicer.util.showStatusMessage(status_msg, 5000)

    def on_place_wand_seed_clicked(self, checked=False):
        """Add another magic wand seed (does NOT clear earlier seeds)."""
        node = self._get_or_create_wand_seed()
        self._enter_place_mode_for_wand(
            node,
            "Click in any view to add a magic wand seed.",
        )
        self._refresh_apply_enabled()

    def on_clear_wand_seed_clicked(self, checked=False):
        """Remove all magic wand seeds (positive and negative) and the preview."""
        self._destroy_wand_seed()
        self._clear_wand_preview_segment()
        self._refresh_apply_enabled()

    def on_clear_preview_wand_clicked(self, checked=False):
        """Hide the magic wand preview overlay; seeds are kept."""
        self._clear_wand_preview_segment()

    def _on_wand_seed_placed(self, caller, event):
        """Seed was placed -- refresh Apply state only (preview is manual)."""
        self._refresh_apply_enabled()

    # -- Magic wand live preview --

    def _get_or_create_wand_preview_segmentation(self):
        """
        Create (or recover) a hidden segmentation node with a single 'preview'
        segment used to visualize the magic wand region before Apply.
        """
        name = self.wand_preview_segment_node_name
        node = self._sel_op_wand_preview_segment_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLSegmentationNode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode"
                )
                node.SetName(name)
            node.HideFromEditorsOn()
            volume_node = self.get_volume_node()
            if volume_node is not None:
                node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            node.CreateDefaultDisplayNodes()
            self._sel_op_wand_preview_segment_node = node

        segmentation = node.GetSegmentation()
        seg_id = self._wand_preview_segment_id
        if not seg_id or not segmentation.GetSegment(seg_id):
            seg_id = segmentation.AddEmptySegment(
                "MagicWandPreview", "MagicWandPreview", [0.95, 0.2, 0.85]
            )
            self._wand_preview_segment_id = seg_id

        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentOpacity2DFill(seg_id, 0.35)
            display_node.SetSegmentOpacity2DOutline(seg_id, 0.9)
            display_node.SetSegmentVisibility(seg_id, True)

        return node, seg_id

    def _clear_wand_preview_segment(self):
        """
        Empty the preview segment's labelmap and hide it. The node itself is
        kept around so the next show is cheap.
        """
        node = self._sel_op_wand_preview_segment_node
        seg_id = self._wand_preview_segment_id
        if node is None or not slicer.mrmlScene.IsNodePresent(node) or not seg_id:
            return
        volume_node = self.get_volume_node()
        image = self.get_image_data()
        if volume_node is None or image is None:
            return
        empty = np.zeros(image.shape, dtype=np.uint8)
        try:
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                empty, node, seg_id, volume_node
            )
        except Exception:
            pass
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, False)

    def _destroy_wand_preview(self):
        """Remove the preview segmentation node entirely (used at cleanup)."""
        node = self._sel_op_wand_preview_segment_node
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._sel_op_wand_preview_segment_node = None
        self._wand_preview_segment_id = None

    def _update_magic_wand_preview(self):
        """
        Recompute the wand mask via nnInteractive from the current seed and
        write it into the preview segment. Safe to call when the wand is not
        the active operand source -- it will just clear the preview.
        """
        # Magic wand source index is 1 in the current ordering.
        if self.ui.cbOperandSource.currentIndex != 1:
            self._clear_wand_preview_segment()
            return
        if not self._is_selection_wand_seed_valid():
            self._clear_wand_preview_segment()
            return

        wand_mask = self._compute_magic_wand_mask()
        if wand_mask is None:
            self._clear_wand_preview_segment()
            return

        node, seg_id = self._get_or_create_wand_preview_segmentation()
        volume_node = self.get_volume_node()
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            wand_mask.astype(np.uint8), node, seg_id, volume_node
        )
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, True)
        if int(wand_mask.sum()) > 0:
            node.GetSegmentation().CollapseBinaryLabelmaps()
            node.CreateClosedSurfaceRepresentation()

    def on_preview_wand_clicked(self, checked=False):
        """Run a one-shot AI wand call and write the result into the preview."""
        if not self._is_selection_wand_seed_valid():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Place a seed before previewing.",
            )
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._update_magic_wand_preview()
        finally:
            qt.QApplication.restoreOverrideCursor()

    def _record_selection_op_undo(self, segment_id, pre_state_uint8):
        """Push a (segment_id, mask) snapshot onto our private undo stack."""
        self._sel_op_undo_stack.append((segment_id, pre_state_uint8))
        while len(self._sel_op_undo_stack) > self._sel_op_undo_stack_limit:
            self._sel_op_undo_stack.pop(0)

    def on_undo_selection_op_clicked(self, checked=False):
        """
        Revert the last Selection Operations Apply from our private undo stack
        (the embedded Segment Editor's history is not always populated for these
        programmatic edits), then resync local state and server.
        """
        if not self._sel_op_undo_stack:
            slicer.util.showStatusMessage(
                "No Selection Operations Apply to undo.", 3000
            )
            return

        segment_id, pre_state = self._sel_op_undo_stack.pop()
        seg_node = self.get_segmentation_node()
        segmentation = seg_node.GetSegmentation() if seg_node is not None else None
        if segmentation is None or not segmentation.GetSegment(segment_id):
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The target segment of the previous Apply no longer exists.",
            )
            return

        self.segment_editor_node.SetSelectedSegmentID(segment_id)
        # show_segmentation re-applies the binary labelmap AND rebuilds the
        # closed surface representation when 3D was being shown, so Show 3D
        # survives Undo. It also updates previous_states and saves to the
        # editor's undo history.
        self.show_segmentation(pre_state)
        self._clear_wand_preview_segment()

        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The segment was reverted locally, but syncing to the server failed.",
            )
        else:
            slicer.util.showStatusMessage("Selection Operation undone.", 3000)

    ###############################################################################
    # Server communication and sync functions
    ###############################################################################

    def update_server(self):
        """
        Reads user-entered server URL from UI, saves to QSettings, updates self.server.
        """
        self.server = self.ui.Server.text.rstrip("/")
        settings = qt.QSettings()
        settings.setValue("SlicerNNInteractive/server", self.server)
        debug_print(f"Server URL updated and saved: {self.server}")

    def test_server_connection(self):
        """
        Sends a lightweight GET request to see if the configured server responds.
        """
        server_text = self.ui.Server.text
        if not server_text.strip():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Test Connection",
                "Please enter a server URL before testing the connection.",
            )
            return

        self.ui.Server.setText(server_text.strip())
        self.update_server()
        server_url = self.server

        if getattr(self, "_test_server_in_progress", False):
            return
        self._test_server_in_progress = True

        slicer.util.showStatusMessage("Testing nnInteractive server connection...", 2000)
        slicer.app.processEvents()

        response = None
        error_message = None
        try:
            response = requests.get(server_url, timeout=5)
        except requests.exceptions.MissingSchema:
            error_message = (
                "Server URL is invalid. Make sure it starts with 'http://' or 'https://'."
            )
        except requests.exceptions.RequestException as exc:
            error_message = str(exc)
        finally:
            self._test_server_in_progress = False
            slicer.util.showStatusMessage("")

        if response is not None:
            info_message = (
                f"Server at '{server_url}' is reachable."
            )
            QMessageBox.information(
                slicer.util.mainWindow(),
                "Test Connection",
                info_message,
            )
            return
        else:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Test Connection",
                f"Failed to reach '{server_url}'.\n\n{error_message}",
            )

    def request_to_server(self, *args, **kwargs):
        """
        Wraps requests.post in a try/except and shows error in pop up windows if necessary.
        """

        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):

            error_message = None
            try:
                response = requests.post(*args, **kwargs)
                debug_print('response:', response)
            except requests.exceptions.MissingSchema as e:
                response = None
                if self.server == "":
                    raise RuntimeError("It seems you have not set the server URL yet. You can configure it in the 'Configuration' tab.")
                else:
                    raise RuntimeError(f"Server URL '{self.server}' is unreachable. You can edit the URL in the 'Configuration' tab.")
            except requests.exceptions.ConnectionError as e:
                response = None
                raise RuntimeError(f"Failed to connect to server '{self.server}'. Please make sure the server is running and check the server URL in the 'Configuration' tab.")
            except requests.exceptions.InvalidSchema as e:
                append_text_to_error_message = ""
                if not args[0].startswith("http://"):
                    append_text_to_error_message = "\n\nHint: Perhaps your Server URL in the 'Configuration' tab should start with 'http://'. For example, if your server runs on localhost and port 1527, 'localhost:1527' would not work as a Server URL, while 'http://localhost:1527' would."
                raise RuntimeError(f'{e}{append_text_to_error_message}')

            if response.status_code != 200:
                status_code = response.status_code
                response = None
                raise RuntimeError(f"Something has gone wrong with your request (Status code {status_code}).")

            t0 = time.time()
            # Try to parse JSON and check for a specific error.
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        debug_print("No image has been uploaded to the server. Please upload an image first.")
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        response = None
                        raise RuntimeError(f"Server error: {resp_json.get('message', 'Unknown error')}")

            debug_print('1157 took', time.time() - t0)

        return response

    def upload_image_to_server(self):
        """
        Gets volume data from Slicer, packs it, and uploads it to the server.
        """
        debug_print("Syncing image with server...")
        try:
            # Retrieve image data, window, and level.
            t0 = time.time()
            image_data = (
                self.get_image_data()
            )  # Expected to return (image_data, window, level)
            debug_print(f"self.get_image_data took {time.time() - t0}")

            if image_data is None:
                debug_print("No image data available to upload.")
                return

            t0 = time.time()
            url = (
                f"{self.server}/upload_image"  # Update this with your actual endpoint.
            )

            buffer = io.BytesIO()
            np.save(buffer, image_data)
            raw_data = buffer.getvalue()
            debug_print(f"len(raw_data): {len(raw_data)}")

            files = {"file": ("volume.npy", raw_data, "application/octet-stream")}

            # Create your MultipartEncoder without gzip headers
            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

            slicer.progress_window = slicer.util.createProgressDialog(autoClose=False)
            slicer.progress_window.minimum = 0
            slicer.progress_window.maximum = 100
            slicer.progress_window.setLabelText("Uploading image...")

            def my_callback(monitor):
                if not hasattr(monitor, "last_update"):
                    monitor.last_update = time.time()
                if time.time() - monitor.last_update <= 0.2:
                    return
                monitor.last_update = time.time()
                slicer.progress_window.setValue(
                    monitor.bytes_read / len(raw_data) * 100
                )
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                slicer.app.processEvents()

            encoder = MultipartEncoder(fields=files)
            monitor = MultipartEncoderMonitor(encoder, my_callback)

            try:
                result = self.request_to_server(
                    url, data=monitor, headers={"Content-Type": monitor.content_type}
                )
            finally:
                slicer.progress_window.close()

            return result
        except Exception as e:
            debug_print(f"Error in upload_image_to_server: {e}")

    def upload_segment_to_server(self):
        """
        Grabs current segmentation labelmap, gzips it, and sends it to the server.
        """
        debug_print("Syncing segment with server...")
        try:
            segment_data = self.get_segment_data()
            files = self.mask_to_np_upload_file(segment_data)
            url = f"{self.server}/upload_segment"  # Update this with your actual endpoint.

            result = self.request_to_server(
                url, files=files, headers={"Content-Encoding": "gzip"}
            )

            return result
        except Exception as e:
            debug_print(f"Error in upload_image_to_server: {e}")

    ###############################################################################
    # Utility / converters functions
    ###############################################################################

    def get_image_data(self):
        """
        Returns the voxel data of the current active (or first available) volume node.
        """
        volume_node = self.get_volume_node()
        if volume_node:
            return slicer.util.arrayFromVolume(volume_node)

        return None

    def get_volume_node(self):
        """
        Retrieves the current source volume node chosen in the segment editor widget.
        If nothing is set then use the most recently added scalar volume
        """
        # Get volume node from segment editor widget
        volumeNode = self.ui.editor_widget.sourceVolumeNode()

        if not volumeNode:
            # Get the most recently added volume node
            volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volumeNodes:
                volumeNode = volumeNodes[-1]
            # Show this volume node in the segment editor widget
            self.ui.editor_widget.setSourceVolumeNode(volumeNode)

        return volumeNode

    def image_changed(self, do_prev_image_update=True):
        """
        Checks if the volume's voxel data changed since the last time we stored it.
        """
        image_data = self.get_image_data()
        if image_data is None:
            debug_print("No volume node found")
            return

        old_image_data = self.previous_states.get("image_data", None)

        image_changed = old_image_data is None or not np.array_equal(
            old_image_data, image_data
        )

        if do_prev_image_update:
            self.previous_states["image_data"] = copy.deepcopy(image_data)

        return image_changed

    def mask_to_np_upload_file(self, mask):
        """
        Converts a numpy mask into a gzipped file object for POSTing.
        """
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())

        files = {"file": ("volume.npy.gz", compressed_data, "application/octet-stream")}

        return files

    def unpack_binary_segmentation(self, binary_data, decompress=False):
        """
        Unpacks data received from server into a full 3D numpy array (bool).
        """
        if decompress:
            binary_data = binary_data = gzip.decompress(binary_data)

        if self.get_image_data() is None:
            self.capture_image()

        vol_shape = self.get_image_data().shape
        total_voxels = np.prod(vol_shape)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        unpacked_bits = unpacked_bits[:total_voxels]

        segmentation_mask = (
            unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        )

        return segmentation_mask

    def ras_to_xyz(self, pos):
        """
        Converts an RAS position to IJK voxel coords in the current volume node.
        """
        volumeNode = self.get_volume_node()

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas
        )
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)

        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
        xyz = [int(round(c)) for c in point_Ijk[0:3]]
        return xyz


    def xyz_from_caller(self, caller, lock_point=True, point_type="control_point"):
        """
        Extract voxel coordinates from a Markups node.
        `point_type` can be either "control_point" or "curve_point".
        """
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0:
                debug_print("No control points found")
                return

            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(n - 1, pos)
            if lock_point:
                caller.SetNthControlPointLocked(n - 1, True)
            xyz = self.ras_to_xyz(pos)
            return xyz
        elif point_type == "curve_point":
            vtk_pts = caller.GetCurvePointsWorld()
            
            if vtk_pts is not None:
                vtk_pts_data = vtk_to_numpy(vtk_pts.GetData())
                xyz = [self.ras_to_xyz(pos) for pos in vtk_pts_data]
                debug_print(xyz)
                return xyz

            return []
        else:
            raise ValueError(f'Unknown point_type {point_type}')

    def lasso_points_to_mask(self, points):
        """
        Given a list of voxel coords (defining a polygon in one slice),
        returns a 3D mask with that polygon filled in the appropriate slice.
        """
        from skimage.draw import polygon

        shape = self.get_image_data().shape
        pts = np.array(points)  # shape (n, 3)

        # Determine which coordinate is constant
        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError(
                "Expected exactly one constant coordinate among the points"
            )
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])

        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Depending on which axis is constant, extract the 2D polygon and fill the corresponding slice.
        # Note: our volume is ordered as (z, y, x)
        if const_axis == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            rr, cc = polygon(y_coords, x_coords, shape=(shape[1], shape[2]))
            mask[const_val, rr, cc] = 1
        elif const_axis == 1:
            x_coords = pts[:, 0]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, x_coords, shape=(shape[0], shape[2]))
            mask[rr, const_val, cc] = 1
        elif const_axis == 0:
            y_coords = pts[:, 1]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, y_coords, shape=(shape[0], shape[1]))
            mask[rr, cc, const_val] = 1

        # Record the lasso plane so show_segmentation can clip the result to
        # this slice +/- N. xyz axis i maps to numpy mask axis (2 - i).
        self._last_lasso_slice = (2 - const_axis, const_val)

        return mask

    ###############################################################################
    # Prompt type toggle (positive / negative)
    ###############################################################################

    @property
    def is_positive(self):
        """
        Returns True if the current prompt is set to "positive",
        False if "negative."
        """
        return self.ui.pbPromptTypePositive.isChecked()

    def on_prompt_type_positive_clicked(self, checked=False):
        """
        Called when user presses the "Positive" prompt button.
        """
        # Update UI
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)
        debug_print("Prompt type set to POSITIVE")

    def on_prompt_type_negative_clicked(self, checked=False):
        """
        Called when user presses the "Negative" prompt button.
        """

        # Update UI
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)
        debug_print("Prompt type set to NEGATIVE")

    def toggle_prompt_type(self, checked=False):
        """
        Toggle between positive and negative (triggered by 'T' key).
        """
        debug_print("Toggling prompt type (positive <> negative)")
        if self.current_prompt_type_positive:
            self.on_prompt_type_negative_clicked()
        else:
            self.on_prompt_type_positive_clicked()


###############################################################################
# Test hook (used by Reload & Test)
###############################################################################
_test_module_path = (
    Path(__file__).resolve().parents[0]
    / "Testing"
    / "Python"
    / "SlicerNNInteractiveSegmentationTest.py"
)

if _test_module_path.exists():
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location(
        "SlicerNNInteractiveSegmentationTest", str(_test_module_path)
    )
    _test_module = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_test_module)
    SlicerNNInteractiveTest = _test_module.SlicerNNInteractiveSegmentationTest
