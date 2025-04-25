import io
import gzip
import requests
import copy
import threading
import time

import importlib.util

import numpy as np

import slicer
import qt
import vtk

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

from urllib.parse import urlparse
import re

def get_path(url: str) -> str:
    """
    Return everything after the host[:port] portion of a URL,
    including the query string and fragment if present.
    """
    parsed = urlparse(url)
    # .path is always there; .query / .fragment may be empty
    rest = parsed.path
    if parsed.query:     # keep ?key=value part, if any
        rest += '?' + parsed.query
    if parsed.fragment:  # keep #anchor part, if any
        rest += '#' + parsed.fragment
    return rest

def record_time(label):
    """
    Decorator that measures execution time (in ms) and stores it in self.timings[label].
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # make sure we have a place to store
            if not hasattr(self, "timings"):
                self.timings = {}
            t0 = time.time()
            result = func(self, *args, **kwargs)
            t1 = time.time()
            
            request_metadata = None
            if label == "request_to_server":
                _add_interaction_re = re.compile(r'^/add_[^/]+_interaction$')
                
                url = args[0]
                path = urlparse(url).path
                
                request_metadata = {
                    "is_request_to_server": True,
                    "is_interaction_to_server": bool(_add_interaction_re.match(path)),
                    "request_url": urlparse(url).path,
                }
            
            if label not in self.timings:
                self.timings[label] = []
            
            self.timings[label].append({
                "time": (t1 - t0) * 1000.0,
                "request_metadata": request_metadata,         
            })
            
            return result

        return wrapper

    return decorator


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
        self.parent.contributors = [
            "Coen de Vente",
            "Kiran Vaidhya Venkadesh",
            "Bram van Ginneken",
            "Clara I. Sanchez",
        ]
        self.parent.helpText = """
            This is an 3D Slicer extension for using nnInteractive.

            Read more about this plugin here: https://github.com/coendevente/SlicerNNInteractive.
            """
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""


###############################################################################
# SlicerNNInteractiveWidget
###############################################################################


class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

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

        self.add_segmentation_widget()

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
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

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

        self.addObserver(
            slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent,
            self.on_interaction_node_modified,
        )

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
            "xxhash": "xxhash==3.5.0",
            "requests_toolbelt": "requests_toolbelt==1.0.0",
            "SimpleITK": "SimpleITK==2.3.1",
            "skimage": "scikit-image==0.22.0",
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
        module_name, module_version = module_name_and_version.split("==")
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False
        else:
            try:
                # For Python 3.8+; if using an older Python version, you might need to install importlib-metadata.
                import importlib.metadata as metadata
            except ImportError:
                debug_print("Use Python 3.8+")

            try:
                version = metadata.version(module_name)
                if version != module_version:
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
            prompt_type["button"].clicked.connect(
                lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(
                    checked, prompt_name
                )
            )
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

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
            if (
                interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place
            ):
                if prompt_type["name"] == "LassoPrompt" and (
                    self.ui.pbInteractionLasso.isChecked()
                ):
                    self.submit_lasso_if_present()
                prompt_type["button"].setChecked(False)
            elif (
                interactionNode.GetCurrentInteractionMode()
                == slicer.vtkMRMLInteractionNode.Place
            ):
                placingThisNode = (
                    selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID()
                )
                prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if (
            interactionNode.GetCurrentInteractionMode()
            == slicer.vtkMRMLInteractionNode.Place
        ):
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
            selectionNode.SetReferenceActivePlaceNodeClassName(
                self.prompt_types[prompt_name]["node_class"]
            )
            selectionNode.SetActivePlaceNodeID(
                self.prompt_types[prompt_name]["node"].GetID()
            )
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
    @record_time("on_point_placed")
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        xyz = self.xyz_from_caller(caller)

        volume_node = self.get_volume_node()
        if volume_node:
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    @record_time("point_prompt")
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
    @record_time("on_bbox_placed")
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
    @record_time("bbox_prompt")
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
    @record_time("on_lasso_placed")
    def on_lasso_placed(self, caller, event):
        """
        Called whenever a new point is added to the lasso.
        """
        pointsDefined = (
            self.prompt_types["lasso"]["node"].GetNumberOfControlPoints() > 0
        )
        self.ui.pbInteractionLassoCancel.setVisible(pointsDefined)

    def on_lasso_cancel_clicked(self):
        """
        Called when the user clicks the cancel button for the lasso.
        """
        self.prompt_types["lasso"]["node"].RemoveAllControlPoints()
        self.ui.pbInteractionLassoCancel.setVisible(False)

    @record_time("submit_lasso_if_present")
    def submit_lasso_if_present(self):
        """
        Submits the currently open lasso. We gather all the control points,
        rasterize them into a mask, and send the mask to the server.
        """
        caller = self.prompt_types["lasso"]["node"]
        xyzs = self.xyz_from_caller(caller, point_type="curve_point")

        if len(xyzs) < 3:
            return

        mask = self.lasso_points_to_mask(xyzs)

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
    @record_time("on_scribble_clicked")
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
    @record_time("lasso_or_scribble_prompt")
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

    @record_time("on_scribble_finished")
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

    def add_segmentation_widget(self):
        """
        Adds the standard Segment Editor widget to the UI of our plugin.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        self.editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self.editor_widget.setMaximumNumberOfUndoStates(10)

        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(
            segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode"
        )

        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(
                self.segment_editor_node
            )

        self.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.editor_widget.setMRMLScene(slicer.mrmlScene)

        # Add the editor widget to the segmentation group
        if hasattr(self.ui, "segmentationGroup"):
            # Create a new layout if needed
            if self.ui.segmentationGroup.layout() is None:
                layout = qt.QVBoxLayout(self.ui.segmentationGroup)
                self.ui.segmentationGroup.setLayout(layout)
            else:
                layout = self.ui.segmentationGroup.layout()

            # Clear any existing widgets in the layout
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

            # Add the editor widget
            layout.addWidget(self.editor_widget)
        else:
            debug_print("Could not find segmentationGroup in UI")

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
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
        self.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        self.editor_widget.updateWidgetFromMRML()

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
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

    @record_time("show_segmentation")
    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        """
        t0 = time.time()
        self.previous_states["segment_data"] = segmentation_mask

        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        )

        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            self.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentation_mask,
                segmentationNode,
                selectedSegmentID,
                self.get_volume_node(),
            )
            if was_3d_shown:
                segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        del segmentation_mask

        debug_print(f"show_segmentation took {time.time() - t0}")

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one.
        """
        # If the segmentation widget has a currently selected segmentation node, return it.
        if hasattr(self, "editor_widget") and self.editor_widget.segmentationNode():
            seg_node = self.editor_widget.segmentationNode()
            if seg_node.GetName() != self.scribble_segment_node_name:
                return seg_node

        # Otherwise, fall back to getting the first segmentation node (or create one if none exists).
        segmentation_node = slicer.mrmlScene.GetFirstNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        if (
            segmentation_node is None
            or segmentation_node.GetName() == self.scribble_segment_node_name
        ):
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode"
            )
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                self.get_volume_node()
            )
            self.editor_widget.setSegmentationNode(segmentation_node)

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
        return self.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

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

    @record_time("request_to_server")
    def request_to_server(self, *args, **kwargs):
        """
        Wraps requests.post in a try/except and shows error in pop up windows if necessary.
        """

        with slicer.util.tryWithErrorDisplay(
            _("Segmentation failed."), waitCursor=True
        ):

            error_message = None
            try:
                response = requests.post(*args, **kwargs)
                debug_print("response:", response)
            except requests.exceptions.MissingSchema as e:
                response = None
                if self.server == "":
                    raise RuntimeError(
                        "It seems you have not set the server URL yet. You can configure it in the 'Configuration' tab."
                    )
                else:
                    raise RuntimeError(
                        f"Server URL '{self.server}' is unreachable. You can edit the URL in the 'Configuration' tab."
                    )
            except requests.exceptions.ConnectionError as e:
                response = None
                raise RuntimeError(
                    f"Failed to connect to server '{self.server}'. Please make sure the server is running and check the server URL in the 'Configuration' tab."
                )

            if response.status_code != 200:
                status_code = response.status_code
                response = None
                raise RuntimeError(
                    f"Something has gone wrong with your request (Status code {status_code})."
                )

            t0 = time.time()
            # Try to parse JSON and check for a specific error.
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        debug_print(
                            "No image has been uploaded to the server. Please upload an image first."
                        )
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        response = None
                        raise RuntimeError(
                            f"Server error: {resp_json.get('message', 'Unknown error')}"
                        )

            debug_print("1157 took", time.time() - t0)

        return response

    @record_time("upload_image_to_server")
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

    @record_time("upload_segment_to_server")
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
        volumeNode = self.editor_widget.sourceVolumeNode()

        if not volumeNode:
            # Get the most recently added volume node
            volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volumeNodes:
                volumeNode = volumeNodes[-1]
            # Show this volume node in the segment editor widget
            self.editor_widget.setSourceVolumeNode(volumeNode)

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

    @record_time("unpack_binary_segmentation")
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
            raise ValueError(f"Unknown point_type {point_type}")

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


from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
import os, slicer, vtk, time, gc


class SlicerNNInteractiveTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.testPointClickAtWorld()

    def selectPointPromptButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbInteractionPoint.click()
        slicer.app.processEvents()

    def selectBBoxPromptButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbInteractionBBox.click()
        slicer.app.processEvents()

    def selectScribblePromptButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbInteractionScribble.click()
        slicer.app.processEvents()

    def selectLassoPromptButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbInteractionLasso.click()
        slicer.app.processEvents()

    def clickNextSegmentButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbNextSegment.click()
        slicer.app.processEvents()

    def clickResetSegmentButton(self):
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        gui.ui.pbResetSegment.click()
        slicer.app.processEvents()

    def get_xy_redview(self, worldPt):
        lm = slicer.app.layoutManager()
        sliceLogic = lm.sliceWidget("Red").sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        sliceNode.SetSliceOffset(worldPt[2])

        lm.sliceWidget("Red").sliceView().renderWindow().Render()
        slicer.app.processEvents()

        redView = lm.sliceWidget("Red").sliceView()
        ren = redView.renderWindow().GetRenderers().GetFirstRenderer()
        ren.SetWorldPoint(worldPt[0], worldPt[1], worldPt[2], 1.0)
        ren.WorldToDisplay()

        w, h = redView.width, redView.height
        x_w = int(w / 2)
        y_w = int(h / 2)

        # self.delayDisplay(f"Clicking at widget coords ({(x_w, y_w)})")
        sliceNode.JumpSliceByCentering(worldPt[0], worldPt[1], worldPt[2])

        return redView, x_w, y_w

    @record_time("clickPoint")
    def clickPoint(self, worldPt, button="Left"):
        redView, x_w, y_w = self.get_xy_redview(worldPt)
        slicer.util.clickAndDrag(
            redView, start=(x_w, y_w), end=(x_w, y_w), steps=0, button=button
        )
        slicer.app.processEvents()

    @record_time("dragPoints")
    def dragPoints(self, middle):
        redView, x_middle, y_middle = self.get_xy_redview(middle)

        print("(x_middle, y_middle):", (x_middle, y_middle))

        slicer.util.clickAndDrag(
            redView,
            start=(x_middle - 25, y_middle - 25),
            end=(x_middle + 25, y_middle + 25),
            steps=2,
            button="Left",
        )
        slicer.app.processEvents()

    def resetTiming(self):
        self.timings = {}
        slicer.modules.SlicerNNInteractiveWidget.timings = {}

    def reportTiming(self):
        widget = slicer.modules.SlicerNNInteractiveWidget
        timings = widget.timings
        print("widget.timings:", timings)
        widget_timings = copy.deepcopy(timings)
        
        server_interactions = [
            t 
            for t in widget_timings["request_to_server"] 
            if t["request_metadata"] is not None
            and t["request_metadata"]["is_interaction_to_server"]
        ]
        print('server_interactions:', server_interactions)
        assert len(server_interactions) == 1
        
        server_time = server_interactions[0]['time']
        
        
        timings = self.timings
        print(timings)
        key_0 = list(timings.keys())[0]
        
        print('timings:', timings)
        
        assert len(timings) == 1
        assert len(timings[key_0]) == 1

        return server_time, widget_timings, copy.deepcopy(timings[key_0][0])

    def makeTimesTable(self, times):
        """
        Produces a markdown table with cells formatted as:
        Total ± sigma_total (Server ± sigma_server), in seconds.
        """
        try:
            import pandas as pd
        except ImportError:
            slicer.util.pip_install('pandas')
            import pandas as pd
        try:
            import tabulate
        except ImportError:
            slicer.util.pip_install('tabulate')
            import tabulate
        import numpy as np
        # Row label for this configuration
        row_label = "Small Image | GPU 1"

        # Mapping keys to column headers
        col_map = {
            "point": "Point",
            "bbox": "Bounding box",
            "scribble": "Scribble",
            "lasso": "Lasso",
        }

        # Prepare data dictionary
        table_data = {}
        for key, col_name in col_map.items():
            entries = times.get(key, [])
            # Extract server and remainder times
            server_times = [e["server_prompt_processing"] for e in entries]
            remainder_times = [e["remainder"] for e in entries]
            # Compute total times
            total_times = [s + r for s, r in zip(server_times, remainder_times)]

            # Compute statistics (mean, std) in seconds
            def stats_ms_to_s(values):
                if not values:
                    return 0.0, 0.0
                mean_ms = np.mean(values)
                std_ms = np.std(values, ddof=1) if len(values) > 1 else 0.0
                return mean_ms / 1000.0, std_ms / 1000.0

            server_mean, server_std = stats_ms_to_s(server_times)
            total_mean, total_std = stats_ms_to_s(total_times)

            # Format string
            cell = (
                f"{total_mean:.1f} s ± {total_std:.2f} s "
                f"({server_mean:.1f} s ± {server_std:.2f} s)"
            )
            table_data[col_name] = [cell]

        # Build DataFrame and print as Markdown
        df = pd.DataFrame(table_data, index=[row_label])
        print(df.to_markdown())
        
        return df

    def deleteCurrentSegment(self):
        """
        Remove the segment that is currently selected in the SlicerNNInteractive widget.
        """
        time.sleep(0.2)
        # Grab the module’s GUI/widget instance
        gui = slicer.util.getModuleGui("SlicerNNInteractive")
        # Ask the widget for its current segmentation node and selected segment ID
        segmentation_node, segment_id = (
            gui.get_selected_segmentation_node_and_segment_id()
        )
        # If there is a segment selected, delete it
        if segment_id:
            segmentation_node.GetSegmentation().RemoveSegment(segment_id)
            # Refresh the editor so the UI knows the segment is gone
            gui.editor_widget.setSegmentationNode(segmentation_node)
            gui.editor_widget.updateWidgetFromMRML()
            slicer.app.processEvents()

    def testPointClickAtWorld(self):
        
        # self.makeTimesTable({'point': [{'server_prompt_processing': 274.81770515441895, 'remainder': 1084.2270851135254}, {'server_prompt_processing': 270.0059413909912, 'remainder': 1018.8159942626953}, {'server_prompt_processing': 367.0217990875244, 'remainder': 1086.721420288086}, {'server_prompt_processing': 272.7179527282715, 'remainder': 1101.8640995025635}, {'server_prompt_processing': 293.9410209655762, 'remainder': 1003.39674949646}, {'server_prompt_processing': 342.46015548706055, 'remainder': 1086.9767665863037}, {'server_prompt_processing': 277.8630256652832, 'remainder': 1120.5148696899414}, {'server_prompt_processing': 388.9279365539551, 'remainder': 1138.554334640503}, {'server_prompt_processing': 363.4791374206543, 'remainder': 1134.6831321716309}, {'server_prompt_processing': 250.1530647277832, 'remainder': 1047.9130744934082}], 'bbox': [{'server_prompt_processing': 267.9722309112549, 'remainder': 1290.557861328125}, {'server_prompt_processing': 277.70495414733887, 'remainder': 1203.4249305725098}, {'server_prompt_processing': 264.57810401916504, 'remainder': 1206.1126232147217}, {'server_prompt_processing': 222.81908988952637, 'remainder': 1307.049036026001}, {'server_prompt_processing': 254.87494468688965, 'remainder': 1220.2939987182617}, {'server_prompt_processing': 318.51911544799805, 'remainder': 1300.35400390625}, {'server_prompt_processing': 355.1318645477295, 'remainder': 1261.366367340088}, {'server_prompt_processing': 362.22290992736816, 'remainder': 1339.8020267486572}, {'server_prompt_processing': 227.5378704071045, 'remainder': 1243.8380718231201}, {'server_prompt_processing': 232.36918449401855, 'remainder': 1364.2230033874512}], 'scribble': [{'server_prompt_processing': 115.55099487304688, 'remainder': 2936.944007873535}, {'server_prompt_processing': 125.45895576477051, 'remainder': 2896.9268798828125}, {'server_prompt_processing': 103.80268096923828, 'remainder': 2852.9422283172607}, {'server_prompt_processing': 111.67621612548828, 'remainder': 2935.222864151001}, {'server_prompt_processing': 109.13205146789551, 'remainder': 2983.234643936157}, {'server_prompt_processing': 116.23072624206543, 'remainder': 3075.700044631958}, {'server_prompt_processing': 119.88496780395508, 'remainder': 2937.4608993530273}, {'server_prompt_processing': 110.96596717834473, 'remainder': 2966.5510654449463}, {'server_prompt_processing': 109.39693450927734, 'remainder': 3133.073091506958}, {'server_prompt_processing': 101.77803039550781, 'remainder': 2975.1129150390625}], 'lasso': [{'server_prompt_processing': 298.4428405761719, 'remainder': 1350.2402305603027}, {'server_prompt_processing': 249.02105331420898, 'remainder': 1265.3090953826904}, {'server_prompt_processing': 236.29307746887207, 'remainder': 1231.6608428955078}, {'server_prompt_processing': 251.4822483062744, 'remainder': 1330.0085067749023}, {'server_prompt_processing': 317.3098564147949, 'remainder': 1347.9561805725098}, {'server_prompt_processing': 358.2158088684082, 'remainder': 1371.7823028564453}, {'server_prompt_processing': 242.71702766418457, 'remainder': 1352.2160053253174}, {'server_prompt_processing': 254.51016426086426, 'remainder': 1396.7690467834473}, {'server_prompt_processing': 342.35405921936035, 'remainder': 1369.4860935211182}, {'server_prompt_processing': 279.97398376464844, 'remainder': 1352.8289794921875}]})
        # return
        
        from pathlib import Path
        
        root_path = Path("~/projects/plain/3d-slicer-uls-plugin/PlainAnnotator/PlainAnnotator/Data/Images").expanduser()
        
        images = [
            {
                "volPath": root_path / "0_cropped_resized.nii.gz",
                "name": "S",
                "slice_offset": 0
            },
            {
                "volPath": root_path / "0_cropped.nii.gz",
                "name": "M",
                "slice_offset": 0
            },
            {
                "volPath": root_path / "0.nii.gz",
                "name": "L",
                "slice_offset": -170
            }
        ]
        
        times = {
            image["name"]: {"point": [], "bbox": [], "scribble": [], "lasso": []}
            for image in images
        }
        
        n_repeats = 10
        for _ in range(n_repeats):
            for image in images:
                image_name = image["name"]
                
                # Remove all loaded volume and segmentation nodes
                for node_class in ["vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]:
                    nodes = slicer.util.getNodesByClass(node_class)
                    for node in nodes:
                        slicer.mrmlScene.RemoveNode(node)

                volumeNode = slicer.util.loadVolume(image['volPath'])

                zres = image["slice_offset"]
                self.assertIsNotNone(volumeNode)

                slicer.app.processEvents()

                slicer.util.selectModule("SlicerNNInteractive")
                slicer.app.processEvents()
                self.selectPointPromptButton()
                slicer.app.processEvents()

                # A point prompt just to trigger the uploading of the image
                self.clickNextSegmentButton()
                self.resetTiming()
                self.clickPoint([54, 170, -173 + zres])
                self.reportTiming()

                

                self.selectPointPromptButton()  # off
                
                # break
                # return

                self.deleteCurrentSegment()
                self.clickNextSegmentButton()
                self.selectPointPromptButton()  # on
                self.resetTiming()
                # self.clickPoint([54, 170, -283])  # renal tumor
                self.clickPoint([54, 170, -173 + zres])  # liver
                server_time, widget_timings, _ = self.reportTiming()
                
                full_time = widget_timings["on_point_placed"][0]["time"]
                
                times[image_name]["point"].append(
                    {
                        "server_prompt_processing": server_time,
                        "remainder": full_time - server_time,
                    }
                )
                # self.selectPointPromptButton()  # off

                self.selectBBoxPromptButton()
                self.deleteCurrentSegment()
                self.clickNextSegmentButton()
                
                # spleen
                # self.clickPoint([-112, 195, -190])
                # self.clickPoint([-190, 81, -190])
                
                # liver
                self.clickPoint([-61, 263, -173 + zres])       
                
                self.resetTiming()
                self.clickPoint([134, 55, -173 + zres])
                server_time, widget_timings, _ = self.reportTiming()
                
                full_time = widget_timings["on_bbox_placed"][0]["time"]
                
                times[image_name]["bbox"].append(
                    {
                        "server_prompt_processing": server_time,
                        "remainder": full_time - server_time,
                    }
                )

                self.selectScribblePromptButton()
                self.deleteCurrentSegment()
                self.clickNextSegmentButton()
                
                self.resetTiming()
                self.dragPoints(middle=[63, 182, -190 + zres])

                server_time, widget_timings, _ = self.reportTiming()
                
                full_time = widget_timings["on_scribble_finished"][0]["time"]
                
                times[image_name]["scribble"].append(
                    {
                        "server_prompt_processing": server_time,
                        "remainder": full_time - server_time,
                    }
                )

                self.selectLassoPromptButton()
                self.deleteCurrentSegment()
                self.clickNextSegmentButton()
                
                # aorta
                # self.clickPoint([-40, 141, -174])
                # self.clickPoint([-53, 127, -174])
                # self.clickPoint([-40, 115, -174])
                # self.clickPoint([-27, 129, -174])
                # self.clickPoint([-40, 150, -174], button="Right")
                
                # liver
                self.clickPoint([-62, 224, -174 + zres])
                self.clickPoint([5, 260, -174 + zres])
                self.clickPoint([88, 224, -174 + zres])
                self.clickPoint([137, 140, -174 + zres])
                self.clickPoint([105, 74, -174 + zres])
                self.clickPoint([33, 71, -174 + zres])
                self.clickPoint([8, 140, -174 + zres])
                self.clickPoint([16, 153, -174 + zres])
                self.clickPoint([4, 177, -174 + zres])
                self.clickPoint([-19, 163, -174 + zres])
                
                self.resetTiming()
                self.clickPoint([-40, 188, -174 + zres], button="Right")
                
                server_time, widget_timings, _ = self.reportTiming()
                
                full_time = widget_timings["submit_lasso_if_present"][0]["time"]
                
                times[image_name]["lasso"].append(
                    {
                        "server_prompt_processing": server_time,
                        "remainder": full_time - server_time,
                    }
                )

                print(times[image_name])
        
        times_tables = [self.makeTimesTable(t) for t in times.values()]

        # times_tables
        import pandas as pd
        times_tables_df = pd.concat(times_tables, ignore_index=True)
        
        md_txt = times_tables_df.to_markdown()
        print(md_txt)
        
        with open(f'/Users/coendevente/projects/plain/nninteractive-slicer/timing_results/{time.time()}.txt', "w", encoding="utf-8") as f:
            f.write(md_txt)
                    
        import json
        with open(f'/Users/coendevente/projects/plain/nninteractive-slicer/timing_results/{time.time()}.json', "w") as f:
            f.write(json.dumps(times, indent=4))
            
        self.delayDisplay("Done!")
