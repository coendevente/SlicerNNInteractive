import contextlib
import os
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox


###############################################################################
# Decorators and utility functions
###############################################################################


DEBUG_MODE = False

# Hard upper bound on the nnInteractive / nninteractive-client versions this extension
# will ever install. nnInteractive 3.x may change the inference API; pin below it so a
# future release can't silently break this plugin. The "up to date" indicator and every
# install command honour this ceiling. Bump deliberately once 3.x is supported.
NNINTERACTIVE_VERSION_CEILING = "3.0.0"
# Hard lower bound: the oldest backend this plugin supports. Earlier releases lack APIs
# the plugin relies on, so both flavors pin at or above it, and a runtime check offers the
# Reinstall / Update flow when an older backend is detected in Slicer's Python (e.g. one
# installed outside the plugin). Applies to BOTH distributions. Bump deliberately.
NNINTERACTIVE_VERSION_FLOOR = "2.5.1"
NNINTERACTIVE_PKG = (
    f"nnInteractive>={NNINTERACTIVE_VERSION_FLOOR},<{NNINTERACTIVE_VERSION_CEILING}"
)
NNINTERACTIVE_CLIENT_PKG = (
    f"nninteractive-client>={NNINTERACTIVE_VERSION_FLOOR},<{NNINTERACTIVE_VERSION_CEILING}"
)

# Packages the PLUGIN itself imports directly and therefore must install explicitly
# with either flavor. Do NOT rely on the backend's dependency tree for these:
# nnInteractive / nninteractive-client >= 2.5 no longer declare scikit-image, which
# the freehand lasso needs (skimage.draw.polygon in lasso_points_to_crop). The Full
# flavor usually still gets it transitively (nnunetv2 keeps scikit-image in its
# requirements as of 2.8), but the thin client does not, and transitive availability
# is not a contract we can rely on.
PLUGIN_DIRECT_DEPS = "scikit-image"

# On Windows, PyPI's default ``pip install torch`` is a CPU-only wheel, so local GPU
# inference would silently fall back to CPU. Point pip at PyTorch's CUDA wheel index
# instead. The SlicerPyTorch extension (PyTorchUtils) is still preferred when
# available -- this is the fallback. To install a different build (a CUDA version for
# an older GPU driver, or a pinned torch version), run pip from Slicer's Python Console
# (see the README).
DEFAULT_WINDOWS_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"

# The README's "Common issues" section, linked from error messages and warnings.
README_COMMON_ISSUES_URL = (
    "https://git.dkfz.de/mic/personal/group2/isensee/slicernninteractive#common-issues"
)

# Marker for an install interrupted by the restart the PyTorch extension requires.
# Written BEFORE that extension is installed (accepting the restart offered by
# Slicer's dialog restarts the app immediately, so there is no later chance), and
# consumed by _resolve_install_on_startup after the restart to continue the install
# without re-asking the user.
PENDING_INSTALL_SETTINGS_KEY = "SlicerNNInteractive/pending_install"

# Human-readable version of THIS extension. Slicer itself identifies extension builds by
# git commit (the SCM revision it stamps at build time -- see _check_plugin_update_async),
# so this string is informational only: it does NOT drive update detection. It is shown in
# the module help and the Configuration tab so users can tell which release they are on.
# Bump on each tagged release and keep it in sync with the git tag ("v" + this string).
PLUGIN_VERSION = "1.0.0"

# The extension's registered name in Slicer's Extensions Manager -- this is the CMake
# project() name in slicer_plugin/CMakeLists.txt, NOT the module title. Used to ask the
# Extensions Manager whether a newer build of THIS extension is published on the server.
PLUGIN_EXTENSION_NAME = "NNInteractive"

# Remembers which installed plugin revision we've already raised the "update available"
# popup for, so the popup appears once per version instead of on every launch. Keyed on
# the installed revision, so it resets naturally once the user actually updates. See
# _maybe_show_plugin_update_popup.
PLUGIN_UPDATE_NOTIFIED_SETTINGS_KEY = "SlicerNNInteractive/plugin_update_notified_revision"


def get_pending_install_flavor():
    """'full' when a Full install was interrupted by the restart the PyTorch
    extension requires, else ''. Only Full ever sets the marker -- the client
    flavor never needs that restart."""
    flavor = slicer.util.settingsValue(PENDING_INSTALL_SETTINGS_KEY, "")
    return flavor if flavor == "full" else ""


def set_pending_install_flavor(flavor):
    settings = qt.QSettings()
    settings.setValue(PENDING_INSTALL_SETTINGS_KEY, flavor)
    settings.sync()  # the PyTorch-extension install may restart Slicer immediately


def reopen_module_for_pending_install():
    """startupCompleted() hook: if an install is waiting on the restart that just
    happened, reopen the module so its startup resolver continues the install
    automatically. No-op on every normal launch."""
    if not get_pending_install_flavor():
        return
    if importlib.util.find_spec("PyTorchUtils") is None:
        # The restart did not activate the PyTorch extension (install failed or is
        # still pending another restart). Leave the marker for a later restart
        # rather than continuing with a degraded plain-pip torch.
        return
    qt.QTimer.singleShot(0, lambda: slicer.util.selectModule("SlicerNNInteractive"))


def cuda_gpu_available():
    """Best-effort check for a usable NVIDIA/CUDA GPU, cheaply and WITHOUT importing torch.

    torch.compile in this plugin only makes sense on an NVIDIA GPU, and this gates the
    config-tab toggle, so the check must stay light: importing torch just to build the
    UI would add seconds for remote-only users and fail outright on client-only installs.
    We therefore probe the NVIDIA driver at the OS level. The authoritative
    ``torch.cuda.is_available()`` check still runs when a local session is built (see
    _construct_local_session), which covers the corner case of an NVIDIA driver present
    but a CPU-only torch wheel installed.

    When torch is already imported, defer to ``torch.cuda.is_available()`` -- it is free
    then and matches exactly what a session will see.
    """
    import glob
    import shutil
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False
    if sys.platform == "darwin":
        return False  # Apple platforms have no CUDA
    # Linux: the NVIDIA kernel driver exposes these when a GPU + driver are present.
    if os.path.isdir("/proc/driver/nvidia/gpus") and os.listdir(
        "/proc/driver/nvidia/gpus"
    ):
        return True
    if glob.glob("/dev/nvidia[0-9]*"):
        return True
    return shutil.which("nvidia-smi") is not None


def torch_compile_unsupported_reason(check_gpu=False):
    """Why torch.compile can't be used here, or None if it can.

    Hard blockers:
    * Windows -- torch.compile (Triton/inductor) is not supported on Windows at all.
    * No NVIDIA/CUDA GPU -- torch.compile here targets CUDA; without a GPU the model runs
      on CPU, where compiling is pointless (and slow to build). Only checked when
      ``check_gpu`` is set, via the torch-free cuda_gpu_available() probe -- the
      module-level default and the session path leave it off so nothing imports torch
      prematurely (the session path enforces the GPU requirement authoritatively via
      torch.cuda.is_available() instead).
    * Elsewhere it compiles small C/CUDA helpers at runtime, which need the Python
      development headers (Python.h). Slicer's bundled Python does not ship them.

    In each case torch.compile would fail or be pointless, so we detect it up front and
    run eager instead.
    """
    import sys
    import sysconfig

    if sys.platform.startswith("win"):
        return "torch.compile is not supported on Windows."

    include_dir = sysconfig.get_paths().get("include")
    if not (include_dir and os.path.isfile(os.path.join(include_dir, "Python.h"))):
        return (
            "torch.compile needs the Python development headers (Python.h), "
            "which are not present in this Python build."
        )
    if check_gpu and not cuda_gpu_available():
        return "torch.compile needs an NVIDIA GPU; none was detected."
    return None


# The single authoritative default for every QSettings-backed setting. The config UI
# and the session constructors both read through get_setting_bool/get_setting_str, so
# a checkbox can never initialize differently from what the session actually uses.
SETTING_DEFAULTS = {
    "autozoom": True,
    # On wherever the automated check says it can work; users who explicitly toggled
    # the checkbox keep their stored choice.
    "use_torch_compile": torch_compile_unsupported_reason() is None,
    "interactions_storage": "auto",
    "device": "cuda:0",
    "checkpoint_path": "",
    "model_id": "",
}


def debug_print(*args):
    if DEBUG_MODE:
        print(*args)


def ensure_synched(func):
    """
    Decorator that ensures a working nnInteractive session exists and that the
    image and segment are synced to it before calling the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        # Make sure we have a live session (constructs a local one or connects
        # to a remote server, depending on the configured mode).
        if not self.ensure_session():
            return

        # The user can delete the layers the session is bound to (its source volume or
        # segmentation) from the Data module while a session is live. The backend keeps
        # its now-stale image/seed, so the next prompt would either segment against the
        # wrong node or drop the backend into an invalid state (random log errors). Catch
        # the deletion here, BEFORE the re-sync below overwrites the recorded identities.
        missing_layers = self._check_session_layers_present()
        if missing_layers:
            self._handle_deleted_session_layers(missing_layers)
            return

        try:
            # Per-step timing (set DEBUG_MODE = True at the top of this file to see it):
            # this is the GUI work that wraps every prompt, so it shows where the
            # end-to-end latency goes beyond the model's prediction itself.
            t0 = time.time()
            if self._handle_active_source_volume_change():
                # Volume changed without the editor observer catching it; prompt state was
                # just reset. The image_changed()/sync below uploads the new image.
                debug_print("Source volume changed before prompt. Prompt state reset.")
            t1 = time.time()
            if self.image_changed():
                debug_print("Image changed (or not previously set). Syncing image to session.")
                if not self.sync_image_to_session():
                    return
            t2 = time.time()
            if self.selected_segment_changed():
                debug_print("Segment changed (or not previously set). Seeding session with segment.")
                self.remove_all_but_last_prompt()
                self.sync_segment_to_session()
            else:
                debug_print("Segment did not change!")
            t3 = time.time()
            result = func(self, *args, **kwargs)
            t4 = time.time()
            debug_print(
                f"[timing] volume-check {t1 - t0:.3f}s | image-check/sync {t2 - t1:.3f}s | "
                f"segment-check/sync {t3 - t2:.3f}s | {func.__name__}+apply {t4 - t3:.3f}s"
            )
            return result
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
            return

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
        # Some Slicer builds surface parent.version in the module panel / about box; set it
        # when supported, but never let an unsupported attribute break module load.
        try:
            self.parent.version = PLUGIN_VERSION
        except Exception:  # noqa: BLE001
            pass
        self.parent.helpText = f"""
            This is an 3D Slicer extension for using nnInteractive (plugin version {PLUGIN_VERSION}).

            Read more about this plugin here: https://git.dkfz.de/mic/personal/group2/isensee/slicernninteractive.
            """
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""

        # Resume an install interrupted by the PyTorch-extension restart (see
        # PENDING_INSTALL_SETTINGS_KEY). Connected on every launch; a no-op unless
        # the marker is set AND the extension actually activated.
        slicer.app.connect("startupCompleted()", reopen_module_for_pending_install)


###############################################################################
# Freehand-lasso input capture
###############################################################################


class _LassoFreehandFilter(qt.QObject):
    """
    Per-slice-view Qt event filter. It does two jobs:

    1. Forwards mouse events to the module widget so it can capture a freehand-lasso
       contour (``handle_lasso_event``).
    2. Re-asserts our coloured prompt cursor on hover/move. Slice views are VTK-backed
       (QVTKOpenGLNativeWidget) and Slicer/markups/effects reset the OS cursor on mode
       changes and on hover, so a one-shot ``setCursor()`` does not stick -- we keep
       re-applying it here while a tool is active (``_reassert_cursor_on_view``).
    """

    def __init__(self, module_widget, slice_widget):
        qt.QObject.__init__(self)
        self._module_widget = module_widget
        self._slice_widget = slice_widget
        self._cursor_event_types = {
            qt.QEvent.Enter,
            qt.QEvent.HoverEnter,
            qt.QEvent.HoverMove,
            qt.QEvent.MouseMove,
        }

    def eventFilter(self, obj, event):
        mw = self._module_widget
        if mw._is_tearing_down():
            return False
        if event.type() in self._cursor_event_types:
            mw._reassert_cursor_on_view(self._slice_widget)
            mw._queue_cursor_reassert(self._slice_widget)
        return mw.handle_lasso_event(self._slice_widget, event, obj)


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

        # nnInteractive session state (constructed lazily on first prompt / Connect).
        self.session = None
        self.target_buffer = None
        self.api_key = ""
        self.server = ""
        # License of the most recently initialized model; merged into the Prompts-tab
        # status line (update_connect_status). Kept after uninitialize.
        self._model_license_text = "Model license: —"
        # Exceptions that mean a remote lease is gone; populated when the remote
        # client is imported. Empty by default so local mode catches nothing here.
        self.SESSION_LOST_ERRORS = tuple()
        self.previous_states = {}
        self._heartbeat_timer = None
        # One visual-undo closure per interaction sent to the session, popped on Ctrl+Z
        # so the prompt marker of the undone interaction disappears together with the
        # reverted segmentation. Kept in lockstep with the session's interaction order.
        self._prompt_undo_stack = []
        # Active click-capture tool ("point"/"bbox") that runs in ViewTransform mode
        # instead of markups Place mode, so right-click zoom/pan stay native (like the
        # lasso). None when no such tool is active. ``_bbox_drag``/``_bbox_preview``
        # hold the in-progress bounding-box drag and its live outline actor.
        self._place_tool = None
        self._bbox_drag = None
        self._bbox_preview = None
        # Compute mode the live session was built for ("local"/"remote"), so the
        # Local/Remote toggle can keep it instead of tearing it down.
        self._session_mode = None
        # True when the live local session fell back to the CPU (no usable CUDA GPU);
        # drives the persistent red warning shown below Initialize.
        self._local_running_on_cpu = False
        # Tracks the editor's selected segment so switching to a different mask clears
        # the displayed prompts. _suppress_segment_switch guards programmatic changes.
        self._last_selected_segment_id = None
        self._suppress_segment_switch = False

        # Freehand-lasso interaction state. The lasso is captured directly from a
        # left-mouse drag on the slice views (see handle_lasso_event), not via markups
        # Place mode.
        self._lasso_active = False
        self._lasso_drawing = False
        self._lasso_last_xy = None
        self._lasso_slice_widget = None
        self._lasso_display_pts = []
        # VTK 2D outline actor for the live freehand contour (see _lasso_overlay_*).
        # The persistent filled prompt is its own overlay segment (see _add_prompt_overlay_segment).
        self._lasso_points = None
        self._lasso_outline_pd = None
        self._lasso_outline_actor = None
        self._lasso_renderer = None
        self._lasso_render_view = None
        self._qt_event_filters = []

        # Per-polarity slice-view cursors (green = positive, red = negative), built
        # lazily and cached. Recoloured whenever the active tool or prompt polarity
        # changes; see _update_prompt_cursor().
        self._prompt_cursor_cache = {}
        self._cursor_reassert_timer_queued = False
        self._pending_cursor_reassert_widgets = []
        self._cleanup_in_progress = False
        self._application_quitting = False
        self._active_source_volume_id = None
        self._handling_source_volume_change = False
        self._scribble_reference_volume_id = None
        self._scribble_callback_in_progress = False

        try:
            slicer.app.aboutToQuit.connect(self._on_application_about_to_quit)
        except Exception:  # noqa: BLE001
            pass

        # Background "update available?" check: a worker thread fetches PyPI while a
        # main-thread poll timer picks up the result (all Qt access stays on the main
        # thread). See _check_for_updates_async / _poll_update_check.
        self._update_poll_timer = None
        self._update_check_result = None

        # Plugin (Slicer extension) update check, driven by Slicer's Extensions Manager
        # (see _check_plugin_update_async). The manager model is Qt/main-thread only, so
        # the check runs via its async signals rather than a worker thread. Guard flags
        # keep us from starting it twice or stacking the one-time popup.
        self._emm = None
        self._plugin_update_check_started = False
        self._plugin_update_popup_shown = False

        # Guards for the install dialogs: never stack a second install prompt on top
        # of an open one, and offer the "backend missing/outdated" update at most
        # once per session (see _offer_backend_update / _prompt_install_choice).
        self._install_prompt_active = False
        self._offered_backend_update = False

        # DON'T install packages here, and never install lazily. Installation happens
        # only from the explicit first-run popup or the Configuration tab's
        # "Reinstall / Update nnInteractive" button. The first-run popup is deferred to
        # the event loop (modal dialogs during setup() are unreliable).
        qt.QTimer.singleShot(0, self._resolve_install_on_startup)

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SlicerNNInteractive.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"

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

        # Surface the 'E' hotkey on the Segment Editor's own Add button: adding a
        # segment there auto-selects it, which resets the prompts just like 'Next
        # segment (E)', so the two are equivalent for the user. AddSegmentButton is
        # internal to qMRMLSegmentEditorWidget -- if a future Slicer renames it, skip
        # the label rather than break module setup.
        try:
            add_button = slicer.util.findChild(self.ui.editor_widget, "AddSegmentButton")
            add_button.setText(f"{add_button.text} (E)")
            add_button.setToolTip(f"{add_button.toolTip}\nKeyboard shortcut: E")
        except Exception as exc:  # noqa: BLE001 - cosmetic only
            debug_print(f"Could not add the hotkey label to the Add segment button: {exc}")

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "button": self.ui.pbInteractionPoint,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "button": self.ui.pbInteractionBBox,
                "button_icon_filename": "bbox_icon.svg",
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "button": self.ui.pbInteractionLasso,
                "button_icon_filename": "lasso_icon.svg",
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        # Prompt-tool buttons are persistent UI widgets, but setup_prompts() runs on
        # every reset / Next segment / volume change. Track which ones we've already
        # wired so we connect each button's `clicked` exactly once -- reconnecting on
        # every cycle would pile up duplicate slots (and fire the handler N times).
        self._connected_prompt_buttons = set()
        self.setup_prompts()

        self.init_ui_functionality()

        self._active_source_volume_id = self._volume_node_id(self.get_volume_node())

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        # Build the Configuration-tab model selection + settings (Local | Remote
        # switch, checkpoint/device/compile, server URL, API key, status, ...). This
        # also loads and wires the saved server URL and sets self.server.
        self.setup_config_ui()

        # Mandatory Initialize control + license banner (top of the Prompts tab) and
        # acknowledgement logos (bottom).
        self.setup_prompts_tab_extras()

        # Set initial prompt type
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.on_next_segment)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)
        self.addObserver(
            self.segment_editor_node,
            vtk.vtkCommand.ModifiedEvent,
            self.on_segment_editor_node_modified,
        )
        # Re-evaluate the Initialize button when volumes are added/removed: Initialize is
        # only available once an image is loaded (so the image is uploaded at init time
        # rather than deferred to the slow first prompt).
        self.addObserver(
            slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.on_scene_nodes_changed
        )
        self.addObserver(
            slicer.mrmlScene, slicer.vtkMRMLScene.NodeRemovedEvent, self.on_scene_nodes_changed
        )

        # Initialization is mandatory: keep every prompt control disabled until a
        # session is live (_on_session_ready re-enables them, gated per model).
        self._set_interaction_ui_enabled(False)

    def setup_config_ui(self):
        """
        Builds the Configuration-tab model selection + settings programmatically,
        mirroring the napari plugin: a Local | Remote segmented switch that shows the
        relevant container, with advanced local options in a collapsible section.
        Controls are stored on ``self.ui`` so the rest of the module references them
        by name.
        """
        import ctk

        layout = self.ui.tabConfig.layout()

        # ===== Installation group (pinned to the bottom of the Configuration tab;
        # placed into the layout after the Model Selection group below) =====
        install_group = qt.QGroupBox("nnInteractive Installation")
        install_layout = qt.QVBoxLayout(install_group)

        self.ui.installFlavorLabel = qt.QLabel()
        self.ui.installFlavorLabel.setWordWrap(True)
        install_layout.addWidget(self.ui.installFlavorLabel)

        self.ui.updateStatusLabel = qt.QLabel()
        self.ui.updateStatusLabel.setWordWrap(True)
        install_layout.addWidget(self.ui.updateStatusLabel)

        self.ui.reinstallButton = qt.QPushButton("Reinstall / Update nnInteractive")
        self.ui.reinstallButton.setMinimumHeight(30)
        self.ui.reinstallButton.setToolTip(
            "Choose Full (local + remote) or Client only (remote), and update the "
            "installed backend to the latest version."
        )
        self.ui.reinstallButton.clicked.connect(
            lambda: self._prompt_install_choice(reinstall=True)
        )
        install_layout.addWidget(self.ui.reinstallButton)

        # Plugin (the Slicer extension itself) update status. Separate from the backend
        # update label above: the extension is rebuilt nightly from the repo's main
        # branch, but Slicer does not notify users, so we surface it here and via a
        # one-time popup. See _check_plugin_update_async.
        self.ui.pluginUpdateStatusLabel = qt.QLabel()
        self.ui.pluginUpdateStatusLabel.setWordWrap(True)
        install_layout.addWidget(self.ui.pluginUpdateStatusLabel)
        # NOTE: install_group is added to the tab layout at the very bottom, after the
        # Model Selection group is inserted (see end of this method).

        switch_style = (
            f"QPushButton {{ {self.unselected_style} }}"
            f"QPushButton:checked {{ {self.selected_style} }}"
        )

        # ===== Model Selection group =====
        model_group = qt.QGroupBox("Model Selection")
        model_layout = qt.QVBoxLayout(model_group)

        # --- Local | Remote segmented switch ---
        switch_row = qt.QHBoxLayout()
        switch_row.setSpacing(0)
        self.ui.localModeButton = qt.QPushButton("Local")
        self.ui.remoteModeButton = qt.QPushButton("Remote")
        self.ui.modeButtonGroup = qt.QButtonGroup(model_group)
        self.ui.modeButtonGroup.setExclusive(True)
        for button in (self.ui.localModeButton, self.ui.remoteModeButton):
            button.setCheckable(True)
            button.setMinimumHeight(28)
            button.setStyleSheet(switch_style)
            self.ui.modeButtonGroup.addButton(button)
            switch_row.addWidget(button)
        model_layout.addLayout(switch_row)
        self.ui.localModeButton.clicked.connect(lambda: self.on_mode_changed("local"))
        self.ui.remoteModeButton.clicked.connect(lambda: self.on_mode_changed("remote"))
        # Local mode needs the full backend; disable it (and fall back to Remote) when
        # only the lightweight remote client is installed. Done before the model dropdown
        # is populated below so a client-only install never triggers a local-only import.
        self._apply_local_mode_availability()

        # --- Local container ---
        self.ui.localContainer = qt.QWidget()
        local_layout = qt.QVBoxLayout(self.ui.localContainer)
        local_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.ui.localContainer)

        # Official-model dropdown, populated from the nnInteractive backend manifest
        # (the authoritative list of selectable models). Left empty/disabled if the
        # list can't be loaded (offline with no cache, or backend not installed yet);
        # the local-checkpoint field below and the default model still work then.
        model_row = qt.QHBoxLayout()
        model_row.addWidget(qt.QLabel("Model:"))
        self.ui.modelComboBox = qt.QComboBox()
        self.ui.modelComboBox.setToolTip(
            "Official nnInteractive model for local inference. Downloaded on first use "
            "into $NNINTERACTIVE_MODEL_DIR (default ~/.nninteractive)."
        )
        self._model_ids = []
        self._populate_model_combo()
        self.ui.modelComboBox.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.ui.modelComboBox)
        local_layout.addLayout(model_row)

        ckpt_row = qt.QHBoxLayout()
        self.ui.checkpointEdit = qt.QLineEdit()
        self.ui.checkpointEdit.setText(self.get_setting_str("checkpoint_path"))
        self.ui.checkpointEdit.setPlaceholderText("Use local checkpoint... (blank = official weights)")
        self.ui.checkpointEdit.editingFinished.connect(
            lambda: self._save_setting("checkpoint_path", self.ui.checkpointEdit.text, reinit=True)
        )
        ckpt_row.addWidget(self.ui.checkpointEdit)
        clear_ckpt_btn = qt.QPushButton("✕")
        clear_ckpt_btn.setFixedWidth(30)
        clear_ckpt_btn.setToolTip("Clear local checkpoint (use the official downloaded weights)")

        def _clear_ckpt():
            self.ui.checkpointEdit.setText("")
            self._save_setting("checkpoint_path", "", reinit=True)

        clear_ckpt_btn.clicked.connect(_clear_ckpt)
        ckpt_row.addWidget(clear_ckpt_btn)
        local_layout.addLayout(ckpt_row)

        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Advanced"
        advanced.collapsed = True
        advanced_form = qt.QFormLayout(advanced)
        local_layout.addWidget(advanced)

        self.ui.deviceCombo = qt.QComboBox()
        self.ui.deviceCombo.addItems(["cuda:0", "cpu"])
        self.ui.deviceCombo.setCurrentText(self.get_local_device())
        self.ui.deviceCombo.currentTextChanged.connect(
            lambda v: self._save_setting("device", v, reinit=True)
        )
        advanced_form.addRow("Device:", self.ui.deviceCombo)

        self.ui.compileCheck = qt.QCheckBox()
        compile_reason = torch_compile_unsupported_reason(check_gpu=True)
        if compile_reason is not None:
            # torch.compile can't work / is pointless here (Windows, no NVIDIA GPU, or no
            # Python.h to build its runtime helpers -- the last is typical in Slicer's
            # bundled Python). Force it OFF and disabled so the user can't turn on
            # something that would fail on the first prediction, and isn't left staring at
            # a stale stored "on" showing as checked. Explain why in the tooltip.
            self.ui.compileCheck.setChecked(False)
            self.ui.compileCheck.setEnabled(False)
            self.ui.compileCheck.setToolTip(f"Unavailable: {compile_reason}")
        else:
            self.ui.compileCheck.setChecked(self.get_setting_bool("use_torch_compile"))
        # Connect AFTER setting the initial check state so the seeding above never fires
        # the save handler (which would reinit the session mid-setup).
        self.ui.compileCheck.toggled.connect(
            lambda v: self._save_setting_bool("use_torch_compile", v, reinit=True)
        )
        advanced_form.addRow("Use torch.compile:", self.ui.compileCheck)

        self.ui.storageCombo = qt.QComboBox()
        self.ui.storageCombo.addItems(["auto", "blosc2", "tensor"])
        self.ui.storageCombo.setCurrentText(self.get_setting_str("interactions_storage"))
        self.ui.storageCombo.currentTextChanged.connect(
            lambda v: self._save_setting("interactions_storage", v, reinit=True)
        )
        advanced_form.addRow("Interaction storage:", self.ui.storageCombo)

        # --- Remote container ---
        self.ui.remoteContainer = qt.QWidget()
        remote_layout = qt.QVBoxLayout(self.ui.remoteContainer)
        remote_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.ui.remoteContainer)

        self.ui.serverUrlEdit = qt.QLineEdit()
        self.ui.serverUrlEdit.setPlaceholderText("http://gpu-box:1527")
        self.ui.serverUrlEdit.setText(slicer.util.settingsValue("SlicerNNInteractive/server", ""))
        self.ui.serverUrlEdit.editingFinished.connect(self.update_server)
        remote_layout.addWidget(self.ui.serverUrlEdit)

        self.ui.apiKeyEdit = qt.QLineEdit()
        self.ui.apiKeyEdit.setEchoMode(qt.QLineEdit.Password)
        self.ui.apiKeyEdit.setPlaceholderText("API key (optional)")
        self.ui.apiKeyEdit.setText(self.api_key)
        # editingFinished (not textChanged) so we only react once the user is done, and
        # a change can disconnect without firing on every keystroke.
        self.ui.apiKeyEdit.editingFinished.connect(self._on_api_key_changed)
        remote_layout.addWidget(self.ui.apiKeyEdit)

        # --- Auto-zoom (common to both modes; see _apply_autozoom_to_session) ---
        autozoom_row = qt.QHBoxLayout()
        self.ui.autozoomCheck = qt.QCheckBox("Auto-zoom")
        self.ui.autozoomCheck.setChecked(self.get_setting_bool("autozoom"))
        self.ui.autozoomCheck.setToolTip(
            "Allows nnInteractive to zoom out so that large objects fit into its field "
            "of view and are segmented completely.\n"
            "Segmentation can take longer, and in rare situations it may segment more "
            "than intended (oversegmentation).\n"
            "Applies to both Local and Remote sessions and takes effect immediately."
        )
        self.ui.autozoomCheck.toggled.connect(self.on_autozoom_toggled)
        autozoom_row.addWidget(self.ui.autozoomCheck)
        autozoom_row.addStretch(1)
        model_layout.addLayout(autozoom_row)

        # --- Status (common to both modes) ---
        # The Initialize/Uninitialize action lives at the TOP of the "nnInteractive
        # Prompts" tab (initialization is mandatory before any prompt can be placed).
        # Keep a status readout here so the session state is visible while the user
        # configures the server / model.
        self.ui.connectStatusLabel = qt.QLabel("Status: not initialized")
        self.ui.connectStatusLabel.setWordWrap(True)
        model_layout.addWidget(self.ui.connectStatusLabel)

        configure_hint = qt.QLabel(
            "Configure here, then click <b>Initialize</b> at the top of the "
            "<b>nnInteractive Prompts</b> tab."
        )
        configure_hint.setWordWrap(True)
        configure_hint.setStyleSheet("color: gray;")
        model_layout.addWidget(configure_hint)

        # The model license is shown only in the Prompts tab (merged into the status
        # line there), where the user actually works -- not duplicated here in the
        # Configuration tab.

        # Insert the model-selection group just above the trailing vertical spacer.
        idx = max(layout.count() - 1, 0)
        layout.insertWidget(idx, model_group)

        # Append the Installation group after the trailing spacer so it stays pinned to
        # the bottom of the Configuration tab, below the model-selection controls.
        layout.addWidget(install_group)

        # Initial switch state + container visibility, and the cached server URL.
        is_local = self.get_mode() == "local"
        self.ui.localModeButton.setChecked(is_local)
        self.ui.remoteModeButton.setChecked(not is_local)
        self._update_mode_visibility()
        self.server = self.ui.serverUrlEdit.text.rstrip("/")

        # Reflect the persisted mode in the status readout. If Local is already the
        # active mode, fill the dropdown right after setup finishes (deferred so it runs
        # once the UI is fully built).
        self.update_connect_status(connected=False)
        if is_local:
            qt.QTimer.singleShot(0, self._ensure_model_combo_populated)

        # Reflect what's installed, then kick off the background "update available?" check.
        self._refresh_install_status_ui()
        self._check_for_updates_async()
        # Also check whether the plugin (extension) itself has a newer build available.
        self._check_plugin_update_async()

    def _on_api_key_changed(self):
        # API keys are intentionally not persisted to QSettings.
        edit = getattr(self.ui, "apiKeyEdit", None)
        if edit is None:
            return
        new_key = edit.text
        if new_key == self.api_key:
            return
        self.api_key = new_key
        # The key changed; drop the remote session so the next Connect uses the new key.
        self._teardown_for_settings_change()

    def on_mode_changed(self, mode):
        if mode == self.get_mode():
            return  # re-click of the active mode -- nothing to do
        self.set_mode(mode)
        self._update_mode_visibility()
        # Mode is a session-affecting setting (like the server URL or a local option):
        # switching it tears down any live session and locks the prompt UI, so the user
        # explicitly re-initializes for the newly selected mode with a single click.
        # (Without this, the old session stays live behind a "not initialized" status,
        # and the first Initialize press would only uninitialize it.)
        self._teardown_for_settings_change()
        if mode == "local":
            # User switched to Local: fill the model dropdown from the already-installed
            # backend. Never installs here (the Full install brings huggingface_hub +
            # model_management up front); the dropdown stays empty if nothing is installed.
            self._ensure_model_combo_populated()

    def _update_mode_visibility(self):
        is_local = self.get_mode() == "local"
        if hasattr(self.ui, "localContainer"):
            self.ui.localContainer.setVisible(is_local)
        if hasattr(self.ui, "remoteContainer"):
            self.ui.remoteContainer.setVisible(not is_local)

    def _local_inference_available(self):
        """True if the full nnInteractive backend (local in-process inference) is
        installed in Slicer's Python. A lightweight 'nninteractive-client'-only
        environment returns False (it ships only the remote client).
        """
        try:
            return (
                importlib.util.find_spec("nnInteractive.inference.inference_session")
                is not None
            )
        except ImportError:
            # The client registers a meta-path finder that raises a friendly
            # ModuleNotFoundError for full-only modules; treat that as "not available".
            return False

    def _model_management_available(self):
        """True if ``nnInteractive.model_management`` (model list + weight downloads)
        is importable. Old full installs that predate it and the lightweight client
        both lack it, and the Local workflow cannot run without it."""
        try:
            return (
                importlib.util.find_spec("nnInteractive.model_management") is not None
            )
        except ImportError:
            return False

    def _full_backend_ready(self):
        """True if the full backend is importable AND recent enough for this plugin,
        i.e. it includes model_management (required to list models and fetch weights
        in Local mode)."""
        return self._local_inference_available() and self._model_management_available()

    def _plugin_direct_deps_available(self):
        """True if the packages the plugin itself imports (PLUGIN_DIRECT_DEPS --
        currently scikit-image, needed by the freehand lasso) are importable.

        The installer adds them with both flavors, but environments installed or
        updated OUTSIDE the plugin can lack them (the backend no longer declares
        scikit-image). Checked at startup so such an environment gets the update
        offer instead of reporting healthy and failing on the first lasso stroke.
        """
        return importlib.util.find_spec("skimage") is not None

    def _installed_backend_versions(self):
        """Map of installed nnInteractive distribution -> version string, for whichever
        of ('nnInteractive', 'nninteractive-client') are present. Empty when neither is."""
        import importlib.metadata as metadata

        versions = {}
        for dist in ("nnInteractive", "nninteractive-client"):
            try:
                versions[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                continue
        return versions

    def _backend_below_min_version(self):
        """List of "``<dist> <version>``" strings for installed nnInteractive
        distributions older than NNINTERACTIVE_VERSION_FLOOR.

        Empty when everything installed meets the floor, nothing is installed, or a
        version can't be parsed (never nag on an unparseable version). Purely local --
        no network -- so it works offline and is safe to call on the main thread.
        """
        try:
            from packaging.version import Version
        except Exception:  # noqa: BLE001
            return []
        floor = Version(NNINTERACTIVE_VERSION_FLOOR)
        outdated = []
        for dist, ver in self._installed_backend_versions().items():
            try:
                if Version(ver) < floor:
                    outdated.append(f"{dist} {ver}")
            except Exception:  # noqa: BLE001
                continue
        return outdated

    def _min_version_reason(self, outdated):
        """Explanation shown in the update prompt / status label when the installed
        backend is below the supported floor. ``outdated`` is _backend_below_min_version()."""
        return (
            "The nnInteractive backend installed in Slicer's Python is older than the "
            f"minimum version this plugin supports ({NNINTERACTIVE_VERSION_FLOOR}): "
            + "; ".join(outdated)
            + "."
        )

    def _apply_local_mode_availability(self):
        """Gate Local mode on whether the full backend is installed.

        Local mode runs the model in-process and needs the full nnInteractive package
        (torch + nnU-Net). When only the lightweight remote client is installed we
        disable the Local button — and fall back to Remote — so the user never lands on
        a Local page that cannot work. The disabled button's tooltip explains how to
        enable Local support.

        Keyed off the recorded install flavor (the user's active choice) rather than a
        live import check: after switching Full -> Client the uninstalled full package
        can linger in sys.modules until restart, but Local must become unavailable
        immediately. ensure_session still verifies the import before building a local
        session.
        """
        available = self.get_install_flavor() == "full"
        btn = getattr(self.ui, "localModeButton", None)
        if btn is not None:
            btn.setEnabled(available)
            if available:
                btn.setToolTip(
                    "Run inference in-process on this machine's GPU (requires a local GPU)."
                )
            else:
                btn.setToolTip(
                    "Local (in-process) inference is not installed in this Slicer.\n"
                    "Only the lightweight remote client (nninteractive-client) is present.\n\n"
                    "To enable Local mode, click 'Reinstall / Update nnInteractive' in the\n"
                    "Installation section below and choose 'Full (local + remote)'."
                )
        if not available and self.get_mode() != "remote":
            self.set_mode("remote")

    def _sync_mode_switch(self):
        """Reflect the persisted mode on the Local | Remote switch (e.g. after the
        first-run dialog, which fires once the UI is already built)."""
        if not hasattr(self.ui, "localModeButton"):
            return
        is_local = self.get_mode() == "local"
        self.ui.localModeButton.setChecked(is_local)  # setChecked emits toggled, not clicked
        self.ui.remoteModeButton.setChecked(not is_local)
        self._update_mode_visibility()
        # Keep the idle Initialize label ("Initialize (Local/Remote)") in step with the
        # mode, including paths that change it without a teardown (startup, reinstall).
        self.update_connect_status(connected=self.session is not None)

    def setup_prompts_tab_extras(self):
        """
        Adds, at the very top of the Prompts tab (the view the user actually works in),
        a mandatory Initialize/Uninitialize control with a combined status + model-license
        readout. The Helmholtz Imaging + DKFZ acknowledgement logos go at the bottom.

        Initialization is mandatory before any prompt can be placed: it loads the local
        model (and runs the torch.compile / cuDNN warmup) or connects to the remote
        server, and uploads the current image so the first prompt is fast. The
        interaction controls below stay disabled until a session is live, and clicking
        Initialize again -- or changing any session-affecting setting -- uninitializes.
        """
        layout = self.ui.tabPrompts.layout()

        # ===== Initialization (top, above the license) =====
        init_group = qt.QGroupBox("")
        init_layout = qt.QVBoxLayout(init_group)
        init_layout.setContentsMargins(8, 6, 8, 6)
        init_layout.setSpacing(4)

        self.ui.initializeButton = qt.QPushButton("Initialize")
        self.ui.initializeButton.setMinimumHeight(34)
        self.ui.initializeButton.setToolTip(
            "Load the model / connect to the server and upload the current image.\n"
            "Required before any prompt can be placed. Click again to uninitialize.\n"
            "Changing the server, API key, model or any local setting also "
            "uninitializes and requires re-initializing."
        )
        self.ui.initializeButton.clicked.connect(self.connect_clicked)
        init_layout.addWidget(self.ui.initializeButton)

        # Single line combining the session status and the model license (the license
        # used to be its own banner; merged to save vertical space).
        self.ui.promptsStatusLabel = qt.QLabel("Status: not initialized")
        self.ui.promptsStatusLabel.setWordWrap(True)
        self.ui.promptsStatusLabel.setAlignment(qt.Qt.AlignCenter)
        init_layout.addWidget(self.ui.promptsStatusLabel)

        # Persistent (red) warning shown only after a local Initialize that fell back to
        # the CPU because no usable CUDA GPU was found. A transient status message is easy
        # to miss, and CPU inference is very slow, so this stays visible until the next
        # (successful) init. Hidden by default; populated by _update_device_warning().
        self.ui.promptsDeviceWarningLabel = qt.QLabel("")
        self.ui.promptsDeviceWarningLabel.setWordWrap(True)
        self.ui.promptsDeviceWarningLabel.setAlignment(qt.Qt.AlignCenter)
        self.ui.promptsDeviceWarningLabel.setTextFormat(qt.Qt.RichText)
        self.ui.promptsDeviceWarningLabel.setOpenExternalLinks(True)
        self.ui.promptsDeviceWarningLabel.setStyleSheet(
            "color: #d9534f; font-weight: bold; padding: 4px;"
        )
        self.ui.promptsDeviceWarningLabel.setVisible(False)
        init_layout.addWidget(self.ui.promptsDeviceWarningLabel)

        layout.insertWidget(0, init_group)

        logos = self._build_logos_widget(height=26)
        if logos is not None:
            layout.addWidget(logos)

        # Reflect the current (idle) session state on the freshly created button.
        self.update_connect_status(connected=self.session is not None)

    def _build_logos_widget(self, height=26):
        """Returns a small white box with the HI and DKFZ logos side by side (or None)."""
        box = qt.QGroupBox("")
        box.setStyleSheet("QGroupBox { background-color: white; }")
        box_layout = qt.QHBoxLayout(box)
        box_layout.setContentsMargins(8, 6, 8, 6)
        box_layout.setSpacing(14)
        box_layout.addStretch(1)
        any_logo = False
        for filename in ("HI_Logo.png", "DKFZ_Logo.png"):
            pixmap = qt.QPixmap(self.resourcePath(f"Logos/{filename}"))
            if pixmap.isNull():
                continue
            pixmap = pixmap.scaledToHeight(height, qt.Qt.SmoothTransformation)
            logo_label = qt.QLabel()
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(qt.Qt.AlignCenter)
            box_layout.addWidget(logo_label)
            any_logo = True
        box_layout.addStretch(1)
        return box if any_logo else None

    def _license_display_text(self, license_str):
        """Format the model license string for display (matches napari)."""
        if not license_str:
            return "Model license: unknown"
        license_str = license_str.strip()
        if license_str == "!!MISSING!!":
            return "Model license: UNKNOWN (warning!)"
        return f"Model license: {license_str}"

    def on_autozoom_toggled(self, checked):
        self._save_setting_bool("autozoom", checked, reinit=False)
        self._apply_autozoom_to_session(checked)

    def _apply_autozoom_to_session(self, enabled=None):
        """Push the auto-zoom setting to the live session. Both the local and the
        remote session expose set_do_autozoom() (the remote client forwards it to the
        server), so this works in either mode -- and the setting is honoured regardless of
        the compute device (auto-zoom on CPU is slower but still respected). No-op when
        nothing is initialized."""
        if self.session is None:
            return
        if enabled is None:
            enabled = self.get_setting_bool("autozoom")
        try:
            self.session.set_do_autozoom(enabled)
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
        except Exception as exc:  # noqa: BLE001 - a failed toggle must not break the UI
            debug_print(f"Could not apply auto-zoom to the session: {exc}")

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "p": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.on_next_segment,
            "r": self.clear_current_segment,
            "Delete": self.on_delete_segment,  # Delete the currently selected segment
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
            "v": self.toggle_segment_visibility,  # Show/hide the segment being worked on
            "Ctrl+Z": self.on_undo,  # Undo the last nnInteractive interaction
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

    @contextlib.contextmanager
    def _busy_dialog(self, message):
        """Indeterminate busy dialog for MAIN-thread blocking work (shared by the pip
        installers and the local model load).

        Showing a window is an asynchronous round trip through the window server, and
        once the caller blocks the main thread nothing repaints -- whatever was
        composited last would stay on screen (isVisible() flips true before
        compositing, so it is NOT a paint signal). So before yielding, pump the event
        loop until the window reports its real expose (windowHandle().isExposed()),
        with a hard cap for slow/remote displays and a fixed 200 ms fallback when the
        binding doesn't offer the signal. The close is pumped out on exit for the
        same reason.
        """
        progress = slicer.util.createProgressDialog(
            parent=slicer.util.mainWindow(),
            maximum=0,  # indeterminate / busy
            labelText=message,
            windowTitle="nnInteractive",
        )
        progress.show()
        start = time.time()
        while time.time() - start < 1.0:
            slicer.app.processEvents()
            try:
                handle = progress.windowHandle()
                exposed = handle is not None and handle.isExposed()
            except Exception:  # noqa: BLE001 - PythonQt/platform differences
                exposed = None
            if exposed or (exposed is None and time.time() - start >= 0.2):
                break
            time.sleep(0.02)
        slicer.app.processEvents()  # one more pass so the label paint lands
        try:
            yield progress
        finally:
            progress.close()
            slicer.app.processEvents()

    def _pip_install(self, command, message):
        """
        Install pip package(s) on the MAIN thread.

        slicer.util.pip_install creates Qt objects, so it MUST NOT run in a worker
        thread -- doing so throws "QObject ... different thread" errors and deadlocks.
        We show a non-cancelable busy dialog; the UI is blocked while pip runs (a
        one-time cost) and pip's output streams to the Python Console. Failures are
        recorded in ``self._last_pip_error`` and surfaced by the caller
        (_install_full / _install_client).
        """
        self._last_pip_error = None
        with self._busy_dialog(message):
            try:
                slicer.util.pip_install(command)
            except Exception as exc:  # noqa: BLE001
                self._last_pip_error = exc
                debug_print(f"pip install '{command}' failed: {exc}")

    def _pip_uninstall(self, command, message):
        """pip-uninstall package(s) on the MAIN thread, with a busy dialog (mirrors
        _pip_install). pip uninstall removes only the named distributions, never their
        dependencies. Failures are recorded in ``self._last_pip_error``; the subsequent
        install (which resets that flag) is the source of truth for success."""
        self._last_pip_error = None
        with self._busy_dialog(message):
            try:
                slicer.util.pip_uninstall(command)
            except Exception as exc:  # noqa: BLE001
                self._last_pip_error = exc
                debug_print(f"pip uninstall '{command}' failed: {exc}")

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
        # Wait on the event with a short timeout (which releases the GIL so the worker
        # runs at full speed) and pump the Qt event loop between waits. A tight
        # processEvents() spin would keep re-acquiring the GIL and starve the worker.
        while not parallel_event.wait(0.03):
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()
        # Make sure the close is actually rendered before the caller potentially
        # blocks the main thread (e.g. the model load right after the weights
        # download) -- otherwise this dialog lingers frozen on screen.
        slicer.app.processEvents()

    def _run_thread_with_message(self, target, args, message):
        """Run ``target(*args, done_event)`` in a worker thread while showing a simple
        modal message dialog -- a label only, NO progress bar -- and pumping the Qt
        event loop so it stays painted.

        Used for opaque, unmeasurable work like the image upload: the client sends the
        image in one blocking call with no byte-level progress callback, so a progress
        bar could only ever jump 0%->done, which is misleading. A plain "please wait"
        message is the honest UI here.
        """
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("nnInteractive")
        dialog.setModal(True)
        # No close/help buttons: the work cannot be cancelled and the dialog is closed
        # programmatically once the worker finishes.
        dialog.setWindowFlags(
            qt.Qt.Dialog | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint
        )
        dialog_layout = qt.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(24, 20, 24, 20)
        dialog_layout.addWidget(qt.QLabel(message))
        dialog.show()
        slicer.app.processEvents()

        done_event = threading.Event()
        worker = threading.Thread(target=target, args=(*args, done_event))
        worker.start()
        # Wait on the event with a short timeout (which releases the GIL so the worker
        # runs at full speed) and pump the Qt event loop between waits to keep the dialog
        # responsive. A tight processEvents() spin would instead keep re-acquiring the
        # GIL and starve the worker, making the work much slower.
        while not done_event.wait(0.03):
            slicer.app.processEvents()
        worker.join()

        dialog.close()
        dialog.deleteLater()
        slicer.app.processEvents()

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        self._cleanup_in_progress = True
        self._pending_cursor_reassert_widgets = []
        self._cursor_reassert_timer_queued = False

        self.removeObservers()
        self._remove_slice_event_filters()
        self._remove_scribble_labelmap_observer()
        # Deactivate Segment Editor effects (Paint etc.) before the scene/widgets are
        # destroyed, so effect-owned Qt timers don't fire during teardown (avoids the
        # "QBasicTimer ... QThread" warnings on exit).
        self._deactivate_segment_editor_effects()

        if not getattr(self, "_application_quitting", False):
            # Restore the default slice-view cursor (we may have set a coloured one).
            self._clear_prompt_cursor()

            # Remove the freehand-lasso overlay actors from their slice view.
            self._lasso_overlay_remove()

        # Release any remote lease / free the local model.
        self.release_session()

        if not getattr(self, "_application_quitting", False):
            # Module reload/close (scene still valid): remove the prompt/scribble nodes
            # we created and drop our references so they are freed rather than leaked.
            # (At app exit, _on_application_about_to_quit already did this while the
            # scene was valid, so skip it here to avoid touching a torn-down scene.)
            try:
                self.remove_prompt_nodes()
            except Exception:  # noqa: BLE001
                pass
            self.target_buffer = None

        # Tear down the programmatic scribble editor widget we own (parentless QObject).
        self._destroy_scribble_editor_widget()

        self._stop_update_poll_timer()
        self._disconnect_plugin_update_signals()

        if not getattr(self, "_application_quitting", False):
            try:
                slicer.app.aboutToQuit.disconnect(self._on_application_about_to_quit)
            except Exception:  # noqa: BLE001
                pass

        self.remove_shortcut_items()

    def _on_application_about_to_quit(self):
        """Stop Qt-driven callbacks before Slicer starts destroying views."""
        self._application_quitting = True
        self._cleanup_in_progress = True
        self._pending_cursor_reassert_widgets = []
        self._cursor_reassert_timer_queued = False
        self._remove_slice_event_filters()
        self._remove_scribble_labelmap_observer()
        self._deactivate_segment_editor_effects()
        self._stop_heartbeat_timer()
        self._stop_update_poll_timer()
        self._disconnect_plugin_update_signals()
        # aboutToQuit fires while the scene is still valid: remove the prompt/scribble
        # nodes we created (and drop our references) now, so their VTK pipelines are
        # freed before the scene is torn down instead of lingering as reported leaks.
        try:
            self.remove_prompt_nodes()
        except Exception:  # noqa: BLE001 - never block shutdown
            pass
        self.target_buffer = None

    def _remove_slice_event_filters(self):
        if hasattr(self, "_qt_event_filters"):
            for target, event_filter in self._qt_event_filters:
                try:
                    target.removeEventFilter(event_filter)
                except Exception:  # noqa: BLE001
                    pass
            self._qt_event_filters = []

    def _deactivate_segment_editor_effects(self):
        """Deactivate the active effect on both Segment Editor widgets.

        qMRMLSegmentEditorWidget effects (e.g. Paint, used by the scribble tool) own
        their own Qt timers and scene observers. If an effect is still active when
        Slicer destroys the widgets / closes the scene at shutdown, those timers get
        poked after the thread's Qt event dispatcher is gone, which Qt reports as
        ``QBasicTimer::start: QBasicTimer can only be used with threads started with
        QThread``. Deactivating the effect first (as Slicer's own Segment Editor module
        does on exit) releases those timers cleanly. Safe to call repeatedly and during
        teardown -- every call is guarded.
        """
        for widget in (
            getattr(self.ui, "editor_widget", None),
            getattr(self, "scribble_editor_widget", None),
        ):
            if widget is None:
                continue
            try:
                widget.setActiveEffectByName("")
            except Exception:  # noqa: BLE001
                pass

    def _destroy_scribble_editor_widget(self):
        """Detach the programmatic scribble Segment Editor widget from MRML and delete it.

        We create this qMRMLSegmentEditorWidget ourselves with no Qt parent, so it would
        otherwise outlive the scene at shutdown (taking its observers/timers with it).
        Cleared references are re-created lazily by setup_scribble_prompt() if needed.
        """
        widget = getattr(self, "scribble_editor_widget", None)
        if widget is None:
            return
        for setter in ("setSegmentationNode", "setMRMLSegmentEditorNode", "setMRMLScene"):
            try:
                getattr(widget, setter)(None)
            except Exception:  # noqa: BLE001
                pass
        try:
            widget.deleteLater()
        except Exception:  # noqa: BLE001
            pass
        self.scribble_editor_widget = None

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def _volume_node_id(self, volume_node):
        if volume_node is None:
            return None
        try:
            return volume_node.GetID()
        except Exception:  # noqa: BLE001
            return None

    def on_segment_editor_node_modified(self, caller, event):
        """Reset transient prompt state when the Segment Editor source volume changes,
        and clear the displayed prompts when the user selects a different segment."""
        self._handle_active_source_volume_change(reupload=True)
        self._handle_selected_segment_change()
        self._update_initialize_button_state()

    def on_scene_nodes_changed(self, caller, event, calldata=None):
        """Volumes added/removed: re-evaluate whether Initialize can be used. Coalesced
        via a queued single-shot: a scene load fires one event per node, and re-scanning
        the scene for volumes on every one of them made big loads O(n^2)."""
        if self._is_tearing_down():
            return
        if getattr(self, "_scene_change_update_queued", False):
            return
        self._scene_change_update_queued = True

        def _update_once():
            self._scene_change_update_queued = False
            if not self._is_tearing_down():
                # Full status sync (not just the button): the status line's "load an
                # image..." hint must clear the moment a volume is loaded, and reappear
                # if all volumes are removed. update_connect_status re-runs
                # _update_initialize_button_state itself.
                self.update_connect_status(connected=self.session is not None)

        qt.QTimer.singleShot(0, _update_once)

    def _handle_selected_segment_change(self):
        """When the user picks a different segment to refine, the session re-seeds it
        (lazily, on the next prompt) and resets its interactions -- so the prompts left
        over from the previous segment must be cleared too, otherwise they bleed into
        the newly selected segment. Programmatic selections are guarded out."""
        # Never touch nodes while the module/scene is being torn down: the Segment Editor
        # node fires Modified events during shutdown, and doing heavy work (reset_all_prompts,
        # node access) on a dying scene crashes the app on exit.
        if self._is_tearing_down():
            return
        if getattr(self, "_suppress_segment_switch", False):
            return
        node = getattr(self, "segment_editor_node", None)
        if node is None:
            return
        seg_id = node.GetSelectedSegmentID()
        prev = getattr(self, "_last_selected_segment_id", None)
        if seg_id == prev:
            return
        self._last_selected_segment_id = seg_id
        # Only act on a genuine switch between two existing segments (not the first
        # selection or a cleared selection), and only once the UI is built.
        if prev is None or not seg_id or not getattr(self, "all_prompt_buttons", None):
            return
        active_tool = self._active_prompt_tool()
        self.reset_all_prompts()  # clears markups + overlays + interactions, like Next/Reset
        self._rearm_tool_later(active_tool)

    def _handle_active_source_volume_change(self, volume_node=None, reupload=False):
        if self._is_tearing_down():
            return False
        if getattr(self, "_handling_source_volume_change", False):
            return False

        if volume_node is None:
            volume_node = self.get_volume_node()
        volume_id = self._volume_node_id(volume_node)
        if volume_id is None:
            return False

        previous_id = getattr(self, "_active_source_volume_id", None)
        if previous_id == volume_id:
            return False

        self._active_source_volume_id = volume_id
        if previous_id is None:
            return False

        self._handling_source_volume_change = True
        try:
            debug_print("Source volume changed. Resetting prompts; uploading the new image.")
            # Keep the live session (no full re-initialize -- that would needlessly reload
            # the model); we just reset the prompts and push the newly selected image to
            # the session below.
            session_live = self.session is not None
            active_tool = (
                self._active_prompt_tool()
                if getattr(self, "all_prompt_buttons", None)
                else None
            )
            if getattr(self, "scribble_editor_widget", None) is not None:
                self.scribble_editor_widget.setActiveEffectByName("")
            self._remove_scribble_labelmap_observer()
            self._cancel_lasso_stroke()
            self.previous_states.pop("image_data", None)
            self.previous_states.pop("segment_fp", None)
            self.target_buffer = None
            self._prev_scribble_masks = {}

            if getattr(self, "prompt_types", None) and getattr(
                self, "all_prompt_buttons", None
            ):
                self.setup_prompts()
                self._rearm_tool_later(active_tool)

            # Push the newly selected image to the live session. Deferred to the event
            # loop so we don't run a blocking upload inside this VTK modified-event
            # callback; _reupload_image_after_volume_change shows the same progress dialog
            # as Initialize, so the user sees why Slicer pauses. Only the editor observer
            # asks for this (reupload=True); the defensive ensure_synched caller lets its
            # own image_changed()/sync handle the upload instead.
            if reupload and session_live:
                qt.QTimer.singleShot(0, self._reupload_image_after_volume_change)
            return True
        finally:
            self._handling_source_volume_change = False

    def _reupload_image_after_volume_change(self):
        """Upload the newly selected source volume to the live session (mirrors what
        Initialize does), keeping the session instead of re-initializing. Shows the
        Initialize progress dialog so a slow (especially remote) upload is visible."""
        if self.session is None:
            return
        if self._is_tearing_down():
            return
        try:
            self._preload_image_and_segment()
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
        except Exception as exc:  # noqa: BLE001 - don't leave a half-synced session
            self.release_session()
            self.update_connect_status(connected=False)
            slicer.util.errorDisplay(
                f"Could not upload the new image to nnInteractive:\n\n{exc}",
                parent=slicer.util.mainWindow(),
            )

    def _get_scribble_prompt_node(self):
        node = getattr(self, "scribble_segment_node", None)
        try:
            if node is not None and node.GetScene() is slicer.mrmlScene:
                return node
        except Exception:  # noqa: BLE001
            pass

        node = slicer.mrmlScene.GetFirstNodeByName(self.scribble_segment_node_name)
        if node is not None:
            self.scribble_segment_node = node
        return node

    def _scribble_prompt_matches_volume(self, volume_node=None):
        if volume_node is None:
            volume_node = self.get_volume_node()
        volume_id = self._volume_node_id(volume_node)
        return (
            volume_id is not None
            and self._get_scribble_prompt_node() is not None
            and getattr(self, "_scribble_reference_volume_id", None) == volume_id
        )

    def _remove_scribble_prompt_node(self):
        self._remove_scribble_labelmap_observer()
        if getattr(self, "scribble_editor_widget", None) is not None:
            self.scribble_editor_widget.setActiveEffectByName("")

        existing_nodes = slicer.mrmlScene.GetNodesByName(self.scribble_segment_node_name)
        if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
            for i in range(existing_nodes.GetNumberOfItems()):
                node = existing_nodes.GetItemAsObject(i)
                slicer.mrmlScene.RemoveNode(node)

        self.scribble_segment_node = None
        self._scribble_reference_volume_id = None
        self._prev_scribble_masks = {}

    def setup_prompts(self, skip_if_exists=False):
        volume_node = self.get_volume_node()
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

            prompt_type["node"] = node
            # Connect each button's `clicked` only once for the widget's lifetime (see
            # _connected_prompt_buttons); the lambda captures only prompt_name/self, so it
            # stays valid even as the underlying prompt node is recreated.
            if prompt_name not in self._connected_prompt_buttons:
                prompt_type["button"].clicked.connect(
                    lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)
                )
                self._connected_prompt_buttons.add(prompt_name)
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}"))
            prompt_type["button"].setIcon(icon)

        scribble_matches_volume = self._scribble_prompt_matches_volume(volume_node)
        if not skip_if_exists or not scribble_matches_volume:
            if skip_if_exists and not scribble_matches_volume:
                self._remove_scribble_prompt_node()
            self.setup_scribble_prompt(volume_node)

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

    def setup_scribble_prompt(self, volume_node=None):
        """
        Creates a hidden "Segment Editor" for the scribble prompt plus a fresh, empty
        scribble segmentation node. The editor widget and its SegmentEditorNode are
        created once and reused -- recreating them on every call would leak an MRML
        node (and a Qt widget) each time, and this runs on every 'Next segment'.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        if volume_node is None:
            volume_node = self.get_volume_node()

        self._remove_scribble_labelmap_observer()
        if getattr(self, "scribble_editor_widget", None) is not None:
            self.scribble_editor_widget.setActiveEffectByName("")

        # Create the background (headless) segment editor once, then reuse it.
        if getattr(self, "scribble_editor_widget", None) is None:
            self.scribble_editor_widget = (
                qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
            )
            self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
            self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        if getattr(self, "scribble_editor_node", None) is None:
            self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)
            # Don't overwrite other segments when painting: positive (fg) and negative
            # (bg) scribbles -- and a scribble overlapping a stamped lasso -- must be able
            # to coexist and stack their (translucent) fills instead of one erasing the
            # other where they overlap. Without this the earlier prompt vanishes under a
            # later overlapping one of the opposite polarity.
            try:
                self.scribble_editor_node.SetOverwriteMode(
                    slicer.vtkMRMLSegmentEditorNode.OverwriteNone
                )
            except Exception as exc:  # noqa: BLE001 - non-fatal display/edit tweak
                debug_print(f"Could not set scribble overwrite mode: {exc}")

        # The previous scribble segmentation node (if any) was removed by
        # remove_prompt_nodes(); create a fresh, empty one.
        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        if volume_node is not None:
            self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(
                volume_node
            )
        self._scribble_reference_volume_id = self._volume_node_id(volume_node)
        if self._scribble_reference_volume_id is not None:
            self._active_source_volume_id = self._scribble_reference_volume_id
        self.scribble_segment_node.SetName(self.scribble_segment_node_name)

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        # bg = negative scribble (red), fg = positive scribble (green).
        segmentation = self.scribble_segment_node.GetSegmentation()
        segmentation.AddEmptySegment("bg", "bg", list(self.COLOR_NEGATIVE))
        segmentation.AddEmptySegment("fg", "fg", list(self.COLOR_POSITIVE))
        # The colour argument to AddEmptySegment is silently ignored on some Slicer
        # builds, leaving auto-assigned colours that don't track prompt polarity. Set
        # the colours explicitly so a positive scribble is always green and a negative
        # one red.
        segmentation.GetSegment("bg").SetColor(*self.COLOR_NEGATIVE)
        segmentation.GetSegment("fg").SetColor(*self.COLOR_POSITIVE)

        dn = self.scribble_segment_node.GetDisplayNode()
        # Fill is kept translucent so the underlying image shows through; the outline
        # is fully opaque so the polarity colour reads clearly.
        for seg_id in ("bg", "fg"):
            dn.SetSegmentOpacity2DFill(seg_id, 0.4)
            dn.SetSegmentOpacity2DOutline(seg_id, 1.0)

        # Per-label ("fg"/"bg") accumulated paint, used to diff out only new strokes.
        self._prev_scribble_masks = {}
        # Fresh node has no per-instance prompt segments yet; restart their numbering.
        self._prompt_overlay_counter = 0

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
            # Drop our Python reference to the just-removed node. Otherwise the dict
            # keeps the node (and the large VTK markups pipeline it owns -- curve
            # generators, measurements, transforms, locators ...) alive until process
            # exit, which is what shows up in vtkDebugLeaks. setup_prompts() repopulates
            # this when it recreates the node.
            prompt_type["node"] = None

        self._remove_scribble_prompt_node()

    def on_interaction_node_modified(self, caller, event):
        """
        Keep the markups prompt buttons in sync with the interaction mode. The lasso
        is handled separately (it is a freehand drag, not a Place interaction) and is
        left untouched here; entering Place mode for another prompt cancels an active
        freehand lasso and scribble.
        """
        # Don't touch (possibly half-released) prompt nodes/widgets during teardown.
        if self._is_tearing_down():
            return
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        in_place = (
            interactionNode.GetCurrentInteractionMode()
            == slicer.vtkMRMLInteractionNode.Place
        )
        # All prompt tools run in ViewTransform mode with their own click capture, so
        # button state is not driven by markups Place mode here. Place mode can still be
        # entered from elsewhere in Slicer (e.g. the Markups module) -- cancel our tools.
        if in_place:
            self.ui.pbInteractionScribble.setChecked(False)
            self.set_lasso_active(False)

        # Recolour the slice-view cursor for the now-active tool (or restore default).
        self._update_prompt_cursor()

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

    # --- Per-interaction visual undo -------------------------------------------------
    # Each prompt sent to the session pushes one closure that knows how to remove just
    # that prompt's on-screen marker. on_undo() pops one after session.undo() succeeds.
    def _push_prompt_undo(self, fn):
        if getattr(self, "_prompt_undo_stack", None) is None:
            self._prompt_undo_stack = []
        self._prompt_undo_stack.append(fn)

    def _pop_prompt_undo(self):
        stack = getattr(self, "_prompt_undo_stack", None)
        if stack:
            fn = stack.pop()
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - visual-only; never block undo
                debug_print(f"prompt-visual undo failed: {exc}")

    def _clear_prompt_undo(self):
        self._prompt_undo_stack = []

    def _make_control_point_undo(self, node_id, cp_index):
        """Undo closure that removes a single placed markups control point (point prompt)."""

        def _undo():
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and 0 <= cp_index < node.GetNumberOfControlPoints():
                node.RemoveNthControlPoint(cp_index)

        return _undo

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        if prompt_name == "lasso":
            # The lasso is a freehand drag, not a markups Place interaction. Keep the
            # views in view-transform mode and toggle our custom drag capture instead.
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            self.set_lasso_active(checked)
            return

        if prompt_name in ("point", "bbox"):
            # Point and BBox use the same model as the lasso: stay in ViewTransform so
            # right-click zoom/pan work natively, and capture left-clicks via the slice-
            # view event filter (point = click; bbox = drag-and-drop). This avoids
            # markups Place mode, which consumed the right-click and forced the
            # second-click-to-zoom behaviour.
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            if checked:
                self._place_tool = prompt_name
                self._cancel_bbox_drag()  # start a fresh box
                # Mutual exclusivity with the other tools.
                self.set_lasso_active(False)
                if self.ui.pbInteractionScribble.isChecked():
                    self.ui.pbInteractionScribble.click()
                sibling = "bbox" if prompt_name == "point" else "point"
                if self.all_prompt_buttons[sibling].isChecked():
                    self.all_prompt_buttons[sibling].setChecked(False)
                self._apply_prompt_cursor(prompt_name)
            else:
                self._place_tool = None
                self._cancel_bbox_drag()
                self._update_prompt_cursor()
            return
        # (no other prompt types exist; scribble has its own handler)

    # Prompt colours: positive = green, negative = red.
    COLOR_POSITIVE = (0.20, 0.85, 0.20)
    COLOR_NEGATIVE = (0.90, 0.15, 0.15)

    def _polarity_color(self, positive):
        """The single positive->green / negative->red mapping used by every prompt visual."""
        return self.COLOR_POSITIVE if positive else self.COLOR_NEGATIVE

    def _set_node_color(self, node, positive):
        """Colour a whole markup node (bbox / lasso) green (pos) or red (neg)."""
        if node is None:
            return
        display_node = node.GetDisplayNode()
        if display_node is None:
            return
        color = self._polarity_color(positive)
        display_node.SetColor(*color)
        display_node.SetSelectedColor(*color)
        display_node.SetActiveColor(*color)
        if hasattr(display_node, "SetSliceProjectionColor"):
            display_node.SetSliceProjectionColor(*color)

    ###############################################################################
    # Interaction cursors (green = positive, red = negative)
    ###############################################################################
    # The slice-view mouse cursor is recoloured per prompt polarity so the user gets
    # immediate feedback about whether the next stroke will be a positive (green) or
    # negative (red) prompt -- without having to glance at the Positive/Negative
    # buttons. Each prompt type also carries a small badge so the four tools stay
    # visually distinct. (Scribble additionally shows a coloured Paint brush circle;
    # should the Paint effect manage its own cursor, that circle still conveys the
    # polarity colour.)

    def _qcolor(self, rgb):
        return qt.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def _get_prompt_cursor(self, tool, positive):
        key = (tool, bool(positive))
        if key not in self._prompt_cursor_cache:
            self._prompt_cursor_cache[key] = self._make_prompt_cursor(tool, positive)
        return self._prompt_cursor_cache[key]

    def _make_prompt_cursor(self, tool, positive):
        """Build a 32x32 colour cursor: a precise crosshair locator plus a per-tool
        badge, drawn with a dark halo so it reads on both bright and dark images."""
        size = 32
        pm = qt.QPixmap(size, size)
        pm.fill(qt.QColor(0, 0, 0, 0))  # transparent
        p = qt.QPainter(pm)
        p.setRenderHint(qt.QPainter.Antialiasing, True)
        color = self._qcolor(self._polarity_color(positive))
        halo = qt.QColor(0, 0, 0, 190)
        cx, cy = 16, 16

        def crosshair(pen_color, w):
            pen = qt.QPen(pen_color)
            pen.setWidth(w)
            pen.setCapStyle(qt.Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(qt.QBrush())
            gap, arm = 3, 7
            p.drawLine(cx - arm, cy, cx - gap, cy)
            p.drawLine(cx + gap, cy, cx + arm, cy)
            p.drawLine(cx, cy - arm, cx, cy - gap)
            p.drawLine(cx, cy + gap, cx, cy + arm)

        def badge(pen_color, w, filled):
            pen = qt.QPen(pen_color)
            pen.setWidth(w)
            pen.setCapStyle(qt.Qt.RoundCap)
            pen.setJoinStyle(qt.Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(qt.QBrush(pen_color) if filled else qt.QBrush())
            if tool == "point":
                p.drawEllipse(21, 3, 8, 8)
            elif tool == "bbox":
                p.drawRect(21, 4, 8, 8)
            elif tool == "lasso":
                p.drawEllipse(20, 3, 10, 8)
                p.drawLine(25, 11, 28, 14)  # little tail
            elif tool == "scribble":
                # A smooth squiggle drawn as a cubic Bezier chain.
                path = qt.QPainterPath()
                path.moveTo(19.5, 9.0)
                path.cubicTo(21.0, 3.0, 23.0, 3.0, 24.5, 8.0)
                path.cubicTo(26.0, 13.0, 28.0, 13.0, 29.5, 6.5)
                p.drawPath(path)

        # Halo pass (wide, dark) then colour pass (narrow) so the glyph stays legible
        # over any image intensity.
        crosshair(halo, 4)
        badge(halo, 4, False)
        crosshair(color, 2)
        badge(color, 2, tool == "point")
        p.end()
        return qt.QCursor(pm, cx, cy)

    def _is_tearing_down(self):
        """True while the widget is being cleaned up or the application is quitting.
        Qt/VTK callbacks must bail out instead of touching half-torn-down state.
        getattr defaults: callbacks can fire before __init__ sets the flags."""
        return getattr(self, "_cleanup_in_progress", False) or getattr(
            self, "_application_quitting", False
        )

    def _slice_widgets(self):
        """Slice widgets whose slice views receive prompt interactions."""
        if self._is_tearing_down():
            return []
        widgets = []
        lm = slicer.app.layoutManager()
        if lm is None:
            return widgets
        for name in lm.sliceViewNames():
            sw = lm.sliceWidget(name)
            if sw is not None:
                widgets.append(sw)
        return widgets

    def _slice_view_widgets(self):
        """The slice-view widgets (where the user clicks/drags prompts)."""
        views = []
        for sw in self._slice_widgets():
            try:
                views.append(sw.sliceView())
            except Exception:  # noqa: BLE001
                pass
        return views

    def _slice_view_cursor_targets(self, slice_widget):
        """Return the slice view plus its VTK/OpenGL child widgets.

        Slicer/VTK often updates the cursor on the lower-level render widget rather
        than on the qMRMLSliceView wrapper, so setting only the slice view leaves the
        stock markups or Paint cursor in control.
        """
        targets = []

        def add(widget):
            if widget is None or not hasattr(widget, "setCursor"):
                return
            if any(widget is existing for existing in targets):
                return
            targets.append(widget)

        try:
            slice_view = slice_widget.sliceView()
        except Exception:  # noqa: BLE001
            slice_view = slice_widget
        add(slice_view)

        def add_children(parent):
            try:
                children = parent.children()
            except Exception:  # noqa: BLE001
                return
            for child in children:
                add(child)
                add_children(child)

        add_children(slice_view)
        return targets

    def _prepare_cursor_target(self, widget):
        try:
            widget.setMouseTracking(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            widget.setAttribute(qt.Qt.WA_Hover, True)
        except Exception:  # noqa: BLE001
            pass

    def _set_cursor_on_slice_widget(self, slice_widget, cursor):
        for widget in self._slice_view_cursor_targets(slice_widget):
            self._prepare_cursor_target(widget)
            try:
                widget.setCursor(cursor)
            except Exception:  # noqa: BLE001
                pass

    def _apply_prompt_cursor(self, tool):
        # The filter is used for every prompt type, not only lasso: markups and Paint
        # can reset the cursor while the mouse is already over the render widget.
        self._install_lasso_filters()
        cursor = self._get_prompt_cursor(tool, self.is_positive)
        for sw in self._slice_widgets():
            self._set_cursor_on_slice_widget(sw, cursor)

    def _clear_prompt_cursor(self):
        for sw in self._slice_widgets():
            for w in self._slice_view_cursor_targets(sw):
                try:
                    w.unsetCursor()
                except Exception:  # noqa: BLE001
                    try:
                        w.setCursor(qt.QCursor(qt.Qt.ArrowCursor))
                    except Exception:  # noqa: BLE001
                        pass

    def _reassert_cursor_on_view(self, slice_widget):
        if self._is_tearing_down():
            return
        if not getattr(self, "all_prompt_buttons", None):
            return
        tool = self._active_prompt_tool()
        if tool is None:
            return
        cursor = self._get_prompt_cursor(tool, self.is_positive)
        self._set_cursor_on_slice_widget(slice_widget, cursor)

    def _queue_cursor_reassert(self, slice_widget):
        if self._is_tearing_down():
            return
        pending = getattr(self, "_pending_cursor_reassert_widgets", None)
        if pending is None:
            pending = []
            self._pending_cursor_reassert_widgets = pending
        if not any(slice_widget is existing for existing in pending):
            pending.append(slice_widget)
        if getattr(self, "_cursor_reassert_timer_queued", False):
            return
        self._cursor_reassert_timer_queued = True
        qt.QTimer.singleShot(0, self._flush_cursor_reasserts)

    def _flush_cursor_reasserts(self):
        self._cursor_reassert_timer_queued = False
        if self._is_tearing_down():
            self._pending_cursor_reassert_widgets = []
            return
        pending = list(getattr(self, "_pending_cursor_reassert_widgets", []))
        self._pending_cursor_reassert_widgets = []
        for slice_widget in pending:
            self._reassert_cursor_on_view(slice_widget)

    def _update_prompt_cursor(self):
        """Single entry point: colour the slice-view cursor for whichever prompt tool
        is active (green/red by polarity), or restore the default cursor when none is.
        Safe to call after any tool- or polarity-state change."""
        if self._is_tearing_down():
            return
        if not getattr(self, "all_prompt_buttons", None):
            return
        tool = self._active_prompt_tool()
        if tool is None:
            self._clear_prompt_cursor()
        else:
            self._apply_prompt_cursor(tool)

    def _restrict_markup_to_2d(self, display_node):
        """Show a prompt markup only in the slice (2D) views, never in 3D.

        nnInteractive prompts are 2D-only by design (lasso/scribble live in a labelmap
        overlay and aren't rendered in 3D), so for consistency the point and bbox
        markups are restricted to the slice views too -- otherwise points (and the bbox
        box) would be the only prompts appearing in the 3D view. Done by limiting the
        display node to the slice view nodes; there is no per-node 3D-visibility flag.
        """
        if display_node is None:
            return
        try:
            slice_nodes = slicer.util.getNodesByClass("vtkMRMLSliceNode")
            if not slice_nodes:
                # No slice views found -- don't restrict to nothing (that would hide the
                # markup everywhere); leave it as-is.
                return
            display_node.SetVisibility2D(True)
            display_node.RemoveAllViewNodeIDs()
            for slice_node in slice_nodes:
                display_node.AddViewNodeID(slice_node.GetID())
        except Exception as exc:  # noqa: BLE001 - display tweak only, never fatal
            debug_print(f"Could not restrict markup to 2D views: {exc}")

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node. Points use the per-control-
        point "selected" flag to pick a colour: selected -> green (positive),
        unselected -> red (negative). See _place_point_from_event().
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(*self.COLOR_NEGATIVE)  # unselected control point = negative
        display_node.SetSelectedColor(*self.COLOR_POSITIVE)  # selected control point = positive
        display_node.SetActiveColor(1.0, 1.0, 0.0)  # while actively placing
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views
        self._restrict_markup_to_2d(display_node)  # prompts are 2D-only (no 3D rendering)

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(1.0)  # crisp, opaque box outline even when nested
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)
        self._restrict_markup_to_2d(display_node)  # prompts are 2D-only (no 3D rendering)

    def display_node_markup_lasso(self, display_node):
        """
        The freehand lasso no longer uses a markups node for drawing -- the live
        contour and its fill are rendered with VTK 2D overlay actors (see
        _lasso_overlay_*). The markups "LassoPrompt" node is kept only so the existing
        prompt/button plumbing stays uniform; hide it so it never renders anything.
        """
        display_node.SetVisibility(False)
        display_node.SetGlyphScale(0)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        """
        Adds a point interaction to the nnInteractive session.
        ``xyz`` is in IJK (i, j, k) order; the session image is [1, k, j, i], so
        coordinates are reversed to match the array's spatial-axis order.
        """
        debug_print(f"{positive_click} point prompt triggered! {xyz}")
        t0 = time.time()
        changed_bbox = self.session.add_point_interaction(
            xyz[::-1],
            include_interaction=bool(positive_click),
            run_prediction=True,
        )
        t1 = time.time()
        self.apply_result(changed_bbox)
        debug_print(
            f"[timing] add_point_interaction {t1 - t0:.3f}s | apply_result {time.time() - t1:.3f}s"
        )

    def _deactivate_capture_tools(self):
        """Turn off the click-capture tools (point/bbox) when another tool takes over."""
        self._place_tool = None
        self._cancel_bbox_drag()
        for btn in (self.ui.pbInteractionPoint, self.ui.pbInteractionBBox):
            if btn.isChecked():
                btn.setChecked(False)

    def _place_point_from_event(self, slice_widget, event, event_widget=None):
        """Place a point prompt at a captured left-click (point tool, ViewTransform mode).

        Mirrors the lasso's display->RAS->IJK conversion, adds the visual control point
        that markups Place mode used to add for us, and sends the point to the session.
        """
        slice_node = slice_widget.mrmlSliceNode()
        slice_view = slice_widget.sliceView()
        if slice_node is None or self.get_volume_node() is None:
            return
        node = self.prompt_types["point"]["node"]
        if node is None:
            return
        xy = self._event_xy_in_slice_view(slice_view, event, event_widget)
        disp = self._event_display_xy(slice_view, xy)
        ras = slice_node.GetXYToRAS().MultiplyPoint([disp[0], disp[1], 0.0, 1.0])
        ras3 = [ras[0], ras[1], ras[2]]
        xyz = self.ras_to_xyz(ras3)
        # Visual marker (markups Place mode used to add this for us); colour by polarity.
        idx = node.AddControlPointWorld(vtk.vtkVector3d(ras3[0], ras3[1], ras3[2]))
        node.SetNthControlPointSelected(idx, self.is_positive)
        node.SetNthControlPointLocked(idx, True)
        self.point_prompt(xyz=xyz, positive_click=self.is_positive)
        self._push_prompt_undo(
            self._make_control_point_undo(node.GetID(), node.GetNumberOfControlPoints() - 1)
        )

    # --- BBox drag-and-drop (press = first corner, drag = live box, release = box) ----
    def _handle_bbox_drag_event(self, slice_widget, event, event_widget=None):
        """Capture a left-button drag as a bounding box. Returns True if consumed."""
        etype = event.type()
        if etype == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            self._begin_bbox_drag(slice_widget, event, event_widget)
            return True
        if etype == qt.QEvent.MouseMove and (event.buttons() & qt.Qt.LeftButton):
            self._update_bbox_drag(event, event_widget)
            return True
        if etype == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
            self._finish_bbox_drag(event, event_widget)
            return True
        if etype == qt.QEvent.MouseButtonDblClick and event.button() == qt.Qt.LeftButton:
            return True
        return False

    def _begin_bbox_drag(self, slice_widget, event, event_widget=None):
        slice_node = slice_widget.mrmlSliceNode()
        slice_view = slice_widget.sliceView()
        if slice_node is None or self.get_volume_node() is None:
            self._cancel_bbox_drag()
            return
        xy = self._event_xy_in_slice_view(slice_view, event, event_widget)
        disp = self._event_display_xy(slice_view, xy)
        ras = slice_node.GetXYToRAS().MultiplyPoint([disp[0], disp[1], 0.0, 1.0])
        color = self._polarity_color(self.is_positive)
        self._bbox_preview_create(slice_view, color)
        self._bbox_drag = {
            "slice_widget": slice_widget,
            "slice_view": slice_view,
            "press_disp": disp,
            "press_ras": [ras[0], ras[1], ras[2]],
        }

    def _update_bbox_drag(self, event, event_widget=None):
        drag = getattr(self, "_bbox_drag", None)
        if not drag:
            return
        xy = self._event_xy_in_slice_view(drag["slice_view"], event, event_widget)
        disp = self._event_display_xy(drag["slice_view"], xy)
        self._bbox_preview_update(drag["press_disp"], disp)

    def _finish_bbox_drag(self, event, event_widget=None):
        drag = getattr(self, "_bbox_drag", None)
        self._bbox_drag = None
        self._bbox_preview_remove()
        if not drag or self.get_volume_node() is None:
            return
        slice_node = drag["slice_widget"].mrmlSliceNode()
        if slice_node is None:
            return
        xy = self._event_xy_in_slice_view(drag["slice_view"], event, event_widget)
        disp = self._event_display_xy(drag["slice_view"], xy)
        ras = slice_node.GetXYToRAS().MultiplyPoint([disp[0], disp[1], 0.0, 1.0])
        first_xyz = self.ras_to_xyz(drag["press_ras"])
        second_xyz = self.ras_to_xyz([ras[0], ras[1], ras[2]])
        if first_xyz == second_xyz:
            return  # zero-size box (a click, not a drag) -- ignore
        self.bbox_prompt(
            outer_point_one=first_xyz,
            outer_point_two=second_xyz,
            positive_click=self.is_positive,
        )
        # Persist the box as a hollow outline in its OWN overlay segment, so it stays
        # visible (with its own outline, stacking with anything it overlaps) and tracks
        # pan/zoom/slice like the scribble/lasso prompts. The live VTK preview is
        # display-space only and is removed on release.
        crop, bbox = self._bbox_outline_crop(first_xyz, second_xyz)
        undo = (
            self._add_prompt_overlay_segment(crop, bbox, self.is_positive)
            if crop is not None
            else None
        )
        self._push_prompt_undo(undo if undo is not None else (lambda: None))

    def _bbox_outline_crop(self, xyz_a, xyz_b):
        """Build a 1-voxel-thick rectangle outline for the box spanning two IJK corners,
        in the (crop, ((k0,k1),(j0,j1),(i0,i1))) form _add_prompt_overlay_segment expects.
        Clamped to the image; returns (None, None) if degenerate or out of bounds."""
        image = self.get_image_data()
        if image is None:
            return None, None
        k_dim, j_dim, i_dim = image.shape  # session image array is (k, j, i)
        ia, ja, ka = xyz_a  # ras_to_xyz returns (i, j, k)
        ib, jb, kb = xyz_b
        i0, i1 = max(0, min(ia, ib)), min(i_dim, max(ia, ib) + 1)
        j0, j1 = max(0, min(ja, jb)), min(j_dim, max(ja, jb) + 1)
        k0, k1 = max(0, min(ka, kb)), min(k_dim, max(ka, kb) + 1)
        if i1 - i0 <= 0 or j1 - j0 <= 0 or k1 - k0 <= 0:
            return None, None
        crop = np.zeros((k1 - k0, j1 - j0, i1 - i0), dtype=np.uint8)
        crop[:, 0, :] = 1
        crop[:, -1, :] = 1
        crop[:, :, 0] = 1
        crop[:, :, -1] = 1
        return crop, ((k0, k1), (j0, j1), (i0, i1))

    def _cancel_bbox_drag(self):
        """Drop any in-progress box drag and its preview."""
        self._bbox_drag = None
        self._bbox_preview_remove()

    def _create_display_outline_actor(self, slice_view, color):
        """Build a live outline overlay (VTK 2D actor in display coordinates) on a
        slice view -- the shared technique behind the bbox preview and the lasso
        stroke. Returns a state dict (points/pd/actor/renderer/view) or None when the
        view has no renderer."""
        renderer = self._lasso_renderer_for(slice_view)
        if renderer is None:
            return None
        points = vtk.vtkPoints()
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)
        pd.SetLines(vtk.vtkCellArray())
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetTransformCoordinate(coordinate)
        mapper.SetInputData(pd)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(2.5)
        renderer.AddActor2D(actor)
        return {
            "points": points,
            "pd": pd,
            "actor": actor,
            "renderer": renderer,
            "view": slice_view,
        }

    def _bbox_preview_create(self, slice_view, color):
        """Live rectangle outline that follows the drag (see _create_display_outline_actor)."""
        self._bbox_preview_remove()
        self._bbox_preview = self._create_display_outline_actor(slice_view, color)
        return self._bbox_preview is not None

    def _bbox_preview_update(self, disp0, disp1):
        prev = getattr(self, "_bbox_preview", None)
        if not prev:
            return
        x0, y0 = disp0
        x1, y1 = disp1
        pts = prev["points"]
        pts.Reset()
        for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            pts.InsertNextPoint(x, y, 0.0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        for i in (0, 1, 2, 3, 0):
            lines.InsertCellPoint(i)
        prev["pd"].SetLines(lines)
        prev["pd"].Modified()
        prev["view"].scheduleRender()

    def _bbox_preview_remove(self):
        prev = getattr(self, "_bbox_preview", None)
        if not prev:
            return
        try:
            prev["renderer"].RemoveActor2D(prev["actor"])
            prev["view"].scheduleRender()
        except Exception:  # noqa: BLE001
            pass
        self._bbox_preview = None

    #
    #  -- Bounding Box
    #
    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """
        Adds a bounding-box interaction to the session. The two corners are given
        in IJK (i, j, k); we reverse to (k, j, i) and build half-open intervals
        ``[[k0, k1], [j0, j1], [i0, i1]]``. One axis is size 1 (the drawn slice),
        which satisfies the 2D-bbox requirement of the public models.
        """
        p1 = outer_point_one[::-1]
        p2 = outer_point_two[::-1]
        bbox = [[int(min(a, b)), int(max(a, b)) + 1] for a, b in zip(p1, p2)]
        changed_bbox = self.session.add_bbox_interaction(
            bbox,
            include_interaction=bool(positive_click),
            run_prediction=True,
        )
        # The box markup is cleared right after it is sent (a fresh box placement is
        # started), so there is no lingering marker to remove; push a no-op to keep the
        # undo stack aligned with the session's interaction count.
        self._push_prompt_undo(lambda: None)
        self.apply_result(changed_bbox)

    #
    #  -- Lasso
    #
    def set_lasso_active(self, active):
        """
        Enable/disable freehand-lasso mode. Unlike the other prompts the lasso is not a
        markups Place interaction: the contour is captured from a left-mouse drag on the
        slice views (see handle_lasso_event) and submitted on release.
        """
        active = bool(active)
        self._lasso_active = active
        if self.ui.pbInteractionLasso.isChecked() != active:
            self.ui.pbInteractionLasso.setChecked(active)
        if active:
            self._install_lasso_filters()
            self._deactivate_capture_tools()  # lasso vs point/bbox are mutually exclusive
            # Lasso and scribble paint are mutually exclusive.
            if self.ui.pbInteractionScribble.isChecked():
                self.ui.pbInteractionScribble.click()
        elif self._lasso_drawing:
            # Toggling the tool off mid-stroke discards the in-progress contour, but a
            # completed lasso stays visible as the active prompt.
            self._cancel_lasso_stroke()
        self._update_prompt_cursor()

    def _install_lasso_filters(self):
        """
        Install event filters on every slice-view cursor target. The filters capture
        freehand-lasso drags when lasso is active and keep the custom prompt cursor
        above Slicer's markups/Paint/VTK cursor resets for all prompt tools.
        """
        if not hasattr(self, "_qt_event_filters"):
            self._qt_event_filters = []
        covered = [target for target, _ in self._qt_event_filters]
        for slice_widget in self._slice_widgets():
            for target in self._slice_view_cursor_targets(slice_widget):
                if any(target is existing for existing in covered):
                    continue
                self._prepare_cursor_target(target)
                event_filter = _LassoFreehandFilter(self, slice_widget)
                target.installEventFilter(event_filter)
                self._qt_event_filters.append((target, event_filter))
                covered.append(target)

    def handle_lasso_event(self, slice_widget, event, event_widget=None):
        """
        Mouse handler routed from the per-slice-view event filter. Returns True to
        consume the event so the drag does not pan / window-level / scroll the view.
        """
        # Point/BBox tools: capture LEFT-click events ourselves (we run in ViewTransform
        # mode, so right/middle button zoom/pan are handled natively and never reach here
        # to be consumed). Only left-button events are swallowed so they don't window/level.
        place_tool = getattr(self, "_place_tool", None)
        if place_tool == "point" and not self._lasso_active:
            etype = event.type()
            if etype == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
                self._place_point_from_event(slice_widget, event, event_widget)
                return True
            if etype == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
                return True
            if etype == qt.QEvent.MouseButtonDblClick and event.button() == qt.Qt.LeftButton:
                return True
            if etype == qt.QEvent.MouseMove and (event.buttons() & qt.Qt.LeftButton):
                return True
        elif place_tool == "bbox" and not self._lasso_active:
            # Drag-and-drop: press = first corner, drag = live rectangle, release = box.
            if self._handle_bbox_drag_event(slice_widget, event, event_widget):
                return True

        if not self._lasso_active:
            return False

        event_type = event.type()
        if event_type == qt.QEvent.MouseButtonPress:
            if event.button() == qt.Qt.LeftButton:
                self._begin_lasso_stroke(slice_widget, event, event_widget)
                return True
            if event.button() == qt.Qt.RightButton and self._lasso_drawing:
                self._cancel_lasso_stroke()
                return True
        elif event_type == qt.QEvent.MouseMove:
            if self._lasso_drawing and (event.buttons() & qt.Qt.LeftButton):
                self._add_lasso_point(slice_widget, event, event_widget=event_widget)
                return True
        elif event_type == qt.QEvent.MouseButtonRelease:
            if event.button() == qt.Qt.LeftButton and self._lasso_drawing:
                self._finish_lasso_stroke()
                return True
        elif event_type == qt.QEvent.MouseButtonDblClick:
            if event.button() == qt.Qt.LeftButton:
                return True
        return False

    def _begin_lasso_stroke(self, slice_widget, event, event_widget=None):
        """Start a freehand contour on the slice the user pressed in."""
        slice_view = slice_widget.sliceView()
        color = self._polarity_color(self.is_positive)
        self._lasso_slice_widget = slice_widget
        self._lasso_display_pts = []
        self._lasso_last_xy = None
        if not self._lasso_overlay_create(slice_view, color):
            self._lasso_drawing = False
            return
        self._lasso_drawing = True
        self._add_lasso_point(slice_widget, event, event_widget=event_widget)

    def _add_lasso_point(self, slice_widget, event, min_step=2.0, event_widget=None):
        """Append the cursor position to the live contour, throttled by pixel distance.

        Points are stored in VTK display coordinates (device px, origin bottom-left) --
        the same system the overlay actor and the slice's XYToRAS matrix both use, so
        the live outline and the rasterized result are guaranteed to agree.
        """
        slice_view = slice_widget.sliceView()
        xy = self._event_xy_in_slice_view(slice_view, event, event_widget)
        if self._lasso_last_xy is not None:
            dx = xy[0] - self._lasso_last_xy[0]
            dy = xy[1] - self._lasso_last_xy[1]
            if (dx * dx + dy * dy) < (min_step * min_step):
                return
        self._lasso_last_xy = xy
        disp = self._event_display_xy(slice_view, xy)
        self._lasso_display_pts.append(disp)
        self._lasso_overlay_add(disp)

    def _finish_lasso_stroke(self):
        """Close the contour on left-button release, then rasterize + fill + submit it."""
        self._lasso_drawing = False
        self._lasso_last_xy = None
        self._lasso_overlay_finalize()  # close the outline loop (brief preview)
        self.submit_lasso_if_present()  # paints the persistent labelmap fill, then sends

    def _cancel_lasso_stroke(self):
        """Discard the in-progress contour without submitting."""
        self._lasso_drawing = False
        self._lasso_last_xy = None
        self._lasso_display_pts = []
        self._lasso_overlay_remove()

    def _event_xy_in_slice_view(self, slice_view, event, event_widget=None):
        """Mouse event position mapped into slice-view logical pixels."""
        if event_widget is not None and event_widget is not slice_view:
            try:
                pos = event_widget.mapTo(slice_view, event.pos())
                return (pos.x(), pos.y())
            except Exception:  # noqa: BLE001
                pass
        return (event.x(), event.y())

    def _event_display_xy(self, slice_view, xy):
        """Qt mouse position -> VTK display coords (device px, origin bottom-left)."""
        try:
            dpr = slice_view.devicePixelRatioF()
        except Exception:  # noqa: BLE001
            dpr = 1.0
        # Qt: origin top-left, logical px. VTK display: origin bottom-left, device px.
        # NB: in Slicer's PythonQt binding QWidget.height is a property (int), not a
        # callable -- ``slice_view.height()`` raises "'int' object is not callable".
        x = xy[0] * dpr
        y = (slice_view.height * dpr) - (xy[1] * dpr)
        return (x, y)

    # --- freehand overlay actors -------------------------------------------------
    # The contour is drawn with native VTK 2D actors layered on the slice view rather
    # than a markups node: a markups closed curve trailed the cursor, lagged a stroke
    # behind on colour, and would not render its fill in 2D. These actors live in
    # display coordinates so the line tracks the cursor exactly while drawing.

    def _lasso_renderer_for(self, slice_view):
        try:
            return slice_view.renderWindow().GetRenderers().GetFirstRenderer()
        except Exception:  # noqa: BLE001
            return None

    def _lasso_overlay_create(self, slice_view, color):
        """Build the live outline actor for a fresh stroke (display coords, follows the
        cursor; see _create_display_outline_actor). The persistent filled prompt is its
        own overlay segment added on release (_add_prompt_overlay_segment), so the
        outline actor is transient. The outline grows one 2-point line segment per
        mouse move (_lasso_overlay_add) instead of rebuilding a polyline cell each
        time, which was O(stroke length) per move and lagged on long strokes."""
        self._lasso_overlay_remove()
        state = self._create_display_outline_actor(slice_view, color)
        if state is None:
            return False
        self._lasso_points = state["points"]
        self._lasso_outline_pd = state["pd"]
        self._lasso_outline_actor = state["actor"]
        self._lasso_renderer = state["renderer"]
        self._lasso_render_view = state["view"]
        return True

    def _lasso_overlay_add(self, disp):
        """Append a point and extend the outline by one segment (O(1) per move)."""
        if self._lasso_points is None:
            return
        self._lasso_points.InsertNextPoint(disp[0], disp[1], 0.0)
        n = self._lasso_points.GetNumberOfPoints()
        if n >= 2:
            lines = self._lasso_outline_pd.GetLines()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(n - 2)
            lines.InsertCellPoint(n - 1)
            lines.Modified()
        self._lasso_outline_pd.Modified()
        self._lasso_render_view.scheduleRender()

    def _lasso_overlay_finalize(self):
        """Close the outline loop (a brief filled-contour preview before the labelmap
        overlay is painted on release)."""
        if self._lasso_points is None or self._lasso_points.GetNumberOfPoints() < 3:
            return
        n = self._lasso_points.GetNumberOfPoints()
        lines = self._lasso_outline_pd.GetLines()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(n - 1)
        lines.InsertCellPoint(0)
        lines.Modified()
        self._lasso_outline_pd.Modified()
        self._lasso_render_view.scheduleRender()

    def _lasso_overlay_remove(self):
        """Remove the live outline actor from the slice view and reset overlay state."""
        renderer = getattr(self, "_lasso_renderer", None)
        view = getattr(self, "_lasso_render_view", None)
        actor = getattr(self, "_lasso_outline_actor", None)
        if renderer is not None and actor is not None:
            renderer.RemoveActor2D(actor)
            if view is not None:
                view.scheduleRender()
        self._lasso_points = None
        self._lasso_outline_pd = None
        self._lasso_outline_actor = None
        self._lasso_renderer = None
        self._lasso_render_view = None

    def submit_lasso_if_present(self):
        """
        Rasterize the freehand contour into a tight 2D crop and send it (with its
        interaction bbox) to the session. Display points are converted to RAS via the
        stroke slice's XYToRAS, then to IJK; skimage fills the closed polygon.
        """
        pts = self._lasso_display_pts
        slice_widget = self._lasso_slice_widget
        if not pts or len(pts) < 3 or slice_widget is None:
            self._lasso_overlay_remove()
            return
        slice_node = slice_widget.mrmlSliceNode()
        if slice_node is None:
            self._lasso_overlay_remove()
            return

        try:
            xy_to_ras = slice_node.GetXYToRAS()
            to_ijk = self._ras_to_ijk_converter()  # hoisted: one setup for the contour
            xyzs = []
            for (x, y) in pts:
                ras = xy_to_ras.MultiplyPoint([x, y, 0.0, 1.0])
                xyzs.append(to_ijk([ras[0], ras[1], ras[2]]))
            crop, bbox = self.lasso_points_to_crop(xyzs)
        except ValueError:
            # No single constant slice axis (e.g. an oblique reformat); ignore.
            self._lasso_overlay_remove()
            slicer.util.showStatusMessage(
                "Lasso must be drawn on an axial, sagittal or coronal slice.", 3000
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Any other failure must still drop the display-space outline (it does not
            # track pan/zoom, so a stranded one sticks to the view) and must be shown
            # to the user -- e.g. a missing scikit-image after a backend update used to
            # leave a stuck lasso with only a Python-console traceback.
            self._lasso_overlay_remove()
            slicer.util.errorDisplay(f"Lasso prompt failed: {exc}")
            return

        if crop is None or self.get_volume_node() is None:
            self._lasso_overlay_remove()
            return

        # Send the prompt FIRST so the model result is painted immediately (exactly as
        # the bbox path does). Only then add the filled region as its own overlay
        # segment so it persists and tracks pan/zoom/slice navigation.
        # _add_prompt_overlay_segment does a full-volume labelmap write that would
        # otherwise delay the result the user is waiting on. Drop the display-space
        # outline actor up front so the live stroke disappears the instant the user
        # releases, regardless of how long the prompt round-trip takes.
        self._lasso_overlay_remove()
        sent = self.lasso_or_scribble_prompt(
            crop=crop,
            interaction_bbox=bbox,
            positive_click=self.is_positive,
            tp="lasso",
        )
        if sent:
            # Persist the lasso as its own distinct overlay segment (so it stays visible
            # and outlined even when nested inside another prompt).
            undo = self._add_prompt_overlay_segment(crop, bbox, self.is_positive)
            # Keep the undo stack paired 1:1 with session interactions.
            self._push_prompt_undo(undo if undo is not None else (lambda: None))

    def _add_prompt_overlay_segment(self, crop, bbox, positive):
        """Render a finished lasso/scribble stroke as its OWN overlay segment.

        Each prompt becomes a distinct, outlined segment (green = positive, red =
        negative) in ``scribble_segment_node`` instead of merging into the shared fg/bg
        fill. With the editor's OverwriteNone mode the segments overlap freely, so a
        prompt drawn inside another keeps its own outline and its translucent fill
        stacks on top -- a scribble inside a lasso (etc.) stays visible. This is purely
        visualization: the model interaction was already sent by the caller.

        Returns an undo closure that removes the segment, or None if nothing was drawn.
        """
        node = getattr(self, "scribble_segment_node", None)
        volume = self.get_volume_node()
        if node is None or volume is None or crop is None:
            return None
        sub = (np.asarray(crop) > 0).astype(np.uint8)
        if int(sub.sum()) == 0:
            return None
        color = self._polarity_color(positive)
        self._prompt_overlay_counter = getattr(self, "_prompt_overlay_counter", 0) + 1
        seg_id = f"prompt_{self._prompt_overlay_counter}"
        segmentation = node.GetSegmentation()
        segmentation.AddEmptySegment(seg_id, seg_id, list(color))
        segment = segmentation.GetSegment(seg_id)
        if segment is not None:
            segment.SetColor(*color)  # explicit (the AddEmptySegment colour is ignored on some builds)
        try:
            # Region write: only the crop's sub-extent is imported, not a full-volume
            # array (the segment is fresh, so MODE_REPLACE over the bbox is exact).
            self._set_segment_region_from_crop(node, seg_id, sub, bbox)
        except Exception as exc:  # noqa: BLE001 - display only; the prompt was already sent
            debug_print(f"_add_prompt_overlay_segment: could not write labelmap: {exc}")
            try:
                segmentation.RemoveSegment(seg_id)
            except Exception:  # noqa: BLE001
                pass
            return None
        dn = node.GetDisplayNode()
        if dn is not None:
            # Translucent fill (so overlaps read as stacked) + opaque outline.
            dn.SetSegmentOpacity2DFill(seg_id, 0.4)
            dn.SetSegmentOpacity2DOutline(seg_id, 1.0)
        return self._make_segment_remove_undo(seg_id)

    def _set_segment_region_from_crop(self, segmentation_node, segment_id, crop, bbox):
        """Write a tight crop into ``segment_id`` over only its sub-extent (no
        full-volume array). ``crop`` is the (k, j, i) sub-region; ``bbox`` its half-open
        absolute extent ``[[k0,k1],[j0,j1],[i0,i1]]``. MODE_REPLACE sets this segment's
        voxels to the crop inside the bbox and leaves it background elsewhere -- correct
        for a freshly created segment. Mirrors the fast path of _update_segment_region.
        """
        from vtk.util import numpy_support

        (k0, k1), (j0, j1), (i0, i1) = bbox
        sub = np.ascontiguousarray(crop, dtype=np.uint8)
        oriented = slicer.vtkOrientedImageData()
        oriented.SetExtent(i0, i1 - 1, j0, j1 - 1, k0, k1 - 1)
        ijk_to_ras = vtk.vtkMatrix4x4()
        self.get_volume_node().GetIJKToRASMatrix(ijk_to_ras)
        oriented.SetImageToWorldMatrix(ijk_to_ras)
        oriented.GetPointData().SetScalars(
            numpy_support.numpy_to_vtk(
                sub.reshape(-1), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
        )
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
            oriented,
            segmentation_node,
            segment_id,
            slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE,
        )

    def _make_segment_remove_undo(self, seg_id):
        """Undo closure: remove a per-instance prompt overlay segment (see
        _add_prompt_overlay_segment)."""

        def _undo():
            node = getattr(self, "scribble_segment_node", None)
            if node is None:
                return
            segmentation = node.GetSegmentation()
            if segmentation is not None and segmentation.GetSegment(seg_id) is not None:
                try:
                    segmentation.RemoveSegment(seg_id)
                except Exception:  # noqa: BLE001
                    pass

        return _undo

    def _clear_scribble_scratch(self, label_name):
        """Empty the live paint scratch segment (fg/bg) after its stroke has been frozen
        into a per-instance overlay segment, and reset that label's diff baseline so the
        next stroke is detected against an empty segment.

        The scribble observer is already removed by the caller before this runs, so this
        labelmap write does not re-trigger the scribble callback.
        """
        node = getattr(self, "scribble_segment_node", None)
        if node is None:
            return
        cleared = False
        try:
            # Zero only THIS segment's own (small) allocated extent in place via the
            # zero-copy internal view -- no full-volume array. The scratch holds just the
            # stroke we already moved out, so its extent is tiny.
            segmentation = node.GetSegmentation()
            segment = segmentation.GetSegment(label_name) if segmentation is not None else None
            vimage = node.GetBinaryLabelmapInternalRepresentation(label_name)
            view = slicer.util.arrayFromSegmentInternalBinaryLabelmap(node, label_name)
            if segment is not None and view is not None and view.size:
                view[view == segment.GetLabelValue()] = 0  # leave any sibling values intact
                if vimage is not None:
                    vimage.Modified()  # numpy write bypasses VTK's MTime
                segment.Modified()
                cleared = True
        except Exception as exc:  # noqa: BLE001
            debug_print(f"_clear_scribble_scratch: in-place clear failed ({exc}); full clear.")
        if not cleared:
            # Fallback: full-volume zeros write (robust if the internal view is unavailable).
            volume = self.get_volume_node()
            image = self.get_image_data()
            if node is not None and volume is not None and image is not None:
                try:
                    slicer.util.updateSegmentBinaryLabelmapFromArray(
                        np.zeros(image.shape, dtype=np.uint8), node, label_name, volume
                    )
                except Exception as exc:  # noqa: BLE001
                    debug_print(f"_clear_scribble_scratch: could not clear '{label_name}': {exc}")
        if isinstance(getattr(self, "_prev_scribble_masks", None), dict):
            self._prev_scribble_masks[label_name] = None

    #
    #  -- Scribble
    #
    def on_scribble_clicked(self, checked=False):
        """
        Activates/deactivates the hidden Segment Editor's Paint effect on the
        scribble segment (bg or fg, depending on prompt type).
        """
        self.setup_prompts(skip_if_exists=True)

        if checked:
            # Scribble paint, freehand lasso and point/bbox are mutually exclusive.
            self.set_lasso_active(False)
            self._deactivate_capture_tools()

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName(
                    ""
                )  # Clears the active effect

            # Optionally clear or reset the segmentation node
            self._remove_scribble_labelmap_observer()

            self._update_prompt_cursor()
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
            self._remove_scribble_labelmap_observer()
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            # The brush thickness comes from the model. Make sure a session exists now so
            # the very first stroke is sized correctly -- otherwise the first scribble
            # uses a fallback relative brush until a prompt lazily creates the session.
            self.ensure_session()
            self._apply_scribble_brush_size(paint_effect)
            self._scribble_labelmap_callback_tag = {
                "node": self.scribble_segment_node,
                "tag": self.scribble_segment_node.AddObserver(
                    vtk.vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        # Apply our colour cursor last (after Paint activation) so it wins initially;
        # the coloured brush circle carries polarity even if Paint later overrides it.
        self._update_prompt_cursor()
        debug_print(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    def _retarget_scribble_if_active(self):
        """
        Re-point an in-progress scribble at the segment for the current polarity
        (fg = positive/green, bg = negative/red). Polarity can change mid-scribble;
        without this the next stroke paints into the previous polarity's segment (wrong
        colour) until on_scribble_finished re-activates it. Toggling the button off/on
        reuses the proven activation path, cleanly swapping the observer + segment.
        """
        if self.ui.pbInteractionScribble.isChecked():
            self.ui.pbInteractionScribble.click()  # off: deactivate + remove observer
            self.ui.pbInteractionScribble.click()  # on:  re-select fg/bg + re-add observer

    #
    #  -- Lasso/scribble
    #
    @ensure_synched
    def lasso_or_scribble_prompt(
        self, crop, interaction_bbox, positive_click=False, tp="lasso"
    ):
        """
        Adds a lasso or scribble interaction to the session, passing only a tight
        crop plus its ``interaction_bbox`` (orders of magnitude less data than the
        full volume; see nnInteractive's API_CHANGES_v2).

        Returns ``True`` if an interaction was actually sent (non-empty crop), else
        ``False``. The caller registers any overlay-undo / baseline bookkeeping AFTER
        this returns, so that work stays off the user-perceived latency path (the
        result is already painted by ``apply_result`` below).
        """
        if crop is None or int(np.sum(crop)) == 0:
            return False

        add_interaction = (
            self.session.add_lasso_interaction
            if tp == "lasso"
            else self.session.add_scribble_interaction
        )
        changed_bbox = add_interaction(
            crop,
            include_interaction=bool(positive_click),
            run_prediction=True,
            interaction_bbox=interaction_bbox,
        )
        self.apply_result(changed_bbox)
        return True

    def _same_mrml_node(self, a, b):
        if a is b:
            return True
        if a is None or b is None:
            return False
        try:
            return a.GetID() == b.GetID()
        except Exception:  # noqa: BLE001
            return False

    def _remove_scribble_labelmap_observer(self, expected_node=None):
        tag_info = getattr(self, "_scribble_labelmap_callback_tag", None)
        if not tag_info:
            return None
        node = tag_info.get("node") or getattr(self, "scribble_segment_node", None)
        if expected_node is not None and not self._same_mrml_node(node, expected_node):
            return None
        tag = tag_info.get("tag")
        label_name = tag_info.get("label_name")
        if node is not None and tag is not None:
            try:
                node.RemoveObserver(tag)
            except Exception:  # noqa: BLE001
                pass
        try:
            del self._scribble_labelmap_callback_tag
        except AttributeError:
            pass
        return label_name

    def _scribble_mask_or_none(self, segmentation_node, label_name, volume_node):
        if segmentation_node is None or label_name is None or volume_node is None:
            return None
        try:
            segmentation = segmentation_node.GetSegmentation()
        except Exception:  # noqa: BLE001
            return None
        if segmentation is None or segmentation.GetSegment(label_name) is None:
            debug_print(f"Ignoring scribble callback for missing segment '{label_name}'.")
            return None

        binary_name = (
            slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        )
        try:
            if not segmentation.ContainsRepresentation(binary_name):
                return None
        except Exception:  # noqa: BLE001
            pass

        try:
            return slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, label_name, volume_node
            )
        except Exception as exc:  # noqa: BLE001
            debug_print(f"Ignoring unreadable scribble labelmap for '{label_name}': {exc}")
            return None

    def _scribble_new_voxels_region(self, node, label_name, volume_node):
        """Fast path for the scribble diff.

        Reads only THIS segment's own allocated labelmap extent -- a few small slices
        for a scribble -- via the zero-copy internal-labelmap view, and diffs it against
        the per-label baseline restricted to the same extent. The old path rasterized
        the segment onto the FULL reference-volume geometry on every stroke (allocate +
        fill the whole volume), then ran a full-volume diff/argwhere/copy; on a large
        image that dominated the perceived latency of a scribble.

        Returns ``(crop, bbox)`` -- the newly painted voxels in ``(k, j, i)`` half-open
        form. Returns ``None`` if the labelmap is not ready yet or nothing new was
        painted. Raises on unexpected API issues so the caller can fall back to the
        robust full-volume path.
        """
        segmentation = node.GetSegmentation()
        if segmentation is None or segmentation.GetSegment(label_name) is None:
            debug_print(f"Ignoring scribble callback for missing segment '{label_name}'.")
            return None
        binary_name = (
            slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        )
        if not segmentation.ContainsRepresentation(binary_name):
            return None
        vimage = node.GetBinaryLabelmapInternalRepresentation(label_name)
        if vimage is None:
            return None
        # Allocated labelmap extent in IJK (i, j, k), inclusive. The scribble node shares
        # the volume's geometry (SetReferenceImageGeometryParameterFromVolumeNode), so
        # these indices address the volume array directly. A SHARED labelmap (fg + bg)
        # reports the union extent, which is why we filter by THIS segment's value below.
        ix0, ix1, iy0, iy1, iz0, iz1 = vimage.GetExtent()
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            return None  # nothing allocated yet -- early AnyEvent before the real stroke
        view = slicer.util.arrayFromSegmentInternalBinaryLabelmap(node, label_name)
        if view is None:
            return None
        label_value = segmentation.GetSegment(label_name).GetLabelValue()
        cur = view == label_value  # this segment's own voxels (labelmap may be shared)

        if not isinstance(getattr(self, "_prev_scribble_masks", None), dict):
            self._prev_scribble_masks = {}
        full_shape = self.get_image_data().shape  # (z, y, x); arrayFromVolume is a view
        base = self._prev_scribble_masks.get(label_name)
        if base is None or base.shape != full_shape:
            base = np.zeros(full_shape, dtype=np.uint8)
            self._prev_scribble_masks[label_name] = base
        base_region = base[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]  # view into the baseline
        # Newly painted on THIS label since the last stroke (boolean "now AND not before"
        # over the small extent only -- no full-volume pass).
        new = cur & (base_region == 0)

        crop, local_bbox = self._mask_to_crop(new)
        if crop is None:
            return None  # nothing newly painted
        (lk0, lk1), (lj0, lj1), (li0, li1) = local_bbox
        bbox = [
            [iz0 + lk0, iz0 + lk1],
            [iy0 + lj0, iy0 + lj1],
            [ix0 + li0, ix0 + li1],
        ]

        return crop, bbox

    def _scribble_new_voxels_full(self, node, label_name, volume_node):
        """Fallback scribble diff over the full reference-volume rasterization, used only
        if ``_scribble_new_voxels_region`` raises. Same ``(crop, bbox)`` contract.
        """
        mask = self._scribble_mask_or_none(node, label_name, volume_node)
        if mask is None:
            # Early AnyEvent before the binary labelmap is ready -- wait for the stroke.
            return None
        if not isinstance(getattr(self, "_prev_scribble_masks", None), dict):
            self._prev_scribble_masks = {}
        # Track the previous paint PER LABEL ("fg" vs "bg"). A single shared previous
        # mask let a negative (bg) scribble be compared against a positive (fg) one.
        prev = self._prev_scribble_masks.get(label_name)
        if prev is None or prev.shape != mask.shape:
            prev = np.zeros_like(mask)
        # Only the voxels newly painted on THIS label since the last stroke. Use a
        # boolean "painted now AND not before" instead of a uint8 subtraction, which
        # underflowed (0 - 1 = 255) and leaked old/other-label scribbles into the prompt.
        new = np.logical_and(mask > 0, prev == 0)
        crop, bbox = self._mask_to_crop(new)
        if crop is None:
            return None
        return crop, bbox

    def on_scribble_finished(self, caller, event):
        if getattr(self, "_scribble_callback_in_progress", False):
            return
        self._scribble_callback_in_progress = True
        try:
            self._on_scribble_finished(caller, event)
        finally:
            self._scribble_callback_in_progress = False

    def _on_scribble_finished(self, caller, event):
        """
        Called when the user completes a scribble stroke in the Paint effect.
        We calculate the diff in the drawn region and send it to the server.
        """
        debug_print("Scribble labelmap event received.")

        tag_info = getattr(self, "_scribble_labelmap_callback_tag", None)
        if not tag_info:
            return

        observed_node = tag_info.get("node")
        if observed_node is not None and not self._same_mrml_node(caller, observed_node):
            return

        label_name = tag_info.get("label_name")
        if not self._same_mrml_node(caller, getattr(self, "scribble_segment_node", None)):
            self._remove_scribble_labelmap_observer(expected_node=caller)
            return

        volume_node = self.get_volume_node()
        if not self._scribble_prompt_matches_volume(volume_node):
            debug_print("Ignoring scribble callback for stale source-volume geometry.")
            self._remove_scribble_labelmap_observer(expected_node=caller)
            return

        # Extract only the newly painted voxels of this stroke. The fast path reads just
        # this segment's allocated labelmap extent; if anything unexpected happens there,
        # fall back to the robust (slower) full-volume diff so a stroke is never dropped.
        try:
            diff = self._scribble_new_voxels_region(caller, label_name, volume_node)
        except Exception as exc:  # noqa: BLE001
            debug_print(f"Region scribble diff failed ({exc}); using full-volume diff.")
            diff = self._scribble_new_voxels_full(caller, label_name, volume_node)

        if diff is None:
            # Segment Editor emits several early AnyEvent notifications before the binary
            # labelmap is ready, and a stroke that paints nothing new yields no diff.
            # Keep the observer and wait for the real stroke.
            return
        crop, bbox = diff

        self._remove_scribble_labelmap_observer(expected_node=caller)
        sent = self.lasso_or_scribble_prompt(
            crop=crop,
            interaction_bbox=bbox,
            positive_click=(label_name == "fg"),
            tp="scribble",
        )
        if sent:
            # Freeze this stroke into its own distinct overlay segment, then empty the
            # fg/bg paint scratch (and its diff baseline) so strokes don't merge into one
            # flat fill and the next stroke is detected against an empty segment. Runs
            # after the result is shown, so it stays off the latency path.
            undo = self._add_prompt_overlay_segment(
                crop, bbox, positive=(label_name == "fg")
            )
            self._clear_scribble_scratch(label_name)
            self._push_prompt_undo(undo if undo is not None else (lambda: None))

        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def reset_all_prompts(self):
        """
        Tear down every interactive prompt before starting/clearing an object:
        remove the displayed point/bbox/lasso markups, rebuild empty scribble
        segments, drop the per-label scribble history, and clear the session's
        interactions. Without this, prompts (especially negative scribbles) from the
        previous object carried over and corrupted the next one.
        """
        if getattr(self, "scribble_editor_widget", None) is not None:
            self.scribble_editor_widget.setActiveEffectByName("")
        self._remove_scribble_labelmap_observer()
        self.setup_prompts()  # removes displayed markups + recreates empty scribble segments
        self._cancel_lasso_stroke()  # clear any persistent freehand-lasso overlay
        self._prev_scribble_masks = {}
        self._clear_prompt_undo()
        if self.session is not None:
            try:
                self.session.reset_interactions()
            except self.SESSION_LOST_ERRORS as exc:
                self.handle_session_expired(exc)
            except Exception:  # noqa: BLE001
                pass

        self._deactivate_all_tools()

        # Invalidate the segment baseline so the next prompt re-seeds the session for
        # whatever segment is then selected.
        self.previous_states.pop("segment_fp", None)

    def on_next_segment(self):
        """
        'Next segment' handler (button / 'e' shortcut): fully reset the current
        prompts and session, then create a fresh empty segment to work on. The
        active interaction tool is preserved so the user can keep working on the new
        object without re-selecting it.
        """
        # The 'e' shortcut bypasses the disabled button, so enforce mandatory init here.
        if not self._require_initialized():
            return
        active_tool = self._active_prompt_tool()

        # DO NOT collapse labelmaps here (or anywhere during editing). Segments may
        # overlap freely, which requires each to stay on its OWN labelmap layer -- a
        # single layer stores one value per voxel and physically cannot represent two
        # segments sharing a voxel. CollapseBinaryLabelmaps(forceToSingleLayer=False)
        # packs segments that don't *currently* overlap onto one shared layer; if the
        # user later grows one into another, the shared-layer write silently zeroes the
        # other in the overlap. That was the sporadic "updates overwrite existing labels"
        # bug -- it only fired when a collapse packed two non-overlapping segments and a
        # later edit made them overlap. Collapsing is a storage optimization only; do it
        # at save/export time if at all, never on the editing path.
        self.reset_all_prompts()
        result = self.make_new_segment()

        # Keep the same tool active on the new segment.
        self._rearm_tool_later(active_tool)
        return result

    def _active_prompt_tool(self):
        """Return the name of the currently active interaction tool, or None."""
        for name, button in self.all_prompt_buttons.items():
            if button.isChecked():
                return name
        return None

    def _activate_prompt_tool(self, name):
        """Re-activate an interaction tool by name (no-op if missing/disabled/active)."""
        if self._is_tearing_down():
            return
        button = self.all_prompt_buttons.get(name)
        if button is None or not button.isEnabled():
            return
        if not button.isChecked():
            button.click()

    def _rearm_tool_later(self, tool):
        """Deferred re-activation of a prompt tool after a reset rebuilt the prompt
        nodes (next event-loop turn, so the rebuild settles first). None -> no-op."""
        if tool is not None:
            qt.QTimer.singleShot(0, lambda: self._activate_prompt_tool(tool))

    def _deactivate_all_tools(self):
        """Deselect every interaction tool (buttons + point/bbox click capture) and
        restore the default cursor. After a reset a tool button could stay "checked"
        but inactive, forcing a second click to re-select it -- deselecting here makes
        re-selection (or the deferred _rearm_tool_later) work with a single click."""
        self._place_tool = None  # also drop point/bbox click-capture
        self._cancel_bbox_drag()
        for button in self.all_prompt_buttons.values():
            button.setChecked(False)
        # setChecked() does not fire the click handlers, so normalize the cursor here.
        self._update_prompt_cursor()

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

        # Create and add the new segment. Suppress the segment-switch handler: this is a
        # programmatic selection (on_next_segment already cleared prompts and will re-arm
        # the tool), so we don't want _handle_selected_segment_change to fire on top.
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self._suppress_segment_switch = True
        try:
            self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

            # Make sure the right node is selected
            self.ui.editor_widget.setSegmentationNode(segmentation_node)
            self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        finally:
            self._last_selected_segment_id = new_segment_id
            self._suppress_segment_switch = False

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment and
        resets the session's interactions for it.
        """
        # The 'r' shortcut bypasses the disabled button, so enforce mandatory init here.
        if not self._require_initialized():
            return
        # Remember the active tool so it can be re-armed after the reset (below).
        active_tool = self._active_prompt_tool()

        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()

        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        if selected_segment_id:
            debug_print(f"Clearing segment: {selected_segment_id}")
            self._clear_segment_labelmap(segmentation_node, selected_segment_id)
            self.setup_prompts()
            self._clear_prompt_undo()
            # Drop the session's interactions for this (now-empty) segment.
            if self.session is not None:
                try:
                    self.session.reset_interactions()
                except self.SESSION_LOST_ERRORS as exc:
                    self.handle_session_expired(exc)
                except Exception:  # noqa: BLE001
                    pass
            # Baseline now reflects the cleared (empty) segment, so the next prompt does
            # not needlessly re-seed.
            self.previous_states["segment_fp"] = self._segment_fingerprint()

            # Re-arm the active tool: setup_prompts() rebuilt the prompt nodes, so the
            # interaction is stale even though the button still looks selected.
            self._deactivate_all_tools()
            self._rearm_tool_later(active_tool)
        else:
            debug_print("No segment selected to clear.")

    def _clear_segment_labelmap(self, segmentation_node, segment_id):
        """Empty a segment's labelmap. Zeroes only the segment's own values inside its
        allocated internal-labelmap extent (in place, no full-volume array -- the same
        technique as _clear_scribble_scratch); falls back to a full-volume zeros write
        if the internal view is unavailable."""
        try:
            segmentation = segmentation_node.GetSegmentation()
            segment = segmentation.GetSegment(segment_id) if segmentation is not None else None
            vimage = segmentation_node.GetBinaryLabelmapInternalRepresentation(segment_id)
            view = slicer.util.arrayFromSegmentInternalBinaryLabelmap(
                segmentation_node, segment_id
            )
            if segment is not None:
                if view is None or view.size == 0:
                    return  # nothing allocated -> already empty
                view[view == segment.GetLabelValue()] = 0  # leave sibling values intact
                if vimage is not None:
                    vimage.Modified()  # numpy write bypasses VTK's MTime
                segment.Modified()
                return
        except Exception as exc:  # noqa: BLE001
            debug_print(f"In-place segment clear failed ({exc}); using full write.")
        self.show_segmentation(np.zeros(self.get_image_data().shape, dtype=np.uint8))

    def on_delete_segment(self):
        """
        'Delete segment' handler ('Del' shortcut): remove the currently selected
        segment entirely, drop its prompts/interactions, and land on a neighbouring
        segment -- or a fresh empty one if it was the last -- with the active tool
        preserved (mirrors Reset / Next segment). Unlike Reset, which only empties the
        segment's labelmap, this removes the segment from the segmentation.

        There is no dedicated button (the Segment Editor below already offers segment
        deletion); this handler backs the 'Del' keyboard shortcut only.
        """
        # 'Del' requires a live session, matching the other prompt-affecting shortcuts.
        if not self._require_initialized():
            return

        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if segmentation_node is None or not selected_segment_id:
            debug_print("No segment selected to delete.")
            return

        active_tool = self._active_prompt_tool()
        segmentation = segmentation_node.GetSegmentation()

        # Remember the deleted segment's position so we can select its neighbour
        # afterwards instead of leaving the editor on an empty selection.
        segment_ids = list(segmentation.GetSegmentIDs())
        try:
            removed_index = segment_ids.index(selected_segment_id)
        except ValueError:
            removed_index = 0

        # One programmatic operation: RemoveSegment moves the selection, which would
        # otherwise fire _handle_selected_segment_change on top of the reset we do here.
        self._suppress_segment_switch = True
        try:
            self.reset_all_prompts()  # clears markups/overlays/session interactions
            segmentation.RemoveSegment(selected_segment_id)

            remaining_ids = list(segmentation.GetSegmentIDs())
            if remaining_ids:
                # Prefer the previous segment; fall back to the new first one.
                new_index = min(max(removed_index - 1, 0), len(remaining_ids) - 1)
                new_segment_id = remaining_ids[new_index]
                self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
                self._last_selected_segment_id = new_segment_id
        finally:
            self._suppress_segment_switch = False

        # Removed the last segment: start a fresh empty one so the user can keep
        # working (make_new_segment manages its own switch-suppression + selection).
        if not remaining_ids:
            self.make_new_segment()

        # Re-arm the active tool on the newly selected / created segment.
        self._rearm_tool_later(active_tool)

    def _update_segment_region(self, mask, bbox, segmentationNode, segmentId):
        """Update only the changed sub-extent of the segment instead of rewriting the whole
        volume. ``bbox`` is the backend's clipped paste bbox -- half-open and directly
        sliceable on the target buffer, whose axes are (k, j, i) in Slicer. Three cases:

        * bbox fits inside the (possibly shared) allocated labelmap extent -> write the region
          IN PLACE via the zero-copy internal-labelmap view, preserving everything outside it;
        * the segment has none of ITS OWN voxels outside the bbox (a fresh segment's first
          prompt -- even in a scene that already has other segments -- or growth that engulfs
          its old content) -> SetBinaryLabelmapToSegment(MODE_REPLACE) the bbox region only
          (the rest of this segment is/becomes background -- nothing of ours to preserve);
        * otherwise (this segment has voxels both inside and outside the bbox) -> return False
          so the caller does the full update, which preserves the outside content.

        Emptiness is judged from THIS segment's own label value, not the labelmap's union
        extent, so siblings in a shared labelmap don't force a needless full update.

        Returns True on success; raises on unexpected API issues, which the caller catches.
        """
        (k0, k1), (j0, j1), (i0, i1) = bbox
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            return False

        segmentation = segmentationNode.GetSegmentation()
        segment = segmentation.GetSegment(segmentId)
        if segment is None:
            return False

        label_value = segment.GetLabelValue()
        vimage = segmentationNode.GetBinaryLabelmapInternalRepresentation(segmentId)
        if vimage is not None:
            # Allocated labelmap extent in IJK (i, j, k), inclusive. For a SHARED labelmap
            # this is the union over all its segments, so it tells us what is *allocated*,
            # not what belongs to THIS segment.
            ix0, ix1, iy0, iy1, iz0, iz1 = vimage.GetExtent()
            if ix1 >= ix0 and iy1 >= iy0 and iz1 >= iz0:  # non-empty allocation
                view = slicer.util.arrayFromSegmentInternalBinaryLabelmap(segmentationNode, segmentId)
                bbox_within_extent = (ix0 <= i0 and i1 - 1 <= ix1
                                      and iy0 <= j0 and j1 - 1 <= iy1
                                      and iz0 <= k0 and k1 - 1 <= iz1)
                if bbox_within_extent:
                    # In-place write. The internal labelmap may be SHARED by several segments
                    # (each stored as its own integer value): write this segment's own value
                    # where the prediction is set; where it is not, clear only THIS segment's
                    # voxels and leave any other segment's values untouched -- preserving the
                    # rest without the disruptive SeparateSegmentLabelmap (which churned the ID).
                    ks = slice(k0 - iz0, k1 - iz0)
                    js = slice(j0 - iy0, j1 - iy0)
                    i_s = slice(i0 - ix0, i1 - ix0)
                    sub_mask = mask[k0:k1, j0:j1, i0:i1]
                    current = view[ks, js, i_s]
                    view[ks, js, i_s] = np.where(
                        sub_mask != 0, label_value, np.where(current == label_value, 0, current)
                    )
                    # A numpy write bypasses VTK's modified time, so bump image + segment.
                    vimage.Modified()
                    segment.Modified()
                    return True

                # The change extends beyond the allocated labelmap, so we can't write in place
                # (a numpy view can't grow the VTK allocation). The region replace below can
                # grow it, but it zeroes THIS segment outside the bbox -- safe only if the
                # segment has no content of its own out there. The labelmap may be SHARED, so
                # inspect THIS segment's OWN voxels, not the union extent (other segments
                # inflate it and would otherwise force a needless full update -- the common
                # case being a fresh segment in a scene that already has others).
                own = view == label_value
                if own.any():
                    kk, jj, ii = np.nonzero(own)
                    own_within_bbox = (
                        i0 <= ix0 + int(ii.min()) and ix0 + int(ii.max()) <= i1 - 1
                        and j0 <= iy0 + int(jj.min()) and iy0 + int(jj.max()) <= j1 - 1
                        and k0 <= iz0 + int(kk.min()) and iz0 + int(kk.max()) <= k1 - 1
                    )
                    if not own_within_bbox:
                        # This segment has voxels outside the bbox -> full update preserves them.
                        return False
                # else: this segment is empty here (the allocation holds only sibling
                # segments), so a region replace drops nothing of ours -> fall through.

        # This segment has nothing of its own outside the bbox (a fresh segment -- possibly
        # in a scene that already holds others -- or growth that engulfs its old content), so
        # SET just the bbox region (the rest of THIS segment stays background). This is the
        # fast path for the first prompt on a new segment, with no full-volume write.
        # SetBinaryLabelmapToSegment maps the foreground to this segment's value and leaves
        # other segments alone; the crop's own extent bounds the replace (zeroing this
        # segment outside it, which is what we want here).
        self._set_segment_region_from_crop(
            segmentationNode, segmentId, mask[k0:k1, j0:j1, i0:i1], bbox
        )
        return True

    def show_segmentation(self, segmentation_mask, changed_bbox=None):
        """
        Updates the currently selected segment with the given binary mask array.

        ``changed_bbox`` (the backend's clipped paste bbox, directly sliceable on the
        target buffer in (k, j, i) order) lets us replace only the changed sub-extent
        instead of rewriting the whole labelmap -- the full path scans + copies + imports
        the entire volume on every prompt, which dominates latency on large images. Falls
        back to the full update when no bbox is given (undo / seeding) or on any error.
        """
        t0 = time.time()
        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())

        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            # NOTE: we deliberately do NOT call saveStateForUndo() here. It snapshots
            # the whole labelmap on every prediction (slow), and undo is handled by the
            # nnInteractive session (Ctrl+Z -> session.undo()), not the segment editor.
            updated_region = False
            t_update = time.time()
            if changed_bbox is not None:
                try:
                    updated_region = self._update_segment_region(
                        segmentation_mask, changed_bbox, segmentationNode, selectedSegmentID
                    )
                except Exception as exc:  # noqa: BLE001 - any failure -> safe full update
                    debug_print(f"region-limited update failed; full update instead ({exc})")
                    updated_region = False
            if not updated_region:
                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    segmentation_mask,
                    segmentationNode,
                    selectedSegmentID,
                    self.get_volume_node(),
                )
            debug_print(
                f"apply path: changed_bbox={'set' if changed_bbox is not None else 'None'}"
                f" region_applied={updated_region} labelmap_write={time.time() - t_update:.3f}s"
            )
            if was_3d_shown:
                segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segment as being edited (can be useful for selective saving of only modified segments)
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        # NOTE: CollapseBinaryLabelmaps() is NOT called here -- nor anywhere on the
        # editing path (see on_next_segment). Besides being O(volume x segments), it
        # packs non-overlapping segments onto a shared layer, which breaks free overlap:
        # a later edit that grows one segment into another then zeroes the other. Keep
        # every segment on its own layer while editing.
        del segmentation_mask

        # Record the post-write fingerprint so selected_segment_changed() sees "no change"
        # next prompt (it only re-seeds on an *external* edit), without re-extracting and
        # comparing the whole labelmap.
        self.previous_states["segment_fp"] = self._segment_fingerprint()

        debug_print(f"show_segmentation took {time.time() - t0}")

    def toggle_segment_visibility(self):
        """
        'V' shortcut: toggle the visibility of the segment currently being worked on.

        A pure display toggle -- it deliberately does NOT require an initialized
        session and never creates a segment. We therefore read the segmentation node
        straight from the editor widget rather than via get_segmentation_node(), which
        would add an empty node when none exists.
        """
        segmentation_node = self.ui.editor_widget.segmentationNode()
        selected_segment_id = self.get_current_segment_id()
        if segmentation_node is None or not selected_segment_id:
            debug_print("No segment selected to toggle visibility.")
            return

        display_node = segmentation_node.GetDisplayNode()
        if display_node is None:
            debug_print("Segmentation has no display node; cannot toggle visibility.")
            return

        new_visibility = not display_node.GetSegmentVisibility(selected_segment_id)
        display_node.SetSegmentVisibility(selected_segment_id, new_visibility)
        debug_print(
            f"Toggled visibility of segment {selected_segment_id} to {new_visibility}"
        )

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one.
        """
        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            if segmentation_node.GetName() != self.scribble_segment_node_name:
                return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            if segmentation_node.GetName() == self.scribble_segment_node_name:
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
        Gets the labelmap array (binary uint8, 0/1) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, self.get_volume_node()
        )
        return (mask != 0).astype(np.uint8)

    def _segment_fingerprint(self):
        """Cheap O(1) identity for the selected segment's labelmap:
        ``(node id, segment id, binary-labelmap MTime)``. The MTime bumps when we write a
        prediction result or the user edits the segment, so this detects "the segment
        changed" without extracting and comparing the whole labelmap every prompt
        (mirrors _image_fingerprint for the volume)."""
        try:
            node, segment_id = self.get_selected_segmentation_node_and_segment_id()
        except Exception:  # noqa: BLE001
            return None
        if node is None or not segment_id:
            return None
        segment = node.GetSegmentation().GetSegment(segment_id)
        if segment is None:
            return None
        bin_name = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        rep = segment.GetRepresentation(bin_name)
        rep_mtime = int(rep.GetMTime()) if rep is not None else 0
        return (node.GetID(), segment_id, rep_mtime)

    def _selected_segment_is_empty(self):
        """Cheaply decide whether the selected segment has any foreground voxels, WITHOUT the
        full-volume resample+extract that get_segment_data() does (~0.1s on large images).
        Reads the zero-copy internal-labelmap view and tests for this segment's own label
        value. Returns True only when the segment is *definitely* empty (no labelmap, empty
        extent, or no matching voxels); returns False when unsure, so the caller falls back
        to the safe full extract."""
        try:
            node, segment_id = self.get_selected_segmentation_node_and_segment_id()
        except Exception:  # noqa: BLE001
            return False
        if node is None or not segment_id:
            return False
        segment = node.GetSegmentation().GetSegment(segment_id)
        if segment is None:
            return False
        vimage = node.GetBinaryLabelmapInternalRepresentation(segment_id)
        if vimage is None:
            return True  # no labelmap representation at all -> empty
        ix0, ix1, iy0, iy1, iz0, iz1 = vimage.GetExtent()
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            return True  # degenerate/empty extent -> empty
        view = slicer.util.arrayFromSegmentInternalBinaryLabelmap(node, segment_id)
        if view is None or view.size == 0:
            return True
        # The labelmap may be shared by several segments; only THIS segment's value counts.
        return not bool((view == segment.GetLabelValue()).any())

    def selected_segment_changed(self):
        """
        Checks (cheaply, via _segment_fingerprint) whether the selected segment changed
        since we last synced it -- i.e. an external edit -- so the session is re-seeded.
        """
        fingerprint = self._segment_fingerprint()
        old_fingerprint = self.previous_states.get("segment_fp", None)
        selected_segment_changed = old_fingerprint is None or old_fingerprint != fingerprint
        debug_print(f"segment fingerprint: {fingerprint} (was {old_fingerprint})")
        debug_print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    ###############################################################################
    # Session management (local in-process or remote server)
    ###############################################################################

    def get_mode(self):
        """Returns the configured compute mode: 'local' or 'remote'."""
        mode = slicer.util.settingsValue("SlicerNNInteractive/mode", "remote")
        return mode if mode in ("local", "remote") else "remote"

    def set_mode(self, mode):
        """Persists the compute mode.

        Switching the Local/Remote toggle does NOT tear down the current session -- a
        round-trip keeps it. A session built for the other mode is rebuilt lazily by
        ensure_session() when it is actually needed.
        """
        qt.QSettings().setValue("SlicerNNInteractive/mode", mode)

    def get_install_flavor(self):
        """Returns the recorded install flavor: 'full', 'client', or '' (none yet)."""
        flavor = slicer.util.settingsValue("SlicerNNInteractive/install_flavor", "")
        return flavor if flavor in ("full", "client") else ""

    def set_install_flavor(self, flavor):
        """Persists the install flavor ('full' or 'client')."""
        qt.QSettings().setValue("SlicerNNInteractive/install_flavor", flavor)

    def _detect_install_flavor(self):
        """Probe the Python environment for which backend is actually importable.

        'full' (in-process local inference) implies the client too, so it wins. Returns
        '' when neither the full backend nor the lightweight client is present.
        """
        if self._local_inference_available():
            return "full"
        try:
            if importlib.util.find_spec("nnInteractive.inference.remote") is not None:
                return "client"
        except ImportError:
            pass
        return ""

    def _backend_installed_for_mode(self, mode):
        """True if the backend needed for ``mode`` is importable. Local needs the full
        in-process backend; remote works with either the client or the full package."""
        if mode == "local":
            return self._local_inference_available()
        return self._detect_install_flavor() != ""

    def get_setting_bool(self, key):
        return slicer.util.settingsValue(
            f"SlicerNNInteractive/{key}", SETTING_DEFAULTS[key], converter=slicer.util.toBool
        )

    def get_setting_str(self, key):
        return slicer.util.settingsValue(f"SlicerNNInteractive/{key}", SETTING_DEFAULTS[key])

    def _save_setting(self, key, value, reinit=False):
        qt.QSettings().setValue(f"SlicerNNInteractive/{key}", value)
        if reinit:
            self._teardown_for_settings_change()

    def _save_setting_bool(self, key, value, reinit=False):
        qt.QSettings().setValue(f"SlicerNNInteractive/{key}", bool(value))
        if reinit:
            self._teardown_for_settings_change()

    def _teardown_for_settings_change(self):
        """A session-affecting setting changed: drop the current session and reset the
        UI so the user explicitly re-initializes with the new settings applied. The
        next session is rebuilt from scratch, so it always reads the updated settings."""
        had_session = self.session is not None
        self.release_session()
        self.update_connect_status(connected=False)
        if had_session:
            slicer.util.showStatusMessage(
                "Settings changed — click Initialize (top of the Prompts tab) to apply.",
                5000,
            )

    def update_server(self):
        """Reads the server URL from the UI and persists it. Changing it disconnects the
        current remote session so the next Connect uses the new address."""
        new_server = self.ui.serverUrlEdit.text.rstrip("/")
        changed = new_server != self.server
        self.server = new_server
        qt.QSettings().setValue("SlicerNNInteractive/server", self.server)
        debug_print(f"Server URL updated and saved: {self.server}")
        if changed:
            # Drop the existing session so the new address actually takes effect.
            self._teardown_for_settings_change()

    def get_local_device(self):
        return self.get_setting_str("device") or "cuda:0"

    def get_image_spacing(self):
        """Image spacing in array (k, j, i) order to match arrayFromVolume()."""
        volume_node = self.get_volume_node()
        if volume_node is None:
            return [1.0, 1.0, 1.0]
        # GetSpacing() is in (i, j, k) order; the numpy array is (k, j, i).
        return [float(s) for s in reversed(volume_node.GetSpacing())]

    def ensure_session(self):
        """
        Make sure ``self.session`` is a live nnInteractive session for the current
        mode. Returns True on success, False otherwise (an error is shown to the
        user on failure).
        """
        # Reuse the existing session only if it was built for the current mode. The
        # Local/Remote toggle does not tear sessions down, so a session for the other
        # mode must be released and rebuilt before it is used here.
        mode = self.get_mode()
        if self.session is not None:
            if getattr(self, "_session_mode", None) == mode:
                return True
            self.release_session()

        # No lazy install: if the backend for the current mode isn't installed, send the
        # user to the Configuration tab instead of silently pip-installing.
        if not self._backend_installed_for_mode(mode):
            need = "Full (local + remote)" if mode == "local" else "the client (or Full)"
            slicer.util.errorDisplay(
                f"nnInteractive is not installed for {mode} mode.\n\n"
                f"Open the Configuration tab and click "
                f"'Reinstall / Update nnInteractive' to install {need}, "
                f"or restart Slicer to get the install prompt.",
                parent=slicer.util.mainWindow(),
            )
            return False

        # Recomputed by _construct_local_session; a remote session never runs on our CPU.
        # (The early-return reuse path above keeps the previous value untouched.)
        self._local_running_on_cpu = False
        try:
            if mode == "local":
                self.session = self._construct_local_session()
            else:
                self.session = self._construct_remote_session()
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
            return False
        except Exception as exc:  # noqa: BLE001 - surface any setup failure
            self.session = None
            slicer.util.errorDisplay(
                f"Could not start nnInteractive ({mode} mode):\n\n{exc}",
                parent=slicer.util.mainWindow(),
            )
            return False

        self._session_mode = mode
        # A fresh session has no image/segment yet; force a re-sync on next prompt.
        self.previous_states.pop("image_data", None)
        self.previous_states.pop("segment_fp", None)
        self.target_buffer = None
        self._on_session_ready()
        return True

    def _construct_local_session(self):
        """Build an in-process nnInteractiveInferenceSession (local compute).

        The slow, torch-heavy work -- importing torch, initializing the device,
        constructing the session, loading the weights and the warmup forward pass --
        runs ON THE MAIN THREAD behind a busy dialog (painted once, then frozen while
        torch works). Main thread ON PURPOSE: prompts run on the main thread and
        PyTorch's cuDNN benchmark/plan cache is THREAD-LOCAL, so a worker-thread
        warmup leaves the first main-thread prediction ~1.1 s slower while cuDNN
        re-benchmarks every conv (measured on an RTX 4090: 1.28 s vs 0.14 s steady
        state). Loading here keeps every cache on the thread that predicts. The wait
        is bounded: ~4.6 s with a warm torch.compile cache, ~15 s on the first-ever
        run (cold inductor cache), less with compile off.

        The weights download runs first (its own progress dialog). Dependencies are
        NOT installed here -- ensure_session() has already verified the full backend
        is installed (installs happen only from the explicit install popup / the
        Configuration tab's Reinstall button).
        """
        checkpoint_path = self.get_checkpoint_path()  # downloads weights if needed

        custom = self.get_setting_str("checkpoint_path").strip()
        if custom:
            model_desc = f"custom checkpoint at {checkpoint_path}"
        else:
            model_id = self.get_setting_str("model_id").strip() or "default"
            model_desc = f"'{model_id}' ({checkpoint_path})"
        print(f"[nnInteractive] Local session using model: {model_desc}")

        # Settings read on the main thread (QSettings access).
        requested_device = self.get_local_device()
        want_autozoom = self.get_setting_bool("autozoom")
        want_compile = self.get_setting_bool("use_torch_compile")
        compile_reason = torch_compile_unsupported_reason()
        storage = self.get_setting_str("interactions_storage")

        use_compile = want_compile
        no_compile_reason = None
        if use_compile and compile_reason is not None:
            # torch.compile can't run here (Windows, or no Python.h to build its
            # runtime helpers); it would fail on the first prediction. Fall back
            # to eager execution and report why below.
            use_compile = False
            no_compile_reason = compile_reason
            print(f"[nnInteractive] {compile_reason} Running without torch.compile.")

        def _build_session(with_compile):
            import torch
            from nnInteractive.inference.inference_session import (
                nnInteractiveInferenceSession,
            )

            if torch.cuda.is_available():
                device = torch.device(requested_device)
                on_cpu = False
            else:
                print("[nnInteractive] No CUDA GPU detected - running on CPU (slow).")
                device = torch.device("cpu")
                on_cpu = True
                # Authoritative GPU gate: torch.compile here targets CUDA, so on CPU it
                # is pointless (and slow to build). Never compile, regardless of the
                # stored setting -- this also catches an NVIDIA driver present but a
                # CPU-only torch wheel, which the config-tab probe can't see.
                with_compile = False

            session = nnInteractiveInferenceSession(
                device=device,
                use_torch_compile=with_compile,
                torch_n_threads=os.cpu_count(),
                verbose=False,
                # Honoured regardless of device: auto-zoom on CPU is slower but the
                # user's choice is respected there too.
                do_autozoom=want_autozoom,
                interactions_storage=storage,
            )
            # Load the weights. initialize_from_trained_model_folder() runs the warmup
            # forward pass internally (torch.compile compilation + cuDNN autotuning);
            # because this is the main thread, the thread-local cuDNN plan cache is
            # warmed for the very thread prompts run on (see the docstring above).
            # Do NOT call warmup() again -- it would run twice.
            session.initialize_from_trained_model_folder(
                checkpoint_path, 0, "checkpoint_final.pth"
            )
            return session, on_cpu

        session = None
        on_cpu = False
        load_error = None
        with self._busy_dialog(
            "Initializing nnInteractive: loading the model and warming it up.\n"
            "This takes a few seconds -- or a few minutes on first use."
        ):
            try:
                session, on_cpu = _build_session(use_compile)
            except ImportError as exc:
                load_error = exc
            except Exception as exc:  # noqa: BLE001
                if use_compile:
                    # torch_compile_unsupported_reason() only proves the headers
                    # exist, not that the inductor/triton toolchain actually works.
                    # Never leave the user blocked by a broken compile stack: retry
                    # once in eager mode (also reloads the weights; acceptable for
                    # this failure path).
                    print(
                        f"[nnInteractive] torch.compile initialization failed ({exc}); "
                        "retrying without torch.compile."
                    )
                    no_compile_reason = "torch.compile failed at initialization."
                    try:
                        session, on_cpu = _build_session(False)
                    except Exception as exc2:  # noqa: BLE001
                        load_error = exc2
                else:
                    load_error = exc

        # No lazy repair: if the import failed, surface a clear message rather than
        # silently reinstalling. ALWAYS include the underlying error -- an ImportError
        # here is often NOT a broken nnInteractive install (find_spec already confirmed
        # the package is on disk) but a torch/torchvision/numpy the user changed
        # themselves that now fails to import. Hiding the reason sends them uselessly to
        # Reinstall, which does not touch torch and cannot fix it.
        if isinstance(load_error, ImportError):
            raise RuntimeError(
                "The local nnInteractive backend could not be imported. The underlying "
                "error was:\n\n"
                f"    {type(load_error).__name__}: {load_error}\n\n"
                "If this mentions torch, torchvision or numpy, a Python package in "
                "Slicer's environment (often one you installed manually) is broken or "
                "mismatched -- fix or uninstall that package. Note that 'Reinstall / "
                "Update nnInteractive' will NOT change torch, so it cannot fix this.\n\n"
                "Otherwise the nnInteractive install may be incomplete: open the "
                "Configuration tab, click 'Reinstall / Update nnInteractive' (choose "
                "Full), then restart Slicer."
            ) from load_error
        if load_error is not None:
            raise load_error

        # The CPU fallback also drives a persistent red warning below Initialize (the
        # transient status message alone is easy to miss); see _update_device_warning().
        self._local_running_on_cpu = on_cpu
        if on_cpu:
            slicer.util.showStatusMessage(
                "No CUDA GPU detected - running nnInteractive on CPU (slow).", 5000
            )
        elif no_compile_reason:
            slicer.util.showStatusMessage(
                f"{no_compile_reason} Running without torch.compile.", 6000
            )
        slicer.util.showStatusMessage(f"nnInteractive model: {model_desc}", 5000)
        return session

    def _construct_remote_session(self):
        """Connect to an official nninteractive-server (remote compute).

        The client must already be installed (ensure_session verifies this); we never
        install here.
        """
        try:
            from nnInteractive.inference.remote.remote_session import (
                ServerAtCapacityError,
                SessionExpiredError,
                nnInteractiveRemoteInferenceSession,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The nnInteractive client is not installed. Open the Configuration tab "
                "and click 'Reinstall / Update nnInteractive'."
            ) from exc

        RemoteSession = nnInteractiveRemoteInferenceSession

        # Remember which exceptions mean "the lease is gone" so callers can reconnect.
        self.SESSION_LOST_ERRORS = (SessionExpiredError,)

        self.update_server()
        if not self.server:
            raise RuntimeError(
                "No server URL set. Enter it in the 'Configuration' tab, then click "
                "Initialize at the top of the 'nnInteractive Prompts' tab."
            )

        api_key = self.api_key or None
        try:
            session = RemoteSession(server_url=self.server, api_key=api_key)
        except ServerAtCapacityError as exc:
            raise RuntimeError(f"The nnInteractive server is at capacity: {exc}") from exc

        # The session starts from the server's auto-zoom default; _on_session_ready()
        # then pushes the user's Auto-zoom choice via set_do_autozoom().
        return session

    def _populate_model_combo(self):
        """Fill the model dropdown from the backend manifest; no-op on failure.

        Loading the manifest can touch the network and needs huggingface_hub, so this
        is best-effort: any failure (offline without a cache, backend not installed)
        just leaves the dropdown empty and we fall back to the default model / a custom
        checkpoint path.
        """
        combo = getattr(self.ui, "modelComboBox", None)
        if combo is None:
            return
        # The model dropdown is a local-inference concern (it lives in the Local
        # container, hidden in remote mode; a remote server owns its own loaded model).
        # Skip in remote mode — otherwise we needlessly import model_management, which a
        # lightweight 'nninteractive-client' install does not ship (full package only).
        if self.get_mode() != "local":
            return
        try:
            from nnInteractive.model_management import get_default_model_id, list_models

            models = list_models()
        except Exception as exc:  # noqa: BLE001 - never block the UI on model discovery
            # Printed unconditionally (not debug_print) so an empty dropdown is never
            # silent.
            print(f"[nnInteractive] Could not load model list: {exc!r}")
            slicer.util.showStatusMessage(
                "nnInteractive: could not load the model list (see Python Console).", 5000
            )
            if isinstance(exc, ModuleNotFoundError):
                # Not a transient failure (offline etc.) but a missing/outdated
                # backend -- e.g. an nnInteractive that predates model_management, or
                # a recorded Full install whose packages are gone. The fix is an
                # install/update, so offer it instead of leaving only a console note.
                self._offer_backend_update()
            return
        if not models:
            return

        saved_id = self.get_setting_str("model_id")
        try:
            default_id = get_default_model_id()
        except Exception:  # noqa: BLE001
            default_id = models[0]["id"]
        selected_id = saved_id or default_id

        combo.blockSignals(True)
        combo.clear()
        self._model_ids = []
        select_index = 0
        for i, model in enumerate(models):
            label = model.get("display_name", model["id"])
            if model.get("downloaded"):
                label += "  (downloaded)"
            combo.addItem(label)
            self._model_ids.append(model["id"])
            if model["id"] == selected_id:
                select_index = i
        combo.setCurrentIndex(select_index)
        combo.blockSignals(False)

    def _ensure_model_combo_populated(self):
        """Fill the model dropdown from the already-installed backend (never installs).

        A Full install brings huggingface_hub + nnInteractive.model_management up front,
        so the dropdown fills here. With nothing installed (or only the client) it is
        left empty -- Local mode is disabled in that case anyway.
        """
        combo = getattr(self.ui, "modelComboBox", None)
        # Note: in Slicer's PythonQt binding, QComboBox.count is a property, not a method.
        if combo is None or combo.count > 0:
            return
        self._populate_model_combo()

    def _on_model_combo_changed(self, index):
        """Persist the chosen model id and invalidate the session so it reloads."""
        if 0 <= index < len(self._model_ids):
            self._save_setting("model_id", self._model_ids[index], reinit=True)

    def get_checkpoint_path(self):
        """Local model folder; downloads the selected official model on first use."""
        custom = self.get_setting_str("checkpoint_path").strip()
        if custom:
            return custom

        from nnInteractive.model_management import ensure_model_available, get_default_model_id

        model_id = self.get_setting_str("model_id").strip() or None
        self._resolved_model_dir = None
        self._model_prep_error = None

        def _download(event):
            try:
                mid = model_id or get_default_model_id()
                self._resolved_model_dir = str(ensure_model_available(mid))
            except Exception as exc:  # surfaced on the main thread below
                self._model_prep_error = exc
            finally:
                event.set()

        self.run_with_progress_bar(
            _download, (), "Preparing nnInteractive model weights (downloads on first use)..."
        )
        if self._model_prep_error is not None:
            raise RuntimeError(str(self._model_prep_error))
        if not self._resolved_model_dir:
            raise RuntimeError("Failed to prepare the nnInteractive model weights.")
        return self._resolved_model_dir

    def _interaction_widgets(self):
        """Controls that require a live session (disabled until Initialize)."""
        widgets = [
            self.ui.pbPromptTypePositive,
            self.ui.pbPromptTypeNegative,
            self.ui.pbResetSegment,
            self.ui.pbNextSegment,
        ]
        if getattr(self, "all_prompt_buttons", None):
            widgets.extend(self.all_prompt_buttons.values())
        return widgets

    def _set_interaction_ui_enabled(self, enabled):
        """Enable/disable everything that needs a live nnInteractive session.

        Initialization is mandatory before prompting: while no session is live, the
        prompt-type toggle, Reset/Next-segment and all interaction tools are disabled.
        When enabling, the individual tool buttons are then further gated by the
        model's supported_interactions (see _on_session_ready). When disabling, any
        active tool is deactivated first so a now-disabled button can't stay 'armed'.
        """
        if not getattr(self, "all_prompt_buttons", None):
            return
        if not enabled:
            self._deactivate_all_tools()
        for widget in self._interaction_widgets():
            widget.setEnabled(enabled)

    def _require_initialized(self):
        """Guard for prompt-affecting actions reachable by keyboard shortcut (which
        bypass the disabled buttons). Returns True if a session is live; otherwise it
        nudges the user to Initialize and returns False."""
        if self.session is not None:
            return True
        slicer.util.showStatusMessage(
            "Click Initialize (top of the Prompts tab) before using nnInteractive.", 4000
        )
        return False

    def _deactivate_all_tools(self):
        """Turn off any active interaction tool (used when uninitializing)."""
        self._place_tool = None
        if self._is_tearing_down():
            # During teardown just drop transient state; don't poke views/cursors.
            return
        self._cancel_bbox_drag()
        self.set_lasso_active(False)
        if (
            self.ui.pbInteractionScribble.isChecked()
            and getattr(self, "scribble_editor_widget", None) is not None
        ):
            self.scribble_editor_widget.setActiveEffectByName("")
        for button in self.all_prompt_buttons.values():
            button.setChecked(False)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)
        self._update_prompt_cursor()

    def _on_session_ready(self):
        """Reflect a freshly created session in the UI (license, capability gating)."""
        session = self.session
        # Shown as part of the status line (see update_connect_status).
        self._model_license_text = self._license_display_text(
            getattr(session, "license", None)
        )

        # Initialized: unlock the interaction UI, then gate the individual tools by
        # what this particular model actually supports (point/bbox/lasso/scribble).
        self._set_interaction_ui_enabled(True)

        supported = getattr(session, "supported_interactions", {}) or {}

        def _supports(*keys):
            if not supported:
                return True
            return any(supported.get(k, False) for k in keys)

        self.ui.pbInteractionPoint.setEnabled(_supports("points"))
        self.ui.pbInteractionBBox.setEnabled(_supports("bbox2d", "bbox3d"))
        self.ui.pbInteractionLasso.setEnabled(_supports("lasso"))
        self.ui.pbInteractionScribble.setEnabled(_supports("scribble"))
        self.update_connect_status(connected=True)

        # Apply the user's auto-zoom choice to the fresh session. The local session was
        # already built with it, but the remote session starts from the server default,
        # so this is what makes the setting take effect for remote sessions after init.
        self._apply_autozoom_to_session()

        # For remote sessions, keep the lease alive with a main-thread Qt timer.
        # The client also has a background daemon heartbeat, but a Qt timer driven by
        # Slicer's event loop is more reliable than a Python daemon thread here.
        if hasattr(session, "heartbeat"):
            self._start_heartbeat_timer()
        else:
            self._stop_heartbeat_timer()

    def _start_heartbeat_timer(self):
        """(Re)start the main-thread heartbeat timer for a remote session."""
        if self._is_tearing_down():
            return
        if getattr(self, "_heartbeat_timer", None) is None:
            self._heartbeat_timer = qt.QTimer()
            self._heartbeat_timer.setSingleShot(False)
            self._heartbeat_timer.timeout.connect(self._send_heartbeat)
        # Server liveness defaults to 60s; beat well inside it (every 30s by default).
        liveness = float(getattr(self.session, "liveness_timeout_seconds", 60.0) or 60.0)
        interval_s = max(5.0, min(30.0, liveness / 2.0))
        self._heartbeat_timer.start(int(interval_s * 1000))

    def _stop_heartbeat_timer(self):
        timer = getattr(self, "_heartbeat_timer", None)
        if timer is None:
            return
        try:
            timer.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            timer.timeout.disconnect(self._send_heartbeat)
        except Exception:  # noqa: BLE001
            pass
        self._heartbeat_timer = None

    def _send_heartbeat(self):
        """Prove liveness to the server. Runs on the main thread via the Qt timer."""
        session = self.session
        if session is None or not hasattr(session, "heartbeat"):
            self._stop_heartbeat_timer()
            return
        try:
            session.heartbeat()
        except self.SESSION_LOST_ERRORS:
            # Lease gone; stop beating and reflect the status (no modal here, since
            # this fires from a timer in the background).
            self._stop_heartbeat_timer()
            self.release_session()
            self.update_connect_status(connected=False)
        except Exception:  # noqa: BLE001
            # Transient network blip; the next beat retries (server tolerates a few
            # missed beats within the liveness window).
            pass

    def _build_image_payload(self):
        """Main-thread MRML reads for an image upload: ``(image_4d, spacing, buffer)``
        -- the [C, X, Y, Z] array nnInteractive expects, its spacing, and a fresh uint8
        target buffer. Returns None when no volume is loaded."""
        image = self.get_image_data()
        if image is None:
            return None
        return (
            np.ascontiguousarray(image[None]),
            self.get_image_spacing(),
            np.zeros(image.shape, dtype=np.uint8),
        )

    def _build_segment_seed(self):
        """The selected segment's mask for session seeding, or None when seeding is
        unsupported or pointless. A fresh segment (the common case on the first prompt
        of a new object) is empty, so detect that cheaply and skip the expensive
        full-volume extract get_segment_data() would otherwise do. Reads MRML -- main
        thread only."""
        if getattr(self.session, "supports_initial_label", True) and not self._selected_segment_is_empty():
            seg = self.get_segment_data()
            if seg.sum() > 0:
                return seg
        return None

    def _run_session_upload(self, upload_fn):
        """Run session-only upload work (no Qt/MRML access) on a worker thread behind
        the modal wait dialog, pumping the event loop so the UI stays painted instead
        of freezing for the (possibly long, especially remote) blocking call. Worker
        errors are re-raised here on the calling (main) thread."""
        error = [None]

        def _worker(done_event):
            try:
                upload_fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                error[0] = exc
            finally:
                done_event.set()

        message = (
            "Sending image to the nnInteractive server ..."
            if self.get_mode() == "remote"
            else "Preparing image for nnInteractive ..."
        )
        self._run_thread_with_message(_worker, (), message)
        if error[0] is not None:
            raise error[0]

    def sync_image_to_session(self):
        """Push the current volume into the session and allocate the target buffer.
        The blocking upload runs on a worker thread behind the wait dialog (it fires
        mid-prompt, so a synchronous call would freeze the UI for its duration)."""
        payload = self._build_image_payload()
        if payload is None:
            debug_print("No image data available to sync.")
            return False
        image_4d, spacing, buffer = payload

        def _upload():
            self.session.set_image(image_4d, {"spacing": spacing})
            self.session.set_target_buffer(buffer)

        self._run_session_upload(_upload)
        self.target_buffer = buffer
        # Re-seeding of the active segment is handled by sync_segment_to_session().
        self.previous_states.pop("segment_fp", None)
        return True

    def sync_segment_to_session(self):
        """Seed the session with the currently selected segment (for editing)."""
        self.session.reset_interactions()
        self._clear_prompt_undo()  # interactions reset -> drop their pending undo markers
        seed = self._build_segment_seed()
        if seed is not None:
            self.session.add_initial_seg_interaction(seed, run_prediction=False)
        self.previous_states["segment_fp"] = self._segment_fingerprint()

    def apply_result(self, changed_bbox=None):
        """Render the session's target buffer into the active segment.

        ``changed_bbox`` is the prediction's clipped paste bbox (from the backend); when
        present we update only that sub-extent instead of rewriting the whole volume.
        """
        if self.target_buffer is not None:
            # No copy needed: updateSegmentBinaryLabelmapFromArray copies the data into
            # VTK, and we no longer keep a reference to the array as a baseline (we use a
            # fingerprint instead), so the session can safely reuse target_buffer.
            self.show_segmentation(self.target_buffer, changed_bbox)

    def on_undo(self):
        """Undo the most recent interaction (if the session supports it)."""
        if self.session is None:
            return
        if not getattr(self.session, "supports_undo", False):
            slicer.util.showStatusMessage("Undo is not supported by this model.", 4000)
            return
        try:
            undone = self.session.undo()
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
            return
        if not undone:
            slicer.util.showStatusMessage("Nothing to undo.", 3000)
            return
        # Remove the marker of the interaction that was just undone.
        self._pop_prompt_undo()
        # Both backends record the region the undo changed in _last_paste_bbox (the
        # local session diffs the target buffer against its snapshot; the remote client
        # stores the server-clipped patch bbox), so update only that sub-extent instead
        # of rewriting the whole volume. None means the undo changed no voxels. Fall
        # back to a full update if the attribute ever disappears from the API.
        if hasattr(self.session, "_last_paste_bbox"):
            changed_bbox = self.session._last_paste_bbox
            if changed_bbox is not None:
                self.apply_result(changed_bbox)
        else:
            self.apply_result()

    def _check_session_layers_present(self):
        """Return human-readable names of the scene layers the *live* session is bound to
        but which no longer exist. Uses the identities recorded at the last image/segment
        sync (``previous_states``), so it reflects exactly what the backend currently
        holds: ``image_data`` is ``(volume_node_id, mtime)`` and ``segment_fp`` is
        ``(segmentation_node_id, segment_id, mtime)``. Returns an empty list when there is
        no live session or nothing has been synced yet (a missing record is "not yet
        synced", not "deleted", so it is never flagged)."""
        if self.session is None:
            return []

        missing = []

        image_fp = self.previous_states.get("image_data")
        if image_fp is not None:
            volume_id = image_fp[0]
            if volume_id and slicer.mrmlScene.GetNodeByID(volume_id) is None:
                missing.append("the source volume")

        segment_fp = self.previous_states.get("segment_fp")
        if segment_fp is not None:
            seg_node_id, segment_id = segment_fp[0], segment_fp[1]
            seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id) if seg_node_id else None
            if seg_node is None:
                missing.append("the segmentation")
            elif segment_id and seg_node.GetSegmentation().GetSegment(segment_id) is None:
                missing.append("the segment being refined")

        return missing

    def _handle_deleted_session_layers(self, missing):
        """A layer the live session depends on was deleted from the scene (see
        _check_session_layers_present). Drop the session and tell the user, so the next
        prompt starts from a clean, consistent state instead of a failed one."""
        self.release_session()
        self.update_connect_status(connected=False)
        layers = " and ".join(missing)
        slicer.util.warningDisplay(
            f"nnInteractive was working on {layers}, but it was removed from the scene "
            "(deleted in the Data module). The session has been reset to avoid an invalid "
            "state. Click Initialize at the top of the 'nnInteractive Prompts' tab to "
            "start again; any segmentation still in the scene is preserved and will be "
            "re-seeded automatically.",
            parent=slicer.util.mainWindow(),
        )

    def handle_session_expired(self, exc=None):
        """A remote lease was lost; drop the session and let the user reconnect."""
        self.release_session()
        self.update_connect_status(connected=False)
        slicer.util.warningDisplay(
            "The connection to the nnInteractive server was lost (session expired or "
            "server restarted). Click Initialize at the top of the 'nnInteractive "
            "Prompts' tab to reconnect; your current segmentation is preserved and "
            "will be re-seeded automatically.",
            parent=slicer.util.mainWindow(),
        )

    def release_session(self):
        """Tear down the current session (if any)."""
        self._stop_heartbeat_timer()
        if self.session is not None:
            close = getattr(self.session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        self.session = None
        self._session_mode = None
        self.target_buffer = None
        self._clear_prompt_undo()
        # Force a re-sync of image/segment on the next prompt.
        self.previous_states.pop("image_data", None)
        # No session -> lock the interaction UI again (re-Initialize to unlock).
        self._set_interaction_ui_enabled(False)

    def connect_clicked(self):
        """Initialize/Uninitialize toggle (top of the 'nnInteractive Prompts' tab).

        Initialize builds the session for the current mode -- loads and warms up the
        local model (the torch.compile / cuDNN dry run happens here), or connects to
        the remote server -- and then uploads the current image so the user's first
        prompt is fast (sending the image is the slow step for remote sessions).
        Clicking again uninitializes. Initialization is mandatory: the prompt controls
        stay disabled until a session is live.
        """
        mode = self.get_mode()
        if self.session is not None:
            self.release_session()
            self.update_connect_status(connected=False)
            slicer.util.showStatusMessage(
                "Disconnected from nnInteractive server."
                if mode == "remote"
                else "Local nnInteractive session released.",
                4000,
            )
            return

        if not self.ensure_session():
            # ensure_session showed the error; leave the UI in the idle state.
            self.update_connect_status(connected=self.session is not None)
            return

        # Front-load the image upload + segment seeding now, so the first prompt does
        # not pay that cost (it is the expensive step for remote: the image is sent
        # over the wire).
        try:
            self._preload_image_and_segment()
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
            return
        except Exception as exc:  # noqa: BLE001 - don't leave a half-initialized session
            self.release_session()
            self.update_connect_status(connected=False)
            slicer.util.errorDisplay(
                f"Could not upload the image to nnInteractive:\n\n{exc}",
                parent=slicer.util.mainWindow(),
            )
            return

        slicer.util.showStatusMessage(
            f"Connected to nnInteractive server at {self.server}."
            if mode == "remote"
            else "Local nnInteractive session ready.",
            4000,
        )
        self.update_connect_status(connected=self.session is not None)

    def _preload_image_and_segment(self):
        """Upload the current image and seed the selected segment right after Initialize.

        Pays the (often slow, especially remote) image-upload cost up front so the first
        prompt is fast, and records the image/segment baselines so ensure_synched does
        not redundantly re-upload on that first prompt. Initialize is gated on a loaded
        volume (see _update_initialize_button_state), so an image is normally present;
        the guard below is just defensive.

        All MRML access (image / segment extraction) happens on the main thread first;
        only the network/compute calls on ``self.session`` run in the worker behind the
        wait dialog (_run_session_upload) -- it never touches Qt or MRML. (Safe for the
        thread-local cuDNN caches: nothing in the worker runs a network forward pass.)
        """
        if self.session is None:
            return
        payload = self._build_image_payload()
        if payload is None:
            return
        image_4d, spacing, buffer = payload
        seed = self._build_segment_seed()

        def _upload():
            self.session.set_image(image_4d, {"spacing": spacing})
            self.session.set_target_buffer(buffer)
            self.session.reset_interactions()
            if seed is not None:
                self.session.add_initial_seg_interaction(seed, run_prediction=False)

        self._run_session_upload(_upload)

        # Worker succeeded: publish the shared target buffer and record the image/segment
        # baselines (main thread) so the first prompt skips a redundant re-upload/re-seed.
        self.target_buffer = buffer
        self._clear_prompt_undo()
        self.previous_states["image_data"] = self._image_fingerprint()
        self.previous_states["segment_fp"] = self._segment_fingerprint()

    def update_connect_status(self, connected):
        """Sync the Initialize/Uninitialize button + status readouts to the session state."""
        mode = self.get_mode()
        if connected:
            # The status line already reports the live mode, so the button stays plain.
            action = "Uninitialize"
            status = "initialized (remote)" if mode == "remote" else "initialized (local)"
        else:
            # Show which mode a press will start, so the user sees it before clicking.
            action = "Initialize (Local)" if mode == "local" else "Initialize (Remote)"
            status = "not initialized"
        # Initialize is disabled until an image is loaded (see
        # _update_initialize_button_state). Flag that in red so a greyed-out button
        # doesn't just look broken.
        no_image = not connected and not self._has_volume_loaded()
        if no_image:
            status = "No image loaded"
        status_style = "color: #d9534f; font-weight: bold;" if no_image else ""
        if hasattr(self.ui, "initializeButton"):
            self.ui.initializeButton.setText(action)
            self.ui.initializeButton.setStyleSheet(
                self.selected_style if connected else self.unselected_style
            )
        label = getattr(self.ui, "connectStatusLabel", None)
        if label is not None:
            label.setText(f"Status: {status}")
            label.setStyleSheet(status_style)
        # The Prompts tab combines status and model license in one line to save space,
        # but with no image there is no model yet, so show just the (red) status.
        label = getattr(self.ui, "promptsStatusLabel", None)
        if label is not None:
            if no_image:
                label.setText(f"Status: {status}")
            else:
                license_text = getattr(self, "_model_license_text", "Model license: —")
                label.setText(f"Status: {status}  |  {license_text}")
            label.setStyleSheet(status_style)
        # After a successful local Initialize the full backend is present; fill the
        # dropdown if it was still empty when the user opened the tab.
        if connected and mode == "local":
            self._ensure_model_combo_populated()
        self._update_initialize_button_state()
        self._update_device_warning(connected)

    def _update_device_warning(self, connected):
        """Show a persistent red warning below Initialize when the live local session
        fell back to the CPU (no usable CUDA GPU). Hidden for remote / GPU sessions and
        whenever no session is live."""
        label = getattr(self.ui, "promptsDeviceWarningLabel", None)
        if label is None:
            return
        show = bool(connected) and getattr(self, "_local_running_on_cpu", False)
        if show:
            label.setText(
                "⚠ No GPU detected — nnInteractive is running on the CPU, which is very "
                "slow. Likely either no compatible GPU is present, or the installed "
                "PyTorch build does not match your GPU. "
                f'See <a href="{README_COMMON_ISSUES_URL}">Common issues</a> in the README.'
            )
        label.setVisible(show)

    def _has_volume_loaded(self):
        """True if a source volume is available to initialize on. Side-effect free
        (unlike get_volume_node, which auto-selects a volume)."""
        if self.ui.editor_widget.sourceVolumeNode() is not None:
            return True
        return len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")) > 0

    def _update_initialize_button_state(self):
        """Enable Initialize only when an image is loaded, so the image is uploaded at
        init time instead of being deferred to the (slow) first prompt. While a session
        is live the button is the Uninitialize action and stays enabled."""
        btn = getattr(self.ui, "initializeButton", None)
        if btn is None:
            return
        if self.session is not None:
            btn.setEnabled(True)
            return
        has_volume = self._has_volume_loaded()
        btn.setEnabled(has_volume)
        if not has_volume:
            btn.setToolTip(
                "Load a volume into Slicer first — Initialize uploads the current image "
                "so your first prompt is fast."
            )
        else:
            btn.setToolTip(
                "Load the model / connect to the server and upload the current image.\n"
                "Required before any prompt can be placed. Click again to uninitialize."
            )

    ###############################################################################
    # Explicit dependency installation (first-run popup / Reinstall button only)
    ###############################################################################

    def ensure_torch_installed(self):
        if importlib.util.find_spec("torch") is not None:
            return
        # Prefer Slicer's PyTorch extension (PyTorchUtils): it installs a build matched
        # to the GPU driver via light-the-torch on every platform (including the correct
        # CUDA wheel on Windows). _ensure_pytorch_extension (run at install-choice time)
        # installs the extension when missing. NOT via PyTorchUtilsLogic.installTorch:
        # that blocks Slicer's main thread through the multi-GB torch download with no
        # output (pip disables progress bars without a tty), freezing the whole UI --
        # _install_torch_with_pytorch_utils runs the same command responsively.
        try:
            import PyTorchUtils  # noqa: F401

            if self._install_torch_with_pytorch_utils():
                return
            if importlib.util.find_spec("torch") is not None:
                return
        except Exception:  # noqa: BLE001
            pass
        # Fallback: plain pip. On Windows the default PyPI torch wheel is CPU-only, so
        # point pip at PyTorch's CUDA index (otherwise GPU inference silently runs on
        # CPU). On Linux the default wheel already bundles CUDA. To install a different
        # build (a CUDA version for an older GPU driver, or a pinned torch version),
        # run pip from Slicer's Python Console -- see the README.
        if os.name == "nt":
            command = f"torch --index-url {DEFAULT_WINDOWS_TORCH_INDEX_URL}"
        else:
            command = "torch"
        self._pip_install(
            command,
            "Installing PyTorch (several GB; Slicer may be unresponsive while the "
            "wheel downloads)...",
        )

    def _install_torch_with_pytorch_utils(self):
        """Install a GPU-matched torch via the PyTorch extension's light-the-torch,
        keeping the UI alive. Returns True when torch is importable afterwards.

        PyTorchUtilsLogic.installTorch runs light-the-torch as a blocking subprocess
        on Slicer's main thread, and pip prints nothing while it downloads the
        multi-GB torch wheel (progress bars are disabled without a tty) -- so Slicer
        freezes for minutes with no feedback. Instead, resolve the install command
        through PyTorchUtils (same Slicer light-the-torch fork, same platform pins),
        then run it in a plain subprocess from a worker thread while a pumped
        please-wait dialog keeps the UI responsive (_run_thread_with_message).
        """
        import shutil
        import subprocess

        import PyTorchUtils

        logic_cls = PyTorchUtils.PyTorchUtilsLogic
        # light-the-torch itself is a tiny package: installing it through the regular
        # main-thread pip path is quick, and reusing the extension's helper keeps its
        # choice of the Slicer fork (maintained for newer CUDA versions).
        try:
            import light_the_torch._patch  # noqa: F401
        except Exception:  # noqa: BLE001
            logic_cls._installLightTheTorch()
        try:
            # Private but stable helper; carries the macOS torch/torchvision/numpy pins.
            args = logic_cls._getPipInstallArguments()
        except Exception:  # noqa: BLE001
            # Private API drifted (extension update). Mirror PyTorchUtils' macOS pins
            # so the fallback doesn't silently install an incompatible combination
            # (Rosetta needs torch>=2.1.2; PyPI's macOS torch requires numpy<2), and
            # say so in the console rather than only in debug output.
            import sys

            print(
                "[nnInteractive] PyTorchUtils' install-arguments helper is "
                "unavailable (extension API changed?); using generic light-the-torch "
                "arguments."
            )
            if sys.platform == "darwin":
                args = ["install", "torch>=2.1.2", "torchvision>=0.16.2", "numpy<2"]
            else:
                args = ["install", "torch", "torchvision"]

        python = shutil.which("PythonSlicer")
        if python is None:
            # Same lookup slicer.util._executePythonModule performs; without it the
            # blocking route would fail too. Let the caller use the plain-pip fallback.
            return False

        result = {}

        def _worker(done_event):
            # Worker thread: plain subprocess only, NO Qt. Output is collected here
            # and printed from the main thread afterwards.
            try:
                kwargs = {}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                proc = subprocess.run(
                    [python, "-m", "light_the_torch", *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **kwargs,
                )
                result["returncode"] = proc.returncode
                result["output"] = proc.stdout or ""
            except Exception as exc:  # noqa: BLE001
                result["returncode"] = -1
                result["output"] = str(exc)
            finally:
                done_event.set()

        self._run_thread_with_message(
            _worker,
            (),
            "Downloading and installing PyTorch (several GB).\n"
            "This can take several minutes depending on your connection...",
        )

        output = result.get("output", "")
        if output:
            print(output)  # same Python-console visibility as slicer.util.pip_install
        importlib.invalidate_caches()
        if result.get("returncode") != 0:
            debug_print(
                f"light-the-torch install failed (rc={result.get('returncode')})."
            )
            return False
        return importlib.util.find_spec("torch") is not None

    def _ensure_pytorch_extension(self):
        """Make PyTorchUtils (Slicer's PyTorch extension) available for a Full install.

        PyTorchUtils picks a torch build matched to the machine's GPU driver via
        light-the-torch on every platform -- far more robust than the plain-pip
        fallback in ensure_torch_installed. Runs BEFORE anything is uninstalled, so
        stopping for a restart leaves the current install fully intact.

        Returns True to proceed with the install now (PyTorchUtils importable, torch
        already present, or the extension genuinely unavailable -- the plain-pip
        fallback then covers torch), False to stop so the user can restart Slicer
        first.
        """
        if importlib.util.find_spec("PyTorchUtils") is not None:
            return True
        if importlib.util.find_spec("torch") is not None:
            return True  # torch already present; ensure_torch_installed will no-op
        try:
            em = slicer.app.extensionsManagerModel()
            if em is None:
                return True
            if not em.isExtensionInstalled("PyTorch"):
                # Record the user's choice BEFORE the extension install: accepting the
                # restart that Slicer's dialog offers restarts the app immediately, and
                # the marker is what lets startup resume the install automatically.
                set_pending_install_flavor("full")
                if not em.installExtensionFromServer("PyTorch"):
                    set_pending_install_flavor("")
                    debug_print(
                        "PyTorch extension install failed or was declined; "
                        "falling back to plain-pip torch."
                    )
                    return True
        except Exception as exc:  # noqa: BLE001
            set_pending_install_flavor("")
            debug_print(f"PyTorch extension unavailable ({exc}); plain-pip torch fallback.")
            return True
        # Extension installed (just now, or earlier without a restart) but PyTorchUtils
        # is not importable in this session: a restart is needed before installTorch can
        # run. Offer plain pip as an escape hatch so a broken extension install can
        # never wedge the Full flavor behind a restart loop.
        msg = qt.QMessageBox(slicer.util.mainWindow())
        msg.setWindowTitle("Restart Slicer to finish installation")
        msg.setIcon(qt.QMessageBox.Information)
        msg.setText(
            "The PyTorch extension is installed but not active in this Slicer "
            "session yet.\n\n"
            "Restart Slicer to activate it -- the nnInteractive installation then "
            "continues automatically, with a PyTorch build matched to your GPU "
            "(recommended).\n\n"
            "Alternatively, continue now with a generic PyTorch build."
        )
        restart_btn = msg.addButton("I'll restart Slicer", qt.QMessageBox.AcceptRole)
        continue_btn = msg.addButton("Continue anyway", qt.QMessageBox.AcceptRole)
        msg.setDefaultButton(restart_btn)
        msg.exec_()
        if msg.clickedButton() == continue_btn:
            set_pending_install_flavor("")
            return True
        # Covers the path where the extension was already installed on entry (marker
        # not set above) and the user opted to restart.
        set_pending_install_flavor("full")
        return False

    # ------------------------------------------------------------------ #
    # Explicit install API. These run ONLY from the first-run popup / the
    # Configuration tab's Reinstall button -- never lazily. Every install
    # is capped below NNINTERACTIVE_VERSION_CEILING.
    # ------------------------------------------------------------------ #

    def _install_full(self):
        """Install the full in-process backend (torch + nnU-Net + nnInteractive), capped.

        The caller uninstalls any existing nnInteractive packages first, so this always
        installs the latest version below the ceiling. torch goes in first (preferring
        SlicerPyTorch for a CUDA-matched build) so the nnInteractive install finds it
        satisfied and doesn't pull a mismatched wheel. nnInteractive declares its own
        up-to-date requirements, so the single install pulls nnU-Net, huggingface_hub,
        blosc2, httpx, etc.; PLUGIN_DIRECT_DEPS covers what the plugin imports itself.
        To install a specific PyTorch build (older CUDA for an old GPU driver, or a
        pinned version), run pip from Slicer's Python Console -- see the README.
        Returns True if pip reported success.
        """
        self.ensure_torch_installed()
        self._pip_install(
            f"{NNINTERACTIVE_PKG} {PLUGIN_DIRECT_DEPS}",
            "Installing nnInteractive (full: local + remote)...",
        )
        if self._last_pip_error is not None:
            return False
        importlib.invalidate_caches()
        self.set_install_flavor("full")
        return True

    def _install_client(self):
        """Install the lightweight, torch-free remote client (capped). The
        'nninteractive-client' distribution declares its own wire-stack requirements
        (numpy/httpx/blosc2); PLUGIN_DIRECT_DEPS covers what the plugin imports itself.
        Returns True if pip reported success."""
        self._pip_install(
            f"{NNINTERACTIVE_CLIENT_PKG} {PLUGIN_DIRECT_DEPS}",
            "Installing nnInteractive client (remote only, no PyTorch)...",
        )
        if self._last_pip_error is not None:
            return False
        importlib.invalidate_caches()
        self.set_install_flavor("client")
        return True

    def _uninstall_nninteractive_packages(self):
        """Uninstall the nnInteractive / nninteractive-client distributions (only those
        packages, NOT their dependencies) before a (re)install, so switching flavor is
        clean -- e.g. reinstalling client-only after a full install actually drops the
        local backend instead of leaving it importable. pip uninstall never removes
        dependencies, so torch / nnU-Net / huggingface_hub / ... are left in place for a
        fast reinstall. No-op when nothing is installed (first run)."""
        import importlib.metadata as metadata

        installed = []
        for dist in ("nnInteractive", "nninteractive-client"):
            try:
                metadata.version(dist)
                installed.append(dist)
            except metadata.PackageNotFoundError:
                continue
        if not installed:
            return
        self._pip_uninstall(
            " ".join(installed), "Removing existing nnInteractive packages ..."
        )
        importlib.invalidate_caches()

    def _prompt_install_choice(self, reinstall=False):
        """Ask the user what to install (Full vs Client only) and run the install.

        This is the ONLY way packages get installed (besides the identical first-run
        path) -- there is no lazy install. Any installed nnInteractive packages are
        uninstalled first (dependencies untouched) so the chosen flavor is the only one
        present afterwards, and the latest capped version is installed. With ``reinstall``
        the dialog is framed as Reinstall/Update. Cancelling installs nothing, so the
        first-run prompt re-appears on the next launch.

        A Full choice first ensures the PyTorch extension is available (its install may
        require a Slicer restart, in which case the flow stops before touching the
        existing packages and a pending-install marker makes startup resume the install
        automatically after the restart -- see _resolve_install_on_startup).
        """
        if self._is_tearing_down() or self._install_prompt_active:
            return
        self._install_prompt_active = True
        try:
            self._prompt_install_choice_impl(reinstall)
        finally:
            self._install_prompt_active = False

    def _prompt_install_choice_impl(self, reinstall):
        prev_flavor = self.get_install_flavor()

        msg = qt.QMessageBox(slicer.util.mainWindow())
        msg.setWindowTitle(
            "Reinstall / Update nnInteractive" if reinstall else "Install nnInteractive"
        )
        msg.setIcon(qt.QMessageBox.Question)
        msg.setText(
            "Which nnInteractive backend should be installed into Slicer's Python?\n\n"
            "Full (local + remote)  -  runs in-process on this machine's GPU AND can "
            "connect to a remote server. Downloads the full nnInteractive + PyTorch "
            "stack (large; a GPU is needed for local inference).\n\n"
            "Client only (remote)  -  lightweight, no PyTorch. Connects to an "
            "nninteractive-server on a GPU machine; Local mode stays disabled."
        )
        full_btn = msg.addButton("Full (local + remote)", qt.QMessageBox.AcceptRole)
        client_btn = msg.addButton("Client only (remote)", qt.QMessageBox.AcceptRole)
        cancel_btn = msg.addButton(qt.QMessageBox.Cancel)
        msg.setDefaultButton(full_btn if prev_flavor == "full" else client_btn)
        msg.exec_()

        clicked = msg.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return
        # Any explicit choice supersedes a pending (restart-interrupted) install:
        # without this, an abandoned Full choice would auto-resume on a later restart
        # and silently override e.g. an explicitly chosen Client-only install.
        # _ensure_pytorch_extension re-arms the marker when Full still needs a restart.
        set_pending_install_flavor("")
        if clicked == full_btn and not self._ensure_pytorch_extension():
            # Restart required to activate the PyTorch extension. Nothing has been
            # uninstalled or installed yet, so the current state is untouched.
            return
        # Clean slate first: remove any installed nnInteractive packages (deps left in
        # place) so the chosen flavor is the only one present -- this is what makes a
        # Full -> Client switch actually drop local support.
        self._uninstall_nninteractive_packages()
        if clicked == full_btn:
            ok = self._install_full()
            new_flavor = "full"
        else:
            ok = self._install_client()
            new_flavor = "client"
        self._post_install(prev_flavor, new_flavor, ok)

    def _post_install(self, prev_flavor, new_flavor, ok):
        """After an install attempt: choose a sensible default mode, tear down any stale
        session, refresh the toggle / dropdown / install + update readouts, and warn if
        the freshly installed backend isn't importable in this session (restart needed).
        """
        if ok:
            # A completed install supersedes any pending restart-interrupted one
            # (belt to _prompt_install_choice_impl's braces; see
            # PENDING_INSTALL_SETTINGS_KEY).
            set_pending_install_flavor("")
        if not ok:
            slicer.util.errorDisplay(
                "nnInteractive installation failed. See the Python Console for the pip "
                "output, check your internet connection, and try again.",
                parent=slicer.util.mainWindow(),
            )
        elif new_flavor == "client":
            self.set_mode("remote")  # local impossible with a client-only install
        elif prev_flavor != "full":
            self.set_mode("local")  # newly gained in-process compute -> default to it

        # Refresh UI regardless of success: a partial install still changes availability.
        self._teardown_for_settings_change()
        self._apply_local_mode_availability()
        self._sync_mode_switch()
        self._ensure_model_combo_populated()
        self._refresh_install_status_ui()
        self._check_for_updates_async()

        if ok and not self._backend_installed_for_mode(self.get_mode()):
            slicer.util.infoDisplay(
                "nnInteractive was installed, but is not loadable in this running "
                "Slicer session yet (this can happen for freshly installed compiled "
                "packages such as PyTorch). Please restart Slicer.",
                parent=slicer.util.mainWindow(),
            )

        if ok and new_flavor == "client" and not slicer.util.settingsValue(
            "SlicerNNInteractive/server", ""
        ):
            # A Client-only install can't do anything until a server address is
            # entered; don't leave the user to discover that on the first prompt.
            self._show_config_tab_for_server_entry()

    def _show_config_tab_for_server_entry(self):
        """Open the Configuration tab, explain that a server address is required
        (a Client-only install is unusable without one), and focus the URL field."""
        tab_widget = getattr(self.ui, "tabWidget", None)
        if tab_widget is not None and hasattr(self.ui, "tabConfig"):
            tab_widget.setCurrentWidget(self.ui.tabConfig)
        slicer.util.infoDisplay(
            "The remote client is installed, but no nnInteractive server address is "
            "configured yet.\n\n"
            "Enter your server's address (e.g. http://gpu-box:1527) in the Server "
            "field of the Configuration tab -- plus the API key if the server "
            "requires one -- then click Initialize in the nnInteractive Prompts tab.",
            parent=slicer.util.mainWindow(),
        )
        edit = getattr(self.ui, "serverUrlEdit", None)
        if edit is not None:
            edit.setFocus()

    def _offer_backend_update(self, reason=None):
        """Explain that the installed nnInteractive backend is missing, outdated, or
        incomplete and offer the install/update dialog.

        Fired when the environment is unusable for this plugin: the Local workflow needs
        ``nnInteractive.model_management`` but it isn't importable (backend predates the
        module, was lost to a new Slicer Python environment, or only the client is
        present while the recorded flavor says Full), a plugin-direct dependency is
        missing (``scikit-image``, see _plugin_direct_deps_available), or the installed
        backend is older than NNINTERACTIVE_VERSION_FLOOR. ``reason`` overrides the
        default explanation (used for the below-minimum case). Shown at most once per
        session, and never on top of an open install dialog.
        """
        if self._offered_backend_update or self._install_prompt_active:
            return
        self._offered_backend_update = True
        default_reason = (
            "The nnInteractive backend installed in Slicer's Python is missing or "
            "incomplete for this plugin (a required module such as "
            "nnInteractive.model_management or scikit-image is not importable)."
        )
        if slicer.util.confirmYesNoDisplay(
            (reason or default_reason) + "\n\nInstall / update nnInteractive now?",
            windowTitle="nnInteractive backend outdated",
            parent=slicer.util.mainWindow(),
        ):
            self._prompt_install_choice(reinstall=True)

    def _resolve_install_on_startup(self):
        """First-run install resolver, deferred from setup() to the event loop.

        If a flavor is already recorded, verify it still holds (a recorded Full whose
        backend can no longer serve Local mode gets an update offer instead of a bare
        console error on the first Local action). Otherwise adopt a pre-existing
        install (e.g. a Slicer-bundled backend) if one is importable and current; if
        nothing usable is installed, prompt the user to choose what to install.
        Dismissing the prompt leaves nothing installed, so it re-appears on the next
        launch.
        """
        if self._is_tearing_down():
            return
        if (
            get_pending_install_flavor()
            and importlib.util.find_spec("PyTorchUtils") is not None
        ):
            # Continue the Full install that the PyTorch-extension restart
            # interrupted (only Full ever sets the marker). The user already chose
            # it, so run dialog-free -- the same steps as _prompt_install_choice
            # after the button click, including its re-entrancy guard so a click on
            # Reinstall can't start a second install mid-flight. If PyTorchUtils is
            # STILL not importable, the marker stays for the restart that actually
            # activates it -- never silently degrade to a generic torch.
            set_pending_install_flavor("")
            self._install_prompt_active = True
            try:
                prev_flavor = self.get_install_flavor()
                self._uninstall_nninteractive_packages()
                ok = self._install_full()
                self._post_install(prev_flavor, "full", ok)
            finally:
                self._install_prompt_active = False
            return
        flavor = self.get_install_flavor()
        if flavor:
            outdated = self._backend_below_min_version()
            if (
                flavor == "full" and not self._full_backend_ready()
            ) or not self._plugin_direct_deps_available():
                self._offer_backend_update()
            elif outdated:
                # Installed but older than the supported floor: offer the update flow so
                # the user lands on a version whose API this plugin actually targets.
                self._offer_backend_update(self._min_version_reason(outdated))
            self._sync_mode_switch()
            return
        detected = self._detect_install_flavor()
        if detected and (
            (detected == "full" and not self._model_management_available())
            or not self._plugin_direct_deps_available()
        ):
            # A pre-existing backend that predates model_management, or one missing a
            # plugin-direct dependency (scikit-image), is incomplete for this plugin:
            # don't adopt it silently (that would suppress the install popup and leave
            # Local / the lasso broken) -- explain and offer the installer.
            self._offer_backend_update()
            return
        if detected:
            outdated = self._backend_below_min_version()
            if outdated:
                # A pre-existing backend older than the supported floor: don't adopt it
                # silently (that would record the flavor and suppress future prompts) --
                # offer the update so the user lands on a supported version.
                self._offer_backend_update(self._min_version_reason(outdated))
                return
            self.set_install_flavor(detected)
            # Match the mode to the adopted flavor, like _post_install does for an
            # explicit install: Full means in-process compute is available, so default
            # to Local -- but only if the user never explicitly picked a mode
            # (get_mode() would otherwise report its "remote" fallback and strand a
            # Full install in Remote mode). Client lands on Remote via the
            # availability fallback below.
            if detected == "full" and not qt.QSettings().contains(
                "SlicerNNInteractive/mode"
            ):
                self.set_mode("local")
            self._apply_local_mode_availability()
            self._refresh_install_status_ui()
            self._sync_mode_switch()
            # In Local mode the dropdown would otherwise stay empty until the first
            # mode toggle (the backend was verified current above, so this is safe).
            self._ensure_model_combo_populated()
            self._check_for_updates_async()
            return
        self._prompt_install_choice(reinstall=False)

    def _refresh_install_status_ui(self):
        """Reflect the recorded install flavor in the Configuration-tab label/button."""
        label = getattr(self.ui, "installFlavorLabel", None)
        if label is not None:
            flavor = self.get_install_flavor()
            if flavor == "full":
                label.setText("Installed: <b>Full</b> (local + remote)")
            elif flavor == "client":
                label.setText("Installed: <b>Client only</b> (remote)")
            else:
                label.setText("Installed: <b>nothing yet</b>")
        btn = getattr(self.ui, "reinstallButton", None)
        if btn is not None:
            btn.setText(
                "Reinstall / Update nnInteractive"
                if self.get_install_flavor()
                else "Install nnInteractive"
            )

    # ------------------------------------------------------------------ #
    # "Update available?" check -- background network, all Qt on the main
    # thread. The worker stores a plain dict; a poll timer renders it.
    # ------------------------------------------------------------------ #

    def _check_for_updates_async(self):
        """Start a background check of whether the installed backend is up to date.

        Fails silently when offline. No-op until a flavor is installed.
        """
        if self._is_tearing_down():
            return
        label = getattr(self.ui, "updateStatusLabel", None)
        if label is None:
            return
        flavor = self.get_install_flavor()
        if not flavor:
            label.setText("")
            return
        # A backend below the hard minimum is a purely local fact: surface it right away
        # (no network) and skip the "newer on PyPI?" probe, which is moot below the floor.
        # The startup resolver also raises a modal update prompt for this same case.
        outdated = self._backend_below_min_version()
        if outdated:
            label.setText(
                "⚠ nnInteractive is outdated ("
                + "; ".join(outdated)
                + f"; minimum supported is {NNINTERACTIVE_VERSION_FLOOR}). "
                "Click 'Reinstall / Update nnInteractive'."
            )
            label.setStyleSheet("color: #c0392b; font-weight: bold;")
            return
        label.setText("Checking for nnInteractive updates ...")
        label.setStyleSheet("color: gray;")
        packages = (
            ["nnInteractive", "nninteractive-client"]
            if flavor == "full"
            else ["nninteractive-client"]
        )
        self._update_check_result = None
        if self._update_poll_timer is None:
            self._update_poll_timer = qt.QTimer()
            self._update_poll_timer.setSingleShot(False)
            self._update_poll_timer.timeout.connect(self._poll_update_check)
        worker = threading.Thread(
            target=self._update_check_worker, args=(packages,), daemon=True
        )
        worker.start()
        self._update_poll_timer.start(300)

    def _update_check_worker(self, packages):
        """Worker thread: collect installed + latest-capped versions. NO Qt access."""
        import importlib.metadata as metadata

        result = {"status": "ok", "packages": []}
        try:
            for pkg in packages:
                try:
                    installed = metadata.version(pkg)
                except Exception:  # noqa: BLE001
                    installed = None
                latest = self._pypi_latest_capped(pkg)
                result["packages"].append(
                    {"name": pkg, "installed": installed, "latest": latest}
                )
            if all(p["latest"] is None for p in result["packages"]):
                result["status"] = "error"  # couldn't reach / parse PyPI
        except Exception:  # noqa: BLE001
            result["status"] = "error"
        self._update_check_result = result

    def _pypi_latest_capped(self, pkg):
        """Latest released version of ``pkg`` on PyPI that is < the version ceiling, or
        None on any failure. Skips pre-releases and fully-yanked releases. Worker-thread
        safe (stdlib + packaging only, no Qt)."""
        import json
        import urllib.request

        try:
            from packaging.version import Version
        except Exception:  # noqa: BLE001
            return None
        try:
            url = f"https://pypi.org/pypi/{pkg}/json"
            with urllib.request.urlopen(url, timeout=4) as resp:
                data = json.load(resp)
        except Exception:  # noqa: BLE001
            return None
        ceiling = Version(NNINTERACTIVE_VERSION_CEILING)
        best = None
        for ver, files in (data.get("releases") or {}).items():
            if not files or all(f.get("yanked") for f in files):
                continue
            try:
                v = Version(ver)
            except Exception:  # noqa: BLE001
                continue
            if v.is_prerelease or v >= ceiling:
                continue
            if best is None or v > best:
                best = v
        return str(best) if best is not None else None

    def _poll_update_check(self):
        """Main thread: pick up the worker's result (if ready) and render the label."""
        result = self._update_check_result
        if result is None:
            return
        self._stop_update_poll_timer()
        self._render_update_label(result)

    def _stop_update_poll_timer(self):
        timer = getattr(self, "_update_poll_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:  # noqa: BLE001
                pass

    def _render_update_label(self, result):
        label = getattr(self.ui, "updateStatusLabel", None)
        if label is None:
            return
        if result.get("status") != "ok":
            label.setText("Could not check for nnInteractive updates.")
            label.setStyleSheet("color: gray;")
            return
        from packaging.version import Version

        outdated = []
        for p in result["packages"]:
            inst, latest = p["installed"], p["latest"]
            if inst is None or latest is None:
                continue
            try:
                if Version(inst) < Version(latest):
                    outdated.append(f"{p['name']} {inst} → {latest}")
            except Exception:  # noqa: BLE001
                continue
        if outdated:
            label.setText(
                "⟳ nnInteractive update available ("
                + "; ".join(outdated)
                + "). Click 'Reinstall / Update nnInteractive'."
            )
            label.setStyleSheet("color: #d35400; font-weight: bold;")
        else:
            label.setText("✓ nnInteractive is up to date")
            label.setStyleSheet("color: #27ae60; font-weight: bold;")

    # ------------------------------------------------------------------ #
    # Plugin (Slicer extension) "update available?" check. Separate from the
    # backend check above: this asks Slicer's Extensions Manager whether a newer
    # build of THIS extension is published on the extensions server for the
    # running Slicer version. The extension is rebuilt nightly from the repo's
    # main branch, but users are not notified -- they must open the Extensions
    # Manager and click Update. We surface that with a Config-tab label and a
    # one-time popup. The manager model is Qt/main-thread only, so the whole
    # flow runs on its async signals (no worker thread):
    #   updateExtensionsMetadataFromServer(force=True, wait=False)
    #     -> updateExtensionsMetadataFromServerCompleted(bool)   [_on_plugin_metadata_fetched]
    #        -> checkForExtensionsUpdates()
    #           -> extensionUpdatesAvailable(bool)               [_on_plugin_updates_available]
    #              -> isExtensionUpdateAvailable(PLUGIN_EXTENSION_NAME)
    # ------------------------------------------------------------------ #

    def _extensions_manager_model(self):
        """Slicer's extensions manager model, or None when unavailable (e.g. the
        extension is loaded from source via an additional module path, or this build
        has the extensions manager disabled)."""
        try:
            return slicer.app.extensionsManagerModel()
        except Exception:  # noqa: BLE001
            return None

    def _plugin_is_managed_extension(self, emm):
        """True only when THIS extension was installed through the Extensions Manager,
        so an update check is meaningful. False when running from source."""
        try:
            return PLUGIN_EXTENSION_NAME in emm.installedExtensions
        except Exception:  # noqa: BLE001
            return False

    def _check_plugin_update_async(self):
        """Ask the Extensions Manager (non-blocking) whether a newer build of this
        extension is available. Fails silently when offline, when the API differs on
        this Slicer build, or when the extension is not a managed install (dev mode)."""
        if self._is_tearing_down() or self._plugin_update_check_started:
            return
        emm = self._extensions_manager_model()
        if emm is None or not self._plugin_is_managed_extension(emm):
            self._render_plugin_update_label(None)  # dev / source checkout: show nothing
            return
        self._plugin_update_check_started = True
        self._emm = emm
        self._render_plugin_update_label("checking")
        try:
            emm.connect(
                "updateExtensionsMetadataFromServerCompleted(bool)",
                self._on_plugin_metadata_fetched,
            )
            emm.connect(
                "extensionUpdatesAvailable(bool)",
                self._on_plugin_updates_available,
            )
            # force=True (ignore the metadata cache TTL), waitForCompletion=False (async).
            emm.updateExtensionsMetadataFromServer(True, False)
        except Exception:  # noqa: BLE001
            self._disconnect_plugin_update_signals()  # drop any partial connection
            self._plugin_update_check_started = False
            self._render_plugin_update_label(None)

    def _on_plugin_metadata_fetched(self, success):
        """Server metadata arrived: run the comparison, which emits
        extensionUpdatesAvailable. NO Qt heavy lifting here."""
        emm = self._emm
        if emm is None or self._is_tearing_down():
            return
        if not success:
            self._render_plugin_update_label("error")
            return
        try:
            emm.checkForExtensionsUpdates()
        except Exception:  # noqa: BLE001
            self._render_plugin_update_label("error")

    def _on_plugin_updates_available(self, _any_available):
        """extensionUpdatesAvailable(bool) fires for the whole catalog; we care only
        about our extension, so query it specifically."""
        emm = self._emm
        if emm is None or self._is_tearing_down():
            return
        try:
            update_available = bool(emm.isExtensionUpdateAvailable(PLUGIN_EXTENSION_NAME))
        except Exception:  # noqa: BLE001
            self._render_plugin_update_label("error")
            return
        self._render_plugin_update_label("available" if update_available else "current")
        if update_available:
            self._maybe_show_plugin_update_popup()

    def _render_plugin_update_label(self, state):
        """Render the Config-tab plugin-update label. ``state`` is one of None (hide --
        dev build or manager unavailable), ``"checking"``, ``"error"``, ``"current"``,
        ``"available"``. Deliberately version-less: builds are pushed to ``main`` without
        bumping PLUGIN_VERSION, so the update status is commit-based (see
        _check_plugin_update_async). Naming a version here would read as "1.0.0 → 1.0.0"."""
        label = getattr(self.ui, "pluginUpdateStatusLabel", None)
        if label is None:
            return
        if state is None:
            label.setText("")
        elif state == "checking":
            label.setText("Checking for nnInteractive plugin updates ...")
            label.setStyleSheet("color: gray;")
        elif state == "error":
            label.setText("Could not check for nnInteractive plugin updates.")
            label.setStyleSheet("color: gray;")
        elif state == "current":
            label.setText("✓ nnInteractive plugin is up to date")
            label.setStyleSheet("color: #27ae60; font-weight: bold;")
        elif state == "available":
            label.setText(
                "⟳ nnInteractive plugin update available. Update it in the Extensions "
                "Manager (Manage Extensions → Update), then restart Slicer."
            )
            label.setStyleSheet("color: #d35400; font-weight: bold;")

    def _installed_plugin_revision(self):
        """Revision string of the currently-installed extension, or '' when unknown.
        Keys the once-per-version popup so it resets after the user updates."""
        emm = self._emm or self._extensions_manager_model()
        if emm is None:
            return ""
        try:
            meta = emm.extensionMetadata(PLUGIN_EXTENSION_NAME)
            return str(meta.get("revision") or meta.get("scm_revision") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _maybe_show_plugin_update_popup(self):
        """One-time modal telling the user a newer plugin build exists and how to get it.
        Shown at most once per installed revision (persisted to QSettings), so it never
        nags on every launch; after the user updates, the installed revision changes and
        a future update will notify again."""
        if self._plugin_update_popup_shown or self._is_tearing_down():
            return
        # Key on the installed revision so the flag resets after an update. Fall back to
        # a fixed marker when the revision can't be read (then: at most one popup ever).
        marker = self._installed_plugin_revision() or "unknown"
        if slicer.util.settingsValue(PLUGIN_UPDATE_NOTIFIED_SETTINGS_KEY, "") == marker:
            return
        self._plugin_update_popup_shown = True

        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Information)
        box.setWindowTitle("nnInteractive plugin update available")
        box.setText("A newer version of the nnInteractive Slicer extension is available.")
        box.setInformativeText(
            "To update, open the <b>Extensions Manager</b>, go to <b>Manage "
            "Extensions</b>, update <b>nnInteractive</b>, then restart Slicer."
        )
        open_btn = box.addButton("Open Extensions Manager", qt.QMessageBox.AcceptRole)
        box.addButton("Later", qt.QMessageBox.RejectRole)
        box.exec_()

        # Record that we've notified for this installed revision regardless of choice, so
        # we don't re-prompt on every launch until the user actually updates.
        settings = qt.QSettings()
        settings.setValue(PLUGIN_UPDATE_NOTIFIED_SETTINGS_KEY, marker)
        settings.sync()

        if box.clickedButton() == open_btn:
            self._open_extensions_manager()

    def _open_extensions_manager(self):
        """Open Slicer's Extensions Manager dialog. No-op on failure -- the Config-tab
        label still tells the user how to update manually."""
        try:
            slicer.app.openExtensionsManagerDialog()
        except Exception:  # noqa: BLE001
            pass

    def _disconnect_plugin_update_signals(self):
        """Drop our connections to the extensions-manager model so its async callbacks
        can't fire into a torn-down widget. Safe to call when never connected."""
        emm = getattr(self, "_emm", None)
        if emm is None:
            return
        for signal, slot in (
            ("updateExtensionsMetadataFromServerCompleted(bool)", self._on_plugin_metadata_fetched),
            ("extensionUpdatesAvailable(bool)", self._on_plugin_updates_available),
        ):
            try:
                emm.disconnect(signal, slot)
            except Exception:  # noqa: BLE001
                pass
        self._emm = None

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

    def _image_fingerprint(self):
        """
        Cheap O(1) identity for the active volume's voxels: ``(node id, voxel MTime)``.
        A node switch or any voxel edit routed through VTK bumps this; metadata-only
        edits do not. Used instead of comparing/copying the whole array each prompt.
        """
        volume_node = self.get_volume_node()
        if volume_node is None:
            return None
        image_data = volume_node.GetImageData()
        if image_data is None:
            return None
        return (volume_node.GetID(), int(image_data.GetMTime()))

    def image_changed(self):
        """
        Checks if the active volume's voxel data changed since the last time we
        synced it, using a lightweight fingerprint rather than scanning/copying the
        full image on every prompt.
        """
        fingerprint = self._image_fingerprint()
        if fingerprint is None:
            debug_print("No volume node found")
            return

        old_fingerprint = self.previous_states.get("image_data", None)
        image_changed = old_fingerprint is None or old_fingerprint != fingerprint

        self.previous_states["image_data"] = fingerprint

        return image_changed

    def _ras_to_ijk_converter(self):
        """One-time setup for RAS -> IJK conversion in the current volume node; returns
        a per-point callable. Hoist this out of loops: building the VTK transform and
        matrix dominates the cost, so per-point construction made long-contour
        conversions (lasso submit) needlessly slow."""
        volumeNode = self.get_volume_node()

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas
        )
        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)

        def _convert(pos):
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
            return [int(round(c)) for c in point_Ijk[0:3]]

        return _convert

    def ras_to_xyz(self, pos):
        """
        Converts an RAS position to IJK voxel coords in the current volume node.
        """
        return self._ras_to_ijk_converter()(pos)


    def lasso_points_to_crop(self, points):
        """
        Rasterizes a polygon (defined in a single slice) into a tight crop plus a
        half-open ``interaction_bbox`` ``[[k0, k1], [j0, j1], [i0, i1]]`` in image
        (k, j, i) = (z, y, x) order. Returns ``(crop, bbox)`` or ``(None, None)``
        if the polygon is empty.

        ``points`` are in IJK (i, j, k) order; the image array is (k, j, i), so
        x->axis2, y->axis1, z->axis0.
        """
        from skimage.draw import polygon

        zyx_shape = self.get_image_data().shape  # (z, y, x)
        pts = np.array(points)  # columns (i=x, j=y, k=z)

        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError("Expected exactly one constant coordinate among the points")
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])

        def _tighten(full2d):
            a, b = np.nonzero(full2d)
            if len(a) == 0:
                return None
            a0, a1 = int(a.min()), int(a.max()) + 1
            b0, b1 = int(b.min()), int(b.max()) + 1
            return full2d[a0:a1, b0:b1], (a0, a1), (b0, b1)

        if const_axis == 2:  # z constant -> slice over (y, x)
            rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(zyx_shape[1], zyx_shape[2]))
            full = np.zeros((zyx_shape[1], zyx_shape[2]), dtype=np.uint8)
            full[rr, cc] = 1
            tightened = _tighten(full)
            if tightened is None:
                return None, None
            crop2d, (y0, y1), (x0, x1) = tightened
            crop = crop2d[None, :, :]  # (1, dy, dx) in (z, y, x)
            bbox = [[const_val, const_val + 1], [y0, y1], [x0, x1]]
        elif const_axis == 1:  # y constant -> slice over (z, x)
            rr, cc = polygon(pts[:, 2], pts[:, 0], shape=(zyx_shape[0], zyx_shape[2]))
            full = np.zeros((zyx_shape[0], zyx_shape[2]), dtype=np.uint8)
            full[rr, cc] = 1
            tightened = _tighten(full)
            if tightened is None:
                return None, None
            crop2d, (z0, z1), (x0, x1) = tightened
            crop = crop2d[:, None, :]  # (dz, 1, dx)
            bbox = [[z0, z1], [const_val, const_val + 1], [x0, x1]]
        else:  # const_axis == 0, x constant -> slice over (z, y)
            rr, cc = polygon(pts[:, 2], pts[:, 1], shape=(zyx_shape[0], zyx_shape[1]))
            full = np.zeros((zyx_shape[0], zyx_shape[1]), dtype=np.uint8)
            full[rr, cc] = 1
            tightened = _tighten(full)
            if tightened is None:
                return None, None
            crop2d, (z0, z1), (y0, y1) = tightened
            crop = crop2d[:, :, None]  # (dz, dy, 1)
            bbox = [[z0, z1], [y0, y1], [const_val, const_val + 1]]

        return np.ascontiguousarray(crop), bbox

    def _mask_to_crop(self, mask):
        """
        Tighten a (z, y, x) binary mask to a crop plus its half-open bbox
        ``[[k0, k1], [j0, j1], [i0, i1]]`` (in the mask's own coordinates). Returns
        ``(None, None)`` if the mask is empty. The caller offsets the bbox when the
        mask is a sub-region rather than the full volume (see the region scribble path).
        """
        nz = np.argwhere(mask > 0)
        if nz.size == 0:
            return None, None
        mins = nz.min(0)
        maxs = nz.max(0) + 1
        bbox = [[int(mins[a]), int(maxs[a])] for a in range(3)]
        crop = mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].astype(np.uint8)
        return np.ascontiguousarray(crop), bbox

    def _apply_scribble_brush_size(self, paint_effect):
        """
        Sets the paint brush size. If a session is connected, use the model's
        preferred scribble thickness (in voxels) converted to an absolute (mm)
        diameter; otherwise fall back to a relative brush.
        """
        thickness = getattr(self.session, "preferred_scribble_thickness", None) if self.session else None
        if thickness:
            spacing = self.get_image_spacing()  # (z, y, x) mm/voxel
            # Half-voxel pad: Paint stencils in voxels whose centers lie strictly
            # inside the brush, so at exactly `thickness` voxels the neighbouring
            # centers sit on the boundary and strokes rasterize one voxel too thin.
            # Padded, a stroke spans `thickness` (occasionally +1) voxels, never fewer.
            diameters_mm = [(t + 0.5) * s for t, s in zip(thickness, spacing)]
            # Use the median extent: the odd-one-out spacing (largest or smallest) is
            # the through-plane axis of the acquisition, so the median is the in-plane
            # voxel size of the plane users typically paint in. min() would undersize
            # the brush whenever slices are thinner than in-plane voxels.
            diameters_mm.sort()
            diameter_mm = max(diameters_mm[len(diameters_mm) // 2], 0.1)
            paint_effect.setParameter("BrushUseAbsoluteSize", "1")
            paint_effect.setParameter("BrushAbsoluteDiameter", str(diameter_mm))
        else:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")
            paint_effect.setParameter("BrushRelativeDiameter", ".75")

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

    def _set_polarity(self, positive):
        """Switch the prompt polarity: restyle the toggle buttons and recolour the
        active tool's cursor / an active scribble's target label."""
        self.ui.pbPromptTypePositive.setStyleSheet(
            self.selected_style if positive else self.unselected_style
        )
        self.ui.pbPromptTypeNegative.setStyleSheet(
            self.unselected_style if positive else self.selected_style
        )
        self.ui.pbPromptTypePositive.setChecked(positive)
        self.ui.pbPromptTypeNegative.setChecked(not positive)
        self._update_prompt_cursor()
        self._retarget_scribble_if_active()
        debug_print(f"Prompt type set to {'POSITIVE' if positive else 'NEGATIVE'}")

    def on_prompt_type_positive_clicked(self, checked=False):
        self._set_polarity(True)

    def on_prompt_type_negative_clicked(self, checked=False):
        self._set_polarity(False)

    def toggle_prompt_type(self, checked=False):
        """
        Toggle between positive and negative (triggered by 'T' key).
        """
        # The 'T' shortcut bypasses the disabled buttons, so enforce mandatory init here.
        if not self._require_initialized():
            return
        self._set_polarity(not self.is_positive)


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
