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
    Decorator that ensures a working nnInteractive session exists and that the
    image and segment are synced to it before calling the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        # Make sure we have a live session (constructs a local one or connects
        # to a remote server, depending on the configured mode).
        if not self.ensure_session():
            return

        try:
            # Per-step timing (set DEBUG_MODE = True at the top of this file to see it):
            # this is the GUI work that wraps every prompt, so it shows where the
            # end-to-end latency goes beyond the model's prediction itself.
            t0 = time.time()
            if self._handle_active_source_volume_change():
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

        # nnInteractive session state (constructed lazily on first prompt / Connect).
        self.session = None
        self.target_buffer = None
        self.api_key = ""
        self.server = ""
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
        # The persistent filled prompt is a labelmap overlay (see _paint_prompt_overlay).
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
        self._active_cursor_tool = None
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

        # DON'T install packages here. Dependency installation is lazy and
        # import-verified (see ensure_session / the _construct_* methods), so an
        # aborted/partial install simply gets retried the next time the user connects
        # or prompts, instead of leaving the module wedged. The first-run mode dialog
        # is deferred to the event loop (modal dialogs during setup() are unreliable),
        # and is also resolved lazily in ensure_session() as a safety net.
        qt.QTimer.singleShot(0, self.resolve_mode_first_run)

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
                # The lasso is captured as a freehand drag (not markups Place mode),
                # so it has no PointPositionDefinedEvent handler.
                "on_placed_function": None,
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
        self._active_source_volume_id = self._volume_node_id(self.get_volume_node())

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        self.ui.uploadProgressGroup.setVisible(False)

        # Build the Configuration-tab model selection + settings (Local | Remote
        # switch, checkpoint/device/compile, server URL, API key, Connect, ...). This
        # also loads and wires the saved server URL and sets self.server.
        self.setup_config_ui()

        # Prominent license banner + acknowledgement logos in the Prompts tab.
        self.setup_prompts_tab_extras()

        # Set initial prompt type
        self.current_prompt_type_positive = True
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

        self.ui.pbInteractionLassoCancel.setVisible(False)  # obsolete with freehand lasso
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)
        self.addObserver(
            self.segment_editor_node,
            vtk.vtkCommand.ModifiedEvent,
            self.on_segment_editor_node_modified,
        )

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

        # The old .ui server widgets are replaced by the model-selection group below.
        self.ui.serverGroup.setVisible(False)
        self.ui.pbTestServer.setVisible(False)

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
        self.ui.checkpointEdit.setText(self.get_setting_str("checkpoint_path", ""))
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
        self.ui.compileCheck.setChecked(self.get_setting_bool("use_torch_compile", False))
        self.ui.compileCheck.toggled.connect(
            lambda v: self._save_setting_bool("use_torch_compile", v, reinit=True)
        )
        if not self._torch_compile_supported():
            # No Python dev headers (Python.h) -> torch.compile/Triton cannot build its
            # runtime helpers (typical in Slicer's bundled Python). Disable the option so
            # the user can't turn on something that would fail on the first prediction.
            self.ui.compileCheck.setEnabled(False)
            self.ui.compileCheck.setToolTip(
                "Unavailable in this Python build: torch.compile needs the Python "
                "development headers (Python.h), which are not present."
            )
        advanced_form.addRow("Use torch.compile:", self.ui.compileCheck)

        self.ui.storageCombo = qt.QComboBox()
        self.ui.storageCombo.addItems(["auto", "blosc2", "tensor"])
        self.ui.storageCombo.setCurrentText(self.get_setting_str("interactions_storage", "auto"))
        self.ui.storageCombo.currentTextChanged.connect(
            lambda v: self._save_setting("interactions_storage", v, reinit=True)
        )
        advanced_form.addRow("Interaction storage:", self.ui.storageCombo)

        # Auto-zoom is a local-only setting (it is baked into the local session at
        # construction). For remote sessions the server decides, so this lives here in
        # the local container rather than the shared Settings group.
        self.ui.autozoomCheck = qt.QCheckBox()
        self.ui.autozoomCheck.setChecked(self.get_setting_bool("autozoom", True))
        self.ui.autozoomCheck.toggled.connect(self.on_autozoom_toggled)
        advanced_form.addRow("Auto-zoom:", self.ui.autozoomCheck)

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

        # --- Connect + status (common to both modes) ---
        self.ui.connectButton = qt.QPushButton("Connect")
        self.ui.connectButton.setMinimumHeight(30)
        self.ui.connectButton.clicked.connect(self.connect_clicked)
        model_layout.addWidget(self.ui.connectButton)

        self.ui.connectStatusLabel = qt.QLabel("Status: not connected")
        self.ui.connectStatusLabel.setWordWrap(True)
        model_layout.addWidget(self.ui.connectStatusLabel)

        # ===== Settings group =====
        settings_group = qt.QGroupBox("Settings")
        settings_form = qt.QFormLayout(settings_group)

        self.ui.licenseLabel = qt.QLabel("Model license: —")
        self.ui.licenseLabel.setWordWrap(True)
        settings_form.addRow(self.ui.licenseLabel)

        # Insert the two groups just above the trailing vertical spacer.
        idx = max(layout.count() - 1, 0)
        layout.insertWidget(idx, model_group)
        layout.insertWidget(idx + 1, settings_group)

        # Initial switch state + container visibility, and the cached server URL.
        is_local = self.get_mode() == "local"
        self.ui.localModeButton.setChecked(is_local)
        self.ui.remoteModeButton.setChecked(not is_local)
        self._update_mode_visibility()
        self.server = self.ui.serverUrlEdit.text.rstrip("/")

        # Reflect the persisted mode on the action button (Connect / Initialize). If
        # Local is already the active mode, fill the dropdown right after setup finishes
        # (deferred so we don't pip-install the lightweight listing deps mid-construction).
        self.update_connect_status(connected=False)
        if is_local:
            qt.QTimer.singleShot(
                0, lambda: self._ensure_model_combo_populated(allow_install=True)
            )

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
            return  # re-click of the active mode -- don't drop the session
        self.set_mode(mode)
        self._update_mode_visibility()
        # Switching the toggle does not disconnect/uninitialize; just reflect whether a
        # session for the now-selected mode already exists.
        self.update_connect_status(connected=self._session_active_for_mode(mode))
        if mode == "local":
            # User explicitly chose Local: prepare the model dropdown now (installing
            # the lightweight listing deps if needed).
            self._ensure_model_combo_populated(allow_install=True)

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

    def _apply_local_mode_availability(self):
        """Gate Local mode on whether the full backend is installed.

        Local mode runs the model in-process and needs the full nnInteractive package
        (torch + nnU-Net). When only the lightweight remote client is installed we
        disable the Local button — and fall back to Remote — so the user never lands on
        a Local page that cannot work. The disabled button's tooltip explains how to
        enable Local support.
        """
        available = self._local_inference_available()
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
                    "To enable Local mode, install the full backend (PyTorch + nnU-Net)\n"
                    "into Slicer's Python and restart Slicer:\n"
                    "    import slicer.util\n"
                    '    slicer.util.pip_install("nnInteractive")'
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

    def setup_prompts_tab_extras(self):
        """
        Adds a prominent model-license banner (top) and the Helmholtz Imaging + DKFZ
        acknowledgement logos (bottom) to the Prompts tab -- the view the user
        actually works in.
        """
        layout = self.ui.tabPrompts.layout()

        self.ui.promptsLicenseLabel = qt.QLabel("Model license: —")
        self.ui.promptsLicenseLabel.setWordWrap(True)
        self.ui.promptsLicenseLabel.setAlignment(qt.Qt.AlignCenter)
        self.ui.promptsLicenseLabel.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.insertWidget(0, self.ui.promptsLicenseLabel)

        logos = self._build_logos_widget(height=26)
        if logos is not None:
            layout.addWidget(logos)

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
        # Auto-zoom is baked into the local session at construction, so changing it
        # tears the session down (like the other session-affecting settings); the user
        # re-initializes to apply it.
        self._save_setting_bool("autozoom", checked, reinit=True)

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.on_next_segment,
            "r": self.clear_current_segment,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
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

    def _pip_install(self, command, message):
        """
        Install pip package(s) on the MAIN thread.

        slicer.util.pip_install creates Qt objects, so it MUST NOT run in a worker
        thread -- doing so throws "QObject ... different thread" errors and deadlocks.
        We show a non-cancelable busy dialog; the UI is blocked while pip runs (a
        one-time cost) and pip's output streams to the Python Console. Failures are
        recorded and ultimately surfaced by the import verification in
        _import_with_install().
        """
        self._last_pip_error = None
        progress = slicer.util.createProgressDialog(
            parent=slicer.util.mainWindow(),
            maximum=0,  # indeterminate / busy
            labelText=message,
            windowTitle="nnInteractive",
        )
        slicer.app.processEvents()
        try:
            slicer.util.pip_install(command)
        except Exception as exc:  # noqa: BLE001
            self._last_pip_error = exc
            debug_print(f"pip install '{command}' failed: {exc}")
        finally:
            progress.close()
            slicer.app.processEvents()

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
        self._cleanup_in_progress = True
        self._pending_cursor_reassert_widgets = []
        self._cursor_reassert_timer_queued = False

        self.removeObservers()
        self._remove_slice_event_filters()
        self._remove_scribble_labelmap_observer()

        if not getattr(self, "_application_quitting", False):
            # Restore the default slice-view cursor (we may have set a coloured one).
            self._clear_prompt_cursor()

            # Remove the freehand-lasso overlay actors from their slice view.
            self._lasso_overlay_remove()

        # Release any remote lease / free the local model.
        self.release_session()

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
        self._stop_heartbeat_timer()

    def _remove_slice_event_filters(self):
        if hasattr(self, "_qt_event_filters"):
            for target, event_filter in self._qt_event_filters:
                try:
                    target.removeEventFilter(event_filter)
                except Exception:  # noqa: BLE001
                    pass
            self._qt_event_filters = []

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
        # Remember the active tool so it can be re-armed after the reset (below).
        active_tool = self._active_prompt_tool()

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
            # interaction is stale even though the button still looks selected. Deselect
            # and (deferred) re-select so it works without a manual second click, exactly
            # like on_next_segment does.
            self._place_tool = None  # also drop point/bbox click-capture
            self._cancel_bbox_drag()
            for button in self.all_prompt_buttons.values():
                button.setChecked(False)
            self._update_prompt_cursor()
            if active_tool is not None:
                qt.QTimer.singleShot(0, lambda: self._activate_prompt_tool(active_tool))
        else:
            debug_print("No segment selected to clear.")

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
        # other segments alone; we don't pass an extent, so it replaces using the small
        # image's own extent (zeroing this segment outside it, which is what we want here).
        from vtk.util import numpy_support

        sub = np.ascontiguousarray(mask[k0:k1, j0:j1, i0:i1], dtype=np.uint8)
        oriented = slicer.vtkOrientedImageData()
        oriented.SetExtent(i0, i1 - 1, j0, j1 - 1, k0, k1 - 1)
        ijk_to_ras = vtk.vtkMatrix4x4()
        self.get_volume_node().GetIJKToRASMatrix(ijk_to_ras)
        oriented.SetImageToWorldMatrix(ijk_to_ras)
        oriented.GetPointData().SetScalars(
            numpy_support.numpy_to_vtk(sub.reshape(-1), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        )
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
            oriented,
            segmentationNode,
            segmentId,
            slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE,
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

        # NOTE: CollapseBinaryLabelmaps() is NOT called here. It is O(volume x segments)
        # and ran on every prediction; collapsing is a storage optimization that we now
        # do once per object (on 'Next segment') instead of on the interactive hot path.
        del segmentation_mask

        # Record the post-write fingerprint so selected_segment_changed() sees "no change"
        # next prompt (it only re-seeds on an *external* edit), without re-extracting and
        # comparing the whole labelmap.
        self.previous_states["segment_fp"] = self._segment_fingerprint()

        debug_print(f"show_segmentation took {time.time() - t0}")

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

    def _session_active_for_mode(self, mode):
        """True if the live session was built for ``mode``."""
        return self.session is not None and getattr(self, "_session_mode", None) == mode

    def get_setting_bool(self, key, default):
        return slicer.util.settingsValue(
            f"SlicerNNInteractive/{key}", default, converter=slicer.util.toBool
        )

    def get_setting_str(self, key, default):
        return slicer.util.settingsValue(f"SlicerNNInteractive/{key}", default)

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
        action button so the user explicitly restarts (Initialize) / reconnects
        (Connect) with the new settings applied. The next session is rebuilt from
        scratch, so it always reads the updated settings."""
        had_session = self.session is not None
        self.release_session()
        self.update_connect_status(connected=False)
        if had_session:
            action = "Initialize" if self.get_mode() == "local" else "Connect"
            slicer.util.showStatusMessage(f"Settings changed — click {action} to apply.", 5000)

    def update_server(self):
        """Reads the server URL from the UI and persists it. Changing it disconnects the
        current remote session so the next Connect uses the new address."""
        edit = getattr(self.ui, "serverUrlEdit", None)
        text = edit.text if edit is not None else self.ui.Server.text
        new_server = text.rstrip("/")
        changed = new_server != self.server
        self.server = new_server
        qt.QSettings().setValue("SlicerNNInteractive/server", self.server)
        debug_print(f"Server URL updated and saved: {self.server}")
        if changed:
            # Drop the existing session so the new address actually takes effect.
            self._teardown_for_settings_change()

    @property
    def auto_run(self):
        # Auto-run is always on (the toggle was removed); a prediction runs as each
        # prompt is added.
        return True

    def get_local_device(self):
        return self.get_setting_str("device", "cuda:0") or "cuda:0"

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

        # Make sure a mode has been chosen (shows the first-run dialog if needed).
        self.resolve_mode_first_run()
        mode = self.get_mode()

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

    def _import_with_install(self, install_fn, import_fn):
        """
        Install the deps for a mode, then import. The import is the source of truth:
        if it fails (deps missing, or a previous install was aborted and left a
        partial/broken package that ``find_spec`` wrongly accepts), force a repair
        reinstall and retry once. This is what makes an aborted install self-heal on
        the next connect/prompt instead of staying wedged.
        """
        install_fn(force=False)
        try:
            return import_fn()
        except ImportError as first_error:
            debug_print(f"Import failed ({first_error}); repairing install and retrying.")
            install_fn(force=True)
            importlib.invalidate_caches()
            return import_fn()

    def _construct_local_session(self):
        """Build an in-process nnInteractiveInferenceSession (local compute)."""
        # Heavy imports are deferred to local mode so remote-only users never load torch.
        def _imp():
            import torch
            from nnInteractive.inference.inference_session import (
                nnInteractiveInferenceSession,
            )
            return torch, nnInteractiveInferenceSession

        torch, nnInteractiveInferenceSession = self._import_with_install(
            self.install_local_dependencies, _imp
        )

        checkpoint_path = self.get_checkpoint_path()  # downloads weights if needed

        custom = self.get_setting_str("checkpoint_path", "").strip()
        if custom:
            model_desc = f"custom checkpoint at {checkpoint_path}"
        else:
            model_id = self.get_setting_str("model_id", "").strip() or "default"
            model_desc = f"'{model_id}' ({checkpoint_path})"
        print(f"[nnInteractive] Local session using model: {model_desc}")
        slicer.util.showStatusMessage(f"nnInteractive model: {model_desc}", 5000)

        if torch.cuda.is_available():
            device = torch.device(self.get_local_device())
        else:
            slicer.util.showStatusMessage(
                "No CUDA GPU detected - running nnInteractive on CPU (slow).", 5000
            )
            device = torch.device("cpu")

        do_autozoom = self.get_setting_bool("autozoom", True) and device.type == "cuda"

        use_torch_compile = self.get_setting_bool("use_torch_compile", False)
        if use_torch_compile and not self._torch_compile_supported():
            # torch.compile (Triton/inductor) compiles C/CUDA helpers at runtime and
            # needs the Python development headers (Python.h). Slicer's bundled Python
            # ships without them, so compilation fails on the first prediction. Fall
            # back to eager execution instead of crashing the session.
            use_torch_compile = False
            msg = (
                "torch.compile is unavailable in this Python (missing development "
                "headers / Python.h); running without it."
            )
            print(f"[nnInteractive] {msg}")
            slicer.util.showStatusMessage(msg, 6000)

        session = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=use_torch_compile,
            torch_n_threads=os.cpu_count(),
            verbose=False,
            do_autozoom=do_autozoom,
            interactions_storage=self.get_setting_str("interactions_storage", "auto"),
        )

        # Load the weights and run a warmup forward pass NOW, so Initialize gets the model
        # fully ready instead of paying the (often large) first-forward cost -- cuDNN
        # autotuning, CUDA kernel init, and torch.compile compilation -- on the user's
        # first prompt. Shown behind a busy dialog because it blocks the main thread.
        progress = slicer.util.createProgressDialog(
            parent=slicer.util.mainWindow(),
            maximum=0,  # indeterminate / busy
            labelText="Loading nnInteractive model ...",
            windowTitle="nnInteractive",
        )
        slicer.app.processEvents()
        try:
            session.initialize_from_trained_model_folder(
                checkpoint_path, 0, "checkpoint_final.pth"
            )
            warmup = getattr(session, "warmup", None)
            if callable(warmup):
                progress.setLabelText("Warming up the model (first run only) ...")
                slicer.app.processEvents()
                try:
                    warmup()
                except Exception as exc:  # noqa: BLE001 - warmup is an optimization only
                    debug_print(f"Model warmup failed (non-fatal): {exc}")
        finally:
            progress.close()
            slicer.app.processEvents()
        return session

    @staticmethod
    def _torch_compile_supported():
        """True only if this Python can build the C/CUDA helpers torch.compile needs.

        torch.compile -> Triton/inductor compiles small C extensions at runtime, which
        requires the Python development headers (Python.h). Slicer's bundled Python does
        not ship them, so we detect their absence and disable torch.compile rather than
        let the first prediction fail with a compiler error.
        """
        import sysconfig

        include_dir = sysconfig.get_paths().get("include")
        return bool(include_dir) and os.path.isfile(os.path.join(include_dir, "Python.h"))

    def _construct_remote_session(self):
        """Connect to an official nninteractive-server (remote compute)."""
        def _imp():
            from nnInteractive.inference.remote.remote_session import (
                ServerAtCapacityError,
                SessionExpiredError,
                nnInteractiveRemoteInferenceSession,
            )
            return (
                nnInteractiveRemoteInferenceSession,
                SessionExpiredError,
                ServerAtCapacityError,
            )

        RemoteSession, SessionExpiredError, ServerAtCapacityError = self._import_with_install(
            self.install_client_dependencies, _imp
        )

        # Remember which exceptions mean "the lease is gone" so callers can reconnect.
        self.SESSION_LOST_ERRORS = (SessionExpiredError,)

        self.update_server()
        if not self.server:
            raise RuntimeError(
                "No server URL set. Enter it in the 'Configuration' tab and click Connect."
            )

        api_key = self.api_key or None
        try:
            session = RemoteSession(server_url=self.server, api_key=api_key)
        except ServerAtCapacityError as exc:
            raise RuntimeError(f"The nnInteractive server is at capacity: {exc}") from exc

        # Auto-zoom is decided by the server for remote sessions; the client does not
        # override it (the Auto-zoom setting is local-only).
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
            # silent. The usual cause during development is a Slicer-bundled
            # nnInteractive that predates the model_management module — install the
            # updated backend into Slicer's Python (see README / status message below).
            print(f"[nnInteractive] Could not load model list: {exc!r}")
            slicer.util.showStatusMessage(
                "nnInteractive: could not load the model list (see Python Console).", 5000
            )
            return
        if not models:
            return

        saved_id = self.get_setting_str("model_id", "")
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

    def _install_listing_dependencies(self):
        """Install just enough to LIST models (no torch): the nnInteractive package
        and huggingface_hub. Much lighter than the full local compute stack, so the
        model dropdown can be filled before the user commits to a local Initialize."""
        # `nnInteractive` is now a PEP 420 namespace package shared with the
        # 'nninteractive-client' distribution, so find_spec("nnInteractive") is truthy
        # even when only the lightweight client is installed. Key the install off a
        # full-package-only module (model_management) so we still install the full
        # package when only the client is present.
        if importlib.util.find_spec("nnInteractive.model_management") is None:
            self._pip_install("--no-deps nnInteractive", "Preparing nnInteractive model list...")
        self._pip_install_if_missing("huggingface_hub", "huggingface_hub")
        importlib.invalidate_caches()

    def _ensure_model_combo_populated(self, allow_install=True):
        """Fill the model dropdown, optionally installing the lightweight listing deps.

        Slicer's bundled Python does not ship huggingface_hub (and may not have the
        nnInteractive package yet in remote mode), so the dropdown is empty until those
        are present. When ``allow_install`` is True (e.g. the user just picked Local)
        we install them on demand; otherwise we only populate if they already exist.
        """
        combo = getattr(self.ui, "modelComboBox", None)
        # Note: in Slicer's PythonQt binding, QComboBox.count is a property, not a method.
        if combo is None or combo.count > 0:
            return
        # Check model_management (a full-package-only module), not bare "nnInteractive":
        # the latter is a namespace package that also exists with only the lightweight
        # client installed, which would wrongly skip installing the listing backend.
        if allow_install and (
            importlib.util.find_spec("nnInteractive.model_management") is None
            or importlib.util.find_spec("huggingface_hub") is None
        ):
            self._install_listing_dependencies()
        self._populate_model_combo()

    def _on_model_combo_changed(self, index):
        """Persist the chosen model id and invalidate the session so it reloads."""
        if 0 <= index < len(self._model_ids):
            self._save_setting("model_id", self._model_ids[index], reinit=True)

    def get_checkpoint_path(self):
        """Local model folder; downloads the selected official model on first use."""
        custom = self.get_setting_str("checkpoint_path", "").strip()
        if custom:
            return custom

        from nnInteractive.model_management import ensure_model_available, get_default_model_id

        model_id = self.get_setting_str("model_id", "").strip() or None
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

    def _on_session_ready(self):
        """Reflect a freshly created session in the UI (license, capability gating)."""
        session = self.session
        license_text = self._license_display_text(getattr(session, "license", None))
        if hasattr(self.ui, "licenseLabel"):
            self.ui.licenseLabel.setText(license_text)
        if hasattr(self.ui, "promptsLicenseLabel"):
            self.ui.promptsLicenseLabel.setText(license_text)

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

        # For remote sessions, keep the lease alive with a main-thread Qt timer.
        # The client also has a background daemon heartbeat, but a Qt timer driven by
        # Slicer's event loop is more reliable than a Python daemon thread here.
        if hasattr(session, "heartbeat"):
            self._start_heartbeat_timer()
        else:
            self._stop_heartbeat_timer()

    def _start_heartbeat_timer(self):
        """(Re)start the main-thread heartbeat timer for a remote session."""
        if getattr(self, "_cleanup_in_progress", False) or getattr(
            self, "_application_quitting", False
        ):
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

    def sync_image_to_session(self):
        """Push the current volume into the session and allocate the target buffer."""
        image = self.get_image_data()
        if image is None:
            debug_print("No image data available to sync.")
            return False

        # nnInteractive expects a 4D array [C, X, Y, Z]; Slicer arrays are [k, j, i].
        image_4d = np.ascontiguousarray(image[None])
        spacing = self.get_image_spacing()

        self.target_buffer = np.zeros(image.shape, dtype=np.uint8)
        self.session.set_image(image_4d, {"spacing": spacing})
        self.session.set_target_buffer(self.target_buffer)
        # Re-seeding of the active segment is handled by sync_segment_to_session().
        self.previous_states.pop("segment_fp", None)
        return True

    def sync_segment_to_session(self):
        """Seed the session with the currently selected segment (for editing)."""
        self.session.reset_interactions()
        self._clear_prompt_undo()  # interactions reset -> drop their pending undo markers
        # Seeding only matters when the segment already holds a mask. A fresh segment (the
        # common case on the first prompt of a new object) is empty, so detect that cheaply
        # and skip the expensive full-volume extract get_segment_data() would otherwise do.
        if getattr(self.session, "supports_initial_label", True) and not self._selected_segment_is_empty():
            seg = self.get_segment_data().astype(np.uint8)
            if seg.sum() > 0:
                self.session.add_initial_seg_interaction(seg, run_prediction=False)
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

    def run_prediction_now(self):
        """Manual 'Run' for when auto-run is off (local sessions only)."""
        if self.session is None:
            return
        predict = getattr(self.session, "_predict", None)
        if predict is None:
            slicer.util.showStatusMessage(
                "Manual run is only available in local mode.", 4000
            )
            return
        try:
            changed_bbox = predict()
        except self.SESSION_LOST_ERRORS as exc:
            self.handle_session_expired(exc)
            return
        self.apply_result(changed_bbox)

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
        self.apply_result()

    def handle_session_expired(self, exc=None):
        """A remote lease was lost; drop the session and let the user reconnect."""
        self.release_session()
        self.update_connect_status(connected=False)
        slicer.util.warningDisplay(
            "The connection to the nnInteractive server was lost (session expired or "
            "server restarted). Reconnect from the 'Configuration' tab; your current "
            "segmentation is preserved and will be re-seeded automatically.",
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

    def connect_clicked(self):
        """Configuration-tab action button: toggle the session for the current mode.

        Remote: Connect / Disconnect. Local: Initialize / Uninitialize. If a session
        is live, clicking tears it down; otherwise it (re)builds one.
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

        if self.ensure_session():
            slicer.util.showStatusMessage(
                f"Connected to nnInteractive server at {self.server}."
                if mode == "remote"
                else "Local nnInteractive session ready.",
                4000,
            )
        # ensure_session updates the status on success (via _on_session_ready); refresh
        # here too so a failed attempt leaves the button in the correct (idle) state.
        self.update_connect_status(connected=self.session is not None)

    def update_connect_status(self, connected):
        """Sync the action button label + status text to the mode and session state."""
        mode = self.get_mode()
        if mode == "remote":
            action = "Disconnect" if connected else "Connect"
            status = "connected" if connected else "not connected"
        else:
            action = "Uninitialize" if connected else "Initialize"
            status = "initialized" if connected else "not initialized"
        if hasattr(self.ui, "connectButton"):
            self.ui.connectButton.setText(action)
        if hasattr(self.ui, "connectStatusLabel"):
            self.ui.connectStatusLabel.setText(f"Status: {status}")
        # After a successful local Initialize the listing deps are present; fill the
        # dropdown if it was still empty when the user opened the tab.
        if connected and mode == "local":
            self._ensure_model_combo_populated(allow_install=False)

    ###############################################################################
    # Dependency installation (mode-gated: client vs local)
    ###############################################################################

    def _pip_install_if_missing(self, import_name, pip_name, force=False):
        # With force=True we reinstall regardless of find_spec, to repair a partial /
        # aborted install that find_spec would otherwise wrongly accept.
        if not force and importlib.util.find_spec(import_name) is not None:
            return
        command = f"--force-reinstall {pip_name}" if force else pip_name
        self._pip_install(command, f"Installing dependency: {pip_name} ...")

    def install_client_dependencies(self, force=False):
        """
        Lightweight, torch-free client deps for remote mode. The remote client lives
        in the dedicated 'nninteractive-client' distribution, which provides
        nnInteractive.inference.remote and pulls only the wire stack
        (numpy/httpx/blosc2) — no torch / nnU-Net. (The full 'nnInteractive' package
        depends on this client, so a full local install includes it too. Installing
        '--no-deps nnInteractive' here would NOT work: the remote client is no longer
        part of that distribution.)
        """
        if force or importlib.util.find_spec("nnInteractive.inference.remote") is None:
            flags = "--force-reinstall " if force else ""
            self._pip_install(
                f"{flags}nninteractive-client",
                "Installing nnInteractive client (no PyTorch)...",
            )
        self._pip_install_if_missing("httpx", "httpx", force=force)
        self._pip_install_if_missing("blosc2", "blosc2", force=force)
        self._pip_install_if_missing("skimage", "scikit-image", force=force)

    def ensure_torch_installed(self):
        if importlib.util.find_spec("torch") is not None:
            return
        # Prefer Slicer's PyTorch extension (PyTorchUtils) for a CUDA-matched build.
        try:
            import PyTorchUtils

            PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=True)
            if importlib.util.find_spec("torch") is not None:
                return
        except Exception:  # noqa: BLE001
            pass
        self._pip_install("torch", "Installing PyTorch (this can take a while)...")

    def install_local_dependencies(self, force=False):
        """Full in-process compute stack (torch + nnU-Net + nnInteractive)."""
        self._pip_install_if_missing("skimage", "scikit-image", force=force)
        self._pip_install_if_missing("httpx", "httpx", force=force)
        self._pip_install_if_missing("blosc2", "blosc2", force=force)
        # Install torch first (preferring SlicerPyTorch) so the nnInteractive install
        # finds it already satisfied and doesn't pull a mismatched wheel.
        self.ensure_torch_installed()
        if force:
            # Repair a partial/broken nnInteractive without clobbering torch:
            # reinstall just the package files (--no-deps), then install any missing
            # dependencies (without --force, so the existing torch is left alone).
            self._pip_install(
                "--force-reinstall --no-deps nnInteractive", "Repairing nnInteractive ..."
            )
            self._pip_install("nnInteractive", "Installing nnInteractive dependencies ...")
        else:
            # Default nnInteractive install brings nnU-Net, acvl_utils, batchgenerators, etc.
            self._pip_install_if_missing("nnunetv2", "nnInteractive")
        self._pip_install_if_missing("huggingface_hub", "huggingface_hub", force=force)

    def resolve_mode_first_run(self):
        """On first use, ask whether to use local GPU compute or a remote server."""
        if getattr(self, "_cleanup_in_progress", False) or getattr(
            self, "_application_quitting", False
        ):
            return
        existing = slicer.util.settingsValue("SlicerNNInteractive/mode", "")
        if existing in ("local", "remote"):
            self._sync_mode_switch()
            return
        if not self._local_inference_available():
            # Only the lightweight remote client is installed: Remote is the only viable
            # mode, so persist it and skip the prompt. (Install the full package and
            # restart to unlock Local — see the Local button's tooltip.)
            qt.QSettings().setValue("SlicerNNInteractive/mode", "remote")
            self._sync_mode_switch()
            return
        msg = qt.QMessageBox(slicer.util.mainWindow())
        msg.setWindowTitle("nnInteractive: choose compute mode")
        msg.setIcon(qt.QMessageBox.Question)
        msg.setText(
            "How would you like to run nnInteractive?\n\n"
            "Remote server  -  lightweight install (no PyTorch); connects to an "
            "nninteractive-server running on a GPU machine.\n\n"
            "Local GPU compute  -  runs in-process inside Slicer; downloads the full "
            "nnInteractive + PyTorch stack (needs a GPU)."
        )
        remote_btn = msg.addButton("Remote server", qt.QMessageBox.AcceptRole)
        local_btn = msg.addButton("Local GPU compute", qt.QMessageBox.AcceptRole)
        msg.exec_()
        mode = "local" if msg.clickedButton() == local_btn else "remote"
        qt.QSettings().setValue("SlicerNNInteractive/mode", mode)
        self._sync_mode_switch()

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

    def image_changed(self, do_prev_image_update=True):
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

        if do_prev_image_update:
            self.previous_states["image_data"] = fingerprint

        return image_changed

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
        self._update_prompt_cursor()  # recolour the active tool's cursor green
        self._retarget_scribble_if_active()  # re-point an active scribble at fg (green)
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
        self._update_prompt_cursor()  # recolour the active tool's cursor red
        self._retarget_scribble_if_active()  # re-point an active scribble at bg (red)
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
