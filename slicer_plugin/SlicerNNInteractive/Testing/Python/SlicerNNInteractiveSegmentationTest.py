import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import slicer
from SampleData import SampleDataLogic
from slicer.ScriptedLoadableModule import *


def positive(coords):
    return {"kind": "point", "coords": np.array(coords, dtype=int), "positive": True}


def negative(coords):
    return {"kind": "point", "coords": np.array(coords, dtype=int), "positive": False}


def bbox(coords_one, coords_two, positive=True):
    return {
        "kind": "bbox",
        "point_one": np.array(coords_one, dtype=int),
        "point_two": np.array(coords_two, dtype=int),
        "positive": bool(positive),
    }


def scribble(name, plane, slice_index, points, positive=True, thickness=1):
    return {
        "kind": "scribble",
        "mask_name": name,
        "plane": plane,
        "slice": int(slice_index),
        "points": [np.array(pt, dtype=float) for pt in points],
        "positive": bool(positive),
        "thickness": int(thickness),
    }


def lasso(name, plane, slice_index, points, positive=True):
    return {
        "kind": "lasso",
        "mask_name": name,
        "plane": plane,
        "slice": int(slice_index),
        "points": [np.array(pt, dtype=float) for pt in points],
        "positive": bool(positive),
    }


PLANE_CONFIGS = {
    "axial": {"slice_axis": 0, "coord_axes": (2, 1)},
    "coronal": {"slice_axis": 1, "coord_axes": (2, 0)},
    "sagittal": {"slice_axis": 2, "coord_axes": (1, 0)},
}


class SlicerNNInteractiveSegmentationTest(ScriptedLoadableModuleTest):
    PROMPTS = [
        ("tumor", [positive([128, 105, 89])]),
        ("brain", [positive([107, 127, 81])]),
        ("right_eye", [positive([108, 69, 41])]),
        ("left_eye", [positive([171, 67, 41])]),
        (
            "full_brain",
            [
                positive([141, 114, 85]),
                positive([109, 114, 58]),
                positive([177, 114, 38]),
            ],
        ),
        (
            "full_brain_with_negative",
            [
                positive([141, 114, 85]),
                positive([109, 114, 58]),
                positive([177, 114, 38]),
                negative([93, 114, 90]),
            ],
        ),
        ("tumor_bbox", [bbox([127, 114, 102], [159, 114, 73])]),
        ("brain_bbox", [bbox([127, 114, 102], [159, 114, 73])]),
        (
            "scribble_tumor",
            [
                scribble(
                    name="scribble_tumor",
                    plane="axial",
                    slice_index=82,
                    points=[
                        [143, 96],
                        [136, 106],
                        [148, 106],
                        [142, 118],
                    ],
                )
            ],
        ),
        (
            "scribble_brain",
            [
                scribble(
                    name="scribble_brain",
                    plane="coronal",
                    slice_index=137,
                    points=[
                        [79, 55],
                        [105, 87],
                        [159, 42],
                        [183, 81],
                    ],
                )
            ],
        ),
        (
            "lasso_tumor",
            [
                lasso(
                    name="lasso_tumor",
                    plane="sagittal",
                    slice_index=140,
                    points=[
                        [89, 92],
                        [92, 73],
                        [117, 76],
                        [123, 87],
                        [115, 98],
                        [99, 102],
                        [89, 92],
                    ],
                )
            ],
        ),
    ]

    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.generate_mode = os.environ.get("SLICER_NNI_GENERATE_TEST_MASK") == "1"
        # self.generate_mode = True
        self.test_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.test_dir / "Data"

    def runTest(self):
        self.setUp()
        try:
            volume_node = self._prepare_volume()
            widget = self._create_widget(volume_node)
            missing = [name for name, _ in self.PROMPTS if not self._reference_path(name).exists()]
            if missing and not self.generate_mode:
                self.fail(
                    "Missing reference masks for prompts: "
                    + ", ".join(missing)
                    + ". Run with SLICER_NNI_GENERATE_TEST_MASK=1 to generate them."
                )

            for prompt_name, sequence in self.PROMPTS:
                print(f"Testing prompt sequence '{prompt_name}'...", sequence)
                widget.clear_current_segment()
                mask = None
                for interaction in sequence:
                    if interaction["kind"] == "point":
                        mask = self._trigger_point_prompt(
                            widget, interaction["coords"], interaction["positive"]
                        )
                    elif interaction["kind"] == "bbox":
                        mask = self._trigger_bbox_prompt(
                            widget,
                            interaction["point_one"],
                            interaction["point_two"],
                            interaction["positive"],
                        )
                    elif interaction["kind"] == "scribble":
                        mask = self._trigger_scribble_prompt(widget, interaction)
                    elif interaction["kind"] == "lasso":
                        mask = self._trigger_lasso_prompt(widget, interaction)
                    else:
                        self.fail(f"Unsupported interaction kind '{interaction['kind']}'.")
                if self.generate_mode:
                    self._store_reference_mask(prompt_name, mask)
                else:
                    reference_mask = self._load_reference_mask(prompt_name)
                    self._verify_mask(
                        reference_mask,
                        mask,
                        prompt_name
                    )
        finally:
            self.tearDown()

        if not self.generate_mode:
            slicer.util.delayDisplay("All SlicerNNInteractive segmentation tests passed.")
            print("All SlicerNNInteractive segmentation tests passed.")

    def _prepare_volume(self):
        logic = SampleDataLogic()
        volume_node = logic.downloadMRBrainTumor2()
        slicer.app.processEvents()
        slicer.util.setSliceViewerLayers(background=volume_node)
        return volume_node

    def _create_widget(self, volume_node):
        slicer.util.selectModule("SlicerNNInteractive")
        widget = slicer.util.getModuleWidget("SlicerNNInteractive")
        segmentation_node = widget.get_segmentation_node()
        widget.ui.editor_widget.setMRMLSegmentEditorNode(widget.segment_editor_node)
        widget.ui.editor_widget.setSegmentationNode(segmentation_node)
        widget.ui.editor_widget.setSourceVolumeNode(volume_node)
        widget.make_new_segment()
        image_data = slicer.util.arrayFromVolume(volume_node).copy()
        widget.previous_states["image_data"] = image_data
        widget.previous_states["segment_data"] = np.zeros_like(image_data, dtype=np.uint8)
        self._ensure_server_is_ready(widget)
        self._upload_volume_before_tests(widget)
        return widget

    def _ensure_server_is_ready(self, widget):
        server_override = os.environ.get("SLICER_NNI_TEST_SERVER_URL", "").strip()
        if server_override:
            widget.server = server_override.rstrip("/")
            widget.ui.Server.setText(widget.server)
        if not getattr(widget, "server", ""):
            self.fail(
                "Server URL not configured. Set it in the Slicer settings or define "
                "SLICER_NNI_TEST_SERVER_URL before running tests."
            )

    def _upload_volume_before_tests(self, widget):
        # Uploading the current image to the nnInteractive server avoids requests failing
        # with "No image uploaded" during the scripted prompts.
        result = widget.upload_image_to_server()
        if result is None:
            self.fail(
                "Failed to upload the volume to the nnInteractive server. "
                "Verify the server is running and reachable."
            )

    def _trigger_point_prompt(self, widget, ijk, positive=True):
        dims = widget.get_image_data().shape  # (k, j, i)
        clamped = [
            int(np.clip(ijk[0], 0, dims[2] - 1)),
            int(np.clip(ijk[1], 0, dims[1] - 1)),
            int(np.clip(ijk[2], 0, dims[0] - 1)),
        ]
        widget.point_prompt(xyz=clamped, positive_click=positive)
        slicer.app.processEvents()
        segmentation_node, segment_id = widget.get_selected_segmentation_node_and_segment_id()
        labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, widget.get_volume_node()
        )
        return labelmap.astype(np.uint8)

    def _trigger_bbox_prompt(self, widget, point_one, point_two, positive=True):
        dims = widget.get_image_data().shape

        def clamp(pt):
            return [
                int(np.clip(pt[0], 0, dims[2] - 1)),
                int(np.clip(pt[1], 0, dims[1] - 1)),
                int(np.clip(pt[2], 0, dims[0] - 1)),
            ]

        widget.bbox_prompt(
            outer_point_one=clamp(point_one),
            outer_point_two=clamp(point_two),
            positive_click=positive,
        )
        slicer.app.processEvents()
        segmentation_node, segment_id = widget.get_selected_segmentation_node_and_segment_id()
        labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, widget.get_volume_node()
        )
        return labelmap.astype(np.uint8)

    def _trigger_scribble_prompt(self, widget, interaction):
        mask = self._build_scribble_mask(widget, interaction)
        self._save_scribble_mask(interaction["mask_name"], mask)
        widget.lasso_or_scribble_prompt(
            mask=mask,
            positive_click=interaction["positive"],
            tp="scribble",
        )
        slicer.app.processEvents()
        segmentation_node, segment_id = widget.get_selected_segmentation_node_and_segment_id()
        labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, widget.get_volume_node()
        )
        return labelmap.astype(np.uint8)

    def _build_scribble_mask(self, widget, interaction):
        dims = widget.get_image_data().shape  # (k, j, i)
        plane = interaction["plane"].lower()

        if plane not in PLANE_CONFIGS:
            self.fail(f"Unsupported scribble plane '{plane}'.")

        slice_axis = PLANE_CONFIGS[plane]["slice_axis"]
        coord_axes = PLANE_CONFIGS[plane]["coord_axes"]

        slice_index = int(np.clip(interaction["slice"], 0, dims[slice_axis] - 1))
        mask = np.zeros(dims, dtype=np.uint8)
        thickness = max(0, int(interaction.get("thickness", 1)))

        points = interaction["points"]
        if len(points) == 0:
            self.fail("Scribble interaction requires at least one point.")

        def clamp_value(value, axis_index):
            return int(np.clip(round(value), 0, dims[axis_index] - 1))

        def stamp(u, v):
            base_idx = [0, 0, 0]
            base_idx[slice_axis] = slice_index
            primary = clamp_value(u, coord_axes[0])
            secondary = clamp_value(v, coord_axes[1])
            for dv in range(-thickness, thickness + 1):
                sec = int(np.clip(secondary + dv, 0, dims[coord_axes[1]] - 1))
                for du in range(-thickness, thickness + 1):
                    prim = int(np.clip(primary + du, 0, dims[coord_axes[0]] - 1))
                    idx = list(base_idx)
                    idx[coord_axes[0]] = prim
                    idx[coord_axes[1]] = sec
                    mask[tuple(idx)] = 1

        if len(points) == 1:
            stamp(points[0][0], points[0][1])
        else:
            for start, end in zip(points[:-1], points[1:]):
                sx, sy = start
                ex, ey = end
                num = int(max(abs(ex - sx), abs(ey - sy)) + 1)
                if num <= 1:
                    stamp(sx, sy)
                    continue
                us = np.linspace(sx, ex, num)
                vs = np.linspace(sy, ey, num)
                for u, v in zip(us, vs):
                    stamp(u, v)

        return mask

    def _scribble_mask_path(self, mask_name):
        return self.data_dir / f"MRBrainTumor2_scribble_{mask_name}.nii.gz"

    def _save_scribble_mask(self, mask_name, mask):
        if not mask_name:
            return
        path = self._scribble_mask_path(mask_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        sitk.WriteImage(image, str(path), useCompression=True)

    def _trigger_lasso_prompt(self, widget, interaction):
        mask = self._build_lasso_mask(widget, interaction)
        self._save_lasso_mask(interaction["mask_name"], mask)
        widget.lasso_or_scribble_prompt(
            mask=mask,
            positive_click=interaction["positive"],
            tp="lasso",
        )
        slicer.app.processEvents()
        segmentation_node, segment_id = widget.get_selected_segmentation_node_and_segment_id()
        labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, widget.get_volume_node()
        )
        return labelmap.astype(np.uint8)

    def _build_lasso_mask(self, widget, interaction):
        dims = widget.get_image_data().shape
        plane = interaction["plane"].lower()

        if plane not in PLANE_CONFIGS:
            self.fail(f"Unsupported lasso plane '{plane}'.")

        slice_axis = PLANE_CONFIGS[plane]["slice_axis"]
        coord_axes = PLANE_CONFIGS[plane]["coord_axes"]

        slice_index = int(np.clip(interaction["slice"], 0, dims[slice_axis] - 1))
        mask = np.zeros(dims, dtype=np.uint8)

        points = interaction["points"]
        if len(points) < 3:
            self.fail("Lasso interaction requires at least three points.")

        axis_from_xyz = {0: 2, 1: 1, 2: 0}  # dims axis -> index in (x, y, z)
        processed_points = []
        for pt in points:
            arr = np.asarray(pt, dtype=float).flatten()
            if arr.size == 3:
                coord_lookup = {0: arr[axis_from_xyz[0]], 1: arr[axis_from_xyz[1]], 2: arr[axis_from_xyz[2]]}
                processed = np.array(
                    [coord_lookup[coord_axes[0]], coord_lookup[coord_axes[1]]], dtype=float
                )
            elif arr.size == 2:
                processed = np.array(arr, dtype=float)
            else:
                self.fail("Lasso points must be 2D plane coords or 3D (x, y, z) tuples.")
            processed_points.append(processed)

        polygon = np.vstack(processed_points)
        
        from matplotlib.path import Path as MplPath
        path = MplPath(polygon)

        grid_primary = np.arange(dims[coord_axes[0]])
        grid_secondary = np.arange(dims[coord_axes[1]])
        gp, gs = np.meshgrid(grid_primary, grid_secondary, indexing="ij")
        coords = np.stack([gp, gs], axis=-1).reshape(-1, 2)
        inside = path.contains_points(coords)
        filled = inside.reshape(len(grid_primary), len(grid_secondary))

        prim_idx, sec_idx = np.nonzero(filled)
        if prim_idx.size == 0:
            return mask

        indices = [None, None, None]
        indices[slice_axis] = np.full_like(prim_idx, slice_index)
        indices[coord_axes[0]] = prim_idx
        indices[coord_axes[1]] = sec_idx
        mask[tuple(indices)] = 1

        return mask

    def _lasso_mask_path(self, mask_name):
        return self.data_dir / f"MRBrainTumor2_lasso_{mask_name}.nii.gz"

    def _save_lasso_mask(self, mask_name, mask):
        if not mask_name:
            return
        path = self._lasso_mask_path(mask_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        sitk.WriteImage(image, str(path), useCompression=True)

    def _reference_path(self, prompt_name):
        out = self.data_dir / f"MRBrainTumor2_point_prompt_{prompt_name}.nii.gz"
        return out

    def _store_reference_mask(self, prompt_name, mask):
        path = self._reference_path(prompt_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        sitk.WriteImage(image, str(path), useCompression=True)
        slicer.util.delayDisplay(
            f"Stored new reference mask for '{prompt_name}' at {path}. Inspect visually, then rerun without SLICER_NNI_GENERATE_TEST_MASK."
        )

    def _load_reference_mask(self, prompt_name):
        path = self._reference_path(prompt_name)
        if not path.exists():
            self.fail(
                f"Reference mask for '{prompt_name}' not found at {path}. "
                "Run once with SLICER_NNI_GENERATE_TEST_MASK=1 to populate it."
            )
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image).astype(np.uint8)

    def _verify_mask(self, reference_mask, result_mask, prompt_name, save_debug=False):
        if save_debug:
            # Write masks as sitk for debug
            reference_mask_sitk = sitk.GetImageFromArray(reference_mask.astype(np.uint8))
            result_mask_sitk = sitk.GetImageFromArray(result_mask.astype(np.uint8))
            debug_dir = self.test_dir / "DebugMasks"
            debug_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                reference_mask_sitk,
                str(debug_dir / f"reference_mask_{prompt_name}.nii.gz"),
                useCompression=True,
            )
            sitk.WriteImage(
                result_mask_sitk,
                str(debug_dir / f"result_mask_{prompt_name}.nii.gz"),
                useCompression=True,
            )
        
        self.assertEqual(reference_mask.shape, result_mask.shape)
        self.assertGreater(result_mask.sum(), 0)
        
        def get_dice():
            reference_mask_bool = reference_mask.astype(bool)
            result_mask_bool = result_mask.astype(bool)
            intersection = np.count_nonzero(reference_mask_bool & result_mask_bool)
            total = reference_mask_bool.sum() + result_mask_bool.sum()
            dice = 1.0 if total == 0 else (2.0 * intersection) / total
            return dice
        
        dice = get_dice()
        print(f'Dice score {prompt_name}: {dice:.4f}')
        
        dice_threshold = 0.99
        try:
            self.assertEqual(reference_mask.shape, result_mask.shape)
            self.assertGreater(result_mask.sum(), 0)
            self.assertGreaterEqual(
                dice,
                dice_threshold,
                msg=(
                    f"Segmentation mismatch for prompt '{prompt_name}'. "
                    f"Dice score {dice:.4f} below threshold {dice_threshold}."
                ),
            )
        except AssertionError:
            print(f"[FAIL] {prompt_name}")
            raise
        print(f"[PASS] {prompt_name}")

    def _describe_prompt_sequence(self, prompt_sequence):
        if not prompt_sequence:
            return "no interactions"

        def as_list(value):
            if isinstance(value, np.ndarray):
                return value.astype(float).tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]

        descriptions = []
        for interaction in prompt_sequence:
            kind = interaction.get("kind", "unknown")
            positive = interaction.get("positive")
            sign = ""
            if positive is not None:
                sign = "positive" if positive else "negative"
            extra = ""
            if kind == "point":
                coords = as_list(interaction.get("coords", []))
                extra = f"coords={coords}"
            elif kind == "bbox":
                p1 = as_list(interaction.get("point_one", []))
                p2 = as_list(interaction.get("point_two", []))
                extra = f"p1={p1}, p2={p2}"
            elif kind in ("scribble", "lasso"):
                plane = interaction.get("plane", "?")
                slice_index = interaction.get("slice", "?")
                count = len(interaction.get("points", []))
                extra = f"{plane} slice={slice_index}, points={count}"
            descriptions.append(
                f"{kind} {sign}".strip() + (f" ({extra})" if extra else "")
            )
        return "; ".join(descriptions)


SlicerNNInteractiveTest = SlicerNNInteractiveSegmentationTest
