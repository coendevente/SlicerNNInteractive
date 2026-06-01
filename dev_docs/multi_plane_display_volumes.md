# Multi-plane display volumes

## Goal

For one patient and one examination, multiple DICOM series may be acquired
with different slice directions. A single series can look sharp in its native
plane but blurry in the other two planes because those views are interpolated.

The client plugin now allows the standard Slicer 2D views to use different
registered scalar volumes as backgrounds:

| Slicer view | Conventional orientation | Intended display volume |
| --- | --- | --- |
| Red | Axial | Axial acquisition |
| Yellow | Sagittal | Sagittal acquisition |
| Green | Coronal | Coronal acquisition |

## Design

This feature deliberately separates:

- **Segmentation source volume**: the volume selected in the embedded Segment
  Editor. Its voxel grid is the *default* canonical output geometry for saved
  segments. This default can be overridden by the high-resolution output
  geometry feature (see "High-resolution output geometry" below).
- **Display volumes**: optional per-view backgrounds used only to improve 2D
  image clarity.
- **Inference working volume**: an optional registered supplemental volume
  uploaded to nnInteractive while native-series inference is enabled.

The implementation changes each standard slice view's
`vtkMRMLSliceCompositeNode` background volume ID. It does not change the
Segment Editor source volume, segmentation reference geometry, server session,
or HTTP protocol.

Markup prompts are stored in Slicer world coordinates (RAS). Existing prompt
conversion maps those world coordinates back into the segmentation source
volume before sending voxel coordinates to the server. Therefore, different
display voxel spacing is supported when the volumes are correctly aligned in
physical space.

## Display-only interaction behavior

The initial implementation supports using a supplemental display volume as a
clearer visual reference while interacting with the current segmentation:

- Point, box, and lasso markups are placed in Slicer world coordinates (RAS)
  and mapped into the Segment Editor source volume before they are sent to the
  nnInteractive server.
- Magic Wand seeds also use world coordinates and are mapped into the source
  volume.
- Scribble and Lasso (3D) use hidden Segment Editor widgets whose source volume
  remains the main segmentation volume. The slice view still provides the
  world-space interaction plane.
- Returned masks are written into the source volume geometry and appear as an
  overlay in all registered display volumes.

This means that users can already inspect a sharper supplemental series,
interact in its slice view, and see the result synchronized to the main
segmentation. However, the nnInteractive model still receives only the main
source image.

## Native-series inference mode

When nnInteractive must analyze the sharper supplemental acquisition, enable
native-series inference and select a separate **inference working volume**. The
fixed main segmentation source volume is not replaced.

Implemented flow:

1. Keep one fixed main segmentation source volume as the canonical output grid.
2. Let the user choose a registered scalar volume as the inference working
   volume.
3. Resample the current main mask onto the inference working grid with nearest
   neighbor interpolation.
4. Upload the working image and resampled mask to the stateful server. Switching
   working volume resets the server interaction history.
5. Convert world-space prompts into the working volume's IJK coordinates.
6. Receive the result mask in the working grid.
7. Resample the binary result mask back onto the main grid with nearest neighbor
   interpolation.
8. Display the source-grid result in a hidden preview segmentation. Do not
   modify the source segment yet.
9. When the user confirms, merge the preview into the main segment using an
   explicit operation: add, replace, subtract, or intersect.

Slicer's `arrayFromSegmentBinaryLabelmap` and
`updateSegmentBinaryLabelmapFromArray` helpers accept a reference volume that
defines origin, spacing, axis directions, and extents. A temporary segmentation
node can therefore be used as a safe staging area while resampling between the
working and main grids.

The default merge mode is **Add to source** so that synchronized results do not
discard useful edits made from another plane. The other modes remain available
for deliberate replacement or boolean editing.

Point, box, and lasso prompts are rasterized on the inference working grid.
Scribble and Lasso (3D) keep their hidden Segment Editor widgets on the main
grid, then resample the generated mask to the working grid before calling the
server. Magic Wand also maps world-space seeds into the working grid. These
paths return source-grid masks before editing or previewing the canonical
segment.

## UI

The `Segment Editor` section contains a new optional `Multi-plane display`
group with:

- `Red (Axial)`
- `Yellow (Sagittal)`
- `Green (Coronal)`
- `Apply view volumes`
- `Use source volume for all`

When applying, an empty selector falls back to the current Segment Editor
source volume. Reset clears all three selectors and restores the source volume
as every standard view's background.

Clicking `Apply view volumes` also enables a sticky per-plane display override.
Hidden Segment Editor effects such as Scribble and Lasso (3D) call
`setSourceVolumeNode` while activating, which resets every slice background to
the source volume. The plugin schedules a silent reapply right after those
activation calls so Red, Yellow, and Green return to the configured display
volumes. Clicking `Use source volume for all` disables this override.

The sticky reapply follows the selections that were locked in *when Apply was
clicked* (a snapshot of the three selector node IDs), not the live selector
values. Changing a selector without clicking Apply again therefore does not
leak into the reapply. If a snapshotted display volume has been removed from
the scene, that view falls back to the current source volume.

The nested `Native-series inference` section contains:

- `Analyze a supplemental series`
- `Working volume`
- `Sync mode`: Add, Replace, Subtract, or Intersect
- `Sync preview to source`
- `Clear preview`

Changing the enabled state or working volume discards any stale preview and
forces the next prompt to upload the correct image and source-grid mask.
Clicking `Clear preview` also restores the current source mask as the server
target so the next prompt does not continue from a discarded candidate.

## Preconditions and limitations

The three display volumes must be aligned in patient space. Being from the same
patient and examination is helpful but does not guarantee exact alignment:
patient motion, differing DICOM orientation metadata, and scanner geometry can
still cause offsets. Confirm alignment visually with slice intersections or
apply a Slicer registration transform before segmentation.

This first version assumes the standard Red, Yellow, and Green slice
orientations are suitable. Oblique acquisitions may still be resampled by the
slice viewer. Supporting native oblique planes would require an additional
feature that changes slice-node orientation.

Display selection is scene-local and is not persisted across Slicer sessions.

## Manual verification

1. Load axial, sagittal, and coronal series for the same examination.
2. Select the intended segmentation source volume in Segment Editor.
3. Select the three registered display volumes and click `Apply view volumes`.
4. Confirm that Red, Yellow, and Green show the expected sharper acquisitions.
5. Enable slice intersections and verify that anatomy aligns around the same
   physical location.
6. Place point, box, lasso, and scribble prompts in each view and confirm that
   masks remain aligned with anatomy.
7. Click `Use source volume for all` and confirm that all three views return to
   the segmentation source volume.
8. Enable `Analyze a supplemental series`, select a registered working volume,
   and place a prompt.
9. Confirm that the preview overlay appears without editing the source segment.
10. Click `Sync preview to source` with the default Add mode and confirm that
    the preview is merged into the source segment.
11. Repeat with Replace, Subtract, and Intersect when validating release
    behavior.

## High-resolution output geometry

By default the canonical segmentation output grid is the (often anisotropic)
Segment Editor source volume, so a stored mask is fine in the source's native
plane but coarse and stair-stepped in the other two planes. The
`High-resolution output (optional)` group decouples the output grid from the
source volume:

- `Store masks on a high-resolution isotropic grid` enables the feature.
- `Isotropic spacing (mm, 0 = auto)`: the target isotropic voxel size; `0`
  uses the finest source spacing. Clamped to `[0.3, 10.0]` mm and coarsened
  automatically if the resulting grid would exceed a voxel budget.

When enabled, a hidden, source-aligned, geometry-only scalar volume
(`NNInteractiveOutputGeometry (do not touch)`) is created with isotropic
spacing covering the source field of view. The segment is stored and resampled
on this grid:

- `get_output_volume_node()` returns this grid (or the source volume when the
  feature is off -- a full no-op, backward compatible).
- The three mask-geometry call sites use it: the segmentation reference
  geometry, the `show_segmentation` write reference, and the `get_segment_data`
  default reference.
- Server results are resampled from the inference grid onto the output grid at
  the single chokepoint `_handle_server_segmentation_result`; native-series
  results are resampled working -> output (preserving high-resolution detail
  instead of collapsing to the coarse source first).
- Selection-operation operands (ROI, magic wand, segment, lasso 3D) are
  rasterized on the source grid and bridged to the output grid via
  `_to_output_grid` inside `apply_boolean_operation`.
- The server upload still samples the segment on the inference grid, so the
  round-trip (write output / upload inference / read output) closes.

**Important expectation**: this is the foundation, not a magic sharpener. A
single inference on the coarse source volume is still coarse; resampling it to a
fine grid only yields smaller stair-steps. True per-plane sharpness comes from
running native-series inference on each high-resolution series and merging the
results (default `Add`), which now land on the shared fine grid at full detail.

Caveats:

- Toggling the feature or changing the spacing re-derives the *current* segment
  onto the new grid; other segments are not auto-migrated.
- Lasso slice-range clipping is skipped while this feature is active (the
  recorded slice index is in source voxels and does not map to the output grid).
- Resampling uses Slicer's segmentation conversion (nearest-neighbor-like), so
  coarse -> fine adds stair-stepping and the fine -> coarse upload loses detail.
- A fine isotropic grid over a large field of view is memory-heavy; the build
  warns and/or coarsens the spacing to stay within the voxel budget. If Slicer
  still cannot resample a mask onto the grid, the feature auto-reverts to the
  source grid (the current result/segment is kept) and a status message is
  shown, rather than failing the operation.
- An empty segment is never exported through the segmentation resampler (it
  would fail); empty masks short-circuit to a zero mask on the target grid.
- Enable state and spacing persist across sessions via `QSettings`
  (`SlicerNNInteractive/high_res_output_enabled`, `.../output_spacing`).
