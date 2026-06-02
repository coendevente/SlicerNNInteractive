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

### Lasso is a single-slice prompt

The lasso (ClosedCurve) prompt is rasterized on **one slice** of the
inference/source volume. Curve points are converted to integer voxel
coordinates (`ras_to_xyz` rounds), so a slice sitting near a voxel boundary
used to scatter the points across two adjacent slices, fail the
"exactly one constant axis" check, and silently clear the lasso with no
segmentation -- experienced as "drew a circle, right-clicked, nothing
happened". `lasso_points_to_mask` now picks the flattest axis and snaps it to
a single slice when its spread is within `LASSO_SLICE_AXIS_MAX_SPREAD` voxels
(2 by default); only genuinely oblique or multi-slice curves (e.g. drawn on a
display plane that is not aligned with the inference/source volume) are
rejected, with a status-bar message instead of a silent no-op. The other
lasso/scribble early-exits (too few points, no source volume, empty mask,
sync failure, non-200 / network error) now also show a status-bar message.

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

## Supplemental-series auto-registration

The whole multi-plane / native-series design assumes every participating volume
shares one patient coordinate system (RAS). That holds only when the series
share a DICOM `FrameOfReferenceUID`. A mid-exam localizer/scout, repositioning,
or patient motion produces a new frame of reference, so two series of the "same
patient, same examination" can be silently misaligned in RAS -- masks then land
on the wrong anatomy with no error.

To guard against this, enabling `Analyze a supplemental series` (or switching
the working volume) compares the working volume's `FrameOfReferenceUID`
(0020,0052) against the source volume's:

- equal -> already aligned, nothing to do;
- different and `Auto-register supplemental series to source` is checked ->
  an asynchronous rigid BRAINSFit registration (moving = working volume, fixed =
  source volume) runs and its linear transform is attached as the working
  volume's parent transform;
- frame of reference unreadable (e.g. non-DICOM import) -> no automatic
  registration; the panel warns and the user can either tick `I confirm these
  series are already aligned (skip registration)` or click `Register now`.

Attaching the transform is the only alignment action needed: `ras_to_xyz` and
`_resample_mask_between_volumes` already follow a volume's parent transform, so
prompt coordinates, the source-mask resample onto the working grid, and the
result resample back all become correct automatically. The source volume and
the high-resolution output grid are never transformed.

The same machinery aligns the three multi-plane display backgrounds. Clicking
`Apply view volumes` runs `_align_display_volumes`, which registers each unique
non-source display volume to the source so a sharper background series drawn
from a different frame of reference is still shown at the correct physical
location. Because registration is asynchronous and the slice view follows the
volume's parent transform live, backgrounds re-place themselves once each
registration finishes. Up to three registrations are serialized through a small
FIFO queue (`_alignment_queue` / `_pump_alignment_queue`) since BRAINSFit runs
one CLI at a time.

Transforms are cached per `(moving, source)` pair and reused on re-apply /
re-enable. A registration whose result is within
`REGISTRATION_IDENTITY_TRANSLATION_MM` and `REGISTRATION_IDENTITY_ROTATION_DEG`
is treated as identity (series already aligned despite differing UIDs) and the
transform is discarded to avoid needless resampling. On success the plugin turns
on slice intersections so the user can verify overlap visually. Transforms stay
attached (volumes remain in their correct physical position) until they are
explicitly cleared via `Clear alignment`, invalidated when the source volume
changes (`_prune_alignment_for_source`), or removed on module cleanup. While a
registration runs, prompts and `Sync preview to source` are blocked so nothing
is sent on stale geometry.

UI controls (under `Native-series inference`): `Auto-register supplemental
series to source` (on by default), `I confirm these series are already aligned`,
`Registration` mode (`Rigid` default / `Affine` escape hatch), `Register now`,
`Clear alignment`, and a status label reporting the outcome plus the registered
translation and rotation magnitudes (flagged for review above
`REGISTRATION_OFFSET_WARN_MM`).

Open follow-up: the BRAINSFit output transform direction (moving->fixed) must be
confirmed against real data and inverted in `_attach_alignment_transform` if a
test shows the moving volume moving the wrong way.

## Preconditions and limitations

Auto-registration relies on Slicer's BRAINSFit module being available and on the
two volumes overlapping enough for mutual-information registration to converge.
When the DICOM frame of reference cannot be read (non-DICOM imports), no
registration is attempted automatically; the user must confirm alignment or
click `Register now`. Only standard Red/Yellow/Green slice orientations are
assumed; oblique acquisitions may still be resampled by the slice viewer.

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

**Important expectation**: by itself this is the foundation, not a magic
sharpener. A single inference on the coarse source volume is still coarse;
nearest-neighbor resampling to a fine grid only yields smaller stair-steps.
Per-plane sharpness comes either from **smooth interpolation** (below) or from
running native-series inference on each high-resolution series and merging the
results (default `Add`), which land on the shared fine grid at full detail.

### Smooth (interpolated) results

`Smooth (interpolate) result between slices` interpolates each coarse server
segmentation result onto the fine output grid so the boundary is smooth in all
three planes instead of stair-stepped. Because smoothing needs the fine grid to
land on, it requires the high-resolution output feature; enabling smoothing
turns that on automatically, and turning the high-resolution grid off turns
smoothing off in lockstep.

- Method: **shape-based (signed-distance) interpolation** in
  `_interpolate_mask_to_output_grid`. The coarse mask's signed distance field
  (mm, via `scipy.ndimage.distance_transform_edt`) is interpolated to the fine
  grid (`ndimage.zoom`, linear) and thresholded at zero. This reconstructs a
  smooth surface between the thick source slices -- it genuinely interpolates
  cross-sections, not just rounds the existing steps. It is valid because the
  output grid is built coplanar with the source (same origin/axes, finer
  spacing), so coarse -> fine is a pure per-axis scale.
- Scope: only the nnInteractive server results that flow through
  `_handle_server_segmentation_result` (point / bbox / lasso / scribble) and the
  native-series preview are smoothed. Manual boolean / Magic Wand operands are
  left exact so deliberate edits are not blurred.
- Supplemental (native-series) working volumes are not coplanar with the output
  grid, so that path falls back to nearest-neighbor resampling followed by a
  Gaussian smoothing pass (rounds steps; lower quality than true SDF interp).
- Strength is a fixed sensible default (no UI control): the SDF zero-crossing
  needs no parameter, and the Gaussian fallback sigma is derived from the source
  slice thickness relative to the isotropic spacing.
- Any failure (no scipy, oversized grid, etc.) returns `None` and the caller
  degrades to the plain resampling path, so a result is never dropped.
- Enable state persists via `QSettings`
  (`SlicerNNInteractive/smooth_interpolate_enabled`), but is only honored at
  startup when high-resolution output is also enabled.
- Manual edits (the built-in Segment Editor Erase / Paint / Scissors effects)
  write the labelmap directly and never pass through the server chokepoint, so
  the auto toggle does not touch them. The `Smooth current segment` button
  re-runs the same SDF smoothing on the whole current segment on demand -- use
  it after erasing or other manual edits. It requires the high-resolution grid
  (clicking enables it if needed) and writes through `show_segmentation`, so it
  is undoable from the Segment Editor.

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
