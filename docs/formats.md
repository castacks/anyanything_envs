# Formats

All coordinates are metres, z up. A primitive patch's frame has x forward (the nominal start sits at the
origin flying +x), the floor at z = 0 and the nominal flight altitude `flight_z` (1.75 m).

## `spec.json`: a primitive patch

Written by `envgen.primitives.save_spec` (`PatchSpec.to_json()`), read by `blender/build_course.py`.

| Key | Type | Meaning |
|---|---|---|
| `test` | int | 5 for every current family (3 and 4 are the earlier fixed-goal series) |
| `family` | str | tier-A family name, `corridor` (tier B) or `arena` (tier C) |
| `seed`, `split` | int, str | the draw; the same family, seed and split give the same patch on any machine |
| `bounds` | [x0, y0, x1, y1] | the patch's extent; the floor grid covers it |
| `flight_z` | float | nominal altitude |
| `params` | dict | the drawn family parameters (widths, offsets, ...), plus `face_spans` / `faces` / `far_faces` / `face_records`: the obstructions in order along x |
| `face_x`, `far_x` | float | near and far x of the (first) obstruction |
| `goal_pos`, `goal_speed`, `goal_radius` | | the nominal goal the patch was drawn for |
| `primitives` | list of dicts | the geometry, see below |

Each primitive is a dict with a `kind`, its geometry and a `region` (the family that placed it):

| `kind` | Geometry keys |
|---|---|
| `box` | `c` centre [x, y, z], `half` half-extents, `rpy` roll/pitch/yaw (rad) |
| `cyl_v` | `c` [x, y], `r` radius, `z0`, `z1` bottom and top |
| `cyl_h` | `a`, `b` end points, `r` radius (bars, wires and branch segments are chains of these) |
| `sphere` | `c` centre, `r` radius |
| `wall` | `p0`, `p1` end points on the floor, `t` thickness, `z0`, `z1`, `slots` (openings cut through it) |

`envgen.primitives.arena` builds and samples them (`box`, `cyl_v`, `cyl_h`, `sphere`, `sample_surfaces`).

## `cloud.npz`: the collision cloud

| Array | dtype | Meaning |
|---|---|---|
| `points` | float64 [N, 3] | obstacle surface samples, one per 10 cm voxel, then the floor grid |
| `region` | int16 [N] | 0 and up for obstacles, -1 for the floor grid (25 cm spacing over `bounds`) |

`envgen.primitives.patch_cloud(spec)` samples the primitives themselves. A Blender course samples every
collidable triangle of the built scene instead (`build_course.py`): alpha-cut surfaces only where alpha reaches
0.5, nothing above `CLOUD_ZMAX` (4.75 m). The same spec, arguments, code and asset library give the same
`cloud.npz` byte for byte.

## `course.json`: what a Blender course contains

Placements, materials, HDRI and exposure, counts, the kit's choices (`chosen`), its changes to the patch
geometry (`semantics_changed`, one record per changed primitive), the cloud's statistics and SHA-256, packing
statistics, timings, and provenance: `git_sha`, `dirty`, `kits_order`, `library_sha`, Blender version and argv
(`blender/provenance.py`).

## Infinigen flight boxes

`infinigen/nature_box.py` writes `<out>/box.json` (seed, scene type, box half-width, margin, terrain bounds,
densities, gin overrides, seconds) and `<out>/scene/scene.blend` with Infinigen's assets folder.
`infinigen/box_views.py` writes into `<out>/views/`:

| File | Content |
|---|---|
| `cloud.npz` | `points` float32 (10 cm voxel centres of every render-visible mesh and instance in the populated square, ground litter under 0.3 m left out), `label` int16 per point with `names` (json: asset family to label), `agl` height over the local ground, `ground_grid` and `grid_x` (ground height every 0.5 m) |
| `band.json` | occupancy per height slice over the ground (`slices`: 0.4-5.5, 0.4-1, 1-3, 3-5.5 m), with blocked share, share of cells at least 1 m clear and median clearance; canopy cover and base; ground relief; asset counts inside the box |
| `band_occ.npy` | the 1-3 m slice's occupancy grid (25 cm cells) |
| `views.json` | camera poses (position, yaw, height over ground), render timings, camera model (640 x 480, 87 deg), backdrop |
| `v<k>_<look>.png`, `v<k>_depth.npy`, `top.png` | drone-eye renders per look (A0 midday, A1 low sun ahead, A2 haze, A3 dusk), metric z-depth (float16, the haze volume hidden), top-down orthographic render |
