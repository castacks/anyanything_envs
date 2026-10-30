# Infinigen flight boxes

Infinigen builds a nature scene around one camera: it only turns placeholders into real assets, and only
scatters ground cover, where that camera can see. A flight test needs the whole area real from every pose.
`infinigen/nature_box.py` runs Infinigen's own `generate_nature` tasks (coarse, populate, fine_terrain)
unchanged except for these replacements, all made at run time:

| Infinigen default | Flight box |
|---|---|
| "in view of the camera" decides what is populated | a square crop: everything within `--half` + `--margin` of the origin, at a fixed design distance (`--design-dist`, which sets asset detail) |
| camera-adaptive terrain mesh (OcMesher / spherical) | `UniformMesher` over the square at `--voxel` (0.15 m) |
| every tree and bush grown individually | `--unique-trees` (12) per species, the rest linked copies with random yaw and +-10 % scale |
| ground-cover scatters live in the viewport | the `scatters` collection is kept out of the viewport depsgraph (see below) |
| the near-camera collider for falling-leaf particles | kept, but not evaluated in the viewport |

## Install (Linux, CPU)

```bash
git clone https://github.com/princeton-vl/infinigen && cd infinigen
git submodule update --init --depth 1        # if gitlab's eigen fails, see below
uv venv --python 3.11 --seed ../venv && . ../venv/bin/activate
INFINIGEN_MINIMAL_INSTALL=False INFINIGEN_INSTALL_TERRAIN=True pip install -e ".[terrain]"
```

Tested at Infinigen `224598d` (2026-10-14) with bpy 4.2. `eigen` is only needed for the OpenGL ground-truth
tool, which we do not build: if its gitlab fetch fails, fetch the other submodules one by one and put any file
into `src/infinigen/datagen/customgt/dependencies/eigen` so the installer does not try again. Keep pip's and
uv's caches and `TMPDIR` on a disk with room (the bpy wheel alone is ~350 MB).

## Run

```bash
python infinigen/nature_box.py --seed 7 --out out/plain_7 --scene-type plain --half 15 --margin 10 \
    --design-dist 8 --no-monocots --configs fast_terrain_assets.gin
python infinigen/box_views.py --scene out/plain_7/scene/scene.blend --box out/plain_7/box.json \
    --out out/plain_7/views --n-views 3 --looks A0 A1 A2 A3 --samples 32 --threads 30
python infinigen/page/make_sheet.py --scenes out/plain_7 --out out/page
python infinigen/page/make_page.py --assets out/page
```

Extra obstacles, Wenshan-style floaters and grounded clutter, are added after generation, so one base scene
gives a whole density ladder:

```bash
python infinigen/scatter_obstacles.py --scene out/plain_7/scene/scene.blend --box out/plain_7/box.json \
    --cloud out/plain_7/views/cloud.npz --out out/plain_7/floaters_80 --floaters 80 --ground 20
python infinigen/box_views.py --scene out/plain_7/floaters_80/scene.blend --box out/plain_7/box.json \
    --out out/plain_7/floaters_80/views --n-views 3 --samples 32
```

They come from Infinigen's own procedural factories (chairs, sofas, tables, lamps, TVs, microwaves,
dishwashers, bookcases, rocks, boulders, cacti), so none is an asset our training data has seen; the floating
placement itself mirrors Wenshan's scenes, which are in our training data, so **the out-of-distribution test
set does not use floaters** (decision 2026-10-29); the script stays for other uses. Placement rejects anything within an object's bounding radius plus `--gap` of the scene's cloud or of
another placement. 80 floaters + 20 grounded took 12.5 min CPU (36 prototypes spawned, a 3 GB scene saved).

`--water type` keeps the scene type's water body (river, coast, arctic); otherwise ponds are off.
`box_views.py --wires` adds thin structures: a pole line with two sagging wires and guy wires across the box,
and ropes between trunks. `--depth-only` re-renders the depth of existing views.

## What it costs (gs2: 32 cores, 62 GB, no GPU)

| Step | Time |
|---|---|
| generate one 50 x 50 m populated square | 26-35 min CPU (terrain 3 min, placeholders 2-4, ground cover 10-15, trees 3-13, fine terrain 1) |
| collision cloud and band map | ~2.5 min |
| one 640 x 480 drone frame, CPU Cycles, 30 threads | 175-225 s at 32 spp (leaf instances and the haze volume dominate) |

At design distance 4 m a scene with two tree species came to 11 GB and needed 53 GB to render, more than
gs2 has; use 8 m or more.

## Traps we hit (all handled in the scripts)

1. **Viewport re-evaluation.** Every operator call during asset generation updates the view layer, which
   re-evaluated the ground-cover scatters; over the whole box that made one mushroom model take 4 min. The
   scatters collection is now hidden from the viewport. `scatter_instances` is imported by name in 20 modules,
   so it is rebound in each.
2. **Monocots** still take ~27 min per model in this setup: `--no-monocots`.
3. **Trees grown one by one** take hours for a dense forest: instanced after 12 unique ones per species.
4. **Geometry-nodes instances** (twigs, leaves, rocks) come from render-hidden prototype collections. Judge an
   instance by its instancer, never by its source, or the canopy goes missing from the cloud.
5. **Water** is meshed coarsely by the uniform mesher and renders as mirror shards. `box_views.py` hides it
   unless `--keep-water`.
6. **Haze in the depth pass.** Infinigen's atmosphere volume scatters some rays in a 1-sample depth pass
   (speckle). The depth render hides volumes.
7. **Nothing beyond the square.** The populated square ends at `--half + --margin`. `box_views.py` adds a
   backdrop (a terrain skirt continuing the edge heights, instanced copies of the scene's own trees at the
   measured density, a dark tone below the horizon); it is outside the collision cloud.
8. **Near-terrain subdivision (what ran mountain out of memory).** Infinigen subdivides its near-camera terrain
   twice (`SUBSURF` levels 2) for the ground cover; with the box crop that is the whole populated square, already
   meshed at `--voxel`. OpenSubdiv's setup costs ~53 KB per face of that mesh whatever the level (measured:
   50k / 100k / 200k faces take 2.6 / 5.2 / 10.6 GB, level 1 as much as level 2); at that rate the step costs
   ~13 GB on plain's square (245k faces) and ~29 GB on cliff's (553k). mountain seed 105 has a floating
   "upside-down mountain" over 70 % of the box: 1.09 M faces in the square (3.4x the flat area), 64 % of them on
   the floating mountain, and the step passed 45 GB and was OOM-killed. Above `--near-subsurf-max-faces` (600k,
   above every type that finished) the near terrain stays as meshed; at 0.15 m it is already finer than
   Infinigen's own subdivided one (its coarse mesh is 0.8 m). `box.json` records `near_terrain`. mountain 105
   then generates in 24.5 min at a 10.8 GB peak (the chopped-tree booleans; the crop step itself 4.2 GB).
   snowy_mountain 108 is steep but a heightfield (423k faces): it keeps the subdivision, 8.1 min at a 23.9 GB
   peak, all of it this step; pass a lower limit to trade that for an unsubdivided near terrain.
9. **Camera rules a crop cannot meet.** Infinigen still poses its camera before anything is populated, and
   snowy_mountain asks for every visible terrain point to be at least 40 m away. Inside the square that almost
   never happens (0 of 3000 proposals on mountain's square; snowy_mountain 108 found 8 of the 100 candidates
   Infinigen wants in 7.5 min, slowing to 87 s each). `nature_box.py` binds
   `keep_cam_pose_proposal.min_terrain_distance=2`, base.gin's value and so a no-op for every other type; the
   camera no longer decides what is populated, and every stage reseeds.
10. **Low-poly terrain up close.** The UniformMesher's faces are flat-shaded, so at drone height its 0.15 m facets
    show on sand and slopes (Infinigen's OcMesher hides them by refining near its camera). `nature_box.py` shades
    the fine terrain smooth before Infinigen saves it, keeping edges sharper than `--smooth-angle` (35 deg) sharp:
    shading everything smooth makes the voronoi rocks, which are part of the terrain mesh, look melted, and the
    angle keeps their edges. Shading only, so the cloud is unchanged: plain seed 101 regenerated with it (and with
    the cave, mountain and indoor changes merged) gives a byte-identical `cloud.npz` to the run from before them.
    `box_views.py` applies the same at render time (`--smooth-angle`, 0 renders the terrain as saved), so scenes
    generated before this render the same way.
11. **Every frame re-syncs the whole scene.** Infinigen's cloud shader carries a driver (`frame * k`, drifting
    clouds) and the grass and fern scatters read Scene Time (wind). Blender re-evaluates both on every render,
    persistent data or not, even when only the camera moves: every shader is recompiled and the cloud volume and the
    scene BVH are rebuilt for each frame, 10 s of a 14 s frame at 1 spp (plain 101, gs2 CPU, 8 threads). Baking
    them at the current frame (`freeze_time` in `tools/bl52_frame.py`, on with `--bench`) leaves 3.9 s: 1.9 s of
    path tracing and 1.9 s of Blender's sync of the 227k instances, which persistent data cannot skip. Frame-1
    renders look the same either way. A multi-frame renderer (flights) needs the same freeze: the closed-loop
    bench's `freeze_scene` clears object animation only, and these scenes have none.

## Scene types

Tested: forest. Running in the first smoke batch: plain, river, canyon, cliff, mountain, desert, arctic,
snowy_mountain, coast. **cave**: one smoke scene, see "Caves" below. **Infinigen-Indoors** is a different
generator, see "Indoors" below. Underwater types (kelp_forest, coral_reef, under_water) are not useful for drones.

**Overhangs (mountain).** mountain draws Infinigen's floating "upside-down mountains" in 40 % of seeds. Seed 105
has one over 70 % of the box, its underside 13-20 m up (5-18 m over the ground). `box_views.py` finds the ground
by casting down from above and leaves the terrain out of the cloud, so under it the ground grid is the island's
top (median 24 m, one of the two smoke views stands on it) and the band map describes that top, as in a cave.
Until `box_views.py` searches for the floor, keep them out of flight boxes with
`-p scene.upsidedown_mountains_chance=0`.

## Caves

```bash
python infinigen/nature_box.py --seed 7 --out out/cave_7 --scene-type cave --half 15 --margin 10 \
    --design-dist 8 --no-monocots --tree-density 0 --configs fast_terrain_assets.gin
python infinigen/box_views.py --scene out/cave_7/scene/scene.blend --box out/cave_7/box.json \
    --out out/cave_7/views --n-views 2 --looks A0 --samples 32 --threads 30
```

The drone flies inside the cave. Both scripts switch on `scene_type == "cave"` (box.json carries it); nothing
changes for the other types.

**Where Infinigen puts the cave.** `cave.gin` sets `scene.caves_chance = 1`. `Caves` is not a terrain element of
its own but an operation on every one of them (ground, land tiles, rocks): a smooth subtraction of the cave's SDF
(`terrain/source/common/elements/caves.h`). The kernel puts one cave in every cell of a Voronoi lattice with
`frequency = 0.01` (100 m cells) and `randomness = 0` (cell centres on the integer lattice), so the cell at the
origin always holds one, and always cave asset 0: an L-system tunnel network (`terrain/assets/caves`: at least 5000
symbols traced as 0.5-unit extrusions turning in 15 degree steps, skinned at radius ~2, remeshed, its SDF stored on a
128^3 grid). The asset's vertex centroid sits at (0, 0, `Caves.height_offset` = -4 m). It is turned about z by a
random angle (`is_horizontal = 1`: no tilt), scaled so its padded bounding box has a 100 m diagonal, and its walls are
pushed in and out by 9-octave Perlin noise of amplitude `noise_scale` (uniform 2-5 m). That noise is as large as the
tunnel radius or larger, so the result is less a tunnel than a cluster of irregular chambers with pillars, dead ends
and some two-level stretches. Infinigen's camera search then samples `cave.gin`'s populated bounds (|x|, |y| <= 25 m,
-25 <= z <= 0), drops each sample to 1.5-2.5 m above the surface below it, and keeps poses in free space that see
terrain in every pixel (`terrain_coverage_range = (1, 1)`), at least 30 % of it cave-tagged, at most 30 % of it
closer than 4 m.

**nature_box.py for a cave:**

| | |
|---|---|
| box | stays centred on the origin, where the cave is |
| caves | `scene.caves_chance=1` after the configs (`fast_terrain_assets.gin` sets it to 0); `caves.load_assets.on_the_fly_instances=1` (only asset 0 reaches the box; one asset takes 7-9 min to build, on one core) |
| terrain z bounds | measured, not fixed (`cave_bounds`): once the terrain elements exist, before meshing, their SDF is sampled on a 1 m x 1 m x 0.5 m grid over the populated square; free cells with rock somewhere above them are cave; the largest connected piece reaching into the flight box sets the z bounds, 2 m below its lowest and above its highest cell. Infinigen's camera search samples inside these bounds too |
| populate | only within two grid cells of that piece (its floor, walls and roof), not the ground above the cave |
| closed sides | the UniformMesher leaves terrain open where it meets the bounds, so a tunnel crossing the side of the square would end in a hole onto the sky. Everything beyond the sides and the bottom counts as rock: tunnels end in a wall 4 m outside the populated square. The top stays open, so a real entrance keeps its sky |
| too little cave | stops after the terrain stage when the flight box holds less than `--cave-min-void` (300 m^3) of it |
| water | off by default (`cave.gin` would flood everything below z = -5 with 80 % chance); `--water type` keeps it |
| trees | not touched, but `--tree-density 0` is recommended: `cave.gin` keeps trees (chance 0.4) on upward-facing land, and a cave floor is that |

**box_views.py for a cave:**

- **Floor from inside the void** (`CaveFloor`). Casting down from z = 200 meets the roof (or the ground above the
  cave) first. Instead, every terrain surface on the vertical line is collected and paired into void intervals: a
  roof (face normal pointing down; Infinigen's terrain normals point out of the rock) and the next surface below it
  with its normal up, the floor. Only roofed intervals whose floor face carries Infinigen's cave tag (`MaskTag`
  values listed as `cave.*` in `MaskTag.json`: the faces the cave carved) count, so voids under other overhangs do
  not. The ground is the floor of the tallest of them (heights capped at the band top, 5.5 m; ties go to the lowest);
  intervals under 1 m are pockets in the walls. A column without one (solid rock, the open ground above the cave, an
  entrance or sinkhole open to the sky) has no floor: `ground_grid` is NaN there.
- **Cave rock in the cloud.** The terrain is not in the cloud of an open scene (the ground is `ground_grid`). In a
  cave the walls and the roof are the obstacles, so terrain samples more than 0.3 m above the floor of their column
  join the cloud with label `cave_rock`; heights over the ground (`agl`) in columns without a floor use the nearest
  floor.
- **Band slices over that floor**; columns without a floor are blocked at every height. `band.json` gets a `cave`
  block: share of the box's columns with a floor, floor heights, headroom (roof minus floor) percentiles, share with
  more than one cave level, share whose topmost surface faces the sky. Its `canopy_*` numbers count the roof.
- **Views** are drawn as in the open scenes (free in the 1-3 m slice, 1.6-2.4 m over the floor, facing the
  longest free sightline). No backdrop. The looks A1-A3 only change the sky; inside a closed cave use A0.
- **top.png is a cut-away**: a copy of the terrain without the faces more than 2 m above the floor (roof, upper
  walls, the ground above), objects wholly above that height hidden, lit by the sky from above. Solid rock shows as
  the black lower half of the sky. `make_sheet.py`'s plan draws columns without a floor as obstacles.

**Light.** `cave.gin` turns glowing rocks on (`compose_nature.glowing_rocks_chance = 1`): `GlowingRocksFactory`
placeholders on cave-tagged faces of any orientation (`normal_thresh = -1.1`), 0.025 per m^2; each is a
translucent rock in a random gem colour with a 400-800 W point light inside, transparent to non-camera rays so its
light gets out. They are practically all the light. In the smoke scene's view 0 the mean sRGB luminance is 111/255
with everything on, 111 with the glowing rocks alone, and 19 (31 % of the pixels below 10) with the sky alone, which
only comes in through openings. `cave.gin` also sets the film exposure to 1.3 (`configure_render_cycles.exposure`;
it is saved in scene.blend, so box_views renders with it). The frames are bright and colourful rather than dark:
walls in the terrain's two materials (a salmon-red rock and a pale lilac-white stone) lit in pools of the rocks'
pink and violet light, the rocks themselves as small bright spots, ferns and dark boulders on the floor, daylight
through an opening in one view; up close the terrain's flat-shaded facets show.

**Smoke scene (seed 7)**, the commands above:

| | |
|---|---|
| generation | 24.1 min: terrain 10.5 (building cave asset 0 alone 7.1, on one core; the land tile 2.0), Infinigen's camera search 5.0, in-view split 3.6, ground cover (pebbles, twigs, ferns, mushrooms) 3.6, populate 0.2, fine terrain 0.6; peak RSS 22.6 GB, reached in the in-view split (which subdivides the near terrain twice; Infinigen's per-stage RSS stays under 3.7 GB); scene.blend 1.9 GB |
| terrain | ground + MultiMountains land tile + voronoi rocks; `cave_bounds`: z bounds -10 to 4 m, 1901 m^3 of roofed cave in the flight box, cave cells between z = -8 and 2 m |
| assets in the box | 60 glowing rocks (67 in the populated square), 4 boulders, ferns, mushrooms, pebbles; no trees (`--tree-density 0`) or bushes (not drawn) |
| floor | 49 % of the box's columns (21 % of the populated square's), z -8.5 to 0.0 m; headroom p10 / median / p90 1.7 / 3.6 / 6.4 m; 7 % with two cave levels; in 88 % of the columns the topmost meshed surface faces the sky (the ground above the cave lies inside the z bounds; the roof is 0.6-3 m thick at the sampled points) |
| slices | 1-3 m: 72 % of the box blocked (the 51 % without a floor included), 9.5 % at least 1 m clear, median clearance 0.7 m; 0.4-5.5 m: 92 % blocked (the roof) |
| cloud | 344,943 voxels, 316,110 of them `cave_rock` |
| views | v0 at (-2.4, -2.1, -4.6) m, 1.8 m over the floor, free sightline 12 m; v1 at (-3.1, 4.9, -4.5), 2.3 m over the floor, 18 m; 34 and 36 s per frame at 32 spp, cut-away 24 s, 2.4 min in all; median depth 3.8 and 7.2 m, v1 sees 3.8 % sky |

An earlier run of the same command with cave asset 0 pre-built (kept with `-p execute_tasks.reset_assets=False`)
took 16.9 min and gave a byte-identical cloud.npz and band.json and the same views.

## Indoors

Infinigen-Indoors (`generate_indoors`) lays out a house with a floor-plan solver, furnishes every room with a
constraint solver and has no terrain to mesh. `infinigen/indoor_box.py` runs it on one storey with
`base_indoors.gin` + `fast_solve.gin` (fewer solver steps and Infinigen's reduced room program: living room,
kitchen, dining room, bedroom(s), bathroom) and writes `<out>/box.json` (every room's floor outline, area, floor
and ceiling height, object counts per factory and per room) and `<out>/scene/scene.blend`.
`infinigen/indoor_views.py` makes the flight-box views; it imports `box_views.py` for the sampling, the voxel
accumulator, the camera and the render helpers.

```bash
python infinigen/indoor_box.py --seed 1 --out out/house_1
python infinigen/indoor_views.py --scene out/house_1/scene/scene.blend --box out/house_1/box.json \
    --out out/house_1/views --n-views 3 --samples 32 --threads 30
```

**Camera culling: none for the house.** The indoor pipeline has no "in view of the camera" test: the solver
places the objects of every room, and `populate_state_placeholders` spawns each one with no camera, at the fixed
detail distance (`scatter_res_distance`, 4 m). `target_face_size` would read the scene camera's lens, but no
scene camera exists before populate (the driver logs it: `scene camera = None`), so every asset gets its 3 mm
floor. The only scatter indoors (fruit in bowls) is called without `dist_max`/`fov`, so it culls nothing.
What does depend on the camera is optional, and the driver forces it off with gin overrides (nothing is patched):

| Infinigen-Indoors | Flight box |
|---|---|
| outdoor terrain around the house, grass and rocks only where the camera looks (`create_outdoor_backdrop`: `split_inview`, `near_distance`) | terrain off, no outdoors (`compose_indoors.terrain_enabled=False`); windows look onto the sky |
| room meshes re-meshed by OcMesher around the cameras (`real_geometry*.gin`) | off: Infinigen's own room meshes |
| `overhead.gin`: rooms other than the solved ones hidden, ceilings invisible, an overhead camera | off: every room and ceiling renders |
| floating objects placed in the camera's view | off |
| a search for camera poses inside the rooms (`pose_cameras`) | skipped: nothing uses it, `indoor_views.py` places its own cameras |
| a 20 % chance that every light is deleted | 0, so the views are lit |
| half the doors nearly shut (0-9 deg), 20 % open, the rest anywhere | `--doors open` (default): every door 84-174 deg open, Infinigen's own `populate_doors.all_open`; `--doors infinigen` keeps the mix |

`box.json` lists the objects in each room and any placeholder left unpopulated (`unpopulated_placeholders`, empty).

**Views.** The flight band runs from 0.4 m over the floor to 0.4 m under the ceiling. The cloud holds every
render-visible mesh and instance (walls, doors, windows with their glass, furniture, lamps, floor, ceiling and the
house's outer shell) as 10 cm voxels, each labelled with its factory or room part (`wall`, `floor`, `ceiling`,
`exterior`); unlike the nature boxes, small meshes are sampled as surfaces too (a cup is not one point), at
2000 samples/m2 so walls have no holes. Occupancy is counted over the house footprint, the room floors closed across
the walls between them: an inner wall is a blocked cell, a doorway a free one. `band.json` reports the slices
0.4-1, 1-2 (the views' slice) and 2 m to the band top, plus the whole band, with connectivity: how many rooms the
largest free region reaches, over all free cells and over those at least 0.5 m clear. Views stand in the most open
spot of the roomiest rooms, 1.2-1.6 m over the floor, facing the longest free sightline; the renders keep
Infinigen's indoor exposure (`film_exposure` 3, saved in the scene). The top view hides the ceiling, the roof and the
ceiling lights from camera rays only, as Infinigen's overhead mode does, so the rooms keep their light.

| Step (gs2, CPU) | Time |
|---|---|
| generate a 5-room house (133 m2, 50 objects) with the smoke settings below | 16 min, peak 1.9 GB RAM |
| cloud (106 k voxels), occupancy and plan | 8 s |
| one 640 x 480 drone frame, 32 spp, 30 threads | 22-35 s (top view 21 s) |

The smoke house (`/mnt/data/infinigen/agents/indoors/house_1` on gs2) was made with

```bash
python infinigen/indoor_box.py --seed 1 --out out/house_1 -p compose_indoors.solve_steps_large=30 \
    compose_indoors.solve_steps_medium=15 SimulatedAnnealingSolver.max_invalid_candidates=2 \
    compose_indoors.solve_small_enabled=False
```

because `fast_solve.gin`'s own steps (100 large, 40 medium, 5 small per room and stage) take hours on a shared gs2:
an annealing step cost 2-3 s in the first room with the box loaded, grows with the objects already placed (10 s in
the living room), and a proposal that cannot fit is retried at full price. In seed 1's kitchen every attempt to add
a kitchen island built a new island and searched up to `max_invalid_candidates` (5) poses for it: 45-125 s per
attempt, every one failing, about an hour for that stage alone. `solve_small_enabled=False` leaves out the small
objects on surfaces (cups, books, plates), nothing a drone flies into. The price is a sparser house: with 30 steps
and 2 candidates no bed was placed in the smoke house (8 proposals failed; the 50-step try placed one) and the
kitchen has an oven and shelves but no counters. For a full house run `fast_solve.gin`'s defaults on a quiet box.

Traps:

1. **Doorways are tight.** A door is 0.7-0.8 x (1.4 m - wall thickness) wide, about 0.85 m, and its leaf and frame
   take some of that: on 25 cm cells a doorway is two or three free cells, so rooms that connect over all free cells
   need not connect at 0.5 m clearance. In the smoke house the free cells at 1-2 m form one region over 4 of the 5
   rooms; the cells at least 0.5 m clear reach 3.
2. **Room floors stop at the walls.** The floor outlines in `box.json` are the rooms' own floors; the walls between
   rooms, and the doorways in them, belong to no room (hence the closed footprint above).
3. **Window glass** is in the cloud (a drone cannot pass it) but almost fully transparent to camera rays, so the
   depth image sees through windows to the sky.
4. **An "open" door can still fill its doorway.** With `--doors open` the smoke house's bathroom door
   (`LiteDoorFactory`) was turned 98 deg and still lies across its 0.7 m opening, on the bathroom side of the wall
   (door voxels from the floor to 2.9 m in the doorway cells), so the bathroom is sealed; the bedroom and entrance
   doors (`PanelDoorFactory`, turned 165 and 127 deg) stand in their rooms. Why this door ends up there is not
   established (a probe of the door meshes was inconclusive). Check `components_free` /
   `rooms_in_largest_component_free` in `band.json` before flying room to room.
5. **The floor is not at z = 0.** Each room is its outline extruded to the wall height and shelled inwards by half a
   wall thickness, so the floor is at `floor_z` (0.15 m here) and the ceiling at wall height minus that; use
   `box.json`, which reads both from the meshes.
