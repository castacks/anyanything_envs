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

## Scene types

Tested: forest. Running in the first smoke batch: plain, river, canyon, cliff, mountain, desert, arctic,
snowy_mountain, coast. Not yet: **cave** (the ground is found by casting down from above, which hits the
roof; needs a floor search from inside). **Infinigen-Indoors** is a different generator, see "Indoors" below.
Underwater types (kelp_forest, coral_reef, under_water) are not useful for drones.

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
