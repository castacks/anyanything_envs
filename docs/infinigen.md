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
placement itself mirrors Wenshan's scenes, so keep a variant without floaters for a clean out-of-distribution
score. Placement rejects anything within an object's bounding radius plus `--gap` of the scene's cloud or of
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
roof; needs a floor search from inside) and **Infinigen-Indoors** (a different generator, `generate_indoors`).
Underwater types (kelp_forest, coral_reef, under_water) are not useful for drones.
