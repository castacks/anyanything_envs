# anyanything_envs

Procedural environments for drone navigation: obstacle layouts, realistic Blender places built around them,
and Infinigen nature scenes turned into flight boxes. Each generator writes a scene and its collision cloud
in the same format, so a planner, a labeller or a policy can fly any of them.

| Generator | What it makes | Runs in |
|---|---|---|
| [`envgen.primitives`](envgen/primitives) | Primitive obstacle patches: 22 single-obstacle families (walls, slots, windows, floating boxes and bars, rooms, rings, tree stands, clutter, wires...), corridors of several obstacles (tier B) and arenas mixing many (tier C). | plain Python (numpy, scipy) |
| [`blender/`](blender) | Blender courses: a patch rebuilt as meshes with the same geometry and dressed as a real place, either one of 4 themes or one of 13 place kits (2 indoor). Asset libraries from Poly Haven and ambientCG (CC0) plus a curated Objaverse slice. | Blender 5.2 |
| [`infinigen/`](infinigen) | Infinigen nature scenes (forest, plain, river, canyon, cliff, mountain, desert, arctic, snowy mountain, coast; caves, flown inside) generated as a fully populated square flight box, with collision cloud, flight-height plan and drone-eye renders in four lighting looks; optional Wenshan-style floating obstacles from Infinigen's procedural asset factories. Infinigen-Indoors houses (a few furnished rooms, one storey) as indoor flight boxes. | Infinigen 1.x (bpy 4.2) |

Wenshan's static Blender scenes are not part of this repo (they are hers).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e .     # numpy, scipy, scikit-fmm

# 1. a primitive patch and its cloud
python scripts/make_spec.py --family wall_slot --seed 2000 --out out/wall_slot_2000/spec.json
python - <<'EOF'
from envgen.primitives import load_spec, patch_cloud
pts, region = patch_cloud(load_spec("out/wall_slot_2000/spec.json"))   # region -1 = floor grid
print(pts.shape)
EOF

# 2. the same patch as a parking garage (Blender 5.2; the v2 asset library, see "Assets")
blender -b --factory-startup -P blender/build_course.py -- --spec out/wall_slot_2000/spec.json \
    --kit garage --assets $BLENDER_LIBRARY --out out/wall_slot_2000/garage --cpu --preview

# 3. a batch: 6 scenes in each indoor kit, built 4 at a time, gated
python blender/batch.py plan3 --batch out/indoor --per-kit 6 --kits garage warehouse
python blender/batch.py build --batch out/indoor --cpu --workers 4 --threads 6 --allow-dirty
python blender/batch.py gates --batch out/indoor --allow-dirty

# 4. an Infinigen forest as a 30 x 30 m flight box (Infinigen installed, see docs/infinigen.md)
python infinigen/nature_box.py --seed 1 --out out/forest_1 --scene-type forest --design-dist 8 --no-monocots \
    --configs fast_terrain_assets.gin
python infinigen/box_views.py --scene out/forest_1/scene/scene.blend --box out/forest_1/box.json \
    --out out/forest_1/views --n-views 3 --samples 32

# 5. an Infinigen-Indoors house (a few furnished rooms) as an indoor flight box
python infinigen/indoor_box.py --seed 1 --out out/house_1
python infinigen/indoor_views.py --scene out/house_1/scene/scene.blend --box out/house_1/box.json \
    --out out/house_1/views --n-views 3 --samples 32
```

## Place kits

| Kit | Indoor | What it builds |
|---|---|---|
| garage | yes | parking garage level: columns, painted bays, parked cars, lamps and services under a concrete ceiling |
| warehouse | yes | distribution warehouse: pallet racking aisles, loads, dock doors, high-bay lamps, forklifts |
| street | | city street: building canyon with shopfronts, pavements, parked cars, lamps, trees, markings |
| alley | | back alley: narrow lane between building backs, fire escapes, dumpsters, pipes, cables overhead |
| port | | container terminal: stacked ISO containers in blocks, painted lanes, light masts |
| industrial | | process plant / factory yard: pipe racks and bridges, tanks, sheds, process frames, yard equipment |
| construction | | building site: concrete frame + scaffolding, stacked materials, cabins, fences, cranes, muddy ground |
| farm | | farmyard: barn and pole shed, paddock fence, hay bales, wagon/tractor, fields and tree lines |
| forest | | woodland: bark trunks under real crowns, undergrowth, logs, mossy rock; conifer/broadleaf/birch/burnt/snow |
| park | | city park: lawns, paths, trees, hedges, benches, lamps, fountain/statue, playground; city beyond |
| ruins | | stone ruins: broken walls, arcades, columns, rubble and overgrowth; cloister, castle, abbey or temple |
| backyard | | suburban backyard: house rear with patio, lawn, board fence, shed, trampoline, swings, washing line |
| coast | | rocky harbour shore: beach/shingle/cove/harbour, sea with foam, pier on pilings, boats, huts, nets |

A kit is one module under [`blender/kits/`](blender/kits) with `NAME`, `INDOOR`, `DESCRIPTION` and
`build(ctx)`; [`kits/__init__.py`](blender/kits/__init__.py) has the contract. The patch's primitives are
always rebuilt with their exact geometry (the obstacle the patch was drawn for is really there); the kit
chooses what they look like and adds structure, props and scenery around them. `--variant` picks a variation,
`--kit-args` passes kit options.

## Assets

The Blender courses draw on asset libraries you fetch once (nothing is vendored here):

```bash
python blender/fetch_library.py --root $BLENDER_LIBRARY            # v2 library for the kits: Poly Haven + ambientCG
python blender/fetch_objaverse.py index|select|fetch|finalise      # optional curated Objaverse slice (see its docstring)
blender -b -P blender/inspect_assets.py -- --root $BLENDER_LIBRARY # measure and catalogue every model
python blender/assets.py --root $BLENDER_ASSETS --theme yard       # v1 library for the four themes
```

Poly Haven and ambientCG are CC0. Objaverse items keep their own licences, which the fetcher records per item.
Default locations, each overridable: `$BLENDER` (Blender 5.2 binary), `$BLENDER_LIBRARY`
(`~/.cache/envgen/library`, the kits' v2 library, ~37 GB with the Objaverse slice), `$BLENDER_ASSETS`
(`~/.cache/envgen/assets_v1`, the themes' v1 library). `$SUPERFLY_BAND` selects the altitude band the planner
and the cloud top follow (`envgen/band.py`; default v9).

## Outputs

Every course directory holds `spec.json` (the patch), `cloud.npz` (`points` float64, `region` int16:
obstacle samples one per 10 cm voxel, region 0 and up; floor grid at 25 cm, region -1), `course.blend`
(packed with `--pack`) and `course.json` (what was placed, materials, HDRI, provenance). Infinigen boxes write
`box.json`, `scene/scene.blend` and, from `box_views.py`, `views/cloud.npz` with per-height-slice occupancy
in `band.json`; indoor houses (`indoor_box.py`, `indoor_views.py`) write the same files, with the rooms in `box.json`
and occupancy over the floor. Details: [docs/formats.md](docs/formats.md).

## Tests

```bash
pytest tests/test_primitives.py      # every family reproduces the original specs and clouds byte for byte
```

## Where the code came from

See [PROVENANCE.md](PROVENANCE.md): every file, the repository and commit it was copied from, and what was
changed.
