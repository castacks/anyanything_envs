# Provenance

Everything here was written by Daniel Kim (with Claude Code sessions) for the anyanything project. It was copied
on 2026-10-29 from three places and is a fresh import: the original history stays where it was.

| Source | Where | Commit |
|---|---|---|
| superfly_expert_sampler (`blender/`, `scenes/`, `scripts/gen_courses.py`, `src/superfly_expert_sampler/{draw_states,tiers,band,sim_sdf}.py`) | branch `dagger-rgb` of the lab's private sampler repository; every copied file's history is Daniel's own commits | `94c7c74` |
| Infinigen flight-box drivers (`infinigen/`) | gs2 `/mnt/data/infinigen/flightbox`, never committed before | 2026-10-29 |
| `scripts/make_spec.py`, `envgen/primitives/spec.py`, `blender/specs.py`, tests, docs | new | this repo |

The primitive families were checked after the move: `tests/golden_specs.json` holds spec and cloud digests
written by the original code for 48 patches (22 tier-A families x 2 seeds, 2 corridors, 2 arenas), and
`tests/test_primitives.py` reproduces every one byte for byte.

Left out on purpose: Wenshan's backdrop kit and her scene tools (`kits/wenshan.py`, `wenshan_*.py`), the
dataset pipeline that labels courses and re-skins datasets (`finish_frames`, `pipeline*`, `reskin*`, the labeller
itself), the policy flyer (`fly_server.py`), and `scenes/obstacle_field.py` (the held-out Isaac
evaluation fields).

## Files

| File | Copied from | Changes |
|---|---|---|
| `blender/alpha_expr.py` | `blender/alpha_expr.py` | copied; unchanged |
| `blender/assets.py` | `blender/assets.py` | copied; unchanged |
| `blender/band_shim.py` | `blender/band_shim.py` | loads `envgen.band` |
| `blender/batch.py` | `blender/batch.py` | plan/plan2/plan3/build/gates/backfill only: the S3 sync, route, label, render and summary stages (lab infrastructure) removed; docstring rewritten |
| `blender/bl_common.py` | `blender/bl_common.py` | copied; unchanged |
| `blender/build_course.py` | `blender/build_course.py` | copied; unchanged |
| `blender/build_gates.py` | `blender/build_gates.py` | specs via `envgen`; the Wenshan backdrop gate removed |
| `blender/course_semantics.py` | `blender/course_semantics.py` | copied; unchanged |
| `blender/fetch_library.py` | `blender/fetch_library.py` | copied; unchanged |
| `blender/fetch_objaverse.py` | `blender/fetch_objaverse.py` | copied; unchanged |
| `blender/flowlib.py` | `blender/flowlib.py` | copied; unchanged |
| `blender/flythrough_path.py` | `blender/flythrough_path.py` | copied; unchanged |
| `blender/flythrough_video.py` | `blender/flythrough_video.py` | copied; unchanged |
| `blender/inspect_assets.py` | `blender/inspect_assets.py` | copied; unchanged |
| `blender/kit_try.py` | `blender/kit_try.py` | copied; unchanged |
| `blender/kits/__init__.py` | `blender/kits/__init__.py` | no backdrop kits (the Wenshan kit stays out) |
| `blender/kits/_farm_util.py` | `blender/kits/_farm_util.py` | copied; unchanged |
| `blender/kits/_forest_util.py` | `blender/kits/_forest_util.py` | copied; unchanged |
| `blender/kits/_garage_util.py` | `blender/kits/_garage_util.py` | copied; unchanged |
| `blender/kits/_industrial_util.py` | `blender/kits/_industrial_util.py` | copied; unchanged |
| `blender/kits/_ruins_util.py` | `blender/kits/_ruins_util.py` | copied; unchanged |
| `blender/kits/_street_util.py` | `blender/kits/_street_util.py` | copied; unchanged |
| `blender/kits/alley.py` | `blender/kits/alley.py` | copied; unchanged |
| `blender/kits/backyard.py` | `blender/kits/backyard.py` | copied; unchanged |
| `blender/kits/base.py` | `blender/kits/base.py` | copied; unchanged |
| `blender/kits/coast.py` | `blender/kits/coast.py` | copied; unchanged |
| `blender/kits/construction.py` | `blender/kits/construction.py` | copied; unchanged |
| `blender/kits/farm.py` | `blender/kits/farm.py` | copied; unchanged |
| `blender/kits/forest.py` | `blender/kits/forest.py` | copied; unchanged |
| `blender/kits/garage.py` | `blender/kits/garage.py` | copied; unchanged |
| `blender/kits/industrial.py` | `blender/kits/industrial.py` | copied; unchanged |
| `blender/kits/library.py` | `blender/kits/library.py` | copied; unchanged |
| `blender/kits/looks.py` | `blender/kits/looks.py` | copied; unchanged |
| `blender/kits/park.py` | `blender/kits/park.py` | copied; unchanged |
| `blender/kits/port.py` | `blender/kits/port.py` | copied; unchanged |
| `blender/kits/ruins.py` | `blender/kits/ruins.py` | copied; unchanged |
| `blender/kits/street.py` | `blender/kits/street.py` | copied; unchanged |
| `blender/kits/warehouse.py` | `blender/kits/warehouse.py` | copied; unchanged |
| `blender/montage.py` | `blender/montage.py` | copied; unchanged |
| `blender/pack_check.py` | `blender/pack_check.py` | copied; unchanged |
| `blender/page_data.py` | `blender/page_data.py` | copied; unchanged |
| `blender/piece_ids.py` | `blender/piece_ids.py` | copied; unchanged |
| `blender/provenance.py` | `blender/provenance.py` | copied; unchanged |
| `blender/render_frames.py` | `blender/render_frames.py` | copied; unchanged |
| `blender/render_io.py` | `blender/render_io.py` | copied; unchanged |
| `blender/render_looks.py` | `blender/render_looks.py` | copied; unchanged |
| `blender/render_poses.py` | `blender/render_poses.py` | copied; unchanged |
| `blender/showcase.py` | `blender/showcase.py` | copied; unchanged |
| `blender/specs.py` | `new` | adapter: the old `build_spec(test, tier, family, seed, split)` signature onto `envgen.primitives` |
| `envgen/band.py` | `src/superfly_expert_sampler/band.py` | copied; unchanged |
| `envgen/primitives/arena.py` | `scenes/arena.py` | copied; unchanged |
| `envgen/primitives/families.py` | `src/superfly_expert_sampler/draw_states.py` | the patch half only (families, `PatchSpec`, `build_patch`, `face_params`, `PatchGrid`); the course-cloud hook, state drawing and attitude synthesis left out; `primitive_clouds` kept as a no-op |
| `envgen/primitives/gen_courses.py` | `scripts/gen_courses.py` | kept the leg builders; dropped the CLI (`main`, which timed fields with the labeller's `sim_episode`) |
| `envgen/primitives/patch.py` | `scenes/patch.py` | `import arena` made package-relative |
| `envgen/primitives/sim_sdf.py` | `src/superfly_expert_sampler/sim_sdf.py` | copied; unchanged |
| `envgen/primitives/spec.py` | `scripts/data_make_dataset.py (`build_spec`)` | rewritten as a small module with the same dispatch |
| `envgen/primitives/tiers.py` | `src/superfly_expert_sampler/tiers.py` | imports repointed to the package |
| `infinigen/box_views.py` | `gs2 flightbox/forest_views.py` | renamed; docstring; `sample_scene` takes `small_m` and `family` (defaults unchanged) for the indoor views |
| `infinigen/indoor_box.py` | `new` | Infinigen-Indoors house driver |
| `infinigen/indoor_views.py` | `new` | indoor flight-box views on top of `box_views.py` |
| `infinigen/nature_box.py` | `gs2 flightbox/forest_box.py` | renamed (any scene type); docstring |
| `infinigen/page/make_page.py` | `gs2 flightbox/make_page.py` | paths in the docstring |
| `infinigen/page/make_sheet.py` | `gs2 flightbox/make_sheet.py` | paths in the docstring |
| `infinigen/tools/bl52_frame.py` | `gs2 flightbox/bl52_frame.py` | copied; unchanged |
| `infinigen/tools/probe_instances.py` | `gs2 flightbox/probe_instances.py` | imports box_views |
