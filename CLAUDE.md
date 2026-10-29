# anyanything_envs: notes for agents

Environment generators for drone navigation. Read `README.md` first, then `docs/formats.md` for the output
contract and `docs/infinigen.md` before touching `infinigen/`.

- **Primitive families are frozen by digests.** `tests/golden_specs.json` holds spec and cloud SHA-256s for 48
  patches. Run `pytest tests/test_primitives.py` after any change under `envgen/`. A deliberate change to a
  family must regenerate the digests in the same commit and say why: every course and label built on that
  family changes with it.
- **Three runtimes.** `envgen` and `blender/batch.py` run in a normal Python with numpy, scipy and scikit-fmm.
  Everything else in `blender/` runs inside Blender 5.2 (`blender -b --factory-startup -P ...`), whose Python has
  no scipy, so keep `blender/build_course.py` and the kits free of `envgen` imports beyond `band_shim`.
  `infinigen/` runs in an Infinigen 1.x environment (bpy 4.2, Python 3.11).
- **Reproducibility.** The same spec, arguments, code and asset library give the same `cloud.npz` byte for byte;
  `batch.py build` pins `PYTHONHASHSEED=0` and refuses a dirty checkout unless `--allow-dirty`. Keep it that way.
- **Assets are never committed.** Fetch them with `blender/fetch_library.py` / `assets.py` / `fetch_objaverse.py`
  into `$BLENDER_LIBRARY` / `$BLENDER_ASSETS`.
- **Not in this repo, on purpose:** Wenshan's static scenes and backdrop kit, the labelling and dataset pipeline,
  the policy flyer, and the held-out Isaac evaluation fields. Do not add generators that replicate held-out
  evaluation environments.
