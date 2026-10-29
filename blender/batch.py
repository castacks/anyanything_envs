#!/usr/bin/env python3
"""A batch of Blender courses: plan the scenes, build them, gate them, backfill the failures.

    # choose the scenes and write their specs (plan: four themes x three levels; plan2: every place kit;
    # plan3: --per-kit scenes per kit, families rotated, levels L1/L2/L3)
    python blender/batch.py plan     --batch <dir>
    python blender/batch.py plan3    --batch <dir> [--per-kit 72] [--kits garage warehouse industrial]
    # build every course in Blender 5.2 (builds refuse a dirty checkout unless --allow-dirty)
    python blender/batch.py build    --batch <dir> [--cpu --no-preview --pack-max-res 1024 --workers 4 --threads 6]
    python blender/batch.py gates    --batch <dir> [--workers 8] [--pack-check] [--expect-git-sha SHA]
    python blender/batch.py backfill --batch <dir>     # scenes.json; prints the spares + rebuilds to do next

Run it with a Python that has this repo's requirements (numpy, scipy, scikit-fmm): the plan stages build
patch specs with ``envgen.primitives``; Blender is only started for ``build`` (``$BLENDER`` or the first
of ``BLENDER_CANDIDATES``). Every stage skips what is already done, so a stage can be re-run after a
failure. A scene is one patch: ``<family>_<seed>`` names its course.

**The plan.** Three levels, four themes:

* level 1 -- tier-A single-obstacle patches, all 14 families, the seed whose
  geometry is *easiest* among 40 candidates (widest slot, largest window,
  smallest box, ...), sparse clutter kept well clear of the openings;
* level 2 -- the same families at the *hardest* of the 40 seeds, normal
  clutter, closer to the openings;
* level 3 -- tier-B corridors (several obstacles in sequence) and tier-C
  arenas (many families on one map), dense clutter, and in the warehouse
  lamps hung low into the flight band.

Themes rotate so every level and most families appear in more than one.

**plan3**: every kit gets
``--per-kit`` scenes, tier A by default, families rotated so each kit covers
all 14 evenly, levels L1/L2/L3 at ``--level-mix`` (40/40/20 %) within each kit
and each family, seeds from ``--seed0`` 10000 (unique per tier). Geometry is *natural*
(the next seed as drawn) rather than plan2's easiest/hardest of 40: those
are the tails of each family's draw (the 2.5 % ends), and the level's clutter knobs already set the difficulty; the 40
candidates also cost 40 ``build_spec`` calls a scene (``--geometry extremes``
keeps the old rule). ``--backfill`` spares per (kit, tier, family, level)
replace scenes that fail the build gates, in a fixed order (``backfill``);
only a scene's own failures spend a spare (``build_gates.SCENE_FAILS``: the
script raised, clearance, slots, gaps, points, CPU build seconds). A build
that was refused, killed, dirty, unpacked or of stale code leaves its slot
pending for a rebuild instead, so neither the box's load nor an operator
slip re-seeds the batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
#: where Blender 5.2.2 lives, first hit wins (``$BLENDER`` overrides)
BLENDER_CANDIDATES = ("~/opt/blender-5.2.2-linux-x64/blender", "/mnt/data/blender/blender-5.2.2-linux-x64/blender")
#: the v1 theme assets (``assets.py``), first hit wins (``$BLENDER_ASSETS`` overrides)
ASSETS_CANDIDATES = ("~/.cache/envgen/assets_v1", "~/.cache/blender_assets", "/mnt/data/blender/assets_v1")


def _first(env: str, candidates, fallback_which: str | None = None) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if Path(c).expanduser().exists():
            return str(Path(c).expanduser())
    if fallback_which and shutil.which(fallback_which):
        return shutil.which(fallback_which)
    return str(Path(candidates[0]).expanduser())


def find_blender() -> str:
    """``$BLENDER``, else the first of ``BLENDER_CANDIDATES`` that exists, else ``blender`` on PATH."""
    return _first("BLENDER", BLENDER_CANDIDATES, "blender")


BLENDER = find_blender()
#: Blender's Python salts str hashes per process and ignores PYTHONHASHSEED unless it runs with
#: ``--python-use-system-env``; a kit loop over a set of names would then build objects in another
#: order on every run (none does today: the 09-27 street rebuild difference was create_uvsphere's
#: face order, fixed in build_course). Builds pin the hash anyway and keep the rest of the
#: caller's Python environment out of Blender.
BLENDER_ENV = {"PYTHONHASHSEED": "0", "PYTHONPATH": "", "PYTHONHOME": "", "PYTHONSTARTUP": "", "PYTHONNOUSERSITE": "1"}
BLENDER_PY = ["--python-use-system-env", "--python-exit-code", "1"]
ASSETS = Path(_first("BLENDER_ASSETS", ASSETS_CANDIDATES))
#: the v2 library (fetch_library.py + inspect_assets.py) the place kits draw from
LIBRARY = Path(os.environ.get("BLENDER_LIBRARY", "~/.cache/envgen/library")).expanduser()
THEMES = ["yard", "woodland", "warehouse", "plaza"]
FAMILIES = ["wall_offset", "wall_sym", "float_box", "float_bar", "wall_slot", "window", "room", "bars_lintel",
            "pillar", "slab", "ring", "stand_sparse", "stand_medium", "stand_dense"]
TIER_FAMILY = {"B": "corridor", "C": "arena"}
STATES = {"A": 10, "B": 10, "C": 15}
LEVEL_KNOBS = {1: {"clutter": 0.5, "keepout": 4.0, "hang": 0.0},
               2: {"clutter": 1.0, "keepout": 2.5, "hang": 0.0},
               3: {"clutter": 1.2, "keepout": 1.5, "hang": 3.0}}


def geometry_hardness(family: str, p: dict) -> float:
    """Bigger = harder, from a tier-A patch's drawn parameters."""
    return {
        "wall_offset": lambda: p["width"], "wall_sym": lambda: p["width"],
        "float_box": lambda: p["width"] * p["height"], "float_bar": lambda: p["width"],
        "wall_slot": lambda: -p["gap"], "window": lambda: -p["width"] * p["height"],
        "room": lambda: -p["door_w"], "bars_lintel": lambda: p["n_bars"] - 2.0 * p["z_under"],
        "pillar": lambda: p["r"], "slab": lambda: p["width"],
        "ring": lambda: -70.0 if p.get("closed") else -p["gap_deg"],
    }.get(family, lambda: p.get("depth", 0.0))()


def fam_args(s: dict) -> list:
    """``--families`` is tier A only: tier B/C patches are all corridor/arena."""
    return ["--families", s["family"]] if s["tier"] == "A" else []


def run(cmd, log: Path, env=None, cwd=REPO) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd,
                               env={**os.environ, **(env or {})})


# --- plan (gs2) ---------------------------------------------------------------------------------

def cmd_plan(a) -> int:
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    import specs as dmd  # noqa: PLC0415
    b = a.batch
    (b / "specs").mkdir(parents=True, exist_ok=True)
    scenes = []

    def add(tier, family, seed, level, theme, spec, note=""):
        pid = f"{spec.family}_{seed}"
        (b / "specs" / f"{pid}.json").write_text(spec.to_json())
        scenes.append({"pid": pid, "tier": tier, "family": family, "seed": seed, "level": level, "theme": theme,
                       "states": STATES[tier], **LEVEL_KNOBS[level], "note": note})

    for fi, fam in enumerate(FAMILIES):
        cands = []
        for seed in range(a.seed0, a.seed0 + a.candidates):
            try:
                sp = dmd.build_spec(5, "A", fam, seed, "train")
            except Exception:  # noqa: BLE001
                continue
            cands.append((geometry_hardness(fam, sp.params), seed, sp))
        cands.sort(key=lambda c: c[0])
        easy, hard = cands[0], cands[-1]
        th1 = "woodland" if fam.startswith("stand") else THEMES[fi % 4]
        th2 = THEMES[(fi + 2) % 4] if not fam.startswith("stand") else ["plaza", "yard", "warehouse"][fi % 3]
        add("A", fam, easy[1], 1, th1, easy[2], f"easiest of {len(cands)} (hardness {easy[0]:.2f})")
        add("A", fam, hard[1], 2, th2, hard[2], f"hardest of {len(cands)} (hardness {hard[0]:.2f})")
        print(f"{fam:14s} L1 seed {easy[1]} ({th1}, {easy[0]:.2f})   L2 seed {hard[1]} ({th2}, {hard[0]:.2f})")

    for tier, n in (("B", a.corridors), ("C", a.arenas)):
        got, seed = 0, a.seed0
        while got < n and seed < a.seed0 + 200:
            try:
                sp = dmd.build_spec(5, tier, TIER_FAMILY[tier], seed, "train")
                add(tier, TIER_FAMILY[tier], seed, 3, THEMES[(got + (1 if tier == "C" else 0)) % 4], sp,
                    f"{len(sp.primitives)} primitives")
                print(f"{TIER_FAMILY[tier]:14s} L3 seed {seed} ({scenes[-1]['theme']}, {len(sp.primitives)} primitives)")
                got += 1
            except Exception as e:  # noqa: BLE001
                print(f"{TIER_FAMILY[tier]} seed {seed}: {type(e).__name__}")
            seed += 1
    plan = {"batch": b.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "level_knobs": LEVEL_KNOBS,
            "scenes": scenes}
    (b / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(scenes)} scenes -> {b / 'plan.json'}")
    return 0


# --- plan2 (gs2): place kits -------------------------------------------------------------------

#: per kit: (tier, level, how many). 24 scenes a kit.
KIT_MIX = [("A", 1, 8), ("A", 2, 8), ("A", 3, 2), ("B", 3, 4), ("C", 3, 2)]


#: backdrop kits (``kits.BACKDROPS``): buildable, never planned by plan2/plan3
BACKDROPS = ["wenshan"]


def kit_names() -> list[str]:
    """Kits with a module in blender/kits/ (read without importing bpy)."""
    names = []
    for line in (HERE / "kits" / "__init__.py").read_text().split("KITS = [", 1)[1].split("]", 1)[0].split(","):
        k = line.strip().strip('"').strip("'").strip()
        if k and (HERE / "kits" / f"{k}.py").exists():
            names.append(k)
    return names


def cmd_plan2(a) -> int:
    """Batch 2: every kit x ``KIT_MIX``. Tier-A families rotate so each kit
    gets a spread and every family appears in most kits; L1 takes the easiest
    geometry of ``--candidates`` fresh seeds, L2 the hardest, L3 a random one.
    Seeds are unique across the batch (a scene's id is ``<family>_<seed>``)."""
    import numpy as np  # noqa: PLC0415
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    import specs as dmd  # noqa: PLC0415
    b = a.batch
    (b / "specs").mkdir(parents=True, exist_ok=True)
    kits = a.kits or kit_names()
    rng = np.random.default_rng(a.seed0)
    next_seed = {"A": a.seed0, "B": a.seed0, "C": a.seed0}
    scenes, fam_i = [], 0

    def fresh(tier, family, n):
        out = []
        while len(out) < n and next_seed[tier] < a.seed0 + 20000:
            seed = next_seed[tier]
            next_seed[tier] += 1
            try:
                sp = dmd.build_spec(5, tier, family, seed, "train")
            except Exception:  # noqa: BLE001
                continue
            out.append((seed, sp))
        return out

    for kit in kits:
        for tier, level, n in KIT_MIX:
            for _ in range(n):
                if tier == "A":
                    fam = FAMILIES[fam_i % len(FAMILIES)]
                    fam_i += 1
                    cands = fresh("A", fam, a.candidates if level < 3 else 1)
                    cands.sort(key=lambda c: geometry_hardness(fam, c[1].params))
                    seed, sp = cands[0] if level == 1 else cands[-1]
                    note = f"{['easiest', 'hardest', 'random'][level - 1]} of {len(cands)}"
                else:
                    fam = TIER_FAMILY[tier]
                    (seed, sp), = fresh(tier, fam, 1)
                    note = f"{len(sp.primitives)} primitives"
                pid = f"{sp.family}_{seed}"
                (b / "specs" / f"{pid}.json").write_text(sp.to_json())
                knobs = dict(LEVEL_KNOBS[level])
                scenes.append({"pid": pid, "tier": tier, "family": fam, "seed": seed, "level": level, "kit": kit,
                               "theme": kit, "variant": 0, "states": STATES[tier], **knobs,
                               "start_keepout": 1.5, "note": note})
        print(f"{kit:13s} {sum(1 for s in scenes if s['kit'] == kit)} scenes")
    plan = {"batch": b.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "level_knobs": LEVEL_KNOBS,
            "kit_mix": KIT_MIX, "kits": kits, "scenes": scenes}
    (b / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(scenes)} scenes over {len(kits)} kits -> {b / 'plan.json'}")
    return 0


# --- plan3 (gs2): the v7 mass run ---------------------------------------------------------------

#: default level mix of a kit's scenes (design: L1 40 / L2 40 / L3 20 %)
LEVEL_MIX = {1: 0.4, 2: 0.4, 3: 0.2}
PLAN3_STATES = 15                   # set_M's five regimes x 3 states
PLAN3_START_KEEPOUT = 1.5           # = kits.base.START_CLEAR, the build gate's clearance
SEED_MAX = 99_999                   # rollout_<seed5>_<family>: five digits


def level_quota(n: int, mix: dict) -> dict:
    """``n`` split over the levels by ``mix`` (largest remainder; ties to the lower level)."""
    tot = sum(mix.values())
    raw = {L: n * w / tot for L, w in mix.items()}
    q = {L: int(v) for L, v in raw.items()}
    for L in sorted(raw, key=lambda L: (-(raw[L] - q[L]), L))[:n - sum(q.values())]:
        q[L] += 1
    return q


class LevelSeats:
    """Levels dealt one seat at a time over a whole batch (Sainte-Lague: the next seat
    goes to the level with the largest ``w / (2 got + 1)``, ties to the lower level).
    House-monotone, so a kit's seats are the difference of two batch totals and a
    small per-kit count still realises the batch mix across kits (``level_quota`` per
    kit gives 1 slot to L1 in every kit)."""

    def __init__(self, mix: dict):
        self.mix, self.got = dict(mix), {L: 0 for L in mix}

    def take(self, n: int) -> dict:
        q = {L: 0 for L in self.mix}
        for _ in range(n):
            L = max(sorted(self.mix), key=lambda L: (self.mix[L] / (2 * self.got[L] + 1), -L))
            self.got[L] += 1
            q[L] += 1
        return q


def plan3_slots(kits: list[str], per_kit: int, mix: dict, bc_share: float = 0.0) -> list[dict]:
    """Every kit's scene slots ``{kit, tier, family, level}``, in plan order.

    Tier A: ``per_kit - round(per_kit * bc_share)`` slots whose families run
    round the 14 ``FAMILIES`` from where the previous kit stopped, so every
    kit covers them evenly (counts differ by at most one) and so does the
    batch. The levels are the kit's seats from a batch-wide ``LevelSeats`` per tier
    (within one of ``level_quota`` per kit); each slot takes, among
    the levels with quota left, the one its family is furthest behind on
    across the batch (``count(f) * mix - count(f, L)``), so every family gets
    the level mix too. Tiers B/C (``bc_share`` > 0): corridors then arenas
    (the odd one a corridor), levels by the same mix."""
    slots, ptr = [], 0
    seen: dict = {}
    got: dict = {}
    seats = {t: LevelSeats(mix) for t in "ABC"}
    for kit in kits:
        n_bc = int(round(per_kit * bc_share))
        n_a = per_kit - n_bc
        left = seats["A"].take(n_a)
        for i in range(n_a):
            fam = FAMILIES[(ptr + i) % len(FAMILIES)]
            seen[fam] = seen.get(fam, 0) + 1
            L = max((L for L in sorted(left) if left[L] > 0),
                    key=lambda L: (seen[fam] * mix[L] - got.get((fam, L), 0), -L))
            left[L] -= 1
            got[fam, L] = got.get((fam, L), 0) + 1
            slots.append({"kit": kit, "tier": "A", "family": fam, "level": L})
        ptr = (ptr + n_a) % len(FAMILIES)
        n_c = n_bc // 2
        for tier, n in (("B", n_bc - n_c), ("C", n_c)):
            q = seats[tier].take(n)
            for L in sorted(q):
                slots += [{"kit": kit, "tier": tier, "family": TIER_FAMILY[tier], "level": L}] * q[L]
    return [dict(sl) for sl in slots]


def plan3_scenes(kits: list[str], per_kit: int, mix: dict, bc_share: float, seed0: int, backfill: int,
                 spec_fn, geometry: str = "natural", candidates: int = 40) -> tuple[list, list, dict]:
    """``(scenes, spares, specs)`` for ``plan3_slots``.

    Seeds count up from ``seed0`` per tier, every seed tried is used up (a
    seed ``spec_fn(tier, family, seed)`` refuses -- it raises -- is skipped),
    so seeds are unique within a tier and pids (``<family>_<seed>``) across
    the plan. ``geometry`` "natural" takes the next seed as it comes, i.e. the
    distribution the family's draw was tuned on (set_M, v7); "extremes" is
    plan2's rule, L1 the easiest and L2 the hardest of ``candidates`` fresh
    seeds by ``geometry_hardness`` (L3 the first). Spares come after every
    primary scene, ``backfill`` per (kit, tier, family, level) group in the
    order the groups first appear, so ``--backfill`` never moves a primary
    seed. ``specs`` maps pid -> the spec object."""
    nxt = {"A": seed0, "B": seed0, "C": seed0}

    def fresh(tier, family):
        while True:
            seed = nxt[tier]
            nxt[tier] += 1
            if seed > SEED_MAX:
                raise RuntimeError(f"tier {tier}: seeds past {SEED_MAX} (rollout names have five digits)")
            try:
                return seed, spec_fn(tier, family, seed)
            except Exception:  # noqa: BLE001 -- the draw refuses this seed for this family
                continue

    def row(sl, seed, note):
        knobs = {**LEVEL_KNOBS[sl["level"]], "start_keepout": PLAN3_START_KEEPOUT}
        return {"pid": f"{sl['family']}_{seed}", "source": "kit", "kit": sl["kit"], "theme": sl["kit"],
                "tier": sl["tier"], "family": sl["family"], "level": sl["level"], "seed": seed, "variant": 0,
                "states": PLAN3_STATES, "knobs": knobs,
                "slot": f"{sl['kit']}/{sl['tier']}/{sl['family']}/L{sl['level']}", "note": note}

    scenes, spares, specs, groups = [], [], {}, []
    for sl in plan3_slots(kits, per_kit, mix, bc_share):
        if geometry == "extremes" and sl["tier"] == "A" and sl["level"] < 3:
            cands = [fresh("A", sl["family"]) for _ in range(candidates)]
            cands.sort(key=lambda c: geometry_hardness(sl["family"], c[1].params))
            seed, sp = cands[0] if sl["level"] == 1 else cands[-1]
            note = f"{['easiest', 'hardest'][sl['level'] - 1]} of {len(cands)}"
        else:
            seed, sp = fresh(sl["tier"], sl["family"])
            note = "natural"
        r = row(sl, seed, note)
        scenes.append(r)
        specs[r["pid"]] = sp
        if r["slot"] not in groups:
            groups.append(r["slot"])
    by_slot = {r["slot"]: r for r in scenes}
    for g in groups:
        sl = {k: by_slot[g][k] for k in ("kit", "tier", "family", "level")}
        for k in range(backfill):
            seed, sp = fresh(sl["tier"], sl["family"])
            r = {**row(sl, seed, "spare"), "spare_rank": k}
            spares.append(r)
            specs[r["pid"]] = sp
    return scenes, spares, specs


def batch1_scenes(src: str, b: Path, start_keepout: float = PLAN3_START_KEEPOUT) -> list[dict]:
    """Batch 1's 42 v1-theme scenes as plan3 rows (``source`` v1, pids unchanged),
    their specs copied into ``<b>/specs``. ``src`` is a batch dir.

    Batch 1 was built with no start keep-out, and 34 of its 42 clouds put
    dressing within 1.5 m of the start or the goal (the build gate); the
    rows therefore get ``start_keepout`` (1.5 m, the kits' disc), which
    re-scatters them. ``start_keepout`` 0 rebuilds batch 1's scatter exactly
    (the same placements), and ``--cloud-voxel 0.08 --cloud-zmax 4.5`` its
    clouds byte for byte -- but only the 37 scenes without spheres: batch 1
    predates build_course's canonical sphere order, so arena_3000-3004 get a
    new cloud (arena_3003: 7,556 of 340,626 points, all on spheres) and their
    batch 1 routes and labels do not carry over."""
    loc = Path(src).expanduser()
    out = []
    for s in json.loads((loc / "plan.json").read_text())["scenes"]:
        shutil.copy(loc / "specs" / f"{s['pid']}.json", b / "specs" / f"{s['pid']}.json")
        out.append({"pid": s["pid"], "source": "v1", "theme": s["theme"], "tier": s["tier"], "family": s["family"],
                    "level": s["level"], "seed": s["seed"], "variant": 0, "states": PLAN3_STATES,
                    "knobs": {"clutter": s["clutter"], "keepout": s["keepout"], "hang": s["hang"],
                              "start_keepout": start_keepout},
                    "slot": f"v1/{s['pid']}", "note": f"batch1 ({s.get('note', '')})".strip()})
    return out


def cmd_plan3(a) -> int:
    """The v7 mass run's plan: ``--per-kit`` scenes for each of ``--kits`` (tier A,
    plus corridors/arenas with ``--bc-share``), ``--backfill`` spares per
    (kit, tier, family, level), optionally batch 1's scenes (``--include-batch1``).
    Writes ``plan.json`` (``scenes``, ``spares``) and ``specs/<pid>.json``."""
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    os.environ.pop("SUPERFLY_SCENE_CLOUDS", None)             # specs are built on primitives, never a course cloud
    import specs as dmd  # noqa: PLC0415
    b = a.batch
    (b / "specs").mkdir(parents=True, exist_ok=True)
    kits = a.kits or kit_names()
    mix = dict(zip((1, 2, 3), (float(v) for v in a.level_mix.split(","))))
    t0 = time.time()
    scenes, spares, specs = plan3_scenes(kits, a.per_kit, mix, a.bc_share, a.seed0, a.backfill,
                                         lambda tier, fam, seed: dmd.build_spec(5, tier, fam, seed, "train"),
                                         a.geometry, a.candidates)
    for pid, sp in specs.items():
        (b / "specs" / f"{pid}.json").write_text(sp.to_json())
    if a.include_batch1:
        scenes += batch1_scenes(a.include_batch1, b, a.batch1_start_keepout)
    pids = [r["pid"] for r in scenes + spares]
    dup = sorted({p for p in pids if pids.count(p) > 1})
    if dup:
        raise RuntimeError(f"duplicate pids in the plan: {dup[:10]}")
    plan = {"batch": b.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "planner": "plan3",
            "kits": kits, "per_kit": a.per_kit, "level_mix": mix, "bc_share": a.bc_share, "seed0": a.seed0,
            "backfill": a.backfill, "geometry": a.geometry, "states": PLAN3_STATES, "level_knobs": LEVEL_KNOBS,
            "include_batch1": a.include_batch1, "batch1_start_keepout": a.batch1_start_keepout,
            "scenes": scenes, "spares": spares}
    (b / "plan.json").write_text(json.dumps(plan, indent=1))
    by = {}
    for r in scenes:
        by.setdefault(r.get("kit") or "v1", []).append(r)
    for k, rs in by.items():
        lv = {L: sum(r["level"] == L for r in rs) for L in (1, 2, 3)}
        print(f"{k:13s} {len(rs):4d} scenes  L1/L2/L3 {lv[1]}/{lv[2]}/{lv[3]}  "
              f"B {sum(r['tier'] == 'B' for r in rs)} C {sum(r['tier'] == 'C' for r in rs)}")
    print(f"{len(scenes)} scenes + {len(spares)} spares over {len(kits)} kits -> {b / 'plan.json'} "
          f"({time.time() - t0:.0f} s; {len(scenes) * PLAN3_STATES} states)")
    return 0


def resolve_scenes(plan: dict, gates: dict, infra_is_failure: bool = False) -> dict:
    """Which scene fills each slot, given the gate rows (``pid -> row`` with ``ok``).

    A primary that passes keeps its slot. One whose row says ``infra`` (the
    build was refused, killed, dirty, unpacked, of stale code: nothing about
    the scene, ``build_gates``) waits for a *rebuild*. One that fails its own
    gates takes the first spare of its (kit, tier, family, level) group not
    yet taken by an earlier failure of the group: a spare that passes
    replaces it, one that fails is used up and the next is tried, one not
    gated yet (or waiting for a rebuild) leaves the slot *pending*. No spare
    left (or a row with no ``slot``: plan/plan2): the slot is dropped.
    ``infra_is_failure`` spends spares on infra rows too (after the rebuilds
    were tried: a scene that kills Blender every time). Deterministic in the
    plan and the gate rows. Returns ``{complete, scenes, pending, rebuild,
    replaced, dropped, failed, in_play}``; ``in_play`` is every spare the rule
    reached (replacing, used up, or reserved: still to gate or rebuild), each
    with ``replaces`` -- the spares to build (``pipeline.py build`` builds
    exactly these)."""
    spares: dict = {}
    for r in sorted(plan.get("spares", []), key=lambda r: r.get("spare_rank", 0)):
        spares.setdefault(r["slot"], []).append(r)

    def state(g):
        if g is None:
            return "pending"
        if g.get("ok"):
            return "ok"
        return "rebuild" if g.get("infra") and not infra_is_failure else "failed"

    used: dict = {}
    out, pending, rebuild, replaced, dropped, failed, in_play = [], [], [], {}, [], {}, []
    for s in plan["scenes"]:
        g = gates.get(s["pid"])
        st = state(g)
        if st == "ok":
            out.append(s)
            continue
        if st in ("pending", "rebuild"):
            (pending if st == "pending" else rebuild).append(s["pid"])
            continue
        failed[s["pid"]] = g.get("fails", [])
        slot = s.get("slot")
        queue = spares.get(slot, []) if slot is not None else []
        k = used.get(slot, 0)
        while k < len(queue):
            sp = queue[k]
            gs = gates.get(sp["pid"])
            k += 1
            in_play.append({**sp, "replaces": s["pid"]})
            sst = state(gs)
            if sst in ("pending", "rebuild"):  # reserved for this slot until it is gated / rebuilt
                (pending if sst == "pending" else rebuild).append(sp["pid"])
                break
            if sst == "ok":
                out.append({**sp, "replaces": s["pid"]})
                replaced[s["pid"]] = sp["pid"]
                break
            failed[sp["pid"]] = gs.get("fails", [])
        else:
            dropped.append(s["pid"])
        used[slot] = k
    return {"complete": not pending and not rebuild, "scenes": out, "pending": pending, "rebuild": rebuild,
            "replaced": replaced, "dropped": dropped, "failed": failed, "in_play": in_play}


def load_plan(b: Path, only=None) -> list[dict]:
    """The plan's scenes; with ``only``, any named scene of the plan *or its spares* (plan3)."""
    plan = json.loads((b / "plan.json").read_text())
    if not only:
        return plan["scenes"]
    return [s for s in plan["scenes"] + plan.get("spares", []) if s["pid"] in only]


def scene_knobs(s: dict) -> dict:
    """A plan row's build knobs: plan3 nests them under ``knobs``, plan/plan2 kept them flat."""
    k = s.get("knobs") or s
    return {"clutter": k["clutter"], "keepout": k["keepout"], "hang": k["hang"],
            "start_keepout": k.get("start_keepout", 0.0)}


# --- build -----------------------------------------------------------

def build_cmd(s: dict, out: Path, spec: Path, a) -> list:
    """The ``build_course.py`` command line of one plan row (run it with
    ``env=BLENDER_ENV``). Kit scenes and plan3's v1 rows are packed
    (``--pack-max-res`` downscales; ``build_gates`` checks the same rule);
    plan/plan2's own v1 rows build as batch 1 did. ``--require-clean``
    unless ``--allow-dirty``."""
    k = scene_knobs(s)
    cmd = [BLENDER, "-b", "--factory-startup", *BLENDER_PY, *(["-t", a.threads] if a.threads else []),
           "-P", HERE / "build_course.py", "--"]
    if s.get("kit"):                    # a place kit, the v2 library
        cmd += ["--kit", s["kit"], "--variant", s.get("variant", 0), "--assets", LIBRARY]
        if s.get("kit_args"):                       # backdrop kits (wenshan): scene, placement
            cmd += ["--kit-args", json.dumps(s["kit_args"], sort_keys=True)]
        if a.cpu:
            cmd += ["--cpu", "--preview-res", "640x360", "--preview-samples", "16", "--threads", a.threads]
    else:                               # a v1 theme (batch 1)
        cmd += ["--assets", ASSETS, "--theme", s["theme"]]
    if s.get("kit") or s.get("source") == "v1":
        cmd += ["--pack", *(["--pack-max-res", a.pack_max_res] if a.pack_max_res else []),
                *(["--pack-cache", a.pack_cache] if a.pack_max_res and getattr(a, "pack_cache", None) else [])]
    if not getattr(a, "allow_dirty", False):    # contract 6: pipeline builds refuse a dirty tree
        cmd += ["--require-clean"]
    return [*cmd, "--spec", spec, "--out", out, "--clutter", k["clutter"], "--keepout", k["keepout"],
            "--hang", k["hang"], "--level", str(s["level"]), "--start-keepout", k["start_keepout"],
            *([] if a.no_preview else ["--preview"])]


def cmd_build(a) -> int:
    """Build every scene of the plan (or ``--only``, which may name spares):
    ``courses/<pid>/`` plus a build record ``builds/<pid>.json`` (rc, wall
    seconds, log) that ``gates`` reads. A scene with a ``course.json`` is
    skipped unless ``--force``, which deletes it first so a failed rebuild
    cannot pass for a good one. Refuses a dirty tree (exit 3 per scene, an
    ``infra`` gate row) unless ``--allow-dirty``."""
    b = a.batch
    (b / "builds").mkdir(parents=True, exist_ok=True)

    def one(s):
        out = b / "courses" / s["pid"]
        if (out / "course.json").exists() and not a.force:
            return True
        (out / "course.json").unlink(missing_ok=True)
        t0 = time.time()
        log = b / "logs" / f"build_{s['pid']}.log"
        rc = run(build_cmd(s, out, b / "specs" / f"{s['pid']}.json", a), log, env=BLENDER_ENV)
        ok = rc == 0 and (out / "course.json").exists()
        rec = {"pid": s["pid"], "rc": rc, "ok": ok, "seconds_wall": round(time.time() - t0, 1), "log": str(log),
               "require_clean": not a.allow_dirty, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (b / "builds" / f"{s['pid']}.json").write_text(json.dumps(rec))
        print(f"build {s['pid']:22s} {s.get('theme', ''):12s} L{s['level']} {'ok' if ok else f'FAILED rc={rc}'} "
              f"{time.time() - t0:5.0f} s", flush=True)
        return ok

    with ThreadPoolExecutor(max(1, a.workers)) as ex:
        oks = list(ex.map(one, load_plan(b, a.only)))
    return 0 if all(oks) else 1


def pack_check(course: Path, out: Path, timeout: float = 900.0) -> dict | None:
    """``pack_check.py`` on ``course/course.blend`` in a fresh Blender (``--pack-check``):
    its result, written to ``out``; None when there is no ``course.blend`` here."""
    blend = course / "course.blend"
    if not blend.exists():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [BLENDER, "-b", str(blend), "--factory-startup", *BLENDER_PY, "-t", "2", "-P", str(HERE / "pack_check.py"),
           "--", "--out", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **BLENDER_ENV})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout:.0f} s"}
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PACK_CHECK ")), None)
    if line is None:
        return {"ok": False, "error": f"no PACK_CHECK line (rc {r.returncode}): {r.stdout[-300:]}"}
    return json.loads(line[len("PACK_CHECK "):])


def _gate_one(args):
    course, s, rec, kw, check_pack, pc_dir = args
    sys.path.insert(0, str(HERE))
    import build_gates as BG  # noqa: PLC0415
    try:
        pc = pack_check(course, pc_dir / f"{s['pid']}.json") if check_pack else None
        return BG.gate_course(course, s, rec, pack_check=pc, **kw)
    except Exception as e:  # noqa: BLE001 -- the gate crashed: infra (rebuild or fix), never a spent spare
        return BG.error_row(s["pid"], e)


def cmd_gates(a) -> int:
    """Gate every built scene (plan scenes and spares with a course or a build
    record): one row per scene in ``gates/<pid>.json`` (``build_gates.gate_course``),
    all rows in ``gates.json``. ``--allow-dirty`` stops gating the recorded
    provenance (trial batches), ``--expect-git-sha`` fails a course built from
    another commit, ``--pack-check`` opens every local ``course.blend`` in a
    fresh Blender (``pack_check.py``; results in ``pack_check/<pid>.json``)."""
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    sys.path.insert(0, str(HERE))
    import build_gates as BG  # noqa: PLC0415
    b = a.batch
    plan = json.loads((b / "plan.json").read_text())
    rows = [s for s in plan["scenes"] + plan.get("spares", []) if not a.only or s["pid"] in a.only]
    (b / "gates").mkdir(parents=True, exist_ok=True)
    kw = {"budget": a.budget or BG.OBSTACLE_BUDGET, "max_build_s": a.max_build_s or BG.MAX_BUILD_S,
          "check_provenance": not a.allow_dirty, "expect": {"git_sha": a.expect_git_sha}}
    todo = []
    for s in rows:
        c, rec = b / "courses" / s["pid"], b / "builds" / f"{s['pid']}.json"
        if not c.exists() and not rec.exists():
            continue
        todo.append((c, s, json.loads(rec.read_text()) if rec.exists() else None, kw, a.pack_check, b / "pack_check"))
    with ProcessPoolExecutor(max(1, a.workers)) as ex:
        for g in ex.map(_gate_one, todo):
            (b / "gates" / f"{g['pid']}.json").write_text(json.dumps(g, indent=1))
            print(f"gate {g['pid']:24s} {'ok  ' if g['ok'] else 'FAIL'} {','.join(g['fails']):22s} "
                  f"obs {g.get('obstacle_points') or 0:>7,}  start {g.get('start_clear_m')}  goal {g.get('goal_clear_m')}"
                  f"  build {g.get('cpu_seconds_build') or g.get('seconds_build')} s"
                  + (f"  infra: {g['infra']}" if g.get("infra") else ""), flush=True)
    allg = [json.loads(f.read_text()) for f in sorted((b / "gates").glob("*.json"))]
    (b / "gates.json").write_text(json.dumps(allg, indent=1))
    bad = [g for g in allg if not g["ok"]]
    shas: dict = {}
    for g in allg:
        if g.get("git_sha"):
            shas.setdefault((g["git_sha"][:10], g.get("theme") if g.get("kit") is None else "kits",
                             (g.get("library_sha") or "")[:10]), []).append(g["pid"])
    for (sha, src, lib), pids in sorted(shas.items()):
        print(f"built from {sha} ({src}, library {lib}): {len(pids)} scenes")
    if len({k[0] for k in shas}) > 1:
        print("NOTE: the batch mixes commits (--expect-git-sha gates one)")
    print(f"{len(allg) - len(bad)}/{len(allg)} gated scenes pass, {sum(bool(g.get('infra')) for g in bad)} of the "
          f"failures infra (rebuild) -> {b / 'gates.json'}")
    return 1 if bad else 0


def load_gates(b: Path) -> dict:
    return {g["pid"]: g for g in (json.loads(f.read_text()) for f in sorted((b / "gates").glob("*.json")))}


def cmd_backfill(a) -> int:
    """``resolve_scenes`` over the gate rows: writes ``scenes.json`` (the scene
    list the later stages run) and prints the spares still to build and gate
    and the infra rows to rebuild (``build --only ... [--force]`` then
    ``gates`` then ``backfill`` again). Exit 2 while anything is pending or
    waits for a rebuild; ``--infra-as-failure`` spends spares on what still
    fails for infra reasons after the rebuilds."""
    b = a.batch
    plan = json.loads((b / "plan.json").read_text())
    res = resolve_scenes(plan, load_gates(b), infra_is_failure=a.infra_as_failure)
    (b / "scenes.json").write_text(json.dumps({"batch": b.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "infra_as_failure": a.infra_as_failure, **res}, indent=1))
    print(f"{len(res['scenes'])}/{len(plan['scenes'])} slots filled, {len(res['replaced'])} by spares, "
          f"{len(res['dropped'])} dropped, {len(res['pending'])} pending, {len(res['rebuild'])} to rebuild "
          f"-> {b / 'scenes.json'}")
    if res["pending"]:
        print("pending (build --only ..., then gates): " + " ".join(res["pending"]))
    if res["rebuild"]:
        print("rebuild (infra: see gates/<pid>.json; build --only ... --force, then gates): " + " ".join(res["rebuild"]))
    if res["dropped"]:
        print("dropped: " + " ".join(res["dropped"]))
    return 2 if res["pending"] or res["rebuild"] else 0


def load_scenes(b: Path) -> list[dict]:
    """The scenes the later stages run: ``scenes.json`` (after ``backfill``) when it
    exists, else the plan's scenes."""
    f = b / "scenes.json"
    return json.loads(f.read_text())["scenes"] if f.exists() else json.loads((b / "plan.json").read_text())["scenes"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["plan", "plan2", "plan3", "build", "gates", "backfill"])
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="scene ids (<family>_<seed>)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--python", default=None, help="interpreter for the repo-side scripts (default: this one)")
    ap.add_argument("--seed0", type=int, default=None, help="first seed (plan/plan2 3000, plan3 10000)")
    ap.add_argument("--candidates", type=int, default=40, help="(plan, plan2, plan3 --geometry extremes)")
    ap.add_argument("--per-kit", type=int, default=72, help="(plan3) scenes per kit")
    ap.add_argument("--level-mix", default="0.4,0.4,0.2", help="(plan3) L1,L2,L3 shares of each kit's scenes")
    ap.add_argument("--bc-share", type=float, default=0.0,
                    help="(plan3) share of each kit's scenes that are tier-B corridors / tier-C arenas (half each)")
    ap.add_argument("--backfill", type=int, default=2,
                    help="(plan3) spare seeds per (kit, tier, family, level) for scenes failing the build gates")
    ap.add_argument("--geometry", choices=["natural", "extremes"], default="natural",
                    help="(plan3) natural = the next seed as drawn; extremes = plan2's easiest/hardest of --candidates")
    ap.add_argument("--include-batch1", default=None,
                    help="(plan3) add a batch-1 plan's scenes as v1 rows (pids unchanged) from this batch dir")
    ap.add_argument("--batch1-start-keepout", type=float, default=PLAN3_START_KEEPOUT,
                    help="(plan3) start/goal keep-out of the batch 1 rows (0 = batch 1's own scatter)")
    ap.add_argument("--corridors", type=int, default=8)
    ap.add_argument("--arenas", type=int, default=6)
    ap.add_argument("--kits", nargs="*", default=None, help="(plan2, plan3) kits to plan; default: every kit with a module")
    ap.add_argument("--cpu", action="store_true", help="(build) Cycles on the CPU for the previews (gs2)")
    ap.add_argument("--threads", type=int, default=8, help="(build) Blender threads (-t; and render threads with --cpu)")
    ap.add_argument("--workers", type=int, default=1, help="(build, gates) scenes at once")
    ap.add_argument("--no-preview", action="store_true", help="(build) no preview renders")
    ap.add_argument("--pack-max-res", type=int, default=0,
                    help="(build) downscale packed 8-bit textures to this longer side (1024 for the 640x480 camera)")
    ap.add_argument("--pack-cache", type=Path, default=None,
                    help="(build) share downscaled textures between builds here (e.g. ~/.cache/envgen/pack_cache)")
    ap.add_argument("--require-clean", action="store_true",
                    help="(build) the default now: refuse a dirty checkout, kit env hook or salted hash (contract 6)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="(build) do not pass --require-clean; (gates) do not gate the recorded provenance. Trial runs")
    ap.add_argument("--expect-git-sha", default=None, help="(gates) fail courses built from another commit")
    ap.add_argument("--pack-check", action="store_true",
                    help="(gates) open each local course.blend in a fresh Blender (pack_check.py) and gate on it")
    ap.add_argument("--infra-as-failure", action="store_true",
                    help="(backfill) spend spares on slots still failing for infra reasons (after rebuilding them)")
    ap.add_argument("--budget", type=int, default=0, help="(gates) obstacle-point budget (0 = build_gates default)")
    ap.add_argument("--max-build-s", type=float, default=0.0, help="(gates) build seconds (0 = build_gates default)")
    a = ap.parse_args(argv)
    if a.require_clean and a.allow_dirty:
        ap.error("--require-clean and --allow-dirty contradict each other")
    a.batch = a.batch.expanduser().resolve()
    if a.seed0 is None:
        a.seed0 = 10000 if a.stage == "plan3" else 3000
    return {"plan": cmd_plan, "plan2": cmd_plan2, "plan3": cmd_plan3, "build": cmd_build, "gates": cmd_gates,
            "backfill": cmd_backfill}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
