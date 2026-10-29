"""The flight band as ONE parameter (PAIR_DRAW_PLAN.md section 4 item 2, 2026-10-02).

Every altitude constant of the stack -- the draw's validity band and its 15/60/25 mixture, the
labeller's floor and climb ceiling (and the terrain per-state band that re-references them), the
cloud top a course is cut at (and the Wenshan kit's removal / collide / PLY cross-check bands that
read it), the QA excursion ceiling, the completeness band, the export retag guard, the sim's
``Z_REF`` / ``Z_CRASH``, the rollout start/goal band and the build gates' clearance band -- is
DERIVED from one :class:`Band`, so switching the preset moves all of them together.

Presets:

* ``v9`` (the default): today's values exactly -- floor 0.4, ceiling 4.25 (flight_z 1.75 +
  climb-max 2.5), draw 0.4-3.85, cloud top 4.75. Nothing changes until the preset is switched.
* ``v10`` (the plan): floor 0.4, ceiling 5.5 (climb-max 3.75), core draw 0.4-4.0 with the
  15/60/25 mixture over 0.4-1 / 1-2.5 / 2.5-4, a 4-7 m high tail for ~6.5 % of the states in a
  quarter of the scenes (ceiling 8.0 there), cloud top = ceiling + 0.75 (6.25; 8.75 in tail scenes).
* ``v11`` (Daniel 2026-10-08, the altitude resampling): the v10 tail numbers on EVERY scene -- floor 0.4,
  ceiling 8.0 (climb-max 6.25), cloud top 8.75 -- no tail; the start height over the local ground drawn from
  five layers (10 % 0.5-1, 40 % 1-2.5, 25 % 2.5-4, 15 % 4-6, 10 % 6-8 m, uniform in each: ``start_mix``) and
  the goal height over ITS ground independently of the start (``goal_mix``: 50 % U(1.2, 2.4); 25 % the start's
  height + U(-1, 1); 25 % U(1, 8); clipped to ``goal_clip`` 0.5-8). The labeller's cost terms stay v10's: label
  units run ``--band v10 --climb-max 6.25`` exactly like v10 tail scenes.

Selecting it: ``$SUPERFLY_BAND`` (read at import, inherited by workers and Blender subprocesses),
or :func:`use` from a ``--band`` flag -- :func:`use` recomputes every module constant that was
bound with :func:`bind`, so a process that parsed its flag after importing the modules still ends
up consistent. Values captured in function defaults at definition time are not rebound: a caller
that must follow the switch passes the module constant explicitly (``draw_v9.Geo`` reads
``Z_BAND`` at call time for that reason).

Dependency-free on purpose (no numpy): Blender's Python loads it by path (``blender/band_shim``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import os

ENV = "SUPERFLY_BAND"
DEFAULT = "v9"


@dataclass(frozen=True)
class Band:
    name: str
    floor: float = 0.4               # the labeller's floor (data_make_dataset.NO_BAND_FLOOR) and the draw's lowest state
    ceiling: float = 4.25            # the labeller's hard ceiling above the patch floor (flight_z + climb_max)
    draw_hi: float | None = None     # the core draw's top; None = ceiling - r_body + 0.05 (today's 3.85)
    mix: tuple = (0.15, 0.60, 0.25)  # shares of the three altitude layers ...
    mix_edges: tuple = (1.0, 2.5)    # ... (floor, e0), (e0, e1), (e1, draw_hi)
    tail: tuple | None = None        # the high tail (lo, hi), drawn only in tail scenes
    tail_share: float = 0.0          # share of a tail scene's states drawn in the tail
    tail_scene_frac: float = 0.0     # share of scenes that get the tail (deterministic in the scene key)
    tail_ceiling: float | None = None  # the labeller's ceiling in a tail scene (None = ceiling)
    cloud_pad: float = 0.5           # cloud top = ceiling + pad (today 4.25 + 0.5 = 4.75)
    r_body: float = 0.45             # the labeller's body radius (recipe v9)
    flight_z: float = 1.75           # patch.FLIGHT_Z: the ceiling is flight_z + climb_max
    sim_pad: float = 0.25            # the sim's Z_REF / Z_CRASH top and the retag guard = ceiling - pad (today 4.0)
    #: v11: the start-height layers ((share, lo, hi), ...) replacing ``mix`` / ``mix_edges`` (None = the 3-layer mix)
    start_mix: tuple | None = None
    #: v11: the goal-height rule ((share, kind, lo, hi), ...), kind "abs" = U(lo, hi) over the goal's own ground,
    #: "rel" = the start's height over its ground + U(lo, hi); None = pair_draw's GOAL_HAG_MIX (80/15/5)
    goal_mix: tuple | None = None
    goal_clip: tuple | None = None   # v11: the goal height over its ground is clipped to this (lo, hi)
    #: v11c: pair_draw class-quota overrides ((class, pct), ...); the other classes are scaled so the shares sum to 100
    quota_pct: tuple | None = None
    #: v12: a goal that fails validity keeps its drawn ALTITUDE and gets a new horizontal position (pair_draw.propose)
    goal_xy_resample: bool = False
    #: v12: class 8 "vertical" never set (the altitude rule alone governs dz; class quotas never cap it)
    vertical_class: bool = True
    #: v12: pair_helpers perturbation kinds left out (the "altitude" kind would break the same-altitude relation)
    perturb_exclude: tuple = ()
    #: v12: a pair counts as "same altitude" for the accepted-set quota when |goal_hag - start_hag| <= this
    same_alt_tol: float | None = None

    # ---- derived: the labeller
    @property
    def z_band(self) -> tuple:
        """``draw_v9.Z_BAND``: the labeller's (floor, ceiling) on a flat floor; ``terrain.z_band``
        re-references it per state."""
        return (self.floor, self.ceiling)

    @property
    def climb_max(self) -> float:
        """``--climb-max`` of recipes v7m / v8 / v9: the ceiling above the flight altitude."""
        return round(self.ceiling - self.flight_z, 6)

    def ceiling_for(self, tail: bool = False) -> float:
        return float(self.tail_ceiling) if (tail and self.tail_ceiling is not None) else self.ceiling

    def climb_max_for(self, tail: bool = False) -> float:
        return round(self.ceiling_for(tail) - self.flight_z, 6)

    def z_band_for(self, tail: bool = False) -> tuple:
        return (self.floor, self.ceiling_for(tail))

    # ---- derived: the cloud
    @property
    def cloud_top(self) -> float:
        """``build_course.CLOUD_ZMAX``: the obstacle cloud is cut here (above the local / highest ground)."""
        return round(self.ceiling + self.cloud_pad, 6)

    def cloud_top_for(self, tail: bool = False) -> float:
        return round(self.ceiling_for(tail) + self.cloud_pad, 6)

    @property
    def label_top(self) -> float:
        """The highest point a label body reaches: ceiling + r_body (``reskin.LABEL_TOP_M``)."""
        return round(self.ceiling + self.r_body, 6)

    # ---- derived: the draw
    @property
    def alt_draw(self) -> tuple:
        """``pair_draw.ALT_BAND``: a drawn state's height over its local ground (validity)."""
        hi = self.draw_hi if self.draw_hi is not None else self.ceiling - self.r_body + 0.05
        return (self.floor, round(float(hi), 6))

    @property
    def alt_a(self) -> tuple:
        """``draw_v9.ALT_A``: family A's validity band."""
        return (0.5, self.alt_draw[1])

    @property
    def alt_mix(self) -> tuple:
        """``pair_draw.ALT_MIX``: ((share, lo, hi), ...) of the three layers."""
        if self.start_mix is not None:
            return tuple(tuple(m) for m in self.start_mix)
        lo, hi = self.alt_draw
        e0, e1 = self.mix_edges
        return ((self.mix[0], lo, e0), (self.mix[1], e0, e1), (self.mix[2], e1, hi))

    def alt_mix_for(self, tail: bool = False) -> tuple:
        """The mixture in a tail scene: the core layers scaled by (1 - tail_share) plus the tail."""
        if not tail or not self.tail or self.tail_share <= 0:
            return self.alt_mix
        s = 1.0 - self.tail_share
        return tuple((sh * s, lo, hi) for sh, lo, hi in self.alt_mix) + ((self.tail_share, *self.tail),)

    def alt_draw_for(self, tail: bool = False) -> tuple:
        return (self.floor, float(self.tail[1])) if (tail and self.tail) else self.alt_draw

    @property
    def hist_hag(self) -> list:
        """``pair_draw.HIST_EDGES['hag']``."""
        lo, hi = self.alt_draw
        if self.start_mix is not None:
            return sorted({float(lo), float(hi), *(float(m[1]) for m in self.start_mix), *(float(m[2]) for m in self.start_mix)})
        edges = [lo, 1.0, 1.5, 2.0, 2.5, 3.0]
        edges = [e for e in edges if e < hi] + [hi]
        if self.tail:
            edges += [float(self.tail[1])]
        return edges

    def tail_scene(self, key) -> bool:
        """Whether scene ``key`` (its seq / pid) draws the high tail: a deterministic share
        ``tail_scene_frac`` of all scenes, by hash of (band name, key)."""
        if not self.tail or self.tail_scene_frac <= 0:
            return False
        h = int.from_bytes(hashlib.sha256(f"tail|{self.name}|{key}".encode()).digest()[:8], "big")
        return (h % 10_000) < int(round(self.tail_scene_frac * 10_000))

    # ---- derived: sim, export, QA, gates, rollouts
    @property
    def sim_z_ref(self) -> tuple:
        """``sim_episode.Z_REF``: the executor's velocity-setpoint z band."""
        return (0.5, round(self.ceiling - self.sim_pad, 6))

    @property
    def sim_z_crash(self) -> tuple:
        """``sim_episode.Z_CRASH``."""
        return (0.3, round(self.ceiling - self.sim_pad, 6))

    @property
    def retag(self) -> tuple:
        """``chunks.RETAG_FLOOR_Z`` / ``RETAG_CEIL_Z``: path samples outside it never validate over/under."""
        return (0.15, round(self.ceiling - self.sim_pad, 6))

    @property
    def rollout(self) -> tuple:
        """``rollouts.Z_BAND``: start / goal heights of the student rollouts -- the nominal middle layer
        of the draw, 0.2 m under / 0.3 m over its edges (today 0.8-2.8)."""
        e0, e1 = self.mix_edges
        return (round(e0 - 0.2, 6), round(e1 + 0.3, 6))

    @property
    def gates(self) -> tuple:
        """``build_gates.BAND``: the band the start/goal clearance gates look at (today 0.3-3.2 =
        ceiling - 1.05: the v7 envelope minus the body and a 0.7 m margin)."""
        return (0.3, round(self.ceiling - 1.05, 6))

    @property
    def qa_zmax_fallback(self) -> float:
        """``pipeline_qa.ZMAX_FALLBACK`` / ``finish_frames.ZMAX``: the cloud top a course.json without
        ``cloud.zmax_m`` is read as (0.25 m under the real top: strict side)."""
        return round(self.cloud_top - 0.25, 6)

    @property
    def reach_top_v7(self) -> float:
        """``_industrial_util.REACH_TOP``: the tallest wall top the v7 drone (r_body 0.35) still clears."""
        return round(self.ceiling - 0.35, 6)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("start_mix", "goal_mix", "goal_clip", "quota_pct", "same_alt_tol"):  # v11 / v12-only fields: absent on
            if d.get(k) is None:                                                    # v9 / v10 (records unchanged)
                d.pop(k, None)
        for k, dflt in (("goal_xy_resample", False), ("vertical_class", True), ("perturb_exclude", ())):
            if d.get(k) == dflt:
                d.pop(k, None)
        d.update(z_band=list(self.z_band), climb_max=self.climb_max, cloud_top=self.cloud_top,
                 alt_draw=list(self.alt_draw), alt_mix=[list(m) for m in self.alt_mix],
                 sim_z_ref=list(self.sim_z_ref), retag=list(self.retag), rollout=list(self.rollout),
                 gates=list(self.gates), label_top=self.label_top,
                 tail_cloud_top=self.cloud_top_for(True), tail_climb_max=self.climb_max_for(True))
        return d


PRESETS = {
    "v9": Band("v9"),
    "v10": Band("v10", floor=0.4, ceiling=5.5, draw_hi=4.0, mix=(0.15, 0.60, 0.25), mix_edges=(1.0, 2.5),
                tail=(4.0, 7.0), tail_share=0.065, tail_scene_frac=0.25, tail_ceiling=8.0, cloud_pad=0.75),
    "v11": Band("v11", floor=0.4, ceiling=8.0, draw_hi=8.0, mix=(0.15, 0.60, 0.25), mix_edges=(1.0, 2.5), cloud_pad=0.75,
                start_mix=((0.10, 0.5, 1.0), (0.40, 1.0, 2.5), (0.25, 2.5, 4.0), (0.15, 4.0, 6.0), (0.10, 6.0, 8.0)),
                goal_mix=((0.50, "abs", 1.2, 2.4), (0.25, "rel", -1.0, 1.0), (0.25, "abs", 1.0, 8.0)), goal_clip=(0.5, 8.0),
                quota_pct=((8, 15.0),)),
    # v12 (Wenshan + Daniel, 2026-10-08): start height uniform 0.5-8.0 (per-metre layers, the accepted set held to them
    # per scene by the alt-quota cells), goal 80 % same altitude (start +- U(0.2)) / 20 % independent U(0.5, 8.0), the
    # 80/20 split held on the accepted set; a failing goal is moved horizontally, never vertically; class 8 off; no
    # altitude perturbation; labeller ceiling 8.25 (climb-max 6.5) so states / goals reach 8.0 under the 0.25 m
    # margin; cloud top 8.75 (the courses rebuilt for v11)
    "v12": Band("v12", floor=0.4, ceiling=8.25, draw_hi=8.0, mix=(0.15, 0.60, 0.25), mix_edges=(1.0, 2.5), cloud_pad=0.5,
                start_mix=((0.5 / 7.5, 0.5, 1.0),) + tuple((1.0 / 7.5, float(a), float(a + 1)) for a in range(1, 8)),
                goal_mix=((0.80, "rel", -0.2, 0.2), (0.20, "abs", 0.5, 8.0)), goal_clip=(0.5, 8.0),
                quota_pct=((8, 0.0),), goal_xy_resample=True, vertical_class=False, perturb_exclude=("altitude",),
                same_alt_tol=0.25),
}

_current: str | None = None
_BOUND: list = []           # (globals dict, name, fn)
_HOOKS: list = []           # fn(band) called after every switch


def name() -> str:
    n = _current or os.environ.get(ENV) or DEFAULT
    if n not in PRESETS:
        raise ValueError(f"{ENV}={n!r}: not a band preset ({sorted(PRESETS)})")
    return n


def current() -> Band:
    """The band in force: :func:`use`'s, else ``$SUPERFLY_BAND``, else ``v9``."""
    return PRESETS[name()]


def get(n: str | None) -> Band:
    return current() if not n else PRESETS[n] if n in PRESETS else _raise(n)


def _raise(n):
    raise ValueError(f"band {n!r}: not a preset ({sorted(PRESETS)})")


def use(n: str | None) -> Band:
    """Switch the band for this process (and, through the environment, its children), and recompute
    every constant bound with :func:`bind`. ``None`` / '' = leave as is."""
    global _current
    if not n:
        return current()
    if n not in PRESETS:
        _raise(n)
    _current = n
    os.environ[ENV] = n
    b = PRESETS[n]
    for g, k, fn in _BOUND:
        g[k] = fn(b)
    for h in _HOOKS:
        h(b)
    return b


def bind(g: dict, **derivs) -> Band:
    """``bind(globals(), Z_BAND=lambda b: b.z_band)``: set each module constant from the current band
    now and register it, so :func:`use` keeps it in step."""
    b = current()
    for k, fn in derivs.items():
        g[k] = fn(b)
        _BOUND.append((g, k, fn))
    return b


def on_change(fn) -> None:
    """Register ``fn(band)`` to run after every :func:`use` (for constants that live in nested
    structures, e.g. a recipe dict); it is also called once now."""
    _HOOKS.append(fn)
    fn(current())


def with_overrides(b: Band, **kw) -> Band:
    return replace(b, **kw)
