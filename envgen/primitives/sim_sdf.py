"""Exact signed distance to the primitives of a patch/arena spec, for the sim.

The spec is the list of primitive dicts ``scenes/arena.py`` writes (``sphere``,
``box`` with an xyz-Euler ``rpy``, ``cyl_v``, ``cyl_h``, ``wall`` with slots).
``PrimitiveSdf`` evaluates the true distance to the nearest surface (negative
inside); ``CloudSdf`` is the fallback for scenes that exist only as a cloud
(distance to the nearest sampled point, so ≈ the surface distance to within
the sampling spacing, and ≥ 0 inside solid-sampled walls).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def euler_xyz(roll, pitch, yaw) -> np.ndarray:
    """The convention of ``scenes/arena._euler_xyz``: R = Rx @ Ry @ Rz."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _sd_box_local(q: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance of points q (M,3) in the box frame to an axis box."""
    d = np.abs(q) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


def _sd_sphere(p, prim):
    return np.linalg.norm(p - np.asarray(prim["c"], float), axis=1) - float(prim["r"])


def _sd_box(p, prim):
    R = euler_xyz(*prim.get("rpy", (0.0, 0.0, 0.0)))
    q = (p - np.asarray(prim["c"], float)) @ R          # world -> box frame (R^T applied)
    return _sd_box_local(q, np.asarray(prim["half"], float))


def _sd_cyl_v(p, prim):
    c = np.asarray(prim["c"], float)
    r, z0, z1 = float(prim["r"]), float(prim["z0"]), float(prim["z1"])
    dr = np.linalg.norm(p[:, :2] - c[None, :], axis=1) - r
    dz = np.abs(p[:, 2] - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0)
    d = np.stack([dr, dz], 1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.max(d, axis=1), 0.0)


def _sd_cyl_h(p, prim):
    a, b = np.asarray(prim["a"], float), np.asarray(prim["b"], float)
    r = float(prim["r"])
    ax = b - a; L = np.linalg.norm(ax); ax = ax / max(L, 1e-9)
    rel = p - a
    s = rel @ ax
    radial = np.linalg.norm(rel - np.outer(s, ax), axis=1) - r
    axial = np.abs(s - 0.5 * L) - 0.5 * L
    d = np.stack([radial, axial], 1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.max(d, axis=1), 0.0)


def _wall_pieces(prim):
    """A slotted wall as the exact union of axis boxes in the wall frame
    (s along the wall, u across, z): full-height pieces between the slots,
    and the pieces above and below each slot. Returns [(s_lo, s_hi, z_lo, z_hi)]."""
    p0, p1 = np.asarray(prim["p0"], float), np.asarray(prim["p1"], float)
    L = float(np.linalg.norm(p1 - p0)); z0, z1 = float(prim["z0"]), float(prim["z1"])
    slots = sorted([list(map(float, sl)) for sl in prim.get("slots", [])], key=lambda sl: sl[0])
    pieces = []; s_cur = 0.0
    for s0, w, sz0, sz1 in slots:
        a, b = max(s0 - w / 2, 0.0), min(s0 + w / 2, L)
        if a > s_cur:
            pieces.append((s_cur, a, z0, z1))
        if sz0 > z0:
            pieces.append((a, b, z0, sz0))
        if sz1 < z1:
            pieces.append((a, b, sz1, z1))
        s_cur = max(s_cur, b)
    if s_cur < L:
        pieces.append((s_cur, L, z0, z1))
    return pieces


def _sd_wall(p, prim):
    p0, p1 = np.asarray(prim["p0"], float), np.asarray(prim["p1"], float)
    d = p1 - p0; L = float(np.linalg.norm(d)); d = d / max(L, 1e-9)
    n = np.array([-d[1], d[0]]); t = float(prim["t"])
    rel = p[:, :2] - p0[None, :]
    s = rel @ d; u = rel @ n; z = p[:, 2]
    best = np.full(len(p), np.inf)
    for s_lo, s_hi, z_lo, z_hi in _wall_pieces(prim):
        q = np.stack([s - 0.5 * (s_lo + s_hi), u, z - 0.5 * (z_lo + z_hi)], 1)
        half = np.array([0.5 * (s_hi - s_lo), 0.5 * t, 0.5 * (z_hi - z_lo)])
        best = np.minimum(best, _sd_box_local(q, half))
    return best


_SD = {"sphere": _sd_sphere, "box": _sd_box, "cyl_v": _sd_cyl_v, "cyl_h": _sd_cyl_h, "wall": _sd_wall}


class PrimitiveSdf:
    """Signed distance to the nearest primitive of a spec (floor excluded;
    the sim's altitude rule handles the ground)."""

    def __init__(self, primitives: list[dict]):
        self.prims = [p for p in primitives if p.get("kind") in _SD]

    def __call__(self, p: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(p, float))
        if not self.prims:
            return np.full(len(P), np.inf)
        return np.min(np.stack([_SD[q["kind"]](P, q) for q in self.prims], 0), axis=0)


class CloudSdf:
    """Distance to the nearest cloud point above the floor (z > 0.3 m; with ``ground`` -- terrain mode,
    ``terrain.ground_of`` -- 0.3 m above the local ground, so the terrain itself is not an obstacle)."""

    def __init__(self, pts: np.ndarray, floor_z: float = 0.3, ground=None):
        if ground is None:
            obs = pts[pts[:, 2] > floor_z]
        else:
            from .terrain import obstacle_mask
            obs = pts[obstacle_mask(pts, ground, floor_z)]
        self.tree = cKDTree(obs if len(obs) else pts[:1])
        self.empty = len(obs) == 0

    def __call__(self, p: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(p, float))
        if self.empty:
            return np.full(len(P), np.inf)
        return self.tree.query(P, workers=-1)[0]
