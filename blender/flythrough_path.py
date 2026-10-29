#!/usr/bin/env python3
"""A camera path that shows a whole course: low through it, then up and round.

    python blender/flythrough_path.py --course <course dir> --out <dir>/poses.json

The low part is planned in the course's own cloud, the one the labels were
planned in: A* over a 2-D occupancy grid of the flight band (0.3-3.2 m),
kept ``--clearance`` from every obstacle point, through each wall opening
(both sides of every slot are targets), across the far end of the patch and
back to the near side the other way (round a wall's end if it must),
then smoothed with a spline whose clearance is re-checked. It is flown at
``--speed`` with a soft start, the camera looking a little ahead along the
path, banked into turns like a quadrotor. The high part climbs on an arc
round the patch centre with the camera on the centre, so the last seconds
are an overview.

Writes ``poses.json``: fps, the camera (width, height, hfov), and per frame
the body pose (p, R_wb) -- the camera is ``flowlib``'s level mount -- plus the
2-D ground path for the video's map.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import interpolate, ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def rot(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """R_wb = Rz(yaw) Ry(pitch) Rx(roll); positive pitch is nose down."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class Grid:
    def __init__(self, pts, bounds, res=0.2, margin=3.0, band=(0.3, 3.2)):
        x0, y0, x1, y1 = bounds
        self.o = np.array([x0 - margin, y0 - margin])
        self.res = res
        self.n = (np.ceil((np.array([x1 + margin, y1 + margin]) - self.o) / res).astype(int) + 1)
        occ = np.zeros(self.n, bool)
        b = pts[(pts[:, 2] > band[0]) & (pts[:, 2] < band[1])]
        ij = np.floor((b[:, :2] - self.o) / res).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1)
        occ[ij[ok, 0], ij[ok, 1]] = True
        self.clear = ndimage.distance_transform_edt(~occ) * res

    def cell(self, xy):
        return tuple(np.clip(np.floor((np.asarray(xy) - self.o) / self.res).astype(int), 0, self.n - 1))

    def xy(self, c):
        return self.o + (np.asarray(c) + 0.5) * self.res

    def clearance_at(self, xy):
        c = (np.asarray(xy) - self.o) / self.res - 0.5
        return ndimage.map_coordinates(self.clear, c.T, order=1, mode="nearest")

    def nearest_free(self, xy, need):
        c = np.array(self.cell(xy))
        ok = np.argwhere(self.clear >= need)
        return tuple(ok[np.argmin(((ok - c) ** 2).sum(1))])

    def astar(self, a, b, need):
        """8-connected A*; cells under ``need`` clearance are walls; each step
        costs its length plus a push away from obstacles."""
        free = self.clear >= need
        g = {a: 0.0}
        came = {}
        pq = [(0.0, a)]
        steps = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
        while pq:
            _, c = heapq.heappop(pq)
            if c == b:
                break
            for di, dj in steps:
                q = (c[0] + di, c[1] + dj)
                if not (0 <= q[0] < self.n[0] and 0 <= q[1] < self.n[1]) or not free[q]:
                    continue
                step = math.hypot(di, dj) * self.res * (1.0 + 0.6 / max(self.clear[q], 0.3))
                ng = g[c] + step
                if ng < g.get(q, math.inf):
                    g[q], came[q] = ng, c
                    heapq.heappush(pq, (ng + math.dist(q, b) * self.res, q))
        if b not in came and a != b:
            raise RuntimeError(f"no path from {self.xy(a)} to {self.xy(b)} at {need} m clearance")
        out = [b]
        while out[-1] != a:
            out.append(came[out[-1]])
        return [self.xy(c) for c in out[::-1]]


def smooth(poly: np.ndarray, grid: Grid, need: float, ds: float = 0.05) -> np.ndarray:
    """Spline through the A* polyline, smoothed as much as the clearance allows."""
    keep = np.concatenate([[0], np.flatnonzero(np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-6) + 1])
    poly = poly[keep]
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    for s in (2.0, 1.0, 0.5, 0.25, 0.1, 0.0):
        tck, _ = interpolate.splprep([poly[:, 0], poly[:, 1]], u=L / L[-1], s=s * len(poly) * 0.01, k=3)
        uu = np.linspace(0, 1, max(int(L[-1] / ds), 50))
        xy = np.stack(interpolate.splev(uu, tck), 1)
        if grid.clearance_at(xy).min() >= need - 0.1:
            return xy
    return xy


def resample(xy: np.ndarray, speed: float, fps: int, ramp: float = 1.2) -> np.ndarray:
    """Positions at each frame for constant ``speed`` after a soft start."""
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    s, out, t = 0.0, [], 0.0
    while s <= L[-1]:
        out.append([np.interp(s, L, xy[:, 0]), np.interp(s, L, xy[:, 1])])
        t += 1 / fps
        s += speed * min(1.0, 0.25 + t / ramp) / fps
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--course", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--z", type=float, default=1.7)
    ap.add_argument("--clearance", type=float, default=0.9)
    ap.add_argument("--overview-s", type=float, default=7.0, help="0: no overview arc")
    ap.add_argument("--max-low-s", type=float, default=0.0, help="cut the low part to this many seconds (0: whole tour)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=91.0)
    a = ap.parse_args(argv)
    spec = json.loads((a.course / "spec.json").read_text())
    pts = np.load(a.course / "cloud.npz")["points"]
    x0, y0, x1, y1 = spec["bounds"]
    grid = Grid(pts, spec["bounds"])

    # targets: start behind the patch, both sides of every opening, the far corners
    targets = [np.array([x0 + 1.0, 0.5 * y1])]
    for p in spec["primitives"]:
        if p["kind"] == "wall" and p.get("slots"):
            p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            n = np.array([-d[1], d[0]])
            for s0, w, z0, z1 in p["slots"]:
                if not (z0 <= a.z <= z1):
                    continue
                c = p0 + d * s0
                side = 1.0 if n @ (c - targets[0]) > 0 else -1.0
                targets += [c - side * n * 2.0, c + side * n * 2.0]
    # then a loop: the far end on the opening's side, across it, and back to the
    # near side the long way (round a wall's end if that is the way)
    first = targets[-1][1] if len(targets) > 1 else 0.0
    side_y = (y0, y1) if first < (y0 + y1) / 2 else (y1, y0)
    targets += [np.array([x1 - 2.0, 0.6 * side_y[0]]), np.array([x1 - 2.0, 0.6 * side_y[1]]),
                np.array([x0 + 5.0, 0.7 * side_y[1]])]
    # each leg at the widest clearance it can have (doors and arena gaps are narrower
    # than 2 x 0.9 m); a target no clearance reaches is skipped, not fatal
    ladder = [c for c in (a.clearance, 0.7, 0.55) if c <= a.clearance]
    cells = [grid.nearest_free(t, ladder[-1]) for t in targets]
    poly, used, skipped, cur = [], [], [], cells[0]
    for c1 in cells[1:]:
        for need in ladder:
            try:
                seg = grid.astar(cur, c1, need)
            except RuntimeError:
                continue
            poly += seg if not poly else seg[1:]
            used.append(need)
            cur = c1
            break
        else:
            skipped.append([float(v) for v in grid.xy(c1)])
    if not poly:
        raise RuntimeError("no leg of the tour could be planned")
    need_min = min(used)
    xy = smooth(np.array(poly), grid, need_min)
    low = resample(xy, a.speed, a.fps)
    if a.max_low_s > 0:
        low = low[: int(a.max_low_s * a.fps)]
    clr = grid.clearance_at(low)

    # low part: look 0.6 s ahead, bank into turns
    n = len(low)
    ahead = int(0.6 * a.fps)
    frames = []
    vel = np.gradient(low, 1 / a.fps, axis=0)
    acc = np.gradient(vel, 1 / a.fps, axis=0)
    yaws = []
    for k in range(n):
        tgt = low[min(k + ahead, n - 1)] if k + ahead < n else low[-1] + vel[-1]
        dv = tgt - low[k]
        yaws.append(math.atan2(dv[1], dv[0]) if np.linalg.norm(dv) > 1e-3 else (yaws[-1] if yaws else 0.0))
    yaws = np.unwrap(np.array(yaws))
    yaws = ndimage.gaussian_filter1d(yaws, 4, mode="nearest")
    for k in range(n):
        h = np.array([math.cos(yaws[k]), math.sin(yaws[k])])
        lat = float(h[0] * acc[k][1] - h[1] * acc[k][0])      # + = turning left
        roll = float(np.clip(-0.5 * math.atan2(lat, 9.81), -0.25, 0.25))
        frames.append({"p": [float(low[k, 0]), float(low[k, 1]), a.z], "R_wb": rot(yaws[k], math.radians(4), roll).tolist(),
                       "phase": "low"})

    # overview: climb on an arc round the centre, camera on the centre
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.5])
    start = np.array(frames[-1]["p"])
    r0 = float(np.linalg.norm(start[:2] - ctr[:2]))
    R1 = max(x1 - x0, y1 - y0) / 2 + 9.0
    ang0 = math.atan2(start[1] - ctr[1], start[0] - ctr[0])
    m = int(a.overview_s * a.fps)            # 0: the loop below adds nothing
    yaw_prev, pitch_prev = yaws[-1], math.radians(4)
    for k in range(1, m + 1):
        s = k / m
        e = s * s * (3 - 2 * s)                              # smoothstep
        ang = ang0 + math.radians(150) * e
        r = r0 + (R1 - r0) * e
        z = a.z + (11.0 - a.z) * e
        p = np.array([ctr[0] + r * math.cos(ang), ctr[1] + r * math.sin(ang), z])
        look = ctr - p
        yaw = math.atan2(look[1], look[0])
        pitch = math.atan2(-look[2], np.linalg.norm(look[:2]))
        b = min(1.0, s / 0.25)                               # blend the first quarter from the low heading
        yaw = yaw_prev + (np.angle(np.exp(1j * (yaw - yaw_prev)))) * b
        pitch = pitch_prev + (pitch - pitch_prev) * b
        frames.append({"p": p.tolist(), "R_wb": rot(yaw, pitch).tolist(), "phase": "overview"})

    out = {"fps": a.fps, "width": a.width, "height": a.height, "hfov_deg": a.hfov,
           "course": spec["family"] + "_" + str(spec["seed"]), "bounds": spec["bounds"],
           "ground_path": np.round(low, 3).tolist(), "targets": [t.tolist() for t in targets],
           "min_clearance_m": round(float(clr.min()), 3), "leg_clearances": used, "skipped_targets": skipped, "low_length_m": round(float(np.sum(np.linalg.norm(np.diff(low, axis=0), axis=1))), 1),
           "frames": frames}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print(f"{len(frames)} frames ({len(frames) / a.fps:.1f} s): low {n} over {out['low_length_m']} m, "
          f"min clearance {out['min_clearance_m']} m; overview {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
