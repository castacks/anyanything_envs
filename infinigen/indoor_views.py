"""Views of one indoor_box house: collision cloud + occupancy per height slice over the floor, a plan map, drone-eye
renders inside rooms, true depth for one view, a top-down render with the ceiling hidden.

    python infinigen/indoor_views.py --scene <out>/scene/scene.blend --box <out>/box.json --out <out>/views \
        [--n-views 3] [--samples 32] [--threads 30] [--no-render]

The flight band indoors runs from 0.4 m over the floor to 0.4 m under the ceiling. Camera = the Starling front camera
(640 x 480, 87 deg horizontal FOV, level, box_views' camera), 1.2-1.6 m over the floor in the most open spot of the
roomiest rooms, facing the longest free sightline. Sampling, voxels, camera and render helpers are box_views'.
Writes <out>/{cloud.npz, band.json, band_occ.npy, plan.png, views.json, v<k>.png, v0_depth.npy, top.png}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np

import box_views as bv

CELL = 0.25                 # occupancy grid, m
CLOSE_M = 0.35              # closing radius that joins the room floors across the walls between them (<= 0.3 m)
FLY_SLICE = (1.0, 2.0)      # the slice the views fly in and the plan's middle panel
ROOM_PARTS = ("wall", "floor", "ceiling", "exterior")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-views", type=int, default=3)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--threads", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=2000.0, help="surface samples per m2 (walls need no holes)")
    ap.add_argument("--top-res", type=int, default=1024, help="long side of the top view, px")
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args()


def family(nm):
    """Room meshes are '<room>.<part>' (Infinigen's split_rooms): label them by part; assets by factory."""
    parts = nm.split(".")
    if len(parts) > 1 and "/" in parts[0] and parts[1] in ROOM_PARTS:
        return parts[1]
    return nm.split("(")[0].split(".")[0]


def room_objects(part):
    col = bpy.data.collections.get(f"unique_assets:room_{part}")
    return [o for o in col.objects] if col is not None else []


# ---------------------------------------------------------------- grid

class Grid:
    """CELL-sized cells over the interior bounds (one cell of padding): room index per cell (-1 off the room floors)
    and the house footprint, the room floors closed across the walls between them. Each floor stops at its wall
    faces, so inner walls and doorways belong to no room; in the footprint they are cells like any other, which the
    wall voxels block and a doorway leaves free."""

    def __init__(self, box):
        from shapely import contains_xy
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        x0, y0, x1, y1 = box["interior_bounds"]
        self.x0 = math.floor(x0 / CELL) * CELL - CELL
        self.y0 = math.floor(y0 / CELL) * CELL - CELL
        self.nx = int(math.ceil((x1 - self.x0) / CELL)) + 1
        self.ny = int(math.ceil((y1 - self.y0) / CELL)) + 1
        X, Y = np.meshgrid(self.x0 + (np.arange(self.nx) + 0.5) * CELL, self.y0 + (np.arange(self.ny) + 0.5) * CELL)
        self.room = np.full((self.ny, self.nx), -1, np.int16)
        polys = [Polygon(r["polygon"]) for r in box["rooms"]]
        for k, poly in enumerate(polys):
            self.room[contains_xy(poly, X, Y) & (self.room < 0)] = k
        self.house = unary_union(polys).buffer(CLOSE_M, join_style="mitre").buffer(-CLOSE_M, join_style="mitre")
        self.interior = contains_xy(self.house, X, Y) | (self.room >= 0)

    def ij(self, xy):
        return ((xy[:, 0] - self.x0) / CELL).astype(int), ((xy[:, 1] - self.y0) / CELL).astype(int)

    def centre(self, i, j):
        return self.x0 + (i + 0.5) * CELL, self.y0 + (j + 0.5) * CELL

    def occupancy(self, P, agl, lo, hi):
        m = (agl >= lo) & (agl <= hi)
        i, j = self.ij(P[m])
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        o = np.zeros((self.ny, self.nx), bool)
        o[j[ok], i[ok]] = True
        return o

    def meta(self):
        return {"x0": self.x0, "y0": self.y0, "cell_m": CELL, "shape": [self.ny, self.nx]}


def clearance(blocked):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~blocked) * CELL


def slice_stats(grid, occ):
    """Over the house footprint: blocked share, share at least 0.5 / 1 m clear, median clearance (outside the
    footprint counts as blocked), and how the free cells connect (4-neighbours), over all free cells and over those
    at least 0.5 m clear: components, the largest one's share, how many rooms it reaches."""
    from scipy.ndimage import label
    inside = grid.interior
    cl = clearance(occ | ~inside)
    free = inside & ~occ
    n = inside.sum()
    st = {"blocked_frac": float((occ & inside).sum() / n),
          "free_frac_clear_0.5m": float((inside & (cl >= 0.5)).sum() / n),
          "free_frac_clear_1m": float((inside & (cl >= 1.0)).sum() / n),
          "median_clearance_m": float(np.median(cl[free])) if free.any() else 0.0}
    for tag, m in (("free", free), ("clear_0.5m", free & (cl >= 0.5))):
        comp, ncomp = label(m)
        sizes = np.bincount(comp.ravel())[1:]
        big = int(np.argmax(sizes)) + 1 if ncomp else 0
        st[f"components_{tag}"] = int(ncomp)
        st[f"largest_component_frac_{tag}"] = float(sizes.max() / m.sum()) if ncomp else 0.0
        st[f"rooms_in_largest_component_{tag}"] = (int(len(set(grid.room[comp == big].tolist()) - {-1}))
                                                   if ncomp else 0)
    return st


# ---------------------------------------------------------------- views

def free_ray(occ_out, grid, x, y, yaw, max_m=30.0):
    d = 0.0
    while d < max_m:
        i = int((x + d * math.cos(yaw) - grid.x0) / CELL)
        j = int((y + d * math.sin(yaw) - grid.y0) / CELL)
        if not (0 <= i < grid.nx and 0 <= j < grid.ny) or occ_out[j, i]:
            break
        d += CELL / 2
    return d


def pick_views(grid, occ_fly, box, n, rng):
    """One view per room, roomiest rooms first (most cells 0.75 m clear in the fly slice): a random cell among the
    room's most open ones (at least 0.5 m and 80 % of its best clearance), 1.2-1.6 m over the floor, yaw = the
    longest free ray of 32."""
    blocked = occ_fly | ~grid.interior
    cl = clearance(blocked)
    rooms = []
    for k in range(len(box["rooms"])):
        m = grid.room == k
        if m.any() and cl[m].max() >= 0.5:
            rooms.append((int((m & (cl >= 0.75)).sum()), float(cl[m].max()), k))
    views = []
    for _, best, k in sorted(rooms, reverse=True)[:n]:
        m = grid.room == k
        jj, ii = np.nonzero(m & (cl >= max(0.5, 0.8 * best)))
        s = int(rng.integers(len(ii)))
        x, y = grid.centre(ii[s], jj[s])
        h = float(rng.uniform(1.2, 1.6))
        ray, yaw = max((free_ray(blocked, grid, x, y, yaw), float(yaw))
                       for yaw in np.linspace(0, 2 * math.pi, 32, endpoint=False))
        r = box["rooms"][k]
        views.append({"p": [round(x, 3), round(y, 3), round(r["floor_z"] + h, 3)], "agl": round(h, 3),
                      "yaw": round(yaw, 4), "room": r["name"], "clearance_m": float(cl[jj[s], ii[s]]),
                      "free_ray_m": round(ray, 2)})
    return views


# ---------------------------------------------------------------- plan map

def plan_map(path, grid, occs, band_occ, box, views):
    """One panel per height slice: blocked cells dark, cells blocked elsewhere in the band light, outside the house
    grey; room outlines and types; views as black dots with heading and FOV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from shapely.geometry import Polygon
    ink, muted = "#141820", "#707a8a"
    cmap = ListedColormap(["#ffffff", "#e1e5ec", "#9aa3b2", "#f2f3f5"])
    ext = [grid.x0, grid.x0 + grid.nx * CELL, grid.y0, grid.y0 + grid.ny * CELL]
    w = (ext[1] - ext[0]) / (ext[3] - ext[2])
    fig, axes = plt.subplots(1, len(occs), figsize=(5.2 * len(occs) * min(1.6, max(0.7, w)), 5.6), dpi=110)
    for ax, (name, (occ, st)) in zip(axes, occs.items()):
        img = np.where(~grid.interior, 3, np.where(occ, 2, np.where(band_occ, 1, 0)))
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        for r in box["rooms"]:
            poly = np.array(r["polygon"])
            ax.plot(poly[:, 0], poly[:, 1], color=ink, lw=0.8)
            c = Polygon(poly).representative_point()
            ax.text(c.x, c.y, r["type"], ha="center", va="center", fontsize=7, color=muted)
        for k, v in enumerate(views):
            x, y, _ = v["p"]
            for da in (-math.radians(bv.HFOV / 2), math.radians(bv.HFOV / 2)):
                ax.plot([x, x + 1.6 * math.cos(v["yaw"] + da)], [y, y + 1.6 * math.sin(v["yaw"] + da)], color=muted,
                        lw=0.6)
            ax.plot([x, x + 0.9 * math.cos(v["yaw"])], [y, y + 0.9 * math.sin(v["yaw"])], color=ink, lw=1.6)
            ax.plot(x, y, "o", color=ink, ms=4)
            ax.text(x + 0.25, y + 0.25, f"v{k}", fontsize=8, color=ink)
        ax.set_title(f"{name} m over the floor: {100 * st['blocked_frac']:.0f} % blocked, "
                     f"{100 * st['free_frac_clear_0.5m']:.0f} % >= 0.5 m clear", fontsize=9, color=ink)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=8, color=muted)
        ax.tick_params(labelsize=7, colors=muted)
    axes[0].set_ylabel("y (m)", fontsize=8, color=muted)
    fig.suptitle("dark: blocked in the slice   light: blocked elsewhere in the flight band   grey: outside the house",
                 fontsize=8, color=muted, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- top view

def render_top(path, box, res):
    """Orthographic top view of the interior. The ceiling, the house's outer shell (roof) and the ceiling lights are
    made invisible to camera rays only, as Infinigen's own overhead mode does, so the rooms keep their lighting."""
    hidden = room_objects("ceiling") + room_objects("exterior")
    hidden += [o for o in bpy.data.objects if "CeilingLight" in o.name]
    for o in hidden:
        o.visible_camera = False
    hb = box.get("house_bounds")
    x0, y0, x1, y1 = (hb[0], hb[1], hb[3], hb[4]) if hb else box["interior_bounds"]
    x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5
    top = bpy.data.objects.new("top_cam", bpy.data.cameras.new("top_cam"))
    bpy.context.scene.collection.objects.link(top)
    top.data.type = "ORTHO"
    top.data.ortho_scale = max(x1 - x0, y1 - y0)
    top.data.sensor_fit = "AUTO"
    top.data.clip_end = 100.0
    top.location = (0.5 * (x0 + x1), 0.5 * (y0 + y1), box["ceiling_z"] + 10.0)
    top.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = top
    r = (x1 - x0) / (y1 - y0)
    bv.render_to(path, (res, max(64, int(res / r))) if r >= 1 else (max(64, int(res * r)), res))
    for o in hidden:
        o.visible_camera = True
    return {"ortho_m": [x1 - x0, y1 - y0], "hidden_from_camera": len(hidden)}


# ---------------------------------------------------------------- main

def main():
    a = parse()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    box = json.loads(a.box.read_text())
    rng = np.random.default_rng(a.seed)
    bpy.ops.wm.open_mainfile(filepath=str(a.scene))
    bv.show_render_visible()

    floor_z, ceil_z = box["floor_z"], box["ceiling_z"]
    hb = box.get("house_bounds") or [*box["interior_bounds"][:2], floor_z, *box["interior_bounds"][2:], ceil_z]
    region = max(abs(v) for v in (hb[0], hb[1], hb[3], hb[4])) + 1.0
    P, L, names = bv.sample_scene(region, ceil_z + 1.0, rng, density=a.density, skip_terrain=True, ground=None,
                                  small_m=0.0, family=family)
    P, L = bv.voxelize(P, L)
    agl = P[:, 2] - floor_z
    np.savez_compressed(a.out / "cloud.npz", points=P.astype(np.float32), label=L, agl=agl.astype(np.float32),
                        names=json.dumps(names), floor_z=np.float32(floor_z), ceiling_z=np.float32(ceil_z))
    print(f"cloud: {len(P)} voxels from {len(names)} families in {time.time() - t0:.0f} s", flush=True)

    # occupancy over the house footprint, per height slice over the floor
    grid = Grid(box)
    top_m = round(box["ceiling_height_m"] - 0.4, 2)
    band = (0.4, top_m)
    slices = {}
    occs = {}
    for lo, hi in [band, (0.4, 1.0), FLY_SLICE, (2.0, top_m)]:
        o = grid.occupancy(P, agl, lo, hi)
        st = slice_stats(grid, o)
        slices[f"{lo:g}-{hi:g}"] = st
        if (lo, hi) != band:
            occs[f"{lo:g}-{hi:g}"] = (o, st)
    band_occ = grid.occupancy(P, agl, *band)
    occ_fly = occs[f"{FLY_SLICE[0]:g}-{FLY_SLICE[1]:g}"][0]
    cl_fly = clearance(occ_fly | ~grid.interior)
    per_room = {}
    for k, r in enumerate(box["rooms"]):
        m = grid.room == k
        free = m & ~occ_fly
        per_room[r["name"]] = {"type": r["type"], "area_m2": r["area_m2"], "fly_blocked_frac": float(occ_fly[m].mean()),
                               "fly_median_clearance_m": float(np.median(cl_fly[free])) if free.any() else 0.0}
    counts = {k: int((L == v).sum()) for k, v in names.items()}
    rec = {"band_m": list(band), "fly_slice_m": list(FLY_SLICE), "floor_z": floor_z, "ceiling_z": ceil_z,
           "slices": slices, "rooms": per_room, "grid": grid.meta(),
           "interior_cells": int(grid.interior.sum()), "interior_area_m2": float(grid.interior.sum() * CELL ** 2),
           "points": int(len(P)), "families": names, "points_per_family": counts,
           "objects": box.get("objects", {})}
    (a.out / "band.json").write_text(json.dumps(rec, indent=1))
    np.save(a.out / "band_occ.npy", occ_fly | ~grid.interior)
    print("band", json.dumps({k: v for k, v in rec.items() if k in ("slices", "interior_area_m2", "points")}),
          flush=True)

    views = pick_views(grid, occ_fly, box, a.n_views, rng)
    plan_map(a.out / "plan.png", grid, occs, band_occ, box, views)
    if a.no_render:
        (a.out / "views.json").write_text(json.dumps({"views": views, "seconds": round(time.time() - t0, 1)}, indent=1))
        return

    bv.setup_render(a.samples, a.threads)
    scene = bpy.context.scene
    cam = bv.make_camera()
    scene.camera = cam
    timings = []
    for k, v in enumerate(views):
        bv.pose_camera(cam, v["p"], v["yaw"])
        t1 = time.time()
        bv.render_to(a.out / f"v{k}.png")
        timings.append({"view": k, "s": round(time.time() - t1, 1)})
        print("rendered", k, v["room"], timings[-1]["s"], "s", flush=True)
        if k == 0:
            bv.render_depth(a.out / "v0_depth.npy")
    scene.cycles.samples = max(16, a.samples // 2)
    t1 = time.time()
    top = render_top(a.out / "top.png", box, a.top_res)
    timings.append({"view": "top", "s": round(time.time() - t1, 1)})
    out = {"views": views, "timings": timings, "camera": {"w": bv.W, "h": bv.H, "hfov_deg": bv.HFOV},
           "samples": a.samples, "threads": a.threads, "film_exposure": scene.cycles.film_exposure,
           "view_transform": scene.view_settings.view_transform, "top": top, "seconds": round(time.time() - t0, 1)}
    (a.out / "views.json").write_text(json.dumps(out, indent=1))
    print("INDOOR_VIEWS_DONE", json.dumps({"seconds": out["seconds"]}))


if __name__ == "__main__":
    sys.exit(main())
