"""Page assets for the forest smoke test: JPEG views, coloured depth, plan maps (top render + flight-band footprint),
and one metrics.json. Plain matplotlib, VIZ_STYLE colours for the plan (dark = obstacle in the flight band, light =
wholly above / below it).

    python infinigen/page/make_sheet.py --scenes out/forest_1 out/plain_1 --out out/page [--ref frame.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

INK, MUTED, GRID = "#141820", "#707a8a", "#eceff4"
DARK, LIGHT = "#9aa3b2", "#e1e5ec"
WIRE = "#eb6834"
BAND = (0.4, 5.5)


def jpg(src: Path, dst: Path, q=88, ev=0.0):
    im = Image.open(src).convert("RGB")
    if ev:
        # auto-exposure stand-in: scale in linear light, back to sRGB
        x = np.asarray(im).astype(np.float32) / 255.0
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * (2.0 ** ev)
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
        im = Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    im.save(dst, quality=q, optimize=True)


def depth_png(npy: Path, dst: Path, dmax=25.0):
    d = np.load(npy).astype(np.float32)
    d[~np.isfinite(d)] = dmax
    v = np.clip(d / dmax, 0, 1)
    rgb = (plt.get_cmap("magma_r")(v)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(dst, optimize=True)
    return float(np.median(np.minimum(d, dmax)))


def plan_png(sc: Path, dst: Path, views, box, fly=(1.0, 3.0)):
    z = np.load(sc / "views" / "cloud.npz")
    P, L, agl = z["points"], z["label"], z["agl"]
    names = json.loads(str(z["names"]))
    half, region = box["half"], box["region"]
    cell = 0.25
    n = int(round(2 * region / cell))
    inband = (agl >= fly[0]) & (agl <= fly[1])
    ij = np.clip(((P[:, :2] + region) / cell).astype(int), 0, n - 1)
    any_occ = np.zeros((n, n), bool)
    any_occ[ij[:, 1], ij[:, 0]] = True
    band_occ = np.zeros((n, n), bool)
    band_occ[ij[inband, 1], ij[inband, 0]] = True
    if box.get("scene_type") == "cave":
        # a cave's columns without a floor (ground_grid NaN) are rock at every height
        gx = z["grid_x"]
        ci = np.clip(np.round((-region + (np.arange(n) + 0.5) * cell - gx[0]) / (gx[1] - gx[0])).astype(int), 0,
                     len(gx) - 1)
        band_occ |= np.isnan(z["ground_grid"])[np.ix_(ci, ci)]
    img = np.ones((n, n, 3))
    img[any_occ] = matplotlib.colors.to_rgb(LIGHT)
    img[band_occ] = matplotlib.colors.to_rgb(DARK)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.imshow(img, origin="lower", extent=(-region, region, -region, region), interpolation="nearest")
    wl = names.get("wires")
    if wl is not None:
        w = L == wl
        ax.scatter(P[w, 0], P[w, 1], s=0.6, c=WIRE, lw=0, label="thin wires and poles")
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], ls=(0, (4, 3)), c=INK, lw=1.2,
            label=f"flight box {2 * half:.0f} x {2 * half:.0f} m")
    for k, v in enumerate(views):
        x, y, _ = v["p"]
        ax.plot(x, y, "o", c=INK, ms=5)
        ax.plot([x, x + 2.2 * np.cos(v["yaw"])], [y, y + 2.2 * np.sin(v["yaw"])], c=INK, lw=1.6)
        ax.text(x + 0.8, y - 1.6, f"v{k}", color=INK, fontsize=9)
    ax.set_xlim(-region, region)
    ax.set_ylim(-region, region)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=MUTED)
    ax.set_ylabel("y (m)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def top_png(sc: Path, dst: Path, box, views):
    im = Image.open(sc / "views" / "top.png").convert("RGB")
    W = im.size[0]
    region, half = box["region"], box["half"]
    s = W / (2 * region)

    def px(x, y):
        return ((x + region) * s, (region - y) * s)

    d = ImageDraw.Draw(im)
    (x0, y0), (x1, y1) = px(-half, half), px(half, -half)
    for t in range(0, int(x1 - x0), 14):
        d.line([(x0 + t, y0), (min(x0 + t + 8, x1), y0)], fill=(255, 255, 255), width=3)
        d.line([(x0 + t, y1), (min(x0 + t + 8, x1), y1)], fill=(255, 255, 255), width=3)
    for t in range(0, int(y1 - y0), 14):
        d.line([(x0, y0 + t), (x0, min(y0 + t + 8, y1))], fill=(255, 255, 255), width=3)
        d.line([(x1, y0 + t), (x1, min(y0 + t + 8, y1))], fill=(255, 255, 255), width=3)
    im.convert("RGB").save(dst, quality=88, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path, required=True)
    ap.add_argument("--ref", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for sc in a.scenes:
        tag = sc.name
        box = json.loads((sc / "box.json").read_text())
        band = json.loads((sc / "views" / "band.json").read_text())
        vj = json.loads((sc / "views" / "views.json").read_text())
        views = vj["views"]
        files = []
        for f in sorted((sc / "views").glob("v*_A*.png")):
            jpg(f, a.out / f"{tag}_{f.stem}.jpg", ev=2.0 if f.stem.endswith("_A3") else 0.0)
            files.append(f"{tag}_{f.stem}.jpg")
        depth_med = {}
        for f in sorted((sc / "views").glob("v*_depth.npy")):
            depth_med[f.stem] = depth_png(f, a.out / f"{tag}_{f.stem}.png")
        plan_png(sc, a.out / f"{tag}_plan.png", views, box, tuple(band.get("fly_slice_m", (1.0, 3.0))))
        top_png(sc, a.out / f"{tag}_top.jpg", box, views)
        metrics[tag] = {"box": box, "band": band, "views": views, "timings": vj["timings"], "samples": vj.get("samples"),
                        "render_seconds": vj["seconds"], "files": files, "depth_median_m": depth_med}
    for r in a.ref:
        im = Image.open(r).convert("RGB")
        im.crop((0, 0, 720, 540)).save(a.out / f"ref_{r.stem}.jpg", quality=88, optimize=True)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("SHEET_DONE", a.out)


if __name__ == "__main__":
    main()
