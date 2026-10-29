#!/usr/bin/env python3
"""Put a rendered flythrough together as two videos.

    python blender/flythrough_video.py --fly <dir>    # render_poses.py's output dir

Writes into ``--fly``:

    flythrough_grid.mp4   1920x1080: RGB | depth / flow | top-down map with the path
    flythrough_rgb.mp4    the RGB alone at the render size

The flow panel is ``flowlib.backward_flow`` between consecutive frames of the
path (the true previous frame, 1/fps earlier), coloured black = still, hue =
direction, on one fixed scale for the whole video; hidden and out-of-view
pixels are shown like the rest (the flow is defined there too).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flowlib as FL  # noqa: E402
from page_data import depth_rgb  # noqa: E402

PANEL = (960, 540)
ACCENT = (240, 133, 63)


def label(img: Image.Image, text: str, font, corner="tl") -> None:
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    x = 14 if corner.endswith("l") else img.width - w - 26
    y = 12 if corner.startswith("t") else img.height - h - 26
    d.rectangle((x - 8, y - 6, x + w + 10, y + h + 12), fill=(10, 14, 13, 170))
    d.text((x, y - y0), text, font=font, fill=(235, 240, 238))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fly", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args(argv)
    fly = a.fly
    P = json.loads((fly / "poses.json").read_text())
    R = json.loads((fly / "render.json").read_text())
    K = np.array(R["K_depth"])
    fps, frames = P["fps"], P["frames"]
    n = len(frames)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    pose = lambda k: (np.array(frames[k]["p"]), np.array(frames[k]["R_wb"]))  # noqa: E731
    depth = lambda k: np.load(fly / "depth" / f"{k:05d}.npy")                  # noqa: E731

    # one flow scale for the whole video: the 95th percentile of per-frame 99th percentiles
    p99 = []
    for k in range(1, n, 10):
        f, st, _ = FL.backward_flow(depth(k), depth(k - 1), pose(k), pose(k - 1), K)
        p99.append(np.percentile(np.hypot(f[..., 0], f[..., 1]), 99))
    scale = float(np.percentile(p99, 95))

    # the map and the world -> map pixel mapping
    mp = R["map"]
    base_map = Image.open(fly / "topmap.png").convert("RGB")
    sx = base_map.width / mp["W"]

    def to_map(xy):
        u = mp["W"] / 2 + (xy[0] - mp["centre"][0]) / mp["metres_per_px"]
        v = mp["H"] / 2 - (xy[1] - mp["centre"][1]) / mp["metres_per_px"]
        return u * sx, v * sx

    path_px = [to_map(f["p"]) for f in frames]
    low_n = sum(1 for f in frames if f.get("phase", "low") == "low")

    grid = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", "1920x1080", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                             "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(fly / "flythrough_grid.mp4")], stdin=subprocess.PIPE)
    prev_d = None
    for k in range(n):
        rgb = Image.open(fly / "rgb" / f"{k:05d}.png").convert("RGB")
        d = depth(k)
        dimg = Image.fromarray(depth_rgb(d)).resize(PANEL, Image.BILINEAR)
        if prev_d is None:
            fimg = Image.new("RGB", PANEL)
        else:
            f, st, _ = FL.backward_flow(d, prev_d, pose(k), pose(k - 1), K)
            fimg = Image.fromarray(FL.flow_to_rgb(f, None, scale, style="dark")).resize(PANEL, Image.BILINEAR)
        prev_d = d

        m = base_map.copy()
        dm = ImageDraw.Draw(m, "RGBA")
        dm.line(path_px[:low_n], fill=(255, 255, 255, 110), width=2)
        dm.line(path_px[:k + 1] if k < low_n else path_px[:low_n], fill=ACCENT + (255,), width=4)
        p, Rwb = pose(k)
        u, v = to_map(p)
        if 0 <= u < m.width and 0 <= v < m.height:
            hu, hv = to_map(p[:2] + Rwb[:2, 0] * 2.0)
            dm.line([(u, v), (hu, hv)], fill=(255, 255, 255, 255), width=3)
            dm.ellipse((u - 7, v - 7, u + 7, v + 7), fill=ACCENT + (255,), outline=(255, 255, 255, 255), width=2)
        m = m.resize(PANEL, Image.BILINEAR)

        rp = rgb.resize(PANEL, Image.LANCZOS)
        spd = np.linalg.norm(np.array(frames[min(k + 1, n - 1)]["p"]) - np.array(frames[max(k - 1, 0)]["p"])) * fps / 2
        label(rp, "RGB  ·  Cycles, 91° drone camera", font)
        label(rp, f"t {k / fps:5.1f} s   {spd:4.1f} m/s   z {p[2]:4.1f} m", small, "bl")
        label(dimg, "Depth  ·  z, 0-20 m (yellow near, violet far)", font)
        label(fimg, f"Optical flow t -> t-1/{fps} s  ·  black = still, full colour = {scale * 2:.0f} px @ 1280", font)
        label(m, "Top view  ·  path so far", font)
        canvas = Image.new("RGB", (1920, 1080))
        for img, xy in ((rp, (0, 0)), (dimg, (960, 0)), (fimg, (0, 540)), (m, (960, 540))):
            canvas.paste(img, xy)
        grid.stdin.write(np.asarray(canvas, np.uint8).tobytes())
        if k % 100 == 0:
            print(f"frame {k}/{n}", flush=True)
    grid.stdin.close()
    grid.wait()

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(fly / "rgb" / "%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf - 3), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(fly / "flythrough_rgb.mp4")], check=True)
    for name in ("flythrough_grid.mp4", "flythrough_rgb.mp4"):
        print(name, round((fly / name).stat().st_size / 1e6, 1), "MB")
    print("flow scale (px @ depth res):", round(scale, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
