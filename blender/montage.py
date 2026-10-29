#!/usr/bin/env python3
"""Cut several flythrough clips (``render_poses.py --rgb-only`` dirs) into one captioned reel.

    python blender/montage.py --clips <dir1> <dir2> ... --captions "..." "..." --out reel.mp4

Each clip is shown whole, a caption bar in the lower left, with a short fade
through black between clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", type=Path, nargs="+", required=True)
    ap.add_argument("--captions", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade", type=int, default=8, help="frames")
    ap.add_argument("--crf", type=int, default=24)
    a = ap.parse_args(argv)
    assert len(a.clips) == len(a.captions)
    first = sorted((a.clips[0] / "rgb").glob("*.png"))[0]
    W, H = Image.open(first).size
    font = ImageFont.load_default(size=max(18, H // 28))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)],
                          stdin=subprocess.PIPE)
    n = 0
    for clip, cap in zip(a.clips, a.captions):
        frames = sorted((clip / "rgb").glob("*.png"))
        for k, f in enumerate(frames):
            im = Image.open(f).convert("RGB")
            d = ImageDraw.Draw(im, "RGBA")
            x0, y0, x1, y1 = d.textbbox((0, 0), cap, font=font)
            pad = H // 60
            d.rectangle((pad, H - (y1 - y0) - 4 * pad, pad + (x1 - x0) + 2 * pad, H - pad), fill=(10, 14, 13, 190))
            d.text((2 * pad, H - (y1 - y0) - 3 * pad - y0), cap, font=font, fill=(240, 244, 242))
            arr = np.asarray(im, np.float32)
            fade = min(1.0, (k + 1) / a.fade, (len(frames) - k) / a.fade)
            ff.stdin.write((arr * fade).astype(np.uint8).tobytes())
            n += 1
    ff.stdin.close()
    ff.wait()
    print(a.out, n, "frames", round(a.out.stat().st_size / 1e6, 1), "MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
