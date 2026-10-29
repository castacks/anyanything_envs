#!/usr/bin/env python3
"""Try a place kit on one patch, on the CPU, and look at it.

    python3 blender/kit_try.py --kit port --spec <spec.json> --out <dir> [--level 2] [--variant 0]
            [--res 960x540] [--samples 24] [--threads 8] [--no-preview]

Builds the course with ``build_course.py --kit`` on the CPU (Cycles, no GPU),
renders the previews (drone's-eye from the start and from the goal, chase,
oblique, top) and writes ``<out>/sheet.jpg`` -- the five shots in one image,
captioned with the kit, HDRI, triangle and cloud counts -- plus the build log.
The level sets the same knobs as a batch (``batch.LEVEL_KNOBS``).

Environment: ``BLENDER`` (default ``batch.find_blender()``: gs2's
/mnt/data/blender/blender-5.2.2-linux-x64/blender, else airstation03's ~/opt/...),
``BLENDER_LIBRARY`` (default /mnt/data/blender/assets).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from batch import BLENDER_ENV, BLENDER_PY, LEVEL_KNOBS, find_blender  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more build_course.py args")
    a = ap.parse_args(argv)
    blender = find_blender()
    lib = os.environ.get("BLENDER_LIBRARY", "/mnt/data/blender/assets")
    a.out.mkdir(parents=True, exist_ok=True)
    k = LEVEL_KNOBS[a.level]
    cmd = [blender, "-b", "--factory-startup", *BLENDER_PY, "-t", str(a.threads), "-P",
           str(HERE / "build_course.py"), "--",
           "--spec", str(a.spec), "--assets", lib, "--out", str(a.out), "--kit", a.kit, "--cpu",
           "--threads", str(a.threads), "--clutter", str(k["clutter"]), "--keepout", str(k["keepout"]),
           "--hang", str(k["hang"]), "--level", str(a.level), "--variant", str(a.variant),
           "--preview-res", a.res, "--preview-samples", str(a.samples), *a.extra]
    if not a.no_preview:
        cmd.append("--preview")
    t0 = time.time()
    with open(a.out / "build.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **BLENDER_ENV})
    dt = time.time() - t0
    if rc != 0 or not (a.out / "course.json").exists():
        tail = (a.out / "build.log").read_text().splitlines()[-40:]
        print("\n".join(tail))
        print(f"BUILD FAILED rc={rc} after {dt:.0f} s (log {a.out / 'build.log'})")
        return 1
    cj = json.loads((a.out / "course.json").read_text())
    line = (f"{a.kit} L{a.level} v{a.variant} {cj['patch']}  hdri {cj.get('hdri')}  exp {cj.get('exposure'):.2f}  "
            f"tris {cj['mesh']['triangles']:,}  cloud {cj['cloud']['obstacle_points']:,} pts  "
            f"build {cj['seconds_build']:.0f} s  total {dt:.0f} s")
    print(line)
    if not a.no_preview:
        try:
            from PIL import Image, ImageDraw  # noqa: PLC0415
        except ImportError:
            print("(no PIL: sheet.jpg skipped)")
            return 0
        shots = ["eye", "eye_back", "chase", "side", "top"]
        ims = [Image.open(a.out / f"preview_{s}.png").convert("RGB") for s in shots if (a.out / f"preview_{s}.png").exists()]
        w, h = ims[0].size
        sheet = Image.new("RGB", (2 * w, 3 * h + 28), (20, 20, 20))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * w, (i // 2) * h + 28))
        ImageDraw.Draw(sheet).text((8, 8), line, fill=(255, 255, 255))
        sheet.save(a.out / "sheet.jpg", quality=88)
        print(f"sheet: {a.out / 'sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
