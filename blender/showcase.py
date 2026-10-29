#!/usr/bin/env python3
"""Showcase data for chosen scenes of a batch: RGB | depth | flow at a few
labelled states, and the labeller's point cloud for a 3-D view.

    python blender/showcase.py --batch <dir> --scenes <pid> ... --out <dir> [--states 3]

Per scene, into ``<out>/<pid>/``:

    s<i>_rgb.jpg, s<i>_depth.jpg, s<i>_flow.jpg   640x480, state i (the writer's frame number)
    cloud.bin        int16 little-endian x,y,z in centimetres, obstacle points only (<= 4.5 m)
    scene.json       bounds, states (pose, speed, goal, label rows at 0.1 s), flow scale, files

The states are the ones with the most obstacle surface within 8 m in view
(stored depth), one per goal first. Depth is re-rendered at 640x480 for
those poses (``render_poses.py``, the course's .blend -- rebuilt if it was
cleaned up); the flow is the stored full-resolution flow; the RGB is the
stored frame.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flowlib as FL  # noqa: E402
import render_io as RIO  # noqa: E402
from page_data import depth_rgb  # noqa: E402

BLENDER = os.environ.get("BLENDER", str(Path.home() / "opt/blender-5.2.2-linux-x64/blender"))


def jpg(arr, path, q=85):
    Image.fromarray(arr).convert("RGB").save(path, quality=q, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--cloud-points", type=int, default=90000)
    a = ap.parse_args(argv)
    b = a.batch
    plan = {s["pid"]: s for s in json.loads((b / "plan.json").read_text())["scenes"]}
    index = []
    for pid in a.scenes:
        s = plan[pid]
        course, data = b / "courses" / pid, b / "data" / pid
        ro = next((data / "train").glob("rollout_*"))
        out = a.out / pid
        out.mkdir(parents=True, exist_ok=True)
        if not (course / "course.blend").exists():
            subprocess.run([sys.executable, HERE / "batch.py", "build", "--batch", b, "--only", pid, "--force"], check=True)
        man = [json.loads(l) for l in (ro / "render_manifest.jsonl").read_text().splitlines() if l.strip()]
        spec = json.loads((course / "spec.json").read_text())

        # choose states: most obstacle surface within 8 m in the stored depth, one per goal first
        scored = []
        for m in man:
            d = np.asarray(Image.open(ro / m["frame"]), np.float32) / 1000.0
            st = json.loads((data / "states" / pid / f"{m['state_k']}.json").read_text())
            near = float(((d > 0.2) & (d < 8.0)).mean())
            scored.append((near, m, st))
        scored.sort(key=lambda t: -t[0])
        picked, goals = [], set()
        for near, m, st in scored:
            if st.get("goal_case") not in goals:
                picked.append((m, st)); goals.add(st.get("goal_case"))
        for near, m, st in scored:
            if len(picked) >= a.states:
                break
            if all(m is not p[0] for p in picked):
                picked.append((m, st))
        picked = picked[: a.states]
        picked.sort(key=lambda t: t[0]["state_k"])

        # re-render full-resolution depth (and RGB) at exactly those poses
        rd = out / "_render"
        poses = {"fps": 15, "width": 640, "height": 480, "hfov_deg": man[0]["cam"]["hfov_deg"],
                 "bounds": spec["bounds"],
                 "frames": [{"p": m["t_world"], "R_wb": np.asarray(m["R_wb"]).reshape(3, 3).tolist(), "phase": "low"}
                            for m, _ in picked]}
        rd.mkdir(exist_ok=True)
        (rd / "poses.json").write_text(json.dumps(poses))
        subprocess.run([BLENDER, "-b", "--factory-startup", "-P", HERE / "render_poses.py", "--",
                        "--course", course, "--poses", rd / "poses.json", "--out", rd, "--depth-scale", "1.0"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        flows = [np.load(ro / RIO.FLOW.format(m["state_k"] + 1))["flow_full"].astype(np.float32) for m, _ in picked]
        fscale = max(float(np.percentile(np.hypot(f[0], f[1]), 99)) for f in flows) or 1.0
        states = []
        for j, ((m, st), fl) in enumerate(zip(picked, flows)):
            i = m["state_k"] + 1
            Image.open(ro / RIO.RGB.format(i)).convert("RGB").save(out / f"s{i}_rgb.jpg", quality=86, optimize=True)
            d = np.load(rd / "depth" / f"{j:05d}.npy")
            jpg(depth_rgb(d), out / f"s{i}_depth.jpg")
            jpg(FL.flow_to_rgb(fl, None, fscale, style="dark"), out / f"s{i}_flow.jpg")
            rows = []
            for r in st.get("labels", []):
                P = np.asarray(r["P"], float)
                rows.append({"P": np.round(P[::2], 2).tolist(), "speeds": np.round(np.asarray(r["speeds"], float)[::2], 2).tolist(),
                             "class": r.get("class_tag"), "cost": round(float(r["cost"]), 3),
                             "arrived_at": r.get("arrived_at")})
            near_d = d[np.isfinite(d)]
            states.append({"i": i, "k": m["state_k"], "p": m["t_world"], "R_wb": m["R_wb"],
                           "speed": st["state"]["speed"], "goal": st["goal"]["position"], "goal_case": st.get("goal_case"),
                           "rows": rows, "files": {k: f"s{i}_{k}.jpg" for k in ("rgb", "depth", "flow")},
                           "nearest_m": round(float(near_d.min()), 2) if near_d.size else None})

        z = np.load(course / "cloud.npz")
        pts = z["points"][z["region"] >= 0]
        rng = np.random.default_rng(0)
        if len(pts) > a.cloud_points:
            pts = pts[rng.choice(len(pts), a.cloud_points, replace=False)]
        q = np.clip(np.rint(pts * 100), -32767, 32767).astype("<i2")
        (out / "cloud.bin").write_bytes(q.tobytes())
        cj = json.loads((course / "course.json").read_text())
        meta = {"pid": pid, "family": s["family"], "seed": s["seed"], "level": s["level"], "theme": s["theme"],
                "tier": s["tier"], "bounds": spec["bounds"], "hdri": cj.get("hdri"), "cloud_points": int(len(pts)),
                "cloud_total": int((z["region"] >= 0).sum()), "flow_scale_px": round(fscale, 1),
                "hfov_deg": poses["hfov_deg"], "states": states,
                "shell": (cj.get("shell") or {}).get("rect")}
        (out / "scene.json").write_text(json.dumps(meta))
        subprocess.run(["rm", "-rf", str(rd)])
        index.append({k: meta[k] for k in ("pid", "family", "seed", "level", "theme", "tier")})
        print(f"{pid:20s} states {[st['i'] for st in states]}  cloud {len(pts)} of {meta['cloud_total']}  flow scale {fscale:.1f} px", flush=True)
    (a.out / "index.json").write_text(json.dumps(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
