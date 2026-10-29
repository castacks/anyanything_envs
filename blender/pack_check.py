"""Does a course.blend open on a render node with nothing but itself?

    blender -b <course.blend> --factory-startup -P blender/pack_check.py [-- --out check.json]

Opens the file in a fresh Blender and checks every image, font, sound and
linked library the scene uses: each must be packed, and each packed image's
pixels must decode (a packed image loads from the .blend, never from its
recorded path; ``--pack-max-res`` builds record a temporary path that no
longer exists, so a pass there proves it). No ``reload()``: on a packed
image that re-packs from the recorded path. Then Blender's own
``report_missing_files`` runs. Prints one ``PACK_CHECK {...}`` JSON line (and
writes it to ``--out``); exits 1 if anything is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_common import script_args  # noqa: E402


def check() -> dict:
    imgs = [im for im in bpy.data.images if im.users and im.source in ("FILE", "SEQUENCE", "TILED")]
    unpacked = sorted(im.name for im in imgs if not im.packed_file)
    failed, sizes = [], []
    for im in imgs:
        if not im.packed_file:
            continue
        w, h = im.size
        if not (w > 0 and h > 0 and len(im.pixels) == w * h * im.channels and im.has_data):
            failed.append(im.name)
        sizes.append(max(w, h))
    fonts = sorted(f.name for f in bpy.data.fonts if f.users and f.filepath != "<builtin>" and not f.packed_file)
    sounds = sorted(s.name for s in bpy.data.sounds if s.users and not s.packed_file)
    libs = sorted(lib.filepath for lib in bpy.data.libraries)
    bpy.ops.file.report_missing_files()
    return {"file": bpy.data.filepath, "images": len(imgs), "packed": len(imgs) - len(unpacked),
            "unpacked": unpacked[:20], "failed_to_load": failed[:20], "fonts_unpacked": fonts,
            "sounds_unpacked": sounds, "libraries": libs,
            "recorded_paths_present": sum(Path(bpy.path.abspath(im.filepath)).exists() for im in imgs),
            "max_side_px": max(sizes) if sizes else 0,
            "sides_px": {str(s): sizes.count(s) for s in sorted(set(sizes))},
            "ok": not (unpacked or failed or fonts or sounds or libs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(script_args())
    r = check()
    line = json.dumps(r)
    print("PACK_CHECK " + line, flush=True)
    if a.out:
        a.out.write_text(line + "\n")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
