#!/usr/bin/env python3
"""Fetch the v2 asset library: most of Poly Haven and a curated slice of
ambientCG, every item tagged so a place kit can ask for "an urban overcast
sky" or "a paving-stone ground" instead of naming files.

    python blender/fetch_library.py --root $BLENDER_LIBRARY [--only ph-models ph-hdris ph-textures acg]

Sources (both CC0, real-world scale):

* **Poly Haven** -- models as glTF (1k textures; 2k for anything 5 m or
  larger, facades and cliffs are seen close up), every HDRI that is not a
  studio (2k .hdr), and every texture that is not a fabric or food (2k jpg:
  diffuse, OpenGL normal, roughness, AO, metal and alpha where present).
  Models smaller than ``--min-model-m`` (hand tools, cutlery) are skipped:
  a drone cannot see them as obstacles.
* **ambientCG** -- PBR materials in the categories a site is made of
  (grounds, paving, asphalt, facades, bricks, metal, fences, bark, snow ...),
  capped per category by popularity, as 2K-JPG (colour, OpenGL normal,
  roughness, AO, metalness, opacity, displacement).

Writes ``<root>/library.json``: ``{"items": {"<source>:<id>": {...}}}`` with
``kind`` (model | hdri | material), ``source``, ``licence``, ``url``,
``categories`` and ``tags`` (lower-case), ``dims_m`` for models, and the
local paths (``path`` for a model's glTF or an HDRI, ``maps`` for a
material, ``alpha_maps`` for models whose cut-outs ship separately). Every
file is md5-checked (Poly Haven) or size-checked (ambientCG), so a re-run
fetches only what is missing. Standard library only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assets import API, UA, fetch, get_json  # noqa: E402

PH_SKIP_MODEL_CATS = {"food", "dishes", "books", "weapons", "firearms", "instrument", "creature",
                      "collection: moon", "wall decoration", "office"}
PH_SKIP_HDRI_CATS = {"studio"}
PH_SKIP_TEX_CATS = {"fabric", "cotton", "stretchy", "leather", "denim", "wool", "fleece", "collection: moon"}
PH_TEX_MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough", "AO": "ao", "Metal": "metal",
               "alpha": "alpha", "Displacement": "disp"}

#: ambientCG display category -> how many (most popular first); absent = skipped
ACG_CAPS = {
    "Ground": 127, "Asphalt": 46, "Gravel": 44, "Road": 31, "Concrete": 61, "Rocks": 26, "Rock": 40,
    "Paving Stones": 70, "Tiles": 25, "Bricks": 50, "Facade": 26, "Snow": 23, "Grass": 8, "Moss": 4,
    "Pathway": 4, "Tactile Paving": 7, "Wood Chips": 3, "Scattered Leaves": 9, "Leaf": 3,
    "Metal": 30, "Metal Plates": 20, "Painted Metal": 18, "Corrugated Steel": 15, "Sheet Metal": 4,
    "Diamond Plate": 10, "Metal Walkway": 14, "Rust": 10, "Grate": 2, "Pipe": 4, "Fence": 12, "Net": 8,
    "Painted Plaster": 18, "Plaster": 7, "Painted Bricks": 5, "Painted Wood": 17, "Wood Siding": 13,
    "Planks": 25, "Wood": 20, "Wood Floor": 10, "Roofing Tiles": 10, "Thatched Roof": 4, "Bark": 15,
    "Logs": 1, "Tree End": 5, "Office Ceiling": 6, "Surface Imperfections": 20, "Marble": 5,
    "Granite": 5, "Terrazzo": 5, "Travertine": 5, "Carpet": 6, "Chipboard": 4, "Cardboard": 4,
    "Plastic": 8, "Rubber": 4, "Styrofoam": 2,
}
ACG_MAPS = {"Color": "diff", "NormalGL": "nor_gl", "Roughness": "rough", "AmbientOcclusion": "ao",
            "Metalness": "metal", "Opacity": "alpha", "Displacement": "disp"}
ACG_API = "https://ambientcg.com/api/v2/full_json"


def _lower(xs) -> list[str]:
    return sorted({str(x).strip().lower() for x in xs if str(x).strip()})


# --- Poly Haven ---------------------------------------------------------------------------------

def ph_items(kinds: set, min_model_m: float) -> list[tuple]:
    """(key, job-builder, entry) per Poly Haven asset to fetch."""
    out = []
    if "ph-models" in kinds:
        for aid, v in get_json(f"{API}/assets?t=models").items():
            cats = set(_lower(v.get("categories", [])))
            dims = [d / 1000.0 for d in (v.get("dimensions") or [])]
            if cats & PH_SKIP_MODEL_CATS or (dims and max(dims) < min_model_m):
                continue
            out.append(("model", aid, v, dims))
    if "ph-hdris" in kinds:
        for aid, v in get_json(f"{API}/assets?t=hdris").items():
            if set(_lower(v.get("categories", []))) & PH_SKIP_HDRI_CATS:
                continue
            out.append(("hdri", aid, v, None))
    if "ph-textures" in kinds:
        for aid, v in get_json(f"{API}/assets?t=textures").items():
            if set(_lower(v.get("categories", []))) & PH_SKIP_TEX_CATS:
                continue
            out.append(("material", aid, v, None))
    return out


def ph_jobs(kind: str, aid: str, v: dict, dims, root: Path) -> tuple[list, dict]:
    files = get_json(f"{API}/files/{aid}")
    base = {"source": "polyhaven", "licence": "CC0-1.0", "url": f"https://polyhaven.com/a/{aid}",
            "categories": _lower(v.get("categories", [])), "tags": _lower(v.get("tags", [])),
            "authors": sorted(v.get("authors", {})), "name": v.get("name", aid)}
    if kind == "model":
        res = "2k" if dims and max(dims) >= 5.0 else "1k"
        g = files["gltf"][res]["gltf"]
        d = root / "ph" / "models" / aid
        jobs = [(g["url"], d / f"{aid}.gltf", g.get("md5"))]
        jobs += [(inc["url"], d / rel, inc.get("md5")) for rel, inc in g.get("include", {}).items()]
        alpha = {}
        for key, per_res in files.items():
            if not key.lower().endswith("alpha") or res not in per_res:
                continue
            fmt = "png" if "png" in per_res[res] else next(iter(per_res[res]))
            f = per_res[res][fmt]
            dest = d / "alpha" / f"{aid}_{key}_{res}.{fmt}"
            jobs.append((f["url"], dest, f.get("md5")))
            alpha[key[:-len("alpha")].rstrip("_").lower()] = str(dest.relative_to(root))
        e = {**base, "kind": "model", "path": str((d / f"{aid}.gltf").relative_to(root)), "dims_m": dims, "res": res}
        if alpha:
            e["alpha_maps"] = alpha
        return jobs, e
    if kind == "hdri":
        h = files["hdri"]["2k"]["hdr"]
        dest = root / "ph" / "hdris" / f"{aid}_2k.hdr"
        return [(h["url"], dest, h.get("md5"))], {**base, "kind": "hdri", "path": str(dest.relative_to(root)),
                                                  "evs_cap": v.get("evs_cap"), "whitebalance": v.get("whitebalance")}
    jobs, maps = [], {}
    for key, short in PH_TEX_MAPS.items():
        if key not in files or "2k" not in files[key]:
            continue
        per = files[key]["2k"]
        fmt = "jpg" if "jpg" in per else ("png" if "png" in per else None)
        if fmt is None:
            continue
        dest = root / "ph" / "textures" / aid / f"{aid}_{short}_2k.{fmt}"
        jobs.append((per[fmt]["url"], dest, per[fmt].get("md5")))
        maps[short] = str(dest.relative_to(root))
    dm = v.get("dimensions")
    return jobs, {**base, "kind": "material", "maps": maps,
                  "tile_m": round(max(dm) / 1000.0, 2) if dm else None}


# --- ambientCG ----------------------------------------------------------------------------------

def acg_items() -> list[dict]:
    got, off = [], 0
    while True:
        d = get_json(f"{ACG_API}?type=Material&limit=250&offset={off}&include=tagData,displayData,downloadData,dimensionsData")
        got += d["foundAssets"]
        off += 250
        if off >= d["numberOfResults"]:
            break
    by = {}
    for a in got:
        by.setdefault(a.get("displayCategory") or "", []).append(a)
    out = []
    for cat, cap in ACG_CAPS.items():
        out += sorted(by.get(cat, []), key=lambda a: -float(a.get("popularityScore") or 0))[:cap]
    return out


def acg_fetch(a: dict, root: Path, tries: int = 4) -> tuple[int, dict | None]:
    """Download the 2K-JPG zip into memory and keep the maps we use."""
    aid = a["assetId"]
    d = root / "acg" / aid
    zips = a["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    z = next((x for x in zips if x["attribute"] == "2K-JPG"), None)
    if z is None:
        return 0, None
    entry = {"source": "ambientcg", "licence": "CC0-1.0", "url": a.get("shortLink") or f"https://ambientcg.com/a/{aid}",
             "categories": _lower([a.get("displayCategory", "")]), "tags": _lower(a.get("tags") or []),
             "name": a.get("displayName", aid), "kind": "material",
             "tile_m": round(max(a.get("dimensionX") or 0, a.get("dimensionY") or 0) / 100.0, 2) or None}
    done = d / ".done"
    if done.exists():
        entry["maps"] = json.loads(done.read_text())
        return 0, entry
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(z["downloadLink"], headers=UA), timeout=300) as r:
                blob = r.read()
            if z.get("size") and len(blob) != int(z["size"]):
                raise IOError(f"size {len(blob)} != {z['size']}")
            maps = {}
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for n in zf.namelist():
                    stem = Path(n).stem
                    for key, short in ACG_MAPS.items():
                        if stem.endswith(f"_{key}") and n.lower().endswith((".jpg", ".png")):
                            dest = d / f"{aid}_{short}{Path(n).suffix.lower()}"
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(zf.read(n))
                            maps[short] = str(dest.relative_to(root))
            done.write_text(json.dumps(maps))
            entry["maps"] = maps
            return len(blob), entry
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return 0, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["ph-models", "ph-hdris", "ph-textures", "acg"])
    ap.add_argument("--min-model-m", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    lib_path = root / "library.json"
    lib = json.loads(lib_path.read_text()) if lib_path.exists() else {"items": {}}
    kinds = set(a.only)
    failed = []
    t0 = time.time()

    todo = ph_items(kinds, a.min_model_m)
    print(f"poly haven: {len(todo)} assets to check", flush=True)

    def one_ph(t):
        kind, aid, v, dims = t
        try:
            jobs, e = ph_jobs(kind, aid, v, dims, root)
            n = sum(fetch(*j) for j in jobs)
            return f"ph:{aid}", e, n, None
        except Exception as ex:  # noqa: BLE001
            return f"ph:{aid}", None, 0, f"{type(ex).__name__}: {ex}"

    got = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (key, e, n, err) in enumerate(ex.map(one_ph, todo)):
            if err:
                failed.append(f"{key}: {err}")
            else:
                lib["items"][key] = e
                got += n
            if i % 100 == 99:
                print(f"  {i + 1}/{len(todo)}  {got / 1e9:.2f} GB  {time.time() - t0:.0f} s", flush=True)
                lib_path.write_text(json.dumps(lib, indent=1, sort_keys=True))

    if "acg" in kinds:
        acg = acg_items()
        print(f"ambientcg: {len(acg)} materials", flush=True)

        def one_acg(x):
            try:
                n, e = acg_fetch(x, root)
                return f"acg:{x['assetId']}", e, n, None
            except Exception as ex:  # noqa: BLE001
                return f"acg:{x['assetId']}", None, 0, f"{type(ex).__name__}: {ex}"

        with ThreadPoolExecutor(max(2, a.workers // 2)) as ex:
            for i, (key, e, n, err) in enumerate(ex.map(one_acg, acg)):
                if err or e is None:
                    failed.append(f"{key}: {err or 'no 2K-JPG'}")
                else:
                    lib["items"][key] = e
                    got += n
                if i % 50 == 49:
                    print(f"  acg {i + 1}/{len(acg)}  {got / 1e9:.2f} GB  {time.time() - t0:.0f} s", flush=True)
                    lib_path.write_text(json.dumps(lib, indent=1, sort_keys=True))

    lib["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    lib_path.write_text(json.dumps(lib, indent=1, sort_keys=True))
    kinds_n = {}
    for e in lib["items"].values():
        kinds_n[e["kind"]] = kinds_n.get(e["kind"], 0) + 1
    print(f"{len(lib['items'])} items {kinds_n}; {got / 1e9:.2f} GB fetched in {time.time() - t0:.0f} s")
    for f in failed:
        print("FAILED", f, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
