"""index.html for the forest smoke test, from make_sheet.py's metrics.json (every number on the page comes from it).

    python infinigen/page/make_page.py --assets out/page [--gen-json <extra.json>]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

LOOK_NAMES = {"A0": "A0 · midday sun (Infinigen's own sky)", "A1": "A1 · low sun ahead of the drone",
              "A2": "A2 · overcast haze", "A3": "A3 · dusk, sun just below the horizon (+2 EV after render, standing in for auto-exposure)"}

CSS = """
:root{
  /* layout: one reading column for text, full-width image grids; scene cards stack */
  --bg:#f5f6f3; --panel:#ffffff; --fg:#17201b; --muted:#5f6a63; --rule:#dde2da;
  --accent:#2d6a4f; --wire:#d4612b; --chip:#e8eee8;
  --display:"Archivo", "Helvetica Neue", Arial, sans-serif;
  --body:"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1311; --panel:#161b18; --fg:#e4eae5; --muted:#9aa69e; --rule:#29312c;
  --accent:#7cc19c; --wire:#f08a55; --chip:#1f2622; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f1311; --panel:#161b18; --fg:#e4eae5; --muted:#9aa69e; --rule:#29312c;
  --accent:#7cc19c; --wire:#f08a55; --chip:#1f2622; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.55;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:28px 64px}
main{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:44px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;letter-spacing:-0.01em}
h1{font-size:2.1rem;font-weight:700;font-stretch:110%}
h2{font-size:1.35rem;font-weight:650}
h3{font-size:1.02rem;font-weight:650}
p{margin:0;max-width:68ch}
.lede{display:flex;flex-direction:column;gap:12px}
.eyebrow{font-family:var(--mono);font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.muted{color:var(--muted)}
section{display:flex;flex-direction:column;gap:16px}
.grid{display:grid;gap:10px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:820px){.g3,.g4{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.g2,.g3,.g4{grid-template-columns:minmax(0,1fr)}}
figure{margin:0;display:flex;flex-direction:column;gap:6px;min-width:0}
figure img{width:100%;height:auto;display:block;border-radius:3px;background:var(--chip)}
figcaption{font-size:.8rem;color:var(--muted);font-family:var(--mono)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:18px;display:flex;flex-direction:column;gap:16px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-family:var(--mono);font-size:.75rem;background:var(--chip);border-radius:999px;padding:3px 10px}
.chip b{font-weight:600}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;min-width:520px;width:100%}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
td.num{font-family:var(--mono)}
ul.plain{margin:0;padding-left:1.1em;display:flex;flex-direction:column;gap:6px;max-width:72ch}
.wire{color:var(--wire);font-weight:600}
.split{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:18px}
@media (max-width:820px){.split{grid-template-columns:minmax(0,1fr)}}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:.88rem}
.kv dt{color:var(--muted)} .kv dd{margin:0;font-family:var(--mono)}
a{color:var(--accent)}
"""


def esc(s):
    return html.escape(str(s))


def fig(src, cap):
    return f'<figure><img src="{esc(src)}" alt="{esc(cap)}" loading="lazy"><figcaption>{esc(cap)}</figcaption></figure>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--gen-json", type=Path, default=None, help="generation timings etc. per scene tag")
    ap.add_argument("--labels", default="{}", help='json {tag: "label"}')
    ap.add_argument("--notes", type=Path, default=None, help="json: cost table, fixes, ladder")
    a = ap.parse_args()
    M = json.loads((a.assets / "metrics.json").read_text())
    G = json.loads(a.gen_json.read_text()) if a.gen_json else {}
    labels = json.loads(a.labels)
    refs = sorted(p.name for p in a.assets.glob("ref_*.jpg"))
    tags = list(M)

    out = ['<title>Forest Smoke Test</title>', '<meta name="claude:viewport-fit-sides" content="cover">',
           '<link rel="preconnect" href="https://fonts.googleapis.com">',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&'
           'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">',
           f"<style>{CSS}</style>", "<main>"]

    # header
    out.append('<header class="lede"><span class="eyebrow">OOD test set · smoke test · Infinigen forests</span>'
               '<h1>Forest Smoke Test</h1>'
               '<p>Two Infinigen forest scenes generated on gs2\'s CPU and rendered with the Starling front camera of '
               'our Blender screen (640 × 480, 87° horizontal FOV, level, 1.6–2.4 m above ground). Each scene is a '
               'square flight box, fully populated, with a collision cloud and the flight-band map the difficulty '
               'axes would be measured on. The point is to see the look before generating a batch.</p></header>')

    # side by side with Wenshan
    if refs:
        cells = [fig(refs[0], "Wenshan test course open_3002178 (\"woodland\"), ViT-B screen flight frame")]
        for t in tags:
            f = [x for x in M[t]["files"] if x.endswith("_v0_A0.jpg")] or M[t]["files"][:1]
            if f:
                cells.append(fig(f[0], f"{labels.get(t, t)} · v0 · A0"))
        out.append('<section><h2>Next to what we test on now</h2>'
                   '<p>Same camera on all three. The Wenshan "woodland" course is a town with floating props and '
                   'scattered trees; the Infinigen scenes are continuous forest with procedural trees, undergrowth, '
                   'logs and uneven ground, built from assets our training data has never seen.</p>'
                   f'<div class="grid g3">{"".join(cells)}</div></section>')

    # scene cards
    for t in tags:
        m = M[t]
        b, box = m["band"], m["box"]
        g = G.get(t, {})
        a0 = [x for x in m["files"] if x.endswith("_A0.jpg")]
        views = "".join(fig(x, f"{x.split('_', 1)[1].rsplit('.', 1)[0]} · {m['views'][int(x.split('_v')[1][0])]['agl']:.1f} m AGL")
                        for x in a0)
        depth = [k for k in m["depth_median_m"]]
        dfig = fig(f"{t}_{depth[0]}.png", f"{depth[0]}: true z-depth from the same pose, 0 m (light) to 25 m and sky "
                   "(dark); the haze volume is hidden for the depth pass") if depth else ""
        rt = [x["s"] for x in m["timings"] if x["view"] != "top"]
        cnt = b.get("assets_in_box", {})
        area = b.get("box_area_m2", (2 * box["half"]) ** 2)
        ntree = cnt.get("TreeFactory", 0)
        chips = [f'<span class="chip">trees in box <b>{ntree}</b> ({ntree / area:.3f}/m²)</span>',
                 f'<span class="chip">bushes in box <b>{cnt.get("BushFactory", 0)}</b></span>',
                 f'<span class="chip">box <b>{2 * box["half"]:.0f} × {2 * box["half"]:.0f} m</b></span>',
                 f'<span class="chip">1–3 m slice blocked <b>{100 * b["blocked_frac"]:.0f} %</b></span>',
                 f'<span class="chip">median clearance at 1–3 m <b>{b["median_clearance_m"]:.1f} m</b></span>',
                 f'<span class="chip">canopy cover <b>{100 * b.get("canopy_cover_frac", 0):.0f} %</b>'
                 + (f', base <b>{b["canopy_base_m"]:.1f} m</b>' if b.get("canopy_base_m") else "") + '</span>',
                 f'<span class="chip">ground relief <b>{b["ground_relief_m"]:.1f} m</b></span>']
        if b.get("thin_points_frac", 0) > 0:
            chips.append('<span class="chip">thin <span class="wire">wires</span> <b>on</b></span>')
        kv = [("seed", box["seed"]), ("Infinigen tree_density", (f'{box["tree_density"]:.2f}/m² (before masks)' if box.get("tree_density")
                                             else f'{box.get("scene_type", "forest")} default')), ("generation (CPU)", f'{g.get("gen_min", float("nan")):.0f} min' if g else "–"),
              ("collision cloud", f'{b["points"]:,} pts (0.1 m voxels)'),
              ("render, 640×480 CPU", f'{min(rt):.0f}–{max(rt):.0f} s / frame at {m.get("samples", 64)} spp' if rt else "–"),
              ("scene on disk", g.get("disk", "–"))]
        out.append(f'<section class="card"><div style="display:flex;flex-direction:column;gap:8px">'
                   f'<span class="eyebrow">{esc(t)}</span><h2>{esc(labels.get(t, t))}</h2>'
                   f'<div class="chips">{"".join(chips)}</div></div>'
                   f'<div class="grid g4">{views}</div>'
                   f'<div class="split"><div class="grid g2">{fig(f"{t}_top.jpg", "top-down render (dashed = flight box)")}'
                   f'{fig(f"{t}_plan.png", "plan: dark = obstacle at 1–3 m over the ground, light = only above or below")}</div>'
                   f'<div style="display:flex;flex-direction:column;gap:14px">{dfig}'
                   f'<dl class="kv">{"".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in kv)}</dl></div></div>'
                   '</section>')

    # appearance axis
    for t in tags:
        looks = [x for x in M[t]["files"] if x.startswith(f"{t}_v0_")]
        if len(looks) > 1:
            cells = "".join(fig(x, LOOK_NAMES.get(x.rsplit("_", 1)[1][:2], x)) for x in looks)
            out.append(f'<section><h2>Appearance axis, same geometry</h2><p>{esc(labels.get(t, t))}, view v0, '
                       're-lit four ways. The geometry and the collision cloud are identical, so the depth oracle '
                       'should score the same across these while the RGB students should drop: the gap between the '
                       'two is the perception penalty.</p>'
                       f'<div class="grid g4">{cells}</div></section>')
            break

    # costs and next steps (authored notes, passed in as json)
    notes = json.loads(a.notes.read_text()) if a.notes else {}
    if notes.get("cost_rows"):
        rows = "".join("<tr>" + "".join(f'<td class="{"num" if i else ""}">{esc(c)}</td>' for i, c in enumerate(r))
                       + "</tr>" for r in notes["cost_rows"])
        head = "".join(f"<th>{esc(h)}</th>" for h in notes["cost_head"])
        out.append(f'<section><h2>What it costs</h2><p>{esc(notes.get("cost_lede", ""))}</p>'
                   f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
                   '</section>')
    if notes.get("fixes"):
        lis = "".join(f"<li>{esc(x)}</li>" for x in notes["fixes"])
        out.append(f'<section><h2>Before generating a batch</h2><ul class="plain">{lis}</ul></section>')
    if notes.get("ladder_rows"):
        rows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in notes["ladder_rows"])
        head = "".join(f"<th>{esc(h)}</th>" for h in notes["ladder_head"])
        out.append(f'<section><h2>Proposed difficulty axes</h2><p>{esc(notes.get("ladder_lede", ""))}</p>'
                   f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
                   '</section>')
    out.append("</main>")
    (a.assets / "index.html").write_text("\n".join(out))
    print("PAGE_DONE", a.assets / "index.html")


if __name__ == "__main__":
    main()
