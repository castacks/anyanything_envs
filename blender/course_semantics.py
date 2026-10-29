"""What a place kit's dressing changes about the tier-A/B/C patch it dresses, in one shape.

    import course_semantics as CS
    CS.of_course(course_json)   # [{prim, kind, change, detail, z_top}, ...] ([] when nothing)
    CS.summary(entries)         # {"n": 3, "kinds": {"coping": 2, "posts_under": 1}}

Standard library only: ``build_course.py`` calls it inside Blender (it writes the normalised list
to course.json's top-level ``semantics_changed``) and the pipeline outside it (finalize QA rows,
the export's ``scenes.jsonl``), reading courses built before that through the same normaliser.

The kits record the change in ``chosen.semantics_changed`` in three formats:

* **strings** (street, alley, port, garage, warehouse, farm, backyard): one sentence per change,
  e.g. ``"wall tops capped (+6 cm)"``;
* **dicts** (forest, park, ruins, coast): ``{prim, kind, change, detail[, z_top]}``, or ``prims``
  (a list) instead of ``prim`` for one change applied to several primitives (``root_flare``);
* **a flag** (industrial, construction): ``False`` plus ``semantics_notes`` -- nothing that changes
  a route.

Normalised, every entry has all five fields (missing ones ``None``): a dict keeps its own, one
entry per primitive of ``prims``; a string becomes ``detail`` with a ``change`` read off its words
(:data:`TEXT_CHANGES`, ``"note"`` when none match); ``False`` / ``None`` is ``[]``; ``True`` is
one ``"unspecified"`` entry carrying the notes.
"""

from __future__ import annotations

from collections import Counter
import re

FIELDS = ("prim", "kind", "change", "detail", "z_top")

#: the change a kit's sentence describes, first match wins (lower case). The names follow the
#: dict kits' where one exists (``coping``, ``posts_under``, ``enclosure``)
TEXT_CHANGES = (
    ("ceiling", r"^(ceiling|roof)\b"),                      # garage / warehouse ceiling_semantics
    ("closed_to_ceiling", r"closed up to the ceiling"),
    ("hung", r"hung from"),
    ("extended", r"continued up|extended to"),
    ("enclosure", r"building (fronts|backs|lines)|container walls"),
    ("overhead", r"cable|laundry|festoon|span-sign"),
    ("posts_under", r"posts under"),
    ("rope_above", r"ropes from"),
    ("lamphead", r"lamp head"),
    ("coping", r"capped|caps on|coping|cone roofs"),
    ("pollard", r"pollard"),
    ("hedge", r"hedge"),
    ("obstacles_added", r"inside the patch"),
)


def text_change(s: str) -> str:
    t = s.lower()
    for name, pat in TEXT_CHANGES:
        if re.search(pat, t):
            return name
    return "note"


def _entry(prim=None, kind=None, change=None, detail=None, z_top=None) -> dict:
    return {"prim": prim, "kind": kind, "change": change or "note", "detail": detail, "z_top": z_top}


def normalise(value, notes=None) -> list[dict]:
    """``chosen.semantics_changed`` in any of the kits' formats (or an already normalised list:
    idempotent) -> ``[{prim, kind, change, detail, z_top}]``."""
    if value is None or value is False:
        return []
    if value is True:
        return [_entry(change="unspecified", detail="; ".join(map(str, notes)) if notes else None)]
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return [_entry(detail=str(value))]
    out = []
    for e in value:
        if isinstance(e, str):
            out.append(_entry(change=text_change(e), detail=e))
        elif isinstance(e, dict):
            if "prim" in e:
                prims = [e["prim"]]
            elif isinstance(e.get("prims"), (list, tuple)) and e["prims"]:
                prims = list(e["prims"])
            else:
                prims = [e.get("prims")]
            for p in prims:
                out.append(_entry(prim=p, kind=e.get("kind"), change=e.get("change"), detail=e.get("detail"),
                                  z_top=e.get("z_top")))
        elif e is not None:
            out.append(_entry(detail=str(e)))
    return out


def from_chosen(chosen: dict | None) -> list[dict]:
    c = chosen or {}
    return normalise(c.get("semantics_changed"), c.get("semantics_notes"))


def of_course(course_json: dict | None) -> list[dict]:
    """A course's normalised list: its top-level ``semantics_changed`` (build_course writes it
    since 2026-09-28), else its kit's ``chosen`` record read through :func:`normalise`; a v1-theme
    course (no kit, no record) has none."""
    cj = course_json or {}
    top = cj.get("semantics_changed")
    if isinstance(top, list):
        return normalise(top)
    return from_chosen(cj.get("chosen"))


def summary(entries: list[dict]) -> dict:
    """``{"n": entries, "kinds": {change: count}}`` -- the scene list's column."""
    return {"n": len(entries), "kinds": dict(sorted(Counter(e["change"] for e in entries).items()))}
