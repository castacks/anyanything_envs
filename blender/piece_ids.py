"""Per-piece instance ids for kit courses (no bpy here, so it is testable): ``cloud.npz inst`` = one id per
placed primitive / kit object, the way Wenshan courses carry ``metadata.json`` instance ids (kits/wenshan.py),
so a dataset catalogue has one object per piece. Off unless ``build_course.py --piece-ids``.

``kits.base.Ctx`` stamps every object it places with a ``piece`` property (one id per ``box`` / ``cylinder`` /
``sphere`` / ``mesh`` / ``panel`` call, one per library ``instance``, one per patch primitive however many
boxes it is made of) and keeps the catalogue row. :func:`labels_for` turns the collidable object names of
``build_course.collidable_triangles`` (``<tag>/<mesh>`` for realised instances, ``<mesh>`` otherwise) into
the per-name id, giving a fresh id to any collidable object a kit made outside the Ctx."""

from __future__ import annotations

from collections import Counter

#: the catalogue class of a piece, in ``pair_draw.OBS_CLS``'s vocabulary
CLASS_OF_KIND = {"prim": "insert", "piece": "prop", "tree": "tree"}


def labels_for(names: list, piece_of_name: dict, pieces: list) -> tuple[dict, list]:
    """``(inst_of_name, catalogue)``: the id of every collidable object name (its own name, or its instance
    tag's) and the catalogue rows (``pieces`` plus one ``other`` row per unmapped name)."""
    cat = [dict(p) for p in pieces]
    ids = {int(p["id"]) for p in cat}
    nxt = (max(ids) + 1) if ids else 0
    inst_of = {}
    for n in names:
        tag = n.split("/")[0]
        pid = piece_of_name.get(n, piece_of_name.get(tag))
        if pid is None:
            pid = nxt
            nxt += 1
            cat.append({"id": pid, "name": tag, "kind": "other", "cls": "building", "key": None, "collide": 1})
            piece_of_name = {**piece_of_name, tag: pid}
        inst_of[n] = int(pid)
    return inst_of, cat


def catalogue(kit: str, cat: list, conventions: dict | None = None) -> dict:
    return {"kind": "kit", "kit": kit, "n_pieces": len(cat), "classes": dict(Counter(p["cls"] for p in cat)),
            "kinds": dict(Counter(p["kind"] for p in cat)), "pieces": cat,
            "conventions": {"inst": "cloud.npz inst = pieces[].id (one per placed primitive / kit object, "
                                    "build_course.py --piece-ids); -1 = the floor grid",
                            "cls": "pair_draw.OBS_CLS vocabulary: insert = the patch primitives, prop / tree = library "
                                   "pieces, building = kit structure (boxes, panels, meshes)", **(conventions or {})}}
