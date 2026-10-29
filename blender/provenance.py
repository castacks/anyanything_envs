"""Where a Blender course came from: the code, the library, the Blender (contract 6).

    from provenance import provenance, check_clean
    prov = provenance(repo, library_files, kits_order=KITS, argv=sys.argv)

Standard library only (runs inside Blender's Python and outside it; the
build gates and the unit tests import it without ``bpy``).

* ``git_sha`` / ``dirty`` -- the checkout the builder ran from. Dirty means
  ``git status --porcelain`` is not empty: a modified tracked file *or* an
  untracked one that is not ignored (an untracked kit module was exactly how
  the 09-25 kits escaped every clean checkout). ``dirty_files`` lists the
  first 50. ``dirty`` is None -- unknown, refused like dirt -- when ``git
  status`` itself fails (a corrupt index, a timeout) while ``rev-parse``
  still names a commit; ``dirty_files`` then carries git's error.
* ``library_sha`` -- SHA-256 over the library index files the build read
  (``library*.json`` + ``catalog*.json`` for the v2 library, ``manifest.json``
  for the v1 themes), each as ``name  sha256`` in name order. A new catalog
  file re-picks pieces for the same seed, so it changes the scene; this is
  the number that says so. The asset files themselves are not hashed (37 GB);
  ``course.json`` ``files`` lists every one a course uses.
* ``env_hooks`` -- kit test hooks set in the environment (``FOREST_KIND``,
  ``PARK_STYLE``): they change a scene without changing its arguments, so a
  pipeline build refuses them like a dirty tree.
* ``hash_randomization`` -- Blender's Python salts ``str`` hashes per process
  and ignores ``PYTHONHASHSEED`` unless started with ``--python-use-system-env``;
  a kit loop over a set of names would then build objects in another order
  on every run. No kit does today (the 09-27 street rebuild difference was
  ``bmesh.ops.create_uvsphere``'s face order, fixed in ``build_course``); a
  pipeline build still refuses a salted hash (``batch.BLENDER_ENV`` pins it).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

#: environment variables a kit reads as a test override (grep os.environ blender/kits)
ENV_HOOKS = ("FOREST_KIND", "PARK_STYLE")
#: the v2 library's index files, as ``kits.library.Library`` globs them
LIBRARY_GLOBS = ("library.json", "library_*.json", "catalog.json", "catalog_*.json")


def _git(repo: Path, *args) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(repo: Path) -> dict:
    """``{git_sha, dirty, dirty_files}`` of the checkout at ``repo`` (sha None outside git)."""
    r = _git(repo, "rev-parse", "HEAD")
    if r is None or r.returncode != 0:
        return {"git_sha": None, "dirty": None, "dirty_files": []}
    st = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    if st is None or st.returncode != 0:          # unknown is not clean: check_clean refuses dirty None
        err = "timed out or not runnable" if st is None else (st.stderr.strip() or f"rc {st.returncode}")
        return {"git_sha": r.stdout.strip(), "dirty": None, "dirty_files": [f"git status failed: {err}"[:300]]}
    lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
    return {"git_sha": r.stdout.strip(), "dirty": bool(lines), "dirty_files": lines[:50]}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_index_files(root: Path) -> list[Path]:
    """The v2 library index files a ``Library(root)`` reads, in its order."""
    root = Path(root).expanduser()
    out = []
    for pat in LIBRARY_GLOBS:
        out += sorted(root.glob(pat)) if "*" in pat else ([root / pat] if (root / pat).exists() else [])
    return out


def files_sha(paths) -> str | None:
    """SHA-256 over ``name  sha256(bytes)`` lines of ``paths`` in name order (None if no files)."""
    rows = sorted(f"{Path(p).name}  {file_sha256(Path(p))}" for p in paths)
    if not rows:
        return None
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def env_hooks(environ=None) -> dict:
    env = os.environ if environ is None else environ
    return {k: env[k] for k in ENV_HOOKS if env.get(k)}


def provenance(repo: Path, library_files, kits_order=None, argv=None, blender: dict | None = None,
               environ=None, git: dict | None = None) -> dict:
    """The provenance block of ``course.json``. ``library_files`` are the
    index files the build read (see module doc); ``blender`` is
    ``{"version": ..., "build_hash": ...}`` when run inside Blender; ``git``
    the ``git_state`` taken when the build started (the code it loaded),
    else taken now."""
    g = git if git is not None else git_state(repo)
    lib = [Path(p) for p in library_files]
    return {**g, "kits_order": list(kits_order) if kits_order is not None else None,
            "library_sha": files_sha(lib),
            "library_files": {p.name: file_sha256(p) for p in sorted(lib, key=lambda p: p.name)},
            "blender": blender or {}, "argv": list(argv) if argv is not None else None,
            "env_hooks": env_hooks(environ), "hash_randomization": bool(sys.flags.hash_randomization)}


def check_clean(prov: dict, recorded: bool = False) -> list[str]:
    """Why a pipeline build (``--require-clean``) must refuse this provenance; [] = fine.

    Fails closed: ``dirty`` must be exactly False (None = ``git status``
    failed). ``recorded`` is the build gate's reading of a ``course.json``
    ``provenance`` block: there a missing ``env_hooks`` or
    ``hash_randomization`` (a build older than the fields) refuses too."""
    why = []
    if not prov.get("git_sha"):
        why.append("not a git checkout")
    elif prov.get("dirty") is None:
        why.append("tree state unknown: " + "; ".join(prov.get("dirty_files", [])[:3] or ["no git status"]))
    elif prov.get("dirty") is not False:
        why.append("dirty tree: " + "; ".join(prov.get("dirty_files", [])[:10]))
    if prov.get("env_hooks"):
        why.append(f"kit test hooks set in the environment: {prov['env_hooks']}")
    elif recorded and "env_hooks" not in prov:
        why.append("env_hooks not recorded")
    if prov.get("hash_randomization") or (recorded and prov.get("hash_randomization") is not False):
        why.append("Python str hashes are salted per process (or not recorded): run Blender with "
                   "PYTHONHASHSEED=0 --python-use-system-env (batch.BLENDER_ENV)")
    return why
