"""The extracted primitive families reproduce the original sampler code byte for byte (golden_specs.json was
written from superfly_expert_sampler before the extraction)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from envgen.primitives import build_spec, patch_cloud

GOLDEN = json.loads((Path(__file__).parent / "golden_specs.json").read_text())["cases"]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: f"{c['tier']}-{c['family']}-{c['seed']}")
def test_spec_and_cloud_match_original(case):
    spec = build_spec(case["family"], case["seed"], tier=case["tier"])
    assert hashlib.sha256(spec.to_json().encode()).hexdigest() == case["spec_sha256"]
    pts, reg = patch_cloud(spec)
    got = hashlib.sha256(np.ascontiguousarray(pts, np.float64).tobytes()
                         + np.ascontiguousarray(reg, np.int16).tobytes()).hexdigest()
    assert len(pts) == case["n_points"]
    assert got == case["cloud_sha256"]
