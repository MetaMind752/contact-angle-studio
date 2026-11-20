"""Known geometry, independent fibre seed, and public serialization contract."""

import json

import cv2
import pytest
from fibrous_fixtures import cases

from contact_angle.fibrous import analyse_fibrous, measure_fibrous_bytes


@pytest.mark.parametrize("case", list(cases(fibres=True, seeds=(202,))), ids=lambda case: case.name)
def test_holdout_geometry_both_sides_under_five(case):
    result = analyse_fibrous(case.image, case.options["baseline"])
    for side, truth in zip(("left", "right"), case.truth):
        assert abs(result[side]["angle"] - truth) < 5
        assert result[side]["min"] <= result[side]["angle"] <= result[side]["max"]
        accepted = [r for r in result["runs"] if r["side"] == side and r["status"] == "accepted"]
        assert accepted
        assert all(r["cv"] <= 1 and r["rms"] <= 1 for r in accepted)
    assert result["baseline"] == case.options["baseline"]


def test_public_measurement_result_serializable_and_annotation():
    case = next(cases(fibres=True, seeds=(202,)))
    _, png = cv2.imencode(".png", case.image)
    result, annotation = measure_fibrous_bytes(png.tobytes(), case.options["baseline"])
    json.dumps(result, allow_nan=False)
    assert len(result["image_sha256"]) == 64
    assert (
        cv2.imdecode(__import__("numpy").frombuffer(annotation, "uint8"), cv2.IMREAD_COLOR).shape
        == case.image.shape
    )
    assert result["runs"]
    assert "not a confidence interval" in result["sensitivity_definition"]


def test_strict_comparison_boundary():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "batch", Path(__file__).parents[1] / "scripts/compare_fibrous_batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.strict_pass(4.999, -4.999)
    assert not module.strict_pass(5, 0)
    assert not module.strict_pass(0, -5)
