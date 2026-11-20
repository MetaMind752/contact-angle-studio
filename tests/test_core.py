"""Algorithm validation from independently rendered analytic silhouettes."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from contact_angle.core import measure_bytes
from contact_angle.models import MeasurementError, MeasurementOptions

_spec = importlib.util.spec_from_file_location(
    "synthetic_fixtures", Path(__file__).resolve().parents[1] / "examples/generate_examples.py"
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def measure(image, config, **overrides):
    options = {**config, "sensitivity": False, **overrides}
    return measure_bytes(fixtures.png_bytes(image), MeasurementOptions.model_validate(options))


@pytest.mark.parametrize("angle", [30, 60, 90, 120, 150])
@pytest.mark.parametrize("method,tolerance", [("circle", 1.5), ("local_polynomial", 3.0)])
def test_analytic_circular_angles(angle, method, tolerance):
    image, config = fixtures.circular_cap(angle)
    output = measure(image, config, method=method)
    result = output.result
    assert abs(result.left.angle_deg - angle) < tolerance
    assert abs(result.right.angle_deg - angle) < tolerance
    assert result.mean_angle_deg == pytest.approx(
        (result.left.angle_deg + result.right.angle_deg) / 2
    )
    assert result.image_width == image.shape[1]
    assert result.image_height == image.shape[0]
    assert len(result.image_sha256) == 64
    annotated = cv2.imdecode(np.frombuffer(output.annotation_png, np.uint8), cv2.IMREAD_COLOR)
    assert annotated.shape[:2] == image.shape


@pytest.mark.parametrize("tilt", [-12, 12])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("angle", [30, 120])
def test_tilt_and_endpoint_order(tilt, reverse, angle):
    image, config = fixtures.circular_cap(angle, tilt_deg=tilt)
    if reverse:
        config["baseline"].reverse()
    result = measure(image, config).result
    assert result.left.angle_deg == pytest.approx(angle, abs=3)
    assert result.right.angle_deg == pytest.approx(angle, abs=3)


def test_light_polarity_and_roi_preserve_global_coordinates():
    image, config = fixtures.circular_cap(90, polarity="light")
    result = measure(image, config, roi={"x": 200, "y": 350, "width": 500, "height": 280}).result
    assert result.mean_angle_deg == pytest.approx(90, abs=3)
    assert result.left.contact_point.x == pytest.approx(230, abs=3)
    assert result.right.contact_point.x == pytest.approx(670, abs=3)
    assert result.left.contact_point.y == pytest.approx(600, abs=0.01)


def test_asymmetric_shape_has_distinct_correct_local_angles():
    image, config, expected = fixtures.asymmetric_drop()
    result = measure(image, config).result
    assert result.left.angle_deg == pytest.approx(expected[0], abs=3)
    assert result.right.angle_deg == pytest.approx(expected[1], abs=3)
    assert result.asymmetry_deg > 10


def test_blank_image_is_rejected():
    image, config = fixtures.circular_cap(90)
    with pytest.raises(MeasurementError):
        measure(np.full_like(image, 240), config)


def test_non_image_is_rejected():
    _, config = fixtures.circular_cap(90)
    with pytest.raises(MeasurementError):
        measure_bytes(
            b"not an image", MeasurementOptions.model_validate({**config, "sensitivity": False})
        )


def test_drop_separated_from_baseline_is_rejected():
    image, config = fixtures.circular_cap(90)
    for point in config["baseline"]:
        point["y"] += 25
    with pytest.raises(MeasurementError):
        measure(image, config)


def test_clipped_drop_is_rejected():
    image, config = fixtures.circular_cap(90)
    with pytest.raises(MeasurementError):
        measure(image, config, roi={"x": 300, "y": 350, "width": 400, "height": 280})


def test_roi_outside_image_is_rejected():
    image, config = fixtures.circular_cap(90)
    with pytest.raises(MeasurementError):
        measure(image, config, roi={"x": 800, "y": 350, "width": 400, "height": 280})


def test_sensitivity_report_is_computed_from_accepted_methods():
    image, config = fixtures.circular_cap(90)
    result = measure(image, config, sensitivity=True).result
    report = result.sensitivity
    assert report is not None
    accepted = [run for run in report.runs if run.status == "accepted"]
    assert len(accepted) >= 3
    assert report.accepted_runs == len(accepted)
    assert report.failed_runs == sum(run.status == "failed" for run in report.runs)
    assert report.excluded_runs == sum(run.status == "excluded" for run in report.runs)
    assert len(report.runs) == report.accepted_runs + report.failed_runs + report.excluded_runs
    for side in ("left", "right"):
        interval = getattr(report, side)
        values = [getattr(run, f"{side}_angle_deg") for run in accepted]
        assert interval.min_deg == pytest.approx(min(values))
        assert interval.max_deg == pytest.approx(max(values))
        assert interval.span_deg == pytest.approx(max(values) - min(values))
        assert interval.min_deg <= interval.median_deg <= interval.max_deg
        assert interval.min_deg <= getattr(result, side).angle_deg <= interval.max_deg
    assert result.parameters.baseline == MeasurementOptions.model_validate(config).baseline
    assert "not a confidence interval" in report.interpretation.lower()


def test_sensitivity_can_be_disabled():
    image, config = fixtures.circular_cap(90)
    assert measure(image, config).result.sensitivity is None


def test_fuzzy_image_preserves_auditable_method_results():
    image, config = fixtures.fuzzy_drop()
    result = measure(image, config, sensitivity=True).result
    report = result.sensitivity
    assert report is not None
    assert report.runs
    for run in report.runs:
        assert run.name
        assert run.method
        assert run.parameters
        if run.status in ("failed", "excluded"):
            assert run.error
        else:
            assert np.isfinite(run.left_angle_deg)
            assert np.isfinite(run.right_angle_deg)
    if report.accepted_runs == 0:
        assert report.left is None
        assert report.right is None
        assert result.status == "review"
    elif max(report.left.span_deg, report.right.span_deg) > report.alert_deg:
        assert report.sensitive
        assert result.status == "review"


def test_excluded_region_outside_image_is_rejected():
    image, config = fixtures.circular_cap(90)
    with pytest.raises(MeasurementError):
        measure(image, config, excluded_regions=[{"x": 850, "y": 400, "width": 200, "height": 50}])


def test_excluding_known_fibres_recovers_visible_cap_and_keeps_review_flag():
    image, config = fixtures.fuzzy_drop()
    unmasked = measure(image, config, sensitivity=True).result
    masked = measure(
        image, config, sensitivity=True, excluded_regions=fixtures.FIBRE_EXCLUSIONS
    ).result
    assert max(abs(unmasked.left.angle_deg - 90), abs(unmasked.right.angle_deg - 90)) > 10
    assert masked.left.angle_deg == pytest.approx(90, abs=3)
    assert masked.right.angle_deg == pytest.approx(90, abs=3)
    assert masked.sensitivity.accepted_runs > unmasked.sensitivity.accepted_runs
    # A successful point estimate does not erase the actual cross-method variation.
    assert masked.status == "review"
    assert masked.sensitivity.sensitive
