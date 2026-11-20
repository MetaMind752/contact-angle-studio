"""Analytic geometry and invalid-observation checks for manual measurements."""

import json
import math

import numpy as np
import pytest

from contact_angle.manual import measure_manual
from contact_angle.models import MeasurementError


@pytest.fixture
def image():
    return np.full((600, 640, 3), 245, dtype=np.uint8)


def points(x, heights):
    return [{"x": float(xx), "y": float(500 - hh)} for xx, hh in zip(x, heights)]


def options(method="quadratic", heights=(0, 10, 25, 40, 60), left=90, right=90):
    hs = np.array(heights, dtype=float)
    return {
        "mode": "manual",
        "manual_method": method,
        "baseline": [{"x": 40, "y": 500}, {"x": 600, "y": 500}],
        "left_points": points(150 + hs / np.tan(np.deg2rad(left)), hs),
        "right_points": points(490 - hs / np.tan(np.deg2rad(right)), hs),
        "sensitivity": True,
    }


@pytest.mark.parametrize("angle", [30, 90, 150])
def test_two_point_lines_inside_angles_and_intersections(image, angle):
    result = measure_manual(image, options("tangent", (0, 40), angle, angle))
    assert result["left"]["angle_deg"] == pytest.approx(angle, abs=1e-8)
    assert result["right"]["angle_deg"] == pytest.approx(angle, abs=1e-8)
    assert result["left"]["contact_point"] == pytest.approx({"x": 150, "y": 500})
    assert result["right"]["contact_point"] == pytest.approx({"x": 490, "y": 500})
    for side in ("left", "right"):
        v = result[side]["tangent_vector"]
        assert math.hypot(v["x"], v["y"]) == pytest.approx(1)
        assert v["y"] < 0
    assert result["sensitivity"]["left"] is None
    assert result["sensitivity"]["right"] is None
    assert result["sensitivity"]["accepted_runs"] == 1
    assert result["status"] == "review"
    json.dumps(result, allow_nan=False)


def test_asymmetric_lines_preserve_both_angles(image):
    result = measure_manual(image, options("tangent", (0, 35), 30, 150))
    assert result["left"]["angle_deg"] == pytest.approx(30)
    assert result["right"]["angle_deg"] == pytest.approx(150)
    assert result["mean_angle_deg"] == pytest.approx(90)
    assert result["asymmetry_deg"] == pytest.approx(120)


@pytest.mark.parametrize("tilt", [-15, 12])
def test_tilted_baseline_rotation_and_reversed_endpoints(tilt):
    picture = np.zeros((700, 800, 3), np.uint8)
    a = math.radians(tilt)
    hs = np.array([0, 12, 30, 45], dtype=float)

    def to_image(u, h):
        return [{"x": float(60 + x * math.cos(a) + y * math.sin(a)),
                 "y": float(500 + x * math.sin(a) - y * math.cos(a))}
                for x, y in zip(np.asarray(u), np.asarray(h))]

    opt = {
        "manual_method": "quadratic",
        "baseline": to_image([600, 0], [0, 0]),
        "left_points": to_image(150 + hs / math.tan(math.radians(55)), hs),
        "right_points": to_image(450 - hs / math.tan(math.radians(120)), hs),
        "sensitivity": False,
    }
    result = measure_manual(picture, opt)
    assert result["left"]["angle_deg"] == pytest.approx(55, abs=1e-8)
    assert result["right"]["angle_deg"] == pytest.approx(120, abs=1e-8)
    assert result["left"]["contact_point"] == pytest.approx(to_image([150], [0])[0])


@pytest.mark.parametrize("method", ["quadratic", "cubic"])
def test_analytic_non_symmetric_curves_and_derivative(image, method):
    hs = np.array([0, 10, 22, 37, 55, 70], dtype=float)
    opt = options(method)
    if method == "quadratic":
        left = 150 + 0.3 * hs + 0.004 * hs**2
        right = 490 + 0.2 * hs - 0.002 * hs**2
        dl, dr = 0.3, 0.2
    else:
        left = 150 + 0.4 * hs - 0.003 * hs**2 + 0.00002 * hs**3
        right = 490 + 0.25 * hs + 0.002 * hs**2 - 0.00001 * hs**3
        dl, dr = 0.4, 0.25
    opt["left_points"], opt["right_points"] = points(left, hs), points(right, hs)
    result = measure_manual(image, opt)
    assert result["left"]["angle_deg"] == pytest.approx(math.degrees(math.atan2(1, dl)), abs=1e-7)
    assert result["right"]["angle_deg"] == pytest.approx(math.degrees(math.atan2(1, -dr)), abs=1e-7)
    assert result["left"]["contact_point"]["x"] == pytest.approx(150, abs=1e-8)
    assert result["right"]["contact_point"]["x"] == pytest.approx(490, abs=1e-8)
    assert result["left"]["fit_rms_px"] < 1e-8
    assert result["sensitivity"]["accepted_runs"] >= 2
    assert result["geometry"]["selected_points"]["left"] == opt["left_points"]
    assert len(result["geometry"]["fits"]) == 2
    assert result["geometry"]["fits"][0]["points"][0] == pytest.approx(result["left"]["contact_point"])


def test_short_analytic_circle_arc_recovers_contact_tangent(image):
    hs = np.array([0, 4, 9, 15, 20], dtype=float)
    radius = 130
    center_h = -radius * math.cos(math.radians(120))
    half = np.sqrt(radius**2 - (hs - center_h)**2)
    opt = options("quadratic")
    opt["left_points"], opt["right_points"] = points(320 - half, hs), points(320 + half, hs)
    result = measure_manual(image, opt)
    assert result["left"]["angle_deg"] == pytest.approx(120, abs=0.5)
    assert result["right"]["angle_deg"] == pytest.approx(120, abs=0.5)


def test_click_order_does_not_change_measurement(image):
    opt = options("cubic", (0, 8, 15, 23, 35, 49, 60))
    original = measure_manual(image, opt)
    opt["baseline"].reverse()
    rng = np.random.default_rng(142)
    rng.shuffle(opt["left_points"])
    rng.shuffle(opt["right_points"])
    shuffled = measure_manual(image, opt)
    for side in ("left", "right"):
        assert shuffled[side]["angle_deg"] == pytest.approx(original[side]["angle_deg"], abs=1e-10)
        assert shuffled[side]["contact_point"] == pytest.approx(original[side]["contact_point"])
    assert shuffled["sensitivity"] == original["sensitivity"]


@pytest.mark.parametrize("method,count", [("quadratic", 2), ("cubic", 3), ("tangent", 1)])
def test_method_never_silently_falls_back(image, method, count):
    opt = options(method)
    opt["left_points"] = opt["left_points"][:count]
    with pytest.raises(MeasurementError, match="需要") as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_point_count"


@pytest.mark.parametrize("bad", [{"x": -1, "y": 490}, {"x": 640, "y": 490}, {"x": 150, "y": 600}])
def test_outside_points_are_rejected(image, bad):
    opt = options()
    opt["left_points"][0] = bad
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_point_bounds"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "oops"])
def test_non_numeric_or_non_finite_points_rejected(image, bad):
    opt = options()
    opt["left_points"][0]["x"] = bad
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "invalid_manual_points"


def test_duplicate_points_rejected(image):
    opt = options()
    opt["left_points"][1] = {"x": 150.1, "y": 499.9}
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "duplicate_manual_points"


@pytest.mark.parametrize("baseline", [None, [], [{"x": 10, "y": 100}, {"x": 10, "y": 400}]])
def test_baseline_required_and_must_define_surface_direction(image, baseline):
    opt = options()
    opt["baseline"] = baseline
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code in ("manual_baseline_required", "invalid_baseline")


def test_points_below_solid_surface_rejected(image):
    opt = options()
    opt["left_points"][0]["y"] = 510
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_below_baseline"


def test_excessive_vertical_extrapolation_rejected(image):
    opt = options("quadratic", (35, 40, 45))
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_excessive_extrapolation"


def test_acute_tangent_cannot_extrapolate_far_horizontally(image):
    opt = options("tangent", (5, 9), 10, 10)
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_excessive_extrapolation"


def test_left_right_assignment_cannot_cross_contacts(image):
    opt = options()
    opt["left_points"], opt["right_points"] = opt["right_points"], opt["left_points"]
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_crossed_contacts"


def test_no_vertical_extent_rejected(image):
    opt = options()
    opt["left_points"] = [{"x": x, "y": 490} for x in (130, 140, 150)]
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "manual_short_arc"


def test_comparison_points_remain_fixed_and_no_range_from_one_candidate(image):
    opt = options("quadratic", (0, 12, 25))
    selected = json.dumps(opt, sort_keys=True)
    result = measure_manual(image, opt)
    assert result["sensitivity"]["accepted_runs"] == 1
    assert result["sensitivity"]["left"] is None
    assert result["sensitivity"]["right"] is None
    assert json.dumps(opt, sort_keys=True) == selected
    opt["sensitivity"] = False
    assert measure_manual(image, opt)["sensitivity"] is None


def test_unknown_method_rejected(image):
    opt = options()
    opt["manual_method"] = "magic"
    with pytest.raises(MeasurementError) as exc:
        measure_manual(image, opt)
    assert exc.value.code == "invalid_manual_method"
