"""Analytic original-coordinate tangents and realistic connected-neck occlusion."""

import cv2
import numpy as np
import pytest

from contact_angle.models import MeasurementError
from contact_angle.pendant_fit import fit_pendant


def circle(angle=122, noisy=False):
    h, w = 470, 650
    rad, cx, y0 = 125.0, 328.0, 100.0
    theta = np.deg2rad(angle)
    cy = y0 - rad * np.cos(theta)
    sy, sx = np.mgrid[:h, :w]
    inside = (sx - cx) ** 2 + (sy - cy) ** 2 <= rad**2
    gray = np.where((sy <= y0) | inside, 19, 242).astype("uint8")
    cv2.circle(gray, (round(cx + 15), round(cy + 0.48 * rad)), 13, 222, -1)
    true_contact = cx + np.array([-1, 1]) * rad * np.sin(theta)
    if noisy:
        lx, rx = np.round(true_contact).astype(int)
        # Dark triangular tufts attached to both the support and the liquid.
        for x, sign in [(lx, -1), (rx, 1)]:
            cv2.fillPoly(
                gray,
                [
                    np.array(
                        [
                            [x - 9 * sign, int(y0)],
                            [x + 19 * sign, int(y0)],
                            [x + 21 * sign, int(y0) + 14],
                            [x + 4 * sign, int(y0) + 22],
                        ],
                        np.int32,
                    )
                ],
                19,
            )
    gray = cv2.GaussianBlur(gray, (5, 5), 0.85)
    surface = [{"x": 0.0, "y": y0}, {"x": w - 1.0, "y": y0}]
    surface_data = {
        "baseline": surface,
        "baseline_points": surface,
        "surface_curve": surface,
        "footprint": [cx - rad, cx + rad],
        "uncertainty_px": 0.85,
        "surface_variants": [
            {
                "name": "tilted",
                "surface_curve": [{"x": 0.0, "y": y0 - 2.0}, {"x": w - 1.0, "y": y0 + 2.0}],
            }
        ],
    }
    phase = np.linspace(np.pi / 2 + theta, np.pi / 2 - theta, 27)
    manual = [{"x": cx + rad * np.cos(t), "y": cy + rad * np.sin(t)} for t in phase]
    return gray, surface_data, manual, true_contact


@pytest.mark.parametrize("angle", [56.0, 95.0, 122.0, 151.0])
def test_circle_liquid_interior_both_sides(angle):
    gray, surface, _, expected = circle(angle)
    r = fit_pendant(gray, surface)
    for i, side in enumerate(("left", "right")):
        out = r[side]
        assert out["contact_point"]["x"] == pytest.approx(expected[i], abs=3.0)
        assert out["contact_point"]["y"] == pytest.approx(100.0, abs=0.01)
        assert out["angle_deg"] == pytest.approx(angle, abs=3.5)
        p = out["tangent_vector"]
        b = out["surface_tangent"]
        assert p["y"] > 0 and b["x"] == (1.0 if i == 0 else -1.0)
        assert out["angle_deg"] == pytest.approx(np.degrees(np.arccos(b["x"] * p["x"])), abs=0.001)
    assert r["sensitivity"]["left"]["span_deg"] > 0
    assert any(
        run["parameters"].get("surface_variant") == "tilted" and run["status"] == "accepted"
        for run in r["sensitivity"]["runs"]
    )


def test_attached_dark_tufts_do_not_create_spurious_hook():
    gray, surface, _, _ = circle(120, True)
    result = fit_pendant(gray, surface)
    for side in ("left", "right"):
        assert result[side]["angle_deg"] == pytest.approx(120.0, abs=4.0)
        assert result["contour_diagnostics"]["neck"][side]["rejected_neck_rows"] > 4


def test_explicit_clicks_reverse_order_and_roi_does_not_manufacture_border():
    gray, s, clicks, _ = circle(122.0)
    r = fit_pendant(gray, s, contour_points=clicks[::-1])
    assert r["mode"] == "manual"
    assert max(abs(r[k]["angle_deg"] - 122) for k in ["left", "right"]) < 2.5
    with pytest.raises(MeasurementError):
        fit_pendant(gray, s, excluded_regions=[{"x": 0, "y": 0, "width": 330, "height": len(gray)}])
    assert fit_pendant(gray, s, sensitivity=False)["sensitivity"] is None


def test_extensive_or_blank_mask_cannot_fabricate_angles():
    gray, s, _, _ = circle()
    with pytest.raises(MeasurementError):
        fit_pendant(np.full_like(gray, 220), s)
