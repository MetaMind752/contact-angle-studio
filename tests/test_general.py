"""Independent analytic outlines and image-formation checks; no paper labels."""

import cv2
import numpy as np
import pytest
from fibrous_fixtures import circle_curve, render

from contact_angle.general import measure_general_bytes
from contact_angle.young_laplace import fit_young_laplace


@pytest.mark.parametrize("angle", [45, 90, 150])
def test_analytic_circle_precision(angle):
    theta = np.linspace(np.deg2rad(2), np.deg2rad(angle), 250)
    for side, sign in [("left", -1), ("right", 1)]:
        points = 180 * np.c_[sign * np.sin(theta), np.cos(theta) - np.cos(np.deg2rad(angle))]
        result = fit_young_laplace(points, side)
        assert result["diagnostics"]["converged"]
        assert abs(result["angle"] - angle) < 0.01


def test_nonconvergence_is_not_a_valid_angle():
    theta = np.linspace(0.01, np.deg2rad(140), 100)
    points = 100 * np.c_[-np.sin(theta), np.cos(theta) - np.cos(np.deg2rad(140))]
    with pytest.raises(ValueError, match="did not converge"):
        fit_young_laplace(points, "left", max_nfev=1, multistart=False)


@pytest.mark.parametrize("angle", [60, 120, 150])
def test_clean_rendered_image_subdegree_with_known_baseline(angle):
    case = render(
        "circle",
        circle_curve(angle, 220),
        (angle, angle),
        geometry="circle",
        fibres=False,
        noise_sigma=0,
        tilt_deg=0,
    )
    ok, data = cv2.imencode(".png", case.image)
    assert ok
    result = measure_general_bytes(data.tobytes(), case.options["baseline"])
    for side in ("left", "right"):
        assert abs(result[side]["angle_deg"] - angle) < 1
