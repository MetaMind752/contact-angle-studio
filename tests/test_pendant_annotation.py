import math

import cv2
import numpy as np
import pytest

from contact_angle.studio_annotation import annotate


def point(x, y):
    return {"x": float(x), "y": float(y)}


def direction(angle):
    return point(math.cos(angle), math.sin(angle))


def has_color(canvas, xy, bgr):
    x, y = np.rint(xy).astype(int)
    patch = canvas[y - 1 : y + 2, x - 1 : x + 2].astype(float)
    return np.min(np.linalg.norm(patch - bgr, axis=2)) < 85


@pytest.mark.parametrize("pendant", [True, False])
def test_original_image_orientation_and_liquid_side_arcs(pendant):
    height, width = 320, 400
    image = np.full((height, width, 3), 242, dtype=np.uint8)
    image[:10, :12] = (20, 25, 33)
    image[-10:, -12:] = (177, 32, 157)
    along = [(110, 120), (290, 120)]
    blue = (225, 120, 55)
    result = {
        "baseline": [point(40, 120), point(360, 120)],
        "geometry": {"fits": [], "contours": [], "selected_points": {}},
        "mode": "auto",
        "algorithm": "test",
        "status": "ok",
        "mean_angle_deg": 120,
    }
    if pendant:
        # A curved surface can differ substantially from the compatibility chord.
        result["surface_curve"] = [
            point(80, 110),
            point(110, 120),
            point(200, 156),
            point(290, 120),
            point(320, 110),
        ]
    checks = []
    for (x, y), side, base, color in zip(
        along,
        ("left", "right"),
        (0.39, math.pi - 0.39) if pendant else (0, math.pi),
        ((28, 155, 230), (170, 155, 20)),
    ):
        sweep = math.radians(120) * (1 if pendant else -1) * (1 if side == "left" else -1)
        result[side] = {
            "angle_deg": 120,
            "contact_point": point(x, y),
            "tangent_vector": direction(base + sweep),
        }
        if pendant:
            result[side]["surface_tangent"] = direction(base)
        xy = np.array([x, y]) + 0.055 * height * np.array(
            [math.cos(base + sweep / 2), math.sin(base + sweep / 2)]
        )
        checks.append((xy, color))
    output = cv2.imdecode(np.frombuffer(annotate(image, result), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert output.shape == (height + 94, width, 3)
    np.testing.assert_array_equal(output[:10, :12], image[:10, :12])
    np.testing.assert_array_equal(output[height - 10 : height, -12:], image[-10:, -12:])
    assert has_color(output, (200, 156 if pendant else 120), blue)
    if pendant:
        np.testing.assert_array_equal(output[120, 200], image[120, 200])
    for xy, color in checks:
        assert has_color(output, xy, color)
