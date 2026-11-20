import cv2
import numpy as np
import pytest
from scipy.optimize import brentq

from contact_angle.models import MeasurementError
from contact_angle.pendant_surface import detect_pendant, estimate_pendant_surface


def specimen(fibers=False, liquid=True, reflective=False):
    h, w = 340, 580
    cx, cy, rad = 264, 138, 88

    def face(x):
        return 96 + 0.047 * (x - cx) - 0.00048 * (x - cx) ** 2

    yy, xx = np.ogrid[:h, :w]
    g = np.full((h, w), 230, np.uint8)
    g[np.broadcast_to(yy <= face(xx), (h, w))] = 15
    # A realistic inner lamination is *not* the solid-air interface.
    g[(yy <= face(xx) - 13) & (yy >= face(xx) - 18)] = 110
    if fibers:
        for x in np.arange(24, w - 10, 11):
            xi = int(x)
            y = int(face(xi))
            cv2.line(g, (xi, y - 2), (xi + 4, y + 10), 35, 1)
    if liquid:
        cv2.circle(g, (cx, cy), rad, 15, -1)
        if reflective:
            cv2.circle(g, (cx, cy + 21), 27, 234, -1)
    return cv2.GaussianBlur(g, (3, 3), 0.7), face, (cx, cy, rad)


@pytest.mark.parametrize("fibers", [False, True])
def test_bowed_face_follows_external_air_wedges(fibers):
    g, face, (cx, cy, r) = specimen(fibers=fibers, reflective=True)
    assert detect_pendant(g)
    out = estimate_pendant_surface(g)
    assert out["footprint"] == pytest.approx([cx - r, cx + r], abs=3)
    xs, ys = np.array([(p["x"], p["y"]) for p in out["surface_curve"]]).T
    assert np.all(np.diff(xs) > 0)
    for lim in ((cx - r, cx), (cx, cx + r)):
        q = brentq(lambda x: (x - cx) ** 2 + (face(x) - cy) ** 2 - r * r, *lim)
        assert np.interp(q, xs, ys) == pytest.approx(face(q), abs=1.6)
    # Observations continue to the shoulders rather than stopping at the
    # maximum diameter, even though the highlight has background brightness.
    assert out["observed_spans"][0][1] > cx - r + 4
    assert out["observed_spans"][1][0] < cx + r - 4
    for p in out["baseline_points"]:
        assert abs(p["y"] - face(p["x"])) < 2.5
    assert len(out["surface_variants"]) == 2


def test_no_top_surface_or_drop_does_not_route_to_pendant():
    hanging, _, _ = specimen()
    nodrop, _, _ = specimen(fibers=True, liquid=False)
    for image in (np.flipud(hanging).copy(), nodrop, np.full_like(nodrop, 127)):
        assert not detect_pendant(image)
        with pytest.raises(MeasurementError):
            estimate_pendant_surface(image)


def test_explicit_surface_marks_work_without_an_automatic_silhouette():
    image = np.full((120, 170), 127, np.uint8)
    points = [{"x": 10, "y": 30}, {"x": 80, "y": 48}, {"x": 156, "y": 31}]
    out = estimate_pendant_surface(image, points)
    assert out["baseline_points"] == points
    assert out["footprint"] == [10, 156] and out["footprint_source"] == "manual_span"
    xs, ys = np.array([(p["x"], p["y"]) for p in out["surface_curve"]]).T
    for p in points:
        assert np.interp(p["x"], xs, ys) == pytest.approx(p["y"], abs=0.01)
