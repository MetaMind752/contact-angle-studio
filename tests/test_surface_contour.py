"""Independent surface pixels and analytic open curves for the studio workflow."""

import io
import json
import math
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from contact_angle.api import app
from contact_angle.manual import measure_manual
from contact_angle.models import MeasurementError
from contact_angle.studio_models import StudioOptions
from contact_angle.studio_service import estimate_surface_points, measure_studio
from contact_angle.surface_points import fit_surface_points


def circle_points(angle=120, count=31, tilt=0):
    radius = 130
    theta = np.deg2rad(angle)
    phase = np.linspace(np.pi / 2 + theta, np.pi / 2 - theta, count)
    uv = np.c_[320 + radius * np.cos(phase), -radius * np.cos(theta) + radius * np.sin(phase)]
    a = np.deg2rad(tilt)
    basis = np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]])
    xy = uv @ basis.T + [20, 450]
    baseline = np.array([[20, 0], [600, 0]]) @ basis.T + [20, 450]
    point = lambda p: {"x": float(p[0]), "y": float(p[1])}
    return [point(p) for p in xy], [point(p) for p in baseline]


def options(angle=120, count=31, tilt=0):
    contour, baseline = circle_points(angle, count, tilt)
    return {
        "mode": "manual",
        "baseline": baseline,
        "contour_points": contour,
        "manual_method": "spline",
        "sensitivity": True,
    }


@pytest.mark.parametrize("angle", [30, 60, 90, 120, 150])
@pytest.mark.parametrize("tilt", [-8, 0, 8])
def test_unified_contour_tangents_match_analytic_circle(angle, tilt):
    image = np.zeros((650, 800, 3), np.uint8)
    opt = options(angle, tilt=tilt)
    result = measure_manual(image, opt)
    for side in ("left", "right"):
        assert result[side]["angle_deg"] == pytest.approx(angle, abs=2)
        curve = next(c for c in result["geometry"]["fits"] if c["side"] == side)
        assert curve["points"][0] == pytest.approx(result[side]["contact_point"])
        assert result[side]["fit_rms_px"] < 0.2
    assert result["sensitivity"]["accepted_runs"] >= 3
    assert result["geometry"]["selected_points"]["contour"] == opt["contour_points"]
    json.dumps(result, allow_nan=False)


def test_reverse_contour_and_baseline_order_preserves_fit():
    image = np.zeros((650, 800, 3), np.uint8)
    opt = options(140)
    first = measure_manual(image, opt)
    opt["contour_points"].reverse()
    opt["baseline"].reverse()
    second = measure_manual(image, opt)
    assert first == second


def test_asymmetric_bezier_does_not_impose_equal_angles():
    image = np.zeros((650, 800, 3), np.uint8)
    controls = np.array([[210, 0], [100, 180], [620, 150], [490, 0]], dtype=float)
    t = np.linspace(0, 1, 41)[:, None]
    uv = (
        (1 - t) ** 3 * controls[0]
        + 3 * (1 - t) ** 2 * t * controls[1]
        + 3 * (1 - t) * t**2 * controls[2]
        + t**3 * controls[3]
    )
    opt = options()
    opt["baseline"] = [{"x": 50, "y": 450}, {"x": 750, "y": 450}]
    opt["contour_points"] = [{"x": float(p[0]), "y": float(450 - p[1])} for p in uv]
    result = measure_manual(image, opt)
    assert result["left"]["angle_deg"] == pytest.approx(math.degrees(math.atan2(180, -110)), abs=2)
    assert result["right"]["angle_deg"] == pytest.approx(math.degrees(math.atan2(150, -130)), abs=2)
    assert result["asymmetry_deg"] > 8


@pytest.mark.parametrize("bad", ["few", "duplicate", "outside", "order"])
def test_bad_contour_observations_are_rejected(bad):
    image = np.zeros((650, 800, 3), np.uint8)
    opt = options()
    if bad == "few":
        opt["contour_points"] = opt["contour_points"][:4]
    elif bad == "duplicate":
        opt["contour_points"][5] = opt["contour_points"][4]
    elif bad == "outside":
        opt["contour_points"][5]["x"] = 900
    else:
        opt["contour_points"][3], opt["contour_points"][20] = (
            opt["contour_points"][20],
            opt["contour_points"][3],
        )
    with pytest.raises(MeasurementError):
        measure_manual(image, opt)


@pytest.mark.parametrize("angle", [45, 90, 150])
@pytest.mark.parametrize("portion", [slice(1, 6), slice(4, -4), slice(-6, -1)])
def test_partial_contour_can_estimate_contacts_from_either_side(angle, portion):
    image = np.zeros((650, 800, 3), np.uint8)
    opt = options(angle, tilt=8)
    opt["contour_points"] = opt["contour_points"][portion]
    result = measure_manual(image, opt)
    assert result["method_label"] == "手动圆弧估算"
    assert result["status"] == "review"
    for side in ("left", "right"):
        assert result[side]["angle_deg"] == pytest.approx(angle, abs=0.1)
        assert result[side]["extrapolation_px"] > 0
        assert result["sensitivity"][side]["max_abs_delta_deg"] < 0.1


def test_manual_contour_preview_does_not_measure_or_save(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTACT_ANGLE_DATA_DIR", str(tmp_path))
    image = np.zeros((650, 800, 3), np.uint8)
    raw = cv2.imencode(".png", image)[1].tobytes()
    opt = options(120)
    with TestClient(app) as client:
        upload = client.post("/api/images", files={"image": ("drop.png", raw, "image/png")})
        uri = f"/api/images/{upload.json()['id']}"
        request = {"contour_points": opt["contour_points"][1:5], "baseline": opt["baseline"]}
        assert client.post(uri + "/contour-preview", json=request).status_code == 422
        request["contour_points"] = opt["contour_points"][1:6]
        response = client.post(uri + "/contour-preview", json=request)
        assert response.is_success
        outline = response.json()
        assert list(outline) == ["points"]
        assert len(outline["points"]) > 5
        assert outline["points"][0] == pytest.approx(opt["contour_points"][0])
        assert outline["points"][-1] == pytest.approx(opt["contour_points"][-1])
        assert client.get(uri + "/results").json() == {"results": []}
        draft = client.get(uri + "/draft").json()["options"]
        assert not draft or not draft["contour_points"]


def test_surface_fit_uses_all_observations_and_rejects_isolated_fiber():
    values = [
        {"x": float(x), "y": float(300 + 0.03 * (x - 300))} for x in [100, 120, 145, 450, 475, 500]
    ]
    values.insert(2, {"x": 130, "y": 282})
    fitted = fit_surface_points(values, 640, 600)
    assert fitted["baseline"][0]["x"] == 100
    assert fitted["baseline"][1]["x"] == 500
    assert fitted["baseline"][0]["y"] == pytest.approx(294, abs=0.05)
    assert fitted["baseline"][1]["y"] == pytest.approx(306, abs=0.05)
    assert fitted["inlier_count"] == 6
    assert fitted["baseline_points"] == values


def synthetic_surface(fibers=False):
    image = np.full((500, 800, 3), 245, np.uint8)
    cv2.circle(image, (400, 310), 135, (12, 12, 12), -1)
    image[310:] = 20
    # Nozzle above the drop is a separate dark component, not the droplet.
    image[:55, 390:410] = 15
    if fibers:
        for x in (178, 204, 590, 610):
            cv2.line(image, (x, 313), (x - 3, 297), (12, 12, 12), 2)
    return image


@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("fibers", [False, True])
def test_auto_baseline_samples_substrate_near_drop_not_image_extremes(light, fibers):
    image = synthetic_surface(fibers)
    if light:
        image = cv2.bitwise_not(image)
    result = estimate_surface_points(image, "light" if light else "dark")
    observed = result["baseline_points"]
    assert len(observed) >= 20
    assert min(p["x"] for p in observed) > 100
    assert max(p["x"] for p in observed) < 700
    assert all(p["x"] < 265 or p["x"] > 535 for p in observed)
    assert len({round(p["x"], 1) for p in observed}) == len(observed)
    assert result["baseline"][0]["y"] == pytest.approx(309.5, abs=0.75)
    assert result["baseline"][1]["y"] == pytest.approx(309.5, abs=0.75)


def test_baseline_samples_are_authoritative_over_stale_derived_line():
    image = np.full((650, 800, 3), 245, np.uint8)
    opt = options()
    opt["baseline_points"] = opt["baseline"]
    opt["baseline"] = [{"x": 0, "y": 600}, {"x": 799, "y": 600}]
    normalized = StudioOptions(**opt).model_dump(mode="json")
    result, _ = measure_studio(cv2.imencode(".png", image)[1].tobytes(), normalized)
    assert result["baseline"][0]["y"] == pytest.approx(450)
    assert result["left"]["angle_deg"] == pytest.approx(120, abs=2)
    assert result["baseline_points"] == opt["baseline_points"]
    assert not any("请核对" in w for w in result["warnings"])


def test_new_api_draft_fit_and_coordinate_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTACT_ANGLE_DATA_DIR", str(tmp_path))
    raw = cv2.imencode(".png", synthetic_surface())[1].tobytes()
    with TestClient(app) as client:
        image = client.post("/api/images", files={"image": ("drop.png", raw, "image/png")}).json()
        detected = client.get(f"/api/images/{image['id']}/baseline").json()
        assert len(detected["baseline_points"]) > 2
        assert detected["warnings"] == []
        opt = options()
        # This test image is only 500x800; the independent analytic clicks still fit.
        opt["baseline_points"] = opt["baseline"]
        assert client.put(f"/api/images/{image['id']}/draft", json=opt).status_code == 200
        saved = client.get(f"/api/images/{image['id']}/draft").json()["options"]
        assert saved["contour_points"] == opt["contour_points"]
        response = client.post("/api/jobs", json={"image_id": image["id"], "options": saved})
        assert response.status_code == 202
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get("/api/jobs/" + response.json()["id"]).json()
            if job["state"] in ("done", "error"):
                break
            time.sleep(0.02)
        assert job["state"] == "done", job
        result = job["result"]
        assert result["method_label"] == "手动轮廓曲线"
        assert result["left"]["angle_deg"] == pytest.approx(120, abs=2)
        with zipfile.ZipFile(
            io.BytesIO(client.get(result["downloads"]["zip_url"]).content)
        ) as bundle:
            coordinates = json.loads(bundle.read("points-and-options.json"))
            assert coordinates["options"]["contour_points"] == opt["contour_points"]
            assert coordinates["options"]["baseline_points"] == opt["baseline_points"]
        invalid = client.put(
            f"/api/images/{image['id']}/draft", json={"baseline_points": [{"x": 10, "y": 20}]}
        )
        assert invalid.status_code == 422


@pytest.mark.parametrize("algorithm", ["general", "local", "fibrous"])
@pytest.mark.parametrize("angle", [90, 120])
def test_pale_demo_substrate_retains_equal_strength_lateral_edges(algorithm, angle):
    image = np.full((480, 700, 3), 246, dtype=np.uint8)
    cy = 360 + 145 * np.cos(np.deg2rad(angle))
    cv2.circle(image, (350, round(cy)), 145, (22, 22, 22), -1, cv2.LINE_AA)
    image[360:] = 225
    fitted = estimate_surface_points(image, algorithm=algorithm)
    assert len(fitted["baseline_points"]) > 2
    assert fitted["baseline"][0]["y"] == pytest.approx(359.5, abs=0.1)
    assert fitted["baseline"][1]["y"] == pytest.approx(359.5, abs=0.1)
