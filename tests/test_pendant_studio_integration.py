"""Exercise the original coordinate convention across uploads, previews, jobs and exports."""

import hashlib
import io
import json
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.optimize import brentq

from contact_angle.api import app
from contact_angle.studio_models import StudioOptions
from contact_angle.studio_service import measure_studio


def inverted_photo():
    width, height = 620, 375
    cx, cy, radius = 318, 146, 91

    def solid(x):
        return 101 + 0.035 * (x - cx) - 0.00031 * (x - cx) ** 2

    yy, xx = np.mgrid[:height, :width]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
    image = np.where((yy <= solid(xx)) | disk, 18, 241).astype("uint8")
    cv2.circle(image, (cx + 10, cy + 27), 17, 225, -1)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    truth = {}
    for side, interval, sign in [("left", (cx - radius, cx), 1), ("right", (cx, cx + radius), -1)]:
        x = brentq(lambda x: (x - cx) ** 2 + (solid(x) - cy) ** 2 - radius**2, *interval)
        y = solid(x)
        tangent = sign * np.array([y - cy, cx - x])
        surface = sign * np.array([1.0, 0.035 - 2 * 0.00031 * (x - cx)])
        degrees = np.rad2deg(
            np.arccos(tangent @ surface / (np.linalg.norm(tangent) * np.linalg.norm(surface)))
        )
        truth[side] = (x, y, degrees)
    return cv2.imencode(".png", image)[1].tobytes(), truth


def test_inverted_photo_uses_original_frame_through_exports(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTACT_ANGLE_DATA_DIR", str(tmp_path))
    data, truth = inverted_photo()
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/images", files={"image": ("inverted.png", data, "image/png")}
        ).json()
        uri = "/api/images/" + uploaded["id"]
        observed = client.get(uri + "/baseline").json()
        assert observed["orientation"] == "pendant"
        assert observed["method"] == "pendant_surface_v1"
        assert len(observed["surface_curve"]) > 2
        options = {
            "baseline_points": observed["baseline_points"],
            "baseline": observed["baseline"],
            "baseline_method": observed["method"],
            "baseline_source": "automatic",
            "orientation": "auto",
        }
        assert client.put(uri + "/draft", json=options).is_success
        jid = client.post(
            "/api/jobs", json={"image_id": uploaded["id"], "options": options}
        ).json()["id"]
        for _ in range(200):
            job = client.get("/api/jobs/" + jid).json()
            if job["state"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert job["state"] == "done", job.get("error")
        result = job["result"]
        assert result["orientation"] == "pendant" and result["provenance"]["rotation_degrees"] == 0
        for side, (x, y, angle) in truth.items():
            got = result[side]
            assert got["contact_point"]["x"] == pytest.approx(x, abs=4)
            assert got["contact_point"]["y"] == pytest.approx(y, abs=2)
            assert got["angle_deg"] == pytest.approx(angle, abs=5)
            assert got["tangent_vector"]["y"] > 0
        with zipfile.ZipFile(io.BytesIO(client.get(result["downloads"]["zip_url"]).content)) as z:
            assert z.read("original.png") == data
            exported = json.loads(z.read("result.json"))
            assert exported["surface_curve"] == result["surface_curve"]
            assert exported["provenance"]["image_sha256"] == hashlib.sha256(data).hexdigest()


def test_inverted_roi_preserves_original_contact_locations():
    data, truth = inverted_photo()
    options = StudioOptions(
        orientation="pendant",
        roi={"x": 150, "y": 35, "width": 350, "height": 255},
        sensitivity=False,
    ).model_dump(mode="json")
    result, _ = measure_studio(data, options)
    assert result["sensitivity"] is None
    assert result["orientation"] == "pendant"
    for side, (x, y, _) in truth.items():
        assert result[side]["contact_point"]["x"] == pytest.approx(x, abs=4)
        assert result[side]["contact_point"]["y"] == pytest.approx(y, abs=2)


@pytest.mark.parametrize("portion", [slice(None), slice(1, 6), slice(-6, -1)])
def test_manual_marks_recover_direction_without_detectable_silhouette(portion):
    photo = cv2.imencode(".png", np.full((300, 450), 185, np.uint8))[1].tobytes()
    angle = np.deg2rad(125)
    radius, cx, cy = 84, 210, 125
    baseline_y = cy + radius * np.cos(angle)
    # From the left contact down around the far apex to the original right contact.
    phase = np.linspace(np.pi / 2 + angle, np.pi / 2 - angle, 33)
    options = StudioOptions(
        mode="manual",
        baseline_source="manual",
        baseline_points=[{"x": 70, "y": baseline_y}, {"x": 350, "y": baseline_y}],
        contour_points=[
            {"x": cx + radius * np.cos(t), "y": cy + radius * np.sin(t)} for t in phase[portion]
        ],
    ).model_dump(mode="json")
    result, _ = measure_studio(photo, options)
    assert result["orientation"] == "pendant"
    assert result["left"]["angle_deg"] == pytest.approx(125, abs=2)
    assert result["right"]["angle_deg"] == pytest.approx(125, abs=2)
