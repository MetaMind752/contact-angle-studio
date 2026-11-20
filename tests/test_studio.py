import csv
import io
import json
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from contact_angle.api import app
from contact_angle.studio_models import StudioOptions
from contact_angle.studio_service import measure_studio
from contact_angle.studio_store import StudioStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTACT_ANGLE_DATA_DIR", str(tmp_path / "workspace"))
    with TestClient(app) as client:
        yield client


def sample_bytes():
    image = np.full((400, 600, 3), 245, dtype=np.uint8)
    cv2.circle(image, (300, 300), 120, (20, 20, 20), -1)
    image[300:] = 230
    return cv2.imencode(".png", image)[1].tobytes()


def manual_options():
    # Independently defined straight edges with known 45deg and135deg angles.
    return {
        "mode": "manual",
        "manual_method": "quadratic",
        "baseline_source": "manual",
        "baseline": [{"x": 10, "y": 300}, {"x": 590, "y": 300}],
        "left_points": [{"x": 150 + h, "y": 300 - h} for h in (0, 5, 15, 30, 45)],
        "right_points": [{"x": 430 + h, "y": 300 - h} for h in (0, 5, 15, 30, 45)],
    }


def wait_job(client, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.03)
    pytest.fail("Measurement worker did not finish")


def upload(client, filename="滴.bmp"):
    response = client.post("/api/images", files={"image": (filename, sample_bytes(), "image/png")})
    assert response.status_code == 201, response.text
    return response.json()


def test_local_workspace_and_original_upload(client):
    info = client.get("/api/info").json()
    assert info["name"] == "Contact Angle Studio"
    image = upload(client)
    assert image["width"] == 600 and image["height"] == 400
    assert upload(client)["id"] == image["id"]
    assert len(client.get("/api/images").json()["images"]) == 1
    preview = client.get(image["preview_url"])
    assert preview.headers["content-type"] == "image/png"
    path = client.get("/api/info").json()["data_dir"]
    from pathlib import Path

    assert (Path(path) / "images" / image["id"] / "original").read_bytes() == sample_bytes()
    assert client.get("/api/images/not-an-id/draft").status_code == 404


def test_draft_and_manual_end_to_end_exports(client):
    image = upload(client, "=SUM(1,2).png")
    options = manual_options()
    assert client.put(f"/api/images/{image['id']}/draft", json=options).status_code == 200
    draft = client.get(f"/api/images/{image['id']}/draft").json()["options"]
    assert draft["left_points"] == options["left_points"]
    response = client.post("/api/jobs", json={"image_id": image["id"], "options": options})
    assert response.status_code == 202
    job = wait_job(client, response.json()["id"])
    assert job["state"] == "done", job
    result = job["result"]
    assert result["left"]["angle_deg"] == pytest.approx(45, abs=0.01)
    assert result["right"]["angle_deg"] == pytest.approx(135, abs=0.01)
    assert result["mean_angle_deg"] == pytest.approx(90, abs=0.01)
    assert result["baseline"] == options["baseline"]
    png = client.get(result["downloads"]["png_url"]).content
    annotated = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert annotated.shape == (494, 600, 3)
    original = cv2.imdecode(np.frombuffer(sample_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.sum(np.any(annotated[:400] != original, axis=2)) > 300
    exported = client.get(result["downloads"]["json_url"]).json()
    assert exported["options"]["left_points"] == options["left_points"]
    table = list(
        csv.DictReader(
            io.StringIO(client.get(result["downloads"]["csv_url"]).content.decode("utf-8-sig"))
        )
    )
    assert float(table[0]["left_angle_deg"]) == pytest.approx(45, abs=0.01)
    assert table[0]["filename"].startswith("'=")
    bundle = client.get(result["downloads"]["zip_url"])
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert set(archive.namelist()) == {
            "result.json",
            "results.csv",
            "annotation.png",
            "points-and-options.json",
            "original.png",
            "说明.txt",
        }
        assert (
            json.loads(archive.read("points-and-options.json"))["options"]["baseline"]
            == options["baseline"]
        )
        assert archive.read("original.png") == sample_bytes()
    assert len(client.get(f"/api/images/{image['id']}/results").json()["results"]) == 1


def test_failed_manual_job_is_preserved_without_result(client):
    image = upload(client)
    options = manual_options()
    options["left_points"] = []
    queued = client.post("/api/jobs", json={"image_id": image["id"], "options": options}).json()
    job = wait_job(client, queued["id"])
    assert job["state"] == "error" and job["result"] is None
    assert "左" in job["error"]["message"]
    assert client.get(f"/api/results/{job['id']}/result.json").status_code == 409
    assert client.get(f"/api/images/{image['id']}/results").json()["results"][0]["state"] == "error"


def test_automatic_analytic_demo_and_roi_mapping(client):
    images = client.get("/api/demo").json()["images"]
    image = next(r for r in images if "120" in r["filename"])
    options = client.get(f"/api/images/{image['id']}/draft").json()["options"]
    assert options["baseline_source"] == "automatic"
    assert len(options["baseline_points"]) > 2
    options["sensitivity"] = False
    options["roi"] = {"x": 130, "y": 100, "width": 440, "height": 300}
    queued = client.post("/api/jobs", json={"image_id": image["id"], "options": options}).json()
    job = wait_job(client, queued["id"])
    assert job["state"] == "done", job
    result = job["result"]
    assert result["left"]["angle_deg"] == pytest.approx(120, abs=2)
    assert result["right"]["angle_deg"] == pytest.approx(120, abs=2)
    assert result["image_width"] == 700 and result["image_height"] == 480
    # A raster transition between rows359/360 is observed at359.5. Check exact
    # image/ROI mapping against that fitted line, plus the independently drawn edge.
    assert result["left"]["contact_point"]["y"] == pytest.approx(
        result["baseline"][0]["y"], abs=1e-6
    )
    assert result["left"]["contact_point"]["y"] == pytest.approx(360, abs=.75)
    assert result["left"]["contact_point"]["x"] < 350 < result["right"]["contact_point"]["x"]
    assert min(p["x"] for c in result["geometry"]["contours"] for p in c["points"]) > 130


def test_persistence_and_interrupted_job_recovery(tmp_path):
    from contact_angle.studio_store import write_json

    store = StudioStore(tmp_path)
    image = store.add_image(sample_bytes(), "image.png")
    store.save_draft(image["id"], manual_options())
    directory = tmp_path / "results" / ("a" * 32)
    directory.mkdir()
    write_json(
        directory / "job.json",
        {"id": "a" * 32, "image_id": image["id"], "state": "running", "result": None},
    )
    reloaded = StudioStore(tmp_path)
    assert reloaded.images()[0]["id"] == image["id"]
    assert reloaded.draft(image["id"])["mode"] == "manual"
    assert reloaded.job("a" * 32)["error"]["code"] == "interrupted"


def test_untrusted_origin_and_bad_input_rejected(client):
    assert client.get("/api/info", headers={"host": "evil.example"}).status_code == 403
    assert (
        client.post(
            "/api/images",
            headers={"origin": "https://evil.example"},
            files={"image": ("test.png", sample_bytes())},
        ).status_code
        == 403
    )
    assert client.post("/api/images", files={"image": ("invalid.png", b"bad")}).status_code == 422
    assert (
        client.post("/api/jobs", json={"image_id": "../secret", "options": {}}).status_code == 422
    )


@pytest.mark.parametrize("algorithm", ["local", "fibrous"])
def test_other_automatic_adapters_return_exportable_geometry(algorithm):
    data = sample_bytes()
    baseline = [{"x": 10, "y": 300}, {"x": 590, "y": 300}]
    truth = 90
    if algorithm == "fibrous":
        from fibrous_fixtures import cases

        case = next(cases(fibres=True, seeds=(202,)))
        data = cv2.imencode(".png", case.image)[1].tobytes()
        baseline = case.options["baseline"]
        truth = case.truth[0]
    options = StudioOptions(algorithm=algorithm, baseline=baseline).model_dump(mode="json")
    result, _ = measure_studio(data, options)
    assert result["left"]["angle_deg"] == pytest.approx(truth, abs=5)
    assert result["geometry"]["contours"]
    assert len(result["geometry"]["fits"]) == 2
    if algorithm == "local":
        for side in ("left", "right"):
            curve = next(c for c in result["geometry"]["fits"] if c["side"] == side)
            assert curve["points"][0]["x"] == pytest.approx(
                result[side]["contact_point"]["x"], abs=1e-7
            )
            assert curve["points"][0]["y"] == pytest.approx(
                result[side]["contact_point"]["y"], abs=1e-7
            )
    json.dumps(result, allow_nan=False)
