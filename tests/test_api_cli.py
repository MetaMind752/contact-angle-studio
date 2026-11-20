import base64
import csv
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from contact_angle import __version__
from contact_angle.api import app
from contact_angle.cli import main
from contact_angle.core import measure_bytes
from contact_angle.models import MeasurementOptions


@pytest.fixture
def sample():
    # A semicircular sessile droplet has a 90-degree analytic contact angle.
    image = np.full((360, 500), 245, dtype=np.uint8)
    cv2.circle(image, (250, 280), 130, 20, -1)
    image[280:, :] = 245
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    options = {"baseline": [{"x": 30, "y": 280}, {"x": 470, "y": 280}], "method": "circle"}
    return encoded.tobytes(), options


def test_health_and_docs():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "version": __version__}
        assert client.get("/").status_code == 200
        assert "/measure" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("annotation", [False, True])
def test_api_measure(sample, annotation):
    image, options = sample
    with TestClient(app) as client:
        response = client.post(
            "/measure",
            files={"image": ("drop.png", image, "image/png")},
            data={"options": json.dumps(options), "include_annotation": str(annotation).lower()},
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["left"]["angle_deg"] == pytest.approx(90, abs=2)
    assert result["right"]["angle_deg"] == pytest.approx(90, abs=2)
    assert result["parameters"]["baseline"] == options["baseline"]
    assert result["sensitivity"]["accepted_runs"] >= 1
    if annotation:
        png = base64.b64decode(result["annotation_png_base64"])
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (360, 500)
    else:
        assert "annotation_png_base64" not in result


@pytest.mark.parametrize("options", ["{", "{}", '{"baseline":[{"x":0,"y":10},{"x":0,"y":10}]}'])
def test_api_invalid_options(sample, options):
    image, _ = sample
    with TestClient(app) as client:
        response = client.post(
            "/measure", files={"image": ("drop.png", image)}, data={"options": options}
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_options"


def test_api_invalid_image(sample):
    _, options = sample
    with TestClient(app) as client:
        response = client.post(
            "/measure",
            files={"image": ("broken.png", b"not an image")},
            data={"options": json.dumps(options)},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"]


def test_api_upload_limit(sample, monkeypatch):
    image, options = sample
    monkeypatch.setattr("contact_angle.api.MAX_UPLOAD_BYTES", 16)
    with TestClient(app) as client:
        response = client.post(
            "/measure", files={"image": ("large.png", image)}, data={"options": json.dumps(options)}
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "image_too_large"


def test_cli_exports_and_no_overwrite(sample, tmp_path, capsys):
    image, options = sample
    source = tmp_path / "drop.png"
    config = tmp_path / "options.json"
    output = tmp_path / "results"
    source.write_bytes(image)
    config.write_text(json.dumps(options))
    args = ["measure", str(source), "--options", str(config), "--output", str(output)]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    result = json.loads((output / "result.json").read_text())
    assert summary["left_angle_deg"] == result["left"]["angle_deg"]
    assert (output / "annotated.png").read_bytes().startswith(b"\x89PNG")
    rows = list(
        csv.DictReader(io.StringIO((output / "results.csv").read_text(encoding="utf-8-sig")))
    )
    assert len(rows) == 1
    assert float(rows[0]["left_angle_deg"]) == result["left"]["angle_deg"]
    sensitivity = result["sensitivity"]
    assert summary["sensitivity"]["left"]["span_deg"] == sensitivity["left"]["span_deg"]
    assert float(rows[0]["left_sensitivity_span_deg"]) == sensitivity["left"]["span_deg"]
    runs = list(
        csv.DictReader(io.StringIO((output / "sensitivity.csv").read_text(encoding="utf-8-sig")))
    )
    assert len(runs) == len(sensitivity["runs"])
    for row, run in zip(runs, sensitivity["runs"]):
        assert row["name"] == run["name"]
        assert row["status"] == run["status"]
        assert json.loads(row["parameters_json"]) == run["parameters"]
    original = (output / "result.json").read_bytes()
    assert main(args) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "output_exists"
    assert (output / "result.json").read_bytes() == original
    assert main([*args, "--overwrite"]) == 0


def test_cli_sensitivity_disabled(sample, tmp_path, capsys):
    image, options = sample
    options["sensitivity"] = False
    source = tmp_path / "drop.png"
    config = tmp_path / "options.json"
    output = tmp_path / "results"
    source.write_bytes(image)
    config.write_text(json.dumps(options))
    assert main(["measure", str(source), "--options", str(config), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["sensitivity"] is None
    assert json.loads((output / "result.json").read_text())["sensitivity"] is None
    rows = list(
        csv.DictReader(io.StringIO((output / "sensitivity.csv").read_text(encoding="utf-8-sig")))
    )
    assert rows == []


def test_cli_no_accepted_sensitivity_runs_has_empty_ranges(sample, tmp_path, capsys, monkeypatch):
    image, options = sample
    outcome = measure_bytes(image, MeasurementOptions.model_validate(options))
    report = outcome.result.sensitivity
    assert report is not None
    # Exercise serialization when every alternative fit was rejected, without coupling
    # this interface test to a particular pathological image or fit rejection heuristic.
    report.left = None
    report.right = None
    report.accepted_runs = 0
    report.excluded_runs = len(report.runs)
    report.failed_runs = 0
    for run in report.runs:
        run.status = "excluded"
        run.error = "Rejected for this test"
    monkeypatch.setattr("contact_angle.cli.measure_bytes", lambda *_: outcome)
    monkeypatch.setattr("contact_angle.api.measure_bytes", lambda *_: outcome)
    with TestClient(app) as client:
        response = client.post(
            "/measure",
            files={"image": ("drop.png", image)},
            data={"options": json.dumps(options)},
        )
    assert response.status_code == 200
    assert response.json()["sensitivity"]["left"] is None
    assert response.json()["sensitivity"]["right"] is None
    source = tmp_path / "drop.png"
    config = tmp_path / "options.json"
    output = tmp_path / "results"
    source.write_bytes(image)
    config.write_text(json.dumps(options))
    assert main(["measure", str(source), "--options", str(config), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["sensitivity"]["left"] is None
    rows = list(
        csv.DictReader(io.StringIO((output / "results.csv").read_text(encoding="utf-8-sig")))
    )
    assert rows[0]["left_sensitivity_span_deg"] == ""
    assert rows[0]["right_sensitivity_min_deg"] == ""
    runs = list(
        csv.DictReader(io.StringIO((output / "sensitivity.csv").read_text(encoding="utf-8-sig")))
    )
    assert all(run["status"] == "excluded" and run["error"] for run in runs)


def test_cli_missing_file_is_readable_error(tmp_path, capsys):
    assert (
        main(
            [
                "measure",
                str(tmp_path / "missing.png"),
                "--options",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "results"),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "file_error"
    assert not (tmp_path / "results").exists()


def test_cli_invalid_options_has_no_traceback(tmp_path, capsys):
    options = tmp_path / "options.json"
    options.write_text("{}")
    assert (
        main(
            ["measure", "unused.png", "--options", str(options), "--output", str(tmp_path / "out")]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_options"
