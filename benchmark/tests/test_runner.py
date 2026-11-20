from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from benchmark.manifest import Manifest, Sample
from benchmark.runner import compare, evaluate, rescore_report, score_sample, summarize


def sample(**changes):
    base = {
        "sample_id": "source_001",
        "image_path": "001.png",
        "source_url": "https://example.org/paper",
        "doi": "10.0000/example",
        "source_type": "real",
        "raw_sha256": "a" * 64,
        "grade": "smooth",
        "grade_reason": "Clear sharp edge; no visible fibre near contact.",
        "group_id": "independent_drop_001",
        "split": "test",
        "reference_kind": "sides",
        "reference_left": 100,
        "reference_right": 100,
        "reference_method": "source manual",
        "reference_locator": "Supplement row 1",
    }
    base.update(changes)
    return Sample(**base)


def manifest(samples):
    return Manifest(
        benchmark_id="test",
        expected_count=len(samples),
        grade_policy="Pregraded without model outputs.",
        grouping_policy="Same source drop stays within one split.",
        samples=samples,
    )


@pytest.mark.parametrize("grade,threshold", [("smooth", 2), ("mild", 3), ("strong", 5)])
def test_strict_threshold_and_no_side_cancellation(grade, threshold):
    reference = sample(grade=grade)
    assert not score_sample(
        reference, {"left_angle_deg": 100 + threshold, "right_angle_deg": 100 - threshold}
    )["passed"]
    assert score_sample(
        reference,
        {"left_angle_deg": 100 + threshold - 1e-8, "right_angle_deg": 100 - threshold + 1e-8},
    )["passed"]


def test_missing_reference_is_invalid():
    with pytest.raises(ValidationError, match="both independently reported"):
        sample(reference_right=None)
    with pytest.raises(ValidationError, match="Mean reference"):
        sample(reference_kind="mean", reference_left=None, reference_right=None)


def test_mean_only_reference_never_fabricates_sides():
    ref = sample(
        reference_kind="mean", reference_left=None, reference_right=None, reference_mean=100
    )
    score = score_sample(ref, {"mean_angle_deg": 100.2})
    assert score["passed"]
    assert score["predicted_left"] is None and score["predicted_right"] is None
    assert "error_left" not in score
    with pytest.raises(ValidationError, match="must not fabricate"):
        sample(reference_kind="mean", reference_mean=100)


def test_duplicate_hash_rejected():
    with pytest.raises(ValidationError, match="Duplicate raw input"):
        manifest([sample(), sample(sample_id="source_002", group_id="drop_002")])


def test_same_drop_cannot_leak_between_splits():
    with pytest.raises(ValidationError, match="leak"):
        manifest([sample(), sample(sample_id="source_002", raw_sha256="b" * 64, split="dev")])


def image_sample(tmp_path: Path, index: int):
    p = tmp_path / f"{index}.png"
    Image.new("RGB", (24, 24), (index * 20, 20, 50)).save(p)
    return sample(
        sample_id=f"source_{index}",
        image_path=p.name,
        raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        group_id=f"drop{index}",
    )


def test_failures_remain_in_denominator_and_labels_not_passed(tmp_path):
    refs = [image_sample(tmp_path, i) for i in (1, 2)]
    calls = []

    def estimator(data, baseline):
        calls.append((data, baseline))
        if len(calls) == 2:
            raise RuntimeError("Unresolved edge")
        return {"left_angle_deg": 100, "right_angle_deg": 100}

    report = evaluate(manifest(refs), tmp_path, estimator)
    summary = report["summary"]["all"]
    assert summary["n"] == 2 and summary["passed"] == 1
    assert summary["pass_rate"] == 0.5 and summary["failed_measurements"] == 1
    assert not summary["target_met"]
    assert len(calls) == 2 and calls[0][1] is None


def test_reencoded_duplicate_pixels_rejected(tmp_path):
    a = image_sample(tmp_path, 1)
    p = tmp_path / "copy.bmp"
    Image.open(tmp_path / a.image_path).save(p)
    b = sample(
        sample_id="copy",
        image_path=p.name,
        raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        group_id="copy",
    )
    with pytest.raises(ValueError, match="Duplicate decoded pixels"):
        evaluate(manifest([a, b]), tmp_path, lambda data, baseline: {})


def test_missing_image_and_tampered_hash_fail_without_vanishing(tmp_path):
    a = image_sample(tmp_path, 1)
    b = sample(
        sample_id="missing", image_path="absent.png", raw_sha256="b" * 64, group_id="missing"
    )
    (tmp_path / a.image_path).write_bytes(b"tampered")
    result = evaluate(manifest([a, b]), tmp_path, lambda data, baseline: {})
    assert result["summary"]["all"]["n"] == 2
    assert result["summary"]["all"]["failed_measurements"] == 2
    assert all(not r["passed"] for r in result["records"])


def test_absent_grade_cannot_satisfy_target():
    report = summarize(
        [
            {
                "passed": True,
                "measured": True,
                "grade": "smooth",
                "split": "test",
                "source_type": "real",
                "reference_kind": "sides",
                "score_abs_error": 0.2,
            }
        ]
    )
    assert not report["all_grade_targets_met"]
    assert not report["all_real_holdout_grade_targets_met"]


def test_before_after_must_be_identical_manifest():
    with pytest.raises(ValueError, match="identical frozen"):
        compare({"manifest_sha256": "a"}, {"manifest_sha256": "b"})


def test_reported_scalar_is_not_assumed_to_be_a_known_mean_or_sides():
    ref = sample(
        reference_kind="reported_scalar",
        reference_left=None,
        reference_right=None,
        reference_scalar=100,
    )
    score = score_sample(ref, {"left_angle_deg": 99.5, "right_angle_deg": 100.7})
    assert score["passed"]
    assert score["error_reported_scalar"] == pytest.approx(0.1)
    assert "error_mean" not in score and "error_left" not in score
    assert score["score_scope"] == "image_level_against_published_scalar_unknown_side_convention"
    assert ref.reference_mean is None
    with pytest.raises(ValidationError, match="reference_scalar"):
        sample(reference_kind="reported_scalar", reference_left=None, reference_right=None)
    with pytest.raises(ValidationError, match="must not fabricate"):
        sample(reference_kind="reported_scalar", reference_scalar=100)


def test_reported_scalar_missing_prediction_still_counts_as_failure(tmp_path):
    source = image_sample(tmp_path, 1)
    ref = source.model_copy(
        update={
            "reference_kind": "reported_scalar",
            "reference_left": None,
            "reference_right": None,
            "reference_scalar": 100,
        }
    )
    report = evaluate(manifest([ref]), tmp_path, lambda data, baseline: {})
    assert report["summary"]["all"]["n"] == 1
    assert report["summary"]["all"]["failed_measurements"] == 1
    assert report["summary"]["all"]["passed"] == 0
    assert report["summary"]["strata"]


def test_revised_thresholds_preserve_measurements_and_failures(tmp_path):
    refs = [image_sample(tmp_path, i) for i in (1, 2)]
    report = evaluate(
        manifest(refs), tmp_path,
        lambda data, baseline: {"left_angle_deg": 101.5, "right_angle_deg": 100},
    )
    report["records"][1].update(measured=False, passed=False, error="Unresolved edge")
    legacy = rescore_report(report, {"smooth": 1, "mild": 2.5, "strong": 5})
    revised = rescore_report(legacy)
    assert legacy["summary"]["all"]["passed"] == 0
    assert revised["summary"]["all"]["passed"] == 1
    assert revised["summary"]["all"]["n"] == 2
    assert revised["records"][0]["predicted_left"] == legacy["records"][0]["predicted_left"]
    assert not revised["records"][1]["passed"]  # Cached angle cannot revive a failed run.
    assert not revised["acceptance_revision"]["measurements_recomputed"]
    assert legacy["thresholds_deg"]["smooth"] == 1  # Input report remains unchanged.
    with pytest.raises(ValueError, match="identical acceptance"):
        compare(legacy, revised)


@pytest.mark.parametrize("thresholds", [{"smooth": 2}, {"smooth": float("nan"), "mild": 3, "strong": 5}])
def test_invalid_revised_thresholds_rejected(thresholds):
    with pytest.raises(ValueError, match="finite positive"):
        rescore_report({}, thresholds)
