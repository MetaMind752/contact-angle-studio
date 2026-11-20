"""Run a frozen source-audited manifest without sending labels to the estimator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from .manifest import TARGET_RATE, THRESHOLDS, Manifest, Sample, load_manifest

SCORE_SCOPES = {
    "sides": "both_sides_against_independently_reported_sides",
    "mean": "image_mean_against_reported_arithmetic_mean",
    "reported_scalar": "image_level_against_published_scalar_unknown_side_convention",
}


def resolve_adapter(specification: str) -> Callable:
    module, separator, function = specification.partition(":")
    if not separator:
        raise ValueError("Adapter must be module:function.")
    adapter = getattr(importlib.import_module(module), function)
    if not callable(adapter):
        raise TypeError("Adapter is not callable.")
    return adapter


def _finite(value, label):
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 180:
        raise ValueError(f"{label} is not a finite angle in [0,180].")
    return numeric


def normalize_prediction(prediction):
    if hasattr(prediction, "result"):
        prediction = prediction.result
    if hasattr(prediction, "model_dump"):
        prediction = prediction.model_dump(mode="json")
    if not isinstance(prediction, dict):
        raise TypeError("Adapter must return a measurement dictionary or MeasurementOutput.")

    def angle(side):
        if isinstance(prediction.get(side), dict):
            return _finite(prediction[side].get("angle_deg"), side)
        return _finite(prediction.get(f"{side}_angle_deg"), side)

    left, right = angle("left"), angle("right")
    mean = _finite(prediction.get("mean_angle_deg"), "mean")
    if left is not None and right is not None:
        derived = (left + right) / 2
        if mean is not None and abs(mean - derived) > 1e-6:
            raise ValueError("Adapter mean conflicts with its left/right arithmetic mean.")
        mean = derived
    return left, right, mean, prediction


def score_sample(sample: Sample, prediction):
    left, right, mean, raw = normalize_prediction(prediction)
    result = {"predicted_left": left, "predicted_right": right, "predicted_mean": mean}
    if sample.reference_kind == "sides":
        if left is None or right is None:
            raise ValueError("Both predicted sides are required for a side-labeled reference.")
        errors = [left - sample.reference_left, right - sample.reference_right]
        result.update(
            error_left=errors[0],
            error_right=errors[1],
            error_mean=mean - (sample.reference_left + sample.reference_right) / 2,
            score_abs_error=max(abs(e) for e in errors),
            side_abs_error_mean=sum(abs(e) for e in errors) / 2,
            score_scope="both_sides_against_independently_reported_sides",
        )
    elif sample.reference_kind == "reported_scalar":
        if mean is None:
            raise ValueError(
                "An image-level predicted mean is required for a reported scalar reference."
            )
        error = mean - sample.reference_scalar
        result.update(
            error_reported_scalar=error,
            score_abs_error=abs(error),
            score_scope="image_level_against_published_scalar_unknown_side_convention",
        )
    else:
        if mean is None:
            raise ValueError("A predicted mean is required for a mean-only reference.")
        error = mean - sample.reference_mean
        result.update(
            error_mean=error,
            score_abs_error=abs(error),
            score_scope="image_mean_against_reported_arithmetic_mean",
        )
    # Equality is a failure. Never average opposing side errors before the sides test.
    result["passed"] = result["score_abs_error"] < sample.threshold
    for side in ("left", "right"):
        interval = (
            raw.get("sensitivity", {}).get(side)
            if isinstance(raw.get("sensitivity"), dict)
            else None
        )
        if interval is None and isinstance(raw.get(side), dict):
            interval = raw[side] if "min_deg" in raw[side] else None
        if interval:
            for key in ("min_deg", "max_deg", "span_deg"):
                value = interval.get(key)
                if value is not None and math.isfinite(float(value)):
                    result[f"sensitivity_{side}_{key}"] = float(value)
    return result


def _identity(sample):
    return {
        "sample_id": sample.sample_id,
        "grade": sample.grade,
        "split": sample.split,
        "group_id": sample.group_id,
        "source_type": sample.source_type,
        "reference_kind": sample.reference_kind,
        "score_scope": SCORE_SCOPES[sample.reference_kind],
        "threshold_deg": sample.threshold,
        "source_url": sample.source_url,
        "doi": sample.doi,
        "image_path": sample.image_path,
        "raw_sha256": sample.raw_sha256,
        "reference_left": sample.reference_left,
        "reference_right": sample.reference_right,
        "reference_mean": sample.reference_mean,
        "reference_scalar": sample.reference_scalar,
        "reference_uncertainty_deg": sample.reference_uncertainty,
        "reference_uncertainty_below_threshold": (
            sample.reference_uncertainty < sample.threshold
            if sample.reference_uncertainty is not None
            else None
        ),
    }


def audit_normalized_duplicates(manifest, base_dir):
    """Catch re-encoded copies too. Bad/missing images remain counted runtime failures."""
    from PIL import Image

    pixels = {}
    for sample in manifest.samples:
        path = base_dir / sample.image_path
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        except (OSError, ValueError):
            continue
        if digest in pixels:
            raise ValueError(f"Duplicate decoded pixels: {pixels[digest]} and {sample.sample_id}")
        pixels[digest] = sample.sample_id


def evaluate(manifest: Manifest, base_dir: Path, adapter: Callable, adapter_name="callback"):
    audit_normalized_duplicates(manifest, base_dir)
    records = []
    for sample in manifest.samples:
        record = _identity(sample)
        try:
            data = (base_dir / sample.image_path).read_bytes()
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != sample.raw_sha256:
                raise ValueError("Input file hash differs from the frozen manifest.")
            baseline = [p.model_dump() for p in sample.baseline] if sample.baseline else None
            # Deliberately pass only image bytes and baseline, never sample ID or labels.
            prediction = adapter(data, baseline=baseline)
            record.update(score_sample(sample, prediction), measured=True, error=None)
        except Exception as exc:  # noqa: BLE001 -- isolate every estimator failure; retain denominator
            record.update(passed=False, measured=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.fingerprint(),
        "adapter": adapter_name,
        "thresholds_deg": THRESHOLDS,
        "strict_less_than": True,
        "target_rate": TARGET_RATE,
        "expected_count": manifest.expected_count,
        "records": records,
        "summary": summarize(records),
        "interpretation": (
            "Agreement with the specified exact-image reference; reference values are not "
            "automatically independently known true angles. Failures remain in denominators. "
            "Real/source-figure and synthetic results are reported separately."
        ),
    }


def _aggregate(records):
    n = len(records)
    passed = sum(r["passed"] for r in records)
    measured = sum(r["measured"] for r in records)
    errors = [r["score_abs_error"] for r in records if r.get("score_abs_error") is not None]
    required = (4 * n + 4) // 5  # Exact ceil(0.8*n), avoiding binary floating point boundaries.
    return {
        "n": n,
        "measured": measured,
        "failed_measurements": n - measured,
        "passed": passed,
        "required_passes": required,
        "pass_rate": passed / n if n else None,
        "target_met": passed >= required if n else None,
        "mean_scored_abs_error_deg": sum(errors) / len(errors) if errors else None,
        "max_scored_abs_error_deg": max(errors) if errors else None,
        "reference_uncertainty_reported": sum(
            r.get("reference_uncertainty_deg") is not None for r in records
        ),
        "reference_uncertainty_unknown": sum(
            r.get("reference_uncertainty_deg") is None for r in records
        ),
    }


def summarize(records):
    summary = {"all": _aggregate(records), "strata": []}
    for dimensions in (
        ("grade",),
        ("split", "grade"),
        ("source_type", "grade"),
        ("reference_kind", "grade"),
        ("split", "source_type", "grade"),
    ):
        groups = defaultdict(list)
        for record in records:
            groups[tuple(record[d] for d in dimensions)].append(record)
        for key, group in sorted(groups.items()):
            summary["strata"].append({**dict(zip(dimensions, key)), **_aggregate(group)})
    summary["grades"] = {
        grade: _aggregate([r for r in records if r["grade"] == grade]) for grade in THRESHOLDS
    }
    summary["reference_kind_counts"] = {
        kind: sum(r["reference_kind"] == kind for r in records) for kind in SCORE_SCOPES
    }
    summary["score_scope_definitions"] = SCORE_SCOPES
    # No absent grade can silently satisfy the three-grade target.
    summary["all_grade_targets_met"] = all(
        value["n"] > 0 and value["target_met"] for value in summary["grades"].values()
    )
    real_test = [r for r in records if r["split"] == "test" and r["source_type"] != "synthetic"]
    summary["real_holdout_grades"] = {
        grade: _aggregate([r for r in real_test if r["grade"] == grade]) for grade in THRESHOLDS
    }
    summary["all_real_holdout_grade_targets_met"] = all(
        value["n"] > 0 and value["target_met"] for value in summary["real_holdout_grades"].values()
    )
    return summary


def compare(before, after):
    if before["manifest_sha256"] != after["manifest_sha256"]:
        raise ValueError("Before/after must use the identical frozen manifest.")
    if before.get("thresholds_deg") != after.get("thresholds_deg"):
        raise ValueError("Before/after must use identical acceptance thresholds.")
    old = {r["sample_id"]: r for r in before["records"]}
    if set(old) != {r["sample_id"] for r in after["records"]}:
        raise ValueError("Before/after sample IDs differ.")
    rows = []
    for current in after["records"]:
        previous = old[current["sample_id"]]
        row = {
            k: current[k] for k in ("sample_id", "grade", "split", "source_type", "reference_kind")
        }
        row["score_scope"] = current.get("score_scope", SCORE_SCOPES[current["reference_kind"]])
        for prefix, record in (("before", previous), ("after", current)):
            for key in (
                "passed",
                "measured",
                "score_abs_error",
                "predicted_left",
                "predicted_right",
                "predicted_mean",
                "error",
                "sensitivity_left_span_deg",
                "sensitivity_right_span_deg",
            ):
                row[f"{prefix}_{key}"] = record.get(key)
        rows.append(row)
    return {
        "manifest_sha256": after["manifest_sha256"],
        "thresholds_deg": after.get("thresholds_deg"),
        "records": rows,
        "before": before["summary"],
        "after": after["summary"],
    }


def rescore_report(report, thresholds=None, reason="User-revised acceptance criteria"):
    """Reapply declared thresholds to cached errors without remeasuring or relabeling."""
    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    if set(thresholds) != set(THRESHOLDS) or any(
        isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0
        for t in thresholds.values()
    ):
        raise ValueError("Thresholds must specify a finite positive value for every grade.")
    result = deepcopy(report)
    source_json = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["acceptance_revision"] = {
        "reason": reason,
        "previous_thresholds_deg": report["thresholds_deg"],
        "source_report_sha256": hashlib.sha256(source_json.encode()).hexdigest(),
        "measurements_recomputed": False,
    }
    result["thresholds_deg"] = thresholds
    result["strict_less_than"] = True
    for record in result["records"]:
        threshold = thresholds[record["grade"]]
        record["threshold_deg"] = threshold
        error = record.get("score_abs_error")
        record["passed"] = bool(
            record["measured"] and error is not None and math.isfinite(error) and error < threshold
        )
        uncertainty = record.get("reference_uncertainty_deg")
        record["reference_uncertainty_below_threshold"] = (
            uncertainty < threshold if uncertainty is not None else None
        )
    result["summary"] = summarize(result["records"])
    return result


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    records = report["records"]
    fields = sorted(set().union(*(r.keys() for r in records))) if records else []
    with (directory / "records.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (directory / "summary.json").write_text(
        json.dumps(
            report.get("summary", {"before": report.get("before"), "after": report.get("after")}),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--adapter", default="benchmark.adapters:current")
    parser.add_argument("--before-adapter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    after = evaluate(manifest, args.manifest.parent, resolve_adapter(args.adapter), args.adapter)
    write_report(after, args.output / "after" if args.before_adapter else args.output)
    if args.before_adapter:
        before = evaluate(
            manifest,
            args.manifest.parent,
            resolve_adapter(args.before_adapter),
            args.before_adapter,
        )
        write_report(before, args.output / "before")
        write_report(compare(before, after), args.output / "comparison")
    print(json.dumps(after["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
