"""Benchmark input contract; grading and grouping must precede model scoring."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# User-revised acceptance criteria, 2026-11-16. Prior reports retain their own
# explicit thresholds; changing these defaults never changes saved measurements.
THRESHOLDS = {"smooth": 2.0, "mild": 3.0, "strong": 5.0}
TARGET_RATE = 0.8


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float
    y: float


class Crop(BaseModel):
    """Original-figure rectangle, for provenance; input image is already cropped."""

    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https?://")
    doi: str | None = None
    source_type: Literal["real", "synthetic", "source-figure"]
    raw_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_asset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    crop: Crop | None = None
    baseline: tuple[Point, Point] | None = None
    grade: Literal["smooth", "mild", "strong"]
    grade_reason: str = Field(min_length=5)
    group_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    reference_kind: Literal["sides", "mean", "reported_scalar"]
    reference_left: float | None = Field(default=None, ge=0, le=180)
    reference_right: float | None = Field(default=None, ge=0, le=180)
    reference_mean: float | None = Field(default=None, ge=0, le=180)
    reference_scalar: float | None = Field(default=None, ge=0, le=180)
    reference_uncertainty: float | None = Field(default=None, ge=0)
    reference_method: str = Field(min_length=3)
    reference_locator: str = Field(min_length=1)
    licence: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def reference_consistency(self):
        if self.reference_kind == "sides":
            if self.reference_left is None or self.reference_right is None:
                raise ValueError(
                    "Sides reference requires both independently reported side angles."
                )
            if self.reference_mean is not None or self.reference_scalar is not None:
                raise ValueError("Leave reference_mean/reference_scalar null for sides references.")
        elif self.reference_kind == "mean":
            if self.reference_mean is None:
                raise ValueError("Mean reference requires the source's exact-image mean angle.")
            if any(
                v is not None
                for v in (self.reference_left, self.reference_right, self.reference_scalar)
            ):
                raise ValueError(
                    "Mean-only references must not fabricate left/right/scalar labels."
                )
        else:
            if self.reference_scalar is None:
                raise ValueError(
                    "Reported scalar reference requires an exact-photo reference_scalar."
                )
            if any(
                v is not None
                for v in (self.reference_left, self.reference_right, self.reference_mean)
            ):
                raise ValueError(
                    "Reported scalar references must not fabricate sides or a known mean."
                )
        if self.baseline:
            a, b = self.baseline
            if (a.x - b.x) ** 2 + (a.y - b.y) ** 2 < 25:
                raise ValueError("Baseline endpoints must be at least 5 pixels apart.")
        return self

    @property
    def threshold(self):
        return THRESHOLDS[self.grade]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    benchmark_id: str = Field(min_length=1)
    version: str = "1"
    expected_count: int = Field(default=100, gt=0)
    grade_policy: str = Field(min_length=10)
    grouping_policy: str = Field(min_length=10)
    frozen_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    samples: list[Sample]

    @model_validator(mode="after")
    def uniqueness_and_separation(self):
        if len(self.samples) != self.expected_count:
            raise ValueError(f"Expected {self.expected_count} samples, got {len(self.samples)}.")
        ids, hashes = set(), set()
        groups = defaultdict(set)
        for sample in self.samples:
            if sample.sample_id in ids:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            if sample.raw_sha256 in hashes:
                raise ValueError(f"Duplicate raw input image hash: {sample.sample_id}")
            ids.add(sample.sample_id)
            hashes.add(sample.raw_sha256)
            groups[sample.group_id].add(sample.split)
        leaking = [group for group, splits in groups.items() if len(splits) > 1]
        if leaking:
            raise ValueError(f"Related-image groups leak across dev/test: {leaking}")
        return self

    def fingerprint(self):
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()


def load_manifest(path: str | Path) -> Manifest:
    return Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
