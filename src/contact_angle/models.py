"""Public data contract. Coordinates are original image pixels, x right, y down."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(Model):
    x: float
    y: float


class ROI(Model):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class MeasurementOptions(Model):
    baseline: tuple[Point, Point]
    roi: ROI | None = None
    excluded_regions: list[ROI] = Field(default_factory=list, max_length=20)
    method: Literal["local_polynomial", "circle"] = "local_polynomial"
    polarity: Literal["dark", "light"] = "dark"
    threshold: int | None = Field(default=None, ge=0, le=255)
    blur_sigma: float = Field(default=0.8, ge=0, le=3)
    fit_fraction: float = Field(default=0.12, ge=0.04, le=0.3)
    polynomial_degree: Literal[2, 3] = 2
    baseline_margin_px: float = Field(default=1.5, ge=0.5, le=5)
    sensitivity: bool = True
    sensitivity_alert_deg: float = Field(default=3.0, gt=0, le=30)

    @model_validator(mode="after")
    def check_baseline(self):
        a, b = self.baseline
        if abs(a.x - b.x) < 5:
            raise ValueError("Baseline endpoints must differ by at least 5 pixels in x.")
        if abs(a.y - b.y) > abs(a.x - b.x):
            raise ValueError("Baseline tilt must be within 45 degrees; rotate the image first.")
        return self


class SideResult(Model):
    angle_deg: float
    contact_point: Point
    tangent_vector: Point
    fit_rms_px: float
    fit_points: int


class SensitivityRun(Model):
    name: str
    method: str
    parameters: dict
    status: Literal["accepted", "excluded", "failed"]
    left_angle_deg: float | None = None
    right_angle_deg: float | None = None
    left_delta_deg: float | None = None
    right_delta_deg: float | None = None
    error: str | None = None


class AngleRange(Model):
    median_deg: float
    min_deg: float
    max_deg: float
    span_deg: float
    max_abs_delta_deg: float


class SensitivityReport(Model):
    interpretation: str = "Range across admissible methods with a fixed baseline; not a confidence interval or accuracy bound. Shared segmentation or baseline errors may remain."
    left: AngleRange | None
    right: AngleRange | None
    failed_runs: int
    excluded_runs: int
    accepted_runs: int
    sensitive: bool
    alert_deg: float
    runs: list[SensitivityRun]


class MeasurementResult(Model):
    algorithm_version: str
    image_sha256: str
    image_width: int
    image_height: int
    method: str
    parameters: MeasurementOptions
    requested_parameters: MeasurementOptions | None = None
    threshold_used: float
    left: SideResult
    right: SideResult
    mean_angle_deg: float
    asymmetry_deg: float
    contour: list[Point]
    status: Literal["ok", "review"]
    warnings: list[str]
    sensitivity: SensitivityReport | None = None


class MeasurementError(Exception):
    """Expected invalid-image/measurement failure (safe to return to the user)."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)
