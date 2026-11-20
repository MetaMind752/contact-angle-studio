"""Local studio request contracts; all coordinates are original image pixels."""

from typing import Literal

from pydantic import Field, model_validator

from .models import ROI, Model, Point

Orientation = Literal["auto", "sessile", "pendant"]


class StudioOptions(Model):
    mode: Literal["auto", "manual"] = "auto"
    orientation: Orientation = "auto"
    algorithm: Literal["general", "fibrous", "local"] = "general"
    baseline: tuple[Point, Point] | None = None
    baseline_points: list[Point] = Field(default_factory=list, max_length=200)
    baseline_source: Literal["automatic", "manual"] = "automatic"
    baseline_method: Literal["dense_substrate_v1", "pendant_surface_v1"] | None = None
    left_points: list[Point] = Field(default_factory=list, max_length=200)
    right_points: list[Point] = Field(default_factory=list, max_length=200)
    contour_points: list[Point] = Field(default_factory=list, max_length=400)
    manual_method: Literal["spline", "quadratic", "cubic", "tangent"] = "spline"
    sensitivity: bool = True
    polarity: Literal["dark", "light"] = "dark"
    roi: ROI | None = None
    excluded_regions: list[ROI] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_baseline(self):
        if len(self.baseline_points) == 1:
            raise ValueError("基线拟合至少需要两个表面点。")
        if self.baseline:
            a, b = self.baseline
            if abs(a.x - b.x) < 5 or abs(a.y - b.y) > abs(a.x - b.x):
                raise ValueError("基线两点横向至少相隔5像素，倾角需在±45°内。")
        return self


class JobRequest(Model):
    image_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    options: StudioOptions


class ContourPreviewRequest(Model):
    contour_points: list[Point] = Field(max_length=400)
    baseline: tuple[Point, Point] | None = None
    orientation: Orientation = "auto"
