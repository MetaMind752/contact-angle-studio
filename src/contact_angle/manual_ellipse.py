"""Bounded convex outlines from manually selected liquid-edge points."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist

from .models import MeasurementError


@dataclass(frozen=True)
class MarkedEllipse:
    center: np.ndarray
    axes: np.ndarray
    angle: float
    rms: float
    model: Literal["ellipse", "circle"]

    def _rotation(self):
        sine, cosine = np.sin(self.angle), np.cos(self.angle)
        return np.array([[cosine, -sine], [sine, cosine]])

    def at(self, phase):
        radii = np.stack((np.cos(phase), np.sin(phase)), axis=-1) * self.axes
        return self.center + radii @ self._rotation().T

    def derivative(self, phase):
        tangent = np.stack((-np.sin(phase), np.cos(phase)), axis=-1) * self.axes
        return tangent @ self._rotation().T

    def phase(self, points):
        coordinates = ((points - self.center) @ self._rotation()) / self.axes
        return np.arctan2(coordinates[..., 1], coordinates[..., 0])


def _coverage(shape, xy):
    phases = np.sort(shape.phase(xy))
    ordered = np.r_[phases, phases[0] + 2 * np.pi]
    return 2 * np.pi - np.max(np.diff(ordered))


def _distance(parameters, xy):
    center = parameters[:2]
    log_radius, log_ratio, angle = parameters[2:]
    axes = np.exp(log_radius + np.array([0.5, -0.5]) * log_ratio)
    sine, cosine = np.sin(angle), np.cos(angle)
    rotated = (xy - center) @ [[cosine, -sine], [sine, cosine]]
    radial = np.linalg.norm(rotated, axis=1)
    radius = np.linalg.norm(rotated / axes, axis=1)
    edge = np.divide(radial, radius, out=np.full(len(xy), axes.min()), where=radius > 1e-9)
    return radial - edge


def _fit_supported_ellipse(xy, origin, extent, max_axis):
    """Only release the shape parameters when the clicks cover a broad arc."""
    try:
        center, diameters, rotation = cv2.fitEllipseDirect(xy.astype("float32"))
    except cv2.error:
        return None
    major, minor = np.sort(diameters)[::-1] / 2
    angle = np.deg2rad(rotation + (90 if diameters[0] < diameters[1] else 0))
    aspect_limit = 3.0
    if not np.isfinite([*center, angle]).all() or not (
        0.04 < minor <= major < max_axis and major / minor < aspect_limit
    ):
        return None
    seed = MarkedEllipse(np.asarray(center), np.array([major, minor]), angle, 0, "ellipse")
    if _coverage(seed, xy) < np.deg2rad(105):
        return None
    # Optimize in units of the marked span and regularize away unsupported ellipticity.
    # Semiaxes share a geometric mean, and their ratio remains bounded throughout.
    log_aspect_limit = np.log(aspect_limit)
    lower = [-max_axis, -max_axis, np.log(0.07), 0.0, -3 * np.pi]
    upper = [max_axis, max_axis, np.log(max_axis), log_aspect_limit, 3 * np.pi]
    start = [*center, np.log(np.sqrt(major * minor)), np.log(major / minor), angle]
    start = np.clip(start, np.asarray(lower) + 1e-8, np.asarray(upper) - 1e-8)

    def residual(parameters):
        aspect_penalty = parameters[3] * np.sqrt(len(xy)) * 0.15 / extent
        return np.r_[_distance(parameters, xy), aspect_penalty]

    fit = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.6 / extent,
        max_nfev=75,
    )
    center = fit.x[:2]
    log_radius, log_ratio, angle = fit.x[2:]
    axes = np.exp(log_radius + np.array([0.5, -0.5]) * log_ratio)
    residual_px = np.abs(_distance(fit.x, xy)) * extent
    rms = float(np.sqrt(np.mean(residual_px**2)))
    if (
        not fit.success
        or axes.max() > max_axis
        or log_ratio > 0.995 * log_aspect_limit
        or rms > max(2.5, 0.025 * extent)
        or np.quantile(residual_px, 0.8) > max(1.4, 0.01 * extent)
    ):
        return None
    result = MarkedEllipse(center, axes, float(angle), rms, "ellipse")
    if _coverage(result, xy) < np.deg2rad(105):
        return None
    return MarkedEllipse(origin + center * extent, axes * extent, float(angle), rms, "ellipse")


def fit_marked_ellipse(xy, width, height):
    """Fit a regularized ellipse, or a circle when the marked arc is too short."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or not np.isfinite(xy).all():
        raise MeasurementError("manual_point_count", "请沿液滴边缘选取有效点后再拟合。")
    xy = xy[np.lexsort((xy[:, 1], xy[:, 0]))]
    extent = float(np.max(pdist(xy)))
    origin = xy.mean(axis=0)
    scatter = np.linalg.svd(xy - origin, compute_uv=False)
    if extent < 3 or scatter[-1] / np.sqrt(len(xy)) < max(0.12, 0.0015 * extent):
        raise MeasurementError("manual_arc_curvature", "选点接近直线，请沿边缘扩展选点范围。")

    diagonal = np.hypot(width, height)
    if len(xy) >= 5 and scatter[-1] >= 0.13 * scatter[0]:
        ellipse = _fit_supported_ellipse(
            (xy - origin) / extent, origin, extent, min(1.8, diagonal / extent)
        )
        if ellipse is not None:
            return ellipse

    # A short local arc seldom constrains five independent ellipse parameters.
    # The circle has three, so it also supports leave-one-out checks with four marks.
    from .manual_arc import fit_circle

    center, radius, rms = fit_circle(xy, min(diagonal, 7 * extent))
    chord_fraction = min(1, extent / (2 * radius))
    sagitta = radius * chord_fraction**2 / (1 + np.sqrt(1 - chord_fraction**2))
    if sagitta < max(0.2, 1.5 * rms):
        raise MeasurementError("manual_arc_curvature", "选点弯曲幅度过小，请沿边缘补点。")
    return MarkedEllipse(center, np.array([radius, radius]), 0.0, rms, "circle")
