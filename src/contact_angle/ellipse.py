"""Robust whole-visible-contour ellipse model; no reference angles used.

An ellipse is a candidate silhouette approximation, not a physical universal drop
model. The caller must compare held-out predictions and alternative side models.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from contact_angle.models import MeasurementError


def _distance(parameters, points):
    cx, cy, loga, logb, theta = parameters
    a, b = np.exp([loga, logb])
    ct, st = np.cos(theta), np.sin(theta)
    centered = points - [cx, cy]
    x = centered[:, 0] * ct + centered[:, 1] * st
    y = -centered[:, 0] * st + centered[:, 1] * ct
    radius = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    gradient = np.sqrt(x * x / a**4 + y * y / b**4)
    return (radius - 1) * radius / np.maximum(gradient, 1e-12)


def fit_ellipse(uv, trim=True):
    """Return {'left': result, 'right': result, 'parameters':...}.

    Each side result has the same core keys as fibrous.fit_side: angle, contact,
    slope, rms, points, rejected, observed_low, high, model, p, keep, prediction.
    Predictions accept scalar or ndarray of local heights, returning local u.
    """
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 40 or not np.isfinite(uv).all():
        raise MeasurementError(
            "short_arc", "Ellipse needs at least 40 finite observed contour pixels."
        )
    if min(np.ptp(uv, axis=0)) < 10:
        raise MeasurementError(
            "insufficient_support", "Visible contour has too little extent for ellipse."
        )
    (cx, cy), (diam_a, diam_b), rotation = cv2.fitEllipse(uv.astype(np.float32).reshape(-1, 1, 2))
    if min(diam_a, diam_b) <= 0:
        raise MeasurementError("fit_failed", "Invalid ellipse initialization.")
    initial = np.array([cx, cy, np.log(diam_a / 2), np.log(diam_b / 2), np.deg2rad(rotation)])
    extent = float(max(np.ptp(uv, axis=0)))
    lower = np.r_[uv.min(axis=0) - 2 * extent, np.log(3), np.log(3), -4 * np.pi]
    upper = np.r_[uv.max(axis=0) + 2 * extent, np.log(10 * extent), np.log(10 * extent), 4 * np.pi]
    initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
    keep = np.ones(len(uv), dtype=bool)
    fit = least_squares(
        _distance,
        initial,
        args=(uv,),
        loss="soft_l1",
        f_scale=0.4,
        x_scale="jac",
        bounds=(lower, upper),
        max_nfev=400,
    )
    if trim:
        for _ in range(3):
            residual = _distance(fit.x, uv)
            median = np.median(residual[keep])
            sigma = max(0.12, 1.4826 * np.median(np.abs(residual[keep] - median)))
            candidate = np.abs(residual) <= max(1.0, 3.5 * sigma)
            if candidate.sum() < 40:
                break
            keep = candidate
            fit = least_squares(
                _distance,
                fit.x,
                args=(uv[keep],),
                loss="soft_l1",
                f_scale=0.3,
                x_scale="jac",
                bounds=(lower, upper),
                max_nfev=300,
            )
    if not fit.success:
        raise MeasurementError("fit_failed", "Ellipse fitting failed to converge.")
    if keep.sum() < 0.6 * len(uv):
        raise MeasurementError(
            "contaminated_arc", "Ellipse rejected more than 40% of observed points."
        )
    cx, cy, loga, logb, theta = fit.x
    a, b = np.exp([loga, logb])
    ct, st = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[ct, st], [-st, ct]])
    q = rotation_matrix.T @ np.diag([a**-2, b**-2]) @ rotation_matrix
    det = float(np.linalg.det(q))
    disc0 = q[0, 0] - det * cy**2
    if disc0 <= 0:
        raise MeasurementError("no_intersection", "Fitted ellipse does not meet the baseline.")
    rms = float(np.sqrt(np.mean(_distance(fit.x, uv[keep]) ** 2)))
    results = {}
    for side, sign in (("left", -1), ("right", 1)):

        def prediction(height, sign=sign):
            height = np.asarray(height, dtype=float)
            d = height - cy
            disc = q[0, 0] - det * d * d
            return (
                cx
                + (-q[0, 1] * d + sign * np.sqrt(np.maximum(0, disc))) / q[0, 0]
                + np.where(disc >= 0, 0, np.nan)
            )

        contact = float(prediction(0))
        gx, gy = q @ [contact - cx, -cy]
        slope = float(-gy / gx)
        angle = float(np.rad2deg(np.arctan2(1, slope if side == "left" else -slope)))
        if not 20 < angle < 175:
            raise MeasurementError("invalid_tangent", "Ellipse tangent outside supported range.")
        if not uv[:, 0].min() < contact < uv[:, 0].max():
            raise MeasurementError(
                "invalid_contact", "Ellipse contact outside observed droplet width."
            )
        # Full observed outline is retained for auditing. Caller may derive side-only CV errors.
        results[side] = {
            "angle": angle,
            "contact": contact,
            "slope": slope,
            "rms": rms,
            "points": int(keep.sum()),
            "rejected": int((~keep).sum()),
            "observed_low": float(np.min(uv[keep, 1])),
            "high": float(np.max(uv[keep, 1])),
            "model": "ellipse",
            "p": uv,
            "keep": keep.copy(),
            "prediction": prediction,
        }
    results["parameters"] = {
        "cx": float(cx),
        "cy": float(cy),
        "a": float(a),
        "b": float(b),
        "theta_rad": float(theta),
    }
    return results
