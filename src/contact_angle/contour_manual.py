"""Manual contact angles from one ordered open liquid-interface curve."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import brentq

from .core import Frame
from .models import MeasurementError, MeasurementOptions


def _fail(code, message):
    raise MeasurementError(code, message)


def _point(p):
    return {"x": float(p[0]), "y": float(p[1])}


def _curve(uv, frame, width, height, sigma):
    distances = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    length = float(distances.sum())
    parameter = np.r_[0, np.cumsum(distances)] / length
    weights = np.ones(len(uv))
    weights[[0, -1]] = 8
    try:
        tck, _ = splprep(uv.T, u=parameter, w=weights, k=3, s=len(uv) * sigma**2)
    except (ValueError, TypeError) as exc:
        raise MeasurementError(
            "manual_fit_failed", "轮廓曲线无法拟合，点位应沿连续边缘排列。"
        ) from exc
    fitted = np.array(splev(parameter, tck)).T
    rms = float(np.sqrt(np.mean(np.sum((fitted - uv) ** 2, axis=1))))
    if rms > 2:
        _fail("manual_inconsistent_points", "轮廓点的拟合残差超过 2 px。")
    allowance = min(0.08, max(12, 0.12 * length) / length)
    apex_t = parameter[np.argmax(uv[:, 1])]

    def contact_near(end):
        low, high = (
            (-allowance, min(apex_t, 0.15)) if end == 0 else (max(apex_t, 0.85), 1 + allowance)
        )
        grid = np.linspace(low, high, 160)
        heights = splev(grid, tck)[1]
        roots = []
        for i in range(len(grid) - 1):
            if abs(heights[i]) < 1e-9:
                roots.append(grid[i])
            if heights[i] * heights[i + 1] < 0:
                roots.append(brentq(lambda t: splev(t, tck)[1], grid[i], grid[i + 1]))
        if abs(heights[-1]) < 1e-9:
            roots.append(grid[-1])
        if not roots:
            _fail("manual_contact_unresolved", "轮廓端点附近未与基线相交，请补充接触区边缘点。")
        return min(roots, key=lambda t: abs(t - end))

    start, stop = contact_near(0), contact_near(1)
    if start >= stop:
        _fail("manual_crossed_contacts", "轮廓与基线的交点顺序交叉。")
    sampled_t = np.linspace(start, stop, 400)
    sampled = np.array(splev(sampled_t, tck)).T
    if np.min(sampled[:, 1]) < -0.75:
        _fail("manual_below_baseline", "拟合轮廓穿入固体表面。")
    contact_uv = np.array(splev([start, stop], tck)).T
    if contact_uv[0, 0] >= contact_uv[1, 0]:
        _fail("manual_crossed_contacts", "左右接触点顺序交叉。")
    sides = {}
    for index, side in enumerate(("left", "right")):
        t = [start, stop][index]
        derivative = np.asarray(splev(t, tck, der=1)) * (1 if index == 0 else -1)
        norm = np.linalg.norm(derivative)
        if norm < 1e-8 or derivative[1] <= 0:
            _fail("manual_invalid_tangent", "接触区切线未朝向液滴内部。")
        theta = math.degrees(math.atan2(derivative[1], derivative[0] * (1 if index == 0 else -1)))
        if not 5 <= theta <= 175:
            _fail("manual_unsupported_angle", "拟合接触角超出 5°–175°。")
        contact = frame.to_image(contact_uv[index])
        if np.any(contact < 0) or np.any(contact > [width - 1, height - 1]):
            _fail("manual_contact_bounds", "拟合接触点落在图片之外。")
        extrapolation = float(np.linalg.norm(contact_uv[index] - uv[0 if index == 0 else -1]))
        if extrapolation > max(12, 0.1 * length):
            _fail("manual_excessive_extrapolation", "接触点外推超过可见轮廓支持范围。")
        sides[side] = {
            "angle_deg": theta,
            "contact_point": _point(contact),
            "tangent_vector": _point(frame.basis @ (derivative / norm)),
            "fit_rms_px": rms,
            "fit_points": len(uv),
            "extrapolation_px": extrapolation,
        }
    apex = int(np.argmax(sampled[:, 1]))
    curves = [
        {"side": "left", "points": [_point(p) for p in frame.to_image(sampled[: apex + 1])]},
        {"side": "right", "points": [_point(p) for p in frame.to_image(sampled[apex:][::-1])]},
    ]
    return sides, curves


def measure_contour(image, options):
    height, width = image.shape[:2]
    values = options.get("contour_points") or []
    if not 7 <= len(values) <= 400:
        _fail("manual_point_count", "轮廓拟合需要 7–400 个连续边缘点。")
    try:
        xy = np.asarray([[p["x"], p["y"]] for p in values], float)
    except (KeyError, ValueError, TypeError) as exc:
        raise MeasurementError("invalid_manual_points", "轮廓点需要 x、y 数值。") from exc
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [width, height]):
        _fail("manual_point_bounds", "轮廓点须位于原图内。")
    delta = xy[:, None] - xy[None, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    if np.min(distance) < 0.5:
        _fail("duplicate_manual_points", "轮廓存在相距不足 0.5 px 的重复点。")
    try:
        validated = MeasurementOptions(baseline=options.get("baseline"), sensitivity=False)
    except ValueError as exc:
        raise MeasurementError("invalid_baseline", "轮廓拟合需要有效的表面基线。") from exc
    frame = Frame.from_options(validated)
    uv = frame.to_local(xy)
    if uv[0, 0] > uv[-1, 0]:
        uv, xy = uv[::-1].copy(), xy[::-1].copy()
    peak = int(np.argmax(uv[:, 1]))
    if peak < 2 or peak > len(uv) - 3:
        _fail("manual_contour_order", "轮廓点应从一侧接触区经过顶点，连续排列到另一侧。")
    if np.min(uv[:, 1]) < -0.75:
        _fail("manual_below_baseline", "轮廓点位于固体基线下方。")
    if max(uv[0, 1], uv[-1, 1]) > max(8, 0.06 * np.ptp(uv[:, 1])):
        _fail("manual_excessive_extrapolation", "轮廓两端离基线过远，需要接触区附近的边缘点。")
    # Reject click-order jumps that would create a bridge across the liquid.
    signs = np.diff(uv[:, 1])
    tolerance = max(1, 0.015 * np.ptp(uv[:, 1]))
    if np.any(signs[:peak] < -tolerance) or np.any(signs[peak:] > tolerance):
        _fail("manual_contour_order", "轮廓点顺序折返，请沿边缘连续排列。")
    sides, curves = _curve(uv, frame, width, height, 0.15)
    warnings = []
    extrapolation = max(s["extrapolation_px"] for s in sides.values())
    if extrapolation > 3:
        warnings.append(f"接触点向基线外推 {extrapolation:.1f} px。")
    sensitivity = None
    if options.get("sensitivity", True):
        runs, accepted = [], []
        candidates = [("主曲线", uv, 0.15), ("插值曲线", uv, 0), ("平滑曲线", uv, 0.35)]
        if len(uv) >= 12:
            for fraction in (0.3, 0.65):
                candidates.append(
                    (
                        f"留一检查 {fraction:.2f}",
                        np.delete(uv, int((len(uv) - 1) * fraction), axis=0),
                        0.15,
                    )
                )
        for name, subset, sigma in candidates:
            run = {
                "name": name,
                "method": "manual_parametric_spline",
                "parameters": {"smoothing_px": sigma, "points": len(subset)},
            }
            try:
                candidate, _ = _curve(subset, frame, width, height, sigma)
                run.update(
                    status="accepted",
                    error=None,
                    **{f"{s}_angle_deg": candidate[s]["angle_deg"] for s in sides},
                )
                accepted.append(candidate)
            except MeasurementError as exc:
                run.update(status="failed", error=exc.message)
            runs.append(run)
        sensitivity = {
            "interpretation": "同一基线和轮廓点下平滑程度及留点检查的角度变化。",
            "runs": runs,
            "accepted_runs": len(accepted),
            "failed_runs": len(runs) - len(accepted),
            "excluded_runs": 0,
        }
        for side in sides:
            values = [c[side]["angle_deg"] for c in accepted]
            sensitivity[side] = (
                None
                if len(values) < 2
                else {
                    "min_deg": min(values),
                    "max_deg": max(values),
                    "span_deg": max(values) - min(values),
                    "median_deg": float(np.median(values)),
                    "max_abs_delta_deg": max(abs(v - sides[side]["angle_deg"]) for v in values),
                }
            )
    return {
        "mode": "manual",
        "algorithm": "manual",
        "method_label": "手动轮廓曲线",
        **sides,
        "baseline": [p.model_dump() for p in sorted(validated.baseline, key=lambda p: p.x)],
        "mean_angle_deg": sum(s["angle_deg"] for s in sides.values()) / 2,
        "asymmetry_deg": abs(sides["left"]["angle_deg"] - sides["right"]["angle_deg"]),
        "status": "review" if warnings else "ok",
        "warnings": warnings,
        "sensitivity": sensitivity,
        "image_width": width,
        "image_height": height,
        "geometry": {
            "contours": [],
            "fits": curves,
            "selected_points": {"contour": [_point(p) for p in xy]},
        },
    }
