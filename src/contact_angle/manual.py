"""Manual, baseline-relative contact-angle fits from user-selected edge points.

Each side is fitted as u(h), where h is height above the supplied baseline.
This admits vertical tangents and overhanging (obtuse-angle) contact regions.
For the left side theta=atan2(1,du/dh); for the right side
theta=atan2(1,-du/dh). These are the angles INSIDE the upper liquid phase.
No image pixels, reference angles, sample identities or expected answers enter
the fit. Point selection remains an explicit human/agent input.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import ValidationError
from scipy.optimize import least_squares

from .core import Frame
from .models import MeasurementError, MeasurementOptions

_DEGREES = {"tangent": 1, "quadratic": 2, "cubic": 3}
_LABELS = {"tangent": "手动直线切线", "quadratic": "手动二次曲线", "cubic": "手动三次曲线"}
_SIDES = {"left": "左侧", "right": "右侧"}


def _error(code, message):
    raise MeasurementError(code, message)


def _point(p):
    return {"x": float(p[0]), "y": float(p[1])}


def _points(values, label, width, height, minimum):
    if not isinstance(values, (list, tuple)) or not minimum <= len(values) <= 200:
        _error("manual_point_count", f"{label}需要 {minimum}–200 个边缘点。")
    try:
        points = np.array([[float(p["x"]), float(p["y"])] for p in values])
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise MeasurementError("invalid_manual_points", f"{label}点坐标格式不正确。") from exc
    if not np.isfinite(points).all():
        _error("invalid_manual_points", f"{label}点坐标必须是有限数值。")
    if np.any(
        (points[:, 0] < 0)
        | (points[:, 0] > width - 1)
        | (points[:, 1] < 0)
        | (points[:, 1] > height - 1)
    ):
        _error("manual_point_bounds", f"{label}点必须位于原图范围内。")
    difference = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    np.fill_diagonal(distance, np.inf)
    if np.min(distance) < 0.5:
        _error("duplicate_manual_points", f"{label}有重复点或相距不足 0.5 像素的点。")
    return points


def _fit_side(points, frame, side, degree, width, height):
    label = _SIDES[side]
    uv = frame.to_local(points)
    # Sorting by observed height makes click order irrelevant. The second key
    # gives deterministic ordering when two points have the same height.
    uv = uv[np.lexsort((uv[:, 0], uv[:, 1]))]
    if np.min(uv[:, 1]) < -0.75:
        _error("manual_below_baseline", f"{label}边缘点应位于基线上方的液滴轮廓。")
    span = float(np.ptp(uv[:, 1]))
    if span < 2:
        _error("manual_short_arc", f"{label}点的高度跨度不足 2 像素，无法确定接触切线。")
    if uv[0, 1] > max(12.0, 0.35 * span):
        _error("manual_excessive_extrapolation", f"{label}最近的边缘点离基线过远，请在接触区附近补点。")
    center_h = float((uv[:, 1].max() + uv[:, 1].min()) / 2)
    scale = span / 2
    center_u = float(np.mean(uv[:, 0]))
    t = (uv[:, 1] - center_h) / scale
    design = np.polynomial.polynomial.polyvander(t, degree)
    if np.linalg.matrix_rank(design) < degree + 1:
        _error("manual_degenerate_points", f"{label}有效高度层数不足，请沿连续边缘分散打点。")
    initial = np.linalg.lstsq(design, uv[:, 0] - center_u, rcond=None)[0]
    fit = least_squares(
        lambda c: design @ c + center_u - uv[:, 0],
        initial,
        loss="soft_l1",
        f_scale=0.7,
        max_nfev=250,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        _error("manual_fit_failed", f"{label}拟合没有收敛，请检查点位。")
    coefficients = fit.x
    derivative = np.polynomial.polynomial.polyder(coefficients)
    contact_t = -center_h / scale
    contact_u = float(center_u + np.polynomial.polynomial.polyval(contact_t, coefficients))
    slope = float(np.polynomial.polynomial.polyval(contact_t, derivative) / scale)
    angle = math.degrees(math.atan2(1.0, slope if side == "left" else -slope))
    if not 5 <= angle <= 175:
        _error("manual_unsupported_angle", f"{label}拟合角度超出 5°–175°，请检查近接触区的点和基线。")
    contact = frame.to_image([contact_u, 0])
    if not (0 <= contact[0] <= width - 1 and 0 <= contact[1] <= height - 1):
        _error("manual_contact_bounds", f"{label}拟合接触点落在图像外，请检查基线和点位。")
    extrapolation = float(np.linalg.norm(np.array([contact_u, 0]) - uv[0]))
    observed_length = float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())
    if extrapolation > max(12.0, 0.45 * observed_length):
        _error("manual_excessive_extrapolation", f"{label}接触点需要过长外推，请在靠近基线处补点。")
    # Report normal-distance approximation in pixels, rather than x residuals
    # alone, so an obtuse/acute side is not assigned an inflated pixel residual.
    local_slopes = np.polynomial.polynomial.polyval(t, derivative) / scale
    normal_residual = fit.fun / np.sqrt(1 + local_slopes * local_slopes)
    rms = float(np.sqrt(np.mean(normal_residual**2)))
    if rms > 8:
        _error("manual_inconsistent_points", f"{label}点位偏离拟合曲线过大，请沿同一侧的连续边缘重新打点。")
    tangent = frame.basis @ (np.array([slope, 1.0]) / math.hypot(slope, 1.0))
    hs = np.linspace(0, float(uv[:, 1].max()), 120)
    xs = center_u + np.polynomial.polynomial.polyval((hs - center_h) / scale, coefficients)
    curve = frame.to_image(np.c_[xs, hs])
    warnings = []
    if rms > 1.5:
        warnings.append(f"{label}拟合残差超过 1.5 像素，请核查边缘点。")
    if extrapolation > max(3.0, 0.12 * observed_length):
        warnings.append(f"{label}接触点由可见边缘外推得到，建议在近基线处补点核查。")
    if len(points) == degree + 1:
        warnings.append(f"{label}点数刚好满足所选曲线，建议增加边缘点交叉检查。")
    if min(angle, 180 - angle) < 20:
        warnings.append(f"{label}角度接近测量边界，对基线和近接触区点位更敏感。")
    return {
        "angle_deg": angle,
        "contact_point": _point(contact),
        "tangent_vector": _point(tangent),
        "fit_rms_px": rms,
        "fit_points": len(points),
        "extrapolation_px": extrapolation,
    }, [_point(p) for p in curve], warnings


def _fit_pair(points, frame, degree, width, height):
    sides, curves, warnings = {}, [], []
    for side in ("left", "right"):
        sides[side], curve, side_warnings = _fit_side(
            points[side], frame, side, degree, width, height
        )
        curves.append({"side": side, "points": curve})
        warnings.extend(side_warnings)
    contact_uv = frame.to_local(
        [[sides[s]["contact_point"]["x"], sides[s]["contact_point"]["y"]] for s in ("left", "right")]
    )
    if contact_uv[0, 0] >= contact_uv[1, 0]:
        _error("manual_crossed_contacts", "左右接触点顺序交叉，请检查左右点是否放反。")
    return sides, curves, warnings


def _range(values, primary):
    return {
        "min_deg": float(min(values)),
        "max_deg": float(max(values)),
        "span_deg": float(max(values) - min(values)),
        "median_deg": float(np.median(values)),
        "max_abs_delta_deg": float(max(abs(v - primary) for v in values)),
    }


def _sensitivity(points, frame, degree, primary, width, height):
    # Compare supported models and near-contact windows of the same observed
    # points. Baseline and clicked point locations remain fixed throughout.
    ordered = {
        side: p[np.lexsort((frame.to_local(p)[:, 0], frame.to_local(p)[:, 1]))]
        for side, p in points.items()
    }
    candidates = [("主方法·全部点", degree, ordered)]
    if all(len(p) >= 3 for p in ordered.values()):
        alternate = 2 if degree != 2 else 3
        if all(len(p) >= alternate + 1 for p in ordered.values()):
            candidates.append((f"交叉检查·{alternate}次曲线", alternate, ordered))
        near_count = {side: max(degree + 1, math.ceil(len(p) * 0.75)) for side, p in ordered.items()}
        near = {side: p[: near_count[side]] for side, p in ordered.items()}
        if any(len(near[s]) < len(ordered[s]) for s in ordered):
            candidates.append(("主方法·近接触区75%点", degree, near))
    if all(len(p) >= 6 for p in ordered.values()):
        for fraction in (0.3, 0.65):
            subset = {
                side: np.delete(p, int((len(p) - 1) * fraction), axis=0)
                for side, p in ordered.items()
            }
            candidates.append((f"主方法·留一检查{fraction:.2f}", degree, subset))
    runs, accepted, seen = [], [], set()
    for name, candidate_degree, subset in candidates:
        key = (candidate_degree, *(tuple(p.ravel()) for p in subset.values()))
        if key in seen:
            continue
        seen.add(key)
        run = {
            "name": name,
            "method": f"manual_degree_{candidate_degree}",
            "parameters": {"degree": candidate_degree, "left_points": len(subset["left"]),
                           "right_points": len(subset["right"])},
        }
        try:
            sides, _, _ = _fit_pair(subset, frame, candidate_degree, width, height)
            run.update(
                left_angle_deg=sides["left"]["angle_deg"],
                right_angle_deg=sides["right"]["angle_deg"],
                left_delta_deg=sides["left"]["angle_deg"] - primary["left"]["angle_deg"],
                right_delta_deg=sides["right"]["angle_deg"] - primary["right"]["angle_deg"],
            )
            if max(sides[s]["fit_rms_px"] for s in sides) > 1.5:
                run.update(status="excluded", error="拟合残差超过 1.5 像素。")
            else:
                run.update(status="accepted", error=None)
                accepted.append(sides)
        except MeasurementError as exc:
            run.update(status="failed", error=exc.message)
        runs.append(run)
    ranges = {side: None for side in ("left", "right")}
    if len(accepted) >= 2:
        ranges = {
            side: _range([s[side]["angle_deg"] for s in accepted], primary[side]["angle_deg"])
            for side in ("left", "right")
        }
    return {
        "interpretation": "同一基线和现有边缘点下，不同拟合阶数、窗口与留点检查的范围；不包含共同基线或共同打点偏差，不是置信区间。",
        **ranges,
        "accepted_runs": len(accepted),
        "excluded_runs": sum(r["status"] == "excluded" for r in runs),
        "failed_runs": sum(r["status"] == "failed" for r in runs),
        "sensitive": any(r is not None and r["span_deg"] > 3 for r in ranges.values()),
        "alert_deg": 3.0,
        "runs": runs,
    }


def measure_manual(image, options):
    """Fit explicit user/agent clicks, in original image coordinates.

    ``quadratic``/``cubic`` mean a degree2/3 polynomial u(h) fitted independently
    to each contact region; ``tangent`` fits a straight u(h) line (two or more
    points). Selected methods never silently fall back to another order.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        _error("invalid_image", "手动测量需要已解码的 BGR 图片。")
    if not isinstance(options, dict):
        _error("invalid_options", "手动测量参数格式不正确。")
    height, width = image.shape[:2]
    method = options.get("manual_method", "quadratic")
    if method == "spline" or options.get("contour_points"):
        from .contour_manual import measure_contour

        return measure_contour(image, options)
    if method not in _DEGREES:
        _error("invalid_manual_method", "请选择二次曲线、三次曲线或直线切线。")
    baseline = options.get("baseline")
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
        _error("manual_baseline_required", "请先在固体表面标出两个基线点。")
    _points(baseline, "基线", width, height, 2)
    try:
        validated = MeasurementOptions(baseline=baseline, sensitivity=False)
    except ValidationError as exc:
        raise MeasurementError("invalid_baseline", "基线两点横向间距至少为5像素，倾斜不超过45°。") from exc
    frame = Frame.from_options(validated)
    degree = _DEGREES[method]
    points = {
        side: _points(options.get(f"{side}_points"), _SIDES[side], width, height, degree + 1)
        for side in ("left", "right")
    }
    sides, curves, warnings = _fit_pair(points, frame, degree, width, height)
    sensitivity = None
    if options.get("sensitivity", True):
        sensitivity = _sensitivity(points, frame, degree, sides, width, height)
        if sensitivity["left"] is None or sensitivity["right"] is None:
            warnings.append("可用的独立拟合/窗口比较不足两项，暂不输出敏感性范围；请增加近接触区边缘点。")
        elif sensitivity["sensitive"]:
            warnings.append("不同合理拟合方法的接触角跨度超过3°，请复核点位、基线和拟合窗口。")
        if sensitivity["failed_runs"] or sensitivity["excluded_runs"]:
            warnings.append("部分敏感性候选未通过几何或残差检查，详情保留在逐方法记录中。")
    asymmetry = abs(sides["left"]["angle_deg"] - sides["right"]["angle_deg"])
    if asymmetry > 5:
        warnings.append("左右接触角差超过5°，请分别报告并核查液滴是否非对称。")
    selected = {side: [_point(p) for p in points[side]] for side in points}
    return {
        "mode": "manual",
        "algorithm": "manual",
        "method_label": _LABELS[method],
        "image_width": width,
        "image_height": height,
        **sides,
        "mean_angle_deg": (sides["left"]["angle_deg"] + sides["right"]["angle_deg"]) / 2,
        "asymmetry_deg": asymmetry,
        "baseline": [p.model_dump() for p in validated.baseline],
        "status": "review" if warnings else "ok",
        "warnings": warnings,
        "sensitivity": sensitivity,
        "geometry": {"contours": [], "fits": curves, "selected_points": selected},
    }
