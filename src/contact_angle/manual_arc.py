"""Manual conic fitting, surface intersections and contact angles."""

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.distance import pdist

from .core import Frame
from .models import MeasurementError, MeasurementOptions

MIN_MANUAL_POINTS = 5


def point(value):
    return {"x": float(value[0]), "y": float(value[1])}


def contour_observations(values, width, height):
    if not MIN_MANUAL_POINTS <= len(values) <= 400:
        raise MeasurementError("manual_point_count", "请选取 5–400 个液滴边缘点。")
    try:
        xy = np.array([[p["x"], p["y"]] for p in values], dtype=float)
    except (KeyError, ValueError, TypeError) as exc:
        raise MeasurementError("invalid_manual_points", "轮廓点需要 x、y 数值。") from exc
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [width, height]):
        raise MeasurementError("manual_point_bounds", "轮廓点须位于原图内。")
    if min(pdist(xy)) < 0.5:
        raise MeasurementError("duplicate_manual_points", "轮廓存在相距不足 0.5 px 的重复点。")
    return xy


def spans_interface(depth, endpoint_limit):
    peak = int(np.argmax(depth))
    return (
        len(depth) >= 7
        and 2 <= peak <= len(depth) - 3
        and max(depth[0], depth[-1]) <= endpoint_limit
    )


def fit_circle(xy, maximum_radius):
    # Normalize the coordinates so the circle equations stay conditioned on large images
    # and on partial arcs far from the image origin.
    origin = xy.mean(axis=0)
    scale = max(np.ptp(xy, axis=0))
    normalized = (xy - origin) / scale
    u, v = normalized.T
    parameters, _, rank, _ = np.linalg.lstsq(
        np.c_[2 * u, 2 * v, np.ones(len(xy))], u**2 + v**2, rcond=None
    )
    center, constant = parameters[:2], parameters[2]
    radius = np.sqrt(max(0.0, constant + center @ center))
    limit = maximum_radius / scale
    if rank < 3 or not 0 < radius < limit:
        raise MeasurementError("manual_arc_curvature", "边缘弧度不足，请沿轮廓拉开选点范围。")

    def residual(parameters):
        return np.linalg.norm(normalized - parameters[:2], axis=1) - parameters[2]

    fit = least_squares(
        residual,
        [*center, radius],
        bounds=([-2 * limit, -2 * limit, 0.01], [2 * limit, 2 * limit, limit]),
        loss="soft_l1",
        f_scale=0.5 / scale,
        max_nfev=80,
    )
    rms = float(np.sqrt(np.mean(fit.fun**2)) * scale)
    if not fit.success or fit.x[2] >= 0.99 * limit or rms > max(1.5, 0.02 * scale):
        raise MeasurementError("manual_arc_fit", "选点无法稳定拟合为圆弧，请沿可见边缘补点。")
    return origin + scale * fit.x[:2], scale * fit.x[2], rms


def liquid_arc(shape, depth, marks):
    phases = np.linspace(0, 2 * np.pi, 721)
    heights = depth(shape.at(phases))
    if not np.any(heights < 0) or not np.any(heights > 0):
        raise MeasurementError("manual_arc_contacts", "拟合轮廓未与基线相交，请补充接触区边缘点。")

    # Start on the solid side so arcs crossing the angular seam are kept in one piece.
    phases += phases[np.argmin(heights)]
    wet = depth(shape.at(phases)) > 0
    transitions = np.flatnonzero(wet[1:] != wet[:-1])
    roots = [brentq(lambda t: depth(shape.at(t)), phases[i], phases[i + 1]) for i in transitions]
    mark_phases = shape.phase(marks)
    mark_phases = phases[0] + (mark_phases - phases[0]) % (2 * np.pi)
    start, end = max(
        zip(roots[::2], roots[1::2]),
        key=lambda pair: (
            np.sum((mark_phases >= pair[0]) & (mark_phases <= pair[1])),
            pair[1] - pair[0],
        ),
    )
    path = shape.at(np.linspace(start, end, 241))
    inward = shape.derivative(np.array([start, end]))
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    inward[1] *= -1
    if path[0, 0] > path[-1, 0]:
        path, inward = path[::-1].copy(), inward[::-1].copy()
    return path, inward


def observed_arc(shape, xy):
    phases = np.sort(shape.phase(xy))
    gaps = np.diff(np.r_[phases, phases[0] + 2 * np.pi])
    cut = int(np.argmax(gaps))
    start = phases[(cut + 1) % len(phases)]
    return shape.at(np.linspace(start, start + 2 * np.pi - gaps[cut], 241))


def preview_contour(xy, width, height, baseline, orientation, surface_curve=()):
    """Fit an outline without computing angles or creating a measurement job."""
    from .manual_ellipse import fit_marked_ellipse

    frame = None
    if baseline:
        try:
            frame = Frame.from_options(MeasurementOptions(baseline=baseline, sensitivity=False))
        except ValueError as exc:
            raise MeasurementError(
                "invalid_baseline", "无法使用该基线，请重新选取表面点。"
            ) from exc
        if orientation == "auto":
            orientation = "pendant" if np.median(frame.to_local(xy)[:, 1]) < 0 else "sessile"

    local = frame is not None and orientation == "sessile"
    marks = frame.to_local(xy) if local else xy
    shape = fit_marked_ellipse(marks, width, height)
    path = None
    if frame:
        if local:

            def depth(p):
                return p[..., 1]
        elif len(surface_curve) >= 2:
            surface_x, surface_y = np.asarray(sorted((p["x"], p["y"]) for p in surface_curve)).T

            def depth(p):
                return p[..., 1] - np.interp(p[..., 0], surface_x, surface_y)
        else:

            def depth(p):
                return -frame.to_local(p)[..., 1]

        try:
            path, _ = liquid_arc(shape, depth, marks)
        except MeasurementError:
            # A wrong or missing contact line still allows a preview of the observed arc.
            pass
    if path is not None:
        image_path = frame.to_image(path) if local else path
        if np.all(image_path >= -1) and np.all(image_path <= [width, height]):
            return image_path
    outline = observed_arc(shape, marks)
    return frame.to_image(outline) if local else outline


def measure_marked_contour(
    xy, width, height, baseline, sensitivity, surface=None, uncertainty=0, *, circle_only=False
):
    from .manual_ellipse import MarkedEllipse, fit_marked_ellipse
    from .studio_service import range_from

    frame = None
    if surface is None:
        try:
            frame = Frame.from_options(MeasurementOptions(baseline=baseline, sensitivity=False))
        except ValueError as exc:
            raise MeasurementError("invalid_baseline", "请先设置有效的表面基线。") from exc
    marks = frame.to_local(xy) if frame else xy

    def depth(p):
        return p[..., 1] if frame else p[..., 1] - surface(p[..., 0])

    if np.min(depth(marks)) < (-0.75 if frame else -3):
        raise MeasurementError("manual_below_baseline", "轮廓点位于固体表面一侧。")

    def calculate(observations, offset=0):
        if circle_only:
            center, radius, rms = fit_circle(observations, np.hypot(width, height))
            shape = MarkedEllipse(center, np.array([radius, radius]), 0.0, rms, "circle")
        else:
            shape = fit_marked_ellipse(observations, width, height)
        path, inward = liquid_arc(shape, lambda p: depth(p) - offset, observations)
        split = int(np.argmax(depth(path)))
        image_path = frame.to_image(path) if frame else path
        contacts = image_path[[0, -1]]
        if np.any(contacts < 0) or np.any(contacts > [width - 1, height - 1]):
            raise MeasurementError(
                "manual_contact_bounds", "拟合的接触点落在原图外，请补充边缘点。"
            )
        fits = []
        sides = {}
        for side, index, contact, derivative in zip(("left", "right"), (0, -1), contacts, inward):
            sign = 1 if side == "left" else -1
            slope = 0 if frame else surface.slope(contact[0])
            along = sign * np.array([1.0, slope]) / np.hypot(1, slope)
            angle = float(np.degrees(np.arccos(np.clip(derivative @ along, -1, 1))))
            if not 5 <= angle <= 175 or derivative[1] - slope * derivative[0] <= 0:
                raise MeasurementError(
                    "manual_invalid_tangent", "拟合轮廓与表面的交点无法支持接触角。"
                )
            tangent = frame.basis @ derivative if frame else derivative
            measured = {
                "angle_deg": angle,
                "contact_point": point(contact),
                "tangent_vector": point(tangent),
                "fit_rms_px": shape.rms,
                "fit_points": len(observations),
                "extrapolation_px": float(np.linalg.norm(observations - path[index], axis=1).min()),
            }
            if not frame:
                measured["surface_tangent"] = point(along)
            sides[side] = measured
            segment = image_path[: split + 1] if side == "left" else image_path[split:][::-1]
            fits.append({"side": side, "points": [point(p) for p in segment]})
        return sides, fits, shape.model

    primary, fits, model = calculate(marks)
    label = (
        "手动圆弧估算" if circle_only else "手动椭圆拟合" if model == "ellipse" else "手动近圆拟合"
    )
    report = None
    if sensitivity:
        alternatives = [("主拟合", marks, 0)]
        # Sample leave-one-out checks across the arc without scaling the work with point count.
        alternatives += [
            (f"去掉第 {i + 1} 点", np.delete(marks, i, axis=0), 0)
            for i in np.linspace(0, len(marks) - 1, min(5, len(marks)), dtype=int)
        ]
        if uncertainty:
            alternatives += [
                (f"表面偏移 {d:+.2f} px", marks, d) for d in (-uncertainty, uncertainty)
            ]
        runs = []
        accepted = []
        for i, (name, subset, offset) in enumerate(alternatives):
            run = {
                "name": name,
                "method": "manual_" + model,
                "parameters": {"surface_shift_px": offset},
            }
            try:
                result, _, variant = (primary, fits, model) if i == 0 else calculate(subset, offset)
                run.update(
                    status="accepted",
                    error=None,
                    method="manual_" + variant,
                    **{f"{s}_angle_deg": v["angle_deg"] for s, v in result.items()},
                )
                accepted.append(result)
            except MeasurementError as exc:
                run.update(status="failed", error=exc.message)
            runs.append(run)
        report = {
            "interpretation": "比较轮廓点选取和可用表面位置变化对接触角的影响。",
            "runs": runs,
            "accepted_runs": len(accepted),
            "failed_runs": len(runs) - len(accepted),
            "excluded_runs": 0,
            **{
                side: range_from(
                    [r[side]["angle_deg"] for r in accepted], primary[side]["angle_deg"]
                )
                for side in primary
            },
        }

    left, right = (primary[s]["angle_deg"] for s in ("left", "right"))
    warnings = []
    if circle_only:
        warnings.append("角度由所选边缘点拟合的同一圆弧外推。")
    elif max(side["extrapolation_px"] for side in primary.values()) > max(
        5, min(width, height) * 0.02
    ):
        warnings.append("至少一侧的接触点由所选轮廓向基线外推。")
    return {
        "mode": "manual",
        "algorithm": "manual",
        "method_label": label,
        **primary,
        "baseline": sorted(baseline, key=lambda p: p["x"]),
        "mean_angle_deg": (left + right) / 2,
        "asymmetry_deg": abs(left - right),
        "status": "review" if warnings else "ok",
        "warnings": warnings,
        "sensitivity": report,
        "image_width": width,
        "image_height": height,
        "geometry": {
            "contours": [],
            "fits": fits,
            "selected_points": {"contour": [point(p) for p in xy]},
        },
    }


def measure_partial_contour(xy, width, height, baseline, sensitivity, surface=None, uncertainty=0):
    return measure_marked_contour(
        xy, width, height, baseline, sensitivity, surface, uncertainty, circle_only=True
    )
