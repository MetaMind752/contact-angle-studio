"""Measure a drop hanging below its support in original (x-right, y-down) pixels."""

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares

from .manual_arc import contour_observations, measure_partial_contour, spans_interface
from .models import MeasurementError


def _error(message):
    raise MeasurementError("pendant_contour", message)


def _point(p):
    return {"x": float(p[0]), "y": float(p[1])}


def _xy(points):
    return np.array([[p["x"], p["y"]] for p in points], dtype=float)


def _unit(v):
    return v / np.linalg.norm(v)


class Surface:
    basis = np.eye(2)

    def __init__(self, surface, offset=0.0):
        curve = _xy(surface.get("surface_curve") or surface["baseline"])
        self.x, self.y = curve[np.argsort(curve[:, 0])].T
        self.offset = offset

    def __call__(self, x):
        return np.interp(x, self.x, self.y) + self.offset

    def slope(self, x):
        return (self(x + 1) - self(x - 1)) / 2

    def to_image(self, pts):
        pts = np.array(pts, float, copy=True)
        pts[..., 1] += self(pts[..., 0])
        return pts


def _orient(side, p, tangent, surf):
    slope = surf.slope(p[0])
    tangent = _unit(tangent)
    if tangent[1] - slope * tangent[0] <= 0:
        _error("接触区液滴切线未指向液体。")
    along = _unit(np.array([1.0, slope])) * (1 if side == "left" else -1)
    angle = np.degrees(np.arccos(np.clip(tangent @ along, -1, 1)))
    return {
        "contact_point": _point(p),
        "tangent_vector": _point(tangent),
        "surface_tangent": _point(along),
        "angle_deg": float(angle),
    }


def _silhouette(gray, surf, foot, exclusions):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.85)
    threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    xx = np.arange(w)
    width = foot[1] - foot[0]
    mid = np.mean(foot)
    clearance = max(2.0, 0.012 * width)
    mask = (blurred <= threshold) & (np.arange(h)[:, None] > surf(xx) + clearance)
    margin = max(0.72 * width, 0.13 * w)
    mask[:, (xx < foot[0] - margin) | (xx > foot[1] + margin)] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    ranked = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 8 or bh < 6 or area < 30 or x + bw < foot[0] or x > foot[1]:
            continue
        # Prefer the liquid body below the surface to detached debris on the sides.
        ranked.append((area / (1 + abs(x + bw / 2 - mid) / max(8, width)), i))
    if not ranked:
        _error("表面下方未检测到可辨识的液滴。")
    i = max(ranked)[1]
    x, y, bw, bh, _ = stats[i]
    yy, xx_local = np.nonzero(labels[y : y + bh, x : x + bw] == i)
    minima, maxima = np.full(bh, w), np.full(bh, -1)
    np.minimum.at(minima, yy, xx_local + x)
    np.maximum.at(maxima, yy, xx_local + x)
    ys = np.arange(y, y + bh)
    sides = {}
    removed = {}
    for side, xs, step in [("left", minima, -1), ("right", maxima, 1)]:
        valid = (xs > 0) & (xs < w - 1)
        xs = xs[valid]
        rows = ys[valid]
        dark = blurred[rows, xs].astype(float)
        light = blurred[rows, xs + step].astype(float)
        # A mask edge at the clearance boundary, crop, or hole isn't a liquid edge.
        valid = (light > threshold) & (dark <= threshold)
        rows = rows[valid]
        xs = xs[valid] + step * (threshold - dark[valid]) / (light - dark)[valid]
        valid = np.ones(len(xs), bool)
        for r in exclusions:
            valid &= ~(
                (xs >= r["x"])
                & (xs < r["x"] + r["width"])
                & (rows >= r["y"])
                & (rows < r["y"] + r["height"])
            )
        sides[side] = np.column_stack([xs[valid], rows[valid]])
        removed[side] = int(np.sum(~valid))
    return (
        sides,
        float(bw),
        {
            "otsu_threshold": float(threshold),
            "liquid_width_px": int(bw),
            "liquid_height_px": int(bh),
            "excluded_edge_rows": removed,
        },
    )


def _fit_side(
    side,
    points,
    surface,
    scale,
    high_fraction,
    low_fraction,
    model,
    h,
    w,
    clean_depth=0.0,
    sparse=False,
):
    min_points = 3 if sparse else 6
    if len(points) < min_points:
        _error("排除遮挡区域后，可见液滴轮廓不足。")
    depth = points[:, 1] - surface(points[:, 0])
    maximum = np.max(depth)
    lo = -3.5 if sparse else max(clean_depth, min(max(1.2, scale * low_fraction), 0.20 * maximum))
    hi = min(max(9.0, scale * high_fraction), 0.76 * maximum)
    subset = points[(depth > lo) & (depth <= hi)]
    if sparse and len(subset) < min(5, len(points)):
        # Sparse clicks can have wide spacing. Use the nearest actual points,
        # never a spline's unsupported endpoint derivative.
        subset = points[np.argsort(depth)[: min(5, len(points))]]
        hi = float(np.max(subset[:, 1] - surface(subset[:, 0])))
    if len(subset) < min_points:
        _error("接触区附近的独立可见边缘行不足。")
    subset = subset[np.argsort(subset[:, 1] - surface(subset[:, 0]))]
    # A few substrate fibres can touch the blob. Trim abrupt lateral excursions
    # against a robust smooth local prediction before estimating the endpoint slope.
    if len(subset) >= 12:
        z = (subset[:, 1] - np.mean(subset[:, 1])) / max(np.ptp(subset[:, 1]), 1.0)
        a = np.column_stack([np.ones_like(z), z, z * z])
        vals = subset[:, 0]
        pred = a @ _robust_linear(a, vals)
        resid = vals - pred
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        subset = subset[np.abs(resid - np.median(resid)) <= max(2.8, 3.5 * mad)]
    if len(subset) < min_points or np.ptp(subset[:, 1]) < 4:
        _error("可辨识液滴边缘跨度不足。")
    # Rotate the local polynomial axes toward the arc chord for both shallow and
    # steep contacts; the image itself and all reported coordinates stay original.
    origin = np.mean(subset, axis=0)
    _, _, vt = np.linalg.svd(subset - origin, full_matrices=False)
    u = vt[0] * (1 if np.dot(vt[0], subset[-1] - subset[0]) > 0 else -1)
    v = np.array([-u[1], u[0]])
    t = (subset - origin) @ u
    q = (subset - origin) @ v
    extent = max(1.0, np.ptp(t))
    a = np.column_stack([np.ones(len(t)), t / extent, (t / extent) ** 2])
    co = _robust_linear(a, q)
    poly = np.polynomial.Polynomial(co)

    def q_at(s):
        return poly(s / extent)

    def slope_at(s):
        return poly.deriv()(s / extent) / extent

    if model == "circle":
        k = 2 * co[2] / extent**2
        if abs(k) < 1 / (100 * scale):
            _error("局部曲率不足以独立确定圆弧。")
        centre0 = np.array([-co[1] / extent / k, co[0] + 1 / k])
        radius0 = np.median(np.hypot(t - centre0[0], q - centre0[1]))
        sol = least_squares(
            lambda p: np.hypot(t - p[0], q - p[1]) - p[2],
            [*centre0, radius0],
            loss="soft_l1",
            f_scale=0.75,
            max_nfev=100,
        )
        tc, qc, radius = sol.x
        if not sol.success or radius <= 0:
            _error("接触区圆弧未收敛。")
        sign = np.sign(np.median(q - qc))

        def q_at(s):
            return qc + sign * np.sqrt(np.maximum(0, radius**2 - (s - tc) ** 2))

        def slope_at(s):
            return -sign * (s - tc) / max(1e-10, np.sqrt(max(0, radius**2 - (s - tc) ** 2)))

    def xy(s):
        return origin + np.asarray(s)[..., None] * u + np.asarray(q_at(s))[..., None] * v

    def gap(s):
        p = xy(s)
        return p[..., 1] - surface(p[..., 0])

    # Find the nearest physical crossing going back from the observed liquid edge.
    grid = np.linspace(
        t.min() - max(16.0, min(0.28 * scale, 0.85 * extent)), np.quantile(t, 0.3), 85
    )
    a0 = gap(grid)
    idx = np.flatnonzero((a0[:-1] <= 0) & (a0[1:] >= 0))
    if model == "circle":
        idx = idx[(np.abs(grid[idx] - tc) < radius) & (np.abs(grid[idx + 1] - tc) < radius)]
    if not len(idx):
        _error("可见液滴边缘无法可靠延伸到表面。")
    j = idx[-1]
    root = brentq(gap, grid[j], grid[j + 1])
    p = xy(root)
    if not 0 <= p[0] < w or not 0 <= p[1] < h:
        _error("拟合的接触点不在原图内。")
    rms = np.sqrt(np.mean((q_at(t) - q) ** 2))
    extrap = np.min(np.linalg.norm(subset - p, axis=1))
    if extrap > max(10.0, 0.9 * extent):
        _error("接触点外推超出可见侧缘支持长度。")
    if np.any(np.diff(gap(np.linspace(root, np.median(t), 30))) < -0.1):
        _error("外推到表面的轮廓出现无依据的折返。")
    if rms > max(3.0, 0.012 * scale):
        _error("液滴边缘残差过大，不能支持该局部模型。")
    info = _orient(side, p, u + v * slope_at(root), surface)
    info.update(
        fit_rms_px=float(rms),
        fit_points=len(subset),
        extrapolation_px=float(extrap),
        fit_depth_window_px=[float(lo), float(hi)],
        model=model,
    )
    return info, {"side": side, "points": [_point(a) for a in xy(np.linspace(root, t.max(), 75))]}


def _robust_linear(a, y):
    co = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(8):
        res = y - a @ co
        mad = max(0.45, 1.4826 * np.median(np.abs(res - np.median(res))))
        weight = 1 / np.sqrt(np.maximum(1.0, np.abs(res) / (1.5 * mad)))
        co = np.linalg.lstsq(a * weight[:, None], y * weight, rcond=None)[0]
    return co


def _clean_depth(side, points, surf, scale, h, w):
    """Find the first sustained departure from the clean flank going toward the solid.

    A dark fibre attached to the body *is* part of a thresholded connected component.
    Its identity changes only when followed up from a coherent deeper interface.
    Compare withheld neck rows to a deep reference arc, measuring normal residuals.
    """
    for model in ("circle", "quadratic"):
        try:
            seed, curve = _fit_side(side, points, surf, scale, 0.39, 0.085, model, h, w)
            break
        except MeasurementError:
            continue
    else:
        return 0.0, {"seed_model": None, "rejected_neck_rows": 0}
    c = _xy(curve["points"])
    d = points[:, 1] - surf(points[:, 0])
    order = np.argsort(d)
    yy, xx = c[:, 1], c[:, 0]
    order_y = np.argsort(yy)
    yy, xx = yy[order_y], xx[order_y]
    dx = np.gradient(xx, yy)
    slope = np.interp(points[:, 1], yy, dx)
    resid = np.abs(points[:, 0] - np.interp(points[:, 1], yy, xx)) / np.hypot(1, slope)
    possible = order[(d[order] <= min(0.16 * scale, 0.4 * d.max())) & (points[order, 1] >= yy[0])]
    bad = resid[possible] > max(1.35, 3 * seed["fit_rms_px"])
    run = 0
    depth = 0.0
    for j in range(len(bad) - 1, -1, -1):
        run = run + 1 if bad[j] else 0
        if run == 3:
            depth = float(d[possible[j + 2]] + 0.3)
            break
    return depth, {
        "seed_model": model,
        "rejected_neck_rows": int(np.sum(d <= depth)),
        "neck_exclusion_depth_px": depth,
        "seed_fit_rms_px": seed["fit_rms_px"],
    }


def fit_pendant(image, surface, sensitivity=True, contour_points=None, excluded_regions=None):
    """Intersect each independent visible interface with its local support curve."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    surf = Surface(surface)
    names = ("left", "right")
    manual = contour_points is not None
    obs = surface.get("baseline_points") or []
    residual = _xy(obs)[:, 1] - surf(_xy(obs)[:, 0]) if obs else np.array([0.0])
    # One pixel of sampling error plus the scatter of observed shoulders. A surface
    # estimator may supply a larger independent bound (e.g. hidden curvature).
    uncertainty = float(
        max(0.75, np.percentile(np.abs(residual), 75), surface.get("uncertainty_px", 0))
    )
    diagnostics = {"surface_shift_px": uncertainty}
    coords = []
    if manual:
        pts = contour_observations(contour_points, w, h)
        depth = pts[:, 1] - surf(pts[:, 0])
        if not spans_interface(depth, max(10, 0.075 * np.ptp(pts[:, 1]))):
            result = measure_partial_contour(
                pts, w, h, surface["baseline"], sensitivity, surf, uncertainty
            )
            result.update(
                surface_curve=surface.get("surface_curve", surface["baseline"]),
                baseline_points=obs,
                contour_diagnostics=diagnostics,
            )
            return result
        if pts[0, 0] > pts[-1, 0]:
            pts = pts[::-1].copy()

        depth = pts[:, 1] - surf(pts[:, 0])
        peak = int(np.argmax(depth))
        tol = max(2.0, 0.02 * np.ptp(depth))
        if (
            peak < 2
            or peak > len(pts) - 3
            or np.any(depth < -3.0)
            or np.any(np.diff(depth[: peak + 1]) < -tol)
            or np.any(np.diff(depth[peak:]) > tol)
        ):
            _error("从一侧接触区沿悬滴轮廓依次打点至另一侧，包含最低点。")
        edges = {"left": pts[: peak + 1], "right": pts[peak:][::-1]}
        scale = np.ptp(pts[:, 0])
        clean = {side: 0.0 for side in names}
        jobs = [
            ("主拟合", "quadratic", 0.30, 0.0, 0.0),
            ("较长窗口", "quadratic", 0.42, 0.0, 0.0),
            ("独立圆弧", "circle", 0.39, 0.0, 0.0),
        ]
        coords = [_point(p) for p in pts]
    else:
        edges, scale, extra = _silhouette(
            gray, surf, surface.get("footprint", [0, w - 1]), excluded_regions or []
        )
        diagnostics.update(extra)

        clean = {}
        diagnostics["neck"] = {}
        for side in names:
            clean[side], diagnostics["neck"][side] = _clean_depth(
                side, edges[side], surf, scale, h, w
            )
        jobs = [
            ("主拟合", "quadratic", 0.31, 0.03, 0.0),
            ("较短轮廓窗口", "quadratic", 0.23, 0.023, 0.0),
            ("较长轮廓窗口", "quadratic", 0.39, 0.05, 0.0),
            ("独立圆弧", "circle", 0.36, 0.04, 0.0),
        ]
    outlines = [
        {
            "side": side,
            "points": [_point(p) for p in edges[side] if manual or p[1] - surf(p[0]) > clean[side]],
        }
        for side in names
    ]

    def calculate(model, hi, lo, shift, variant=None):
        s = Surface(variant or surface, shift)
        parts = {}
        curves = []
        for side in names:
            parts[side], curve = _fit_side(
                side,
                edges[side],
                s,
                scale,
                hi,
                lo,
                model,
                h,
                w,
                clean_depth=max(0.0, clean[side] - shift),
                sparse=manual,
            )
            curves.append(curve)
        if parts["left"]["contact_point"]["x"] >= parts["right"]["contact_point"]["x"]:
            _error("左右接触点交叉。")
        return parts, curves

    results = []
    runs = []
    method = None
    # Any fallback must use the same observed pixels and return a physical intersection.
    for job in jobs:
        label, model, hi, lo, shift = job
        rec = {
            "name": label,
            "method": model,
            "parameters": {"window": hi, "clearance": lo, "surface_shift_px": shift},
        }
        try:
            parts, curves = calculate(model, hi, lo, shift)
            if method is None:
                primary, fits, method = parts, curves, job
            results.append(parts)
            rec.update(
                status="accepted",
                error=None,
                **{s + "_angle_deg": parts[s]["angle_deg"] for s in names},
            )
        except MeasurementError as exc:
            rec.update(status="failed", error=exc.message)
        runs.append(rec)
        if method and not sensitivity:
            break
    if method is None:
        _error("；".join(dict.fromkeys(r["error"] for r in runs)))
    se = None
    if sensitivity:
        variants = [(f"表面位置 {d:+.2f} px", d, None) for d in (-uncertainty, uncertainty)]
        variants.extend(
            (f"肩部曲面 {v['name']}", 0.0, v)
            for v in surface.get("surface_variants", [])
            if v.get("surface_curve")
        )
        for label, offset, variant in variants:
            rec = {
                "name": label,
                "method": method[1],
                "parameters": {
                    "surface_shift_px": offset,
                    "surface_variant": variant["name"] if variant else "primary",
                },
            }
            try:
                parts, _ = calculate(*method[1:4], offset, variant)
                results.append(parts)
                rec.update(
                    status="accepted",
                    error=None,
                    **{s + "_angle_deg": parts[s]["angle_deg"] for s in names},
                )
            except MeasurementError as exc:
                rec.update(status="failed", error=exc.message)
            runs.append(rec)
        se = {
            "interpretation": "同一原图可见轮廓在不同拟合及表面定位下的角度变化；区间取有效方案的最小值与最大值。",
            "runs": runs,
            "accepted_runs": len(results),
            "failed_runs": len(runs) - len(results),
            "excluded_runs": 0,
        }
        for side in names:
            vals = np.array([r[side]["angle_deg"] for r in results])
            se[side] = (
                None
                if len(vals) < 2
                else {
                    "min_deg": float(vals.min()),
                    "max_deg": float(vals.max()),
                    "median_deg": float(np.median(vals)),
                    "span_deg": float(np.ptp(vals)),
                    "max_abs_delta_deg": float(np.max(np.abs(vals - primary[side]["angle_deg"]))),
                }
            )
    warnings = []
    if method != jobs[0]:
        warnings.append(f"主拟合缺少支持，采用{method[0]}。")
    if se and (se["accepted_runs"] < 2 or any(se[s] and se[s]["span_deg"] > 10 for s in names)):
        warnings.append("接触角对表面或轮廓拟合选择敏感。")
    angles = [primary[s]["angle_deg"] for s in names]
    return dict(
        mode="manual" if manual else "auto",
        algorithm="manual" if manual else "pendant",
        method_label="悬滴手动轮廓" if manual else "悬滴局部轮廓",
        **primary,
        mean_angle_deg=float(np.mean(angles)),
        asymmetry_deg=abs(angles[0] - angles[1]),
        status="review" if warnings else "ok",
        warnings=warnings,
        sensitivity=se,
        image_width=w,
        image_height=h,
        baseline=surface["baseline"],
        surface_curve=surface.get("surface_curve", surface["baseline"]),
        baseline_points=obs,
        contour_diagnostics=diagnostics,
        geometry={"contours": outlines, "fits": fits, "selected_points": {"contour": coords}},
    )
