"""Fit a surface from observed pixels on both sides of the droplet."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .models import MeasurementError


def gradient_surface_seed(image):
    """Lateral gradient consensus also supports a pale, low-contrast substrate.

    This is only a location seed: the final fit re-samples near the drop. Equal
    strength edge pixels are retained; a uniform synthetic interface must not
    fail because every lateral column has the same contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1).astype(float)
    h, w = gray.shape
    gradient = np.abs(np.gradient(gray, axis=0))
    columns = np.r_[
        np.arange(int(0.06 * w), int(0.26 * w)), np.arange(int(0.74 * w), int(0.94 * w))
    ]
    lo, hi = int(0.2 * h), int(0.95 * h)
    supported = cv2.dilate(gradient, np.ones((9, 1), np.uint8))
    score = np.median(supported[lo:hi, columns], axis=1)
    best = lo + int(np.argmax(score))
    low, high = max(lo, best - 12), min(hi, best + 13)
    band = gradient[low:high, columns]
    rows = np.argmax(band, axis=0) + low
    strength = np.max(band, axis=0)
    keep = strength >= max(1, float(np.quantile(strength, 0.15)))
    if keep.sum() < 12:
        raise MeasurementError("surface_unresolved", "两侧固体表面没有足够的可见亮度边缘。")
    # Keep a small, evenly distributed subset to satisfy the shared point bound.
    xy = np.c_[columns[keep], rows[keep]]
    xy = xy[np.linspace(0, len(xy) - 1, min(100, len(xy))).astype(int)]
    return fit_surface_points([{"x": float(x), "y": float(y)} for x, y in xy], w, h)["baseline"]


def fit_surface_points(values, width, height):
    """Robust y(x) line with end handles at the observed x span, not image edges."""
    try:
        xy = np.array([[p["x"], p["y"]] for p in values], dtype=float)
    except (ValueError, KeyError, TypeError) as exc:
        raise MeasurementError("invalid_baseline_points", "基线点需要 x、y 数值。") from exc
    if xy.shape[0] < 2 or xy.shape[0] > 200 or xy.shape[1:] != (2,):
        raise MeasurementError("invalid_baseline_points", "基线需要 2–200 个表面边缘点。")
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [width, height]):
        raise MeasurementError("invalid_baseline_points", "基线点须位于原图内。")
    if np.ptp(xy[:, 0]) < 5:
        raise MeasurementError("invalid_baseline", "基线点横向跨度不足 5 px。")
    center = float(np.mean(xy[:, 0]))
    x = xy[:, 0] - center
    initial = np.polyfit(x, xy[:, 1], 1)[::-1]
    fit = least_squares(
        lambda p: (p[0] + p[1] * x - xy[:, 1]) / np.hypot(1, p[1]),
        initial,
        loss="soft_l1",
        f_scale=0.8,
    )
    residual = fit.fun
    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    keep = np.abs(residual - np.median(residual)) <= max(1.5, 3 * mad)
    if keep.sum() >= 2 and np.ptp(x[keep]) >= 5:
        fit = least_squares(
            lambda p: (p[0] + p[1] * x[keep] - xy[keep, 1]) / np.hypot(1, p[1]),
            fit.x,
            loss="soft_l1",
            f_scale=0.8,
        )
    if not fit.success or abs(fit.x[1]) > 1:
        raise MeasurementError("invalid_baseline", "拟合基线倾角超出 ±45°。")
    lo, hi = np.min(xy[:, 0]), np.max(xy[:, 0])
    baseline = [
        {"x": float(xx), "y": float(fit.x[0] + fit.x[1] * (xx - center))} for xx in (lo, hi)
    ]
    if any(not 0 <= p["y"] < height for p in baseline):
        raise MeasurementError("invalid_baseline", "拟合基线超出图片范围。")
    return {
        "baseline": baseline,
        "baseline_points": [{"x": float(p[0]), "y": float(p[1])} for p in xy],
        "fit_rms_px": float(np.sqrt(np.mean(fit.fun**2))),
        "inlier_count": int(keep.sum()),
    }


def estimate_surface_points(image, seed):
    """Use substrate transition pixels outside the drop silhouette's lateral bounds.

    The seed supplies an approximate height only. The returned fit is based on
    the continuous substrate transition after narrow dark fibers are removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    a, b = sorted(seed, key=lambda p: p["x"])
    slope = (b["y"] - a["y"]) / (b["x"] - a["x"])
    ys, xs = np.mgrid[:h, :w]
    surface_y = a["y"] + slope * (xs - a["x"])
    above = surface_y - ys
    valid = above > max(4, 0.008 * h)
    threshold, _ = cv2.threshold(gray[valid].reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
    mask = np.uint8((gray <= threshold) & valid)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max(30, w * h * 0.0005):
            continue
        py, px = np.where(labels == label)
        if np.min(above[py, px]) > max(18, 0.1 * h):
            continue
        candidates.append((area, label))
    if not candidates:
        raise MeasurementError("surface_unresolved", "未找到接近表面的液滴轮廓。")
    drop = labels == max(candidates)[1]
    projections = drop.sum(axis=0)
    occupied = np.flatnonzero(projections >= max(4, 0.02 * np.max(projections)))
    xleft, xright = float(occupied.min()), float(occupied.max())
    drop_width = xright - xleft
    pad = max(5, drop_width * 0.025)
    reach = max(15, min(0.18 * drop_width, 0.12 * w))
    margin = max(3, 0.04 * w)
    intervals = [
        (max(margin, xleft - reach), xleft - pad),
        (xright + pad, min(w - 1 - margin, xright + reach)),
    ]
    if any(end - start < 5 for start, end in intervals):
        raise MeasurementError("surface_unresolved", "液滴两侧的可见固体表面不足以拟合基线。")
    # A dark fiber can have a stronger gradient than the substrate itself.
    # Grayscale closing is an opening of the dark phase: it removes narrow
    # protrusions while preserving a planar half-space, including tilted ones.
    # Its scale follows the droplet width, never an expected angle or filename.
    support_width = max(25, min(101, round(drop_width * 0.20) | 1))
    # Align the filter with the observed plane. A horizontal filter on a
    # tilted low-contrast surface can retain the bases of slanted fibers.
    center_x = (w - 1) / 2
    aligned = cv2.warpAffine(
        gray, np.array([[1., 0., 0.], [-slope, 1., slope * center_x]]),
        (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    continuous = cv2.morphologyEx(
        aligned, cv2.MORPH_CLOSE, np.ones((1, support_width), np.uint8)
    )
    continuous = cv2.GaussianBlur(continuous, (0, 0), 0.7).astype(float)
    points = []
    band = max(24, int(h * 0.08))
    persistence = max(6, min(16, int(h * 0.02)))
    for start, end in intervals:
        for xx in np.unique(np.rint(np.linspace(start, end, 25)).astype(int)):
            yy = a["y"] + slope * (center_x - a["x"])
            lo, hi = max(1, int(yy) - band), min(h - 2, int(yy) + band)
            if hi - lo < 2 * persistence:
                continue
            profile = continuous[lo : hi + 1, xx]
            background = float(np.median(profile[:persistence]))
            substrate = float(np.median(profile[-persistence:]))
            contrast = background - substrate
            if contrast < 6:
                continue
            level = (background + substrate) / 2
            dark = profile <= level
            candidates = []
            for index in np.flatnonzero((~dark[:-1]) & dark[1:]) + 1:
                below = dark[index : index + persistence]
                if len(below) < persistence or np.mean(below) < 0.9:
                    continue
                # A substrate transition has bright support above it, rather
                # than a second edge inside an already opaque region.
                above = profile[max(0, index - persistence) : index]
                if np.mean(above > level) < 0.6:
                    continue
                candidates.append(int(index))
            if not candidates:
                continue
            index = candidates[0]
            removed = continuous[lo + max(0, index - persistence) : lo + index, xx] - aligned[
                lo + max(0, index - persistence) : lo + index, xx
            ]
            if index >= 4 and len(removed) and np.max(removed) > max(10, 0.25 * contrast):
                # A dense dark fiber bed can shade even the remaining gaps.
                # Refine the contrast against the local exposed background,
                # instead of treating a far brighter background as this edge.
                local_background = float(np.median(profile[max(0, index - 4) : index - 1]))
                local_substrate = float(np.median(profile[index + 3 : index + persistence]))
                if local_background - local_substrate >= 6:
                    local_level = (local_background + local_substrate) / 2
                    for candidate in range(max(1, index - 2), min(len(profile), index + 4)):
                        if profile[candidate - 1] > local_level >= profile[candidate]:
                            index, level = candidate, local_level
                            break
            upper, lower = profile[index - 1], profile[index]
            fraction = (upper - level) / max(upper - lower, 1e-9)
            points.append({"x": float(xx), "y": float(lo + index - 1 + fraction + slope * (xx - center_x))})
    if len(points) < 12:
        raise MeasurementError("surface_unresolved", "固体表面边缘的有效采样不足 12 点。")
    fit = fit_surface_points(points, w, h)
    # Retain the selected substrate observations for editing and export.
    fit["source"] = "automatic"
    fit["method"] = "dense_substrate_v1"
    fit["support_width_px"] = support_width
    return fit
