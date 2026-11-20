"""Infer the lower face of a solid above a drop, in original image coordinates.

Outside each contact the air wedge is connected to the surrounding background.
Inside it the next exposed interface is the *bottom of the drop*. Distinguishing
those boundaries before fitting keeps the dense interior of a curved solid, the
enclosed highlight of a drop and isolated hairs out of the surface model.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import gaussian_filter1d, median_filter

from .models import MeasurementError
from .surface_points import fit_surface_points


def _xy(x, y):
    return [{"x": float(i), "y": float(j)} for i, j in zip(x, y)]


def _mad(a):
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def _poly(x, y, degree, origin, scale):
    """Huber regression on scaled polynomial coordinates; equal weight per column."""
    u = (np.asarray(x) - origin) / scale
    mat = np.polynomial.polynomial.polyvander(u, degree)
    weight = np.ones(len(x))
    for _ in range(6):
        beta = np.linalg.lstsq(mat * weight[:, None], y * weight, rcond=None)[0]
        residual = y - mat @ beta
        limit = max(0.7, 1.35 * _mad(residual))
        weight = np.sqrt(np.minimum(1.0, limit / np.maximum(np.abs(residual), 1e-6)))
    return np.polynomial.Polynomial(beta, domain=[origin - scale, origin + scale])


def _geometry(gray):
    if gray.ndim != 2 or min(gray.shape) < 24:
        raise MeasurementError("surface_unresolved", "悬滴分析需要至少 24×24 像素的灰度图。")
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    h, w = smooth.shape
    if np.ptp(smooth) < 7:
        raise MeasurementError("surface_unresolved", "图中没有足够对比度的悬滴。")
    thresh, mask = cv2.threshold(smooth, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    wide = (
        np.where(
            (stats[1:, cv2.CC_STAT_WIDTH] >= 0.81 * w)
            & (stats[1:, cv2.CC_STAT_AREA] >= 0.018 * w * h)
        )[0]
        + 1
    )
    if not len(wide):
        raise MeasurementError("no_pendant", "未找到横跨液滴两侧的上方基体。")
    lab = wide[np.argmax(stats[wide, cv2.CC_STAT_AREA])]
    solid = labels == lab
    present = solid.any(axis=0)
    # Lowermost point of the *same* dark solid+liquid component finds the maximum
    # lateral span of the protrusion, even if its wet contact is much narrower.
    x = np.arange(w)
    y = h - 1 - np.argmax(solid[::-1], axis=0)
    ys = median_filter(y, size=2 * max(1, round(w * 0.0025)) + 1, mode="nearest")
    lo, hi = x[present][[0, -1]]
    outside = present & ((x < lo + 0.115 * w) | (x > hi - 0.115 * w))
    origin, scale = (lo + hi) / 2, max(1.0, (hi - lo) / 2)
    chord = _poly(x[outside], ys[outside], 1, origin, scale)
    delta = ys - chord(x)
    delta[~present] = -h
    peak = int(np.argmax(delta))
    depth = float(delta[peak])
    base_threshold = max(7.0, 0.043 * h)
    if not (depth > base_threshold and y[peak] < h - 1 and lo < peak < hi):
        raise MeasurementError("no_pendant", "上表面下方没有独立的液滴轮廓。")
    for degree in (1, 2):
        # First exclude the body of the drop using the side-to-side chord. Refit
        # the bowed substrate to exposed columns, then locate the true overhang.
        active = delta > max(5, (0.42 if degree == 1 else 0.22) * depth)
        left = np.flatnonzero(~active[:peak])
        right = np.flatnonzero(~active[peak:])
        if not len(left) or not len(right):
            raise MeasurementError("no_pendant", "液滴两侧缺少可见基体。")
        a, b = int(left[-1] + 1), int(peak + right[0] - 1)
        if degree == 1:
            pad = max(3.0, (b - a) * 0.055)
            valid = present & ((x < a - pad) | (x > b + pad))
            if valid.sum() < 8:
                raise MeasurementError("no_pendant", "液滴两侧缺少可见基体。")
            seed = _poly(x[valid], ys[valid], 2, origin, scale)
            delta = ys - seed(x)
            delta[~present] = -h
            peak = int(np.argmax(delta))
            depth = float(delta[peak])
    d = b - a
    if (
        d < max(8, 0.026 * w)
        or d > 0.85 * w
        or a - lo < max(3, 0.011 * w)
        or hi - b < max(3, 0.011 * w)
        or depth < base_threshold
        or depth / d < 0.15
        or depth / d > 2.35
    ):
        raise MeasurementError("no_pendant", "上表面下方没有足够宽且连续的液滴。")
    # A broad smooth arc should have a real air gap underneath. Reject flat
    # solids and bottom-attached sessile silhouettes before routing to this fit.
    bg_test = mask[int(min(h - 1, y[peak] + max(3, h * 0.015))) :, a : b + 1]
    if not bg_test.size or np.mean(bg_test == 0) < 0.65:
        raise MeasurementError("no_pendant", "悬滴下方没有可分辨的空气区域。")
    return smooth, thresh, seed, a, b, present


def detect_pendant(image) -> bool:
    """True only for a broad overhead substrate with a localized liquid protrusion."""
    try:
        _geometry(np.asarray(image))
    except MeasurementError:
        return False
    return True


def _air_edge(smooth, threshold, radius, seed, a, b, present):
    h, w = smooth.shape
    x = np.arange(w)
    d = b - a
    # Close thin dark filaments in grayscale before segmentation. At broad
    # interfaces this small disk leaves the intensity transition in place.
    k = 2 * radius + 1
    clean = cv2.morphologyEx(
        smooth, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    air = (clean > threshold).astype("uint8")
    _, lab, stats, _ = cv2.connectedComponentsWithStats(air, 8)
    floor = np.unique(lab[-1])
    floor = floor[floor > 0]
    if not len(floor):
        raise MeasurementError("surface_unresolved", "未找到悬滴下方的外界区域。")
    ambient = lab == floor[np.argmax(stats[floor, cv2.CC_STAT_AREA])]
    # A thin substrate can leave an additional *upper* air band joining the main
    # air at an image edge. Start the search below its upper face, near our seed.
    tol = max(6.0, 0.085 * d)
    ambient &= np.arange(h)[:, None] >= np.maximum(1, seed(x) - tol)
    row = np.argmax(ambient, axis=0)
    prev = np.maximum(0, row - 1)
    v0 = clean[prev, x].astype(float)
    v1 = clean[row, x].astype(float)
    edge = prev + np.clip((threshold - v0) / np.maximum(1, v1 - v0), 0, 1)
    # Require an actual dark-to-light crossing and a short depth of external
    # air, rather than a glint, a pinhole in the solid or the far side of a drop.
    gap = max(2, round(0.014 * d))
    air_y = np.minimum(h - 1, row + gap)
    keep = (
        present
        & ambient.any(axis=0)
        & (v0 <= threshold)
        & ambient[air_y, x]
        & (np.abs(edge - seed(x)) < tol)
    )
    return edge, keep


def _flanks(edge, keep, a, b):
    d = b - a
    w = len(edge)
    x = np.arange(w)
    # The visible surface at an overhang can continue several dozen pixels
    # inward past the largest drop diameter, up to the air wedge's narrow tip.
    groups = []
    for left in (True, False):
        if left:
            potential = x[(x >= a - 0.62 * d) & (x <= a + 0.44 * d)]
        else:
            potential = x[(x >= b - 0.44 * d) & (x <= b + 0.62 * d)][::-1]
        if not len(potential):
            raise MeasurementError("surface_unresolved", "接触区附近没有可见表面。")
        # Stop advancing once a contiguous stretch of the external surface is
        # hidden; do not resume on a detached highlight inside the footprint.
        gap = max(4, round(d * 0.035))
        bad = 0
        seen = False
        accepted = []
        for j in potential:
            if keep[j]:
                seen = True
                bad = 0
                accepted.append(j)
            elif seen:
                bad += 1
                if bad >= gap and (j >= a - 0.14 * d if left else j <= b + 0.14 * d):
                    break
        if len(accepted) < 4:
            raise MeasurementError("surface_unresolved", "接触区的一侧缺少足够的固体外缘。")
        groups.append(np.asarray(sorted(accepted)))
    return groups


def _fitted(edge, groups, d, window, h):
    # A quadratic estimated from BOTH flanks captures the overall substrate
    # curvature. Local linear residuals correct a change of slope near one
    # contact; the unobserved segment is joined with continuous local tangents.
    l, r = groups
    xl, xr = float(l[-1]), float(r[0])
    loc = (xl + xr) / 2
    scale = max(5.0, d * 0.5)
    xl_all = np.r_[l, r]
    global_fit = _poly(xl_all, edge[xl_all], 2, loc, scale)
    xs = []
    offsets = []
    slopes = []
    for arr, q in ((l, xl), (r, xr)):
        near = arr[np.abs(arr - q) <= max(8, d * window)]
        if len(near) < 4:
            near = arr
        residual = edge[near] - global_fit(near)
        local = _poly(near, residual, 1, q, scale)
        # Angular scatter due to fine, irregular hairs should not extrapolate
        # across an entire drop. Pool the slope with the two-flank estimate.
        tilt = float(local.deriv()(q))
        scatter = max(0.45, _mad(residual - local(near)))
        slope_evidence = abs(tilt) * max(1, np.ptp(near))
        shrink = slope_evidence / (slope_evidence + 3 * scatter)
        tilt *= shrink
        offsets.append(float(local(q)))
        slopes.append(tilt)
        xs.append(q)
    bridge = CubicHermiteSpline(xs, offsets, slopes)
    # Retain the actual observed edge away from the contact neighborhoods. A
    # tangent estimated near a contact should not drift progressively into
    # the material a hundred pixels further out, where pixels are available.
    visible = []
    for arr in (l, r):
        xi = np.arange(arr[0], arr[-1] + 1)
        actual = np.interp(xi, arr, edge[arr])
        local = median_filter(actual, size=2 * max(1, round(0.01 * d)) + 1, mode="nearest")
        visible.append((xi, gaussian_filter1d(local, sigma=max(0.8, 0.018 * d), mode="nearest")))

    def curve(query):
        q = np.asarray(query)
        cor = bridge(np.clip(q, xl, xr))
        cor = np.where(q < xl, offsets[0] + slopes[0] * (q - xl), cor)
        cor = np.where(q > xr, offsets[1] + slopes[1] * (q - xr), cor)
        yy = global_fit(q) + cor
        for (xi, obs), anchor, sign in zip(visible, (xl, xr), (-1.0, 1.0)):
            # Smoothstep blends the measured wing into its extrapolation while
            # preserving the local tangent exactly at either end of the gap.
            u = np.clip(sign * (q - anchor) / max(3, 0.14 * d), 0, 1)
            blend = u * u * (3 - 2 * u)
            yy = (1 - blend) * yy + blend * np.interp(q, xi, obs)
        return np.clip(yy, 0, h - 1)

    return curve


def estimate_pendant_surface(image, points=None) -> dict:
    """Returns ordered image-space surface observations and the gap interpolation.

    ``footprint`` brackets the *visible lateral drop extent*, not the two contact
    coordinates. Manual surface marks take precedence even on an empty image.
    """
    gray = np.asarray(image)
    h, w = gray.shape
    if points is not None and len(points):
        output = fit_surface_points(points, w, h)
        marks = output["baseline_points"]
        x = np.array([p["x"] for p in marks])
        y = np.array([p["y"] for p in marks])
        a, b = x.min(), x.max()
        unique = np.unique(x)
        # Two distinct marks specify a line. Three or more marks with lateral
        # coverage determine a curved face without using any automatic pixels.
        deg = (
            2
            if len(unique) >= 3 and np.any((x > a + 0.12 * (b - a)) & (x < b - 0.12 * (b - a)))
            else 1
        )
        model = _poly(x, y, deg, (a + b) / 2, (b - a) / 2)
        try:
            _, _, _, fa, fb, _ = _geometry(gray)
            bounds = "image"
        except MeasurementError:
            fa, fb = int(a), int(b)
            bounds = "manual_span"
        q = np.unique(np.r_[np.linspace(min(a, fa), max(b, fb), 161), x])
        err = y - model(x)
        return dict(
            output,
            method="pendant_surface_v1",
            source="manual",
            baseline=_xy([a, b], model([a, b])),
            surface_curve=_xy(q, np.clip(model(q), 0, h - 1)),
            footprint=[fa, fb],
            footprint_source=bounds,
            uncertainty_px=max(0.5, _mad(err)),
            surface_uncertainty_px=max(0.5, _mad(err)),
            fit_rms_px=float(np.sqrt(np.mean(err**2))),
            surface_variants=[],
        )

    smooth, threshold, seed, a, b, present = _geometry(gray)
    d = b - a
    radius = max(1, int(np.floor(0.011 * d + 0.5)))
    edge, keep = _air_edge(smooth, threshold, radius, seed, a, b, present)
    groups = _flanks(edge, keep, a, b)
    l, r = groups
    curve = _fitted(edge, groups, d, 0.29, h)
    # Marks remain *measured* subpixel crossings. Thin the returned set by local
    # medians, so the returned handles and the JSON remain readable/reproducible.
    markx = []
    marky = []
    for side in (l, r):
        for chunk in np.array_split(side, min(16, max(2, len(side) // 5))):
            i = chunk[len(chunk) // 2]
            markx.append(i)
            marky.append(np.median(edge[chunk]))
    q = np.unique(np.r_[np.linspace(l[0], r[-1], 161), l[-1], r[0]])
    near = np.r_[l[l >= l[-1] - 0.29 * d], r[r <= r[0] + 0.29 * d]]
    residual = edge[near] - curve(near)
    spread = max(0.5, _mad(residual))
    variants = []
    for name, window, rad in [
        ("short_flanks", 0.19, max(1, round(radius * 0.65))),
        ("long_flanks", 0.45, max(1, round(radius * 1.4))),
    ]:
        try:
            v_edge, v_keep = _air_edge(smooth, threshold, rad, seed, a, b, present)
            v_groups = _flanks(v_edge, v_keep, a, b)
            v_curve = _fitted(v_edge, v_groups, d, window, h)
        except MeasurementError:
            continue
        variants.append({"name": name, "surface_curve": _xy(q, v_curve(q))})
    # The two compatibility handles sit on observed shoulders near the drop;
    # their chord is not the local surface tangent used to compute either angle.
    end = np.array(
        [l[np.argmin(abs(l - (l[-1] - 0.13 * d)))], r[np.argmin(abs(r - (r[0] + 0.13 * d)))]]
    )
    return {
        "method": "pendant_surface_v1",
        "source": "automatic",
        "baseline": _xy(end, curve(end)),
        "baseline_points": _xy(markx, marky),
        "surface_curve": _xy(q, curve(q)),
        "footprint": [int(a), int(b)],
        "footprint_source": "image",
        "uncertainty_px": spread,
        "surface_uncertainty_px": spread,
        "fit_rms_px": float(np.sqrt(np.mean(residual**2))),
        "inlier_count": len(near),
        "observed_spans": [[int(l[0]), int(l[-1])], [int(r[0]), int(r[-1])]],
        "surface_variants": variants,
    }
