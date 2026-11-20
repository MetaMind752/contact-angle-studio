"""General side-view photo extraction and physically constrained drop fitting."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .core import Frame, decode_image
from .models import MeasurementError, MeasurementOptions
from .young_laplace import fit_young_laplace


def local_contrast_edge(xy, gray):
    ix = np.rint(xy[:, 0]).astype(int)
    iy = np.rint(xy[:, 1]).astype(int)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[iy, ix]
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[iy, ix]
    normals = np.c_[gx, gy]
    mag = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(mag[:, None], 1e-8)
    steps = np.linspace(-4, 4, 65)
    q = xy[:, None, :] + normals[:, None, :] * steps[None, :, None]
    vals = cv2.remap(
        gray.astype(np.float32),
        q[:, :, 0].astype(np.float32),
        q[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )
    low = np.median(vals[:, :8], axis=1)
    high = np.median(vals[:, -8:], axis=1)
    target = (low + high) / 2
    signed = vals - target[:, None]
    cross = (signed[:, :-1] <= 0) & (signed[:, 1:] > 0)
    valid = cross.any(axis=1) & (high - low > 10) & (mag > 1)
    # Take the crossing closest to original edge, preserving outer-edge identity.
    dist = np.abs((steps[:-1] + steps[1:]) / 2)
    indices = np.argmin(np.where(cross, dist[None, :], 1e6), axis=1)
    rows = np.arange(len(xy))
    v0 = signed[rows, indices]
    v1 = signed[rows, indices + 1]
    delta = steps[indices] - v0 / np.maximum(v1 - v0, 1e-8) * (steps[1] - steps[0])
    delta[~valid] = 0
    return xy + delta[:, None] * normals


def normalize_background(gray, above):
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    x = (xx - w / 2) / w
    y = (yy - h / 2) / h
    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    sample = above & (gray > 20)
    indices = np.flatnonzero(sample.ravel())
    indices = indices[:: max(1, len(indices) // 5000)]
    a = design.reshape(-1, 6)[indices]
    b = gray.ravel()[indices].astype(float)
    p = np.linalg.lstsq(a, b, rcond=None)[0]
    for _ in range(4):
        expected = a @ p
        keep = b > 0.65 * np.maximum(expected, 20)
        if keep.sum() < 50:
            break
        fit = least_squares(
            lambda p, aa=a[keep], bb=b[keep]: aa @ p - bb, p, loss="soft_l1", f_scale=8
        )
        p = fit.x
    bg = np.maximum(20, design @ p)
    return np.uint8(np.clip(gray.astype(float) / bg * 235, 0, 255))


def gradient_peak_edge(xy, gray):
    """Choose rising outer-edge derivative peak nearest a rough contour seed."""
    ix = np.clip(np.rint(xy[:, 0]).astype(int), 1, gray.shape[1] - 2)
    iy = np.clip(np.rint(xy[:, 1]).astype(int), 1, gray.shape[0] - 2)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[iy, ix]
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[iy, ix]
    normal = np.c_[gx, gy]
    mag = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(mag[:, None], 1e-8)
    steps = np.linspace(-3, 3, 97)
    q = xy[:, None, :] + normal[:, None, :] * steps[None, :, None]
    vals = cv2.remap(
        gray.astype(np.float32),
        q[:, :, 0].astype(np.float32),
        q[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )
    vals = gaussian_filter1d(vals, 3.2, axis=1, mode="nearest")
    d = np.gradient(vals, steps, axis=1)
    # Positive derivative follows dark-to-light direction of the outer seed.
    peak = (d[:, 1:-1] >= d[:, :-2]) & (d[:, 1:-1] > d[:, 2:]) & (d[:, 1:-1] > 3)
    dpeak = d[:, 1:-1]
    maximum = np.max(dpeak, axis=1)
    eligible = peak & (dpeak > 0.5 * maximum[:, None])
    distance = abs(steps[1:-1])[None, :]
    k = np.argmin(np.where(eligible, distance, 1e8), axis=1) + 1
    row = np.arange(len(xy))
    ym = d[row, k - 1]
    y0 = d[row, k]
    yp = d[row, k + 1]
    denom = ym - 2 * y0 + yp
    offset = np.clip(0.5 * (ym - yp) / np.where(abs(denom) > 1e-8, denom, 1), -0.8, 0.8) * (
        steps[1] - steps[0]
    )
    shift = steps[k] + offset
    valid = eligible.any(axis=1) & (mag > 1)
    shift[~valid] = 0
    return xy + shift[:, None] * normal


def estimate_surface(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1)
    h, w = gray.shape
    gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    xs = np.r_[np.arange(int(0.07 * w), int(0.25 * w)), np.arange(int(0.75 * w), int(0.93 * w))]
    lo, hi = int(0.25 * h), int(0.95 * h)
    # Consensus across lateral columns favors a straight material interface over
    # the drop, needle, image vignetting or isolated surface debris.
    window = cv2.dilate(gradient, (np.ones((5, 1), np.uint8)))
    score = np.median(window[lo:hi, xs], axis=1)
    best = lo + int(np.argmax(score))
    band = gradient[max(lo, best - 8) : min(hi, best + 9), xs]
    yy = np.argmax(band, axis=0) + max(lo, best - 8)
    weights = np.max(band, axis=0)
    keep = weights > max(2, float(np.quantile(weights, 0.15)))
    x, y = xs[keep], yy[keep]
    if len(x) < 20:
        raise MeasurementError("surface_unresolved", "No lateral image evidence for a baseline.")
    fit = least_squares(
        lambda v: v[0] + v[1] * (x - w / 2) - y, [np.median(y), 0], loss="soft_l1", f_scale=1
    )
    c, m = fit.x
    if abs(m) > 0.1:
        raise MeasurementError(
            "surface_tilt", "Rotate the image before automatic baseline detection."
        )
    return [{"x": float(xx), "y": float(c + m * (xx - w / 2))} for xx in (5, w - 6)]


def extract_photo(image, baseline, clearance=2, threshold=None, normalize=False):
    opt = MeasurementOptions(baseline=baseline, sensitivity=False)
    frame = Frame.from_options(opt)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]]
    hh = (xx - frame.origin[0]) * frame.basis[0, 1] + (yy - frame.origin[1]) * frame.basis[1, 1]
    above = hh > clearance
    segmentation_gray = normalize_background(gray, above) if normalize else gray
    if normalize == "rim":
        k = max(15, int(min(gray.shape) * 0.07) | 1)
        background = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        segmentation_gray = 255 - cv2.subtract(background, gray)
    if threshold is None:
        threshold, _ = cv2.threshold(
            segmentation_gray[above].reshape(-1, 1), 0, 255, cv2.THRESH_OTSU
        )
    mask = np.uint8((segmentation_gray < threshold) & above) * 255
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    choices = []
    for c in cs:
        area = cv2.contourArea(c)
        p = c[:, 0, :].astype(float)
        uv = frame.to_local(p)
        if area < 50 or np.min(uv[:, 1]) > clearance + 3:
            continue
        if np.any(p[:, 0] < 2) or np.any(p[:, 0] > gray.shape[1] - 3):
            continue
        choices.append((area, c))
    if not choices:
        raise MeasurementError("no_drop", "No separated visible droplet above the baseline.")
    c = max(choices, key=lambda x: x[0])[1]
    _x, _y, w, h = cv2.boundingRect(c)
    hollow = cv2.contourArea(c) / (w * h) < 0.3
    silhouette = np.zeros_like(mask)
    cv2.drawContours(silhouette, [c], -1, 255, cv2.FILLED)
    for row in np.flatnonzero(silhouette.any(axis=1)):
        xs = np.flatnonzero(silhouette[row])
        silhouette[row, xs[0] : xs[-1] + 1] = 255
    outer, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw = max(outer, key=cv2.contourArea)[:, 0, :].astype(float)
    p = (gradient_peak_edge if hollow else local_contrast_edge)(raw, gray)
    uv = frame.to_local(p)
    good = frame.to_local(raw)[:, 1] > clearance + 1
    if good.all():
        raise MeasurementError("no_contact", "No contact-region contour.")
    start = np.flatnonzero(~good)[0]
    uv = np.roll(uv, -start, axis=0)
    good = np.roll(good, -start)
    ids = np.flatnonzero(good)
    ids = max(np.split(ids, np.where(np.diff(ids) > 1)[0] + 1), key=len)
    uv = uv[ids]
    if uv[0, 0] > uv[-1, 0]:
        uv = uv[::-1].copy()
    if len(uv) < 30:
        raise MeasurementError("short_contour", "Not enough droplet pixels.")
    return uv, frame, float(threshold)


def measure_general_bytes(data, baseline=None):
    image = decode_image(data)
    baseline = baseline if baseline is not None else estimate_surface(image)
    attempts = []
    best = None
    for mode in (False, True, "rim"):
        for clearance in (2, 5, 8, 12, 16, 24, 32):
            try:
                uv, frame, threshold = extract_photo(
                    image, baseline, clearance=clearance, normalize=mode
                )
                center = (uv[:, 0].min() + uv[:, 0].max()) / 2
                height = float(uv[:, 1].max())
                if clearance > 0.2 * height:
                    raise MeasurementError(
                        "occluded_contact", "Too much of the contact region is obscured."
                    )
                results = {}
                for side in ("left", "right"):
                    p = uv[(uv[:, 0] < center) if side == "left" else (uv[:, 0] > center)]
                    p = p[p[:, 1] < height * 0.98]
                    try:
                        fit = fit_young_laplace(p, side=side, max_nfev=80, multistart=False)
                    except ValueError as e:
                        if not any(
                            term in str(e)
                            for term in (
                                "did not converge",
                                "No converged",
                                "outside fitted",
                                "does not reach",
                            )
                        ):
                            raise
                        fit = fit_young_laplace(p, side=side, max_nfev=200, multistart=True)
                    results[side] = {
                        "angle_deg": float(fit["angle"]),
                        "fit_rms_px": float(fit["rms"]),
                        "contact_point": frame.to_image([fit["contact"], 0]).tolist(),
                        "parameters": fit["parameters"],
                        "diagnostics": fit["diagnostics"],
                    }
                error = max(results[side]["fit_rms_px"] for side in ("left", "right"))
                result = {
                    "left": results["left"],
                    "right": results["right"],
                    "baseline": baseline,
                    "threshold": threshold,
                    "clearance_px": clearance,
                    "background_mode": str(mode),
                    "method": "young_laplace_side",
                }
                if best is None or error < best[0]:
                    best = (error, result)
                if error < 0.3:
                    return result
                attempts.append(f"Fit RMS{error:.3f}px exceeds first-pass cutoff.")
            except (MeasurementError, ValueError, RuntimeError) as e:
                attempts.append(str(e))
        if best is not None and best[0] < 0.5:
            return best[1]
    if best is not None and best[0] < 1.5:
        best[1]["warnings"] = ["Fit residual requires review."]
        return best[1]
    raise MeasurementError("unresolved_profile", "No reliable full side profile: " + attempts[-1])
