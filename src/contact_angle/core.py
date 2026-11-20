"""Sessile-drop measurement in a baseline-aligned coordinate frame.

The baseline is an observed input, never inferred by assuming a flat image bottom.
Angles are through the liquid. No silhouette completion or Young-Laplace claims.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.optimize import least_squares

from . import __version__
from .models import (
    AngleRange,
    MeasurementError,
    MeasurementOptions,
    MeasurementResult,
    Point,
    SensitivityReport,
    SensitivityRun,
    SideResult,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


@dataclass
class MeasurementOutput:
    result: MeasurementResult
    annotation_png: bytes


@dataclass
class Frame:
    origin: np.ndarray
    basis: np.ndarray

    @classmethod
    def from_options(cls, options):
        a, b = sorted(options.baseline, key=lambda p: p.x)
        origin = np.array([a.x, a.y])
        along = np.array([b.x - a.x, b.y - a.y])
        along /= np.linalg.norm(along)
        up = np.array([along[1], -along[0]])
        return cls(origin, np.stack([along, up], axis=1))

    def to_local(self, points):
        return (np.asarray(points) - self.origin) @ self.basis

    def to_image(self, points):
        return np.asarray(points) @ self.basis.T + self.origin


def _error(code, text):
    raise MeasurementError(code, text)


def decode_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        _error("image_too_large", "Image exceeds 20 MiB upload limit.")
    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}:
                _error("unsupported_image", "Use PNG, JPEG, TIFF, BMP or WebP.")
            w, h = im.size
            if max(w, h) >= 32767:
                _error("image_too_large", "Each image dimension must be below 32767 pixels.")
            if w * h > MAX_IMAGE_PIXELS or w < 24 or h < 24:
                _error("image_too_large", "Use an image between 24x24 and 12 million pixels.")
            if getattr(im, "n_frames", 1) > 1:
                _error("unsupported_image", "Export one frame before measurement.")
            # Keep original stored pixel coordinates; do not silently rotate EXIF images.
            if im.getexif().get(274, 1) != 1:
                _error(
                    "image_orientation",
                    "Normalize image orientation before selecting baseline coordinates.",
                )
            if im.mode not in {"L", "RGB", "RGBA", "P"}:
                _error("unsupported_image", "Convert image to 8-bit grayscale or RGB first.")
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (255, 255, 255, 255))
            rgb = np.array(Image.alpha_composite(background, rgba).convert("RGB"))
    except MeasurementError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise MeasurementError("invalid_image", "Cannot decode this image.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _point(xy):
    return Point(x=float(xy[0]), y=float(xy[1]))


def _subpixel_contour(xy, gray, offset, threshold, polarity):
    """Locate threshold crossings along image gradients, retaining observed edges."""
    local = xy - offset
    ix = np.rint(local[:, 0]).astype(int)
    iy = np.rint(local[:, 1]).astype(int)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[iy, ix]
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[iy, ix]
    normals = np.c_[gx, gy]
    if polarity == "light":
        normals *= -1
    magnitude = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(magnitude[:, None], 1e-8)
    steps = np.linspace(-1.5, 2.0, 15)
    probes = local[:, None, :] + normals[:, None, :] * steps[None, :, None]
    # remap also limits the destination height to signed 16-bit coordinates.
    source = gray.astype(np.float32)
    values = np.concatenate(
        [
            cv2.remap(
                source,
                chunk[:, :, 0].astype(np.float32),
                chunk[:, :, 1].astype(np.float32),
                cv2.INTER_LINEAR,
            )
            for chunk in np.array_split(probes, max(1, math.ceil(len(probes) / 16000)))
        ]
    )
    signed = values - threshold
    if polarity == "light":
        signed *= -1
    crossing = (signed[:, :-1] <= 0) & (signed[:, 1:] > 0)
    valid = crossing.any(axis=1) & (magnitude > 1)
    k = np.argmax(crossing, axis=1)
    rows = np.arange(len(xy))
    v0, v1 = signed[rows, k], signed[rows, k + 1]
    fraction = -v0 / np.maximum(v1 - v0, 1e-8)
    delta = steps[k] + np.clip(fraction, 0, 1) * (steps[1] - steps[0])
    delta[~valid] = 0
    return xy + normals * delta[:, None]


def _extract(image, options, frame):
    height, width = image.shape[:2]
    for p in options.baseline:
        if not (0 <= p.x < width and 0 <= p.y < height):
            _error("invalid_baseline", "Baseline endpoints must lie inside the original image.")
    x0, y0, rw, rh = 0, 0, width, height
    if options.roi:
        r = options.roi
        x0, y0, rw, rh = r.x, r.y, r.width, r.height
        if x0 + rw > width or y0 + rh > height:
            _error("invalid_roi", "ROI extends outside the image.")
    for r in options.excluded_regions:
        if r.x + r.width > width or r.y + r.height > height:
            _error("invalid_exclusion", "An excluded region extends outside the image.")
    gray = cv2.cvtColor(image[y0 : y0 + rh, x0 : x0 + rw], cv2.COLOR_BGR2GRAY)
    if options.blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), options.blur_sigma)
    yy, xx = np.mgrid[y0 : y0 + rh, x0 : x0 + rw]
    local_height = (xx - frame.origin[0]) * frame.basis[0, 1] + (
        yy - frame.origin[1]
    ) * frame.basis[1, 1]
    above = local_height >= 0.5
    samples = gray[above]
    if len(samples) < 100 or float(np.std(samples)) < 2:
        _error("no_contrast", "No usable droplet contrast above the baseline.")
    if options.threshold is None:
        threshold, _ = cv2.threshold(samples.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
    else:
        threshold = float(options.threshold)
    foreground = gray <= threshold if options.polarity == "dark" else gray > threshold
    mask = np.uint8(foreground & above) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        xy = c[:, 0, :].astype(float) + [x0, y0]
        uv = frame.to_local(xy)
        if np.min(uv[:, 1]) <= 2.5 and np.max(uv[:, 1]) >= 12:
            candidates.append((area, xy, uv))
    if not candidates:
        _error(
            "no_drop",
            "No resolved droplet connected to the baseline; check baseline, ROI and polarity.",
        )
    candidates.sort(key=lambda c: c[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.2:
        _error("ambiguous_drop", "Multiple baseline-connected objects; select a tighter ROI.")
    _, xy, uv = candidates[0]
    if (
        np.any(xy[:, 0] <= x0)
        or np.any(xy[:, 0] >= x0 + rw - 1)
        or np.any(xy[:, 1] <= y0)
        or np.any((xy[:, 1] >= y0 + rh - 1) & (uv[:, 1] > 2.5))
    ):
        _error(
            "clipped_drop", "Droplet touches image/ROI boundary; provide the full visible outline."
        )
    # Subpixel threshold crossings reduce staircase bias in short tangent fits.
    xy = _subpixel_contour(xy, gray, np.array([x0, y0]), threshold, options.polarity)
    uv = frame.to_local(xy)
    # Remove the artificial baseline closing segment, preserving the longest ordered arc.
    good = uv[:, 1] >= options.baseline_margin_px
    if good.all():
        _error("missing_contact", "Contour does not intersect the baseline exclusion band.")
    start = int(np.flatnonzero(~good)[0])
    uv, xy, good = np.roll(uv, -start, axis=0), np.roll(xy, -start, axis=0), np.roll(good, -start)
    indices = np.flatnonzero(good)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    idx = max(runs, key=len)
    if len(idx) < 30:
        _error("short_contour", "Too few contour pixels for a stable fit.")
    uv, xy = uv[idx], xy[idx]
    if uv[0, 0] > uv[-1, 0]:
        uv, xy = uv[::-1].copy(), xy[::-1].copy()
    if uv[-1, 0] - uv[0, 0] < 16:
        _error("small_drop", "Contact width is too small; use a higher-resolution image.")
    usable = np.ones(len(xy), dtype=bool)
    for r in options.excluded_regions:
        usable &= ~(
            (xy[:, 0] >= r.x)
            & (xy[:, 0] < r.x + r.width)
            & (xy[:, 1] >= r.y)
            & (xy[:, 1] < r.y + r.height)
        )
    if np.count_nonzero(usable) < 30:
        _error(
            "insufficient_visible_contour",
            "Too little visible contour outside the excluded regions.",
        )
    return uv, xy, usable, threshold


def _side(angle, contact, tangent, rms, count, frame):
    tangent = tangent / np.linalg.norm(tangent)
    return SideResult(
        angle_deg=float(angle),
        contact_point=_point(frame.to_image(contact)),
        tangent_vector=_point(tangent @ frame.basis.T),
        fit_rms_px=float(rms),
        fit_points=int(count),
    )


def _local_side(uv, usable, frame, options, side, *, with_curve=False):
    if side == "right":
        uv, usable = uv[::-1], usable[::-1]
    visible = np.flatnonzero(usable)
    if len(visible) < 12:
        _error("insufficient_visible_contour", f"Too few visible points for the {side} fit.")
    # Bridge excluded gaps only for the arc-length coordinate; no invented pixels
    # enter the fit. Keep the initial missing length to bound extrapolation.
    first_gap = np.linalg.norm(uv[visible[0]] - uv[0])
    arc = first_gap + np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(uv[visible], axis=0), axis=1))]
    distance = np.interp(np.arange(len(uv)), visible, arc)
    length = max(12.0, distance[-1] * options.fit_fraction)
    chosen = (distance <= length) & usable
    # No filling occlusions: only visible points are used, with bounded extrapolation.
    if chosen.sum() < 12:
        _error(
            "insufficient_visible_contour", f"Too few visible points for the {side} contact fit."
        )
    t = distance[chosen] / length
    points = uv[chosen]
    if t[0] > 0.35:
        _error(
            "contact_occluded",
            f"The {side} contact region is obscured; extrapolation would be excessive.",
        )
    degree = options.polynomial_degree
    design = np.polynomial.polynomial.polyvander(t, degree)
    initial = np.linalg.lstsq(design, points, rcond=None)[0]
    fit = least_squares(
        lambda c: (design @ c.reshape(degree + 1, 2) - points).ravel(),
        initial.ravel(),
        loss="soft_l1",
        f_scale=0.7,
        max_nfev=200,
    )
    if not fit.success:
        _error("fit_failed", f"Local {side} fit did not converge.")
    coefficients = fit.x.reshape(degree + 1, 2)
    roots = np.polynomial.polynomial.polyroots(coefficients[:, 1])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-7 and -0.4 <= r.real <= t[0] + 0.03]
    if not real_roots:
        _error("no_intersection", f"Local {side} fit does not reliably reach the baseline.")
    root = min(real_roots, key=lambda r: abs(r - t[0]))
    if t[0] - root > 0.4:
        _error("contact_occluded", f"The {side} contact point needs excessive extrapolation.")
    derivative = np.polynomial.polynomial.polyval(
        root, coefficients[1:] * np.arange(1, degree + 1)[:, None]
    )
    if derivative[1] <= 0:
        _error("invalid_tangent", f"The {side} tangent points away from the droplet.")
    x = np.polynomial.polynomial.polyval(root, coefficients[:, 0])
    sign = 1 if side == "left" else -1
    angle = math.degrees(math.atan2(derivative[1], sign * derivative[0]))
    if not 5 <= angle <= 175:
        _error(
            "unsupported_angle",
            "Angles outside 5–175 degrees require dedicated imaging and fitting.",
        )
    rms = np.sqrt(np.mean(np.sum((design @ coefficients - points) ** 2, axis=1)))
    result = _side(angle, [x, 0], derivative, rms, len(t), frame)
    if with_curve:
        sample = np.linspace(root, t[-1], 120)
        curve = np.column_stack(
            [np.polynomial.polynomial.polyval(sample, coefficients[:, axis]) for axis in (0, 1)]
        )
        return result, frame.to_image(curve)
    return result


def _circle(uv, usable, frame):
    p = uv[usable]
    # Algebraic estimate, followed by robust geometric least squares in pixels.
    a = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = np.sum(p * p, axis=1)
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = np.sqrt(max(1.0, c + cx * cx + cy * cy))
    fit = least_squares(
        lambda v: np.linalg.norm(p - v[:2], axis=1) - v[2],
        [cx, cy, radius],
        loss="soft_l1",
        f_scale=0.7,
        bounds=([-np.inf, -np.inf, 1], [np.inf, np.inf, np.inf]),
        max_nfev=300,
    )
    if not fit.success:
        _error("fit_failed", "Circle fit did not converge.")
    cx, cy, radius = fit.x
    if abs(cy) >= radius:
        _error("no_intersection", "Fitted circle does not intersect the baseline.")
    half = np.sqrt(radius * radius - cy * cy)
    if half < 8:
        _error("small_drop", "Fitted contact width is too small.")
    rms = np.sqrt(np.mean(fit.fun**2))
    angle = math.degrees(math.acos(-cy / radius))
    if not 5 <= angle <= 175:
        _error(
            "unsupported_angle",
            "Angles outside 5–175 degrees require dedicated imaging and fitting.",
        )
    return (
        _side(angle, [cx - half, 0], [-cy, half], rms, len(p), frame),
        _side(angle, [cx + half, 0], [cy, half], rms, len(p), frame),
    )


def _measure(image, data_hash, options):
    frame = Frame.from_options(options)
    uv, xy, usable, threshold = _extract(image, options, frame)
    warnings = []
    if options.method == "circle":
        left, right = _circle(uv, usable, frame)
        warnings.append(
            "Circle assumes a circular, symmetric silhouette; equal angles are imposed by the model."
        )
    else:
        left = _local_side(uv, usable, frame, options, "left")
        right = _local_side(uv, usable, frame, options, "right")
    if (
        frame.to_local([left.contact_point.x, left.contact_point.y])[0]
        >= frame.to_local([right.contact_point.x, right.contact_point.y])[0]
    ):
        _error("invalid_contacts", "Fitted left and right contact points cross.")
    if max(left.fit_rms_px, right.fit_rms_px) > 1.5:
        warnings.append(
            "Fit residual exceeds 1.5 pixels; inspect roughness, fibres or the silhouette model."
        )
    asymmetry = abs(left.angle_deg - right.angle_deg)
    if asymmetry > 5:
        warnings.append("Left/right angles differ by more than 5 degrees; report them separately.")
    if options.excluded_regions:
        warnings.append(
            "Excluded regions were omitted from fitting; inspect contact extrapolation and remaining contour."
        )
    if min(left.angle_deg, right.angle_deg) < 20 or max(left.angle_deg, right.angle_deg) > 160:
        warnings.append(
            "Extreme contact angles are especially sensitive to image resolution and baseline placement."
        )
    # Cap response geometry while keeping full contour internally for fitting.
    step = max(1, math.ceil(len(xy) / 2000))
    return MeasurementResult(
        algorithm_version=__version__,
        image_sha256=data_hash,
        image_width=image.shape[1],
        image_height=image.shape[0],
        method=options.method,
        parameters=options,
        threshold_used=threshold,
        left=left,
        right=right,
        mean_angle_deg=(left.angle_deg + right.angle_deg) / 2,
        asymmetry_deg=asymmetry,
        contour=[_point(p) for p in xy[::step]],
        status="review" if warnings else "ok",
        warnings=warnings,
    )


def _quality_reason(result):
    if max(result.left.fit_rms_px, result.right.fit_rms_px) > 1.5:
        return "Fit RMS exceeds 1.5 px."
    return None


def _sensitivity(image, data_hash, options, primary):
    # Alternatives are plausible analysis choices, not perturbed physical geometry.
    # All use exactly the supplied baseline, ROI, exclusions and polarity.
    runs = []
    accepted = []
    candidates = [("primary", primary.parameters.model_copy(update={"sensitivity": False}))]
    for degree in (2, 3):
        for fraction in (0.08, 0.12, 0.18):
            candidates.append(
                (
                    f"local_d{degree}_window{fraction:.2f}",
                    options.model_copy(
                        update={
                            "method": "local_polynomial",
                            "polynomial_degree": degree,
                            "fit_fraction": fraction,
                            "sensitivity": False,
                        }
                    ),
                )
            )
    for sigma in (0.5, 1.2):
        candidates.append(
            (
                f"local_smoothing{sigma}",
                options.model_copy(
                    update={
                        "method": "local_polynomial",
                        "blur_sigma": sigma,
                        "sensitivity": False,
                    }
                ),
            )
        )
    candidates.append(
        (
            "circle_shape_check",
            options.model_copy(update={"method": "circle", "sensitivity": False}),
        )
    )
    # Deduplicate choices so the default method is not given extra weight.
    seen = set()
    circles = []
    for name, config in candidates:
        key = (
            config.method,
            config.polynomial_degree if config.method == "local_polynomial" else 0,
            config.fit_fraction if config.method == "local_polynomial" else 0,
            config.blur_sigma,
        )
        if key in seen:
            continue
        seen.add(key)
        params = {
            "polynomial_degree": config.polynomial_degree,
            "fit_fraction": config.fit_fraction,
            "blur_sigma": config.blur_sigma,
            "threshold": config.threshold,
        }
        try:
            result = primary if name == "primary" else _measure(image, data_hash, config)
            reason = _quality_reason(result)
            run = SensitivityRun(
                name=name,
                method=config.method,
                parameters=params,
                status="excluded" if reason else "accepted",
                error=reason,
                left_angle_deg=result.left.angle_deg,
                right_angle_deg=result.right.angle_deg,
                left_delta_deg=result.left.angle_deg - primary.left.angle_deg,
                right_delta_deg=result.right.angle_deg - primary.right.angle_deg,
            )
            if config.method == "circle":
                circles.append(run)
            elif not reason:
                accepted.append(run)
            runs.append(run)
        except MeasurementError as exc:
            runs.append(
                SensitivityRun(
                    name=name,
                    method=config.method,
                    parameters=params,
                    status="failed",
                    error=f"{exc.code}: {exc.message}",
                )
            )
    # Circle is admitted only if it also agrees with independent local sides.
    for run in circles:
        if run.status != "accepted":
            continue
        if not accepted:
            run.status, run.error = (
                "excluded",
                "No admissible independent local fits to check circular assumption.",
            )
            continue
        ml = float(np.median([r.left_angle_deg for r in accepted]))
        mr = float(np.median([r.right_angle_deg for r in accepted]))
        if max(abs(run.left_angle_deg - ml), abs(run.right_angle_deg - mr), abs(ml - mr)) > 5:
            run.status, run.error = (
                "excluded",
                "Circle inconsistent with independent local contact angles.",
            )
        else:
            accepted.append(run)

    def summarize(side, base):
        values = [getattr(r, f"{side}_angle_deg") for r in accepted]
        if not values:
            return None
        low, high = float(min(values)), float(max(values))
        return AngleRange(
            median_deg=float(np.median(values)),
            min_deg=low,
            max_deg=high,
            span_deg=high - low,
            max_abs_delta_deg=max(abs(v - base) for v in values),
        )

    left = summarize("left", primary.left.angle_deg)
    right = summarize("right", primary.right.angle_deg)
    failed = sum(r.status == "failed" for r in runs)
    excluded = sum(r.status == "excluded" for r in runs)
    sensitive = (
        left is None
        or right is None
        or max(left.span_deg, right.span_deg) > options.sensitivity_alert_deg
    )
    report = SensitivityReport(
        left=left,
        right=right,
        failed_runs=failed,
        excluded_runs=excluded,
        accepted_runs=len(accepted),
        sensitive=sensitive,
        alert_deg=options.sensitivity_alert_deg,
        runs=runs,
    )
    primary.sensitivity = report
    if not accepted:
        primary.warnings.append(
            "No admissible method: primary angles are diagnostic only; no aggregate or range can be reported."
        )
    if sensitive and accepted:
        primary.warnings.append(
            "Admissible methods span more than the sensitivity threshold; review the image before choosing a value."
        )
    if failed or excluded or len(accepted) < 3:
        primary.warnings.append(
            "Some methods failed/were excluded, or fewer than three were admissible; inspect the method table."
        )
    if primary.warnings:
        primary.status = "review"


def _annotate(image, result):
    canvas = image.copy()
    xy = np.array([[p.x, p.y] for p in result.contour], dtype=np.int32)
    cv2.polylines(canvas, [xy], False, (30, 190, 30), 1, cv2.LINE_AA)
    a, b = result.parameters.baseline
    cv2.line(
        canvas, (round(a.x), round(a.y)), (round(b.x), round(b.y)), (255, 150, 0), 2, cv2.LINE_AA
    )
    for r in result.parameters.excluded_regions:
        cv2.rectangle(canvas, (r.x, r.y), (r.x + r.width, r.y + r.height), (180, 0, 180), 2)
    length = min(canvas.shape[:2]) * 0.12
    for name, side in (("L", result.left), ("R", result.right)):
        p = np.array([side.contact_point.x, side.contact_point.y])
        v = np.array([side.tangent_vector.x, side.tangent_vector.y])
        cv2.line(
            canvas,
            tuple(np.rint(p).astype(int)),
            tuple(np.rint(p + v * length).astype(int)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, tuple(np.rint(p).astype(int)), 4, (0, 0, 255), -1)
    labels = [
        f"Primary L {result.left.angle_deg:.2f} deg | R {result.right.angle_deg:.2f} deg",
        f"{result.method} | {result.status}",
    ]
    if result.sensitivity and result.sensitivity.accepted_runs == 0:
        labels.append("NO ADMISSIBLE METHOD - primary angles are diagnostic only")
    if result.sensitivity and result.sensitivity.left and result.sensitivity.right:
        s = result.sensitivity
        labels.extend(
            [
                f"Methods L {s.left.min_deg:.2f}-{s.left.max_deg:.2f} | R {s.right.min_deg:.2f}-{s.right.max_deg:.2f}",
                f"Median L {s.left.median_deg:.2f} | R {s.right.median_deg:.2f} (not confidence bounds)",
            ]
        )
    font_scale = min(0.55, max(0.25, canvas.shape[1] / 1400))
    for i, label in enumerate(labels):
        pos = (8, 20 + i * 22)
        cv2.putText(
            canvas,
            label,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), 1, cv2.LINE_AA
        )
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        _error("encode_failed", "Could not encode annotated image.")
    return encoded.tobytes()


def measure_bytes(data: bytes, options: MeasurementOptions) -> MeasurementOutput:
    image = decode_image(data)
    digest = hashlib.sha256(data).hexdigest()
    requested_failure = None
    try:
        result = _measure(image, digest, options)
    except MeasurementError as original:
        if not options.sensitivity or original.code not in {
            "fit_failed",
            "insufficient_visible_contour",
            "no_intersection",
            "contact_occluded",
            "invalid_tangent",
            "unsupported_angle",
            "invalid_contacts",
        }:
            raise
        requested_failure = f"{original.code}: {original.message}"
        # If the requested fit fails, try other local models with the same geometry.
        # Record requested and actual parameters; never silently change the baseline.
        result = None
        for degree, fraction in ((2, 0.18), (2, 0.08), (3, 0.18), (3, 0.12)):
            alternative = options.model_copy(
                update={
                    "method": "local_polynomial",
                    "polynomial_degree": degree,
                    "fit_fraction": fraction,
                }
            )
            try:
                result = _measure(image, digest, alternative)
                result.requested_parameters = options
                result.warnings.append(
                    f"Requested fit failed ({original.code}); primary uses recorded fallback parameters."
                )
                result.status = "review"
                break
            except MeasurementError:
                continue
        if result is None:
            raise
    if options.sensitivity:
        _sensitivity(image, digest, options, result)
        if requested_failure:
            result.sensitivity.runs.insert(
                0,
                SensitivityRun(
                    name="requested_fit_failed",
                    method=options.method,
                    parameters=options.model_dump(mode="json"),
                    status="failed",
                    error=requested_failure,
                ),
            )
            result.sensitivity.failed_runs += 1
    return MeasurementOutput(result=result, annotation_png=_annotate(image, result))
