"""Robust visible side-arc analysis for fibrous sessile-drop images.

No human angles or sample identifiers are inputs. Contact points are extrapolated
from observed liquid edges to a supplied baseline. All excluded points are kept
for auditing; no generated edge pixels enter a fit.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .core import Frame, _subpixel_contour
from .models import MeasurementError, MeasurementOptions


def estimate_baseline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 160 or h < 120:
        raise MeasurementError(
            "small_image", "Automatic fibrous-surface analysis needs at least 160x120 pixels."
        )
    smooth = cv2.GaussianBlur(gray, (0, 0), 2)
    points = []
    for x in np.r_[
        np.arange(int(w * 0.03), int(w * 0.2), 2), np.arange(int(w * 0.8), int(w * 0.97), 2)
    ]:
        average = np.convolve((smooth[:, x] < 110).astype(float), np.ones(11) / 11, "same")
        ids = (
            np.where((average[1:] >= 0.85) & (average[:-1] < 0.85) & (np.arange(1, h) > h * 0.38))[
                0
            ]
            + 1
        )
        if len(ids):
            points.append((x, ids[0] - 4))
    if len(points) < 30:
        raise MeasurementError("baseline_unresolved", "Insufficient lateral surface pixels.")
    points = np.array(points)
    x, y = points.T
    fit = least_squares(
        lambda v: v[0] + v[1] * (x - w / 2) - y, [np.median(y), 0], loss="soft_l1", f_scale=2
    )
    c, m = fit.x
    line = [{"x": float(x), "y": float(c + m * (x - w / 2))} for x in (50, w - 50)]
    return line, float(np.median(abs(fit.fun)))


def extract_visible(image, baseline, threshold=None, clearance=8):
    options = MeasurementOptions(baseline=baseline, sensitivity=False)
    if any(not (0 <= p.x < image.shape[1] and 0 <= p.y < image.shape[0]) for p in options.baseline):
        raise MeasurementError("invalid_baseline", "Baseline endpoints must be inside the image.")
    frame = Frame.from_options(options)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]]
    height = (xx - frame.origin[0]) * frame.basis[0, 1] + (yy - frame.origin[1]) * frame.basis[1, 1]
    if threshold is None:
        threshold, _ = cv2.threshold(
            gray[height > clearance].reshape(-1, 1), 0, 255, cv2.THRESH_OTSU
        )
    mask = np.uint8((gray <= threshold) & (height > clearance)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise MeasurementError("no_drop", "No visible contour above the surface exclusion band.")
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    if (
        np.any(contour[:, 0] <= 1)
        or np.any(contour[:, 0] >= image.shape[1] - 2)
        or np.any(contour[:, 1] <= 1)
    ):
        raise MeasurementError("clipped_drop", "Visible droplet touches the image edge.")
    xy = _subpixel_contour(contour, gray, np.array([0, 0]), threshold, "dark")
    uv = frame.to_local(xy)
    good = uv[:, 1] > clearance + 2
    if good.all():
        raise MeasurementError("no_drop", "Droplet does not extend to the surface band.")
    start = np.flatnonzero(~good)[0]
    good = np.roll(good, -start)
    uv = np.roll(uv, -start, axis=0)
    ids = np.flatnonzero(good)
    ids = max(np.split(ids, np.where(np.diff(ids) > 1)[0] + 1), key=len)
    uv = uv[ids]
    if len(uv) < 50:
        raise MeasurementError("short_contour", "Insufficient visible outline.")
    if uv[0, 0] > uv[-1, 0]:
        uv = uv[::-1].copy()
    return uv, frame, float(threshold)


def circle_residual(parameters, p):
    return np.linalg.norm(p - parameters[:2], axis=1) - parameters[2]


def fit_circle(p, trim=True):
    if len(p) < 20:
        raise MeasurementError("short_arc", "At least 20 observed arc pixels required.")
    # Seed away from the ambiguous lowest contact pixels. These points are used
    # only for initialization; every supported visible inlier can enter the fit.
    seed = p[p[:, 1] >= np.min(p[:, 1]) + 0.3 * np.ptp(p[:, 1])]
    aa = np.c_[2 * seed[:, 0], 2 * seed[:, 1], np.ones(len(seed))]
    cx, cy, c = np.linalg.lstsq(aa, np.sum(seed * seed, axis=1), rcond=None)[0]
    radius = np.sqrt(max(1, c + cx * cx + cy * cy))
    fit = least_squares(
        lambda v: circle_residual(v, seed),
        [cx, cy, radius],
        loss="soft_l1",
        f_scale=0.35,
        max_nfev=300,
        bounds=([-1e5, -1e5, 1], [1e5, 1e5, 1e5]),
    )
    keep = np.abs(circle_residual(fit.x, p)) <= 1.5
    if keep.sum() < 20:
        raise MeasurementError(
            "contaminated_arc", "Insufficient pixels consistent with the upper side arc."
        )
    if trim:
        for _ in range(3):
            error = circle_residual(fit.x, p)
            scale = max(0.15, 1.4826 * np.median(abs(error[keep] - np.median(error[keep]))))
            candidate = abs(error) <= max(1.0, 3.5 * scale)
            if candidate.sum() < 20:
                break
            keep = candidate
            fit = least_squares(
                lambda v, selected=p[keep]: circle_residual(v, selected),
                fit.x,
                loss="soft_l1",
                f_scale=0.3,
                max_nfev=200,
                bounds=([-1e5, -1e5, 1], [1e5, 1e5, 1e5]),
            )
    if not fit.success:
        raise MeasurementError("fit_failed", "Local circle did not converge.")
    return fit.x, keep, float(np.sqrt(np.mean(circle_residual(fit.x, p[keep]) ** 2)))


def side_points(uv, side, low, high):
    mid = (np.min(uv[:, 0]) + np.max(uv[:, 0])) / 2
    good = (
        (uv[:, 1] >= low)
        & (uv[:, 1] <= high)
        & ((uv[:, 0] < mid) if side == "left" else (uv[:, 0] > mid))
    )
    p = uv[good]
    return p[np.argsort(p[:, 1])]


def fit_side(uv, side, low, high, model="circle"):
    p = side_points(uv, side, low, high)
    if len(p) < 20:
        raise MeasurementError("short_arc", "Not enough observed side pixels.")
    if model == "circle":
        params, keep, rms = fit_circle(p)
        cx, cy, radius = params
        if abs(cy) >= radius:
            raise MeasurementError(
                "no_intersection", "Side circle does not meet the supplied baseline."
            )
        half = np.sqrt(radius * radius - cy * cy)
        contact = cx - half if side == "left" else cx + half
        slope = (-cy / half) * (1 if side == "left" else -1)
        prediction = lambda y: (
            cx + (-1 if side == "left" else 1) * np.sqrt(np.maximum(0, radius**2 - (y - cy) ** 2))
        )
    else:
        scale = high
        t = p[:, 1] / scale
        mat = np.polynomial.polynomial.polyvander(t, 3)
        initial = np.linalg.lstsq(mat, p[:, 0], rcond=None)[0]
        fit = least_squares(lambda c: mat @ c - p[:, 0], initial, loss="soft_l1", f_scale=0.3)
        residual = mat @ fit.x - p[:, 0]
        keep = abs(residual) < max(1, 3.5 * 1.4826 * np.median(abs(residual - np.median(residual))))
        if keep.sum() < 20:
            raise MeasurementError("short_arc", "Not enough inlier pixels.")
        fit = least_squares(
            lambda c: mat[keep] @ c - p[keep, 0], fit.x, loss="soft_l1", f_scale=0.3
        )
        slope = fit.x[1] / scale
        contact = fit.x[0]
        rms = float(np.sqrt(np.mean((mat[keep] @ fit.x - p[keep, 0]) ** 2)))
        prediction = lambda y: np.polynomial.polynomial.polyval(np.asarray(y) / scale, fit.x)
    angle = float(np.degrees(np.arctan2(1, slope if side == "left" else -slope)))
    if not 20 < angle < 175:
        raise MeasurementError("invalid_tangent", "Side tangent outside supported range.")
    # No low-residual contact may jump across the opposite side or far from observed side.
    if not uv[:, 0].min() < contact < uv[:, 0].max():
        raise MeasurementError("invalid_contact", "Contact outside visible droplet width.")
    if keep.sum() < 0.6 * len(p):
        raise MeasurementError("contaminated_arc", "More than 40% of local arc points rejected.")
    if np.ptp(p[keep, 1]) < 0.55 * (high - low):
        raise MeasurementError(
            "insufficient_support", "Inliers cover too little of the selected vertical interval."
        )
    if p[keep, 1].min() > high * 0.4:
        raise MeasurementError(
            "excessive_extrapolation", "Nearest reliable contour is too far from the baseline."
        )
    return {
        "angle": angle,
        "contact": float(contact),
        "slope": float(slope),
        "rms": rms,
        "points": int(keep.sum()),
        "rejected": int((~keep).sum()),
        "observed_low": float(p[keep, 1].min()),
        "high": float(high),
        "model": model,
        "p": p,
        "keep": keep,
        "prediction": prediction,
    }


def _polynomial_model(p, degree=3):
    if len(p) < 20:
        raise MeasurementError("short_arc", "Too few side pixels for polynomial model.")
    scale = max(p[:, 1])
    mat = np.polynomial.polynomial.polyvander(p[:, 1] / scale, degree)
    fit = least_squares(
        lambda c: mat @ c - p[:, 0],
        np.linalg.lstsq(mat, p[:, 0], rcond=None)[0],
        loss="soft_l1",
        f_scale=0.3,
    )
    residual = mat @ fit.x - p[:, 0]
    keep = abs(residual) <= max(1, 3.5 * 1.4826 * np.median(abs(residual - np.median(residual))))
    if keep.sum() < 20:
        raise MeasurementError("short_arc", "Too few inlier pixels.")
    fit = least_squares(lambda c: mat[keep] @ c - p[keep, 0], fit.x, loss="soft_l1", f_scale=0.3)
    return {
        "predict": lambda y: np.polynomial.polynomial.polyval(np.asarray(y) / scale, fit.x),
        "slope": fit.x[1] / scale,
        "contact": fit.x[0],
        "keep": keep,
        "rms": float(np.sqrt(np.mean((mat[keep] @ fit.x - p[keep, 0]) ** 2))),
    }


def _circle_model(p, side):
    params, keep, rms = fit_circle(p)
    cx, cy, radius = params
    if abs(cy) >= radius:
        raise MeasurementError("no_intersection", "Circle misses baseline.")
    sign = -1 if side == "left" else 1
    half = np.sqrt(radius * radius - cy * cy)
    return {
        "predict": lambda y: (
            cx + sign * np.sqrt(np.maximum(0, radius * radius - (np.asarray(y) - cy) ** 2))
        ),
        "slope": sign * cy / half,
        "contact": cx + sign * half,
        "keep": keep,
        "rms": rms,
    }


def _model(p, side, method):
    return _circle_model(p, side) if method == "local_circle" else _polynomial_model(p)


def fit_candidates(uv, side, low, ellipse_model=None):
    height = float(uv[:, 1].max())
    runs = []
    # Compare model families on the same observed side, holding out the lowest
    # 20% of its vertical interval for a prediction check. No human angles enter.
    for fraction in (0.3, 0.4, 0.5):
        p = side_points(uv, side, low, height * fraction)
        # Independent global model is only a coarse spike guard. If not a good
        # fit to the shape it cannot reject genuine points.
        if ellipse_model is not None and ellipse_model.get("guard", False):
            predicted = ellipse_model[side]["prediction"](p[:, 1])
            p = p[abs(p[:, 0] - predicted) < max(2.0, 6 * ellipse_model["rms"])]
        for method in ("local_circle", "height_polynomial"):
            run = {"side": side, "method": method, "fraction": fraction, "status": "failed"}
            try:
                if len(p) < 30:
                    raise MeasurementError("short_arc", "Fewer than 30 visible pixels.")
                model = _model(p, side, method)
                keep = model["keep"]
                q = p[keep]
                if len(q) < 20 or keep.sum() < len(p) * 0.6:
                    raise MeasurementError("contaminated_arc", "Insufficient inlier support.")
                if q[:, 1].min() > (height * fraction) * 0.4:
                    raise MeasurementError(
                        "excessive_extrapolation", "Contact-region gap is too large."
                    )
                split = np.quantile(q[:, 1], 0.22)
                train = q[q[:, 1] > split]
                test = q[q[:, 1] <= split]
                trained = _model(train, side, method)
                cv = float(np.sqrt(np.mean((trained["predict"](test[:, 1]) - test[:, 0]) ** 2)))
                angle = float(
                    np.degrees(np.arctan2(1, model["slope"] if side == "left" else -model["slope"]))
                )
                if not 20 < angle < 175:
                    raise MeasurementError("invalid_tangent", "Angle outside supported range.")
                if not uv[:, 0].min() < model["contact"] < uv[:, 0].max():
                    raise MeasurementError("invalid_contact", "Contact outside droplet width.")
                if (
                    not np.isfinite([model["rms"], cv, angle, model["contact"]]).all()
                    or model["rms"] > 1.0
                    or cv > 1.0
                ):
                    raise MeasurementError(
                        "model_mismatch", "Model fails residual or held-out edge prediction check."
                    )
                run.update(
                    status="eligible",
                    angle=angle,
                    contact=float(model["contact"]),
                    slope=float(model["slope"]),
                    rms=model["rms"],
                    cv=cv,
                    points=len(q),
                    rejected=int(len(p) - len(q)),
                    observed_low=float(q[:, 1].min()),
                    model_object=model,
                    p=q,
                )
            except (MeasurementError, ValueError, np.linalg.LinAlgError) as e:
                run["reason"] = str(e)
            runs.append(run)
    return runs


def analyse_fibrous(image, baseline=None):
    """Per-side model selection by held-out edge prediction, independent of labels."""
    from .ellipse import fit_ellipse

    if baseline is None:
        baseline, baseline_rms = estimate_baseline(image)
    else:
        baseline_rms = None
    runs = []
    contexts = {}
    for clearance in (12, 20, 28):
        uv, frame, threshold = extract_visible(image, baseline, clearance=clearance)
        contexts[clearance] = (uv, frame, threshold)
        ell = None
        try:
            ell = fit_ellipse(uv)
            ell["rms"] = ell["left"]["rms"]
            ell["guard"] = ell["rms"] < 0.5 and ell["left"]["rejected"] < 0.05 * len(uv)
        except (MeasurementError, ValueError):
            pass
        for side in ("left", "right"):
            candidates = fit_candidates(uv, side, clearance + 2, ell)
            for r in candidates:
                r.update(clearance=clearance, threshold=threshold)
            runs.extend(candidates)
            if ell is not None:
                item = ell[side]
                run = {
                    "side": side,
                    "method": "ellipse",
                    "fraction": 1.0,
                    "clearance": clearance,
                    "threshold": threshold,
                    "status": "failed",
                }
                try:
                    height = float(uv[:, 1].max())
                    test = side_points(uv, side, clearance + 2, height * 0.22)
                    train = uv[uv[:, 1] > height * 0.22]
                    held = fit_ellipse(train)
                    cv = float(
                        np.sqrt(np.mean((held[side]["prediction"](test[:, 1]) - test[:, 0]) ** 2))
                    )
                    if (
                        not np.isfinite([item["rms"], cv, item["angle"]]).all()
                        or item["rms"] > 1
                        or cv > 1
                    ):
                        raise MeasurementError(
                            "model_mismatch",
                            "Ellipse fails whole-arc residual or withheld lower-arc prediction.",
                        )
                    run.update(
                        status="eligible",
                        angle=item["angle"],
                        contact=item["contact"],
                        slope=item["slope"],
                        rms=item["rms"],
                        cv=cv,
                        points=item["points"],
                        rejected=item["rejected"],
                        observed_low=item["observed_low"],
                        model_object={"predict": item["prediction"]},
                        p=uv,
                    )
                except (MeasurementError, ValueError) as e:
                    run["reason"] = str(e)
                runs.append(run)
    sides = {}
    for side in ("left", "right"):
        eligible = [r for r in runs if r["side"] == side and r["status"] == "eligible"]
        if not eligible:
            raise MeasurementError(
                "no_admissible_model", f"No side model passes held-out prediction for {side}."
            )
        best = min(r["cv"] for r in eligible)
        # 0.15 px is a pixel-level prediction tolerance, never an angle-agreement gate.
        chosen = [r for r in eligible if r["cv"] <= best + 0.15]
        for r in eligible:
            r["status"] = "accepted" if r in chosen else "excluded"
            if r["status"] == "excluded":
                r["reason"] = "Held-out lower-edge prediction exceeds best candidate by >0.15 px."
        # Equal family weights avoid duplicating the effect of nearby fit windows.
        families = {}
        for method in sorted({r["method"] for r in chosen}):
            families[method] = float(
                np.median([r["angle"] for r in chosen if r["method"] == method])
            )
        angle = float(np.median(list(families.values())))
        primary = min(chosen, key=lambda r: abs(r["angle"] - angle))
        values = [r["angle"] for r in chosen]
        sides[side] = {
            "angle": angle,
            "primary": primary,
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
            "families": families,
            "eligible_min": min(r["angle"] for r in eligible),
            "eligible_max": max(r["angle"] for r in eligible),
        }
    return {
        "left": sides["left"],
        "right": sides["right"],
        "baseline": baseline,
        "baseline_residual_px": baseline_rms,
        "runs": runs,
        "contexts": contexts,
    }


def measure_fibrous_bytes(data, baseline=None):
    """Return serializable results and a PNG; no comparison labels are accepted."""
    import hashlib

    from .core import decode_image

    image = decode_image(data)
    analysis = analyse_fibrous(image, baseline)
    result = {
        "algorithm_version": "0.2.0-fibrous",
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "baseline": analysis["baseline"],
        "baseline_residual_px": analysis["baseline_residual_px"],
        "status": "review",
        "sensitivity_definition": "min/max across models passing residual and held-out edge prediction; fixed baseline; not a confidence interval",
        "protocol": {
            "clearances_px": [12, 20, 28],
            "height_fractions": [0.3, 0.4, 0.5],
            "models": ["local_circle", "height_polynomial", "ellipse"],
            "pixel_cv_tolerance": 0.15,
            "max_fit_rms_px": 1,
            "max_cv_rms_px": 1,
            "primary": "equal family median of accepted candidate angles",
        },
        "runs": [],
    }
    canvas = image.copy()
    frame = next(iter(analysis["contexts"].values()))[1]
    for clearance, (uv, _, _) in analysis["contexts"].items():
        xy = frame.to_image(uv)
        if clearance == 20:
            cv2.polylines(canvas, [np.rint(xy).astype(np.int32)], False, (130, 130, 130), 1)
    line = analysis["baseline"]
    a, b = [tuple(np.rint([p["x"], p["y"]]).astype(int)) for p in line]
    cv2.line(canvas, a, b, (255, 160, 0), 2, cv2.LINE_AA)
    for side in ("left", "right"):
        s = analysis[side]
        accepted = [r for r in analysis["runs"] if r["side"] == side and r["status"] == "accepted"]
        # Same family weighting for location as for angle.
        contacts = [
            float(np.median([r["contact"] for r in accepted if r["method"] == family]))
            for family in s["families"]
        ]
        contact = float(np.median(contacts))
        angle = s["angle"]
        local_contact = np.array([contact, 0])
        p = frame.to_image(local_contact)
        t = np.array(
            [(1 if side == "left" else -1) * np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
        )
        tangent = t @ frame.basis.T
        primary = s["primary"]
        for r in accepted:
            q = frame.to_image(r["p"])
            cv2.polylines(canvas, [np.rint(q).astype(np.int32)], False, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.rint(p).astype(int)), 4, (0, 0, 255), -1)
        cv2.line(
            canvas,
            tuple(np.rint(p).astype(int)),
            tuple(np.rint(p + 65 * tangent).astype(int)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        result[side] = {
            "angle_deg": angle,
            "contact_point": {"x": float(p[0]), "y": float(p[1])},
            "min_deg": s["min"],
            "max_deg": s["max"],
            "span_deg": s["span"],
            "all_eligible_min_deg": s["eligible_min"],
            "all_eligible_max_deg": s["eligible_max"],
            "family_angles": s["families"],
            "accepted_runs": len(accepted),
            "representative_method": primary["method"],
            "representative_fit_rms_px": primary["rms"],
            "representative_cv_rms_px": primary["cv"],
        }
    for r in analysis["runs"]:
        result["runs"].append({k: v for k, v in r.items() if k not in ("model_object", "p")})
    result["mean_angle_deg"] = (result["left"]["angle_deg"] + result["right"]["angle_deg"]) / 2
    result["warnings"] = [
        "Contact points are extrapolated to the supplied/estimated baseline; inspect the annotated outline.",
        "Sensitivity holds baseline fixed; shared baseline error is not bounded by this range.",
    ]
    for side in ("left", "right"):
        if len(result[side]["family_angles"]) < 2:
            result["warnings"].append(
                f"{side}: only one model family passed; a narrow range can understate model uncertainty."
            )
    labels = [
        f"Ensemble L {result['left']['angle_deg']:.2f} | R {result['right']['angle_deg']:.2f} deg",
        f"L range {result['left']['min_deg']:.2f}-{result['left']['max_deg']:.2f} | R {result['right']['min_deg']:.2f}-{result['right']['max_deg']:.2f}",
        "Fixed baseline | model-selection range, not confidence bounds",
    ]
    for i, text in enumerate(labels):
        pos = (10, 22 + 24 * i)
        cv2.putText(
            canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA
        )
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    success, png = cv2.imencode(".png", canvas)
    if not success:
        raise MeasurementError("encode_failed", "Cannot encode annotation.")
    return result, png.tobytes()
