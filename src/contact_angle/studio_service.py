"""Adapt measured geometry into one studio result and reproducible export bundle."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .core import Frame, _extract, _local_side, decode_image, measure_bytes
from .manual_arc import MIN_MANUAL_POINTS
from .models import MeasurementError, MeasurementOptions
from .studio_annotation import annotate
from .studio_store import write_json

RANGE_NOTE = "同一基线下合理方法或拟合窗口之间的范围，不是置信区间，也不包含共同基线误差。"


def points(values):
    return [{"x": float(x), "y": float(y)} for x, y in np.asarray(values).reshape(-1, 2)]


def frame_for(baseline):
    return Frame.from_options(MeasurementOptions(baseline=baseline, sensitivity=False))


def tangent(angle, side, frame):
    theta = math.radians(angle)
    vector = np.array([(1 if side == "left" else -1) * math.cos(theta), math.sin(theta)])
    xy = vector @ frame.basis.T
    return {"x": float(xy[0]), "y": float(xy[1])}


def pendant_orientation(image, polarity, orientation):
    if orientation != "auto":
        return orientation == "pendant"
    from .pendant_surface import detect_pendant

    return detect_pendant(pendant_gray(image, polarity))


def pendant_gray(image, polarity):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.bitwise_not(gray) if polarity == "light" else gray


def manual_orientation(options):
    orientation = options.get("orientation", "auto")
    contour = options.get("contour_points") or []
    surface = options.get("baseline_points") or options.get("baseline") or []
    if (
        orientation != "auto"
        or options.get("mode") != "manual"
        or len(contour) < MIN_MANUAL_POINTS
        or len(surface) < 2
    ):
        return orientation
    # In low-contrast images the user's marked profile can identify the liquid side
    # even if the silhouette cannot be separated from the photographic background.
    xy = np.asarray([(p["x"], p["y"]) for p in contour])
    anchors = sorted((p["x"], p["y"]) for p in surface)
    delta = xy[:, 1] - np.interp(xy[:, 0], *np.asarray(anchors).T)
    if np.median(delta) > 4:
        return "pendant"
    if np.median(delta) < -4:
        return "sessile"
    return orientation


def estimate_surface_points(
    image, polarity="dark", algorithm="general", orientation="auto", marks=None
):
    if pendant_orientation(image, polarity, orientation):
        from .pendant_surface import estimate_pendant_surface

        surface = estimate_pendant_surface(pendant_gray(image, polarity), points=marks)
        return {**surface, "method": "pendant_surface_v1", "orientation": "pendant"}
    if marks:
        from .surface_points import fit_surface_points

        return {
            **fit_surface_points(marks, image.shape[1], image.shape[0]),
            "orientation": "sessile",
            "source": "manual",
        }
    from .general import estimate_surface
    from .surface_points import estimate_surface_points as sample_surface
    from .surface_points import gradient_surface_seed

    work = cv2.bitwise_not(image) if polarity == "light" else image
    try:
        if algorithm == "fibrous":
            from .fibrous import estimate_baseline

            line, _ = estimate_baseline(work)
        else:
            line = estimate_surface(work)
    except MeasurementError:
        line = gradient_surface_seed(work)
    validate_line(image, line)
    return {**sample_surface(work, line), "orientation": "sessile"}


def estimate(image, polarity="dark", algorithm="general"):
    return estimate_surface_points(image, polarity, algorithm)["baseline"]


def validate_line(image, line):
    opt = MeasurementOptions(baseline=line, sensitivity=False)
    h, w = image.shape[:2]
    if any(not (0 <= p.x < w and 0 <= p.y < h) for p in opt.baseline):
        raise MeasurementError("invalid_baseline", "基线两点必须位于原图内。")
    return opt


def range_from(values, primary):
    if len(values) < 2:
        return None
    return {
        "min_deg": float(min(values)),
        "max_deg": float(max(values)),
        "span_deg": float(max(values) - min(values)),
        "median_deg": float(np.median(values)),
        "max_abs_delta_deg": float(max(abs(v - primary) for v in values)),
    }


def supplement_sensitivity(data, baseline, result, exclusions=None):
    """Compare supported local alternatives with the selected YL result; do not move baseline."""
    runs = [
        {
            "name": "young_laplace_primary",
            "method": "young_laplace",
            "status": "accepted",
            "left_angle_deg": result["left"]["angle_deg"],
            "right_angle_deg": result["right"]["angle_deg"],
            "error": None,
        }
    ]
    try:
        alternative = measure_bytes(
            data,
            MeasurementOptions(
                baseline=baseline, sensitivity=True, excluded_regions=exclusions or []
            ),
        ).result
        if alternative.sensitivity:
            runs.extend(r.model_dump(mode="json") for r in alternative.sensitivity.runs)
    except (MeasurementError, ValueError) as exc:
        runs.append({"name": "local_alternatives", "status": "failed", "error": str(exc)})
    accepted = [r for r in runs if r["status"] == "accepted"]
    report = {
        "interpretation": RANGE_NOTE,
        "accepted_runs": len(accepted),
        "failed_runs": sum(r["status"] == "failed" for r in runs),
        "excluded_runs": sum(r["status"] == "excluded" for r in runs),
        "runs": runs,
    }
    for side in ("left", "right"):
        values = [
            r[f"{side}_angle_deg"] for r in accepted if r.get(f"{side}_angle_deg") is not None
        ]
        report[side] = range_from(values, result[side]["angle_deg"])
    return report


def _general(data, image, baseline, options):
    from .general import extract_photo, measure_general_bytes
    from .young_laplace import solve_profile

    raw = measure_general_bytes(data, baseline)
    uv, frame, _ = extract_photo(
        image,
        baseline,
        clearance=raw["clearance_px"],
        normalize={"False": False, "True": True, "rim": "rim"}[raw["background_mode"]],
    )
    geometry = {
        "contours": [
            {"side": "both", "points": points(frame.to_image(uv[:: max(1, len(uv) // 1800)]))}
        ],
        "fits": [],
        "selected_points": {"left": [], "right": []},
    }
    result = {
        "method_label": "Young–Laplace 左右独立轮廓拟合",
        "geometry": geometry,
        "warnings": [],
        "status": "ok",
    }
    center = (uv[:, 0].min() + uv[:, 0].max()) / 2
    for side in ("left", "right"):
        s = dict(raw[side])
        if not isinstance(s["contact_point"], dict):
            s["contact_point"] = {
                "x": float(s["contact_point"][0]),
                "y": float(s["contact_point"][1]),
            }
        p = s["parameters"]
        profile = solve_profile(float(p["Bo"]))
        height = np.linspace(0, min(uv[:, 1].max() * 0.98, p["A"] * 0.998), 220)
        zz = (p["A"] - height) / p["R0"]
        radius = p["R0"] * profile.radius_of_z(zz)
        xx = p["cx"] + (-1 if side == "left" else 1) * radius
        fit = np.c_[xx, height]
        fit = fit[np.isfinite(fit).all(axis=1)]
        geometry["fits"].append({"side": side, "points": points(frame.to_image(fit))})
        s["tangent_vector"] = tangent(s["angle_deg"], side, frame)
        s["fit_points"] = int(
            np.sum((uv[:, 0] < center) if side == "left" else (uv[:, 0] > center))
        )
        result[side] = s
    result["sensitivity"] = (
        supplement_sensitivity(data, baseline, result) if options["sensitivity"] else None
    )
    return result


def _fibrous(image, baseline, options):
    from .fibrous import analyse_fibrous

    analysis = analyse_fibrous(image, baseline)
    uv, frame, _ = next(iter(analysis["contexts"].values()))
    result = {
        "method_label": "纤维表面 · 可见边缘多模型汇总",
        "status": "review",
        "warnings": [
            "接触点由可见侧边外推至基线。",
            "曲线显示代表模型，角度为通过筛选的模型族等权汇总。",
        ],
        "geometry": {
            "contours": [
                {"side": "both", "points": points(frame.to_image(uv[:: max(1, len(uv) // 1500)]))}
            ],
            "fits": [],
            "selected_points": {"left": [], "right": []},
        },
    }
    report = {
        "interpretation": RANGE_NOTE,
        "runs": [],
        "accepted_runs": 0,
        "failed_runs": 0,
        "excluded_runs": 0,
    }
    for side in ("left", "right"):
        s = analysis[side]
        candidates = [
            r for r in analysis["runs"] if r["side"] == side and r["status"] == "accepted"
        ]
        contact = float(
            np.median(
                [
                    np.median([r["contact"] for r in candidates if r["method"] == family])
                    for family in s["families"]
                ]
            )
        )
        primary = s["primary"]
        p = frame.to_image([contact, 0])
        result[side] = {
            "angle_deg": s["angle"],
            "contact_point": {"x": float(p[0]), "y": float(p[1])},
            "tangent_vector": tangent(s["angle"], side, frame),
            "fit_rms_px": primary["rms"],
            "fit_points": len(primary["p"]),
        }
        hh = np.linspace(0, float(primary["p"][:, 1].max()), 180)
        model = primary.get("model_object")
        if model and "predict" in model:
            curve = frame.to_image(np.c_[model["predict"](hh), hh])
            # A fitted ellipse can end just below the highest observed pixel.
            # Display only its supported finite arc, without completing a contour.
            curve = curve[np.isfinite(curve).all(axis=1)]
            result["geometry"]["fits"].append({"side": side, "points": points(curve)})
        report[side] = (
            range_from([r["angle"] for r in candidates], s["angle"])
            if len(s["families"]) >= 2
            else None
        )
        if len(s["families"]) < 2:
            result["warnings"].append(
                f"{'左' if side == 'left' else '右'}侧仅一种模型族合格，敏感性范围证据不足。"
            )
    for r in analysis["runs"]:
        report["runs"].append({k: v for k, v in r.items() if k not in ("p", "model_object")})
        report[
            {"accepted": "accepted_runs", "failed": "failed_runs", "excluded": "excluded_runs"}[
                r["status"]
            ]
        ] += 1
    result["sensitivity"] = report if options["sensitivity"] else None
    return result


def _local(data, baseline, options):
    outcome = measure_bytes(
        data,
        MeasurementOptions(
            baseline=baseline,
            sensitivity=options["sensitivity"],
            excluded_regions=options["excluded_regions"],
        ),
    )
    raw = outcome.result.model_dump(mode="json")
    config = outcome.result.parameters
    frame = Frame.from_options(config)
    uv, _, usable, _ = _extract(decode_image(data), config, frame)
    curves = []
    for side in ("left", "right"):
        _, curve = _local_side(uv, usable, frame, config, side, with_curve=True)
        curves.append({"side": side, "points": points(curve)})
    return {
        "left": raw["left"],
        "right": raw["right"],
        "sensitivity": raw["sensitivity"],
        "status": raw["status"],
        "warnings": raw["warnings"],
        "method_label": "局部参数曲线与几何模型对照",
        "geometry": {
            "contours": [{"side": "both", "points": raw["contour"]}],
            "fits": curves,
            "selected_points": {"left": [], "right": []},
        },
    }


def _crop(image, baseline, options):
    roi = options.get("roi")
    if not roi:
        return image, baseline, options, np.array([0.0, 0.0])
    x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
    if w < 24 or h < 24 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise MeasurementError("invalid_roi", "分析区域需在原图内，且至少24×24像素。")
    a, b = sorted(baseline, key=lambda p: p["x"])
    slope = (b["y"] - a["y"]) / (b["x"] - a["x"])
    line = [
        {"x": float(xx), "y": float(a["y"] + slope * (xx + x - a["x"]) - y)} for xx in (1, w - 2)
    ]
    cropped = image[y : y + h, x : x + w].copy()
    validate_line(cropped, line)
    revised = dict(options)
    regions = []
    for r in options["excluded_regions"]:
        x0 = max(x, r["x"])
        y0 = max(y, r["y"])
        x1 = min(x + w, r["x"] + r["width"])
        y1 = min(y + h, r["y"] + r["height"])
        if x1 > x0 and y1 > y0:
            regions.append({"x": x0 - x, "y": y0 - y, "width": x1 - x0, "height": y1 - y0})
    revised["excluded_regions"] = regions
    return cropped, line, revised, np.array([float(x), float(y)])


def measure_studio(data, options):
    image = decode_image(data)
    options = dict(options)
    baseline = options.get("baseline")
    baseline_points = options.get("baseline_points") or []
    pendant = pendant_orientation(image, options["polarity"], manual_orientation(options))
    if baseline_points and not pendant:
        from .surface_points import fit_surface_points

        fitted = fit_surface_points(baseline_points, image.shape[1], image.shape[0])
        baseline = fitted["baseline"]
        options["baseline"] = baseline
    actual_source = (
        options.get("baseline_source", "manual") if baseline or baseline_points else "automatic"
    )
    if pendant:
        from .pendant_studio import pendant_job

        result = pendant_job(pendant_gray(image, options["polarity"]), options)
        baseline, baseline_points = result["baseline"], result["baseline_points"]
    elif options["mode"] == "manual":
        from .manual import measure_manual

        if not baseline:
            raise MeasurementError("baseline_required", "手动测量需要至少两个固体表面边缘点。")
        result = measure_manual(image, options)
        result["algorithm"] = "manual"
        if options.get("roi") or options.get("excluded_regions"):
            result["warnings"].append(
                "手动拟合仅使用已标记的边缘点；分析框与自动排除区域不改变这些点。"
            )
            result["status"] = "review"
    else:
        if not baseline:
            detected = estimate_surface_points(
                image, options["polarity"], options["algorithm"], "sessile"
            )
            baseline, baseline_points = detected["baseline"], detected["baseline_points"]
        validate_line(image, baseline)
        if options["excluded_regions"] and options["algorithm"] != "local":
            raise MeasurementError(
                "unsupported_exclusions",
                "遮挡排除区域仅适用于局部曲线模式；其他模式请改用手动边缘点避开遮挡。",
            )
        work = cv2.bitwise_not(image) if options["polarity"] == "light" else image
        work, line, config, offset = _crop(work, baseline, options)
        encoded = cv2.imencode(".png", work)[1].tobytes()
        if options["algorithm"] == "general":
            result = _general(encoded, work, line, config)
        elif options["algorithm"] == "fibrous":
            result = _fibrous(work, line, config)
        else:
            result = _local(encoded, line, config)
        for side in ("left", "right"):
            p = result[side]["contact_point"]
            p["x"] += float(offset[0])
            p["y"] += float(offset[1])
        for group in ("contours", "fits"):
            for curve in result["geometry"][group]:
                for p in curve["points"]:
                    p["x"] += float(offset[0])
                    p["y"] += float(offset[1])
        result["algorithm"] = options["algorithm"]
    left, right = [result[side]["contact_point"] for side in ("left", "right")]
    if pendant:
        crossed = left["x"] >= right["x"]
    else:
        local = Frame.from_options(validate_line(image, baseline))
        crossed = (
            local.to_local([left["x"], left["y"]])[0] >= local.to_local([right["x"], right["y"]])[0]
        )
    if crossed:
        raise MeasurementError("invalid_contacts", "拟合的左右接触点交叉，请检查基线和所选边缘。")
    for side in ("left", "right"):
        p = result[side]["contact_point"]
        if not (0 <= p["x"] < image.shape[1] and 0 <= p["y"] < image.shape[0]):
            raise MeasurementError(
                "invalid_contact", "拟合接触点落在图片之外，请扩大图像或重新选择可见边缘。"
            )
    result.update(
        mode=options["mode"],
        orientation="pendant" if pendant else "sessile",
        baseline=baseline,
        baseline_points=baseline_points,
        image_width=image.shape[1],
        image_height=image.shape[0],
        mean_angle_deg=(result["left"]["angle_deg"] + result["right"]["angle_deg"]) / 2,
        asymmetry_deg=abs(result["left"]["angle_deg"] - result["right"]["angle_deg"]),
    )
    report = result.get("sensitivity")
    if report:
        if not pendant:
            report["interpretation"] = RANGE_NOTE
        if any(report.get(s) is None for s in ("left", "right")):
            result["warnings"].append("可用候选不足，部分敏感性范围无法估计。")
            result["status"] = "review"
        if any(report.get(s) and report[s]["span_deg"] > 3 for s in ("left", "right")):
            result["warnings"].append("合理拟合方法的角度跨度超过 3°。")
            result["status"] = "review"
    if result["asymmetry_deg"] > 5:
        result["warnings"].append("左右接触角相差超过 5°。")
        result["status"] = "review"
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    result["provenance"] = {
        "algorithm_version": __version__,
        "baseline_source": actual_source,
        "rotation_degrees": 0,
        "coordinate_system": "original",
    }
    return result, image


def execute_job(store, job):
    image_info = store.image(job["image_id"])
    data = (store.directory("images", job["image_id"]) / "original").read_bytes()
    result, image = measure_studio(data, job["options"])
    identifier = job["id"]
    result["filename"] = image_info["filename"]
    result["provenance"]["image_sha256"] = image_info["sha256"]
    base = f"/api/results/{identifier}"
    result["annotation_url"] = base + "/annotation.png"
    result["downloads"] = {
        "json_url": base + "/result.json",
        "csv_url": base + "/results.csv",
        "png_url": base + "/annotation.png",
        "zip_url": base + "/bundle.zip",
    }
    directory = store.directory("results", identifier)
    (directory / "annotation.png").write_bytes(annotate(image, result))
    write_json(directory / "result.json", {**result, "options": job["options"]})
    row = {
        "filename": image_info["filename"],
        "image_sha256": image_info["sha256"],
        "mode": result["mode"],
        "algorithm": result["algorithm"],
        "orientation": result["orientation"],
        "status": result["status"],
        "left_angle_deg": result["left"]["angle_deg"],
        "right_angle_deg": result["right"]["angle_deg"],
        "mean_angle_deg": result["mean_angle_deg"],
        "warnings": "; ".join(result["warnings"]),
    }
    for side in ("left", "right"):
        interval = (result.get("sensitivity") or {}).get(side) or {}
        for name in ("min_deg", "max_deg", "span_deg"):
            row[f"{side}_sensitivity_{name}"] = interval.get(name, "")
    # Prevent spreadsheet formula interpretation of user-controlled filename/text.
    safe = {
        k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
        for k, v in row.items()
    }
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(safe)
    (directory / "results.csv").write_text(handle.getvalue(), encoding="utf-8-sig")
    with zipfile.ZipFile(
        directory / "bundle.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ("result.json", "results.csv", "annotation.png"):
            archive.write(directory / name, name)
        archive.writestr(
            "points-and-options.json",
            json.dumps(
                {
                    "format": "contact-angle-studio-coordinates-v1",
                    "image": {
                        "filename": image_info["filename"],
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "sha256": image_info["sha256"],
                    },
                    "options": job["options"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        with Image.open(io.BytesIO(data)) as original:
            suffix = {"PNG": "png", "JPEG": "jpg", "TIFF": "tif", "BMP": "bmp", "WEBP": "webp"}[
                original.format
            ]
        archive.writestr("original." + suffix, data)
        archive.writestr(
            "说明.txt",
            "坐标相对原始图片。PNG下方附统计栏，原图坐标未移动。\n方法敏感性范围不是置信区间。原始图片未修改。\n",
        )
    return result
