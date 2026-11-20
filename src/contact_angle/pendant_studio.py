"""Adapt inverted-drop measurements to the studio's original-pixel coordinate contract."""

from .models import MeasurementError
from .pendant_fit import fit_pendant
from .pendant_surface import estimate_pendant_surface


def translated(points, dx, dy):
    return [{"x": p["x"] + dx, "y": p["y"] + dy} for p in points]


def pendant_job(gray, options):
    height, width = gray.shape
    manual = options["mode"] == "manual"
    marks = None
    if options.get("baseline_source") == "manual":
        marks = options.get("baseline_points") or options.get("baseline")
    contours = options.get("contour_points") if manual else None
    if manual and not contours:
        raise MeasurementError("contour_required", "请沿悬滴边缘选取至少 5 个点。")

    ex = list(options.get("excluded_regions") or []) if not manual else []
    roi = options.get("roi") if not manual else None
    x, y, x1, y1 = 0, 0, width, height
    if roi:
        x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
        x1, y1 = x + w, y + h
        if w < 24 or h < 24 or x1 > width or y1 > height:
            raise MeasurementError("invalid_roi", "分析区域需在原图内，且至少24×24像素。")
        if marks:
            # The user can measure in an ROI using surface observations outside it.
            # Keep the original surface context and restrict the liquid observations.
            ex += [
                {"x": xx, "y": yy, "width": ww, "height": hh}
                for xx, yy, ww, hh in (
                    (0, 0, x, height),
                    (x1, 0, width - x1, height),
                    (x, 0, w, y),
                    (x, y1, w, height - y1),
                )
                if ww and hh
            ]
            x, y, x1, y1 = 0, 0, width, height
    excluded = []
    for r in ex:
        xx, yy = max(r["x"], x), max(r["y"], y)
        end_x, end_y = min(r["x"] + r["width"], x1), min(r["y"] + r["height"], y1)
        if xx < end_x and yy < end_y:
            excluded.append({"x": xx - x, "y": yy - y, "width": end_x - xx, "height": end_y - yy})
    crop = gray[y:y1, x:x1]
    surface = estimate_pendant_surface(crop, points=translated(marks, -x, -y) if marks else None)
    result = fit_pendant(
        crop,
        surface,
        sensitivity=options["sensitivity"],
        contour_points=contours,
        excluded_regions=excluded,
        manual_method=options.get("manual_method", "ellipse"),
    )
    for side in ("left", "right"):
        result[side]["contact_point"] = translated([result[side]["contact_point"]], x, y)[0]
    geometry = result["geometry"]
    for curve in geometry["contours"] + geometry["fits"]:
        curve["points"] = translated(curve["points"], x, y)
    for side, points in geometry["selected_points"].items():
        geometry["selected_points"][side] = translated(points, x, y)
    for key in ("baseline", "baseline_points", "surface_curve"):
        result[key] = translated(surface[key], x, y)
    result["surface_diagnostics"] = {
        key: surface[key]
        for key in ("uncertainty_px", "fit_rms_px", "inlier_count", "footprint_source")
        if key in surface
    }
    result["surface_diagnostics"]["observed_spans"] = [
        [a + x, b + x] for a, b in surface.get("observed_spans", [])
    ]
    result["surface_diagnostics"]["footprint"] = [xx + x for xx in surface["footprint"]]
    result["algorithm"] = "manual" if manual else "pendant"
    result["baseline_method"] = "pendant_surface_v1"
    return result
