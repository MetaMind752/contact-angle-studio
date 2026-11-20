"""Export the original photograph with observed curves, tangents and interior arcs."""

import math

import cv2
import numpy as np

from .models import MeasurementError


def annotate(image, result):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    scale = max(0.5, min(w, h) / 650)
    thickness = max(1, round(scale * 2))

    def point(p):
        return np.array([p["x"], p["y"]], dtype=float)

    def line(points, color, width=1):
        arr = np.array([[p["x"], p["y"]] for p in points], dtype=float)
        if len(arr) > 1 and np.isfinite(arr).all():
            cv2.polylines(canvas, [np.rint(arr).astype(np.int32)], False, color, width, cv2.LINE_AA)

    def draw_surface(curve, left_x, right_x):
        # The physical surface is obscured between the contacts. Keep this inferred
        # part dashed, with the dash lengths measured along the fitted curve.
        xy = np.array([point(p) for p in curve])
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        start, stop = np.interp([left_x, right_x], xy[:, 0], distance)
        dash = max(4, 7 * scale)
        segments = [(0.0, start), (stop, distance[-1])]
        segments.extend((s, min(stop, s + dash)) for s in np.arange(start, stop, dash * 1.8))
        for a, b in segments:
            if b <= a:
                continue
            d = np.r_[a, distance[(a < distance) & (distance < b)], b]
            piece = np.column_stack([np.interp(d, distance, axis) for axis in xy.T])
            cv2.polylines(
                canvas,
                [np.rint(piece).astype(np.int32)],
                False,
                (225, 120, 55),
                thickness,
                cv2.LINE_AA,
            )

    for curve in result["geometry"]["contours"]:
        line(curve["points"], (170, 160, 135), 1)
    colors = {"left": (28, 155, 230), "right": (170, 155, 20)}
    for curve in result["geometry"]["fits"]:
        if result.get("mode") == "manual":
            line(curve["points"], (53, 45, 198), max(1, round(scale)))
        else:
            line(curve["points"], colors.get(curve["side"], (170, 150, 130)), thickness)
    a, b = sorted(result["baseline"], key=lambda p: p["x"])
    along = point(b) - point(a)
    along /= np.linalg.norm(along)
    if result.get("surface_curve"):
        draw_surface(
            result["surface_curve"],
            result["left"]["contact_point"]["x"],
            result["right"]["contact_point"]["x"],
        )
    else:
        line([a, b], (225, 120, 55), thickness)
    for selected in result.get("baseline_points", []):
        cv2.circle(
            canvas,
            tuple(np.rint(point(selected)).astype(int)),
            max(2, thickness),
            (225, 120, 55),
            -1,
            cv2.LINE_AA,
        )
    for selected in result["geometry"]["selected_points"].get("contour", []):
        cv2.circle(
            canvas,
            tuple(np.rint(point(selected)).astype(int)),
            max(2, thickness),
            (95, 95, 95),
            -1,
            cv2.LINE_AA,
        )
    contact_gap = np.linalg.norm(
        point(result["right"]["contact_point"]) - point(result["left"]["contact_point"])
    )
    radius = min(max(9, min(w, h) * 0.055), contact_gap * 0.36)
    for side in ("left", "right"):
        result_side = result[side]
        p = point(result_side["contact_point"])
        vector = point(result_side["tangent_vector"])
        vector /= np.linalg.norm(vector)
        color = colors[side]
        surface_tangent = result_side.get("surface_tangent")
        base = (
            point(surface_tangent)
            if surface_tangent
            else along.copy() * (1 if side == "left" else -1)
        )
        base /= np.linalg.norm(base)
        if surface_tangent:
            end = p + base * radius * 1.45
            cv2.line(
                canvas,
                tuple(np.rint(p).astype(int)),
                tuple(np.rint(end).astype(int)),
                (225, 120, 55),
                thickness,
                cv2.LINE_AA,
            )
        # Both vectors leave the contact into the liquid. Their cross product determines
        # the signed interior sweep in the original, downward-positive image coordinates.
        cross = base[0] * vector[1] - base[1] * vector[0]
        sweep = math.copysign(math.radians(result_side["angle_deg"]), cross)
        theta = math.atan2(base[1], base[0]) + np.linspace(0, sweep, 65)
        arc = p + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        cv2.polylines(canvas, [np.rint(arc).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
        cv2.line(
            canvas,
            tuple(np.rint(p - vector * radius * 0.35).astype(int)),
            tuple(np.rint(p + vector * radius * 2.1).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas, tuple(np.rint(p).astype(int)), max(3, thickness + 1), color, -1, cv2.LINE_AA
        )
        for selected in result["geometry"]["selected_points"].get(side, []):
            cv2.circle(
                canvas, tuple(np.rint(point(selected)).astype(int)), max(2, thickness), color, -1
            )
        text = f"{'L' if side == 'left' else 'R'} {result_side['angle_deg']:.1f}"
        font_size = max(0.5, min(1.35, min(w, h) / 640))
        stroke = max(1, round(2 * font_size))
        (text_width, text_height), descent = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_size, stroke
        )
        degree_radius = max(2, round(3.5 * font_size))
        label_width = text_width + 3 * degree_radius + 2
        # Put the figures outward of each contact. A light halo keeps the darker
        # print colors readable on both solid and air when the PNG is reduced in Word.
        offset = radius * 1.6
        label_x = p[0] - label_width - offset if side == "left" else p[0] + offset
        x = int(np.clip(label_x, 4, max(4, w - label_width - 4)))
        liquid_y = 1 if (base[0] * sweep) > 0 else -1
        y = int(np.clip(p[1] + liquid_y * radius * 1.35, text_height + 5, h - descent - 4))
        degree_center = (x + text_width + degree_radius + 2, y - text_height + degree_radius + 1)
        ink = (20, 59, 150) if side == "left" else (72, 83, 6)
        halo = stroke + max(3, round(font_size * 3))
        cv2.circle(canvas, degree_center, degree_radius, (255, 255, 255), halo, cv2.LINE_AA)
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            halo,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, ink, stroke, cv2.LINE_AA
        )
        cv2.circle(canvas, degree_center, degree_radius, ink, stroke, cv2.LINE_AA)
    # Footer is outside the original image: original pixel coordinates are preserved.
    footer_h = 94
    exported = np.full((h + footer_h, w, 3), 249, dtype=np.uint8)
    exported[:h] = canvas
    texts = [
        f"L {result['left']['angle_deg']:.2f} deg   R {result['right']['angle_deg']:.2f} deg   Mean {result['mean_angle_deg']:.2f} deg",
        f"{result['mode']} / {result['algorithm']}   {result['status']}   Original {w} x {h} px",
    ]
    sensitivity = result.get("sensitivity") or {}
    sides = []
    for side in ("left", "right"):
        interval = sensitivity.get(side)
        if interval:
            sides.append(
                f"{side[0].upper()} {interval['min_deg']:.1f}-{interval['max_deg']:.1f} deg"
            )
    if sides:
        texts.append("Sensitivity    " + "    ".join(sides))
    for index, text in enumerate(texts):
        factor = min(
            0.52,
            max(
                0.2, (w - 16) / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0])
            ),
        )
        cv2.putText(
            exported,
            text,
            (8, h + 24 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            factor,
            (55, 65, 70),
            1,
            cv2.LINE_AA,
        )
    ok, png = cv2.imencode(".png", exported)
    if not ok:
        raise MeasurementError("encode_failed", "无法导出测量标注图。")
    return png.tobytes()
