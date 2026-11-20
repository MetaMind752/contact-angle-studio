"""Generate synthetic fixtures with analytic reference angles (not experimental data)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FIBRE_EXCLUSIONS = [
    {"x": 207, "y": 536, "width": 32, "height": 58},
    {"x": 661, "y": 536, "width": 37, "height": 53},
]


def _render_curve(
    curve: np.ndarray,
    *,
    tilt_deg: float = 0,
    polarity: str = "dark",
    origin: tuple[float, float] = (450, 600),
    size: tuple[int, int] = (900, 760),
    scale: int = 4,
) -> tuple[np.ndarray, dict]:
    """Render a closed drop bounded by an upper curve and its substrate chord."""
    tilt = np.deg2rad(tilt_deg)
    tangent = np.array([np.cos(tilt), np.sin(tilt)])
    normal = np.array([np.sin(tilt), -np.cos(tilt)])
    center = np.array(origin, dtype=float)
    image_points = center + curve[:, :1] * tangent + curve[:, 1:] * normal
    background, foreground = (240, 20) if polarity == "dark" else (20, 240)
    width, height = size
    image = np.full((height * scale, width * scale), background, dtype=np.uint8)
    polygon = np.rint(image_points * scale).astype(np.int32)
    cv2.fillPoly(image, [polygon], foreground)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    baseline = [center - 330 * tangent, center + 330 * tangent]
    options = {
        "baseline": [{"x": float(p[0]), "y": float(p[1])} for p in baseline],
        "polarity": polarity,
        "method": "local_polynomial",
    }
    return image, options


def circular_cap(
    angle_deg: float,
    *,
    radius: float = 220,
    tilt_deg: float = 0,
    polarity: str = "dark",
    origin: tuple[float, float] = (450, 600),
    size: tuple[int, int] = (900, 760),
) -> tuple[np.ndarray, dict]:
    """Circle centered at height -R*cos(theta); exact internal angle is theta."""
    theta = np.deg2rad(angle_deg)
    phi = np.linspace(-theta, theta, 2401)
    curve = radius * np.column_stack((np.sin(phi), np.cos(phi) - np.cos(theta)))
    return _render_curve(curve, tilt_deg=tilt_deg, polarity=polarity, origin=origin, size=size)


def asymmetric_drop() -> tuple[np.ndarray, dict, tuple[float, float]]:
    """Cubic Bezier cap with independent endpoint tangents known analytically."""
    control = np.array([[-180, 0], [-130, 140], [110, 100], [180, 0]], dtype=float)
    t = np.linspace(0, 1, 2401)[:, None]
    curve = (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )
    image, options = _render_curve(curve)
    expected = (float(np.degrees(np.arctan2(140, 50))), float(np.degrees(np.arctan2(100, 70))))
    return image, options, expected


def fuzzy_drop() -> tuple[np.ndarray, dict]:
    """A deliberately corrupted synthetic 90-degree silhouette for review testing."""
    image, options = circular_cap(90)
    # Thin protrusions near the contact areas emulate visible fibres, not hidden geometry.
    for x, y, dx, dy in [
        (230, 585, -16, -8),
        (232, 570, -19, -12),
        (236, 551, -20, -10),
        (669, 580, 18, -5),
        (665, 553, 17, -12),
    ]:
        cv2.line(image, (x, y), (x + dx, y + dy), 20, 2, cv2.LINE_AA)
    image = cv2.GaussianBlur(image, (0, 0), 1.7)
    noise = np.random.default_rng(28).normal(0, 4, image.shape)
    return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8), options


def png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode synthetic image")
    return encoded.tobytes()


def main() -> None:
    root = Path(__file__).resolve().parent
    ground_truth = {}
    for angle in (90, 120):
        name = f"demo_{angle}"
        image, options = circular_cap(angle)
        (root / f"{name}.png").write_bytes(png_bytes(image))
        (root / f"{name}.options.json").write_text(json.dumps(options, indent=2) + "\n")
        ground_truth[name] = {"synthetic": True, "left_angle_deg": angle, "right_angle_deg": angle}
    image, options = fuzzy_drop()
    (root / "demo_fuzzy.png").write_bytes(png_bytes(image))
    (root / "demo_fuzzy.options.json").write_text(json.dumps(options, indent=2) + "\n")
    masked_options = {**options, "excluded_regions": FIBRE_EXCLUSIONS}
    (root / "demo_fuzzy_masked.options.json").write_text(
        json.dumps(masked_options, indent=2) + "\n"
    )
    ground_truth["demo_fuzzy"] = {
        "synthetic": True,
        "underlying_cap_angle_deg": 90,
        "note": "Blur, deterministic noise and protrusions alter the visible boundary; for sensitivity demonstration only.",
    }
    (root / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2) + "\n")


if __name__ == "__main__":
    main()
