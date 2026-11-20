"""Geometric regression cases for the dense solid boundary beneath surface fibers.

The true substrate is a separately rendered plane. Fibers protrude above it and
cover most tested lateral columns; their tips are not samples of that plane.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from contact_angle.studio_service import estimate_surface_points
from contact_angle.surface_points import estimate_surface_points as sample_surface


def dense_substrate_image(
    *, kind="dense", tilt=0.0, fiber_height=16.0, substrate=20,
    light=False, noise=0.6, seed=821,
):
    """Return an independent supersampled image and its exact substrate plane."""
    width, height, scale = 800, 500, 4
    origin = np.array([400.0, 330.25])
    angle = math.radians(tilt)
    along = np.array([math.cos(angle), math.sin(angle)])
    up = np.array([math.sin(angle), -math.cos(angle)])
    rng = np.random.default_rng(seed)

    def image_points(points):
        uv = np.asarray(points, dtype=float)
        return origin + uv[:, :1] * along + uv[:, 1:] * up

    def pixels(points):
        return np.rint(image_points(points) * scale).astype(np.int32)

    raw = np.full((height * scale, width * scale), 245, np.uint8)
    # A 130-degree circular cap, independent of all substrate/fiber parameters.
    theta, radius = math.radians(130), 128.0
    phase = np.linspace(-theta, theta, 600)
    cap = np.c_[radius * np.sin(phase), radius * (np.cos(phase) - math.cos(theta))]
    cv2.fillPoly(raw, [pixels(cap)], 14)
    cv2.fillPoly(raw, [pixels([[-700, 0], [700, 0], [700, -500], [-700, -500]])], substrate)
    # Draw well outside the drop so lateral observations cannot be liquid edge.
    if kind != "none":
        spacing = 18.0 if kind == "sparse" else 8.5 if kind == "bundles" else 5.0
        for lo, hi in [(-335.0, -135.0), (135.0, 335.0)]:
            for u in np.arange(lo, hi, spacing):
                fh = fiber_height + rng.uniform(-1.0, 1.0)
                if kind == "bundles":
                    # Multiple neighboring stalks form clumps but leave visible gaps.
                    for offset in (-1.25, 0, 1.25):
                        cv2.line(raw, tuple(pixels([[u + offset, -1]])[0]),
                                 tuple(pixels([[u + offset + 1.5, fh]])[0]), 12, 2 * scale)
                else:
                    lean = fh * 0.28 if kind == "slanted" else rng.uniform(-0.5, 0.5)
                    cv2.line(raw, tuple(pixels([[u, -1]])[0]),
                             tuple(pixels([[u + lean, fh]])[0]), 12,
                             round((1.2 if kind == "sparse" else 2.5) * scale))
    raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
    raw = cv2.GaussianBlur(raw, (0, 0), 0.45)
    if noise:
        raw = np.clip(raw.astype(float) + rng.normal(0, noise, raw.shape), 0, 255).astype(np.uint8)
    if light:
        raw = cv2.bitwise_not(raw)
    image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    def true_y(x):
        return float(origin[1] + math.tan(angle) * (x - origin[0]))

    seed_line = [{"x": float(x), "y": true_y(x) - 7.0} for x in (70, 730)]
    return image, true_y, seed_line


LOWER_CASES = [
    ("none", -8, 0, 20), ("none", 0, 0, 20), ("none", 8, 0, 20),
    ("sparse", -6, 17, 20), ("sparse", 6, 17, 20),
    ("dense", -6, 14, 20), ("dense", 0, 14, 20), ("dense", 6, 14, 20),
    ("dense", 0, 25, 20), ("dense", 0, 36, 20),
    ("bundles", -5, 18, 20), ("bundles", 5, 18, 20),
    ("slanted", -6, 20, 20), ("slanted", 6, 20, 20),
    ("dense", -4, 16, 215), ("dense", 4, 16, 215),
]


@pytest.mark.parametrize("kind,tilt,fiber_height,substrate", LOWER_CASES)
def test_local_sampler_tracks_continuous_substrate_beneath_fibers(
    kind, tilt, fiber_height, substrate,
):
    image, true_y, biased_seed = dense_substrate_image(
        kind=kind, tilt=tilt, fiber_height=fiber_height, substrate=substrate,
    )
    result = sample_surface(image, biased_seed)
    assert len(result["baseline_points"]) >= 12
    errors = [p["y"] - true_y(p["x"]) for p in result["baseline"]]
    assert max(abs(value) for value in errors) < 2.0, errors
    observed = [p["y"] - true_y(p["x"]) for p in result["baseline_points"]]
    assert abs(float(np.median(observed))) < 2.0, observed


@pytest.mark.parametrize("algorithm", ["general", "local", "fibrous"])
@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("substrate", [20, 215])
def test_service_handles_dense_fibers_without_raising_surface_to_the_tips(algorithm, light, substrate):
    image, true_y, _ = dense_substrate_image(kind="dense", tilt=0, fiber_height=14, substrate=substrate, light=light)
    result = estimate_surface_points(image, "light" if light else "dark", algorithm)
    errors = [p["y"] - true_y(p["x"]) for p in result["baseline"]]
    assert max(abs(value) for value in errors) < 2.0, errors
