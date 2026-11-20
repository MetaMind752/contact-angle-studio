"""Independent geometry fixtures for fibrous-substrate contact-angle validation.

Analytic target angles are through the liquid. Synthetic curves only test image/
geometry implementation, not a physical Young-Laplace model of a real drop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Case:
    name: str
    image: np.ndarray  # BGR uint8
    options: dict
    truth: tuple[float, float]
    contour_xy: np.ndarray
    fibre_mask: np.ndarray
    geometry: str
    metadata: dict


def circle_curve(angle: float, radius: float = 220) -> np.ndarray:
    a = np.deg2rad(angle)
    t = np.linspace(-a, a, 3001)
    return radius * np.column_stack((np.sin(t), np.cos(t) - np.cos(a)))


def ellipse_curve(angle: float, radius_x: float = 220, ratio: float = 0.85) -> np.ndarray:
    """Ellipse scaled so the endpoint tangent's true angle equals angle."""
    a = np.deg2rad(angle)
    radius_y = radius_x * ratio
    t_contact = np.arctan2(radius_x * np.sin(a), radius_y * np.cos(a))
    t = np.linspace(-t_contact, t_contact, 3001)
    return np.column_stack((radius_x * np.sin(t), radius_y * (np.cos(t) - np.cos(t_contact))))


def bezier_curve(left_angle=135, right_angle=150, width=240, handle=310):
    a, b = np.deg2rad([left_angle, right_angle])
    p0 = np.array([-width / 2, 0.0])
    p3 = np.array([width / 2, 0.0])
    p1 = p0 + handle * np.array([np.cos(a), np.sin(a)])
    p2 = p3 + handle * np.array([-np.cos(b), np.sin(b)])
    t = np.linspace(0, 1, 3001)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def render(
    name: str,
    curve: np.ndarray,
    truth: tuple[float, float],
    *,
    geometry: str,
    tilt_deg=-1,
    fibres=True,
    seed=101,
    size=(1024, 768),
    origin=(512, 650),
    scale=3,
    blur_sigma=0.8,
    noise_sigma=1.5,
) -> Case:
    width, height = size
    theta = np.deg2rad(tilt_deg)
    along = np.array([np.cos(theta), np.sin(theta)])
    up = np.array([np.sin(theta), -np.cos(theta)])
    origin = np.array(origin, dtype=float)

    def to_img(p):
        p = np.asarray(p)
        return origin + p[..., 0, None] * along + p[..., 1, None] * up

    # Render original geometric outline first, then the independently drawn solid.
    image = np.full((height * scale, width * scale), 245, dtype=np.uint8)
    image_curve = to_img(curve)
    cv2.fillPoly(image, [np.rint(image_curve * scale).astype(np.int32)], 12)
    substrate = to_img(np.array([[-800, 0], [800, 0], [800, -600], [-800, -600]]))
    cv2.fillPoly(image, [np.rint(substrate * scale).astype(np.int32)], 12)

    # Internal bright reflection: should never be taken as an outer boundary.
    centroid = np.mean(curve, axis=0)
    spot = to_img(centroid)
    spot_radius = min(18.0, max(9.0, np.ptp(curve[:, 0]) * 0.04))
    cv2.circle(
        image,
        tuple(np.rint(spot * scale).astype(int)),
        int(spot_radius * scale),
        225,
        -1,
    )

    mask = np.zeros_like(image)
    rng = np.random.default_rng(seed)
    if fibres:
        # Surface fibres are generated independently of target angle and manual labels.
        for u in rng.uniform(-500, 500, 140):
            h = rng.uniform(3, 13)
            d = rng.uniform(-7, 7)
            a, b = to_img([[u, 0], [u + d, h]])
            cv2.line(
                mask,
                tuple(np.rint(a * scale).astype(int)),
                tuple(np.rint(b * scale).astype(int)),
                255,
                scale,
            )
        # Contact-adjacent protrusions are tall/thick enough to mimic problem images.
        for contact in (curve[0, 0], curve[-1, 0]):
            for _ in range(9):
                u = contact + rng.uniform(-22, 22)
                h = rng.uniform(10, 25)
                d = rng.uniform(-13, 13)
                a, b = to_img([[u, -1], [u + d, h]])
                cv2.line(
                    mask,
                    tuple(np.rint(a * scale).astype(int)),
                    tuple(np.rint(b * scale).astype(int)),
                    255,
                    int(rng.choice([1, 2])) * scale,
                )
        image[mask > 0] = 20
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    if blur_sigma:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    if noise_sigma:
        image = np.clip(
            image.astype(float) + rng.normal(0, noise_sigma, image.shape), 0, 255
        ).astype(np.uint8)
    baseline = to_img([[-450, 0], [450, 0]])
    options = {
        "baseline": [{"x": float(p[0]), "y": float(p[1])} for p in baseline],
        "polarity": "dark",
        "sensitivity": False,
    }
    return Case(
        name,
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        options,
        truth,
        image_curve,
        mask,
        geometry,
        {
            "tilt_deg": tilt_deg,
            "fibres": fibres,
            "seed": seed,
            "blur_sigma": blur_sigma,
            "noise_sigma": noise_sigma,
        },
    )


def cases(*, fibres=True, seeds=(101,)):
    for seed in seeds:
        for radius in (140, 220, 300):
            for angle in (125, 140, 155):
                yield render(
                    f"circle_r{radius}_a{angle}_s{seed}",
                    circle_curve(angle, radius),
                    (angle, angle),
                    geometry="circle",
                    fibres=fibres,
                    seed=seed,
                )
        for radius in (180, 240):
            for ratio in (0.8, 1.2):
                for angle in (135, 150):
                    yield render(
                        f"ellipse_r{radius}_q{ratio}_a{angle}_s{seed}",
                        ellipse_curve(angle, radius, ratio),
                        (angle, angle),
                        geometry="ellipse",
                        fibres=fibres,
                        seed=seed,
                    )
        for a, b in ((125, 145), (145, 125), (135, 150), (150, 135)):
            yield render(
                f"asym_a{a}_b{b}_s{seed}",
                bezier_curve(a, b),
                (a, b),
                geometry="asymmetric_bezier",
                fibres=fibres,
                seed=seed,
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/validation_new/images"))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    records = []
    for case in cases():
        cv2.imwrite(str(args.output / (case.name + ".png")), case.image)
        (args.output / (case.name + ".options.json")).write_text(json.dumps(case.options, indent=2))
        records.append(
            {
                "name": case.name,
                "geometry": case.geometry,
                "left": case.truth[0],
                "right": case.truth[1],
                **case.metadata,
            }
        )
    (args.output / "ground_truth.json").write_text(json.dumps(records, indent=2))
    print(f"Generated {len(records)} fixtures")


if __name__ == "__main__":
    main()
