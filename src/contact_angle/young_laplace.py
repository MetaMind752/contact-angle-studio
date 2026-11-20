"""Independent axisymmetric Young--Laplace side-profile fit.

Coordinates: points[:,0] = baseline-aligned u, points[:,1] = height h above
baseline; apex height A is positive. Dimensionless z points down from apex.
The implemented convention is dr/ds=cos(phi), dz/ds=sin(phi),
dphi/ds=2+Bo*z-sin(phi)/r, with apex radius R0 and Bo in [0,3].
No image labels or reference contact angles are used by the fitting routine.

Run this file to execute independent analytic-circle and generated-YL checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


@dataclass
class Profile:
    bond_number: float
    z: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    radius_of_z: object
    angle_of_z: object


@lru_cache(maxsize=256)
def solve_profile(bond_number: float, max_angle_deg: float = 179.9) -> Profile:
    """Integrate dimensionless apex-radius-one Young--Laplace equations.

    solve_ivp, rather than contact-angle data or a third-party implementation,
    generates every profile. Values are not quantized in the optimization path.
    """
    bo = float(bond_number)
    if not np.isfinite(bo) or not 0 <= bo <= 3:
        raise ValueError("Bond number must be finite and lie in [0,3].")
    limit = np.deg2rad(max_angle_deg)
    eps = 1e-5

    def equations(s, state):
        r, z, phi = state
        return [np.cos(phi), np.sin(phi), 2.0 + bo * z - np.sin(phi) / r]

    def reached_angle(s, state):
        return state[2] - limit

    reached_angle.terminal = True
    reached_angle.direction = 1
    solution = solve_ivp(
        equations,
        (eps, 8.0),
        [eps, eps * eps / 2, eps],
        events=reached_angle,
        rtol=2e-9,
        atol=2e-11,
        max_step=0.015,
    )
    if not solution.success or not len(solution.t_events[0]):
        raise ValueError("Young--Laplace integration did not reach the angle limit.")
    r, z, phi = solution.y
    z, r, phi = np.r_[0.0, z], np.r_[0.0, r], np.r_[0.0, phi]
    if np.any(np.diff(z) <= 0) or not np.isfinite(r).all():
        raise ValueError("Profile is not invertible before the angle limit.")
    return Profile(
        bo,
        z,
        r,
        phi,
        PchipInterpolator(z, r, extrapolate=False),
        PchipInterpolator(z, phi, extrapolate=False),
    )


def generate_profile(
    contact_angle_deg=140.0,
    apex_radius_px=220.0,
    bond_number=0.25,
    center_u=0.0,
    side="left",
    count=300,
):
    """Generate a complete side arc from a prescribed angle and YL parameters."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    profile = solve_profile(float(bond_number))
    if not 0 < contact_angle_deg <= 174:
        raise ValueError("Generator supports contact angles in (0,174] degrees.")
    z_contact = float(PchipInterpolator(profile.phi, profile.z)(np.deg2rad(contact_angle_deg)))
    z = np.linspace(z_contact, max(1e-5, z_contact * 0.001), count)
    radius = float(apex_radius_px)
    apex = radius * z_contact
    sign = -1 if side == "left" else 1
    uv = np.c_[center_u + sign * radius * profile.radius_of_z(z), apex - radius * z]
    return uv, {
        "cx": float(center_u),
        "A": apex,
        "R0": radius,
        "Bo": float(bond_number),
        "contact_angle_deg": float(contact_angle_deg),
    }


def _circle_initial(points):
    matrix = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]
    cx, cy, constant = np.linalg.lstsq(matrix, np.sum(points * points, axis=1), rcond=None)[0]
    radius = math.sqrt(max(1.0, constant + cx * cx + cy * cy))
    return float(cx), float(cy + radius), float(radius)


def fit_young_laplace(points, side="left", initial=None, max_nfev=100, multistart=True):
    """Fit a single observed side to the supplied baseline h=0.

    Returns angle/contact/slope/rms, convergence and identifiability diagnostics,
    fitted parameters, and a callable prediction(height). Single-side fits are
    not proof of axisymmetry; the caller must compare sides and other models.
    A fit may have low RMS but poor parameter identification, reported explicitly.
    """
    points = np.asarray(points, dtype=float)
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 30:
        raise ValueError("Need at least 30 observed [u,h] points for one side.")
    if not np.isfinite(points).all():
        raise ValueError("Observed points must be finite.")
    # Uniformly subsample only for runtime; no smoothing or invented data points.
    if len(points) > 400:
        points = points[np.linspace(0, len(points) - 1, 400).astype(int)]
    height = float(np.max(points[:, 1]))
    extent = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 20))
    if np.ptp(points[:, 1]) < 10:
        raise ValueError("Observed side has insufficient vertical extent.")
    min_u, max_u = points[:, 0].min(), points[:, 0].max()
    lower = np.array([min_u - 3 * extent, max(1e-3, height - 0.5), 3.0, 0.0])
    upper = np.array([max_u + 3 * extent, height + 3 * extent, 6 * extent, 3.0])
    sign = -1 if side == "left" else 1
    cx, apex, radius = _circle_initial(points)
    start = np.array([cx, max(apex, height + 0.01), radius, 0.05])
    if initial is not None:
        if isinstance(initial, dict):
            start = np.array([initial[k] for k in ("cx", "A", "R0", "Bo")], dtype=float)
        else:
            start = np.array(initial, dtype=float)
    start = np.clip(start, lower + 1e-7, upper - 1e-7)

    def residual(parameters):
        cx, apex, radius, bo = parameters
        profile = solve_profile(float(bo))
        z = (apex - points[:, 1]) / radius
        clipped = np.clip(z, 1e-8, profile.z[-1] - 1e-8)
        pred = cx + sign * radius * profile.radius_of_z(clipped)
        angle = profile.angle_of_z(clipped)
        slope = -sign * np.cos(angle) / np.maximum(np.sin(angle), 1e-8)
        normal = (pred - points[:, 0]) / np.sqrt(1 + slope * slope)
        # A point above the apex or past the model limit must be penalized,
        # not silently assigned the nearest valid shape coordinate.
        outside = (z - clipped) * radius
        return normal + outside

    starts = [start]
    if multistart:
        alternate = start.copy()
        alternate[3] = 0.5 if start[3] < 0.25 else 0.02
        starts.append(alternate)
    attempts = []
    for candidate in starts:
        fit = least_squares(
            residual,
            candidate,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.4,
            x_scale="jac",
            diff_step=2e-4,
            ftol=2e-8,
            xtol=2e-8,
            gtol=2e-8,
            max_nfev=max_nfev,
        )
        attempts.append(fit)
    fits = [f for f in attempts if f.success]
    if not fits:
        raise ValueError("Young-Laplace optimizer did not converge within the iteration budget.")
    # Compare only geometrically feasible converged attempts. A lower residual
    # profile which stops above the supplied baseline cannot define a contact.
    feasible = []
    for candidate in fits:
        _, apex_c, radius_c, bo_c = candidate.x
        profile_c = solve_profile(float(bo_c))
        if 0 < apex_c / radius_c < profile_c.z[-1]:
            observed = (apex_c - points[:, 1]) / radius_c
            if np.all(observed >= -1e-3) and np.all(observed <= profile_c.z[-1] + 1e-3):
                feasible.append(candidate)
    if not feasible:
        raise ValueError("No converged YL profile reaches the baseline and all visible points.")
    fit = min(feasible, key=lambda f: float(np.mean(residual(f.x) ** 2)))
    cx, apex, radius, bo = fit.x
    profile = solve_profile(float(bo))
    z0 = apex / radius
    if not 0 < z0 < profile.z[-1]:
        raise ValueError(
            "Fitted profile does not reach the supplied baseline within 179.9 degrees."
        )
    contact_angle = float(np.rad2deg(profile.angle_of_z(z0)))
    contact = float(cx + sign * radius * profile.radius_of_z(z0))
    slope = float(-sign / np.tan(np.deg2rad(contact_angle)))
    observed_z = (apex - points[:, 1]) / radius
    if np.any(observed_z < -1e-3) or np.any(observed_z > profile.z[-1] + 1e-3):
        raise ValueError("Observed contour lies outside fitted YL profile.")

    def prediction(h):
        z = (apex - np.asarray(h, dtype=float)) / radius
        return cx + sign * radius * profile.radius_of_z(z)

    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
    boundary = [name for name, mask in zip(("cx", "A", "R0", "Bo"), fit.active_mask) if mask]
    rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return {
        "angle": contact_angle,
        "angle_deg": contact_angle,
        "contact": contact,
        "slope": slope,
        "rms": rms,
        "full_fit_rms_px": rms,
        "points": len(points),
        "prediction": prediction,
        "parameters": {"cx": float(cx), "A": float(apex), "R0": float(radius), "Bo": float(bo)},
        "diagnostics": {
            "converged": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "jacobian_condition": condition,
            "active_bounds": boundary,
            "attempts": len(attempts),
            "nearest_visible_height_px": float(points[:, 1].min()),
            "visible_vertical_span_px": float(np.ptp(points[:, 1])),
            "model_convention": "2+Bo*z;apex_radius_one",
            "warning": "Single-side model may be weakly identified; RMS does not establish accuracy.",
        },
    }


def _self_test():
    rows = []
    # Exact circles test analytic integration as well as fitted side orientation.
    circle = solve_profile(0.0)
    assert np.max(np.abs(circle.r - np.sin(circle.phi))) < 1e-6
    assert np.max(np.abs(circle.z - (1 - np.cos(circle.phi)))) < 1e-6
    for side in ("left", "right"):
        for bo, angle in ((0.0, 45), (0.0, 145), (0.15, 70), (0.4, 135), (1.0, 160)):
            uv, _truth = generate_profile(angle, 180, bo, 150, side)
            # Do not supply true fit parameters as the initializer.
            result = fit_young_laplace(uv[uv[:, 1] > 2], side, max_nfev=120)
            error = result["angle"] - angle
            rows.append(
                {
                    "side": side,
                    "Bo": bo,
                    "angle": angle,
                    "measured": result["angle"],
                    "error": error,
                    "rms": result["rms"],
                    "fit": result["parameters"],
                    "converged": result["diagnostics"]["converged"],
                }
            )
            assert result["diagnostics"]["converged"], rows[-1]
            assert abs(error) < 0.2, rows[-1]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    _self_test()
