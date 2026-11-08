"""Per-pixel unit ray directions in the camera optical frame (x right, y down, z forward).

Precomputed once, so any lens model costs nothing per frame. Pixel centres sit at integer
coordinates (OpenCV convention), so a calibration's (fx, fy, cx, cy) can be used as is.
"""
import math

import numpy as np

LENS_MODELS = ("pinhole", "kb4")


def intrinsics_from_fov(width, height, fov_deg, fov_type="horizontal", lens_model="pinhole"):
    """(fx, fy, cx, cy) for square pixels and a centred principal point."""
    if lens_model not in LENS_MODELS:
        raise ValueError(f"lens_model must be one of {LENS_MODELS}, got {lens_model!r}")
    if fov_type == "horizontal":
        r_px = width / 2.0
    elif fov_type == "diagonal":
        r_px = math.hypot(width, height) / 2.0
    else:
        raise ValueError(f"fov_type must be 'horizontal' or 'diagonal', got {fov_type!r}")
    half = math.radians(fov_deg) / 2.0
    # kb4 with zero coefficients is equidistant (r = f*theta), not pinhole (r = f*tan(theta))
    f = r_px / math.tan(half) if lens_model == "pinhole" else r_px / half
    return f, f, (width - 1) / 2.0, (height - 1) / 2.0


def _kb4_theta(theta_d, k):
    """Invert theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8) by Newton."""
    k1, k2, k3, k4 = k
    theta = theta_d.copy()
    for _ in range(30):
        t2 = theta * theta
        poly = 1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
        dpoly = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = theta - (theta * poly - theta_d) / dpoly
    return theta


def ray_lut(width, height, fx, fy, cx, cy, lens_model="pinhole", kb4_coeffs=(0.0, 0.0, 0.0, 0.0)):
    """(H, W, 3) float32 unit rays; ray[y, x] is the direction seen by pixel (x, y)."""
    if lens_model not in LENS_MODELS:
        raise ValueError(f"lens_model must be one of {LENS_MODELS}, got {lens_model!r}")
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    mx = (u - cx) / fx
    my = (v - cy) / fy
    if lens_model == "pinhole":
        d = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    else:
        theta_d = np.hypot(mx, my)
        theta = _kb4_theta(theta_d, kb4_coeffs)
        if np.any(theta >= math.pi) or np.any(~np.isfinite(theta)):
            raise ValueError("kb4 coefficients give rays at or beyond 180 deg off-axis")
        safe = np.maximum(theta_d, 1e-12)
        scale = np.where(theta_d > 1e-12, np.sin(theta) / safe, 1.0)
        d = np.stack([mx * scale, my * scale, np.cos(theta)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.astype(np.float32)
