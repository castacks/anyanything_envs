"""Camera poses and ego-motion optical flow for the Blender courses.

numpy only, so the same file is imported inside Blender's bundled python
(``render_frames.py``) and by the repo side (``finish_frames.py``, tests).

Conventions -- the labeller's, so nothing here is new:

* world: z up, floor at z = 0; a state's ``R`` is R_wb (world <- body),
  body x forward, z up (``draw_states.flatness_rotation``);
* camera: OpenCV axes (x right, y down, z forward), mounted level through
  ``R_BC`` (``render_depth.R_BC``) at ``t_bc`` in the body frame (FLU, m; the
  body origin when omitted -- ``agile_91``; ``starling_front_v1`` sits at
  (0.071, 0, -0.007)); intrinsics from the named ``render_depth.Camera`` --
  640x480, 91 or 87 deg horizontal fov, principal point at the centre of the
  pixel grid ``(w - 1) / 2``;
* depth: z-depth (distance along the optical axis), metres.

**The previous frame.** States are drawn one at a time, so the frame the
drone would have seen ``dt`` earlier is synthesised from the state itself:
``p' = p - v dt + a dt^2 / 2`` and ``R' = R exp(-[w]x dt)`` with the body
rates ``w``. In a flown episode the true previous frame replaces this.

**The flow** is backward flow on the *current* frame's pixel grid: for pixel
x of frame t, ``flow[x]`` is where that surface point was in frame t - dt,
minus x. It is computed from depth(t) and the two poses (exact for a static
scene) and stored for every pixel; a separate status marks the pixels whose
point was hidden or out of view at t - dt. Pixels with no surface (sky) get
the rotation-only flow of a point at infinity, which is what a camera sees
there.
"""

from __future__ import annotations

import math

import numpy as np

#: columns: the camera's axes in the body frame (render_depth.R_BC)
R_BC = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]])

#: OpenCV camera axes -> Blender camera axes (x right, y up, looking down -z)
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])


def intrinsics(width: int = 640, height: int = 480, hfov_deg: float = 91.0) -> np.ndarray:
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def skew(w) -> np.ndarray:
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w) -> np.ndarray:
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def previous_pose(p, R_wb, v, a, w_body, dt: float):
    """(p', R_wb') of the body ``dt`` seconds earlier, to second order in p."""
    p, v, a = (np.asarray(x, float) for x in (p, v, a))
    return p - v * dt + 0.5 * a * dt * dt, np.asarray(R_wb, float) @ so3_exp(-np.asarray(w_body, float) * dt)


def camera_pose(p, R_wb, t_bc=None):
    """(t_wc, R_wc) of the OpenCV camera: columns of R_wc are its axes in the world;
    ``t_bc`` the mount offset in the body frame (FLU, m), None = the body origin."""
    R_wb = np.asarray(R_wb, float)
    t = np.asarray(p, float)
    if t_bc is not None:
        t = t + R_wb @ np.asarray(t_bc, float)
    return t, R_wb @ R_BC


def blender_matrix(p, R_wb, t_bc=None) -> np.ndarray:
    """4x4 ``matrix_world`` of a Blender camera at this body pose."""
    t, R = camera_pose(p, R_wb, t_bc)
    M = np.eye(4)
    M[:3, :3] = R @ CV_TO_BLENDER
    M[:3, 3] = t
    return M


def pixel_rays(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H, W, 3) camera-frame rays with z = 1 through the pixel centres."""
    u = (np.arange(width) - K[0, 2]) / K[0, 0]
    v = (np.arange(height) - K[1, 2]) / K[1, 1]
    U, V = np.meshgrid(u, v)
    return np.stack([U, V, np.ones_like(U)], -1)


#: per-pixel flow status (``backward_flow``)
VALID, OCCLUDED, OUT_OF_VIEW = 0, 1, 2


def _neighbour_range(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """(min, max, any_inf) of ``img`` over the 2x2 pixels around (u, v) -- the
    footprint a bilinear lookup would blend. Infinite (no surface) pixels are
    left out of min/max and reported in ``any_inf``."""
    H, W = img.shape
    u0 = np.clip(np.floor(u).astype(int), 0, W - 1); v0 = np.clip(np.floor(v).astype(int), 0, H - 1)
    u1 = np.minimum(u0 + 1, W - 1); v1 = np.minimum(v0 + 1, H - 1)
    stack = np.stack([img[v0, u0], img[v0, u1], img[v1, u0], img[v1, u1]])
    fin = np.isfinite(stack)
    lo = np.where(fin, stack, np.inf).min(0)
    hi = np.where(fin, stack, -np.inf).max(0)
    return lo, hi, (~fin).any(0)


def _slope_tolerance(depth: np.ndarray) -> np.ndarray:
    """How much the depth changes from one pixel to the next here (the larger
    of the two axes), which is what a sub-pixel reprojection error costs in
    depth on a surface seen at a grazing angle."""
    d = np.where(np.isfinite(depth), depth, np.nan)
    gx = np.abs(np.diff(d, axis=1, append=d[:, -1:]))
    gy = np.abs(np.diff(d, axis=0, append=d[-1:, :]))
    g = np.fmax(gx, gy)
    return np.nan_to_num(g, nan=0.0)


def backward_flow(depth_t: np.ndarray, depth_prev: np.ndarray, pose_t, pose_prev, K: np.ndarray,
                  rel_tol: float = 0.02, abs_tol: float = 0.02, slope_k: float = 1.5, grazing_cap: float = 0.25,
                  t_bc=None):
    """Flow of frame t back to frame t - dt, on frame t's grid.

    ``depth_*``: (H, W) z-depth in metres, ``inf`` (or <= 0) where no surface.
    ``pose_*``: (p, R_wb) body poses. Returns ``(flow (H, W, 2) float32,
    status (H, W) uint8, at_infinity (H, W) bool)``.

    The flow is defined -- and stored -- for **every** pixel: a surface point
    hidden or outside the frame at t - dt still has a well-defined previous
    image position. ``status`` says which pixels a photometric comparison may
    trust: ``VALID``; ``OCCLUDED`` (the point was behind something at
    t - dt); ``OUT_OF_VIEW`` (it was outside the image, or behind the camera).

    Visibility: the reprojected depth z' must fall inside the range of the
    previous depth over the 2x2 pixels a bilinear lookup would blend, widened
    by ``rel_tol * z' + abs_tol`` and by ``slope_k`` times the local
    pixel-to-pixel depth change of frame t (up to ``grazing_cap`` of z').
    The slope term is what keeps far ground seen at a grazing angle from
    reading as occluded: there one pixel of reprojection error is metres of
    depth, and a fixed relative tolerance fails in sawtooth wedges.
    """
    H, W = depth_t.shape
    t1, R1 = camera_pose(*pose_t, t_bc)
    t0, R0 = camera_pose(*pose_prev, t_bc)
    rays = pixel_rays(K, W, H)
    finite = np.isfinite(depth_t) & (depth_t > 0)
    Rrel = R0.T @ R1                                   # frame-t camera -> frame-(t-dt) camera
    # surface points: X0 = R0^T (R1 X1 + t1 - t0); points at infinity: rotation only
    X1 = rays * np.where(finite, depth_t, 1.0)[..., None]
    X0 = X1 @ Rrel.T + np.where(finite, 1.0, 0.0)[..., None] * (R0.T @ (t1 - t0))
    z0 = X0[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u0 = K[0, 0] * X0[..., 0] / z0 + K[0, 2]
        v0 = K[1, 1] * X0[..., 1] / z0 + K[1, 2]
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    flow = np.stack([u0 - uu, v0 - vv], -1)
    in_view = (z0 > 1e-6) & (u0 >= -0.5) & (u0 <= W - 0.5) & (v0 >= -0.5) & (v0 <= H - 0.5)
    prev = np.where(np.isfinite(depth_prev) & (depth_prev > 0), depth_prev, np.inf)
    lo, hi, any_inf = _neighbour_range(prev, np.nan_to_num(u0), np.nan_to_num(v0))
    zt = np.where(finite, z0, np.nan)
    tol = rel_tol * zt + abs_tol + np.minimum(slope_k * _slope_tolerance(depth_t), grazing_cap * zt)
    with np.errstate(invalid="ignore"):
        seen_surface = (zt >= lo - tol) & (zt <= hi + tol)
    visible = np.where(finite, seen_surface, any_inf)   # sky stays sky
    status = np.full((H, W), OCCLUDED, np.uint8)
    status[visible] = VALID
    status[~in_view] = OUT_OF_VIEW
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return flow, status, ~finite


def resize_flow(flow: np.ndarray, valid: np.ndarray, out: int, resize) -> tuple[np.ndarray, np.ndarray]:
    """Flow at ``out`` x ``out`` (the stored frame), in that grid's pixels.
    ``resize`` is ``render_depth.resize_bilinear`` so flow and depth share one
    resampler; a stored pixel is valid only if every source pixel under its
    bilinear footprint was."""
    H, W = valid.shape
    fu = resize(flow[..., 0].astype(np.float32), out) * (out / W)
    fv = resize(flow[..., 1].astype(np.float32), out) * (out / H)
    ok = resize(valid.astype(np.float32), out) > 0.999
    return np.stack([fu, fv], 0), ok


def flow_to_rgb(flow: np.ndarray, valid: np.ndarray | None = None, max_mag: float | None = None,
                style: str = "dark") -> np.ndarray:
    """Colour-wheel flow picture: hue = direction. ``style="dark"``: brightness
    = magnitude, black = still (the FlyingThings-style look); ``"light"``:
    saturation = magnitude, white = still (Middlebury). ``max_mag`` fixes the
    scale -- pass one value for a whole sequence so frames are comparable.
    ``valid``, if given, blacks out the pixels it marks False."""
    u, v = (flow[0], flow[1]) if flow.shape[0] == 2 else (flow[..., 0], flow[..., 1])
    mag = np.hypot(u, v)
    m = max_mag or max(float(np.percentile(mag, 99)), 1e-6)
    hue = (np.arctan2(-v, -u) / math.pi + 1) / 2
    amt = np.clip(mag / m, 0, 1)
    sat, val = (np.ones_like(amt), amt ** 0.7) if style == "dark" else (amt, np.ones_like(amt))
    h6 = hue * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
    lut = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    rgb = np.zeros(u.shape + (3,))
    for k, (r, g, b) in enumerate(lut):
        sel = i == k
        rgb[sel] = np.stack([r[sel], g[sel], b[sel]], -1)
    if valid is not None:
        rgb[~valid] = 0.0
    return (rgb * 255).astype(np.uint8)


def status_to_rgb(status: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """The status mask for eyes: valid pixels show ``base`` dimmed (or grey),
    occluded ones orange, out-of-view ones blue."""
    H, W = status.shape
    out = (np.full((H, W, 3), 60.0) if base is None else base.astype(float) * 0.35)
    out[status == OCCLUDED] = [240, 133, 63]
    out[status == OUT_OF_VIEW] = [70, 140, 240]
    return out.astype(np.uint8)
