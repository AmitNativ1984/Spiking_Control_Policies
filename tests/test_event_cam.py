"""Analytic checks of the warp event camera (sensors/event_camera) on synthetic scenes.

Each test compares DIFF3D output against a closed-form expectation: a straight albedo
edge swept across the image leaves a band of ceil/floor(displacement) pixels per frame,
each holding floor(ln(I1/I2)/C) events. No Isaac Gym sim is built.
"""
import math

import numpy as np
import pytest
import torch
import trimesh as tm
import warp as wp

from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig
from sensors.event_camera import WarpEventCam, intrinsics_from_fov, ray_lut
from sensors.event_camera.synthetic import SyntheticScene, camera_quat, random_forest, wall_x

DEV = "cuda:0"
MS = 1_000_000  # ns

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="warp kernels need CUDA")

# Deterministic sensor: no threshold noise, no background activity, I = rho exactly.
CLEAN = dict(
    c_sigma_pixel=0.0, c_sigma_temporal=0.0, ba_rate_hz_range=[0.0, 0.0],
    c_on_mean_range=[0.25, 0.25], c_off_mean_range=[0.25, 0.25],
    ambient_range=[1.0, 1.0], diffuse_range=[0.0, 0.0], sky_intensity_range=[0.5, 0.5],
    frame_period_s=0.010, num_frames=1, oversample=1,
)


def make_cam(scene, num_envs=1, **overrides):
    wp.init()
    cfg = type("Cfg", (GenX320EventCamConfig,), {**CLEAN, **overrides})
    cam = WarpEventCam(num_envs, cfg, scene.mesh_ids_array, DEV, meshes=scene.meshes)
    cam.scene = scene  # the camera only holds mesh ids; a collected wp.Mesh frees its BVH
    return cam


class Pose:
    def __init__(self, cam, n=1):
        self.pos = torch.zeros((n, 1, 3), device=DEV)
        self.quat = torch.zeros((n, 1, 4), device=DEV)
        self.set(yaw=0.0)
        cam.set_pose_tensor(self.pos, self.quat)

    def set(self, xyz=None, yaw=None):
        if xyz is not None:
            self.pos[:, 0] = torch.tensor(xyz, dtype=torch.float32, device=DEV)
        if yaw is not None:
            self.quat[:, 0] = torch.tensor(camera_quat(yaw), dtype=torch.float32, device=DEV)


def edge_wall(z):
    """Wall at world x = z split at y = 0: asset 0 is image-right, asset 1 image-left."""
    return SyntheticScene([[wall_x(z, -50, 0, -50, 50), wall_x(z, 0, 50, -50, 50)]], DEV)


def set_albedo(cam, *rho):
    for i, r in enumerate(rho):
        cam.albedo[:, i] = r


def run_sweep(z, v, steps, rho=(0.2, 0.8), step_ms=10, **cfg):
    """Translate along +y at v m/s in front of the edge wall; return the closed frames."""
    scene = edge_wall(z)
    cam = make_cam(scene, **cfg)
    set_albedo(cam, *rho)
    pose = Pose(cam)
    cam.capture(0)
    frames = []
    for k in range(1, steps + 1):
        pose.set(xyz=[0.0, v * k * step_ms * 1e-3, 0.0])
        frames.append(cam.capture(k * step_ms * MS)[0, -1].clone())
    return cam, frames


# --- geometry: motion -> events ------------------------------------------------------------


@pytest.mark.parametrize("z,v", [(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (1.0, 6.0)])
def test_translation_band_width_is_f_v_T_over_Z(z, v):
    cam, frames = run_sweep(z, v, steps=12)
    w = cam.fx * v * cam.cfg.frame_period_s / z
    widths = [int((f[cam.height // 2] != 0).sum()) for f in frames]
    assert set(widths) <= {math.floor(w), math.ceil(w)}, (w, widths)
    # pixel centres crossed in total = floor(edge position) difference
    x0 = cam.cx
    x_end = cam.cx + cam.fx * v * 12 * cam.cfg.frame_period_s / z
    assert sum(widths) == math.floor(x_end) - math.floor(x0)


def test_band_value_is_floor_log_ratio_over_C_and_uniform_over_rows():
    cam, frames = run_sweep(2.0, 4.0, steps=5)
    expected = math.floor(math.log(0.8 / 0.2) / 0.25)
    f = frames[-1]
    assert set(torch.unique(f).tolist()) == {0, expected}
    # lateral translation past a fronto-parallel wall shifts every row equally
    cols = (f != 0).any(dim=0)
    assert torch.equal((f != 0), cols.expand_as(f))


def test_opposite_motion_gives_opposite_polarity():
    _, fwd = run_sweep(2.0, 4.0, steps=4)
    _, back = run_sweep(2.0, -4.0, steps=4)
    assert fwd[-1].max() > 0 and fwd[-1].min() == 0
    assert back[-1].min() < 0 and back[-1].max() == 0


@pytest.mark.parametrize("omega", [1.0, 3.0])
def test_yaw_band_matches_tan_projection_with_interpolated_samples(omega):
    """Pure yaw, 30 ms captures, 10 ms frames: two of three samples are slerp-interpolated."""
    scene = edge_wall(3.0)
    cam = make_cam(scene, num_frames=3)
    set_albedo(cam, 0.2, 0.8)
    pose = Pose(cam)
    cam.capture(0)
    T, total, step = cam.cfg.frame_period_s, 0, 30
    for k in range(1, 6):
        pose.set(yaw=omega * k * step * 1e-3)
        frames = cam.capture(k * step * MS)[0]
        for j in range(3):  # the three frames closed by this capture, oldest first
            t1 = ((k - 1) * 3 + j + 1) * T
            lo = cam.cx + cam.fx * math.tan(omega * (t1 - T))
            hi = cam.cx + cam.fx * math.tan(omega * t1)
            width = int((frames[j, cam.height // 2] != 0).sum())
            assert width == math.floor(hi) - math.floor(lo), (k, j, width, hi - lo)
            total += width
    assert total == math.floor(cam.cx + cam.fx * math.tan(omega * 15 * T)) - math.floor(cam.cx)


def test_static_camera_is_silent():
    _, frames = run_sweep(2.0, 0.0, steps=5)
    assert all(int(f.abs().sum()) == 0 for f in frames)


def test_output_does_not_depend_on_how_time_is_split_into_env_steps():
    """Randomized substeps change the capture times, not the frame boundaries."""
    def run(step_ms):
        scene = SyntheticScene([random_forest(np.random.default_rng(0), 25)], DEV)
        torch.manual_seed(0)
        cam = make_cam(scene, num_frames=6)
        pose = Pose(cam)
        pose.set(xyz=[0.0, 0.0, 1.5])  # start on the trajectory: it must be linear in t throughout
        cam.capture(0)
        got, t = {}, 0
        for dt in step_ms:
            t += dt
            s = t * 1e-3
            pose.set(xyz=[3.0 * s, 0.5 * s, 1.5], yaw=0.8 * s)
            frames = cam.capture(t * MS)[0]
            latest = t // 10
            for j in range(6):
                got[latest - 5 + j] = frames[j].clone()
        return got

    a = run([30] * 10)
    b = run([10, 40, 20, 50, 0, 30, 30, 40, 10, 20, 50])
    common = [i for i in a if i in b and i > 0]
    assert len(common) >= 25
    assert sum(int((a[i] != 0).sum()) for i in common) > 10_000  # the run produced events
    # float32 lerp/slerp at different alphas can flip a pixel sitting exactly on an edge
    mismatch = sum(int((a[i] != b[i]).sum()) for i in common)
    assert mismatch <= 1e-4 * len(common) * a[common[0]].numel(), mismatch


def test_oversampling_changes_net_counts_by_at_most_one():
    """Frame-boundary sampling vs 16x oversampling: totals agree to +-1 per pixel (C+ = C-)."""
    def run(oversample):
        scene = SyntheticScene([random_forest(np.random.default_rng(1), 30)], DEV)
        torch.manual_seed(1)
        cam = make_cam(scene, num_frames=3, oversample=oversample,
                       ambient_range=[0.2, 0.2], diffuse_range=[0.7, 0.7])
        pose = Pose(cam)
        pose.set(xyz=[0.0, 0.0, 1.5])
        cam.capture(0)
        total = torch.zeros((cam.height, cam.width), dtype=torch.int32, device=DEV)
        per_frame = []
        for k in range(1, 21):
            s = k * 0.03
            pose.set(xyz=[4.0 * s, 0.8 * math.sin(3 * s), 1.5 + 0.3 * math.sin(5 * s)],
                     yaw=0.6 * math.sin(2 * s))
            f = cam.capture(k * 30 * MS)[0].to(torch.int32)
            total += f.sum(0)
            per_frame.append(f)
        return total, torch.stack(per_frame)

    t1, f1 = run(1)
    t16, f16 = run(16)
    assert int((t1 - t16).abs().max()) <= 1
    assert int((f1 - f16).abs().max()) <= 2
    assert int((f1 != 0).sum()) > 1000  # the trajectory actually produced events


# --- photometry -------------------------------------------------------------------------


def _single_sided_wall(x):
    """Two triangles in the plane world-x = x, wound so the geometric normal is -x."""
    v = np.array([[x, -50, -50], [x, 50, -50], [x, 50, 50], [x, -50, 50]], dtype=float)
    m = tm.Trimesh(vertices=v, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if m.face_normals[0][0] > 0:
        m.invert()
    return m


def test_lambert_intensity_and_back_face_flip():
    """Front face lit head-on: I = rho(a+b). Same plane from behind, light on the far side: I = rho*a."""
    scene = SyntheticScene([[_single_sided_wall(2.0)]] * 2, DEV)
    cam = make_cam(scene, num_envs=2, ambient_range=[0.2, 0.2], diffuse_range=[0.6, 0.6])
    cam.albedo[:, 0] = 0.5
    cam.light[:] = torch.tensor([-1.0, 0.0, 0.0], device=DEV)  # towards the front (x < 2) side
    pose = Pose(cam, n=2)
    pose.pos[1, 0] = torch.tensor([4.0, 0.0, 0.0], device=DEV)
    pose.quat[1, 0] = torch.tensor(camera_quat(math.pi), dtype=torch.float32, device=DEV)
    cam.capture(0)
    I = torch.exp(cam.l_ref[:, cam.height // 2, cam.width // 2])
    assert I[0].item() == pytest.approx(0.5 * (0.2 + 0.6), rel=1e-5)
    assert I[1].item() == pytest.approx(0.5 * 0.2, rel=1e-5)


def test_lambert_cosine_falloff():
    scene = SyntheticScene([[_single_sided_wall(2.0)]], DEV)
    cam = make_cam(scene, ambient_range=[0.1, 0.1], diffuse_range=[0.8, 0.8])
    cam.albedo[:, 0] = 1.0
    ang = math.radians(60)
    cam.light[:] = torch.tensor([-math.cos(ang), 0.0, math.sin(ang)], device=DEV)
    Pose(cam)
    cam.capture(0)
    I = torch.exp(cam.l_ref[0, cam.height // 2, cam.width // 2]).item()
    assert I == pytest.approx(0.1 + 0.8 * math.cos(ang), rel=1e-5)


def test_depth_output_is_planar_depth():
    scene = edge_wall(3.0)
    cam = make_cam(scene, write_depth=True)
    Pose(cam)
    cam.capture(0)
    assert torch.allclose(cam.depth[0], torch.full_like(cam.depth[0], 3.0), atol=1e-4)


# --- noise ------------------------------------------------------------------------------


def _sky_only():
    box = tm.creation.box(extents=[0.1, 0.1, 0.1])
    box.apply_translation([-5.0, 0.0, 0.0])  # behind the camera: every ray sees the sky
    return SyntheticScene([[box]], DEV)


@pytest.mark.parametrize("oversample", [1, 4])
def test_background_activity_rate(oversample):
    """Net count per frame is Skellam: mean 0, variance = rate * T * (1 - I)."""
    cam = make_cam(_sky_only(), oversample=oversample, ba_rate_hz_range=[200.0, 200.0],
                   sky_intensity_range=[0.5, 0.5])
    Pose(cam)
    cam.capture(0)
    vals = torch.stack([cam.capture(k * 10 * MS)[0, -1].float() for k in range(1, 21)])
    lam = 200.0 * 0.010 * (1 - 0.5)
    assert vals.mean().item() == pytest.approx(0.0, abs=0.01)
    assert vals.var().item() == pytest.approx(lam, rel=0.02)


def test_no_background_activity_at_full_brightness():
    cam = make_cam(_sky_only(), ba_rate_hz_range=[200.0, 200.0], sky_intensity_range=[1.0, 1.0])
    Pose(cam)
    cam.capture(0)
    assert all(int(cam.capture(k * 10 * MS).abs().sum()) == 0 for k in range(1, 6))


def test_pixel_mismatch_is_fixed_within_an_episode_and_changes_across():
    """Same sweep twice in one episode -> identical frames; after reset -> different."""
    scene = edge_wall(2.0)
    cam = make_cam(scene, c_sigma_pixel=0.05, c_min=0.05, c_max=1.0)
    set_albedo(cam, 0.2, 0.8)
    pose = Pose(cam)

    def sweep(t0):
        out = []
        for k in range(1, 6):
            pose.set(xyz=[0.0, 0.04 * k, 0.0])
            out.append(cam.capture(t0 + k * 10 * MS)[0, -1].clone())
        pose.set(xyz=[0.0, 0.0, 0.0])
        return torch.stack(out)

    cam.capture(0)
    a = sweep(0)
    cam.reinit[:] = 1  # back to the start pose without a new episode
    cam.capture(60 * MS)
    b = sweep(60 * MS)
    assert torch.equal(a, b)
    vals = torch.unique(a[a != 0])
    assert len(vals) > 1  # thresholds really vary across pixels

    cam.reset_idx([0])
    set_albedo(cam, 0.2, 0.8)
    cam.capture(120 * MS)
    c = sweep(120 * MS)
    assert not torch.equal(a, c)


# --- DIFF3D encoding and episode handling ----------------------------------------------------


@pytest.mark.parametrize("bits,overflow,expected", [(8, "wrap", 18), (4, "wrap", 2), (4, "saturate", 7),
                                                    (2, "wrap", -2)])
def test_diff3d_counter_width_and_overflow(bits, overflow, expected):
    """rho 0.01 -> 1.0 is ln(100)/0.25 = 18.4 -> 18 events; 4-bit wrap: (18+8)%16-8 = 2."""
    _, frames = run_sweep(2.0, 4.0, steps=3, rho=(0.01, 1.0), log_eps=1e-4,
                          diff3d_bits=bits, diff3d_overflow=overflow)
    assert set(torch.unique(frames[-1]).tolist()) == {0, expected}


def test_reset_does_not_emit_a_teleport_burst():
    scene = edge_wall(2.0)
    cam = make_cam(scene, num_frames=2)
    set_albedo(cam, 0.2, 0.8)
    pose = Pose(cam)
    cam.capture(0)
    pose.set(xyz=[0.0, 0.1, 0.0])
    assert int(cam.capture(20 * MS).abs().sum()) > 0
    pose.set(xyz=[0.0, -3.0, 0.5], yaw=0.4)  # teleport, as an env reset does
    cam.reset_idx([0])
    assert int(cam.capture(40 * MS).abs().sum()) == 0
    assert int(cam.capture(60 * MS).abs().sum()) == 0


def test_clock_must_not_go_backwards():
    cam = make_cam(edge_wall(2.0))
    Pose(cam)
    cam.capture(10 * MS)
    with pytest.raises(ValueError):
        cam.capture(5 * MS)


# --- lens --------------------------------------------------------------------------------


def test_pinhole_lut_matches_upstream_warp_camera_rays():
    """Upstream: ray = normalize(K^-1 (x, y, 1)) with u0 = W/2, alpha_u = u0 / tan(hfov/2)."""
    W, H, hfov = 320, 240, math.radians(87.0)
    u0, v0 = W / 2, H / 2
    f = u0 / math.tan(hfov / 2)
    vfov = 2 * math.atan(H / (2 * f))
    av = v0 / math.tan(vfov / 2)
    lut = ray_lut(W, H, f, av, u0, v0, "pinhole")
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    ref = np.stack([(xs - u0) / f, (ys - v0) / av, np.ones_like(xs, dtype=float)], -1)
    ref /= np.linalg.norm(ref, axis=-1, keepdims=True)
    assert np.allclose(lut, ref, atol=1e-6)


def test_kb4_with_zero_coefficients_is_equidistant():
    fx, fy, cx, cy = intrinsics_from_fov(320, 320, 104.0, "diagonal", "kb4")
    lut = ray_lut(320, 320, fx, fy, cx, cy, "kb4", (0.0, 0.0, 0.0, 0.0))
    xs, ys = np.meshgrid(np.arange(320), np.arange(320))
    r = np.hypot((xs - cx) / fx, (ys - cy) / fy)
    assert np.allclose(np.arccos(np.clip(lut[..., 2], -1, 1)), r, atol=1e-5)
    # corner ray sits at half the diagonal FOV
    corner = math.degrees(math.acos(lut[0, 0, 2]))
    assert corner == pytest.approx(52.0 * math.hypot(cx, cy) / math.hypot(160, 160), rel=1e-3)


def test_pinhole_fov_types():
    fh = intrinsics_from_fov(320, 320, 104.0, "horizontal")[0]
    fd = intrinsics_from_fov(320, 320, 104.0, "diagonal")[0]
    assert math.degrees(2 * math.atan(160 / fh)) == pytest.approx(104.0)
    assert math.degrees(2 * math.atan(160 / fd)) == pytest.approx(84.29, abs=0.01)


# --- aerial_gym integration ----------------------------------------------------------------


def test_aerial_gym_env_produces_event_frames(tmp_path):
    """Real sim with the f450_event_cam robot. Subprocess: Isaac Gym allows one sim per process."""
    import os
    import subprocess
    import sys

    out = tmp_path / "ev.mp4"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "tools.event_cam_demo", "--scene", "aerial_gym", "--num_envs", "2",
         "--seconds", "1", "--num_obstacles", "20", "--out", str(out)],
        cwd=repo, capture_output=True, text=True, timeout=900,
    )
    assert "per-step sum stats" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert out.exists()


# --- polarity convention: ON (+) = brighter, OFF (-) = darker --------------------------------


def test_polarity_follows_brightness_change_without_motion():
    cam = make_cam(SyntheticScene([[wall_x(2.0, -50, 50, -50, 50)]], DEV))
    cam.albedo[:, 0] = 0.3
    Pose(cam)
    cam.capture(0)
    cam.albedo[:, 0] = 0.6  # ln 2 / 0.25 = 2.77 -> +2 everywhere
    assert set(torch.unique(cam.capture(10 * MS)[0, -1]).tolist()) == {2}
    cam.albedo[:, 0] = 0.3
    assert set(torch.unique(cam.capture(20 * MS)[0, -1]).tolist()) == {-2}


def test_dark_object_moving_over_bright_background_is_off_ahead_on_behind():
    """Leading edge covers bright wall (darker -> OFF); trailing edge uncovers it (brighter -> ON)."""
    scene = SyntheticScene([[wall_x(4.0, -50, 50, -50, 50), wall_x(2.0, -0.25, 0.25, -50, 50)]], DEV)
    cam = make_cam(scene, write_depth=True)
    set_albedo(cam, 0.9, 0.1)
    pose = Pose(cam)
    cam.capture(0)
    pose.set(xyz=[0.0, 0.05, 0.0])  # camera moves left -> post moves right in the image
    row = cam.capture(10 * MS)[0, -1, cam.height // 2].to(torch.int32)
    post = torch.nonzero(cam.depth[0, cam.height // 2] < 3.0).flatten()
    n = math.floor(math.log(0.9 / 0.1) / 0.25)
    on, off = torch.nonzero(row > 0).flatten(), torch.nonzero(row < 0).flatten()
    assert set(row[on].tolist()) == {n} and set(row[off].tolist()) == {-n}
    assert int(on.max()) < int(post.min())  # ON trails the post
    assert int(post.min()) <= int(off.min()) and int(off.max()) <= int(post.max())  # OFF inside its new extent, at the front


# --- procedural texture ----------------------------------------------------------------------

TEX = dict(texture_enabled=True, texture_scale_range=[0.1, 0.1], texture_contrast_range=[0.3, 0.3],
           texture_probability=1.0)


def _log_albedo_on_wall(z, n=8, **overrides):
    """ln(I) over a head-on wall at distance z, I = rho exactly; n envs = n independent patterns."""
    scene = SyntheticScene([[wall_x(z, -50, 50, -50, 50)]] * n, DEV)
    cam = make_cam(scene, num_envs=n, **{**TEX, **overrides})
    cam.albedo[:, 0] = 0.05  # low enough that the rho <= 1 clip never binds
    Pose(cam, n=n)
    cam.capture(0)
    return cam, cam.l_ref.clone()


def test_texture_contrast_is_the_std_of_log_albedo():
    _, logi = _log_albedo_on_wall(1.0)  # 1 m: all four octaves resolved
    assert logi.std().item() == pytest.approx(0.3, rel=0.1)
    assert logi.mean().item() == pytest.approx(math.log(0.05), abs=0.05)


def test_texture_fades_with_distance_instead_of_aliasing():
    _, near = _log_albedo_on_wall(1.0)
    _, far = _log_albedo_on_wall(12.0)  # footprint 6.8 cm: octaves < 10 cm gone, 10 cm half-faded
    assert far.std().item() < 0.5 * near.std().item()


def test_textured_flat_wall_fires_inside_its_silhouette():
    def active_fraction(texture):
        cfg = TEX if texture else {}
        cam = make_cam(SyntheticScene([[wall_x(2.0, -50, 50, -50, 50)]], DEV), **cfg)
        cam.albedo[:, 0] = 0.5
        pose = Pose(cam)
        cam.capture(0)
        pose.set(xyz=[0.0, 0.04, 0.0])  # 4 m/s sideways for 10 ms
        return (cam.capture(10 * MS)[0, -1] != 0).float().mean().item()

    assert active_fraction(texture=False) == 0.0
    assert active_fraction(texture=True) > 0.1


def test_texture_is_world_anchored():
    """Static camera: silent. Leave and come back within an episode: same intensities."""
    cam = make_cam(SyntheticScene([[wall_x(2.0, -50, 50, -50, 50)]], DEV), **TEX)
    pose = Pose(cam)
    cam.capture(0)
    first = cam.l_ref[0].clone()
    assert all(int(cam.capture(k * 10 * MS).abs().sum()) == 0 for k in range(1, 4))
    pose.set(xyz=[0.0, 0.5, 0.0])
    cam.capture(40 * MS)
    pose.set(xyz=[0.0, 0.0, 0.0])
    cam.reinit[:] = 1
    cam.capture(50 * MS)
    assert torch.allclose(cam.l_ref[0], first)


def test_texture_pattern_changes_across_episodes():
    cam, first = _log_albedo_on_wall(2.0, n=1)
    cam.reset_idx([0])
    cam.albedo[:, 0] = 0.05
    cam.capture(10 * MS)
    assert not torch.allclose(cam.l_ref, first, atol=1e-3)


def test_texture_probability_mixes_textured_and_flat_assets():
    """Many assets per env, p = 0.5: about half get sigma = 0 (flat), half keep their contrast."""
    walls = [wall_x(2.0 + 0.01 * i, -50, 50, -50, 50) for i in range(200)]
    cam = make_cam(SyntheticScene([walls], DEV), **{**TEX, "texture_probability": 0.5})
    flat = (cam.tex_sigma[0, :200] == 0).float().mean().item()
    assert 0.35 < flat < 0.65
    kept = cam.tex_sigma[0, :200][cam.tex_sigma[0, :200] > 0]
    assert torch.allclose(kept, torch.full_like(kept, 0.3))

    cam.tex_sigma[0, 0] = 0.0  # the visible wall (asset 0) flat: no texture anywhere in the image
    cam.albedo[:, 0] = 0.5
    Pose(cam)
    cam.capture(0)
    assert torch.allclose(cam.l_ref[0], torch.full_like(cam.l_ref[0], math.log(0.5)))


# --- smooth normals and supersampling ----------------------------------------------------------


def _sphere_shading_error(smooth):
    """Mean |ln I - ln I_true| over a tessellated sphere, against exact sphere normals."""
    from sensors.event_camera.synthetic import _R_OPTICAL_TO_WORLD

    centre = np.array([3.0, 0.0, 0.0])
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.apply_translation(centre)
    cam = make_cam(SyntheticScene([[sphere]], DEV), smooth_normals=smooth, write_depth=True,
                   ambient_range=[0.2, 0.2], diffuse_range=[0.8, 0.8])
    cam.albedo[:, 0] = 1.0
    light = np.array([-1.0, 0.4, 0.5]) / np.linalg.norm([-1.0, 0.4, 0.5])
    cam.light[:] = torch.tensor(light, dtype=torch.float32, device=DEV)
    Pose(cam)
    cam.capture(0)
    d_world = cam.ray_lut_np @ _R_OPTICAL_TO_WORLD.T
    depth = cam.depth[0].cpu().numpy()
    hit = depth < cam.cfg.max_range
    t = depth / cam.ray_lut_np[..., 2]
    n_true = t[..., None] * d_world - centre
    n_true /= np.linalg.norm(n_true, axis=-1, keepdims=True)
    i_true = 0.2 + 0.8 * np.clip(n_true @ light, 0.0, None)
    err = np.abs(cam.l_ref[0].cpu().numpy() - np.log(i_true))
    return err[hit].mean()


def test_smooth_normals_match_exact_sphere_shading():
    flat, smooth = _sphere_shading_error(False), _sphere_shading_error(True)
    assert smooth < 0.3 * flat, (flat, smooth)  # also fails if the barycentric order is wrong
    assert smooth < 0.02


def test_crease_keeps_box_faces_flat():
    """A box's corners are shared by perpendicular faces; averaging them would curve the face."""
    box = tm.creation.box(extents=[1.0, 1.0, 1.0])
    box.apply_translation([3.0, 0.0, 0.0])
    cam = make_cam(SyntheticScene([[box]], DEV), ambient_range=[0.2, 0.2], diffuse_range=[0.8, 0.8])
    cam.albedo[:, 0] = 1.0
    cam.light[:] = torch.tensor([-0.6, 0.48, 0.64], device=DEV)
    Pose(cam)
    cam.capture(0)
    c, half = cam.height // 2, 20  # the front face spans about +-35 px around the centre
    face = cam.l_ref[0, c - half:c + half, c - half:c + half]
    assert float(face.max() - face.min()) < 1e-5


def test_smooth_normals_follow_geometry_moved_on_reset():
    """aerial_gym re-places obstacles on reset; the corner normals must be recomputed."""
    scene = SyntheticScene([[wall_x(2.0, -1, 1, -1, 1)]], DEV)
    cam = make_cam(scene, ambient_range=[0.0, 0.0], diffuse_range=[1.0, 1.0])
    cam.albedo[:, 0] = 1.0
    cam.light[:] = torch.tensor([-1.0, 0.0, 0.0], device=DEV)
    Pose(cam)
    cam.capture(0)
    c = cam.height // 2
    assert torch.exp(cam.l_ref[0, c, c]).item() == pytest.approx(1.0, rel=1e-4)

    points = scene._buffers[0][0]  # rotate the wall 30 deg about z through its centre
    a = math.radians(30.0)
    R = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                      [0.0, 0.0, 1.0]], device=DEV)
    pivot = torch.tensor([2.0, 0.0, 0.0], device=DEV)
    points[:] = (points - pivot) @ R.T + pivot
    scene.meshes[0].refit()
    cam.reset_idx([0])
    cam.albedo[:, 0] = 1.0
    cam.light[:] = torch.tensor([-1.0, 0.0, 0.0], device=DEV)
    cam.capture(10 * MS)
    assert torch.exp(cam.l_ref[0, c, c]).item() == pytest.approx(math.cos(a), rel=1e-3)


def test_supersampling_averages_partial_pixel_coverage():
    """Edge through a pixel centre: 2x2 sub-rays split 2/2, so I is the mean of both sides."""
    z = 2.0
    scene = edge_wall(z)
    cam = make_cam(scene, supersample=2)
    set_albedo(cam, 0.2, 0.8)
    pose = Pose(cam)
    pose.set(xyz=[0.0, 0.5 * z / cam.fx, 0.0])  # shift the edge half a pixel, onto column 160
    cam.capture(0)
    row = torch.exp(cam.l_ref[0, cam.height // 2]).cpu()
    assert row[160].item() == pytest.approx(0.5, rel=1e-4)
    assert row[159].item() == pytest.approx(0.8, rel=1e-4)
    assert row[161].item() == pytest.approx(0.2, rel=1e-4)


def test_supersampling_conserves_events_per_crossed_pixel():
    """Partial coverage spreads an edge crossing over several frames, not its total count."""
    cam, frames = run_sweep(2.0, 4.0, steps=10, supersample=2)
    total = torch.stack(frames).to(torch.int32).sum(0)[cam.height // 2].cpu()
    x_end = cam.cx + cam.fx * 4.0 * 0.1 / 2.0  # edge position after 100 ms
    last = round(x_end)  # pixel the edge sits in: sub-rays at last -+ 0.25
    assert x_end - (last - 0.25) > 0 and (last + 0.25) - x_end > 0  # exactly half covered
    full = total[160:last]
    assert len(full) > 20
    assert set(full.tolist()) == {math.floor(math.log(0.8 / 0.2) / 0.25)}  # 0.2 -> 0.8: 5
    assert total[last].item() == math.floor(math.log(0.5 / 0.2) / 0.25)  # 0.2 -> 0.5: 3
    assert int((total[:160] != 0).sum()) == 0 and int((total[last + 1:] != 0).sum()) == 0
