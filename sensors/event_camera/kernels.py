"""Fused warp kernel: ray cast -> Lambert intensity -> log -> ESIM events -> DIFF3D.

One thread owns one pixel of one env and walks every sample instant that falls inside the
current env step, so L_ref, the accumulator and the RNG state stay in registers.
"""
import warp as wp


@wp.func
def render_intensity(
    mesh: wp.uint64,
    ro: wp.vec3,
    rd: wp.vec3,
    far: float,
    env: int,
    albedo: wp.array2d(dtype=float),
    light: wp.vec3,
    ambient: float,
    diffuse: float,
    sky: float,
):
    """(I, hit distance); distance is -1 for a miss."""
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    if wp.mesh_query_ray(mesh, ro, rd, far, t, u, v, sign, n, f):
        m = wp.mesh_get(mesh)
        # velocities[:, 1] carries the asset index within the env (velocities[:, 0] is segmentation)
        aid = wp.clamp(int(m.velocities[m.indices[f * 3]][1]), 0, albedo.shape[1] - 1)
        nn = wp.normalize(n)
        # winding is not guaranteed across assets: shade the side facing the camera
        if wp.dot(nn, rd) > 0.0:
            nn = -nn
        intensity = albedo[env, aid] * (ambient + diffuse * wp.max(0.0, wp.dot(nn, light)))
        return wp.vec2(intensity, t)
    return wp.vec2(sky, -1.0)


@wp.func
def diff3d_encode(count: int, half: int, wrap: int):
    """Signed count -> DIFF3D cell. half = 2^(bits-1); wrap mimics the sensor's roll-over."""
    v = count
    if wrap != 0:
        m = (count + half) % (2 * half)
        if m < 0:
            m = m + 2 * half
        v = m - half
    else:
        v = wp.clamp(count, -half, half - 1)
    return wp.int8(v)


@wp.func
def depth_value(hit_t: float, ray_z: float, far: float):
    if hit_t < 0.0:
        return far
    return hit_t * ray_z


@wp.kernel
def event_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    pos_prev: wp.array(dtype=wp.vec3),
    quat_prev: wp.array(dtype=wp.quat),
    pos_curr: wp.array(dtype=wp.vec3),
    quat_curr: wp.array(dtype=wp.quat),
    ray_lut: wp.array2d(dtype=wp.vec3),
    far: float,
    albedo: wp.array2d(dtype=float),
    light: wp.array(dtype=wp.vec3),
    ambient: wp.array(dtype=float),
    diffuse: wp.array(dtype=float),
    sky: wp.array(dtype=float),
    c_on_mean: wp.array(dtype=float),
    c_off_mean: wp.array(dtype=float),
    ba_rate: wp.array(dtype=float),
    env_seed: wp.array(dtype=wp.int32),
    reinit: wp.array(dtype=wp.int32),
    sigma_pixel: float,
    sigma_temporal: float,
    c_min: float,
    c_max: float,
    log_eps: float,
    n_samples: int,
    alpha0: float,
    dalpha: float,
    s0: int,
    oversample: int,
    num_frames: int,
    t_sample: float,
    step_seed: int,
    diff3d_half: int,
    wrap: int,
    l_ref: wp.array3d(dtype=float),
    acc: wp.array3d(dtype=wp.int32),
    ring: wp.array4d(dtype=wp.int8),
    depth: wp.array3d(dtype=float),
    write_depth: int,
):
    env, y, x = wp.tid()
    pix = y * ray_lut.shape[1] + x
    mesh = mesh_ids[env]
    d_cam = ray_lut[y, x]

    p0 = pos_prev[env]
    q0 = quat_prev[env]
    p1 = pos_curr[env]
    q1 = quat_curr[env]
    if q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0.0:
        q1 = wp.quat(-q1[0], -q1[1], -q1[2], -q1[3])

    seed = env_seed[env]
    lit = light[env]
    amb = ambient[env]
    dif = diffuse[env]
    sky_i = sky[env]

    # per-pixel threshold mismatch: same draw every call within an episode (seeded by pixel)
    fp = wp.rand_init(seed, pix)
    c_on_px = c_on_mean[env] + sigma_pixel * wp.randn(fp)
    c_off_px = c_off_mean[env] + sigma_pixel * wp.randn(fp)

    if reinit[env] != 0:
        r = render_intensity(mesh, p1, wp.normalize(wp.quat_rotate(q1, d_cam)), far, env,
                             albedo, lit, amb, dif, sky_i)
        l_ref[env, y, x] = wp.log(wp.max(r[0], log_eps))
        acc[env, y, x] = 0
        for b in range(num_frames):
            ring[env, b, y, x] = wp.int8(0)
        if write_depth != 0:
            depth[env, y, x] = depth_value(r[1], d_cam[2], far)
        return

    lref = l_ref[env, y, x]
    count = acc[env, y, x]
    st = wp.rand_init(seed ^ step_seed, pix)

    for k in range(n_samples):
        alpha = alpha0 + float(k) * dalpha
        ro = wp.lerp(p0, p1, alpha)
        q = wp.quat_slerp(q0, q1, alpha)
        r = render_intensity(mesh, ro, wp.normalize(wp.quat_rotate(q, d_cam)), far, env,
                             albedo, lit, amb, dif, sky_i)
        intensity = r[0]
        dl = wp.log(wp.max(intensity, log_eps)) - lref

        c_on = wp.clamp(c_on_px + sigma_temporal * wp.randn(st), c_min, c_max)
        c_off = wp.clamp(c_off_px + sigma_temporal * wp.randn(st), c_min, c_max)
        if dl > 0.0:
            n = int(wp.floor(dl / c_on))
            lref = lref + float(n) * c_on
            count = count + n
        else:
            n = int(wp.floor(-dl / c_off))
            lref = lref - float(n) * c_off
            count = count - n

        # background activity: Poisson, uniform polarity, more in the dark; does not touch L_ref
        lam = 0.5 * ba_rate[env] * t_sample * (1.0 - wp.clamp(intensity, 0.0, 1.0))
        count = count + int(wp.poisson(st, lam)) - int(wp.poisson(st, lam))

        s = s0 + k
        if s % oversample == 0:
            ring[env, (s / oversample) % num_frames, y, x] = diff3d_encode(count, diff3d_half, wrap)
            count = 0

    l_ref[env, y, x] = lref
    acc[env, y, x] = count

    if write_depth != 0:
        r = render_intensity(mesh, p1, wp.normalize(wp.quat_rotate(q1, d_cam)), far, env,
                             albedo, lit, amb, dif, sky_i)
        depth[env, y, x] = depth_value(r[1], d_cam[2], far)
