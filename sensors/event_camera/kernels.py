"""Fused warp kernel: ray cast -> Lambert intensity -> log -> ESIM events -> DIFF3D.

One thread owns one pixel of one env and walks every sample instant that falls inside the
current env step, so L_ref, the accumulator and the RNG state stay in registers.
"""
import warp as wp

# std of value_noise() (uniform [-1, 1] lattice, quintic trilinear), measured over 2e6 points
VALUE_NOISE_STD = wp.constant(0.4007)


@wp.struct
class Texture:
    """World-anchored fBm albedo texture: rho *= exp(sigma * F(p / scale)), F unit-std."""

    enabled: int
    scale: wp.array2d(dtype=float)  # (env, asset) feature size [m]
    sigma: wp.array2d(dtype=float)  # (env, asset) std of ln(rho)
    octaves: int
    norm: float  # VALUE_NOISE_STD * sqrt(sum_k 0.25^k), so F has unit std
    pixel_angle: float  # [rad]
    min_footprints: float  # an octave fades out below this many pixel footprints


@wp.struct
class Geometry:
    """Crease-aware smooth normals, one per triangle corner, for every env's mesh."""

    smooth: int
    corner_normals: wp.array(dtype=wp.vec3)  # 3 per face, faces of env e start at face_offset[e]
    face_offset: wp.array(dtype=wp.int32)


@wp.func
def lattice(seed: int, ix: int, iy: int, iz: int):
    st = wp.rand_init(seed, ix * 73856093 + iy * 19349663 + iz * 83492791)
    return wp.randf(st) * 2.0 - 1.0


@wp.func
def quintic(t: float):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
def value_noise(p: wp.vec3, seed: int):
    fx = wp.floor(p[0])
    fy = wp.floor(p[1])
    fz = wp.floor(p[2])
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    ux = quintic(p[0] - fx)
    uy = quintic(p[1] - fy)
    uz = quintic(p[2] - fz)
    x00 = wp.lerp(lattice(seed, ix, iy, iz), lattice(seed, ix + 1, iy, iz), ux)
    x10 = wp.lerp(lattice(seed, ix, iy + 1, iz), lattice(seed, ix + 1, iy + 1, iz), ux)
    x01 = wp.lerp(lattice(seed, ix, iy, iz + 1), lattice(seed, ix + 1, iy, iz + 1), ux)
    x11 = wp.lerp(lattice(seed, ix, iy + 1, iz + 1), lattice(seed, ix + 1, iy + 1, iz + 1), ux)
    return wp.lerp(wp.lerp(x00, x10, uy), wp.lerp(x01, x11, uy), uz)


@wp.func
def texture_log_albedo(tex: Texture, p: wp.vec3, env: int, aid: int, seed: int, footprint: float):
    """sigma * F(p): fBm with gain 0.5 per octave; octaves finer than the footprint are faded out."""
    scale = tex.scale[env, aid]
    acc = float(0.0)
    amp = float(1.0)
    freq = 1.0 / scale
    for k in range(tex.octaves):
        # weight 1 when the wavelength spans >= min_footprints footprints, 0 at half that
        r = (1.0 / freq) / (tex.min_footprints * footprint)
        w = wp.clamp(2.0 * r - 1.0, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        if w > 0.0:
            oseed = seed + aid * 1000003 + k * 7919
            # random lattice offset: value noise is not stationary within a cell, and a surface
            # lying on a lattice plane (e.g. a wall at x = 1 m, s = 0.1 m) would get ~12% more contrast
            st = wp.rand_init(oseed, 0)
            off = wp.vec3(wp.randf(st), wp.randf(st), wp.randf(st)) * 1000.0
            acc = acc + amp * w * value_noise(p * freq + off, oseed)
        amp = amp * 0.5
        freq = freq * 2.0
    return tex.sigma[env, aid] * acc / tex.norm


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
    tex: Texture,
    geom: Geometry,
    seed: int,
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
        nf = wp.normalize(n)
        # winding is not guaranteed across assets: shade the side facing the camera
        facing = float(1.0)
        if wp.dot(nf, rd) > 0.0:
            facing = -1.0
        nn = nf
        if geom.smooth != 0:
            # barycentric convention of mesh_query_ray: p = u*p0 + v*p1 + (1-u-v)*p2
            c = 3 * (geom.face_offset[env] + f)
            ns = (u * geom.corner_normals[c] + v * geom.corner_normals[c + 1]
                  + (1.0 - u - v) * geom.corner_normals[c + 2])
            if wp.length(ns) > 0.0:
                nn = wp.normalize(ns)
        nn = facing * nn
        rho = albedo[env, aid]
        if tex.enabled != 0 and tex.sigma[env, aid] > 0.0:
            # footprint grows at grazing incidence (geometric normal); 0.05 caps it near edge-on
            footprint = t * tex.pixel_angle / wp.max(wp.abs(wp.dot(nf, rd)), 0.05)
            rho = wp.min(rho * wp.exp(texture_log_albedo(tex, ro + t * rd, env, aid, seed, footprint)), 1.0)
        intensity = rho * (ambient + diffuse * wp.max(0.0, wp.dot(nn, light)))
        return wp.vec2(intensity, t)
    return wp.vec2(sky, -1.0)


@wp.func
def pixel_intensity(
    mesh: wp.uint64,
    ro: wp.vec3,
    q: wp.quat,
    ray_lut: wp.array3d(dtype=wp.vec3),
    y: int,
    x: int,
    far: float,
    env: int,
    albedo: wp.array2d(dtype=float),
    light: wp.vec3,
    ambient: float,
    diffuse: float,
    sky: float,
    tex: Texture,
    geom: Geometry,
    seed: int,
):
    """Box-filtered linear intensity over the pixel's sub-rays (a pixel integrates light, then logs)."""
    total = float(0.0)
    for i in range(ray_lut.shape[2]):
        rd = wp.normalize(wp.quat_rotate(q, ray_lut[y, x, i]))
        r = render_intensity(mesh, ro, rd, far, env, albedo, light, ambient, diffuse, sky, tex, geom, seed)
        total = total + r[0]
    return total / float(ray_lut.shape[2])


@wp.func
def face_cross(mesh: wp.uint64, f: int):
    m = wp.mesh_get(mesh)
    p0 = m.points[m.indices[f * 3]]
    return wp.cross(m.points[m.indices[f * 3 + 1]] - p0, m.points[m.indices[f * 3 + 2]] - p0)


@wp.kernel
def corner_normals_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    dirty: wp.array(dtype=wp.int32),
    face_offset: wp.array(dtype=wp.int32),
    num_faces: wp.array(dtype=wp.int32),
    vert_offset: wp.array(dtype=wp.int32),
    adj_start: wp.array(dtype=wp.int32),
    adj_faces: wp.array(dtype=wp.int32),
    cos_crease: float,
    out: wp.array(dtype=wp.vec3),
):
    """Corner normal = area-weighted sum of the faces at that vertex within the crease angle."""
    env, fl = wp.tid()
    if dirty[env] == 0 or fl >= num_faces[env]:
        return
    mesh = mesh_ids[env]
    m = wp.mesh_get(mesh)
    own = face_cross(mesh, fl)
    own_n = wp.vec3()
    if wp.length(own) > 0.0:
        own_n = wp.normalize(own)
    for c in range(3):
        g = vert_offset[env] + m.indices[fl * 3 + c]
        acc = wp.vec3()
        for a in range(adj_start[g], adj_start[g + 1]):
            other = face_cross(mesh, adj_faces[a])
            if wp.length(other) > 0.0:
                if wp.dot(wp.normalize(other), own_n) >= cos_crease:
                    acc = acc + other
        res = own_n
        if wp.length(acc) > 0.0:
            res = wp.normalize(acc)
        out[3 * (face_offset[env] + fl) + c] = res


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
    ray_lut: wp.array3d(dtype=wp.vec3),
    depth_lut: wp.array2d(dtype=wp.vec3),
    far: float,
    albedo: wp.array2d(dtype=float),
    tex: Texture,
    geom: Geometry,
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
    pix = y * depth_lut.shape[1] + x
    mesh = mesh_ids[env]
    d_cam = depth_lut[y, x]

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
        i0 = pixel_intensity(mesh, p1, q1, ray_lut, y, x, far, env, albedo, lit, amb, dif, sky_i,
                             tex, geom, seed)
        l_ref[env, y, x] = wp.log(wp.max(i0, log_eps))
        acc[env, y, x] = 0
        for b in range(num_frames):
            ring[env, b, y, x] = wp.int8(0)
        if write_depth != 0:
            r = render_intensity(mesh, p1, wp.normalize(wp.quat_rotate(q1, d_cam)), far, env,
                                 albedo, lit, amb, dif, sky_i, tex, geom, seed)
            depth[env, y, x] = depth_value(r[1], d_cam[2], far)
        return

    lref = l_ref[env, y, x]
    count = acc[env, y, x]
    st = wp.rand_init(seed ^ step_seed, pix)

    for k in range(n_samples):
        alpha = alpha0 + float(k) * dalpha
        ro = wp.lerp(p0, p1, alpha)
        q = wp.quat_slerp(q0, q1, alpha)
        intensity = pixel_intensity(mesh, ro, q, ray_lut, y, x, far, env, albedo, lit, amb, dif,
                                    sky_i, tex, geom, seed)
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
                             albedo, lit, amb, dif, sky_i, tex, geom, seed)
        depth[env, y, x] = depth_value(r[1], d_cam[2], far)
