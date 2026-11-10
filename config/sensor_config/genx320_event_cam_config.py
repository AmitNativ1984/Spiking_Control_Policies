from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig


class GenX320EventCamConfig(BaseSensorConfig):
    """Prophesee GenX320 event camera (320x320, M6 104 deg lens) emitting DIFF3D frames.

    Event model, noise and scene parameters are v2e/EVIS-style placeholders: no real
    GenX320 recording exists yet to calibrate them against. Ranges are [lo, hi] and are
    resampled per env every episode; set lo == hi to fix a value.
    """

    num_sensors = 1
    sensor_type = "event_camera"

    # --- optics --------------------------------------------------------------------------
    width = 320
    height = 320
    lens_model = "pinhole"  # "pinhole" | "kb4" (Kannala-Brandt; zero coeffs = equidistant)
    intrinsics = None  # (fx, fy, cx, cy) from a calibration; None -> derived from fov_deg
    kb4_coeffs = [0.0, 0.0, 0.0, 0.0]
    fov_deg = 104.0
    fov_type = "diagonal"  # the vendor page does not say which; diagonal is the usual lens spec
    max_range = 20.0  # ray far plane [m]; misses see the sky
    supersample = 1  # k x k sub-rays per pixel, averaged in linear intensity (cost x k^2)

    # --- DIFF3D output -------------------------------------------------------------------
    frame_period_s = 0.010  # DIFF3D integration period T
    num_frames = 3  # output = last num_frames closed frames, (N, num_frames, H, W) int8
    oversample = 1  # samples per frame; >1 only resolves intra-frame ON/OFF hysteresis (+-1)
    diff3d_bits = 8  # 2..8, as the sensor's pixellayout "0p/<bits>n"
    diff3d_overflow = "wrap"  # sensor default is roll-over; "saturate" clamps instead
    write_depth = False  # also write metric depth at the capture pose (+1 ray per pixel)

    # --- event model: L = log(max(I, eps)), N = floor(|L - L_ref| / C) -------------------
    log_eps = 1.0e-3
    c_on_mean_range = [0.15, 0.35]
    c_off_mean_range = [0.15, 0.35]
    c_sigma_pixel = 0.03  # per-pixel mismatch, fixed within an episode (v2e)
    c_sigma_temporal = 0.01  # redrawn every sample
    c_min = 0.10
    c_max = 0.50

    # --- background activity: Poisson, rate * (1 - I), uniform polarity --------------------
    ba_rate_hz_range = [0.5, 2.0]  # per pixel; v2e default is 1 Hz

    # --- scene: I = rho * (a + b * max(0, n . l)), intensities in [0, 1] -------------------
    albedo_range = [0.1, 1.0]  # rho, one per asset
    ambient_range = [0.1, 0.3]  # a
    diffuse_range = [0.5, 0.7]  # b
    sky_intensity_range = [0.5, 1.0]  # I where a ray hits nothing
    light_elevation_deg_range = [20.0, 80.0]  # directional light; azimuth is U(0, 360)
    max_assets_per_env = 1024
    smooth_normals = True  # interpolate corner normals; needs the env meshes (bind_meshes)
    crease_angle_deg = 40.0  # faces meeting at a sharper angle than this keep a hard edge

    # --- texture: rho *= exp(sigma * F(p / s)), F = unit-std fBm anchored to the hit point ----
    texture_enabled = False
    texture_scale_range = [0.2, 1.0]  # s [m], per asset per episode, log-uniform
    texture_contrast_range = [0.05, 0.5]  # sigma = std of ln(rho); compare with C (~0.25)
    texture_probability = 0.7  # chance each asset is textured this episode; the rest stay flat
    texture_octaves = 2  # wavelengths s, s/2, s/4, s/8
    texture_min_footprints = 2.0  # fade octaves whose wavelength < this many pixel footprints

    # --- mount (sensor frame -> robot base_link) -------------------------------------------
    euler_frame_rot_deg = [-90.0, 0.0, -90.0]  # optical (z forward) -> body (x forward)
    # upstream WarpSensor leaves the translation at zero unless randomize_placement is True,
    # so a fixed mount is expressed as min == max
    randomize_placement = True
    min_translation = [0.10, 0.0, -0.03]
    max_translation = [0.10, 0.0, -0.03]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    # --- fields upstream robot_manager / WarpSensor read for every warp camera -------------
    return_pointcloud = False
    segmentation_camera = False
    min_range = 0.0
    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
