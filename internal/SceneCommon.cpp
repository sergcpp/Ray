#include "SceneCommon.h"

#include "AtmosphereRef.h"
#include "Core.h"

namespace Ray {
Ref::fvec4 rgb_to_rgbe(const Ref::fvec4 &rgb) {
    const float max_component = fmaxf(fmaxf(rgb.get<0>(), rgb.get<1>()), rgb.get<2>());
    if (max_component < 1e-32) {
        return Ref::fvec4{0.0f};
    }

    int exponent;
    const float factor = frexpf(max_component, &exponent) * 256.0f / max_component;

    return Ref::fvec4{rgb.get<0>() * factor, rgb.get<1>() * factor, rgb.get<2>() * factor, float(exponent + 128)};
}
} // namespace Ray

void Ray::SceneCommon::GetEnvironment(environment_desc_t &env) {
    std::shared_lock<std::shared_timed_mutex> lock(mtx_);

    memcpy(env.env_col, env_.env_col, 3 * sizeof(float));
    env.env_map = TextureHandle{env_.env_map};
    memcpy(env.back_col, env_.back_col, 3 * sizeof(float));
    env.back_map = TextureHandle{env_.back_map};
    env.env_map_rotation = env_.env_map_rotation;
    env.back_map_rotation = env_.back_map_rotation;
    env.importance_sample = env_.importance_sample;
    env.envmap_resolution = env_.envmap_resolution;
    env.atmosphere = env_.atmosphere;
}

void Ray::SceneCommon::SetEnvironment(const environment_desc_t &env) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    memcpy(env_.env_col, env.env_col, 3 * sizeof(float));
    env_.env_map = env.env_map._index;
    memcpy(env_.back_col, env.back_col, 3 * sizeof(float));
    env_.back_map = env.back_map._index;
    env_.env_map_rotation = env.env_map_rotation;
    env_.back_map_rotation = env.back_map_rotation;
    env_.importance_sample = env.importance_sample;
    env_.envmap_resolution = env.envmap_resolution;
    env_.atmosphere = env.atmosphere;

    UpdateSkyTransmittanceLUT(env_.atmosphere);
    UpdateMultiscatterLUT(env_.atmosphere);
}

Ray::CameraHandle Ray::SceneCommon::AddCamera(const camera_desc_t &c) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> al = cams_.Allocate(1);
    const Ray::CameraHandle i = {al.first, al.second};

    SetCamera_nolock(i, c);
    if (current_cam_ == InvalidCameraHandle) {
        current_cam_ = i;
    }
    return i;
}

void Ray::SceneCommon::GetCamera(const CameraHandle i, camera_desc_t &c) const {
    std::shared_lock<std::shared_timed_mutex> lock(mtx_);

    const camera_t &cam = cams_[i._index];
    c.type = cam.type;
    c.view_transform = cam.view_transform;
    c.exposure = cam.exposure;
    c.gamma = cam.gamma;
    if (c.type != eCamType::Geo) {
        c.filter = cam.filter;
        c.filter_width = cam.filter_width;
        memcpy(&c.origin[0], &cam.origin[0], 3 * sizeof(float));
        memcpy(&c.fwd[0], &cam.fwd[0], 3 * sizeof(float));
        memcpy(&c.up[0], &cam.up[0], 3 * sizeof(float));
        memcpy(&c.shift[0], &cam.shift[0], 2 * sizeof(float));
        c.fov = cam.fov;
        c.focus_distance = cam.focus_distance;
        c.focal_length = cam.focal_length;
        c.fstop = cam.fstop;
        c.sensor_height = cam.sensor_height;
        c.lens_rotation = cam.lens_rotation;
        c.lens_ratio = cam.lens_ratio;
        c.lens_blades = cam.lens_blades;
        c.clip_start = cam.clip_start;
        c.clip_end = cam.clip_end;
    } else {
        c.mi_index = cam.mi_index;
        c.uv_index = cam.uv_index;
    }

    c.lighting_only = (cam.pass_settings.flags & ePassFlags::LightingOnly);
    c.skip_direct_lighting = (cam.pass_settings.flags & ePassFlags::SkipDirectLight);
    c.skip_indirect_lighting = (cam.pass_settings.flags & ePassFlags::SkipIndirectLight);
    c.no_background = (cam.pass_settings.flags & ePassFlags::NoBackground);
    c.output_sh = (cam.pass_settings.flags & ePassFlags::OutputSH);

    c.max_diff_depth = cam.pass_settings.max_diff_depth;
    c.max_spec_depth = cam.pass_settings.max_spec_depth;
    c.max_refr_depth = cam.pass_settings.max_refr_depth;
    c.max_transp_depth = cam.pass_settings.max_transp_depth;
    c.max_total_depth = cam.pass_settings.max_total_depth;
    c.min_total_depth = cam.pass_settings.min_total_depth;
    c.min_transp_depth = cam.pass_settings.min_transp_depth;

    c.clamp_direct = cam.pass_settings.clamp_direct;
    c.clamp_indirect = cam.pass_settings.clamp_indirect;

    c.min_samples = cam.pass_settings.min_samples;
    c.variance_threshold = cam.pass_settings.variance_threshold;
    c.regularize_alpha = cam.pass_settings.regularize_alpha;
}

void Ray::SceneCommon::RemoveCamera(const CameraHandle i) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);
    cams_.Erase(i._block);
}

void Ray::SceneCommon::SetCamera_nolock(const CameraHandle i, const camera_desc_t &c) {
    assert(i._index < uint32_t(cams_.size()));
    camera_t &cam = cams_[i._index];
    if (c.type != eCamType::Geo) {
        if (c.ltype == eLensUnits::FOV) {
            ConstructCamera(c.type, c.filter, c.filter_width, c.view_transform, c.origin, c.fwd, c.up, c.shift, c.fov,
                            c.sensor_height, c.exposure, c.gamma, c.focus_distance, c.fstop, c.lens_rotation,
                            c.lens_ratio, c.lens_blades, c.clip_start, c.clip_end, &cam);
        } else if (c.ltype == eLensUnits::FLength) {
        }
    } else {
        cam.type = eCamType::Geo;
        cam.exposure = c.exposure;
        cam.gamma = c.gamma;
        cam.mi_index = c.mi_index;
        cam.uv_index = c.uv_index;
    }

    cam.pass_settings.flags = {};
    if (c.lighting_only) {
        cam.pass_settings.flags |= ePassFlags::LightingOnly;
    }
    if (c.skip_direct_lighting) {
        cam.pass_settings.flags |= ePassFlags::SkipDirectLight;
    }
    if (c.skip_indirect_lighting) {
        cam.pass_settings.flags |= ePassFlags::SkipIndirectLight;
    }
    if (c.no_background) {
        cam.pass_settings.flags |= ePassFlags::NoBackground;
    }
    if (c.output_sh) {
        cam.pass_settings.flags |= ePassFlags::OutputSH;
    }

    cam.pass_settings.max_diff_depth = c.max_diff_depth;
    cam.pass_settings.max_spec_depth = c.max_spec_depth;
    cam.pass_settings.max_refr_depth = c.max_refr_depth;
    cam.pass_settings.max_transp_depth = c.max_transp_depth;
    cam.pass_settings.max_total_depth = c.max_total_depth;
    cam.pass_settings.min_total_depth = c.min_total_depth;
    cam.pass_settings.min_transp_depth = c.min_transp_depth;

    // make sure to not exceed allowed bounces
    while (cam.pass_settings.max_transp_depth + cam.pass_settings.max_total_depth > MAX_BOUNCES) {
        cam.pass_settings.max_transp_depth = std::max(cam.pass_settings.max_transp_depth - 1, 0);
        cam.pass_settings.max_total_depth = std::max(cam.pass_settings.max_total_depth - 1, 0);
    }

    cam.pass_settings.clamp_direct = c.clamp_direct;
    cam.pass_settings.clamp_indirect = c.clamp_indirect;

    cam.pass_settings.min_samples = c.min_samples;
    cam.pass_settings.variance_threshold = c.variance_threshold;
    cam.pass_settings.regularize_alpha = c.regularize_alpha;
}

//#define DUMP_SKY_ENV
#ifdef DUMP_SKY_ENV
extern "C" {
int SaveEXR(const float *data, int width, int height, int components, const int save_as_fp16, const char *outfilename,
            const char **err);
}
#endif

void Ray::SceneCommon::UpdateSkyTransmittanceLUT(const atmosphere_params_t &params) {
    sky_transmittance_lut_.resize(4 * SKY_TRANSMITTANCE_LUT_W * SKY_TRANSMITTANCE_LUT_H);
    for (int y = 0; y < SKY_TRANSMITTANCE_LUT_H; ++y) {
        const float v = float(y) / SKY_TRANSMITTANCE_LUT_H;
        for (int x = 0; x < SKY_TRANSMITTANCE_LUT_W; ++x) {
            const float u = float(x) / SKY_TRANSMITTANCE_LUT_W;

            const Ref::fvec2 uv = {u, v};

            float view_height, view_zenith_cos_angle;
            UvToLutTransmittanceParams(params, uv, view_height, view_zenith_cos_angle);

            const Ref::fvec4 world_pos = {0.0f, view_height - params.planet_radius, 0.0f, 0.0f};
            const Ref::fvec4 world_dir = {0.0f, view_zenith_cos_angle,
                                          -sqrtf(1.0f - view_zenith_cos_angle * view_zenith_cos_angle), 0.0f};

            const Ref::fvec4 optical_depthlight = IntegrateOpticalDepth(params, world_pos, world_dir);
            const Ref::fvec4 transmittance = exp(-optical_depthlight);

            transmittance.store_to(&sky_transmittance_lut_[4 * (y * SKY_TRANSMITTANCE_LUT_W + x)], Ref::vector_aligned);
        }
    }
}

void Ray::SceneCommon::UpdateMultiscatterLUT(const atmosphere_params_t &params) {
    atmosphere_params_t _params = params;
    _params.moon_radius = 0.0f;

    const float SphereSolidAngle = 4.0f * PI;
    const float IsotropicPhase = 1.0f / SphereSolidAngle;

    const float PlanetRadiusOffset = 0.01f;
    const int RaysCountSqrt = 8;

    // Taken from: https://github.com/sebh/UnrealEngineSkyAtmosphere

    sky_multiscatter_lut_.resize(4 * SKY_MULTISCATTER_LUT_RES * SKY_MULTISCATTER_LUT_RES);
    for (int j = 0; j < SKY_MULTISCATTER_LUT_RES; ++j) {
        const float y = (j + 0.5f) / SKY_MULTISCATTER_LUT_RES;
        for (int i = 0; i < SKY_MULTISCATTER_LUT_RES; ++i) {
            const float x = (i + 0.5f) / SKY_MULTISCATTER_LUT_RES;

            const Ref::fvec2 uv = {from_sub_uvs_to_unit(x, SKY_MULTISCATTER_LUT_RES),
                                   from_sub_uvs_to_unit(y, SKY_MULTISCATTER_LUT_RES)};

            const float cos_sun_zenith_angle = uv.get<0>() * 2.0f - 1.0f;
            const Ref::fvec4 sun_dir = {0.0f, cos_sun_zenith_angle,
                                        -sqrtf(saturate(1.0f - cos_sun_zenith_angle * cos_sun_zenith_angle)), 0.0f};

            const float view_height =
                saturate(uv.get<1>() + PlanetRadiusOffset) * (params.atmosphere_height - PlanetRadiusOffset);

            const Ref::fvec4 world_pos = {0.0f, view_height, 0.0f, 0.0f};
            Ref::fvec4 world_dir = {0.0f, 1.0f, 0.0f, 0.0f};

            std::pair<Ray::Ref::fvec4, Ray::Ref::fvec4> total_res = {};

            for (int rj = 0; rj < RaysCountSqrt; ++rj) {
                const float rv = (rj + 0.5f) / RaysCountSqrt;
                const float phi = acosf(1.0f - 2.0f * rv);

                const float cos_phi = cosf(phi), sin_phi = sinf(phi);
                for (int ri = 0; ri < RaysCountSqrt; ++ri) {
                    const float ru = (ri + 0.5f) / RaysCountSqrt;
                    const float theta = 2.0f * PI * ru;

                    const float cos_theta = cosf(theta), sin_theta = sinf(theta);

                    world_dir.set<0>(cos_theta * sin_phi);
                    world_dir.set<1>(cos_phi);
                    world_dir.set<2>(-sin_theta * sin_phi);

                    Ref::fvec4 transmittance = 1.0f;
                    const std::pair<Ray::Ref::fvec4, Ray::Ref::fvec4> res =
                        Ref::IntegrateScatteringMain<true>(_params, world_pos, world_dir, MAX_DIST, sun_dir, {}, 1.0f,
                                                           sky_transmittance_lut_, {}, 0.0f, 32, transmittance);

                    total_res.first += res.first;
                    total_res.second += res.second;
                }
            }

            total_res.first *= SphereSolidAngle / (RaysCountSqrt * RaysCountSqrt);
            total_res.second *= SphereSolidAngle / (RaysCountSqrt * RaysCountSqrt);

            const Ref::fvec4 in_scattered_luminance = total_res.first * IsotropicPhase;
            const Ref::fvec4 multi_scat_as_1 = total_res.second * IsotropicPhase;

            // For a serie, sum_{n=0}^{n=+inf} = 1 + r + r^2 + r^3 + ... + r^n = 1 / (1.0 - r), see
            // https://en.wikipedia.org/wiki/Geometric_series
            const Ref::fvec4 r = multi_scat_as_1;
            const Ref::fvec4 sum_of_all_multiscattering_events_contribution = 1.0f / (1.0f - r);
            const Ref::fvec4 L = in_scattered_luminance * sum_of_all_multiscattering_events_contribution;

            L.store_to(&sky_multiscatter_lut_[4 * (j * SKY_MULTISCATTER_LUT_RES + i)], Ref::vector_aligned);
        }
    }
}

std::vector<Ray::color_rgba8_t>
Ray::SceneCommon::CalcSkyEnvTexture(const atmosphere_params_t &params, const int res[2], const light_t lights[],
                                    Span<const uint32_t> dir_lights,
                                    const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    std::vector<color_rgba8_t> rgbe_pixels(res[0] * res[1]);
#ifdef DUMP_SKY_ENV
    std::vector<color_rgb_t> rgb_pixels(res[0] * res[1]);
#endif

    parallel_for(0, res[1], [&](const int y) {
        const float theta = PI * float(y) / float(res[1]);
        for (int x = 0; x < res[0]; ++x) {
            const uint32_t px_hash = Ref::hash((x << 16)| y);

            const float phi = 2.0f * PI * (x + 0.5f) / float(res[0]);
            auto ray_dir = Ref::fvec4{sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi), 0.0f};

            Ref::fvec4 color = 0.0f;

            // Evaluate light sources
            if (!dir_lights.empty()) {
                for (const uint32_t li_index : dir_lights) {
                    const light_t &l = lights[li_index];

                    const Ref::fvec4 light_dir = {l.dir.dir[0], l.dir.dir[1], l.dir.dir[2], 0.0f};
                    Ref::fvec4 light_col = {l.col[0], l.col[1], l.col[2], 0.0f};
                    if (l.dir.angle != 0.0f) {
                        const float radius = tanf(l.dir.angle);
                        light_col *= (PI * radius * radius);
                    }

                    color += IntegrateScattering(params, Ref::fvec4{0.0f, params.viewpoint_height, 0.0f, 0.0f}, ray_dir,
                                                 MAX_DIST, light_dir, l.dir.angle, light_col, sky_transmittance_lut_,
                                                 sky_multiscatter_lut_, px_hash);
                }
            } else if (params.stars_brightness > 0.0f) {
                // Use fake lightsource (to light up the moon)
                const Ref::fvec4 light_dir = {0.0f, -1.0f, 0.0f, 0.0f},
                                 light_col = {144809.866891f, 129443.618266f, 127098.894121f, 0.0f};

                color += IntegrateScattering(params, Ref::fvec4{0.0f, params.viewpoint_height, 0.0f, 0.0f}, ray_dir,
                                             MAX_DIST, light_dir, 0.0f, light_col, sky_transmittance_lut_,
                                             sky_multiscatter_lut_, px_hash);
            }

#ifdef DUMP_SKY_ENV
            rgb_pixels[y * res[0] + x].v[0] = color.get<0>();
            rgb_pixels[y * res[0] + x].v[1] = color.get<1>();
            rgb_pixels[y * res[0] + x].v[2] = color.get<2>();
#endif

            color = rgb_to_rgbe(color);

            rgbe_pixels[y * res[0] + x].v[0] = uint8_t(color.get<0>());
            rgbe_pixels[y * res[0] + x].v[1] = uint8_t(color.get<1>());
            rgbe_pixels[y * res[0] + x].v[2] = uint8_t(color.get<2>());
            rgbe_pixels[y * res[0] + x].v[3] = uint8_t(color.get<3>());
        }
    });

#ifdef DUMP_SKY_ENV
    const char *err = nullptr;
    SaveEXR(&rgb_pixels[0].v[0], res[0], res[1], 3, 0, "sky.exr", &err);
#endif

    return rgbe_pixels;
}