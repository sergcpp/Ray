#pragma once

#include <utility>

#include "CoreRef.h"

namespace Ray::Ref {
fvec4 IntegrateOpticalDepth(const atmosphere_params_t &params, const fvec4 &ray_start, const fvec4 &ray_dir);

template <bool UniformPhase = false>
std::pair<fvec4, fvec4>
IntegrateScatteringMain(const atmosphere_params_t &params, const fvec4 &ray_start, const fvec4 &ray_dir,
                        float ray_length, const fvec4 &light_dir, const fvec4 &moon_dir, const fvec4 &moon_irradiance,
                        const fvec4 &light_color, Span<const float> transmittance_lut,
                        Span<const float> multiscatter_lut, float rand_offset_main, float rand_offset_shadow,
                        int sample_count, bool use_clouds_shadow, fvec4 &inout_transmittance);

// Bake AA data for a disc: its angular cap seen from the bake ray origin plus the lat-long bake cell bounds
struct moon_disc_aa_t {
    fvec4 cap_dir;
    float ang_radius;
    float theta0, theta1, phi0, phi1;
};

fvec4 IntegrateScattering(const atmosphere_params_t &params, fvec4 ray_start, const fvec4 &ray_dir, float ray_length,
                          const fvec4 &light_dir, float light_angle, const fvec4 &light_color,
                          const fvec4 &light_color_point, Span<const float> transmittance_lut,
                          Span<const float> multiscatter_lut, uint32_t rand_hash, float sun_disc_coverage = -1.0f,
                          const moon_disc_aa_t *moon_disc_aa = nullptr);

// Fraction of the lat-long cell [theta0,theta1]x[phi0,phi1] covered by a cap of angular radius rho around cap_dir
float SphericalCapCellCoverage(const fvec4 &cap_dir, float rho, float theta0, float theta1, float phi0, float phi1);

// Transmittance LUT function parameterisation from Bruneton 2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
void UvToLutTransmittanceParams(const atmosphere_params_t &params, fvec2 uv, float &view_height,
                                float &view_zenith_cos_angle);
fvec2 LutTransmittanceParamsToUv(const atmosphere_params_t &params, float view_height, float view_zenith_cos_angle);

void ShadeSky(const pass_settings_t &ps, float limit, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
              Span<const uint32_t> ray_indices, const scene_data_t &sc, int iteration, int img_w,
              color_rgba_t *out_color);
void ShadeSkyPrimary(const pass_settings_t &ps, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                     Span<const uint32_t> ray_indices, const scene_data_t &sc, int iteration, int img_w,
                     color_rgba_t *out_color);
void ShadeSkySecondary(const pass_settings_t &ps, float clamp_direct, Span<const hit_data_t> inters,
                       Span<const ray_data_t> rays, Span<const uint32_t> ray_indices, const scene_data_t &sc,
                       int iteration, int img_w, color_rgba_t *out_color);
} // namespace Ray::Ref
