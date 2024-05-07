#pragma once

#include <utility>

#include "CoreRef.h"

namespace Ray {
namespace Ref {
fvec4 IntegrateOpticalDepth(const atmosphere_params_t &params, const fvec4 &ray_start, const fvec4 &ray_dir);

template <bool UniformPhase = false>
std::pair<fvec4, fvec4> IntegrateScatteringMain(const atmosphere_params_t &params, const fvec4 &ray_start,
                                                const fvec4 &ray_dir, float ray_length, const fvec4 &light_dir,
                                                const fvec4 &moon_dir, const fvec4 &light_color,
                                                Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
                                                float rand_offset, int sample_count, fvec4 &inout_transmittance);

fvec4 IntegrateScattering(const atmosphere_params_t &params, fvec4 ray_start, const fvec4 &ray_dir, float ray_length,
                          const fvec4 &light_dir, float light_angle, const fvec4 &light_color,
                          Span<const float> transmittance_lut, Span<const float> multiscatter_lut, uint32_t rand_hash);

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
} // namespace Ref
} // namespace Ray
