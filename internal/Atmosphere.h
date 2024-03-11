#pragma once

#include <utility>

#include "CoreRef.h"

namespace Ray {
const int TRANSMITTANCE_LUT_W = 256, TRANSMITTANCE_LUT_H = 64;
const int MULTISCATTER_LUT_RES = 32;

force_inline float from_unit_to_sub_uvs(float u, float resolution) {
    return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f));
}
force_inline float from_sub_uvs_to_unit(float u, float resolution) {
    return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f));
}

Ref::fvec4 IntegrateOpticalDepth(const atmosphere_params_t &params, const Ref::fvec4 &ray_start,
                                      const Ref::fvec4 &ray_dir);

template <bool ExpSampleDistribution = true, bool UniformPhase = false>
std::pair<Ref::fvec4, Ref::fvec4>
IntegrateScatteringMain(const atmosphere_params_t &params, const Ref::fvec4 &ray_start,
                        const Ref::fvec4 &ray_dir, float ray_length, const Ref::fvec4 &light_dir,
                        const Ref::fvec4 &moon_dir, const Ref::fvec4 &light_color,
                        Span<const float> transmittance_lut, Span<const float> multiscatter_lut, float rand_offset,
                        int sample_count, Ref::fvec4 &inout_transmittance);

Ref::fvec4 IntegrateScattering(const atmosphere_params_t &params, Ref::fvec4 ray_start,
                                    const Ref::fvec4 &ray_dir, float ray_length, const Ref::fvec4 &light_dir,
                                    float light_angle, const Ref::fvec4 &light_color,
                                    Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
                                    uint32_t rand_hash);

// Transmittance LUT function parameterisation from Bruneton 2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
void UvToLutTransmittanceParams(const atmosphere_params_t &params, Ref::fvec2 uv, float &view_height,
                                float &view_zenith_cos_angle);
Ref::fvec2 LutTransmittanceParamsToUv(const atmosphere_params_t &params, float view_height,
                                           float view_zenith_cos_angle);
} // namespace Ray
