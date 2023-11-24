#pragma once

#include "CoreRef.h"

// Based on: https://github.com/sebh/UnrealEngineSkyAtmosphere

namespace Ray {
const int TRANSMITTANCE_LUT_W = 256, TRANSMITTANCE_LUT_H = 64;

Ref::simd_fvec4 IntegrateOpticalDepth(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                      const Ref::simd_fvec4 &ray_dir);

template <bool ExpSampleDistribution = true>
Ref::simd_fvec4 IntegrateScatteringMain(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                        const Ref::simd_fvec4 &ray_dir, float ray_length,
                                        const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &moon_dir,
                                        const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut,
                                        float rand_offset, int sample_count, Ref::simd_fvec4 &inout_transmittance);

Ref::simd_fvec4 IntegrateScattering(const atmosphere_params_t &params, Ref::simd_fvec4 ray_start,
                                    const Ref::simd_fvec4 &ray_dir, float ray_length, const Ref::simd_fvec4 &light_dir,
                                    float light_angle, const Ref::simd_fvec4 &light_color,
                                    Span<const float> transmittance_lut, uint32_t rand_hash);

// Transmittance LUT function parameterisation from Bruneton 2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
void UvToLutTransmittanceParams(const atmosphere_params_t &params, Ref::simd_fvec2 uv, float &view_height,
                                float &view_zenith_cos_angle);
Ref::simd_fvec2 LutTransmittanceParamsToUv(const atmosphere_params_t &params, float view_height,
                                           float view_zenith_cos_angle);
} // namespace Ray
