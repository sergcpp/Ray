#pragma once

#include "CoreRef.h"

// Based on: https://github.com/sebh/UnrealEngineSkyAtmosphere

namespace Ray {
const int TRANSMITTANCE_LUT_W = 256, TRANSMITTANCE_LUT_H = 64;

// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the
// atmosphere). This function calculates just that for our three atmospheric elements: R: Rayleigh G: Mie B: Ozone If
// you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in
// total".
Ref::simd_fvec4 IntegrateOpticalDepth(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                      const Ref::simd_fvec4 &ray_dir);

// Calculate a luminance transmittance value from optical depth.
Ref::simd_fvec4 Absorb(const atmosphere_params_t &params, const Ref::simd_fvec4 &opticalDepth);

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
Ref::simd_fvec4 IntegrateScattering(const atmosphere_params_t &params, Ref::simd_fvec4 ray_start,
                                    const Ref::simd_fvec4 &ray_dir, float ray_length, const Ref::simd_fvec4 &light_dir,
                                    float light_angle, const Ref::simd_fvec4 &light_color,
                                    Span<const float> transmittance_lut, uint32_t rand_hash,
                                    Ref::simd_fvec4 &transmittance);

// Transmittance LUT function parameterisation from Bruneton 2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
void UvToLutTransmittanceParams(const atmosphere_params_t &params, Ref::simd_fvec2 uv, float &view_height,
                                float &view_zenith_cos_angle);
Ref::simd_fvec2 LutTransmittanceParamsToUv(const atmosphere_params_t &params, float view_height,
                                           float view_zenith_cos_angle);
} // namespace Ray