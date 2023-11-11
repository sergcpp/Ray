#pragma once

#include "CoreRef.h"

// Based on: https://github.com/sebh/UnrealEngineSkyAtmosphere

namespace Ray {
struct AtmosphereParameters {
    float planet_radius = 6371000.0f;
    float atmosphere_height = 100000.0f;
    float rayleigh_height = atmosphere_height * 0.08f;
    float mie_height = atmosphere_height * 0.012f;
    // The ozone layer is represented as a tent function
    float ozone_height_center = 25000.0f, ozone_half_width = 15000.0f;
    Ref::simd_fvec4 rayleigh_scattering = Ref::simd_fvec4{5.802f, 13.558f, 33.100f, 0.0f} * 1e-6f;
    Ref::simd_fvec4 mie_scattering = Ref::simd_fvec4{3.996f, 3.996f, 3.996f, 0.0f} * 1e-6f;
    Ref::simd_fvec4 ozone_absorbtion = Ref::simd_fvec4{0.650f, 1.881f, 0.085f, 0.0f} * 1e-6f;
    float atmosphere_density = 1.0f;
    Ref::simd_fvec4 ground_albedo = {0.05f, 0.05f, 0.05f, 0.0f};
};

const int TRANSMITTANCE_LUT_W = 256, TRANSMITTANCE_LUT_H = 64;

// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the
// atmosphere). This function calculates just that for our three atmospheric elements: R: Rayleigh G: Mie B: Ozone If
// you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in
// total".
Ref::simd_fvec4 IntegrateOpticalDepth(const AtmosphereParameters &params, const Ref::simd_fvec4 &ray_start,
                                      const Ref::simd_fvec4 &ray_dir);

// Calculate a luminance transmittance value from optical depth.
Ref::simd_fvec4 Absorb(const AtmosphereParameters &params, const Ref::simd_fvec4 &opticalDepth);

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
Ref::simd_fvec4 IntegrateScattering(const AtmosphereParameters &params, Ref::simd_fvec4 ray_start,
                                    const Ref::simd_fvec4 &ray_dir, float ray_length, const Ref::simd_fvec4 &light_dir,
                                    const Ref::simd_fvec4 &light_color, Span<const Ref::simd_fvec4> transmittance_lut,
                                    Ref::simd_fvec4 &transmittance);

// Transmittance LUT function parameterisation from Bruneton 2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
void UvToLutTransmittanceParams(const AtmosphereParameters &params, Ref::simd_fvec2 uv, float &view_height,
                                float &view_zenith_cos_angle);
Ref::simd_fvec2 LutTransmittanceParamsToUv(const AtmosphereParameters &params, float view_height,
                                           float view_zenith_cos_angle);
} // namespace Ray