#pragma once

#include "CoreRef.h"

namespace Ray {
// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the
// atmosphere). This function calculates just that for our three atmospheric elements: R: Rayleigh G: Mie B: Ozone If
// you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in
// total".
Ref::simd_fvec4 IntegrateOpticalDepth(const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir);

// Calculate a luminance transmittance value from optical depth.
Ref::simd_fvec4 Absorb(const Ref::simd_fvec4 &opticalDepth);

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
Ref::simd_fvec4 IntegrateScattering(Ref::simd_fvec4 ray_start, const Ref::simd_fvec4 &ray_dir, float ray_length,
                                    const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &light_color,
                                    Ref::simd_fvec4 &transmittance);
}