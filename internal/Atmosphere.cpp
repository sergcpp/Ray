// Copyright (c) 2021 Felix Westin
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Atmosphere.h"

#include <cmath>

namespace Ray {
// const float ATMOSPHERE_EPS = 1e-6f;
// #define INFINITY 1.0 / 0.0
const float PLANET_RADIUS = 6371000.0f;
const Ref::simd_fvec4 PLANET_CENTER = Ref::simd_fvec4(0, -PLANET_RADIUS, 0, 0);
const float ATMOSPHERE_HEIGHT = 100000.0f;
const float RAYLEIGH_HEIGHT = (ATMOSPHERE_HEIGHT * 0.08f);
const float MIE_HEIGHT = (ATMOSPHERE_HEIGHT * 0.012f);

const Ref::simd_fvec4 C_RAYLEIGH = Ref::simd_fvec4{5.802f, 13.558f, 33.100f, 0.0f} * 1e-6f;
const Ref::simd_fvec4 C_MIE = Ref::simd_fvec4{3.996f, 3.996f, 3.996f, 0.0f} * 1e-6f;
const Ref::simd_fvec4 C_OZONE = Ref::simd_fvec4{0.650f, 1.881f, 0.085f, 0.0f} * 1e-6f;

const float ATMOSPHERE_DENSITY = 1.0f;

force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}

// Math
Ref::simd_fvec2 SphereIntersection(Ref::simd_fvec4 ray_start, const Ref::simd_fvec4 &ray_dir,
                                   const Ref::simd_fvec4 &sphere_center, const float sphere_radius) {
    ray_start -= sphere_center;
    const float a = dot(ray_dir, ray_dir);
    const float b = 2.0f * dot(ray_start, ray_dir);
    const float c = dot(ray_start, ray_start) - (sphere_radius * sphere_radius);
    float d = b * b - 4 * a * c;
    if (d < 0) {
        return Ref::simd_fvec2{-1};
    } else {
        d = sqrt(d);
        return Ref::simd_fvec2{-b - d, -b + d} / (2 * a);
    }
}

Ref::simd_fvec2 PlanetIntersection(const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir) {
    return SphereIntersection(ray_start, ray_dir, PLANET_CENTER, PLANET_RADIUS);
}
Ref::simd_fvec2 AtmosphereIntersection(const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir) {
    return SphereIntersection(ray_start, ray_dir, PLANET_CENTER, PLANET_RADIUS + ATMOSPHERE_HEIGHT);
}

// Phase functions
float PhaseRayleigh(const float costh) { return 3 * (1 + costh * costh) / (16 * PI); }
float PhaseMie(float costh, float g = 0.85f) {
    g = fminf(g, 0.9381f);
    float k = 1.55f * g - 0.55f * g * g * g;
    float kcosth = k * costh;
    return (1 - k * k) / ((4 * PI) * (1 - kcosth) * (1 - kcosth));
}

// Atmosphere
float AtmosphereHeight(const Ref::simd_fvec4 &position_ws) {
    return length(position_ws - PLANET_CENTER) - PLANET_RADIUS;
}

// NOTE: simd_fvec4 is used to workaround flt exception in std::exp due to unused lanes
force_inline Ref::simd_fvec4 DensityRayleigh(const Ref::simd_fvec4 &h) {
    return exp(-max(0.0f, h / RAYLEIGH_HEIGHT));
}
force_inline Ref::simd_fvec4 DensityMie(const Ref::simd_fvec4 &h) { return exp(-max(0.0f, h / MIE_HEIGHT)); }
force_inline Ref::simd_fvec4 DensityOzone(const Ref::simd_fvec4 &h) {
    // The ozone layer is represented as a tent function with a width of 30km, centered around an altitude of 25km.
    return max(0.0f, 1.0f - abs(h - 25000.0f) / 15000.0f);
}
force_inline Ref::simd_fvec4 AtmosphereDensity(float h) {
    //return Ref::simd_fvec4{DensityRayleigh(h), DensityMie(h), DensityOzone(h), 0.0f};

    const Ref::simd_fvec4 x = DensityRayleigh(h);
    const Ref::simd_fvec4 y = DensityMie(h);
    const Ref::simd_fvec4 z = DensityOzone(h);

    return Ref::simd_fvec4{x.get<0>(), y.get<0>(), z.get<0>(), 0.0f};
}
} // namespace Ray

Ray::Ref::simd_fvec4 Ray::IntegrateOpticalDepth(const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir) {
    Ref::simd_fvec2 intersection = AtmosphereIntersection(ray_start, ray_dir);
    float ray_length = intersection[1];

    const int SampleCount = 8;
    float stepSize = ray_length / SampleCount;

    Ref::simd_fvec4 optical_depth = 0.0f;

    for (int i = 0; i < SampleCount; i++) {
        Ref::simd_fvec4 local_pos = ray_start + ray_dir * (i + 0.5f) * stepSize;
        float local_height = AtmosphereHeight(local_pos);
        Ref::simd_fvec4 local_density = AtmosphereDensity(local_height);

        optical_depth += local_density * stepSize;
    }

    return optical_depth;
}

// Calculate a luminance transmittance value from optical depth.
Ray::Ref::simd_fvec4 Ray::Absorb(const Ref::simd_fvec4 &opticalDepth) {
    // Note that Mie results in slightly more light absorption than scattering, about 10%
    return exp(
        -(opticalDepth.get<0>() * C_RAYLEIGH + opticalDepth.get<1>() * C_MIE * 1.1f + opticalDepth.get<2>() * C_OZONE) *
        ATMOSPHERE_DENSITY);
}

Ray::Ref::simd_fvec4 Ray::IntegrateScattering(Ref::simd_fvec4 ray_start, const Ref::simd_fvec4 &ray_dir,
                                              float ray_length, const Ref::simd_fvec4 &light_dir,
                                              const Ref::simd_fvec4 &light_color, Ref::simd_fvec4 &transmittance) {
    // We can reduce the number of atmospheric samples required to converge by spacing them exponentially closer to the
    // camera. This breaks space view however, so let's compensate for that with an exponent that "fades" to 1 as we
    // leave the atmosphere.
    float ray_height = AtmosphereHeight(ray_start);
    float sample_distribution_exponent =
        1.0f + clamp(1.0f - ray_height / ATMOSPHERE_HEIGHT, 0.0f, 1.0f) * 8.0f; // Slightly arbitrary max exponent of 9

    const Ref::simd_fvec2 intersection = AtmosphereIntersection(ray_start, ray_dir);
    ray_length = fminf(ray_length, intersection.get<1>());
    if (intersection.get<0>() > 0) {
        // Advance ray to the atmosphere entry point
        ray_start += ray_dir * intersection.get<0>();
        ray_length -= intersection.get<0>();
    }

    float costh = dot(ray_dir, light_dir);
    float phase_r = PhaseRayleigh(costh);
    float phase_m = PhaseMie(costh);

    const int SampleCount = 64;

    Ref::simd_fvec4 optical_depth = 0.0f;
    Ref::simd_fvec4 rayleigh = 0.0f;
    Ref::simd_fvec4 mie = 0.0f;

    float prev_ray_time = 0;

    for (int i = 0; i < SampleCount; i++) {
        float ray_time = pow((float)i / SampleCount, sample_distribution_exponent) * ray_length;
        // Because we are distributing the samples exponentially, we have to calculate the step size per sample.
        float step_size = (ray_time - prev_ray_time);

        Ref::simd_fvec4 local_position = ray_start + ray_dir * ray_time;
        float local_height = AtmosphereHeight(local_position);
        Ref::simd_fvec4 local_density = AtmosphereDensity(local_height);

        optical_depth += local_density * step_size;

        // The atmospheric transmittance from rayStart to localPosition
        Ref::simd_fvec4 view_transmittance = Absorb(optical_depth);

        Ref::simd_fvec4 optical_depthlight = IntegrateOpticalDepth(local_position, light_dir);
        // The atmospheric transmittance of light reaching localPosition
        Ref::simd_fvec4 light_transmittance = Absorb(optical_depthlight);

        rayleigh += view_transmittance * light_transmittance * phase_r * local_density.get<0>() * step_size;
        mie += view_transmittance * light_transmittance * phase_m * local_density.get<1>() * step_size;

        prev_ray_time = ray_time;
    }

    transmittance = Absorb(optical_depth);

    return (rayleigh * C_RAYLEIGH + mie * C_MIE) * light_color;
}
