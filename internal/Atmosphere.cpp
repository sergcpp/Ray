#include "Atmosphere.h"

#include <cmath>

namespace Ray {
force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline float saturate(const float val) { return clamp(val, 0.0f, 1.0f); }

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

Ref::simd_fvec2 PlanetIntersection(const AtmosphereParameters &params, const Ref::simd_fvec4 &ray_start,
                                   const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius);
}
Ref::simd_fvec2 AtmosphereIntersection(const AtmosphereParameters &params, const Ref::simd_fvec4 &ray_start,
                                       const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius + params.atmosphere_height);
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
float AtmosphereHeight(const AtmosphereParameters &params, const Ref::simd_fvec4 &position_ws,
                       Ref::simd_fvec4 &up_vector) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    up_vector = (position_ws - planet_center);
    const float height = length(up_vector);
    up_vector /= height;
    return height - params.planet_radius;
}

force_inline Ref::simd_fvec4 AtmosphereDensity(const AtmosphereParameters &params, const float h) {
#if 1 // expf bug workaround (fp exception on unused simd lanes)
    const Ref::simd_fvec4 density_rayleigh = exp(Ref::simd_fvec4{-fmaxf(0.0f, h / params.rayleigh_height)});
    const Ref::simd_fvec4 density_mie = exp(Ref::simd_fvec4{-fmaxf(0.0f, h / params.mie_height)});
#else
    const float density_rayleigh = expf(-fmaxf(0.0f, h / params.rayleigh_height));
    const float density_mie = expf(-fmaxf(0.0f, h / params.mie_height));
#endif
    const float density_ozone = fmaxf(0.0f, 1.0f - fabsf(h - params.ozone_height_center) / params.ozone_half_width);

    return Ref::simd_fvec4{density_rayleigh.get<0>(), density_mie.get<0>(), density_ozone, 0.0f};
}

Ref::simd_fvec4 SampleTransmittanceLUT(Span<const Ref::simd_fvec4> transmittance_lut, Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H) - 0.5f;
    auto iuv0 = Ref::simd_ivec2(uv);
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});
    const Ref::simd_ivec2 iuv1 =
        min(iuv0 + Ref::simd_ivec2{1, 1}, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});

    const Ref::simd_fvec4 tr00 = transmittance_lut[iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>()],
                          tr01 = transmittance_lut[iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>()],
                          tr10 = transmittance_lut[iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>()],
                          tr11 = transmittance_lut[iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>()];

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 tr0 = tr01 * k.get<0>() + tr00 * (1.0f - k.get<0>()),
                          tr1 = tr11 * k.get<0>() + tr10 * (1.0f - k.get<0>());

    return (tr1 * k.get<1>() + tr0 * (1.0f - k.get<1>()));
}
} // namespace Ray

Ray::Ref::simd_fvec4 Ray::IntegrateOpticalDepth(const AtmosphereParameters &params, const Ref::simd_fvec4 &ray_start,
                                                const Ref::simd_fvec4 &ray_dir) {
    Ref::simd_fvec2 intersection = AtmosphereIntersection(params, ray_start, ray_dir);
    float ray_length = intersection[1];

    const int SampleCount = 64;
    float stepSize = ray_length / SampleCount;

    Ref::simd_fvec4 optical_depth = 0.0f;

    for (int i = 0; i < SampleCount; i++) {
        Ref::simd_fvec4 local_pos = ray_start + ray_dir * (i + 0.5f) * stepSize, up_vector;
        const float local_height = AtmosphereHeight(params, local_pos, up_vector);
        const Ref::simd_fvec4 local_density = AtmosphereDensity(params, local_height);

        optical_depth += local_density * stepSize;
    }

    return optical_depth;
}

// Calculate a luminance transmittance value from optical depth.
Ray::Ref::simd_fvec4 Ray::Absorb(const AtmosphereParameters &params, const Ref::simd_fvec4 &opticalDepth) {
    // Note that Mie results in slightly more light absorption than scattering, about 10%
    return exp(-(opticalDepth.get<0>() * params.rayleigh_scattering +
                 opticalDepth.get<1>() * params.mie_scattering * 1.1f +
                 opticalDepth.get<2>() * params.ozone_absorbtion) *
               params.atmosphere_density);
}

Ray::Ref::simd_fvec4 Ray::IntegrateScattering(const AtmosphereParameters &params, Ref::simd_fvec4 ray_start,
                                              const Ref::simd_fvec4 &ray_dir, float ray_length,
                                              const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &light_color,
                                              Span<const Ref::simd_fvec4> transmittance_lut,
                                              Ref::simd_fvec4 &transmittance) {
    // We can reduce the number of atmospheric samples required to converge by spacing them exponentially closer to the
    // camera. This breaks space view however, so let's compensate for that with an exponent that "fades" to 1 as we
    // leave the atmosphere.
    Ref::simd_fvec4 _unused;
    const float ray_height = AtmosphereHeight(params, ray_start, _unused);
    const float sample_distribution_exponent =
        1.0f + saturate(1.0f - ray_height / params.atmosphere_height) * 8.0f; // Slightly arbitrary max exponent of 9

    const Ref::simd_fvec2 atm_intersection = AtmosphereIntersection(params, ray_start, ray_dir);
    ray_length = fminf(ray_length, atm_intersection.get<1>());
    if (atm_intersection.get<0>() > 0) {
        // Advance ray to the atmosphere entry point
        ray_start += ray_dir * atm_intersection.get<0>();
        ray_length -= atm_intersection.get<0>();
    }

    const Ref::simd_fvec2 planet_intersection = PlanetIntersection(params, ray_start, ray_dir);
    if (planet_intersection.get<0>() > 0) {
        ray_length = fminf(ray_length, planet_intersection.get<0>());
    }

    if (ray_length <= 0.0f) {
        return Ref::simd_fvec4{0.0f};
    }

    const float costh = dot(ray_dir, light_dir), phase_r = PhaseRayleigh(costh), phase_m = PhaseMie(costh);

    const int SampleCount = 64;

    Ref::simd_fvec4 optical_depth = 0.0f, rayleigh = 0.0f, mie = 0.0f;

    float prev_ray_time = 0;

    for (int i = 0; i < SampleCount; i++) {
        const float ray_time = powf(float(i) / SampleCount, sample_distribution_exponent) * ray_length;
        // Because we are distributing the samples exponentially, we have to calculate the step size per sample.
        const float step_size = (ray_time - prev_ray_time);

        const Ref::simd_fvec4 local_position = ray_start + ray_dir * ray_time;
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);
        const Ref::simd_fvec4 local_density = AtmosphereDensity(params, local_height);

        optical_depth += local_density * step_size;

        // The atmospheric transmittance from ray_start to localPosition
        const Ref::simd_fvec4 view_transmittance = Absorb(params, optical_depth);

        Ref::simd_fvec4 light_transmittance;
        if (transmittance_lut.empty()) {
            const Ref::simd_fvec4 optical_depthlight = IntegrateOpticalDepth(params, local_position, light_dir);
            // The atmospheric transmittance of light reaching localPosition
            light_transmittance = Absorb(params, optical_depthlight);
        } else {
            const float view_zenith_cos_angle = dot(light_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
        }

        rayleigh += view_transmittance * light_transmittance * phase_r * local_density.get<0>() * step_size;
        mie += view_transmittance * light_transmittance * phase_m * local_density.get<1>() * step_size;

        prev_ray_time = ray_time;
    }

    transmittance = Absorb(params, optical_depth);

    Ref::simd_fvec4 ground_color = 0.0f;
    if (planet_intersection.get<0>() > 0) { // planet ground
        const Ref::simd_fvec4 local_position = ray_start + ray_dir * ray_length;
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);

        const Ref::simd_fvec4 view_transmittance = Absorb(params, optical_depth);

        Ref::simd_fvec4 light_transmittance;
        if (transmittance_lut.empty()) {
            const Ref::simd_fvec4 optical_depthlight = IntegrateOpticalDepth(params, local_position, light_dir);
            // The atmospheric transmittance of light reaching localPosition
            light_transmittance = Absorb(params, optical_depthlight);
        } else {
            const float view_zenith_cos_angle = dot(light_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
        }
        ground_color =
            params.ground_albedo * saturate(dot(up_vector, light_dir)) * view_transmittance * light_transmittance;
    }

    return (ground_color + rayleigh * params.rayleigh_scattering + mie * params.mie_scattering) * light_color;
}

void Ray::UvToLutTransmittanceParams(const AtmosphereParameters &params, Ref::simd_fvec2 uv, float &view_height,
                                     float &view_zenith_cos_angle) {
    const float top_radius = params.planet_radius + params.atmosphere_height;

    const float x_mu = uv.get<0>(), x_r = uv.get<1>();

    const float H = sqrtf(top_radius * top_radius - params.planet_radius * params.planet_radius);
    const float rho = H * x_r;
    view_height = sqrtf(rho * rho + params.planet_radius * params.planet_radius);

    const float d_min = top_radius - view_height;
    const float d_max = rho + H;
    const float d = d_min + x_mu * (d_max - d_min);
    view_zenith_cos_angle = d == 0.0f ? 1.0f : (H * H - rho * rho - d * d) / (2.0f * view_height * d);
    view_zenith_cos_angle = clamp(view_zenith_cos_angle, -1.0f, 1.0f);
}

Ray::Ref::simd_fvec2 Ray::LutTransmittanceParamsToUv(const AtmosphereParameters &params, const float view_height,
                                                     const float view_zenith_cos_angle) {
    const float top_radius = params.planet_radius + params.atmosphere_height;

    const float H = sqrtf(fmaxf(0.0f, top_radius * top_radius - params.planet_radius * params.planet_radius));
    const float rho = sqrtf(fmaxf(0.0f, view_height * view_height - params.planet_radius * params.planet_radius));

    const float discriminant =
        view_height * view_height * (view_zenith_cos_angle * view_zenith_cos_angle - 1.0) + top_radius * top_radius;
    const float d =
        fmaxf(0.0f, (-view_height * view_zenith_cos_angle + sqrtf(discriminant))); // Distance to atmosphere boundary

    const float d_min = top_radius - view_height;
    const float d_max = rho + H;
    const float x_mu = (d - d_min) / (d_max - d_min);
    const float x_r = rho / H;

    return Ref::simd_fvec2{x_mu, x_r};
}
