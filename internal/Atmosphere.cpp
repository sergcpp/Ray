#include "Atmosphere.h"

#include <cmath>

// Based on: https://github.com/sebh/UnrealEngineSkyAtmosphere

namespace Ray {
#include "precomputed/__3d_noise_tex.inl"
#include "precomputed/__cirrus_tex.inl"
#include "precomputed/__moon_tex.inl"
#include "precomputed/__weather_tex.inl"

const float MoonSunRelation = 0.0000001f;

force_inline float remap(float value, float original_min) {
    return saturate((value - original_min) / (1.000001f - original_min));
}

// GPU PRO 7 - Real-time Volumetric Cloudscapes
// https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
// https://github.com/sebh/TileableVolumeNoise
force_inline float remap(float value, float original_min, float original_max, float new_min, float new_max) {
    return new_min +
           (saturate((value - original_min) / (0.0000001f + original_max - original_min)) * (new_max - new_min));
}

force_inline float linstep(float smin, float smax, float x) { return saturate((x - smin) / (smax - smin)); }

force_inline float smoothstep(float edge0, float edge1, float x) {
    const float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
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

Ref::simd_fvec2 PlanetIntersection(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                   const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius);
}

Ref::simd_fvec2 AtmosphereIntersection(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                       const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius + params.atmosphere_height);
}

Ref::simd_fvec4 CloudsIntersection(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                   const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    const Ref::simd_fvec2 beg =
        SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius + params.clouds_height_beg);
    const Ref::simd_fvec2 end =
        SphereIntersection(ray_start, ray_dir, planet_center, params.planet_radius + params.clouds_height_end);
    return {beg.get<0>(), beg.get<1>(), end.get<0>(), end.get<1>()};
}

Ref::simd_fvec2 MoonIntersection(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                 const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 planet_center =
        Ref::simd_fvec4{params.moon_dir, Ref::simd_mem_aligned} * params.moon_distance;
    return SphereIntersection(ray_start, ray_dir, planet_center, params.moon_radius);
}

// Phase functions
float PhaseRayleigh(const float costh) { return 3 * (1 + costh * costh) / (16 * PI); }
float PhaseMie(const float costh, float g = 0.85f) {
    g = fminf(g, 0.9381f);
    float k = 1.55f * g - 0.55f * g * g * g;
    float kcosth = k * costh;
    return (1 - k * k) / ((4 * PI) * (1 - kcosth) * (1 - kcosth));
}

const float WrenningePhaseScale = 0.9f;
const Ref::simd_fvec2 WrenningePhaseParameters = Ref::simd_fvec2(-0.2f, 0.8f);

force_inline float HenyeyGreenstein(const float mu, const float inG) {
    return (1.0f - inG * inG) / (powf(1.0f + inG * inG - 2.0f * inG * mu, 1.5f) * 4.0f * PI);
}

force_inline float CloudPhaseFunction(const float mu) {
    return mix(HenyeyGreenstein(mu, WrenningePhaseParameters.get<0>()),
               HenyeyGreenstein(mu, WrenningePhaseParameters.get<1>()), 0.7f);
}

Ref::simd_fvec4 PhaseWrenninge(float mu) {
    Ref::simd_fvec4 phase = 0.0f;
    // Wrenninge multiscatter approximation
    phase.set<0>(CloudPhaseFunction(mu));
    phase.set<1>(CloudPhaseFunction(mu * WrenningePhaseScale));
    phase.set<2>(CloudPhaseFunction(mu * WrenningePhaseScale * WrenningePhaseScale));
    return phase;
}

// dl is the density sampled along the light ray for the given sample position.
// dC is the low lod sample of density at the given sample position.
float GetLightEnergy(const float dl, const float dC, const Ref::simd_fvec4 &phase_probability) {
    // Wrenninge multi scatter approximation
    const auto exp_scale = Ref::simd_fvec4(0.8f, 0.1f, 0.002f, 0.0f);
    const auto total_scale = Ref::simd_fvec4(2.0f, 0.8f, 0.4f, 0.0f);
    const Ref::simd_fvec4 intensity_curve = exp(-dl * exp_scale);
    return dot(total_scale * phase_probability, intensity_curve);
}

// Atmosphere
float AtmosphereHeight(const atmosphere_params_t &params, const Ref::simd_fvec4 &position_ws,
                       Ref::simd_fvec4 &up_vector) {
    const Ref::simd_fvec4 planet_center = Ref::simd_fvec4(0, -params.planet_radius, 0, 0);
    up_vector = (position_ws - planet_center);
    const float height = length(up_vector);
    up_vector /= height;
    return height - params.planet_radius;
}

force_inline Ref::simd_fvec4 AtmosphereDensity(const atmosphere_params_t &params, const float h) {
#if 1 // expf bug workaround (fp exception on unused simd lanes)
    const Ref::simd_fvec4 density_rayleigh = exp(Ref::simd_fvec4{-fmaxf(0.0f, h / params.rayleigh_height)});
    const Ref::simd_fvec4 density_mie = exp(Ref::simd_fvec4{-fmaxf(0.0f, h / params.mie_height)});
#else
    const float density_rayleigh = expf(-fmaxf(0.0f, h / params.rayleigh_height));
    const float density_mie = expf(-fmaxf(0.0f, h / params.mie_height));
#endif
    const float density_ozone = fmaxf(0.0f, 1.0f - fabsf(h - params.ozone_height_center) / params.ozone_half_width);

    return params.atmosphere_density *
           Ref::simd_fvec4{density_rayleigh.get<0>(), density_mie.get<0>(), density_ozone, 0.0f};
}

struct atmosphere_medium_t {
    Ref::simd_fvec4 scattering;
    Ref::simd_fvec4 absorption;
    Ref::simd_fvec4 extinction;

    Ref::simd_fvec4 scattering_mie;
    Ref::simd_fvec4 absorption_mie;
    Ref::simd_fvec4 extinction_mie;

    Ref::simd_fvec4 scattering_ray;
    Ref::simd_fvec4 absorption_ray;
    Ref::simd_fvec4 extinction_ray;

    Ref::simd_fvec4 scattering_ozo;
    Ref::simd_fvec4 absorption_ozo;
    Ref::simd_fvec4 extinction_ozo;
};

force_inline atmosphere_medium_t SampleAtmosphereMedium(const atmosphere_params_t &params, const float h) {
    const Ref::simd_fvec4 local_density = AtmosphereDensity(params, h);

    atmosphere_medium_t s;

    s.scattering_mie = local_density.get<1>() * Ref::simd_fvec4{params.mie_scattering, Ref::simd_mem_aligned};
    s.absorption_mie = local_density.get<1>() * Ref::simd_fvec4{params.mie_absorption, Ref::simd_mem_aligned};
    s.extinction_mie = local_density.get<1>() * Ref::simd_fvec4{params.mie_extinction, Ref::simd_mem_aligned};

    s.scattering_ray = local_density.get<0>() * Ref::simd_fvec4{params.rayleigh_scattering, Ref::simd_mem_aligned};
    s.absorption_ray = 0.0f;
    s.extinction_ray = s.scattering_ray + s.absorption_ray;

    s.scattering_ozo = 0.0;
    s.absorption_ozo = local_density.get<2>() * Ref::simd_fvec4{params.ozone_absorbtion, Ref::simd_mem_aligned};
    s.extinction_ozo = s.scattering_ozo + s.absorption_ozo;

    s.scattering = s.scattering_mie + s.scattering_ray + s.scattering_ozo;
    s.absorption = s.absorption_mie + s.absorption_ray + s.absorption_ozo;
    s.extinction = s.extinction_mie + s.extinction_ray + s.extinction_ozo;
    s.extinction.set<3>(1.0f); // make it safe divisor

    return s;
}

Ref::simd_fvec4 SampleTransmittanceLUT(Span<const float> lut, Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    auto iuv0 = Ref::simd_ivec2(uv);
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});
    const Ref::simd_ivec2 iuv1 = min(iuv0 + 1, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});

    const auto tr00 = Ref::simd_fvec4(&lut[4 * (iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               tr01 = Ref::simd_fvec4(&lut[4 * (iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>())],
                                      Ref::simd_mem_aligned),
               tr10 = Ref::simd_fvec4(&lut[4 * (iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               tr11 = Ref::simd_fvec4(&lut[4 * (iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>())],
                                      Ref::simd_mem_aligned);

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 tr0 = tr01 * k.get<0>() + tr00 * (1.0f - k.get<0>()),
                          tr1 = tr11 * k.get<0>() + tr10 * (1.0f - k.get<0>());

    return (tr1 * k.get<1>() + tr0 * (1.0f - k.get<1>()));
}

Ref::simd_fvec4 SampleMultiscatterLUT(Span<const float> lut, Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(MULTISCATTER_LUT_RES);
    auto iuv0 = Ref::simd_ivec2(uv);
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{MULTISCATTER_LUT_RES - 1});
    const Ref::simd_ivec2 iuv1 = min(iuv0 + 1, Ref::simd_ivec2{MULTISCATTER_LUT_RES - 1});

    const auto ms00 = Ref::simd_fvec4(&lut[4 * (iuv0.get<1>() * MULTISCATTER_LUT_RES + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               ms01 = Ref::simd_fvec4(&lut[4 * (iuv0.get<1>() * MULTISCATTER_LUT_RES + iuv1.get<0>())],
                                      Ref::simd_mem_aligned),
               ms10 = Ref::simd_fvec4(&lut[4 * (iuv1.get<1>() * MULTISCATTER_LUT_RES + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               ms11 = Ref::simd_fvec4(&lut[4 * (iuv1.get<1>() * MULTISCATTER_LUT_RES + iuv1.get<0>())],
                                      Ref::simd_mem_aligned);

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 ms0 = ms01 * k.get<0>() + ms00 * (1.0f - k.get<0>()),
                          ms1 = ms11 * k.get<0>() + ms10 * (1.0f - k.get<0>());

    return (ms1 * k.get<1>() + ms0 * (1.0f - k.get<1>()));
}

force_inline Ref::simd_fvec4 FetchWeatherTex(const int x, const int y) {
    return Ref::simd_fvec4{float(__weather_tex[3 * (y * WEATHER_TEX_RES + x) + 0]),
                           float(__weather_tex[3 * (y * WEATHER_TEX_RES + x) + 1]),
                           float(__weather_tex[3 * (y * WEATHER_TEX_RES + x) + 2]), 0.0f};
}

Ref::simd_fvec4 SampleWeatherTex(Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(WEATHER_TEX_RES);
    auto iuv0 = Ref::simd_ivec2{uv};
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{WEATHER_TEX_RES - 1});
    const Ref::simd_ivec2 iuv1 = (iuv0 + 1) & Ref::simd_ivec2{WEATHER_TEX_RES - 1};

    const Ref::simd_fvec4 w00 = FetchWeatherTex(iuv0.get<0>(), iuv0.get<1>()),
                          w01 = FetchWeatherTex(iuv1.get<0>(), iuv0.get<1>()),
                          w10 = FetchWeatherTex(iuv0.get<0>(), iuv1.get<1>()),
                          w11 = FetchWeatherTex(iuv1.get<0>(), iuv1.get<1>());

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 w0 = w01 * k.get<0>() + w00 * (1.0f - k.get<0>()),
                          w1 = w11 * k.get<0>() + w10 * (1.0f - k.get<0>());

    return (w1 * k.get<1>() + w0 * (1.0f - k.get<1>())) * (1.0f / 255.0f);
}

// Taken from https://github.com/armory3d/armory_ci/blob/master/build_untitled/compiled/Shaders/world_pass.frag.glsl
float GetDensityHeightGradientForPoint(float height, float cloud_type) {
    const auto stratusGrad = Ref::simd_fvec4(0.02f, 0.05f, 0.09f, 0.11f);
    const auto stratocumulusGrad = Ref::simd_fvec4(0.02f, 0.2f, 0.48f, 0.625f);
    const auto cumulusGrad = Ref::simd_fvec4(0.01f, 0.0625f, 0.78f, 1.0f);
    float stratus = 1.0f - clamp(cloud_type * 2.0f, 0, 1);
    float stratocumulus = 1.0f - abs(cloud_type - 0.5f) * 2.0f;
    float cumulus = clamp(cloud_type - 0.5f, 0, 1) * 2.0f;
    Ref::simd_fvec4 cloudGradient = stratusGrad * stratus + stratocumulusGrad * stratocumulus + cumulusGrad * cumulus;
    return smoothstep(cloudGradient.get<0>(), cloudGradient.get<1>(), height) -
           smoothstep(cloudGradient.get<2>(), cloudGradient.get<3>(), height);
}

force_inline float Fetch3dNoiseTex(const int x, const int y, const int z) {
    return __3d_noise_tex[z * NOISE_3D_RES * NOISE_3D_RES + y * NOISE_3D_RES + x];
}

float Sample3dNoiseTex(Ref::simd_fvec4 uvw) {
    uvw *= NOISE_3D_RES;
    Ref::simd_ivec4 iuvw0 = Ref::simd_ivec4(uvw);
    iuvw0 = clamp(iuvw0, Ref::simd_ivec4{0}, Ref::simd_ivec4{NOISE_3D_RES - 1});
    const Ref::simd_ivec4 iuvw1 = (iuvw0 + 1) & Ref::simd_ivec4{NOISE_3D_RES - 1};

    const float n000 = Fetch3dNoiseTex(iuvw0.get<0>(), iuvw0.get<1>(), iuvw0.get<2>()),
                n001 = Fetch3dNoiseTex(iuvw1.get<0>(), iuvw0.get<1>(), iuvw0.get<2>()),
                n010 = Fetch3dNoiseTex(iuvw0.get<0>(), iuvw1.get<1>(), iuvw0.get<2>()),
                n011 = Fetch3dNoiseTex(iuvw1.get<0>(), iuvw1.get<1>(), iuvw0.get<2>()),
                n100 = Fetch3dNoiseTex(iuvw0.get<0>(), iuvw0.get<1>(), iuvw1.get<2>()),
                n101 = Fetch3dNoiseTex(iuvw1.get<0>(), iuvw0.get<1>(), iuvw1.get<2>()),
                n110 = Fetch3dNoiseTex(iuvw0.get<0>(), iuvw1.get<1>(), iuvw1.get<2>()),
                n111 = Fetch3dNoiseTex(iuvw1.get<0>(), iuvw1.get<1>(), iuvw1.get<2>());

    const Ref::simd_fvec4 k = fract(uvw);

    const float n00x = (1.0f - k.get<0>()) * n000 + k.get<0>() * n001,
                n01x = (1.0f - k.get<0>()) * n010 + k.get<0>() * n011,
                n10x = (1.0f - k.get<0>()) * n100 + k.get<0>() * n101,
                n11x = (1.0f - k.get<0>()) * n110 + k.get<0>() * n111;

    const float n0xx = (1.0f - k.get<1>()) * n00x + k.get<1>() * n01x,
                n1xx = (1.0f - k.get<1>()) * n10x + k.get<1>() * n11x;

    return ((1.0f - k.get<2>()) * n0xx + k.get<2>() * n1xx) / 255.0f;
}

float GetCloudsDensity(const atmosphere_params_t &params, Ref::simd_fvec4 local_position, float &out_local_height,
                       float &out_height_fraction, Ref::simd_fvec4 &out_up_vector) {
    out_local_height = AtmosphereHeight(params, local_position, out_up_vector);
    out_height_fraction =
        (out_local_height - params.clouds_height_beg) / (params.clouds_height_end - params.clouds_height_beg);

    Ref::simd_fvec2 weather_uv = {local_position.get<0>() + params.clouds_offset_x,
                                  local_position.get<2>() + params.clouds_offset_z};
    weather_uv = fract(weather_uv * 0.00007f);

    const Ref::simd_fvec4 weather_sample = SampleWeatherTex(weather_uv);

    float cloud_coverage = mix(weather_sample.get<2>(), weather_sample.get<1>(), params.clouds_variety);
    cloud_coverage = remap(cloud_coverage, saturate(1.0f - params.clouds_density + 0.5f * out_height_fraction));

    const float cloud_type = weather_sample.get<0>();
    cloud_coverage *= GetDensityHeightGradientForPoint(out_height_fraction, cloud_type);

    if (out_height_fraction > 1.0f || cloud_coverage < 0.01f) {
        return 0.0f;
    }

    local_position /= 1.5f * (params.clouds_height_end - params.clouds_height_beg);
    local_position = fract(local_position);

    const float noise_read = Sample3dNoiseTex(local_position);
    return 0.12f * mix(fmaxf(0.0f, 1.0f - cloud_type * 2.0f), 1.0f, out_height_fraction) *
           powf(5.0f * remap(cloud_coverage, 0.6f * noise_read), 1.0f - out_height_fraction);
}

float TraceCloudShadow(const atmosphere_params_t &params, const uint32_t rand_hash, Ref::simd_fvec4 ray_start,
                       const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 clouds_intersection = CloudsIntersection(params, ray_start, ray_dir);
    if (clouds_intersection.get<3>() > 0) {
        const int SampleCount = 32;
        const float StepSize = 16.0f;

        Ref::simd_fvec4 pos = ray_start + Ref::construct_float(rand_hash) * ray_dir * StepSize;

        float ret = 0.0f;
        for (int i = 0; i < SampleCount; ++i) {
            float local_height, height_fraction;
            Ref::simd_fvec4 up_vector;
            const float local_density = GetCloudsDensity(params, pos, local_height, height_fraction, up_vector);
            ret += local_density;
            pos += ray_dir * StepSize;
        }

        return ret * StepSize;
    }
    return 1.0f;
}

// https://www.shadertoy.com/view/NtsBzB
Ref::simd_fvec4 stars_hash(Ref::simd_fvec4 p) {
    p = Ref::simd_fvec4{dot(p, Ref::simd_fvec4{127.1f, 311.7f, 74.7f, 0.0f}),
                        dot(p, Ref::simd_fvec4{269.5f, 183.3f, 246.1f, 0.0f}),
                        dot(p, Ref::simd_fvec4{113.5f, 271.9f, 124.6f, 0.0f}), 0.0f};

    p.set<0>(sinf(p.get<0>()));
    p.set<1>(sinf(p.get<1>()));
    p.set<2>(sinf(p.get<2>()));

    return -1.0f + 2.0f * fract(p * 43758.5453123f);
}

float stars_noise(const Ref::simd_fvec4 &p) {
    Ref::simd_fvec4 i = floor(p);
    Ref::simd_fvec4 f = fract(p);

    Ref::simd_fvec4 u = f * f * (3.0f - 2.0f * f);

    return mix(mix(mix(dot(stars_hash(i + Ref::simd_fvec4(0.0f, 0.0f, 0.0f, 0.0f)),
                           f - Ref::simd_fvec4(0.0f, 0.0f, 0.0f, 0.0f)),
                       dot(stars_hash(i + Ref::simd_fvec4(1.0f, 0.0f, 0.0f, 0.0f)),
                           f - Ref::simd_fvec4(1.0f, 0.0f, 0.0f, 0.0f)),
                       u.get<0>()),
                   mix(dot(stars_hash(i + Ref::simd_fvec4(0.0f, 1.0f, 0.0f, 0.0f)),
                           f - Ref::simd_fvec4(0.0f, 1.0f, 0.0f, 0.0f)),
                       dot(stars_hash(i + Ref::simd_fvec4(1.0f, 1.0f, 0.0f, 0.0f)),
                           f - Ref::simd_fvec4(1.0f, 1.0f, 0.0f, 0.0f)),
                       u.get<0>()),
                   u.get<1>()),
               mix(mix(dot(stars_hash(i + Ref::simd_fvec4(0.0f, 0.0f, 1.0f, 0.0f)),
                           f - Ref::simd_fvec4(0.0f, 0.0f, 1.0f, 0.0f)),
                       dot(stars_hash(i + Ref::simd_fvec4(1.0f, 0.0f, 1.0f, 0.0f)),
                           f - Ref::simd_fvec4(1.0f, 0.0f, 1.0f, 0.0f)),
                       u.get<0>()),
                   mix(dot(stars_hash(i + Ref::simd_fvec4(0.0f, 1.0f, 1.0f, 0.0f)),
                           f - Ref::simd_fvec4(0.0f, 1.0f, 1.0f, 0.0f)),
                       dot(stars_hash(i + Ref::simd_fvec4(1.0f, 1.0f, 1.0f, 0.0f)),
                           f - Ref::simd_fvec4(1.0f, 1.0f, 1.0f, 0.0f)),
                       u.get<0>()),
                   u.get<1>()),
               u.get<2>());
}

force_inline Ref::simd_fvec4 FetchMoonTex(const int x, const int y) {
    return Ref::simd_fvec4{float(__moon_tex[3 * (y * MOON_TEX_W + x) + 0]),
                           float(__moon_tex[3 * (y * MOON_TEX_W + x) + 1]),
                           float(__moon_tex[3 * (y * MOON_TEX_W + x) + 2]), 0.0f};
}

Ref::simd_fvec4 SampleMoonTex(Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(MOON_TEX_W, MOON_TEX_H);
    auto iuv0 = Ref::simd_ivec2{uv};
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{MOON_TEX_W - 1, MOON_TEX_H - 1});
    const Ref::simd_ivec2 iuv1 = (iuv0 + 1) & Ref::simd_ivec2{MOON_TEX_W - 1, MOON_TEX_H - 1};

    const Ref::simd_fvec4 m00 = FetchMoonTex(iuv0.get<0>(), iuv0.get<1>()),
                          m01 = FetchMoonTex(iuv1.get<0>(), iuv0.get<1>()),
                          m10 = FetchMoonTex(iuv0.get<0>(), iuv1.get<1>()),
                          m11 = FetchMoonTex(iuv1.get<0>(), iuv1.get<1>());

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 m0 = m01 * k.get<0>() + m00 * (1.0f - k.get<0>()),
                          m1 = m11 * k.get<0>() + m10 * (1.0f - k.get<0>());

    return srgb_to_rgb((m1 * k.get<1>() + m0 * (1.0f - k.get<1>())) * (1.0f / 255.0f));
}

force_inline Ref::simd_fvec2 FetchCirrusTex(const int x, const int y) {
    return Ref::simd_fvec2{float(__cirrus_tex[2 * (y * CIRRUS_TEX_W + x) + 0]),
                           float(__cirrus_tex[2 * (y * CIRRUS_TEX_W + x) + 1])};
}

Ref::simd_fvec2 SampleCirrusTex(Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(CIRRUS_TEX_W, CIRRUS_TEX_H);
    auto iuv0 = Ref::simd_ivec2{uv};
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{CIRRUS_TEX_W - 1, CIRRUS_TEX_H - 1});
    const Ref::simd_ivec2 iuv1 = (iuv0 + 1) & Ref::simd_ivec2{CIRRUS_TEX_W - 1, CIRRUS_TEX_H - 1};

    const Ref::simd_fvec2 m00 = FetchCirrusTex(iuv0.get<0>(), iuv0.get<1>()),
                          m01 = FetchCirrusTex(iuv1.get<0>(), iuv0.get<1>()),
                          m10 = FetchCirrusTex(iuv0.get<0>(), iuv1.get<1>()),
                          m11 = FetchCirrusTex(iuv1.get<0>(), iuv1.get<1>());

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec2 m0 = m01 * k.get<0>() + m00 * (1.0f - k.get<0>()),
                          m1 = m11 * k.get<0>() + m10 * (1.0f - k.get<0>());

    return srgb_to_rgb((m1 * k.get<1>() + m0 * (1.0f - k.get<1>())) * (1.0f / 255.0f));
}

} // namespace Ray

Ray::Ref::simd_fvec4 Ray::IntegrateOpticalDepth(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                                                const Ref::simd_fvec4 &ray_dir) {
    Ref::simd_fvec2 intersection = AtmosphereIntersection(params, ray_start, ray_dir);
    float ray_length = intersection[1];

    const int SampleCount = 64;
    float step_size = ray_length / SampleCount;

    Ref::simd_fvec4 optical_depth = 0.0f;

    for (int i = 0; i < SampleCount; i++) {
        Ref::simd_fvec4 local_pos = ray_start + ray_dir * (i + 0.5f) * step_size, up_vector;
        const float local_height = AtmosphereHeight(params, local_pos, up_vector);
        const atmosphere_medium_t medium = SampleAtmosphereMedium(params, local_height);
        optical_depth += medium.extinction * step_size;
    }

    return optical_depth;
}

template <bool ExpSampleDistribution, bool UniformPhase>
std::pair<Ray::Ref::simd_fvec4, Ray::Ref::simd_fvec4>
Ray::IntegrateScatteringMain(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
                             const Ref::simd_fvec4 &ray_dir, float ray_length, const Ref::simd_fvec4 &light_dir,
                             const Ref::simd_fvec4 &moon_dir, const Ref::simd_fvec4 &light_color,
                             Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
                             const float rand_offset, const int sample_count, Ref::simd_fvec4 &inout_transmittance) {
    const Ref::simd_fvec2 atm_intersection = AtmosphereIntersection(params, ray_start, ray_dir);
    ray_length = fminf(ray_length, atm_intersection.get<1>());
    const Ref::simd_fvec2 planet_intersection = PlanetIntersection(params, ray_start, ray_dir);
    if (planet_intersection.get<0>() > 0) {
        ray_length = fminf(ray_length, planet_intersection.get<0>());
    }

    Ref::simd_fvec4 _unused;
    const float ray_height = AtmosphereHeight(params, ray_start, _unused);
    const float sample_distribution_exponent = 1.0f + saturate(1.0f - ray_height / params.atmosphere_height) * 8.0f;

    const float costh = dot(ray_dir, light_dir);
    const float phase_r = PhaseRayleigh(costh), phase_m = PhaseMie(costh);

    const float moon_costh = dot(ray_dir, moon_dir);
    const float moon_phase_r = PhaseRayleigh(moon_costh), moon_phase_m = PhaseMie(moon_costh);

    const float phase_uniform = 1.0f / (4.0f * PI);

    Ref::simd_fvec4 radiance = 0.0f, multiscat_as_1 = 0.0f;

    //
    // Atmosphere
    //
    float prev_ray_time = 0;
    for (int i = 0; i < sample_count; ++i) {
        float ray_time;
        if (ExpSampleDistribution) {
            ray_time = powf(float(i) / sample_count, sample_distribution_exponent) * ray_length;
        } else {
            ray_time = (float(i) / sample_count) * ray_length;
        }
        const float step_size = (ray_time - prev_ray_time);

        const Ref::simd_fvec4 local_position = ray_start + ray_dir * (ray_time - 0.1f * rand_offset * step_size);
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);
        const atmosphere_medium_t medium = SampleAtmosphereMedium(params, local_height);
        const Ref::simd_fvec4 optical_depth = medium.extinction * step_size;
        const Ref::simd_fvec4 local_transmittance = exp(-optical_depth);

        Ref::simd_fvec4 S = 0.0f;

        if (light_dir.get<1>() > -0.025f) {
            // main light contribution
            const float view_zenith_cos_angle = dot(light_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

            const Ref::simd_fvec2 planet_intersection = PlanetIntersection(params, local_position, light_dir);
            const float planet_shadow = planet_intersection.get<0>() > 0 ? 0.0f : 1.0f;

            Ref::simd_fvec4 multiscattered_lum = 0.0f;
            if (!multiscatter_lut.empty()) {
                Ref::simd_fvec2 uv = saturate(
                    Ref::simd_fvec2(view_zenith_cos_angle * 0.5f + 0.5f, local_height / params.atmosphere_height));
                uv = Ref::simd_fvec2(from_unit_to_sub_uvs(uv.get<0>(), MULTISCATTER_LUT_RES),
                                     from_unit_to_sub_uvs(uv.get<1>(), MULTISCATTER_LUT_RES));

                multiscattered_lum = SampleMultiscatterLUT(multiscatter_lut, uv);
            }

            const Ref::simd_fvec4 phase_times_scattering =
                UniformPhase ? medium.scattering * phase_uniform
                             : medium.scattering_ray * phase_r + medium.scattering_mie * phase_m;
            S += (planet_shadow * light_transmittance * phase_times_scattering +
                  multiscattered_lum * medium.scattering) *
                 light_color;
        } else if (params.moon_radius > 0.0f) {
            // moon reflection contribution  (totally fake)
            const float view_zenith_cos_angle = dot(moon_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

            Ref::simd_fvec4 multiscattered_lum = 0.0f;
            if (!multiscatter_lut.empty()) {
                Ref::simd_fvec2 uv = saturate(
                    Ref::simd_fvec2(view_zenith_cos_angle * 0.5f + 0.5f, local_height / params.atmosphere_height));
                uv = Ref::simd_fvec2(from_unit_to_sub_uvs(uv.get<0>(), MULTISCATTER_LUT_RES),
                                     from_unit_to_sub_uvs(uv.get<1>(), MULTISCATTER_LUT_RES));

                multiscattered_lum = SampleMultiscatterLUT(multiscatter_lut, uv);
            }

            const Ref::simd_fvec4 phase_times_scattering =
                medium.scattering_ray * moon_phase_r + medium.scattering_mie * moon_phase_m;
            S += MoonSunRelation *
                 (light_transmittance * phase_times_scattering + multiscattered_lum * medium.scattering) * light_color;
        }

        // 1 is the integration of luminance over the 4pi of a sphere, and assuming an isotropic phase function
        // of 1.0/(4*PI)
        const Ref::simd_fvec4 MS = medium.scattering * 1.0f;
        const Ref::simd_fvec4 MS_int = (MS - MS * local_transmittance) / medium.extinction;
        multiscat_as_1 += inout_transmittance * MS_int;

#if 0
        radiance += inout_transmittance * S * step_size;
#else
        const Ref::simd_fvec4 S_int = (S - S * local_transmittance) / medium.extinction;
        radiance += inout_transmittance * S_int;
#endif
        inout_transmittance *= local_transmittance;

        prev_ray_time = ray_time;
    }

    //
    // Ground 'floor'
    //
    if (planet_intersection.get<0>() > 0) {
        const Ref::simd_fvec4 local_position = ray_start + ray_dir * planet_intersection.get<0>();
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);

        const float view_zenith_cos_angle = dot(light_dir, up_vector);
        const Ref::simd_fvec2 uv =
            LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
        const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
        radiance += Ref::simd_fvec4{params.ground_albedo, Ref::simd_mem_aligned} * saturate(dot(up_vector, light_dir)) *
                    inout_transmittance * light_transmittance * light_color;
    }

    return std::make_pair(radiance, multiscat_as_1);
}

template std::pair<Ray::Ref::simd_fvec4, Ray::Ref::simd_fvec4> Ray::IntegrateScatteringMain<false, false>(
    const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir,
    float ray_length, const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &moon_dir,
    const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
    float rand_offset, int sample_count, Ref::simd_fvec4 &inout_transmittance);
template std::pair<Ray::Ref::simd_fvec4, Ray::Ref::simd_fvec4> Ray::IntegrateScatteringMain<false, true>(
    const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir,
    float ray_length, const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &moon_dir,
    const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
    float rand_offset, int sample_count, Ref::simd_fvec4 &inout_transmittance);
template std::pair<Ray::Ref::simd_fvec4, Ray::Ref::simd_fvec4> Ray::IntegrateScatteringMain<true, false>(
    const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir,
    float ray_length, const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &moon_dir,
    const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
    float rand_offset, int sample_count, Ref::simd_fvec4 &inout_transmittance);
template std::pair<Ray::Ref::simd_fvec4, Ray::Ref::simd_fvec4> Ray::IntegrateScatteringMain<true, true>(
    const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start, const Ref::simd_fvec4 &ray_dir,
    float ray_length, const Ref::simd_fvec4 &light_dir, const Ref::simd_fvec4 &moon_dir,
    const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut, Span<const float> multiscatter_lut,
    float rand_offset, int sample_count, Ref::simd_fvec4 &inout_transmittance);

Ray::Ref::simd_fvec4 Ray::IntegrateScattering(const atmosphere_params_t &params, Ref::simd_fvec4 ray_start,
                                              const Ref::simd_fvec4 &ray_dir, float ray_length,
                                              const Ref::simd_fvec4 &light_dir, const float light_angle,
                                              const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut,
                                              Span<const float> multiscatter_lut, uint32_t rand_hash) {
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

    const Ref::simd_fvec2 moon_intersection = MoonIntersection(params, ray_start, ray_dir);
    Ref::simd_fvec4 moon_dir = Ref::simd_fvec4{params.moon_dir, Ref::simd_mem_aligned};
    const Ref::simd_fvec4 moon_point = moon_dir * params.moon_distance + 0.5f * light_dir * params.moon_radius;
    moon_dir = normalize(moon_point);

    const float costh = dot(ray_dir, light_dir);
    const Ref::simd_fvec4 phase_w = PhaseWrenninge(costh);

    const float moon_costh = dot(ray_dir, moon_dir);
    const Ref::simd_fvec4 moon_phase_w = PhaseWrenninge(moon_costh);

    const int PreAtmosphereSampleCount = 8, MainAtmosphereSampleCount = 24, CloudsSampleCount = 128;

    const float light_brightness = light_color.get<0>() + light_color.get<1>() + light_color.get<2>();

    Ref::simd_fvec4 total_radiance = 0.0f, total_transmittance = 1.0f;

    const Ref::simd_fvec4 clouds_intersection = CloudsIntersection(params, ray_start, ray_dir);

    //
    // Atmosphere before clouds
    //
    if (clouds_intersection.get<1>() > 0 && light_brightness > 0.0f) {
        const float pre_atmosphere_ray_length = fminf(ray_length, clouds_intersection.get<1>());

        const float rand_offset = Ref::construct_float(rand_hash);
        rand_hash = Ref::hash(rand_hash);

        total_radiance +=
            IntegrateScatteringMain<false>(params, ray_start, ray_dir, pre_atmosphere_ray_length, light_dir, moon_dir,
                                           light_color, transmittance_lut, multiscatter_lut, rand_offset,
                                           PreAtmosphereSampleCount, total_transmittance)
                .first;
    }

    //
    // Main clouds
    //
    const float CloudsHorizonCutoff = 0.005f;
    if (planet_intersection.get<0>() < 0 && clouds_intersection.get<1>() > 0 && params.clouds_density > 0.0f &&
        light_brightness > 0.0f && ray_dir.get<1>() > CloudsHorizonCutoff) {

        float clouds_ray_length = fminf(ray_length, clouds_intersection.get<3>());
        Ref::simd_fvec4 clouds_ray_start = ray_start + ray_dir * clouds_intersection.get<1>();
        clouds_ray_length -= clouds_intersection.get<1>();

        if (clouds_ray_length > 0.0f) {
            const float step_size = clouds_ray_length / float(CloudsSampleCount);

            Ref::simd_fvec4 local_position = clouds_ray_start + ray_dir * Ref::construct_float(rand_hash) * step_size;
            rand_hash = Ref::hash(rand_hash);

            Ref::simd_fvec4 clouds = 0.0f;

            // NOTE: We assume transmittance is constant along the clouds range (~500m)
            Ref::simd_fvec4 light_transmittance, moon_transmittance, multiscattered_lum = 0.0f,
                                                                     moon_multiscattered_lum = 0.0f;
            {
                Ref::simd_fvec4 up_vector;
                const float local_height = AtmosphereHeight(params, local_position, up_vector);
                {
                    const float view_zenith_cos_angle = dot(light_dir, up_vector);
                    const Ref::simd_fvec2 uv =
                        LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
                    light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

                    if (!multiscatter_lut.empty()) {
                        Ref::simd_fvec2 uv = saturate(Ref::simd_fvec2(view_zenith_cos_angle * 0.5f + 0.5f,
                                                                      local_height / params.atmosphere_height));
                        uv = Ref::simd_fvec2(from_unit_to_sub_uvs(uv.get<0>(), MULTISCATTER_LUT_RES),
                                             from_unit_to_sub_uvs(uv.get<1>(), MULTISCATTER_LUT_RES));

                        multiscattered_lum = SampleMultiscatterLUT(multiscatter_lut, uv);
                    }
                }
                {
                    const float view_zenith_cos_angle = dot(moon_dir, up_vector);
                    const Ref::simd_fvec2 uv =
                        LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
                    moon_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

                    if (!multiscatter_lut.empty()) {
                        Ref::simd_fvec2 uv = saturate(Ref::simd_fvec2(view_zenith_cos_angle * 0.5f + 0.5f,
                                                                      local_height / params.atmosphere_height));
                        uv = Ref::simd_fvec2(from_unit_to_sub_uvs(uv.get<0>(), MULTISCATTER_LUT_RES),
                                             from_unit_to_sub_uvs(uv.get<1>(), MULTISCATTER_LUT_RES));

                        moon_multiscattered_lum = SampleMultiscatterLUT(multiscatter_lut, uv);
                    }
                }
            }

            Ref::simd_fvec4 transmittance_before = total_transmittance;

            for (int i = 0; i < CloudsSampleCount /*&& total_transmittance.get<1>() > 0.001f*/; ++i) {
                float local_height, height_fraction;
                Ref::simd_fvec4 up_vector;
                const float local_density =
                    GetCloudsDensity(params, local_position, local_height, height_fraction, up_vector);
                if (local_density > 0.0f) {
                    const float local_transmittance = expf(-local_density * step_size);
                    const float ambient_visibility = (0.75f + 1.5f * fmaxf(0.0f, height_fraction - 0.1f));

                    if (light_dir.get<1>() > -0.025f) {
                        // main light contribution
                        const Ref::simd_fvec2 planet_intersection =
                            PlanetIntersection(params, local_position, light_dir);
                        const float planet_shadow = planet_intersection.get<0>() > 0 ? 0.0f : 1.0f;
                        const float cloud_shadow = TraceCloudShadow(params, rand_hash, local_position, light_dir);

                        clouds += total_transmittance *
                                  (planet_shadow * GetLightEnergy(cloud_shadow, local_density, phase_w) +
                                   ambient_visibility * multiscattered_lum) *
                                  (1.0f - local_transmittance) * light_transmittance;
                    } else if (params.moon_radius > 0.0f) {
                        // moon reflection contribution (totally fake)
                        const float cloud_shadow = TraceCloudShadow(params, rand_hash, local_position, moon_dir);

                        clouds += MoonSunRelation * total_transmittance *
                                  (GetLightEnergy(cloud_shadow, local_density, moon_phase_w) +
                                   ambient_visibility * moon_multiscattered_lum) *
                                  (1.0f - local_transmittance) * moon_transmittance;
                    }

                    total_transmittance *= local_transmittance;
                }
                local_position += ray_dir * step_size;
            }

            // NOTE: totally arbitrary cloud blending
            float cloud_blend = linstep(CloudsHorizonCutoff + 0.25f, CloudsHorizonCutoff, ray_dir.get<1>());
            cloud_blend = 1.0f - powf(cloud_blend, 5.0f);

            total_radiance += cloud_blend * clouds * light_color;
            total_transmittance = mix(transmittance_before, total_transmittance, cloud_blend);
        }
    }

    //
    // Cirrus clouds
    //
    if (planet_intersection.get<0>() < 0 && clouds_intersection.get<1>() > 0 && params.cirrus_clouds_amount > 0.0f &&
        light_brightness > 0.0f) {
        Ref::simd_fvec2 cirrus_coords =
            3e-4f * Ref::simd_fvec2{params.clouds_offset_z, params.clouds_offset_x} +
            0.8f * (Ref::simd_fvec2{ray_dir.get<2>(), ray_dir.get<0>()}) / (fabsf(ray_dir.get<1>()) + 0.02f);
        cirrus_coords.set<1>(cirrus_coords.get<1>() + 1.75f);

        float noise_read = 1.0f - Sample3dNoiseTex(fract(Ref::simd_fvec4{0.0f, cirrus_coords.get<0>() * 0.03f,
                                                                         cirrus_coords.get<1>() * 0.03f, 0.0f}));
        noise_read =
            saturate(noise_read - 1.0f + params.cirrus_clouds_amount * 0.6f) / (params.cirrus_clouds_amount + 1e-9f);

        float dC = 1.2f * smoothstep(0.0f, 1.0f, noise_read) * SampleCirrusTex(fract(cirrus_coords * 0.5f)).get<0>();

        //
        cirrus_coords.set<0>(cirrus_coords.get<0>() + 0.25f);
        noise_read = 1.0f - Sample3dNoiseTex(fract(Ref::simd_fvec4{0.7f, cirrus_coords.get<0>() * 0.02f,
                                                                   cirrus_coords.get<1>() * 0.02f, 0.0f}));
        noise_read =
            saturate(noise_read - 1.0f + params.cirrus_clouds_amount * 0.7f) / (params.cirrus_clouds_amount + 1e-9f);

        dC += 0.6f * smoothstep(0.0f, 1.0f, noise_read) * SampleCirrusTex(fract(cirrus_coords * 0.25f)).get<1>();

        Ref::simd_fvec4 local_position = ray_start + ray_dir * params.cirrus_clouds_height;

        Ref::simd_fvec4 light_transmittance, moon_transmittance;
        {
            Ref::simd_fvec4 up_vector;
            const float local_height = AtmosphereHeight(params, local_position, up_vector);
            {
                const float view_zenith_cos_angle = dot(light_dir, up_vector);
                const Ref::simd_fvec2 uv =
                    LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
                light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
            }
            {
                const float view_zenith_cos_angle = dot(moon_dir, up_vector);
                const Ref::simd_fvec2 uv =
                    LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
                moon_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
            }
        }

        if (light_dir.get<1>() > -0.025f) {
            total_radiance +=
                total_transmittance * GetLightEnergy(0.002f, dC, phase_w) * light_transmittance * dC * light_color;
        } else if (params.moon_radius > 0.0f) {
            total_radiance += MoonSunRelation * total_transmittance * GetLightEnergy(0.002f, dC, moon_phase_w) *
                              moon_transmittance * dC * light_color;
        }
        total_transmittance *= expf(-dC * 0.002f * 1000.0f);
    }

    //
    // Main atmosphere
    //
    if (planet_intersection.get<0>() < 0 && light_brightness > 0.0f) {
        float main_ray_length = ray_length;
        Ref::simd_fvec4 main_ray_start = ray_start + ray_dir * clouds_intersection.get<3>();
        main_ray_length -= clouds_intersection.get<1>();

        const float rand_offset = Ref::construct_float(rand_hash);
        rand_hash = Ref::hash(rand_hash);

        total_radiance += IntegrateScatteringMain(params, main_ray_start, ray_dir, main_ray_length, light_dir, moon_dir,
                                                  light_color, transmittance_lut, multiscatter_lut, rand_offset,
                                                  MainAtmosphereSampleCount, total_transmittance)
                              .first;
    }

    //
    // Sun disk (bake directional light into the texture)
    //
    if (light_angle > 0.0f && planet_intersection.get<0>() < 0.0f && light_brightness > 0.0f) {
        const float cos_theta = cosf(light_angle);
        const float BlendVal = 0.000005f;
        Ref::simd_fvec4 sun_disk = total_transmittance * smoothstep(cos_theta - BlendVal, cos_theta + BlendVal, costh);
        // 'de-multiply' by disk area (to get original brightness)
        const float radius = tanf(light_angle);
        sun_disk /= (PI * radius * radius);

        total_radiance += sun_disk * light_color;
    }

    //
    // Stars
    //
    if (params.stars_brightness > 0.0f && planet_intersection.get<0>() < 0 && moon_intersection.get<0>() < 0) {
        const float StarsThreshold = 10.0f;
        total_radiance +=
            total_transmittance *
            (powf(clamp(stars_noise(ray_dir * 200.0f), 0.0f, 1.0f), StarsThreshold) * params.stars_brightness);
    }

    //
    // Moon
    //
    if (planet_intersection.get<0>() < 0 && moon_intersection.get<0>() > 0 && params.moon_radius > 0.0f &&
        light_brightness > 0.0f) {
        const Ref::simd_fvec4 moon_center =
            Ref::simd_fvec4{params.moon_dir, Ref::simd_mem_aligned} * params.moon_distance;
        const Ref::simd_fvec4 moon_normal = normalize(ray_start + moon_intersection.get<0>() * ray_dir - moon_center);

        const float theta = acosf(clamp(moon_normal.get<1>(), -1.0f, 1.0f)) / PI;

        float phi = atan2f(moon_normal.get<2>(), moon_normal.get<0>());
        if (phi < 0) {
            phi += 2 * PI;
        }
        if (phi > 2 * PI) {
            phi -= 2 * PI;
        }

        const float u = Ref::fract(0.5f * phi / PI);

        const Ref::simd_fvec2 uvs = Ref::simd_fvec2(u, theta);
        const Ref::simd_fvec4 albedo = SampleMoonTex(uvs);

        total_radiance += total_transmittance * fmaxf(dot(moon_normal, light_dir), 0.0f) * albedo;
    }

    return total_radiance;
}

void Ray::UvToLutTransmittanceParams(const atmosphere_params_t &params, Ref::simd_fvec2 uv, float &view_height,
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

Ray::Ref::simd_fvec2 Ray::LutTransmittanceParamsToUv(const atmosphere_params_t &params, const float view_height,
                                                     const float view_zenith_cos_angle) {
    const float top_radius = params.planet_radius + params.atmosphere_height;

    const float H = sqrtf(fmaxf(0.0f, top_radius * top_radius - params.planet_radius * params.planet_radius));
    const float rho = sqrtf(fmaxf(0.0f, view_height * view_height - params.planet_radius * params.planet_radius));

    const float discriminant =
        view_height * view_height * (view_zenith_cos_angle * view_zenith_cos_angle - 1.0f) + top_radius * top_radius;
    const float d =
        fmaxf(0.0f, (-view_height * view_zenith_cos_angle + sqrtf(discriminant))); // Distance to atmosphere boundary

    const float d_min = top_radius - view_height;
    const float d_max = rho + H;
    const float x_mu = (d - d_min) / (d_max - d_min);
    const float x_r = rho / H;

    return Ref::simd_fvec2{x_mu, x_r};
}
