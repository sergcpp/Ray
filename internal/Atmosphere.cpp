#include "Atmosphere.h"

#include <cmath>

namespace Ray {
#include "precomputed/__3d_noise_tex.inl"
#include "precomputed/__moon_tex.inl"
#include "precomputed/__weather_tex.inl"

force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline float saturate(const float val) { return clamp(val, 0.0f, 1.0f); }

force_inline float remap(float value, float original_min) {
    return saturate((value - original_min) / (1.000001f - original_min));
}

force_inline float mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }

// GPU PRO 7 - Real-time Volumetric Cloudscapes
// https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
// https://github.com/sebh/TileableVolumeNoise
force_inline float remap(float value, float original_min, float original_max, float new_min, float new_max) {
    return new_min +
           (saturate((value - original_min) / (0.0000001f + original_max - original_min)) * (new_max - new_min));
}

force_inline float linstep(float smin, float smax, float x) { return saturate((x - smin) / (smax - smin)); }

force_inline float smoothstep(float edge0, float edge1, float x) {
    const float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
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

    return Ref::simd_fvec4{density_rayleigh.get<0>(), density_mie.get<0>(), density_ozone, 0.0f};
}

Ref::simd_fvec4 SampleTransmittanceLUT(Span<const float> transmittance_lut, Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    auto iuv0 = Ref::simd_ivec2(uv);
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});
    const Ref::simd_ivec2 iuv1 = min(iuv0 + 1, Ref::simd_ivec2{TRANSMITTANCE_LUT_W - 1, TRANSMITTANCE_LUT_H - 1});

    const auto tr00 = Ref::simd_fvec4(&transmittance_lut[4 * (iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               tr01 = Ref::simd_fvec4(&transmittance_lut[4 * (iuv0.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>())],
                                      Ref::simd_mem_aligned),
               tr10 = Ref::simd_fvec4(&transmittance_lut[4 * (iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv0.get<0>())],
                                      Ref::simd_mem_aligned),
               tr11 = Ref::simd_fvec4(&transmittance_lut[4 * (iuv1.get<1>() * TRANSMITTANCE_LUT_W + iuv1.get<0>())],
                                      Ref::simd_mem_aligned);

    const Ref::simd_fvec2 k = fract(uv);

    const Ref::simd_fvec4 tr0 = tr01 * k.get<0>() + tr00 * (1.0f - k.get<0>()),
                          tr1 = tr11 * k.get<0>() + tr10 * (1.0f - k.get<0>());

    return (tr1 * k.get<1>() + tr0 * (1.0f - k.get<1>()));
}

force_inline Ref::simd_fvec4 FetchWeatherTex(const int x, const int y) {
    return Ref::simd_fvec4{
        float(__weather_tex[4 * (y * WEATHER_TEX_W + x) + 0]), float(__weather_tex[4 * (y * WEATHER_TEX_W + x) + 1]),
        float(__weather_tex[4 * (y * WEATHER_TEX_W + x) + 2]), float(__weather_tex[4 * (y * WEATHER_TEX_W + x) + 3])};
}

Ref::simd_fvec4 SampleWeatherTex(Ref::simd_fvec2 uv) {
    uv = uv * Ref::simd_fvec2(WEATHER_TEX_W, WEATHER_TEX_H);
    auto iuv0 = Ref::simd_ivec2{uv};
    iuv0 = clamp(iuv0, Ref::simd_ivec2{0, 0}, Ref::simd_ivec2{WEATHER_TEX_W - 1, WEATHER_TEX_H - 1});
    const Ref::simd_ivec2 iuv1 = (iuv0 + 1) & Ref::simd_ivec2{WEATHER_TEX_W - 1, WEATHER_TEX_H - 1};

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
    return __3d_noise_tex[z * NOISE_3D_RES * NOISE_3D_RES + y * NOISE_3D_RES + x] / 255.0f;
}

float Sample3dNoiseTex(const Ref::simd_fvec4 &uvw) {
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

    return (1.0f - k.get<2>()) * n0xx + k.get<2>() * n1xx;
}

float GetCloudsDensity(const atmosphere_params_t &params, Ref::simd_fvec4 local_position, float &out_local_height,
                       Ref::simd_fvec4 &out_up_vector) {
    out_local_height = AtmosphereHeight(params, local_position, out_up_vector);

    Ref::simd_fvec2 weather_uv = {local_position.get<0>() + params.clouds_offset_x,
                                  local_position.get<2>() + params.clouds_offset_z};
    weather_uv = fract(weather_uv * 0.00007f);

    Ref::simd_fvec4 weather_sample = SampleWeatherTex(weather_uv);
    // weather_sample = srgb_to_rgb(weather_sample);

    const float height_fraction =
        (out_local_height - params.clouds_height_beg) / (params.clouds_height_end - params.clouds_height_beg);

    float cloud_coverage = mix(weather_sample.get<2>(), weather_sample.get<1>(), params.clouds_variety);
    cloud_coverage = remap(cloud_coverage, saturate(1.0f - params.clouds_density + 0.5f * height_fraction));

    float cloud_type = weather_sample.get<0>();
    cloud_coverage *= GetDensityHeightGradientForPoint(height_fraction, cloud_type);

    if (height_fraction > 1.0f || cloud_coverage < 0.01f) {
        return 0.0f;
    }

    local_position /= 1.5f * (params.clouds_height_end - params.clouds_height_beg);

    local_position = fract(local_position);
    local_position *= NOISE_3D_RES;

    const float noise_read = 1.0f - Sample3dNoiseTex(local_position);

    return remap(cloud_coverage, 0.6f * noise_read);
    return 0.12f * mix(fmaxf(0.0f, 1.0f - cloud_type * 2.0f), 1.0f, height_fraction) *
           powf(5.0f * remap(cloud_coverage, 0.6f * noise_read), 1.0f - height_fraction);
}

float TraceCloudShadow(const atmosphere_params_t &params, const uint32_t rand_hash, Ref::simd_fvec4 ray_start,
                       const Ref::simd_fvec4 &ray_dir) {
    const Ref::simd_fvec4 clouds_intersection = CloudsIntersection(params, ray_start, ray_dir);
    if (clouds_intersection.get<3>() > 0) {
        const int SampleCount = 24;
        const float StepSize = 16.0f;

        Ref::simd_fvec4 pos = ray_start + Ref::construct_float(rand_hash) * StepSize;

        float ret = 0.0f;
        for (int i = 0; i < SampleCount; ++i) {
            float local_height;
            Ref::simd_fvec4 up_vector;
            const float local_density = GetCloudsDensity(params, pos, local_height, up_vector);
            ret += local_density;
            pos += ray_dir * StepSize;
        }

        return ret * StepSize;
    }
    return 1.0f;
}

// https://www.shadertoy.com/view/NtsBzB
Ref::simd_fvec4 _hash(Ref::simd_fvec4 p) {
    p = Ref::simd_fvec4{dot(p, Ref::simd_fvec4{127.1f, 311.7f, 74.7f, 0.0f}),
                        dot(p, Ref::simd_fvec4{269.5f, 183.3f, 246.1f, 0.0f}),
                        dot(p, Ref::simd_fvec4{113.5f, 271.9f, 124.6f, 0.0f}), 0.0f};

    p.set<0>(sinf(p.get<0>()));
    p.set<1>(sinf(p.get<1>()));
    p.set<2>(sinf(p.get<2>()));

    return -1.0f + 2.0f * fract(p * 43758.5453123f);
}

float noise(const Ref::simd_fvec4 &p) {
    Ref::simd_fvec4 i = floor(p);
    Ref::simd_fvec4 f = fract(p);

    Ref::simd_fvec4 u = f * f * (3.0f - 2.0f * f);

    return mix(
        mix(mix(dot(_hash(i + Ref::simd_fvec4(0.0f, 0.0f, 0.0f, 0.0f)), f - Ref::simd_fvec4(0.0f, 0.0f, 0.0f, 0.0f)),
                dot(_hash(i + Ref::simd_fvec4(1.0f, 0.0f, 0.0f, 0.0f)), f - Ref::simd_fvec4(1.0f, 0.0f, 0.0f, 0.0f)),
                u.get<0>()),
            mix(dot(_hash(i + Ref::simd_fvec4(0.0f, 1.0f, 0.0f, 0.0f)), f - Ref::simd_fvec4(0.0f, 1.0f, 0.0f, 0.0f)),
                dot(_hash(i + Ref::simd_fvec4(1.0f, 1.0f, 0.0f, 0.0f)), f - Ref::simd_fvec4(1.0f, 1.0f, 0.0f, 0.0f)),
                u.get<0>()),
            u.get<1>()),
        mix(mix(dot(_hash(i + Ref::simd_fvec4(0.0f, 0.0f, 1.0f, 0.0f)), f - Ref::simd_fvec4(0.0f, 0.0f, 1.0f, 0.0f)),
                dot(_hash(i + Ref::simd_fvec4(1.0f, 0.0f, 1.0f, 0.0f)), f - Ref::simd_fvec4(1.0f, 0.0f, 1.0f, 0.0f)),
                u.get<0>()),
            mix(dot(_hash(i + Ref::simd_fvec4(0.0f, 1.0f, 1.0f, 0.0f)), f - Ref::simd_fvec4(0.0f, 1.0f, 1.0f, 0.0f)),
                dot(_hash(i + Ref::simd_fvec4(1.0f, 1.0f, 1.0f, 0.0f)), f - Ref::simd_fvec4(1.0f, 1.0f, 1.0f, 0.0f)),
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

} // namespace Ray

Ray::Ref::simd_fvec4 Ray::IntegrateOpticalDepth(const atmosphere_params_t &params, const Ref::simd_fvec4 &ray_start,
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
Ray::Ref::simd_fvec4 Ray::Absorb(const atmosphere_params_t &params, const Ref::simd_fvec4 &opticalDepth) {
    // Note that Mie results in slightly more light absorption than scattering, about 10%
    return exp(-(opticalDepth.get<0>() * Ref::simd_fvec4{params.rayleigh_scattering, Ref::simd_mem_aligned} +
                 opticalDepth.get<1>() * Ref::simd_fvec4{params.mie_scattering, Ref::simd_mem_aligned} * 1.1f +
                 opticalDepth.get<2>() * Ref::simd_fvec4{params.ozone_absorbtion, Ref::simd_mem_aligned}) *
               params.atmosphere_density);
}

Ray::Ref::simd_fvec4 Ray::IntegrateScattering(const atmosphere_params_t &params, Ref::simd_fvec4 ray_start,
                                              const Ref::simd_fvec4 &ray_dir, float ray_length,
                                              const Ref::simd_fvec4 &light_dir, const float light_angle,
                                              const Ref::simd_fvec4 &light_color, Span<const float> transmittance_lut,
                                              uint32_t rand_hash, Ref::simd_fvec4 &transmittance) {
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

    const Ref::simd_fvec2 moon_intersection = MoonIntersection(params, ray_start, ray_dir);
    Ref::simd_fvec4 moon_dir = Ref::simd_fvec4{params.moon_dir, Ref::simd_mem_aligned};
    const Ref::simd_fvec4 moon_point = moon_dir * params.moon_distance + 0.5f * light_dir * params.moon_radius;
    moon_dir = normalize(moon_point);

    const float costh = dot(ray_dir, light_dir);
    const float phase_r = PhaseRayleigh(costh), phase_m = PhaseMie(costh);

    const float moon_costh = dot(ray_dir, moon_dir);
    const float moon_phase_r = PhaseRayleigh(moon_costh), moon_phase_m = PhaseMie(moon_costh);

    const int SampleCount = 32;

    Ref::simd_fvec4 optical_depth = 0.0f, rayleigh = 0.0f, mie = 0.0f;

    const float rand_offset = Ref::construct_float(rand_hash);
    rand_hash = Ref::hash(rand_hash);

    const float light_brightness = light_color.get<0>() + light_color.get<1>() + light_color.get<2>();

    //
    // Main atmosphere
    //
    float prev_ray_time = 0;
    for (int i = 0; i < SampleCount && light_brightness > 0.0f; ++i) {
        const float ray_time = powf(float(i) / SampleCount, sample_distribution_exponent) * ray_length;
        const float step_size = (ray_time - prev_ray_time);

        const Ref::simd_fvec4 local_position = ray_start + ray_dir * (ray_time - 0.1f * rand_offset * step_size);
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);
        const Ref::simd_fvec4 local_density = AtmosphereDensity(params, local_height);

        optical_depth += local_density * step_size;

        // The atmospheric transmittance from ray_start to local_osition
        const Ref::simd_fvec4 view_transmittance = Absorb(params, optical_depth);

        { // main light contribution
            const float view_zenith_cos_angle = dot(light_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

            const Ref::simd_fvec2 planet_intersection = PlanetIntersection(params, local_position, light_dir);
            const float planet_shadow = planet_intersection.get<0>() > 0 ? 0.0f : 1.0f;

            rayleigh +=
                planet_shadow * view_transmittance * light_transmittance * phase_r * local_density.get<0>() * step_size;
            mie +=
                planet_shadow * view_transmittance * light_transmittance * phase_m * local_density.get<1>() * step_size;
        }

        if (params.moon_radius > 0.0f) {
            // moon reflection contribution  (totally fake)
            const float view_zenith_cos_angle = dot(moon_dir, up_vector);
            const Ref::simd_fvec2 uv =
                LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
            const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);

            rayleigh +=
                0.0001f * view_transmittance * light_transmittance * moon_phase_r * local_density.get<0>() * step_size;
            mie +=
                0.0001f * view_transmittance * light_transmittance * moon_phase_m * local_density.get<1>() * step_size;
        }

        prev_ray_time = ray_time;
    }

    transmittance = Absorb(params, optical_depth);

    Ref::simd_fvec4 total_radiance = (rayleigh * Ref::simd_fvec4{params.rayleigh_scattering, Ref::simd_mem_aligned} +
                                      mie * Ref::simd_fvec4{params.mie_scattering, Ref::simd_mem_aligned}) *
                                     light_color;

    //
    // Main clouds
    //
    const Ref::simd_fvec4 clouds_intersection = CloudsIntersection(params, ray_start, ray_dir);
    if (planet_intersection.get<0>() < 0 && clouds_intersection.get<1>() > 0 && params.clouds_density > 0.0f &&
        light_brightness > 0.0f) {
        ray_length = fminf(ray_length, clouds_intersection.get<3>());
        ray_start += ray_dir * clouds_intersection.get<1>();
        ray_length -= clouds_intersection.get<1>();

        if (ray_length > 0.0f) {
            const int SampleCount = 128;
            const float step_size = ray_length / float(SampleCount);

            const Ref::simd_fvec4 phase_w = PhaseWrenninge(costh), moon_phase_w = PhaseWrenninge(moon_costh);

            // NOTE: this is incorrect, must use transmittance before clouds, but it's ok
            Ref::simd_fvec4 view_transmittance = transmittance;

            Ref::simd_fvec4 local_position = ray_start + Ref::construct_float(rand_hash) * step_size;
            rand_hash = Ref::hash(rand_hash);

            Ref::simd_fvec4 clouds = 0.0f;

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

            for (int i = 0; i < SampleCount; ++i) {
                float local_height;
                Ref::simd_fvec4 up_vector;
                const float local_density = GetCloudsDensity(params, local_position, local_height, up_vector);
                if (local_density > 0.0f) {
                    const float local_transmittance = expf(-local_density * step_size);

                    { // main light contribution
                        const Ref::simd_fvec2 planet_intersection =
                            PlanetIntersection(params, local_position, light_dir);
                        const float planet_shadow = planet_intersection.get<0>() > 0 ? 0.0f : 1.0f;

                        const float cloud_shadow = TraceCloudShadow(params, rand_hash, local_position, light_dir);

                        clouds += planet_shadow * view_transmittance *
                                  GetLightEnergy(cloud_shadow, local_density, phase_w) * (1.0f - local_transmittance) *
                                  light_transmittance;
                    }

                    if (params.moon_radius > 0.0f) {
                        // moon reflection contribution (totally fake)
                        const float cloud_shadow = TraceCloudShadow(params, rand_hash, local_position, moon_dir);

                        clouds += 0.0001f * view_transmittance *
                                  GetLightEnergy(cloud_shadow, local_density, moon_phase_w) *
                                  (1.0f - local_transmittance) * moon_transmittance;
                    }

                    view_transmittance *= local_transmittance;
                }
                local_position += ray_dir * step_size;
            }

            clouds *= step_size;
            transmittance = view_transmittance;

            // NOTE: totally arbitrary cloud blending
            float cloud_blend = fmaxf(0.15f, ray_dir.get<1>());
            clouds *= 0.5f * cloud_blend * cloud_blend;

            total_radiance += clouds * light_color;
        }
    }

    //
    // Ground 'floor'
    //
    if (planet_intersection.get<0>() > 0 && light_brightness > 0.0f) {
        const Ref::simd_fvec4 local_position = ray_start + ray_dir * ray_length;
        Ref::simd_fvec4 up_vector;
        const float local_height = AtmosphereHeight(params, local_position, up_vector);

        const Ref::simd_fvec4 view_transmittance = Absorb(params, optical_depth);

        const float view_zenith_cos_angle = dot(light_dir, up_vector);
        const Ref::simd_fvec2 uv =
            LutTransmittanceParamsToUv(params, local_height + params.planet_radius, view_zenith_cos_angle);
        const Ref::simd_fvec4 light_transmittance = SampleTransmittanceLUT(transmittance_lut, uv);
        total_radiance += Ref::simd_fvec4{params.ground_albedo, Ref::simd_mem_aligned} *
                          saturate(dot(up_vector, light_dir)) * view_transmittance * light_transmittance * light_color;
    }

    //
    // Sun disk (bake directional light into the texture)
    //
    if (light_angle > 0.0f && planet_intersection.get<0>() < 0.0f && light_brightness > 0.0f) {
        const float cos_theta = cosf(light_angle);
        const float BlendVal = 0.000005f;
        Ref::simd_fvec4 sun_disk = transmittance * smoothstep(cos_theta - BlendVal, cos_theta + BlendVal, costh);
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
        total_radiance += transmittance *
                          (powf(clamp(noise(ray_dir * 200.0f), 0.0f, 1.0f), StarsThreshold) * params.stars_brightness);
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
        const float r =
            sqrtf(moon_normal.get<0>() * moon_normal.get<0>() + moon_normal.get<2>() * moon_normal.get<2>());

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

        total_radiance += transmittance * fmaxf(dot(moon_normal, light_dir), 0.0f) * albedo;
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
