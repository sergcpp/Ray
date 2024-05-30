#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#include "shade_sky_interface.h"
#include "common.glsl"
#include "envmap.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout (binding = ATMOSPHERE_PARAMS_BUF_SLOT, std140) uniform AtmosphereParams {
    atmosphere_params_t g_atmosphere_params;
};

#if !BAKE
layout(std430, binding = RAY_INDICES_BUF_SLOT) readonly buffer RayIndices {
    uint g_ray_indices[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = HITS_BUF_SLOT) readonly buffer Hits {
    hit_data_t g_hits[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

layout(binding = ENV_QTREE_TEX_SLOT) uniform texture2D g_env_qtree;
#endif

layout(binding = TRANSMITTANCE_LUT_SLOT) uniform sampler2D g_trasmittance_lut;
layout(binding = MULTISCATTER_LUT_SLOT) uniform sampler2D g_multiscatter_lut;

layout(binding = MOON_TEX_SLOT) uniform sampler2D g_moon_tex;
layout(binding = WEATHER_TEX_SLOT) uniform sampler2D g_weather_tex;
layout(binding = CIRRUS_TEX_SLOT) uniform sampler2D g_cirrus_tex;
layout(binding = NOISE3D_TEX_SLOT) uniform sampler3D g_noise3d_tex;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform image2D g_out_img;

vec2 SphereIntersection(vec3 ray_start, const vec3 ray_dir, const vec3 sphere_center, const float sphere_radius) {
    ray_start -= sphere_center;
    const float a = dot(ray_dir, ray_dir);
    const float b = 2.0 * dot(ray_start, ray_dir);
    const float c = dot(ray_start, ray_start) - (sphere_radius * sphere_radius);
    float d = b * b - 4 * a * c;
    if (d < 0) {
        return vec2(-1);
    } else {
        d = sqrt(d);
        return vec2(-b - d, -b + d) / (2 * a);
    }
}

vec2 PlanetIntersection(const vec3 ray_start, const vec3 ray_dir) {
    const vec3 planet_center = vec3(0, -g_atmosphere_params.planet_radius, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, g_atmosphere_params.planet_radius);
}

vec2 AtmosphereIntersection(const vec3 ray_start, const vec3 ray_dir) {
    const vec3 planet_center = vec3(0, -g_atmosphere_params.planet_radius, 0);
    return SphereIntersection(ray_start, ray_dir, planet_center, g_atmosphere_params.planet_radius + g_atmosphere_params.atmosphere_height);
}

vec4 CloudsIntersection(const vec3 ray_start, const vec3 ray_dir) {
    const vec3 planet_center = vec3(0, -g_atmosphere_params.planet_radius, 0);
    const vec2 beg =
        SphereIntersection(ray_start, ray_dir, planet_center, g_atmosphere_params.planet_radius + g_atmosphere_params.clouds_height_beg);
    const vec2 end =
        SphereIntersection(ray_start, ray_dir, planet_center, g_atmosphere_params.planet_radius + g_atmosphere_params.clouds_height_end);
    return vec4(beg.x, beg.y, end.x, end.y);
}

vec2 MoonIntersection(const vec3 ray_start, const vec3 ray_dir) {
    const vec3 planet_center = g_atmosphere_params.moon_dir.xyz * g_atmosphere_params.moon_distance;
    return SphereIntersection(ray_start, ray_dir, planet_center, g_atmosphere_params.moon_radius);
}

// Phase functions
float PhaseRayleigh(const float costh) { return 3 * (1 + costh * costh) / (16 * PI); }
float PhaseMie(const float costh, float g) {
    g = min(g, 0.9381);
    float k = 1.55 * g - 0.55 * g * g * g;
    float kcosth = k * costh;
    return (1 - k * k) / ((4 * PI) * (1 - kcosth) * (1 - kcosth));
}

const float WrenningePhaseScale = 0.9;
const vec2 WrenningePhaseParameters = vec2(-0.2, 0.8);

float HenyeyGreenstein(const float mu, const float inG) {
    return (1.0 - inG * inG) / (pow(1.0 + inG * inG - 2.0 * inG * mu, 1.5) * 4.0 * PI);
}

float CloudPhaseFunction(const float mu) {
    return mix(HenyeyGreenstein(mu, WrenningePhaseParameters.x),
               HenyeyGreenstein(mu, WrenningePhaseParameters.y), 0.7);
}

vec3 PhaseWrenninge(float mu) {
    vec3 phase = vec3(0.0);
    // Wrenninge multiscatter approximation
    phase.x = CloudPhaseFunction(mu);
    phase.y = CloudPhaseFunction(mu * WrenningePhaseScale);
    phase.z = CloudPhaseFunction(mu * WrenningePhaseScale * WrenningePhaseScale);
    return phase;
}

// dl is the density sampled along the light ray for the given sample position.
// dC is the low lod sample of density at the given sample position.
float GetLightEnergy(const float dl, const float dC, const vec3 phase_probability) {
    // Wrenninge multi scatter approximation
    const vec3 exp_scale = vec3(0.8, 0.1, 0.002);
    const vec3 total_scale = vec3(2.0, 0.8, 0.4);
    const vec3 intensity_curve = exp(-dl * exp_scale);
    return dot(total_scale * phase_probability, intensity_curve);
}

// Atmosphere
float AtmosphereHeight(const vec3 position_ws, out vec3 up_vector) {
    const vec3 planet_center = vec3(0, -g_atmosphere_params.planet_radius, 0);
    up_vector = (position_ws - planet_center);
    const float height = length(up_vector);
    up_vector /= height;
    return height - g_atmosphere_params.planet_radius;
}

vec3 AtmosphereDensity(const float h) {
    const float density_rayleigh = exp(-max(0.0, h / g_atmosphere_params.rayleigh_height));
    const float density_mie = exp(-max(0.0, h / g_atmosphere_params.mie_height));
    const float density_ozone = max(0.0, 1.0 - abs(h - g_atmosphere_params.ozone_height_center) / g_atmosphere_params.ozone_half_width);
    return g_atmosphere_params.atmosphere_density * vec3(density_rayleigh, density_mie, density_ozone);
}

struct atmosphere_medium_t {
    vec3 scattering;
    vec3 absorption;
    vec3 extinction;

    vec3 scattering_mie;
    vec3 absorption_mie;
    vec3 extinction_mie;

    vec3 scattering_ray;
    vec3 absorption_ray;
    vec3 extinction_ray;

    vec3 scattering_ozo;
    vec3 absorption_ozo;
    vec3 extinction_ozo;
};

atmosphere_medium_t SampleAtmosphereMedium(const float h) {
    const vec3 local_density = AtmosphereDensity(h);

    atmosphere_medium_t s;

    s.scattering_mie = local_density.y * g_atmosphere_params.mie_scattering.xyz;
    s.absorption_mie = local_density.y * g_atmosphere_params.mie_absorption.xyz;
    s.extinction_mie = local_density.y * g_atmosphere_params.mie_extinction.xyz;

    s.scattering_ray = local_density.x * g_atmosphere_params.rayleigh_scattering.xyz;
    s.absorption_ray = vec3(0.0);
    s.extinction_ray = s.scattering_ray + s.absorption_ray;

    s.scattering_ozo = vec3(0.0);
    s.absorption_ozo = local_density.z * g_atmosphere_params.ozone_absorbtion.xyz;
    s.extinction_ozo = s.scattering_ozo + s.absorption_ozo;

    s.scattering = s.scattering_mie + s.scattering_ray + s.scattering_ozo;
    s.absorption = s.absorption_mie + s.absorption_ray + s.absorption_ozo;
    s.extinction = s.extinction_mie + s.extinction_ray + s.extinction_ozo;

    return s;
}

void UvToLutTransmittanceParams(const vec3 uv, out float view_height, out float view_zenith_cos_angle) {
    const float top_radius = g_atmosphere_params.planet_radius + g_atmosphere_params.atmosphere_height;

    const float x_mu = uv.x, x_r = uv.y;

    const float H = sqrt(top_radius * top_radius - g_atmosphere_params.planet_radius * g_atmosphere_params.planet_radius);
    const float rho = H * x_r;
    view_height = sqrt(rho * rho + g_atmosphere_params.planet_radius * g_atmosphere_params.planet_radius);

    const float d_min = top_radius - view_height;
    const float d_max = rho + H;
    const float d = d_min + x_mu * (d_max - d_min);
    view_zenith_cos_angle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * view_height * d);
    view_zenith_cos_angle = clamp(view_zenith_cos_angle, -1.0, 1.0);
}

vec2 LutTransmittanceParamsToUv(const float view_height, const float view_zenith_cos_angle) {
    const float top_radius = g_atmosphere_params.planet_radius + g_atmosphere_params.atmosphere_height;

    const float H = sqrt(max(0.0, top_radius * top_radius - g_atmosphere_params.planet_radius * g_atmosphere_params.planet_radius));
    const float rho = sqrt(max(0.0, view_height * view_height - g_atmosphere_params.planet_radius * g_atmosphere_params.planet_radius));

    const float discriminant =
        view_height * view_height * (view_zenith_cos_angle * view_zenith_cos_angle - 1.0) + top_radius * top_radius;
    const float d =
        max(0.0, (-view_height * view_zenith_cos_angle + sqrt(discriminant))); // Distance to atmosphere boundary

    const float d_min = top_radius - view_height;
    const float d_max = rho + H;
    const float x_mu = (d - d_min) / (d_max - d_min);
    const float x_r = rho / H;

    return vec2(x_mu, x_r);
}

vec3 IntegrateScatteringMain(const vec3 ray_start, const vec3 ray_dir, float ray_length, float rand_offset, const int sample_count, inout vec3 inout_transmittance) {
    const vec2 atm_intersection = AtmosphereIntersection(ray_start, ray_dir);
    ray_length = min(ray_length, atm_intersection.y);
    const vec2 planet_intersection = PlanetIntersection(ray_start, ray_dir);
    if (planet_intersection.x > 0) {
        ray_length = min(ray_length, planet_intersection.x);
    }

    const float costh = dot(ray_dir, g_params.light_dir.xyz);
    const float phase_r = PhaseRayleigh(costh), phase_m = PhaseMie(costh, 0.85);

    const float moon_costh = dot(ray_dir, g_atmosphere_params.moon_dir.xyz);
    const float moon_phase_r = PhaseRayleigh(moon_costh), moon_phase_m = PhaseMie(moon_costh, 0.85);

    vec3 radiance = vec3(0.0), multiscat_as_1 = vec3(0.0);

    const float step_size = ray_length / float(sample_count);
    float ray_time = 0.1 * rand_offset * step_size;
    for (int i = 0; i < sample_count; ++i) {
        const vec3 local_position = ray_start + ray_dir * ray_time;
        vec3 up_vector;
        const float local_height = AtmosphereHeight(local_position, up_vector);
        const atmosphere_medium_t medium = SampleAtmosphereMedium(local_height);
        const vec3 optical_depth = medium.extinction * step_size;
        const vec3 local_transmittance = exp(-optical_depth);

        vec3 S = vec3(0.0);

        if (g_params.light_dir.y > -0.025) {
            // main light contribution
            const float view_zenith_cos_angle = dot(g_params.light_dir.xyz, up_vector);
            const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
            const vec3 light_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;

            const vec2 planet_intersection = PlanetIntersection(local_position, g_params.light_dir.xyz);
            const float planet_shadow = planet_intersection.x > 0 ? 0.0 : 1.0;

            vec2 uv2 = saturate(vec2(view_zenith_cos_angle * 0.5 + 0.5, local_height / g_atmosphere_params.atmosphere_height));
            uv2 = vec2(from_unit_to_sub_uvs(uv2.x, SKY_MULTISCATTER_LUT_RES), from_unit_to_sub_uvs(uv2.y, SKY_MULTISCATTER_LUT_RES));
            const vec3 multiscattered_lum = textureLod(g_multiscatter_lut, uv2, 0.0).xyz;

            const vec3 phase_times_scattering = medium.scattering_ray * phase_r + medium.scattering_mie * phase_m;
            S += (planet_shadow * light_transmittance * phase_times_scattering + multiscattered_lum * medium.scattering) * g_params.light_col.xyz;
        } else if (g_atmosphere_params.moon_radius > 0.0) {
            // moon reflection contribution  (totally fake)
            const float view_zenith_cos_angle = dot(g_atmosphere_params.moon_dir.xyz, up_vector);
            const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
            const vec3 light_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;

            vec2 uv2 = saturate(vec2(view_zenith_cos_angle * 0.5 + 0.5, local_height / g_atmosphere_params.atmosphere_height));
            uv2 = vec2(from_unit_to_sub_uvs(uv2.x, SKY_MULTISCATTER_LUT_RES), from_unit_to_sub_uvs(uv2.y, SKY_MULTISCATTER_LUT_RES));
            const vec3 multiscattered_lum = textureLod(g_multiscatter_lut, uv2, 0.0).xyz;

            const vec3 phase_times_scattering = medium.scattering_ray * moon_phase_r + medium.scattering_mie * moon_phase_m;
            S += SKY_MOON_SUN_RELATION * (light_transmittance * phase_times_scattering + multiscattered_lum * medium.scattering) * g_params.light_col.xyz;
        }

        // 1 is the integration of luminance over the 4pi of a sphere, and assuming an isotropic phase function
        // of 1.0/(4*PI)
        const vec3 MS = medium.scattering * 1.0;
        const vec3 MS_int = (MS - MS * local_transmittance) / medium.extinction;
        multiscat_as_1 += inout_transmittance * MS_int;

        const vec3 S_int = (S - S * local_transmittance) / medium.extinction;
        radiance += inout_transmittance * S_int;

        inout_transmittance *= local_transmittance;

        ray_time += step_size;
    }

    //
    // Ground 'floor'
    //
    if (planet_intersection.x > 0.0) {
        const vec3 local_position = ray_start + ray_dir * planet_intersection.x;
        vec3 up_vector;
        const float local_height = AtmosphereHeight(local_position, up_vector);

        const float view_zenith_cos_angle = dot(g_params.light_dir.xyz, up_vector);
        const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
        const vec3 light_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;
        radiance += g_atmosphere_params.ground_albedo.xyz * saturate(dot(up_vector, g_params.light_dir.xyz)) *
                    inout_transmittance * light_transmittance * g_params.light_col.xyz;
    }

    return radiance;
}

// https://www.shadertoy.com/view/NtsBzB
vec3 stars_hash(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));
    p = sin(p);
    return -1.0 + 2.0 * fract(p * 43758.5453123);
}

float stars_noise(const vec3 p) {
    const vec3 i = floor(p), f = fract(p);
    const vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            mix(dot(stars_hash(i + vec3(0.0, 0.0, 0.0)), f - vec3(0.0, 0.0, 0.0)),
                dot(stars_hash(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0)), u.x),
            mix(dot(stars_hash(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0)),
                dot(stars_hash(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0)), u.x),
            u.y),
        mix(
            mix(dot(stars_hash(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0)),
                dot(stars_hash(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0)), u.x),
            mix(dot(stars_hash(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0)),
                dot(stars_hash(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0)), u.x),
            u.y),
        u.z);
}

float remap(const float value, const float original_min) {
    return saturate((value - original_min) / (1.000001 - original_min));
}

// GPU PRO 7 - Real-time Volumetric Cloudscapes
// https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
// https://github.com/sebh/TileableVolumeNoise
float remap(float value, float original_min, float original_max, float new_min, float new_max) {
    return new_min +
           (saturate((value - original_min) / (0.0000001 + original_max - original_min)) * (new_max - new_min));
}

// Taken from https://github.com/armory3d/armory_ci/blob/master/build_untitled/compiled/Shaders/world_pass.frag.glsl
float GetDensityHeightGradientForPoint(float height, float cloud_type) {
    const vec4 stratusGrad = vec4(0.02, 0.05, 0.09, 0.11);
    const vec4 stratocumulusGrad = vec4(0.02, 0.2, 0.48, 0.625);
    const vec4 cumulusGrad = vec4(0.01, 0.0625, 0.78, 1.0);
    float stratus = 1.0 - clamp(cloud_type * 2.0, 0.0, 1.0);
    float stratocumulus = 1.0 - abs(cloud_type - 0.5) * 2.0;
    float cumulus = clamp(cloud_type - 0.5, 0, 1) * 2.0;
    vec4 cloudGradient = stratusGrad * stratus + stratocumulusGrad * stratocumulus + cumulusGrad * cumulus;
    return smoothstep(cloudGradient.x, cloudGradient.y, height) -
           smoothstep(cloudGradient.z, cloudGradient.w, height);
}

float GetCloudsDensity(vec3 local_position, out float out_local_height, out float out_height_fraction, out vec3 out_up_vector) {
    out_local_height = AtmosphereHeight(local_position, out_up_vector);
    out_height_fraction =
        (out_local_height - g_atmosphere_params.clouds_height_beg) / (g_atmosphere_params.clouds_height_end - g_atmosphere_params.clouds_height_beg);

    vec2 weather_uv = vec2(local_position.x + g_atmosphere_params.clouds_offset_x,
                           local_position.z + g_atmosphere_params.clouds_offset_z);
    weather_uv *= 0.00007;

    const vec3 weather_sample = textureLod(g_weather_tex, weather_uv, 0.0).xyz;

    float cloud_coverage = mix(weather_sample.z, weather_sample.y, g_atmosphere_params.clouds_variety);
    cloud_coverage = remap(cloud_coverage, saturate(1.0 - g_atmosphere_params.clouds_density + 0.5 * out_height_fraction));

    const float cloud_type = weather_sample.x;
    cloud_coverage *= GetDensityHeightGradientForPoint(out_height_fraction, cloud_type);

    if (out_height_fraction > 1.0 || cloud_coverage < 0.01) {
        return 0.0;
    }

    local_position /= 1.5 * (g_atmosphere_params.clouds_height_end - g_atmosphere_params.clouds_height_beg);

    const float noise_read = textureLod(g_noise3d_tex, local_position, 0.0).x;
    return 3.0 * mix(max(0.0, 1.0 - cloud_type * 2.0), 1.0, out_height_fraction) *
           remap(cloud_coverage, 0.6 * noise_read);
}

float TraceCloudShadow(const uint rand_hash, vec3 ray_start, const vec3 ray_dir) {
    const vec4 clouds_intersection = CloudsIntersection(ray_start, ray_dir);
    if (clouds_intersection.w > 0) {
        const int SampleCount = 24;

        const float StepSize = 16.0;
        vec3 pos = ray_start + construct_float(rand_hash) * ray_dir * StepSize;

        float ret = 0.0;
        for (int i = 0; i < SampleCount; ++i) {
            float local_height, height_fraction;
            vec3 up_vector;
            const float local_density = GetCloudsDensity(pos, local_height, height_fraction, up_vector);
            ret += local_density;
            pos += ray_dir * StepSize;
        }

        return ret * StepSize;
    }
    return 1.0;
}

vec3 IntegrateScattering(vec3 ray_start, const vec3 ray_dir, float ray_length, uint rand_hash) {
    const vec2 atm_intersection = AtmosphereIntersection(ray_start, ray_dir);
    ray_length = min(ray_length, atm_intersection.y);
    if (atm_intersection.x > 0) {
        // Advance ray to the atmosphere entry point
        ray_start += ray_dir * atm_intersection.x;
        ray_length -= atm_intersection.x;
    }

    const vec2 planet_intersection = PlanetIntersection(ray_start, ray_dir);
    if (planet_intersection.x > 0) {
        ray_length = min(ray_length, planet_intersection.x);
    }

    if (ray_length <= 0.0) {
        return vec3(0.0);
    }

    const vec2 moon_intersection = MoonIntersection(ray_start, ray_dir);
    vec3 moon_dir = g_atmosphere_params.moon_dir.xyz;
    const vec3 moon_point = moon_dir * g_atmosphere_params.moon_distance + 0.5 * g_params.light_dir.xyz * g_atmosphere_params.moon_radius;
    moon_dir = normalize(moon_point);

    const float costh = dot(ray_dir, g_params.light_dir.xyz);
    const vec3 phase_w = PhaseWrenninge(costh);

    const float moon_costh = dot(ray_dir, moon_dir);
    const vec3 moon_phase_w = PhaseWrenninge(moon_costh);

    const float light_brightness = g_params.light_col.x + g_params.light_col.y + g_params.light_col.z;

    vec3 total_radiance = vec3(0.0), total_transmittance = vec3(1.0);

    const vec4 clouds_intersection = CloudsIntersection(ray_start, ray_dir);

    //
    // Atmosphere before clouds
    //
    if (clouds_intersection.y > 0.0 && light_brightness > 0.0) {
        const float pre_atmosphere_ray_length = min(ray_length, clouds_intersection.y);

        const float rand_offset = construct_float(rand_hash);
        rand_hash = hash(rand_hash);

        total_radiance += IntegrateScatteringMain(ray_start, ray_dir, pre_atmosphere_ray_length, rand_offset, SKY_PRE_ATMOSPHERE_SAMPLE_COUNT, total_transmittance);
    }

    //
    // Main clouds
    //

    if (planet_intersection.x < 0 && clouds_intersection.y > 0 && g_atmosphere_params.clouds_density > 0.0 &&
        light_brightness > 0.0 && ray_dir.y > SKY_CLOUDS_HORIZON_CUTOFF) {

        float clouds_ray_length = min(ray_length, clouds_intersection.w);
        vec3 clouds_ray_start = ray_start + ray_dir * clouds_intersection.y;
        clouds_ray_length -= clouds_intersection.y;

        if (clouds_ray_length > 0.0) {
            const float step_size = clouds_ray_length / float(SKY_CLOUDS_SAMPLE_COUNT);

            vec3 local_position = clouds_ray_start + ray_dir * construct_float(rand_hash) * step_size;
            rand_hash = hash(rand_hash);

            vec3 clouds = vec3(0.0);

            // NOTE: We assume transmittance is constant along the clouds range (~500m)
            vec3 light_transmittance, moon_transmittance, multiscattered_lum = vec3(0.0), moon_multiscattered_lum = vec3(0.0);
            {
                vec3 up_vector;
                const float local_height = AtmosphereHeight(local_position, up_vector);
                {
                    const float view_zenith_cos_angle = dot(g_params.light_dir.xyz, up_vector);
                    const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
                    light_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;

                    vec2 uv2 = saturate(vec2(view_zenith_cos_angle * 0.5 + 0.5, local_height / g_atmosphere_params.atmosphere_height));
                    uv2 = vec2(from_unit_to_sub_uvs(uv2.x, SKY_MULTISCATTER_LUT_RES),
                               from_unit_to_sub_uvs(uv2.y, SKY_MULTISCATTER_LUT_RES));
                    multiscattered_lum = textureLod(g_multiscatter_lut, uv2, 0.0).xyz;
                }
                {
                    const float view_zenith_cos_angle = dot(moon_dir, up_vector);
                    const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
                    moon_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;

                    vec2 uv2 = saturate(vec2(view_zenith_cos_angle * 0.5 + 0.5, local_height / g_atmosphere_params.atmosphere_height));
                    uv2 = vec2(from_unit_to_sub_uvs(uv2.x, SKY_MULTISCATTER_LUT_RES),
                               from_unit_to_sub_uvs(uv2.y, SKY_MULTISCATTER_LUT_RES));
                    moon_multiscattered_lum = textureLod(g_multiscatter_lut, uv2, 0.0).xyz;
                }
            }

            vec3 transmittance_before = total_transmittance;

            for (int i = 0; i < SKY_CLOUDS_SAMPLE_COUNT; ++i) {
                float local_height, height_fraction;
                vec3 up_vector;
                const float local_density = GetCloudsDensity(local_position, local_height, height_fraction, up_vector);
                if (local_density > 0.0) {
                    const float local_transmittance = exp(-local_density * step_size);
                    const float ambient_visibility = (0.75 + 1.5 * max(0.0, height_fraction - 0.1));

                    if (g_params.light_dir.y > -0.025) {
                        // main light contribution
                        const vec2 planet_intersection = PlanetIntersection(local_position, g_params.light_dir.xyz);
                        const float planet_shadow = planet_intersection.x > 0 ? 0.0 : 1.0;
                        const float cloud_shadow = TraceCloudShadow(rand_hash, local_position, g_params.light_dir.xyz);

                        clouds += total_transmittance *
                                (planet_shadow * GetLightEnergy(cloud_shadow, local_density, phase_w) +
                                ambient_visibility * multiscattered_lum) *
                                (1.0 - local_transmittance) * light_transmittance;
                    } else if (g_atmosphere_params.moon_radius > 0.0) {
                        // moon reflection contribution (totally fake)
                        const float cloud_shadow = TraceCloudShadow(rand_hash, local_position, moon_dir);

                        clouds += SKY_MOON_SUN_RELATION * total_transmittance *
                                (GetLightEnergy(cloud_shadow, local_density, moon_phase_w) +
                                ambient_visibility * moon_multiscattered_lum) *
                                (1.0 - local_transmittance) * moon_transmittance;
                    }

                    total_transmittance *= local_transmittance;
                    if ((total_transmittance.x + total_transmittance.y + total_transmittance.z) < 0.01) {
                        break;
                    }
                }
                local_position += ray_dir * step_size;
            }

            // NOTE: totally arbitrary cloud blending
            float cloud_blend = linstep(SKY_CLOUDS_HORIZON_CUTOFF + 0.25, SKY_CLOUDS_HORIZON_CUTOFF, ray_dir.y);
            cloud_blend = 1.0 - pow(cloud_blend, 5.0);

            total_radiance += cloud_blend * clouds * g_params.light_col.xyz;
            total_transmittance = mix(transmittance_before, total_transmittance, cloud_blend);
        }
    }

    //
    // Cirrus clouds
    //
    if (planet_intersection.x < 0 && clouds_intersection.y > 0 && g_atmosphere_params.cirrus_clouds_amount > 0.0 && light_brightness > 0.0) {
        vec2 cirrus_coords =
            3e-4 * vec2(g_atmosphere_params.clouds_offset_z, g_atmosphere_params.clouds_offset_x) + 0.8 * ray_dir.zx / (abs(ray_dir.y) + 0.02);
        cirrus_coords.y += 1.75;

        float noise_read = 1.0 - textureLod(g_noise3d_tex, fract(vec3(0.0, cirrus_coords.xy * 0.03)), 0.0).x;
        noise_read =
            saturate(noise_read - 1.0 + g_atmosphere_params.cirrus_clouds_amount * 0.6) / (g_atmosphere_params.cirrus_clouds_amount + 1e-9);

        float dC = 1.2 * smoothstep(0.0, 1.0, noise_read) * _srgb_to_linear(textureLod(g_cirrus_tex, fract(cirrus_coords * 0.5), 0.0).x);

        //
        cirrus_coords.x += 0.25;
        noise_read = 1.0 - textureLod(g_noise3d_tex, fract(vec3(0.7, cirrus_coords.xy * 0.02)), 0.0).x;
        noise_read =
            saturate(noise_read - 1.0 + g_atmosphere_params.cirrus_clouds_amount * 0.7) / (g_atmosphere_params.cirrus_clouds_amount + 1e-9);

        dC += 0.6 * smoothstep(0.0, 1.0, noise_read) * _srgb_to_linear(textureLod(g_cirrus_tex, fract(cirrus_coords * 0.25), 0.0).y);

        vec3 local_position = ray_start + ray_dir * g_atmosphere_params.cirrus_clouds_height;

        vec3 light_transmittance, moon_transmittance;
        {
            vec3 up_vector;
            const float local_height = AtmosphereHeight(local_position, up_vector);
            {
                const float view_zenith_cos_angle = dot(g_params.light_dir.xyz, up_vector);
                const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
                light_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;
            }
            {
                const float view_zenith_cos_angle = dot(moon_dir, up_vector);
                const vec2 uv = LutTransmittanceParamsToUv(local_height + g_atmosphere_params.planet_radius, view_zenith_cos_angle);
                moon_transmittance = textureLod(g_trasmittance_lut, uv, 0.0).xyz;
            }
        }

        if (g_params.light_dir.y > -0.025) {
            total_radiance += total_transmittance * GetLightEnergy(0.002, dC, phase_w) * light_transmittance * dC * g_params.light_col.xyz;
        } else if (g_atmosphere_params.moon_radius > 0.0) {
            total_radiance += SKY_MOON_SUN_RELATION * total_transmittance * GetLightEnergy(0.002, dC, moon_phase_w) *
                            moon_transmittance * dC * g_params.light_col.xyz;
        }
        total_transmittance *= exp(-dC * 0.002 * 1000.0);
    }

    if (total_transmittance.x + total_transmittance.y + total_transmittance.z < 0.001) {
        return total_radiance;
    }

    //
    // Main atmosphere
    //
    if (planet_intersection.x < 0 && light_brightness > 0.0) {
        float main_ray_length = ray_length;
        vec3 main_ray_start = ray_start + ray_dir * clouds_intersection.w;
        main_ray_length -= clouds_intersection.y;

        const float rand_offset = construct_float(rand_hash);
        rand_hash = hash(rand_hash);

        total_radiance += IntegrateScatteringMain(main_ray_start, ray_dir, main_ray_length, rand_offset, SKY_MAIN_ATMOSPHERE_SAMPLE_COUNT, total_transmittance);
    }

    //
    // Sun disk (bake directional light into the texture)
    //
    if (g_params.light_dir.w > 0.0 && planet_intersection.x < 0.0 && light_brightness > 0.0) {
        const float cos_theta = cos(g_params.light_dir.w);
        vec3 sun_disk = total_transmittance * smoothstep(cos_theta - SKY_SUN_BLEND_VAL, cos_theta + SKY_SUN_BLEND_VAL, costh);
        // 'de-multiply' by disk area (to get original brightness)
        const float radius = tan(g_params.light_dir.w);
        sun_disk /= (PI * radius * radius);

        total_radiance += sun_disk * g_params.light_col.xyz;
    }

    //
    // Stars
    //
    if (g_atmosphere_params.stars_brightness > 0.0 && planet_intersection.x < 0 && moon_intersection.x < 0) {
        total_radiance +=
            total_transmittance *
            (pow(clamp(stars_noise(ray_dir * 400.0), 0.0, 1.0), SKY_STARS_THRESHOLD) * g_atmosphere_params.stars_brightness);
    }

    //
    // Moon
    //
    if (planet_intersection.x < 0 && moon_intersection.x > 0 && g_atmosphere_params.moon_radius > 0.0 && light_brightness > 0.0) {
        const vec3 moon_center = g_atmosphere_params.moon_dir.xyz * g_atmosphere_params.moon_distance;
        const vec3 moon_normal = normalize(ray_start + moon_intersection.x * ray_dir - moon_center);

        const float theta = acos(clamp(moon_normal.y, -1.0, 1.0)) / PI;

        float phi = atan(moon_normal.z, moon_normal.x);
        if (phi < 0) {
            phi += 2 * PI;
        }
        if (phi > 2 * PI) {
            phi -= 2 * PI;
        }

        const float u = fract(0.5 * phi / PI);

        const vec2 uvs = vec2(u, theta);
        const vec3 albedo = textureLod(g_moon_tex, uvs, 0.0).rgb;

        total_radiance += total_transmittance * saturate(dot(moon_normal, g_params.light_dir.xyz)) * albedo;
    }

    return total_radiance;
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if BAKE
    const int x = int(gl_GlobalInvocationID.x),
              y = int(gl_GlobalInvocationID.y);

    const uint px_hash = hash((x << 16) | y);
    const uint rand_hash = px_hash;

    const float phi = 2.0 * PI * (x + 0.5) / float(g_params.res[0]);
    const float theta = PI * float(y) / float(g_params.res[1]);
    const vec3 I = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));

    const vec3 sky_color = IntegrateScattering(vec3(0.0, g_atmosphere_params.viewpoint_height, 0.0), I, MAX_DIST, rand_hash);
#else // BAKE
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[5]) {
        return;
    }

    const uint ray_index = g_ray_indices[index];
    const ray_data_t r = g_rays[ray_index];
    const hit_data_t inter = g_hits[ray_index];

    const int x = int((r.xy >> 16) & 0xffff), y = int(r.xy & 0xffff);

    const uint px_hash = hash(r.xy);
    const uint rand_hash = hash_combine(px_hash, g_params.rand_seed);

    const float env_map_rotation = is_indirect(r.depth) ? g_params.env_rotation : g_params.back_rotation;
    const vec3 I = vec3(r.d[0] * cos(env_map_rotation) - r.d[2] * sin(env_map_rotation), r.d[1],
                        r.d[0] * sin(env_map_rotation) + r.d[2] * cos(env_map_rotation));

    vec3 sky_color = IntegrateScattering(vec3(0.0, g_atmosphere_params.viewpoint_height, 0.0), I, MAX_DIST, rand_hash);

    const vec3 env_col = is_indirect(r.depth) ? g_params.env_col.xyz : g_params.back_col.xyz;
    sky_color *= env_col;

#if USE_NEE

#if USE_HIERARCHICAL_NEE
    const float pdf_factor = (get_total_depth(r.depth) < g_params.max_total_depth) ? (1.0 / inter.u) : -1.0;
#else
    const float pdf_factor = (get_total_depth(r.depth) < g_params.max_total_depth) ? float(g_params.li_count) : -1.0;
#endif

    if (g_params.env_light_index != -1 && pdf_factor >= 0.0 && is_indirect(r.depth)) {
        if (g_params.env_qtree_levels > 0) {
            const float light_pdf = Evaluate_EnvQTree(env_map_rotation, g_env_qtree, g_params.env_qtree_levels, I) / pdf_factor;
            const float bsdf_pdf = r.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            sky_color *= mis_weight;
        } else {
            const float light_pdf = 0.5 / (PI * float(g_params.li_count));
            const float bsdf_pdf = r.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            sky_color *= mis_weight;
        }
    }
#endif // USE_NEE

    sky_color *= vec3(r.c[0], r.c[1], r.c[2]);
    const float sum = (sky_color.x + sky_color.y + sky_color.z);
    if (sum > g_params.limit) {
        sky_color *= (g_params.limit / sum);
    }
#endif // BAKE
    const vec3 col = imageLoad(g_out_img, ivec2(x, y)).xyz;
    imageStore(g_out_img, ivec2(x, y), vec4(col + sky_color, 1.0));
}