#version 450
#extension GL_GOOGLE_include_directive : require

#include "shade_interface.h"
#include "common.glsl"
#include "envmap.glsl"
#include "texture.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HITS_BUF_SLOT) readonly buffer Hits {
    hit_data_t g_hits[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

layout(std430, binding = LIGHTS_BUF_SLOT) readonly buffer Lights {
    light_t g_lights[];
};

layout(std430, binding = LI_INDICES_BUF_SLOT) readonly buffer LiIndices {
    uint g_li_indices[];
};

layout(std430, binding = TRIS_BUF_SLOT) readonly buffer Tris {
    tri_accel_t g_tris[];
};

layout(std430, binding = TRI_MATERIALS_BUF_SLOT) readonly buffer TriMaterials {
    uint g_tri_materials[];
};

layout(std430, binding = MATERIALS_BUF_SLOT) readonly buffer Materials {
    material_t g_materials[];
};

layout(std430, binding = TRANSFORMS_BUF_SLOT) readonly buffer Transforms {
    transform_t g_transforms[];
};

layout(std430, binding = MESH_INSTANCES_BUF_SLOT) readonly buffer MeshInstances {
    mesh_instance_t g_mesh_instances[];
};

layout(std430, binding = VERTICES_BUF_SLOT) readonly buffer Vertices {
    vertex_t g_vertices[];
};

layout(std430, binding = VTX_INDICES_BUF_SLOT) readonly buffer VtxIndices {
    uint g_vtx_indices[];
};

layout(std430, binding = RANDOM_SEQ_BUF_SLOT) readonly buffer Random {
    float g_random_seq[];
};

layout(binding = ENV_QTREE_TEX_SLOT) uniform sampler2D g_env_qtree;

#if PRIMARY
layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
#else
layout(binding = OUT_IMG_SLOT, rgba32f) uniform image2D g_out_img;
#endif

layout(std430, binding = OUT_RAYS_BUF_SLOT) writeonly buffer OutRays {
    ray_data_t g_out_rays[];
};

layout(std430, binding = OUT_SH_RAYS_BUF_SLOT) writeonly buffer OutShRays {
    shadow_ray_t g_out_sh_rays[];
};

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer InoutCounters {
    uint g_inout_counters[];
};

float power_heuristic(float a, float b) {
    float t = a * a;
    return t / (b * b + t);
}

float pow5(float v) { return (v * v) * (v * v) * v; }

float schlick_weight(float u) {
    const float m = clamp(1.0 - u, 0.0, 1.0);
    return pow5(m);
}

float fresnel_dielectric_cos(float cosi, float eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    float c = abs(cosi);
    float g = eta * eta - 1 + c * c;
    float result;

    if (g > 0) {
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        result = 0.5 * A * A * (1 + B * B);
    } else {
        result = 1.0; // TIR (no refracted component)
    }

    return result;
}

float safe_sqrtf(float f) { return sqrt(max(f, 0.0)); }

// Taken from Cycles
vec3 ensure_valid_reflection(vec3 Ng, vec3 I, vec3 N) {
    const vec3 R = 2 * dot(N, I) * N - I;

    // Reflection rays may always be at least as shallow as the incoming ray.
    const float threshold = min(0.9 * dot(Ng, I), 0.01);
    if (dot(Ng, R) >= threshold) {
        return N;
    }

    // Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
    // The X axis is found by normalizing the component of N that's orthogonal to Ng.
    // The Y axis isn't actually needed.
    const float NdotNg = dot(N, Ng);
    const vec3 X = normalize(N - NdotNg * Ng);

    // Calculate N.z and N.x in the local coordinate system.
    //
    // The goal of this computation is to find a N' that is rotated towards Ng just enough
    // to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
    //
    // According to the standard reflection equation,
    // this means that we want dot(2*dot(N', I)*N' - I, Ng) = t.
    //
    // Since the Z axis of our local coordinate system is Ng, dot(x, Ng) is just x.z, so we get
    // 2*dot(N', I)*N'.z - I.z = t.
    //
    // The rotation is simple to express in the coordinate system we formed -
    // since N lies in the X-Z-plane, we know that N' will also lie in the X-Z-plane,
    // so N'.y = 0 and therefore dot(N', I) = N'.x*I.x + N'.z*I.z .
    //
    // Furthermore, we want N' to be normalized, so N'.x = sqrt(1 - N'.z^2).
    //
    // With these simplifications,
    // we get the final equation 2*(sqrt(1 - N'.z^2)*I.x + N'.z*I.z)*N'.z - I.z = t.
    //
    // The only unknown here is N'.z, so we can solve for that.
    //
    // The equation has four solutions in general:
    //
    // N'.z = +-sqrt(0.5*(+-sqrt(I.x^2*(I.x^2 + I.z^2 - t^2)) + t*I.z + I.x^2 + I.z^2)/(I.x^2 + I.z^2))
    // We can simplify this expression a bit by grouping terms:
    //
    // a = I.x^2 + I.z^2
    // b = sqrt(I.x^2 * (a - t^2))
    // c = I.z*t + a
    // N'.z = +-sqrt(0.5*(+-b + c)/a)
    //
    // Two solutions can immediately be discarded because they're negative so N' would lie in the
    // lower hemisphere.

    const float Ix = dot(I, X), Iz = dot(I, Ng);
    const float Ix2 = (Ix * Ix), Iz2 = (Iz * Iz);
    const float a = Ix2 + Iz2;

    const float b = safe_sqrtf(Ix2 * (a - (threshold * threshold)));
    const float c = Iz * threshold + a;

    // Evaluate both solutions.
    // In many cases one can be immediately discarded (if N'.z would be imaginary or larger than
    // one), so check for that first. If no option is viable (might happen in extreme cases like N
    // being in the wrong hemisphere), give up and return Ng.
    const float fac = 0.5 / a;
    const float N1_z2 = fac * (b + c), N2_z2 = fac * (-b + c);
    bool valid1 = (N1_z2 > 1e-5) && (N1_z2 <= (1.0 + 1e-5));
    bool valid2 = (N2_z2 > 1e-5) && (N2_z2 <= (1.0 + 1e-5));

    vec2 N_new;
    if (valid1 && valid2) {
        // If both are possible, do the expensive reflection-based check.
        const vec2 N1 = vec2(safe_sqrtf(1.0 - N1_z2), safe_sqrtf(N1_z2));
        const vec2 N2 = vec2(safe_sqrtf(1.0 - N2_z2), safe_sqrtf(N2_z2));

        const float R1 = 2 * (N1[0] * Ix + N1[1] * Iz) * N1[1] - Iz;
        const float R2 = 2 * (N2[0] * Ix + N2[1] * Iz) * N2[1] - Iz;

        valid1 = (R1 >= 1e-5);
        valid2 = (R2 >= 1e-5);
        if (valid1 && valid2) {
            // If both solutions are valid, return the one with the shallower reflection since it will be
            // closer to the input (if the original reflection wasn't shallow, we would not be in this
            // part of the function).
            N_new = (R1 < R2) ? N1 : N2;
        } else {
            // If only one reflection is valid (= positive), pick that one.
            N_new = (R1 > R2) ? N1 : N2;
        }
    } else if (valid1 || valid2) {
        // Only one solution passes the N'.z criterium, so pick that one.
        const float Nz2 = valid1 ? N1_z2 : N2_z2;
        N_new = vec2(safe_sqrtf(1.0 - Nz2), safe_sqrtf(Nz2));
    } else {
        return Ng;
    }

    return N_new[0] * X + N_new[1] * Ng;
}

void get_lobe_weights(const float base_color_lum, const float spec_color_lum, const float specular,
                      const float metallic, const float transmission, const float clearcoat, out float out_diffuse_weight,
                      out float out_specular_weight, out float out_clearcoat_weight, out float out_refraction_weight) {
    // taken from Cycles
    out_diffuse_weight = base_color_lum * (1.0 - metallic) * (1.0 - transmission);
    const float final_transmission = transmission * (1.0 - metallic);
    out_specular_weight =
        (specular != 0.0 || metallic != 0.0) ? spec_color_lum * (1.0 - final_transmission) : 0.0;
    out_clearcoat_weight = 0.25 * clearcoat * (1.0 - metallic);
    out_refraction_weight = final_transmission * base_color_lum;

    const float total_weight =
        out_diffuse_weight + out_specular_weight + out_clearcoat_weight + out_refraction_weight;
    if (total_weight != 0.0) {
        out_diffuse_weight /= total_weight;
        out_specular_weight /= total_weight;
        out_clearcoat_weight /= total_weight;
        out_refraction_weight /= total_weight;
    }
}

vec3 rotate_around_axis(vec3 p, vec3 axis, float angle) {
    const float costheta = cos(angle);
    const float sintheta = sin(angle);
    vec3 r;

    r[0] = ((costheta + (1.0 - costheta) * axis[0] * axis[0]) * p[0]) +
           (((1.0 - costheta) * axis[0] * axis[1] - axis[2] * sintheta) * p[1]) +
           (((1.0 - costheta) * axis[0] * axis[2] + axis[1] * sintheta) * p[2]);

    r[1] = (((1.0 - costheta) * axis[0] * axis[1] + axis[2] * sintheta) * p[0]) +
           ((costheta + (1.0 - costheta) * axis[1] * axis[1]) * p[1]) +
           (((1.0 - costheta) * axis[1] * axis[2] - axis[0] * sintheta) * p[2]);

    r[2] = (((1.0 - costheta) * axis[0] * axis[2] - axis[1] * sintheta) * p[0]) +
           (((1.0 - costheta) * axis[1] * axis[2] + axis[0] * sintheta) * p[1]) +
           ((costheta + (1.0 - costheta) * axis[2] * axis[2]) * p[2]);

    return r;
}

vec3 offset_ray(vec3 p, vec3 n) {
    const float Origin = 1.0 / 32.0;
    const float FloatScale = 1.0 / 65536.0;
    const float IntScale = 128.0; // 256.0;

    const ivec3 of_i = ivec3(IntScale * n);

    const vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p[0]) + ((p[0] < 0.0) ? -of_i[0] : of_i[0])),
                          intBitsToFloat(floatBitsToInt(p[1]) + ((p[1] < 0.0) ? -of_i[1] : of_i[1])),
                          intBitsToFloat(floatBitsToInt(p[2]) + ((p[2] < 0.0) ? -of_i[2] : of_i[2])));

    return vec3(abs(p[0]) < Origin ? (p[0] + FloatScale * n[0]) : p_i[0],
                abs(p[1]) < Origin ? (p[1] + FloatScale * n[1]) : p_i[1],
                abs(p[2]) < Origin ? (p[2] + FloatScale * n[2]) : p_i[2]);
}

// http://jcgt.org/published/0007/04/01/paper.pdf by Eric Heitz
// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
vec3 SampleGGX_VNDF(const vec3 Ve, float alpha_x, float alpha_y, float U1, float U2) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    const vec3 Vh = normalize(vec3(alpha_x * Ve[0], alpha_y * Ve[1], Ve[2]));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    const float lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
    const vec3 T1 =
        lensq > 0.0 ? vec3(-Vh[1], Vh[0], 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    const vec3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    const float r = sqrt(U1);
    const float phi = 2.0 * PI * U2;
    const float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    const float s = 0.5 * (1.0 + Vh[2]);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    const vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    const vec3 Ne = normalize(vec3(alpha_x * Nh[0], alpha_y * Nh[1], max(0.0, Nh[2])));
    return Ne;
}

// Smith shadowing function
float G1(vec3 Ve, float alpha_x, float alpha_y) {
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    const float delta =
        (-1.0 + sqrt(1.0 + (alpha_x * Ve[0] * Ve[0] + alpha_y * Ve[1] * Ve[1]) / (Ve[2] * Ve[2]))) / 2.0;
    return 1.0 / (1.0 + delta);
}

float D_GTR1(float NDotH, float a) {
    if (a >= 1.0) {
        return 1.0 / PI;
    }
    const float a2 = a * a;
    const float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

float D_GTR2(const float N_dot_H, const float a) {
    const float a2 = a * a;
    const float t = 1.0 + (a2 - 1.0) * N_dot_H * N_dot_H;
    return a2 / (PI * t * t);
}

float D_GGX(const vec3 H, const float alpha_x, const float alpha_y) {
    if (H[2] == 0.0) {
        return 0.0;
    }
    const float sx = -H[0] / (H[2] * alpha_x);
    const float sy = -H[1] / (H[2] * alpha_y);
    const float s1 = 1.0 + sx * sx + sy * sy;
    const float cos_theta_h4 = H[2] * H[2] * H[2] * H[2];
    return 1.0 / ((s1 * s1) * PI * alpha_x * alpha_y * cos_theta_h4);
}

vec3 world_from_tangent(vec3 T, vec3 B, vec3 N, vec3 V) {
    return V[0] * T + V[1] * B + V[2] * N;
}

vec3 tangent_from_world(vec3 T, vec3 B, vec3 N, vec3 V) {
    return vec3(dot(V, T), dot(V, B), dot(V, N));
}

float BRDF_PrincipledDiffuse(vec3 V, vec3 N, vec3 L, vec3 H, float roughness) {
    const float N_dot_L = dot(N, L);
    const float N_dot_V = dot(N, V);
    if (N_dot_L <= 0.0) {
        return 0.0;
    }

    const float FL = schlick_weight(N_dot_L);
    const float FV = schlick_weight(N_dot_V);

    const float L_dot_H = dot(L, H);
    const float Fd90 = 0.5 + 2.0 * L_dot_H * L_dot_H * roughness;
    const float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    return Fd;
}

vec4 Evaluate_OrenDiffuse_BSDF(vec3 V, vec3 N, vec3 L, float roughness, vec3 base_color) {
    const float sigma = roughness;
    const float div = 1.0 / (PI + ((3.0 * PI - 4.0) / 6.0) * sigma);

    const float a = 1.0 * div;
    const float b = sigma * div;

    ////

    const float nl = max(dot(N, L), 0.0);
    const float nv = max(dot(N, V), 0.0);
    float t = dot(L, V) - nl * nv;

    if (t > 0.0) {
        t /= max(nl, nv) + FLT_MIN;
    }
    const float is = nl * (a + b * t);

    vec3 diff_col = is * base_color;

    return vec4(diff_col, 0.5 / PI);
}

vec4 Sample_OrenDiffuse_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, float roughness,
                             vec3 base_color, float rand_u, float rand_v, out vec3 out_V) {
    const float phi = 2 * PI * rand_v;

    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

    const float dir = sqrt(1.0 - rand_u * rand_u);
    vec3 V = vec3(dir * cos_phi, dir * sin_phi, rand_u); // in tangent-space

    out_V = world_from_tangent(T, B, N, V);
    return Evaluate_OrenDiffuse_BSDF(-I, N, out_V, roughness, base_color);
}

vec4 Evaluate_PrincipledDiffuse_BSDF(vec3 V, vec3 N, vec3 L, float roughness, vec3 base_color, vec3 sheen_color, bool uniform_sampling) {
    float weight, pdf;
    if (uniform_sampling) {
        weight = 2 * dot(N, L);
        pdf = 0.5 / PI;
    } else {
        weight = 1.0;
        pdf = dot(N, L) / PI;
    }

    vec3 H = normalize(L + V);
    if (dot(V, H) < 0.0) {
        H = -H;
    }

    vec3 diff_col = base_color * (weight * BRDF_PrincipledDiffuse(V, N, L, H, roughness));

    const float FH = PI * schlick_weight(dot(L, H));
    diff_col += FH * sheen_color;

    return vec4(diff_col, pdf);
}

vec4 Sample_PrincipledDiffuse_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, float roughness, vec3 base_color,
                                   vec3 sheen_color, bool uniform_sampling, float rand_u, float rand_v,
                                   out vec3 out_V) {
    const float phi = 2 * PI * rand_v;

    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

    vec3 V;
    if (uniform_sampling) {
        const float dir = sqrt(1.0 - rand_u * rand_u);
        V = vec3(dir * cos_phi, dir * sin_phi, rand_u); // in tangent-space
    } else {
        const float dir = sqrt(rand_u);
        const float k = sqrt(1.0 - rand_u);
        V = vec3(dir * cos_phi, dir * sin_phi, k); // in tangent-space
    }

    out_V = world_from_tangent(T, B, N, V);
    return Evaluate_PrincipledDiffuse_BSDF(-I, N, out_V, roughness, base_color, sheen_color, uniform_sampling);
}

vec4 Evaluate_GGXSpecular_BSDF(vec3 view_dir_ts, vec3 sampled_normal_ts,
                               vec3 reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior,
                               float spec_F0, vec3 spec_col) {
#if USE_VNDF_GGX_SAMPLING == 1
    const float D = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
#else
    const float D = D_GTR2(sampled_normal_ts[2], alpha_x);
#endif

    const float G = G1(view_dir_ts, alpha_x, alpha_y) * G1(reflected_dir_ts, alpha_x, alpha_y);

    const float FH =
        (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0 - spec_F0);
    vec3 F = mix(spec_col, vec3(1.0), FH);

    const float denom = 4.0 * abs(view_dir_ts[2] * reflected_dir_ts[2]);
    F *= (denom != 0.0) ? (D * G / denom) : 0.0;

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1(view_dir_ts, alpha_x, alpha_y) * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0, 1.0) /
                abs(view_dir_ts[2]);
    const float div = 4.0 * dot(view_dir_ts, sampled_normal_ts);
    if (div != 0.0) {
        pdf /= div;
    }
#else
    const float pdf = D * sampled_normal_ts[2] / (4.0 * dot(view_dir_ts, sampled_normal_ts));
#endif

    F *= max(reflected_dir_ts[2], 0.0);

    return vec4(F, pdf);
}

vec4 Sample_GGXSpecular_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, const float roughness,
                             const float anisotropic, const float spec_ior, const float spec_F0,
                             const vec3 spec_col, const float rand_u, const float rand_v, out vec3 out_V) {
    const float roughness2 = roughness * roughness;
    const float aspect = sqrt(1.0 - 0.9 * anisotropic);

    const float alpha_x = roughness2 / aspect;
    const float alpha_y = roughness2 * aspect;

    [[dont_flatten]] if (alpha_x * alpha_y < 1e-7) {
        const vec3 V = reflect(I, N);
        const float FH = (fresnel_dielectric_cos(dot(V, N), spec_ior) - spec_F0) / (1.0 - spec_F0);
        vec3 F = mix(spec_col, vec3(1.0), FH);
        out_V = V;
        return vec4(F[0] * 1e6f, F[1] * 1e6f, F[2] * 1e6f, 1e6f);
    }

    const vec3 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
#if USE_VNDF_GGX_SAMPLING == 1
    const vec3 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha_x, alpha_y, rand_u, rand_v);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif
    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const vec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);
    return Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha_x, alpha_y, spec_ior,
                                     spec_F0, spec_col);
}

vec4 Evaluate_PrincipledClearcoat_BSDF(vec3 view_dir_ts, vec3 sampled_normal_ts, vec3 reflected_dir_ts,
                                       float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0) {
    const float D = D_GTR1(sampled_normal_ts[2], clearcoat_roughness2);
    // Always assume roughness of 0.25 for clearcoat
    const float clearcoat_alpha = (0.25 * 0.25);

    const float G =
        G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) * G1(reflected_dir_ts, clearcoat_alpha, clearcoat_alpha);

    const float FH = (fresnel_dielectric_cos(dot(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
                     (1.0 - clearcoat_F0);
    float F = mix(0.04, 1.0, FH);

    const float denom = 4.0 * abs(view_dir_ts[2]) * abs(reflected_dir_ts[2]);
    F *= (denom != 0.0) ? D * G / denom : 0.0;

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) *
                clamp(dot(view_dir_ts, sampled_normal_ts), 0.0, 1.0) / abs(view_dir_ts[2]);
    const float div = 4.0 * dot(view_dir_ts, sampled_normal_ts);
    if (div != 0.0) {
        pdf /= div;
    }
#else
    float pdf = D * sampled_normal_ts[2] / (4.0 * dot(view_dir_ts, sampled_normal_ts));
#endif

    F *= clamp(reflected_dir_ts[2], 0.0, 1.0);
    return vec4(F, F, F, pdf);
}

vec4 Sample_PrincipledClearcoat_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, float clearcoat_roughness2,
                                     float clearcoat_ior, float clearcoat_F0, float rand_u, float rand_v,
                                     out vec3 out_V) {
    [[dont_flatten]] if (clearcoat_roughness2 * clearcoat_roughness2 < 1e-7) {
        const vec3 V = reflect(I, N);

        const float FH = (fresnel_dielectric_cos(dot(V, N), clearcoat_ior) - clearcoat_F0) / (1.0 - clearcoat_F0);
        const float F = mix(0.04, 1.0, FH);

        out_V = V;
        return vec4(F * 1e6f, F * 1e6f, F * 1e6f, 1e6f);
    }

    const vec3 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    // NOTE: GTR1 distribution is not used for sampling because Cycles does it this way (???!)
#if USE_VNDF_GGX_SAMPLING == 1
    const vec3 sampled_normal_ts =
        SampleGGX_VNDF(view_dir_ts, clearcoat_roughness2, clearcoat_roughness2, rand_u, rand_v);
#else
    const vec3 sampled_normal_ts = sample_GGX_NDF(clearcoat_roughness2, rand_u, rand_v);
#endif
    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const vec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);

    return Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, clearcoat_roughness2,
                                             clearcoat_ior, clearcoat_F0);
}

vec4 Evaluate_GGXRefraction_BSDF(vec3 view_dir_ts, vec3 sampled_normal_ts,
                                 vec3 refr_dir_ts, float roughness2, float eta,
                                 vec3 refr_col) {
    if (refr_dir_ts[2] >= 0.0 || view_dir_ts[2] <= 0.0) {
        return vec4(0.0);
    }

#if USE_VNDF_GGX_SAMPLING == 1
    const float D = D_GGX(sampled_normal_ts, roughness2, roughness2);
#else
    const float D = D_GTR2(sampled_normal_ts[2], roughness2);
#endif

    const float G1o = G1(refr_dir_ts, roughness2, roughness2);
    const float G1i = G1(view_dir_ts, roughness2, roughness2);

    const float denom = dot(refr_dir_ts, sampled_normal_ts) + dot(view_dir_ts, sampled_normal_ts) * eta;
    const float jacobian = clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0, 1.0) / (denom * denom);

    float F = D * G1i * G1o * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0, 1.0) * jacobian /
              (/*-refr_dir_ts[2] */ view_dir_ts[2]);

#if USE_VNDF_GGX_SAMPLING == 1
    float pdf = D * G1o * clamp(dot(view_dir_ts, sampled_normal_ts), 0.0, 1.0) * jacobian / view_dir_ts[2];
#else
    // const float pdf = D * std::max(sampled_normal_ts[2], 0.0) * jacobian;
    const float pdf = D * sampled_normal_ts[2] * clamp(-dot(refr_dir_ts, sampled_normal_ts), 0.0, 1.0) / denom;
#endif

    return vec4(F * refr_col, pdf);
}

vec4 Sample_GGXRefraction_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, float roughness, float eta,
                               vec3 refr_col, float rand_u, float rand_v, out vec4 out_V) {
    const float roughness2 = (roughness * roughness);
    [[dont_flatten]] if (roughness2 * roughness2 < 1e-7) {
        const float cosi = -dot(I, N);
        const float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
        if (cost2 < 0) {
            return vec4(0.0);
        }
        const float m = eta * cosi - sqrt(cost2);
        const vec3 V = normalize(eta * I + m * N);

        out_V = vec4(V[0], V[1], V[2], m);
        return vec4(refr_col[0] * 1e6f, refr_col[1] * 1e6f, refr_col[2] * 1e6f, 1e6f);
    }

    const vec3 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
#if USE_VNDF_GGX_SAMPLING == 1
    const vec3 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, roughness2, roughness2, rand_u, rand_v);
#else
    const vec3 sampled_normal_ts = sample_GGX_NDF(roughness2, rand_u, rand_v);
#endif

    const float cosi = dot(view_dir_ts, sampled_normal_ts);
    const float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (cost2 < 0) {
        return vec4(0.0);
    }
    const float m = eta * cosi - sqrt(cost2);
    const vec3 refr_dir_ts = normalize(-eta * view_dir_ts + m * sampled_normal_ts);

    const vec4 F =
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, roughness2, eta, refr_col);

    const vec3 V = world_from_tangent(T, B, N, refr_dir_ts);
    out_V = vec4(V[0], V[1], V[2], m);
    return F;
}

struct light_sample_t {
    vec3 col, L;
    float area, dist, pdf, cast_shadow;
};

void create_tbn(const vec3 N, out vec3 out_T, out vec3 out_B) {
    vec3 U;
    [[flatten]] if (abs(N[1]) < 0.999) {
        U = vec3(0.0, 1.0, 0.0);
    } else {
        U = vec3(1.0, 0.0, 0.0);
    }

    out_T = normalize(cross(U, N));
    out_B = cross(N, out_T);
}

vec3 MapToCone(float r1, float r2, vec3 N, float radius) {
    const vec2 offset = 2.0 * vec2(r1, r2) - vec2(1.0);

    if (offset[0] == 0.0 && offset[1] == 0.0) {
        return N;
    }

    float theta, r;

    if (abs(offset[0]) > abs(offset[1])) {
        r = offset[0];
        theta = 0.25 * PI * (offset[1] / offset[0]);
    } else {
        r = offset[1];
        theta = 0.5 * PI * (1.0 - 0.5 * (offset[0] / offset[1]));
    }

    const vec2 uv = vec2(radius * r * cos(theta), radius * r * sin(theta));

    vec3 LT, LB;
    create_tbn(N, LT, LB);

    return N + uv[0] * LT + uv[1] * LB;
}

void SampleLightSource(vec3 P, int hi, vec2 sample_off, inout light_sample_t ls) {
    const float u1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_PICK] + sample_off[0]);
    const uint light_index = min(uint(u1 * g_params.li_count), uint(g_params.li_count - 1));

    const light_t l = g_lights[g_li_indices[light_index]];

    ls.col = uintBitsToFloat(l.type_and_param0.yzw);
    ls.col *= float(g_params.li_count);
    ls.cast_shadow = (l.type_and_param0.x & (1 << 5)) != 0 ? 1.0 : 0.0;

    const uint l_type = (l.type_and_param0.x & 0x1f);
    [[dont_flatten]] if (l_type == LIGHT_TYPE_SPHERE) {
        const float r1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]);
        const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

        vec3 center_to_surface = P - l.SPH_POS;
        float dist_to_center = length(center_to_surface);

        center_to_surface /= dist_to_center;

        // sample hemisphere
        const float r = sqrt(clamp(1.0 - r1 * r1, 0.0, 1.0));
        const float phi = 2.0 * PI * r2;
        vec3 sampled_dir = vec3(r * cos(phi), r * sin(phi), r1);

        vec3 LT, LB;
        create_tbn(center_to_surface, LT, LB);

        sampled_dir = LT * sampled_dir[0] + LB * sampled_dir[1] + center_to_surface * sampled_dir[2];

        const vec3 light_surf_pos = l.SPH_POS + sampled_dir * l.SPH_RADIUS;

        ls.L = light_surf_pos - P;
        ls.dist = length(ls.L);
        ls.L /= ls.dist;

        ls.area = l.SPH_AREA;
        const vec3 light_forward = normalize(light_surf_pos - l.SPH_POS);

        const float cos_theta = abs(dot(ls.L, light_forward));
        [[flatten]] if (cos_theta > 0.0) {
            ls.pdf = (ls.dist * ls.dist) / (0.5 * ls.area * cos_theta);
        }

        [[dont_flatten]] if (l.SPH_SPOT > 0.0) {
            const float _dot = -dot(ls.L, l.SPH_DIR);
            if (_dot > 0.0) {
                const float _angle = acos(clamp(_dot, 0.0, 1.0));
                ls.col *= clamp((l.SPH_SPOT - _angle) / l.SPH_BLEND, 0.0, 1.0);
            } else {
                ls.col *= 0.0;
            }
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_DIR) {
        ls.L = l.DIR_DIR;
        if (l.DIR_ANGLE != 0.0){
            const float r1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]);
            const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

            const float radius = tan(l.DIR_ANGLE);
            ls.L = normalize(MapToCone(r1, r2, ls.L, radius));
        }
        ls.area = 0.0;
        ls.dist = MAX_DIST;
        ls.pdf = 1.0;

        if ((l.type_and_param0.x & (1 << 6)) == 0) { // !visible
            ls.area = 0.0;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_RECT) {
        const vec3 light_pos = l.RECT_POS;
        const vec3 light_u = l.RECT_U;
        const vec3 light_v = l.RECT_V;

        const float r1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]) - 0.5;
        const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]) - 0.5;
        const vec3 lp = light_pos + light_u * r1 + light_v * r2;

        const vec3 to_light = lp - P;
        ls.dist = length(to_light);
        ls.L = (to_light / ls.dist);

        ls.area = l.RECT_AREA;
        vec3 light_forward = normalize(cross(light_u, light_v));

        const float cos_theta = dot(-ls.L, light_forward);
        if (cos_theta > 0.0) {
            ls.pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
        }

        if ((l.type_and_param0.x & (1 << 6)) == 0) { // !visible
            ls.area = 0.0;
        }

        [[dont_flatten]] if ((l.type_and_param0.x & (1 << 7)) != 0) { // sky portal
            vec3 env_col = g_params.env_col.xyz;
            const uint env_map = floatBitsToUint(g_params.env_col.w);
            if (env_map != 0xffffffff) {
#if BINDLESS
                env_col *= SampleLatlong_RGBE(env_map, ls.L, g_params.env_rotation);
#else
                env_col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation);
#endif
            }
            ls.col *= env_col;
            ls.dist = MAX_DIST;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_DISK) {
        const vec3 light_pos = l.DISK_POS;
        const vec3 light_u = l.DISK_U;
        const vec3 light_v = l.DISK_V;

        const float r1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]);
        const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

        vec2 offset = 2.0 * vec2(r1, r2) - vec2(1.0);
        if (offset[0] != 0.0 && offset[1] != 0.0) {
            float theta, r;
            if (abs(offset[0]) > abs(offset[1])) {
                r = offset[0];
                theta = 0.25 * PI * (offset[1] / offset[0]);
            } else {
                r = offset[1];
                theta = 0.5 * PI - 0.25 * PI * (offset[0] / offset[1]);
            }

            offset[0] = 0.5 * r * cos(theta);
            offset[1] = 0.5 * r * sin(theta);
        }

        const vec3 lp = light_pos + light_u * offset[0] + light_v * offset[1];

        const vec3 to_light = lp - P;
        ls.dist = length(to_light);
        ls.L = (to_light / ls.dist);

        ls.area = l.DISK_AREA;
        vec3 light_forward = normalize(cross(light_u, light_v));

        const float cos_theta = dot(-ls.L, light_forward);
        [[flatten]] if (cos_theta > 0.0) {
            ls.pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
        }

        if ((l.type_and_param0.x & (1 << 6)) == 0) { // !visible
            ls.area = 0.0;
        }

        [[dont_flatten]] if ((l.type_and_param0.x & (1 << 7)) != 0) { // sky portal
            vec3 env_col = g_params.env_col.xyz;
            const uint env_map = floatBitsToUint(g_params.env_col.w);
            if (env_map != 0xffffffff) {
#if BINDLESS
                env_col *= SampleLatlong_RGBE(env_map, ls.L, g_params.env_rotation);
#else
                env_col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation);
#endif
            }
            ls.col *= env_col;
            ls.dist = MAX_DIST;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_LINE) {
        const vec3 light_pos = l.LINE_POS;
        const vec3 light_dir = l.LINE_V;

        const float r1 = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]);
        const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

        const vec3 center_to_surface = P - light_pos;

        vec3 light_u = normalize(cross(center_to_surface, light_dir));
        vec3 light_v = cross(light_u, light_dir);

        const float phi = PI * r1;
        const vec3 normal = cos(phi) * light_u + sin(phi) * light_v;

        const vec3 lp = light_pos + normal * l.LINE_RADIUS + (r2 - 0.5) * light_dir * l.LINE_HEIGHT;

        const vec3 to_light = lp - P;
        ls.dist = length(to_light);
        ls.L = (to_light / ls.dist);

        ls.area = l.LINE_AREA;

        const float cos_theta = 1.0 - abs(dot(ls.L, light_dir));
        [[flatten]] if (cos_theta != 0.0) {
            ls.pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
        }

        if ((l.type_and_param0.x & (1 << 6)) == 0) { // !visible
            ls.area = 0.0;
        }

        // probably can not be a portal, but still..
        [[dont_flatten]] if ((l.type_and_param0.x & (1 << 7)) != 0) { // sky portal
            vec3 env_col = g_params.env_col.xyz;
            const uint env_map = floatBitsToUint(g_params.env_col.w);
            if (env_map != 0xffffffff) {
#if BINDLESS
                env_col *= SampleLatlong_RGBE(env_map, ls.L, g_params.env_rotation);
#else
                env_col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation);
#endif
            }
            ls.col *= env_col;
            ls.dist = MAX_DIST;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_TRI) {
        const uint ltri_index = floatBitsToUint(l.TRI_TRI_INDEX);
        const transform_t ltr = g_transforms[floatBitsToUint(l.TRI_XFORM_INDEX)];

        const vertex_t v1 = g_vertices[g_vtx_indices[ltri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[ltri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[ltri_index * 3 + 2]];

        const vec3 p1 = vec3(v1.p[0], v1.p[1], v1.p[2]),
                   p2 = vec3(v2.p[0], v2.p[1], v2.p[2]),
                   p3 = vec3(v3.p[0], v3.p[1], v3.p[2]);
        const vec2 uv1 = vec2(v1.t[0][0], v1.t[0][1]),
                   uv2 = vec2(v2.t[0][0], v2.t[0][1]),
                   uv3 = vec2(v3.t[0][0], v3.t[0][1]);

        const float r1 = sqrt(fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]));
        const float r2 = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

        const vec2 luvs = uv1 * (1.0 - r1) + r1 * (uv2 * (1.0 - r2) + uv3 * r2);
        const vec3 lp = (ltr.xform * vec4(p1 * (1.0 - r1) + r1 * (p2 * (1.0 - r2) + p3 * r2), 1.0)).xyz;

        vec3 light_forward = (ltr.xform * vec4(cross(p2 - p1, p3 - p1), 0.0)).xyz;
        ls.area = 0.5 * length(light_forward);
        light_forward = normalize(light_forward);

        const vec3 to_light = lp - P;
        ls.dist = length(to_light);
        ls.L = (to_light / ls.dist);

        const float cos_theta = abs(dot(ls.L, light_forward)); // abs for doublesided light
        [[flatten]] if (cos_theta > 0.0) {
            ls.pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
        }

        const material_t lmat = g_materials[(g_tri_materials[ltri_index] >> 16u) & MATERIAL_INDEX_BITS];
        if (lmat.textures[BASE_TEXTURE] != 0xffffffff) {
            ls.col *= SampleBilinear(lmat.textures[BASE_TEXTURE], luvs, 0 /* lod */).xyz;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_ENV) {
        const float rand = u1 * float(g_params.li_count) - float(light_index);

        const float rx = fract(g_random_seq[hi + RAND_DIM_LIGHT_U] + sample_off[0]);
        const float ry = fract(g_random_seq[hi + RAND_DIM_LIGHT_V] + sample_off[1]);

        const vec4 dir_and_pdf = Sample_EnvQTree(g_params.env_rotation, g_env_qtree, g_params.env_qtree_levels, rand, rx, ry);

        ls.L = dir_and_pdf.xyz;
        ls.col *= g_params.env_col.xyz;

        const uint env_map = floatBitsToUint(g_params.env_col.w);
#if BINDLESS
        ls.col *= SampleLatlong_RGBE(env_map, ls.L, g_params.env_rotation);
#else
        ls.col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation);
#endif

        ls.area = 1.0;
        ls.dist = MAX_DIST;
        ls.pdf = dir_and_pdf.w;
    }
}

vec3 ShadeSurface(hit_data_t inter, ray_data_t ray) {
    const vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    const vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);

    [[dont_flatten]] if (inter.mask == 0) {
#if PRIMARY
        vec3 env_col = g_params.back_col.xyz;
        const uint env_map = floatBitsToUint(g_params.back_col.w);
        const float env_map_rotation = g_params.back_rotation;
#else
        vec3 env_col = (ray.depth & 0x00ffffff) != 0 ? g_params.env_col.xyz : g_params.back_col.xyz;
        const uint env_map = (ray.depth & 0x00ffffff) != 0 ? floatBitsToUint(g_params.env_col.w) : floatBitsToUint(g_params.back_col.w);
        const float env_map_rotation = (ray.depth & 0x00ffffff) != 0 ? g_params.env_rotation : g_params.back_rotation;
#endif
        if (env_map != 0xffffffff) {
#if BINDLESS
            env_col *= SampleLatlong_RGBE(env_map, rd, env_map_rotation);
#else
            env_col *= SampleLatlong_RGBE(g_textures[env_map], rd, env_map_rotation);
#endif
            if (g_params.env_qtree_levels > 0) {
                const float light_pdf = Evaluate_EnvQTree(env_map_rotation, g_env_qtree, g_params.env_qtree_levels, rd);
                const float bsdf_pdf = ray.pdf;

                const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                env_col *= mis_weight;
            }
        }
        return vec3(ray.c[0] * env_col[0], ray.c[1] * env_col[1], ray.c[2] * env_col[2]);
    }

    const vec3 I = rd;
    const vec3 P = ro + inter.t * rd;

    [[dont_flatten]] if (inter.obj_index < 0) { // Area light intersection
        const light_t l = g_lights[-inter.obj_index - 1];

        vec3 lcol = uintBitsToFloat(l.type_and_param0.yzw);
        [[dont_flatten]] if ((l.type_and_param0.x & (1 << 7)) != 0) { // sky portal
            vec3 env_col = g_params.env_col.xyz;
            const uint env_map = floatBitsToUint(g_params.env_col.w);
            if (env_map != 0xffffffff) {
#if BINDLESS
                env_col *= SampleLatlong_RGBE(env_map, I, g_params.env_rotation);
#else
                env_col *= SampleLatlong_RGBE(g_textures[env_map], I, g_params.env_rotation);
#endif
            }
            lcol *= env_col;
        }

        const uint l_type = (l.type_and_param0.x & 0x1f);
        if (l_type == LIGHT_TYPE_SPHERE) {
            const vec3 light_pos = l.SPH_POS;
            const float light_area = l.SPH_AREA;

            const vec3 op = light_pos - ro;
            const float b = dot(op, rd);
            const float det = sqrt(b * b - dot(op, op) + l.SPH_RADIUS * l.SPH_RADIUS);

            const float cos_theta = dot(rd, normalize(light_pos - P));

            const float light_pdf = (inter.t * inter.t) / (0.5 * light_area * cos_theta);
            const float bsdf_pdf = ray.pdf;

            float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;

            [[dont_flatten]] if (l.SPH_SPOT > 0.0 && l.SPH_BLEND > 0.0) {
                const float _dot = -dot(I, l.SPH_DIR);
                const float _angle = acos(clamp(_dot, 0.0, 1.0));
                [[flatten]] if (l.SPH_BLEND > 0.0) {
                    lcol *= clamp((l.SPH_SPOT - _angle) / l.SPH_BLEND, 0.0, 1.0);
                }
            }
        } else if (l_type == LIGHT_TYPE_RECT) {
            const vec3 light_pos = l.RECT_POS;
            const vec3 light_u = l.RECT_U;
            const vec3 light_v = l.RECT_V;

            const vec3 light_forward = normalize(cross(light_u, light_v));
            const float light_area = l.RECT_AREA;

            const float plane_dist = dot(light_forward, light_pos);
            const float cos_theta = dot(rd, light_forward);

            const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        } else if (l_type == LIGHT_TYPE_DISK) {
            const vec3 light_pos = l.DISK_POS;
            const vec3 light_u = l.DISK_U;
            const vec3 light_v = l.DISK_V;

            const vec3 light_forward = normalize(cross(light_u, light_v));
            const float light_area = l.DISK_AREA;

            const float plane_dist = dot(light_forward, light_pos);
            const float cos_theta = dot(rd, light_forward);

            const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        } else if (l_type == LIGHT_TYPE_LINE) {
            const vec3 light_dir = l.LINE_V;
            const float light_area = l.LINE_AREA;

            const float cos_theta = 1.0 - abs(dot(rd, light_dir));

            const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        }

        return vec3(ray.c[0] * lcol[0], ray.c[1] * lcol[1], ray.c[2] * lcol[2]);
    }

    const bool is_backfacing = (inter.prim_index < 0);
    const uint tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

    material_t mat = g_materials[(g_tri_materials[tri_index] >> 16u) & MATERIAL_INDEX_BITS];

    const transform_t tr = g_transforms[floatBitsToUint(g_mesh_instances[inter.obj_index].bbox_min.w)];

    const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
    const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
    const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

    const vec3 p1 = vec3(v1.p[0], v1.p[1], v1.p[2]);
    const vec3 p2 = vec3(v2.p[0], v2.p[1], v2.p[2]);
    const vec3 p3 = vec3(v3.p[0], v3.p[1], v3.p[2]);

    const float w = 1.0 - inter.u - inter.v;
    vec3 N = normalize(vec3(v1.n[0], v1.n[1], v1.n[2]) * w +
                       vec3(v2.n[0], v2.n[1], v2.n[2]) * inter.u +
                       vec3(v3.n[0], v3.n[1], v3.n[2]) * inter.v);
    vec2 uvs = vec2(v1.t[0][0], v1.t[0][1]) * w +
               vec2(v2.t[0][0], v2.t[0][1]) * inter.u +
               vec2(v3.t[0][0], v3.t[0][1]) * inter.v;

    vec3 plane_N = cross(vec3(p2 - p1), vec3(p3 - p1));
    const float pa = length(plane_N);
    plane_N /= pa;

    vec3 B = vec3(v1.b[0], v1.b[1], v1.b[2]) * w +
             vec3(v2.b[0], v2.b[1], v2.b[2]) * inter.u +
             vec3(v3.b[0], v3.b[1], v3.b[2]) * inter.v;
    vec3 T = cross(B, N);

    if (is_backfacing) {
        if ((g_tri_materials[tri_index] & 0xffff) == 0xffff) {
            return vec3(0.0);
        } else {
            mat = g_materials[g_tri_materials[tri_index] & MATERIAL_INDEX_BITS];
            plane_N = -plane_N;
            N = -N;
            B = -B;
            T = -T;
        }
    }

    plane_N = TransformNormal(plane_N, tr.inv_xform);
    N = TransformNormal(N, tr.inv_xform);
    B = TransformNormal(B, tr.inv_xform);
    T = TransformNormal(T, tr.inv_xform);

#ifdef USE_RAY_DIFFERENTIALS
    #error "Not implemented"

    //const auto do_dx = simd_fvec4{ray.do_dx[0], ray.do_dx[1], ray.do_dx[2], 0.0};
    //const auto do_dy = simd_fvec4{ray.do_dy[0], ray.do_dy[1], ray.do_dy[2], 0.0};
    //const auto dd_dx = simd_fvec4{ray.dd_dx[0], ray.dd_dx[1], ray.dd_dx[2], 0.0};
    //const auto dd_dy = simd_fvec4{ray.dd_dy[0], ray.dd_dy[1], ray.dd_dy[2], 0.0};

    //derivatives_t surf_der;
    //ComputeDerivatives(I, inter.t, do_dx, do_dy, dd_dx, dd_dy, v1, v2, v3, plane_N, *tr, surf_der);
#else
    const float ta = abs((v2.t[0][0] - v1.t[0][0]) * (v3.t[0][1] - v1.t[0][1]) -
                         (v3.t[0][0] - v1.t[0][0]) * (v2.t[0][1] - v1.t[0][1]));

    const float cone_width = ray.cone_width + ray.cone_spread * inter.t;

    float lambda = 0.5 * log2(ta / pa);
    lambda += log2(cone_width);
    // lambda += 0.5 * fast_log2(tex_res.x * tex_res.y);
    // lambda -= fast_log2(std::abs(dot(I, plane_N)));
#endif

    vec2 sample_off = vec2(construct_float(hash(ray.xy)), construct_float(hash(hash(ray.xy))));

    vec3 col = vec3(0.0);

    const int diff_depth = ray.depth & 0x000000ff;
    const int spec_depth = (ray.depth >> 8) & 0x000000ff;
    const int refr_depth = (ray.depth >> 16) & 0x000000ff;
    const int transp_depth = (ray.depth >> 24) & 0x000000ff;
    // NOTE: transparency depth is not accounted here
    const int total_depth = diff_depth + spec_depth + refr_depth;

    const int hi = g_params.hi + (total_depth + transp_depth) * RAND_DIM_BOUNCE_COUNT;

    float mix_rand = fract(g_random_seq[hi + RAND_DIM_BSDF_PICK] + sample_off[0]);
    float mix_weight = 1.0;

    // resolve mix material
    while (mat.type == MixNode) {
        float mix_val = mat.tangent_rotation_or_strength;
        if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
            mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], uvs, 0).r;
        }

        const float eta = is_backfacing ? (mat.ext_ior / mat.int_ior) : (mat.int_ior / mat.ext_ior);
        const float RR = mat.int_ior != 0.0 ? fresnel_dielectric_cos(dot(I, N), eta) : 1.0;

        mix_val *= clamp(RR, 0.0, 1.0);

        if (mix_rand > mix_val) {
            mix_weight *= (mat.flags & MAT_FLAG_MIX_ADD) != 0 ? 1.0 / (1.0 - mix_val) : 1.0;

            mat = g_materials[mat.textures[MIX_MAT1]];
            mix_rand = (mix_rand - mix_val) / (1.0 - mix_val);
        } else {
            mix_weight *= (mat.flags & MAT_FLAG_MIX_ADD) != 0 ? 1.0 / mix_val : 1.0;

            mat = g_materials[mat.textures[MIX_MAT2]];
            mix_rand = mix_rand / mix_val;
        }
    }

    // apply normal map
    [[dont_flatten]] if (mat.textures[NORMALS_TEXTURE] != 0xffffffff) {
        vec3 normals = vec3(SampleBilinear(mat.textures[NORMALS_TEXTURE], uvs, 0).xy, 1.0);
#if BINDLESS
        normals = normals * 2.0 - 1.0;
        if ((mat.textures[NORMALS_TEXTURE] & TEX_RECONSTRUCT_Z_BIT) != 0) {
            normals.z = sqrt(1.0 - dot(normals.xy, normals.xy));
        }
#else
        normals = normals * 2.0 - 1.0;
        if ((g_textures[mat.textures[NORMALS_TEXTURE]].size & ATLAS_TEX_RECONSTRUCT_Z_BIT) != 0) {
            normals.z = sqrt(1.0 - dot(normals.xy, normals.xy));
        }
#endif
        vec3 in_normal = N;
        N = normalize(normals[0] * T + normals[2] * N + normals[1] * B);
        if ((mat.normal_map_strength_unorm & 0xffff) != 0xffff) {
            N = normalize(in_normal + (N - in_normal) * unpack_unorm_16(mat.normal_map_strength_unorm & 0xffff));
        }
        N = ensure_valid_reflection(plane_N, -I, N);
    }

#if 0
    //create_tbn_matrix(N, _tangent_from_world);
#else
    // Find radial tangent in local space
    const vec3 P_ls = p1 * w + p2 * inter.u + p3 * inter.v;
    // rotate around Y axis by 90 degrees in 2d
    vec3 tangent = vec3(-P_ls[2], 0.0, P_ls[0]);
    tangent = TransformNormal(tangent, tr.inv_xform);
    if (length2(cross(tangent, N)) == 0.0) {
        tangent = TransformNormal(P_ls, tr.inv_xform);
    }
    if (mat.tangent_rotation_or_strength != 0.0) {
        tangent = rotate_around_axis(tangent, N, mat.tangent_rotation_or_strength);
    }

    B = normalize(cross(tangent, N));
    T = cross(N, B);
#endif

#if USE_NEE
    light_sample_t ls;
    ls.col = ls.L = vec3(0.0);
    ls.area = ls.pdf = ls.dist = 0;
    if (/*pi.should_add_direct_light() &&*/ g_params.li_count != 0 && mat.type != EmissiveNode) {
        SampleLightSource(P, hi, sample_off, ls);
    }
    const float N_dot_L = dot(N, ls.L);
#endif

    const float mat_ior = is_backfacing ? mat.ext_ior : mat.int_ior;

    vec3 base_color = vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
    [[dont_flatten]] if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
        const float base_lod = get_texture_lod(texSize(mat.textures[BASE_TEXTURE]), lambda);
        base_color *= SampleBilinear(mat.textures[BASE_TEXTURE], uvs, int(base_lod), true /* YCoCg */, true /* SRGB */).rgb;
    }

    vec3 tint_color = vec3(0.0);

    const float base_color_lum = lum(base_color);
    [[flatten]] if (base_color_lum > 0.0) {
        tint_color = base_color / base_color_lum;
    }

    float roughness = unpack_unorm_16(mat.roughness_and_anisotropic & 0xffff);
    [[dont_flatten]] if (mat.textures[ROUGH_TEXTURE] != 0xffffffff) {
        const float roughness_lod = get_texture_lod(texSize(mat.textures[ROUGH_TEXTURE]), lambda);
        roughness *= SampleBilinear(mat.textures[ROUGH_TEXTURE], uvs, int(roughness_lod), false /* YCoCg */, true /* SRGB */).r;
    }

    const float rand_u = fract(g_random_seq[hi + RAND_DIM_BSDF_U] + sample_off[0]);
    const float rand_v = fract(g_random_seq[hi + RAND_DIM_BSDF_V] + sample_off[1]);

    ray_data_t new_ray;
    new_ray.c[0] = new_ray.c[1] = new_ray.c[2] = 0.0;
#ifndef USE_RAY_DIFFERENTIALS
    new_ray.cone_width = cone_width;
    new_ray.cone_spread = ray.cone_spread;
#endif
    new_ray.xy = ray.xy;
    new_ray.pdf = 0.0;

    ///

    [[dont_flatten]] if (mat.type == DiffuseNode) {
#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && N_dot_L > 0.0) {
            const vec4 diff_col = Evaluate_OrenDiffuse_BSDF(-I, N, ls.L, roughness, base_color);
            const float bsdf_pdf = diff_col[3];

            float mis_weight = 1.0;
            if (ls.area > 0.0) {
                mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
            }

            const vec3 lcol = ls.col * diff_col.xyz * (mix_weight * mis_weight / ls.pdf);

            [[dont_flatten]] if (ls.cast_shadow > 0.5) {
                // schedule shadow ray
                shadow_ray_t sh_r;

                vec3 new_o = offset_ray(P, plane_N);
                sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
                sh_r.d[0] = ls.L[0]; sh_r.d[1] = ls.L[1]; sh_r.d[2] = ls.L[2];
                sh_r.dist = ls.dist - 10.0 * HIT_BIAS;
                sh_r.c[0] = ray.c[0] * lcol[0];
                sh_r.c[1] = ray.c[1] * lcol[1];
                sh_r.c[2] = ray.c[2] * lcol[2];
                sh_r.xy = ray.xy;
                sh_r.depth = ray.depth;

                const uint index = atomicAdd(g_inout_counters[2], 1);
                g_out_sh_rays[index] = sh_r;
            } else {
                // apply light immediately
                col += lcol;
            }
        }
#endif

        [[dont_flatten]] if (diff_depth < g_params.max_diff_depth && total_depth < g_params.max_total_depth) {
            vec3 V;
            const vec4 F = Sample_OrenDiffuse_BSDF(T, B, N, I, roughness, base_color, rand_u, rand_v, V);

            new_ray.depth = ray.depth + 0x00000001;

            vec3 new_o = offset_ray(P, plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

            new_ray.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            new_ray.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            new_ray.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

#ifdef USE_RAY_DIFFERENTIALS
            // TODO: ...
#endif
        }
    } else [[dont_flatten]] if (mat.type == GlossyNode) {
        const float specular = 0.5;
        const float spec_ior = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;
        const float spec_F0 = fresnel_dielectric_cos(1.0, spec_ior);
        const float roughness2 = roughness * roughness;

#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && roughness2 * roughness2 >= 1e-7 && N_dot_L > 0.0) {
            const vec3 H = normalize(ls.L - I);

            const vec3 view_dir_ts = tangent_from_world(T, B, N, -I);
            const vec3 light_dir_ts = tangent_from_world(T, B, N, ls.L);
            const vec3 sampled_normal_ts = tangent_from_world(T, B, N, H);

            const vec4 spec_col = Evaluate_GGXSpecular_BSDF(
                view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2, spec_ior, spec_F0, base_color);
            const float bsdf_pdf = spec_col[3];

            float mis_weight = 1.0;
            if (ls.area > 0.0) {
                mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
            }
            const vec3 lcol = ls.col * spec_col.rgb * (mix_weight * mis_weight / ls.pdf);

            [[dont_flatten]] if (ls.cast_shadow > 0.5) {
                // schedule shadow ray
                shadow_ray_t sh_r;

                vec3 new_o = offset_ray(P, plane_N);
                sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
                sh_r.d[0] = ls.L[0]; sh_r.d[1] = ls.L[1]; sh_r.d[2] = ls.L[2];
                sh_r.dist = ls.dist - 10.0 * HIT_BIAS;
                sh_r.c[0] = ray.c[0] * lcol[0];
                sh_r.c[1] = ray.c[1] * lcol[1];
                sh_r.c[2] = ray.c[2] * lcol[2];
                sh_r.xy = ray.xy;
                sh_r.depth = ray.depth;

                const uint index = atomicAdd(g_inout_counters[2], 1);
                g_out_sh_rays[index] = sh_r;
            } else {
                // apply light immediately
                col += lcol;
            }
        }
#endif

        [[dont_flatten]] if (spec_depth < g_params.max_spec_depth && total_depth < g_params.max_total_depth) {
            vec3 V;
            const vec4 F =
                Sample_GGXSpecular_BSDF(T, B, N, I, roughness, 0.0, spec_ior, spec_F0, base_color, rand_u, rand_v, V);

            new_ray.depth = ray.depth + 0x00000100;

            vec3 new_o = offset_ray(P, plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

            new_ray.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            new_ray.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            new_ray.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

#ifdef USE_RAY_DIFFERENTIALS
            // TODO: ...
#endif
        }
    } else [[dont_flatten]] if (mat.type == RefractiveNode) {
        const float eta = is_backfacing ? (mat.int_ior / mat.ext_ior) : (mat.ext_ior / mat.int_ior);
        const float roughness2 = roughness * roughness;

#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && roughness2 * roughness2 >= 1e-7 && N_dot_L < 0.0) {
            const vec3 H = normalize(ls.L - I * eta);
            const vec3 view_dir_ts = tangent_from_world(T, B, N, -I);
            const vec3 light_dir_ts = tangent_from_world(T, B, N, ls.L);
            const vec3 sampled_normal_ts = tangent_from_world(T, B, N, H);

            const vec4 refr_col =
                Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, eta, base_color);
            const float bsdf_pdf = refr_col[3];

            float mis_weight = 1.0;
            if (ls.area > 0.0) {
                mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
            }
            const vec3 lcol = ls.col * refr_col.rgb * (mix_weight * mis_weight / ls.pdf);

            [[dont_flatten]] if (ls.cast_shadow > 0.5) {
                // schedule shadow ray
                shadow_ray_t sh_r;

                vec3 new_o = offset_ray(P, -plane_N);
                sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
                sh_r.d[0] = ls.L[0]; sh_r.d[1] = ls.L[1]; sh_r.d[2] = ls.L[2];
                sh_r.dist = ls.dist - 10.0 * HIT_BIAS;
                sh_r.c[0] = ray.c[0] * lcol[0];
                sh_r.c[1] = ray.c[1] * lcol[1];
                sh_r.c[2] = ray.c[2] * lcol[2];
                sh_r.xy = ray.xy;
                sh_r.depth = ray.depth;

                const uint index = atomicAdd(g_inout_counters[2], 1);
                g_out_sh_rays[index] = sh_r;
            } else {
                // apply light immediately
                col += lcol;
            }
        }
#endif

        [[dont_flatten]] if (refr_depth < g_params.max_refr_depth && total_depth < g_params.max_total_depth) {
            vec4 _V;
            const vec4 F = Sample_GGXRefraction_BSDF(T, B, N, I, roughness, eta, base_color, rand_u, rand_v, _V);

            const vec3 V = _V.xyz;
            const float m = _V[3];

            new_ray.depth = ray.depth + 0x00010000;

            new_ray.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
            new_ray.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
            new_ray.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

            vec3 new_o = offset_ray(P, -plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];
#ifdef USE_RAY_DIFFERENTIALS
            // TODO: ...
#endif
        }
    } else [[dont_flatten]] if (mat.type == EmissiveNode) {
        float mis_weight = 1.0;
#if USE_NEE && !PRIMARY
        [[dont_flatten]] if ((mat.flags & MAT_FLAG_MULT_IMPORTANCE) != 0) {
            const vec3 p1 = vec3(v1.p[0], v1.p[1], v1.p[2]),
                       p2 = vec3(v2.p[0], v2.p[1], v2.p[2]),
                       p3 = vec3(v3.p[0], v3.p[1], v3.p[2]);

            vec3 light_forward = (tr.xform * vec4(cross(p2 - p1, p3 - p1), 0.0)).xyz;
            const float light_forward_len = length(light_forward);
            light_forward /= light_forward_len;
            const float tri_area = 0.5 * light_forward_len;

            const float cos_theta = abs(dot(I, light_forward)); // abs for doublesided light
            if (cos_theta > 0.0) {
                const float light_pdf = (inter.t * inter.t) / (tri_area * cos_theta);
                const float bsdf_pdf = ray.pdf;

                mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            }
        }
#endif
        col += mix_weight * mis_weight * mat.tangent_rotation_or_strength * base_color;
    } else [[dont_flatten]] if (mat.type == PrincipledNode) {
        float metallic = unpack_unorm_16((mat.tint_and_metallic >> 16) & 0xffff);
        [[dont_flatten]] if (mat.textures[METALLIC_TEXTURE] != 0xffffffff) {
            const float metallic_lod = get_texture_lod(texSize(mat.textures[METALLIC_TEXTURE]), lambda);
            metallic *= SampleBilinear(mat.textures[METALLIC_TEXTURE], uvs, int(metallic_lod)).r;
        }

        float specular = unpack_unorm_16(mat.specular_and_specular_tint & 0xffff);
        [[dont_flatten]] if (mat.textures[SPECULAR_TEXTURE] != 0xffffffff) {
            const float specular_lod = get_texture_lod(texSize(mat.textures[SPECULAR_TEXTURE]), lambda);
            specular *= SampleBilinear(mat.textures[SPECULAR_TEXTURE], uvs, int(specular_lod)).r;
        }

        const float specular_tint = unpack_unorm_16((mat.specular_and_specular_tint >> 16) & 0xffff);
        const float transmission = unpack_unorm_16(mat.transmission_and_transmission_roughness & 0xffff);
        const float clearcoat = unpack_unorm_16(mat.clearcoat_and_clearcoat_roughness & 0xffff);
        const float clearcoat_roughness = unpack_unorm_16((mat.clearcoat_and_clearcoat_roughness >> 16) & 0xffff);
        const float sheen = unpack_unorm_16(mat.sheen_and_sheen_tint & 0xffff);
        const float sheen_tint = unpack_unorm_16((mat.sheen_and_sheen_tint >> 16) & 0xffff);

        vec3 spec_tmp_col = mix(vec3(1.0), tint_color, specular_tint);
        spec_tmp_col = mix(specular * 0.08 * spec_tmp_col, base_color, metallic);

        const float spec_ior = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;
        const float spec_F0 = fresnel_dielectric_cos(1.0, spec_ior);

        // Approximation of FH (using shading normal)
        const float FN = (fresnel_dielectric_cos(dot(I, N), spec_ior) - spec_F0) / (1.0 - spec_F0);

        const vec3 approx_spec_col = mix(spec_tmp_col, vec3(1.0), FN);
        const float spec_color_lum = lum(approx_spec_col);

        float diffuse_weight, specular_weight, clearcoat_weight, refraction_weight;
        get_lobe_weights(mix(base_color_lum, 1.0, sheen), spec_color_lum, specular, metallic, transmission, clearcoat,
                         diffuse_weight, specular_weight, clearcoat_weight, refraction_weight);

        const vec3 _base_color = /*pi.should_consider_albedo()*/ true ? base_color : vec3(1.0);
        const vec3 sheen_color = sheen * mix(vec3(1.0), tint_color, sheen_tint);

        const float eta = is_backfacing ? (mat.int_ior / mat.ext_ior) : (mat.ext_ior / mat.int_ior);
        const float fresnel = fresnel_dielectric_cos(dot(I, N), 1.0 / eta);

        const float clearcoat_ior = (2.0 / (1.0 - sqrt(0.08 * clearcoat))) - 1.0;
        const float clearcoat_F0 = fresnel_dielectric_cos(1.0, clearcoat_ior);
        const float clearcoat_roughness2 = clearcoat_roughness * clearcoat_roughness;

        const float transmission_roughness =
            1.0 - (1.0 - roughness) * (1.0 - unpack_unorm_16((mat.transmission_and_transmission_roughness >> 16) & 0xffff));
        const float transmission_roughness2 = transmission_roughness * transmission_roughness;

#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0) {
            vec3 lcol = vec3(0.0);
            float bsdf_pdf = 0.0;

            [[dont_flatten]] if (diffuse_weight > 1e-7 && N_dot_L > 0.0) {
                vec4 diff_col = Evaluate_PrincipledDiffuse_BSDF(-I, N, ls.L, roughness, _base_color, sheen_color,
                                                                false);
                bsdf_pdf += diffuse_weight * diff_col[3];
                diff_col *= (1.0 - metallic);

                lcol += ls.col * N_dot_L * diff_col.rgb / (PI * ls.pdf);
            }

            vec3 H;
            [[flatten]] if (N_dot_L > 0.0) {
                H = normalize(ls.L - I);
            } else {
                H = normalize(ls.L - I * eta);
            }

            const float roughness2 = roughness * roughness;
            const float aspect = sqrt(1.0 - 0.9 * unpack_unorm_16((mat.roughness_and_anisotropic >> 16) & 0xffff));

            const float alpha_x = roughness2 / aspect;
            const float alpha_y = roughness2 * aspect;

            const vec3 view_dir_ts = tangent_from_world(T, B, N, -I);
            const vec3 light_dir_ts = tangent_from_world(T, B, N, ls.L);
            const vec3 sampled_normal_ts = tangent_from_world(T, B, N, H);

            [[dont_flatten]] if (specular_weight > 0.0 && alpha_x * alpha_y >= 1e-7 && N_dot_L > 0.0) {
                const vec4 spec_col = Evaluate_GGXSpecular_BSDF(
                    view_dir_ts, sampled_normal_ts, light_dir_ts, alpha_x, alpha_y, spec_ior, spec_F0, spec_tmp_col);
                bsdf_pdf += specular_weight * spec_col[3];
                lcol += ls.col * spec_col.rgb / ls.pdf;
            }

            [[dont_flatten]] if (clearcoat_weight > 0.0 && clearcoat_roughness2 * clearcoat_roughness2 >= 1e-7 && N_dot_L > 0.0) {
                const vec4 clearcoat_col = Evaluate_PrincipledClearcoat_BSDF(
                    view_dir_ts, sampled_normal_ts, light_dir_ts, clearcoat_roughness2, clearcoat_ior, clearcoat_F0);
                bsdf_pdf += clearcoat_weight * clearcoat_col[3];
                lcol += 0.25 * ls.col * clearcoat_col.rgb / ls.pdf;
            }

            [[dont_flatten]] if (refraction_weight > 0.0) {
                [[dont_flatten]] if (fresnel != 0.0 && roughness2 * roughness2 >= 1e-7 && N_dot_L > 0.0) {
                    const vec4 spec_col =
                        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2,
                                                  1.0 /* ior */, 0.0 /* F0 */, vec3(1.0));
                    bsdf_pdf += refraction_weight * fresnel * spec_col[3];
                    lcol += ls.col * spec_col.rgb * (fresnel / ls.pdf);
                }

                [[dont_flatten]] if (fresnel != 1.0 && transmission_roughness2 * transmission_roughness2 >= 1e-7 && N_dot_L < 0.0) {
                    const vec4 refr_col = Evaluate_GGXRefraction_BSDF(
                        view_dir_ts, sampled_normal_ts, light_dir_ts, transmission_roughness2, eta, base_color);
                    bsdf_pdf += refraction_weight * (1.0 - fresnel) * refr_col[3];
                    lcol += ls.col * refr_col.rgb * ((1.0 - fresnel) / ls.pdf);
                }
            }

            float mis_weight = 1.0;
            [[flatten]] if (ls.area > 0.0) {
                mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
            }
            lcol *= mix_weight * mis_weight;

            [[dont_flatten]] if (ls.cast_shadow > 0.5) {
                // schedule shadow ray
                shadow_ray_t sh_r;

                vec3 new_o = offset_ray(P, N_dot_L < 0.0 ? -plane_N : plane_N);
                sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
                sh_r.d[0] = ls.L[0]; sh_r.d[1] = ls.L[1]; sh_r.d[2] = ls.L[2];
                sh_r.dist = ls.dist - 10.0 * HIT_BIAS;
                sh_r.c[0] = ray.c[0] * lcol[0];
                sh_r.c[1] = ray.c[1] * lcol[1];
                sh_r.c[2] = ray.c[2] * lcol[2];
                sh_r.xy = ray.xy;
                sh_r.depth = ray.depth;

                const uint index = atomicAdd(g_inout_counters[2], 1);
                g_out_sh_rays[index] = sh_r;
            } else {
                // apply light immediately
                col += lcol;
            }
        }
#endif

        [[dont_flatten]] if (mix_rand < diffuse_weight) {
            //
            // Diffuse lobe
            //
            if (diff_depth < g_params.max_diff_depth && total_depth < g_params.max_total_depth) {
                vec3 V;
                vec4 diff_col = Sample_PrincipledDiffuse_BSDF(T, B, N, I, roughness, _base_color, sheen_color,
                                false, rand_u, rand_v, V);
                diff_col.rgb *= (1.0 - metallic);

                new_ray.depth = ray.depth + 0x00000001;

                const vec3 new_o = offset_ray(P, plane_N);
                new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
                new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

                new_ray.c[0] = ray.c[0] * diff_col[0] * mix_weight / diffuse_weight;
                new_ray.c[1] = ray.c[1] * diff_col[1] * mix_weight / diffuse_weight;
                new_ray.c[2] = ray.c[2] * diff_col[2] * mix_weight / diffuse_weight;
                new_ray.pdf = diff_col[3];

#ifdef USE_RAY_DIFFERENTIALS
                // TODO: ...
#endif
            }
        } else [[dont_flatten]] if (mix_rand < diffuse_weight + specular_weight) {
            //
            // Main specular lobe
            //
            if (spec_depth < g_params.max_spec_depth && total_depth < g_params.max_total_depth) {
                vec3 V;
                vec4 F = Sample_GGXSpecular_BSDF(T, B, N, I, roughness, unpack_unorm_16((mat.roughness_and_anisotropic >> 16) & 0xffff),
                                                 spec_ior, spec_F0, spec_tmp_col, rand_u, rand_v, V);
                F[3] *= specular_weight;

                new_ray.depth = ray.depth + 0x00000100;

                new_ray.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
                new_ray.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
                new_ray.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
                new_ray.pdf = F[3];

                const vec3 new_o = offset_ray(P, plane_N);
                new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
                new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

#ifdef USE_RAY_DIFFERENTIALS
                // TODO: ...
#endif
            }
        } else [[dont_flatten]] if (mix_rand < diffuse_weight + specular_weight + clearcoat_weight) {
            //
            // Clearcoat lobe (secondary specular)
            //
            if (spec_depth < g_params.max_spec_depth && total_depth < g_params.max_total_depth) {
                vec3 V;
                vec4 F = Sample_PrincipledClearcoat_BSDF(T, B, N, I, clearcoat_roughness2, clearcoat_ior,
                                                         clearcoat_F0, rand_u, rand_v, V);
                F[3] *= clearcoat_weight;

                new_ray.depth = ray.depth + 0x00000100;

                new_ray.c[0] = 0.25 * ray.c[0] * F[0] * mix_weight / F[3];
                new_ray.c[1] = 0.25 * ray.c[1] * F[1] * mix_weight / F[3];
                new_ray.c[2] = 0.25 * ray.c[2] * F[2] * mix_weight / F[3];
                new_ray.pdf = F[3];

                const vec3 new_o = offset_ray(P, plane_N);
                new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
                new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

#ifdef USE_RAY_DIFFERENTIALS
                // TODO: ...
#endif
            }
        } else /*if (mix_rand < diffuse_weight + specular_weight + clearcoat_weight + refraction_weight)*/ {
            //
            // Refraction/reflection lobes
            //
            [[dont_flatten]] if (((mix_rand >= fresnel && refr_depth < g_params.max_refr_depth) ||
                                  (mix_rand < fresnel && spec_depth < g_params.max_spec_depth)) &&
                                   total_depth < g_params.max_total_depth) {
                mix_rand -= diffuse_weight + specular_weight + clearcoat_weight;
                mix_rand /= refraction_weight;

                //////////////////

                vec4 F;
                vec3 V;
                [[dont_flatten]] if (mix_rand < fresnel) {
                    F = Sample_GGXSpecular_BSDF(T, B, N, I, roughness, 0.0 /* anisotropic */, 1.0 /* ior */,
                                                0.0 /* F0 */, vec3(1.0), rand_u, rand_v, V);

                    new_ray.depth = ray.depth + 0x00000100;

                    const vec3 new_o = offset_ray(P, plane_N);
                    new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];

#ifdef USE_RAY_DIFFERENTIALS
                    // TODO: ...
#endif
                } else {
                    vec4 _V;
                    F = Sample_GGXRefraction_BSDF(T, B, N, I, transmission_roughness, eta, base_color, rand_u, rand_v,
                                                  _V);
                    V = _V.xyz;

                    new_ray.depth = ray.depth + 0x00010000;

                    const vec3 new_o = offset_ray(P, -plane_N);
                    new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];

#ifdef USE_RAY_DIFFERENTIALS
                    // TODO: ...
#endif
                }

                F[3] *= refraction_weight;

                new_ray.c[0] = ray.c[0] * F[0] * mix_weight / F[3];
                new_ray.c[1] = ray.c[1] * F[1] * mix_weight / F[3];
                new_ray.c[2] = ray.c[2] * F[2] * mix_weight / F[3];
                new_ray.pdf = F[3];

                new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

#ifdef USE_RAY_DIFFERENTIALS
                // TODO: ...
#endif
            }
        }
    } /*else [[dont_flatten]] if (mat.type == TransparentNode) {
        [[dont_flatten]] if (transp_depth < g_params.max_transp_depth && total_depth < g_params.max_total_depth) {
            new_ray.depth = ray.depth + 0x01000000;
            new_ray.pdf = ray.pdf;

            const vec3 new_o = offset_ray(P, -plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = ray.d[0]; new_ray.d[1] = ray.d[1]; new_ray.d[2] = ray.d[2];
            new_ray.c[0] = ray.c[0]; new_ray.c[1] = ray.c[1]; new_ray.c[2] = ray.c[2];

#ifdef USE_RAY_DIFFERENTIALS
            // TODO: ...
#endif
        }
    }*/

#if USE_PATH_TERMINATION
    const bool can_terminate_path = total_depth > g_params.min_total_depth;
#else
    const bool can_terminate_path = false;
#endif

    const float lum = max(new_ray.c[0], max(new_ray.c[1], new_ray.c[2]));
    const float p = fract(g_random_seq[hi + RAND_DIM_TERMINATE] + sample_off[0]);
    const float q = can_terminate_path ? max(0.05, 1.0 - lum) : 0.0;
    [[dont_flatten]] if (p >= q && lum > 0.0 && new_ray.pdf > 0.0) {
        new_ray.c[0] /= (1.0 - q);
        new_ray.c[1] /= (1.0 - q);
        new_ray.c[2] /= (1.0 - q);
        const uint index = atomicAdd(g_inout_counters[0], 1);
        g_out_rays[index] = new_ray;
    }

    return vec3(ray.c[0] * col[0], ray.c[1] * col[1], ray.c[2] * col[2]);
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if PRIMARY
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);

    const int index = y * int(g_params.img_size.x) + x;
#else
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_inout_counters[1]) {
        return;
    }

    const int x = (g_rays[index].xy >> 16) & 0xffff;
    const int y = (g_rays[index].xy & 0xffff);
#endif

    hit_data_t inter = g_hits[index];
    ray_data_t ray = g_rays[index];

    vec3 col = ShadeSurface(inter, ray);
#if !PRIMARY
    col += imageLoad(g_out_img, ivec2(x, y)).rgb;
#endif
    imageStore(g_out_img, ivec2(x, y), vec4(col, 1.0));
}
