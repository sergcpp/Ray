#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#include "shade_interface.h"
#include "common.glsl"
#include "envmap.glsl"
#include "texture.glsl"
#include "light_bvh.glsl"
#include "traverse_bvh.glsl"
#if CACHE_UPDATE || CACHE_QUERY
    #include "spatial_radiance_cache.glsl"
#endif

layout(push_constant) uniform UniformParams {
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
    uint g_random_seq[];
};

layout(std430, binding = LIGHT_WNODES_BUF_SLOT) readonly buffer WNodes {
    light_wbvh_node_t g_light_wnodes[];
};

layout(binding = ENV_QTREE_TEX_SLOT) uniform texture2D g_env_qtree;

#if CACHE_QUERY
layout(std430, binding = CACHE_ENTRIES_BUF_SLOT) readonly buffer CacheEntries {
    uvec2 g_cache_entries[];
};

layout(std430, binding = CACHE_VOXELS_BUF_SLOT) readonly buffer CacheVoxels {
    uvec4 g_cache_voxels[];
};

bool hash_map_find(const uint64_t hash_key, inout uint cache_entry) {
    const uint hash = hash64(hash_key);
    const uint slot = hash % HASH_GRID_CACHE_ENTRIES_COUNT;
    const uint base_slot = hash_map_base_slot(slot);
    for (uint bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t stored_hash_key = (uint64_t(g_cache_entries[base_slot + bucket_offset].y) << 32u) | g_cache_entries[base_slot + bucket_offset].x;
        if (stored_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        } else if (HASH_GRID_ALLOW_COMPACTION && stored_hash_key == HASH_GRID_INVALID_HASH_KEY) {
            return false;
        }
    }
    return false;
}

uint find_entry(const vec3 p, const vec3 n, const cache_grid_params_t params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    uint cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
    hash_map_find(hash_key, cache_entry);
    return cache_entry;
}

#endif

layout(binding = OUT_IMG_SLOT, rgba32f) uniform image2D g_out_img;

layout(std430, binding = OUT_RAYS_BUF_SLOT) writeonly buffer OutRays {
    ray_data_t g_out_rays[];
};

layout(std430, binding = OUT_SH_RAYS_BUF_SLOT) writeonly buffer OutShRays {
    shadow_ray_t g_out_sh_rays[];
};

#if DETAILED_SKY
layout(std430, binding = OUT_SKY_RAYS_BUF_SLOT) writeonly buffer OutSkyRays {
    uint g_out_sky_rays[];
};
#endif

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer InoutCounters {
    uint g_inout_counters[];
};

#if OUTPUT_BASE_COLOR
    layout(binding = OUT_BASE_COLOR_IMG_SLOT, rgba32f) uniform image2D g_out_base_color_img;
#endif
#if OUTPUT_DEPTH_NORMALS || CACHE_UPDATE
    layout(binding = OUT_DEPTH_NORMALS_IMG_SLOT, rgba32f) uniform image2D g_out_depth_normals_img;
#endif

vec2 get_scrambled_2d_rand(const uint dim, const uint seed, const int _sample) {
    const uint i_seed = hash_combine(seed, dim),
               x_seed = hash_combine(seed, 2 * dim + 0),
               y_seed = hash_combine(seed, 2 * dim + 1);

    const uint shuffled_dim = uint(nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1));
    const uint shuffled_i = uint(nested_uniform_scramble_base2(_sample, i_seed) & (RAND_SAMPLES_COUNT - 1));
    return vec2(scramble_unorm(x_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0]),
                scramble_unorm(y_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1]));
}

float pow5(float v) { return (v * v) * (v * v) * v; }

float schlick_weight(float u) {
    const float m = saturate(1.0 - u);
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

vec2 calc_alpha(const float roughness, const float anisotropy, const float regularize_alpha) {
    const float roughness2 = sqr(roughness);
    const float aspect = sqrt(1.0 - 0.9 * anisotropy);

    vec2 alpha = vec2(roughness2 / aspect, roughness2 * aspect);
    [[flatten]] if (alpha.x < regularize_alpha) {
        alpha.x = clamp(2 * alpha.x, 0.25 * regularize_alpha, regularize_alpha);
    }
    [[flatten]] if (alpha.y < regularize_alpha) {
        alpha.y = clamp(2 * alpha.y, 0.25 * regularize_alpha, regularize_alpha);
    }
    return alpha;
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
    const float Ix2 = sqr(Ix), Iz2 = sqr(Iz);
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

// "Stratified Sampling of Spherical Triangles" https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
// Based on https://www.shadertoy.com/view/4tGGzd
float SampleSphericalTriangle(const vec3 P, const vec3 p1, const vec3 p2, const vec3 p3, const vec2 Xi, out vec3 out_dir) {
    // setup spherical triangle
    const vec3 A = normalize(p1 - P), B = normalize(p2 - P), C = normalize(p3 - P);

    // calculate internal angles of spherical triangle: alpha, beta and gamma
    const vec3 BA = orthogonalize(A, B - A);
    const vec3 CA = orthogonalize(A, C - A);
    const vec3 AB = orthogonalize(B, A - B);
    const vec3 CB = orthogonalize(B, C - B);
    const vec3 BC = orthogonalize(C, B - C);
    const vec3 AC = orthogonalize(C, A - C);

    const float alpha = angle_between(BA, CA);
    const float beta = angle_between(AB, CB);
    const float gamma = angle_between(BC, AC);

    const float area = alpha + beta + gamma - PI;
    if (area <= SPHERICAL_AREA_THRESHOLD) {
        return 0.0;
    }

    // calculate arc lengths for edges of spherical triangle
    const float b = portable_acosf(clamp(dot(C, A), -1.0, 1.0));
    const float c = portable_acosf(clamp(dot(A, B), -1.0, 1.0));

    // Use one random variable to select the new area
    const float area_S = Xi.x * area;

    // Save the sine and cosine of the angle delta
    const float p = sin(area_S - alpha);
    const float q = cos(area_S - alpha);

    // Compute the pair(u; v) that determines sin(beta_s) and cos(beta_s)
    const float u = q - cos(alpha);
    const float v = p + sin(alpha) * cos(c);

    // Compute the s coordinate as normalized arc length from A to C_s
    const float denom = (v * p + u * q) * sin(alpha);
    const float a1 = ((v * q - u * p) * cos(alpha) - v) / denom;
    const float s = (1.0 / b) * portable_acosf(clamp(a1, -1.0, 1.0));

    // Compute the third vertex of the sub - triangle
    const vec3 C_s = slerp(A, C, s);

    // Compute the t coordinate using C_s and Xi[1]
    const float b0 = dot(C_s, B);
    const float denom2 = portable_acosf(clamp(b0, -1.0, 1.0));
    const float c0 = 1.0 - Xi.y * (1.0 - dot(C_s, B));
    const float t = portable_acosf(clamp(c0, -1.0, 1.0)) / denom2;

    // Construct the corresponding point on the sphere.
    out_dir = slerp(B, C_s, t);

    // return pdf
    return (1.0 / area);
}

// "An Area-Preserving Parametrization for Spherical Rectangles"
// https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
// NOTE: no precomputation is done, everything is calculated in-place
float SampleSphericalRectangle(const vec3 P, const vec3 light_pos, const vec3 axis_u,
                               const vec3 axis_v, const vec2 Xi, out vec3 out_p) {
    const vec3 corner = light_pos - 0.5 * axis_u - 0.5 * axis_v;

    float axisu_len, axisv_len;
    const vec3 x = normalize_len(axis_u, axisu_len), y = normalize_len(axis_v, axisv_len);
    vec3 z = cross(x, y);

    // compute rectangle coords in local reference system
    const vec3 dir = corner - P;
    float z0 = dot(dir, z);
    // flip z to make it point against Q
    if (z0 > 0.0) {
        z = -z;
        z0 = -z0;
    }
    const float x0 = dot(dir, x);
    const float y0 = dot(dir, y);
    const float x1 = x0 + axisu_len;
    const float y1 = y0 + axisv_len;
    // compute internal angles (gamma_i)
    const vec4 diff = vec4(x0, y1, x1, y0) - vec4(x1, y0, x0, y1);
    vec4 nz = vec4(y0, x1, y1, x0) * diff;
    nz = nz / sqrt(z0 * z0 * diff * diff + nz * nz);
    const float g0 = portable_acosf(clamp(-nz.x * nz.y, -1.0, 1.0));
    const float g1 = portable_acosf(clamp(-nz.y * nz.z, -1.0, 1.0));
    const float g2 = portable_acosf(clamp(-nz.z * nz.w, -1.0, 1.0));
    const float g3 = portable_acosf(clamp(-nz.w * nz.x, -1.0, 1.0));
    // compute predefined constants
    const float b0 = nz.x;
    const float b1 = nz.z;
    const float b0sq = b0 * b0;
    const float k = 2 * PI - g2 - g3;
    // compute solid angle from internal angles
    const float area = g0 + g1 - k;
    if (area <= SPHERICAL_AREA_THRESHOLD) {
        return 0.0f;
    }

    // compute cu
    const float au = Xi.x * area + k;
    const float fu = (cos(au) * b0 - b1) / sin(au);
    float cu = 1.0 / sqrt(fu * fu + b0sq) * (fu > 0.0 ? 1.0 : -1.0);
    cu = clamp(cu, -1.0, 1.0);
    // compute xu
    float xu = -(cu * z0) / max(sqrt(1.0 - cu * cu), 1e-7);
    xu = clamp(xu, x0, x1);
    // compute yv
    const float z0sq = z0 * z0;
    const float y0sq = y0 * y0;
    const float y1sq = y1 * y1;
    const float d = sqrt(xu * xu + z0sq);
    const float h0 = y0 / sqrt(d * d + y0sq);
    const float h1 = y1 / sqrt(d * d + y1sq);
    const float hv = h0 + Xi.y * (h1 - h0), hv2 = hv * hv;
    const float yv = (hv2 < 1.0 - 1e-6) ? (hv * d) / sqrt(1.0 - hv2) : y1;

    // transform (xu, yv, z0) to world coords
    out_p = P + xu * x + yv * y + z0 * z;

    return (1.0 / area);
}

struct lobe_weights_t {
    float diffuse, specular, clearcoat, refraction;
};

lobe_weights_t get_lobe_weights(const float base_color_lum, const float spec_color_lum, const float specular,
                                const float metallic, const float transmission, const float clearcoat) {
    lobe_weights_t weights;

    // taken from Cycles
    weights.diffuse = base_color_lum * (1.0 - metallic) * (1.0 - transmission);
    const float final_transmission = transmission * (1.0 - metallic);
    weights.specular =
        (specular != 0.0 || metallic != 0.0) ? spec_color_lum * (1.0 - final_transmission) : 0.0;
    weights.clearcoat = 0.25 * clearcoat * (1.0 - metallic);
    weights.refraction = final_transmission * base_color_lum;

    const float total_weight = weights.diffuse + weights.specular + weights.clearcoat + weights.refraction;
    if (total_weight != 0.0) {
        weights.diffuse /= total_weight;
        weights.specular /= total_weight;
        weights.clearcoat /= total_weight;
        weights.refraction /= total_weight;
    }

    return weights;
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

    return vec3(abs(p[0]) < Origin ? fma(FloatScale, n[0], p[0]) : p_i[0],
                abs(p[1]) < Origin ? fma(FloatScale, n[1], p[1]) : p_i[1],
                abs(p[2]) < Origin ? fma(FloatScale, n[2], p[2]) : p_i[2]);
}

// http://jcgt.org/published/0007/04/01/paper.pdf
vec3 SampleVNDF_Hemisphere_CrossSect(const vec3 Vh, float U1, float U2) {
    // orthonormal basis (with special case if cross product is zero)
    const float lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
    const vec3 T1 =
        lensq > 0.0 ? vec3(-Vh[1], Vh[0], 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    const vec3 T2 = cross(Vh, T1);
    // parameterization of the projected area
    const float r = sqrt(U1);
    const float phi = 2.0 * PI * U2;
    const float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    const float s = 0.5 * (1.0 + Vh[2]);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    // reprojection onto hemisphere
    const vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    // normalization will be done later
    return Nh;
}

// https://arxiv.org/pdf/2306.05044.pdf
vec3 SampleVNDF_Hemisphere_SphCap(const vec3 Vh, const vec2 rand) {
    // sample a spherical cap in (-Vh.z, 1]
    const float phi = 2.0f * PI * rand.x;
    const float z = fma(1.0 - rand.y, 1.0 + Vh.z, -Vh.z);
    const float sin_theta = sqrt(saturate(1.0 - z * z));
    const float x = sin_theta * cos(phi);
    const float y = sin_theta * sin(phi);
    const vec3 c = vec3(x, y, z);
    // normalization will be done later
    return c + Vh;
}

// https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
vec3 SampleVNDF_Hemisphere_SphCap_Bounded(const vec3 Ve, const vec3 Vh, const vec2 alpha, const vec2 rand) {
    // sample a spherical cap in (-Vh.z, 1]
    const float phi = 2.0 * PI * rand.x;
    const float a = saturate(min(alpha.x, alpha.y));
    const float s = 1.0 + length(Ve.xy);
    const float a2 = a * a, s2 = s * s;
    const float k = (1.0 - a2) * s2 / (s2 + a2 * Ve.z * Ve.z);
    const float b = (Ve.z > 0.0) ? k * Vh.z : Vh.z;
    const float z = fma(1.0 - rand.y, 1.0f + b, -b);
    const float sin_theta = sqrt(saturate(1.0 - z * z));
    const float x = sin_theta * cos(phi);
    const float y = sin_theta * sin(phi);
    const vec3 c = vec3(x, y, z);
    // normalization will be done later
    return c + Vh;
}

vec3 SampleGGX_VNDF(const vec3 Ve, const vec2 alpha, const vec2 rand) {
    // transforming the view direction to the hemisphere configuration
    const vec3 Vh = normalize(vec3(alpha.x * Ve[0], alpha.y * Ve[1], Ve[2]));
    // sample the hemisphere
    const vec3 Nh = SampleVNDF_Hemisphere_SphCap(Vh, rand);
    // transforming the normal back to the ellipsoid configuration
    const vec3 Ne = normalize(vec3(alpha.x * Nh[0], alpha.y * Nh[1], max(0.0, Nh[2])));
    return Ne;
}

vec3 SampleGGX_VNDF_Bounded(const vec3 Ve, vec2 alpha, vec2 rand) {
    // transforming the view direction to the hemisphere configuration
    const vec3 Vh = normalize(vec3(alpha.x * Ve[0], alpha.y * Ve[1], Ve.z));
    // sample the hemisphere
    const vec3 Nh = SampleVNDF_Hemisphere_SphCap_Bounded(Ve, Vh, alpha, rand);
    // transforming the normal back to the ellipsoid configuration
    const vec3 Ne = normalize(vec3(alpha.x * Nh[0], alpha.y * Nh[1], max(0.0, Nh[2])));
    return Ne;
}

float GGX_VNDF_Reflection_Bounded_PDF(const float D, const vec3 view_dir_ts, const vec2 alpha) {
    const vec2 ai = alpha * view_dir_ts.xy;
    const float len2 = dot(ai, ai);
    const float t = sqrt(len2 + view_dir_ts.z * view_dir_ts.z);
    if (view_dir_ts.z >= 0.0) {
        const float a = saturate(min(alpha.x, alpha.y));
        const float s = 1.0 + length(view_dir_ts.xy);
        const float a2 = a * a, s2 = s * s;
        const float k = (1.0 - a2) * s2 / (s2 + a2 * view_dir_ts.z * view_dir_ts.z);
        return D / (2.0 * (k * view_dir_ts.z + t));
    }
    return D * (t - view_dir_ts.z) / (2.0 * len2);
}

// Smith shadowing function
float G1(vec3 Ve, vec2 alpha) {
    alpha *= alpha;
    const float delta =
        (-1.0 + sqrt(1.0 + (alpha.x * Ve[0] * Ve[0] + alpha.y * Ve[1] * Ve[1]) / (Ve[2] * Ve[2]))) / 2.0;
    return 1.0 / (1.0 + delta);
}

float D_GTR1(float NDotH, float a) {
    if (a >= 1.0) {
        return 1.0 / PI;
    }
    const float a2 = sqr(a);
    const float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

float D_GTR2(const float N_dot_H, const float a) {
    const float a2 = sqr(a);
    const float t = 1.0 + (a2 - 1.0) * N_dot_H * N_dot_H;
    return a2 / (PI * t * t);
}

float D_GGX(const vec3 H, const vec2 alpha) {
    if (H[2] == 0.0) {
        return 0.0;
    }
    const float sx = -H[0] / (H[2] * alpha.x);
    const float sy = -H[1] / (H[2] * alpha.y);
    const float s1 = 1.0 + sx * sx + sy * sy;
    const float cos_theta_h4 = H[2] * H[2] * H[2] * H[2];
    return 1.0 / ((s1 * s1) * PI * alpha.x * alpha.y * cos_theta_h4);
}

vec3 world_from_tangent(vec3 T, vec3 B, vec3 N, vec3 V) {
    return V[0] * T + V[1] * B + V[2] * N;
}

vec3 tangent_from_world(vec3 T, vec3 B, vec3 N, vec3 V) {
    return vec3(dot(V, T), dot(V, B), dot(V, N));
}

float exchange(inout float old_value, float new_value) {
    float temp = old_value;
    old_value = new_value;
    return temp;
}

bool exchange(inout bool old_value, bool new_value) {
    bool temp = old_value;
    old_value = new_value;
    return temp;
}

void push_ior_stack(inout float stack[4], const float val) {
    if (stack[0] < 0.0) {
        stack[0] = val;
        return;
    }
    if (stack[1] < 0.0) {
        stack[1] = val;
        return;
    }
    if (stack[2] < 0.0) {
        stack[2] = val;
        return;
    }
    // replace the last value regardless of sign
    stack[3] = val;
}

float pop_ior_stack(inout float stack[4], float default_value) {
    if (stack[3] > 0.0) {
        return exchange(stack[3], -1.0);
    }
    if (stack[2] > 0.0) {
        return exchange(stack[2], -1.0);
    }
    if (stack[1] > 0.0) {
        return exchange(stack[1], -1.0);
    }
    if (stack[0] > 0.0) {
        return exchange(stack[0], -1.0);
    }
    return default_value;
}

float peek_ior_stack(float stack[4], bool skip_first, float default_value) {
    if (stack[3] > 0.0) {
        if (!exchange(skip_first, false)) {
            return stack[3];
        }
    }
    if (stack[2] > 0.0) {
        if (!exchange(skip_first, false)) {
            return stack[2];
        }
    }
    if (stack[1] > 0.0) {
        if (!exchange(skip_first, false)) {
            return stack[1];
        }
    }
    if (stack[0] > 0.0) {
        if (!exchange(skip_first, false)) {
            return stack[0];
        }
    }
    return default_value;
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
                             vec3 base_color, vec2 rand, out vec3 out_V) {
    const float phi = 2 * PI * rand.y;
    const float cos_phi = cos(phi), sin_phi = sin(phi);

    const float dir = sqrt(1.0 - rand.x * rand.x);
    vec3 V = vec3(dir * cos_phi, dir * sin_phi, rand.x); // in tangent-space

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
                                   vec3 sheen_color, bool uniform_sampling, const vec2 rand,
                                   out vec3 out_V) {
    const float phi = 2 * PI * rand.y;
    const float cos_phi = cos(phi), sin_phi = sin(phi);

    vec3 V;
    if (uniform_sampling) {
        const float dir = sqrt(1.0 - rand.x * rand.x);
        V = vec3(dir * cos_phi, dir * sin_phi, rand.x); // in tangent-space
    } else {
        const float dir = sqrt(rand.x);
        const float k = sqrt(1.0 - rand.x);
        V = vec3(dir * cos_phi, dir * sin_phi, k); // in tangent-space
    }

    out_V = world_from_tangent(T, B, N, V);
    return Evaluate_PrincipledDiffuse_BSDF(-I, N, out_V, roughness, base_color, sheen_color, uniform_sampling);
}

vec4 Evaluate_GGXSpecular_BSDF(const vec3 view_dir_ts, const vec3 sampled_normal_ts,
                               const vec3 reflected_dir_ts, const vec2 alpha, const float spec_ior,
                               const float spec_F0, const vec3 spec_col, const vec3 spec_col_90) {
    const float D = D_GGX(sampled_normal_ts, alpha);
    const float G = G1(view_dir_ts, alpha) * G1(reflected_dir_ts, alpha);

    const float FH =
        (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0 - spec_F0);
    vec3 F = mix(spec_col, spec_col_90, FH);

    const float denom = 4.0 * abs(view_dir_ts[2] * reflected_dir_ts[2]);
    F *= (denom != 0.0) ? (D * G / denom) : 0.0;
    F *= max(reflected_dir_ts[2], 0.0);

    const float pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, alpha);
    return vec4(F, pdf);
}

vec4 Sample_GGXSpecular_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, const vec2 alpha, const float spec_ior,
                             const float spec_F0, const vec3 spec_col, const vec3 spec_col_90,
                             const vec2 rand, out vec3 out_V) {
    [[dont_flatten]] if (alpha.x * alpha.y < 1e-7) {
        const vec3 V = reflect(I, N);
        const float FH = (fresnel_dielectric_cos(dot(V, N), spec_ior) - spec_F0) / (1.0 - spec_F0);
        vec3 F = mix(spec_col, spec_col_90, FH);
        out_V = V;
        return vec4(F[0] * 1e6f, F[1] * 1e6f, F[2] * 1e6f, 1e6f);
    }

    const vec3 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    const vec3 sampled_normal_ts = SampleGGX_VNDF_Bounded(view_dir_ts, alpha, rand);

    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const vec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);
    return Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha, spec_ior,
                                     spec_F0, spec_col, spec_col_90);
}

vec4 Evaluate_PrincipledClearcoat_BSDF(vec3 view_dir_ts, vec3 sampled_normal_ts, vec3 reflected_dir_ts,
                                       float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0) {
    const float D = D_GTR1(sampled_normal_ts[2], clearcoat_roughness2);
    // Always assume roughness of 0.25 for clearcoat
    const vec2 clearcoat_alpha = vec2(0.25 * 0.25, 0.25 * 0.25);

    const float G = G1(view_dir_ts, clearcoat_alpha) * G1(reflected_dir_ts, clearcoat_alpha);

    const float FH = (fresnel_dielectric_cos(dot(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
                     (1.0 - clearcoat_F0);
    float F = mix(0.04, 1.0, FH);

    const float denom = 4.0 * abs(view_dir_ts[2]) * abs(reflected_dir_ts[2]);
    F *= (denom != 0.0) ? D * G / denom : 0.0;
    F *= saturate(reflected_dir_ts[2]);

    const float pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, clearcoat_alpha);
    return vec4(F, F, F, pdf);
}

vec4 Sample_PrincipledClearcoat_BSDF(vec3 T, vec3 B, vec3 N, vec3 I, float clearcoat_roughness2,
                                     float clearcoat_ior, float clearcoat_F0, const vec2 rand,
                                     out vec3 out_V) {
    [[dont_flatten]] if (sqr(clearcoat_roughness2) < 1e-7) {
        const vec3 V = reflect(I, N);

        const float FH = (fresnel_dielectric_cos(dot(V, N), clearcoat_ior) - clearcoat_F0) / (1.0 - clearcoat_F0);
        const float F = mix(0.04, 1.0, FH);

        out_V = V;
        return vec4(F * 1e6f, F * 1e6f, F * 1e6f, 1e6f);
    }

    const vec3 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    // NOTE: GTR1 distribution is not used for sampling because Cycles does it this way (???!)
    const vec3 sampled_normal_ts = SampleGGX_VNDF_Bounded(view_dir_ts, vec2(clearcoat_roughness2), rand);

    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const vec3 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);

    return Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, clearcoat_roughness2,
                                             clearcoat_ior, clearcoat_F0);
}

vec4 Evaluate_GGXRefraction_BSDF(const vec3 view_dir_ts, const vec3 sampled_normal_ts,
                                 const vec3 refr_dir_ts, const vec2 alpha, const float eta,
                                 const vec3 refr_col) {
    if (refr_dir_ts[2] >= 0.0 || view_dir_ts[2] <= 0.0) {
        return vec4(0.0);
    }

    const float D = D_GGX(sampled_normal_ts, alpha);
    const float G1o = G1(refr_dir_ts, alpha), G1i = G1(view_dir_ts, alpha);

    const float denom = dot(refr_dir_ts, sampled_normal_ts) + dot(view_dir_ts, sampled_normal_ts) * eta;
    const float jacobian = saturate(-dot(refr_dir_ts, sampled_normal_ts)) / (denom * denom);

    const float F = D * G1i * G1o * saturate(dot(view_dir_ts, sampled_normal_ts)) * jacobian /
              (/*-refr_dir_ts[2] */ view_dir_ts[2]);

    const float pdf = D * G1o * saturate(dot(view_dir_ts, sampled_normal_ts)) * jacobian / view_dir_ts[2];

    return vec4(F * refr_col, pdf);
}

vec4 Sample_GGXRefraction_BSDF(const vec3 T, const vec3 B, const vec3 N, const vec3 I, const vec2 alpha,
                               const float eta, const vec3 refr_col, const vec2 rand, out vec4 out_V) {
    [[dont_flatten]] if (alpha.x * alpha.y < 1e-7) {
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
    const vec3 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha, rand);

    const float cosi = dot(view_dir_ts, sampled_normal_ts);
    const float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (cost2 < 0) {
        return vec4(0.0);
    }
    const float m = eta * cosi - sqrt(cost2);
    const vec3 refr_dir_ts = normalize(-eta * view_dir_ts + m * sampled_normal_ts);

    const vec4 F =
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, alpha, eta, refr_col);

    const vec3 V = world_from_tangent(T, B, N, refr_dir_ts);
    out_V = vec4(V[0], V[1], V[2], m);
    return F;
}

struct light_sample_t {
    vec3 col, L, lp;
    float area, dist_mul, pdf;
    uint ray_flags;
    bool cast_shadow, from_env;
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

vec3 map_to_cone(float r1, float r2, vec3 N, float radius) {
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

float sphere_intersection(const vec3 center, const float radius, const vec3 ro, const vec3 rd) {
    const vec3 oc = ro - center;
    const float a = dot(rd, rd);
    const float b = 2 * dot(oc, rd);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - 4 * a * c;
    return (-b - sqrt(max(discriminant, 0.0f))) / (2 * a);
}

void SampleLightSource(vec3 P, vec3 T, vec3 B, vec3 N, const float rand_pick_light, const vec2 rand_light_uv,
                       const vec2 rand_tex_uv, inout light_sample_t ls) {
    float u1 = rand_pick_light;

#if USE_HIERARCHICAL_NEE
    float factor = 1.0;
    uint cur = 0; // start from root
    while ((cur & LEAF_NODE_BIT) == 0) {
        light_wbvh_node_t n = g_light_wnodes[cur];

        float importance[8];
        const float total_importance = calc_lnode_importance(n, P, importance);
        [[dont_flatten]] if (total_importance == 0.0) {
            // failed to find lightsource for sampling
            return;
        }

        // normalize
        [[unroll]] for (int j = 0; j < 8; ++j) {
            importance[j] /= total_importance;
        }

        float importance_cdf[9];
        importance_cdf[0] = 0.0;
        [[unroll]] for (int j = 0; j < 8; ++j) {
            importance_cdf[j + 1] = importance_cdf[j] + importance[j];
        }
        // make sure cdf ends with 1.0
        [[unroll]] for (int j = 0; j < 8; ++j) {
            [[flatten]] if (importance_cdf[j + 1] == importance_cdf[8]) {
                importance_cdf[j + 1] = 1.01;
            }
        }

        int next = 0;
        [[unroll]] for (int j = 1; j < 9; ++j) {
            if (importance_cdf[j] <= u1) {
                ++next;
            }
        }

        u1 = fract((u1 - importance_cdf[next]) / importance[next]);
        cur = n.child[next];
        factor *= importance[next];
    }
    const uint light_index = (cur & PRIM_INDEX_BITS);
    factor = (1.0 / factor);
#else
    uint light_index = min(uint(u1 * g_params.li_count), uint(g_params.li_count - 1));
    u1 = u1 * float(g_params.li_count) - float(light_index);
    light_index = g_li_indices[light_index];
    const float factor = float(g_params.li_count);
#endif

    const light_t l = g_lights[light_index];

    ls.col = uintBitsToFloat(l.type_and_param0.yzw);
    ls.cast_shadow = LIGHT_CAST_SHADOW(l);
    ls.from_env = false;
    ls.ray_flags = 0;

    const uint l_type = LIGHT_TYPE(l);
    [[dont_flatten]] if (l_type == LIGHT_TYPE_SPHERE) {
        const float r1 = rand_light_uv.x, r2 = rand_light_uv.y;

        const vec3 surface_to_center = l.SPH_POS - P;
        float disk_dist;
        const vec3 sampled_dir = normalize_len(map_to_cone(r1, r2, surface_to_center, l.SPH_RADIUS), disk_dist);

        if (l.SPH_RADIUS > 0.0) {
            const float ls_dist = sphere_intersection(l.SPH_POS, l.SPH_RADIUS, P, sampled_dir);

            const vec3 light_surf_pos = P + sampled_dir * ls_dist;
            const vec3 light_forward = normalize(light_surf_pos - l.SPH_POS);

            ls.lp = offset_ray(light_surf_pos, light_forward);
            ls.pdf = (disk_dist * disk_dist) / (PI * l.SPH_RADIUS * l.SPH_RADIUS);
        } else {
            ls.lp = l.SPH_POS;
            ls.pdf = (disk_dist * disk_dist) / PI;
        }

        ls.L = sampled_dir;
        ls.area = l.SPH_AREA;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        if (!LIGHT_VISIBLE(l)) {
            ls.area = 0.0;
        }

        [[dont_flatten]] if (l.SPH_SPOT > 0.0) {
            const float _dot = -dot(ls.L, l.SPH_DIR);
            if (_dot > 0.0) {
                const float _angle = acos(saturate(_dot));
                ls.col *= saturate((l.SPH_SPOT - _angle) / l.SPH_BLEND);
            } else {
                ls.col *= 0.0;
            }
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_DIR) {
        ls.L = l.DIR_DIR;
        ls.area = 0.0;
        ls.pdf = 1.0;
        ls.dist_mul = MAX_DIST;
        [[dont_flatten]] if (l.DIR_ANGLE != 0.0) {
            const float r1 = rand_light_uv.x, r2 = rand_light_uv.y;

            const float radius = tan(l.DIR_ANGLE);
            ls.L = normalize(map_to_cone(r1, r2, ls.L, radius));
            ls.area = PI * radius * radius;

            const float cos_theta = dot(ls.L, l.DIR_DIR);
            ls.pdf = 1.0 / (ls.area * cos_theta);
        }
        ls.lp = P + ls.L;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        if (!LIGHT_VISIBLE(l)) {
            ls.area = 0.0;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_RECT) {
        const vec3 light_pos = l.RECT_POS;
        const vec3 light_u = l.RECT_U, light_v = l.RECT_V;
        const vec3 light_forward = normalize(cross(light_u, light_v));

        vec3 lp;
        float pdf;

#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
        pdf = SampleSphericalRectangle(P, light_pos, light_u, light_v, rand_light_uv, lp);
        if (pdf <= 0.0)
#endif
        {
            const float r1 = rand_light_uv.x - 0.5, r2 = rand_light_uv.y - 0.5;
            lp = light_pos + light_u * r1 + light_v * r2;
        }

        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        const float cos_theta = dot(-ls.L, light_forward);
        if (cos_theta > 0.0) {
            ls.lp = offset_ray(lp, light_forward);
            ls.pdf = (pdf > 0.0) ? pdf : (ls_dist * ls_dist) / (ls.area * cos_theta);
            ls.area = l.RECT_AREA;
            if (!LIGHT_VISIBLE(l)) {
                ls.area = 0.0;
            }
            [[dont_flatten]] if (LIGHT_SKY_PORTAL(l)) {
                vec3 env_col = g_params.env_col.xyz;
                const uint env_map = floatBitsToUint(g_params.env_col.w);
                if (env_map != 0xffffffff) {
#if BINDLESS
                    env_col *= SampleLatlong_RGBE(env_map, ivec2(g_params.env_map_res >> 16u, g_params.env_map_res & 0xffff), ls.L, g_params.env_rotation, rand_tex_uv);
#else
                    env_col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation, rand_tex_uv);
#endif
                }
                ls.col *= env_col;
                ls.from_env = true;
            }
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_DISK) {
        const vec3 light_pos = l.DISK_POS;
        const vec3 light_u = l.DISK_U;
        const vec3 light_v = l.DISK_V;

        vec2 offset = 2.0 * rand_light_uv - vec2(1.0);
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
        const vec3 light_forward = normalize(cross(light_u, light_v));

        ls.lp = offset_ray(lp, light_forward);
        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);
        ls.area = l.DISK_AREA;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        const float cos_theta = dot(-ls.L, light_forward);
        [[flatten]] if (cos_theta > 0.0) {
            ls.pdf = (ls_dist * ls_dist) / (ls.area * cos_theta);
        }

        if (!LIGHT_VISIBLE(l)) {
            ls.area = 0.0;
        }

        [[dont_flatten]] if (LIGHT_SKY_PORTAL(l)) {
            vec3 env_col = g_params.env_col.xyz;
            const uint env_map = floatBitsToUint(g_params.env_col.w);
            if (env_map != 0xffffffff) {
#if BINDLESS
                env_col *= SampleLatlong_RGBE(env_map, ivec2(g_params.env_map_res >> 16u, g_params.env_map_res & 0xffff), ls.L, g_params.env_rotation, rand_tex_uv);
#else
                env_col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation, rand_tex_uv);
#endif
            }
            ls.col *= env_col;
            ls.from_env = true;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_LINE) {
        const vec3 light_pos = l.LINE_POS;
        const vec3 light_dir = l.LINE_V;

        const float r1 = rand_light_uv.x, r2 = rand_light_uv.y;

        const vec3 center_to_surface = P - light_pos;

        vec3 light_u = normalize(cross(center_to_surface, light_dir));
        vec3 light_v = cross(light_u, light_dir);

        const float phi = PI * r1;
        const vec3 normal = cos(phi) * light_u + sin(phi) * light_v;

        const vec3 lp = light_pos + normal * l.LINE_RADIUS + (r2 - 0.5) * light_dir * l.LINE_HEIGHT;

        ls.lp = lp;
        float ls_dist;
        ls.L = normalize_len(lp - P, ls_dist);

        ls.area = l.LINE_AREA;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        const float cos_theta = 1.0 - abs(dot(ls.L, light_dir));
        [[flatten]] if (cos_theta != 0.0) {
            ls.pdf = (ls_dist * ls_dist) / (ls.area * cos_theta);
        }

        if (!LIGHT_VISIBLE(l)) {
            ls.area = 0.0;
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_TRI) {
        const uint ltri_index = floatBitsToUint(l.TRI_TRI_INDEX);
        const mesh_instance_t lmi = g_mesh_instances[floatBitsToUint(l.TRI_MI_INDEX)];

        const vertex_t v1 = g_vertices[g_vtx_indices[ltri_index * 3 + 0]];
        const vertex_t v2 = g_vertices[g_vtx_indices[ltri_index * 3 + 1]];
        const vertex_t v3 = g_vertices[g_vtx_indices[ltri_index * 3 + 2]];

        const vec3 p1 = (lmi.xform * vec4(v1.p[0], v1.p[1], v1.p[2], 1.0)).xyz,
                   p2 = (lmi.xform * vec4(v2.p[0], v2.p[1], v2.p[2], 1.0)).xyz,
                   p3 = (lmi.xform * vec4(v3.p[0], v3.p[1], v3.p[2], 1.0)).xyz;
        const vec2 uv1 = vec2(v1.t[0], v1.t[1]),
                   uv2 = vec2(v2.t[0], v2.t[1]),
                   uv3 = vec2(v3.t[0], v3.t[1]);

        const vec3 e1 = p2 - p1, e2 = p3 - p1;
        float light_fwd_len;
        vec3 light_forward = normalize_len(cross(e1, e2), light_fwd_len);
        ls.area = 0.5 * light_fwd_len;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);

        vec3 lp;
        vec2 luvs;
        float pdf;

#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
        // Spherical triangle sampling
        pdf = SampleSphericalTriangle(P, p1, p2, p3, rand_light_uv, ls.L);
        if (pdf > 0.0) {
            // find u, v of intersection point
            const vec3 pvec = cross(ls.L, e2);
            const vec3 tvec = P - p1, qvec = cross(tvec, e1);

            const float inv_det = 1.0 / dot(e1, pvec);
            const float tri_u = dot(tvec, pvec) * inv_det, tri_v = dot(ls.L, qvec) * inv_det;

            lp = (1.0 - tri_u - tri_v) * p1 + tri_u * p2 + tri_v * p3;
            luvs = (1.0 - tri_u - tri_v) * uv1 + tri_u * uv2 + tri_v * uv3;
        } else
#endif
        {
            // Flat triangle sampling
            const float r1 = sqrt(rand_light_uv.x), r2 = rand_light_uv.y;

            luvs = uv1 * (1.0 - r1) + r1 * (uv2 * (1.0 - r2) + uv3 * r2);
            lp = p1 * (1.0 - r1) + r1 * (p2 * (1.0 - r2) + p3 * r2);

            float ls_dist;
            ls.L = normalize_len(lp - P, ls_dist);

            const float cos_theta = -dot(ls.L, light_forward);
            pdf = (ls_dist * ls_dist) / (ls.area * cos_theta);
        }

        float cos_theta = -dot(ls.L, light_forward);
        ls.lp = offset_ray(lp, cos_theta >= 0.0 ? light_forward : -light_forward);
        if (LIGHT_DOUBLE_SIDED(l)) { // doublesided
            cos_theta = abs(cos_theta);
        }
        [[dont_flatten]] if (cos_theta > 0.0) {
            ls.pdf = pdf;
            const uint tex_index = floatBitsToUint(l.TRI_TEX_INDEX);
            if (tex_index != 0xffffffff) {
                ls.col *= SampleBilinear(tex_index, luvs, 0 /* lod */, rand_tex_uv, true /* YCoCg */, true /* SRGB */).xyz;
            }
        }
    } else [[dont_flatten]] if (l_type == LIGHT_TYPE_ENV) {
        const float rx = rand_light_uv.x, ry = rand_light_uv.y;

        vec4 dir_and_pdf;
        if (g_params.env_qtree_levels > 0) {
            // Sample environment using quadtree
            dir_and_pdf = Sample_EnvQTree(g_params.env_rotation, g_env_qtree, g_params.env_qtree_levels, u1, rx, ry);
        } else {
            // Sample environment as hemishpere
            const float phi = 2 * PI * ry;
            const float cos_phi = cos(phi), sin_phi = sin(phi);

            const float dir = sqrt(1.0 - rx * rx);
            vec3 V = vec3(dir * cos_phi, dir * sin_phi, rx); // in tangent-space

            dir_and_pdf.xyz = world_from_tangent(T, B, N, V);
            dir_and_pdf.w = 0.5 / PI;
        }

        ls.L = dir_and_pdf.xyz;
        ls.col *= g_params.env_col.xyz;

        const uint env_map = floatBitsToUint(g_params.env_col.w);
        if (env_map != 0xffffffff) {
#if BINDLESS
            ls.col *= SampleLatlong_RGBE(env_map, ivec2(g_params.env_map_res >> 16u, g_params.env_map_res & 0xffff), ls.L, g_params.env_rotation, rand_tex_uv);
#else
            ls.col *= SampleLatlong_RGBE(g_textures[env_map], ls.L, g_params.env_rotation, rand_tex_uv);
#endif
        }

        ls.area = 1.0;
        ls.lp = P + ls.L;
        ls.dist_mul = MAX_DIST;
        ls.pdf = dir_and_pdf.w;
        ls.from_env = true;
        ls.ray_flags = LIGHT_RAY_VISIBILITY(l);
    }

    ls.pdf /= factor;
}

shared uint g_stack[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];
shared float g_stack_factors[LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y][MAX_STACK_SIZE];

float EvalTriLightFactor(const vec3 P, const vec3 ro, uint tri_index) {
    uint stack_size = 0;
    g_stack_factors[gl_LocalInvocationIndex][stack_size] = 1.0;
    g_stack[gl_LocalInvocationIndex][stack_size++] = 0;

    while (stack_size != 0) {
        const uint cur = g_stack[gl_LocalInvocationIndex][--stack_size];
        const float cur_factor = g_stack_factors[gl_LocalInvocationIndex][stack_size];

        if ((cur & LEAF_NODE_BIT) == 0) {
            light_wbvh_node_t n = g_light_wnodes[cur];

            float importance[8];
            const float total_importance = calc_lnode_importance(n, ro, importance);

            for (int j = 0; j < 8; ++j) {
                if (importance[j] > 0.0 && _bbox_test(P, vec3(n.bbox_min[0][j], n.bbox_min[1][j], n.bbox_min[2][j]),
                                                         vec3(n.bbox_max[0][j], n.bbox_max[1][j], n.bbox_max[2][j]))) {
                    g_stack_factors[gl_LocalInvocationIndex][stack_size] = cur_factor * importance[j] / total_importance;
                    g_stack[gl_LocalInvocationIndex][stack_size++] = n.child[j];
                }
            }
        } else {
            const int light_index = int(cur & PRIM_INDEX_BITS);

            light_t l = g_lights[light_index];
            if (LIGHT_TYPE(l) == LIGHT_TYPE_TRI && floatBitsToUint(l.TRI_TRI_INDEX) == tri_index) {
                // needed triangle found
                return 1.0 / cur_factor;
            }
        }
    }

    return 1.0;
}

vec3 Evaluate_EnvColor(const ray_data_t ray, const float pdf_factor, const vec2 tex_rand) {
    const vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);
#if PRIMARY
    vec3 env_col = g_params.back_col.xyz;
    const uint env_map = floatBitsToUint(g_params.back_col.w);
    const uint env_map_res = g_params.back_map_res;
    const float env_map_rotation = g_params.back_rotation;
#else
    vec3 env_col = is_indirect(ray.depth) ? g_params.env_col.xyz : g_params.back_col.xyz;
    const uint env_map = is_indirect(ray.depth) ? floatBitsToUint(g_params.env_col.w) : floatBitsToUint(g_params.back_col.w);
    const uint env_map_res = is_indirect(ray.depth) ? g_params.env_map_res : g_params.back_map_res;
    const float env_map_rotation = is_indirect(ray.depth) ? g_params.env_rotation : g_params.back_rotation;
#endif
    if (env_map != 0xffffffff) {
#if BINDLESS
        env_col *= SampleLatlong_RGBE(env_map, ivec2(env_map_res >> 16u, env_map_res & 0xffff), rd, env_map_rotation, tex_rand);
#else
        env_col *= SampleLatlong_RGBE(g_textures[env_map], rd, env_map_rotation, tex_rand);
#endif
    }

#if USE_NEE
    if (g_params.env_light_index != 0xffffffff && pdf_factor >= 0.0 && is_indirect(ray.depth)) {
        if (g_params.env_qtree_levels > 0) {
            const float light_pdf = Evaluate_EnvQTree(env_map_rotation, g_env_qtree, g_params.env_qtree_levels, rd) / pdf_factor;
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            env_col *= mis_weight;
        } else {
            const float light_pdf = 0.5 / (PI * float(g_params.li_count));
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            env_col *= mis_weight;
        }
    }
#endif

    return env_col;
}

vec3 Evaluate_LightColor(const ray_data_t ray, const hit_data_t inter, const vec2 tex_rand) {
    const vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    const vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);

    const vec3 P = ro + inter.t * rd;

    const light_t l = g_lights[-inter.obj_index - 1];
#if USE_HIERARCHICAL_NEE
    const float pdf_factor = (1.0 / inter.u);
#else
    const float pdf_factor = float(g_params.li_count);
#endif

    vec3 lcol = uintBitsToFloat(l.type_and_param0.yzw);
    [[dont_flatten]] if (LIGHT_SKY_PORTAL(l)) {
        vec3 env_col = g_params.env_col.xyz;
        const uint env_map = floatBitsToUint(g_params.env_col.w);
        if (env_map != 0xffffffff) {
#if BINDLESS
            env_col *= SampleLatlong_RGBE(env_map, ivec2(g_params.env_map_res >> 16u, g_params.env_map_res & 0xffff), rd, g_params.env_rotation, tex_rand);
#else
            env_col *= SampleLatlong_RGBE(g_textures[env_map], rd, g_params.env_rotation, tex_rand);
#endif
        }
        lcol *= env_col;
    }

    const uint l_type = LIGHT_TYPE(l);
    if (l_type == LIGHT_TYPE_SPHERE) {
        const vec3 disk_normal = normalize(ro - l.SPH_POS);
        const float disk_dist = dot(ro, disk_normal) - dot(l.SPH_POS, disk_normal);

        const float light_pdf = (disk_dist * disk_dist) / (PI * l.SPH_RADIUS * l.SPH_RADIUS * pdf_factor);
        const float bsdf_pdf = ray.pdf;

        const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
        lcol *= mis_weight;

        [[dont_flatten]] if (l.SPH_SPOT > 0.0 && l.SPH_BLEND > 0.0) {
            const float _dot = -dot(rd, l.SPH_DIR);
            const float _angle = acos(saturate(_dot));
            [[flatten]] if (l.SPH_BLEND > 0.0) {
                lcol *= saturate((l.SPH_SPOT - _angle) / l.SPH_BLEND);
            }
        }
    } else if (l_type == LIGHT_TYPE_DIR) {
        const float radius = tan(l.DIR_ANGLE);
        const float light_area = PI * radius * radius;

        const float cos_theta = dot(rd, l.DIR_DIR);

        const float light_pdf = 1.0 / (light_area * cos_theta * pdf_factor);
        const float bsdf_pdf = ray.pdf;

        const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
        lcol *= mis_weight;
    } else if (l_type == LIGHT_TYPE_RECT) {
        const vec3 light_pos = l.RECT_POS;
        const vec3 light_u = l.RECT_U, light_v = l.RECT_V;

        float light_pdf;
#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
        vec3 _unused;
        light_pdf = SampleSphericalRectangle(ro, light_pos, light_u, light_v, vec2(0.0), _unused) / pdf_factor;
        if (light_pdf == 0.0)
#endif
        {
            const vec3 light_forward = normalize(cross(light_u, light_v));
            const float light_area = l.RECT_AREA;
            const float cos_theta = dot(rd, light_forward);
            light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
        }

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

        const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
        const float bsdf_pdf = ray.pdf;

        const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
        lcol *= mis_weight;
    } else if (l_type == LIGHT_TYPE_LINE) {
        const vec3 light_dir = l.LINE_V;
        const float light_area = l.LINE_AREA;

        const float cos_theta = 1.0 - abs(dot(rd, light_dir));

        const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
        const float bsdf_pdf = ray.pdf;

        const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
        lcol *= mis_weight;
    }
    return lcol;
}

struct surface_t {
    vec3 P, T, B, N, plane_N;
    vec2 uvs;
};

vec3 Evaluate_DiffuseNode(const light_sample_t ls, const ray_data_t ray, const surface_t surf,
                          const vec3 base_color, const float roughness, const float mix_weight,
                          const bool use_mis, inout shadow_ray_t sh_r) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);

    const vec4 diff_col = Evaluate_OrenDiffuse_BSDF(-I, surf.N, ls.L, roughness, base_color);
    const float bsdf_pdf = diff_col[3];

    float mis_weight = 1.0;
    if (use_mis && ls.area > 0.0) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }

    const vec3 lcol = ls.col * diff_col.xyz * (mix_weight * mis_weight / ls.pdf);

    [[dont_flatten]] if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    vec3 new_o = offset_ray(surf.P, surf.plane_N);
    sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
    sh_r.c[0] = lcol[0]; sh_r.c[1] = lcol[1]; sh_r.c[2] = lcol[2];
    sh_r.xy = ray.xy;
    sh_r.depth = ray.depth;

    return vec3(0.0);
}

void Sample_DiffuseNode(const ray_data_t ray, const surface_t surf, const vec3 base_color,
                        const float roughness, const vec2 rand, const float mix_weight, inout ray_data_t new_ray) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);

    vec3 V;
    const vec4 F = Sample_OrenDiffuse_BSDF(surf.T, surf.B, surf.N, I, roughness, base_color, rand, V);

    new_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(1, 0, 0, 0);

    vec3 new_o = offset_ray(surf.P, surf.plane_N);
    new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
    new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

    new_ray.c[0] = F[0] * mix_weight / F[3];
    new_ray.c[1] = F[1] * mix_weight / F[3];
    new_ray.c[2] = F[2] * mix_weight / F[3];
    new_ray.pdf = F[3];
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT;
}

vec3 Evaluate_GlossyNode(const light_sample_t ls, const ray_data_t ray, const surface_t surf,
                         const vec3 base_color, const float roughness, const float regularize_alpha,
                         const float spec_ior, const float spec_F0, const float mix_weight,
                         const bool use_mis, inout shadow_ray_t sh_r) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);
    const vec3 H = normalize(ls.L - I);

    const vec3 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const vec3 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const vec3 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    const vec2 alpha = calc_alpha(roughness, 0.0, regularize_alpha);
    if (alpha.x * alpha.y < 1e-7f) {
        return vec3(0.0);
    }

    const vec4 spec_col = Evaluate_GGXSpecular_BSDF(
        view_dir_ts, sampled_normal_ts, light_dir_ts, alpha, spec_ior, spec_F0, base_color, base_color);
    const float bsdf_pdf = spec_col[3];

    float mis_weight = 1.0;
    if (use_mis && ls.area > 0.0) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    const vec3 lcol = ls.col * spec_col.rgb * (mix_weight * mis_weight / ls.pdf);

    [[dont_flatten]] if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    vec3 new_o = offset_ray(surf.P, surf.plane_N);
    sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
    sh_r.c[0] = lcol[0];
    sh_r.c[1] = lcol[1];
    sh_r.c[2] = lcol[2];
    sh_r.xy = ray.xy;
    sh_r.depth = ray.depth;

    return vec3(0.0);
}

void Sample_GlossyNode(const ray_data_t ray, const surface_t surf, const vec3 base_color,
                       const float roughness, const float regularize_alpha, const float spec_ior,
                       const float spec_F0, const vec2 rand, const float mix_weight,
                       inout ray_data_t new_ray) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);
    const vec2 alpha = calc_alpha(roughness, 0.0, regularize_alpha);

    vec3 V;
    const vec4 F = Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha, spec_ior, spec_F0, base_color, base_color, rand, V);

    new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

    vec3 new_o = offset_ray(surf.P, surf.plane_N);
    new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
    new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

    new_ray.c[0] = F[0] * mix_weight / F[3];
    new_ray.c[1] = F[1] * mix_weight / F[3];
    new_ray.c[2] = F[2] * mix_weight / F[3];
    new_ray.pdf = F[3];
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * min(alpha.x, alpha.y);
}

vec3 Evaluate_RefractiveNode(const light_sample_t ls, const ray_data_t ray, const surface_t surf,
                             const vec3 base_color, const float roughness, const float regularize_alpha,
                             const float eta, const float mix_weight, const bool use_mis, inout shadow_ray_t sh_r) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);
    const vec3 H = normalize(ls.L - I * eta);
    const vec3 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const vec3 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const vec3 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    const vec4 refr_col =
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                    calc_alpha(roughness, 0.0, regularize_alpha), eta, base_color);
    const float bsdf_pdf = refr_col[3];

    float mis_weight = 1.0;
    if (use_mis && ls.area > 0.0) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    const vec3 lcol = ls.col * refr_col.rgb * (mix_weight * mis_weight / ls.pdf);

    [[dont_flatten]] if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    vec3 new_o = offset_ray(surf.P, -surf.plane_N);
    sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
    sh_r.c[0] = lcol[0]; sh_r.c[1] = lcol[1]; sh_r.c[2] = lcol[2];
    sh_r.xy = ray.xy;
    sh_r.depth = ray.depth;

    return vec3(0.0);
}

void Sample_RefractiveNode(const ray_data_t ray, const surface_t surf, const vec3 base_color,
                           const float roughness, const float regularize_alpha, const bool is_backfacing, const float int_ior,
                           const float ext_ior, const vec2 rand, const float mix_weight, inout ray_data_t new_ray) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);
    const vec2 alpha = calc_alpha(roughness, 0.0, regularize_alpha);
    const float eta = is_backfacing ? (int_ior / ext_ior) : (ext_ior / int_ior);

    vec4 _V;
    const vec4 F = Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, I, alpha, eta, base_color, rand, _V);

    const vec3 V = _V.xyz;
    const float m = _V[3];

    new_ray.depth = pack_ray_type(RAY_TYPE_REFR);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 0, 1, 0);

    new_ray.c[0] = F[0] * mix_weight / F[3];
    new_ray.c[1] = F[1] * mix_weight / F[3];
    new_ray.c[2] = F[2] * mix_weight / F[3];
    new_ray.pdf = F[3];

    if (!is_backfacing) {
        // Entering the surface, push new value
        push_ior_stack(new_ray.ior, int_ior);
    } else {
        // Exiting the surface, pop the last ior value
        pop_ior_stack(new_ray.ior, 1.0);
    }

    vec3 new_o = offset_ray(surf.P, -surf.plane_N);
    new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
    new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * min(alpha.x, alpha.y);
}

struct diff_params_t {
    vec3 base_color;
    vec3 sheen_color;
    float roughness;
};

struct spec_params_t {
    vec3 tmp_col;
    float roughness;
    float ior;
    float F0;
    float anisotropy;
};

struct clearcoat_params_t {
    float roughness;
    float ior;
    float F0;
};

struct transmission_params_t {
    float roughness;
    float int_ior;
    float eta;
    float fresnel;
    bool backfacing;
};

vec3 Evaluate_PrincipledNode(const light_sample_t ls, const ray_data_t ray,
                             const surface_t surf, const lobe_weights_t lobe_weights,
                             const diff_params_t diff, const spec_params_t spec,
                             const clearcoat_params_t coat, const transmission_params_t trans,
                             const float metallic, const float transmission, const float N_dot_L,
                             const float mix_weight, const bool use_mis, const float regularize_alpha,
                             inout shadow_ray_t sh_r) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);

    vec3 lcol = vec3(0.0);
    float bsdf_pdf = 0.0;

    [[dont_flatten]] if (lobe_weights.diffuse > 1e-7 && (ls.ray_flags & RAY_TYPE_DIFFUSE_BIT) != 0 && N_dot_L > 0.0) {
        vec4 diff_col = Evaluate_PrincipledDiffuse_BSDF(-I, surf.N, ls.L, diff.roughness, diff.base_color,
                                                        diff.sheen_color, false);
        bsdf_pdf += lobe_weights.diffuse * diff_col[3];
        diff_col *= (1.0 - metallic) * (1.0 - transmission);

        lcol += ls.col * N_dot_L * diff_col.rgb / (PI * ls.pdf);
    }

    vec3 H;
    [[flatten]] if (N_dot_L > 0.0) {
        H = normalize(ls.L - I);
    } else {
        H = normalize(ls.L - I * trans.eta);
    }

    const vec3 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const vec3 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const vec3 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    const vec2 spec_alpha = calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha);
    [[dont_flatten]] if (lobe_weights.specular > 0.0 && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0 && spec_alpha.x * spec_alpha.y >= 1e-7 && N_dot_L > 0.0) {
        const vec4 spec_col = Evaluate_GGXSpecular_BSDF(
            view_dir_ts, sampled_normal_ts, light_dir_ts, spec_alpha, spec.ior, spec.F0, spec.tmp_col, vec3(1.0));
        bsdf_pdf += lobe_weights.specular * spec_col[3];
        lcol += ls.col * spec_col.rgb / ls.pdf;
    }

    const vec2 coat_alpha = calc_alpha(coat.roughness, 0.0, regularize_alpha);
    [[dont_flatten]] if (lobe_weights.clearcoat > 0.0 && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0 && coat_alpha.x * coat_alpha.y >= 1e-7 && N_dot_L > 0.0) {
        const vec4 clearcoat_col = Evaluate_PrincipledClearcoat_BSDF(
            view_dir_ts, sampled_normal_ts, light_dir_ts, coat_alpha.x, coat.ior, coat.F0);
        bsdf_pdf += lobe_weights.clearcoat * clearcoat_col[3];
        lcol += 0.25 * ls.col * clearcoat_col.rgb / ls.pdf;
    }

    [[dont_flatten]] if (lobe_weights.refraction > 0.0) {
        const vec2 refr_spec_alpha = calc_alpha(spec.roughness, 0.0, regularize_alpha);
        [[dont_flatten]] if (trans.fresnel != 0.0 && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0 && refr_spec_alpha.x * refr_spec_alpha.y >= 1e-7 && N_dot_L > 0.0) {
            const vec4 spec_col =
                Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, refr_spec_alpha,
                                          1.0 /* ior */, 0.0 /* F0 */, vec3(1.0), vec3(1.0));
            bsdf_pdf += lobe_weights.refraction * trans.fresnel * spec_col[3];
            lcol += ls.col * spec_col.rgb * (trans.fresnel / ls.pdf);
        }

        const vec2 refr_trans_alpha = calc_alpha(trans.roughness, 0.0, regularize_alpha);
        [[dont_flatten]] if (trans.fresnel != 1.0 && (ls.ray_flags & RAY_TYPE_REFR_BIT) != 0 && refr_trans_alpha.x * refr_trans_alpha.y >= 1e-7 && N_dot_L < 0.0) {
            const vec4 refr_col = Evaluate_GGXRefraction_BSDF(
                view_dir_ts, sampled_normal_ts, light_dir_ts, refr_trans_alpha, trans.eta, diff.base_color);
            bsdf_pdf += lobe_weights.refraction * (1.0 - trans.fresnel) * refr_col[3];
            lcol += ls.col * refr_col.rgb * ((1.0 - trans.fresnel) / ls.pdf);
        }
    }

    float mis_weight = 1.0;
    [[flatten]] if (use_mis && ls.area > 0.0) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    lcol *= mix_weight * mis_weight;

    [[dont_flatten]] if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    vec3 new_o = offset_ray(surf.P, N_dot_L < 0.0 ? -surf.plane_N : surf.plane_N);
    sh_r.o[0] = new_o[0]; sh_r.o[1] = new_o[1]; sh_r.o[2] = new_o[2];
    sh_r.c[0] = lcol[0]; sh_r.c[1] = lcol[1]; sh_r.c[2] = lcol[2];
    sh_r.xy = ray.xy;
    sh_r.depth = ray.depth;

    return vec3(0.0);
}

void Sample_PrincipledNode(const ray_data_t ray, const surface_t surf,
                           const lobe_weights_t lobe_weights, const diff_params_t diff,
                           const spec_params_t spec, const clearcoat_params_t coat,
                           const transmission_params_t trans, const float metallic, const float transmission,
                           const vec2 rand, float mix_rand, const float mix_weight, const float regularize_alpha,
                           inout ray_data_t new_ray) {
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);

    const int diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
                           refr_depth = get_refr_depth(ray.depth), transp_depth = get_transp_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const int total_depth = diff_depth + spec_depth + refr_depth;

    [[dont_flatten]] if (mix_rand < lobe_weights.diffuse) {
        //
        // Diffuse lobe
        //
        if (diff_depth < get_diff_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            vec3 V;
            vec4 F = Sample_PrincipledDiffuse_BSDF(surf.T, surf.B, surf.N, I, diff.roughness,
                                                   diff.base_color, diff.sheen_color, false, rand, V);
            F.rgb *= (1.0 - metallic) * (1.0 - transmission);
            //F[3] *= lobe_weights.diffuse;

            new_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(1, 0, 0, 0);

            const vec3 new_o = offset_ray(surf.P, surf.plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];

            new_ray.c[0] = F[0] * mix_weight / lobe_weights.diffuse;
            new_ray.c[1] = F[1] * mix_weight / lobe_weights.diffuse;
            new_ray.c[2] = F[2] * mix_weight / lobe_weights.diffuse;
            new_ray.pdf = F[3];
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT;
        }
    } else [[dont_flatten]] if (mix_rand < lobe_weights.diffuse + lobe_weights.specular) {
        //
        // Main specular lobe
        //
        if (spec_depth < get_spec_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            const vec2 alpha = calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha);
            vec3 V;
            vec4 F = Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha,
                                             spec.ior, spec.F0, spec.tmp_col, vec3(1.0), rand, V);
            F[3] *= lobe_weights.specular;

            new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

            new_ray.c[0] = F[0] * mix_weight / F[3];
            new_ray.c[1] = F[1] * mix_weight / F[3];
            new_ray.c[2] = F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

            const vec3 new_o = offset_ray(surf.P, surf.plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V [0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * min(alpha.x, alpha.y);
        }
    } else [[dont_flatten]] if (mix_rand < lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat) {
        //
        // Clearcoat lobe (secondary specular)
        //
        if (spec_depth < get_spec_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            const float alpha = calc_alpha(coat.roughness, 0.0, regularize_alpha).x;
            vec3 V;
            vec4 F = Sample_PrincipledClearcoat_BSDF(surf.T, surf.B, surf.N, I, alpha,
                                                     coat.ior, coat.F0, rand, V);
            F[3] *= lobe_weights.clearcoat;

            new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

            new_ray.c[0] = 0.25 * F[0] * mix_weight / F[3];
            new_ray.c[1] = 0.25 * F[1] * mix_weight / F[3];
            new_ray.c[2] = 0.25 * F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

            const vec3 new_o = offset_ray(surf.P, surf.plane_N);
            new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * alpha;
        }
    } else /*if (mix_rand < lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat + lobe_weights.refraction)*/ {
        //
        // Refraction/reflection lobes
        //
        mix_rand -= lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat;
        mix_rand /= lobe_weights.refraction;
        [[dont_flatten]] if (((mix_rand >= trans.fresnel && refr_depth < get_refr_depth(g_params.max_ray_depth)) ||
                                (mix_rand < trans.fresnel && spec_depth < get_spec_depth(g_params.max_ray_depth))) &&
                                total_depth < g_params.max_total_depth) {
            vec4 F;
            vec3 V;
            [[dont_flatten]] if (mix_rand < trans.fresnel) {
                const vec2 alpha = calc_alpha(spec.roughness, 0.0f, regularize_alpha);
                F = Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha,
                                            1.0 /* ior */, 0.0 /* F0 */, vec3(1.0), vec3(1.0), rand, V);

                new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
                new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

                const vec3 new_o = offset_ray(surf.P, surf.plane_N);
                new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
                new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * min(alpha.x, alpha.y);
            } else {
                const vec2 alpha = calc_alpha(trans.roughness, 0.0f, regularize_alpha);
                vec4 _V;
                F = Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, I, alpha,
                                              trans.eta, diff.base_color, rand, _V);
                V = _V.xyz;

                new_ray.depth = pack_ray_type(RAY_TYPE_REFR);
                new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 0, 1, 0);

                const vec3 new_o = offset_ray(surf.P, -surf.plane_N);
                new_ray.o[0] = new_o[0]; new_ray.o[1] = new_o[1]; new_ray.o[2] = new_o[2];
                new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * min(alpha.x, alpha.y);

                if (!trans.backfacing) {
                    // Entering the surface, push new value
                    push_ior_stack(new_ray.ior, trans.int_ior);
                } else {
                    // Exiting the surface, pop the last ior value
                    pop_ior_stack(new_ray.ior, 1.0);
                }
            }

            F[3] *= lobe_weights.refraction;

            new_ray.c[0] = F[0] * mix_weight / F[3];
            new_ray.c[1] = F[1] * mix_weight / F[3];
            new_ray.c[2] = F[2] * mix_weight / F[3];
            new_ray.pdf = F[3];

            new_ray.d[0] = V[0]; new_ray.d[1] = V[1]; new_ray.d[2] = V[2];
        }
    }
}

vec3 ShadeSurface(const int ray_index, const hit_data_t inter, const ray_data_t ray, inout vec3 out_base_color, inout vec3 out_normals) {
    const vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    const vec3 rd = vec3(ray.d[0], ray.d[1], ray.d[2]);

    const int diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
                           refr_depth = get_refr_depth(ray.depth), transp_depth = get_transp_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const int total_depth = diff_depth + spec_depth + refr_depth;

    const uint px_hash = hash(ray.xy);
    const uint rand_hash = hash_combine(px_hash, g_params.rand_seed);

    const uint rand_dim = RAND_DIM_BASE_COUNT + (total_depth + transp_depth) * RAND_DIM_BOUNCE_COUNT;

    const vec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, g_params.iteration - 1);

    [[dont_flatten]] if (inter.v < 0.0) {
#if DETAILED_SKY
        if (ray.cone_spread < g_params.sky_map_spread_angle) {
            const uint index = atomicAdd(g_inout_counters[4], 1);
            g_out_sky_rays[index] = ray_index;
            return vec3(0.0);
        }
#endif
#if USE_HIERARCHICAL_NEE
        const float pdf_factor = (total_depth < g_params.max_total_depth) ? (1.0 / inter.u) : -1.0;
#else
        const float pdf_factor = (total_depth < g_params.max_total_depth) ? float(g_params.li_count) : -1.0;
#endif

        vec3 env_col = Evaluate_EnvColor(ray, pdf_factor, tex_rand);
#if !CACHE_UPDATE
        env_col *= vec3(ray.c[0], ray.c[1], ray.c[2]);
#endif

        const float sum = env_col.r + env_col.g + env_col.b;
        if (sum > g_params.limit_direct) {
            env_col *= (g_params.limit_direct / sum);
        }

        return env_col;
    }

    const vec3 I = rd;

    surface_t surf;
    surf.P = ro + inter.t * rd;

    [[dont_flatten]] if (inter.obj_index < 0) { // Area light intersection
        vec3 lcol = Evaluate_LightColor(ray, inter, tex_rand);
#if !CACHE_UPDATE
        lcol *= vec3(ray.c[0], ray.c[1], ray.c[2]);
#endif

        const float sum = lcol.r + lcol.g + lcol.b;
        if (sum > g_params.limit_direct) {
            lcol *= (g_params.limit_direct / sum);
        }

        return lcol;
    }

    const bool is_backfacing = (inter.prim_index < 0);
    const uint tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

    material_t mat = g_materials[g_tri_materials[tri_index] & MATERIAL_INDEX_BITS];

    const mesh_instance_t mi = g_mesh_instances[inter.obj_index];

    const vertex_t v1 = g_vertices[g_vtx_indices[tri_index * 3 + 0]];
    const vertex_t v2 = g_vertices[g_vtx_indices[tri_index * 3 + 1]];
    const vertex_t v3 = g_vertices[g_vtx_indices[tri_index * 3 + 2]];

    const vec3 p1 = vec3(v1.p[0], v1.p[1], v1.p[2]);
    const vec3 p2 = vec3(v2.p[0], v2.p[1], v2.p[2]);
    const vec3 p3 = vec3(v3.p[0], v3.p[1], v3.p[2]);

    const float w = 1.0 - inter.u - inter.v;
    surf.N = normalize(vec3(v1.n[0], v1.n[1], v1.n[2]) * w +
                       vec3(v2.n[0], v2.n[1], v2.n[2]) * inter.u +
                       vec3(v3.n[0], v3.n[1], v3.n[2]) * inter.v);
    surf.uvs = vec2(v1.t[0], v1.t[1]) * w +
               vec2(v2.t[0], v2.t[1]) * inter.u +
               vec2(v3.t[0], v3.t[1]) * inter.v;

    float pa;
    surf.plane_N = normalize_len(cross(vec3(p2 - p1), vec3(p3 - p1)), pa);

    surf.B = vec3(v1.b[0], v1.b[1], v1.b[2]) * w +
             vec3(v2.b[0], v2.b[1], v2.b[2]) * inter.u +
             vec3(v3.b[0], v3.b[1], v3.b[2]) * inter.v;
    surf.T = cross(surf.B, surf.N);

    if (is_backfacing) {
        if (((g_tri_materials[tri_index] >> 16u) & 0xffff) == 0xffff) {
            return vec3(0.0);
        } else {
            mat = g_materials[(g_tri_materials[tri_index] >> 16u) & MATERIAL_INDEX_BITS];
            surf.plane_N = -surf.plane_N;
            surf.N = -surf.N;
            surf.B = -surf.B;
            surf.T = -surf.T;
        }
    }

    surf.plane_N = TransformNormal(surf.plane_N, mi.inv_xform);
    surf.N = TransformNormal(surf.N, mi.inv_xform);
    surf.B = TransformNormal(surf.B, mi.inv_xform);
    surf.T = TransformNormal(surf.T, mi.inv_xform);

    // normalize vectors (scaling might have been applied)
    surf.plane_N = normalize(surf.plane_N);
    surf.N = normalize(surf.N);
    surf.B = normalize(surf.B);
    surf.T = normalize(surf.T);

    const float ta = abs((v2.t[0] - v1.t[0]) * (v3.t[1] - v1.t[1]) -
                         (v3.t[0] - v1.t[0]) * (v2.t[1] - v1.t[1]));

    const float cone_width = ray.cone_width + ray.cone_spread * inter.t;

    float lambda = 0.5 * log2(ta / pa);
    lambda += log2(cone_width);
    // lambda += 0.5 * fast_log2(tex_res.x * tex_res.y);
    // lambda -= fast_log2(std::abs(dot(I, plane_N)));

    const float ext_ior = peek_ior_stack(ray.ior, is_backfacing, 1.0 /* default_value */);

    vec3 col = vec3(0.0);

    const vec2 mix_term_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF_PICK, rand_hash, g_params.iteration - 1);

    float mix_rand = mix_term_rand.x;
    float mix_weight = 1.0;

    // resolve mix material
    while (mat.type == MixNode) {
        float mix_val = mat.tangent_rotation_or_strength;
        if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
            mix_val *= SampleBilinear(mat.textures[BASE_TEXTURE], surf.uvs, 0, tex_rand, true /* YCoCg */, true /* SRGB */).r;
        }

        const float eta = is_backfacing ? (ext_ior / mat.ior) : (mat.ior / ext_ior);
        const float RR = mat.ior != 0.0 ? fresnel_dielectric_cos(dot(I, surf.N), eta) : 1.0;

        mix_val *= saturate(RR);

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
        vec3 normals = vec3(SampleBilinear(mat.textures[NORMALS_TEXTURE], surf.uvs, 0, tex_rand).xy, 1.0);
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
        vec3 in_normal = surf.N;
        surf.N = normalize(normals[0] * surf.T + normals[2] * surf.N + normals[1] * surf.B);
        if ((mat.normal_map_strength_unorm & 0xffff) != 0xffff) {
            surf.N = normalize(in_normal + (surf.N - in_normal) * unpack_unorm_16(mat.normal_map_strength_unorm & 0xffff));
        }
        surf.N = ensure_valid_reflection(surf.plane_N, -I, surf.N);
    }

#if 0
    //create_tbn_matrix(N, _tangent_from_world);
#else
    // Find radial tangent in local space
    const vec3 P_ls = p1 * w + p2 * inter.u + p3 * inter.v;
    // rotate around Y axis by 90 degrees in 2d
    vec3 tangent = vec3(-P_ls[2], 0.0, P_ls[0]);
    tangent = TransformNormal(tangent, mi.inv_xform);
    if (length2(cross(tangent, surf.N)) == 0.0) {
        tangent = TransformNormal(P_ls, mi.inv_xform);
    }
    if (mat.tangent_rotation_or_strength != 0.0) {
        tangent = rotate_around_axis(tangent, surf.N, mat.tangent_rotation_or_strength);
    }

    surf.B = normalize(cross(tangent, surf.N));
    surf.T = cross(surf.N, surf.B);
#endif

#if CACHE_QUERY
    if (mat.type != EmissiveNode) {
        cache_grid_params_t params;
        params.cam_pos_curr = g_params.cam_pos_and_exposure.xyz;
        params.log_base = RAD_CACHE_GRID_LOGARITHM_BASE;
        params.scale = RAD_CACHE_GRID_SCALE;
        params.exposure = g_params.cam_pos_and_exposure.w;

        const vec2 cache_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_CACHE, rand_hash, g_params.iteration - 1);

        const uint grid_level = calc_grid_level(surf.P, params);
        const float voxel_size = calc_voxel_size(grid_level, params);

        bool use_cache = cone_width > mix(1.0, 1.5, cache_rand.x) * voxel_size;
        use_cache = use_cache && (inter.t > mix(1.0, 2.0, cache_rand.y) * voxel_size);
        if (use_cache) {
            const uint cache_entry = find_entry(surf.P, surf.plane_N, params);
            if (cache_entry != HASH_GRID_INVALID_CACHE_ENTRY) {
                const uvec4 voxel = g_cache_voxels[cache_entry];
                const cache_voxel_t unpacked = unpack_voxel_data(voxel);
                if (unpacked.sample_count >= RAD_CACHE_SAMPLE_COUNT_MIN) {
                    vec3 color = unpacked.radiance / float(unpacked.sample_count);
                    color /= params.exposure;
                    color *= vec3(ray.c[0], ray.c[1], ray.c[2]);
                    return color;
                }
            }
        }
    }
#endif

#if USE_NEE
    light_sample_t ls;
    ls.col = ls.L = vec3(0.0);
    ls.area = ls.pdf = 0.0;
    ls.dist_mul = 1.0;
    if (/*pi.should_add_direct_light() &&*/ g_params.li_count != 0 && mat.type != EmissiveNode) {
        const float rand_pick_light = get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT_PICK, rand_hash, g_params.iteration - 1).x;
        const vec2 rand_light_uv = get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT, rand_hash, g_params.iteration - 1);

        SampleLightSource(surf.P, surf.T, surf.B, surf.N, rand_pick_light, rand_light_uv, tex_rand, ls);
    }
    const float N_dot_L = dot(surf.N, ls.L);
#endif

    vec3 base_color = vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
    [[dont_flatten]] if (mat.textures[BASE_TEXTURE] != 0xffffffff) {
        const float base_lod = get_texture_lod(texSize(mat.textures[BASE_TEXTURE]), lambda);
        base_color *= SampleBilinear(mat.textures[BASE_TEXTURE], surf.uvs, int(base_lod), tex_rand, true /* YCoCg */, true /* SRGB */).rgb;
    }

    out_base_color = base_color;
#if !CACHE_UPDATE
    out_normals = surf.N;
#else
    out_normals = surf.plane_N;
#endif

    vec3 tint_color = vec3(0.0);

    const float base_color_lum = lum(base_color);
    [[flatten]] if (base_color_lum > 0.0) {
        tint_color = base_color / base_color_lum;
    }

    float roughness = unpack_unorm_16(mat.roughness_and_anisotropic & 0xffff);
    [[dont_flatten]] if (mat.textures[ROUGH_TEXTURE] != 0xffffffff) {
        const float roughness_lod = get_texture_lod(texSize(mat.textures[ROUGH_TEXTURE]), lambda);
        roughness *= SampleBilinear(mat.textures[ROUGH_TEXTURE], surf.uvs, int(roughness_lod), tex_rand, false /* YCoCg */, true /* SRGB */).r;
    }
#if CACHE_UPDATE
    roughness = max(roughness, RAD_CACHE_MIN_ROUGHNESS);
#endif

    const vec2 rand_bsdf_uv = get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF, rand_hash, g_params.iteration - 1);

    ray_data_t new_ray;
    new_ray.c[0] = new_ray.c[1] = new_ray.c[2] = 0.0;
    [[unroll]] for (int i = 0; i < 4; ++i) {
        new_ray.ior[i] = ray.ior[i];
    }
    new_ray.cone_width = cone_width;
    new_ray.cone_spread = ray.cone_spread;
    new_ray.xy = ray.xy;
    new_ray.pdf = 0.0;

    shadow_ray_t sh_r;
    sh_r.c[0] = sh_r.c[1] = sh_r.c[2] = 0.0;
    sh_r.depth = ray.depth;
    sh_r.xy = ray.xy;

    ///

    const float regularize_alpha = (get_diff_depth(ray.depth) > 0) ? g_params.regularize_alpha : 0.0;

    [[dont_flatten]] if (mat.type == DiffuseNode) {
#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && (ls.ray_flags & RAY_TYPE_DIFFUSE_BIT) != 0 && N_dot_L > 0.0) {
            col += Evaluate_DiffuseNode(ls, ray, surf, base_color, roughness, mix_weight,
                                        (total_depth < g_params.max_total_depth), sh_r);
        }
#endif
        [[dont_flatten]] if (diff_depth < get_diff_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            Sample_DiffuseNode(ray, surf, base_color, roughness, rand_bsdf_uv, mix_weight, new_ray);
        }
    } else [[dont_flatten]] if (mat.type == GlossyNode) {
        const float specular = 0.5;
        const float spec_ior = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;
        const float spec_F0 = fresnel_dielectric_cos(1.0, spec_ior);

#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0 && N_dot_L > 0.0) {
            col += Evaluate_GlossyNode(ls, ray, surf, base_color, roughness, regularize_alpha, spec_ior,
                                       spec_F0, mix_weight, (total_depth < g_params.max_total_depth), sh_r);
        }
#endif
        [[dont_flatten]] if (spec_depth < get_spec_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            Sample_GlossyNode(ray, surf, base_color, roughness, regularize_alpha, spec_ior, spec_F0, rand_bsdf_uv,
                              mix_weight, new_ray);
        }
    } else [[dont_flatten]] if (mat.type == RefractiveNode) {
#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0 && (ls.ray_flags & RAY_TYPE_REFR_BIT) != 0 && N_dot_L < 0.0) {
            const float eta = is_backfacing ? (mat.ior / ext_ior) : (ext_ior / mat.ior);
            col += Evaluate_RefractiveNode(ls, ray, surf, base_color, roughness, regularize_alpha, eta, mix_weight,
                                           (total_depth < g_params.max_total_depth), sh_r);
        }
#endif
        [[dont_flatten]] if (refr_depth < get_refr_depth(g_params.max_ray_depth) && total_depth < g_params.max_total_depth) {
            Sample_RefractiveNode(ray, surf, base_color, roughness, regularize_alpha, is_backfacing, mat.ior, ext_ior, rand_bsdf_uv,
                                  mix_weight, new_ray);
        }
    } else [[dont_flatten]] if (mat.type == EmissiveNode) {
        float mis_weight = 1.0;
#if USE_NEE && !PRIMARY
        [[dont_flatten]] if ((mat.flags & MAT_FLAG_IMP_SAMPLE) != 0) {
#if USE_HIERARCHICAL_NEE
            // TODO: maybe this can be done more efficiently
            const float pdf_factor = EvalTriLightFactor(surf.P, ro, tri_index);
#else
            const float pdf_factor = float(g_params.li_count);
#endif

            const vec3 p1 = vec3(v1.p[0], v1.p[1], v1.p[2]),
                       p2 = vec3(v2.p[0], v2.p[1], v2.p[2]),
                       p3 = vec3(v3.p[0], v3.p[1], v3.p[2]);

            float light_forward_len;
            vec3 light_forward = normalize_len((mi.xform * vec4(cross(p2 - p1, p3 - p1), 0.0)).xyz, light_forward_len);
            const float tri_area = 0.5 * light_forward_len;

            const float cos_theta = abs(dot(I, light_forward)); // abs for doublesided light
            if (cos_theta > 0.0) {
                float light_pdf;
#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
                const vec3 P = (mi.inv_xform * vec4(ro, 1.0)).xyz;

                vec3 _unused;
                light_pdf = SampleSphericalTriangle(P, p1, p2, p3, vec2(0.0), _unused) / pdf_factor;
                if (light_pdf == 0.0)
#endif // USE_SPHERICAL_AREA_LIGHT_SAMPLING
                {
                    light_pdf = (inter.t * inter.t) / (tri_area * cos_theta * pdf_factor);
                }
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
            metallic *= SampleBilinear(mat.textures[METALLIC_TEXTURE], surf.uvs, int(metallic_lod), tex_rand).r;
        }

        float specular = unpack_unorm_16(mat.specular_and_specular_tint & 0xffff);
        [[dont_flatten]] if (mat.textures[SPECULAR_TEXTURE] != 0xffffffff) {
            const float specular_lod = get_texture_lod(texSize(mat.textures[SPECULAR_TEXTURE]), lambda);
            specular *= SampleBilinear(mat.textures[SPECULAR_TEXTURE], surf.uvs, int(specular_lod), tex_rand).r;
        }

        const float specular_tint = unpack_unorm_16((mat.specular_and_specular_tint >> 16) & 0xffff);
        const float transmission = unpack_unorm_16(mat.transmission_and_transmission_roughness & 0xffff);
        const float clearcoat = unpack_unorm_16(mat.clearcoat_and_clearcoat_roughness & 0xffff);
        const float clearcoat_roughness = unpack_unorm_16((mat.clearcoat_and_clearcoat_roughness >> 16) & 0xffff);
        const float sheen = 2.0 * unpack_unorm_16(mat.sheen_and_sheen_tint & 0xffff);
        const float sheen_tint = unpack_unorm_16((mat.sheen_and_sheen_tint >> 16) & 0xffff);

        diff_params_t diff;
        diff.base_color = base_color;
        diff.sheen_color = sheen * mix(vec3(1.0), tint_color, sheen_tint);
        diff.roughness = roughness;

        spec_params_t spec;
        spec.tmp_col = mix(vec3(1.0), tint_color, specular_tint);
        spec.tmp_col = mix(specular * 0.08 * spec.tmp_col, base_color, metallic);
        spec.roughness = roughness;
        spec.ior = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;
        spec.F0 = fresnel_dielectric_cos(1.0, spec.ior);
        spec.anisotropy = unpack_unorm_16((mat.roughness_and_anisotropic >> 16) & 0xffff);

        clearcoat_params_t coat;
        coat.roughness = clearcoat_roughness;
        coat.ior = (2.0 / (1.0 - sqrt(0.08 * clearcoat))) - 1.0;
        coat.F0 = fresnel_dielectric_cos(1.0, coat.ior);

        transmission_params_t trans;
        trans.roughness =
            1.0 - (1.0 - roughness) * (1.0 - unpack_unorm_16((mat.transmission_and_transmission_roughness >> 16) & 0xffff));
        trans.int_ior = mat.ior;
        trans.eta = is_backfacing ? (mat.ior / ext_ior) : (ext_ior / mat.ior);
        trans.fresnel = fresnel_dielectric_cos(dot(I, surf.N), 1.0 / trans.eta);
        trans.backfacing = is_backfacing;

        // Approximation of FH (using shading normal)
        const float FN = (fresnel_dielectric_cos(dot(I, surf.N), spec.ior) - spec.F0) / (1.0 - spec.F0);

        const vec3 approx_spec_col = mix(spec.tmp_col, vec3(1.0), FN);
        const float spec_color_lum = lum(approx_spec_col);

        lobe_weights_t lobe_weights =
            get_lobe_weights(mix(base_color_lum, 1.0, sheen), spec_color_lum, specular, metallic, transmission, clearcoat);

#if USE_NEE
        [[dont_flatten]] if (ls.pdf > 0.0) {
            col += Evaluate_PrincipledNode(ls, ray, surf, lobe_weights, diff, spec, coat, trans, metallic, transmission, N_dot_L,
                                           mix_weight, (total_depth < g_params.max_total_depth), regularize_alpha, sh_r);
        }
#endif
        Sample_PrincipledNode(ray, surf, lobe_weights, diff, spec, coat, trans, metallic, transmission, rand_bsdf_uv, mix_rand,
                              mix_weight, regularize_alpha, new_ray);
    }

#if USE_PATH_TERMINATION
    const bool can_terminate_path = total_depth > g_params.min_total_depth;
#else
    const bool can_terminate_path = false;
#endif

#if !CACHE_UPDATE
    new_ray.c[0] *= ray.c[0]; new_ray.c[1] *= ray.c[1]; new_ray.c[2] *= ray.c[2];
#endif
    const float lum = max(new_ray.c[0], max(new_ray.c[1], new_ray.c[2]));
    const float p = mix_term_rand.y;
    const float q = can_terminate_path ? max(0.05, 1.0 - lum) : 0.0;
    [[dont_flatten]] if (p >= q && lum > 0.0 && new_ray.pdf > 0.0) {
        new_ray.pdf = min(new_ray.pdf, 1e6f);
        new_ray.c[0] /= (1.0 - q);
        new_ray.c[1] /= (1.0 - q);
        new_ray.c[2] /= (1.0 - q);
        const uint index = atomicAdd(g_inout_counters[0], 1);
        g_out_rays[index] = new_ray;
    }

#if USE_NEE
    #if !CACHE_UPDATE
        sh_r.c[0] *= ray.c[0]; sh_r.c[1] *= ray.c[1]; sh_r.c[2] *= ray.c[2];
    #endif
    const float sh_lum = max(sh_r.c[0], max(sh_r.c[1], sh_r.c[2]));
    [[dont_flatten]] if (sh_lum > 0.0) {
        // actual ray direction accouning for bias from both ends
        float sh_dist;
        const vec3 to_light = normalize_len(ls.lp - vec3(sh_r.o[0], sh_r.o[1], sh_r.o[2]), sh_dist);
        sh_dist *= ls.dist_mul;
        if (ls.from_env) {
            // NOTE: hacky way to identify env ray
            sh_dist = -sh_dist;
        }
        sh_r.d[0] = to_light[0]; sh_r.d[1] = to_light[1]; sh_r.d[2] = to_light[2];
        sh_r.dist = sh_dist;
        const uint index = atomicAdd(g_inout_counters[2], 1);
        g_out_sh_rays[index] = sh_r;
    }
#endif

#if !CACHE_UPDATE
    col *= vec3(ray.c[0], ray.c[1], ray.c[2]);
#endif

    const float sum = col.r + col.g + col.b;
    if (sum > g_params.limit_indirect) {
        col *= (g_params.limit_indirect / sum);
    }

    return col;
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
#if !INDIRECT
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    const int x = int(g_params.rect.x + gl_GlobalInvocationID.x);
    const int y = int(g_params.rect.y + gl_GlobalInvocationID.y);

    const int index = int(gl_GlobalInvocationID.y * g_params.rect.z) + x;
#else
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_inout_counters[1]) {
        return;
    }

    const int x = int((g_rays[index].xy >> 16) & 0xffff);
    const int y = int(g_rays[index].xy & 0xffff);
#endif

    const hit_data_t inter = g_hits[index];
    const ray_data_t ray = g_rays[index];

    vec3 base_color = vec3(0.0), normals = vec3(0.0);
    vec3 col = ShadeSurface(index, inter, ray, base_color, normals);

#if !PRIMARY && !CACHE_UPDATE
    col += imageLoad(g_out_img, ivec2(x, y)).rgb;
#endif
    imageStore(g_out_img, ivec2(x, y), vec4(col, 1.0));
#if OUTPUT_BASE_COLOR && !CACHE_UPDATE
    const float norm_factor = max(max(base_color.x, base_color.y), max(base_color.z, 1.0));
    base_color /= norm_factor;
    imageStore(g_out_base_color_img, ivec2(x, y), vec4(base_color, 0.0));
#endif
#if OUTPUT_DEPTH_NORMALS || CACHE_UPDATE
    imageStore(g_out_depth_normals_img, ivec2(x, y), vec4(normals, inter.t));
#endif
}
