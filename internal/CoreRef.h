#pragma once

#include <cfloat>
#include <vector>

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS Ref
#if defined(_M_AMD64) || defined(_M_X64) || (!defined(__ANDROID__) && defined(__x86_64__)) ||                          \
    (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE2
// #pragma message("Ray::Ref::simd_vec will use SSE2")
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#define USE_NEON
// #pragma message("Ray::Ref::simd_vec will use NEON")
#elif defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define USE_SSE2
// #pragma message("Ray::Ref::simd_vec will use SSE2")
#else
#pragma message("Ray::Ref::simd_vec will not use SIMD")
#endif

#include "simd/simd.h"

#undef USE_SSE2
#undef USE_NEON
#undef NS

#pragma pop_macro("NS")

#define USE_SAFE_MATH 1

namespace Ray {
namespace Cpu {
class TexStorageBase;
template <typename T, int N> class TexStorageLinear;
template <typename T, int N> class TexStorageTiled;
template <typename T, int N> class TexStorageSwizzled;
using TexStorageRGBA = TexStorageSwizzled<uint8_t, 4>;
using TexStorageRGB = TexStorageSwizzled<uint8_t, 3>;
using TexStorageRG = TexStorageSwizzled<uint8_t, 2>;
using TexStorageR = TexStorageSwizzled<uint8_t, 1>;
} // namespace Cpu
namespace Ref {
//
// Useful constants for debugging
//
const bool USE_NEE = true;
const bool USE_PATH_TERMINATION = true;
const bool USE_HIERARCHICAL_NEE = true;
const bool USE_SPHERICAL_AREA_LIGHT_SAMPLING = true;

// Generic ray structure
struct ray_data_t {
    // origin, direction and PDF
    float o[3], d[3], pdf;
    // throughput color of ray
    float c[3];
    // stack of ior values
    float ior[4];
    // ray cone params
    float cone_width, cone_spread;
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    uint32_t xy;
    // four 7-bit ray depth counters
    uint32_t depth;
};
static_assert(sizeof(ray_data_t) == 72, "!");

// Shadow ray structure
struct shadow_ray_t {
    // origin
    float o[3];
    // four 7-bit ray depth counters
    uint32_t depth;
    // direction and distance
    float d[3], dist;
    // throughput color of ray
    float c[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    uint32_t xy;
};
static_assert(sizeof(shadow_ray_t) == 48, "!");

// Ray hit structure
struct hit_data_t {
    // index of an object that was hit by ray
    int obj_index;
    // index of a primitive that was hit by ray
    int prim_index;
    // distance and baricentric coordinates of a hit point
    float t, u, v;

    explicit hit_data_t(eUninitialize) {}
    hit_data_t() {
        obj_index = -1;
        prim_index = -1;
        t = MAX_DIST;
        u = 0.0f;
        v = -1.0f; // negative v means 'no intersection'
    }
};

// Surface at the hit point
struct surface_t {
    // position, tangent, bitangent, smooth normal and planar normal
    fvec4 P, T, B, N, plane_N;
    // texture coordinates
    fvec2 uvs;
};

// Surface derivatives at the hit point
struct derivatives_t {
    fvec4 do_dx, do_dy, dd_dx, dd_dy;
    fvec2 duv_dx, duv_dy;
    fvec4 dndx, dndy;
    float ddn_dx, ddn_dy;
};

struct light_sample_t {
    fvec4 col, L, lp;
    float area = 0, dist_mul = 1, pdf = 0;
    uint32_t cast_shadow : 1;
    uint32_t from_env : 1;
    uint32_t ray_flags : 8;
    uint32_t _pad0 : 22;
};
static_assert(sizeof(light_sample_t) == 64, "!");

force_inline constexpr uint32_t hash(uint32_t x) {
    // finalizer from murmurhash3
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

force_inline uint32_t hash_combine(uint32_t seed, uint32_t v) { return seed ^ (v + (seed << 6) + (seed >> 2)); }

force_inline float construct_float(uint32_t m) {
    static const uint32_t ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    static const uint32_t ieeeOne = 0x3F800000u;      // 1.0 in IEEE binary32

    m &= ieeeMantissa; // Keep only mantissa bits (fractional part)
    m |= ieeeOne;      // Add fractional part to 1.0

    union {
        uint32_t i;
        float f;
    } ret = {m};         // Range [1:2]
    return ret.f - 1.0f; // Range [0:1]
}

force_inline float fract(const float v) { return v - floorf(v); }

force_inline float safe_sqrt(float val) {
#if USE_SAFE_MATH
    return sqrtf(fmaxf(val, 0.0f));
#else
    return sqrtf(val);
#endif
}

force_inline float safe_div(const float a, const float b) {
#if USE_SAFE_MATH
    return b != 0.0f ? (a / b) : FLT_MAX;
#else
    return (a / b)
#endif
}

force_inline float safe_div_pos(const float a, const float b) {
#if USE_SAFE_MATH
    return a / fmaxf(b, FLT_EPS);
#else
    return (a / b)
#endif
}

force_inline float safe_div_neg(const float a, const float b) {
#if USE_SAFE_MATH
    return a / fminf(b, -FLT_EPS);
#else
    return (a / b)
#endif
}

force_inline void safe_invert(const float v[3], float out_v[3]) {
    for (int i = 0; i < 3; ++i) {
        out_v[i] = 1.0f / ((fabsf(v[i]) > FLT_EPS) ? v[i] : copysignf(FLT_EPS, v[i]));
    }
}

force_inline fvec4 safe_normalize(const fvec4 &a) {
#if USE_SAFE_MATH
    const float l = length(a);
    return l > 0.0f ? (a / l) : a;
#else
    return normalize(a);
#endif
}

force_inline fvec4 srgb_to_linear(const fvec4 &col) {
    fvec4 ret;
    UNROLLED_FOR(i, 3, {
        if (col.get<i>() > 0.04045f) {
            ret.set<i>(powf((col.get<i>() + 0.055f) / 1.055f, 2.4f));
        } else {
            ret.set<i>(col.get<i>() / 12.92f);
        }
    })
    ret.set<3>(col[3]);

    return ret;
}

force_inline fvec2 srgb_to_linear(const fvec2 &col) {
    fvec2 ret;
    UNROLLED_FOR(i, 2, {
        if (col.get<i>() > 0.04045f) {
            ret.set<i>(powf((col.get<i>() + 0.055f) / 1.055f, 2.4f));
        } else {
            ret.set<i>(col.get<i>() / 12.92f);
        }
    })
    return ret;
}

force_inline fvec4 rgbe_to_rgb(const color_t<uint8_t, 4> &rgbe) {
    const float f = exp2f(float(rgbe.v[3]) - 128.0f);
    return fvec4{to_norm_float(rgbe.v[0]) * f, to_norm_float(rgbe.v[1]) * f, to_norm_float(rgbe.v[2]) * f, 1.0f};
}

force_inline fvec4 YCoCg_to_RGB(const fvec4 &col) {
    const float scale = (col.get<2>() * (255.0f / 8.0f)) + 1.0f;
    const float Y = col.get<3>();
    const float Co = (col.get<0>() - (0.5f * 256.0f / 255.0f)) / scale;
    const float Cg = (col.get<1>() - (0.5f * 256.0f / 255.0f)) / scale;

    fvec4 col_rgb = 1.0f;
    col_rgb.set<0>(Y + Co - Cg);
    col_rgb.set<1>(Y + Cg);
    col_rgb.set<2>(Y - Co - Cg);

    return saturate(col_rgb);
}

force_inline uint32_t mask_ray_depth(const uint32_t depth) { return depth & 0x0fffffff; }
force_inline uint32_t pack_ray_type(const int ray_type) {
    assert(ray_type < 0xf);
    return uint32_t(ray_type << 28);
}
force_inline uint32_t pack_ray_depth(const int diff_depth, const int spec_depth, const int refr_depth,
                                     const int transp_depth) {
    assert(diff_depth < 0x7f && spec_depth < 0x7f && refr_depth < 0x7f && transp_depth < 0x7f);
    uint32_t ret = 0;
    ret |= (diff_depth << 0);
    ret |= (spec_depth << 7);
    ret |= (refr_depth << 14);
    ret |= (transp_depth << 21);
    return ret;
}
force_inline int get_diff_depth(const uint32_t depth) { return int(depth & 0x7f); }
force_inline int get_spec_depth(const uint32_t depth) { return int(depth >> 7) & 0x7f; }
force_inline int get_refr_depth(const uint32_t depth) { return int(depth >> 14) & 0x7f; }
force_inline int get_transp_depth(const uint32_t depth) { return int(depth >> 21) & 0x7f; }
force_inline int get_total_depth(const uint32_t depth) {
    return get_diff_depth(depth) + get_spec_depth(depth) + get_refr_depth(depth) + get_transp_depth(depth);
}
force_inline int get_ray_type(const uint32_t depth) { return int(depth >> 28) & 0xf; }

force_inline bool is_indirect(const uint32_t depth) {
    // not only transparency ray
    return (depth & 0x001fffff) != 0;
}

force_inline fvec4 make_fvec3(const float *f) { return fvec4{f[0], f[1], f[2], 0.0f}; }

force_inline fvec4 cross(const fvec4 &v1, const fvec4 &v2) {
    return fvec4{v1.get<1>() * v2.get<2>() - v1.get<2>() * v2.get<1>(),
                 v1.get<2>() * v2.get<0>() - v1.get<0>() * v2.get<2>(),
                 v1.get<0>() * v2.get<1>() - v1.get<1>() * v2.get<0>(), 0.0f};
}

force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}
force_inline float saturate(const float val) { return clamp(val, 0.0f, 1.0f); }

force_inline float sqr(const float x) { return x * x; }

force_inline fvec4 world_from_tangent(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &V) {
    return V.get<0>() * T + V.get<1>() * B + V.get<2>() * N;
}

force_inline fvec4 tangent_from_world(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &V) {
    return fvec4{dot(V, T), dot(V, B), dot(V, N), 0.0f};
}

float portable_cos(float a);
float portable_sin(float a);
fvec2 portable_sincos(float a);

fvec2 get_scrambled_2d_rand(const uint32_t dim, const uint32_t seed, const int sample, const uint32_t rand_seq[]);

float SampleSphericalRectangle(const fvec4 &P, const fvec4 &light_pos, const fvec4 &axis_u, const fvec4 &axis_v,
                               fvec2 Xi, fvec4 *out_p);
float SampleSphericalTriangle(const fvec4 &P, const fvec4 &p1, const fvec4 &p2, const fvec4 &p3, const fvec2 Xi,
                              fvec4 *out_dir);

// Generation of rays
void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, const uint32_t rand_seq[],
                         uint32_t rand_seed, const float filter_table[], int iteration,
                         const uint16_t required_samples[], aligned_vector<ray_data_t> &out_rays,
                         aligned_vector<hit_data_t> &out_inters);
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const mesh_instance_t &mi,
                              const uint32_t *vtx_indices, const vertex_t *vertices, const rect_t &r, int w, int h,
                              const uint32_t rand_seq[], aligned_vector<ray_data_t> &out_rays,
                              aligned_vector<hit_data_t> &out_inters);

// Sorting of rays
int SortRays_CPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3], uint32_t *hash_values,
                 uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
int SortRays_GPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3], uint32_t *hash_values,
                 int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp,
                 uint32_t *skeleton);

// Intersect primitives
bool IntersectTris_ClosestHit(const float ro[3], const float rd[3], const tri_accel_t *tris, int tri_start, int tri_end,
                              int obj_index, hit_data_t &out_inter);
bool IntersectTris_ClosestHit(const float ro[3], const float rd[3], const mtri_accel_t *mtris, int tri_start,
                              int tri_end, int obj_index, hit_data_t &out_inter);
bool IntersectTris_AnyHit(const float ro[3], const float rd[3], const tri_accel_t *tris,
                          const tri_mat_data_t *materials, const uint32_t *indices, int tri_start, int tri_end,
                          int obj_index, hit_data_t &out_inter);
bool IntersectTris_AnyHit(const float ro[3], const float rd[3], const mtri_accel_t *mtris,
                          const tri_mat_data_t *materials, const uint32_t *indices, int tri_start, int tri_end,
                          int obj_index, hit_data_t &out_inter);

// traditional bvh traversal with stack for outer nodes
bool Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], uint32_t ray_flags,
                                        const bvh_node_t *nodes, uint32_t root_index,
                                        const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                        const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], uint32_t ray_flags,
                                        const bvh2_node_t *nodes, uint32_t root_index,
                                        const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                        const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], uint32_t ray_flags,
                                        const wbvh_node_t *oct_nodes, uint32_t root_index,
                                        const mesh_instance_t *mesh_instances, const mtri_accel_t *mtris,
                                        const uint32_t *tri_indices, hit_data_t &inter);
// returns whether hit was solid
bool Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], int ray_type, const bvh_node_t *nodes,
                                    uint32_t root_index, const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], int ray_type, const bvh2_node_t *nodes,
                                    uint32_t root_index, const mesh_instance_t *mesh_instances, const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], int ray_type, const wbvh_node_t *nodes,
                                    uint32_t root_index, const mesh_instance_t *mesh_instances,
                                    const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                    const uint32_t *tri_indices, hit_data_t &inter);
// traditional bvh traversal with stack for inner nodes
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                        const bvh_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                        int obj_index, hit_data_t &inter);
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                        const bvh2_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                        int obj_index, hit_data_t &inter);
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                        const wbvh_node_t *nodes, uint32_t root_index, const mtri_accel_t *mtris,
                                        int obj_index, hit_data_t &inter);
// returns whether hit was solid
bool Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3], const bvh_node_t *nodes,
                                    uint32_t root_index, const tri_accel_t *tris, const tri_mat_data_t *materials,
                                    const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                    const bvh2_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index,
                                    hit_data_t &inter);
bool Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                    const wbvh_node_t *nodes, uint32_t root_index, const mtri_accel_t *mtris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index,
                                    hit_data_t &inter);

// Transform
void TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3], float out_rd[3]);
fvec4 TransformPoint(const fvec4 &p, const float *xform);
fvec4 TransformDirection(const fvec4 &p, const float *xform);
fvec4 TransformNormal(const fvec4 &n, const float *inv_xform);

force_inline float lum(const fvec3 &color) {
    return 0.212671f * color.get<0>() + 0.715160f * color.get<1>() + 0.072169f * color.get<2>();
}

force_inline float lum(const fvec4 &color) {
    return 0.212671f * color.get<0>() + 0.715160f * color.get<1>() + 0.072169f * color.get<2>();
}

force_inline float fast_log2(float val) {
    // From https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c
    union {
        float val;
        int32_t x;
    } u = {val};
    auto log_2 = float(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
    return (log_2);
}

float get_texture_lod(const Cpu::TexStorageBase *const textures[], const uint32_t index, const fvec2 &duv_dx,
                      const fvec2 &duv_dy);
float get_texture_lod(const Cpu::TexStorageBase *const textures[], const uint32_t index, const float lambda);

force_inline float power_heuristic(const float a, const float b) {
    const float t = a * a;
    return t / (b * b + t);
}

//
// From "A Fast and Robust Method for Avoiding Self-Intersection"
//

force_inline int32_t float_as_int(const float v) {
    union {
        float f;
        int32_t i;
    } ret = {v};
    return ret.i;
}
force_inline float int_as_float(const int32_t v) {
    union {
        int32_t i;
        float f;
    } ret = {v};
    return ret.f;
}

inline fvec4 offset_ray(const fvec4 &p, const fvec4 &n) {
    const float Origin = 1.0f / 32.0f;
    const float FloatScale = 1.0f / 65536.0f;
    const float IntScale = 128.0f; // 256.0f;

    const ivec4 of_i(IntScale * n);

    const fvec4 p_i(int_as_float(float_as_int(p.get<0>()) + ((p.get<0>() < 0.0f) ? -of_i.get<0>() : of_i.get<0>())),
                    int_as_float(float_as_int(p.get<1>()) + ((p.get<1>() < 0.0f) ? -of_i.get<1>() : of_i.get<1>())),
                    int_as_float(float_as_int(p.get<2>()) + ((p.get<2>() < 0.0f) ? -of_i.get<2>() : of_i.get<2>())),
                    0.0f);

    return fvec4{fabsf(p.get<0>()) < Origin ? (p.get<0>() + FloatScale * n.get<0>()) : p_i.get<0>(),
                 fabsf(p.get<1>()) < Origin ? (p.get<1>() + FloatScale * n.get<1>()) : p_i.get<1>(),
                 fabsf(p.get<2>()) < Origin ? (p.get<2>() + FloatScale * n.get<2>()) : p_i.get<2>(), 0.0f};
}

// Sample Texture
fvec4 SampleNearest(const Cpu::TexStorageBase *const textures[], uint32_t index, const fvec2 &uvs, int lod);
fvec4 SampleBilinear(const Cpu::TexStorageBase *const textures[], uint32_t index, const fvec2 &uvs, int lod,
                     const fvec2 &rand);
fvec4 SampleBilinear(const Cpu::TexStorageBase &storage, uint32_t tex, const fvec2 &iuvs, int lod, const fvec2 &rand);
fvec4 SampleTrilinear(const Cpu::TexStorageBase *const textures[], uint32_t index, const fvec2 &uvs, float lod,
                      const fvec2 &rand);
fvec4 SampleAnisotropic(const Cpu::TexStorageBase *const textures[], uint32_t index, const fvec2 &uvs,
                        const fvec2 &duv_dx, const fvec2 &duv_dy);
fvec4 SampleLatlong_RGBE(const Cpu::TexStorageRGBA &storage, uint32_t index, const fvec4 &dir, float y_rotation,
                         const fvec2 &rand);

// Trace rays through scene hierarchy
void IntersectScene(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth, const uint32_t rand_seq[],
                    uint32_t random_seed, int iteration, const scene_data_t &sc, uint32_t root_index,
                    const Cpu::TexStorageBase *const textures[], Span<hit_data_t> out_inter);
fvec4 IntersectScene(const shadow_ray_t &r, int max_transp_depth, const scene_data_t &sc, uint32_t node_index,
                     const uint32_t rand_seq[], uint32_t random_seed, int iteration,
                     const Cpu::TexStorageBase *const textures[]);

// Pick point on any light source for evaluation
void SampleLightSource(const fvec4 &P, const fvec4 &T, const fvec4 &B, const fvec4 &N, const scene_data_t &sc,
                       const Cpu::TexStorageBase *const textures[], float rand_pick_light, fvec2 rand_light_uv,
                       fvec2 rand_tex_uv, light_sample_t &ls);

// Account for visible lights contribution
void IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights, Span<const light_cwbvh_node_t> nodes,
                         Span<hit_data_t> inout_inters);
void IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights, Span<const light_wbvh_node_t> nodes,
                         Span<hit_data_t> inout_inters);
void IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights, Span<const light_bvh_node_t> nodes,
                         Span<hit_data_t> inout_inters);
float IntersectAreaLights(const shadow_ray_t &ray, Span<const light_t> lights, Span<const light_wbvh_node_t> nodes);
float IntersectAreaLights(const shadow_ray_t &ray, Span<const light_t> lights, Span<const light_cwbvh_node_t> nodes);
float EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                         Span<const light_bvh_node_t> nodes);
float EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                         Span<const light_wbvh_node_t> nodes);
float EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                         Span<const light_cwbvh_node_t> nodes);

float Evaluate_EnvQTree(float y_rotation, const fvec4 *const *qtree_mips, int qtree_levels, const fvec4 &L);
fvec4 Sample_EnvQTree(float y_rotation, const fvec4 *const *qtree_mips, int qtree_levels, float rand, float rx,
                      float ry);

void TraceRays(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth, const scene_data_t &sc,
               uint32_t node_index, bool trace_lights, const Cpu::TexStorageBase *const textures[],
               const uint32_t rand_seq[], uint32_t random_seed, int iteration, Span<hit_data_t> out_inter);
void TraceShadowRays(Span<const shadow_ray_t> rays, int max_transp_depth, float clamp_val, const scene_data_t &sc,
                     uint32_t node_index, const uint32_t rand_seq[], uint32_t random_seed, int iteration,
                     const Cpu::TexStorageBase *const textures[], int img_w, color_rgba_t *out_color);

// Get environment color at direction
fvec4 Evaluate_EnvColor(const ray_data_t &ray, const environment_t &env, const Cpu::TexStorageRGBA &tex_storage,
                        float pdf_factor, const fvec2 &rand);
// Get light color at intersection point
fvec4 Evaluate_LightColor(const ray_data_t &ray, const hit_data_t &inter, const environment_t &env,
                          const Cpu::TexStorageRGBA &tex_storage, Span<const light_t> lights, uint32_t lights_count,
                          const fvec2 &rand);

} // namespace Ref
} // namespace Ray

#undef USE_SAFE_MATH