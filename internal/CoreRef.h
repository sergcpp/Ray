#pragma once

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
    int depth;
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
    uint32_t _pad0 : 30;
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

force_inline float fract(const float v) {
    return v - floorf(v);
}

force_inline fvec4 srgb_to_rgb(const fvec4 &col) {
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

force_inline fvec2 srgb_to_rgb(const fvec2 &col) {
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
                                        const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                        const mesh_t *meshes, const tri_accel_t *tris, const uint32_t *tri_indices,
                                        hit_data_t &inter);
bool Traverse_TLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], uint32_t ray_flags,
                                        const wbvh_node_t *oct_nodes, uint32_t root_index,
                                        const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                        const mesh_t *meshes, const mtri_accel_t *mtris, const uint32_t *tri_indices,
                                        hit_data_t &inter);
// returns whether hit was solid
bool Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], int ray_type, const bvh_node_t *nodes,
                                    uint32_t root_index, const mesh_instance_t *mesh_instances,
                                    const uint32_t *mi_indices, const mesh_t *meshes, const mtri_accel_t *mtris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_TLAS_WithStack_AnyHit(const float ro[3], const float rd[3], int ray_type, const wbvh_node_t *nodes,
                                    uint32_t root_index, const mesh_instance_t *mesh_instances,
                                    const uint32_t *mi_indices, const mesh_t *meshes, const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t &inter);
// traditional bvh traversal with stack for inner nodes
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                        const bvh_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                        int obj_index, hit_data_t &inter);
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const float inv_d[3],
                                        const wbvh_node_t *nodes, uint32_t root_index, const mtri_accel_t *mtris,
                                        int obj_index, hit_data_t &inter);
// returns whether hit was solid
bool Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3], const bvh_node_t *nodes,
                                    uint32_t root_index, const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                    const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const float inv_d[3],
                                    const wbvh_node_t *nodes, uint32_t root_index, const tri_accel_t *tris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index,
                                    hit_data_t &inter);

// BRDFs
float BRDF_PrincipledDiffuse(const fvec4 &V, const fvec4 &N, const fvec4 &L, const fvec4 &H, float roughness);

fvec4 Evaluate_OrenDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L, float roughness,
                                const fvec4 &base_color);
fvec4 Sample_OrenDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, float roughness,
                              const fvec4 &base_color, fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_PrincipledDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L, float roughness,
                                      const fvec4 &base_color, const fvec4 &sheen_color, bool uniform_sampling);
fvec4 Sample_PrincipledDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, float roughness,
                                    const fvec4 &base_color, const fvec4 &sheen_color, bool uniform_sampling,
                                    fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_GGXSpecular_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts, const fvec4 &reflected_dir_ts,
                                fvec2 alpha, float spec_ior, float spec_F0, const fvec4 &spec_col,
                                const fvec4 &spec_col_90);
fvec4 Sample_GGXSpecular_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, fvec2 alpha,
                              float spec_ior, float spec_F0, const fvec4 &spec_col, const fvec4 &spec_col_90,
                              fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_GGXRefraction_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts, const fvec4 &refr_dir_ts,
                                  fvec2 slpha, float eta, const fvec4 &refr_col);
fvec4 Sample_GGXRefraction_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, fvec2 alpha, float eta,
                                const fvec4 &refr_col, fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_PrincipledClearcoat_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                        const fvec4 &reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior,
                                        float clearcoat_F0);
fvec4 Sample_PrincipledClearcoat_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                      float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0, fvec2 rand,
                                      fvec4 &out_V);

float Evaluate_EnvQTree(float y_rotation, const fvec4 *const *qtree_mips, int qtree_levels, const fvec4 &L);
fvec4 Sample_EnvQTree(float y_rotation, const fvec4 *const *qtree_mips, int qtree_levels, float rand, float rx,
                      float ry);

// Transform
void TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3], float out_rd[3]);
fvec4 TransformPoint(const fvec4 &p, const float *xform);
fvec4 TransformDirection(const fvec4 &p, const float *xform);
fvec4 TransformNormal(const fvec4 &n, const float *inv_xform);

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
void IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights, Span<const light_wbvh_node_t> nodes,
                         Span<hit_data_t> inout_inters);
void IntersectAreaLights(Span<const ray_data_t> rays, Span<const light_t> lights, Span<const light_bvh_node_t> nodes,
                         Span<hit_data_t> inout_inters);
float IntersectAreaLights(const shadow_ray_t &ray, Span<const light_t> lights, Span<const light_wbvh_node_t> nodes);
float EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                         Span<const light_bvh_node_t> nodes);
float EvalTriLightFactor(const fvec4 &P, const fvec4 &ro, uint32_t tri_index, Span<const light_t> lights,
                         Span<const light_wbvh_node_t> nodes);

void TraceRays(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth, const scene_data_t &sc,
               uint32_t node_index, bool trace_lights, const Cpu::TexStorageBase *const textures[],
               const uint32_t rand_seq[], uint32_t random_seed, int iteration, Span<hit_data_t> out_inter);
void TraceShadowRays(Span<const shadow_ray_t> rays, int max_transp_depth, float clamp_val, const scene_data_t &sc,
                     uint32_t node_index, const uint32_t rand_seq[], uint32_t random_seed, int iteration,
                     const Cpu::TexStorageBase *const textures[], int img_w, color_rgba_t *out_color);

// Get environment collor at direction
fvec4 Evaluate_EnvColor(const ray_data_t &ray, const environment_t &env, const Cpu::TexStorageRGBA &tex_storage,
                        float pdf_factor, const fvec2 &rand);
// Get light color at intersection point
fvec4 Evaluate_LightColor(const ray_data_t &ray, const hit_data_t &inter, const environment_t &env,
                          const Cpu::TexStorageRGBA &tex_storage, Span<const light_t> lights, uint32_t lights_count,
                          const fvec2 &rand);

// Evaluate individual nodes
fvec4 Evaluate_DiffuseNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                           const fvec4 &base_color, float roughness, float mix_weight, bool use_mis,
                           shadow_ray_t &sh_r);
void Sample_DiffuseNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                        fvec2 rand, float mix_weight, ray_data_t &new_ray);

fvec4 Evaluate_GlossyNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                          const fvec4 &base_color, float roughness, float regularize_alpha, float spec_ior,
                          float spec_F0, float mix_weight, bool use_mis, shadow_ray_t &sh_r);
void Sample_GlossyNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                       float regularize_alpha, float spec_ior, float spec_F0, fvec2 rand, float mix_weight,
                       ray_data_t &new_ray);

fvec4 Evaluate_RefractiveNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                              const fvec4 &base_color, float roughness, float regularize_alpha, float eta,
                              float mix_weight, bool use_mis, shadow_ray_t &sh_r);
void Sample_RefractiveNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                           float regularize_alpha, bool is_backfacing, float int_ior, float ext_ior, fvec2 rand,
                           float mix_weight, ray_data_t &new_ray);

struct diff_params_t {
    fvec4 base_color;
    fvec4 sheen_color;
    float roughness;
};

struct spec_params_t {
    fvec4 tmp_col;
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

struct lobe_weights_t {
    float diffuse, specular, clearcoat, refraction;
};

fvec4 Evaluate_PrincipledNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                              const lobe_weights_t &lobe_weights, const diff_params_t &diff, const spec_params_t &spec,
                              const clearcoat_params_t &coat, const transmission_params_t &trans, float metallic,
                              float transmission, float N_dot_L, float mix_weight, bool use_mis, float regularize_alpha,
                              shadow_ray_t &sh_r);
void Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t &ray, const surface_t &surf,
                           const lobe_weights_t &lobe_weights, const diff_params_t &diff, const spec_params_t &spec,
                           const clearcoat_params_t &coat, const transmission_params_t &trans, float metallic,
                           float transmission, fvec2 rand, float mix_rand, float mix_weight, float regularize_alpha,
                           ray_data_t &new_ray);

// Shade
color_rgba_t ShadeSurface(const pass_settings_t &ps, const float limits[2], const hit_data_t &inter,
                          const ray_data_t &ray, const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                          const scene_data_t &sc, uint32_t node_index, const Cpu::TexStorageBase *const textures[],
                          ray_data_t *out_secondary_rays, int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                          int *out_shadow_rays_count, color_rgba_t *out_base_color, color_rgba_t *out_depth_normal);
void ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                  const uint32_t rand_seq[], uint32_t rand_seed, int iteration, const scene_data_t &sc,
                  uint32_t node_index, const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                  int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count, int img_w,
                  float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                  color_rgba_t *out_depth_normal);
void ShadeSecondary(const pass_settings_t &ps, float clamp_direct, Span<const hit_data_t> inters,
                    Span<const ray_data_t> rays, const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                    const scene_data_t &sc, uint32_t node_index, const Cpu::TexStorageBase *const textures[],
                    ray_data_t *out_secondary_rays, int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                    int *out_shadow_rays_count, int img_w, color_rgba_t *out_color);

// Denoise
template <int WINDOW_SIZE = 7, int NEIGHBORHOOD_SIZE = 3>
void JointNLMFilter(const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
                    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight,
                    const color_rgba_t feature1[], float feature1_weight, const rect_t &output_rect, int output_stride,
                    color_rgba_t output[]);

template <int InChannels, int OutChannels, int OutPxPitch, ePostOp PostOp = ePostOp::None,
          eActivation Activation = eActivation::ReLU>
void Convolution3x3_Direct(const float data[], const rect_t &rect, int w, int h, int stride, const float weights[],
                           const float biases[], float output[], int output_stride);
template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None, ePostOp PostOp = ePostOp::None,
          eActivation Activation = eActivation::ReLU>
void Convolution3x3_GEMM(const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w,
                         int in_h, int w, int h, int stride, const float weights[], const float biases[],
                         float output[], int output_stride);

template <int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3_Direct(const float data1[], const float data2[], const rect_t &rect, int w, int h,
                                 int stride1, int stride2, const float weights[], const float biases[], float output[],
                                 int output_stride);
template <int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3_GEMM(const float data1[], const float data2[], const rect_t &rect, int w, int h,
                               const float weights[], const float biases[], float output[]);
template <int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
          ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
          ePreOp PreOp4 = ePreOp::None, ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3_1Direct_2GEMM(const float data1[], const float data2[], const float data3[],
                                        const float data4[], const rect_t &rect, int w, int h, int w2, int h2,
                                        int stride1, int stride2, const float weights[], const float biases[],
                                        float output[], int output_stride);
void ClearBorders(const rect_t &rect, int w, int h, bool downscaled, int out_channels, float output[]);

// Tonemap

// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
force_inline fvec4 vectorcall reversible_tonemap(const fvec4 c) {
    return c / (fmaxf(c.get<0>(), fmaxf(c.get<1>(), c.get<2>())) + 1.0f);
}

force_inline fvec4 vectorcall reversible_tonemap_invert(const fvec4 c) {
    return c / (1.0f - fmaxf(c.get<0>(), fmaxf(c.get<1>(), c.get<2>())));
}

struct tonemap_params_t {
    eViewTransform view_transform;
    float inv_gamma;
};

force_inline fvec4 vectorcall TonemapStandard(fvec4 c) {
    UNROLLED_FOR(i, 3, {
        if (c.get<i>() < 0.0031308f) {
            c.set<i>(12.92f * c.get<i>());
        } else {
            c.set<i>(1.055f * powf(c.get<i>(), (1.0f / 2.4f)) - 0.055f);
        }
    })
    return c;
}

fvec4 vectorcall TonemapFilmic(eViewTransform view_transform, fvec4 color);

force_inline fvec4 vectorcall Tonemap(const tonemap_params_t &params, fvec4 c) {
    if (params.view_transform == eViewTransform::Standard) {
        c = TonemapStandard(c);
    } else {
        c = TonemapFilmic(params.view_transform, c);
    }

    if (params.inv_gamma != 1.0f) {
        c = pow(c, fvec4{params.inv_gamma, params.inv_gamma, params.inv_gamma, 1.0f});
    }

    return saturate(c);
}

} // namespace Ref
} // namespace Ray
