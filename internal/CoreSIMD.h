// #pragma once
//  This file is compiled many times for different simd architectures (SSE, NEON...).
//  Macro 'NS' defines a namespace in which everything will be located, so it should be set before including this file.
//  Macros 'USE_XXX' define template instantiation of simd_fvec, simd_ivec classes.
//  Template parameter S defines width of vectors used. Usualy it is equal to ray packet size.

#include <vector>

#include <cfloat>

#include "TextureStorageRef.h"

#include "simd/simd_vec.h"

#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant

//
// Useful macros for debugging
//
#define USE_VNDF_GGX_SAMPLING 1
#define USE_NEE 1
#define USE_PATH_TERMINATION 1
// #define FORCE_TEXTURE_LOD 0
#define USE_SAFE_MATH 1

namespace Ray {
#ifndef RAY_EXCHANGE_DEFINED
template <class T, class U = T> T exchange(T &obj, U &&new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}
#define RAY_EXCHANGE_DEFINED
#endif

namespace Ref {
class TexStorageBase;
template <typename T, int N> class TexStorageLinear;
template <typename T, int N> class TexStorageTiled;
template <typename T, int N> class TexStorageSwizzled;
using TexStorageRGBA = TexStorageSwizzled<uint8_t, 4>;
using TexStorageRGB = TexStorageSwizzled<uint8_t, 3>;
using TexStorageRG = TexStorageSwizzled<uint8_t, 2>;
using TexStorageR = TexStorageSwizzled<uint8_t, 1>;
} // namespace Ref
namespace NS {
// Up to 4x4 rays
// [ 0] [ 1] [ 4] [ 5]
// [ 2] [ 3] [ 6] [ 7]
// [ 8] [ 9] [12] [13]
// [10] [11] [14] [15]
alignas(64) const int ray_packet_layout_x[] = {0, 1, 0, 1,  // NOLINT
                                               2, 3, 2, 3,  // NOLINT
                                               0, 1, 0, 1,  // NOLINT
                                               2, 3, 2, 3}; // NOLINT
alignas(64) const int ray_packet_layout_y[] = {0, 0, 1, 1,  // NOLINT
                                               0, 0, 1, 1,  // NOLINT
                                               2, 2, 3, 3,  // NOLINT
                                               2, 2, 3, 3}; // NOLINT

// Usefull to make index argument for a gather instruction
alignas(64) const int ascending_counter[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int S> struct ray_data_t {
    // origins of rays in packet
    simd_fvec<S> o[3];
    // directions of rays in packet
    simd_fvec<S> d[3], pdf;
    // throughput color of ray
    simd_fvec<S> c[3];
    // stack of ior values
    simd_fvec<S> ior[4];
    // ray cone params
    simd_fvec<S> cone_width, cone_spread;
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_ivec<S> xy;
    // four 8-bit ray depth counters
    simd_ivec<S> depth;
};

template <int S> struct shadow_ray_t {
    // origins of rays in packet
    simd_fvec<S> o[3];
    // four 8-bit ray depth counters
    simd_ivec<S> depth;
    // directions of rays in packet
    simd_fvec<S> d[3], dist;
    // throughput color of ray
    simd_fvec<S> c[3];
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_ivec<S> xy;
};

template <int S> struct hit_data_t {
    simd_ivec<S> mask;
    simd_ivec<S> obj_index;
    simd_ivec<S> prim_index;
    simd_fvec<S> t, u, v;

    explicit hit_data_t(eUninitialize) {}
    force_inline hit_data_t() {
        mask = {0};
        obj_index = {-1};
        prim_index = {-1};
        t = MAX_DIST;
        u = v = 0.0f;
    }
};

template <int S> struct surface_t {
    simd_fvec<S> P[3] = {0.0f, 0.0f, 0.0f}, T[3], B[3], N[3], plane_N[3];
    simd_fvec<S> uvs[2];

    force_inline surface_t() = default;
};

template <int S> struct light_sample_t {
    simd_fvec<S> col[3] = {0.0f, 0.0f, 0.0f}, L[3] = {0.0f, 0.0f, 0.0f}, lp[3] = {0.0f, 0.0f, 0.0f};
    simd_fvec<S> area = 0.0f, dist_mul = 1.0f, pdf = 0.0f;
    // TODO: merge these two into bitflags
    simd_ivec<S> cast_shadow = -1, from_env = 0;

    force_inline light_sample_t() = default;
};

template <int S> force_inline simd_ivec<S> total_depth(const ray_data_t<S> &r) {
    const simd_ivec<S> diff_depth = r.depth & 0x000000ff;
    const simd_ivec<S> spec_depth = (r.depth >> 8) & 0x000000ff;
    const simd_ivec<S> refr_depth = (r.depth >> 16) & 0x000000ff;
    const simd_ivec<S> transp_depth = (r.depth >> 24) & 0x000000ff;
    return diff_depth + spec_depth + refr_depth + transp_depth;
}

template <int S> force_inline int total_depth(const shadow_ray_t<S> &r) {
    const simd_ivec<S> diff_depth = r.depth & 0x000000ff;
    const simd_ivec<S> spec_depth = (r.depth >> 8) & 0x000000ff;
    const simd_ivec<S> refr_depth = (r.depth >> 16) & 0x000000ff;
    const simd_ivec<S> transp_depth = (r.depth >> 24) & 0x000000ff;
    return diff_depth + spec_depth + refr_depth + transp_depth;
}

// Generating rays
template <int DimX, int DimY>
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float random_seq[],
                         aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                         aligned_vector<simd_ivec<DimX * DimY>> &out_masks);
template <int DimX, int DimY>
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr,
                              const uint32_t *vtx_indices, const vertex_t *vertices, const rect_t &r, int w, int h,
                              const float random_seq[], aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                              aligned_vector<hit_data_t<DimX * DimY>> &out_inters);

// Sorting rays
template <int S>
void SortRays_CPU(ray_data_t<S> *rays, simd_ivec<S> *ray_masks, int &secondary_rays_count, const float root_min[3],
                  const float cell_size[3], simd_ivec<S> *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                  ray_chunk_t *chunks_temp);
template <int S>
void SortRays_GPU(ray_data_t<S> *rays, simd_ivec<S> *ray_masks, int &secondary_rays_count, const float root_min[3],
                  const float cell_size[3], simd_ivec<S> *hash_values, int *head_flags, uint32_t *scan_values,
                  ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

// Intersect primitives
template <int S>
bool IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                              const tri_accel_t *tris, uint32_t num_tris, int obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                              const tri_accel_t *tris, int tri_start, int tri_end, int obj_index,
                              hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_ClosestHit(const float o[3], const float d[3], int i, const tri_accel_t *tris, int tri_start,
                              int tri_end, int obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_ClosestHit(const float o[3], const float d[3], const mtri_accel_t *mtris, int tri_start, int tri_end,
                              int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v);
template <int S>
bool IntersectTris_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                          const tri_accel_t *tris, uint32_t num_tris, int obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                          const tri_accel_t *tris, int tri_start, int tri_end, int obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_AnyHit(const float o[3], const float d[3], int i, const tri_accel_t *tris,
                          const tri_mat_data_t *materials, const uint32_t *indices, int tri_start, int tri_end,
                          int obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris_AnyHit(const float o[3], const float d[3], const mtri_accel_t *mtris,
                          const tri_mat_data_t *materials, const uint32_t *indices, int tri_start, int tri_end,
                          int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v);

// Traverse acceleration structure
template <int S>
bool Traverse_MacroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                             const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                             const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                             const mesh_t *meshes, const transform_t *transforms,
                                             const tri_accel_t *tris, const uint32_t *tri_indices,
                                             hit_data_t<S> &inter);
template <int S>
bool Traverse_MacroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                             const simd_ivec<S> &ray_mask, const mbvh_node_t *mnodes,
                                             uint32_t node_index, const mesh_instance_t *mesh_instances,
                                             const uint32_t *mi_indices, const mesh_t *meshes,
                                             const transform_t *transforms, const mtri_accel_t *mtris,
                                             const uint32_t *tri_indices, hit_data_t<S> &inter);
template <int S>
simd_ivec<S>
Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                    const bvh_node_t *nodes, uint32_t node_index, const mesh_instance_t *mesh_instances,
                                    const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                    const tri_accel_t *tris, const tri_mat_data_t *materials,
                                    const uint32_t *tri_indices, hit_data_t<S> &inter);
template <int S>
simd_ivec<S>
Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                    const mbvh_node_t *mnodes, uint32_t node_index,
                                    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                    const mesh_t *meshes, const transform_t *transforms, const mtri_accel_t *mtris,
                                    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t<S> &inter);
// traditional bvh traversal with stack for inner nodes
template <int S>
bool Traverse_MicroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                             const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index,
                                             hit_data_t<S> &inter);
template <int S>
bool Traverse_MicroTree_WithStack_ClosestHit(const float ro[3], const float rd[3], const mbvh_node_t *mnodes,
                                             uint32_t node_index, const mtri_accel_t *mtris,
                                             const uint32_t *tri_indices, int &inter_prim_index, float &inter_t,
                                             float &inter_u, float &inter_v);
// returns 0 - no hit, 1 - hit, 2 - solid hit (no need to check for transparency)
template <int S>
simd_ivec<S> Traverse_MicroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                 const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                 uint32_t node_index, const tri_accel_t *tris,
                                                 const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                 int obj_index, hit_data_t<S> &inter);
template <int S>
int Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], const mbvh_node_t *mnodes,
                                        uint32_t node_index, const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                        const uint32_t *tri_indices, int &inter_prim_index, float &inter_t,
                                        float &inter_u, float &inter_v);

// BRDFs
template <int S>
simd_fvec<S> BRDF_PrincipledDiffuse(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                                    const simd_fvec<S> H[3], const simd_fvec<S> &roughness);

template <int S>
void Evaluate_OrenDiffuse_BSDF(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                               const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                               simd_fvec<S> out_color[4]);
template <int S>
void Sample_OrenDiffuse_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                             const simd_fvec<S> I[3], const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                             const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                             simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_PrincipledDiffuse_BSDF(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                                     const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                                     const simd_fvec<S> sheen_color[3], bool uniform_sampling,
                                     simd_fvec<S> out_color[4]);
template <int S>
void Sample_PrincipledDiffuse_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                   const simd_fvec<S> I[3], const simd_fvec<S> &roughness,
                                   const simd_fvec<S> base_color[3], const simd_fvec<S> sheen_color[3],
                                   bool uniform_sampling, const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                                   simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_GGXSpecular_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                               const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> &alpha_x,
                               const simd_fvec<S> &alpha_y, const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                               const simd_fvec<S> spec_col[3], simd_fvec<S> out_color[4]);
template <int S>
void Sample_GGXSpecular_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                             const simd_fvec<S> I[3], const simd_fvec<S> &roughness, const simd_fvec<S> &anisotropic,
                             const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0, const simd_fvec<S> spec_col[3],
                             const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                             simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_GGXRefraction_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                 const simd_fvec<S> refr_dir_ts[3], const simd_fvec<S> &roughness2,
                                 const simd_fvec<S> &eta, const simd_fvec<S> refr_col[3], simd_fvec<S> out_color[4]);
template <int S>
void Sample_GGXRefraction_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                               const simd_fvec<S> I[3], const simd_fvec<S> &roughness, const simd_fvec<S> &eta,
                               const simd_fvec<S> refr_col[3], const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                               simd_fvec<S> out_V[4], simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_PrincipledClearcoat_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                       const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> &clearcoat_roughness2,
                                       const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                       simd_fvec<S> out_color[4]);
template <int S>
void Sample_PrincipledClearcoat_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> I[3], const simd_fvec<S> &clearcoat_roughness2,
                                     const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                     const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                                     simd_fvec<S> out_color[4]);

template <int S>
simd_fvec<S> Evaluate_EnvQTree(float y_rotation, const simd_fvec4 *const *qtree_mips, int qtree_levels,
                               const simd_fvec<S> L[3]);
template <int S>
void Sample_EnvQTree(float y_rotation, const simd_fvec4 *const *qtree_mips, int qtree_levels, const simd_fvec<S> &rand,
                     const simd_fvec<S> &rx, const simd_fvec<S> &ry, simd_fvec<S> out_V[4]);

// Transform
template <int S>
void TransformRay(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const float *xform, simd_fvec<S> out_ro[3],
                  simd_fvec<S> out_rd[3]);
template <int S> void TransformPoint(const simd_fvec<S> p[3], const float *xform, simd_fvec<S> out_p[3]);
template <int S> void TransformDirection(const simd_fvec<S> xform[16], simd_fvec<S> p[3]);
template <int S> void TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]);
template <int S> void TransformNormal(const simd_fvec<S> n[3], const simd_fvec<S> inv_xform[16], simd_fvec<S> out_n[3]);
template <int S> void TransformNormal(const simd_fvec<S> inv_xform[16], simd_fvec<S> inout_n[3]);

void TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3], float out_rd[3]);

template <int S> void CanonicalToDir(const simd_fvec<S> p[2], float y_rotation, simd_fvec<S> out_d[3]);
template <int S> void DirToCanonical(const simd_fvec<S> d[3], float y_rotation, simd_fvec<S> out_p[2]);

template <int S>
void rotate_around_axis(const simd_fvec<S> p[3], const simd_fvec<S> axis[3], const simd_fvec<S> &angle,
                        simd_fvec<S> out_p[3]);

// Sample texture
template <int S>
void SampleNearest(const Ref::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                   const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Ref::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                    const simd_ivec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleTrilinear(const Ref::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                     const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleLatlong_RGBE(const Ref::TexStorageRGBA &storage, uint32_t index, const simd_fvec<S> dir[3], float y_rotation,
                        const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]);

// Trace rays through scene hierarchy
template <int S>
void IntersectScene(ray_data_t<S> &r, const simd_ivec<S> &ray_mask, int min_transp_depth, int max_transp_depth,
                    const float random_seq[], const scene_data_t &sc, uint32_t root_index,
                    const Ref::TexStorageBase *const textures[], hit_data_t<S> &inter);
template <int S>
void IntersectScene(const shadow_ray_t<S> &r, const simd_ivec<S> &mask, int max_transp_depth, const scene_data_t &sc,
                    uint32_t node_index, const Ref::TexStorageBase *const textures[], simd_fvec<S> rc[3]);

// Pick point on any light source for evaluation
template <int S>
void SampleLightSource(const simd_fvec<S> P[3], const simd_fvec<S> T[3], const simd_fvec<S> B[3],
                       const simd_fvec<S> N[3], const scene_data_t &sc, const Ref::TexStorageBase *const tex_atlases[],
                       const float random_seq[], const simd_ivec<S> &rand_index, const simd_fvec<S> sample_off[2],
                       const simd_ivec<S> &ray_mask, light_sample_t<S> &ls);

// Account for visible lights contribution
template <int S>
void IntersectAreaLights(const ray_data_t<S> &r, const simd_ivec<S> &ray_mask, const light_t lights[],
                         Span<const uint32_t> visible_lights, const transform_t transforms[],
                         hit_data_t<S> &inout_inter);
template <int S>
simd_fvec<S> IntersectAreaLights(const shadow_ray_t<S> &r, simd_ivec<S> ray_mask, const light_t lights[],
                                 Span<const uint32_t> blocker_lights, const transform_t transforms[]);

// Get environment collor at direction
template <int S>
void Evaluate_EnvColor(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const environment_t &env,
                       const Ref::TexStorageRGBA &tex_storage, simd_fvec<S> env_col[4]);
// Get light color at intersection point
template <int S>
void Evaluate_LightColor(const simd_fvec<S> P[3], const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                         const hit_data_t<S> &inter, const environment_t &env, const light_t *lights,
                         const Ref::TexStorageRGBA &tex_storage, simd_fvec<S> light_col[3]);

// Evaluate individual nodes
template <int S>
simd_ivec<S> Evaluate_DiffuseNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                  const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                  const simd_fvec<S> &roughness, const simd_fvec<S> &mix_weight,
                                  simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_DiffuseNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                        const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness, const simd_fvec<S> &rand_u,
                        const simd_fvec<S> &rand_v, const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray);

template <int S>
simd_ivec<S> Evaluate_GlossyNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                 const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                 const simd_fvec<S> &roughness, const simd_fvec<S> &spec_ior,
                                 const simd_fvec<S> &spec_F0, const simd_fvec<S> &mix_weight, simd_fvec<S> out_col[3],
                                 shadow_ray_t<S> &sh_r);
template <int S>
void Sample_GlossyNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                       const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness, const simd_fvec<S> &spec_ior,
                       const simd_fvec<S> &spec_F0, const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                       const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray);

template <int S>
simd_ivec<S> Evaluate_RefractiveNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                     const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                     const simd_fvec<S> &roughness2, const simd_fvec<S> &eta,
                                     const simd_fvec<S> &mix_weight, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_RefractiveNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                           const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                           const simd_ivec<S> &is_backfacing, const simd_fvec<S> &int_ior, const simd_fvec<S> &ext_ior,
                           const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, const simd_fvec<S> &mix_weight,
                           ray_data_t<S> &new_ray);

template <int S> struct diff_params_t {
    simd_fvec<S> base_color[3];
    simd_fvec<S> sheen_color[3];
    simd_fvec<S> roughness;
};

template <int S> struct spec_params_t {
    simd_fvec<S> tmp_col[3];
    simd_fvec<S> roughness;
    simd_fvec<S> ior;
    simd_fvec<S> F0;
    simd_fvec<S> anisotropy;
};

template <int S> struct clearcoat_params_t {
    simd_fvec<S> roughness;
    simd_fvec<S> ior;
    simd_fvec<S> F0;
};

template <int S> struct transmission_params_t {
    simd_fvec<S> roughness;
    simd_fvec<S> int_ior;
    simd_fvec<S> eta;
    simd_fvec<S> fresnel;
    simd_ivec<S> backfacing;
};

template <int S> struct lobe_weights_t {
    simd_fvec<S> diffuse, specular, clearcoat, refraction;
};

template <int S>
simd_ivec<S> Evaluate_PrincipledNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                     const surface_t<S> &surf, const lobe_weights_t<S> &lobe_weights,
                                     const diff_params_t<S> &diff, const spec_params_t<S> &spec,
                                     const clearcoat_params_t<S> &coat, const transmission_params_t<S> &trans,
                                     const simd_fvec<S> &metallic, const simd_fvec<S> &N_dot_L,
                                     const simd_fvec<S> &mix_weight, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                           const surface_t<S> &surf, const lobe_weights_t<S> &lobe_weights,
                           const diff_params_t<S> &diff, const spec_params_t<S> &spec,
                           const clearcoat_params_t<S> &coat, const transmission_params_t<S> &trans,
                           const simd_fvec<S> &metallic, const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                           simd_fvec<S> mix_rand, const simd_fvec<S> &mix_weight, simd_ivec<S> &secondary_mask,
                           ray_data_t<S> &new_ray);

// Shade
template <int S>
void ShadeSurface(const pass_settings_t &ps, const float *random_seq, const hit_data_t<S> &inter,
                  const ray_data_t<S> &ray, const scene_data_t &sc, uint32_t node_index,
                  const Ref::TexStorageBase *const tex_atlases[], simd_fvec<S> out_rgba[4],
                  simd_ivec<S> out_secondary_masks[], ray_data_t<S> out_secondary_rays[], int *out_secondary_rays_count,
                  simd_ivec<S> out_shadow_masks[], shadow_ray_t<S> out_shadow_rays[], int *out_shadow_rays_count);
} // namespace NS
} // namespace Ray

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace Ray {
namespace NS {
template <int S> force_inline simd_fvec<S> safe_inv(const simd_fvec<S> &a) {
#if USE_SAFE_MATH
    simd_fvec<S> denom = a;
    where(denom == 0.0f, denom) = FLT_EPS;
    return 1.0f / denom;
#else
    return 1.0f / a;
#endif
}

template <int S> force_inline simd_fvec<S> safe_inv_pos(const simd_fvec<S> &a) {
#if USE_SAFE_MATH
    return 1.0f / max(a, FLT_EPS);
#else
    return 1.0f / a;
#endif
}

template <int S> force_inline simd_fvec<S> safe_div(const simd_fvec<S> &a, const simd_fvec<S> &b) {
#if USE_SAFE_MATH
    simd_fvec<S> denom = b;
    where(denom == 0.0f, denom) = FLT_EPS;
    return a / denom;
#else
    return a / b;
#endif
}

template <int S> force_inline simd_fvec<S> safe_div_pos(const simd_fvec<S> &a, const simd_fvec<S> &b) {
#if USE_SAFE_MATH
    return a / max(b, FLT_EPS);
#else
    return a / b;
#endif
}

template <int S> force_inline simd_fvec<S> safe_div_pos(const float a, const simd_fvec<S> &b) {
#if USE_SAFE_MATH
    return a / max(b, FLT_EPS);
#else
    return a / b;
#endif
}

template <int S> force_inline simd_fvec<S> safe_div_pos(const simd_fvec<S> &a, const float b) {
#if USE_SAFE_MATH
    return a / std::max(b, FLT_EPS);
#else
    return a / b;
#endif
}

force_inline float safe_div_pos(const float a, const float b) {
#if USE_SAFE_MATH
    return a / std::max(b, FLT_EPS);
#else
    return a / b;
#endif
}

template <int S> force_inline simd_fvec<S> safe_div_neg(const simd_fvec<S> &a, const simd_fvec<S> &b) {
#if USE_SAFE_MATH
    return a / min(b, -FLT_EPS);
#else
    return a / b;
#endif
}

template <int S> force_inline simd_fvec<S> safe_sqrt(const simd_fvec<S> &a) {
#if USE_SAFE_MATH
    return sqrt(max(a, 0.0f));
#else
    return sqrt(a);
#endif
}

template <int S> force_inline void safe_normalize(simd_fvec<S> v[3]) {
    simd_fvec<S> l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
#if USE_SAFE_MATH
    const simd_fvec<S> mask = (l != 0.0f);
    where(~mask, l) = FLT_EPS;

    where(mask, v[0]) /= l;
    where(mask, v[1]) /= l;
    where(mask, v[2]) /= l;
#else
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
#endif
}

template <int S> force_inline simd_fvec<S> safe_sqrtf(const simd_fvec<S> &f) { return sqrt(max(f, 0.0f)); }

#define sqr(x) ((x) * (x))

template <typename T, int S>
force_inline void swap_elements(simd_vec<T, S> &v1, const int i1, simd_vec<T, S> &v2, const int i2) {
    const T temp = v1[i1];
    v1.set(i1, v2[i2]);
    v2.set(i2, temp);
}

#define _dot(x, y) ((x)[0] * (y)[0] + (x)[1] * (y)[1] + (x)[2] * (y)[2])

template <int S>
force_inline simd_ivec<S> IntersectTri(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                       const tri_accel_t &tri, uint32_t prim_index, hit_data_t<S> &inter) {
    const simd_fvec<S> det = _dot(rd, tri.n_plane);
    const simd_fvec<S> dett = tri.n_plane[3] - _dot(ro, tri.n_plane);

    const simd_ivec<S> imask = simd_cast(dett >= 0.0f) != simd_cast(det * inter.t - dett >= 0.0f);
    if (imask.all_zeros()) {
        return simd_ivec<S>{0};
    }

    const simd_fvec<S> p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1], det * ro[2] + dett * rd[2]};
    const simd_fvec<S> detu = _dot(p, tri.u_plane) + det * tri.u_plane[3];
    const simd_ivec<S> imask1 = simd_cast(detu >= 0.0f) != simd_cast(det - detu >= 0.0f);
    if (imask1.all_zeros()) {
        return simd_ivec<S>{0};
    }

    const simd_fvec<S> detv = _dot(p, tri.v_plane) + det * tri.v_plane[3];
    const simd_ivec<S> imask2 = simd_cast(detv >= 0.0f) != simd_cast(det - detu - detv >= 0.0f);
    if (imask2.all_zeros()) {
        return simd_ivec<S>{0};
    }

    const simd_fvec<S> rdet = 1.0f / det;
    const simd_fvec<S> t = dett * rdet;

    const simd_fvec<S> bar_u = detu * rdet;
    const simd_fvec<S> bar_v = detv * rdet;

    const simd_fvec<S> &fmask = simd_cast(imask);

    inter.mask = inter.mask | imask;

    where(imask, inter.prim_index) = simd_ivec<S>{reinterpret_cast<const int &>(prim_index)};
    where(fmask, inter.t) = t;
    where(fmask, inter.u) = bar_u;
    where(fmask, inter.v) = bar_v;

    return imask;
}

#undef _dot

template <int S>
force_inline bool IntersectTri(const float o[3], const float d[3], int i, const tri_accel_t &tri,
                               const uint32_t prim_index, hit_data_t<S> &inter) {
#define _sign_of(f) (((f) >= 0) ? 1 : -1)
#define _dot(x, y) ((x)[0] * (y)[0] + (x)[1] * (y)[1] + (x)[2] * (y)[2])

    const float det = _dot(d, tri.n_plane);
    const float dett = tri.n_plane[3] - _dot(o, tri.n_plane);
    if (_sign_of(dett) != _sign_of(det * inter.t[i] - dett)) {
        return false;
    }

    const float p[3] = {det * o[0] + dett * d[0], det * o[1] + dett * d[1], det * o[2] + dett * d[2]};

    const float detu = _dot(p, tri.u_plane) + det * tri.u_plane[3];
    if (_sign_of(detu) != _sign_of(det - detu)) {
        return false;
    }

    const float detv = _dot(p, tri.v_plane) + det * tri.v_plane[3];
    if (_sign_of(detv) != _sign_of(det - detu - detv)) {
        return false;
    }

    const float rdet = (1.0f / det);
    const float t = dett * rdet;

    if (t > 0.0f && t < inter.t[i]) {
        inter.mask[i] = 0xffffffff;
        inter.prim_index[i] = (det < 0.0f) ? int(prim_index) : -int(prim_index) - 1;
        inter.t[i] = t;
        inter.u[i] = detu * rdet;
        inter.v[i] = detv * rdet;

        return true;
    }

    return false;

#undef _dot
#undef _sign_of
}

template <int S>
bool IntersectTri(const float ro[3], const float rd[3], const mtri_accel_t &tri, const uint32_t prim_index,
                  int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v) {
    simd_ivec<S> _mask = 0, _prim_index;
    simd_fvec<S> _t = inter_t, _u, _v;

    for (int i = 0; i < 8; i += S) {
        const simd_fvec<S> det = rd[0] * simd_fvec<S>{&tri.n_plane[0][i], simd_mem_aligned} +
                                 rd[1] * simd_fvec<S>{&tri.n_plane[1][i], simd_mem_aligned} +
                                 rd[2] * simd_fvec<S>{&tri.n_plane[2][i], simd_mem_aligned};
        const simd_fvec<S> dett = simd_fvec<S>{&tri.n_plane[3][i], simd_mem_aligned} -
                                  ro[0] * simd_fvec<S>{&tri.n_plane[0][i], simd_mem_aligned} -
                                  ro[1] * simd_fvec<S>{&tri.n_plane[1][i], simd_mem_aligned} -
                                  ro[2] * simd_fvec<S>{&tri.n_plane[2][i], simd_mem_aligned};

        // compare sign bits
        simd_ivec<S> is_active_lane = ~srai(simd_cast(dett ^ (det * _t - dett)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        const simd_fvec<S> p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1], det * ro[2] + dett * rd[2]};

        const simd_fvec<S> detu = p[0] * simd_fvec<S>{&tri.u_plane[0][i], simd_mem_aligned} +
                                  p[1] * simd_fvec<S>{&tri.u_plane[1][i], simd_mem_aligned} +
                                  p[2] * simd_fvec<S>{&tri.u_plane[2][i], simd_mem_aligned} +
                                  det * simd_fvec<S>{&tri.u_plane[3][i], simd_mem_aligned};

        // compare sign bits
        is_active_lane &= ~srai(simd_cast(detu ^ (det - detu)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        const simd_fvec<S> detv = p[0] * simd_fvec<S>{&tri.v_plane[0][i], simd_mem_aligned} +
                                  p[1] * simd_fvec<S>{&tri.v_plane[1][i], simd_mem_aligned} +
                                  p[2] * simd_fvec<S>{&tri.v_plane[2][i], simd_mem_aligned} +
                                  det * simd_fvec<S>{&tri.v_plane[3][i], simd_mem_aligned};

        // compare sign bits
        is_active_lane &= ~srai(simd_cast(detv ^ (det - detu - detv)), 31);
        if (is_active_lane.all_zeros()) {
            continue;
        }

        const simd_fvec<S> rdet = safe_inv(det);

        simd_ivec<S> prim = -(int(prim_index) + simd_ivec<S>{&ascending_counter[i], simd_mem_aligned}) - 1;
        where(det < 0.0f, prim) = int(prim_index) + simd_ivec<S>{&ascending_counter[i], simd_mem_aligned};

        _mask |= is_active_lane;
        where(is_active_lane, _prim_index) = prim;
        where(is_active_lane, _t) = dett * rdet;
        where(is_active_lane, _u) = detu * rdet;
        where(is_active_lane, _v) = detv * rdet;
    }

    long mask = _mask.movemask();
    if (!mask) {
        return false;
    }

    const long i1 = GetFirstBit(mask);
    mask = ClearBit(mask, i1);

    long min_i = i1;
    inter_prim_index = _prim_index[i1];
    inter_t = _t[i1];
    inter_u = _u[i1];
    inter_v = _v[i1];

    if (mask == 0) { // Only one triangle was hit
        return true;
    }

    do {
        const long i2 = GetFirstBit(mask);
        mask = ClearBit(mask, i2);

        if (_t[i2] < _t[min_i]) {
            inter_prim_index = _prim_index[i2];
            inter_t = _t[i2];
            inter_u = _u[i2];
            inter_v = _v[i2];
            min_i = i2;
        }
    } while (mask != 0);

    return true;
}

template <>
bool IntersectTri<16>(const float ro[3], const float rd[3], const mtri_accel_t &tri, const uint32_t prim_index,
                      int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v) {
    simd_ivec<8> _mask = 0, _prim_index;
    simd_fvec<8> _t = inter_t, _u, _v;

    { // intersect 8 triangles
        const simd_fvec<8> det = rd[0] * simd_fvec<8>{&tri.n_plane[0][0], simd_mem_aligned} +
                                 rd[1] * simd_fvec<8>{&tri.n_plane[1][0], simd_mem_aligned} +
                                 rd[2] * simd_fvec<8>{&tri.n_plane[2][0], simd_mem_aligned};
        const simd_fvec<8> dett = simd_fvec<8>{&tri.n_plane[3][0], simd_mem_aligned} -
                                  ro[0] * simd_fvec<8>{&tri.n_plane[0][0], simd_mem_aligned} -
                                  ro[1] * simd_fvec<8>{&tri.n_plane[1][0], simd_mem_aligned} -
                                  ro[2] * simd_fvec<8>{&tri.n_plane[2][0], simd_mem_aligned};

        // compare sign bits
        simd_ivec<8> is_active_lane = ~srai(simd_cast(dett ^ (det * _t - dett)), 31);
        if (!is_active_lane.all_zeros()) {
            const simd_fvec<8> p[3] = {det * ro[0] + dett * rd[0], det * ro[1] + dett * rd[1],
                                       det * ro[2] + dett * rd[2]};

            const simd_fvec<8> detu = p[0] * simd_fvec<8>{&tri.u_plane[0][0], simd_mem_aligned} +
                                      p[1] * simd_fvec<8>{&tri.u_plane[1][0], simd_mem_aligned} +
                                      p[2] * simd_fvec<8>{&tri.u_plane[2][0], simd_mem_aligned} +
                                      det * simd_fvec<8>{&tri.u_plane[3][0], simd_mem_aligned};

            // compare sign bits
            is_active_lane &= ~srai(simd_cast(detu ^ (det - detu)), 31);
            if (!is_active_lane.all_zeros()) {
                const simd_fvec<8> detv = p[0] * simd_fvec<8>{&tri.v_plane[0][0], simd_mem_aligned} +
                                          p[1] * simd_fvec<8>{&tri.v_plane[1][0], simd_mem_aligned} +
                                          p[2] * simd_fvec<8>{&tri.v_plane[2][0], simd_mem_aligned} +
                                          det * simd_fvec<8>{&tri.v_plane[3][0], simd_mem_aligned};

                // compare sign bits
                is_active_lane &= ~srai(simd_cast(detv ^ (det - detu - detv)), 31);
                if (!is_active_lane.all_zeros()) {
                    const simd_fvec<8> rdet = safe_inv(det);

                    simd_ivec<8> prim = -(int(prim_index) + simd_ivec<8>{&ascending_counter[0], simd_mem_aligned}) - 1;
                    where(det < 0.0f, prim) = int(prim_index) + simd_ivec<8>{&ascending_counter[0], simd_mem_aligned};

                    _mask |= is_active_lane;
                    where(is_active_lane, _prim_index) = prim;
                    where(is_active_lane, _t) = dett * rdet;
                    where(is_active_lane, _u) = detu * rdet;
                    where(is_active_lane, _v) = detv * rdet;
                }
            }
        }
    }

    long mask = _mask.movemask();
    if (!mask) {
        return false;
    }

    const long i1 = GetFirstBit(mask);
    mask = ClearBit(mask, i1);

    long min_i = i1;
    inter_prim_index = _prim_index[i1];
    inter_t = _t[i1];
    inter_u = _u[i1];
    inter_v = _v[i1];

    if (mask == 0) { // Only one triangle was hit
        return true;
    }

    do {
        const long i2 = GetFirstBit(mask);
        mask = ClearBit(mask, i2);

        if (_t[i2] < _t[min_i]) {
            inter_prim_index = _prim_index[i2];
            inter_t = _t[i2];
            inter_u = _u[i2];
            inter_v = _v[i2];
            min_i = i2;
        }
    } while (mask != 0);

    return true;
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t,
                                    const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec<S> low, high, tmin, tmax;

    low = inv_d[0] * (_bbox_min[0] - o[0]);
    high = inv_d[0] * (_bbox_max[0] - o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = inv_d[1] * (_bbox_min[1] - o[1]);
    high = inv_d[1] * (_bbox_max[1] - o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = inv_d[2] * (_bbox_min[2] - o[2]);
    high = inv_d[2] * (_bbox_max[2] - o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024f;

    const simd_fvec<S> mask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
    return reinterpret_cast<const simd_ivec<S> &>(mask);
}

template <int S>
force_inline simd_ivec<S> bbox_test_fma(const simd_fvec<S> inv_d[3], const simd_fvec<S> inv_d_o[3],
                                        const simd_fvec<S> &t, const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec<S> low, high, tmin, tmax;

    low = fmsub(inv_d[0], _bbox_min[0], inv_d_o[0]);
    high = fmsub(inv_d[0], _bbox_max[0], inv_d_o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = fmsub(inv_d[1], _bbox_min[1], inv_d_o[1]);
    high = fmsub(inv_d[1], _bbox_max[1], inv_d_o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fmsub(inv_d[2], _bbox_min[2], inv_d_o[2]);
    high = fmsub(inv_d[2], _bbox_max[2], inv_d_o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024f;

    simd_fvec<S> mask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);

    return simd_cast(mask);
}

template <int S>
force_inline void bbox_test_oct(const float inv_d[3], const float inv_d_o[3], const float t,
                                const simd_fvec<S> bbox_min[3], const simd_fvec<S> bbox_max[3], simd_ivec<S> &out_mask,
                                simd_fvec<S> &out_dist) {
    simd_fvec<S> low, high, tmin, tmax;

    low = fmsub(inv_d[0], bbox_min[0], inv_d_o[0]);
    high = fmsub(inv_d[0], bbox_max[0], inv_d_o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = fmsub(inv_d[1], bbox_min[1], inv_d_o[1]);
    high = fmsub(inv_d[1], bbox_max[1], inv_d_o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fmsub(inv_d[2], bbox_min[2], inv_d_o[2]);
    high = fmsub(inv_d[2], bbox_max[2], inv_d_o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024f;

    const simd_fvec<S> fmask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
    out_mask = reinterpret_cast<const simd_ivec<S> &>(fmask);
    out_dist = tmin;
}

template <int S>
force_inline long bbox_test_oct(const float inv_d[3], const float inv_d_o[3], const float t, const float bbox_min[3][8],
                                const float bbox_max[3][8], float out_dist[8]) {
    simd_fvec<S> low, high, tmin, tmax;
    long res = 0;

    static const int LanesCount = (8 / S);

    ITERATE_R(LanesCount, {
        low = fmsub(inv_d[0], simd_fvec<S>{&bbox_min[0][S * i], simd_mem_aligned}, inv_d_o[0]);
        high = fmsub(inv_d[0], simd_fvec<S>{&bbox_max[0][S * i], simd_mem_aligned}, inv_d_o[0]);
        tmin = min(low, high);
        tmax = max(low, high);

        low = fmsub(inv_d[1], simd_fvec<S>{&bbox_min[1][S * i], simd_mem_aligned}, inv_d_o[1]);
        high = fmsub(inv_d[1], simd_fvec<S>{&bbox_max[1][S * i], simd_mem_aligned}, inv_d_o[1]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        low = fmsub(inv_d[2], simd_fvec<S>{&bbox_min[2][S * i], simd_mem_aligned}, inv_d_o[2]);
        high = fmsub(inv_d[2], simd_fvec<S>{&bbox_max[2][S * i], simd_mem_aligned}, inv_d_o[2]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));
        tmax *= 1.00000024f;

        const simd_fvec<S> fmask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
        res <<= S;
        res |= simd_cast(fmask).movemask();
        tmin.copy_to(&out_dist[S * i], simd_mem_aligned);
    })

    return res;
}

template <>
force_inline long bbox_test_oct<16>(const float inv_d[3], const float inv_d_o[3], const float t,
                                    const float bbox_min[3][8], const float bbox_max[3][8], float out_dist[8]) {
    simd_fvec<8> low = fmsub(inv_d[0], simd_fvec<8>{&bbox_min[0][0], simd_mem_aligned}, inv_d_o[0]);
    simd_fvec<8> high = fmsub(inv_d[0], simd_fvec<8>{&bbox_max[0][0], simd_mem_aligned}, inv_d_o[0]);
    simd_fvec<8> tmin = min(low, high);
    simd_fvec<8> tmax = max(low, high);

    low = fmsub(inv_d[1], simd_fvec<8>{&bbox_min[1][0], simd_mem_aligned}, inv_d_o[1]);
    high = fmsub(inv_d[1], simd_fvec<8>{&bbox_max[1][0], simd_mem_aligned}, inv_d_o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = fmsub(inv_d[2], simd_fvec<8>{&bbox_min[2][0], simd_mem_aligned}, inv_d_o[2]);
    high = fmsub(inv_d[2], simd_fvec<8>{&bbox_max[2][0], simd_mem_aligned}, inv_d_o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));
    tmax *= 1.00000024f;

    const simd_fvec<8> fmask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);

    long res = simd_cast(fmask).movemask();
    tmin.copy_to(&out_dist[0], simd_mem_aligned);

    return res;
}

template <int S>
force_inline void bbox_test_oct(const float p[3], const simd_fvec<S> bbox_min[3], const simd_fvec<S> bbox_max[3],
                                simd_ivec<S> &out_mask) {
    const simd_fvec<S> mask = (bbox_min[0] < p[0]) & (bbox_max[0] > p[0]) & (bbox_min[1] < p[1]) &
                              (bbox_max[1] > p[1]) & (bbox_min[2] < p[2]) & (bbox_max[2] > p[2]);
    out_mask = reinterpret_cast<const simd_ivec<S> &>(mask);
}

force_inline bool bbox_test(const float inv_d[3], const float inv_do[3], const float t, const float bbox_min[3],
                            const float bbox_max[3]) {
    float lo_x = inv_d[0] * bbox_min[0] - inv_do[0];
    float hi_x = inv_d[0] * bbox_max[0] - inv_do[0];
    if (lo_x > hi_x) {
        const float tmp = lo_x;
        lo_x = hi_x;
        hi_x = tmp;
    }

    float lo_y = inv_d[1] * bbox_min[1] - inv_do[1];
    float hi_y = inv_d[1] * bbox_max[1] - inv_do[1];
    if (lo_y > hi_y) {
        const float tmp = lo_y;
        lo_y = hi_y;
        hi_y = tmp;
    }

    float lo_z = inv_d[2] * bbox_min[2] - inv_do[2];
    float hi_z = inv_d[2] * bbox_max[2] - inv_do[2];
    if (lo_z > hi_z) {
        const float tmp = lo_z;
        lo_z = hi_z;
        hi_z = tmp;
    }

    float tmin = lo_x > lo_y ? lo_x : lo_y;
    if (lo_z > tmin) {
        tmin = lo_z;
    }
    float tmax = hi_x < hi_y ? hi_x : hi_y;
    if (hi_z < tmax) {
        tmax = hi_z;
    }
    tmax *= 1.00000024f;

    return tmin <= tmax && tmin <= t && tmax > 0;
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> p[3], const float _bbox_min[3], const float _bbox_max[3]) {
    const simd_fvec<S> mask = (p[0] > _bbox_min[0]) & (p[0] < _bbox_max[0]) & (p[1] > _bbox_min[1]) &
                              (p[1] < _bbox_max[1]) & (p[2] > _bbox_min[2]) & (p[2] < _bbox_max[2]);
    return reinterpret_cast<const simd_ivec<S> &>(mask);
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t,
                                    const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox_min, node.bbox_max);
}

template <int S>
force_inline simd_ivec<S> bbox_test_fma(const simd_fvec<S> inv_d[3], const simd_fvec<S> inv_d_o[3],
                                        const simd_fvec<S> &t, const bvh_node_t &node) {
    return bbox_test_fma(inv_d, inv_d_o, t, node.bbox_min, node.bbox_max);
}

template <int S> force_inline simd_ivec<S> bbox_test(const simd_fvec<S> p[3], const bvh_node_t &node) {
    return bbox_test(p, node.bbox_min, node.bbox_max);
}

template <int S>
force_inline uint32_t near_child(const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask, const bvh_node_t &node) {
    const simd_fvec<S> dir_neg_fmask = rd[node.prim_count >> 30] < 0.0f;
    const auto dir_neg_imask = reinterpret_cast<const simd_ivec<S> &>(dir_neg_fmask);
    if (dir_neg_imask.all_zeros(ray_mask)) {
        return node.left_child;
    } else {
        assert(and_not(dir_neg_imask, ray_mask).all_zeros());
        return (node.right_child & RIGHT_CHILD_BITS);
    }
}

force_inline uint32_t other_child(const bvh_node_t &node, uint32_t cur_child) {
    return node.left_child == cur_child ? (node.right_child & RIGHT_CHILD_BITS) : node.left_child;
}

force_inline bool is_leaf_node(const bvh_node_t &node) { return (node.prim_index & LEAF_NODE_BIT) != 0; }
force_inline bool is_leaf_node(const mbvh_node_t &node) { return (node.child[0] & LEAF_NODE_BIT) != 0; }

template <int S, int StackSize> struct TraversalStateStack_Multi {
    struct {
        simd_ivec<S> mask;
        uint32_t stack[StackSize];
        uint32_t stack_size;
    } queue[S];

    force_inline void push_children(const simd_fvec<S> rd[3], const bvh_node_t &node) {
        const simd_fvec<S> dir_neg_mask = rd[node.prim_count >> 30] < 0.0f;
        const auto mask1 = simd_cast(dir_neg_mask) & queue[index].mask;
        if (mask1.all_zeros()) {
            queue[index].stack[queue[index].stack_size++] = (node.right_child & RIGHT_CHILD_BITS);
            queue[index].stack[queue[index].stack_size++] = node.left_child;
        } else {
            const simd_ivec<S> mask2 = and_not(mask1, queue[index].mask);
            if (mask2.all_zeros()) {
                queue[index].stack[queue[index].stack_size++] = node.left_child;
                queue[index].stack[queue[index].stack_size++] = (node.right_child & RIGHT_CHILD_BITS);
            } else {
                queue[num].stack_size = queue[index].stack_size;
                memcpy(queue[num].stack, queue[index].stack, sizeof(uint32_t) * queue[index].stack_size);
                queue[num].stack[queue[num].stack_size++] = (node.right_child & RIGHT_CHILD_BITS);
                queue[num].stack[queue[num].stack_size++] = node.left_child;
                queue[num].mask = mask2;
                num++;
                queue[index].stack[queue[index].stack_size++] = node.left_child;
                queue[index].stack[queue[index].stack_size++] = (node.right_child & RIGHT_CHILD_BITS);
                queue[index].mask = mask1;
            }
        }
    }

    force_inline void push_children(const bvh_node_t &node) {
        queue[index].stack[queue[index].stack_size++] = node.left_child;
        queue[index].stack[queue[index].stack_size++] = (node.right_child & RIGHT_CHILD_BITS);
        assert(queue[index].stack_size < StackSize);
    }

    int index = 0, num = 1;
};

struct stack_entry_t {
    uint32_t index;
    float dist;
};

template <int StackSize> class TraversalStateStack_Single {
  public:
    stack_entry_t stack[StackSize];
    uint32_t stack_size = 0;

    force_inline void push(uint32_t index, float dist) {
        stack[stack_size++] = {index, dist};
        assert(stack_size < StackSize && "Traversal stack overflow!");
    }

    force_inline stack_entry_t pop() {
        return stack[--stack_size];
        assert(stack_size >= 0 && "Traversal stack underflow!");
    }

    force_inline uint32_t pop_index() { return stack[--stack_size].index; }

    force_inline bool empty() const { return stack_size == 0; }

    void sort_top3() {
        assert(stack_size >= 3);
        const uint32_t i = stack_size - 3;

        if (stack[i].dist > stack[i + 1].dist) {
            if (stack[i + 1].dist > stack[i + 2].dist) {
                return;
            } else if (stack[i].dist > stack[i + 2].dist) {
                std::swap(stack[i + 1], stack[i + 2]);
            } else {
                const stack_entry_t tmp = stack[i];
                stack[i] = stack[i + 2];
                stack[i + 2] = stack[i + 1];
                stack[i + 1] = tmp;
            }
        } else {
            if (stack[i].dist > stack[i + 2].dist) {
                std::swap(stack[i], stack[i + 1]);
            } else if (stack[i + 2].dist > stack[i + 1].dist) {
                std::swap(stack[i], stack[i + 2]);
            } else {
                const stack_entry_t tmp = stack[i];
                stack[i] = stack[i + 1];
                stack[i + 1] = stack[i + 2];
                stack[i + 2] = tmp;
            }
        }

        assert(stack[stack_size - 3].dist >= stack[stack_size - 2].dist &&
               stack[stack_size - 2].dist >= stack[stack_size - 1].dist);
    }

    void sort_top4() {
        assert(stack_size >= 4);
        const uint32_t i = stack_size - 4;

        if (stack[i + 0].dist < stack[i + 1].dist) {
            std::swap(stack[i + 0], stack[i + 1]);
        }
        if (stack[i + 2].dist < stack[i + 3].dist) {
            std::swap(stack[i + 2], stack[i + 3]);
        }
        if (stack[i + 0].dist < stack[i + 2].dist) {
            std::swap(stack[i + 0], stack[i + 2]);
        }
        if (stack[i + 1].dist < stack[i + 3].dist) {
            std::swap(stack[i + 1], stack[i + 3]);
        }
        if (stack[i + 1].dist < stack[i + 2].dist) {
            std::swap(stack[i + 1], stack[i + 2]);
        }

        assert(stack[stack_size - 4].dist >= stack[stack_size - 3].dist &&
               stack[stack_size - 3].dist >= stack[stack_size - 2].dist &&
               stack[stack_size - 2].dist >= stack[stack_size - 1].dist);
    }

    void sort_topN(int count) {
        assert(stack_size >= uint32_t(count));
        const int start = int(stack_size - count);

        for (int i = start + 1; i < int(stack_size); i++) {
            const stack_entry_t key = stack[i];

            int j = i - 1;

            while (j >= start && stack[j].dist < key.dist) {
                stack[j + 1] = stack[j];
                j--;
            }

            stack[j + 1] = key;
        }

#ifndef NDEBUG
        for (int j = 0; j < count - 1; j++) {
            assert(stack[stack_size - count + j].dist >= stack[stack_size - count + j + 1].dist);
        }
#endif
    }
};

template <int S>
force_inline void comp_aux_inv_values(const simd_fvec<S> o[3], const simd_fvec<S> d[3], simd_fvec<S> inv_d[3],
                                      simd_fvec<S> inv_d_o[3]) {
    for (int i = 0; i < 3; i++) {
        simd_fvec<S> denom = d[i];
        where(denom == 0.0f, denom) = FLT_EPS;

        inv_d[i] = 1.0f / denom;
        inv_d_o[i] = o[i] * inv_d[i];

        const simd_fvec<S> d_is_plus_zero = (d[i] <= FLT_EPS) & (d[i] >= 0);
        where(d_is_plus_zero, inv_d[i]) = MAX_DIST;
        where(d_is_plus_zero, inv_d_o[i]) = MAX_DIST;

        const simd_fvec<S> d_is_minus_zero = (d[i] >= -FLT_EPS) & (d[i] < 0);
        where(d_is_minus_zero, inv_d[i]) = -MAX_DIST;
        where(d_is_minus_zero, inv_d_o[i]) = -MAX_DIST;
    }
}

force_inline void comp_aux_inv_values(const float o[3], const float d[3], float inv_d[3], float inv_d_o[3]) {
    if (d[0] <= FLT_EPS && d[0] >= 0) {
        inv_d[0] = MAX_DIST;
        inv_d_o[0] = MAX_DIST;
    } else if (d[0] >= -FLT_EPS && d[0] < 0) {
        inv_d[0] = -MAX_DIST;
        inv_d_o[0] = -MAX_DIST;
    } else {
        inv_d[0] = 1.0f / d[0];
        inv_d_o[0] = inv_d[0] * o[0];
    }

    if (d[1] <= FLT_EPS && d[1] >= 0) {
        inv_d[1] = MAX_DIST;
        inv_d_o[1] = MAX_DIST;
    } else if (d[1] >= -FLT_EPS && d[1] < 0) {
        inv_d[1] = -MAX_DIST;
        inv_d_o[1] = -MAX_DIST;
    } else {
        inv_d[1] = 1.0f / d[1];
        inv_d_o[1] = inv_d[1] * o[1];
    }

    if (d[2] <= FLT_EPS && d[2] >= 0) {
        inv_d[2] = MAX_DIST;
        inv_d_o[2] = MAX_DIST;
    } else if (d[2] >= -FLT_EPS && d[2] < 0) {
        inv_d[2] = -MAX_DIST;
        inv_d_o[2] = -MAX_DIST;
    } else {
        inv_d[2] = 1.0f / d[2];
        inv_d_o[2] = inv_d[2] * o[2];
    }
}

template <int S> force_inline simd_fvec<S> dot3(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S> force_inline simd_fvec<S> dot3(const simd_fvec<S> v1[3], const float v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S> force_inline simd_fvec<S> dot3(const float v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

force_inline float dot3(const float v1[3], const float v2[3]) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }

template <int S> force_inline void cross(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3], simd_fvec<S> res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template <int S> force_inline void cross(const simd_fvec<S> &v1, const simd_fvec<S> &v2, simd_fvec<S> &res) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

force_inline void cross(const float v1[3], const float v2[3], float res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template <int S> force_inline void normalize(simd_fvec<S> v[3]) {
    const simd_fvec<S> l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
}

force_inline void normalize(float v[3]) {
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
}

template <int S> force_inline simd_fvec<S> length2(const simd_fvec<S> v[3]) {
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <int S> force_inline simd_fvec<S> length(const simd_fvec<S> v[3]) { return sqrt(length2(v)); }

template <int S> force_inline simd_fvec<S> length2_2d(const simd_fvec<S> v[2]) { return v[0] * v[0] + v[1] * v[1]; }

template <int S> force_inline simd_fvec<S> distance(const simd_fvec<S> p1[3], const simd_fvec<S> p2[3]) {
    const simd_fvec<S> temp[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    return length(temp);
}

template <int S> force_inline simd_fvec<S> clamp(const simd_fvec<S> &v, float min, float max) {
    simd_fvec<S> ret = v;
    where(ret < min, ret) = min;
    where(ret > max, ret) = max;
    return ret;
}

template <int S> force_inline simd_ivec<S> clamp(const simd_ivec<S> &v, int min, int max) {
    simd_ivec<S> ret = v;
    where(ret < min, ret) = min;
    where(ret > max, ret) = max;
    return ret;
}

force_inline int hash(int x) {
    unsigned ret = reinterpret_cast<const unsigned &>(x);
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return reinterpret_cast<const int &>(ret);
}

template <int S> force_inline simd_ivec<S> hash(const simd_ivec<S> &x) {
    simd_ivec<S> ret;
    ret = ((x >> 16) ^ x) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return ret;
}

force_inline float floor(float x) { return float(int(x) - (x < 0.0f)); }

template <int S>
force_inline void reflect(const simd_fvec<S> I[3], const simd_fvec<S> N[3], const simd_fvec<S> &dot_N_I,
                          simd_fvec<S> res[3]) {
    res[0] = I[0] - 2.0f * dot_N_I * N[0];
    res[1] = I[1] - 2.0f * dot_N_I * N[1];
    res[2] = I[2] - 2.0f * dot_N_I * N[2];
}

force_inline float pow5(const float v) { return (v * v) * (v * v) * v; }

force_inline void TransformDirection(const float d[3], const float *xform, float out_d[3]) {
    out_d[0] = xform[0] * d[0] + xform[4] * d[1] + xform[8] * d[2];
    out_d[1] = xform[1] * d[0] + xform[5] * d[1] + xform[9] * d[2];
    out_d[2] = xform[2] * d[0] + xform[6] * d[1] + xform[10] * d[2];
}

template <int S> force_inline simd_fvec<S> pow5(const simd_fvec<S> &v) { return (v * v) * (v * v) * v; }

template <int S>
simd_ivec<S> get_ray_hash(const ray_data_t<S> &r, const simd_ivec<S> &mask, const float root_min[3],
                          const float cell_size[3]) {
    simd_ivec<S> x = clamp((simd_ivec<S>)((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
                 y = clamp((simd_ivec<S>)((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
                 z = clamp((simd_ivec<S>)((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    simd_ivec<S> omega_index = clamp((simd_ivec<S>)((1.0f + r.d[2]) / omega_step), 0, 32),
                 phi_index_i = clamp((simd_ivec<S>)((1.0f + r.d[1]) / phi_step), 0, 16),
                 phi_index_j = clamp((simd_ivec<S>)((1.0f + r.d[0]) / phi_step), 0, 16);

    simd_ivec<S> o, p;

    UNROLLED_FOR_S(i, S, {
        if (mask[i]) {
            x.template set<i>(morton_table_256[x.template get<i>()]);
            y.template set<i>(morton_table_256[y.template get<i>()]);
            z.template set<i>(morton_table_256[z.template get<i>()]);
            o.template set<i>(morton_table_16[int(omega_table[omega_index.template get<i>()])]);
            p.template set<i>(
                morton_table_16[int(phi_table[phi_index_i.template get<i>()][phi_index_j.template get<i>()])]);
        } else {
            o.template set<i>(0xFFFFFFFF);
            p.template set<i>(0xFFFFFFFF);
            x.template set<i>(0xFFFFFFFF);
            y.template set<i>(0xFFFFFFFF);
            z.template set<i>(0xFFFFFFFF);
        }
    });

    return (o << 25) | (p << 24) | (y << 2) | (z << 1) | (x << 0);
}

void _radix_sort_lsb(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1, unsigned maxshift) {
    ray_chunk_t *end1 = begin1 + (end - begin);

    for (unsigned shift = 0; shift <= maxshift; shift += 8) {
        size_t count[0x100] = {};
        for (ray_chunk_t *p = begin; p != end; p++) {
            count[(p->hash >> shift) & 0xFF]++;
        }
        ray_chunk_t *bucket[0x100], *q = begin1;
        for (int i = 0; i < 0x100; q += count[i++]) {
            bucket[i] = q;
        }
        for (ray_chunk_t *p = begin; p != end; p++) {
            *bucket[(p->hash >> shift) & 0xFF]++ = *p;
        }
        std::swap(begin, begin1);
        std::swap(end, end1);
    }
}

force_inline void radix_sort(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1) {
    _radix_sort_lsb(begin, end, begin1, 24);
}

template <int S> force_inline simd_fvec<S> construct_float(const simd_ivec<S> &_m) {
    const simd_ivec<S> ieeeMantissa = {0x007FFFFF}; // binary32 mantissa bitmask
    const simd_ivec<S> ieeeOne = {0x3F800000};      // 1.0 in IEEE binary32

    simd_ivec<S> m = _m & ieeeMantissa; // Keep only mantissa bits (fractional part)
    m = m | ieeeOne;                    // Add fractional part to 1.0

    const simd_fvec<S> f = simd_cast(m); // Range [1:2]
    return f - simd_fvec<S>{1.0f};       // Range [0:1]
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
    return log_2;
}

template <int S> force_inline simd_fvec<S> fast_log2(const simd_fvec<S> &val) {
    // From https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c
    union {
        simd_fvec<S> val;
        simd_ivec<S> x;
    } u = {val};
    simd_fvec<S> log_2 = simd_fvec<S>(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
    return log_2;
}
template <int S> force_inline simd_fvec<S> lum(const simd_fvec<S> color[3]) {
    return 0.212671f * color[0] + 0.715160f * color[1] + 0.072169f * color[2];
}

template <int S> force_inline void srgb_to_rgb(const simd_fvec<S> in_col[4], simd_fvec<S> out_col[4]) {
    UNROLLED_FOR(i, 3, {
        simd_fvec<S> temp = in_col[i] / 12.92f;
        where(in_col[i] > 0.04045f, temp) = pow((in_col[i] + 0.055f) / 1.055f, 2.4f);
        out_col[i] = temp;
    })
    out_col[3] = in_col[3];
}

template <int S>
simd_fvec<S> get_texture_lod(const Ref::TexStorageBase *textures[], const uint32_t index, const simd_fvec<S> duv_dx[2],
                             const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD
    const simd_fvec<S> lod = float(FORCE_TEXTURE_LOD);
#else
    float sz[2];
    textures[index >> 28]->GetFRes(int(index & 0x00ffffff), 0, sz);

    const simd_fvec<S> _duv_dx[2] = {duv_dx[0] * sz[0], duv_dx[1] * sz[1]};
    const simd_fvec<S> _duv_dy[2] = {duv_dy[0] * sz[0], duv_dy[1] * sz[1]};

    const simd_fvec<S> _diagonal[2] = {_duv_dx[0] + _duv_dy[0], _duv_dx[1] + _duv_dy[1]};

    const simd_fvec<S> dim = min(min(length2_2d(_duv_dx), length2_2d(_duv_dy)), length2_2d(_diagonal));

    simd_fvec<S> lod = 0.5f * fast_log2(dim) - 1.0f;

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);
#endif
    return lod;
}

template <int S>
simd_fvec<S> get_texture_lod(const Ref::TexStorageBase *const textures[], const uint32_t index,
                             const simd_fvec<S> &lambda, const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD
    const simd_fvec<S> lod = float(FORCE_TEXTURE_LOD);
#else
    float sz[2];
    textures[index >> 28]->GetFRes(int(index & 0x00ffffff), 0, sz);

    simd_fvec<S> lod = 0.0f;

    UNROLLED_FOR_S(i, S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask).template get<i>()) {
            lod.template set<i>(lambda.template get<i>() + 0.5f * fast_log2(sz[0] * sz[1]) - 1.0f);
        }
    })

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);
#endif
    return lod;
}

template <int S>
simd_fvec<S> get_texture_lod(const simd_ivec<S> &width, const simd_ivec<S> &height, const simd_fvec<S> duv_dx[2],
                             const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD
    const simd_fvec<S> lod = float(FORCE_TEXTURE_LOD);
#else
    const simd_fvec<S> _duv_dx[2] = {duv_dx[0] * simd_fvec<S>(width), duv_dx[1] * simd_fvec<S>(height)};
    const simd_fvec<S> _duv_dy[2] = {duv_dy[0] * simd_fvec<S>(width), duv_dy[1] * simd_fvec<S>(height)};

    const simd_fvec<S> _diagonal[2] = {_duv_dx[0] + _duv_dy[0], _duv_dx[1] + _duv_dy[1]};

    const simd_fvec<S> dim = min(min(length2_2d(_duv_dx), length2_2d(_duv_dy)), length2_2d(_diagonal));

    simd_fvec<S> lod = 0.0f;

    UNROLLED_FOR_S(i, S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask).template get<i>()) {
            lod.template set<i>(0.5f * fast_log2(dim.template get<i>()) - 1.0f);
        }
    })

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);
#endif
    return lod;
}

template <int S>
simd_fvec<S> get_texture_lod(const simd_ivec<S> &width, const simd_ivec<S> &height, const simd_fvec<S> &lambda,
                             const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD
    const simd_fvec<S> lod = float(FORCE_TEXTURE_LOD);
#else
    simd_fvec<S> lod;

    UNROLLED_FOR_S(i, S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask).template get<i>()) {
            lod[i] = lambda.template get<i>() + 0.5f * fast_log2(width * height) - 1.0f;
        } else {
            lod[i] = 0.0f;
        }
    })

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);
#endif
    return lod;
}

template <int S> force_inline simd_fvec<S> conv_unorm_16(const simd_ivec<S> &v) { return simd_fvec<S>(v) / 65535.0f; }

template <int S>
void FetchTransformAndRecalcBasis(const transform_t *sc_transforms, const simd_ivec<S> &tr_index,
                                  const simd_fvec<S> P_ls[3], simd_fvec<S> inout_plane_N[3], simd_fvec<S> inout_N[3],
                                  simd_fvec<S> inout_B[3], simd_fvec<S> inout_T[3], simd_fvec<S> inout_tangent[3],
                                  simd_fvec<S> out_transform[16]) {
    const float *transforms = &sc_transforms[0].xform[0];
    const float *inv_transforms = &sc_transforms[0].inv_xform[0];
    const int TransformsStride = sizeof(transform_t) / sizeof(float);

    simd_fvec<S> inv_transform[16];
    UNROLLED_FOR(i, 16, {
        out_transform[i] = gather(transforms + i, tr_index * TransformsStride);
        inv_transform[i] = gather(inv_transforms + i, tr_index * TransformsStride);
    })

    simd_fvec<S> temp[3];
    cross(inout_tangent, inout_N, temp);
    const simd_fvec<S> mask = length2(temp) == 0.0f;
    UNROLLED_FOR(i, 3, { where(mask, inout_tangent[i]) = P_ls[i]; })

    TransformNormal(inv_transform, inout_plane_N);
    TransformNormal(inv_transform, inout_N);
    TransformNormal(inv_transform, inout_B);
    TransformNormal(inv_transform, inout_T);

    TransformNormal(inv_transform, inout_tangent);
}

template <int S>
void FetchVertexAttribute3(const float *attribs, const simd_ivec<S> vtx_indices[3], const simd_fvec<S> &u,
                           const simd_fvec<S> &v, const simd_fvec<S> &w, simd_fvec<S> out_A[3]) {
    static const int VtxStride = sizeof(vertex_t) / sizeof(float);

    const simd_fvec<S> A1[3] = {gather(attribs + 0, vtx_indices[0] * VtxStride),
                                gather(attribs + 1, vtx_indices[0] * VtxStride),
                                gather(attribs + 2, vtx_indices[0] * VtxStride)};
    const simd_fvec<S> A2[3] = {gather(attribs + 0, vtx_indices[1] * VtxStride),
                                gather(attribs + 1, vtx_indices[1] * VtxStride),
                                gather(attribs + 2, vtx_indices[1] * VtxStride)};
    const simd_fvec<S> A3[3] = {gather(attribs + 0, vtx_indices[2] * VtxStride),
                                gather(attribs + 1, vtx_indices[2] * VtxStride),
                                gather(attribs + 2, vtx_indices[2] * VtxStride)};

    UNROLLED_FOR(i, 3, { out_A[i] = A1[i] * w + A2[i] * u + A3[i] * v; })
}

template <int S>
void EnsureValidReflection(const simd_fvec<S> Ng[3], const simd_fvec<S> I[3], simd_fvec<S> inout_N[3]) {
    simd_fvec<S> R[3];
    UNROLLED_FOR(i, 3, { R[i] = 2.0f * dot3(inout_N, I) * inout_N[i] - I[i]; })

    // Reflection rays may always be at least as shallow as the incoming ray.
    const simd_fvec<S> threshold = min(0.9f * dot3(Ng, I), 0.01f);

    const simd_ivec<S> early_mask = simd_cast(dot3(Ng, R) < threshold);
    if (early_mask.all_zeros()) {
        return;
    }

    // Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
    // The X axis is found by normalizing the component of N that's orthogonal to Ng.
    // The Y axis isn't actually needed.
    const simd_fvec<S> NdotNg = dot3(inout_N, Ng);

    simd_fvec<S> X[3];
    UNROLLED_FOR(i, 3, { X[i] = inout_N[i] - NdotNg * Ng[i]; })
    safe_normalize(X);

    const simd_fvec<S> Ix = dot3(I, X), Iz = dot3(I, Ng);
    const simd_fvec<S> Ix2 = sqr(Ix), Iz2 = sqr(Iz);
    const simd_fvec<S> a = Ix2 + Iz2;

    const simd_fvec<S> b = safe_sqrtf(Ix2 * (a - (threshold * threshold)));
    const simd_fvec<S> c = Iz * threshold + a;

    // Evaluate both solutions.
    // In many cases one can be immediately discarded (if N'.z would be imaginary or larger than
    // one), so check for that first. If no option is viable (might happen in extreme cases like N
    // being in the wrong hemisphere), give up and return Ng.
    const simd_fvec<S> fac = safe_div(simd_fvec<S>{0.5f}, a);
    const simd_fvec<S> N1_z2 = fac * (b + c), N2_z2 = fac * (-b + c);

    simd_ivec<S> valid1 = simd_cast((N1_z2 > 1e-5f) & (N1_z2 <= (1.0f + 1e-5f)));
    simd_ivec<S> valid2 = simd_cast((N2_z2 > 1e-5f) & (N2_z2 <= (1.0f + 1e-5f)));

    simd_fvec<S> N_new[2];

    if ((valid1 & valid2).not_all_zeros()) {
        // If both are possible, do the expensive reflection-based check.
        const simd_fvec<S> N1[2] = {safe_sqrtf(1.0f - N1_z2), safe_sqrtf(N1_z2)};
        const simd_fvec<S> N2[2] = {safe_sqrtf(1.0f - N2_z2), safe_sqrtf(N2_z2)};

        const simd_fvec<S> R1 = 2 * (N1[0] * Ix + N1[1] * Iz) * N1[1] - Iz;
        const simd_fvec<S> R2 = 2 * (N2[0] * Ix + N2[1] * Iz) * N2[1] - Iz;

        valid1 = simd_cast(R1 >= 1e-5f);
        valid2 = simd_cast(R2 >= 1e-5f);

        const simd_ivec<S> mask = valid1 & valid2;

        const simd_ivec<S> mask1 = mask & simd_cast(R1 < R2);
        UNROLLED_FOR(i, 2, { where(mask1, N_new[i]) = N1[i]; })
        const simd_ivec<S> mask2 = mask & ~simd_cast(R1 < R2);
        UNROLLED_FOR(i, 2, { where(mask2, N_new[i]) = N2[i]; })

        const simd_ivec<S> mask3 = ~mask & simd_cast(R1 > R2);
        UNROLLED_FOR(i, 2, { where(mask3, N_new[i]) = N1[i]; })
        const simd_ivec<S> mask4 = ~mask & ~simd_cast(R1 > R2);
        UNROLLED_FOR(i, 2, { where(mask4, N_new[i]) = N2[i]; })
    }

    if ((valid1 | valid2).not_all_zeros()) {
        const simd_ivec<S> exclude = ~(valid1 & valid2);

        // Only one solution passes the N'.z criterium, so pick that one.
        simd_fvec<S> Nz2 = N2_z2;
        where(valid1, Nz2) = N1_z2;

        where(exclude & (valid1 | valid2), N_new[0]) = safe_sqrtf(1.0f - Nz2);
        where(exclude & (valid1 | valid2), N_new[1]) = safe_sqrtf(Nz2);
    }

    UNROLLED_FOR(i, 3, {
        where(early_mask, inout_N[i]) = N_new[0] * X[i] + N_new[1] * Ng[i];
        where(early_mask & ~valid1 & ~valid2, inout_N[i]) = Ng[i];
    })
}

template <int S>
force_inline void world_from_tangent(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> V[3], simd_fvec<S> out_V[3]) {
    UNROLLED_FOR(i, 3, { out_V[i] = V[0] * T[i] + V[1] * B[i] + V[2] * N[i]; })
}

template <int S>
force_inline void tangent_from_world(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> V[3], simd_fvec<S> out_V[3]) {
    out_V[0] = dot3(V, T);
    out_V[1] = dot3(V, B);
    out_V[2] = dot3(V, N);
}

template <int S> force_inline simd_fvec<S> cos(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, { ret.template set<i>(std::cos(v.template get<i>())); })
    return ret;
}

template <int S> force_inline simd_fvec<S> sin(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, { ret.template set<i>(std::sin(v.template get<i>())); })
    return ret;
}

//
// From "A Fast and Robust Method for Avoiding Self-Intersection"
//
template <int S> void offset_ray(const simd_fvec<S> p[3], const simd_fvec<S> n[3], simd_fvec<S> out_p[3]) {
    static const float Origin = 1.0f / 32.0f;
    static const float FloatScale = 1.0f / 65536.0f;
    static const float IntScale = 128.0f; // 256.0f;

    simd_ivec<S> of_i[3] = {simd_ivec<S>{IntScale * n[0]}, simd_ivec<S>{IntScale * n[1]},
                            simd_ivec<S>{IntScale * n[2]}};
    UNROLLED_FOR(i, 3, { where(p[i] < 0.0f, of_i[i]) = -of_i[i]; })

    const simd_fvec<S> p_i[3] = {simd_cast(simd_cast(p[0]) + of_i[0]), simd_cast(simd_cast(p[1]) + of_i[1]),
                                 simd_cast(simd_cast(p[2]) + of_i[2])};

    UNROLLED_FOR(i, 3, {
        out_p[i] = p_i[i];
        where(abs(p[i]) < Origin, out_p[i]) = fmadd(simd_fvec<S>{FloatScale}, n[i], p[i]);
    })
}

// http://jcgt.org/published/0007/04/01/paper.pdf by Eric Heitz
// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
template <int S>
void SampleGGX_VNDF(const simd_fvec<S> Ve[3], const simd_fvec<S> &alpha_x, const simd_fvec<S> &alpha_y,
                    const simd_fvec<S> &U1, const simd_fvec<S> &U2, simd_fvec<S> out_V[3]) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    simd_fvec<S> Vh[3] = {alpha_x * Ve[0], alpha_y * Ve[1], Ve[2]};
    safe_normalize(Vh);
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    const simd_fvec<S> lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];

    simd_fvec<S> T1[3] = {{1.0f}, {0.0f}, {0.0f}};
    const simd_fvec<S> denom = safe_sqrt(lensq);
    where(lensq > 0.0f, T1[0]) = -safe_div_pos(Vh[1], denom);
    where(lensq > 0.0f, T1[1]) = safe_div_pos(Vh[0], denom);

    simd_fvec<S> T2[3];
    cross(Vh, T1, T2);
    // Section 4.2: parameterization of the projected area
    const simd_fvec<S> r = sqrt(U1);
    const simd_fvec<S> phi = 2.0f * PI * U2;
    simd_fvec<S> t1;
    UNROLLED_FOR_S(i, S, { t1.template set<i>(r.template get<i>() * std::cos(phi.template get<i>())); })
    simd_fvec<S> t2;
    UNROLLED_FOR_S(i, S, { t2.template set<i>(r.template get<i>() * std::sin(phi.template get<i>())); })
    const simd_fvec<S> s = 0.5f * (1.0f + Vh[2]);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    simd_fvec<S> Nh[3];
    UNROLLED_FOR(i, 3, { Nh[i] = t1 * T1[i] + t2 * T2[i] + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh[i]; })
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    out_V[0] = alpha_x * Nh[0];
    out_V[1] = alpha_y * Nh[1];
    out_V[2] = max(0.0f, Nh[2]);
    safe_normalize(out_V);
}

// Smith shadowing function
template <int S> force_inline simd_fvec<S> G1(const simd_fvec<S> Ve[3], simd_fvec<S> alpha_x, simd_fvec<S> alpha_y) {
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    const simd_fvec<S> delta =
        (-1.0f + safe_sqrt(1.0f + safe_div(alpha_x * Ve[0] * Ve[0] + alpha_y * Ve[1] * Ve[1], Ve[2] * Ve[2]))) / 2.0f;
    return 1.0f / (1.0f + delta);
}

template <int S> simd_fvec<S> D_GTR1(const simd_fvec<S> &NDotH, const simd_fvec<S> &a) {
    simd_fvec<S> ret = 1.0f / PI;
    const simd_fvec<S> a2 = sqr(a);
    const simd_fvec<S> t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    where(a < 1.0f, ret) = safe_div(a2 - 1.0f, PI * log(a2) * t);
    return ret;
}

template <int S> simd_fvec<S> D_GGX(const simd_fvec<S> H[3], const simd_fvec<S> &alpha_x, const simd_fvec<S> &alpha_y) {
    simd_fvec<S> ret = 0.0f;

    const simd_fvec<S> sx = -safe_div(H[0], H[2] * alpha_x);
    const simd_fvec<S> sy = -safe_div(H[1], H[2] * alpha_y);
    const simd_fvec<S> s1 = 1.0f + sx * sx + sy * sy;
    const simd_fvec<S> cos_theta_h4 = H[2] * H[2] * H[2] * H[2];

    where(H[2] != 0.0f, ret) = safe_inv_pos((s1 * s1) * PI * alpha_x * alpha_y * cos_theta_h4);
    return ret;
}

template <int S> void create_tbn(const simd_fvec<S> N[3], simd_fvec<S> out_T[3], simd_fvec<S> out_B[3]) {
    simd_fvec<S> U[3] = {1.0f, 0.0f, 0.0f};
    where(N[1] < 0.999f, U[0]) = 0.0f;
    where(N[1] < 0.999f, U[1]) = 1.0f;

    cross(U, N, out_T);
    safe_normalize(out_T);

    cross(N, out_T, out_B);
}

template <int S>
void MapToCone(const simd_fvec<S> &r1, const simd_fvec<S> &r2, const simd_fvec<S> N[3], float radius,
               simd_fvec<S> out_V[3]) {
    const simd_fvec<S> offset[2] = {2.0f * r1 - 1.0f, 2.0f * r2 - 1.0f};

    UNROLLED_FOR(i, 3, { out_V[i] = N[i]; })

    simd_fvec<S> r = offset[1];
    simd_fvec<S> theta = 0.5f * PI * (1.0f - 0.5f * safe_div(offset[0], offset[1]));

    where(abs(offset[0]) > abs(offset[1]), r) = offset[0];
    where(abs(offset[0]) > abs(offset[1]), theta) = 0.25f * PI * safe_div(offset[1], offset[0]);

    const simd_fvec<S> uv[2] = {radius * r * cos(theta), radius * r * sin(theta)};

    simd_fvec<S> LT[3], LB[3];
    create_tbn(N, LT, LB);

    UNROLLED_FOR(i, 3, { out_V[i] = N[i] + uv[0] * LT[i] + uv[1] * LB[i]; })

    const simd_fvec<S> mask = (offset[0] == 0.0f & offset[1] == 0.0f);
    UNROLLED_FOR(i, 3, { where(mask, out_V[i]) = N[i]; })
}

template <int S> force_inline simd_fvec<S> schlick_weight(const simd_fvec<S> &u) {
    const simd_fvec<S> m = clamp(1.0f - u, 0.0f, 1.0f);
    return pow5(m);
}

force_inline float fresnel_dielectric_cos(float cosi, float eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    float c = std::abs(cosi);
    float g = eta * eta - 1 + c * c;
    float result;

    if (g > 0) {
        g = std::sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        result = 0.5f * A * A * (1 + B * B);
    } else {
        result = 1.0f; /* TIR (no refracted component) */
    }

    return result;
}

template <int S> simd_fvec<S> fresnel_dielectric_cos(const simd_fvec<S> &cosi, const simd_fvec<S> &eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    simd_fvec<S> c = abs(cosi);
    simd_fvec<S> g = eta * eta - 1 + c * c;
    const simd_fvec<S> mask = (g > 0.0f);

    simd_fvec<S> result = 1.0f; // TIR (no refracted component)

    g = safe_sqrt(g);
    const simd_fvec<S> A = safe_div(g - c, g + c);
    const simd_fvec<S> B = safe_div(c * (g + c) - 1, c * (g - c) + 1);
    where(mask, result) = 0.5f * A * A * (1 + B * B);

    return result;
}

template <int S>
void get_lobe_weights(const simd_fvec<S> &base_color_lum, const simd_fvec<S> &spec_color_lum,
                      const simd_fvec<S> &specular, const simd_fvec<S> &metallic, const float transmission,
                      const float clearcoat, lobe_weights_t<S> &out_weights) {
    // taken from Cycles
    out_weights.diffuse = base_color_lum * (1.0f - metallic) * (1.0f - transmission);
    const simd_fvec<S> final_transmission = transmission * (1.0f - metallic);
    //(*out_specular_weight) =
    //    (specular != 0.0f || metallic != 0.0f) ? spec_color_lum * (1.0f - final_transmission) : 0.0f;
    out_weights.specular = 0.0f;

    auto temp_mask = (specular != 0.0f | metallic != 0.0f);
    where(temp_mask, out_weights.specular) = spec_color_lum * (1.0f - final_transmission);

    out_weights.clearcoat = 0.25f * clearcoat * (1.0f - metallic);
    out_weights.refraction = final_transmission * base_color_lum;

    const simd_fvec<S> total_weight =
        out_weights.diffuse + out_weights.specular + out_weights.clearcoat + out_weights.refraction;

    where(total_weight != 0.0f, out_weights.diffuse) = safe_div_pos(out_weights.diffuse, total_weight);
    where(total_weight != 0.0f, out_weights.specular) = safe_div_pos(out_weights.specular, total_weight);
    where(total_weight != 0.0f, out_weights.clearcoat) = safe_div_pos(out_weights.clearcoat, total_weight);
    where(total_weight != 0.0f, out_weights.refraction) = safe_div_pos(out_weights.refraction, total_weight);
}

template <int S> force_inline simd_fvec<S> power_heuristic(const simd_fvec<S> &a, const simd_fvec<S> &b) {
    const simd_fvec<S> t = a * a;
    return safe_div_pos(t, b * b + t);
}

template <int S>
force_inline simd_ivec<S> quadratic(const simd_fvec<S> &a, const simd_fvec<S> &b, const simd_fvec<S> &c,
                                    simd_fvec<S> &t0, simd_fvec<S> &t1) {
    const simd_fvec<S> d = b * b - 4.0f * a * c;
    const simd_fvec<S> sqrt_d = safe_sqrt(d);
    simd_fvec<S> q = -0.5f * (b + sqrt_d);
    where(b < 0.0f, q) = -0.5f * (b - sqrt_d);
    t0 = safe_div(q, a);
    t1 = safe_div(c, q);
    return simd_cast(d >= 0.0f);
}

template <int S> force_inline simd_fvec<S> ngon_rad(const simd_fvec<S> &theta, const float n) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, {
        ret.template set<i>(std::cos(PI / n) /
                            std::cos(theta.template get<i>() -
                                     (2.0f * PI / n) * std::floor((n * theta.template get<i>() + PI) / (2.0f * PI))));
    })
    return ret;
}

template <int S>
void get_pix_dirs(const float w, const float h, const camera_t &cam, const float k, const float fov_k,
                  const simd_fvec<S> &x, const simd_fvec<S> &y, const simd_fvec<S> origin[3], simd_fvec<S> d[3]) {
    simd_fvec<S> _dx = 2 * fov_k * (x / w + cam.shift[0] / k) - fov_k;
    simd_fvec<S> _dy = 2 * fov_k * (-y / h + cam.shift[1]) + fov_k;

    d[0] = cam.origin[0] + k * _dx * cam.side[0] + _dy * cam.up[0] + cam.fwd[0] * cam.focus_distance;
    d[1] = cam.origin[1] + k * _dx * cam.side[1] + _dy * cam.up[1] + cam.fwd[1] * cam.focus_distance;
    d[2] = cam.origin[2] + k * _dx * cam.side[2] + _dy * cam.up[2] + cam.fwd[2] * cam.focus_distance;

    d[0] = d[0] - origin[0];
    d[1] = d[1] - origin[1];
    d[2] = d[2] - origin[2];

    normalize(d);
}

template <int S> void push_ior_stack(const simd_ivec<S> &_mask, simd_fvec<S> stack[4], const simd_fvec<S> &val) {
    simd_fvec<S> active_lanes = simd_cast(_mask);
    // 0
    simd_fvec<S> mask = active_lanes & (stack[0] < 0.0f);
    where(mask, stack[0]) = val;
    active_lanes &= ~mask;
    // 1
    mask = active_lanes & (stack[1] < 0.0f);
    where(mask, stack[1]) = val;
    active_lanes &= ~mask;
    // 2
    mask = active_lanes & (stack[2] < 0.0f);
    where(mask, stack[2]) = val;
    active_lanes &= ~mask;
    // 3
    // replace the last value regardless of sign
    where(active_lanes, stack[3]) = val;
}

template <int S>
simd_fvec<S> pop_ior_stack(const simd_ivec<S> &_mask, simd_fvec<S> stack[4],
                           const simd_fvec<S> &default_value = {1.0f}) {
    simd_fvec<S> ret = default_value;
    simd_fvec<S> active_lanes = simd_cast(_mask);
    // 3
    simd_fvec<S> mask = active_lanes & (stack[3] > 0.0f);
    where(mask, ret) = stack[3];
    where(mask, stack[3]) = -1.0f;
    active_lanes &= ~mask;
    // 2
    mask = active_lanes & (stack[2] > 0.0f);
    where(mask, ret) = stack[2];
    where(mask, stack[2]) = -1.0f;
    active_lanes &= ~mask;
    // 1
    mask = active_lanes & (stack[1] > 0.0f);
    where(mask, ret) = stack[1];
    where(mask, stack[1]) = -1.0f;
    active_lanes &= ~mask;
    // 0
    mask = active_lanes & (stack[0] > 0.0f);
    where(mask, ret) = stack[0];
    where(mask, stack[0]) = -1.0f;

    return ret;
}

template <int S>
simd_fvec<S> peek_ior_stack(const simd_fvec<S> stack[4], const simd_ivec<S> &_skip_first,
                            const simd_fvec<S> &default_value = {1.0f}) {
    simd_fvec<S> ret = default_value;
    simd_fvec<S> skip_first = simd_cast(_skip_first);
    // 3
    simd_fvec<S> mask = (stack[3] > 0.0f);
    mask &= ~exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[3];
    simd_fvec<S> active_lanes = ~mask;
    // 2
    mask = active_lanes & (stack[2] > 0.0f);
    mask &= ~exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[2];
    active_lanes &= ~mask;
    // 1
    mask = active_lanes & (stack[1] > 0.0f);
    mask &= ~exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[1];
    active_lanes &= ~mask;
    // 0
    mask = active_lanes & (stack[0] > 0.0f);
    mask &= ~exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[0];

    return ret;
}

} // namespace NS
} // namespace Ray

template <int DimX, int DimY>
void Ray::NS::GeneratePrimaryRays(const int iteration, const camera_t &cam, const rect_t &r, int w, int h,
                                  const float random_seq[], aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                                  aligned_vector<simd_ivec<DimX * DimY>> &out_masks) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    const float k = float(w) / float(h);

    const float temp = std::tan(0.5f * cam.fov * PI / 180.0f);
    const float fov_k = temp * cam.focus_distance;
    const float spread_angle = std::atan(2.0f * temp / float(h));

    const auto off_x = simd_ivec<S>{ray_packet_layout_x, simd_mem_aligned},
               off_y = simd_ivec<S>{ray_packet_layout_y, simd_mem_aligned};

    const int x_res = (r.w + DimX - 1) / DimX, y_res = (r.h + DimY - 1) / DimY;

    size_t i = 0;
    out_rays.resize(x_res * y_res);
    out_masks.resize(x_res * y_res);

    for (int y = r.y; y < r.y + r.h; y += DimY) {
        for (int x = r.x; x < r.x + r.w; x += DimX) {
            simd_ivec<S> &out_mask = out_masks[i];
            ray_data_t<S> &out_r = out_rays[i++];

            const simd_ivec<S> ixx = x + off_x, iyy = y + off_y;

            out_mask = (ixx < w) & (iyy < h);

            const simd_ivec<S> index = iyy * w + ixx;

            auto fxx = (simd_fvec<S>)ixx, fyy = (simd_fvec<S>)iyy;

            const simd_ivec<S> hash_val = hash(index);
            const simd_fvec<S> sample_off[2] = {construct_float(hash_val), construct_float(hash(hash_val))};

            simd_fvec<S> rxx = fract(random_seq[RAND_DIM_FILTER_U] + sample_off[0]),
                         ryy = fract(random_seq[RAND_DIM_FILTER_V] + sample_off[1]);

            simd_fvec<S> offset[2] = {0.0f, 0.0f};
            if (cam.fstop > 0.0f) {
                const simd_fvec<S> r1 = fract(random_seq[RAND_DIM_LENS_U] + sample_off[0]);
                const simd_fvec<S> r2 = fract(random_seq[RAND_DIM_LENS_V] + sample_off[1]);

                offset[0] = 2.0f * r1 - 1.0f;
                offset[1] = 2.0f * r2 - 1.0f;

                simd_fvec<S> r = offset[1], theta = 0.5f * PI - 0.25f * PI * safe_div(offset[0], offset[1]);
                where(abs(offset[0]) > abs(offset[1]), r) = offset[0];
                where(abs(offset[0]) > abs(offset[1]), theta) = 0.25f * PI * safe_div(offset[1], offset[0]);

                if (cam.lens_blades) {
                    r *= ngon_rad(theta, float(cam.lens_blades));
                }

                theta += cam.lens_rotation;

                where(offset[0] != 0.0f & offset[1] != 0.0f, offset[0]) = 0.5f * r * cos(theta) / cam.lens_ratio;
                where(offset[0] != 0.0f & offset[1] != 0.0f, offset[1]) = 0.5f * r * sin(theta);

                const float coc = 0.5f * (cam.focal_length / cam.fstop);
                offset[0] *= coc * cam.sensor_height;
                offset[1] *= coc * cam.sensor_height;
            }

            if (cam.filter == Tent) {
                simd_fvec<S> temp = rxx;
                rxx = 1.0f - sqrt(2.0f - 2.0f * temp);
                where(temp < 0.5f, rxx) = sqrt(2.0f * temp) - 1.0f;

                temp = ryy;
                ryy = 1.0f - sqrt(2.0f - 2.0f * temp);
                where(temp < 0.5f, ryy) = sqrt(2.0f * temp) - 1.0f;

                rxx += 0.5f;
                ryy += 0.5f;
            }

            fxx += rxx;
            fyy += ryy;

            const simd_fvec<S> _origin[3] = {{cam.origin[0] + cam.side[0] * offset[0] + cam.up[0] * offset[1]},
                                             {cam.origin[1] + cam.side[1] * offset[0] + cam.up[1] * offset[1]},
                                             {cam.origin[2] + cam.side[2] * offset[0] + cam.up[2] * offset[1]}};

            simd_fvec<S> _d[3], _dx[3], _dy[3];
            get_pix_dirs(float(w), float(h), cam, k, fov_k, fxx, fyy, _origin, _d);
            get_pix_dirs(float(w), float(h), cam, k, fov_k, fxx + 1.0f, fyy, _origin, _dx);
            get_pix_dirs(float(w), float(h), cam, k, fov_k, fxx, fyy + 1.0f, _origin, _dy);

            const simd_fvec<S> clip_start = cam.clip_start / dot3(_d, cam.fwd);

            for (int j = 0; j < 3; j++) {
                out_r.d[j] = _d[j];
                out_r.o[j] = _origin[j] + _d[j] * clip_start;
                out_r.c[j] = {1.0f};
            }

            // air ior is implicit
            out_r.ior[0] = out_r.ior[1] = out_r.ior[2] = out_r.ior[3] = -1.0f;

            out_r.cone_width = 0.0f;
            out_r.cone_spread = spread_angle;

            out_r.pdf = {1e6f};
            out_r.xy = (ixx << 16) | iyy;
            out_r.depth = 0;
        }
    }
}

template <int DimX, int DimY>
void Ray::NS::SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                       const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                                       const rect_t &r, int width, int height, const float random_seq[],
                                       aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                                       aligned_vector<hit_data_t<DimX * DimY>> &out_inters) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    out_rays.resize(r.w * r.h / S + ((r.w * r.h) % S != 0));
    out_inters.resize(out_rays.size());

    const auto off_x = simd_ivec<S>{ray_packet_layout_x}, off_y = simd_ivec<S>{ray_packet_layout_y};

    size_t count = 0;
    for (int y = r.y; y < r.y + r.h - (r.h & (DimY - 1)); y += DimY) {
        for (int x = r.x; x < r.x + r.w - (r.w & (DimX - 1)); x += DimX) {
            const simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

            ray_data_t<S> &out_ray = out_rays[count];
            hit_data_t<S> &out_inter = out_inters[count];
            count++;

            out_ray.xy = (ixx << 16) | iyy;
            out_ray.c[0] = out_ray.c[1] = out_ray.c[2] = 1.0f;
            out_ray.cone_width = 0.0f;
            out_ray.cone_spread = 0.0f;
            out_inter.mask = 0;
        }
    }

    const simd_ivec4 irect_min = {r.x, r.y, 0, 0}, irect_max = {r.x + r.w - 1, r.y + r.h - 1, 0, 0};
    const simd_fvec4 size = {float(width), float(height), 0.0f, 0.0f};

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const vertex_t &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const vertex_t &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const vertex_t &v2 = vertices[vtx_indices[tri * 3 + 2]];

        const auto t0 = simd_fvec4{v0.t[uv_layer][0], 1.0f - v0.t[uv_layer][1], 0.0f, 0.0f} * size;
        const auto t1 = simd_fvec4{v1.t[uv_layer][0], 1.0f - v1.t[uv_layer][1], 0.0f, 0.0f} * size;
        const auto t2 = simd_fvec4{v2.t[uv_layer][0], 1.0f - v2.t[uv_layer][1], 0.0f, 0.0f} * size;

        simd_fvec4 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        simd_ivec4 ibbox_min = (simd_ivec4)(bbox_min),
                   ibbox_max = simd_ivec4{int(std::round(bbox_max[0])), int(std::round(bbox_max[1])), 0, 0};

        if (ibbox_max[0] < irect_min[0] || ibbox_max[1] < irect_min[1] || ibbox_min[0] > irect_max[0] ||
            ibbox_min[1] > irect_max[1]) {
            continue;
        }

        ibbox_min = max(ibbox_min, irect_min);
        ibbox_max = min(ibbox_max, irect_max);

        ibbox_min.set<0>(ibbox_min[0] - (ibbox_min[0] % DimX));
        ibbox_min.set<1>(ibbox_min[1] - (ibbox_min[1] % DimY));
        ibbox_max.set<0>(ibbox_max[0] + (((ibbox_max[0] + 1) % DimX) ? (DimX - (ibbox_max[0] + 1) % DimX) : 0));
        ibbox_max.set<1>(ibbox_max[1] + (((ibbox_max[1] + 1) % DimY) ? (DimY - (ibbox_max[1] + 1) % DimY) : 0));

        const simd_fvec4 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

        float area = d01[0] * d20[1] - d20[0] * d01[1];
        if (area < FLT_EPS) {
            continue;
        }

        const float inv_area = 1.0f / area;

        for (int y = ibbox_min[1]; y <= ibbox_max[1]; y += DimY) {
            for (int x = ibbox_min[0]; x <= ibbox_max[0]; x += DimX) {
                const simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

                const int ndx = ((y - r.y) / DimY) * (r.w / DimX) + (x - r.x) / DimX;
                ray_data_t<S> &out_ray = out_rays[ndx];
                hit_data_t<S> &out_inter = out_inters[ndx];

                simd_fvec<S> rxx = construct_float(hash(out_ray.xy));
                simd_fvec<S> ryy = construct_float(hash(hash(out_ray.xy)));

                UNROLLED_FOR_S(i, S, {
                    float _unused;
                    rxx.template set<i>(std::modf(random_seq[RAND_DIM_FILTER_U] + rxx.template get<i>(), &_unused));
                    ryy.template set<i>(std::modf(random_seq[RAND_DIM_FILTER_V] + ryy.template get<i>(), &_unused));
                })

                const simd_fvec<S> fxx = simd_fvec<S>{ixx} + rxx, fyy = simd_fvec<S>{iyy} + ryy;

                simd_fvec<S> u = d01[0] * (fyy - t0[1]) - d01[1] * (fxx - t0[0]),
                             v = d12[0] * (fyy - t1[1]) - d12[1] * (fxx - t1[0]),
                             w = d20[0] * (fyy - t2[1]) - d20[1] * (fxx - t2[0]);

                const simd_fvec<S> fmask = (u >= -FLT_EPS) & (v >= -FLT_EPS) & (w >= -FLT_EPS);
                const simd_ivec<S> imask = simd_cast(fmask);

                if (imask.not_all_zeros()) {
                    u *= inv_area;
                    v *= inv_area;
                    w *= inv_area;

                    const simd_fvec<S> _p[3] = {v0.p[0] * v + v1.p[0] * w + v2.p[0] * u,
                                                v0.p[1] * v + v1.p[1] * w + v2.p[1] * u,
                                                v0.p[2] * v + v1.p[2] * w + v2.p[2] * u},
                                       _n[3] = {v0.n[0] * v + v1.n[0] * w + v2.n[0] * u,
                                                v0.n[1] * v + v1.n[1] * w + v2.n[1] * u,
                                                v0.n[2] * v + v1.n[2] * w + v2.n[2] * u};

                    simd_fvec<S> p[3], n[3];

                    TransformPoint(_p, tr.xform, p);
                    TransformNormal(_n, tr.inv_xform, n);

                    UNROLLED_FOR(i, 3, { where(fmask, out_ray.o[i]) = p[i] + n[i]; })
                    UNROLLED_FOR(i, 3, { where(fmask, out_ray.d[i]) = -n[i]; })
                    // where(fmask, out_ray.ior) = 1.0f;
                    where(fmask, out_ray.depth) = 0;

                    out_inter.mask = (out_inter.mask | imask);
                    where(imask, out_inter.prim_index) = tri;
                    where(imask, out_inter.obj_index) = obj_index;
                    where(fmask, out_inter.t) = 1.0f;
                    where(fmask, out_inter.u) = w;
                    where(fmask, out_inter.v) = u;
                }
            }
        }
    }
}

template <int S>
void Ray::NS::SortRays_CPU(ray_data_t<S> *rays, simd_ivec<S> *ray_masks, int &rays_count, const float root_min[3],
                           const float cell_size[3], simd_ivec<S> *hash_values, uint32_t *scan_values,
                           ray_chunk_t *chunks, ray_chunk_t *chunks_temp) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (int i = 0; i < rays_count; i++) {
        hash_values[i] = get_ray_hash(rays[i], ray_masks[i], root_min, cell_size);
    }

    size_t chunks_count = 0;

    // compress codes into spans of indentical values (makes sorting stage faster)
    int *flat_hash_values = value_ptr(hash_values[0]);
    for (int start = 0, end = 1; end <= rays_count * S; end++) {
        if (end == (rays_count * S) || (flat_hash_values[start] != flat_hash_values[end])) {
            chunks[chunks_count].hash = flat_hash_values[start];
            chunks[chunks_count].base = start;
            chunks[chunks_count++].size = end - start;
            start = end;
        }
    }

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    // decompress sorted spans
    size_t counter = 0;
    for (uint32_t i = 0; i < chunks_count; i++) {
        for (uint32_t j = 0; j < chunks[i].size; j++) {
            scan_values[counter++] = chunks[i].base + j;
        }
    }

    { // reorder rays
        for (int i = 0; i < rays_count * S; i++) {
            int j;
            while (i != (j = int(scan_values[i]))) {
                const int k = int(scan_values[j]);

                {
                    const int jj = j / S, _jj = j % S, kk = k / S, _kk = k % S;

                    swap_elements(rays[jj].o[0], _jj, rays[kk].o[0], _kk);
                    swap_elements(rays[jj].o[1], _jj, rays[kk].o[1], _kk);
                    swap_elements(rays[jj].o[2], _jj, rays[kk].o[2], _kk);

                    swap_elements(rays[jj].d[0], _jj, rays[kk].d[0], _kk);
                    swap_elements(rays[jj].d[1], _jj, rays[kk].d[1], _kk);
                    swap_elements(rays[jj].d[2], _jj, rays[kk].d[2], _kk);

                    swap_elements(rays[jj].pdf, _jj, rays[kk].pdf, _kk);

                    swap_elements(rays[jj].c[0], _jj, rays[kk].c[0], _kk);
                    swap_elements(rays[jj].c[1], _jj, rays[kk].c[1], _kk);
                    swap_elements(rays[jj].c[2], _jj, rays[kk].c[2], _kk);

                    swap_elements(rays[jj].ior[0], _jj, rays[kk].ior[0], _kk);
                    swap_elements(rays[jj].ior[1], _jj, rays[kk].ior[1], _kk);
                    swap_elements(rays[jj].ior[2], _jj, rays[kk].ior[2], _kk);
                    swap_elements(rays[jj].ior[3], _jj, rays[kk].ior[3], _kk);

                    swap_elements(rays[jj].cone_width, _jj, rays[kk].cone_width, _kk);
                    swap_elements(rays[jj].cone_spread, _jj, rays[kk].cone_spread, _kk);

                    swap_elements(rays[jj].xy, _jj, rays[kk].xy, _kk);
                    swap_elements(rays[jj].depth, _jj, rays[kk].depth, _kk);

                    swap_elements(ray_masks[jj], _jj, ray_masks[kk], _kk);
                }

                std::swap(flat_hash_values[i], flat_hash_values[j]);
                std::swap(scan_values[i], scan_values[j]);
            }
        }
    }

    // remove non-active rays
    while (rays_count && ray_masks[rays_count - 1].all_zeros()) {
        rays_count--;
    }
}

template <int S>
void Ray::NS::SortRays_GPU(ray_data_t<S> *rays, simd_ivec<S> *ray_masks, int &rays_count, const float root_min[3],
                           const float cell_size[3], simd_ivec<S> *hash_values, int *head_flags, uint32_t *scan_values,
                           ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute ray hash values
    for (int i = 0; i < rays_count; i++) {
        hash_values[i] = get_ray_hash(rays[i], ray_masks[i], root_min, cell_size);
    }

    // set head flags
    head_flags[0] = 1;
    for (int i = 1; i < rays_count * S; i++) {
        head_flags[i] = hash_values[i / S][i % S] != hash_values[(i - 1) / S][(i - 1) % S];
    }

    int chunks_count = 0;

    { // perform exclusive scan on head flags
        uint32_t cur_sum = 0;
        for (int i = 0; i < rays_count * S; i++) {
            scan_values[i] = cur_sum;
            cur_sum += head_flags[i];
        }
        chunks_count = int(cur_sum);
    }

    // init ray chunks hash and base index
    for (int i = 0; i < rays_count * S; i++) {
        if (head_flags[i]) {
            chunks[scan_values[i]].hash = uint32_t(hash_values[i / S][i % S]);
            chunks[scan_values[i]].base = uint32_t(i);
        }
    }

    // init ray chunks size
    for (int i = 0; i < chunks_count - 1; i++) {
        chunks[i].size = chunks[i + 1].base - chunks[i].base;
    }
    chunks[chunks_count - 1].size = (uint32_t)rays_count * S - chunks[chunks_count - 1].base;

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    { // perform exclusive scan on chunks size
        uint32_t cur_sum = 0;
        for (int i = 0; i < chunks_count; i++) {
            scan_values[i] = cur_sum;
            cur_sum += chunks[i].size;
        }
    }

    std::fill(&skeleton[0], &skeleton[0] + rays_count * S, 1);
    std::fill(&head_flags[0], &head_flags[0] + rays_count * S, 0);

    // init skeleton and head flags array
    for (int i = 0; i < chunks_count; i++) {
        skeleton[scan_values[i]] = chunks[i].base;
        head_flags[scan_values[i]] = 1;
    }

    { // perform a segmented scan on skeleton array
        uint32_t cur_sum = 0;
        for (int i = 0; i < rays_count * S; i++) {
            if (head_flags[i]) {
                cur_sum = 0;
            }
            cur_sum += skeleton[i];
            scan_values[i] = cur_sum;
        }
    }

    { // reorder rays
        for (int i = 0; i < rays_count * S; i++) {
            int j;
            while (i != (j = scan_values[i])) {
                const int k = scan_values[j];

                {
                    const int jj = j / S, _jj = j % S, kk = k / S, _kk = k % S;

                    swap_elements(hash_values[jj], _jj, hash_values[kk], _kk);

                    swap_elements(rays[jj].o[0], _jj, rays[kk].o[0], _kk);
                    swap_elements(rays[jj].o[1], _jj, rays[kk].o[1], _kk);
                    swap_elements(rays[jj].o[2], _jj, rays[kk].o[2], _kk);

                    swap_elements(rays[jj].d[0], _jj, rays[kk].d[0], _kk);
                    swap_elements(rays[jj].d[1], _jj, rays[kk].d[1], _kk);
                    swap_elements(rays[jj].d[2], _jj, rays[kk].d[2], _kk);

                    swap_elements(rays[jj].pdf, _jj, rays[kk].pdf, _kk);

                    swap_elements(rays[jj].c[0], _jj, rays[kk].c[0], _kk);
                    swap_elements(rays[jj].c[1], _jj, rays[kk].c[1], _kk);
                    swap_elements(rays[jj].c[2], _jj, rays[kk].c[2], _kk);

                    swap_elements(rays[jj].cone_width, _jj, rays[kk].cone_width, _kk);
                    swap_elements(rays[jj].cone_spread, _jj, rays[kk].cone_spread, _kk);

                    swap_elements(rays[jj].xy, _jj, rays[kk].xy, _kk);

                    swap_elements(ray_masks[jj], _jj, ray_masks[kk], _kk);
                }

                std::swap(scan_values[i], scan_values[j]);
            }
        }
    }

    // remove non-active rays
    while (rays_count && ray_masks[rays_count - 1].all_zeros()) {
        rays_count--;
    }
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                       const tri_accel_t *tris, uint32_t num_tris, int obj_index,
                                       hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = {Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    out_inter.mask |= inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter.mask, out_inter.u) = inter.u;
    where(inter.mask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                       const tri_accel_t *tris, const int tri_start, const int tri_end,
                                       const int obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter{Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    out_inter.mask |= inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter.mask, out_inter.u) = inter.u;
    where(inter.mask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const float o[3], const float d[3], int i, const tri_accel_t *tris,
                                       const int tri_start, const int tri_end, const int obj_index,
                                       hit_data_t<S> &out_inter) {
    bool res = false;

    for (int j = tri_start; j < tri_end; j++) {
        res |= _IntersectTri(o, d, i, tris[j], j, out_inter);
    }

    if (res) {
        out_inter.obj_index[i] = obj_index;
    }

    return res;
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const float o[3], const float d[3], const mtri_accel_t *mtris,
                                       const int tri_start, const int tri_end, int &inter_prim_index, float &inter_t,
                                       float &inter_u, float &inter_v) {
    bool res = false;

    for (int j = tri_start / 8; j < (tri_end + 7) / 8; ++j) {
        res |= IntersectTri<S>(o, d, mtris[j], j * 8, inter_prim_index, inter_t, inter_u, inter_v);
    }

    return res;
}

template <int S>
bool Ray::NS::IntersectTris_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                   const tri_accel_t *tris, uint32_t num_tris, int obj_index,
                                   hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = {Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    out_inter.mask |= inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter.mask, out_inter.u) = inter.u;
    where(inter.mask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                   const tri_accel_t *tris, const int tri_start, const int tri_end, const int obj_index,
                                   hit_data_t<S> &out_inter) {
    hit_data_t<S> inter{Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    out_inter.mask |= inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter.mask, out_inter.u) = inter.u;
    where(inter.mask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_AnyHit(const float o[3], const float d[3], int i, const tri_accel_t *tris,
                                   const tri_mat_data_t *materials, const uint32_t *indices, const int tri_start,
                                   const int tri_end, const int obj_index, hit_data_t<S> &out_inter) {
    bool res = false;

    for (int j = tri_start; j < tri_end; j++) {
        res |= _IntersectTri(o, d, i, tris[j], j, out_inter);
        // if (res && ((inter.prim_index > 0 && (materials[indices[i]].front_mi & MATERIAL_SOLID_BIT)) ||
        //             (inter.prim_index < 0 && (materials[indices[i]].back_mi & MATERIAL_SOLID_BIT)))) {
        //     break;
        // }
    }

    if (res) {
        out_inter.obj_index[i] = obj_index;
    }

    return res;
}

template <int S>
bool Ray::NS::IntersectTris_AnyHit(const float o[3], const float d[3], const mtri_accel_t *mtris,
                                   const tri_mat_data_t *materials, const uint32_t *indices, const int tri_start,
                                   const int tri_end, int &inter_prim_index, float &inter_t, float &inter_u,
                                   float &inter_v) {
    bool res = false;

    for (int j = tri_start / 8; j < (tri_end + 7) / 8; j++) {
        res |= IntersectTri<S>(o, d, mtris[j], j * 8, inter_prim_index, inter_t, inter_u, inter_v);
        // if (res && ((inter.prim_index > 0 && (materials[indices[i]].front_mi & MATERIAL_SOLID_BIT)) ||
        //             (inter.prim_index < 0 && (materials[indices[i]].back_mi & MATERIAL_SOLID_BIT)))) {
        //     break;
        // }
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MacroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                      const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                      uint32_t node_index, const mesh_instance_t *mesh_instances,
                                                      const uint32_t *mi_indices, const mesh_t *meshes,
                                                      const transform_t *transforms, const tri_accel_t *tris,
                                                      const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    TraversalStateStack_Multi<S, MAX_STACK_SIZE> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            uint32_t cur = stack[--stack_size];

            simd_ivec<S> mask1 = bbox_test_fma(inv_d, inv_d_o, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            simd_ivec<S> mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(rd, nodes[cur]);
            } else {
                const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);
                for (uint32_t i = prim_index; i < prim_index + nodes[cur].prim_count; i++) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    simd_ivec<S> bbox_mask =
                        bbox_test_fma(inv_d, inv_d_o, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                    if (bbox_mask.all_zeros()) {
                        continue;
                    }

                    simd_fvec<S> _ro[3], _rd[3];
                    TransformRay(ro, rd, tr.inv_xform, _ro, _rd);

                    res |= Traverse_MicroTree_WithStack_ClosestHit(_ro, _rd, bbox_mask, nodes, m.node_index, tris,
                                                                   tri_indices, int(mi_indices[i]), inter);
                }
            }
        }
        st.index++;
    }

    // resolve primitive index indirection
    const simd_ivec<S> is_backfacing = (inter.prim_index < 0);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    inter.prim_index = gather(reinterpret_cast<const int *>(tri_indices), inter.prim_index);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return res;
}

template <int S>
bool Ray::NS::Traverse_MacroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                      const simd_ivec<S> &ray_mask, const mbvh_node_t *nodes,
                                                      uint32_t node_index, const mesh_instance_t *mesh_instances,
                                                      const uint32_t *mi_indices, const mesh_t *meshes,
                                                      const transform_t *transforms, const mtri_accel_t *mtris,
                                                      const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    alignas(S * 4) float _ro[3][S], _rd[3][S];
    ro[0].copy_to(_ro[0], simd_mem_aligned);
    ro[1].copy_to(_ro[1], simd_mem_aligned);
    ro[2].copy_to(_ro[2], simd_mem_aligned);
    rd[0].copy_to(_rd[0], simd_mem_aligned);
    rd[1].copy_to(_rd[1], simd_mem_aligned);
    rd[2].copy_to(_rd[2], simd_mem_aligned);

    alignas(S * 4) int ray_masks[S], inter_mask[S], inter_prim_index[S], inter_obj_index[S];
    alignas(S * 4) float inter_t[S], inter_u[S], inter_v[S];
    ray_mask.copy_to(ray_masks, simd_mem_aligned);
    inter.mask.copy_to(inter_mask, simd_mem_aligned);
    inter.prim_index.copy_to(inter_prim_index, simd_mem_aligned);
    inter.obj_index.copy_to(inter_obj_index, simd_mem_aligned);
    inter.t.copy_to(inter_t, simd_mem_aligned);
    inter.u.copy_to(inter_u, simd_mem_aligned);
    inter.v.copy_to(inter_v, simd_mem_aligned);

    for (int ri = 0; ri < S; ri++) {
        if (!ray_masks[ri]) {
            continue;
        }

        // recombine in AoS layout
        const float r_o[3] = {_ro[0][ri], _ro[1][ri], _ro[2][ri]}, r_d[3] = {_rd[0][ri], _rd[1][ri], _rd[2][ri]};
        const float _inv_d[3] = {inv_d[0][ri], inv_d[1][ri], inv_d[2][ri]},
                    _inv_d_o[3] = {inv_d_o[0][ri], inv_d_o[1][ri], inv_d_o[2][ri]};

        TraversalStateStack_Single<MAX_STACK_SIZE> st;
        st.push(node_index, 0.0f);

        while (!st.empty()) {
            stack_entry_t cur = st.pop();

            if (cur.dist > inter_t[ri]) {
                continue;
            }

        TRAVERSE:
            if (!is_leaf_node(nodes[cur.index])) {
                alignas(32) float res_dist[8];
                long mask = bbox_test_oct<S>(_inv_d, _inv_d_o, inter_t[ri], nodes[cur.index].bbox_min,
                                             nodes[cur.index].bbox_max, res_dist);
                if (mask) {
                    long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (mask == 0) { // only one box was hit
                        cur.index = nodes[cur.index].child[i];
                        goto TRAVERSE;
                    }

                    const long i2 = GetFirstBit(mask);
                    mask = ClearBit(mask, i2);
                    if (mask == 0) { // two boxes were hit
                        if (res_dist[i] < res_dist[i2]) {
                            st.push(nodes[cur.index].child[i2], res_dist[i2]);
                            cur.index = nodes[cur.index].child[i];
                        } else {
                            st.push(nodes[cur.index].child[i], res_dist[i]);
                            cur.index = nodes[cur.index].child[i2];
                        }
                        goto TRAVERSE;
                    }

                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    st.push(nodes[cur.index].child[i2], res_dist[i2]);

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    if (mask == 0) { // three boxes were hit
                        st.sort_top3();
                        cur.index = st.pop_index();
                        goto TRAVERSE;
                    }

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    if (mask == 0) { // four boxes were hit
                        st.sort_top4();
                        cur.index = st.pop_index();
                        goto TRAVERSE;
                    }

                    const uint32_t size_before = st.stack_size;

                    // from five to eight boxes were hit
                    do {
                        i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        st.push(nodes[cur.index].child[i], res_dist[i]);
                    } while (mask != 0);

                    const int count = int(st.stack_size - size_before + 4);
                    st.sort_topN(count);
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }
            } else {
                const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);
                for (uint32_t j = prim_index; j < prim_index + nodes[cur.index].child[1]; j++) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[j]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    if (!bbox_test(_inv_d, _inv_d_o, inter_t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, tr.inv_xform, tr_ro, tr_rd);

                    const bool lres = Traverse_MicroTree_WithStack_ClosestHit<S>(
                        tr_ro, tr_rd, nodes, m.node_index, mtris, tri_indices, inter_prim_index[ri], inter_t[ri],
                        inter_u[ri], inter_v[ri]);
                    if (lres) {
                        inter_mask[ri] = -1;
                        inter_obj_index[ri] = int(mi_indices[j]);
                    }
                    res |= lres;
                }
            }
        }
    }

    inter.mask = simd_ivec<S>{inter_mask, simd_mem_aligned};
    inter.prim_index = simd_ivec<S>{inter_prim_index, simd_mem_aligned};
    inter.obj_index = simd_ivec<S>{inter_obj_index, simd_mem_aligned};
    inter.t = simd_fvec<S>{inter_t, simd_mem_aligned};
    inter.u = simd_fvec<S>{inter_u, simd_mem_aligned};
    inter.v = simd_fvec<S>{inter_v, simd_mem_aligned};

    // resolve primitive index indirection
    simd_ivec<S> prim_index = (ray_mask & inter.prim_index);

    const simd_ivec<S> is_backfacing = (prim_index < 0);
    where(is_backfacing, prim_index) = -prim_index - 1;

    where(ray_mask, inter.prim_index) = gather(reinterpret_cast<const int *>(tri_indices), prim_index);
    where(ray_mask & is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return res;
}

template <int S>
Ray::NS::simd_ivec<S> Ray::NS::Traverse_MacroTree_WithStack_AnyHit(
    const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
    uint32_t node_index, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const tri_accel_t *tris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<S> &inter) {
    simd_ivec<S> solid_hit_mask = {0};

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    TraversalStateStack_Multi<S, MAX_STACK_SIZE> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            const uint32_t cur = stack[--stack_size];

            simd_ivec<S> mask1 = bbox_test_fma(inv_d, inv_d_o, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            simd_ivec<S> mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(rd, nodes[cur]);
            } else {
                const uint32_t prim_index = (nodes[cur].prim_index & PRIM_INDEX_BITS);
                for (uint32_t i = prim_index; i < prim_index + nodes[cur].prim_count; i++) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[i]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    const simd_ivec<S> bbox_mask =
                        bbox_test_fma(inv_d, inv_d_o, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                    if (bbox_mask.all_zeros()) {
                        continue;
                    }

                    simd_fvec<S> _ro[3], _rd[3];
                    TransformRay(ro, rd, tr.inv_xform, _ro, _rd);

                    solid_hit_mask |=
                        Traverse_MicroTree_WithStack_AnyHit(_ro, _rd, bbox_mask, nodes, m.node_index, tris, materials,
                                                            tri_indices, int(mi_indices[i]), inter);
                }
            }
        }
        st.index++;
    }

    // resolve primitive index indirection
    simd_ivec<S> prim_index = (ray_mask & inter.prim_index);

    const simd_ivec<S> is_backfacing = (prim_index < 0);
    where(is_backfacing, prim_index) = -prim_index - 1;

    where(ray_mask, inter.prim_index) = gather(reinterpret_cast<const int *>(tri_indices), prim_index);
    where(ray_mask & is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return solid_hit_mask;
}

template <int S>
Ray::NS::simd_ivec<S> Ray::NS::Traverse_MacroTree_WithStack_AnyHit(
    const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask, const mbvh_node_t *nodes,
    uint32_t node_index, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const mtri_accel_t *mtris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<S> &inter) {
    simd_ivec<S> solid_hit_mask = {0};

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    alignas(S * 4) int ray_masks[S], inter_prim_index[S];
    alignas(S * 4) float inter_t[S], inter_u[S], inter_v[S];
    ray_mask.copy_to(ray_masks, simd_mem_aligned);
    inter.prim_index.copy_to(inter_prim_index, simd_mem_aligned);
    inter.t.copy_to(inter_t, simd_mem_aligned);
    inter.u.copy_to(inter_u, simd_mem_aligned);
    inter.v.copy_to(inter_v, simd_mem_aligned);

    for (int ri = 0; ri < S; ri++) {
        if (!ray_masks[ri]) {
            continue;
        }

        // recombine in AoS layout
        const float r_o[3] = {ro[0][ri], ro[1][ri], ro[2][ri]}, r_d[3] = {rd[0][ri], rd[1][ri], rd[2][ri]};
        const float _inv_d[3] = {inv_d[0][ri], inv_d[1][ri], inv_d[2][ri]},
                    _inv_d_o[3] = {inv_d_o[0][ri], inv_d_o[1][ri], inv_d_o[2][ri]};

        TraversalStateStack_Single<MAX_STACK_SIZE> st;
        st.push(node_index, 0.0f);

        while (!st.empty()) {
            stack_entry_t cur = st.pop();

            if (cur.dist > inter.t[ri]) {
                continue;
            }

        TRAVERSE:
            if (!is_leaf_node(nodes[cur.index])) {
                alignas(32) float res_dist[8];
                long mask = bbox_test_oct<S>(_inv_d, _inv_d_o, inter.t[ri], nodes[cur.index].bbox_min,
                                             nodes[cur.index].bbox_max, res_dist);
                if (mask) {
                    long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (mask == 0) { // only one box was hit
                        cur.index = nodes[cur.index].child[i];
                        goto TRAVERSE;
                    }

                    const long i2 = GetFirstBit(mask);
                    mask = ClearBit(mask, i2);
                    if (mask == 0) { // two boxes were hit
                        if (res_dist[i] < res_dist[i2]) {
                            st.push(nodes[cur.index].child[i2], res_dist[i2]);
                            cur.index = nodes[cur.index].child[i];
                        } else {
                            st.push(nodes[cur.index].child[i], res_dist[i]);
                            cur.index = nodes[cur.index].child[i2];
                        }
                        goto TRAVERSE;
                    }

                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    st.push(nodes[cur.index].child[i2], res_dist[i2]);

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    if (mask == 0) { // three boxes were hit
                        st.sort_top3();
                        cur.index = st.pop_index();
                        goto TRAVERSE;
                    }

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                    if (mask == 0) { // four boxes were hit
                        st.sort_top4();
                        cur.index = st.pop_index();
                        goto TRAVERSE;
                    }

                    const uint32_t size_before = st.stack_size;

                    // from five to eight boxes were hit
                    do {
                        i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        st.push(nodes[cur.index].child[i], res_dist[i]);
                    } while (mask != 0);

                    const int count = int(st.stack_size - size_before + 4);
                    st.sort_topN(count);
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }
            } else {
                const uint32_t prim_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);
                for (uint32_t j = prim_index; j < prim_index + nodes[cur.index].child[1]; j++) {
                    const mesh_instance_t &mi = mesh_instances[mi_indices[j]];
                    const mesh_t &m = meshes[mi.mesh_index];
                    const transform_t &tr = transforms[mi.tr_index];

                    if (!bbox_test(_inv_d, _inv_d_o, inter_t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, tr.inv_xform, tr_ro, tr_rd);

                    const int hit_type = Traverse_MicroTree_WithStack_AnyHit<S>(
                        tr_ro, tr_rd, nodes, m.node_index, mtris, materials, tri_indices, inter_prim_index[ri],
                        inter_t[ri], inter_u[ri], inter_v[ri]);
                    if (hit_type) {
                        inter.mask.set(ri, -1);
                        inter.obj_index.set(ri, int(mi_indices[j]));
                    }
                    if (hit_type == 2) {
                        solid_hit_mask.set(ri, -1);
                        break;
                    }
                }
            }

            if (solid_hit_mask[ri]) {
                break;
            }
        }
    }

    inter.prim_index = simd_ivec<S>{inter_prim_index, simd_mem_aligned};
    inter.t = simd_fvec<S>{inter_t, simd_mem_aligned};
    inter.u = simd_fvec<S>{inter_u, simd_mem_aligned};
    inter.v = simd_fvec<S>{inter_v, simd_mem_aligned};

    // resolve primitive index indirection
    const simd_ivec<S> is_backfacing = (inter.prim_index < 0);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    inter.prim_index = gather(reinterpret_cast<const int *>(tri_indices), inter.prim_index);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return solid_hit_mask;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                      const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                      uint32_t node_index, const tri_accel_t *tris,
                                                      const uint32_t *tri_indices, int obj_index,
                                                      hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    TraversalStateStack_Multi<S, MAX_STACK_SIZE> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            uint32_t cur = stack[--stack_size];

            simd_ivec<S> mask1 = bbox_test_fma(inv_d, inv_d_o, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            simd_ivec<S> mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(rd, nodes[cur]);
            } else {
                const int tri_start = int(nodes[cur].prim_index & PRIM_INDEX_BITS),
                          tri_end = int(tri_start + nodes[cur].prim_count);
                res |= IntersectTris_ClosestHit(ro, rd, st.queue[st.index].mask, tris, tri_start, tri_end, obj_index,
                                                inter);
            }
        }
        st.index++;
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack_ClosestHit(const float ro[3], const float rd[3], const mbvh_node_t *nodes,
                                                      uint32_t node_index, const mtri_accel_t *mtris,
                                                      const uint32_t *tri_indices, int &inter_prim_index,
                                                      float &inter_t, float &inter_u, float &inter_v) {
    bool res = false;

    float _inv_d[3], _inv_d_o[3];
    comp_aux_inv_values(ro, rd, _inv_d, _inv_d_o);

    TraversalStateStack_Single<MAX_STACK_SIZE> st;
    st.push(node_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter_t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(32) float res_dist[8];
            long mask = bbox_test_oct<S>(_inv_d, _inv_d_o, inter_t, nodes[cur.index].bbox_min,
                                         nodes[cur.index].bbox_max, res_dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (res_dist[i] < res_dist[i2]) {
                        st.push(nodes[cur.index].child[i2], res_dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], res_dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], res_dist[i]);
                st.push(nodes[cur.index].child[i2], res_dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], res_dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], res_dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                const uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int tri_start = int(nodes[cur.index].child[0] & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur.index].child[1]);
            res |= IntersectTris_ClosestHit<S>(ro, rd, mtris, tri_start, tri_end, inter_prim_index, inter_t, inter_u,
                                               inter_v);
        }
    }

    return res;
}

template <int S>
Ray::NS::simd_ivec<S>
Ray::NS::Traverse_MicroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                             const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                             const tri_accel_t *tris, const tri_mat_data_t *materials,
                                             const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter) {
    simd_ivec<S> solid_hit_mask = 0;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    TraversalStateStack_Multi<S, MAX_STACK_SIZE> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            const uint32_t cur = stack[--stack_size];

            const simd_ivec<S> mask1 = bbox_test_fma(inv_d, inv_d_o, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            const simd_ivec<S> mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(rd, nodes[cur]);
            } else {
                const int tri_start = nodes[cur].prim_index & PRIM_INDEX_BITS,
                          tri_end = tri_start + nodes[cur].prim_count;
                const bool hit_found =
                    IntersectTris_AnyHit(ro, rd, st.queue[st.index].mask, tris, tri_start, tri_end, obj_index, inter);
                unused(hit_found);
                /*if (hit_found) {
                    const bool is_backfacing = inter.prim_index < 0;
                    const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                    if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                        (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                        return true;
                    }
                }
                res |= hit_found;*/
            }
        }
        st.index++;
    }

    return solid_hit_mask;
}

template <int S>
int Ray::NS::Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], const mbvh_node_t *nodes,
                                                 uint32_t node_index, const mtri_accel_t *mtris,
                                                 const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                 int &inter_prim_index, float &inter_t, float &inter_u,
                                                 float &inter_v) {
    bool res = false;

    float _inv_d[3], _inv_d_o[3];
    comp_aux_inv_values(ro, rd, _inv_d, _inv_d_o);

    TraversalStateStack_Single<MAX_STACK_SIZE> st;
    st.push(node_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter_t) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(32) float res_dist[8];
            long mask = bbox_test_oct<S>(_inv_d, _inv_d_o, inter_t, nodes[cur.index].bbox_min,
                                         nodes[cur.index].bbox_max, res_dist);
            if (mask) {
                long i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                if (mask == 0) { // only one box was hit
                    cur.index = nodes[cur.index].child[i];
                    goto TRAVERSE;
                }

                const long i2 = GetFirstBit(mask);
                mask = ClearBit(mask, i2);
                if (mask == 0) { // two boxes were hit
                    if (res_dist[i] < res_dist[i2]) {
                        st.push(nodes[cur.index].child[i2], res_dist[i2]);
                        cur.index = nodes[cur.index].child[i];
                    } else {
                        st.push(nodes[cur.index].child[i], res_dist[i]);
                        cur.index = nodes[cur.index].child[i2];
                    }
                    goto TRAVERSE;
                }

                st.push(nodes[cur.index].child[i], res_dist[i]);
                st.push(nodes[cur.index].child[i2], res_dist[i2]);

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], res_dist[i]);
                if (mask == 0) { // three boxes were hit
                    st.sort_top3();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                i = GetFirstBit(mask);
                mask = ClearBit(mask, i);
                st.push(nodes[cur.index].child[i], res_dist[i]);
                if (mask == 0) { // four boxes were hit
                    st.sort_top4();
                    cur.index = st.pop_index();
                    goto TRAVERSE;
                }

                const uint32_t size_before = st.stack_size;

                // from five to eight boxes were hit
                do {
                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i]);
                } while (mask != 0);

                const int count = int(st.stack_size - size_before + 4);
                st.sort_topN(count);
                cur.index = st.pop_index();
                goto TRAVERSE;
            }
        } else {
            const int tri_start = int(nodes[cur.index].child[0] & PRIM_INDEX_BITS),
                      tri_end = int(tri_start + nodes[cur.index].child[1]);
            const bool hit_found = IntersectTris_AnyHit<S>(ro, rd, mtris, materials, tri_indices, tri_start, tri_end,
                                                           inter_prim_index, inter_t, inter_u, inter_v);
            if (hit_found) {
                const bool is_backfacing = inter_prim_index < 0;
                const uint32_t prim_index = is_backfacing ? -inter_prim_index - 1 : inter_prim_index;

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return 2;
                }
            }
            res |= hit_found;
        }
    }

    return res ? 1 : 0;
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::BRDF_PrincipledDiffuse(const simd_fvec<S> V[3], const simd_fvec<S> N[3],
                                                      const simd_fvec<S> L[3], const simd_fvec<S> H[3],
                                                      const simd_fvec<S> &roughness) {
    const simd_fvec<S> N_dot_L = dot3(N, L);
    const simd_fvec<S> N_dot_V = dot3(N, V);

    const simd_fvec<S> FL = schlick_weight(N_dot_L);
    const simd_fvec<S> FV = schlick_weight(N_dot_V);

    const simd_fvec<S> L_dot_H = dot3(L, H);
    const simd_fvec<S> Fd90 = 0.5f + 2.0f * L_dot_H * L_dot_H * roughness;
    const simd_fvec<S> Fd = mix(simd_fvec<S>{1.0f}, Fd90, FL) * mix(simd_fvec<S>{1.0f}, Fd90, FV);

    simd_fvec<S> ret = 0.0f;
    where(N_dot_L > 0.0f, ret) = Fd;

    return ret;
}

template <int S>
void Ray::NS::Evaluate_OrenDiffuse_BSDF(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                                        const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                                        simd_fvec<S> out_color[4]) {
    const simd_fvec<S> sigma = roughness;
    const simd_fvec<S> div = 1.0f / (PI + ((3.0f * PI - 4.0f) / 6.0f) * sigma);

    const simd_fvec<S> a = 1.0f * div;
    const simd_fvec<S> b = sigma * div;

    ////

    const simd_fvec<S> nl = max(dot3(N, L), 0.0f);
    const simd_fvec<S> nv = max(dot3(N, V), 0.0f);
    simd_fvec<S> t = dot3(L, V) - nl * nv;

    where(t > 0.0f, t) /= (max(nl, nv) + FLT_MIN);

    const simd_fvec<S> is = nl * (a + b * t);

    UNROLLED_FOR(i, 3, { out_color[i] = is * base_color[i]; })
    out_color[3] = 0.5f / PI;
}

template <int S>
void Ray::NS::Sample_OrenDiffuse_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                      const simd_fvec<S> I[3], const simd_fvec<S> &roughness,
                                      const simd_fvec<S> base_color[3], const simd_fvec<S> &rand_u,
                                      const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]) {
    const simd_fvec<S> phi = 2 * PI * rand_v;
    const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);
    const simd_fvec<S> dir = sqrt(1.0f - rand_u * rand_u);

    const simd_fvec<S> V[3] = {dir * cos_phi, dir * sin_phi, rand_u}; // in tangent-space
    world_from_tangent(T, B, N, V, out_V);

    const simd_fvec<S> neg_I[3] = {-I[0], -I[1], -I[2]};
    Evaluate_OrenDiffuse_BSDF(neg_I, N, out_V, roughness, base_color, out_color);
}

template <int S>
void Ray::NS::Evaluate_PrincipledDiffuse_BSDF(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                                              const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                                              const simd_fvec<S> sheen_color[3], const bool uniform_sampling,
                                              simd_fvec<S> out_color[4]) {
    simd_fvec<S> weight, pdf;
    if (uniform_sampling) {
        weight = 2 * dot3(N, L);
        pdf = 0.5f / PI;
    } else {
        weight = 1.0f;
        pdf = dot3(N, L) / PI;
    }

    simd_fvec<S> H[3] = {L[0] + V[0], L[1] + V[1], L[2] + V[2]};
    safe_normalize(H);

    const simd_fvec<S> dot_VH = dot3(V, H);
    UNROLLED_FOR(i, 3, { where(dot_VH < 0.0f, H[i]) = -H[i]; })

    weight *= BRDF_PrincipledDiffuse(V, N, L, H, roughness);
    UNROLLED_FOR(i, 3, { out_color[i] = base_color[i] * weight; })

    const simd_fvec<S> FH = PI * schlick_weight(dot3(L, H));
    UNROLLED_FOR(i, 3, { out_color[i] += FH * sheen_color[i]; })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_PrincipledDiffuse_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                            const simd_fvec<S> I[3], const simd_fvec<S> &roughness,
                                            const simd_fvec<S> base_color[3], const simd_fvec<S> sheen_color[3],
                                            const bool uniform_sampling, const simd_fvec<S> &rand_u,
                                            const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                                            simd_fvec<S> out_color[4]) {
    const simd_fvec<S> phi = 2 * PI * rand_v;
    const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);

    simd_fvec<S> V[3];
    if (uniform_sampling) {
        const simd_fvec<S> dir = sqrt(1.0f - rand_u * rand_u);

        // in tangent-space
        V[0] = dir * cos_phi;
        V[1] = dir * sin_phi;
        V[2] = rand_u;
    } else {
        const simd_fvec<S> dir = sqrt(rand_u);
        const simd_fvec<S> k = sqrt(1.0f - rand_u);

        // in tangent-space
        V[0] = dir * cos_phi;
        V[1] = dir * sin_phi;
        V[2] = k;
    }

    world_from_tangent(T, B, N, V, out_V);

    const simd_fvec<S> neg_I[3] = {-I[0], -I[1], -I[2]};
    Evaluate_PrincipledDiffuse_BSDF(neg_I, N, out_V, roughness, base_color, sheen_color, uniform_sampling, out_color);
}

template <int S>
void Ray::NS::Evaluate_GGXSpecular_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                        const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> &alpha_x,
                                        const simd_fvec<S> &alpha_y, const simd_fvec<S> &spec_ior,
                                        const simd_fvec<S> &spec_F0, const simd_fvec<S> spec_col[3],
                                        simd_fvec<S> out_color[4]) {
#if USE_VNDF_GGX_SAMPLING == 1
    const simd_fvec<S> D = D_GGX(sampled_normal_ts, alpha_x, alpha_y);
#else
    const float D = D_GTR2(sampled_normal_ts[2], alpha_x);
#endif

    const simd_fvec<S> G = G1(view_dir_ts, alpha_x, alpha_y) * G1(reflected_dir_ts, alpha_x, alpha_y);

    const simd_fvec<S> FH =
        (fresnel_dielectric_cos(dot3(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0f - spec_F0);

    simd_fvec<S> F[3];
    UNROLLED_FOR(i, 3, { F[i] = mix(spec_col[i], simd_fvec<S>{1.0f}, FH); })

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2] * reflected_dir_ts[2]);
    UNROLLED_FOR(i, 3, {
        F[i] *= safe_div_pos(D * G, denom);
        where(denom == 0.0f, F[i]) = 0.0f;
    })

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf = safe_div_pos(
        D * G1(view_dir_ts, alpha_x, alpha_y) * max(dot3(view_dir_ts, sampled_normal_ts), 0.0f), abs(view_dir_ts[2]));
    where(abs(view_dir_ts[2]) == 0.0f, pdf) = 0.0f;

    const simd_fvec<S> div = 4.0f * dot3(view_dir_ts, sampled_normal_ts);
    where(div != 0.0f, pdf) = safe_div(pdf, div);
#else
    const float pdf = D * sampled_normal_ts[2] / (4.0f * dot3(view_dir_ts, sampled_normal_ts));
#endif

    UNROLLED_FOR(i, 3, { out_color[i] = F[i] * max(reflected_dir_ts[2], 0.0f); })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_GGXSpecular_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                      const simd_fvec<S> I[3], const simd_fvec<S> &roughness,
                                      const simd_fvec<S> &anisotropic, const simd_fvec<S> &spec_ior,
                                      const simd_fvec<S> &spec_F0, const simd_fvec<S> spec_col[3],
                                      const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                                      simd_fvec<S> out_color[4]) {
    const simd_fvec<S> roughness2 = sqr(roughness);
    const simd_fvec<S> aspect = sqrt(1.0f - 0.9f * anisotropic);

    const simd_fvec<S> alpha_x = roughness2 / aspect;
    const simd_fvec<S> alpha_y = roughness2 * aspect;

    const simd_ivec<S> is_mirror = simd_cast(alpha_x * alpha_y < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        reflect(I, N, dot3(N, I), out_V);
        const simd_fvec<S> FH = (fresnel_dielectric_cos(dot3(out_V, N), spec_ior) - spec_F0) / (1.0f - spec_F0);
        UNROLLED_FOR(i, 3, { out_color[i] = mix(spec_col[i], simd_fvec<S>{1.0f}, FH) * 1e6f; })
        out_color[3] = 1e6f;
    }

    const simd_ivec<S> is_glossy = ~is_mirror;
    if (is_glossy.all_zeros()) {
        return;
    }

    const simd_fvec<S> nI[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, nI, view_dir_ts);
    safe_normalize(view_dir_ts);

    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, max(alpha_x, FLT_EPS), max(alpha_y, FLT_EPS), rand_u, rand_v, sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif

    const simd_fvec<S> dot_N_V = -dot3(sampled_normal_ts, view_dir_ts);
    simd_fvec<S> reflected_dir_ts[3];
    const simd_fvec<S> _view_dir_ts[3] = {-view_dir_ts[0], -view_dir_ts[1], -view_dir_ts[2]};
    reflect(_view_dir_ts, sampled_normal_ts, dot_N_V, reflected_dir_ts);
    safe_normalize(reflected_dir_ts);

    simd_fvec<S> glossy_V[3], glossy_F[4];
    world_from_tangent(T, B, N, reflected_dir_ts, glossy_V);
    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, max(alpha_x, 1e-7f),
                              max(alpha_y, 1e-7f), spec_ior, spec_F0, spec_col, glossy_F);

    UNROLLED_FOR(i, 3, { where(is_glossy, out_V[i]) = glossy_V[i]; })
    UNROLLED_FOR(i, 4, { where(is_glossy, out_color[i]) = glossy_F[i]; })
}

template <int S>
void Ray::NS::Evaluate_GGXRefraction_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                          const simd_fvec<S> refr_dir_ts[3], const simd_fvec<S> &roughness2,
                                          const simd_fvec<S> &eta, const simd_fvec<S> refr_col[3],
                                          simd_fvec<S> out_color[4]) {
#if USE_VNDF_GGX_SAMPLING == 1
    const simd_fvec<S> D = D_GGX(sampled_normal_ts, roughness2, roughness2);
#else
    const float D = D_GTR2(sampled_normal_ts[2], roughness2);
#endif

    const simd_fvec<S> G1o = G1(refr_dir_ts, roughness2, roughness2);
    const simd_fvec<S> G1i = G1(view_dir_ts, roughness2, roughness2);

    const simd_fvec<S> denom = dot3(refr_dir_ts, sampled_normal_ts) + dot3(view_dir_ts, sampled_normal_ts) * eta;
    const simd_fvec<S> jacobian = safe_div_pos(max(-dot3(refr_dir_ts, sampled_normal_ts), 0.0f), denom * denom);

    simd_fvec<S> F = safe_div(D * G1i * G1o * max(dot3(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian,
                              (/*-refr_dir_ts[2] */ view_dir_ts[2]));

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf = safe_div(D * G1o * max(dot3(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian, view_dir_ts[2]);
#else
    // const float pdf = D * std::max(sampled_normal_ts[2], 0.0f) * jacobian;
    const float pdf = safe_div(D * sampled_normal_ts[2] * std::max(-dot3(refr_dir_ts, sampled_normal_ts), 0.0f), denom);
#endif

    const simd_fvec<S> is_valid = (refr_dir_ts[2] < 0.0f) & (view_dir_ts[2] > 0.0f);

    UNROLLED_FOR(i, 3, {
        out_color[i] = 0.0f;
        where(is_valid, out_color[i]) = F * refr_col[i];
    })
    out_color[3] = 0.0f;
    where(is_valid, out_color[3]) = pdf;
}

template <int S>
void Ray::NS::Sample_GGXRefraction_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                        const simd_fvec<S> I[3], const simd_fvec<S> &roughness, const simd_fvec<S> &eta,
                                        const simd_fvec<S> refr_col[3], const simd_fvec<S> &rand_u,
                                        const simd_fvec<S> &rand_v, simd_fvec<S> out_V[4], simd_fvec<S> out_color[4]) {
    const simd_fvec<S> roughness2 = sqr(roughness);
    const simd_ivec<S> is_mirror = simd_cast(sqr(roughness2) < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        const simd_fvec<S> cosi = -dot3(I, N);
        const simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);

        const simd_fvec<S> m = eta * cosi - safe_sqrt(cost2);
        UNROLLED_FOR(i, 3, { out_V[i] = eta * I[i] + m * N[i]; })
        safe_normalize(out_V);

        out_V[3] = m;
        UNROLLED_FOR(i, 3, {
            out_color[i] = refr_col[i] * 1e6f;
            where(cost2 < 0, out_color[i]) = 0.0f;
        })
        out_color[3] = 1e6f;
        where(cost2 < 0, out_color[3]) = 0.0f;
    }

    const simd_ivec<S> is_glossy = ~is_mirror;
    if (is_glossy.all_zeros()) {
        return;
    }

    const simd_fvec<S> neg_I[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, neg_I, view_dir_ts);
    safe_normalize(view_dir_ts);

    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, roughness2, roughness2, rand_u, rand_v, sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif

    const simd_fvec<S> cosi = dot3(view_dir_ts, sampled_normal_ts);
    const simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);

    UNROLLED_FOR(i, 4, { where(is_glossy, out_color[i]) = 0.0f; })

    const simd_ivec<S> cost2_positive = simd_cast(cost2 >= 0.0f);
    if ((is_glossy & cost2_positive).not_all_zeros()) {
        const simd_fvec<S> m = eta * cosi - safe_sqrt(cost2);
        simd_fvec<S> refr_dir_ts[3];
        UNROLLED_FOR(i, 3, { refr_dir_ts[i] = -eta * view_dir_ts[i] + m * sampled_normal_ts[i]; })
        safe_normalize(refr_dir_ts);

        simd_fvec<S> glossy_V[3], glossy_F[4];
        world_from_tangent(T, B, N, refr_dir_ts, glossy_V);
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, roughness2, eta, refr_col, glossy_F);

        UNROLLED_FOR(i, 3, { where(is_glossy & cost2_positive, out_V[i]) = glossy_V[i]; })
        UNROLLED_FOR(i, 4, { where(is_glossy & cost2_positive, out_color[i]) = glossy_F[i]; })
    }
}

template <int S>
void Ray::NS::Evaluate_PrincipledClearcoat_BSDF(const simd_fvec<S> view_dir_ts[3],
                                                const simd_fvec<S> sampled_normal_ts[3],
                                                const simd_fvec<S> reflected_dir_ts[3],
                                                const simd_fvec<S> &clearcoat_roughness2,
                                                const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                                simd_fvec<S> out_color[4]) {
    const simd_fvec<S> D = D_GTR1(sampled_normal_ts[2], simd_fvec<S>{clearcoat_roughness2});
    // Always assume roughness of 0.25 for clearcoat
    const simd_fvec<S> clearcoat_alpha = (0.25f * 0.25f);
    const simd_fvec<S> G =
        G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) * G1(reflected_dir_ts, clearcoat_alpha, clearcoat_alpha);

    const simd_fvec<S> FH =
        (fresnel_dielectric_cos(dot3(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
        (1.0f - clearcoat_F0);
    simd_fvec<S> F = mix(simd_fvec<S>{0.04f}, simd_fvec<S>{1.0f}, FH);

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2]) * abs(reflected_dir_ts[2]);
    F *= safe_div_pos(D * G, denom);
    where(denom == 0.0f, F) = 0.0f;

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf = safe_div_pos(D * G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) *
                                        max(dot3(view_dir_ts, sampled_normal_ts), 0.0f),
                                    abs(view_dir_ts[2]));
    const simd_fvec<S> div = 4.0f * dot3(view_dir_ts, sampled_normal_ts);
    where(div != 0.0f, pdf) = safe_div_pos(pdf, div);
#else
    float pdf = D * sampled_normal_ts[2] / (4.0f * dot3(view_dir_ts, sampled_normal_ts));
#endif

    F *= max(reflected_dir_ts[2], 0.0f);

    UNROLLED_FOR(i, 3, { out_color[i] = F; })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_PrincipledClearcoat_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                              const simd_fvec<S> I[3], const simd_fvec<S> &clearcoat_roughness2,
                                              const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                              const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                                              simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]) {
    const simd_ivec<S> is_mirror = simd_cast(sqr(clearcoat_roughness2) < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        reflect(I, N, dot3(N, I), out_V);

        const simd_fvec<S> FH =
            (fresnel_dielectric_cos(dot3(out_V, N), clearcoat_ior) - clearcoat_F0) / (1.0f - clearcoat_F0);
        const simd_fvec<S> F = mix(simd_fvec<S>{0.04f}, simd_fvec<S>{1.0f}, FH);

        UNROLLED_FOR(i, 3, { out_color[i] = F * 1e6f; })
        out_color[3] = 1e6f;
    }

    const simd_ivec<S> is_glossy = ~is_mirror;
    if (is_glossy.all_zeros()) {
        return;
    }

    const simd_fvec<S> neg_I[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, neg_I, view_dir_ts);
    safe_normalize(view_dir_ts);

    // NOTE: GTR1 distribution is not used for sampling because Cycles does it this way (???!)
    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, clearcoat_roughness2, clearcoat_roughness2, rand_u, rand_v, sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(clearcoat_roughness2, rand_u, rand_v);
#endif

    const simd_fvec<S> dot_N_V = -dot3(sampled_normal_ts, view_dir_ts);
    simd_fvec<S> reflected_dir_ts[3];
    const simd_fvec<S> _view_dir_ts[3] = {-view_dir_ts[0], -view_dir_ts[1], -view_dir_ts[2]};
    reflect(_view_dir_ts, sampled_normal_ts, dot_N_V, reflected_dir_ts);
    safe_normalize(reflected_dir_ts);

    world_from_tangent(T, B, N, reflected_dir_ts, out_V);

    simd_fvec<S> glossy_F[4];
    Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, clearcoat_roughness2,
                                      clearcoat_ior, clearcoat_F0, glossy_F);

    UNROLLED_FOR(i, 4, { where(is_glossy, out_color[i]) = glossy_F[i]; })
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::Evaluate_EnvQTree(const float y_rotation, const simd_fvec4 *const *qtree_mips,
                                                 const int qtree_levels, const simd_fvec<S> L[3]) {
    int res = 2;
    int lod = qtree_levels - 1;

    simd_fvec<S> p[2];
    DirToCanonical(L, -y_rotation, p);
    simd_fvec<S> factor = 1.0f;

    while (lod >= 0) {
        const simd_ivec<S> x = clamp(simd_ivec<S>(p[0] * float(res)), 0, res - 1);
        const simd_ivec<S> y = clamp(simd_ivec<S>(p[1] * float(res)), 0, res - 1);

        simd_ivec<S> index = 0;
        index |= (x & 1) << 0;
        index |= (y & 1) << 1;

        const simd_ivec<S> qx = x / 2;
        const simd_ivec<S> qy = y / 2;

        simd_fvec<S> quad[4];
        UNROLLED_FOR_S(i, S, {
            const simd_fvec4 q = qtree_mips[lod][qy.template get<i>() * res / 2 + qx.template get<i>()];

            quad[0].template set<i>(q.template get<0>());
            quad[1].template set<i>(q.template get<1>());
            quad[2].template set<i>(q.template get<2>());
            quad[3].template set<i>(q.template get<3>());
        })
        const simd_fvec<S> total = quad[0] + quad[1] + quad[2] + quad[3];

        const simd_ivec<S> mask = simd_cast(total > 0.0f);
        if (mask.all_zeros()) {
            break;
        }

        where(mask, factor) *=
            4.0f * gather(value_ptr(quad[0]), index * S + simd_ivec<S>(ascending_counter, simd_mem_aligned)) / total;

        --lod;
        res *= 2;
    }

    return factor / (4.0f * PI);
}

template <int S>
void Ray::NS::Sample_EnvQTree(float y_rotation, const simd_fvec4 *const *qtree_mips, int qtree_levels,
                              const simd_fvec<S> &rand, const simd_fvec<S> &rx, const simd_fvec<S> &ry,
                              simd_fvec<S> out_V[4]) {
    int res = 2;
    float step = 1.0f / float(res);

    simd_fvec<S> sample = rand;
    int lod = qtree_levels - 1;

    simd_fvec<S> origin[2] = {{0.0f}, {0.0f}};
    simd_fvec<S> factor = 1.0f;

    while (lod >= 0) {
        const simd_ivec<S> qx = simd_ivec<S>(origin[0] * float(res)) / 2;
        const simd_ivec<S> qy = simd_ivec<S>(origin[1] * float(res)) / 2;

        simd_fvec<S> quad[4];
        UNROLLED_FOR_S(i, S, {
            const simd_fvec4 q = qtree_mips[lod][qy.template get<i>() * res / 2 + qx.template get<i>()];

            quad[0].template set<i>(q.template get<0>());
            quad[1].template set<i>(q.template get<1>());
            quad[2].template set<i>(q.template get<2>());
            quad[3].template set<i>(q.template get<3>());
        })

        const simd_fvec<S> top_left = quad[0];
        const simd_fvec<S> top_right = quad[1];
        simd_fvec<S> partial = top_left + quad[2];
        const simd_fvec<S> total = partial + top_right + quad[3];

        const simd_ivec<S> mask = simd_cast(total > 0.0f);
        if (mask.all_zeros()) {
            break;
        }

        simd_fvec<S> boundary = partial / total;

        simd_ivec<S> index = 0;

        { // left or right decision
            const simd_ivec<S> left_mask = simd_cast(sample < boundary);

            where(left_mask, sample) /= boundary;
            where(left_mask, boundary) = top_left / partial;

            partial = total - partial;
            where(~left_mask & mask, origin[0]) += step;
            where(~left_mask, sample) = (sample - boundary) / (1.0f - boundary);
            where(~left_mask, boundary) = top_right / partial;
            where(~left_mask, index) |= (1 << 0);
        }

        { // bottom or up decision
            const simd_ivec<S> bottom_mask = simd_cast(sample < boundary);

            where(bottom_mask, sample) /= boundary;

            where(~bottom_mask & mask, origin[1]) += step;
            where(~bottom_mask, sample) = (sample - boundary) / (1.0f - boundary);
            where(~bottom_mask, index) |= (1 << 1);
        }

        where(mask, factor) *=
            4.0f * gather(value_ptr(quad[0]), index * S + simd_ivec<S>(ascending_counter, simd_mem_aligned)) / total;

        --lod;
        res *= 2;
        step *= 0.5f;
    }

    origin[0] += 2.0f * step * rx;
    origin[1] += 2.0f * step * ry;

    CanonicalToDir(origin, y_rotation, out_V);
    out_V[3] = factor / (4.0f * PI);
}

template <int S>
void Ray::NS::TransformRay(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const float *xform,
                           simd_fvec<S> out_ro[3], simd_fvec<S> out_rd[3]) {
    out_ro[0] = ro[0] * xform[0] + ro[1] * xform[4] + ro[2] * xform[8] + xform[12];
    out_ro[1] = ro[0] * xform[1] + ro[1] * xform[5] + ro[2] * xform[9] + xform[13];
    out_ro[2] = ro[0] * xform[2] + ro[1] * xform[6] + ro[2] * xform[10] + xform[14];

    out_rd[0] = rd[0] * xform[0] + rd[1] * xform[4] + rd[2] * xform[8];
    out_rd[1] = rd[0] * xform[1] + rd[1] * xform[5] + rd[2] * xform[9];
    out_rd[2] = rd[0] * xform[2] + rd[1] * xform[6] + rd[2] * xform[10];
}

void Ray::NS::TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3], float out_rd[3]) {
    out_ro[0] = ro[0] * xform[0] + ro[1] * xform[4] + ro[2] * xform[8] + xform[12];
    out_ro[1] = ro[0] * xform[1] + ro[1] * xform[5] + ro[2] * xform[9] + xform[13];
    out_ro[2] = ro[0] * xform[2] + ro[1] * xform[6] + ro[2] * xform[10] + xform[14];

    out_rd[0] = rd[0] * xform[0] + rd[1] * xform[4] + rd[2] * xform[8];
    out_rd[1] = rd[0] * xform[1] + rd[1] * xform[5] + rd[2] * xform[9];
    out_rd[2] = rd[0] * xform[2] + rd[1] * xform[6] + rd[2] * xform[10];
}

template <int S> void Ray::NS::TransformPoint(const simd_fvec<S> p[3], const float *xform, simd_fvec<S> out_p[3]) {
    out_p[0] = xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2] + xform[12];
    out_p[1] = xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2] + xform[13];
    out_p[2] = xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2] + xform[14];
}

template <int S> void Ray::NS::TransformDirection(const simd_fvec<S> xform[16], simd_fvec<S> p[3]) {
    const simd_fvec<S> temp0 = xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2];
    const simd_fvec<S> temp1 = xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2];
    const simd_fvec<S> temp2 = xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2];

    p[0] = temp0;
    p[1] = temp1;
    p[2] = temp2;
}

template <int S> void Ray::NS::TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]) {
    out_n[0] = n[0] * inv_xform[0] + n[1] * inv_xform[1] + n[2] * inv_xform[2];
    out_n[1] = n[0] * inv_xform[4] + n[1] * inv_xform[5] + n[2] * inv_xform[6];
    out_n[2] = n[0] * inv_xform[8] + n[1] * inv_xform[9] + n[2] * inv_xform[10];
}

template <int S>
void Ray::NS::TransformNormal(const simd_fvec<S> n[3], const simd_fvec<S> inv_xform[16], simd_fvec<S> out_n[3]) {
    out_n[0] = n[0] * inv_xform[0] + n[1] * inv_xform[1] + n[2] * inv_xform[2];
    out_n[1] = n[0] * inv_xform[4] + n[1] * inv_xform[5] + n[2] * inv_xform[6];
    out_n[2] = n[0] * inv_xform[8] + n[1] * inv_xform[9] + n[2] * inv_xform[10];
}

template <int S> void Ray::NS::TransformNormal(const simd_fvec<S> inv_xform[16], simd_fvec<S> inout_n[3]) {
    simd_fvec<S> temp0 = inout_n[0] * inv_xform[0] + inout_n[1] * inv_xform[1] + inout_n[2] * inv_xform[2];
    simd_fvec<S> temp1 = inout_n[0] * inv_xform[4] + inout_n[1] * inv_xform[5] + inout_n[2] * inv_xform[6];
    simd_fvec<S> temp2 = inout_n[0] * inv_xform[8] + inout_n[1] * inv_xform[9] + inout_n[2] * inv_xform[10];

    inout_n[0] = temp0;
    inout_n[1] = temp1;
    inout_n[2] = temp2;
}

template <int S> void Ray::NS::CanonicalToDir(const simd_fvec<S> p[2], float y_rotation, simd_fvec<S> out_d[3]) {
    const simd_fvec<S> cos_theta = 2 * p[0] - 1;
    simd_fvec<S> phi = 2 * PI * p[1] + y_rotation;
    where(phi < 0, phi) += 2 * PI;
    where(phi > 2 * PI, phi) -= 2 * PI;

    const simd_fvec<S> sin_theta = sqrt(1 - cos_theta * cos_theta);

    const simd_fvec<S> sin_phi = sin(phi);
    const simd_fvec<S> cos_phi = cos(phi);

    out_d[0] = sin_theta * cos_phi;
    out_d[1] = cos_theta;
    out_d[2] = -sin_theta * sin_phi;
}

template <int S> void Ray::NS::DirToCanonical(const simd_fvec<S> d[3], float y_rotation, simd_fvec<S> out_p[2]) {
    const simd_fvec<S> cos_theta = clamp(d[1], -1.0f, 1.0f);

    simd_fvec<S> phi;
    UNROLLED_FOR_S(i, S, { phi.template set<i>(-std::atan2(d[2].template get<i>(), d[0].template get<i>())); })

    phi += y_rotation;
    where(phi < 0, phi) += 2 * PI;
    where(phi > 2 * PI, phi) -= 2 * PI;

    out_p[0] = (cos_theta + 1.0f) / 2.0f;
    out_p[1] = phi / (2.0f * PI);
}

template <int S>
void Ray::NS::rotate_around_axis(const simd_fvec<S> p[3], const simd_fvec<S> axis[3], const simd_fvec<S> &angle,
                                 simd_fvec<S> out_p[3]) {
    const simd_fvec<S> costheta = cos(angle), sintheta = sin(angle);

    const simd_fvec<S> temp0 = ((costheta + (1.0f - costheta) * axis[0] * axis[0]) * p[0]) +
                               (((1.0f - costheta) * axis[0] * axis[1] - axis[2] * sintheta) * p[1]) +
                               (((1.0f - costheta) * axis[0] * axis[2] + axis[1] * sintheta) * p[2]);

    const simd_fvec<S> temp1 = (((1.0f - costheta) * axis[0] * axis[1] + axis[2] * sintheta) * p[0]) +
                               ((costheta + (1.0f - costheta) * axis[1] * axis[1]) * p[1]) +
                               (((1.0f - costheta) * axis[1] * axis[2] - axis[0] * sintheta) * p[2]);

    const simd_fvec<S> temp2 = (((1.0f - costheta) * axis[0] * axis[2] - axis[1] * sintheta) * p[0]) +
                               (((1.0f - costheta) * axis[1] * axis[2] + axis[0] * sintheta) * p[1]) +
                               ((costheta + (1.0f - costheta) * axis[2] * axis[2]) * p[2]);

    out_p[0] = temp0;
    out_p[1] = temp1;
    out_p[2] = temp2;
}

template <int S>
void Ray::NS::SampleNearest(const Ref::TexStorageBase *const textures[], const uint32_t index,
                            const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask,
                            simd_fvec<S> out_rgba[4]) {
    const Ref::TexStorageBase &storage = *textures[index >> 28];
    auto _lod = (simd_ivec<S>)lod;

    where(_lod > MAX_MIP_LEVEL, _lod) = MAX_MIP_LEVEL;

    for (int j = 0; j < S; j++) {
        if (!mask[j]) {
            continue;
        }

        const auto &pix = storage.Fetch(index & 0x00ffffff, uvs[0][j], uvs[1][j], _lod[j]);

        UNROLLED_FOR(i, 4, { out_rgba[i].set(j, static_cast<float>(pix.v[i])); });
    }

    const float k = 1.0f / 255.0f;
    UNROLLED_FOR(i, 4, { out_rgba[i] *= k; })
}

template <int S>
void Ray::NS::SampleBilinear(const Ref::TexStorageBase *const textures[], const uint32_t index,
                             const simd_fvec<S> uvs[2], const simd_ivec<S> &lod, const simd_ivec<S> &mask,
                             simd_fvec<S> out_rgba[4]) {
    const Ref::TexStorageBase &storage = *textures[index >> 28];

    const int tex = int(index & 0x00ffffff);

    simd_fvec<S> _uvs[2];
    _uvs[0] = fract(uvs[0]);
    _uvs[1] = fract(uvs[1]);

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        float img_size[2];
        storage.GetFRes(tex, lod[i], img_size);

        _uvs[0].set(i, _uvs[0][i] * img_size[0] - 0.5f);
        _uvs[1].set(i, _uvs[1][i] * img_size[1] - 0.5f);
    }

    const simd_fvec<S> k[2] = {fract(_uvs[0]), fract(_uvs[1])};

    simd_fvec<S> p0[4] = {}, p1[4] = {};

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = storage.Fetch(tex, int(_uvs[0][i]), int(_uvs[1][i]), lod[i]);
        const auto &p01 = storage.Fetch(tex, int(_uvs[0][i] + 1), int(_uvs[1][i]), lod[i]);
        const auto &p10 = storage.Fetch(tex, int(_uvs[0][i]), int(_uvs[1][i] + 1), lod[i]);
        const auto &p11 = storage.Fetch(tex, int(_uvs[0][i] + 1), int(_uvs[1][i] + 1), lod[i]);

        p0[0].set(i, p01.v[0] * k[0][i] + p00.v[0] * (1 - k[0][i]));
        p0[1].set(i, p01.v[1] * k[0][i] + p00.v[1] * (1 - k[0][i]));
        p0[2].set(i, p01.v[2] * k[0][i] + p00.v[2] * (1 - k[0][i]));
        p0[3].set(i, p01.v[3] * k[0][i] + p00.v[3] * (1 - k[0][i]));

        p1[0].set(i, p11.v[0] * k[0][i] + p10.v[0] * (1 - k[0][i]));
        p1[1].set(i, p11.v[1] * k[0][i] + p10.v[1] * (1 - k[0][i]));
        p1[2].set(i, p11.v[2] * k[0][i] + p10.v[2] * (1 - k[0][i]));
        p1[3].set(i, p11.v[3] * k[0][i] + p10.v[3] * (1 - k[0][i]));
    }

    where(mask, out_rgba[0]) = (p1[0] * k[1] + p0[0] * (1.0f - k[1]));
    where(mask, out_rgba[1]) = (p1[1] * k[1] + p0[1] * (1.0f - k[1]));
    where(mask, out_rgba[2]) = (p1[2] * k[1] + p0[2] * (1.0f - k[1]));
    where(mask, out_rgba[3]) = (p1[3] * k[1] + p0[3] * (1.0f - k[1]));
}

template <int S>
void Ray::NS::SampleTrilinear(const Ref::TexStorageBase *const textures[], const uint32_t index,
                              const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask,
                              simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> col1[4];
    SampleBilinear(textures, index, uvs, (simd_ivec<S>)floor(lod), mask, col1);
    simd_fvec<S> col2[4];
    SampleBilinear(textures, index, uvs, (simd_ivec<S>)ceil(lod), mask, col2);

    const simd_fvec<S> k = fract(lod);
    UNROLLED_FOR(i, 4, { out_rgba[i] = col1[i] * (1.0f - k) + col2[i] * k; })
}

template <int S>
void Ray::NS::SampleLatlong_RGBE(const Ref::TexStorageRGBA &storage, const uint32_t index, const simd_fvec<S> dir[3],
                                 const float y_rotation, const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]) {
    const simd_fvec<S> y = clamp(dir[1], -1.0f, 1.0f);

    simd_fvec<S> theta = 0.0f, phi = 0.0f;
    UNROLLED_FOR_S(i, S, {
        if (mask.template get<i>()) {
            theta.template set<i>(std::acos(y.template get<i>()) / PI);
            phi.template set<i>(std::atan2(dir[2].template get<i>(), dir[0].template get<i>()) + y_rotation);
        }
    })
    where(phi < 0.0f, phi) += 2 * PI;
    where(phi > 2 * PI, phi) -= 2 * PI;

    const simd_fvec<S> u = 0.5f * phi / PI;

    const int tex = int(index & 0x00ffffff);
    float sz[2];
    storage.GetFRes(tex, 0, sz);

    const simd_fvec<S> uvs[2] = {clamp(u * sz[0], 0.0f, sz[0] - 1.0f), clamp(theta * sz[1], 0.0f, sz[1] - 1.0f)};

    const simd_fvec<S> k[2] = {fract(uvs[0]), fract(uvs[1])};

    simd_fvec<S> _p00[3] = {}, _p01[3] = {}, _p10[3] = {}, _p11[3] = {};

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = storage.Get(tex, int(uvs[0][i] + 0), int(uvs[1][i] + 0), 0);
        const auto &p01 = storage.Get(tex, int(uvs[0][i] + 1), int(uvs[1][i] + 0), 0);
        const auto &p10 = storage.Get(tex, int(uvs[0][i] + 0), int(uvs[1][i] + 1), 0);
        const auto &p11 = storage.Get(tex, int(uvs[0][i] + 1), int(uvs[1][i] + 1), 0);

        float f = std::exp2(float(p00.v[3]) - 128.0f);
        _p00[0].set(i, to_norm_float(p00.v[0]) * f);
        _p00[1].set(i, to_norm_float(p00.v[1]) * f);
        _p00[2].set(i, to_norm_float(p00.v[2]) * f);

        f = std::exp2(float(p01.v[3]) - 128.0f);
        _p01[0].set(i, to_norm_float(p01.v[0]) * f);
        _p01[1].set(i, to_norm_float(p01.v[1]) * f);
        _p01[2].set(i, to_norm_float(p01.v[2]) * f);

        f = std::exp2(float(p10.v[3]) - 128.0f);
        _p10[0].set(i, to_norm_float(p10.v[0]) * f);
        _p10[1].set(i, to_norm_float(p10.v[1]) * f);
        _p10[2].set(i, to_norm_float(p10.v[2]) * f);

        f = std::exp2(float(p11.v[3]) - 128.0f);
        _p11[0].set(i, to_norm_float(p11.v[0]) * f);
        _p11[1].set(i, to_norm_float(p11.v[1]) * f);
        _p11[2].set(i, to_norm_float(p11.v[2]) * f);
    }

    const simd_fvec<S> p0X[3] = {_p01[0] * k[0] + _p00[0] * (1 - k[0]), _p01[1] * k[0] + _p00[1] * (1 - k[0]),
                                 _p01[2] * k[0] + _p00[2] * (1 - k[0])};
    const simd_fvec<S> p1X[3] = {_p11[0] * k[0] + _p10[0] * (1 - k[0]), _p11[1] * k[0] + _p10[1] * (1 - k[0]),
                                 _p11[2] * k[0] + _p10[2] * (1 - k[0])};

    out_rgb[0] = p1X[0] * k[1] + p0X[0] * (1.0f - k[1]);
    out_rgb[1] = p1X[1] * k[1] + p0X[1] * (1.0f - k[1]);
    out_rgb[2] = p1X[2] * k[1] + p0X[2] * (1.0f - k[1]);
}

template <int S>
void Ray::NS::IntersectScene(ray_data_t<S> &r, const simd_ivec<S> &mask, const int min_transp_depth,
                             const int max_transp_depth, const float random_seq[], const scene_data_t &sc,
                             const uint32_t root_index, const Ref::TexStorageBase *const textures[],
                             hit_data_t<S> &inter) {
    simd_fvec<S> ro[3] = {r.o[0], r.o[1], r.o[2]};

    const simd_fvec<S> rand_offset = construct_float(hash(r.xy));
    simd_ivec<S> rand_index = total_depth(r) * RAND_DIM_BOUNCE_COUNT;

    simd_ivec<S> keep_going = mask;
    while (keep_going.not_all_zeros()) {
        const simd_fvec<S> t_val = inter.t;

        if (sc.mnodes) {
            NS::Traverse_MacroTree_WithStack_ClosestHit(ro, r.d, keep_going, sc.mnodes, root_index, sc.mesh_instances,
                                                        sc.mi_indices, sc.meshes, sc.transforms, sc.mtris,
                                                        sc.tri_indices, inter);
        } else {
            NS::Traverse_MacroTree_WithStack_ClosestHit(ro, r.d, keep_going, sc.nodes, root_index, sc.mesh_instances,
                                                        sc.mi_indices, sc.meshes, sc.transforms, sc.tris,
                                                        sc.tri_indices, inter);
        }

        keep_going &= inter.mask;
        if (keep_going.all_zeros()) {
            break;
        }

        simd_ivec<S> tri_index = inter.prim_index;
        const simd_ivec<S> is_backfacing = (tri_index < 0);
        where(is_backfacing, tri_index) = -tri_index - 1;

        simd_ivec<S> mat_index = gather(reinterpret_cast<const int *>(sc.tri_materials), tri_index);

        where(~is_backfacing, mat_index) = mat_index & 0xffff; // use front material index
        where(is_backfacing, mat_index) = mat_index >> 16;     // use back material index
        where(~keep_going, mat_index) = 0xffff;

        const simd_ivec<S> solid_hit = (mat_index & MATERIAL_SOLID_BIT) != 0;
        keep_going &= ~solid_hit;
        if (keep_going.all_zeros()) {
            break;
        }

        mat_index &= MATERIAL_INDEX_BITS;

        const simd_fvec<S> w = 1.0f - inter.u - inter.v;

        const simd_ivec<S> vtx_indices[3] = {gather(reinterpret_cast<const int *>(sc.vtx_indices + 0), tri_index * 3),
                                             gather(reinterpret_cast<const int *>(sc.vtx_indices + 1), tri_index * 3),
                                             gather(reinterpret_cast<const int *>(sc.vtx_indices + 2), tri_index * 3)};

        simd_fvec<S> uvs[2];

        { // Fetch vertex uvs
            const float *vtx_uvs = &sc.vertices[0].t[0][0];
            const int VtxUVsStride = sizeof(vertex_t) / sizeof(float);

            UNROLLED_FOR(i, 2, {
                const simd_fvec<S> temp1 = gather(vtx_uvs + i, vtx_indices[0] * VtxUVsStride);
                const simd_fvec<S> temp2 = gather(vtx_uvs + i, vtx_indices[1] * VtxUVsStride);
                const simd_fvec<S> temp3 = gather(vtx_uvs + i, vtx_indices[2] * VtxUVsStride);

                uvs[i] = temp1 * w + temp2 * inter.u + temp3 * inter.v;
            })
        }

        simd_fvec<S> rand_pick = fract(gather(random_seq + RAND_DIM_BSDF_PICK, rand_index) + rand_offset);
        const simd_fvec<S> rand_term = fract(gather(random_seq + RAND_DIM_TERMINATE, rand_index) + rand_offset);

        { // resolve material
            simd_ivec<S> ray_queue[S];
            int index = 0, num = 1;

            ray_queue[0] = keep_going;

            while (index != num) {
                const int mask = ray_queue[index].movemask();
                uint32_t first_mi = mat_index[GetFirstBit(mask)];

                simd_ivec<S> same_mi = (mat_index == first_mi);
                simd_ivec<S> diff_mi = and_not(same_mi, ray_queue[index]);

                if (diff_mi.not_all_zeros()) {
                    ray_queue[num++] = diff_mi;
                }

                ray_queue[index] &= same_mi;

                if (first_mi != 0xffff) {
                    const material_t *mat = &sc.materials[first_mi];

                    while (mat->type == MixNode) {
                        simd_fvec<S> _mix_val = 1.0f;

                        if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
                            simd_fvec<S> mix[4] = {};
                            SampleBilinear(textures, mat->textures[BASE_TEXTURE], uvs, {0}, same_mi, mix);
                            _mix_val *= mix[0];
                        }
                        _mix_val *= mat->strength;

                        first_mi = 0xffff;

                        for (int i = 0; i < S; i++) {
                            if (!same_mi[i]) {
                                continue;
                            }

                            float mix_val = _mix_val[i];

                            if (rand_pick[i] > mix_val) {
                                mat_index.set(i, mat->textures[MIX_MAT1]);
                                rand_pick.set(i, safe_div_pos(rand_pick[i] - mix_val, 1.0f - mix_val));
                            } else {
                                mat_index.set(i, mat->textures[MIX_MAT2]);
                                rand_pick.set(i, safe_div_pos(rand_pick[i], mix_val));
                            }

                            if (first_mi == 0xffff) {
                                first_mi = mat_index[i];
                            }
                        }

                        const simd_ivec<S> _same_mi = (mat_index == first_mi);
                        diff_mi = and_not(_same_mi, same_mi);
                        same_mi = _same_mi;

                        if (diff_mi.not_all_zeros()) {
                            ray_queue[num++] = diff_mi;
                        }

                        ray_queue[index] &= same_mi;

                        mat = &sc.materials[first_mi];
                    }

                    if (mat->type != TransparentNode) {
                        where(ray_queue[index], keep_going) = 0;
                        index++;
                        continue;
                    }

#if USE_PATH_TERMINATION
                    const simd_ivec<S> can_terminate_path = (r.depth >> 24) > min_transp_depth;
#else
                    const simd_ivec<S> can_terminate_path = 0;
#endif
                    const simd_fvec<S> lum = max(r.c[0], max(r.c[1], r.c[2]));
                    const simd_fvec<S> &p = rand_term;
                    simd_fvec<S> q = 0.0f;
                    where(can_terminate_path, q) = max(0.05f, 1.0f - lum);

                    const simd_ivec<S> terminate =
                        simd_cast(p < q) | simd_cast(lum == 0.0f) | ((r.depth >> 24) + 1 >= max_transp_depth);

                    UNROLLED_FOR(i, 3, {
                        where(ray_queue[index] & terminate, r.c[i]) = 0.0f;
                        where(ray_queue[index] & ~terminate, r.c[i]) *= safe_div_pos(mat->base_color[i], 1.0f - q);
                    })
                }

                index++;
            }
        }

        const simd_fvec<S> t = inter.t + HIT_BIAS;
        UNROLLED_FOR(i, 3, { where(keep_going, ro[i]) += r.d[i] * t; })

        // discard current intersection
        where(keep_going, inter.mask) = 0;
        where(keep_going, inter.t) = t_val - inter.t;

        where(keep_going, r.depth) += 0x01000000;

        rand_index += RAND_DIM_BOUNCE_COUNT;
    }

    inter.t += distance(r.o, ro);
}

template <int S>
void Ray::NS::IntersectScene(const shadow_ray_t<S> &r, const simd_ivec<S> &mask, const int max_transp_depth,
                             const scene_data_t &sc, uint32_t node_index, const Ref::TexStorageBase *const textures[],
                             simd_fvec<S> rc[3]) {
    simd_fvec<S> ro[3] = {r.o[0], r.o[1], r.o[2]};
    UNROLLED_FOR(i, 3, { rc[i] = r.c[i]; })
    simd_fvec<S> dist = r.dist;
    where(r.dist < 0.0f, dist) = MAX_DIST;
    simd_ivec<S> depth = (r.depth >> 24);

    simd_ivec<S> keep_going = simd_cast(dist > HIT_EPS) & mask;
    while (keep_going.not_all_zeros()) {
        hit_data_t<S> inter;
        inter.t = dist;

        simd_ivec<S> solid_hit;
        if (sc.mnodes) {
            solid_hit = Traverse_MacroTree_WithStack_AnyHit(ro, r.d, keep_going, sc.mnodes, node_index,
                                                            sc.mesh_instances, sc.mi_indices, sc.meshes, sc.transforms,
                                                            sc.mtris, sc.tri_materials, sc.tri_indices, inter);
        } else {
            solid_hit = Traverse_MacroTree_WithStack_AnyHit(ro, r.d, keep_going, sc.nodes, node_index,
                                                            sc.mesh_instances, sc.mi_indices, sc.meshes, sc.transforms,
                                                            sc.tris, sc.tri_materials, sc.tri_indices, inter);
        }

        const simd_ivec<S> terminate_mask = solid_hit | (depth > max_transp_depth);
        UNROLLED_FOR(i, 3, { where(terminate_mask, rc[i]) = 0.0f; })

        keep_going &= inter.mask & ~terminate_mask;
        if (keep_going.all_zeros()) {
            break;
        }

        const simd_fvec<S> w = 1.0f - inter.u - inter.v;

        simd_ivec<S> tri_index = inter.prim_index;
        const simd_ivec<S> is_backfacing = (tri_index < 0);
        where(is_backfacing, tri_index) = -tri_index - 1;

        const simd_ivec<S> vtx_indices[3] = {gather(reinterpret_cast<const int *>(sc.vtx_indices + 0), tri_index * 3),
                                             gather(reinterpret_cast<const int *>(sc.vtx_indices + 1), tri_index * 3),
                                             gather(reinterpret_cast<const int *>(sc.vtx_indices + 2), tri_index * 3)};

        simd_fvec<S> sh_uvs[2];

        { // Fetch vertex uvs
            const float *vtx_uvs = &sc.vertices[0].t[0][0];
            const int VtxUVsStride = sizeof(vertex_t) / sizeof(float);

            UNROLLED_FOR(i, 2, {
                const simd_fvec<S> temp1 = gather(vtx_uvs + i, vtx_indices[0] * VtxUVsStride);
                const simd_fvec<S> temp2 = gather(vtx_uvs + i, vtx_indices[1] * VtxUVsStride);
                const simd_fvec<S> temp3 = gather(vtx_uvs + i, vtx_indices[2] * VtxUVsStride);

                sh_uvs[i] = temp1 * w + temp2 * inter.u + temp3 * inter.v;
            })
        }

        simd_ivec<S> mat_index = gather(reinterpret_cast<const int *>(sc.tri_materials), tri_index) &
                                 simd_ivec<S>((MATERIAL_INDEX_BITS << 16) | MATERIAL_INDEX_BITS);

        where(~is_backfacing, mat_index) = mat_index & 0xffff; // use front material index
        where(is_backfacing, mat_index) = mat_index >> 16;     // use back material index
        where(~inter.mask, mat_index) = 0xffff;

        { // resolve material
            simd_ivec<S> ray_queue[S];
            int index = 0, num = 1;

            ray_queue[0] = inter.mask;

            while (index != num) {
                const int mask = ray_queue[index].movemask();
                const uint32_t first_mi = mat_index[GetFirstBit(mask)];

                simd_ivec<S> same_mi = (mat_index == first_mi);
                simd_ivec<S> diff_mi = and_not(same_mi, ray_queue[index]);

                if (diff_mi.not_all_zeros()) {
                    ray_queue[num] = diff_mi;
                    num++;
                }

                if (first_mi != 0xffff) {
                    struct {
                        uint32_t index;
                        simd_fvec<S> weight;
                    } stack[16];
                    int stack_size = 0;

                    stack[stack_size++] = {first_mi, 1.0f};

                    simd_fvec<S> throughput[3] = {};

                    while (stack_size--) {
                        const material_t *mat = &sc.materials[stack[stack_size].index];
                        const simd_fvec<S> weight = stack[stack_size].weight;

                        // resolve mix material
                        if (mat->type == MixNode) {
                            simd_fvec<S> mix_val = mat->strength;
                            if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
                                simd_fvec<S> mix[4] = {};
                                SampleBilinear(textures, mat->textures[BASE_TEXTURE], sh_uvs, {0}, same_mi, mix);
                                mix_val *= mix[0];
                            }

                            stack[stack_size++] = {mat->textures[MIX_MAT1], weight * (1.0f - mix_val)};
                            stack[stack_size++] = {mat->textures[MIX_MAT2], weight * mix_val};
                        } else if (mat->type == TransparentNode) {
                            UNROLLED_FOR(i, 3, { throughput[i] += weight * mat->base_color[i]; })
                        }
                    }

                    UNROLLED_FOR(i, 3, { where(same_mi & keep_going, rc[i]) *= throughput[i]; })
                }

                index++;
            }
        }

        simd_fvec<S> t = inter.t + HIT_BIAS;
        UNROLLED_FOR(i, 3, { ro[i] += r.d[i] * t; })
        dist -= t;

        where(keep_going, depth) += 1;

        // update mask
        keep_going &= simd_cast(dist > HIT_EPS) & simd_cast(lum(rc) >= FLT_EPS);
    }
}

// Pick point on any light source for evaluation
template <int S>
void Ray::NS::SampleLightSource(const simd_fvec<S> P[3], const simd_fvec<S> T[3], const simd_fvec<S> B[3],
                                const simd_fvec<S> N[3], const scene_data_t &sc,
                                const Ref::TexStorageBase *const textures[], const float random_seq[],
                                const simd_ivec<S> &rand_index, const simd_fvec<S> sample_off[2],
                                const simd_ivec<S> &ray_mask, light_sample_t<S> &ls) {
    const simd_fvec<S> ri = fract(gather(random_seq + RAND_DIM_LIGHT_PICK, rand_index) + sample_off[0]);
    const simd_fvec<S> ru = fract(gather(random_seq + RAND_DIM_LIGHT_U, rand_index) + sample_off[0]);
    const simd_fvec<S> rv = fract(gather(random_seq + RAND_DIM_LIGHT_V, rand_index) + sample_off[1]);

    const simd_ivec<S> light_index = min(simd_ivec<S>{ri * float(sc.li_indices.size())}, int(sc.li_indices.size() - 1));

    simd_ivec<S> ray_queue[S];
    ray_queue[0] = ray_mask;

    int index = 0, num = 1;
    while (index != num) {
        const long mask = ray_queue[index].movemask();
        const uint32_t first_li = light_index[GetFirstBit(mask)];

        const simd_ivec<S> same_li = (light_index == first_li);
        const simd_ivec<S> diff_li = and_not(same_li, ray_queue[index]);

        if (diff_li.not_all_zeros()) {
            ray_queue[index] &= same_li;
            ray_queue[num++] = diff_li;
        }

        const light_t &l = sc.lights[sc.li_indices[first_li]];

        UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) = l.col[i] * float(sc.li_indices.size()); })
        where(ray_queue[index], ls.cast_shadow) = l.cast_shadow ? -1 : 0;

        if (l.type == LIGHT_TYPE_SPHERE) {
            simd_fvec<S> center_to_surface[3];
            UNROLLED_FOR(i, 3, { center_to_surface[i] = P[i] - l.sph.pos[i]; })

            const simd_fvec<S> dist_to_center = length(center_to_surface);
            UNROLLED_FOR(i, 3, { center_to_surface[i] /= dist_to_center; })

            // sample hemisphere
            const simd_fvec<S> r = sqrt(max(0.0f, 1.0f - ru * ru));
            const simd_fvec<S> phi = 2.0f * PI * rv;

            const simd_fvec<S> sampled_dir[3] = {r * cos(phi), r * sin(phi), ru};

            simd_fvec<S> LT[3], LB[3];
            create_tbn(center_to_surface, LT, LB);

            simd_fvec<S> _sampled_dir[3];
            UNROLLED_FOR(i, 3, {
                _sampled_dir[i] =
                    LT[i] * sampled_dir[0] + LB[i] * sampled_dir[1] + center_to_surface[i] * sampled_dir[2];
            })

            simd_fvec<S> light_surf_pos[3];
            UNROLLED_FOR(i, 3, { light_surf_pos[i] = l.sph.pos[i] + _sampled_dir[i] * l.sph.radius; })

            simd_fvec<S> L[3];
            UNROLLED_FOR(i, 3, { L[i] = light_surf_pos[i] - P[i]; })

            simd_fvec<S> light_forward[3];
            UNROLLED_FOR(i, 3, { light_forward[i] = light_surf_pos[i] - l.sph.pos[i]; })
            normalize(light_forward);

            simd_fvec<S> lp_biased[3];
            offset_ray(light_surf_pos, light_forward, lp_biased);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })

            const simd_fvec<S> ls_dist = length(L);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = L[i] / ls_dist; })

            where(ray_queue[index], ls.area) = l.sph.area;

            const simd_fvec<S> cos_theta = abs(dot3(ls.L, light_forward));

            simd_fvec<S> pdf = safe_div_pos(ls_dist * ls_dist, 0.5f * ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sph.spot > 0.0f) {
                simd_fvec<S> _dot =
                    min(-ls.L[0] * l.sph.dir[0] - ls.L[1] * l.sph.dir[1] - ls.L[2] * l.sph.dir[2], 1.0f);
                simd_ivec<S> mask = simd_cast(_dot > 0.0f);
                if (mask.not_all_zeros()) {
                    simd_fvec<S> _angle = 0.0f;
                    UNROLLED_FOR_S(i, S, {
                        if (mask.template get<i>()) {
                            _angle.template set<i>(std::acos(_dot.template get<i>()));
                        }
                    })
                    const simd_fvec<S> k = clamp((l.sph.spot - _angle) / l.sph.blend, 0.0f, 1.0f);
                    UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= k; })
                }
                UNROLLED_FOR(i, 3, { where(~mask & ray_queue[index], ls.col[i]) = 0.0f; })
            }
        } else if (l.type == LIGHT_TYPE_DIR) {
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = l.dir.dir[i]; })
            if (l.dir.angle != 0.0f) {
                const float radius = std::tan(l.dir.angle);

                simd_fvec<S> V[3];
                MapToCone(ru, rv, ls.L, radius, V);
                safe_normalize(V);

                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = V[i]; })
            }

            where(ray_queue[index], ls.area) = 0.0f;
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = P[i] + ls.L[i]; })
            where(ray_queue[index], ls.dist_mul) = MAX_DIST;
            where(ray_queue[index], ls.pdf) = 1.0f;
        } else if (l.type == LIGHT_TYPE_RECT) {
            const simd_fvec<S> r1 = ru - 0.5f, r2 = rv - 0.5f;

            const simd_fvec<S> lp[3] = {l.rect.pos[0] + l.rect.u[0] * r1 + l.rect.v[0] * r2,
                                        l.rect.pos[1] + l.rect.u[1] * r1 + l.rect.v[1] * r2,
                                        l.rect.pos[2] + l.rect.u[2] * r1 + l.rect.v[2] * r2};

            simd_fvec<S> to_light[3];
            UNROLLED_FOR(i, 3, { to_light[i] = lp[i] - P[i]; })

            float light_forward[3];
            cross(l.rect.u, l.rect.v, light_forward);
            normalize(light_forward);

            simd_fvec<S> lp_biased[3], _light_forward[3] = {light_forward[0], light_forward[1], light_forward[2]};
            offset_ray(lp, _light_forward, lp_biased);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })

            const simd_fvec<S> ls_dist = length(to_light);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = to_light[i] / ls_dist; })

            where(ray_queue[index], ls.area) = l.rect.area;

            const simd_fvec<S> cos_theta =
                -ls.L[0] * light_forward[0] - ls.L[1] * light_forward[1] - ls.L[2] * light_forward[2];
            simd_fvec<S> pdf = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sky_portal != 0) {
                simd_fvec<S> env_col[3] = {sc.env.env_col[0], sc.env.env_col[1], sc.env.env_col[2]};
                if (sc.env.env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Ref::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                       sc.env.env_map_rotation, ray_queue[index], tex_col);
                    UNROLLED_FOR(i, 3, { env_col[i] *= tex_col[i]; })
                }
                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= sc.env.env_col[i]; })
                where(ray_queue[index], ls.from_env) = -1;
            }
        } else if (l.type == LIGHT_TYPE_DISK) {
            simd_fvec<S> offset[2] = {2.0f * ru - 1.0f, 2.0f * rv - 1.0f};
            const simd_ivec<S> mask = simd_cast(offset[0] != 0.0f & offset[1] != 0.0f);
            if (mask.not_all_zeros()) {
                simd_fvec<S> theta = 0.5f * PI - 0.25f * PI * safe_div(offset[0], offset[1]), r = offset[1];

                where(abs(offset[0]) > abs(offset[1]), r) = offset[0];
                where(abs(offset[0]) > abs(offset[1]), theta) = 0.25f * PI * safe_div(offset[1], offset[0]);

                where(mask, offset[0]) = 0.5f * r * cos(theta);
                where(mask, offset[1]) = 0.5f * r * sin(theta);
            }

            const simd_fvec<S> lp[3] = {l.disk.pos[0] + l.disk.u[0] * offset[0] + l.disk.v[0] * offset[1],
                                        l.disk.pos[1] + l.disk.u[1] * offset[0] + l.disk.v[1] * offset[1],
                                        l.disk.pos[2] + l.disk.u[2] * offset[0] + l.disk.v[2] * offset[1]};

            simd_fvec<S> to_light[3];
            UNROLLED_FOR(i, 3, { to_light[i] = lp[i] - P[i]; })

            const simd_fvec<S> ls_dist = length(to_light);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = to_light[i] / ls_dist; })

            where(ray_queue[index], ls.area) = l.disk.area;

            float light_forward[3];
            cross(l.disk.u, l.disk.v, light_forward);
            normalize(light_forward);

            simd_fvec<S> lp_biased[3], _light_forward[3] = {light_forward[0], light_forward[1], light_forward[2]};
            offset_ray(lp, _light_forward, lp_biased);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })

            const simd_fvec<S> cos_theta =
                -ls.L[0] * light_forward[0] - ls.L[1] * light_forward[1] - ls.L[2] * light_forward[2];
            simd_fvec<S> pdf = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sky_portal != 0) {
                simd_fvec<S> env_col[3] = {sc.env.env_col[0], sc.env.env_col[1], sc.env.env_col[2]};
                if (sc.env.env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Ref::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                       sc.env.env_map_rotation, ray_queue[index], tex_col);
                    UNROLLED_FOR(i, 3, { env_col[i] *= tex_col[i]; })
                }
                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= sc.env.env_col[i]; })
                where(ray_queue[index], ls.from_env) = -1;
            }
        } else if (l.type == LIGHT_TYPE_LINE) {
            simd_fvec<S> center_to_surface[3];
            UNROLLED_FOR(i, 3, { center_to_surface[i] = P[i] - l.line.pos[i]; })

            const float *light_dir = l.line.v;

            simd_fvec<S> light_u[3] = {center_to_surface[1] * light_dir[2] - center_to_surface[2] * light_dir[1],
                                       center_to_surface[2] * light_dir[0] - center_to_surface[0] * light_dir[2],
                                       center_to_surface[0] * light_dir[1] - center_to_surface[1] * light_dir[0]};
            normalize(light_u);

            const simd_fvec<S> light_v[3] = {light_u[1] * light_dir[2] - light_u[2] * light_dir[1],
                                             light_u[2] * light_dir[0] - light_u[0] * light_dir[2],
                                             light_u[0] * light_dir[1] - light_u[1] * light_dir[0]};

            const simd_fvec<S> phi = PI * ru;
            const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);

            const simd_fvec<S> normal[3] = {cos_phi * light_u[0] - sin_phi * light_v[0],
                                            cos_phi * light_u[1] - sin_phi * light_v[1],
                                            cos_phi * light_u[2] - sin_phi * light_v[2]};

            const simd_fvec<S> lp[3] = {
                l.line.pos[0] + normal[0] * l.line.radius + (rv - 0.5f) * light_dir[0] * l.line.height,
                l.line.pos[1] + normal[1] * l.line.radius + (rv - 0.5f) * light_dir[1] * l.line.height,
                l.line.pos[2] + normal[2] * l.line.radius + (rv - 0.5f) * light_dir[2] * l.line.height};

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp[i]; })

            simd_fvec<S> to_light[3];
            UNROLLED_FOR(i, 3, { to_light[i] = lp[i] - P[i]; })

            const simd_fvec<S> ls_dist = length(to_light);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = to_light[i] / ls_dist; })

            where(ray_queue[index], ls.area) = l.line.area;

            const simd_fvec<S> cos_theta = 1.0f - abs(dot3(ls.L, light_dir));
            simd_fvec<S> pdf = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            where(cos_theta == 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }
        } else if (l.type == LIGHT_TYPE_TRI) {
            const transform_t &ltr = sc.transforms[l.tri.xform_index];
            const uint32_t ltri_index = l.tri.tri_index;

            const vertex_t &v1 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 0]];
            const vertex_t &v2 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 1]];
            const vertex_t &v3 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 2]];

            const simd_fvec<S> r1 = sqrt(ru), r2 = rv;

            const simd_fvec<S> luvs[2] = {v1.t[0][0] * (1.0f - r1) + r1 * (v2.t[0][0] * (1.0f - r2) + v3.t[0][0] * r2),
                                          v1.t[0][1] * (1.0f - r1) + r1 * (v2.t[0][1] * (1.0f - r2) + v3.t[0][1] * r2)};

            const simd_fvec<S> lp_ls[3] = {v1.p[0] * (1.0f - r1) + r1 * (v2.p[0] * (1.0f - r2) + v3.p[0] * r2),
                                           v1.p[1] * (1.0f - r1) + r1 * (v2.p[1] * (1.0f - r2) + v3.p[1] * r2),
                                           v1.p[2] * (1.0f - r1) + r1 * (v2.p[2] * (1.0f - r2) + v3.p[2] * r2)};

            simd_fvec<S> lp[3];
            TransformPoint(lp_ls, ltr.xform, lp);

            const float temp1[3] = {v2.p[0] - v1.p[0], v2.p[1] - v1.p[1], v2.p[2] - v1.p[2]};
            const float temp2[3] = {v3.p[0] - v1.p[0], v3.p[1] - v1.p[1], v3.p[2] - v1.p[2]};
            float _light_forward[3], light_forward[3];
            cross(temp1, temp2, _light_forward);
            TransformDirection(_light_forward, ltr.xform, light_forward);

            const float light_fwd_len =
                std::sqrt(light_forward[0] * light_forward[0] + light_forward[1] * light_forward[1] +
                          light_forward[2] * light_forward[2]);
            where(ray_queue[index], ls.area) = 0.5f * light_fwd_len;
            UNROLLED_FOR(i, 3, { light_forward[i] /= light_fwd_len; })

            const simd_fvec<S> to_light[3] = {lp[0] - P[0], lp[1] - P[1], lp[2] - P[2]};
            const simd_fvec<S> ls_dist = length(to_light);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = safe_div_pos(to_light[i], ls_dist); })

            simd_fvec<S> cos_theta = dot3(ls.L, light_forward);

            simd_fvec<S> lp_biased[3], vlight_forward[3] = {light_forward[0], light_forward[1], light_forward[2]};
            UNROLLED_FOR(i, 3, { where(cos_theta >= 0.0f, vlight_forward[i]) = -vlight_forward[i]; })
            offset_ray(lp, vlight_forward, lp_biased);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })

            cos_theta = abs(cos_theta);
            where(simd_cast(cos_theta > 0.0f) & ray_queue[index], ls.pdf) =
                safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);

            const material_t &lmat = sc.materials[sc.tri_materials[ltri_index].front_mi & MATERIAL_INDEX_BITS];
            if (lmat.textures[BASE_TEXTURE] != 0xffffffff) {
                simd_fvec<S> tex_col[4] = {};
                SampleBilinear(textures, lmat.textures[BASE_TEXTURE], luvs, simd_ivec<S>{0}, ray_mask, tex_col);
                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= tex_col[i]; })
            }
        } else if (l.type == LIGHT_TYPE_ENV) {
            simd_fvec<S> dir_and_pdf[4];
            if (sc.env.qtree_levels) {
                // Sample environment using quadtree
                const auto *qtree_mips = reinterpret_cast<const simd_fvec4 *const *>(sc.env.qtree_mips);

                const simd_fvec<S> rand = ri * float(sc.li_indices.size()) - simd_fvec<S>(light_index);
                Sample_EnvQTree(sc.env.env_map_rotation, qtree_mips, sc.env.qtree_levels, rand, ru, rv, dir_and_pdf);
            } else {
                // Sample environment as hemishpere
                const simd_fvec<S> phi = 2 * PI * rv;
                const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);
                const simd_fvec<S> dir = sqrt(1.0f - ru * ru);

                const simd_fvec<S> V[3] = {dir * cos_phi, dir * sin_phi, ru}; // in tangent-space
                world_from_tangent(T, B, N, V, dir_and_pdf);
                dir_and_pdf[3] = 0.5f / PI;
            }

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = dir_and_pdf[i]; })

            simd_fvec<S> tex_col[3] = {1.0f, 1.0f, 1.0f};
            if (sc.env.env_map != 0xffffffff) {
                SampleLatlong_RGBE(*static_cast<const Ref::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                   sc.env.env_map_rotation, ray_queue[index], tex_col);
            }
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= sc.env.env_col[i] * tex_col[i]; })

            where(ray_queue[index], ls.area) = 1.0f;
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = P[i] + ls.L[i]; })
            where(ray_queue[index], ls.dist_mul) = MAX_DIST;
            where(ray_queue[index], ls.pdf) = dir_and_pdf[3];
            where(ray_queue[index], ls.from_env) = -1;
        }

        ++index;
    }
}

template <int S>
void Ray::NS::IntersectAreaLights(const ray_data_t<S> &r, const simd_ivec<S> &_ray_mask, const light_t lights[],
                                  Span<const uint32_t> visible_lights, const transform_t transforms[],
                                  hit_data_t<S> &inout_inter) {
    for (uint32_t li = 0; li < uint32_t(visible_lights.size()); ++li) {
        const uint32_t light_index = visible_lights[li];
        const light_t &l = lights[light_index];
        // Portal lights affect only missed rays
        const simd_ivec<S> ray_mask =
            _ray_mask & ~((l.sky_portal ? simd_ivec<S>{-1} : simd_ivec<S>{0}) & inout_inter.mask);
        if (ray_mask.all_zeros()) {
            continue;
        }

        const simd_fvec<S> no_shadow = simd_cast(l.cast_shadow ? simd_ivec<S>{0} : simd_ivec<S>{-1});
        if (l.type == LIGHT_TYPE_SPHERE) {
            const simd_fvec<S> op[3] = {l.sph.pos[0] - r.o[0], l.sph.pos[1] - r.o[1], l.sph.pos[2] - r.o[2]};
            const simd_fvec<S> b = dot3(op, r.d);
            simd_fvec<S> det = b * b - dot3(op, op) + l.sph.radius * l.sph.radius;

            simd_ivec<S> imask = simd_cast(det >= 0.0f) & ray_mask;
            if (imask.not_all_zeros()) {
                det = safe_sqrt(det);
                const simd_fvec<S> t1 = b - det, t2 = b + det;

                simd_fvec<S> mask1 = (t1 > HIT_EPS) & ((t1 < inout_inter.t) | no_shadow) & simd_cast(imask);
                const simd_fvec<S> mask2 =
                    (t2 > HIT_EPS) & ((t2 < inout_inter.t) | no_shadow) & simd_cast(imask) & ~mask1;

                if (l.sph.spot > 0.0f) {
                    const simd_fvec<S> _dot =
                        min(-r.d[0] * l.sph.dir[0] - r.d[1] * l.sph.dir[1] - r.d[2] * l.sph.dir[2], 1.0f);
                    mask1 &= (_dot > 0.0f);
                    const simd_ivec<S> imask1 = simd_cast(mask1);
                    if (imask1.not_all_zeros()) {
                        simd_fvec<S> _angle = 0.0f;
                        UNROLLED_FOR_S(i, S, {
                            if (imask1.template get<i>()) {
                                _angle.template set<i>(std::acos(_dot.template get<i>()));
                            }
                        })
                        mask1 &= (_angle <= l.sph.spot);
                    }
                }

                inout_inter.mask |= simd_cast(mask1 | mask2);

                where(mask1 | mask2, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                where(mask1, inout_inter.t) = t1;
                where(mask2, inout_inter.t) = t2;
            }
        } else if (l.type == LIGHT_TYPE_RECT) {
            float light_fwd[3];
            cross(l.rect.u, l.rect.v, light_fwd);
            normalize(light_fwd);

            const float plane_dist = dot3(light_fwd, l.rect.pos);

            const simd_fvec<S> cos_theta = dot3(r.d, light_fwd);
            const simd_fvec<S> t = safe_div_neg(plane_dist - dot3(light_fwd, r.o), cos_theta);

            const simd_ivec<S> imask =
                simd_cast((cos_theta < 0.0f) & (t > HIT_EPS) & ((t < inout_inter.t) | no_shadow)) & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot3(l.rect.u, l.rect.u);
                const float dot_v = dot3(l.rect.v, l.rect.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.rect.pos[0], p[1] - l.rect.pos[1], p[2] - l.rect.pos[2]};

                const simd_fvec<S> a1 = dot3(l.rect.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot3(l.rect.v, vi) / dot_v;

                const simd_fvec<S> final_mask =
                    (a1 >= -0.5f & a1 <= 0.5f) & (a2 >= -0.5f & a2 <= 0.5f) & simd_cast(imask);

                inout_inter.mask |= simd_cast(final_mask);
                where(final_mask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                where(final_mask, inout_inter.t) = t;
            }
        } else if (l.type == LIGHT_TYPE_DISK) {
            float light_fwd[3];
            cross(l.disk.u, l.disk.v, light_fwd);
            normalize(light_fwd);

            const float plane_dist = dot3(light_fwd, l.disk.pos);

            const simd_fvec<S> cos_theta = dot3(r.d, light_fwd);
            const simd_fvec<S> t = safe_div_neg(plane_dist - dot3(light_fwd, r.o), cos_theta);

            const simd_ivec<S> imask =
                simd_cast((cos_theta < 0.0f) & (t > HIT_EPS) & ((t < inout_inter.t) | no_shadow)) & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot3(l.disk.u, l.disk.u);
                const float dot_v = dot3(l.disk.v, l.disk.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.disk.pos[0], p[1] - l.disk.pos[1], p[2] - l.disk.pos[2]};

                const simd_fvec<S> a1 = dot3(l.disk.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot3(l.disk.v, vi) / dot_v;

                const simd_fvec<S> final_mask = (sqrt(a1 * a1 + a2 * a2) <= 0.5f) & simd_cast(imask);

                inout_inter.mask |= simd_cast(final_mask);
                where(final_mask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                where(final_mask, inout_inter.t) = t;
            }
        } else if (l.type == LIGHT_TYPE_LINE) {
            const float *light_dir = l.line.v;

            float light_v[3];
            cross(l.line.u, light_dir, light_v);

            simd_fvec<S> _ro[3] = {r.o[0] - l.line.pos[0], r.o[1] - l.line.pos[1], r.o[2] - l.line.pos[2]};
            const simd_fvec<S> ro[3] = {dot3(_ro, light_dir), dot3(_ro, l.line.u), dot3(_ro, light_v)};
            const simd_fvec<S> rd[3] = {dot3(r.d, light_dir), dot3(r.d, l.line.u), dot3(r.d, light_v)};

            const simd_fvec<S> A = rd[2] * rd[2] + rd[1] * rd[1];
            const simd_fvec<S> B = 2.0f * (rd[2] * ro[2] + rd[1] * ro[1]);
            const simd_fvec<S> C = ro[2] * ro[2] + ro[1] * ro[1] - l.line.radius * l.line.radius;

            simd_fvec<S> t0, t1;
            simd_ivec<S> imask = quadratic(A, B, C, t0, t1);
            imask &= simd_cast(t0 > HIT_EPS) & simd_cast(t1 > HIT_EPS);

            const simd_fvec<S> t = min(t0, t1);
            const simd_fvec<S> p[3] = {fmadd(rd[0], t, ro[0]), fmadd(rd[1], t, ro[1]), fmadd(rd[2], t, ro[2])};

            imask &= simd_cast(abs(p[0]) < 0.5f * l.line.height) & simd_cast((t < inout_inter.t) | no_shadow);

            inout_inter.mask |= imask;
            where(imask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
            where(imask, inout_inter.t) = t;
        }
    }
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::IntersectAreaLights(const shadow_ray_t<S> &r, simd_ivec<S> ray_mask,
                                                   const light_t lights[], Span<const uint32_t> blocker_lights,
                                                   const transform_t transforms[]) {
    const simd_fvec<S> rdist = abs(r.dist);
    const simd_ivec<S> env_ray = simd_cast(r.dist < 0.0f);
    simd_fvec<S> ret = 1.0f;

    for (uint32_t li = 0; li < uint32_t(blocker_lights.size()) && ray_mask.not_all_zeros(); ++li) {
        const uint32_t light_index = blocker_lights[li];
        const light_t &l = lights[light_index];
        const simd_ivec<S> portal_mask = l.sky_portal ? env_ray : -1;
        if (l.type == LIGHT_TYPE_RECT) {
            float light_fwd[3];
            cross(l.rect.u, l.rect.v, light_fwd);
            normalize(light_fwd);

            const float plane_dist = dot3(light_fwd, l.rect.pos);

            const simd_fvec<S> cos_theta = dot3(r.d, light_fwd);
            const simd_fvec<S> t = safe_div_neg(plane_dist - dot3(light_fwd, r.o), cos_theta);

            const simd_ivec<S> imask =
                simd_cast((cos_theta < 0.0f) & (t > HIT_EPS) & (t < rdist)) & portal_mask & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot3(l.rect.u, l.rect.u);
                const float dot_v = dot3(l.rect.v, l.rect.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.rect.pos[0], p[1] - l.rect.pos[1], p[2] - l.rect.pos[2]};

                const simd_fvec<S> a1 = dot3(l.rect.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot3(l.rect.v, vi) / dot_v;

                const simd_fvec<S> final_mask =
                    (a1 >= -0.5f & a1 <= 0.5f) & (a2 >= -0.5f & a2 <= 0.5f) & simd_cast(imask);

                ray_mask &= ~simd_cast(final_mask);
                where(final_mask, ret) = 0.0f;
            }
        } else if (l.type == LIGHT_TYPE_DISK) {
            float light_fwd[3];
            cross(l.disk.u, l.disk.v, light_fwd);
            normalize(light_fwd);

            const float plane_dist = dot3(light_fwd, l.disk.pos);

            const simd_fvec<S> cos_theta = dot3(r.d, light_fwd);
            const simd_fvec<S> t = safe_div_neg(plane_dist - dot3(light_fwd, r.o), cos_theta);

            const simd_ivec<S> imask =
                simd_cast((cos_theta < 0.0f) & (t > HIT_EPS) & (t < rdist)) & portal_mask & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot3(l.disk.u, l.disk.u);
                const float dot_v = dot3(l.disk.v, l.disk.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.disk.pos[0], p[1] - l.disk.pos[1], p[2] - l.disk.pos[2]};

                const simd_fvec<S> a1 = dot3(l.disk.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot3(l.disk.v, vi) / dot_v;

                const simd_fvec<S> final_mask = (sqrt(a1 * a1 + a2 * a2) <= 0.5f) & simd_cast(imask);

                ray_mask &= ~simd_cast(final_mask);
                where(final_mask, ret) = 0.0f;
            }
        }
    }

    return ret;
}

template <int S>
void Ray::NS::Evaluate_EnvColor(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const environment_t &env,
                                const Ref::TexStorageRGBA &tex_storage, simd_fvec<S> env_col[4]) {
    const uint32_t env_map = env.env_map;
    const float env_map_rotation = env.env_map_rotation;
    const simd_ivec<S> env_map_mask = (ray.depth & 0x00ffffff) != 0;

    if ((mask & env_map_mask).not_all_zeros()) {
        UNROLLED_FOR(i, 3, { env_col[i] = 1.0f; });
        if (env_map != 0xffffffff) {
            SampleLatlong_RGBE(tex_storage, env_map, ray.d, env_map_rotation, (mask & env_map_mask), env_col);
        }
        if (env.qtree_levels) {
            const auto *qtree_mips = reinterpret_cast<const simd_fvec4 *const *>(env.qtree_mips);

            const simd_fvec<S> light_pdf = Evaluate_EnvQTree(env_map_rotation, qtree_mips, env.qtree_levels, ray.d);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { env_col[i] *= mis_weight; })
        } else if (env.multiple_importance) {
            const simd_fvec<S> light_pdf = 0.5f / PI;
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { env_col[i] *= mis_weight; })
        }
    }
    UNROLLED_FOR(i, 3, { where(env_map_mask, env_col[i]) *= env.env_col[i]; })

    const uint32_t back_map = env.back_map;
    const float back_map_rotation = env.back_map_rotation;
    const simd_ivec<S> back_map_mask = ~env_map_mask;

    if (back_map != 0xffffffff && (mask & back_map_mask).not_all_zeros()) {
        simd_fvec<S> back_col[3];
        SampleLatlong_RGBE(tex_storage, back_map, ray.d, back_map_rotation, (mask & back_map_mask), back_col);
        UNROLLED_FOR(i, 3, { where(back_map_mask, env_col[i]) = back_col[i]; })
    }
    UNROLLED_FOR(i, 3, { where(back_map_mask, env_col[i]) *= env.back_col[i]; })
}

template <int S>
void Ray::NS::Evaluate_LightColor(const simd_fvec<S> P[3], const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                  const hit_data_t<S> &inter, const environment_t &env, const light_t *lights,
                                  const Ref::TexStorageRGBA &tex_storage, simd_fvec<S> light_col[3]) {
    simd_ivec<S> ray_queue[S];
    ray_queue[0] = mask;

    int index = 0, num = 1;
    while (index != num) {
        const int mask = ray_queue[index].movemask();
        const uint32_t first_li = inter.obj_index[GetFirstBit(mask)];

        const simd_ivec<S> same_li = (inter.obj_index == first_li);
        const simd_ivec<S> diff_li = and_not(same_li, ray_queue[index]);

        if (diff_li.not_all_zeros()) {
            ray_queue[index] &= same_li;
            ray_queue[num++] = diff_li;
        }

        const light_t &l = lights[-int(first_li) - 1];

        simd_fvec<S> lcol[3] = {l.col[0], l.col[1], l.col[2]};
        if (l.sky_portal) {
            UNROLLED_FOR(i, 3, { lcol[i] *= env.env_col[i]; })
            if (env.env_map != 0xffffffff) {
                simd_fvec<S> tex_col[3];
                SampleLatlong_RGBE(tex_storage, env.env_map, ray.d, env.env_map_rotation, ray_queue[index], tex_col);
                UNROLLED_FOR(i, 3, { lcol[i] *= tex_col[i]; })
            }
        }

        if (l.type == LIGHT_TYPE_SPHERE) {
            simd_fvec<S> dd[3] = {l.sph.pos[0] - P[0], l.sph.pos[1] - P[1], l.sph.pos[2] - P[2]};
            normalize(dd);

            const simd_fvec<S> cos_theta = dot3(ray.d, dd);

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, 0.5f * l.sph.area * cos_theta);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; });

            if (l.sph.spot > 0.0f && l.sph.blend > 0.0f) {
                const simd_fvec<S> _dot =
                    -(ray.d[0] * l.sph.dir[0] + ray.d[1] * l.sph.dir[1] + ray.d[2] * l.sph.dir[2]);
                assert((ray_queue[index] & simd_cast(_dot <= 0.0f)).all_zeros());

                simd_fvec<S> _angle = 0.0f;
                UNROLLED_FOR_S(i, S, {
                    if (ray_queue[index].template get<i>()) {
                        _angle.template set<i>(std::acos(_dot.template get<i>()));
                    }
                })
                assert((ray_queue[index] & simd_cast(_angle > l.sph.spot)).all_zeros());
                if (l.sph.blend > 0.0f) {
                    const simd_fvec<S> spot_weight = clamp((l.sph.spot - _angle) / l.sph.blend, 0.0f, 1.0f);
                    UNROLLED_FOR(i, 3, { lcol[i] *= spot_weight; })
                }
            }
        } else if (l.type == LIGHT_TYPE_RECT) {
            float light_fwd[3];
            cross(l.rect.u, l.rect.v, light_fwd);
            normalize(light_fwd);

            const simd_fvec<S> cos_theta = dot3(ray.d, light_fwd);

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, l.rect.area * cos_theta);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; });
        } else if (l.type == LIGHT_TYPE_DISK) {
            float light_fwd[3];
            cross(l.disk.u, l.disk.v, light_fwd);
            normalize(light_fwd);

            const simd_fvec<S> cos_theta = dot3(ray.d, light_fwd);

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, l.disk.area * cos_theta);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; });
        } else if (l.type == LIGHT_TYPE_LINE) {
            const float *light_dir = l.line.v;

            const simd_fvec<S> cos_theta = 1.0f - abs(dot3(ray.d, light_dir));

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, l.line.area * cos_theta);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; });
        }

        where(ray_queue[index], light_col[0]) = lcol[0];
        where(ray_queue[index], light_col[1]) = lcol[1];
        where(ray_queue[index], light_col[2]) = lcol[2];

        ++index;
    }
};

template <int S>
Ray::NS::simd_ivec<S>
Ray::NS::Evaluate_DiffuseNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                              const surface_t<S> &surf, const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                              const simd_fvec<S> &mix_weight, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};

    simd_fvec<S> diff_col[4];
    Evaluate_OrenDiffuse_BSDF(nI, surf.N, ls.L, roughness, base_color, diff_col);
    const simd_fvec<S> &bsdf_pdf = diff_col[3];

    simd_fvec<S> mis_weight = 1.0f;
    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    UNROLLED_FOR(i, 3, { where(mask, sh_r.o[i]) = P_biased[i]; })
    UNROLLED_FOR(i, 3, {
        const simd_fvec<S> temp = ls.col[i] * diff_col[i] * safe_div(mix_weight * mis_weight, ls.pdf);
        where(mask, sh_r.c[i]) = ray.c[i] * temp;
        where(mask & ~ls.cast_shadow, out_col[i]) += temp;
    })

    return (mask & ls.cast_shadow);
}

template <int S>
void Ray::NS::Sample_DiffuseNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                                 const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                                 const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, const simd_fvec<S> &mix_weight,
                                 ray_data_t<S> &new_ray) {
    simd_fvec<S> V[3], F[4];
    Sample_OrenDiffuse_BSDF(surf.T, surf.B, surf.N, ray.d, roughness, base_color, rand_u, rand_v, V, F);

    where(mask, new_ray.depth) = ray.depth + 0x00000001;

    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    UNROLLED_FOR(i, 3, {
        where(mask, new_ray.o[i]) = P_biased[i];
        where(mask, new_ray.d[i]) = V[i];
        where(mask, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3];
    })
    where(mask, new_ray.pdf) = F[3];
}

template <int S>
Ray::NS::simd_ivec<S>
Ray::NS::Evaluate_GlossyNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                             const surface_t<S> &surf, const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                             const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0, const simd_fvec<S> &mix_weight,
                             simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};
    simd_fvec<S> H[3] = {ls.L[0] - ray.d[0], ls.L[1] - ray.d[1], ls.L[2] - ray.d[2]};
    safe_normalize(H);

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    simd_fvec<S> spec_col[4];
    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, sqr(roughness), sqr(roughness),
                              simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0}, base_color, spec_col);
    const simd_fvec<S> &bsdf_pdf = spec_col[3];

    simd_fvec<S> mis_weight = 1.0f;
    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    UNROLLED_FOR(i, 3, { where(mask, sh_r.o[i]) = P_biased[i]; })
    UNROLLED_FOR(i, 3, {
        const simd_fvec<S> temp = ls.col[i] * spec_col[i] * safe_div_pos(mix_weight * mis_weight, ls.pdf);
        where(mask, sh_r.c[i]) = ray.c[i] * temp;
        where(mask & ~ls.cast_shadow, out_col[i]) += temp;
    })

    return (mask & ls.cast_shadow);
}

template <int S>
void Ray::NS::Sample_GlossyNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                                const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                                const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0, const simd_fvec<S> &rand_u,
                                const simd_fvec<S> &rand_v, const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray) {
    simd_fvec<S> V[3], F[4];
    Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, roughness, simd_fvec<S>{0.0f}, spec_ior, spec_F0, base_color,
                            rand_u, rand_v, V, F);

    where(mask, new_ray.depth) = ray.depth + 0x00000100;

    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    UNROLLED_FOR(i, 3, {
        where(mask, new_ray.o[i]) = P_biased[i];
        where(mask, new_ray.d[i]) = V[i];
        where(mask, new_ray.c[i]) = ray.c[i] * F[i] * safe_div_pos(mix_weight, F[3]);
    })
    where(mask, new_ray.pdf) = F[3];
}

template <int S>
Ray::NS::simd_ivec<S> Ray::NS::Evaluate_RefractiveNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray,
                                                       const simd_ivec<S> &mask, const surface_t<S> &surf,
                                                       const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness2,
                                                       const simd_fvec<S> &eta, const simd_fvec<S> &mix_weight,
                                                       simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};
    simd_fvec<S> H[3] = {ls.L[0] - ray.d[0] * eta, ls.L[1] - ray.d[1] * eta, ls.L[2] - ray.d[2] * eta};
    safe_normalize(H);

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    simd_fvec<S> refr_col[4];
    Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, simd_fvec<S>{eta}, base_color,
                                refr_col);
    const simd_fvec<S> &bsdf_pdf = refr_col[3];

    simd_fvec<S> mis_weight = 1.0f;
    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

    simd_fvec<S> P_biased[3];
    const simd_fvec<S> _plane_N[3] = {-surf.plane_N[0], -surf.plane_N[1], -surf.plane_N[2]};
    offset_ray(surf.P, _plane_N, P_biased);

    UNROLLED_FOR(i, 3, { where(mask, sh_r.o[i]) = P_biased[i]; })
    UNROLLED_FOR(i, 3, {
        const simd_fvec<S> temp = ls.col[i] * refr_col[i] * safe_div_pos(mix_weight * mis_weight, ls.pdf);
        where(mask, sh_r.c[i]) = ray.c[i] * temp;
        where(mask & ~ls.cast_shadow, out_col[i]) += temp;
    })

    return (mask & ls.cast_shadow);
}

template <int S>
void Ray::NS::Sample_RefractiveNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                                    const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                                    const simd_ivec<S> &is_backfacing, const simd_fvec<S> &int_ior,
                                    const simd_fvec<S> &ext_ior, const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                                    const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray) {
    simd_fvec<S> eta = (ext_ior / int_ior);
    where(is_backfacing, eta) = (int_ior / ext_ior);

    simd_fvec<S> V[4], F[4];
    Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, ray.d, roughness, eta, base_color, rand_u, rand_v, V, F);

    where(mask, new_ray.depth) = ray.depth + 0x00010000;

    simd_fvec<S> P_biased[3];
    const simd_fvec<S> _plane_N[3] = {-surf.plane_N[0], -surf.plane_N[1], -surf.plane_N[2]};
    offset_ray(surf.P, _plane_N, P_biased);

    UNROLLED_FOR(i, 3, {
        where(mask, new_ray.o[i]) = P_biased[i];
        where(mask, new_ray.d[i]) = V[i];
        where(mask, new_ray.c[i]) = ray.c[i] * F[i] * safe_div_pos(mix_weight, F[3]);
    })

    pop_ior_stack(is_backfacing & mask, new_ray.ior);
    push_ior_stack(~is_backfacing & mask, new_ray.ior, int_ior);

    where(mask, new_ray.pdf) = F[3];
}

template <int S>
Ray::NS::simd_ivec<S> Ray::NS::Evaluate_PrincipledNode(
    const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
    const lobe_weights_t<S> &lobe_weights, const diff_params_t<S> &diff, const spec_params_t<S> &spec,
    const clearcoat_params_t<S> &coat, const transmission_params_t<S> &trans, const simd_fvec<S> &metallic,
    const simd_fvec<S> &N_dot_L, const simd_fvec<S> &mix_weight, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};

    const simd_ivec<S> _is_backfacing = simd_cast(N_dot_L < 0.0f);
    const simd_ivec<S> _is_frontfacing = simd_cast(N_dot_L > 0.0f);

    simd_fvec<S> lcol[3] = {0.0f, 0.0f, 0.0f};
    simd_fvec<S> bsdf_pdf = 0.0f;

    const simd_ivec<S> eval_diff_lobe = simd_cast(lobe_weights.diffuse > 0.0f) & _is_frontfacing & mask;
    if (eval_diff_lobe.not_all_zeros()) {
        simd_fvec<S> diff_col[4];
        Evaluate_PrincipledDiffuse_BSDF(nI, surf.N, ls.L, diff.roughness, diff.base_color, diff.sheen_color, false,
                                        diff_col);

        where(eval_diff_lobe, bsdf_pdf) += lobe_weights.diffuse * diff_col[3];
        UNROLLED_FOR(i, 3, {
            diff_col[i] *= (1.0f - metallic);
            where(eval_diff_lobe, lcol[i]) += safe_div_pos(ls.col[i] * N_dot_L * diff_col[i], PI * ls.pdf);
        })
    }

    simd_fvec<S> H[3];
    UNROLLED_FOR(i, 3, {
        H[i] = ls.L[i] - ray.d[i] * trans.eta;
        where(_is_frontfacing, H[i]) = ls.L[i] - ray.d[i];
    })
    safe_normalize(H);

    const simd_fvec<S> roughness2 = sqr(spec.roughness);
    const simd_fvec<S> aspect = sqrt(1.0f - 0.9f * spec.anisotropy);

    const simd_fvec<S> alpha_x = roughness2 / aspect;
    const simd_fvec<S> alpha_y = roughness2 * aspect;

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    const simd_ivec<S> eval_spec_lobe =
        simd_cast(lobe_weights.specular > 0.0f) & simd_cast(alpha_x * alpha_y >= 1e-7f) & _is_frontfacing & mask;
    if (eval_spec_lobe.not_all_zeros()) {
        simd_fvec<S> spec_col[4];
        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, max(alpha_x, 1e-7f),
                                  max(alpha_y, 1e-7f), spec.ior, spec.F0, spec.tmp_col, spec_col);

        where(eval_spec_lobe, bsdf_pdf) += lobe_weights.specular * spec_col[3];

        UNROLLED_FOR(i, 3, { where(eval_spec_lobe, lcol[i]) += safe_div_pos(ls.col[i] * spec_col[i], ls.pdf); })
    }

    const simd_fvec<S> clearcoat_roughness2 = sqr(coat.roughness);

    const simd_ivec<S> eval_coat_lobe = simd_cast(lobe_weights.clearcoat > 0.0f) &
                                        simd_cast(sqr(clearcoat_roughness2) >= 1e-7f) & _is_frontfacing & mask;
    if (eval_coat_lobe.not_all_zeros()) {
        simd_fvec<S> clearcoat_col[4];
        Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, clearcoat_roughness2, coat.ior,
                                          coat.F0, clearcoat_col);

        where(eval_coat_lobe, bsdf_pdf) += lobe_weights.clearcoat * clearcoat_col[3];

        UNROLLED_FOR(i, 3,
                     { where(eval_coat_lobe, lcol[i]) += safe_div_pos(0.25f * ls.col[i] * clearcoat_col[i], ls.pdf); })
    }

    const simd_ivec<S> eval_refr_spec_lobe = simd_cast(trans.fresnel != 0.0f) &
                                             simd_cast(lobe_weights.refraction > 0.0f) &
                                             simd_cast(sqr(roughness2) >= 1e-7f) & _is_frontfacing & mask;
    if (eval_refr_spec_lobe.not_all_zeros()) {
        simd_fvec<S> spec_col[4], spec_temp_col[3] = {1.0f, 1.0f, 1.0f};
        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2,
                                  simd_fvec<S>{1.0f} /* ior */, simd_fvec<S>{0.0f} /* F0 */, spec_temp_col, spec_col);
        where(eval_refr_spec_lobe, bsdf_pdf) += lobe_weights.refraction * trans.fresnel * spec_col[3];

        UNROLLED_FOR(i, 3, {
            where(eval_refr_spec_lobe, lcol[i]) += ls.col[i] * spec_col[i] * safe_div_pos(trans.fresnel, ls.pdf);
        })
    }

    const simd_fvec<S> transmission_roughness2 = sqr(trans.roughness);

    const simd_ivec<S> eval_refr_trans_lobe = simd_cast(trans.fresnel != 1.0f) &
                                              simd_cast(lobe_weights.refraction > 0.0f) &
                                              simd_cast(sqr(transmission_roughness2) >= 1e-7f) & _is_backfacing & mask;
    if (eval_refr_trans_lobe.not_all_zeros()) {
        simd_fvec<S> refr_col[4];
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, transmission_roughness2, trans.eta,
                                    diff.base_color, refr_col);
        where(eval_refr_trans_lobe, bsdf_pdf) += lobe_weights.refraction * (1.0f - trans.fresnel) * refr_col[3];

        UNROLLED_FOR(i, 3, {
            where(eval_refr_trans_lobe, lcol[i]) +=
                ls.col[i] * refr_col[i] * safe_div_pos(1.0f - trans.fresnel, ls.pdf);
        })
    }

    simd_fvec<S> mis_weight = 1.0f;
    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);
    UNROLLED_FOR(i, 3, { where(mask, lcol[i]) *= mix_weight * mis_weight; })

    ///
    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    const simd_fvec<S> neg_plane_N[3] = {-surf.plane_N[0], -surf.plane_N[1], -surf.plane_N[2]};
    simd_fvec<S> back_P_biased[3];
    offset_ray(surf.P, neg_plane_N, back_P_biased);

    UNROLLED_FOR(i, 3, {
        where(N_dot_L < 0.0f, P_biased[i]) = back_P_biased[i];
        where(mask, sh_r.o[i]) = P_biased[i];
    })
    UNROLLED_FOR(i, 3, {
        where(mask, sh_r.c[i]) = ray.c[i] * lcol[i];
        where(mask & ~ls.cast_shadow, out_col[i]) += lcol[i];
    })

    return (mask & ls.cast_shadow);
}

template <int S>
void Ray::NS::Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                    const surface_t<S> &surf, const lobe_weights_t<S> &lobe_weights,
                                    const diff_params_t<S> &diff, const spec_params_t<S> &spec,
                                    const clearcoat_params_t<S> &coat, const transmission_params_t<S> &trans,
                                    const simd_fvec<S> &metallic, const simd_fvec<S> &rand_u,
                                    const simd_fvec<S> &rand_v, simd_fvec<S> mix_rand, const simd_fvec<S> &mix_weight,
                                    simd_ivec<S> &secondary_mask, ray_data_t<S> &new_ray) {
    const simd_ivec<S> diff_depth = ray.depth & 0x000000ff;
    const simd_ivec<S> spec_depth = (ray.depth >> 8) & 0x000000ff;
    const simd_ivec<S> refr_depth = (ray.depth >> 16) & 0x000000ff;
    // NOTE: transparency depth is not accounted here
    const simd_ivec<S> total_depth = diff_depth + spec_depth + refr_depth;

    const simd_ivec<S> sample_diff_lobe = (diff_depth < ps.max_diff_depth) & (total_depth < ps.max_total_depth) &
                                          simd_cast(mix_rand < lobe_weights.diffuse) & mask;
    if (sample_diff_lobe.not_all_zeros()) {
        simd_fvec<S> V[3], diff_col[4];
        Sample_PrincipledDiffuse_BSDF(surf.T, surf.B, surf.N, ray.d, diff.roughness, diff.base_color, diff.sheen_color,
                                      false, rand_u, rand_v, V, diff_col);

        UNROLLED_FOR(i, 3, { diff_col[i] *= (1.0f - metallic); })

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_diff_lobe, new_ray.depth) = ray.depth + 0x00000001;

        UNROLLED_FOR(i, 3, {
            where(sample_diff_lobe, new_ray.o[i]) = new_p[i];
            where(sample_diff_lobe, new_ray.d[i]) = V[i];
            where(sample_diff_lobe, new_ray.c[i]) =
                safe_div_pos(ray.c[i] * diff_col[i] * mix_weight, lobe_weights.diffuse);
        })
        where(sample_diff_lobe, new_ray.pdf) = diff_col[3];

        assert((secondary_mask & sample_diff_lobe).all_zeros());
        secondary_mask |= sample_diff_lobe;
    }

    const simd_ivec<S> sample_spec_lobe = (spec_depth < ps.max_spec_depth) & (total_depth < ps.max_total_depth) &
                                          simd_cast(mix_rand >= lobe_weights.diffuse) &
                                          simd_cast(mix_rand < lobe_weights.diffuse + lobe_weights.specular) & mask;
    if (sample_spec_lobe.not_all_zeros()) {
        simd_fvec<S> V[3], F[4];
        Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, spec.roughness, spec.anisotropy, spec.ior, spec.F0,
                                spec.tmp_col, rand_u, rand_v, V, F);
        F[3] *= lobe_weights.specular;

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_spec_lobe, new_ray.depth) = ray.depth + 0x00000100;

        UNROLLED_FOR(i, 3, {
            where(sample_spec_lobe, new_ray.o[i]) = new_p[i];
            where(sample_spec_lobe, new_ray.d[i]) = V[i];
            where(sample_spec_lobe, new_ray.c[i]) = safe_div_pos(ray.c[i] * F[i] * mix_weight, F[3]);
        })
        where(sample_spec_lobe, new_ray.pdf) = F[3];

        assert(((sample_spec_lobe & mask) != sample_spec_lobe).all_zeros());
        assert((secondary_mask & sample_spec_lobe).all_zeros());
        secondary_mask |= sample_spec_lobe;
    }

    const simd_ivec<S> sample_coat_lobe =
        (spec_depth < ps.max_spec_depth) & (total_depth < ps.max_total_depth) &
        simd_cast(mix_rand >= lobe_weights.diffuse + lobe_weights.specular) &
        simd_cast(mix_rand < lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat) & mask;
    if (sample_coat_lobe.not_all_zeros()) {
        simd_fvec<S> V[3], F[4];
        Sample_PrincipledClearcoat_BSDF(surf.T, surf.B, surf.N, ray.d, sqr(coat.roughness), coat.ior, coat.F0, rand_u,
                                        rand_v, V, F);
        F[3] *= lobe_weights.clearcoat;

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_coat_lobe, new_ray.depth) = ray.depth + 0x00000100;

        UNROLLED_FOR(i, 3, {
            where(sample_coat_lobe, new_ray.o[i]) = new_p[i];
            where(sample_coat_lobe, new_ray.d[i]) = V[i];
            where(sample_coat_lobe, new_ray.c[i]) = 0.25f * ray.c[i] * F[i] * safe_div_pos(mix_weight, F[3]);
        })
        where(sample_coat_lobe, new_ray.pdf) = F[3];

        assert((secondary_mask & sample_coat_lobe).all_zeros());
        secondary_mask |= sample_coat_lobe;
    }

    const simd_ivec<S> sample_trans_lobe =
        simd_cast(mix_rand >= lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat) &
        ((simd_cast(mix_rand >= trans.fresnel) & (refr_depth < ps.max_refr_depth)) |
         (simd_cast(mix_rand < trans.fresnel) & (spec_depth < ps.max_spec_depth))) &
        (total_depth < ps.max_total_depth) & mask;
    if (sample_trans_lobe.not_all_zeros()) {
        where(sample_trans_lobe, mix_rand) -= lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat;
        where(sample_trans_lobe, mix_rand) = safe_div_pos(mix_rand, lobe_weights.refraction);

        simd_fvec<S> F[4] = {}, V[3] = {};

        const simd_ivec<S> sample_trans_spec_lobe = simd_cast(mix_rand < trans.fresnel) & sample_trans_lobe;
        if (sample_trans_spec_lobe.not_all_zeros()) {
            const simd_fvec<S> _spec_tmp_col[3] = {{1.0f}, {1.0f}, {1.0f}};
            Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, spec.roughness, simd_fvec<S>{0.0f} /* anisotropic */,
                                    simd_fvec<S>{1.0f} /* ior */, simd_fvec<S>{0.0f} /* F0 */, _spec_tmp_col, rand_u,
                                    rand_v, V, F);

            simd_fvec<S> new_p[3];
            offset_ray(surf.P, surf.plane_N, new_p);

            where(sample_trans_spec_lobe, new_ray.depth) = ray.depth + 0x00000100;

            UNROLLED_FOR(i, 3, { where(sample_trans_spec_lobe, new_ray.o[i]) = new_p[i]; })
        }

        const simd_ivec<S> sample_trans_refr_lobe = ~sample_trans_spec_lobe & sample_trans_lobe;
        if (sample_trans_refr_lobe.not_all_zeros()) {
            simd_fvec<S> temp_F[4], temp_V[4];
            Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, ray.d, trans.roughness, trans.eta, diff.base_color,
                                      rand_u, rand_v, temp_V, temp_F);

            const simd_fvec<S> _plane_N[3] = {-surf.plane_N[0], -surf.plane_N[1], -surf.plane_N[2]};
            simd_fvec<S> new_p[3];
            offset_ray(surf.P, _plane_N, new_p);

            where(sample_trans_refr_lobe, new_ray.depth) = ray.depth + 0x00010000;

            UNROLLED_FOR(i, 4, { where(sample_trans_refr_lobe, F[i]) = temp_F[i]; })
            UNROLLED_FOR(i, 3, {
                where(sample_trans_refr_lobe, V[i]) = temp_V[i];
                where(sample_trans_refr_lobe, new_ray.o[i]) = new_p[i];
            })

            pop_ior_stack(trans.backfacing & sample_trans_refr_lobe, new_ray.ior);
            push_ior_stack(~trans.backfacing & sample_trans_refr_lobe, new_ray.ior, trans.int_ior);
        }

        F[3] *= lobe_weights.refraction;

        UNROLLED_FOR(i, 3, {
            where(sample_trans_lobe, new_ray.d[i]) = V[i];
            where(sample_trans_lobe, new_ray.c[i]) = safe_div_pos(ray.c[i] * F[i] * mix_weight, F[3]);
        })
        where(sample_trans_lobe, new_ray.pdf) = F[3];

        assert((sample_trans_spec_lobe & sample_trans_refr_lobe).all_zeros());
        assert((secondary_mask & sample_trans_lobe).all_zeros());
        secondary_mask |= sample_trans_lobe;
    }
}

template <int S>
void Ray::NS::ShadeSurface(const pass_settings_t &ps, const float *random_seq, const hit_data_t<S> &inter,
                           const ray_data_t<S> &ray, const scene_data_t &sc, const uint32_t node_index,
                           const Ref::TexStorageBase *const textures[], simd_fvec<S> out_rgba[4],
                           simd_ivec<S> out_secondary_masks[], ray_data_t<S> out_secondary_rays[],
                           int *out_secondary_rays_count, simd_ivec<S> out_shadow_masks[],
                           shadow_ray_t<S> out_shadow_rays[], int *out_shadow_rays_count) {
    out_rgba[0] = out_rgba[1] = out_rgba[2] = {0.0f};
    out_rgba[3] = {1.0f};

    const simd_ivec<S> ino_hit = ~inter.mask;
    if (ino_hit.not_all_zeros()) {
        simd_fvec<S> env_col[4] = {{1.0f}, {1.0f}, {1.0f}, {1.0f}};
        Evaluate_EnvColor(ray, ino_hit, sc.env, *static_cast<const Ref::TexStorageRGBA *>(textures[0]), env_col);

        where(ino_hit, out_rgba[0]) = ray.c[0] * env_col[0];
        where(ino_hit, out_rgba[1]) = ray.c[1] * env_col[1];
        where(ino_hit, out_rgba[2]) = ray.c[2] * env_col[2];
        where(ino_hit, out_rgba[3]) = env_col[3];
    }

    simd_ivec<S> is_active_lane = inter.mask;
    if (is_active_lane.all_zeros()) {
        return;
    }

    const simd_fvec<S> *I = ray.d;

    surface_t<S> surf;
    UNROLLED_FOR(i, 3, { where(inter.mask, surf.P[i]) = fmadd(inter.t, ray.d[i], ray.o[i]); })

    const simd_ivec<S> is_light_hit = is_active_lane & (inter.obj_index < 0); // Area light intersection
    if (is_light_hit.not_all_zeros()) {
        simd_fvec<S> light_col[3] = {};
        Evaluate_LightColor(surf.P, ray, is_light_hit, inter, sc.env, sc.lights,
                            *static_cast<const Ref::TexStorageRGBA *>(textures[0]), light_col);

        UNROLLED_FOR(i, 3, { where(is_light_hit, out_rgba[i]) = ray.c[i] * light_col[i]; })
        where(is_light_hit, out_rgba[3]) = 1.0f;

        is_active_lane &= ~is_light_hit;
    }

    if (is_active_lane.all_zeros()) {
        return;
    }

    simd_ivec<S> tri_index = inter.prim_index;
    const simd_ivec<S> is_backfacing = (tri_index < 0);
    where(is_backfacing, tri_index) = -tri_index - 1;

    simd_ivec<S> obj_index = inter.obj_index;
    where(~is_active_lane, obj_index) = 0;

    simd_ivec<S> mat_index = gather(reinterpret_cast<const int *>(sc.tri_materials), tri_index) &
                             simd_ivec<S>((MATERIAL_INDEX_BITS << 16) | MATERIAL_INDEX_BITS);

    const int *tr_indices = reinterpret_cast<const int *>(&sc.mesh_instances[0].tr_index);
    const int TrIndicesStride = sizeof(mesh_instance_t) / sizeof(int);

    const simd_ivec<S> tr_index = gather(tr_indices, obj_index * TrIndicesStride);

    const simd_ivec<S> vtx_indices[3] = {gather(reinterpret_cast<const int *>(sc.vtx_indices + 0), tri_index * 3),
                                         gather(reinterpret_cast<const int *>(sc.vtx_indices + 1), tri_index * 3),
                                         gather(reinterpret_cast<const int *>(sc.vtx_indices + 2), tri_index * 3)};

    const simd_fvec<S> w = 1.0f - inter.u - inter.v;

    simd_fvec<S> p1[3], p2[3], p3[3], P_ls[3];
    { // Fetch vertex positions
        const float *vtx_positions = &sc.vertices[0].p[0];
        const int VtxPositionsStride = sizeof(vertex_t) / sizeof(float);

        UNROLLED_FOR(i, 3, {
            p1[i] = gather(vtx_positions + i, vtx_indices[0] * VtxPositionsStride);
            p2[i] = gather(vtx_positions + i, vtx_indices[1] * VtxPositionsStride);
            p3[i] = gather(vtx_positions + i, vtx_indices[2] * VtxPositionsStride);

            P_ls[i] = p1[i] * w + p2[i] * inter.u + p3[i] * inter.v;
        });
    }

    FetchVertexAttribute3(&sc.vertices[0].n[0], vtx_indices, inter.u, inter.v, w, surf.N);
    normalize(surf.N);

    simd_fvec<S> u1[2], u2[2], u3[2];
    { // Fetch vertex uvs
        const float *vtx_uvs = &sc.vertices[0].t[0][0];
        const int VtxUVStride = sizeof(vertex_t) / sizeof(float);

        UNROLLED_FOR(i, 2, {
            u1[i] = gather(vtx_uvs + i, vtx_indices[0] * VtxUVStride);
            u2[i] = gather(vtx_uvs + i, vtx_indices[1] * VtxUVStride);
            u3[i] = gather(vtx_uvs + i, vtx_indices[2] * VtxUVStride);

            surf.uvs[i] = u1[i] * w + u2[i] * inter.u + u3[i] * inter.v;
        })
    }

    { // calc planar normal
        simd_fvec<S> e21[3], e31[3];
        UNROLLED_FOR(i, 3, {
            e21[i] = p2[i] - p1[i];
            e31[i] = p3[i] - p1[i];
        })
        cross(e21, e31, surf.plane_N);
    }
    const simd_fvec<S> pa = length(surf.plane_N);
    UNROLLED_FOR(i, 3, { surf.plane_N[i] /= pa; })

    FetchVertexAttribute3(&sc.vertices[0].b[0], vtx_indices, inter.u, inter.v, w, surf.B);
    cross(surf.B, surf.N, surf.T);

    { // return black for non-existing backfacing material
        simd_ivec<S> no_back_mi = (mat_index >> 16) == 0xffff;
        no_back_mi &= is_backfacing & is_active_lane;
        UNROLLED_FOR(i, 4, { where(no_back_mi, out_rgba[i]) = 0.0f; })
        is_active_lane &= ~no_back_mi;
    }

    if (is_active_lane.all_zeros()) {
        return;
    }

    where(~is_backfacing, mat_index) = mat_index & 0xffff; // use front material index
    where(is_backfacing, mat_index) = mat_index >> 16;     // use back material index

    UNROLLED_FOR(i, 3, {
        where(is_backfacing, surf.plane_N[i]) = -surf.plane_N[i];
        where(is_backfacing, surf.N[i]) = -surf.N[i];
        where(is_backfacing, surf.B[i]) = -surf.B[i];
        where(is_backfacing, surf.T[i]) = -surf.T[i];
    });

    simd_fvec<S> tangent[3] = {-P_ls[2], {0.0f}, P_ls[0]};

    simd_fvec<S> transform[16];
    FetchTransformAndRecalcBasis(sc.transforms, tr_index, P_ls, surf.plane_N, surf.N, surf.B, surf.T, tangent,
                                 transform);

    // normalize vectors (scaling might have been applied)
    safe_normalize(surf.plane_N);
    safe_normalize(surf.N);
    safe_normalize(surf.B);
    safe_normalize(surf.T);

    //////////////////////////////////

    const simd_fvec<S> ta = abs((u2[0] - u1[0]) * (u3[1] - u1[1]) - (u3[0] - u1[0]) * (u2[1] - u1[1]));

    const simd_fvec<S> cone_width = ray.cone_width + ray.cone_spread * inter.t;

    simd_fvec<S> lambda = 0.5f * fast_log2(ta / pa);
    lambda += fast_log2(cone_width);
    // lambda += 0.5 * fast_log2(tex_res.x * tex_res.y);
    // lambda -= fast_log2(abs(dot3(I, surf.plane_N)));

    //////////////////////////////////

    static const int MatDWORDStride = sizeof(material_t) / sizeof(float);

    // used to randomize random sequence among pixels
    const simd_fvec<S> sample_off[2] = {construct_float(hash(ray.xy)), construct_float(hash(hash(ray.xy)))};
    const simd_fvec<S> ext_ior = peek_ior_stack(ray.ior, is_backfacing);

    const simd_ivec<S> diff_depth = ray.depth & 0x000000ff;
    const simd_ivec<S> spec_depth = (ray.depth >> 8) & 0x000000ff;
    const simd_ivec<S> refr_depth = (ray.depth >> 16) & 0x000000ff;
    const simd_ivec<S> transp_depth = (ray.depth >> 24) & 0x000000ff;
    // NOTE: transparency depth is not accounted here
    const simd_ivec<S> total_depth = diff_depth + spec_depth + refr_depth;

    // offset of the sequence
    const simd_ivec<S> rand_index = (total_depth + transp_depth) * RAND_DIM_BOUNCE_COUNT;

    const simd_ivec<S> mat_type =
        gather(reinterpret_cast<const int *>(&sc.materials[0].type), mat_index * sizeof(material_t) / sizeof(int)) &
        0xff;

    simd_fvec<S> mix_rand = fract(gather(random_seq + RAND_DIM_BSDF_PICK, rand_index) + sample_off[0]);
    simd_fvec<S> mix_weight = 1.0f;

    // resolve mix material
    const simd_ivec<S> is_mix_mat = mat_type == MixNode;
    if (is_mix_mat.not_all_zeros()) {
        const float *mix_values = &sc.materials[0].strength;
        simd_fvec<S> mix_val = gather(mix_values, mat_index * MatDWORDStride);

        const int *base_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[BASE_TEXTURE]);
        const simd_ivec<S> base_texture = gather(base_textures, mat_index * MatDWORDStride);

        const simd_ivec<S> has_texture = (base_texture != -1) & is_active_lane;
        if (has_texture.not_all_zeros()) {
            simd_ivec<S> ray_queue[S];
            ray_queue[0] = has_texture;

            int index = 0, num = 1;
            while (index != num) {
                const long mask = ray_queue[index].movemask();
                const uint32_t first_t = base_texture[GetFirstBit(mask)];

                const simd_ivec<S> same_t = (base_texture == first_t);
                const simd_ivec<S> diff_t = and_not(same_t, ray_queue[index]);

                if (diff_t.not_all_zeros()) {
                    ray_queue[index] &= same_t;
                    ray_queue[num++] = diff_t;
                }

                const simd_fvec<S> base_lod = get_texture_lod(textures, first_t, lambda, ray_queue[index]);

                simd_fvec<S> tex_color[4] = {};
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(base_lod), ray_queue[index], tex_color);

                where(ray_queue[index], mix_val) *= tex_color[0];

                ++index;
            }
        }

        const float *iors = &sc.materials[0].ior;

        const simd_fvec<S> ior = gather(iors, mat_index * MatDWORDStride);

        simd_fvec<S> eta = safe_div_pos(ext_ior, ior);
        where(is_backfacing, eta) = safe_div_pos(ior, ext_ior);

        simd_fvec<S> RR = fresnel_dielectric_cos(dot3(I, surf.N), eta);
        where(ior == 0.0f, RR) = 1.0f;

        mix_val *= clamp(RR, 0.0f, 1.0f);

        const simd_ivec<S> use_mat1 = simd_cast(mix_rand > mix_val) & is_mix_mat;
        const simd_ivec<S> use_mat2 = ~use_mat1 & is_mix_mat;

        const int *all_mat_flags = reinterpret_cast<const int *>(&sc.materials[0].flags);
        const simd_ivec<S> is_add = (gather(all_mat_flags, mat_index * MatDWORDStride) & MAT_FLAG_MIX_ADD) != 0;

        const int *all_mat_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[0]);
        const simd_ivec<S> mat1_index = gather(&all_mat_textures[MIX_MAT1], mat_index * MatDWORDStride);
        const simd_ivec<S> mat2_index = gather(&all_mat_textures[MIX_MAT2], mat_index * MatDWORDStride);

        where(is_add & use_mat1, mix_weight) = safe_div_pos(mix_weight, 1.0f - mix_val);
        where(use_mat1, mat_index) = mat1_index;
        where(use_mat1, mix_rand) = safe_div_pos(mix_rand - mix_val, 1.0f - mix_val);

        where(is_add & use_mat2, mix_weight) = safe_div_pos(mix_weight, mix_val);
        where(use_mat2, mat_index) = mat2_index;
        where(use_mat2, mix_rand) = safe_div_pos(mix_rand, mix_val);
    }

    { // apply normal map
        const int *norm_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[NORMALS_TEXTURE]);
        const simd_ivec<S> normals_texture = gather(norm_textures, mat_index * MatDWORDStride);

        const simd_ivec<S> has_texture = (normals_texture != -1) & is_active_lane;
        if (has_texture.not_all_zeros()) {
            simd_ivec<S> ray_queue[S];
            ray_queue[0] = has_texture;

            simd_fvec<S> normals_tex[4] = {{0.0f}, {1.0f}, {0.0f}, {0.0f}};
            simd_ivec<S> reconstruct_z = 0;

            int index = 0, num = 1;
            while (index != num) {
                const long mask = ray_queue[index].movemask();
                const uint32_t first_t = normals_texture[GetFirstBit(mask)];

                const simd_ivec<S> same_t = (normals_texture == first_t);
                const simd_ivec<S> diff_t = and_not(same_t, ray_queue[index]);

                if (diff_t.not_all_zeros()) {
                    ray_queue[index] &= same_t;
                    ray_queue[num++] = diff_t;
                }

                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>{0}, ray_queue[index], normals_tex);
                if (first_t & TEX_RECONSTRUCT_Z_BIT) {
                    reconstruct_z |= ray_queue[index];
                }

                ++index;
            }

            UNROLLED_FOR(i, 2, { normals_tex[i] = normals_tex[i] * 2.0f - 1.0f; })
            normals_tex[2] = 1.0f;
            if (reconstruct_z.not_all_zeros()) {
                where(reconstruct_z, normals_tex[2]) =
                    safe_sqrt(1.0f - normals_tex[0] * normals_tex[0] - normals_tex[1] * normals_tex[1]);
            }

            simd_fvec<S> new_normal[3];
            UNROLLED_FOR(i, 3, {
                new_normal[i] = normals_tex[0] * surf.T[i] + normals_tex[2] * surf.N[i] + normals_tex[1] * surf.B[i];
            })
            normalize(new_normal);

            const int *normalmap_strengths = reinterpret_cast<const int *>(&sc.materials[0].normal_map_strength_unorm);
            const simd_ivec<S> normalmap_strength = gather(normalmap_strengths, mat_index * MatDWORDStride) & 0xffff;

            const simd_fvec<S> fstrength = conv_unorm_16(normalmap_strength);
            UNROLLED_FOR(i, 3, { new_normal[i] = surf.N[i] + (new_normal[i] - surf.N[i]) * fstrength; })
            normalize(new_normal);

            const simd_fvec<S> nI[3] = {-I[0], -I[1], -I[2]};
            EnsureValidReflection(surf.plane_N, nI, new_normal);

            UNROLLED_FOR(i, 3, { where(has_texture, surf.N[i]) = new_normal[i]; })
        }
    }

#if 0

#else
    simd_fvec<S> tangent_rotation;
    { // fetch anisotropic rotations
        const float *tangent_rotations = &sc.materials[0].tangent_rotation;
        tangent_rotation = gather(tangent_rotations, mat_index * MatDWORDStride);
    }

    const simd_ivec<S> has_rotation = simd_cast(tangent_rotation != 0.0f);
    if (has_rotation.not_all_zeros()) {
        rotate_around_axis(tangent, surf.N, tangent_rotation, tangent);
    }

    cross(tangent, surf.N, surf.B);
    normalize(surf.B);
    cross(surf.N, surf.B, surf.T);
#endif

#if USE_NEE
    light_sample_t<S> ls;
    if (!sc.li_indices.empty()) {
        SampleLightSource(surf.P, surf.T, surf.B, surf.N, sc, textures, random_seq, rand_index, sample_off,
                          is_active_lane, ls);
    }
    const simd_fvec<S> N_dot_L = dot3(surf.N, ls.L);
#endif

    simd_fvec<S> base_color[3];
    { // Fetch material base color
        const float *base_colors = &sc.materials[0].base_color[0];
        UNROLLED_FOR(i, 3, { base_color[i] = gather(base_colors + i, mat_index * MatDWORDStride); })

        const int *base_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[BASE_TEXTURE]);
        const simd_ivec<S> base_texture = gather(base_textures, mat_index * MatDWORDStride);

        const simd_ivec<S> has_texture = (base_texture != -1) & is_active_lane;
        if (has_texture.not_all_zeros()) {
            simd_ivec<S> ray_queue[S];
            ray_queue[0] = has_texture;

            int index = 0, num = 1;
            while (index != num) {
                const long mask = ray_queue[index].movemask();
                const uint32_t first_t = base_texture[GetFirstBit(mask)];

                const simd_ivec<S> same_t = (base_texture == first_t);
                const simd_ivec<S> diff_t = and_not(same_t, ray_queue[index]);

                if (diff_t.not_all_zeros()) {
                    ray_queue[index] &= same_t;
                    ray_queue[num++] = diff_t;
                }

                const simd_fvec<S> base_lod = get_texture_lod(textures, first_t, lambda, ray_queue[index]);

                simd_fvec<S> tex_color[4] = {};
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(base_lod), ray_queue[index], tex_color);
                if (first_t & TEX_SRGB_BIT) {
                    srgb_to_rgb(tex_color, tex_color);
                }

                UNROLLED_FOR(i, 3, { where(ray_queue[index], base_color[i]) *= tex_color[i]; })

                ++index;
            }
        }
    }

    simd_fvec<S> tint_color[3] = {{0.0f}, {0.0f}, {0.0f}};

    const simd_fvec<S> base_color_lum = lum(base_color);
    UNROLLED_FOR(i, 3, { where(base_color_lum > 0.0f, tint_color[i]) = safe_div_pos(base_color[i], base_color_lum); })

    simd_fvec<S> roughness;
    { // fetch material roughness
        const int *roughnesses = reinterpret_cast<const int *>(&sc.materials[0].roughness_unorm);
        roughness = conv_unorm_16(gather(roughnesses, mat_index * MatDWORDStride) & 0xffff);

        const int *roughness_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[ROUGH_TEXTURE]);
        const simd_ivec<S> roughness_texture = gather(roughness_textures, mat_index * MatDWORDStride);

        const simd_ivec<S> has_texture = (roughness_texture != -1) & is_active_lane;
        if (has_texture.not_all_zeros()) {
            simd_ivec<S> ray_queue[S];
            ray_queue[0] = has_texture;

            int index = 0, num = 1;
            while (index != num) {
                const long mask = ray_queue[index].movemask();
                const uint32_t first_t = roughness_texture[GetFirstBit(mask)];

                const simd_ivec<S> same_t = (roughness_texture == first_t);
                const simd_ivec<S> diff_t = and_not(same_t, ray_queue[index]);

                if (diff_t.not_all_zeros()) {
                    ray_queue[index] &= same_t;
                    ray_queue[num++] = diff_t;
                }

                const simd_fvec<S> roughness_lod = get_texture_lod(textures, first_t, lambda, ray_queue[index]);

                simd_fvec<S> roughness_color[4] = {};
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(roughness_lod), ray_queue[index],
                               roughness_color);
                if (first_t & TEX_SRGB_BIT) {
                    srgb_to_rgb(roughness_color, roughness_color);
                }
                where(ray_queue[index], roughness) *= roughness_color[0];

                ++index;
            }
        }
    }

    simd_fvec<S> col[3] = {0.0f, 0.0f, 0.0f};

    const simd_fvec<S> rand_u = fract(gather(random_seq + RAND_DIM_BSDF_U, rand_index) + sample_off[0]);
    const simd_fvec<S> rand_v = fract(gather(random_seq + RAND_DIM_BSDF_V, rand_index) + sample_off[1]);

    simd_ivec<S> secondary_mask = {0}, shadow_mask = {0};

    ray_data_t<S> &new_ray = out_secondary_rays[*out_secondary_rays_count];
    new_ray.o[0] = new_ray.o[1] = new_ray.o[2] = 0.0f;
    new_ray.d[0] = new_ray.d[1] = new_ray.d[2] = 0.0f;
    new_ray.pdf = 0.0f;
    UNROLLED_FOR(i, 4, { new_ray.ior[i] = ray.ior[i]; })
    new_ray.c[0] = new_ray.c[1] = new_ray.c[2] = 0.0f;
    new_ray.cone_width = cone_width;
    new_ray.cone_spread = ray.cone_spread;
    new_ray.xy = ray.xy;
    new_ray.depth = 0;

    shadow_ray_t<S> &sh_r = out_shadow_rays[*out_shadow_rays_count];
    sh_r = {};
    sh_r.depth = ray.depth;
    sh_r.xy = ray.xy;

    { // Sample materials
        simd_ivec<S> ray_queue[S];
        ray_queue[0] = is_active_lane;

        simd_ivec<S> lanes_processed = 0;

        int index = 0, num = 1;
        while (index != num) {
            const long mask = ray_queue[index].movemask();
            const uint32_t first_mi = mat_index[GetFirstBit(mask)];

            const simd_ivec<S> same_mi = (mat_index == first_mi);
            const simd_ivec<S> diff_mi = and_not(same_mi, ray_queue[index]);

            if (diff_mi.not_all_zeros()) {
                ray_queue[index] &= same_mi;
                ray_queue[num++] = diff_mi;
            }

            assert((lanes_processed & ray_queue[index]).all_zeros());
            lanes_processed |= ray_queue[index];

            const material_t *mat = &sc.materials[first_mi];
            if (mat->type == DiffuseNode) {
#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |=
                        Evaluate_DiffuseNode(ls, ray, eval_light, surf, base_color, roughness, mix_weight, col, sh_r);
                }
#endif
                const simd_ivec<S> gen_ray =
                    (diff_depth < ps.max_diff_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_DiffuseNode(ray, gen_ray, surf, base_color, roughness, rand_u, rand_v, mix_weight, new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == GlossyNode) {
                const float specular = 0.5f;
                const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
                const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);
                const simd_fvec<S> roughness2 = sqr(roughness);

#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(sqr(roughness2) >= 1e-7f) &
                                                simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |=
                        Evaluate_GlossyNode(ls, ray, eval_light, surf, base_color, roughness, simd_fvec<S>{spec_ior},
                                            simd_fvec<S>{spec_F0}, mix_weight, col, sh_r);
                };
#endif

                const simd_ivec<S> gen_ray =
                    (spec_depth < ps.max_spec_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_GlossyNode(ray, gen_ray, surf, base_color, roughness, simd_fvec<S>{spec_ior},
                                      simd_fvec<S>{spec_F0}, rand_u, rand_v, mix_weight, new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == RefractiveNode) {
#if USE_NEE
                const simd_fvec<S> roughness2 = sqr(roughness);
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(sqr(roughness2) >= 1e-7f) &
                                                simd_cast(N_dot_L < 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    simd_fvec<S> eta = (ext_ior / mat->ior);
                    where(is_backfacing, eta) = (mat->ior / ext_ior);
                    shadow_mask |= Evaluate_RefractiveNode(ls, ray, eval_light, surf, base_color, roughness2, eta,
                                                           mix_weight, col, sh_r);
                }
#endif
                const simd_ivec<S> gen_ray =
                    (refr_depth < ps.max_refr_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_RefractiveNode(ray, gen_ray, surf, base_color, roughness, is_backfacing,
                                          simd_fvec<S>{mat->ior}, ext_ior, rand_u, rand_v, mix_weight, new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == EmissiveNode) {
                simd_fvec<S> mis_weight = 1.0f;
#if USE_NEE
                if ((ray.depth & 0x00ffffff).not_all_zeros() && (mat->flags & MAT_FLAG_MULT_IMPORTANCE)) {
                    const simd_fvec<S> v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]},
                                       v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};

                    simd_fvec<S> light_forward[3];
                    cross(v1, v2, light_forward);
                    TransformDirection(transform, light_forward);

                    const simd_fvec<S> light_forward_len = length(light_forward);
                    UNROLLED_FOR(i, 3, { light_forward[i] /= light_forward_len; })
                    const simd_fvec<S> tri_area = 0.5f * light_forward_len;

                    const simd_fvec<S> cos_theta = abs(dot3(I, light_forward));
                    const simd_fvec<S> light_pdf = (inter.t * inter.t) / (tri_area * cos_theta);
                    const simd_fvec<S> &bsdf_pdf = ray.pdf;

                    where((cos_theta > 0.0f) & simd_cast((ray.depth & 0x00ffffff) != 0), mis_weight) =
                        power_heuristic(bsdf_pdf, light_pdf);
                }
#endif
                UNROLLED_FOR(i, 3, {
                    where(ray_queue[index], col[i]) += mix_weight * mis_weight * mat->strength * base_color[i];
                })
            } else if (mat->type == PrincipledNode) {
                simd_fvec<S> metallic = unpack_unorm_16(mat->metallic_unorm);
                if (mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
                    const uint32_t metallic_tex = mat->textures[METALLIC_TEXTURE];
                    const simd_fvec<S> metallic_lod = get_texture_lod(textures, metallic_tex, lambda, ray_queue[index]);
                    simd_fvec<S> metallic_color[4] = {};
                    SampleBilinear(textures, metallic_tex, surf.uvs, simd_ivec<S>(metallic_lod), ray_queue[index],
                                   metallic_color);

                    metallic *= metallic_color[0];
                }

                simd_fvec<S> specular = unpack_unorm_16(mat->specular_unorm);
                if (mat->textures[SPECULAR_TEXTURE] != 0xffffffff) {
                    const uint32_t specular_tex = mat->textures[SPECULAR_TEXTURE];
                    const simd_fvec<S> specular_lod = get_texture_lod(textures, specular_tex, lambda, ray_queue[index]);
                    simd_fvec<S> specular_color[4] = {};
                    SampleBilinear(textures, specular_tex, surf.uvs, simd_ivec<S>(specular_lod), ray_queue[index],
                                   specular_color);
                    if (specular_tex & TEX_SRGB_BIT) {
                        srgb_to_rgb(specular_color, specular_color);
                    }
                    specular *= specular_color[0];
                }

                const float transmission = unpack_unorm_16(mat->transmission_unorm);
                const float clearcoat = unpack_unorm_16(mat->clearcoat_unorm);
                const float clearcoat_roughness = unpack_unorm_16(mat->clearcoat_roughness_unorm);
                const float sheen = 2.0f * unpack_unorm_16(mat->sheen_unorm);
                const float sheen_tint = unpack_unorm_16(mat->sheen_tint_unorm);

                diff_params_t<S> diff;
                UNROLLED_FOR(i, 3, { diff.base_color[i] = base_color[i]; })
                UNROLLED_FOR(i, 3,
                             { diff.sheen_color[i] = sheen * mix(simd_fvec<S>{1.0f}, tint_color[i], sheen_tint); })
                diff.roughness = roughness;

                spec_params_t<S> spec;
                UNROLLED_FOR(i, 3, {
                    spec.tmp_col[i] = mix(simd_fvec<S>{1.0f}, tint_color[i], unpack_unorm_16(mat->specular_tint_unorm));
                    spec.tmp_col[i] = mix(specular * 0.08f * spec.tmp_col[i], base_color[i], metallic);
                })
                spec.roughness = roughness;
                spec.ior = (2.0f / (1.0f - sqrt(0.08f * specular))) - 1.0f;
                spec.F0 = fresnel_dielectric_cos(simd_fvec<S>{1.0f}, spec.ior);
                spec.anisotropy = unpack_unorm_16(mat->anisotropic_unorm);

                clearcoat_params_t<S> coat;
                coat.roughness = clearcoat_roughness;
                coat.ior = (2.0f / (1.0f - std::sqrt(0.08f * clearcoat))) - 1.0f;
                coat.F0 = fresnel_dielectric_cos(simd_fvec<S>{1.0f}, coat.ior);

                transmission_params_t<S> trans;
                trans.roughness =
                    1.0f - (1.0f - roughness) * (1.0f - unpack_unorm_16(mat->transmission_roughness_unorm));
                trans.int_ior = mat->ior;
                trans.eta = (ext_ior / mat->ior);
                where(is_backfacing, trans.eta) = (mat->ior / ext_ior);
                trans.fresnel = fresnel_dielectric_cos(dot3(I, surf.N), 1.0f / trans.eta);
                trans.backfacing = is_backfacing;

                // Approximation of FH (using shading normal)
                const simd_fvec<S> FN =
                    (fresnel_dielectric_cos(dot3(I, surf.N), spec.ior) - spec.F0) / (1.0f - spec.F0);

                simd_fvec<S> approx_spec_col[3];
                UNROLLED_FOR(i, 3, { approx_spec_col[i] = mix(spec.tmp_col[i], simd_fvec<S>{1.0f}, FN); })

                const simd_fvec<S> spec_color_lum = lum(approx_spec_col);

                lobe_weights_t<S> lobe_weights;
                get_lobe_weights(mix(base_color_lum, simd_fvec<S>{1.0f}, sheen), spec_color_lum, specular, metallic,
                                 transmission, clearcoat, lobe_weights);

#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= Evaluate_PrincipledNode(ls, ray, eval_light, surf, lobe_weights, diff, spec, coat,
                                                           trans, metallic, N_dot_L, mix_weight, col, sh_r);
                }
#endif
                Sample_PrincipledNode(ps, ray, ray_queue[index], surf, lobe_weights, diff, spec, coat, trans, metallic,
                                      rand_u, rand_v, mix_rand, mix_weight, secondary_mask, new_ray);
            } /*else if (mat->type == TransparentNode) {
                assert(false);
            }*/

            ++index;
        }
    }

#if USE_PATH_TERMINATION
    const simd_ivec<S> can_terminate_path = total_depth > int(ps.min_total_depth);
#else
    const simd_ivec<S> can_terminate_path = 0;
#endif

    const simd_fvec<S> lum = max(new_ray.c[0], max(new_ray.c[1], new_ray.c[2]));
    const simd_fvec<S> p = fract(gather(random_seq + RAND_DIM_TERMINATE, rand_index) + sample_off[0]);
    simd_fvec<S> q = 0.0f;
    where(can_terminate_path, q) = max(0.05f, 1.0f - lum);

    secondary_mask &= simd_cast(p >= q) & simd_cast(lum > 0.0f) & simd_cast(new_ray.pdf > 0.0f);
    if (secondary_mask.not_all_zeros()) {
        UNROLLED_FOR(i, 3, { new_ray.c[i] = safe_div_pos(new_ray.c[i], 1.0f - q); })

        // TODO: check if this is needed
        new_ray.pdf = min(new_ray.pdf, 1e6f);

        // TODO: get rid of this!
        UNROLLED_FOR(i, 3, { where(~secondary_mask, new_ray.d[i]) = 0.0f; })

        const int index = (*out_secondary_rays_count)++;
        out_secondary_masks[index] = secondary_mask;
    }

    if (shadow_mask.not_all_zeros()) {
        // actual ray direction accouning for bias from both ends
        simd_fvec<S> to_light[3];
        UNROLLED_FOR(i, 3, { to_light[i] = ls.lp[i] - sh_r.o[i]; })

        sh_r.dist = length(to_light);
        UNROLLED_FOR(i, 3, { where(shadow_mask, sh_r.d[i]) = safe_div_pos(to_light[i], sh_r.dist); })
        sh_r.dist *= ls.dist_mul;
        // NOTE: hacky way to identify env ray
        where(ls.from_env & shadow_mask, sh_r.dist) = -sh_r.dist;

        const int index = (*out_shadow_rays_count)++;
        out_shadow_masks[index] = shadow_mask;
    }

    UNROLLED_FOR(i, 3, { where(is_active_lane, out_rgba[i]) = ray.c[i] * col[i]; })
    where(is_active_lane, out_rgba[3]) = 1.0f;
}

#undef sqr

#undef USE_VNDF_GGX_SAMPLING
#undef USE_NEE
#undef USE_PATH_TERMINATION
#undef FORCE_TEXTURE_LOD

#pragma warning(pop)