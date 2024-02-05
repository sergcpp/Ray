#pragma once
//  This file is compiled many times for different simd architectures (SSE, NEON...).
//  Macro 'NS' defines a namespace in which everything will be located, so it should be set before including this file.
//  Macros 'USE_XXX' define template instantiation of simd_vec classes.
//  Template parameter S defines width of vectors used. Usualy it is equal to ray packet size.

#include <vector>

#include <cfloat>

#include "simd/simd_vec.h"

#include "Convolution.h"
#include "TextureStorageCPU.h"

#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant

namespace Ray {
//
// Useful macros for debugging
//
#define USE_NEE 1
#define USE_HIERARCHICAL_NEE 1
#define USE_PATH_TERMINATION 1
// #define FORCE_TEXTURE_LOD 0
#define USE_STOCH_TEXTURE_FILTERING 1
#define USE_SPHERICAL_AREA_LIGHT_SAMPLING 1
#define USE_SAFE_MATH 1

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
namespace NS {
// Up to 4x4 rays
// [ 0] [ 1] [ 4] [ 5]
// [ 2] [ 3] [ 6] [ 7]
// [ 8] [ 9] [12] [13]
// [10] [11] [14] [15]
alignas(64) const int rays_layout_x[] = {0, 1, 0, 1,  // NOLINT
                                         2, 3, 2, 3,  // NOLINT
                                         0, 1, 0, 1,  // NOLINT
                                         2, 3, 2, 3}; // NOLINT
alignas(64) const int rays_layout_y[] = {0, 0, 1, 1,  // NOLINT
                                         0, 0, 1, 1,  // NOLINT
                                         2, 2, 3, 3,  // NOLINT
                                         2, 2, 3, 3}; // NOLINT

// Usefull to make index argument for a gather instruction
alignas(64) const int ascending_counter[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int S> struct ray_data_t {
    // active rays mask
    simd_ivec<S> mask;
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
    simd_uvec<S> xy;
    // four 8-bit ray depth counters
    simd_uvec<S> depth;
};

template <int S> struct shadow_ray_t {
    // active rays mask
    simd_ivec<S> mask;
    // origins of rays in packet
    simd_fvec<S> o[3];
    // four 8-bit ray depth counters
    simd_uvec<S> depth;
    // directions of rays in packet
    simd_fvec<S> d[3], dist;
    // throughput color of ray
    simd_fvec<S> c[3];
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_uvec<S> xy;
};

template <int S> struct hit_data_t {
    simd_ivec<S> obj_index;
    simd_ivec<S> prim_index;
    simd_fvec<S> t, u, v;

    explicit hit_data_t(eUninitialize) {}
    force_inline hit_data_t() {
        obj_index = {-1};
        prim_index = {-1};
        t = MAX_DIST;
        u = 0.0f;
        v = -1.0f; // negative v means 'no intersection'
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

template <int S> force_inline simd_uvec<S> mask_ray_depth(const simd_uvec<S> depth) { return depth & 0x0fffffff; }
force_inline uint32_t pack_ray_type(const int ray_type) {
    assert(ray_type < 0xf);
    return uint32_t(ray_type << 28);
}
template <int S>
force_inline simd_uvec<S> pack_depth(const simd_ivec<S> &diff_depth, const simd_ivec<S> &spec_depth,
                                     const simd_ivec<S> &refr_depth, const simd_ivec<S> &transp_depth) {
    assert((diff_depth >= 0x7f).all_zeros() && (spec_depth >= 0x7f).all_zeros() && (refr_depth >= 0x7f).all_zeros() &&
           (transp_depth >= 0x7f).all_zeros());
    simd_uvec<S> ret = 0u;
    ret |= simd_uvec<S>(diff_depth) << 0u;
    ret |= simd_uvec<S>(spec_depth) << 7u;
    ret |= simd_uvec<S>(refr_depth) << 14u;
    ret |= simd_uvec<S>(transp_depth) << 21u;
    return ret;
}
template <int S> force_inline simd_ivec<S> get_diff_depth(const simd_uvec<S> &depth) {
    return simd_ivec<S>(depth & 0x7f);
}
template <int S> force_inline simd_ivec<S> get_spec_depth(const simd_uvec<S> &depth) {
    return simd_ivec<S>(depth >> 7) & 0x7f;
}
template <int S> force_inline simd_ivec<S> get_refr_depth(const simd_uvec<S> &depth) {
    return simd_ivec<S>(depth >> 14) & 0x7f;
}
template <int S> force_inline simd_ivec<S> get_transp_depth(const simd_uvec<S> &depth) {
    return simd_ivec<S>(depth >> 21) & 0x7f;
}
template <int S> force_inline simd_ivec<S> get_total_depth(const simd_uvec<S> &depth) {
    return get_diff_depth(depth) + get_spec_depth(depth) + get_refr_depth(depth) + get_transp_depth(depth);
}
template <int S> force_inline simd_ivec<S> get_ray_type(const simd_uvec<S> &depth) {
    return simd_ivec<S>(depth >> 28) & 0xf;
}

template <int S> force_inline simd_ivec<S> is_indirect(const simd_uvec<S> &depth) {
    // not only transparency ray
    return simd_ivec<S>((depth & 0x001fffff) != 0u);
}

// Generating rays
template <int DimX, int DimY>
void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, const uint32_t rand_seq[],
                         uint32_t rand_seed, const float filter_table[], int iteration,
                         const uint16_t required_samples[], aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                         aligned_vector<hit_data_t<DimX * DimY>> &out_inters);
template <int DimX, int DimY>
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const mesh_instance_t &mi,
                              const uint32_t *vtx_indices, const vertex_t *vertices, const rect_t &r, int w, int h,
                              const uint32_t rand_seq[], aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                              aligned_vector<hit_data_t<DimX * DimY>> &out_inters);

// Sorting rays
template <int S>
int SortRays_CPU(Span<ray_data_t<S>> rays, const float root_min[3], const float cell_size[3], simd_ivec<S> *hash_values,
                 uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
template <int S>
int SortRays_GPU(Span<ray_data_t<S>> rays, const float root_min[3], const float cell_size[3], simd_ivec<S> *hash_values,
                 int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp,
                 uint32_t *skeleton);

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
bool Traverse_TLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                        const simd_uvec<S> &ray_flags, const simd_ivec<S> &ray_mask,
                                        const bvh_node_t *nodes, uint32_t node_index,
                                        const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                        const mesh_t *meshes, const tri_accel_t *tris, const uint32_t *tri_indices,
                                        hit_data_t<S> &inter);
template <int S>
bool Traverse_TLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                        const simd_uvec<S> &ray_flags, const simd_ivec<S> &ray_mask,
                                        const wbvh_node_t *nodes, uint32_t node_index,
                                        const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                        const mesh_t *meshes, const mtri_accel_t *mtris, const uint32_t *tri_indices,
                                        hit_data_t<S> &inter);
template <int S>
simd_ivec<S> Traverse_TLAS_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], int ray_type,
                                            const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                            const mesh_t *meshes, const tri_accel_t *tris,
                                            const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                            hit_data_t<S> &inter);
template <int S>
simd_ivec<S> Traverse_TLAS_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], int ray_type,
                                            const simd_ivec<S> &ray_mask, const wbvh_node_t *nodes, uint32_t node_index,
                                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                            const mesh_t *meshes, const mtri_accel_t *mtris,
                                            const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                            hit_data_t<S> &inter);
// traditional bvh traversal with stack for inner nodes
template <int S>
bool Traverse_BLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                        const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                        const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index,
                                        hit_data_t<S> &inter);
template <int S>
bool Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const wbvh_node_t *nodes,
                                        uint32_t node_index, const mtri_accel_t *mtris, const uint32_t *tri_indices,
                                        int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v);
// returns 0 - no hit, 1 - hit, 2 - solid hit (no need to check for transparency)
template <int S>
simd_ivec<S> Traverse_BLAS_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                            const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                            const tri_accel_t *tris, const tri_mat_data_t *materials,
                                            const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);
template <int S>
int Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const wbvh_node_t *nodes, uint32_t node_index,
                                   const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                   const uint32_t *tri_indices, int &inter_prim_index, float &inter_t, float &inter_u,
                                   float &inter_v);

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
                                   bool uniform_sampling, const simd_fvec<S> rand[2], simd_fvec<S> out_V[3],
                                   simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_GGXSpecular_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                               const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> alpha[2],
                               const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                               const simd_fvec<S> spec_col[3], const simd_fvec<S> spec_col_90[3],
                               simd_fvec<S> out_color[4]);
template <int S>
void Sample_GGXSpecular_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                             const simd_fvec<S> I[3], const simd_fvec<S> alpha[2], const simd_fvec<S> &spec_ior,
                             const simd_fvec<S> &spec_F0, const simd_fvec<S> spec_col[3],
                             const simd_fvec<S> spec_col_90[3], const simd_fvec<S> rand[2], simd_fvec<S> out_V[3],
                             simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_GGXRefraction_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                 const simd_fvec<S> refr_dir_ts[3], const simd_fvec<S> alpha[2],
                                 const simd_fvec<S> &eta, const simd_fvec<S> refr_col[3], simd_fvec<S> out_color[4]);
template <int S>
void Sample_GGXRefraction_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                               const simd_fvec<S> I[3], const simd_fvec<S> alpha[2], const simd_fvec<S> &eta,
                               const simd_fvec<S> refr_col[3], const simd_fvec<S> rand[2], simd_fvec<S> out_V[4],
                               simd_fvec<S> out_color[4]);

template <int S>
void Evaluate_PrincipledClearcoat_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                       const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> &clearcoat_roughness2,
                                       const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                       simd_fvec<S> out_color[4]);
template <int S>
void Sample_PrincipledClearcoat_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> I[3], const simd_fvec<S> &clearcoat_roughness2,
                                     const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                     const simd_fvec<S> rand[2], simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]);

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
template <int S> void TransformPoint(const simd_fvec<S> xform[16], simd_fvec<S> out_p[3]);
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
void SampleNearest(const Cpu::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                   const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Cpu::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                    const simd_ivec<S> &lod, const simd_fvec<S> rand[2], const simd_ivec<S> &mask,
                    simd_fvec<S> out_rgba[4]);
template <int S>
void SampleTrilinear(const Cpu::TexStorageBase *const textures[], uint32_t index, const simd_fvec<S> uvs[2],
                     const simd_fvec<S> &lod, const simd_fvec<S> rand[2], const simd_ivec<S> &mask,
                     simd_fvec<S> out_rgba[4]);
template <int S>
void SampleLatlong_RGBE(const Cpu::TexStorageRGBA &storage, uint32_t index, const simd_fvec<S> dir[3], float y_rotation,
                        const simd_fvec<S> rand[2], const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]);

// Trace rays through scene hierarchy
template <int S>
void IntersectScene(ray_data_t<S> &r, int min_transp_depth, int max_transp_depth, const uint32_t rand_seq[],
                    uint32_t rand_seed, int iteration, const scene_data_t &sc, uint32_t root_index,
                    const Cpu::TexStorageBase *const textures[], hit_data_t<S> &inter);
template <int S>
void IntersectScene(const shadow_ray_t<S> &r, int max_transp_depth, const scene_data_t &sc, uint32_t node_index,
                    const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                    const Cpu::TexStorageBase *const textures[], simd_fvec<S> rc[3]);

// Pick point on any light source for evaluation
template <int S>
void SampleLightSource(const simd_fvec<S> P[3], const simd_fvec<S> T[3], const simd_fvec<S> B[3],
                       const simd_fvec<S> N[3], const scene_data_t &sc, const Cpu::TexStorageBase *const tex_atlases[],
                       const simd_fvec<S> &rand_pick_light, const simd_fvec<S> rand_light_uv[2],
                       const simd_fvec<S> rand_tex_uv[2], simd_ivec<S> ray_mask, light_sample_t<S> &ls);

// Account for visible lights contribution
template <int S>
void IntersectAreaLights(const ray_data_t<S> &r, Span<const light_t> lights, Span<const light_wbvh_node_t> nodes,
                         hit_data_t<S> &inout_inter);
template <int S>
simd_fvec<S> IntersectAreaLights(const shadow_ray_t<S> &r, Span<const light_t> lights,
                                 Span<const light_wbvh_node_t> nodes);
template <int S>
simd_fvec<S> EvalTriLightFactor(const simd_fvec<S> P[3], const simd_fvec<S> ro[3], const simd_ivec<S> &mask,
                                const simd_ivec<S> &tri_index, Span<const light_t> lights,
                                Span<const light_wbvh_node_t> nodes);

template <int S>
void TraceRays(Span<ray_data_t<S>> rays, int min_transp_depth, int max_transp_depth, const scene_data_t &sc,
               uint32_t root_index, bool trace_lights, const Cpu::TexStorageBase *const textures[],
               const uint32_t rand_seq[], uint32_t random_seed, int iteration, Span<hit_data_t<S>> out_inter);
template <int S>
void TraceShadowRays(Span<const shadow_ray_t<S>> rays, int max_transp_depth, float clamp_val, const scene_data_t &sc,
                     uint32_t root_index, const uint32_t rand_seq[], uint32_t random_seed, int iteration,
                     const Cpu::TexStorageBase *const textures[], int img_w, color_rgba_t *out_color);

// Get environment collor at direction
template <int S>
void Evaluate_EnvColor(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const environment_t &env,
                       const Cpu::TexStorageRGBA &tex_storage, const simd_fvec<S> &pdf_factor,
                       const simd_fvec<S> rand[2], simd_fvec<S> env_col[4]);
// Get light color at intersection point
template <int S>
void Evaluate_LightColor(const simd_fvec<S> P[3], const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                         const hit_data_t<S> &inter, const environment_t &env, Span<const light_t> lights,
                         uint32_t lights_count, const Cpu::TexStorageRGBA &tex_storage, const simd_fvec<S> rand[2],
                         simd_fvec<S> light_col[3]);

// Evaluate individual nodes
template <int S>
simd_ivec<S> Evaluate_DiffuseNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                  const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                  const simd_fvec<S> &roughness, const simd_fvec<S> &mix_weight,
                                  const simd_ivec<S> &mis_mask, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_DiffuseNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                        const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness, const simd_fvec<S> &rand_u,
                        const simd_fvec<S> &rand_v, const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray);

template <int S>
simd_ivec<S> Evaluate_GlossyNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, simd_ivec<S> mask,
                                 const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                 const simd_fvec<S> &roughness, const simd_fvec<S> &regularize_alpha,
                                 const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                                 const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask, simd_fvec<S> out_col[3],
                                 shadow_ray_t<S> &sh_r);
template <int S>
void Sample_GlossyNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                       const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                       const simd_fvec<S> &regularize_alpha, const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                       const simd_fvec<S> rand[2], const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray);

template <int S>
simd_ivec<S> Evaluate_RefractiveNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                     const surface_t<S> &surf, const simd_fvec<S> base_color[3],
                                     const simd_fvec<S> &roughness, const simd_fvec<S> &regularize_alpha,
                                     const simd_fvec<S> &eta, const simd_fvec<S> &mix_weight,
                                     const simd_ivec<S> &mis_mask, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_RefractiveNode(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const surface_t<S> &surf,
                           const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                           const simd_fvec<S> &regularize_alpha, const simd_ivec<S> &is_backfacing,
                           const simd_fvec<S> &int_ior, const simd_fvec<S> &ext_ior, const simd_fvec<S> rand[2],
                           const simd_fvec<S> &mix_weight, ray_data_t<S> &new_ray);

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
simd_ivec<S>
Evaluate_PrincipledNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                        const surface_t<S> &surf, const lobe_weights_t<S> &lobe_weights, const diff_params_t<S> &diff,
                        const spec_params_t<S> &spec, const clearcoat_params_t<S> &coat,
                        const transmission_params_t<S> &trans, const simd_fvec<S> &metallic, float transmission,
                        const simd_fvec<S> &N_dot_L, const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask,
                        const simd_fvec<S> &regularize_alpha, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r);
template <int S>
void Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                           const surface_t<S> &surf, const lobe_weights_t<S> &lobe_weights,
                           const diff_params_t<S> &diff, const spec_params_t<S> &spec,
                           const clearcoat_params_t<S> &coat, const transmission_params_t<S> &trans,
                           const simd_fvec<S> &metallic, float transmission, const simd_fvec<S> rand[2],
                           simd_fvec<S> mix_rand, const simd_fvec<S> &mix_weight, const simd_fvec<S> &regularize_alpha,
                           simd_ivec<S> &secondary_mask, ray_data_t<S> &new_ray);

// Shade
template <int S>
void ShadeSurface(const pass_settings_t &ps, const float limits[2], const uint32_t rand_seq[], uint32_t rand_seed,
                  int iteration, const hit_data_t<S> &inter, const ray_data_t<S> &ray, const scene_data_t &sc,
                  uint32_t node_index, const Cpu::TexStorageBase *const tex_atlases[], simd_fvec<S> out_rgba[4],
                  ray_data_t<S> out_secondary_rays[], int *out_secondary_rays_count, shadow_ray_t<S> out_shadow_rays[],
                  int *out_shadow_rays_count, simd_fvec<S> out_base_color[4], simd_fvec<S> out_depth_normals[4]);
template <int S>
void ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t<S>> inters, Span<const ray_data_t<S>> rays,
                  const uint32_t rand_seq[], uint32_t rans_seed, int iteration, const scene_data_t &sc,
                  uint32_t node_index, const Cpu::TexStorageBase *const textures[], ray_data_t<S> *out_secondary_rays,
                  int *out_secondary_rays_count, shadow_ray_t<S> *out_shadow_rays, int *out_shadow_rays_count,
                  int img_w, float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                  color_rgba_t *out_depth_normals);
template <int S>
void ShadeSecondary(const pass_settings_t &ps, float clamp_direct, Span<const hit_data_t<S>> inters,
                    Span<const ray_data_t<S>> rays, const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                    const scene_data_t &sc, uint32_t node_index, const Cpu::TexStorageBase *const textures[],
                    ray_data_t<S> *out_secondary_rays, int *out_secondary_rays_count, shadow_ray_t<S> *out_shadow_rays,
                    int *out_shadow_rays_count, int img_w, color_rgba_t *out_color);

template <int S, int InChannels, int OutChannels, int OutPxPitch = OutChannels, ePostOp PostOp = ePostOp::None,
          eActivation Activation = eActivation::ReLU>
void Convolution3x3_Direct(const float data[], const rect_t &rect, int w, int h, int stride, const float weights[],
                           const float biases[], float output[], int output_stride);

template <int S, int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3_Direct(const float data1[], const float data2[], const rect_t &rect, int w, int h,
                                 int stride1, int stride2, const float weights[], const float biases[], float output[],
                                 int output_stride);
template <int S, int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
          Ray::ePreOp PreOp1, Ray::ePreOp PreOp2, Ray::ePreOp PreOp3, Ray::ePreOp PreOp4, Ray::ePostOp PostOp,
          Ray::eActivation Activation>
void ConvolutionConcat3x3_1Direct_2GEMM(const float data1[], const float data2[], const float data3[],
                                        const float data4[], const rect_t &rect, int w, int h, int w2, int h2,
                                        int stride1, int stride2, const float weights[], const float biases[],
                                        float output[], int output_stride);

class SIMDPolicyBase {
  public:
    using RayDataType = ray_data_t<RPSize>;
    using ShadowRayType = shadow_ray_t<RPSize>;
    using HitDataType = hit_data_t<RPSize>;
    using RayHashType = simd_ivec<RPSize>;

  protected:
    static force_inline void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, const int w, const int h,
                                                 const uint32_t rand_seq[], const uint32_t rand_seed,
                                                 const float filter_table[], const int iteration,
                                                 const uint16_t required_samples[],
                                                 aligned_vector<RayDataType> &out_rays,
                                                 aligned_vector<HitDataType> &out_inters) {
        NS::GeneratePrimaryRays<RPDimX, RPDimY>(cam, r, w, h, rand_seq, rand_seed, filter_table, iteration,
                                                required_samples, out_rays, out_inters);
    }

    static force_inline void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                                      const mesh_instance_t &mi, const uint32_t *vtx_indices,
                                                      const vertex_t *vertices, const rect_t &r, int w, int h,
                                                      const uint32_t rand_seq[], aligned_vector<RayDataType> &out_rays,
                                                      aligned_vector<HitDataType> &out_inters) {
        NS::SampleMeshInTextureSpace<RPDimX, RPDimY>(iteration, obj_index, uv_layer, mesh, mi, vtx_indices, vertices, r,
                                                     w, h, rand_seq, out_rays, out_inters);
    }

    static force_inline void TraceRays(Span<RayDataType> rays, int min_transp_depth, int max_transp_depth,
                                       const scene_data_t &sc, uint32_t root_index, bool trace_lights,
                                       const Cpu::TexStorageBase *const textures[], const uint32_t rand_seq[],
                                       const uint32_t rand_seed, const int iteration, Span<HitDataType> out_inter) {
        NS::TraceRays<RPSize>(rays, min_transp_depth, max_transp_depth, sc, root_index, trace_lights, textures,
                              rand_seq, rand_seed, iteration, out_inter);
    }

    static force_inline void TraceShadowRays(Span<const ShadowRayType> rays, int max_transp_depth, float clamp_val,
                                             const scene_data_t &sc, uint32_t node_index, const uint32_t rand_seq[],
                                             const uint32_t rand_seed, const int iteration,
                                             const Cpu::TexStorageBase *const textures[], int img_w,
                                             color_rgba_t *out_color) {
        NS::TraceShadowRays<RPSize>(rays, max_transp_depth, clamp_val, sc, node_index, rand_seq, rand_seed, iteration,
                                    textures, img_w, out_color);
    }

    static force_inline int SortRays_CPU(Span<RayDataType> rays, const float root_min[3], const float cell_size[3],
                                         RayHashType *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                                         ray_chunk_t *chunks_temp) {
        return NS::SortRays_CPU<RPSize>(rays, root_min, cell_size, hash_values, scan_values, chunks, chunks_temp);
    }

    static force_inline void ShadePrimary(const pass_settings_t &ps, Span<const HitDataType> inters,
                                          Span<const RayDataType> rays, const uint32_t rand_seq[],
                                          const uint32_t rand_seed, const int iteration, const scene_data_t &sc,
                                          uint32_t node_index, const Cpu::TexStorageBase *const textures[],
                                          RayDataType *out_secondary_rays, int *out_secondary_rays_count,
                                          ShadowRayType *out_shadow_rays, int *out_shadow_rays_count, int img_w,
                                          float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                                          color_rgba_t *out_depth_normal) {
        NS::ShadePrimary<RPSize>(ps, inters, rays, rand_seq, rand_seed, iteration, sc, node_index, textures,
                                 out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count,
                                 img_w, mix_factor, out_color, out_base_color, out_depth_normal);
    }

    static force_inline void ShadeSecondary(const pass_settings_t &ps, const float clamp_direct,
                                            Span<const HitDataType> inters, Span<const RayDataType> rays,
                                            const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                                            const scene_data_t &sc, uint32_t node_index,
                                            const Cpu::TexStorageBase *const textures[],
                                            RayDataType *out_secondary_rays, int *out_secondary_rays_count,
                                            ShadowRayType *out_shadow_rays, int *out_shadow_rays_count, int img_w,
                                            color_rgba_t *out_color) {
        NS::ShadeSecondary<RPSize>(ps, clamp_direct, inters, rays, rand_seq, rand_seed, iteration, sc, node_index,
                                   textures, out_secondary_rays, out_secondary_rays_count, out_shadow_rays,
                                   out_shadow_rays_count, img_w, out_color);
    }

    template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels,
              ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
              ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void Convolution3x3_GEMM(const float data1[], const float data2[], const float data3[],
                                                 const rect_t &rect, int in_w, int in_h, int w, int h, int stride,
                                                 const float weights[], const float biases[], float output[],
                                                 int output_stride) {
        NS::Convolution3x3_GEMM<RPSize, InChannels1, InChannels2, InChannels3, PxPitch, OutChannels, PreOp1, PreOp2,
                                PreOp3, PostOp, Activation>(data1, data2, data3, rect, in_w, in_h, w, h, stride,
                                                            weights, biases, output, output_stride);
    }

    template <int InChannels, int OutChannels, int OutPxPitch = OutChannels, ePostOp PostOp = ePostOp::None,
              eActivation Activation = eActivation::ReLU>
    static force_inline void Convolution3x3_Direct(const float data[], const rect_t &rect, int w, int h, int stride,
                                                   const float weights[], const float biases[], float output[],
                                                   int output_stride) {
        NS::Convolution3x3_Direct<RPSize, InChannels, OutChannels, OutPxPitch, PostOp, Activation>(
            data, rect, w, h, stride, weights, biases, output, output_stride);
    }

    template <int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
              ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void ConvolutionConcat3x3_Direct(const float data1[], const float data2[], const rect_t &rect,
                                                         int w, int h, int stride1, int stride2, const float weights[],
                                                         const float biases[], float output[], int output_stride) {
        NS::ConvolutionConcat3x3_Direct<RPSize, InChannels1, InChannels2, OutChannels, PreOp1, PostOp, Activation>(
            data1, data2, rect, w, h, stride1, stride2, weights, biases, output, output_stride);
    }

    template <int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
              ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
              ePreOp PreOp4 = ePreOp::None, ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void
    ConvolutionConcat3x3_1Direct_2GEMM(const float data1[], const float data2[], const float data3[],
                                       const float data4[], const rect_t &rect, int w, int h, int w2, int h2,
                                       int stride1, int stride2, const float weights[], const float biases[],
                                       float output[], int output_stride) {
        NS::ConvolutionConcat3x3_1Direct_2GEMM<RPSize, InChannels1, InChannels2, InChannels3, InChannels4, PxPitch2,
                                               OutChannels, PreOp1, PreOp2, PreOp3, PreOp4, PostOp, Activation>(
            data1, data2, data3, data4, rect, w, h, w2, h2, stride1, stride2, weights, biases, output, output_stride);
    }

    static force_inline void ClearBorders(const rect_t &rect, int w, int h, bool downscaled, int out_channels,
                                          float output[]) {
        NS::ClearBorders(rect, w, h, downscaled, out_channels, output);
    }
};
} // namespace NS
} // namespace Ray

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace Ray {
namespace NS {
template <int S> force_inline simd_fvec<S> safe_inv(const simd_fvec<S> &a) {
#if USE_SAFE_MATH
    const simd_fvec<S> denom = select(a != 0.0f, a, simd_fvec<S>{FLT_EPS});
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
    const simd_fvec<S> denom = select(b != 0.0f, b, simd_fvec<S>{FLT_EPS});
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
    return a / fmaxf(b, FLT_EPS);
#else
    return a / b;
#endif
}

force_inline float safe_div_pos(const float a, const float b) {
#if USE_SAFE_MATH
    return a / fmaxf(b, FLT_EPS);
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

    UNROLLED_FOR_R(i, LanesCount, {
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
        tmin.store_to(&out_dist[S * i], simd_mem_aligned);
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
    tmin.store_to(&out_dist[0], simd_mem_aligned);

    return res;
}

template <int S>
force_inline void bbox_test_oct(const float p[3], const simd_fvec<S> bbox_min[3], const simd_fvec<S> bbox_max[3],
                                simd_ivec<S> &out_mask) {
    const simd_fvec<S> mask = (bbox_min[0] < p[0]) & (bbox_max[0] > p[0]) & (bbox_min[1] < p[1]) &
                              (bbox_max[1] > p[1]) & (bbox_min[2] < p[2]) & (bbox_max[2] > p[2]);
    out_mask = simd_cast(mask);
}

template <int S>
force_inline long bbox_test_oct(const float p[3], const float bbox_min[3][8], const float bbox_max[3][8]) {
    long res = 0;

    static const int LanesCount = (8 / S);

    UNROLLED_FOR_R(i, LanesCount, {
        const simd_fvec<S> fmask = (simd_fvec<S>{&bbox_min[0][S * i], simd_mem_aligned} <= p[0]) &
                                   (simd_fvec<S>{&bbox_max[0][S * i], simd_mem_aligned} >= p[0]) &
                                   (simd_fvec<S>{&bbox_min[1][S * i], simd_mem_aligned} <= p[1]) &
                                   (simd_fvec<S>{&bbox_max[1][S * i], simd_mem_aligned} >= p[1]) &
                                   (simd_fvec<S>{&bbox_min[2][S * i], simd_mem_aligned} <= p[2]) &
                                   (simd_fvec<S>{&bbox_max[2][S * i], simd_mem_aligned} >= p[2]);

        res <<= S;
        res |= simd_cast(fmask).movemask();
    })

    return res;
}

template <>
force_inline long bbox_test_oct<16>(const float p[3], const float bbox_min[3][8], const float bbox_max[3][8]) {
    const simd_fvec<8> fmask = (simd_fvec<8>{&bbox_min[0][0], simd_mem_aligned} <= p[0]) &
                               (simd_fvec<8>{&bbox_max[0][0], simd_mem_aligned} >= p[0]) &
                               (simd_fvec<8>{&bbox_min[1][0], simd_mem_aligned} <= p[1]) &
                               (simd_fvec<8>{&bbox_max[1][0], simd_mem_aligned} >= p[1]) &
                               (simd_fvec<8>{&bbox_min[2][0], simd_mem_aligned} <= p[2]) &
                               (simd_fvec<8>{&bbox_max[2][0], simd_mem_aligned} >= p[2]);
    return simd_cast(fmask).movemask();
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
force_inline bool is_leaf_node(const wbvh_node_t &node) { return (node.child[0] & LEAF_NODE_BIT) != 0; }

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

struct light_stack_entry_t {
    uint32_t index;
    float dist;
    float factor;
};

template <int StackSize, typename T = stack_entry_t> class TraversalStateStack_Single {
  public:
    T stack[StackSize];
    uint32_t stack_size = 0;

    template <class... Args> force_inline void push(Args &&...args) {
        stack[stack_size++] = {std::forward<Args>(args)...};
        assert(stack_size < StackSize && "Traversal stack overflow!");
    }

    force_inline T pop() {
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
                const T tmp = stack[i];
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
                const T tmp = stack[i];
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
            const T key = stack[i];

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
        const simd_fvec<S> denom = select(d[i] != 0.0f, d[i], simd_fvec<S>{FLT_EPS});

        inv_d[i] = 1.0f / denom;
        inv_d_o[i] = o[i] * inv_d[i];

        const simd_fvec<S> d_is_plus_zero = (d[i] <= FLT_EPS) & (d[i] >= 0.0f);
        where(d_is_plus_zero, inv_d[i]) = MAX_DIST;
        where(d_is_plus_zero, inv_d_o[i]) = MAX_DIST;

        const simd_fvec<S> d_is_minus_zero = (d[i] >= -FLT_EPS) & (d[i] < 0.0f);
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

template <int S> force_inline void cross(const simd_fvec<S> v1[3], const float v2[3], simd_fvec<S> res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

force_inline void cross(const float v1[3], const float v2[3], float res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template <int S> force_inline simd_fvec<S> normalize(simd_fvec<S> v[3]) {
    const simd_fvec<S> l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] = safe_div_pos(v[0], l);
    v[1] = safe_div_pos(v[1], l);
    v[2] = safe_div_pos(v[2], l);
    return l;
}

force_inline void normalize(float v[3]) {
    const float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
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

template <int S> force_inline simd_uvec<S> hash(simd_uvec<S> x) {
    // finalizer from murmurhash3
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

template <int S> force_inline simd_uvec<S> hash_combine(const simd_uvec<S> &seed, const simd_uvec<S> &v) {
    return seed ^ (v + (seed << 6) + (seed >> 2));
}

template <int S> force_inline simd_uvec<S> hash_combine(const simd_uvec<S> &seed, const uint32_t v) {
    return seed ^ (v + (seed << 6) + (seed >> 2));
}

template <int S> force_inline simd_uvec<S> reverse_bits(simd_uvec<S> x) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));
}

template <int S> force_inline simd_uvec<S> laine_karras_permutation(simd_uvec<S> x, const simd_uvec<S> &seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

template <int S> force_inline simd_uvec<S> nested_uniform_scramble_base2(simd_uvec<S> x, const simd_uvec<S> &seed) {
    x = reverse_bits(x);
    x = laine_karras_permutation(x, seed);
    x = reverse_bits(x);
    return x;
}

template <int S> force_inline simd_fvec<S> scramble_flt(const simd_uvec<S> &seed, const simd_fvec<S> &val) {
    simd_uvec<S> u = simd_uvec<S>(val * 16777216.0f) << 8;
    u = nested_uniform_scramble_base2(u, seed);
    return simd_fvec<S>(u >> 8) / 16777216.0f;
}

template <int S> force_inline simd_fvec<S> scramble_unorm(const simd_uvec<S> &seed, simd_uvec<S> val) {
    val = nested_uniform_scramble_base2(val, seed);
    return simd_fvec<S>(val >> 8) / 16777216.0f;
}

template <int S>
void get_scrambled_2d_rand(const simd_uvec<S> &dim, const simd_uvec<S> &seed, const int sample,
                           const uint32_t rand_seq[], simd_fvec<S> out_val[2]) {
    const simd_uvec<S> i_seed = hash_combine(seed, dim), x_seed = hash_combine(seed, 2 * dim + 0u),
                       y_seed = hash_combine(seed, 2 * dim + 1);

    const auto shuffled_dim = simd_ivec<S>(nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1));
    const auto shuffled_i =
        simd_ivec<S>(nested_uniform_scramble_base2(simd_uvec<S>(uint32_t(sample)), i_seed) & (RAND_SAMPLES_COUNT - 1));

    out_val[0] = scramble_unorm(x_seed, gather(rand_seq, shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0));
    out_val[1] = scramble_unorm(y_seed, gather(rand_seq, shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1));
}

// Gram-Schmidt method
template <int S>
force_inline void orthogonalize(const simd_fvec<S> a[3], const simd_fvec<S> b[3], simd_fvec<S> out_v[3]) {
    // we assume that a is normalized
    const simd_fvec<S> temp = dot3(a, b);
    UNROLLED_FOR(i, 3, { out_v[i] = b[i] - temp * a[i]; })
    normalize(out_v);
}

template <int S> force_inline simd_fvec<S> acos(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, { ret.set(i, acosf(v[i])); })
    return ret;
}

template <int S> force_inline simd_fvec<S> asin(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, { ret.set(i, asinf(v[i])); })
    return ret;
}

template <int S>
force_inline void slerp(const simd_fvec<S> start[3], const simd_fvec<S> end[3], const simd_fvec<S> &percent,
                        simd_fvec<S> out_v[3]) {
    // Dot product - the cosine of the angle between 2 vectors.
    simd_fvec<S> cos_theta = dot3(start, end);
    // Clamp it to be in the range of Acos()
    // This may be unnecessary, but floating point
    // precision can be a fickle mistress.
    cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    // Acos(dot) returns the angle between start and end,
    // And multiplying that by percent returns the angle between
    // start and the final result.
    const simd_fvec<S> theta = acos(cos_theta) * percent;
    simd_fvec<S> relative_vec[3];
    UNROLLED_FOR(i, 3, { relative_vec[i] = end[i] - start[i] * cos_theta; })
    safe_normalize(relative_vec);
    // Orthonormal basis
    // The final result.
    const simd_fvec<S> cos_theta2 = cos(theta), sin_theta = sin(theta);
    UNROLLED_FOR(i, 3, { out_v[i] = start[i] * cos_theta2 + relative_vec[i] * sin_theta; })
}

// Return arcsine(x) given that .57 < x
template <int S> force_inline simd_fvec<S> asin_tail(const simd_fvec<S> &x) {
    return (PI / 2) - ((x + 2.71745038f) * x + 14.0375338f) * (0.00440413551f * ((x - 8.31223679f) * x + 25.3978882f)) *
                          sqrt(1 - x);
}

template <int S> force_inline simd_fvec<S> portable_asinf(const simd_fvec<S> &x) {
    simd_fvec<S> ret;

    const simd_fvec<S> mask = abs(x) > 0.57f;
    where(mask, ret) = asin_tail(abs(x));
    where(x < 0.0f, ret) = -ret;

    const simd_fvec<S> x2 = x * x;
    where(~mask, ret) = x + (0.0517513789f * ((x2 + 1.83372748f) * x2 + 1.56678128f)) * x *
                                (x2 * ((x2 - 1.48268414f) * x2 + 2.05554748f));

    return ret;
}

// Equivalent to acosf(dot(a, b)), but more numerically stable
// Taken from PBRT source code
template <int S> simd_fvec<S> angle_between(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3]) {
    const simd_fvec<S> dot_mask = dot3(v1, v2) < 0.0f;

    simd_fvec<S> arg[3];
    UNROLLED_FOR(i, 3, {
        arg[i] = v2[i] - v1[i];
        where(dot_mask, arg[i]) = v1[i] + v2[i];
    })

    simd_fvec<S> ret = 2 * portable_asinf(length(arg) / 2);
    where(dot_mask, ret) = PI - ret;
    return ret;
}

template <int S> force_inline simd_fvec<S> acos_positive_tail(const simd_fvec<S> &x) {
    return (((x + 2.71850395f) * x + 14.7303705f)) * (0.00393401226f * ((x - 8.60734272f) * x + 27.0927486f)) *
           sqrt(1 - x);
}

template <int S> force_inline simd_fvec<S> acos_negative_tail(const simd_fvec<S> &x) {
    return PI - (((x - 2.71850395f) * x + 14.7303705f)) * (0.00393401226f * ((x + 8.60734272f) * x + 27.0927486f)) *
                    sqrt(1 + x);
}

template <int S> force_inline simd_fvec<S> portable_acosf(const simd_fvec<S> &x) {
    const simd_fvec<S> mask1 = (x < -0.62f);
    const simd_fvec<S> mask2 = (x <= 0.62f);

    simd_fvec<S> ret;

    where(mask1, ret) = acos_negative_tail(x);

    const simd_fvec<S> x2 = x * x;
    where(~mask1 & mask2, ret) =
        (PI / 2) - x -
        (0.0700945929f * x * ((x2 + 1.57144082f) * x2 + 1.25210774f)) * (x2 * ((x2 - 1.53757966f) * x2 + 1.89929986f));

    where(~mask1 & ~mask2, ret) = acos_positive_tail(x);

    return ret;
}

// "Stratified Sampling of Spherical Triangles"
// https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
// Based on https://www.shadertoy.com/view/4tGGzd
template <int S>
simd_fvec<S> SampleSphericalTriangle(const simd_fvec<S> P[3], const simd_fvec<S> p1[3], const simd_fvec<S> p2[3],
                                     const simd_fvec<S> p3[3], const simd_fvec<S> Xi[2], simd_fvec<S> out_dir[3]) {
    // setup spherical triangle
    simd_fvec<S> A[3], B[3], C[3];
    UNROLLED_FOR(i, 3, { A[i] = p1[i] - P[i]; })
    UNROLLED_FOR(i, 3, { B[i] = p2[i] - P[i]; })
    UNROLLED_FOR(i, 3, { C[i] = p3[i] - P[i]; })
    normalize(A);
    normalize(B);
    normalize(C);

    simd_fvec<S> BA[3], CA[3], AB[3], CB[3], BC[3], AC[3];
    // calculate internal angles of spherical triangle: alpha, beta and gamma
    for (int i = 0; i < 3; ++i) {
        BA[i] = B[i] - A[i];
        CA[i] = C[i] - A[i];
        AB[i] = A[i] - B[i];
        CB[i] = C[i] - B[i];
        BC[i] = B[i] - C[i];
        AC[i] = A[i] - C[i];
    }
    orthogonalize(A, BA, BA);
    orthogonalize(A, CA, CA);
    orthogonalize(B, AB, AB);
    orthogonalize(B, CB, CB);
    orthogonalize(C, BC, BC);
    orthogonalize(C, AC, AC);
    const simd_fvec<S> alpha = angle_between(BA, CA);
    const simd_fvec<S> beta = angle_between(AB, CB);
    const simd_fvec<S> gamma = angle_between(BC, AC);

    const simd_fvec<S> area = alpha + beta + gamma - PI;
    simd_ivec<S> mask = simd_cast(area > SPHERICAL_AREA_THRESHOLD);
    if (mask.all_zeros()) {
        return 0.0f;
    }

    if (out_dir) {
        // calculate arc lengths for edges of spherical triangle
        const simd_fvec<S> b = portable_acosf(clamp(dot3(C, A), -1.0f, 1.0f));
        const simd_fvec<S> c = portable_acosf(clamp(dot3(A, B), -1.0f, 1.0f));

        // Use one random variable to select the new area
        const simd_fvec<S> area_S = Xi[0] * area;

        // Save the sine and cosine of the angle delta
        const simd_fvec<S> p = sin(area_S - alpha);
        const simd_fvec<S> q = cos(area_S - alpha);

        // Compute the pair(u; v) that determines sin(beta_s) and cos(beta_s)
        const simd_fvec<S> u = q - cos(alpha);
        const simd_fvec<S> v = p + sin(alpha) * cos(c);

        // Compute the s coordinate as normalized arc length from A to C_s
        const simd_fvec<S> denom = ((v * p + u * q) * sin(alpha));
        const simd_fvec<S> s = safe_div(simd_fvec<S>{1.0f}, b) *
                               portable_acosf(clamp(safe_div(((v * q - u * p) * cos(alpha) - v), denom), -1.0f, 1.0f));

        // Compute the third vertex of the sub - triangle.
        simd_fvec<S> C_s[3];
        slerp(A, C, s, C_s);

        // Compute the t coordinate using C_s and Xi[1]
        const simd_fvec<S> denom2 = portable_acosf(clamp(dot3(C_s, B), -1.0f, 1.0f));
        const simd_fvec<S> t =
            safe_div(portable_acosf(clamp(1.0f - Xi[1] * (1.0f - dot3(C_s, B)), -1.0f, 1.0f)), denom2);

        // Construct the corresponding point on the sphere
        slerp(B, C_s, t, out_dir);
    }

    return select(mask, safe_div_pos(1.0f, area), simd_fvec<S>{0.0f});
}

// "An Area-Preserving Parametrization for Spherical Rectangles"
// https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
// NOTE: no precomputation is done, everything is calculated in-place
template <int S>
simd_fvec<S> SampleSphericalRectangle(const simd_fvec<S> P[3], const simd_fvec<S> light_pos[3],
                                      const simd_fvec<S> axis_u[3], const simd_fvec<S> axis_v[3],
                                      const simd_fvec<S> Xi[2], simd_fvec<S> out_p[3]) {
    simd_fvec<S> corner[3], x[3], y[3], z[3];
    UNROLLED_FOR(i, 3, {
        corner[i] = light_pos[i] - 0.5f * axis_u[i] - 0.5f * axis_v[i];
        x[i] = axis_u[i];
        y[i] = axis_v[i];
    })

    const simd_fvec<S> axisu_len = normalize(x), axisv_len = normalize(y);
    cross(x, y, z);

    // compute rectangle coords in local reference system
    simd_fvec<S> dir[3];
    UNROLLED_FOR(i, 3, { dir[i] = corner[i] - P[i]; })
    simd_fvec<S> z0 = dot3(dir, z);
    // flip z to make it point against Q
    UNROLLED_FOR(i, 3, { where(z0 > 0.0f, z[i]) = -z[i]; })
    where(z0 > 0.0f, z0) = -z0;

    const simd_fvec<S> x0 = dot3(dir, x);
    const simd_fvec<S> y0 = dot3(dir, y);
    const simd_fvec<S> x1 = x0 + axisu_len;
    const simd_fvec<S> y1 = y0 + axisv_len;

    // compute internal angles (gamma_i)
    simd_fvec<S> diff[4] = {x0 - x1, y1 - y0, x1 - x0, y0 - y1}, nz[4] = {y0, x1, y1, x0};
    UNROLLED_FOR(i, 4, {
        nz[i] *= diff[i];
        nz[i] /= sqrt(z0 * z0 * diff[i] * diff[i] + nz[i] * nz[i]);
    })
    const simd_fvec<S> g0 = portable_acosf(clamp(-nz[0] * nz[1], -1.0f, 1.0f));
    const simd_fvec<S> g1 = portable_acosf(clamp(-nz[1] * nz[2], -1.0f, 1.0f));
    const simd_fvec<S> g2 = portable_acosf(clamp(-nz[2] * nz[3], -1.0f, 1.0f));
    const simd_fvec<S> g3 = portable_acosf(clamp(-nz[3] * nz[0], -1.0f, 1.0f));
    // compute predefined constants
    const simd_fvec<S> b0 = nz[0];
    const simd_fvec<S> b1 = nz[2];
    const simd_fvec<S> b0sq = b0 * b0;
    const simd_fvec<S> k = 2 * PI - g2 - g3;
    // compute solid angle from internal angles
    const simd_fvec<S> area = g0 + g1 - k;
    const simd_ivec<S> mask = simd_cast(area > SPHERICAL_AREA_THRESHOLD);
    if (mask.all_zeros()) {
        return 0.0f;
    }

    if (out_p) {
        // compute cu
        const simd_fvec<S> au = Xi[0] * area + k;
        const simd_fvec<S> fu = safe_div((cos(au) * b0 - b1), sin(au));
        simd_fvec<S> cu = 1.0f / sqrt(fu * fu + b0sq);
        where(fu <= 0.0f, cu) = -cu;
        cu = clamp(cu, -1.0f, 1.0f);
        // compute xu
        simd_fvec<S> xu = -(cu * z0) / max(sqrt(1.0f - cu * cu), 1e-7f);
        xu = min(max(xu, x0), x1);
        // compute yv
        const simd_fvec<S> z0sq = z0 * z0;
        const simd_fvec<S> y0sq = y0 * y0;
        const simd_fvec<S> y1sq = y1 * y1;
        const simd_fvec<S> d = sqrt(xu * xu + z0sq);
        const simd_fvec<S> h0 = y0 / sqrt(d * d + y0sq);
        const simd_fvec<S> h1 = y1 / sqrt(d * d + y1sq);
        const simd_fvec<S> hv = h0 + Xi[1] * (h1 - h0), hv2 = hv * hv;
        simd_fvec<S> yv = y1;
        where(hv2 < 1.0f - 1e-6f, yv) = safe_div_pos(hv * d, sqrt(1.0f - hv2));

        // transform (xu, yv, z0) to world coords
        UNROLLED_FOR(i, 3, { out_p[i] = P[i] + xu * x[i] + yv * y[i] + z0 * z[i]; })
    }

    return select(mask, safe_div_pos(1.0f, area), simd_fvec<S>{0.0f});
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

force_inline void TransformPoint(const float p[3], const float *xform, float out_p[3]) {
    out_p[0] = xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2] + xform[12];
    out_p[1] = xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2] + xform[13];
    out_p[2] = xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2] + xform[14];
}

force_inline void TransformDirection(const float d[3], const float *xform, float out_d[3]) {
    out_d[0] = xform[0] * d[0] + xform[4] * d[1] + xform[8] * d[2];
    out_d[1] = xform[1] * d[0] + xform[5] * d[1] + xform[9] * d[2];
    out_d[2] = xform[2] * d[0] + xform[6] * d[1] + xform[10] * d[2];
}

template <int S> force_inline simd_fvec<S> pow5(const simd_fvec<S> &v) { return (v * v) * (v * v) * v; }

template <int S> simd_ivec<S> get_ray_hash(const ray_data_t<S> &r, const float root_min[3], const float cell_size[3]) {
    simd_ivec<S> x = clamp(simd_ivec<S>((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
                 y = clamp(simd_ivec<S>((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
                 z = clamp(simd_ivec<S>((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    simd_ivec<S> omega_index = clamp(simd_ivec<S>((1.0f + r.d[2]) / omega_step), 0, 32),
                 phi_index_i = clamp(simd_ivec<S>((1.0f + r.d[1]) / phi_step), 0, 16),
                 phi_index_j = clamp(simd_ivec<S>((1.0f + r.d[0]) / phi_step), 0, 16);

    simd_ivec<S> o, p;

    UNROLLED_FOR_S(i, S, {
        if (r.mask[i]) {
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
    })

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
        out_col[i] = select(in_col[i] > 0.04045f, pow((in_col[i] + 0.055f) / 1.055f, 2.4f), in_col[i] / 12.92f);
    })
    out_col[3] = in_col[3];
}

template <int S> force_inline void YCoCg_to_RGB(const simd_fvec<S> in_col[4], simd_fvec<S> out_col[3]) {
    const simd_fvec<S> scale = (in_col[2] * (255.0f / 8.0f)) + 1.0f;
    const simd_fvec<S> Y = in_col[3];
    const simd_fvec<S> Co = (in_col[0] - (0.5f * 256.0f / 255.0f)) / scale;
    const simd_fvec<S> Cg = (in_col[1] - (0.5f * 256.0f / 255.0f)) / scale;

    out_col[0] = saturate(Y + Co - Cg);
    out_col[1] = saturate(Y + Cg);
    out_col[2] = saturate(Y - Co - Cg);
}

template <int S>
simd_fvec<S> get_texture_lod(const Cpu::TexStorageBase *textures[], const uint32_t index, const simd_fvec<S> duv_dx[2],
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
simd_fvec<S> get_texture_lod(const Cpu::TexStorageBase *const textures[], const uint32_t index,
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
void FetchTransformAndRecalcBasis(const mesh_instance_t *sc_mesh_instances, const simd_ivec<S> &mi_index,
                                  const simd_fvec<S> P_ls[3], simd_fvec<S> inout_plane_N[3], simd_fvec<S> inout_N[3],
                                  simd_fvec<S> inout_B[3], simd_fvec<S> inout_T[3], simd_fvec<S> inout_tangent[3],
                                  simd_fvec<S> inout_ro_ls[3], simd_fvec<S> out_transform[16]) {
    const float *transforms = &sc_mesh_instances[0].xform[0];
    const float *inv_transforms = &sc_mesh_instances[0].inv_xform[0];
    const int MeshInstancesStride = sizeof(mesh_instance_t) / sizeof(float);

    simd_fvec<S> inv_transform[16];
    UNROLLED_FOR(i, 16, {
        out_transform[i] = gather(transforms + i, mi_index * MeshInstancesStride);
        inv_transform[i] = gather(inv_transforms + i, mi_index * MeshInstancesStride);
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
    TransformPoint(inv_transform, inout_ro_ls);
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
        const simd_fvec<S> Nz2 = select(valid1, N1_z2, N2_z2);

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
    UNROLLED_FOR_S(i, S, { ret.template set<i>(cosf(v.template get<i>())); })
    return ret;
}

template <int S> force_inline simd_fvec<S> sin(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, { ret.template set<i>(sinf(v.template get<i>())); })
    return ret;
}

template <int S>
force_inline void calc_alpha(const simd_fvec<S> &roughness, const simd_fvec<S> &anisotropy,
                             const simd_fvec<S> &regularize_alpha, simd_fvec<S> out_alpha[2]) {
    const simd_fvec<S> roughness2 = sqr(roughness);
    const simd_fvec<S> aspect = sqrt(1.0f - 0.9f * anisotropy);

    out_alpha[0] = (roughness2 / aspect);
    out_alpha[1] = (roughness2 * aspect);

    where(out_alpha[0] < regularize_alpha, out_alpha[0]) =
        clamp(2 * out_alpha[0], 0.25f * regularize_alpha, regularize_alpha);
    where(out_alpha[1] < regularize_alpha, out_alpha[1]) =
        clamp(2 * out_alpha[1], 0.25f * regularize_alpha, regularize_alpha);
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

// http://jcgt.org/published/0007/04/01/paper.pdf
template <int S>
void SampleVNDF_Hemisphere_CrossSect(const simd_fvec<S> Vh[3], const simd_fvec<S> &U1, const simd_fvec<S> &U2,
                                     simd_fvec<S> out_Nh[3]) {
    // orthonormal basis (with special case if cross product is zero)
    const simd_fvec<S> lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];

    simd_fvec<S> T1[3] = {{1.0f}, {0.0f}, {0.0f}};
    const simd_fvec<S> denom = safe_sqrt(lensq);
    where(lensq > 0.0f, T1[0]) = -safe_div_pos(Vh[1], denom);
    where(lensq > 0.0f, T1[1]) = safe_div_pos(Vh[0], denom);

    simd_fvec<S> T2[3];
    cross(Vh, T1, T2);
    // parameterization of the projected area
    const simd_fvec<S> r = sqrt(U1);
    const simd_fvec<S> phi = 2.0f * PI * U2;
    simd_fvec<S> t1;
    UNROLLED_FOR_S(i, S, { t1.template set<i>(r.template get<i>() * cosf(phi.template get<i>())); })
    simd_fvec<S> t2;
    UNROLLED_FOR_S(i, S, { t2.template set<i>(r.template get<i>() * sinf(phi.template get<i>())); })
    const simd_fvec<S> s = 0.5f * (1.0f + Vh[2]);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    // reprojection onto hemisphere
    UNROLLED_FOR(i, 3, { out_Nh[i] = t1 * T1[i] + t2 * T2[i] + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh[i]; })
}

// https://arxiv.org/pdf/2306.05044.pdf
template <int S>
void SampleVNDF_Hemisphere_SphCap(const simd_fvec<S> Vh[3], const simd_fvec<S> rand[2], simd_fvec<S> out_Nh[3]) {
    const simd_fvec<S> phi = 2.0f * PI * rand[0];
    const simd_fvec<S> z = fmadd(1.0f - rand[1], 1.0f + Vh[2], -Vh[2]);
    const simd_fvec<S> sin_theta = sqrt(saturate(1.0f - z * z));
    out_Nh[0] = Vh[0] + sin_theta * cos(phi);
    out_Nh[1] = Vh[1] + sin_theta * sin(phi);
    out_Nh[2] = Vh[2] + z;
}

// https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
template <int S>
void SampleVNDF_Hemisphere_SphCap_Bounded(const simd_fvec<S> Ve[3], const simd_fvec<S> Vh[3],
                                          const simd_fvec<S> alpha[2], const simd_fvec<S> rand[2],
                                          simd_fvec<S> out_Nh[3]) {
    // sample a spherical cap in (-Vh.z, 1]
    const simd_fvec<S> phi = 2.0f * PI * rand[0];
    const simd_fvec<S> a = saturate(min(alpha[0], alpha[1]));
    const simd_fvec<S> s = 1.0f + sqrt(Ve[0] * Ve[0] + Ve[1] * Ve[1]);
    const simd_fvec<S> a2 = a * a, s2 = s * s;
    const simd_fvec<S> k = (1.0f - a2) * s2 / (s2 + a2 * Ve[2] * Ve[2]);
    const simd_fvec<S> b = select(Ve[2] > 0.0f, k * Vh[2], Vh[2]);
    const simd_fvec<S> z = fmadd(1.0f - rand[1], 1.0f + b, -b);
    const simd_fvec<S> sin_theta = sqrt(saturate(1.0f - z * z));
    const simd_fvec<S> x = sin_theta * cos(phi);
    const simd_fvec<S> y = sin_theta * sin(phi);
    out_Nh[0] = x + Vh[0];
    out_Nh[1] = y + Vh[1];
    out_Nh[2] = z + Vh[2];
}

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
template <int S>
void SampleGGX_VNDF(const simd_fvec<S> Ve[3], const simd_fvec<S> alpha[2], const simd_fvec<S> rand[2],
                    simd_fvec<S> out_V[3]) {
    // transforming the view direction to the hemisphere configuration
    simd_fvec<S> Vh[3] = {alpha[0] * Ve[0], alpha[1] * Ve[1], Ve[2]};
    safe_normalize(Vh);
    // sample the hemisphere
    simd_fvec<S> Nh[3];
    SampleVNDF_Hemisphere_SphCap(Vh, rand, Nh);
    // transforming the normal back to the ellipsoid configuration
    out_V[0] = alpha[0] * Nh[0];
    out_V[1] = alpha[1] * Nh[1];
    out_V[2] = max(0.0f, Nh[2]);
    safe_normalize(out_V);
}

template <int S>
void SampleGGX_VNDF_Bounded(const simd_fvec<S> Ve[3], const simd_fvec<S> alpha[2], const simd_fvec<S> rand[2],
                            simd_fvec<S> out_V[3]) {
    // transforming the view direction to the hemisphere configuration
    simd_fvec<S> Vh[3] = {alpha[0] * Ve[0], alpha[1] * Ve[1], Ve[2]};
    safe_normalize(Vh);
    // sample the hemisphere
    simd_fvec<S> Nh[3];
    SampleVNDF_Hemisphere_SphCap_Bounded(Ve, Vh, alpha, rand, Nh);
    // transforming the normal back to the ellipsoid configuration
    out_V[0] = alpha[0] * Nh[0];
    out_V[1] = alpha[1] * Nh[1];
    out_V[2] = max(0.0f, Nh[2]);
    safe_normalize(out_V);
}

template <int S>
simd_fvec<S> GGX_VNDF_Reflection_Bounded_PDF(const simd_fvec<S> &D, const simd_fvec<S> view_dir_ts[3],
                                             const simd_fvec<S> alpha[2]) {
    const simd_fvec<S> ai[2] = {alpha[0] * view_dir_ts[0], alpha[1] * view_dir_ts[1]};
    const simd_fvec<S> len2 = ai[0] * ai[0] + ai[1] * ai[1];
    const simd_fvec<S> t = sqrt(len2 + view_dir_ts[2] * view_dir_ts[2]);

    simd_fvec<S> ret = D * safe_div_pos(t - view_dir_ts[2], 2.0f * len2);

    const simd_fvec<S> a = saturate(min(alpha[0], alpha[1]));
    const simd_fvec<S> s = 1.0f + sqrt(view_dir_ts[0] * view_dir_ts[0] + view_dir_ts[1] * view_dir_ts[1]);
    const simd_fvec<S> a2 = a * a, s2 = s * s;
    const simd_fvec<S> k = (1.0f - a2) * s2 / (s2 + a2 * view_dir_ts[2] * view_dir_ts[2]);
    where(view_dir_ts[2] >= 0.0f, ret) = safe_div(D, 2.0f * (k * view_dir_ts[2] + t));

    return ret;
}

// Smith shadowing function
template <int S> force_inline simd_fvec<S> G1(const simd_fvec<S> Ve[3], simd_fvec<S> alpha_x, simd_fvec<S> alpha_y) {
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    const simd_fvec<S> delta =
        (-1.0f + safe_sqrt(1.0f + safe_div_pos(alpha_x * Ve[0] * Ve[0] + alpha_y * Ve[1] * Ve[1], Ve[2] * Ve[2]))) /
        2.0f;
    return 1.0f / (1.0f + delta);
}

template <int S> simd_fvec<S> D_GTR1(const simd_fvec<S> &NDotH, const simd_fvec<S> &a) {
    const simd_fvec<S> a2 = sqr(a);
    const simd_fvec<S> t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return select(a < 1.0f, safe_div(a2 - 1.0f, PI * log(a2) * t), simd_fvec<S>{1.0f / PI});
}

template <int S> simd_fvec<S> D_GGX(const simd_fvec<S> H[3], const simd_fvec<S> &alpha_x, const simd_fvec<S> &alpha_y) {
    const simd_fvec<S> sx = -safe_div(H[0], H[2] * alpha_x);
    const simd_fvec<S> sy = -safe_div(H[1], H[2] * alpha_y);
    const simd_fvec<S> s1 = 1.0f + sx * sx + sy * sy;
    const simd_fvec<S> cos_theta_h4 = H[2] * H[2] * H[2] * H[2];
    return select(H[2] != 0.0f, safe_inv_pos((s1 * s1) * PI * alpha_x * alpha_y * cos_theta_h4), simd_fvec<S>{0.0f});
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
void map_to_cone(const simd_fvec<S> &r1, const simd_fvec<S> &r2, const simd_fvec<S> N[3], float radius,
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

template <int S>
force_inline simd_fvec<S> sphere_intersection(const float center[3], const float radius, const simd_fvec<S> ro[3],
                                              const simd_fvec<S> rd[3]) {
    const simd_fvec<S> oc[3] = {ro[0] - center[0], ro[1] - center[1], ro[2] - center[2]};
    const simd_fvec<S> a = dot3(rd, rd);
    const simd_fvec<S> b = 2 * dot3(oc, rd);
    const simd_fvec<S> c = dot3(oc, oc) - radius * radius;
    const simd_fvec<S> discriminant = b * b - 4 * a * c;
    return (-b - sqrt(max(discriminant, 0.0f))) / (2 * a);
}

template <int S> force_inline simd_fvec<S> schlick_weight(const simd_fvec<S> &u) {
    const simd_fvec<S> m = saturate(1.0f - u);
    return pow5(m);
}

force_inline float fresnel_dielectric_cos(float cosi, float eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    float result;

    if (g > 0) {
        g = sqrtf(g);
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

    g = safe_sqrt(g);
    const simd_fvec<S> A = safe_div(g - c, g + c);
    const simd_fvec<S> B = safe_div(c * (g + c) - 1, c * (g - c) + 1);

    return select(mask, 0.5f * A * A * (1 + B * B), simd_fvec<S>{1.0f} /* TIR (no refracted component) */);
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
    const simd_fvec<S> q = select(b < 0.0f, -0.5f * (b - sqrt_d), -0.5f * (b + sqrt_d));
    t0 = safe_div(q, a);
    t1 = safe_div(c, q);
    return simd_cast(d >= 0.0f);
}

template <int S> force_inline simd_fvec<S> ngon_rad(const simd_fvec<S> &theta, const float n) {
    simd_fvec<S> ret;
    UNROLLED_FOR_S(i, S, {
        ret.template set<i>(
            cosf(PI / n) /
            cosf(theta.template get<i>() - (2.0f * PI / n) * floorf((n * theta.template get<i>() + PI) / (2.0f * PI))));
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
    mask &= ~std::exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[3];
    simd_fvec<S> active_lanes = ~mask;
    // 2
    mask = active_lanes & (stack[2] > 0.0f);
    mask &= ~std::exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[2];
    active_lanes &= ~mask;
    // 1
    mask = active_lanes & (stack[1] > 0.0f);
    mask &= ~std::exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[1];
    active_lanes &= ~mask;
    // 0
    mask = active_lanes & (stack[0] > 0.0f);
    mask &= ~std::exchange(skip_first, skip_first & ~mask);
    where(mask, ret) = stack[0];

    return ret;
}

template <int S> simd_fvec<S> approx_atan2(const simd_fvec<S> &y, const simd_fvec<S> &x) {
    simd_fvec<S> t0, t1, t3, t4;

    t3 = abs(x);
    t1 = abs(y);
    t0 = max(t3, t1);
    t1 = min(t3, t1);
    t3 = safe_inv(t0);
    t3 = t1 * t3;

    t4 = t3 * t3;
    t0 = -0.013480470f;
    t0 = t0 * t4 + 0.057477314f;
    t0 = t0 * t4 - 0.121239071f;
    t0 = t0 * t4 + 0.195635925f;
    t0 = t0 * t4 - 0.332994597f;
    t0 = t0 * t4 + 0.999995630f;
    t3 = t0 * t3;

    where(abs(y) > abs(x), t3) = 1.570796327f - t3;
    where(x < 0.0f, t3) = 3.141592654f - t3;
    where(y < 0.0f, t3) = -t3;

    return t3;
}

template <int S> force_inline simd_fvec<S> approx_cos(simd_fvec<S> x) { // max error is 0.056010f
    const float tp = 1.0f / (2.0f * PI);
    x *= tp;
    x -= 0.25f + floor(x + 0.25f);
    x *= 16.0f * (abs(x) - 0.5f);
    return x;
}

template <int S> force_inline simd_fvec<S> approx_acos(simd_fvec<S> x) { // max error is 0.000068f
    const simd_fvec<S> negate = select(x < 0.0f, simd_fvec<S>{1.0f}, simd_fvec<S>{0.0f});
    x = abs(x);
    simd_fvec<S> ret = -0.0187293f;
    ret = ret * x;
    ret = ret + 0.0742610f;
    ret = ret * x;
    ret = ret - 0.2121144f;
    ret = ret * x;
    ret = ret + 1.5707288f;
    ret = ret * sqrt(1.0f - min(x, 1.0f));
    ret = ret - 2 * negate * ret;
    return negate * PI + ret;
}

template <int S> void calc_lnode_importance(const light_wbvh_node_t &n, const float P[3], float importance[8]) {
    for (int i = 0; i < 8; i += S) {
        simd_fvec<S> mul = 1.0f, v_len2 = 1.0f;

        const simd_ivec<S> mask = simd_cast(simd_fvec<S>{&n.bbox_min[0][i], simd_mem_aligned} > -MAX_DIST);
        if (mask.not_all_zeros()) {
            simd_fvec<S> v[3] = {P[0] - 0.5f * (simd_fvec<S>{&n.bbox_min[0][i], simd_mem_aligned} +
                                                simd_fvec<S>{&n.bbox_max[0][i], simd_mem_aligned}),
                                 P[1] - 0.5f * (simd_fvec<S>{&n.bbox_min[1][i], simd_mem_aligned} +
                                                simd_fvec<S>{&n.bbox_max[1][i], simd_mem_aligned}),
                                 P[2] - 0.5f * (simd_fvec<S>{&n.bbox_min[2][i], simd_mem_aligned} +
                                                simd_fvec<S>{&n.bbox_max[2][i], simd_mem_aligned})};
            const simd_fvec<S> ext[3] = {
                simd_fvec<S>{&n.bbox_max[0][i], simd_mem_aligned} - simd_fvec<S>{&n.bbox_min[0][i], simd_mem_aligned},
                simd_fvec<S>{&n.bbox_max[1][i], simd_mem_aligned} - simd_fvec<S>{&n.bbox_min[1][i], simd_mem_aligned},
                simd_fvec<S>{&n.bbox_max[2][i], simd_mem_aligned} - simd_fvec<S>{&n.bbox_min[2][i], simd_mem_aligned}};

            const simd_fvec<S> extent = 0.5f * length(ext);
            where(mask, v_len2) = length2(v);
            const simd_fvec<S> v_len = sqrt(v_len2);
            const simd_fvec<S> omega_u = approx_atan2(extent, v_len) + 0.000005f;

            const simd_fvec<S> axis[3] = {simd_fvec<S>{&n.axis[0][i], simd_mem_aligned},
                                          simd_fvec<S>{&n.axis[1][i], simd_mem_aligned},
                                          simd_fvec<S>{&n.axis[2][i], simd_mem_aligned}};

            UNROLLED_FOR(j, 3, { v[j] /= v_len; })
            const simd_fvec<S> omega = approx_acos(min(dot3(axis, v), 1.0f)) - 0.00007f;
            const simd_fvec<S> omega_ = max(0.0f, omega - simd_fvec<S>{&n.omega_n[i], simd_mem_aligned} - omega_u);
            where(mask, mul) = 0.0f;
            where(mask & simd_cast(omega_ < simd_fvec<S>{&n.omega_e[i], simd_mem_aligned}), mul) =
                approx_cos(omega_) + 0.057f;
        }

        const simd_fvec<S> imp = simd_fvec<S>{&n.flux[i], simd_mem_aligned} * mul / v_len2;
        imp.store_to(&importance[i], simd_mem_aligned);
    }
}

template <int S>
void calc_lnode_importance(const light_wbvh_node_t &n, const simd_fvec<S> P[3], simd_fvec<S> importance[8]) {
    for (int i = 0; i < 8; ++i) {
        simd_fvec<S> mul = 1.0f, v_len2 = 1.0f;

        if (n.bbox_min[0][i] > -MAX_DIST) {
            simd_fvec<S> v[3] = {P[0] - 0.5f * (n.bbox_min[0][i] + n.bbox_max[0][i]),
                                 P[1] - 0.5f * (n.bbox_min[1][i] + n.bbox_max[1][i]),
                                 P[2] - 0.5f * (n.bbox_min[2][i] + n.bbox_max[2][i])};

            const float ext[3] = {n.bbox_max[0][i] - n.bbox_min[0][i], n.bbox_max[1][i] - n.bbox_min[1][i],
                                  n.bbox_max[2][i] - n.bbox_min[2][i]};
            const float extent = 0.5f * sqrtf(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]);

            v_len2 = length2(v);
            const simd_fvec<S> v_len = sqrt(v_len2);
            const simd_fvec<S> omega_u = approx_atan2(simd_fvec<S>{extent}, v_len) + 0.000005f;

            const float axis[3] = {n.axis[0][i], n.axis[1][i], n.axis[2][i]};

            UNROLLED_FOR(j, 3, { v[j] = safe_div_pos(v[j], v_len); })
            const simd_fvec<S> omega = approx_acos(min(dot3(axis, v), 1.0f)) - 0.00007f;
            const simd_fvec<S> omega_ = max(0.0f, omega - n.omega_n[i] - omega_u);
            mul = select(omega_ < n.omega_e[i], approx_cos(omega_) + 0.057f, simd_fvec<S>{0.0f});
        }

        importance[i] = safe_div_pos(n.flux[i] * mul, v_len2);
    }
}

} // namespace NS
} // namespace Ray

template <int DimX, int DimY>
void Ray::NS::GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, const uint32_t rand_seq[],
                                  const uint32_t rand_seed, const float filter_table[], const int iteration,
                                  const uint16_t required_samples[], aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                                  aligned_vector<hit_data_t<DimX * DimY>> &out_inters) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    const float k = float(w) / float(h);

    const float temp = tanf(0.5f * cam.fov * PI / 180.0f);
    const float fov_k = temp * cam.focus_distance;
    const float spread_angle = atanf(2.0f * temp / float(h));

    const auto off_x = simd_ivec<S>{rays_layout_x, simd_mem_aligned},
               off_y = simd_ivec<S>{rays_layout_y, simd_mem_aligned};

    const int x_res = (r.w + DimX - 1) / DimX, y_res = (r.h + DimY - 1) / DimY;

    size_t i = 0;
    out_rays.resize(x_res * y_res);
    out_inters.resize(x_res * y_res);

    for (int y = r.y; y < r.y + r.h; y += DimY) {
        for (int x = r.x; x < r.x + r.w; x += DimX) {
            ray_data_t<S> &out_r = out_rays[i];

            const simd_ivec<S> ixx = x + off_x, iyy = y + off_y;
            const simd_ivec<S> ixx_clamped = min(ixx, w - 1), iyy_clamped = min(iyy, h - 1);

            simd_ivec<S> req_samples;
            UNROLLED_FOR_S(i, S, {
                req_samples.template set<i>(
                    required_samples[iyy_clamped.template get<i>() * w + ixx_clamped.template get<i>()]);
            })

            out_r.mask = (ixx < w) & (iyy < h) & (req_samples >= iteration);

            auto fxx = simd_fvec<S>(ixx), fyy = simd_fvec<S>(iyy);

            const simd_uvec<S> px_hash = hash(simd_uvec<S>((ixx << 16) | iyy));
            const simd_uvec<S> rand_hash = hash_combine(px_hash, rand_seed);

            simd_fvec<S> filter_rand[2];
            get_scrambled_2d_rand(simd_uvec<S>(uint32_t(RAND_DIM_FILTER)), rand_hash, iteration - 1, rand_seq,
                                  filter_rand);

            if (cam.filter != ePixelFilter::Box) {
                filter_rand[0] *= float(FILTER_TABLE_SIZE - 1);
                filter_rand[1] *= float(FILTER_TABLE_SIZE - 1);

                const simd_ivec<S> index_x = min(simd_ivec<S>(filter_rand[0]), FILTER_TABLE_SIZE - 1),
                                   index_y = min(simd_ivec<S>(filter_rand[1]), FILTER_TABLE_SIZE - 1);

                const simd_ivec<S> nindex_x = min(index_x + 1, FILTER_TABLE_SIZE - 1),
                                   nindex_y = min(index_y + 1, FILTER_TABLE_SIZE - 1);

                const simd_fvec<S> tx = filter_rand[0] - simd_fvec<S>(index_x),
                                   ty = filter_rand[1] - simd_fvec<S>(index_y);

                const simd_fvec<S> data0_x = gather(filter_table, index_x), data1_x = gather(filter_table, nindex_x);
                const simd_fvec<S> data0_y = gather(filter_table, index_y), data1_y = gather(filter_table, nindex_y);

                filter_rand[0] = (1.0f - tx) * data0_x + tx * data1_x;
                filter_rand[1] = (1.0f - ty) * data0_y + ty * data1_y;
            }

            fxx += filter_rand[0];
            fyy += filter_rand[1];

            simd_fvec<S> offset[2] = {0.0f, 0.0f};
            if (cam.fstop > 0.0f) {
                simd_fvec<S> lens_rand[2];
                get_scrambled_2d_rand(simd_uvec<S>(uint32_t(RAND_DIM_LENS)), rand_hash, iteration - 1, rand_seq,
                                      lens_rand);

                offset[0] = 2.0f * lens_rand[0] - 1.0f;
                offset[1] = 2.0f * lens_rand[1] - 1.0f;

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
            out_r.xy = simd_uvec<S>((ixx << 16) | iyy);
            out_r.depth = pack_ray_type(RAY_TYPE_CAMERA);
            out_r.depth |= pack_depth(simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0});

            hit_data_t<S> &out_i = out_inters[i++];
            out_i = {};
            out_i.t = (cam.clip_end / dot3(_d, cam.fwd)) - clip_start;
        }
    }

    out_rays.resize(i);
    out_inters.resize(i);
}

template <int DimX, int DimY>
void Ray::NS::SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                       const mesh_instance_t &mi, const uint32_t *vtx_indices, const vertex_t *vertices,
                                       const rect_t &r, int width, int height, const uint32_t rand_seq[],
                                       aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                                       aligned_vector<hit_data_t<DimX * DimY>> &out_inters) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    out_rays.resize(r.w * r.h / S + ((r.w * r.h) % S != 0));
    out_inters.resize(out_rays.size());

    const auto off_x = simd_ivec<S>{rays_layout_x}, off_y = simd_ivec<S>{rays_layout_y};

    size_t count = 0;
    for (int y = r.y; y < r.y + r.h - (r.h & (DimY - 1)); y += DimY) {
        for (int x = r.x; x < r.x + r.w - (r.w & (DimX - 1)); x += DimX) {
            const simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

            ray_data_t<S> &out_ray = out_rays[count];
            hit_data_t<S> &out_inter = out_inters[count];
            count++;

            out_ray.xy = simd_uvec<S>((ixx << 16) | iyy);
            out_ray.c[0] = out_ray.c[1] = out_ray.c[2] = 1.0f;
            out_ray.cone_width = 0.0f;
            out_ray.cone_spread = 0.0f;
            out_inter.v = -1.0f;
        }
    }

    const simd_ivec4 irect_min = {r.x, r.y, 0, 0}, irect_max = {r.x + r.w - 1, r.y + r.h - 1, 0, 0};
    const simd_fvec4 size = {float(width), float(height), 0.0f, 0.0f};

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const vertex_t &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const vertex_t &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const vertex_t &v2 = vertices[vtx_indices[tri * 3 + 2]];

        // TODO: use uv_layer
        const auto t0 = simd_fvec4{v0.t[0], 1.0f - v0.t[1], 0.0f, 0.0f} * size;
        const auto t1 = simd_fvec4{v1.t[0], 1.0f - v1.t[1], 0.0f, 0.0f} * size;
        const auto t2 = simd_fvec4{v2.t[0], 1.0f - v2.t[1], 0.0f, 0.0f} * size;

        simd_fvec4 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        simd_ivec4 ibbox_min = simd_ivec4(bbox_min),
                   ibbox_max = simd_ivec4{int(roundf(bbox_max[0])), int(roundf(bbox_max[1])), 0, 0};

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

                // NOTE: temporarily broken
                simd_fvec<S> rxx = 0.0f;
                simd_fvec<S> ryy = 0.0f;

                // UNROLLED_FOR_S(i, S, {
                //     float _unused;
                //     rxx.template set<i>(modff(rand_seq[RAND_DIM_FILTER_U] + rxx.template get<i>(), &_unused));
                //     ryy.template set<i>(modff(rand_seq[RAND_DIM_FILTER_V] + ryy.template get<i>(), &_unused));
                // })

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

                    TransformPoint(_p, mi.xform, p);
                    TransformNormal(_n, mi.inv_xform, n);

                    UNROLLED_FOR(i, 3, { where(fmask, out_ray.o[i]) = p[i] + n[i]; })
                    UNROLLED_FOR(i, 3, { where(fmask, out_ray.d[i]) = -n[i]; })
                    // where(fmask, out_ray.ior) = 1.0f;
                    where(fmask, out_ray.depth) = pack_ray_type(RAY_TYPE_DIFFUSE);
                    where(fmask, out_ray.depth) |=
                        pack_depth(simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0});

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
int Ray::NS::SortRays_CPU(Span<ray_data_t<S>> rays, const float root_min[3], const float cell_size[3],
                          simd_ivec<S> *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                          ray_chunk_t *chunks_temp) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]
    int rays_count = int(rays.size());

    // compute ray hash values
    for (int i = 0; i < rays_count; i++) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
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

                    swap_elements(rays[jj].mask, _jj, rays[kk].mask, _kk);

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
                }

                std::swap(flat_hash_values[i], flat_hash_values[j]);
                std::swap(scan_values[i], scan_values[j]);
            }
        }
    }

    // remove non-active rays
    while (rays_count && rays[rays_count - 1].mask.all_zeros()) {
        rays_count--;
    }

    return rays_count;
}

template <int S>
int Ray::NS::SortRays_GPU(Span<ray_data_t<S>> rays, const float root_min[3], const float cell_size[3],
                          simd_ivec<S> *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks,
                          ray_chunk_t *chunks_temp, uint32_t *skeleton) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]
    int rays_count = int(rays.size());

    // compute ray hash values
    for (int i = 0; i < rays_count; i++) {
        hash_values[i] = get_ray_hash(rays[i], root_min, cell_size);
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
    chunks[chunks_count - 1].size = uint32_t(rays_count) * S - chunks[chunks_count - 1].base;

    radix_sort(chunks, chunks + chunks_count, chunks_temp);

    { // perform exclusive scan on chunks size
        uint32_t cur_sum = 0;
        for (int i = 0; i < chunks_count; i++) {
            scan_values[i] = cur_sum;
            cur_sum += chunks[i].size;
        }
    }

    std::fill(skeleton, skeleton + rays_count * S, 1);
    memset(head_flags, 0, rays_count * S);

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

                    swap_elements(rays[jj].mask, _jj, rays[kk].mask, _kk);

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
                }

                std::swap(scan_values[i], scan_values[j]);
            }
        }
    }

    // remove non-active rays
    while (rays_count && rays[rays_count - 1].mask.all_zeros()) {
        rays_count--;
    }

    return rays_count;
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                       const tri_accel_t *tris, uint32_t num_tris, int obj_index,
                                       hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = {Uninitialize};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    const simd_ivec<S> inter_mask = simd_cast(inter.v >= 0.0f);

    where(inter_mask, out_inter.obj_index) = inter.obj_index;
    where(inter_mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter_mask, out_inter.u) = inter.u;
    where(inter_mask, out_inter.v) = inter.v;

    return inter_mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                       const tri_accel_t *tris, const int tri_start, const int tri_end,
                                       const int obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter{Uninitialize};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    const simd_ivec<S> inter_mask = simd_cast(inter.v >= 0.0f);

    where(inter_mask, out_inter.obj_index) = inter.obj_index;
    where(inter_mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter_mask, out_inter.u) = inter.u;
    where(inter_mask, out_inter.v) = inter.v;

    return inter_mask.not_all_zeros();
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
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    const simd_ivec<S> inter_mask = simd_cast(inter.v >= 0.0f);

    where(inter_mask, out_inter.obj_index) = inter.obj_index;
    where(inter_mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter_mask, out_inter.u) = inter.u;
    where(inter_mask, out_inter.v) = inter.v;

    return inter_mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const simd_ivec<S> &ray_mask,
                                   const tri_accel_t *tris, const int tri_start, const int tri_end, const int obj_index,
                                   hit_data_t<S> &out_inter) {
    hit_data_t<S> inter{Uninitialize};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;
    inter.v = -1.0f;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, ray_mask, tris[i], i, inter);
    }

    const simd_ivec<S> inter_mask = simd_cast(inter.v >= 0.0f);

    where(inter_mask, out_inter.obj_index) = inter.obj_index;
    where(inter_mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(inter_mask, out_inter.u) = inter.u;
    where(inter_mask, out_inter.v) = inter.v;

    return inter_mask.not_all_zeros();
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
bool Ray::NS::Traverse_TLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                 const simd_uvec<S> &ray_flags, const simd_ivec<S> &ray_mask,
                                                 const bvh_node_t *nodes, uint32_t node_index,
                                                 const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                                 const mesh_t *meshes, const tri_accel_t *tris,
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

                    simd_ivec<S> bbox_mask = simd_ivec<S>((mi.ray_visibility & ray_flags) != 0u) &
                                             bbox_test_fma(inv_d, inv_d_o, inter.t, mi.bbox_min, mi.bbox_max) &
                                             st.queue[st.index].mask;
                    if (bbox_mask.all_zeros()) {
                        continue;
                    }

                    simd_fvec<S> _ro[3], _rd[3];
                    TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

                    res |= Traverse_BLAS_WithStack_ClosestHit(_ro, _rd, bbox_mask, nodes, m.node_index, tris,
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
bool Ray::NS::Traverse_TLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                 const simd_uvec<S> &ray_flags, const simd_ivec<S> &ray_mask,
                                                 const wbvh_node_t *nodes, uint32_t node_index,
                                                 const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                                 const mesh_t *meshes, const mtri_accel_t *mtris,
                                                 const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    alignas(S * 4) float _ro[3][S], _rd[3][S];
    ro[0].store_to(_ro[0], simd_mem_aligned);
    ro[1].store_to(_ro[1], simd_mem_aligned);
    ro[2].store_to(_ro[2], simd_mem_aligned);
    rd[0].store_to(_rd[0], simd_mem_aligned);
    rd[1].store_to(_rd[1], simd_mem_aligned);
    rd[2].store_to(_rd[2], simd_mem_aligned);

    alignas(S * 4) int ray_masks[S], inter_prim_index[S], inter_obj_index[S];
    alignas(S * 4) float inter_t[S], inter_u[S], inter_v[S];
    ray_mask.store_to(ray_masks, simd_mem_aligned);
    inter.prim_index.store_to(inter_prim_index, simd_mem_aligned);
    inter.obj_index.store_to(inter_obj_index, simd_mem_aligned);
    inter.t.store_to(inter_t, simd_mem_aligned);
    inter.u.store_to(inter_u, simd_mem_aligned);
    inter.v.store_to(inter_v, simd_mem_aligned);

    alignas(S * 4) unsigned _ray_flags[S];
    ray_flags.store_to(_ray_flags, simd_mem_aligned);

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
                    if ((mi.ray_visibility & _ray_flags[ri]) == 0 ||
                        !bbox_test(_inv_d, _inv_d_o, inter_t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    const mesh_t &m = meshes[mi.mesh_index];

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, mi.inv_xform, tr_ro, tr_rd);

                    const bool lres = Traverse_BLAS_WithStack_ClosestHit<S>(tr_ro, tr_rd, nodes, m.node_index, mtris,
                                                                            tri_indices, inter_prim_index[ri],
                                                                            inter_t[ri], inter_u[ri], inter_v[ri]);
                    if (lres) {
                        inter_obj_index[ri] = int(mi_indices[j]);
                    }
                    res |= lres;
                }
            }
        }
    }

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
Ray::NS::simd_ivec<S>
Ray::NS::Traverse_TLAS_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], int ray_type,
                                        const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                        const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                        const mesh_t *meshes, const tri_accel_t *tris, const tri_mat_data_t *materials,
                                        const uint32_t *tri_indices, hit_data_t<S> &inter) {
    const int ray_vismask = (1u << ray_type);

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
                    if ((mi.ray_visibility & ray_vismask) == 0) {
                        continue;
                    }

                    const mesh_t &m = meshes[mi.mesh_index];

                    const simd_ivec<S> bbox_mask =
                        bbox_test_fma(inv_d, inv_d_o, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                    if (bbox_mask.all_zeros()) {
                        continue;
                    }

                    simd_fvec<S> _ro[3], _rd[3];
                    TransformRay(ro, rd, mi.inv_xform, _ro, _rd);

                    solid_hit_mask |= Traverse_BLAS_WithStack_AnyHit(_ro, _rd, bbox_mask, nodes, m.node_index, tris,
                                                                     materials, tri_indices, int(mi_indices[i]), inter);
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
Ray::NS::simd_ivec<S> Ray::NS::Traverse_TLAS_WithStack_AnyHit(
    const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], int ray_type, const simd_ivec<S> &ray_mask,
    const wbvh_node_t *nodes, uint32_t node_index, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
    const mesh_t *meshes, const mtri_accel_t *mtris, const tri_mat_data_t *materials, const uint32_t *tri_indices,
    hit_data_t<S> &inter) {
    const int ray_vismask = (1u << ray_type);

    simd_ivec<S> solid_hit_mask = {0};

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    alignas(S * 4) int ray_masks[S], inter_prim_index[S];
    alignas(S * 4) float inter_t[S], inter_u[S], inter_v[S];
    ray_mask.store_to(ray_masks, simd_mem_aligned);
    inter.prim_index.store_to(inter_prim_index, simd_mem_aligned);
    inter.t.store_to(inter_t, simd_mem_aligned);
    inter.u.store_to(inter_u, simd_mem_aligned);
    inter.v.store_to(inter_v, simd_mem_aligned);

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
                    if ((mi.ray_visibility & ray_vismask) == 0) {
                        continue;
                    }

                    const mesh_t &m = meshes[mi.mesh_index];

                    if (!bbox_test(_inv_d, _inv_d_o, inter_t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, mi.inv_xform, tr_ro, tr_rd);

                    const int hit_type = Traverse_BLAS_WithStack_AnyHit<S>(tr_ro, tr_rd, nodes, m.node_index, mtris,
                                                                           materials, tri_indices, inter_prim_index[ri],
                                                                           inter_t[ri], inter_u[ri], inter_v[ri]);
                    if (hit_type) {
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
bool Ray::NS::Traverse_BLAS_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                 const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                 uint32_t node_index, const tri_accel_t *tris,
                                                 const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter) {
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
bool Ray::NS::Traverse_BLAS_WithStack_ClosestHit(const float ro[3], const float rd[3], const wbvh_node_t *nodes,
                                                 uint32_t node_index, const mtri_accel_t *mtris,
                                                 const uint32_t *tri_indices, int &inter_prim_index, float &inter_t,
                                                 float &inter_u, float &inter_v) {
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
Ray::NS::Traverse_BLAS_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
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
int Ray::NS::Traverse_BLAS_WithStack_AnyHit(const float ro[3], const float rd[3], const wbvh_node_t *nodes,
                                            uint32_t node_index, const mtri_accel_t *mtris,
                                            const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                            int &inter_prim_index, float &inter_t, float &inter_u, float &inter_v) {
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

    return select(N_dot_L > 0.0f, Fd, simd_fvec<S>{0.0f});
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
                                            const bool uniform_sampling, const simd_fvec<S> rand[2],
                                            simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]) {
    const simd_fvec<S> phi = 2 * PI * rand[1];
    const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);

    simd_fvec<S> V[3];
    if (uniform_sampling) {
        const simd_fvec<S> dir = sqrt(1.0f - rand[0] * rand[0]);

        // in tangent-space
        V[0] = dir * cos_phi;
        V[1] = dir * sin_phi;
        V[2] = rand[0];
    } else {
        const simd_fvec<S> dir = sqrt(rand[0]);
        const simd_fvec<S> k = sqrt(1.0f - rand[0]);

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
                                        const simd_fvec<S> reflected_dir_ts[3], const simd_fvec<S> alpha[2],
                                        const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                                        const simd_fvec<S> spec_col[3], const simd_fvec<S> spec_col_90[3],
                                        simd_fvec<S> out_color[4]) {
    const simd_fvec<S> D = D_GGX(sampled_normal_ts, alpha[0], alpha[1]);

    const simd_fvec<S> G = G1(view_dir_ts, alpha[0], alpha[1]) * G1(reflected_dir_ts, alpha[0], alpha[1]);

    const simd_fvec<S> FH =
        (fresnel_dielectric_cos(dot3(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0f - spec_F0);

    simd_fvec<S> F[3];
    UNROLLED_FOR(i, 3, { F[i] = mix(spec_col[i], spec_col_90[i], FH); })

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2] * reflected_dir_ts[2]);
    UNROLLED_FOR(i, 3, { F[i] = select(denom != 0.0f, F[i] * safe_div_pos(D * G, denom), simd_fvec<S>{0.0f}); })

    const simd_fvec<S> pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, alpha);

    UNROLLED_FOR(i, 3, { out_color[i] = F[i] * max(reflected_dir_ts[2], 0.0f); })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_GGXSpecular_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                      const simd_fvec<S> I[3], const simd_fvec<S> alpha[2],
                                      const simd_fvec<S> &spec_ior, const simd_fvec<S> &spec_F0,
                                      const simd_fvec<S> spec_col[3], const simd_fvec<S> spec_col_90[3],
                                      const simd_fvec<S> rand[2], simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]) {
    const simd_ivec<S> is_mirror = simd_cast(alpha[0] * alpha[1] < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        reflect(I, N, dot3(N, I), out_V);
        const simd_fvec<S> FH = (fresnel_dielectric_cos(dot3(out_V, N), spec_ior) - spec_F0) / (1.0f - spec_F0);
        UNROLLED_FOR(i, 3, { out_color[i] = mix(spec_col[i], spec_col_90[i], FH) * 1e6f; })
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
    SampleGGX_VNDF_Bounded(view_dir_ts, alpha, rand, sampled_normal_ts);

    const simd_fvec<S> dot_N_V = -dot3(sampled_normal_ts, view_dir_ts);
    simd_fvec<S> reflected_dir_ts[3];
    const simd_fvec<S> _view_dir_ts[3] = {-view_dir_ts[0], -view_dir_ts[1], -view_dir_ts[2]};
    reflect(_view_dir_ts, sampled_normal_ts, dot_N_V, reflected_dir_ts);
    safe_normalize(reflected_dir_ts);

    simd_fvec<S> glossy_V[3], glossy_F[4];
    world_from_tangent(T, B, N, reflected_dir_ts, glossy_V);
    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha, spec_ior, spec_F0, spec_col,
                              spec_col_90, glossy_F);

    UNROLLED_FOR(i, 3, { where(is_glossy, out_V[i]) = glossy_V[i]; })
    UNROLLED_FOR(i, 4, { where(is_glossy, out_color[i]) = glossy_F[i]; })
}

template <int S>
void Ray::NS::Evaluate_GGXRefraction_BSDF(const simd_fvec<S> view_dir_ts[3], const simd_fvec<S> sampled_normal_ts[3],
                                          const simd_fvec<S> refr_dir_ts[3], const simd_fvec<S> alpha[2],
                                          const simd_fvec<S> &eta, const simd_fvec<S> refr_col[3],
                                          simd_fvec<S> out_color[4]) {
    const simd_fvec<S> D = D_GGX(sampled_normal_ts, alpha[0], alpha[1]);

    const simd_fvec<S> G1o = G1(refr_dir_ts, alpha[0], alpha[1]), G1i = G1(view_dir_ts, alpha[0], alpha[1]);

    const simd_fvec<S> denom = dot3(refr_dir_ts, sampled_normal_ts) + dot3(view_dir_ts, sampled_normal_ts) * eta;
    const simd_fvec<S> jacobian = safe_div_pos(max(-dot3(refr_dir_ts, sampled_normal_ts), 0.0f), denom * denom);

    simd_fvec<S> F = safe_div(D * G1i * G1o * max(dot3(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian,
                              (/*-refr_dir_ts[2] */ view_dir_ts[2]));

    const simd_fvec<S> pdf =
        safe_div(D * G1o * max(dot3(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian, view_dir_ts[2]);

    // const float pdf = D * fmaxf(sampled_normal_ts[2], 0.0f) * jacobian;
    // const float pdf = safe_div(D * sampled_normal_ts[2] * fmaxf(-dot3(refr_dir_ts, sampled_normal_ts), 0.0f), denom);

    const simd_fvec<S> is_valid =
        (refr_dir_ts[2] < 0.0f) & (view_dir_ts[2] > 0.0f) & (alpha[0] >= 1e-7f) & (alpha[1] >= 1e-7f);

    UNROLLED_FOR(i, 3, { out_color[i] = select(is_valid, F * refr_col[i], simd_fvec<S>{0.0f}); })
    out_color[3] = select(is_valid, pdf, simd_fvec<S>{0.0f});
}

template <int S>
void Ray::NS::Sample_GGXRefraction_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                        const simd_fvec<S> I[3], const simd_fvec<S> alpha[2], const simd_fvec<S> &eta,
                                        const simd_fvec<S> refr_col[3], const simd_fvec<S> rand[2],
                                        simd_fvec<S> out_V[4], simd_fvec<S> out_color[4]) {
    const simd_ivec<S> is_mirror = simd_cast(alpha[0] * alpha[1] < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        const simd_fvec<S> cosi = -dot3(I, N);
        const simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);

        const simd_fvec<S> m = eta * cosi - safe_sqrt(cost2);
        UNROLLED_FOR(i, 3, { out_V[i] = eta * I[i] + m * N[i]; })
        safe_normalize(out_V);

        out_V[3] = m;
        UNROLLED_FOR(i, 3, { out_color[i] = select(cost2 >= 0.0f, refr_col[i] * 1e6f, simd_fvec<S>{0.0f}); })
        out_color[3] = select(cost2 >= 0.0f, simd_fvec<S>{1e6f}, simd_fvec<S>{0.0f});
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
    SampleGGX_VNDF(view_dir_ts, alpha, rand, sampled_normal_ts);

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
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, alpha, eta, refr_col, glossy_F);

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
    const simd_fvec<S> clearcoat_alpha[2] = {0.25f * 0.25f, 0.25f * 0.25f};
    const simd_fvec<S> G = G1(view_dir_ts, clearcoat_alpha[0], clearcoat_alpha[1]) *
                           G1(reflected_dir_ts, clearcoat_alpha[0], clearcoat_alpha[1]);

    const simd_fvec<S> FH =
        (fresnel_dielectric_cos(dot3(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
        (1.0f - clearcoat_F0);
    simd_fvec<S> F = mix(simd_fvec<S>{0.04f}, simd_fvec<S>{1.0f}, FH);

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2]) * abs(reflected_dir_ts[2]);
    F = select(denom != 0.0f, safe_div_pos(F * D * G, denom), simd_fvec<S>{0.0f});
    F *= max(reflected_dir_ts[2], 0.0f);

    const simd_fvec<S> pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, clearcoat_alpha);

    UNROLLED_FOR(i, 3, { out_color[i] = F; })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_PrincipledClearcoat_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                              const simd_fvec<S> I[3], const simd_fvec<S> &clearcoat_roughness2,
                                              const simd_fvec<S> &clearcoat_ior, const simd_fvec<S> &clearcoat_F0,
                                              const simd_fvec<S> rand[2], simd_fvec<S> out_V[3],
                                              simd_fvec<S> out_color[4]) {
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
    simd_fvec<S> sampled_normal_ts[3], alpha[2] = {clearcoat_roughness2, clearcoat_roughness2};
    SampleGGX_VNDF_Bounded(view_dir_ts, alpha, rand, sampled_normal_ts);

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

template <int S> void Ray::NS::TransformPoint(const simd_fvec<S> xform[16], simd_fvec<S> p[3]) {
    const simd_fvec<S> temp0 = xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2] + xform[12];
    const simd_fvec<S> temp1 = xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2] + xform[13];
    const simd_fvec<S> temp2 = xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2] + xform[14];

    p[0] = temp0;
    p[1] = temp1;
    p[2] = temp2;
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
    where(phi < 0.0f, phi) += 2 * PI;
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
    UNROLLED_FOR_S(i, S, { phi.template set<i>(-atan2f(d[2].template get<i>(), d[0].template get<i>())); })

    phi += y_rotation;
    where(phi < 0.0f, phi) += 2 * PI;
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
void Ray::NS::SampleNearest(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                            const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask,
                            simd_fvec<S> out_rgba[4]) {
    const Cpu::TexStorageBase &storage = *textures[index >> 28];
    auto _lod = (simd_ivec<S>)lod;

    where(_lod > MAX_MIP_LEVEL, _lod) = MAX_MIP_LEVEL;

    for (int j = 0; j < S; j++) {
        if (!mask[j]) {
            continue;
        }

        const auto &pix = storage.Fetch(index & 0x00ffffff, uvs[0][j], uvs[1][j], _lod[j]);

        UNROLLED_FOR(i, 4, { out_rgba[i].set(j, static_cast<float>(pix.v[i])); })
    }

    const float k = 1.0f / 255.0f;
    UNROLLED_FOR(i, 4, { out_rgba[i] *= k; })
}

template <int S>
void Ray::NS::SampleBilinear(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                             const simd_fvec<S> uvs[2], const simd_ivec<S> &lod, const simd_fvec<S> rand[2],
                             const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    const Cpu::TexStorageBase &storage = *textures[index >> 28];

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

#if USE_STOCH_TEXTURE_FILTERING
    // Jitter UVs
    _uvs[0] += rand[0];
    _uvs[1] += rand[1];

    for (int i = 0; i < S; ++i) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = storage.Fetch(tex, int(_uvs[0][i]), int(_uvs[1][i]), lod[i]);

        out_rgba[0].set(i, p00.v[0]);
        out_rgba[1].set(i, p00.v[1]);
        out_rgba[2].set(i, p00.v[2]);
        out_rgba[3].set(i, p00.v[3]);
    }
#else  // USE_STOCH_TEXTURE_FILTERING
    const simd_fvec<S> k[2] = {fract(_uvs[0]), fract(_uvs[1])};

    simd_fvec<S> p0[4] = {}, p1[4] = {};

    for (int i = 0; i < S; ++i) {
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
#endif // USE_STOCH_TEXTURE_FILTERING
}

template <int S>
void Ray::NS::SampleTrilinear(const Cpu::TexStorageBase *const textures[], const uint32_t index,
                              const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_fvec<S> rand[2],
                              const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> col1[4];
    SampleBilinear(textures, index, uvs, (simd_ivec<S>)floor(lod), rand, mask, col1);
    simd_fvec<S> col2[4];
    SampleBilinear(textures, index, uvs, (simd_ivec<S>)ceil(lod), rand, mask, col2);

    const simd_fvec<S> k = fract(lod);
    UNROLLED_FOR(i, 4, { out_rgba[i] = col1[i] * (1.0f - k) + col2[i] * k; })
}

template <int S>
void Ray::NS::SampleLatlong_RGBE(const Cpu::TexStorageRGBA &storage, const uint32_t index, const simd_fvec<S> dir[3],
                                 const float y_rotation, const simd_fvec<S> rand[2], const simd_ivec<S> &mask,
                                 simd_fvec<S> out_rgb[3]) {
    const simd_fvec<S> y = clamp(dir[1], -1.0f, 1.0f);

    simd_fvec<S> theta = 0.0f, phi = 0.0f;
    UNROLLED_FOR_S(i, S, {
        if (mask.template get<i>()) {
            theta.template set<i>(acosf(y.template get<i>()) / PI);
            phi.template set<i>(atan2f(dir[2].template get<i>(), dir[0].template get<i>()) + y_rotation);
        }
    })
    where(phi < 0.0f, phi) += 2 * PI;
    where(phi > 2 * PI, phi) -= 2 * PI;

    const simd_fvec<S> u = 0.5f * phi / PI;

    const int tex = int(index & 0x00ffffff);
    float sz[2];
    storage.GetFRes(tex, 0, sz);

    simd_fvec<S> uvs[2] = {clamp(u * sz[0], 0.0f, sz[0] - 1.0f), clamp(theta * sz[1], 0.0f, sz[1] - 1.0f)};

#if USE_STOCH_TEXTURE_FILTERING
    uvs[0] += rand[0] - 0.5f;
    uvs[1] += rand[1] - 0.5f;

    const simd_ivec<S> iuvs[2] = {simd_ivec<S>(uvs[0]), simd_ivec<S>(uvs[1])};

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = storage.Get(tex, iuvs[0][i], iuvs[1][i], 0);

        const float f = exp2f(float(p00.v[3]) - 128.0f);
        out_rgb[0].set(i, to_norm_float(p00.v[0]) * f);
        out_rgb[1].set(i, to_norm_float(p00.v[1]) * f);
        out_rgb[2].set(i, to_norm_float(p00.v[2]) * f);
    }
#else  // USE_STOCH_TEXTURE_FILTERING
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

        float f = exp2f(float(p00.v[3]) - 128.0f);
        _p00[0].set(i, to_norm_float(p00.v[0]) * f);
        _p00[1].set(i, to_norm_float(p00.v[1]) * f);
        _p00[2].set(i, to_norm_float(p00.v[2]) * f);

        f = exp2f(float(p01.v[3]) - 128.0f);
        _p01[0].set(i, to_norm_float(p01.v[0]) * f);
        _p01[1].set(i, to_norm_float(p01.v[1]) * f);
        _p01[2].set(i, to_norm_float(p01.v[2]) * f);

        f = exp2f(float(p10.v[3]) - 128.0f);
        _p10[0].set(i, to_norm_float(p10.v[0]) * f);
        _p10[1].set(i, to_norm_float(p10.v[1]) * f);
        _p10[2].set(i, to_norm_float(p10.v[2]) * f);

        f = exp2f(float(p11.v[3]) - 128.0f);
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
#endif // USE_STOCH_TEXTURE_FILTERING
}

template <int S>
void Ray::NS::IntersectScene(ray_data_t<S> &r, const int min_transp_depth, const int max_transp_depth,
                             const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                             const scene_data_t &sc, const uint32_t root_index,
                             const Cpu::TexStorageBase *const textures[], hit_data_t<S> &inter) {
    simd_fvec<S> ro[3] = {r.o[0], r.o[1], r.o[2]};

    const simd_uvec<S> ray_flags = simd_uvec<S>(1 << get_ray_type(r.depth));

    const simd_uvec<S> px_hash = hash(r.xy);
    const simd_uvec<S> rand_hash = hash_combine(px_hash, rand_seed);

    auto rand_dim = simd_uvec<S>(RAND_DIM_BASE_COUNT + get_total_depth(r.depth) * RAND_DIM_BOUNCE_COUNT);

    simd_ivec<S> keep_going = r.mask;
    while (keep_going.not_all_zeros()) {
        const simd_fvec<S> t_val = inter.t;

        if (sc.wnodes) {
            NS::Traverse_TLAS_WithStack_ClosestHit(ro, r.d, ray_flags, keep_going, sc.wnodes, root_index,
                                                   sc.mesh_instances, sc.mi_indices, sc.meshes, sc.mtris,
                                                   sc.tri_indices, inter);
        } else {
            NS::Traverse_TLAS_WithStack_ClosestHit(ro, r.d, ray_flags, keep_going, sc.nodes, root_index,
                                                   sc.mesh_instances, sc.mi_indices, sc.meshes, sc.tris, sc.tri_indices,
                                                   inter);
        }

        keep_going &= simd_cast(inter.v >= 0.0f);
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
            const float *vtx_uvs = &sc.vertices[0].t[0];
            const int VtxUVsStride = sizeof(vertex_t) / sizeof(float);

            UNROLLED_FOR(i, 2, {
                const simd_fvec<S> temp1 = gather(vtx_uvs + i, vtx_indices[0] * VtxUVsStride);
                const simd_fvec<S> temp2 = gather(vtx_uvs + i, vtx_indices[1] * VtxUVsStride);
                const simd_fvec<S> temp3 = gather(vtx_uvs + i, vtx_indices[2] * VtxUVsStride);

                uvs[i] = temp1 * w + temp2 * inter.u + temp3 * inter.v;
            })
        }

        simd_fvec<S> mix_term_rand[2];
        get_scrambled_2d_rand(rand_dim + unsigned(RAND_DIM_BSDF_PICK), rand_hash, iteration - 1, rand_seq, mix_term_rand);

        simd_fvec<S> tex_rand[2];
        get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq, tex_rand);

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

                    while (mat->type == eShadingNode::Mix) {
                        simd_fvec<S> _mix_val = 1.0f;

                        const uint32_t first_t = mat->textures[BASE_TEXTURE];
                        if (first_t != 0xffffffff) {
                            simd_fvec<S> mix[4] = {};
                            SampleBilinear(textures, first_t, uvs, {0}, tex_rand, same_mi, mix);
                            if (first_t & TEX_YCOCG_BIT) {
                                YCoCg_to_RGB(mix, mix);
                            }
                            if (first_t & TEX_SRGB_BIT) {
                                srgb_to_rgb(mix, mix);
                            }
                            _mix_val *= mix[0];
                        }
                        _mix_val *= mat->strength;

                        first_mi = 0xffff;

                        for (int i = 0; i < S; i++) {
                            if (!same_mi[i]) {
                                continue;
                            }

                            float mix_val = _mix_val[i];

                            if (mix_term_rand[0][i] > mix_val) {
                                mat_index.set(i, mat->textures[MIX_MAT1]);
                                mix_term_rand[0].set(i, safe_div_pos(mix_term_rand[0][i] - mix_val, 1.0f - mix_val));
                            } else {
                                mat_index.set(i, mat->textures[MIX_MAT2]);
                                mix_term_rand[0].set(i, safe_div_pos(mix_term_rand[0][i], mix_val));
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

                    if (mat->type != eShadingNode::Transparent) {
                        where(ray_queue[index], keep_going) = 0;
                        index++;
                        continue;
                    }

#if USE_PATH_TERMINATION
                    const simd_ivec<S> can_terminate_path = get_transp_depth(r.depth) > min_transp_depth;
#else
                    const simd_ivec<S> can_terminate_path = 0;
#endif
                    const simd_fvec<S> lum = max(r.c[0], max(r.c[1], r.c[2]));
                    const simd_fvec<S> &p = mix_term_rand[1];
                    simd_fvec<S> q = 0.0f;
                    where(can_terminate_path, q) = max(0.05f, 1.0f - lum);

                    const simd_ivec<S> _terminate =
                        simd_cast(p < q) | simd_cast(lum == 0.0f) | (get_transp_depth(r.depth) + 1 >= max_transp_depth);

                    UNROLLED_FOR(i, 3, {
                        where(ray_queue[index] & _terminate, r.c[i]) = 0.0f;
                        where(ray_queue[index] & ~_terminate, r.c[i]) *= safe_div_pos(mat->base_color[i], 1.0f - q);
                    })

                    keep_going &= ~_terminate;
                }

                index++;
            }
        }

        const simd_fvec<S> t = inter.t + HIT_BIAS;
        UNROLLED_FOR(i, 3, { where(keep_going, ro[i]) += r.d[i] * t; })

        // discard current intersection
        where(keep_going, inter.v) = -1.0f;
        where(keep_going, inter.t) = t_val - inter.t;

        where(keep_going, r.depth) += pack_depth(simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{1});

        rand_dim += RAND_DIM_BOUNCE_COUNT;
    }

    inter.t += distance(r.o, ro);
}

template <int S>
void Ray::NS::TraceRays(Span<ray_data_t<S>> rays, int min_transp_depth, int max_transp_depth, const scene_data_t &sc,
                        uint32_t root_index, bool trace_lights, const Cpu::TexStorageBase *const textures[],
                        const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                        Span<hit_data_t<S>> out_inter) {
    for (int i = 0; i < rays.size(); ++i) {
        ray_data_t<S> &r = rays[i];
        hit_data_t<S> &inter = out_inter[i];

        IntersectScene(r, min_transp_depth, max_transp_depth, rand_seq, rand_seed, iteration, sc, root_index, textures,
                       inter);
        if (trace_lights && sc.visible_lights_count) {
            IntersectAreaLights(r, sc.lights, sc.light_wnodes, inter);
        }
    }
}

template <int S>
void Ray::NS::TraceShadowRays(Span<const shadow_ray_t<S>> rays, int max_transp_depth, float _clamp_val,
                              const scene_data_t &sc, const uint32_t root_index, const uint32_t rand_seq[],
                              const uint32_t rand_seed, const int iteration,
                              const Cpu::TexStorageBase *const textures[], int img_w, color_rgba_t *out_color) {
    const float limit = (_clamp_val != 0.0f) ? 3.0f * _clamp_val : FLT_MAX;
    for (int i = 0; i < rays.size(); ++i) {
        const shadow_ray_t<S> &sh_r = rays[i];

        simd_fvec<S> rc[3];
        IntersectScene(sh_r, max_transp_depth, sc, root_index, rand_seq, rand_seed, iteration, textures, rc);
        if (sc.blocker_lights_count) {
            const simd_fvec<S> k = IntersectAreaLights(sh_r, sc.lights, sc.light_wnodes);
            UNROLLED_FOR(j, 3, { rc[j] *= k; })
        }
        const simd_fvec<S> sum = rc[0] + rc[1] + rc[2];
        UNROLLED_FOR(j, 3, { where(sum > limit, rc[j]) = safe_div_pos(rc[j] * limit, sum); })

        const simd_uvec<S> x = sh_r.xy >> 16, y = sh_r.xy & 0x0000FFFF;

        // TODO: match layouts!
        UNROLLED_FOR_S(i, S, {
            if (sh_r.mask.template get<i>()) {
                auto old_val =
                    simd_fvec4(out_color[y.template get<i>() * img_w + x.template get<i>()].v, simd_mem_aligned);
                old_val += simd_fvec4(rc[0].template get<i>(), rc[1].template get<i>(), rc[2].template get<i>(), 0.0f);
                old_val.store_to(out_color[y.template get<i>() * img_w + x.template get<i>()].v, simd_mem_aligned);
            }
        })
    }
}

template <int S>
void Ray::NS::IntersectScene(const shadow_ray_t<S> &r, const int max_transp_depth, const scene_data_t &sc,
                             const uint32_t node_index, const uint32_t rand_seq[], const uint32_t rand_seed,
                             const int iteration, const Cpu::TexStorageBase *const textures[], simd_fvec<S> rc[3]) {
    simd_fvec<S> ro[3] = {r.o[0], r.o[1], r.o[2]};
    UNROLLED_FOR(i, 3, { rc[i] = r.c[i]; })
    simd_fvec<S> dist = select(r.dist >= 0.0f, r.dist, simd_fvec<S>{MAX_DIST});
    simd_ivec<S> depth = get_transp_depth(r.depth);

    const simd_uvec<S> px_hash = hash(r.xy);
    const simd_uvec<S> rand_hash = hash_combine(px_hash, rand_seed);

    auto rand_dim = simd_uvec<S>(RAND_DIM_BASE_COUNT + get_total_depth(r.depth) * RAND_DIM_BOUNCE_COUNT);

    simd_ivec<S> keep_going = simd_cast(dist > HIT_EPS) & r.mask;
    while (keep_going.not_all_zeros()) {
        hit_data_t<S> inter;
        inter.t = dist;

        simd_ivec<S> solid_hit;
        if (sc.wnodes) {
            solid_hit = Traverse_TLAS_WithStack_AnyHit(ro, r.d, RAY_TYPE_SHADOW, keep_going, sc.wnodes, node_index,
                                                       sc.mesh_instances, sc.mi_indices, sc.meshes, sc.mtris,
                                                       sc.tri_materials, sc.tri_indices, inter);
        } else {
            solid_hit = Traverse_TLAS_WithStack_AnyHit(ro, r.d, RAY_TYPE_SHADOW, keep_going, sc.nodes, node_index,
                                                       sc.mesh_instances, sc.mi_indices, sc.meshes, sc.tris,
                                                       sc.tri_materials, sc.tri_indices, inter);
        }

        const simd_ivec<S> terminate_mask = solid_hit | (depth > max_transp_depth);
        UNROLLED_FOR(i, 3, { where(terminate_mask, rc[i]) = 0.0f; })

        keep_going &= simd_cast(inter.v >= 0.0f) & ~terminate_mask;
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
            const float *vtx_uvs = &sc.vertices[0].t[0];
            const int VtxUVsStride = sizeof(vertex_t) / sizeof(float);

            UNROLLED_FOR(i, 2, {
                const simd_fvec<S> temp1 = gather(vtx_uvs + i, vtx_indices[0] * VtxUVsStride);
                const simd_fvec<S> temp2 = gather(vtx_uvs + i, vtx_indices[1] * VtxUVsStride);
                const simd_fvec<S> temp3 = gather(vtx_uvs + i, vtx_indices[2] * VtxUVsStride);

                sh_uvs[i] = temp1 * w + temp2 * inter.u + temp3 * inter.v;
            })
        }

        simd_fvec<S> tex_rand[2];
        get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq, tex_rand);

        simd_ivec<S> mat_index = gather(reinterpret_cast<const int *>(sc.tri_materials), tri_index) &
                                 simd_ivec<S>((MATERIAL_INDEX_BITS << 16) | MATERIAL_INDEX_BITS);

        where(~is_backfacing, mat_index) = mat_index & 0xffff; // use front material index
        where(is_backfacing, mat_index) = mat_index >> 16;     // use back material index
        where(inter.v < 0.0f, mat_index) = 0xffff;

        { // resolve material
            simd_ivec<S> ray_queue[S];
            int index = 0, num = 1;

            ray_queue[0] = simd_cast(inter.v >= 0.0f);

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
                        if (mat->type == eShadingNode::Mix) {
                            simd_fvec<S> mix_val = mat->strength;
                            const uint32_t first_t = mat->textures[BASE_TEXTURE];
                            if (first_t != 0xffffffff) {
                                simd_fvec<S> mix[4] = {};
                                SampleBilinear(textures, first_t, sh_uvs, {0}, tex_rand, same_mi, mix);
                                if (first_t & TEX_YCOCG_BIT) {
                                    YCoCg_to_RGB(mix, mix);
                                }
                                if (first_t & TEX_SRGB_BIT) {
                                    srgb_to_rgb(mix, mix);
                                }
                                mix_val *= mix[0];
                            }

                            stack[stack_size++] = {mat->textures[MIX_MAT1], weight * (1.0f - mix_val)};
                            stack[stack_size++] = {mat->textures[MIX_MAT2], weight * mix_val};
                        } else if (mat->type == eShadingNode::Transparent) {
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

        rand_dim += RAND_DIM_BOUNCE_COUNT;
    }
}

// Pick point on any light source for evaluation
template <int S>
void Ray::NS::SampleLightSource(const simd_fvec<S> P[3], const simd_fvec<S> T[3], const simd_fvec<S> B[3],
                                const simd_fvec<S> N[3], const scene_data_t &sc,
                                const Cpu::TexStorageBase *const textures[], const simd_fvec<S> &rand_pick_light,
                                const simd_fvec<S> rand_light_uv[2], const simd_fvec<S> rand_tex_uv[2],
                                simd_ivec<S> ray_mask, light_sample_t<S> &ls) {
    simd_fvec<S> ri = rand_pick_light;

#if USE_HIERARCHICAL_NEE
    simd_ivec<S> light_index = -1;
    simd_fvec<S> factor = 1.0f;

    { // Traverse light tree structure
        struct {
            simd_ivec<S> mask;
            uint32_t i;
        } queue[S];

        queue[0].mask = ray_mask;
        queue[0].i = 0; // start from root

        int index = 0, num = 1;
        while (index != num) {
            uint32_t i = queue[index].i;
            while (!is_leaf_node(sc.light_wnodes[i])) {
                simd_fvec<S> importance[8];
                calc_lnode_importance(sc.light_wnodes[i], P, importance);

                simd_fvec<S> total_importance = importance[0];
                UNROLLED_FOR(j, 7, { total_importance += importance[j + 1]; })

                queue[index].mask &= simd_cast(total_importance > 0.0f);
                if (queue[index].mask.all_zeros()) {
                    // failed to find lightsource for sampling
                    break;
                }

                simd_fvec<S> factors[8];
                UNROLLED_FOR(j, 8, { factors[j] = safe_div_pos(importance[j], total_importance); })

                simd_fvec<S> factors_cdf[9] = {};
                UNROLLED_FOR(j, 8, { factors_cdf[j + 1] = factors_cdf[j] + factors[j]; })
                // make sure cdf ends with 1.0
                UNROLLED_FOR(j, 8, { where(factors_cdf[j + 1] == factors_cdf[8], factors_cdf[j + 1]) = 1.01f; })

                simd_ivec<S> next = 0;
                UNROLLED_FOR(j, 8, { where(factors_cdf[j + 1] <= ri, next) += 1; })
                assert((next >= 8).all_zeros());

                const int first_next = next[GetFirstBit(queue[index].mask.movemask())];

                const simd_ivec<S> same_next = (next == first_next);
                const simd_ivec<S> diff_next = and_not(same_next, queue[index].mask);

                if (diff_next.not_all_zeros()) {
                    queue[index].mask &= same_next;
                    // TODO: Avoid visiting node twice!
                    queue[num++] = {diff_next, i};
                }

                where(queue[index].mask, ri) = fract(safe_div_pos(ri - factors_cdf[first_next], factors[first_next]));
                i = sc.light_wnodes[i].child[first_next];
                where(queue[index].mask, factor) *= factors[first_next];
            }

            where(queue[index].mask, light_index) = (sc.light_wnodes[i].child[0] & PRIM_INDEX_BITS);

            ++index;
        }
    }

    ray_mask &= (light_index != -1);
    if (ray_mask.all_zeros()) {
        // failed to find lightsources for sampling
        return;
    }
    factor = 1.0f / factor;
#else
    simd_ivec<S> light_index = min(simd_ivec<S>{ri * float(sc.li_indices.size())}, int(sc.li_indices.size() - 1));
    ri = ri * float(sc.li_indices.size()) - simd_fvec<S>(light_index);
    light_index = gather(reinterpret_cast<const int *>(sc.li_indices.data()), light_index);
    const simd_fvec<S> factor = float(sc.li_indices.size());
#endif

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

        const light_t &l = sc.lights[first_li];

        UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) = l.col[i]; })
        where(ray_queue[index], ls.cast_shadow) = l.cast_shadow ? -1 : 0;

        if (l.type == LIGHT_TYPE_SPHERE) {
            const simd_fvec<S> r1 = rand_light_uv[0], r2 = rand_light_uv[1];

            const float *center = l.sph.pos;
            const simd_fvec<S> surface_to_center[3] = {center[0] - P[0], center[1] - P[1], center[2] - P[2]};
            simd_fvec<S> sampled_dir[3];
            map_to_cone(r1, r2, surface_to_center, l.sph.radius, sampled_dir);
            const simd_fvec<S> disk_dist = normalize(sampled_dir);

            if (l.sph.radius > 0.0f) {
                const simd_fvec<S> ls_dist = sphere_intersection(center, l.sph.radius, P, sampled_dir);

                const simd_fvec<S> light_surf_pos[3] = {
                    P[0] + sampled_dir[0] * ls_dist, P[1] + sampled_dir[1] * ls_dist, P[2] + sampled_dir[2] * ls_dist};
                simd_fvec<S> light_forward[3] = {light_surf_pos[0] - center[0], light_surf_pos[1] - center[1],
                                                 light_surf_pos[2] - center[2]};
                normalize(light_forward);

                simd_fvec<S> lp_biased[3];
                offset_ray(light_surf_pos, light_forward, lp_biased);

                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })
                where(ray_queue[index], ls.pdf) = safe_div_pos(disk_dist * disk_dist, PI * l.sph.radius * l.sph.radius);
            } else {
                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = center[i]; })
                where(ray_queue[index], ls.pdf) = (disk_dist * disk_dist) / PI;
            }

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = sampled_dir[i]; })
            where(ray_queue[index], ls.area) = l.sph.area;

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
                            _angle.template set<i>(acosf(_dot.template get<i>()));
                        }
                    })
                    const simd_fvec<S> k = saturate((l.sph.spot - _angle) / l.sph.blend);
                    UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= k; })
                }
                UNROLLED_FOR(i, 3, { where(~mask & ray_queue[index], ls.col[i]) = 0.0f; })
            }
        } else if (l.type == LIGHT_TYPE_DIR) {
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = l.dir.dir[i]; })
            where(ray_queue[index], ls.area) = 0.0f;
            where(ray_queue[index], ls.pdf) = 1.0f;
            where(ray_queue[index], ls.dist_mul) = MAX_DIST;
            if (l.dir.angle != 0.0f) {
                const float radius = tanf(l.dir.angle);

                simd_fvec<S> V[3];
                map_to_cone(rand_light_uv[0], rand_light_uv[1], ls.L, radius, V);
                safe_normalize(V);

                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = V[i]; })
                where(ray_queue[index], ls.area) = PI * radius * radius;

                const simd_fvec<S> cos_theta = dot3(ls.L, l.dir.dir);
                where(ray_queue[index], ls.pdf) = safe_div_pos(1.0f, ls.area * cos_theta);
            }
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = P[i] + ls.L[i]; })

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }
        } else if (l.type == LIGHT_TYPE_RECT) {
            simd_fvec<S> lp[3];
#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
            const simd_fvec<S> vp[3] = {l.rect.pos[0], l.rect.pos[1], l.rect.pos[2]},
                               vu[3] = {l.rect.u[0], l.rect.u[1], l.rect.u[2]},
                               vv[3] = {l.rect.v[0], l.rect.v[1], l.rect.v[2]};
            simd_fvec<S> pdf = SampleSphericalRectangle(P, vp, vu, vv, rand_light_uv, lp);
            const simd_ivec<S> invalid_pdf = ray_queue[index] & simd_cast(pdf <= 0.0f);
            if (invalid_pdf.not_all_zeros())
#endif
            {
                const simd_fvec<S> r1 = rand_light_uv[0] - 0.5f, r2 = rand_light_uv[1] - 0.5f;
                UNROLLED_FOR(i, 3, { lp[i] = l.rect.pos[i] + l.rect.u[i] * r1 + l.rect.v[i] * r2; })
            }

            simd_fvec<S> to_light[3];
            UNROLLED_FOR(i, 3, { to_light[i] = lp[i] - P[i]; })
            const simd_fvec<S> ls_dist = normalize(to_light);

            float light_forward[3];
            cross(l.rect.u, l.rect.v, light_forward);
            normalize(light_forward);

            simd_fvec<S> lp_biased[3], _light_forward[3] = {light_forward[0], light_forward[1], light_forward[2]};
            offset_ray(lp, _light_forward, lp_biased);
            UNROLLED_FOR(i, 3, {
                where(ray_queue[index], ls.lp[i]) = lp_biased[i];
                where(ray_queue[index], ls.L[i]) = to_light[i];
            })

            where(ray_queue[index], ls.area) = l.rect.area;

            const simd_fvec<S> cos_theta =
                -ls.L[0] * light_forward[0] - ls.L[1] * light_forward[1] - ls.L[2] * light_forward[2];
            where(invalid_pdf, pdf) = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sky_portal != 0) {
                simd_fvec<S> env_col[3] = {sc.env.env_col[0], sc.env.env_col[1], sc.env.env_col[2]};
                if (sc.env.env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                       sc.env.env_map_rotation, rand_tex_uv, ray_queue[index], tex_col);
                    UNROLLED_FOR(i, 3, { env_col[i] *= tex_col[i]; })
                }
                UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.col[i]) *= sc.env.env_col[i]; })
                where(ray_queue[index], ls.from_env) = -1;
            }
        } else if (l.type == LIGHT_TYPE_DISK) {
            simd_fvec<S> offset[2] = {2.0f * rand_light_uv[0] - 1.0f, 2.0f * rand_light_uv[1] - 1.0f};
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
            const simd_fvec<S> ls_dist = normalize(to_light);

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = to_light[i]; })
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
                    SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                       sc.env.env_map_rotation, rand_tex_uv, ray_queue[index], tex_col);
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

            const simd_fvec<S> phi = PI * rand_light_uv[0];
            const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);

            const simd_fvec<S> normal[3] = {cos_phi * light_u[0] - sin_phi * light_v[0],
                                            cos_phi * light_u[1] - sin_phi * light_v[1],
                                            cos_phi * light_u[2] - sin_phi * light_v[2]};

            const simd_fvec<S> lp[3] = {
                l.line.pos[0] + normal[0] * l.line.radius + (rand_light_uv[1] - 0.5f) * light_dir[0] * l.line.height,
                l.line.pos[1] + normal[1] * l.line.radius + (rand_light_uv[1] - 0.5f) * light_dir[1] * l.line.height,
                l.line.pos[2] + normal[2] * l.line.radius + (rand_light_uv[1] - 0.5f) * light_dir[2] * l.line.height};

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp[i]; })

            simd_fvec<S> to_light[3];
            UNROLLED_FOR(i, 3, { to_light[i] = lp[i] - P[i]; })
            const simd_fvec<S> ls_dist = normalize(to_light);

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = to_light[i]; })
            where(ray_queue[index], ls.area) = l.line.area;

            const simd_fvec<S> cos_theta = 1.0f - abs(dot3(ls.L, light_dir));
            simd_fvec<S> pdf = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            where(cos_theta == 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }
        } else if (l.type == LIGHT_TYPE_TRI) {
            const mesh_instance_t &lmi = sc.mesh_instances[l.tri.mi_index];
            const uint32_t ltri_index = l.tri.tri_index;

            const vertex_t &v1 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 0]],
                           &v2 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 1]],
                           &v3 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 2]];

            float p1[3], p2[3], p3[3];
            TransformPoint(v1.p, lmi.xform, p1);
            TransformPoint(v2.p, lmi.xform, p2);
            TransformPoint(v3.p, lmi.xform, p3);

            const simd_fvec<S> vp1[3] = {p1[0], p1[1], p1[2]}, vp2[3] = {p2[0], p2[1], p2[2]},
                               vp3[3] = {p3[0], p3[1], p3[2]};

            const float e1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]},
                        e2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
            float light_forward[3];
            cross(e1, e2, light_forward);

            const float light_fwd_len =
                sqrtf(light_forward[0] * light_forward[0] + light_forward[1] * light_forward[1] +
                      light_forward[2] * light_forward[2]);
            where(ray_queue[index], ls.area) = 0.5f * light_fwd_len;
            UNROLLED_FOR(i, 3, { light_forward[i] /= light_fwd_len; })

            simd_fvec<S> lp[3] = {};
            simd_fvec<S> luvs[2] = {};
            simd_fvec<S> pdf = {};

#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
            // Spherical triangle sampling
            simd_fvec<S> dir[3];
            pdf = SampleSphericalTriangle(P, vp1, vp2, vp3, rand_light_uv, dir);
            const simd_ivec<S> pdf_positive = ray_queue[index] & simd_cast(pdf > 0.0f);
            if (pdf_positive.not_all_zeros()) {
                // find u, v, t of intersection point
                simd_fvec<S> pvec[3];
                cross(dir, e2, pvec);
                simd_fvec<S> tvec[3];
                UNROLLED_FOR(i, 3, { tvec[i] = P[i] - p1[i]; })
                simd_fvec<S> qvec[3];
                cross(tvec, e1, qvec);

                const simd_fvec<S> inv_det = safe_div(simd_fvec<S>{1.0f}, dot3(e1, pvec));
                const simd_fvec<S> tri_u = dot3(tvec, pvec) * inv_det, tri_v = dot3(dir, qvec) * inv_det;

                UNROLLED_FOR(i, 3, {
                    where(pdf_positive, lp[i]) = (1.0f - tri_u - tri_v) * p1[i] + tri_u * p2[i] + tri_v * p3[i];
                })
                UNROLLED_FOR(i, 2, {
                    where(pdf_positive, luvs[i]) = (1.0f - tri_u - tri_v) * v1.t[i] + tri_u * v2.t[i] + tri_v * v3.t[i];
                })

                UNROLLED_FOR(i, 3, { where(pdf_positive, ls.L[i]) = dir[i]; })
            }
            const simd_ivec<S> pdf_negative = ray_queue[index] & ~pdf_positive;
#else  // USE_SPHERICAL_AREA_LIGHT_SAMPLING
            const simd_ivec<S> pdf_negative = -1;
#endif // USE_SPHERICAL_AREA_LIGHT_SAMPLING
            if (pdf_negative.not_all_zeros()) {
                // Simple area sampling
                const simd_fvec<S> r1 = sqrt(rand_light_uv[0]), r2 = rand_light_uv[1];

                UNROLLED_FOR(i, 2, {
                    where(pdf_negative, luvs[i]) = v1.t[i] * (1.0f - r1) + r1 * (v2.t[i] * (1.0f - r2) + v3.t[i] * r2);
                })
                UNROLLED_FOR(i, 3, {
                    where(pdf_negative, lp[i]) = p1[i] * (1.0f - r1) + r1 * (p2[i] * (1.0f - r2) + p3[i] * r2);
                })

                simd_fvec<S> to_light[3] = {lp[0] - P[0], lp[1] - P[1], lp[2] - P[2]};
                const simd_fvec<S> ls_dist = normalize(to_light);

                UNROLLED_FOR(i, 3, { where(pdf_negative, ls.L[i]) = to_light[i]; })
                const simd_fvec<S> cos_theta = -dot3(ls.L, light_forward);
                where(pdf_negative, pdf) = safe_div_pos(ls_dist * ls_dist, ls.area * cos_theta);
            }

            simd_fvec<S> cos_theta = -dot3(ls.L, light_forward);

            simd_fvec<S> lp_biased[3], vlight_forward[3] = {light_forward[0], light_forward[1], light_forward[2]};
            UNROLLED_FOR(i, 3, { where(cos_theta < 0.0f, vlight_forward[i]) = -vlight_forward[i]; })
            offset_ray(lp, vlight_forward, lp_biased);
            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.lp[i]) = lp_biased[i]; })
            if (l.doublesided) {
                cos_theta = abs(cos_theta);
            }
            simd_ivec<S> accept = simd_cast(cos_theta > 0.0f) & ray_queue[index];
            if (accept.not_all_zeros()) {
                where(accept, ls.pdf) = pdf;
                if (l.tri.tex_index != 0xffffffff) {
                    simd_fvec<S> tex_col[4] = {};
                    SampleBilinear(textures, l.tri.tex_index, luvs, simd_ivec<S>{0}, rand_tex_uv, accept, tex_col);
                    if (l.tri.tex_index & TEX_YCOCG_BIT) {
                        YCoCg_to_RGB(tex_col, tex_col);
                    }
                    if (l.tri.tex_index & TEX_SRGB_BIT) {
                        srgb_to_rgb(tex_col, tex_col);
                    }
                    UNROLLED_FOR(i, 3, { where(accept, ls.col[i]) *= tex_col[i]; })
                }
            }
        } else if (l.type == LIGHT_TYPE_ENV) {
            simd_fvec<S> dir_and_pdf[4];
            if (sc.env.qtree_levels) {
                // Sample environment using quadtree
                const auto *qtree_mips = reinterpret_cast<const simd_fvec4 *const *>(sc.env.qtree_mips);

                Sample_EnvQTree(sc.env.env_map_rotation, qtree_mips, sc.env.qtree_levels, ri, rand_light_uv[0],
                                rand_light_uv[1], dir_and_pdf);
            } else {
                // Sample environment as hemishpere
                const simd_fvec<S> phi = 2 * PI * rand_light_uv[1];
                const simd_fvec<S> cos_phi = cos(phi), sin_phi = sin(phi);
                const simd_fvec<S> dir = sqrt(1.0f - rand_light_uv[0] * rand_light_uv[0]);

                const simd_fvec<S> V[3] = {dir * cos_phi, dir * sin_phi, rand_light_uv[0]}; // in tangent-space
                world_from_tangent(T, B, N, V, dir_and_pdf);
                dir_and_pdf[3] = 0.5f / PI;
            }

            UNROLLED_FOR(i, 3, { where(ray_queue[index], ls.L[i]) = dir_and_pdf[i]; })

            simd_fvec<S> tex_col[3] = {1.0f, 1.0f, 1.0f};
            if (sc.env.env_map != 0xffffffff) {
                SampleLatlong_RGBE(*static_cast<const Cpu::TexStorageRGBA *>(textures[0]), sc.env.env_map, ls.L,
                                   sc.env.env_map_rotation, rand_tex_uv, ray_queue[index], tex_col);
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

    where(ray_mask, ls.pdf) /= factor;
}

template <int S>
void Ray::NS::IntersectAreaLights(const ray_data_t<S> &r, Span<const light_t> lights,
                                  Span<const light_wbvh_node_t> nodes, hit_data_t<S> &inout_inter) {
    const int SS = S <= 8 ? S : 8;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(r.o, r.d, inv_d, inv_d_o);

    alignas(S * 4) int ray_masks[S];
    alignas(S * 4) float inter_t[S];
    r.mask.store_to(ray_masks, simd_mem_aligned);
    inout_inter.t.store_to(inter_t, simd_mem_aligned);

    for (int ri = 0; ri < S; ri++) {
        if (!ray_masks[ri]) {
            continue;
        }

        // recombine in AoS layout
        const float _ro[3] = {r.o[0][ri], r.o[1][ri], r.o[2][ri]},
                    _inv_d[3] = {inv_d[0][ri], inv_d[1][ri], inv_d[2][ri]},
                    _inv_d_o[3] = {inv_d_o[0][ri], inv_d_o[1][ri], inv_d_o[2][ri]};

        TraversalStateStack_Single<MAX_STACK_SIZE, light_stack_entry_t> st;
        st.push(0u /* root_index */, 0.0f, 1.0f);

        while (!st.empty()) {
            light_stack_entry_t cur = st.pop();

            if (cur.dist > inter_t[ri] || cur.factor == 0.0f) {
                continue;
            }

        TRAVERSE:
            if (!is_leaf_node(nodes[cur.index])) {
                alignas(32) float res_dist[8];
                long mask = bbox_test_oct<SS>(_inv_d, _inv_d_o, inter_t[ri], nodes[cur.index].bbox_min,
                                              nodes[cur.index].bbox_max, res_dist);
                if (mask) {
                    simd_fvec<SS> importance[8 / SS];
                    calc_lnode_importance<SS>(nodes[cur.index], _ro, value_ptr(importance[0]));

                    simd_fvec<SS> total_importance_v = 0.0f;
                    UNROLLED_FOR_S(i, 8 / SS, { total_importance_v += importance[i]; })
                    const float total_importance = hsum(total_importance_v);
                    assert(total_importance > 0.0f);

                    alignas(32) float factors[8];
                    UNROLLED_FOR_S(i, 8 / SS, {
                        importance[i] /= total_importance;
                        importance[i].store_to(&factors[i * SS], simd_mem_aligned);
                    })

                    long i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    if (mask == 0) { // only one box was hit
                        cur.index = nodes[cur.index].child[i];
                        cur.factor *= factors[i];
                        goto TRAVERSE;
                    }

                    const long i2 = GetFirstBit(mask);
                    mask = ClearBit(mask, i2);
                    if (mask == 0) { // two boxes were hit
                        if (res_dist[i] < res_dist[i2]) {
                            st.push(nodes[cur.index].child[i2], res_dist[i2], cur.factor * factors[i2]);
                            cur.index = nodes[cur.index].child[i];
                            cur.factor *= factors[i];
                        } else {
                            st.push(nodes[cur.index].child[i], res_dist[i], cur.factor * factors[i]);
                            cur.index = nodes[cur.index].child[i2];
                            cur.factor *= factors[i2];
                        }
                        goto TRAVERSE;
                    }

                    st.push(nodes[cur.index].child[i], res_dist[i], cur.factor * factors[i]);
                    st.push(nodes[cur.index].child[i2], res_dist[i2], cur.factor * factors[i2]);

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // three boxes were hit
                        st.sort_top3();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    i = GetFirstBit(mask);
                    mask = ClearBit(mask, i);
                    st.push(nodes[cur.index].child[i], res_dist[i], cur.factor * factors[i]);
                    if (mask == 0) { // four boxes were hit
                        st.sort_top4();
                        cur = st.pop();
                        goto TRAVERSE;
                    }

                    const uint32_t size_before = st.stack_size;

                    // from five to eight boxes were hit
                    do {
                        i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        st.push(nodes[cur.index].child[i], res_dist[i], cur.factor * factors[i]);
                    } while (mask != 0);

                    const int count = int(st.stack_size - size_before + 4);
                    st.sort_topN(count);
                    cur = st.pop();
                    goto TRAVERSE;
                }
            } else {
                const uint32_t light_index = (nodes[cur.index].child[0] & PRIM_INDEX_BITS);
                assert(nodes[cur.index].child[1] == 1);
                const light_t &l = lights[light_index];
                if (!l.visible) {
                    continue;
                }
                // Portal lights affect only missed rays
                // TODO: actually process multiple rays
                simd_ivec<S> ray_mask = 0;
                ray_mask.set(ri, -1);
                ray_mask &= ~(simd_ivec<S>{l.sky_portal ? -1 : 0} & simd_cast(inout_inter.v >= 0.0f));
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
                                        _angle.template set<i>(acosf(_dot.template get<i>()));
                                    }
                                })
                                mask1 &= (_angle <= l.sph.spot);
                            }
                        }

                        where(mask1 | mask2, inout_inter.v) = 0.0f;
                        where(mask1 | mask2, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                        where(mask1, inout_inter.t) = t1;
                        where(mask2, inout_inter.t) = t2;
                        where(mask1 | mask2, inout_inter.u) = cur.factor;
                        inout_inter.t.store_to(inter_t, simd_mem_aligned);
                    }
                } else if (l.type == LIGHT_TYPE_DIR) {
                    const simd_fvec<S> cos_theta = dot3(r.d, l.dir.dir);
                    const simd_ivec<S> imask = simd_cast(cos_theta > cosf(l.dir.angle)) & ray_mask &
                                               (simd_cast(inout_inter.v < 0.0f) | simd_cast(no_shadow));
                    where(imask, inout_inter.v) = 0.0f;
                    where(imask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                    where(imask, inout_inter.t) = safe_div_pos(1.0f, cos_theta);
                    where(imask, inout_inter.u) = cur.factor;
                    inout_inter.t.store_to(inter_t, simd_mem_aligned);
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

                        where(final_mask, inout_inter.v) = 0.0f;
                        where(final_mask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                        where(final_mask, inout_inter.t) = t;
                        where(final_mask, inout_inter.u) = cur.factor;
                        inout_inter.t.store_to(inter_t, simd_mem_aligned);
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

                        where(final_mask, inout_inter.v) = 0.0f;
                        where(final_mask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                        where(final_mask, inout_inter.t) = t;
                        where(final_mask, inout_inter.u) = cur.factor;
                        inout_inter.t.store_to(inter_t, simd_mem_aligned);
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

                    imask &= simd_cast(abs(p[0]) < 0.5f * l.line.height) & simd_cast((t < inout_inter.t) | no_shadow) &
                             ray_mask;

                    where(imask, inout_inter.v) = 0.0f;
                    where(imask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                    where(imask, inout_inter.t) = t;
                    where(imask, inout_inter.u) = cur.factor;
                    inout_inter.t.store_to(inter_t, simd_mem_aligned);
                } else if (l.type == LIGHT_TYPE_ENV) {
                    // NOTE: mask remains empty
                    where(simd_cast(inout_inter.v < 0.0f) & ray_mask, inout_inter.obj_index) =
                        -simd_ivec<S>(light_index) - 1;
                    where(simd_cast(inout_inter.v < 0.0f) & ray_mask, inout_inter.u) = cur.factor;
                }
            }
        }
    }
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::IntersectAreaLights(const shadow_ray_t<S> &r, Span<const light_t> lights,
                                                   Span<const light_wbvh_node_t> nodes) {
    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(r.o, r.d, inv_d, inv_d_o);

    const simd_fvec<S> rdist = abs(r.dist);
    const simd_ivec<S> env_ray = simd_cast(r.dist < 0.0f);
    simd_fvec<S> ret = 1.0f;

    simd_ivec<S> ray_mask = r.mask;

    alignas(S * 4) int ray_masks[S];
    alignas(S * 4) float inter_t[S];
    ray_mask.store_to(ray_masks, simd_mem_aligned);
    rdist.store_to(inter_t, simd_mem_aligned);

    for (int ri = 0; ri < S; ri++) {
        if (!ray_masks[ri]) {
            continue;
        }

        // recombine in AoS layout
        const float _inv_d[3] = {inv_d[0][ri], inv_d[1][ri], inv_d[2][ri]},
                    _inv_d_o[3] = {inv_d_o[0][ri], inv_d_o[1][ri], inv_d_o[2][ri]};

        TraversalStateStack_Single<MAX_STACK_SIZE> st;
        st.push(0u /* root_index */, 0.0f);

        while (!st.empty() && ray_masks[ri]) {
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
                const int light_index = int(nodes[cur.index].child[0] & PRIM_INDEX_BITS);
                assert(nodes[cur.index].child[1] == 1);
                const light_t &l = lights[light_index];
                if (!l.blocking) {
                    continue;
                }
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
                        ray_mask.store_to(ray_masks, simd_mem_aligned);
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
                        ray_mask.store_to(ray_masks, simd_mem_aligned);
                    }
                }
            }
        }
    }

    return ret;
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::EvalTriLightFactor(const simd_fvec<S> P[3], const simd_fvec<S> ro[3],
                                                  const simd_ivec<S> &mask, const simd_ivec<S> &tri_index,
                                                  Span<const light_t> lights, Span<const light_wbvh_node_t> nodes) {
    const int SS = S <= 8 ? S : 8;

    simd_fvec<S> ret = 1.0f;

    for (int ri = 0; ri < S; ri++) {
        if (!mask[ri]) {
            continue;
        }

        // recombine in AoS layout
        const float _p[3] = {P[0][ri], P[1][ri], P[2][ri]}, _ro[3] = {ro[0][ri], ro[1][ri], ro[2][ri]};

        uint32_t stack[MAX_STACK_SIZE];
        float stack_factors[MAX_STACK_SIZE];
        uint32_t stack_size = 0;

        stack_factors[stack_size] = 1.0f;
        stack[stack_size++] = 0;

        while (stack_size) {
            const uint32_t cur = stack[--stack_size];
            const float cur_factor = stack_factors[stack_size];

            if (!is_leaf_node(nodes[cur])) {
                long mask = bbox_test_oct<S>(_p, nodes[cur].bbox_min, nodes[cur].bbox_max);
                if (mask) {
                    alignas(32) float importance[8];
                    calc_lnode_importance<SS>(nodes[cur], _ro, importance);

                    float total_importance = 0.0f;
                    UNROLLED_FOR(j, 8, { total_importance += importance[j]; })
                    assert(total_importance > 0.0f);

                    do {
                        const long i = GetFirstBit(mask);
                        mask = ClearBit(mask, i);
                        if (importance[i] > 0.0f) {
                            stack_factors[stack_size] = cur_factor * importance[i] / total_importance;
                            stack[stack_size++] = nodes[cur].child[i];
                        }
                    } while (mask != 0);
                }
            } else {
                const int light_index = int(nodes[cur].child[0] & PRIM_INDEX_BITS);
                assert((nodes[cur].child[1] & PRIM_COUNT_BITS) == 1);

                const light_t &l = lights[light_index];
                if (l.type == LIGHT_TYPE_TRI && l.tri.tri_index == tri_index[ri]) {
                    // needed triangle found
                    ret.set(ri, 1.0f / cur_factor);
                }
            }
        }
    }

    return ret;
}

template <int S>
void Ray::NS::Evaluate_EnvColor(const ray_data_t<S> &ray, const simd_ivec<S> &mask, const environment_t &env,
                                const Cpu::TexStorageRGBA &tex_storage, const simd_fvec<S> &pdf_factor,
                                const simd_fvec<S> rand[2], simd_fvec<S> env_col[4]) {
    const uint32_t env_map = env.env_map;
    const float env_map_rotation = env.env_map_rotation;
    const simd_ivec<S> env_map_mask = is_indirect(ray.depth);

    if ((mask & env_map_mask).not_all_zeros()) {
        UNROLLED_FOR(i, 3, { env_col[i] = 1.0f; })
        if (env_map != 0xffffffff) {
            SampleLatlong_RGBE(tex_storage, env_map, ray.d, env_map_rotation, rand, (mask & env_map_mask), env_col);
        }
#if USE_NEE
        const simd_ivec<S> mis_mask =
            simd_ivec<S>((env.light_index != 0xffffffff) ? -1 : 0) & simd_cast(pdf_factor >= 0.0f) & env_map_mask;
        if (mis_mask.not_all_zeros()) {
            if (env.qtree_levels) {
                const auto *qtree_mips = reinterpret_cast<const simd_fvec4 *const *>(env.qtree_mips);

                const simd_fvec<S> light_pdf =
                    safe_div_pos(Evaluate_EnvQTree(env_map_rotation, qtree_mips, env.qtree_levels, ray.d), pdf_factor);
                const simd_fvec<S> bsdf_pdf = ray.pdf;

                const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                UNROLLED_FOR(i, 3, { where(mis_mask, env_col[i]) *= mis_weight; })
            } else {
                const simd_fvec<S> light_pdf = safe_div_pos(0.5f, PI * pdf_factor);
                const simd_fvec<S> bsdf_pdf = ray.pdf;

                const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                UNROLLED_FOR(i, 3, { where(mis_mask, env_col[i]) *= mis_weight; })
            }
        }
#endif
    }
    UNROLLED_FOR(i, 3, { where(env_map_mask, env_col[i]) *= env.env_col[i]; })

    const uint32_t back_map = env.back_map;
    const float back_map_rotation = env.back_map_rotation;
    const simd_ivec<S> back_map_mask = ~env_map_mask;

    if (back_map != 0xffffffff && (mask & back_map_mask).not_all_zeros()) {
        simd_fvec<S> back_col[3] = {};
        SampleLatlong_RGBE(tex_storage, back_map, ray.d, back_map_rotation, rand, (mask & back_map_mask), back_col);
        UNROLLED_FOR(i, 3, { where(back_map_mask, env_col[i]) = back_col[i]; })
    }
    UNROLLED_FOR(i, 3, { where(back_map_mask, env_col[i]) *= env.back_col[i]; })
}

template <int S>
void Ray::NS::Evaluate_LightColor(const simd_fvec<S> P[3], const ray_data_t<S> &ray, const simd_ivec<S> &mask,
                                  const hit_data_t<S> &inter, const environment_t &env, Span<const light_t> lights,
                                  const uint32_t lights_count, const Cpu::TexStorageRGBA &tex_storage,
                                  const simd_fvec<S> rand[2], simd_fvec<S> light_col[3]) {
#if USE_HIERARCHICAL_NEE
    const simd_fvec<S> pdf_factor = safe_div_pos(1.0f, inter.u);
#else
    const float pdf_factor = float(lights_count);
#endif

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
                SampleLatlong_RGBE(tex_storage, env.env_map, ray.d, env.env_map_rotation, rand, ray_queue[index],
                                   tex_col);
                UNROLLED_FOR(i, 3, { lcol[i] *= tex_col[i]; })
            }
        }

        if (l.type == LIGHT_TYPE_SPHERE) {
            simd_fvec<S> disk_normal[3] = {ray.o[0] - l.sph.pos[0], ray.o[1] - l.sph.pos[1], ray.o[2] - l.sph.pos[2]};
            normalize(disk_normal);
            const simd_fvec<S> disk_dist = dot3(ray.o, disk_normal) - dot3(l.sph.pos, disk_normal);

            const simd_fvec<S> light_pdf =
                safe_div(disk_dist * disk_dist, PI * l.sph.radius * l.sph.radius * pdf_factor);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; })

            if (l.sph.spot > 0.0f && l.sph.blend > 0.0f) {
                const simd_fvec<S> _dot =
                    -(ray.d[0] * l.sph.dir[0] + ray.d[1] * l.sph.dir[1] + ray.d[2] * l.sph.dir[2]);
                assert((ray_queue[index] & simd_cast(_dot <= 0.0f)).all_zeros());

                simd_fvec<S> _angle = 0.0f;
                UNROLLED_FOR_S(i, S, {
                    if (ray_queue[index].template get<i>()) {
                        _angle.template set<i>(acosf(_dot.template get<i>()));
                    }
                })
                assert((ray_queue[index] & simd_cast(_angle > l.sph.spot)).all_zeros());
                if (l.sph.blend > 0.0f) {
                    const simd_fvec<S> spot_weight = saturate((l.sph.spot - _angle) / l.sph.blend);
                    UNROLLED_FOR(i, 3, { lcol[i] *= spot_weight; })
                }
            }
        } else if (l.type == LIGHT_TYPE_DIR) {
            const float radius = tanf(l.dir.angle);
            const float light_area = PI * radius * radius;

            const simd_fvec<S> cos_theta = dot3(ray.d, l.dir.dir);

            const simd_fvec<S> light_pdf = safe_div(simd_fvec<S>{1.0f}, light_area * cos_theta * pdf_factor);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; })
        } else if (l.type == LIGHT_TYPE_RECT) {
            float light_fwd[3];
            cross(l.rect.u, l.rect.v, light_fwd);
            normalize(light_fwd);

            const simd_fvec<S> cos_theta = dot3(ray.d, light_fwd);

            simd_fvec<S> light_pdf = 0.0f;
#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
            const simd_fvec<S> vp[3] = {l.rect.pos[0], l.rect.pos[1], l.rect.pos[2]},
                               vu[3] = {l.rect.u[0], l.rect.u[1], l.rect.u[2]},
                               vv[3] = {l.rect.v[0], l.rect.v[1], l.rect.v[2]};
            light_pdf = SampleSphericalRectangle<S>(ray.o, vp, vu, vv, nullptr, nullptr) / pdf_factor;
#endif
            where(light_pdf == 0.0f, light_pdf) = safe_div(inter.t * inter.t, l.rect.area * cos_theta * pdf_factor);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; })
        } else if (l.type == LIGHT_TYPE_DISK) {
            float light_fwd[3];
            cross(l.disk.u, l.disk.v, light_fwd);
            normalize(light_fwd);

            const simd_fvec<S> cos_theta = dot3(ray.d, light_fwd);

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, l.disk.area * cos_theta * pdf_factor);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; })
        } else if (l.type == LIGHT_TYPE_LINE) {
            const float *light_dir = l.line.v;

            const simd_fvec<S> cos_theta = 1.0f - abs(dot3(ray.d, light_dir));

            const simd_fvec<S> light_pdf = safe_div(inter.t * inter.t, l.line.area * cos_theta * pdf_factor);
            const simd_fvec<S> bsdf_pdf = ray.pdf;

            const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            UNROLLED_FOR(i, 3, { lcol[i] *= mis_weight; })
        }

        where(ray_queue[index], light_col[0]) = lcol[0];
        where(ray_queue[index], light_col[1]) = lcol[1];
        where(ray_queue[index], light_col[2]) = lcol[2];

        ++index;
    }
};

template <int S>
Ray::NS::simd_ivec<S> Ray::NS::Evaluate_DiffuseNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray,
                                                    const simd_ivec<S> &mask, const surface_t<S> &surf,
                                                    const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                                                    const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask,
                                                    simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};

    simd_fvec<S> diff_col[4];
    Evaluate_OrenDiffuse_BSDF(nI, surf.N, ls.L, roughness, base_color, diff_col);
    const simd_fvec<S> &bsdf_pdf = diff_col[3];

    const simd_fvec<S> mis_weight =
        select(mis_mask & simd_cast(ls.area > 0.0f), power_heuristic(ls.pdf, bsdf_pdf), simd_fvec<S>{1.0f});

    simd_fvec<S> P_biased[3];
    offset_ray(surf.P, surf.plane_N, P_biased);

    UNROLLED_FOR(i, 3, { where(mask, sh_r.o[i]) = P_biased[i]; })
    UNROLLED_FOR(i, 3, {
        const simd_fvec<S> temp = ls.col[i] * diff_col[i] * safe_div_pos(mix_weight * mis_weight, ls.pdf);
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

    where(mask, new_ray.depth) = pack_ray_type(RAY_TYPE_DIFFUSE);
    where(mask, new_ray.depth) |=
        mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0});

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
Ray::NS::Evaluate_GlossyNode(const light_sample_t<S> &ls, const ray_data_t<S> &ray, simd_ivec<S> mask,
                             const surface_t<S> &surf, const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                             const simd_fvec<S> &regularize_alpha, const simd_fvec<S> &spec_ior,
                             const simd_fvec<S> &spec_F0, const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask,
                             simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};
    simd_fvec<S> H[3] = {ls.L[0] - ray.d[0], ls.L[1] - ray.d[1], ls.L[2] - ray.d[2]};
    safe_normalize(H);

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    simd_fvec<S> spec_col[4], alpha[2];
    calc_alpha(roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
    mask &= simd_cast(alpha[0] * alpha[1] >= 1e-7f);

    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, alpha, simd_fvec<S>{spec_ior},
                              simd_fvec<S>{spec_F0}, base_color, base_color, spec_col);
    const simd_fvec<S> &bsdf_pdf = spec_col[3];

    const simd_fvec<S> mis_weight =
        select(mis_mask & simd_cast(ls.area > 0.0f), power_heuristic(ls.pdf, bsdf_pdf), simd_fvec<S>{1.0f});

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
                                const simd_fvec<S> &regularize_alpha, const simd_fvec<S> &spec_ior,
                                const simd_fvec<S> &spec_F0, const simd_fvec<S> rand[2], const simd_fvec<S> &mix_weight,
                                ray_data_t<S> &new_ray) {
    simd_fvec<S> alpha[2];
    calc_alpha(roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);

    simd_fvec<S> V[3], F[4];
    Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, alpha, spec_ior, spec_F0, base_color, base_color, rand, V,
                            F);

    where(mask, new_ray.depth) = pack_ray_type(RAY_TYPE_SPECULAR);
    where(mask, new_ray.depth) |=
        mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0});

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
                                                       const simd_fvec<S> base_color[3], const simd_fvec<S> &roughness,
                                                       const simd_fvec<S> &regularize_alpha, const simd_fvec<S> &eta,
                                                       const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask,
                                                       simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
    const simd_fvec<S> nI[3] = {-ray.d[0], -ray.d[1], -ray.d[2]};
    simd_fvec<S> H[3] = {ls.L[0] - ray.d[0] * eta, ls.L[1] - ray.d[1] * eta, ls.L[2] - ray.d[2] * eta};
    safe_normalize(H);

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    simd_fvec<S> refr_col[4], alpha[2];
    calc_alpha(roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
    Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, alpha, simd_fvec<S>{eta}, base_color,
                                refr_col);
    const simd_fvec<S> &bsdf_pdf = refr_col[3];

    const simd_fvec<S> mis_weight =
        select(mis_mask & simd_cast(ls.area > 0.0f), power_heuristic(ls.pdf, bsdf_pdf), simd_fvec<S>{1.0f});

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
                                    const simd_fvec<S> &regularize_alpha, const simd_ivec<S> &is_backfacing,
                                    const simd_fvec<S> &int_ior, const simd_fvec<S> &ext_ior,
                                    const simd_fvec<S> rand[2], const simd_fvec<S> &mix_weight,
                                    ray_data_t<S> &new_ray) {
    const simd_fvec<S> eta = select(is_backfacing, (int_ior / ext_ior), (ext_ior / int_ior));

    simd_fvec<S> V[4], F[4], alpha[2];
    calc_alpha(roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
    Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, ray.d, alpha, eta, base_color, rand, V, F);

    where(mask, new_ray.depth) = pack_ray_type(RAY_TYPE_REFR);
    where(mask, new_ray.depth) |=
        mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0});

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
    const float transmission, const simd_fvec<S> &N_dot_L, const simd_fvec<S> &mix_weight, const simd_ivec<S> &mis_mask,
    const simd_fvec<S> &regularize_alpha, simd_fvec<S> out_col[3], shadow_ray_t<S> &sh_r) {
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
            diff_col[i] *= (1.0f - metallic) * (1.0f - transmission);
            where(eval_diff_lobe, lcol[i]) += safe_div_pos(ls.col[i] * N_dot_L * diff_col[i], PI * ls.pdf);
        })
    }

    simd_fvec<S> H[3];
    UNROLLED_FOR(i, 3, { H[i] = select(_is_frontfacing, ls.L[i] - ray.d[i], ls.L[i] - ray.d[i] * trans.eta); })
    safe_normalize(H);

    const simd_fvec<S> spec_col_90[3] = {1.0f, 1.0f, 1.0f};

    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
    tangent_from_world(surf.T, surf.B, surf.N, nI, view_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, ls.L, light_dir_ts);
    tangent_from_world(surf.T, surf.B, surf.N, H, sampled_normal_ts);

    simd_fvec<S> spec_alpha[2];
    calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha, spec_alpha);
    const simd_ivec<S> eval_spec_lobe = simd_cast(lobe_weights.specular > 0.0f) &
                                        simd_cast(spec_alpha[0] * spec_alpha[1] >= 1e-7f) & _is_frontfacing & mask;
    if (eval_spec_lobe.not_all_zeros()) {
        simd_fvec<S> spec_col[4], _alpha[2] = {max(spec_alpha[0], 1e-7f), max(spec_alpha[1], 1e-7f)};
        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, _alpha, spec.ior, spec.F0, spec.tmp_col,
                                  spec_col_90, spec_col);

        where(eval_spec_lobe, bsdf_pdf) += lobe_weights.specular * spec_col[3];

        UNROLLED_FOR(i, 3, { where(eval_spec_lobe, lcol[i]) += safe_div_pos(ls.col[i] * spec_col[i], ls.pdf); })
    }

    simd_fvec<S> coat_alpha[2];
    calc_alpha(coat.roughness, simd_fvec<S>{0.0f}, regularize_alpha, coat_alpha);
    const simd_ivec<S> eval_coat_lobe = simd_cast(lobe_weights.clearcoat > 0.0f) &
                                        simd_cast(coat_alpha[0] * coat_alpha[1] >= 1e-7f) & _is_frontfacing & mask;
    if (eval_coat_lobe.not_all_zeros()) {
        simd_fvec<S> clearcoat_col[4];
        Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, coat_alpha[0], coat.ior,
                                          coat.F0, clearcoat_col);

        where(eval_coat_lobe, bsdf_pdf) += lobe_weights.clearcoat * clearcoat_col[3];

        UNROLLED_FOR(i, 3,
                     { where(eval_coat_lobe, lcol[i]) += safe_div_pos(0.25f * ls.col[i] * clearcoat_col[i], ls.pdf); })
    }

    simd_fvec<S> refr_spec_alpha[2];
    calc_alpha(spec.roughness, simd_fvec<S>{0.0f}, regularize_alpha, refr_spec_alpha);
    const simd_ivec<S> eval_refr_spec_lobe =
        simd_cast(trans.fresnel != 0.0f) & simd_cast(lobe_weights.refraction > 0.0f) &
        simd_cast(refr_spec_alpha[0] * refr_spec_alpha[1] >= 1e-7f) & _is_frontfacing & mask;
    if (eval_refr_spec_lobe.not_all_zeros()) {
        simd_fvec<S> spec_col[4], spec_temp_col[3] = {1.0f, 1.0f, 1.0f};
        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, refr_spec_alpha,
                                  simd_fvec<S>{1.0f} /* ior */, simd_fvec<S>{0.0f} /* F0 */, spec_temp_col, spec_col_90,
                                  spec_col);
        where(eval_refr_spec_lobe, bsdf_pdf) += lobe_weights.refraction * trans.fresnel * spec_col[3];

        UNROLLED_FOR(i, 3, {
            where(eval_refr_spec_lobe, lcol[i]) += ls.col[i] * spec_col[i] * safe_div_pos(trans.fresnel, ls.pdf);
        })
    }

    simd_fvec<S> refr_trans_alpha[2];
    calc_alpha(trans.roughness, simd_fvec<S>{0.0f}, regularize_alpha, refr_trans_alpha);
    const simd_ivec<S> eval_refr_trans_lobe =
        simd_cast(trans.fresnel != 1.0f) & simd_cast(lobe_weights.refraction > 0.0f) &
        simd_cast(refr_trans_alpha[0] * refr_trans_alpha[1] >= 1e-7f) & _is_backfacing & mask;
    if (eval_refr_trans_lobe.not_all_zeros()) {
        simd_fvec<S> refr_col[4];
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, refr_trans_alpha, trans.eta,
                                    diff.base_color, refr_col);
        where(eval_refr_trans_lobe, bsdf_pdf) += lobe_weights.refraction * (1.0f - trans.fresnel) * refr_col[3];

        UNROLLED_FOR(i, 3, {
            where(eval_refr_trans_lobe, lcol[i]) +=
                ls.col[i] * refr_col[i] * safe_div_pos(1.0f - trans.fresnel, ls.pdf);
        })
    }

    const simd_fvec<S> mis_weight =
        select(mis_mask & simd_cast(ls.area > 0.0f), power_heuristic(ls.pdf, bsdf_pdf), simd_fvec<S>{1.0f});
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
                                    const simd_fvec<S> &metallic, const float transmission, const simd_fvec<S> rand[2],
                                    simd_fvec<S> mix_rand, const simd_fvec<S> &mix_weight,
                                    const simd_fvec<S> &regularize_alpha, simd_ivec<S> &secondary_mask,
                                    ray_data_t<S> &new_ray) {
    const simd_ivec<S> diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
                       refr_depth = get_refr_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const simd_ivec<S> total_depth = diff_depth + spec_depth + refr_depth;

    const simd_ivec<S> sample_diff_lobe = (diff_depth < ps.max_diff_depth) & (total_depth < ps.max_total_depth) &
                                          simd_cast(mix_rand < lobe_weights.diffuse) & mask;
    if (sample_diff_lobe.not_all_zeros()) {
        simd_fvec<S> V[3], F[4];
        Sample_PrincipledDiffuse_BSDF(surf.T, surf.B, surf.N, ray.d, diff.roughness, diff.base_color, diff.sheen_color,
                                      false, rand, V, F);
        // F[3] *= lobe_weights.diffuse;

        UNROLLED_FOR(i, 3, { F[i] *= (1.0f - metallic) * (1.0f - transmission); })

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_diff_lobe, new_ray.depth) = pack_ray_type(RAY_TYPE_DIFFUSE);
        where(sample_diff_lobe, new_ray.depth) |=
            mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{0});

        UNROLLED_FOR(i, 3, {
            where(sample_diff_lobe, new_ray.o[i]) = new_p[i];
            where(sample_diff_lobe, new_ray.d[i]) = V[i];
            where(sample_diff_lobe, new_ray.c[i]) = safe_div_pos(ray.c[i] * F[i] * mix_weight, lobe_weights.diffuse);
        })
        where(sample_diff_lobe, new_ray.pdf) = F[3];

        assert((secondary_mask & sample_diff_lobe).all_zeros());
        secondary_mask |= sample_diff_lobe;
    }

    const simd_ivec<S> sample_spec_lobe = (spec_depth < ps.max_spec_depth) & (total_depth < ps.max_total_depth) &
                                          simd_cast(mix_rand >= lobe_weights.diffuse) &
                                          simd_cast(mix_rand < lobe_weights.diffuse + lobe_weights.specular) & mask;
    if (sample_spec_lobe.not_all_zeros()) {
        const simd_fvec<S> spec_col_90[3] = {1.0f, 1.0f, 1.0f};

        simd_fvec<S> V[3], F[4], alpha[2];
        calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha, alpha);
        Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, alpha, spec.ior, spec.F0, spec.tmp_col, spec_col_90,
                                rand, V, F);
        F[3] *= lobe_weights.specular;

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_spec_lobe, new_ray.depth) = pack_ray_type(RAY_TYPE_SPECULAR);
        where(sample_spec_lobe, new_ray.depth) |=
            mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0});

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
        simd_fvec<S> V[3], F[4], alpha[2];
        calc_alpha(coat.roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
        Sample_PrincipledClearcoat_BSDF(surf.T, surf.B, surf.N, ray.d, alpha[0], coat.ior, coat.F0, rand, V, F);
        F[3] *= lobe_weights.clearcoat;

        simd_fvec<S> new_p[3];
        offset_ray(surf.P, surf.plane_N, new_p);

        where(sample_spec_lobe, new_ray.depth) = pack_ray_type(RAY_TYPE_SPECULAR);
        where(sample_coat_lobe, new_ray.depth) |=
            mask_ray_depth(ray.depth) + pack_depth(simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0});

        UNROLLED_FOR(i, 3, {
            where(sample_coat_lobe, new_ray.o[i]) = new_p[i];
            where(sample_coat_lobe, new_ray.d[i]) = V[i];
            where(sample_coat_lobe, new_ray.c[i]) = 0.25f * ray.c[i] * F[i] * safe_div_pos(mix_weight, F[3]);
        })
        where(sample_coat_lobe, new_ray.pdf) = F[3];

        assert((secondary_mask & sample_coat_lobe).all_zeros());
        secondary_mask |= sample_coat_lobe;
    }

    simd_ivec<S> sample_trans_lobe =
        simd_cast(mix_rand >= lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat) &
        (total_depth < ps.max_total_depth) & mask;

    mix_rand -= lobe_weights.diffuse + lobe_weights.specular + lobe_weights.clearcoat;
    mix_rand = safe_div_pos(mix_rand, lobe_weights.refraction);

    sample_trans_lobe &= ((simd_cast(mix_rand >= trans.fresnel) & (refr_depth < ps.max_refr_depth)) |
                          (simd_cast(mix_rand < trans.fresnel) & (spec_depth < ps.max_spec_depth)));
    if (sample_trans_lobe.not_all_zeros()) {
        simd_fvec<S> F[4] = {}, V[3] = {};

        const simd_ivec<S> sample_trans_spec_lobe = simd_cast(mix_rand < trans.fresnel) & sample_trans_lobe;
        if (sample_trans_spec_lobe.not_all_zeros()) {
            simd_fvec<S> _spec_tmp_col[3] = {1.0f, 1.0f, 1.0f}, alpha[2];
            calc_alpha(spec.roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
            Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, ray.d, alpha, simd_fvec<S>{1.0f} /* ior */,
                                    simd_fvec<S>{0.0f} /* F0 */, _spec_tmp_col, _spec_tmp_col, rand, V, F);

            simd_fvec<S> new_p[3];
            offset_ray(surf.P, surf.plane_N, new_p);

            where(sample_trans_spec_lobe, new_ray.depth) = pack_ray_type(RAY_TYPE_SPECULAR);
            where(sample_trans_spec_lobe, new_ray.depth) |=
                mask_ray_depth(ray.depth) +
                pack_depth(simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0}, simd_ivec<S>{0});

            UNROLLED_FOR(i, 3, { where(sample_trans_spec_lobe, new_ray.o[i]) = new_p[i]; })
        }

        const simd_ivec<S> sample_trans_refr_lobe = ~sample_trans_spec_lobe & sample_trans_lobe;
        if (sample_trans_refr_lobe.not_all_zeros()) {
            simd_fvec<S> temp_F[4], temp_V[4], alpha[2];
            calc_alpha(trans.roughness, simd_fvec<S>{0.0f}, regularize_alpha, alpha);
            Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, ray.d, alpha, trans.eta, diff.base_color, rand, temp_V,
                                      temp_F);

            const simd_fvec<S> _plane_N[3] = {-surf.plane_N[0], -surf.plane_N[1], -surf.plane_N[2]};
            simd_fvec<S> new_p[3];
            offset_ray(surf.P, _plane_N, new_p);

            where(sample_trans_refr_lobe, new_ray.depth) = pack_ray_type(RAY_TYPE_REFR);
            where(sample_trans_refr_lobe, new_ray.depth) |=
                mask_ray_depth(ray.depth) +
                pack_depth(simd_ivec<S>{0}, simd_ivec<S>{0}, simd_ivec<S>{1}, simd_ivec<S>{0});

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
void Ray::NS::ShadeSurface(const pass_settings_t &ps, const float limits[2], const uint32_t rand_seq[],
                           const uint32_t rand_seed, const int iteration, const hit_data_t<S> &inter,
                           const ray_data_t<S> &ray, const scene_data_t &sc, const uint32_t node_index,
                           const Cpu::TexStorageBase *const textures[], simd_fvec<S> out_rgba[4],
                           ray_data_t<S> out_secondary_rays[], int *out_secondary_rays_count,
                           shadow_ray_t<S> out_shadow_rays[], int *out_shadow_rays_count,
                           simd_fvec<S> out_base_color[4], simd_fvec<S> out_depth_normals[4]) {
    out_rgba[0] = out_rgba[1] = out_rgba[2] = {0.0f};
    out_rgba[3] = {1.0f};

    // used to randomize random sequence among pixels
    const simd_uvec<S> px_hash = hash(ray.xy);
    const simd_uvec<S> rand_hash = hash_combine(px_hash, rand_seed);

    const simd_ivec<S> diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
                       refr_depth = get_refr_depth(ray.depth), transp_depth = get_transp_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const simd_ivec<S> total_depth = diff_depth + spec_depth + refr_depth;

    // offset of the sequence
    const auto rand_dim = simd_uvec<S>(RAND_DIM_BASE_COUNT + (total_depth + transp_depth) * RAND_DIM_BOUNCE_COUNT);

    simd_fvec<S> tex_rand[2];
    get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq, tex_rand);

    const simd_ivec<S> ino_hit = simd_cast(inter.v < 0.0f);
    if (ino_hit.not_all_zeros()) {
        simd_fvec<S> env_col[4] = {{1.0f}, {1.0f}, {1.0f}, {1.0f}};
        const simd_fvec<S> pdf_factor = select(total_depth < ps.max_total_depth,
#if USE_HIERARCHICAL_NEE
                                               safe_div_pos(1.0f, inter.u),
#else
                                               float(sc.li_indices.size()),
#endif
                                               simd_fvec<S>{-1.0f});
        Evaluate_EnvColor(ray, ino_hit, sc.env, *static_cast<const Cpu::TexStorageRGBA *>(textures[0]), pdf_factor,
                          tex_rand, env_col);
        UNROLLED_FOR(i, 3, { env_col[i] = ray.c[i] * env_col[i]; })

        const simd_fvec<S> sum = env_col[0] + env_col[1] + env_col[2];
        UNROLLED_FOR(i, 3, {
            where(sum > limits[0], env_col[i]) = safe_div_pos(env_col[i] * limits[0], sum);
            where(ino_hit, out_rgba[i]) = env_col[i];
        })
        where(ino_hit, out_rgba[3]) = env_col[3];
    }

    simd_ivec<S> is_active_lane = simd_cast(inter.v >= 0.0f);
    if (is_active_lane.all_zeros()) {
        return;
    }

    const simd_fvec<S> *I = ray.d;

    surface_t<S> surf;
    UNROLLED_FOR(i, 3, { where(inter.v >= 0.0f, surf.P[i]) = fmadd(inter.t, ray.d[i], ray.o[i]); })

    const simd_ivec<S> is_light_hit = is_active_lane & (inter.obj_index < 0); // Area light intersection
    if (is_light_hit.not_all_zeros()) {
        simd_fvec<S> light_col[3] = {};
        Evaluate_LightColor(surf.P, ray, is_light_hit, inter, sc.env, sc.lights, uint32_t(sc.li_indices.size()),
                            *static_cast<const Cpu::TexStorageRGBA *>(textures[0]), tex_rand, light_col);
        UNROLLED_FOR(i, 3, { light_col[i] = ray.c[i] * light_col[i]; })

        const simd_fvec<S> sum = light_col[0] + light_col[1] + light_col[2];
        UNROLLED_FOR(i, 3, {
            where(sum > limits[0], light_col[i]) = safe_div_pos(light_col[i] * limits[0], sum);
            where(is_light_hit, out_rgba[i]) = light_col[i];
        })
        where(is_light_hit, out_rgba[3]) = 1.0f;

        is_active_lane &= ~is_light_hit;
    }

    if (is_active_lane.all_zeros()) {
        return;
    }

    const simd_ivec<S> is_backfacing = (inter.prim_index < 0);
    const simd_ivec<S> tri_index = select(is_backfacing, -inter.prim_index - 1, inter.prim_index);

    const simd_ivec<S> obj_index = select(is_active_lane, inter.obj_index, simd_ivec<S>{0});

    simd_ivec<S> mat_index = gather(reinterpret_cast<const int *>(sc.tri_materials), tri_index) &
                             simd_ivec<S>((MATERIAL_INDEX_BITS << 16) | MATERIAL_INDEX_BITS);

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
        })
    }

    FetchVertexAttribute3(&sc.vertices[0].n[0], vtx_indices, inter.u, inter.v, w, surf.N);
    safe_normalize(surf.N);

    simd_fvec<S> u1[2], u2[2], u3[2];
    { // Fetch vertex uvs
        const float *vtx_uvs = &sc.vertices[0].t[0];
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
    const simd_fvec<S> pa = normalize(surf.plane_N);

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
    })

    simd_fvec<S> tangent[3] = {-P_ls[2], {0.0f}, P_ls[0]};

    simd_fvec<S> transform[16], ro_ls[3] = {ray.o[0], ray.o[1], ray.o[2]};
    FetchTransformAndRecalcBasis(sc.mesh_instances, obj_index, P_ls, surf.plane_N, surf.N, surf.B, surf.T, tangent,
                                 ro_ls, transform);

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

    const simd_fvec<S> ext_ior = peek_ior_stack(ray.ior, is_backfacing);

    simd_ivec<S> mat_type =
        gather(reinterpret_cast<const int *>(&sc.materials[0].type), mat_index * sizeof(material_t) / sizeof(int)) &
        0xff;

    simd_fvec<S> mix_term_rand[2];
    get_scrambled_2d_rand(rand_dim + unsigned(RAND_DIM_BSDF_PICK), rand_hash, iteration - 1, rand_seq, mix_term_rand);

    simd_fvec<S> mix_rand = mix_term_rand[0];
    simd_fvec<S> mix_weight = 1.0f;

    // resolve mix material
    simd_ivec<S> is_mix_mat = (mat_type == int(eShadingNode::Mix));
    while (is_mix_mat.not_all_zeros()) {
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
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(base_lod), tex_rand, ray_queue[index],
                               tex_color);
                if (first_t & TEX_YCOCG_BIT) {
                    YCoCg_to_RGB(tex_color, tex_color);
                }
                if (first_t & TEX_SRGB_BIT) {
                    srgb_to_rgb(tex_color, tex_color);
                }

                where(ray_queue[index], mix_val) *= tex_color[0];

                ++index;
            }
        }

        const float *iors = &sc.materials[0].ior;

        const simd_fvec<S> ior = gather(iors, mat_index * MatDWORDStride);

        const simd_fvec<S> eta = select(is_backfacing, safe_div_pos(ext_ior, ior), safe_div_pos(ior, ext_ior));
        const simd_fvec<S> RR = select(ior != 0.0f, fresnel_dielectric_cos(dot3(I, surf.N), eta), simd_fvec<S>{1.0f});

        mix_val *= saturate(RR);

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

        mat_type =
            gather(reinterpret_cast<const int *>(&sc.materials[0].type), mat_index * sizeof(material_t) / sizeof(int)) &
            0xff;
        is_mix_mat = (mat_type == int(eShadingNode::Mix));
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

                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>{0}, tex_rand, ray_queue[index], normals_tex);
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
    safe_normalize(surf.B);
    cross(surf.N, surf.B, surf.T);
#endif

#if USE_NEE
    light_sample_t<S> ls;
    if (!sc.light_wnodes.empty()) {
        simd_fvec<S> rand_pick_light[2];
        get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT_PICK, rand_hash, iteration - 1, rand_seq, rand_pick_light);

        simd_fvec<S> rand_light_uv[2];
        get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT, rand_hash, iteration - 1, rand_seq, rand_light_uv);

        SampleLightSource(surf.P, surf.T, surf.B, surf.N, sc, textures, rand_pick_light[0], rand_light_uv, tex_rand,
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
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(base_lod), tex_rand, ray_queue[index],
                               tex_color);
                if (first_t & TEX_YCOCG_BIT) {
                    YCoCg_to_RGB(tex_color, tex_color);
                }
                if (first_t & TEX_SRGB_BIT) {
                    srgb_to_rgb(tex_color, tex_color);
                }

                UNROLLED_FOR(i, 3, { where(ray_queue[index], base_color[i]) *= tex_color[i]; })

                ++index;
            }
        }
    }

    if (out_base_color) {
        UNROLLED_FOR(i, 3, { where(is_active_lane, out_base_color[i]) = base_color[i]; })
    }
    if (out_depth_normals) {
        UNROLLED_FOR(i, 3, { where(is_active_lane, out_depth_normals[i]) = surf.N[i]; })
        where(is_active_lane, out_depth_normals[3]) = inter.t;
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
                SampleBilinear(textures, first_t, surf.uvs, simd_ivec<S>(roughness_lod), tex_rand, ray_queue[index],
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

    simd_fvec<S> rand_uv[2];
    get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF, rand_hash, iteration - 1, rand_seq, rand_uv);

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
    new_ray.depth = simd_uvec<S>{0u};

    shadow_ray_t<S> &sh_r = out_shadow_rays[*out_shadow_rays_count];
    sh_r = {};
    sh_r.depth = ray.depth;
    sh_r.xy = ray.xy;

    simd_fvec<S> regularize_alpha = 0.0f;
    where(get_diff_depth(ray.depth) > 0, regularize_alpha) = ps.regularize_alpha;

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
            if (mat->type == eShadingNode::Diffuse) {
#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= Evaluate_DiffuseNode(ls, ray, eval_light, surf, base_color, roughness, mix_weight,
                                                        (total_depth < ps.max_total_depth), col, sh_r);
                }
#endif
                const simd_ivec<S> gen_ray =
                    (diff_depth < ps.max_diff_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_DiffuseNode(ray, gen_ray, surf, base_color, roughness, rand_uv[0], rand_uv[1], mix_weight,
                                       new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == eShadingNode::Glossy) {
                const float specular = 0.5f;
                const float spec_ior = (2.0f / (1.0f - sqrtf(0.08f * specular))) - 1.0f;
                const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);

#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= Evaluate_GlossyNode(ls, ray, eval_light, surf, base_color, roughness,
                                                       regularize_alpha, simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0},
                                                       mix_weight, (total_depth < ps.max_total_depth), col, sh_r);
                };
#endif

                const simd_ivec<S> gen_ray =
                    (spec_depth < ps.max_spec_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_GlossyNode(ray, gen_ray, surf, base_color, roughness, regularize_alpha,
                                      simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0}, rand_uv, mix_weight, new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == eShadingNode::Refractive) {
#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(N_dot_L < 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    assert((shadow_mask & eval_light).all_zeros());
                    const simd_fvec<S> eta = select(is_backfacing, mat->ior / ext_ior, ext_ior / mat->ior);
                    shadow_mask |=
                        Evaluate_RefractiveNode(ls, ray, eval_light, surf, base_color, roughness, regularize_alpha, eta,
                                                mix_weight, (total_depth < ps.max_total_depth), col, sh_r);
                }
#endif
                const simd_ivec<S> gen_ray =
                    (refr_depth < ps.max_refr_depth) & (total_depth < ps.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    Sample_RefractiveNode(ray, gen_ray, surf, base_color, roughness, regularize_alpha, is_backfacing,
                                          simd_fvec<S>{mat->ior}, ext_ior, rand_uv, mix_weight, new_ray);
                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == eShadingNode::Emissive) {
                simd_fvec<S> mis_weight = 1.0f;
#if USE_NEE
                if ((ray.depth & 0x00ffffff).not_all_zeros() && (mat->flags & MAT_FLAG_MULT_IMPORTANCE)) {
#if USE_HIERARCHICAL_NEE
                    const simd_fvec<S> pdf_factor =
                        EvalTriLightFactor(surf.P, ray.o, ray_queue[index], tri_index, sc.lights, sc.light_wnodes);
#else  // USE_HIERARCHICAL_NEE
                    const float pdf_factor = float(sc.li_indices.size());
#endif // USE_HIERARCHICAL_NEE

                    const simd_fvec<S> v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]},
                                       v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};

                    simd_fvec<S> light_forward[3];
                    cross(v1, v2, light_forward);
                    TransformDirection(transform, light_forward);
                    const simd_fvec<S> tri_area = 0.5f * normalize(light_forward);

                    const simd_fvec<S> cos_theta = abs(dot3(I, light_forward));
                    const simd_ivec<S> emissive_mask =
                        ray_queue[index] & simd_cast(cos_theta > 0.0f) & (simd_ivec<S>(ray.depth & 0x00ffffff) != 0);
                    if (emissive_mask.not_all_zeros()) {
#if USE_SPHERICAL_AREA_LIGHT_SAMPLING
                        simd_fvec<S> light_pdf =
                            SampleSphericalTriangle<S>(ro_ls, p1, p2, p3, nullptr, nullptr) / pdf_factor;
                        where(light_pdf == 0.0f, light_pdf) =
                            safe_div_pos(inter.t * inter.t, tri_area * cos_theta * pdf_factor);
#else  // USE_SPHERICAL_AREA_LIGHT_SAMPLING
                        const simd_fvec<S> light_pdf =
                            safe_div_pos(inter.t * inter.t, tri_area * cos_theta * pdf_factor);
#endif // USE_SPHERICAL_AREA_LIGHT_SAMPLING
                        const simd_fvec<S> &bsdf_pdf = ray.pdf;

                        where(emissive_mask, mis_weight) = power_heuristic(bsdf_pdf, light_pdf);
                    }
                }
#endif // USE_NEE
                UNROLLED_FOR(i, 3, {
                    where(ray_queue[index], col[i]) += mix_weight * mis_weight * mat->strength * base_color[i];
                })
            } else if (mat->type == eShadingNode::Principled) {
                simd_fvec<S> metallic = unpack_unorm_16(mat->metallic_unorm);
                if (mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
                    const uint32_t metallic_tex = mat->textures[METALLIC_TEXTURE];
                    const simd_fvec<S> metallic_lod = get_texture_lod(textures, metallic_tex, lambda, ray_queue[index]);
                    simd_fvec<S> metallic_color[4] = {};
                    SampleBilinear(textures, metallic_tex, surf.uvs, simd_ivec<S>(metallic_lod), tex_rand,
                                   ray_queue[index], metallic_color);

                    metallic *= metallic_color[0];
                }

                simd_fvec<S> specular = unpack_unorm_16(mat->specular_unorm);
                if (mat->textures[SPECULAR_TEXTURE] != 0xffffffff) {
                    const uint32_t specular_tex = mat->textures[SPECULAR_TEXTURE];
                    const simd_fvec<S> specular_lod = get_texture_lod(textures, specular_tex, lambda, ray_queue[index]);
                    simd_fvec<S> specular_color[4] = {};
                    SampleBilinear(textures, specular_tex, surf.uvs, simd_ivec<S>(specular_lod), tex_rand,
                                   ray_queue[index], specular_color);
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
                coat.ior = (2.0f / (1.0f - sqrtf(0.08f * clearcoat))) - 1.0f;
                coat.F0 = fresnel_dielectric_cos(simd_fvec<S>{1.0f}, coat.ior);

                transmission_params_t<S> trans;
                trans.roughness =
                    1.0f - (1.0f - roughness) * (1.0f - unpack_unorm_16(mat->transmission_roughness_unorm));
                trans.int_ior = mat->ior;
                trans.eta = select(is_backfacing, (mat->ior / ext_ior), (ext_ior / mat->ior));
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
                    shadow_mask |= Evaluate_PrincipledNode(
                        ls, ray, eval_light, surf, lobe_weights, diff, spec, coat, trans, metallic, transmission,
                        N_dot_L, mix_weight, (total_depth < ps.max_total_depth), regularize_alpha, col, sh_r);
                }
#endif
                Sample_PrincipledNode(ps, ray, ray_queue[index], surf, lobe_weights, diff, spec, coat, trans, metallic,
                                      transmission, rand_uv, mix_rand, mix_weight, regularize_alpha, secondary_mask,
                                      new_ray);
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
    const simd_fvec<S> &p = mix_term_rand[1];
    const simd_fvec<S> q = select(can_terminate_path, max(0.05f, 1.0f - lum), simd_fvec<S>{0.0f});

    secondary_mask &= simd_cast(p >= q) & simd_cast(lum > 0.0f) & simd_cast(new_ray.pdf > 0.0f);
    if (secondary_mask.not_all_zeros()) {
        UNROLLED_FOR(i, 3, { new_ray.c[i] = safe_div_pos(new_ray.c[i], 1.0f - q); })

        // TODO: check if this is needed
        new_ray.pdf = min(new_ray.pdf, 1e6f);

        // TODO: get rid of this!
        UNROLLED_FOR(i, 3, { where(~secondary_mask, new_ray.d[i]) = 0.0f; })

        (*out_secondary_rays_count)++;
        new_ray.mask = secondary_mask;
    }

#if USE_NEE
    if (shadow_mask.not_all_zeros()) {
        // actual ray direction accouning for bias from both ends
        simd_fvec<S> to_light[3];
        UNROLLED_FOR(i, 3, { to_light[i] = ls.lp[i] - sh_r.o[i]; })
        sh_r.dist = normalize(to_light);
        UNROLLED_FOR(i, 3, { where(shadow_mask, sh_r.d[i]) = to_light[i]; })
        sh_r.dist *= ls.dist_mul;
        // NOTE: hacky way to identify env ray
        where(ls.from_env & shadow_mask, sh_r.dist) = -sh_r.dist;

        (*out_shadow_rays_count)++;
        sh_r.mask = shadow_mask;
    }
#endif

    UNROLLED_FOR(i, 3, { where(is_active_lane, col[i]) = ray.c[i] * col[i]; })

    const simd_fvec<S> sum = col[0] + col[1] + col[2];
    UNROLLED_FOR(i, 3, {
        where(sum > limits[1], col[i]) = safe_div_pos(col[i] * limits[1], sum);
        where(is_active_lane, out_rgba[i]) = col[i];
    })
    where(is_active_lane, out_rgba[3]) = 1.0f;
}

template <int S>
void Ray::NS::ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t<S>> inters, Span<const ray_data_t<S>> rays,
                           const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                           const scene_data_t &sc, const uint32_t node_index,
                           const Cpu::TexStorageBase *const textures[], ray_data_t<S> *out_secondary_rays,
                           int *out_secondary_rays_count, shadow_ray_t<S> *out_shadow_rays, int *out_shadow_rays_count,
                           int img_w, float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                           color_rgba_t *out_depth_normals) {
    const float limits[2] = {(ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX,
                             (ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX};
    for (int i = 0; i < inters.size(); ++i) {
        const ray_data_t<S> &r = rays[i];
        const hit_data_t<S> &inter = inters[i];

        simd_fvec<S> col[4] = {}, base_color[3] = {}, depth_normal[4] = {};
        ShadeSurface(ps, limits, rand_seq, rand_seed, iteration, inter, r, sc, node_index, textures, col,
                     out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count, base_color,
                     depth_normal);

        const simd_uvec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;

        // TODO: match layouts!
        UNROLLED_FOR_S(j, S, {
            if (r.mask.template get<j>()) {
                UNROLLED_FOR(k, 4, {
                    out_color[y.template get<j>() * img_w + x.template get<j>()].v[k] = col[k].template get<j>();
                })
                { // base color
                    auto old_val = simd_fvec4(out_base_color[y.template get<j>() * img_w + x.template get<j>()].v,
                                              simd_mem_aligned);
                    old_val += (simd_fvec4{base_color[0].template get<j>(), base_color[1].template get<j>(),
                                           base_color[2].template get<j>(), 0.0f} -
                                old_val) *
                               mix_factor;
                    old_val.store_to(out_base_color[y.template get<j>() * img_w + x.template get<j>()].v,
                                     simd_mem_aligned);
                }
                { // depth-normals
                    auto old_val = simd_fvec4(out_depth_normals[y.template get<j>() * img_w + x.template get<j>()].v,
                                              simd_mem_aligned);
                    old_val += (simd_fvec4{depth_normal[0].template get<j>(), depth_normal[1].template get<j>(),
                                           depth_normal[2].template get<j>(), depth_normal[3].template get<j>()} -
                                old_val) *
                               mix_factor;
                    old_val.store_to(out_depth_normals[y.template get<j>() * img_w + x.template get<j>()].v,
                                     simd_mem_aligned);
                }
            }
        })
    }
}

template <int S>
void Ray::NS::ShadeSecondary(const pass_settings_t &ps, const float clamp_direct, Span<const hit_data_t<S>> inters,
                             Span<const ray_data_t<S>> rays, const uint32_t rand_seq[], const uint32_t rand_seed,
                             const int iteration, const scene_data_t &sc, uint32_t node_index,
                             const Cpu::TexStorageBase *const textures[], ray_data_t<S> *out_secondary_rays,
                             int *out_secondary_rays_count, shadow_ray_t<S> *out_shadow_rays,
                             int *out_shadow_rays_count, int img_w, color_rgba_t *out_color) {
    const float limits[2] = {(clamp_direct != 0.0f) ? 3.0f * clamp_direct : FLT_MAX,
                             (ps.clamp_indirect != 0.0f) ? 3.0f * ps.clamp_indirect : FLT_MAX};
    for (int i = 0; i < inters.size(); ++i) {
        const ray_data_t<S> &r = rays[i];
        const hit_data_t<S> &inter = inters[i];

        simd_fvec<S> col[4] = {0.0f};
        Ray::NS::ShadeSurface(ps, limits, rand_seq, rand_seed, iteration, inter, r, sc, node_index, textures, col,
                              out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count,
                              (simd_fvec<S> *)nullptr, (simd_fvec<S> *)nullptr);

        const simd_uvec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;

        // TODO: match layouts!
        UNROLLED_FOR_S(j, S, {
            if (r.mask.template get<j>()) {
                auto old_val =
                    simd_fvec4(out_color[y.template get<j>() * img_w + x.template get<j>()].v, simd_mem_aligned);
                old_val +=
                    simd_fvec4(col[0].template get<j>(), col[1].template get<j>(), col[2].template get<j>(), 0.0f);
                old_val.store_to(out_color[y.template get<j>() * img_w + x.template get<j>()].v, simd_mem_aligned);
            }
        })
    }
}

template <int S, int InChannels, int OutChannels, int OutPxPitch, Ray::ePostOp PostOp, Ray::eActivation Activation>
void Ray::NS::Convolution3x3_Direct(const float data[], const rect_t &rect, int w, int h, int stride,
                                    const float weights[], const float biases[], float output[], int output_stride) {
    static_assert((InChannels % S) == 0, "!");

    if (!output_stride) {
        if (PostOp == ePostOp::Downscale) {
            output_stride = (w + 1) / 2;
        } else {
            output_stride = w;
        }
    }

    if (PostOp == ePostOp::Downscale) {
        if (OutChannels == OutPxPitch) {
            for (int y = (rect.y / 2); y < (rect.y + rect.h + 1) / 2; ++y) {
                float *ptr = &output[OutChannels * (y * output_stride + (rect.x / 2))];
                std::fill(ptr, ptr + ((rect.w + 1) / 2) * OutChannels, 0.0f);
            }
        } else {
            for (int y = (rect.y / 2); y < (rect.y + rect.h + 1) / 2; ++y) {
                for (int x = (rect.x / 2); x < (rect.x + rect.w + 1) / 2; ++x) {
                    for (int c = 0; c < OutChannels; ++c) {
                        output[OutPxPitch * (y * output_stride + (x / 2)) + c] = 0.0f;
                    }
                }
            }
        }
    }

    static_assert((InChannels % S) == 0, "!");

    int y = rect.y;
    for (; y < rect.y + rect.h - 7; y += 8) {
        Convolution3x3_Direct_ProcessRows<8, S, InChannels, OutChannels, OutPxPitch, PostOp, Activation>(
            y, data, rect, w, h, stride, weights, biases, output, output_stride);
    }

    for (; y < rect.y + rect.h - 3; y += 4) {
        Convolution3x3_Direct_ProcessRows<4, S, InChannels, OutChannels, OutPxPitch, PostOp, Activation>(
            y, data, rect, w, h, stride, weights, biases, output, output_stride);
    }

    for (; y < rect.y + rect.h; ++y) {
        Convolution3x3_Direct_ProcessRows<1, S, InChannels, OutChannels, OutPxPitch, PostOp, Activation>(
            y, data, rect, w, h, stride, weights, biases, output, output_stride);
    }
}

template <int S, int InChannels1, int InChannels2, int OutChannels, Ray::ePreOp PreOp1, Ray::ePostOp PostOp,
          Ray::eActivation Activation>
void Ray::NS::ConvolutionConcat3x3_Direct(const float data1[], const float data2[], const rect_t &rect, int w, int h,
                                          int stride1, int stride2, const float weights[], const float biases[],
                                          float output[], int output_stride) {
    static_assert((InChannels1 % S) == 0 && (InChannels2 % S) == 0, "!");

    if (!output_stride) {
        if (PostOp == ePostOp::Downscale) {
            output_stride = (w + 1) / 2;
        } else {
            output_stride = w;
        }
    }

    int y = rect.y;
    for (; y < rect.y + rect.h - 7; y += 8) {
        ConvolutionConcat3x3_Direct_ProcessRows<8, S, InChannels1, InChannels2, OutChannels, PreOp1, PostOp,
                                                Activation>(y, data1, data2, rect, w, h, stride1, stride2, weights,
                                                            biases, output, output_stride);
    }

    for (; y < rect.y + rect.h - 3; y += 4) {
        ConvolutionConcat3x3_Direct_ProcessRows<4, S, InChannels1, InChannels2, OutChannels, PreOp1, PostOp,
                                                Activation>(y, data1, data2, rect, w, h, stride1, stride2, weights,
                                                            biases, output, output_stride);
    }

    for (; y < rect.y + rect.h; ++y) {
        ConvolutionConcat3x3_Direct_ProcessRows<1, S, InChannels1, InChannels2, OutChannels, PreOp1, PostOp,
                                                Activation>(y, data1, data2, rect, w, h, stride1, stride2, weights,
                                                            biases, output, output_stride);
    }
}

template <int S, int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
          Ray::ePreOp PreOp1, Ray::ePreOp PreOp2, Ray::ePreOp PreOp3, Ray::ePreOp PreOp4, Ray::ePostOp PostOp,
          Ray::eActivation Activation>
void Ray::NS::ConvolutionConcat3x3_1Direct_2GEMM(const float data1[], const float data2[], const float data3[],
                                                 const float data4[], const rect_t &rect, int w, int h, int w2, int h2,
                                                 int stride1, int stride2, const float weights[], const float biases[],
                                                 float output[], int output_stride) {
    static_assert((InChannels1 % S) == 0, "!");

    int y = rect.y;
    for (; y < rect.y + rect.h - 7; y += 8) {
        ConvolutionConcat3x3_1Direct_2GEMM_ProcessRows<8, S, InChannels1, InChannels2, InChannels3, InChannels4,
                                                       PxPitch2, OutChannels, PreOp1, PreOp2, PreOp3, PreOp4, PostOp,
                                                       Activation>(y, data1, data2, data3, data4, rect, w, h, w2, h2,
                                                                   stride1, stride2, weights, biases, output,
                                                                   output_stride);
    }

    for (; y < rect.y + rect.h - 3; y += 4) {
        ConvolutionConcat3x3_1Direct_2GEMM_ProcessRows<4, S, InChannels1, InChannels2, InChannels3, InChannels4,
                                                       PxPitch2, OutChannels, PreOp1, PreOp2, PreOp3, PreOp4, PostOp,
                                                       Activation>(y, data1, data2, data3, data4, rect, w, h, w2, h2,
                                                                   stride1, stride2, weights, biases, output,
                                                                   output_stride);
    }

    for (; y < rect.y + rect.h; ++y) {
        ConvolutionConcat3x3_1Direct_2GEMM_ProcessRows<1, S, InChannels1, InChannels2, InChannels3, InChannels4,
                                                       PxPitch2, OutChannels, PreOp1, PreOp2, PreOp3, PreOp4, PostOp,
                                                       Activation>(y, data1, data2, data3, data4, rect, w, h, w2, h2,
                                                                   stride1, stride2, weights, biases, output,
                                                                   output_stride);
    }
}

#undef sqr

#undef USE_NEE
#undef USE_PATH_TERMINATION
#undef FORCE_TEXTURE_LOD
#undef USE_SAFE_MATH
#undef USE_STOCH_TEXTURE_FILTERING

#pragma warning(pop)