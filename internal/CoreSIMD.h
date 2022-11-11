//#pragma once
// This file is compiled many times for different simd architectures (SSE, NEON...).
// Macro 'NS' defines a namespace in which everything will be located, so it should be set before including this file.
// Macros 'USE_XXX' define template instantiation of simd_fvec, simd_ivec classes.
// Template parameter S defines width of vectors used. Usualy it is equal to ray packet size.

#include <vector>

#include <cfloat>

#include "TextureAtlasRef.h"

#include "simd/simd_vec.h"

#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant

//
// Useful macros for debugging
//
#define USE_VNDF_GGX_SAMPLING 1
#define USE_NEE 1
#define USE_PATH_TERMINATION 1
#define FORCE_TEXTURE_LOD0 0

namespace Ray {
namespace Ref {
class TextureAtlasBase;
template <typename T, int N> class TextureAtlasLinear;
template <typename T, int N> class TextureAtlasTiled;
template <typename T, int N> class TextureAtlasSwizzled;
using TextureAtlasRGBA = TextureAtlasSwizzled<uint8_t, 4>;
using TextureAtlasRGB = TextureAtlasSwizzled<uint8_t, 3>;
using TextureAtlasRG = TextureAtlasSwizzled<uint8_t, 2>;
using TextureAtlasR = TextureAtlasSwizzled<uint8_t, 1>;
} // namespace Ref
namespace NS {

alignas(64) const int ray_packet_layout_x[] = {0, 1, 0, 1,  // NOLINT
                                               2, 3, 2, 3,  // NOLINT
                                               0, 1, 0, 1,  // NOLINT
                                               2, 3, 2, 3}; // NOLINT

alignas(64) const int ray_packet_layout_y[] = {0, 0, 1, 1,  // NOLINT
                                               0, 0, 1, 1,  // NOLINT
                                               2, 2, 3, 3,  // NOLINT
                                               2, 2, 3, 3}; // NOLINT

alignas(64) const int ascending_counter[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int S> struct ray_data_t {
    // origins of rays in packet
    simd_fvec<S> o[3];
    // directions of rays in packet
    simd_fvec<S> d[3], pdf;
    // throughput color of ray
    simd_fvec<S> c[3];
#ifdef USE_RAY_DIFFERENTIALS
    // derivatives
    simd_fvec<S> do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
#else
    // ray cone params
    simd_fvec<S> cone_width, cone_spread;
#endif
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_ivec<S> xy;
    // four 8-bit ray depth counters
    simd_ivec<S> ray_depth;
};

template <int S> struct shadow_ray_t {
    // origins of rays in packet
    simd_fvec<S> o[3];
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

    hit_data_t(eUninitialize) {}
    force_inline hit_data_t() {
        mask = {0};
        obj_index = {-1};
        prim_index = {-1};
        t = {MAX_DIST};
    }
};

template <int S> struct derivatives_t {
    simd_fvec<S> do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
    simd_fvec<S> duv_dx[2], duv_dy[2];
    simd_fvec<S> dndx[3], dndy[3];
    simd_fvec<S> ddn_dx, ddn_dy;
};

template <int S> struct light_sample_t {
    simd_fvec<S> col[3], L[3];
    simd_fvec<S> area = 0.0f, dist, pdf = 0.0f;
};

// Generating rays
template <int DimX, int DimY>
void GeneratePrimaryRays(const int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton,
                         aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                         aligned_vector<simd_ivec<DimX * DimY>> &out_masks);
template <int DimX, int DimY>
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr,
                              const uint32_t *vtx_indices, const vertex_t *vertices, const rect_t &r, int w, int h,
                              const float *halton, aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
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
bool IntersectTris_ClosestHit(const float o[3], const float d[3], int i, const mtri_accel_t *mtris, int tri_start,
                              int tri_end, int obj_index, hit_data_t<S> &out_inter);
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
bool IntersectTris_AnyHit(const float o[3], const float d[3], int i, const mtri_accel_t *mtris,
                          const tri_mat_data_t *materials, const uint32_t *indices, int tri_start, int tri_end,
                          int obj_index, hit_data_t<S> &out_inter);

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
bool Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                         const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                         const mesh_t *meshes, const transform_t *transforms, const tri_accel_t *tris,
                                         const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                         hit_data_t<S> &inter);
template <int S>
bool Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                         const simd_ivec<S> &ray_mask, const mbvh_node_t *mnodes, uint32_t node_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                         const mesh_t *meshes, const transform_t *transforms, const mtri_accel_t *mtris,
                                         const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                         hit_data_t<S> &inter);
// traditional bvh traversal with stack for inner nodes
template <int S>
bool Traverse_MicroTree_WithStack_ClosestHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                             const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index,
                                             hit_data_t<S> &inter);
template <int S>
bool Traverse_MicroTree_WithStack_ClosestHit(const float ro[3], const float rd[3], int i, const mbvh_node_t *mnodes,
                                             uint32_t node_index, const mtri_accel_t *mtris,
                                             const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);
template <int S>
bool Traverse_MicroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                         const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                         const tri_accel_t *tris, const tri_mat_data_t *materials,
                                         const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);
template <int S>
bool Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], int i, const mbvh_node_t *mnodes,
                                         uint32_t node_index, const mtri_accel_t *mtris,
                                         const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index,
                                         hit_data_t<S> &inter);

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
                                     const simd_fvec<S> I[3], float clearcoat_roughness2, float clearcoat_ior,
                                     float clearcoat_F0, const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v,
                                     simd_fvec<S> out_V[3], simd_fvec<S> out_color[4]);

// Transform
template <int S>
void TransformRay(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const float *xform, simd_fvec<S> out_ro[3],
                  simd_fvec<S> out_rd[3]);
template <int S> void TransformPoint(const simd_fvec<S> p[3], const float *xform, simd_fvec<S> out_p[3]);
template <int S> void TransformDirection(const simd_fvec<S> xform[16], simd_fvec<S> p[3]);
template <int S> void TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]);
template <int S> void TransformNormal(const simd_fvec<S> n[3], const simd_fvec<S> inv_xform[16], simd_fvec<S> out_n[3]);
template <int S> void TransformNormal(const simd_fvec<S> inv_xform[16], simd_fvec<S> inout_n[3]);
template <int S>
void TransformUVs(const simd_fvec<S> _uvs[2], float sx, float sy, const texture_t &t, const simd_ivec<S> &mip_level,
                  simd_fvec<S> out_res[2]);

void TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3], float out_rd[3]);

template <int S>
void rotate_around_axis(const simd_fvec<S> p[3], const simd_fvec<S> axis[3], const simd_fvec<S> &angle,
                        simd_fvec<S> out_p[3]);

// Sample texture
template <int S>
void SampleNearest(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                   const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                    const simd_ivec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Ref::TextureAtlasBase &atlas, const simd_fvec<S> uvs[2], const simd_ivec<S> &page,
                    const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleTrilinear(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                     const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleAnisotropic(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                       const simd_fvec<S> duv_dx[2], const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask,
                       simd_fvec<S> out_rgba[4]);
template <int S>
void SampleLatlong_RGBE(const Ref::TextureAtlasRGBA &atlas, const texture_t &t, const simd_fvec<S> dir[3],
                        const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]);

// Get visibility between two points accounting for transparent materials
template <int S>
simd_fvec<S> ComputeVisibility(const simd_fvec<S> p1[3], const simd_fvec<S> d[3], simd_fvec<S> dist,
                               const simd_ivec<S> &mask, const float rand_val, const simd_ivec<S> &rand_hash2,
                               const scene_data_t &sc, uint32_t node_index, const Ref::TextureAtlasBase *atlases[]);

// Compute derivatives at hit point
template <int S>
void ComputeDerivatives(const simd_fvec<S> I[3], const simd_fvec<S> &t, const simd_fvec<S> do_dx[3],
                        const simd_fvec<S> do_dy[3], const simd_fvec<S> dd_dx[3], const simd_fvec<S> dd_dy[3],
                        const simd_fvec<S> p1[3], const simd_fvec<S> p2[3], const simd_fvec<S> p3[3],
                        const simd_fvec<S> n1[3], const simd_fvec<S> n2[3], const simd_fvec<S> n3[3],
                        const simd_fvec<S> u1[2], const simd_fvec<S> u2[2], const simd_fvec<S> u3[2],
                        const simd_fvec<S> plane_N[3], const simd_fvec<S> xform[16], derivatives_t<S> &out_der);

// Pick point on any light source for evaluation
template <int S>
void SampleLightSource(const simd_fvec<S> P[3], const scene_data_t &sc, const Ref::TextureAtlasBase *tex_atlases[],
                       const float halton[], const simd_fvec<S> sample_off[2], const simd_ivec<S> &ray_mask,
                       light_sample_t<S> &ls);

// Account for visible lights contribution
template <int S>
void IntersectAreaLights(const ray_data_t<S> &r, const simd_ivec<S> &ray_mask, const light_t lights[],
                         Span<const uint32_t> visible_lights, const transform_t transforms[],
                         hit_data_t<S> &inout_inter);

// Shade
template <int S>
void ShadeSurface(const simd_ivec<S> &px_index, const pass_info_t &pi, const float *halton, const hit_data_t<S> &inter,
                  const ray_data_t<S> &ray, const scene_data_t &sc, uint32_t node_index,
                  const Ref::TextureAtlasBase *tex_atlases[], simd_fvec<S> out_rgba[4],
                  simd_ivec<S> out_secondary_masks[], ray_data_t<S> out_secondary_rays[], int *out_secondary_rays_count,
                  simd_ivec<S> out_shadow_masks[], shadow_ray_t<S> out_shadow_rays[], int *out_shadow_rays_count);
} // namespace NS
} // namespace Ray

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace Ray {
namespace NS {
template <int S>
force_inline simd_ivec<S> _IntersectTri(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                        const simd_ivec<S> &ray_mask, const tri_accel_t &tri, uint32_t prim_index,
                                        hit_data_t<S> &inter) {
#define _dot(x, y) ((x)[0] * (y)[0] + (x)[1] * (y)[1] + (x)[2] * (y)[2])

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
#undef _dot
}

template <int S>
force_inline bool _IntersectTri(const float o[3], const float d[3], int i, const tri_accel_t &tri,
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
force_inline bool _IntersectTri(const float ro[3], const float rd[3], int j, const mtri_accel_t &tri,
                                const uint32_t prim_index, hit_data_t<S> &inter) {
    static const int LanesCount = 8 / S;

    simd_ivec<S> _mask = 0, _prim_index;
    simd_fvec<S> _t = inter.t[j], _u, _v;

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

        const simd_fvec<S> rdet = (1.0f / det);

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

    inter.mask[j] = 0xffffffff;

    const long i1 = GetFirstBit(mask);
    mask = ClearBit(mask, i1);

    long min_i = i1;
    inter.prim_index[j] = _prim_index[i1];
    inter.t[j] = _t[i1];
    inter.u[j] = _u[i1];
    inter.v[j] = _v[i1];

    if (mask == 0) { // Only one triangle was hit
        return true;
    }

    do {
        const long i2 = GetFirstBit(mask);
        mask = ClearBit(mask, i2);

        if (_t[i2] < _t[min_i]) {
            inter.prim_index[j] = _prim_index[i2];
            inter.t[j] = _t[i2];
            inter.u[j] = _u[i2];
            inter.v[j] = _v[i2];
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

    return reinterpret_cast<const simd_ivec<S> &>(mask);
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

    static const int LanesCount = 8 / S;

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
        const auto mask1 = reinterpret_cast<const simd_ivec<S> &>(dir_neg_mask) & queue[index].mask;
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

template <int S> force_inline void safe_invert(const simd_fvec<S> v[3], simd_fvec<S> inv_v[3]) {
    inv_v[0] = {1.0f / v[0]};
    where((v[0] <= FLT_EPS) & (v[0] >= 0), inv_v[0]) = MAX_DIST;
    where((v[0] >= -FLT_EPS) & (v[0] < 0), inv_v[0]) = -MAX_DIST;

    inv_v[1] = {1.0f / v[1]};
    where((v[1] <= FLT_EPS) & (v[1] >= 0), inv_v[1]) = MAX_DIST;
    where((v[1] >= -FLT_EPS) & (v[1] < 0), inv_v[1]) = -MAX_DIST;

    inv_v[2] = {1.0f / v[2]};
    where((v[2] <= FLT_EPS) & (v[2] >= 0), inv_v[2]) = MAX_DIST;
    where((v[2] >= -FLT_EPS) & (v[2] < 0), inv_v[2]) = -MAX_DIST;
}

force_inline void safe_invert(const float v[3], float out_v[3]) {
    out_v[0] = 1.0f / v[0];
    out_v[1] = 1.0f / v[1];
    out_v[2] = 1.0f / v[2];

    if (v[0] <= FLT_EPS && v[0] >= 0) {
        out_v[0] = MAX_DIST;
    } else if (v[0] >= -FLT_EPS && v[0] < 0) {
        out_v[0] = -MAX_DIST;
    }

    if (v[1] <= FLT_EPS && v[1] >= 0) {
        out_v[1] = MAX_DIST;
    } else if (v[1] >= -FLT_EPS && v[1] < 0) {
        out_v[1] = -MAX_DIST;
    }

    if (v[2] <= FLT_EPS && v[2] >= 0) {
        out_v[2] = MAX_DIST;
    } else if (v[2] >= -FLT_EPS && v[2] < 0) {
        out_v[2] = -MAX_DIST;
    }
}

template <int S>
force_inline void comp_aux_inv_values(const simd_fvec<S> o[3], const simd_fvec<S> d[3], simd_fvec<S> inv_d[3],
                                      simd_fvec<S> inv_d_o[3]) {
    for (int i = 0; i < 3; i++) {
        inv_d[i] = {1.0f / d[i]};
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
    inv_d[0] = 1.0f / d[0];
    inv_d[1] = 1.0f / d[1];
    inv_d[2] = 1.0f / d[2];

    inv_d_o[0] = inv_d[0] * o[0];
    inv_d_o[1] = inv_d[1] * o[1];
    inv_d_o[2] = inv_d[2] * o[2];

    if (d[0] <= FLT_EPS && d[0] >= 0) {
        inv_d[0] = MAX_DIST;
        inv_d_o[0] = MAX_DIST;
    } else if (d[0] >= -FLT_EPS && d[0] < 0) {
        inv_d[0] = -MAX_DIST;
        inv_d_o[0] = -MAX_DIST;
    }

    if (d[1] <= FLT_EPS && d[1] >= 0) {
        inv_d[1] = MAX_DIST;
        inv_d_o[1] = MAX_DIST;
    } else if (d[1] >= -FLT_EPS && d[1] < 0) {
        inv_d[1] = -MAX_DIST;
        inv_d_o[1] = -MAX_DIST;
    }

    if (d[2] <= FLT_EPS && d[2] >= 0) {
        inv_d[2] = MAX_DIST;
        inv_d_o[2] = MAX_DIST;
    } else if (d[2] >= -FLT_EPS && d[2] < 0) {
        inv_d[2] = -MAX_DIST;
        inv_d_o[2] = -MAX_DIST;
    }
}

template <int S> force_inline simd_fvec<S> dot(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S> force_inline simd_fvec<S> dot(const simd_fvec<S> v1[3], const float v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S> force_inline simd_fvec<S> dot(const float v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

force_inline float dot(const float v1[3], const float v2[3]) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }

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

template <int S> force_inline simd_fvec<S> length(const simd_fvec<S> v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <int S> force_inline simd_fvec<S> length2_2d(const simd_fvec<S> v[2]) { return v[0] * v[0] + v[1] * v[1]; }

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

force_inline float length(const simd_fvec2 &x) { return sqrtf(x[0] * x[0] + x[1] * x[1]); }

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
force_inline simd_ivec<S> get_ray_hash(const ray_data_t<S> &r, const simd_ivec<S> &mask, const float root_min[3],
                                       const float cell_size[3]) {
    simd_ivec<S> x = clamp((simd_ivec<S>)((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
                 y = clamp((simd_ivec<S>)((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
                 z = clamp((simd_ivec<S>)((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    simd_ivec<S> omega_index = clamp((simd_ivec<S>)((1.0f + r.d[2]) / omega_step), 0, 32),
                 phi_index_i = clamp((simd_ivec<S>)((1.0f + r.d[1]) / phi_step), 0, 16),
                 phi_index_j = clamp((simd_ivec<S>)((1.0f + r.d[0]) / phi_step), 0, 16);

    simd_ivec<S> o, p;

    ITERATE(S, {
        if (mask[i]) {
            x[i] = morton_table_256[x[i]];
            y[i] = morton_table_256[y[i]];
            z[i] = morton_table_256[z[i]];
            o[i] = morton_table_16[int(omega_table[omega_index[i]])];
            p[i] = morton_table_16[int(phi_table[phi_index_i[i]][phi_index_j[i]])];
        } else {
            o[i] = p[i] = 0xFFFFFFFF;
            x[i] = y[i] = z[i] = 0xFFFFFFFF;
        }
    });

    return (o << 25) | (p << 24) | (y << 2) | (z << 1) | (x << 0);
}

force_inline void _radix_sort_lsb(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1, unsigned maxshift) {
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

    const simd_fvec<S> f = reinterpret_cast<simd_fvec<S> &>(m); // Range [1:2]
    return f - simd_fvec<S>{1.0f};                              // Range [0:1]
}

force_inline float fast_log2(float val) {
    // From https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c
    union {
        float val;
        int32_t x;
    } u = {val};
    float log_2 = float(((u.x >> 23) & 255) - 128);
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
    ITERATE_3({
        simd_fvec<S> temp = in_col[i] / 12.92f;
        where(in_col[i] > 0.04045f, temp) = pow((in_col[i] + 0.055f) / 1.055f, 2.4f);
        out_col[i] = temp;
    })
    out_col[3] = in_col[3];
}

template <int S>
simd_fvec<S> get_texture_lod(const texture_t &t, const simd_fvec<S> duv_dx[2], const simd_fvec<S> duv_dy[2],
                             const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD0
    const simd_fvec<S> lod = 0.0f;
#else
    const int width = int(t.width & TEXTURE_WIDTH_BITS), height = int(t.height & TEXTURE_HEIGHT_BITS);

    const simd_fvec<S> _duv_dx[2] = {duv_dx[0] * float(width), duv_dx[1] * float(height)};
    const simd_fvec<S> _duv_dy[2] = {duv_dy[0] * float(width), duv_dy[1] * float(height)};

    const simd_fvec<S> _diagonal[2] = {_duv_dx[0] + _duv_dy[0], _duv_dx[1] + _duv_dy[1]};

    const simd_fvec<S> dim = min(min(length2_2d(_duv_dx), length2_2d(_duv_dy)), length2_2d(_diagonal));

    simd_fvec<S> lod = 0.5f * fast_log2(dim) - 1.0f;

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);
#endif
    return lod;
}

template <int S>
simd_fvec<S> get_texture_lod(const texture_t &t, const simd_fvec<S> &lambda, const simd_ivec<S> &mask) {
#if FORCE_TEXTURE_LOD0
    const simd_fvec<S> lod = 0.0f;
#else
    const float width = float(t.width & TEXTURE_WIDTH_BITS), height = float(t.height & TEXTURE_HEIGHT_BITS);

    simd_fvec<S> lod;

    ITERATE(S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask)[i]) {
            lod[i] = lambda[i] + 0.5f * fast_log2(width * height) - 1.0f;
        } else {
            lod[i] = 0.0f;
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
#if FORCE_TEXTURE_LOD0
    const simd_fvec<S> lod = 0.0f;
#else
    const simd_fvec<S> _duv_dx[2] = {duv_dx[0] * simd_fvec<S>(width), duv_dx[1] * simd_fvec<S>(height)};
    const simd_fvec<S> _duv_dy[2] = {duv_dy[0] * simd_fvec<S>(width), duv_dy[1] * simd_fvec<S>(height)};

    const simd_fvec<S> _diagonal[2] = {_duv_dx[0] + _duv_dy[0], _duv_dx[1] + _duv_dy[1]};

    const simd_fvec<S> dim = min(min(length2_2d(_duv_dx), length2_2d(_duv_dy)), length2_2d(_diagonal));

    simd_fvec<S> lod;

    ITERATE(S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask)[i]) {
            lod[i] = 0.5f * fast_log2(dim[i]) - 1.0f;
        } else {
            lod[i] = 0.0f;
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
#if FORCE_TEXTURE_LOD0
    const simd_fvec<S> lod = 0.0f;
#else
    simd_fvec<S> lod;

    ITERATE(S, {
        if (reinterpret_cast<const simd_ivec<S> &>(mask)[i]) {
            lod[i] = lambda[i] + 0.5f * fast_log2(width * height) - 1.0f;
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

template <int S> force_inline simd_fvec<S> safe_sqrtf(const simd_fvec<S> &f) { return sqrt(max(f, 0.0f)); }

template <int S>
void ensure_valid_reflection(const simd_fvec<S> Ng[3], const simd_fvec<S> I[3], simd_fvec<S> inout_N[3]) {
    simd_fvec<S> R[3];
    ITERATE_3({ R[i] = 2.0f * dot(inout_N, I) * inout_N[i] - I[i]; })

    // Reflection rays may always be at least as shallow as the incoming ray.
    const simd_fvec<S> threshold = min(0.9f * dot(Ng, I), 0.01f);

    const simd_ivec<S> early_mask = simd_cast(dot(Ng, R) < threshold);
    if (early_mask.all_zeros()) {
        return;
    }

    // Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
    // The X axis is found by normalizing the component of N that's orthogonal to Ng.
    // The Y axis isn't actually needed.
    const simd_fvec<S> NdotNg = dot(inout_N, Ng);

    simd_fvec<S> X[3];
    ITERATE_3({ X[i] = inout_N[i] - NdotNg * Ng[i]; })
    normalize(X);

    const simd_fvec<S> Ix = dot(I, X), Iz = dot(I, Ng);
    const simd_fvec<S> Ix2 = (Ix * Ix), Iz2 = (Iz * Iz);
    const simd_fvec<S> a = Ix2 + Iz2;

    const simd_fvec<S> b = safe_sqrtf(Ix2 * (a - (threshold * threshold)));
    const simd_fvec<S> c = Iz * threshold + a;

    // Evaluate both solutions.
    // In many cases one can be immediately discarded (if N'.z would be imaginary or larger than
    // one), so check for that first. If no option is viable (might happen in extreme cases like N
    // being in the wrong hemisphere), give up and return Ng.
    const simd_fvec<S> fac = 0.5f / a;
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
        ITERATE_2({ where(mask1, N_new[i]) = N1[i]; })
        const simd_ivec<S> mask2 = mask & ~simd_cast(R1 < R2);
        ITERATE_2({ where(mask2, N_new[i]) = N2[i]; })

        const simd_ivec<S> mask3 = ~mask & simd_cast(R1 > R2);
        ITERATE_2({ where(mask3, N_new[i]) = N1[i]; })
        const simd_ivec<S> mask4 = ~mask & ~simd_cast(R1 > R2);
        ITERATE_2({ where(mask4, N_new[i]) = N2[i]; })
    }

    if ((valid1 | valid2).not_all_zeros()) {
        const simd_ivec<S> exclude = ~(valid1 & valid2);

        // Only one solution passes the N'.z criterium, so pick that one.
        simd_fvec<S> Nz2 = N2_z2;
        where(valid1, Nz2) = N1_z2;

        where(exclude & (valid1 | valid2), N_new[0]) = safe_sqrtf(1.0f - Nz2);
        where(exclude & (valid1 | valid2), N_new[1]) = safe_sqrtf(Nz2);
    }

    ITERATE_3({ where(early_mask, inout_N[i]) = N_new[0] * X[i] + N_new[1] * Ng[i]; })
    ITERATE_3({ where(early_mask & ~valid1 & ~valid2, inout_N[i]) = Ng[i]; })
}

template <int S>
force_inline void world_from_tangent(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> V[3], simd_fvec<S> out_V[3]) {
    ITERATE_3({ out_V[i] = V[0] * T[i] + V[1] * B[i] + V[2] * N[i]; })
}

template <int S>
force_inline void tangent_from_world(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                     const simd_fvec<S> V[3], simd_fvec<S> out_V[3]) {
    out_V[0] = dot(V, T);
    out_V[1] = dot(V, B);
    out_V[2] = dot(V, N);
}

template <int S> force_inline simd_fvec<S> cos(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    ITERATE(S, { ret[i] = std::cos(v[i]); })
    return ret;
}

template <int S> force_inline simd_fvec<S> sin(const simd_fvec<S> &v) {
    simd_fvec<S> ret;
    ITERATE(S, { ret[i] = std::sin(v[i]); })
    return ret;
}

//
// From "A Fast and Robust Method for Avoiding Self-Intersection"
//
template <int S> void offset_ray(const simd_fvec<S> p[3], const simd_fvec<S> n[3], simd_fvec<S> out_p[3]) {
    static const float Origin = 1.0f / 32.0f;
    static const float FloatScale = 1.0f / 65536.0f;
    static const float IntScale = 256.0f;

    simd_ivec<S> of_i[3] = {simd_ivec<S>{IntScale * n[0]}, simd_ivec<S>{IntScale * n[1]},
                            simd_ivec<S>{IntScale * n[2]}};
    ITERATE_3({ where(p[i] < 0.0f, of_i[i]) = -of_i[i]; })

    const simd_fvec<S> p_i[3] = {simd_cast(simd_cast(p[0]) + of_i[0]), simd_cast(simd_cast(p[1]) + of_i[1]),
                                 simd_cast(simd_cast(p[2]) + of_i[2])};

    ITERATE_3({ out_p[i] = p_i[i]; })
    ITERATE_3({ where(abs(p[i]) < Origin, out_p[i]) = fmadd(simd_fvec<S>{FloatScale}, n[i], p[i]); })
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
    normalize(Vh);
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    const simd_fvec<S> lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];

    simd_fvec<S> T1[3] = {{1.0f}, {0.0f}, {0.0f}};
    where(lensq > 0.0f, T1[0]) = -Vh[1] / sqrt(lensq);
    where(lensq > 0.0f, T1[1]) = Vh[0] / sqrt(lensq);

    simd_fvec<S> T2[3];
    cross(Vh, T1, T2);
    // Section 4.2: parameterization of the projected area
    const simd_fvec<S> r = sqrt(U1);
    const simd_fvec<S> phi = 2.0f * PI * U2;
    simd_fvec<S> t1;
    ITERATE(S, { t1[i] = r[i] * std::cos(phi[i]); })
    simd_fvec<S> t2;
    ITERATE(S, { t2[i] = r[i] * std::sin(phi[i]); })
    const simd_fvec<S> s = 0.5f * (1.0f + Vh[2]);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    simd_fvec<S> Nh[3];
    ITERATE_3({ Nh[i] = t1 * T1[i] + t2 * T2[i] + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh[i]; })
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    out_V[0] = alpha_x * Nh[0];
    out_V[1] = alpha_y * Nh[1];
    out_V[2] = max(0.0f, Nh[2]);
    normalize(out_V);
}

// Smith shadowing function
template <int S> force_inline simd_fvec<S> G1(const simd_fvec<S> Ve[3], simd_fvec<S> alpha_x, simd_fvec<S> alpha_y) {
    alpha_x *= alpha_x;
    alpha_y *= alpha_y;
    const simd_fvec<S> delta =
        (-1.0f + sqrt(1.0f + (alpha_x * Ve[0] * Ve[0] + alpha_y * Ve[1] * Ve[1]) / (Ve[2] * Ve[2]))) / 2.0f;
    return 1.0f / (1.0f + delta);
}

template <int S> simd_fvec<S> D_GTR1(const simd_fvec<S> &NDotH, const simd_fvec<S> &a) {
    simd_fvec<S> ret = 1.0f / PI;
    const simd_fvec<S> a2 = a * a;
    const simd_fvec<S> t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    where(a < 1.0f, ret) = (a2 - 1.0f) / (PI * log(a2) * t);
    return ret;
}

template <int S> simd_fvec<S> D_GGX(const simd_fvec<S> H[3], const simd_fvec<S> &alpha_x, const simd_fvec<S> &alpha_y) {
    simd_fvec<S> ret = 0.0f;

    const simd_fvec<S> sx = -H[0] / (H[2] * alpha_x);
    const simd_fvec<S> sy = -H[1] / (H[2] * alpha_y);
    const simd_fvec<S> s1 = 1.0f + sx * sx + sy * sy;
    const simd_fvec<S> cos_theta_h4 = H[2] * H[2] * H[2] * H[2];

    where(H[2] != 0.0f, ret) = 1.0f / ((s1 * s1) * PI * alpha_x * alpha_y * cos_theta_h4);
    return ret;
}

template <int S> void create_tbn(const simd_fvec<S> N[3], simd_fvec<S> out_T[3], simd_fvec<S> out_B[3]) {

    simd_fvec<S> U[3] = {1.0f, 0.0f, 0.0f};
    where(N[1] < 0.999f, U[0]) = 0.0f;
    where(N[1] < 0.999f, U[1]) = 1.0f;

    cross(U, N, out_T);
    normalize(out_T);

    cross(N, out_T, out_B);
}

template <int S>
void MapToCone(const simd_fvec<S> &r1, const simd_fvec<S> &r2, const simd_fvec<S> N[3], float radius,
               simd_fvec<S> out_V[3]) {
    const simd_fvec<S> offset[2] = {2.0f * r1 - 1.0f, 2.0f * r2 - 1.0f};

    ITERATE_3({ out_V[i] = N[i]; })

    simd_fvec<S> r = offset[1];
    simd_fvec<S> theta = 0.5f * PI * (1.0f - 0.5f * (offset[0] / offset[1]));

    where(abs(offset[0]) > abs(offset[1]), r) = offset[0];
    where(abs(offset[0]) > abs(offset[1]), theta) = 0.25f * PI * (offset[1] / offset[0]);

    const simd_fvec<S> uv[2] = {radius * r * cos(theta), radius * r * sin(theta)};

    simd_fvec<S> LT[3], LB[3];
    create_tbn(N, LT, LB);

    ITERATE_3({ out_V[i] = N[i] + uv[0] * LT[i] + uv[1] * LB[i]; })

    const simd_fvec<S> mask = (offset[0] == 0.0f & offset[1] == 0.0f);
    ITERATE_3({ where(mask, out_V[i]) = N[i]; })
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

    simd_fvec<S> result = 1.0f; // TIR (no refracted component)

    g = sqrt(g);
    const simd_fvec<S> A = (g - c) / (g + c);
    const simd_fvec<S> B = (c * (g + c) - 1) / (c * (g - c) + 1);
    where(g > 0.0f, result) = 0.5f * A * A * (1 + B * B);

    return result;
}

template <int S>
void get_lobe_weights(const simd_fvec<S> &base_color_lum, const simd_fvec<S> &spec_color_lum, const float specular,
                      const simd_fvec<S> &metallic, const float transmission, const float clearcoat,
                      simd_fvec<S> *out_diffuse_weight, simd_fvec<S> *out_specular_weight,
                      simd_fvec<S> *out_clearcoat_weight, simd_fvec<S> *out_refraction_weight) {
    // taken from Cycles
    (*out_diffuse_weight) = base_color_lum * (1.0f - metallic) * (1.0f - transmission);
    const simd_fvec<S> final_transmission = transmission * (1.0f - metallic);
    //(*out_specular_weight) =
    //    (specular != 0.0f || metallic != 0.0f) ? spec_color_lum * (1.0f - final_transmission) : 0.0f;
    (*out_specular_weight) = 0.0f;

    auto temp_mask = simd_fvec<S>{specular} != 0.0f | metallic != 0.0f;
    where(temp_mask, *out_specular_weight) = spec_color_lum * (1.0f - final_transmission);

    (*out_clearcoat_weight) = 0.25f * clearcoat * (1.0f - metallic);
    (*out_refraction_weight) = final_transmission * base_color_lum;

    const simd_fvec<S> total_weight =
        (*out_diffuse_weight) + (*out_specular_weight) + (*out_clearcoat_weight) + (*out_refraction_weight);

    where(total_weight != 0.0f, *out_diffuse_weight) = (*out_diffuse_weight) / total_weight;
    where(total_weight != 0.0f, *out_specular_weight) = (*out_specular_weight) / total_weight;
    where(total_weight != 0.0f, *out_clearcoat_weight) = (*out_clearcoat_weight) / total_weight;
    where(total_weight != 0.0f, *out_refraction_weight) = (*out_refraction_weight) / total_weight;
}

template <int S> force_inline simd_fvec<S> power_heuristic(const simd_fvec<S> &a, const simd_fvec<S> &b) {
    const simd_fvec<S> t = a * a;
    return t / (b * b + t);
}

} // namespace NS
} // namespace Ray

template <int DimX, int DimY>
void Ray::NS::GeneratePrimaryRays(const int iteration, const camera_t &cam, const rect_t &r, int w, int h,
                                  const float *halton, aligned_vector<ray_data_t<DimX * DimY>> &out_rays,
                                  aligned_vector<simd_ivec<DimX * DimY>> &out_masks) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    simd_fvec<S> ww = {float(w)}, hh = {float(h)};

    const float k = float(w) / h;

    const float focus_distance = cam.focus_distance;
    const float temp = std::tan(0.5f * cam.fov * PI / 180.0f);
    const float fov_k = temp * focus_distance;
    const float spread_angle = std::atan(2.0f * temp / float(h));

    const simd_fvec<S> fwd[3] = {{cam.fwd[0]}, {cam.fwd[1]}, {cam.fwd[2]}},
                       side[3] = {{cam.side[0]}, {cam.side[1]}, {cam.side[2]}},
                       up[3] = {{cam.up[0]}, {cam.up[1]}, {cam.up[2]}},
                       cam_origin[3] = {{cam.origin[0]}, {cam.origin[1]}, {cam.origin[2]}};

    auto get_pix_dirs = [k, fov_k, focus_distance, &fwd, &side, &up, &cam_origin, &ww,
                         &hh](const simd_fvec<S> &x, const simd_fvec<S> &y, const simd_fvec<S> origin[3],
                              simd_fvec<S> d[3]) {
        const int S = DimX * DimY;

        simd_fvec<S> _dx = 2 * fov_k * x / ww - fov_k;
        simd_fvec<S> _dy = 2 * fov_k * -y / hh + fov_k;

        d[0] = cam_origin[0] + k * _dx * side[0] + _dy * up[0] + fwd[0] * focus_distance;
        d[1] = cam_origin[1] + k * _dx * side[1] + _dy * up[1] + fwd[1] * focus_distance;
        d[2] = cam_origin[2] + k * _dx * side[2] + _dy * up[2] + fwd[2] * focus_distance;

        d[0] = d[0] - origin[0];
        d[1] = d[1] - origin[1];
        d[2] = d[2] - origin[2];

        simd_fvec<DimX *DimY> len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        d[0] /= len;
        d[1] /= len;
        d[2] /= len;
    };

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

            simd_fvec<S> fxx = (simd_fvec<S>)ixx, fyy = (simd_fvec<S>)iyy;

            const simd_ivec<S> hash_val = hash(index);
            simd_fvec<S> rxx = construct_float(hash_val);
            simd_fvec<S> ryy = construct_float(hash(hash_val));
            simd_fvec<S> sxx, syy;

            for (int j = 0; j < S; j++) {
                float _unused;
                sxx[j] = cam.focus_factor * (-0.5f + std::modf(halton[RAND_DIM_LENS_U] + rxx[j], &_unused));
                syy[j] = cam.focus_factor * (-0.5f + std::modf(halton[RAND_DIM_LENS_V] + ryy[j], &_unused));
                rxx[j] = std::modf(halton[RAND_DIM_FILTER_U] + rxx[j], &_unused);
                ryy[j] = std::modf(halton[RAND_DIM_FILTER_V] + ryy[j], &_unused);
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

            const simd_fvec<S> _origin[3] = {{cam_origin[0] + side[0] * sxx + up[0] * syy},
                                             {cam_origin[1] + side[1] * sxx + up[1] * syy},
                                             {cam_origin[2] + side[2] * sxx + up[2] * syy}};

            simd_fvec<S> _d[3], _dx[3], _dy[3];
            get_pix_dirs(fxx, fyy, _origin, _d);
            get_pix_dirs(fxx + 1.0f, fyy, _origin, _dx);
            get_pix_dirs(fxx, fyy + 1.0f, _origin, _dy);

            for (int j = 0; j < 3; j++) {
                out_r.d[j] = _d[j];
                out_r.o[j] = _origin[j];
                out_r.c[j] = {1.0f};

#ifdef USE_RAY_DIFFERENTIALS
                out_r.do_dx[j] = {0.0f};
                out_r.dd_dx[j] = _dx[j] - out_r.d[j];
                out_r.do_dy[j] = {0.0f};
                out_r.dd_dy[j] = _dy[j] - out_r.d[j];
#endif
            }

#ifndef USE_RAY_DIFFERENTIALS
            out_r.cone_width = 0.0f;
            out_r.cone_spread = spread_angle;
#endif

            out_r.pdf = {1e6f};
            out_r.xy = (ixx << 16) | iyy;
            out_r.ray_depth = {0};
        }
    }
}

template <int DimX, int DimY>
void Ray::NS::SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                       const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                                       const rect_t &r, int width, int height, const float *halton,
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
#ifdef USE_RAY_DIFFERENTIALS
            out_ray.do_dx[0] = out_ray.do_dx[1] = out_ray.do_dx[2] = 0.0f;
            out_ray.dd_dx[0] = out_ray.dd_dx[1] = out_ray.dd_dx[2] = 0.0f;
            out_ray.do_dy[0] = out_ray.do_dy[1] = out_ray.do_dy[2] = 0.0f;
            out_ray.dd_dy[0] = out_ray.dd_dy[1] = out_ray.dd_dy[2] = 0.0f;
#else
            out_ray.cone_width = 0.0f;
            out_ray.cone_spread = 0.0f;
#endif
            out_inter.mask = 0;
        }
    }

    const simd_ivec2 irect_min = {r.x, r.y}, irect_max = {r.x + r.w - 1, r.y + r.h - 1};
    const simd_fvec2 size = {float(width), float(height)};

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const vertex_t &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const vertex_t &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const vertex_t &v2 = vertices[vtx_indices[tri * 3 + 2]];

        const auto t0 = simd_fvec2{v0.t[uv_layer][0], 1.0f - v0.t[uv_layer][1]} * size;
        const auto t1 = simd_fvec2{v1.t[uv_layer][0], 1.0f - v1.t[uv_layer][1]} * size;
        const auto t2 = simd_fvec2{v2.t[uv_layer][0], 1.0f - v2.t[uv_layer][1]} * size;

        simd_fvec2 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        simd_ivec2 ibbox_min = (simd_ivec2)(bbox_min),
                   ibbox_max = simd_ivec2{int(std::round(bbox_max[0])), int(std::round(bbox_max[1]))};

        if (ibbox_max[0] < irect_min[0] || ibbox_max[1] < irect_min[1] || ibbox_min[0] > irect_max[0] ||
            ibbox_min[1] > irect_max[1]) {
            continue;
        }

        ibbox_min = max(ibbox_min, irect_min);
        ibbox_max = min(ibbox_max, irect_max);

        ibbox_min[0] -= ibbox_min[0] % DimX;
        ibbox_min[1] -= ibbox_min[1] % DimY;
        ibbox_max[0] += ((ibbox_max[0] + 1) % DimX) ? (DimX - (ibbox_max[0] + 1) % DimX) : 0;
        ibbox_max[1] += ((ibbox_max[1] + 1) % DimY) ? (DimY - (ibbox_max[1] + 1) % DimY) : 0;

        const simd_fvec2 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

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

                const simd_ivec<S> index = iyy * width + ixx;
                const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

                const simd_ivec<S> hash_val = hash(index);
                simd_fvec<S> rxx = construct_float(hash_val);
                simd_fvec<S> ryy = construct_float(hash(hash_val));

                for (int i = 0; i < S; i++) {
                    float _unused;
                    rxx[i] = std::modf(halton[hi + 0] + rxx[i], &_unused);
                    ryy[i] = std::modf(halton[hi + 1] + ryy[i], &_unused);
                }

                const simd_fvec<S> fxx = simd_fvec<S>{ixx} + rxx, fyy = simd_fvec<S>{iyy} + ryy;

                simd_fvec<S> u = d01[0] * (fyy - t0[1]) - d01[1] * (fxx - t0[0]),
                             v = d12[0] * (fyy - t1[1]) - d12[1] * (fxx - t1[0]),
                             w = d20[0] * (fyy - t2[1]) - d20[1] * (fxx - t2[0]);

                const simd_fvec<S> fmask = (u >= -FLT_EPS) & (v >= -FLT_EPS) & (w >= -FLT_EPS);
                const auto &imask = reinterpret_cast<const simd_ivec<S> &>(fmask);

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

                    ITERATE_3({ where(fmask, out_ray.o[i]) = p[i] + n[i]; })
                    ITERATE_3({ where(fmask, out_ray.d[i]) = -n[i]; })
                    // where(fmask, out_ray.ior) = 1.0f;
                    where(reinterpret_cast<const simd_ivec<S> &>(fmask), out_ray.ray_depth) = {0};

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
    for (int start = 0, end = 1; end <= rays_count * S; end++) {
        if (end == (rays_count * S) || (hash_values[start / S][start % S] != hash_values[end / S][end % S])) {
            chunks[chunks_count].hash = hash_values[start / S][start % S];
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
            while (i != (j = scan_values[i])) {
                const int k = scan_values[j];

                {
                    const int jj = j / S, _jj = j % S, kk = k / S, _kk = k % S;

                    std::swap(hash_values[jj][_jj], hash_values[kk][_kk]);

                    std::swap(rays[jj].o[0][_jj], rays[kk].o[0][_kk]);
                    std::swap(rays[jj].o[1][_jj], rays[kk].o[1][_kk]);
                    std::swap(rays[jj].o[2][_jj], rays[kk].o[2][_kk]);

                    std::swap(rays[jj].d[0][_jj], rays[kk].d[0][_kk]);
                    std::swap(rays[jj].d[1][_jj], rays[kk].d[1][_kk]);
                    std::swap(rays[jj].d[2][_jj], rays[kk].d[2][_kk]);

                    std::swap(rays[jj].pdf[_jj], rays[kk].pdf[_kk]);

                    std::swap(rays[jj].c[0][_jj], rays[kk].c[0][_kk]);
                    std::swap(rays[jj].c[1][_jj], rays[kk].c[1][_kk]);
                    std::swap(rays[jj].c[2][_jj], rays[kk].c[2][_kk]);

#ifdef USE_RAY_DIFFERENTIALS
                    std::swap(rays[jj].do_dx[0][_jj], rays[kk].do_dx[0][_kk]);
                    std::swap(rays[jj].do_dx[1][_jj], rays[kk].do_dx[1][_kk]);
                    std::swap(rays[jj].do_dx[2][_jj], rays[kk].do_dx[2][_kk]);

                    std::swap(rays[jj].dd_dx[0][_jj], rays[kk].dd_dx[0][_kk]);
                    std::swap(rays[jj].dd_dx[1][_jj], rays[kk].dd_dx[1][_kk]);
                    std::swap(rays[jj].dd_dx[2][_jj], rays[kk].dd_dx[2][_kk]);

                    std::swap(rays[jj].do_dy[0][_jj], rays[kk].do_dy[0][_kk]);
                    std::swap(rays[jj].do_dy[1][_jj], rays[kk].do_dy[1][_kk]);
                    std::swap(rays[jj].do_dy[2][_jj], rays[kk].do_dy[2][_kk]);

                    std::swap(rays[jj].dd_dy[0][_jj], rays[kk].dd_dy[0][_kk]);
                    std::swap(rays[jj].dd_dy[1][_jj], rays[kk].dd_dy[1][_kk]);
                    std::swap(rays[jj].dd_dy[2][_jj], rays[kk].dd_dy[2][_kk]);
#else
                    std::swap(rays[jj].cone_width[_jj], rays[kk].cone_width[_kk]);
                    std::swap(rays[jj].cone_spread[_jj], rays[kk].cone_spread[_kk]);
#endif

                    std::swap(rays[jj].xy[_jj], rays[kk].xy[_kk]);
                    std::swap(rays[jj].ray_depth[_jj], rays[kk].ray_depth[_kk]);

                    std::swap(ray_masks[jj][_jj], ray_masks[kk][_kk]);
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
        chunks_count = cur_sum;
    }

    // init ray chunks hash and base index
    for (int i = 0; i < rays_count * S; i++) {
        if (head_flags[i]) {
            chunks[scan_values[i]].hash = reinterpret_cast<const uint32_t &>(hash_values[i / S][i % S]);
            chunks[scan_values[i]].base = (uint32_t)i;
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

                    std::swap(hash_values[jj][_jj], hash_values[kk][_kk]);

                    std::swap(rays[jj].o[0][_jj], rays[kk].o[0][_kk]);
                    std::swap(rays[jj].o[1][_jj], rays[kk].o[1][_kk]);
                    std::swap(rays[jj].o[2][_jj], rays[kk].o[2][_kk]);

                    std::swap(rays[jj].d[0][_jj], rays[kk].d[0][_kk]);
                    std::swap(rays[jj].d[1][_jj], rays[kk].d[1][_kk]);
                    std::swap(rays[jj].d[2][_jj], rays[kk].d[2][_kk]);

                    std::swap(rays[jj].pdf[_jj], rays[kk].pdf[_kk]);

                    std::swap(rays[jj].c[0][_jj], rays[kk].c[0][_kk]);
                    std::swap(rays[jj].c[1][_jj], rays[kk].c[1][_kk]);
                    std::swap(rays[jj].c[2][_jj], rays[kk].c[2][_kk]);

#ifdef USE_RAY_DIFFERENTIALS
                    std::swap(rays[jj].do_dx[0][_jj], rays[kk].do_dx[0][_kk]);
                    std::swap(rays[jj].do_dx[1][_jj], rays[kk].do_dx[1][_kk]);
                    std::swap(rays[jj].do_dx[2][_jj], rays[kk].do_dx[2][_kk]);

                    std::swap(rays[jj].dd_dx[0][_jj], rays[kk].dd_dx[0][_kk]);
                    std::swap(rays[jj].dd_dx[1][_jj], rays[kk].dd_dx[1][_kk]);
                    std::swap(rays[jj].dd_dx[2][_jj], rays[kk].dd_dx[2][_kk]);

                    std::swap(rays[jj].do_dy[0][_jj], rays[kk].do_dy[0][_kk]);
                    std::swap(rays[jj].do_dy[1][_jj], rays[kk].do_dy[1][_kk]);
                    std::swap(rays[jj].do_dy[2][_jj], rays[kk].do_dy[2][_kk]);

                    std::swap(rays[jj].dd_dy[0][_jj], rays[kk].dd_dy[0][_kk]);
                    std::swap(rays[jj].dd_dy[1][_jj], rays[kk].dd_dy[1][_kk]);
                    std::swap(rays[jj].dd_dy[2][_jj], rays[kk].dd_dy[2][_kk]);
#else
                    std::swap(rays[jj].cone_width[_jj], rays[kk].cone_width[_kk]);
                    std::swap(rays[jj].cone_spread[_jj], rays[kk].cone_spread[_kk]);
#endif

                    std::swap(rays[jj].xy[_jj], rays[kk].xy[_kk]);

                    std::swap(ray_masks[jj][_jj], ray_masks[kk][_kk]);
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
    hit_data_t<S> inter = {Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
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
bool Ray::NS::IntersectTris_ClosestHit(const float o[3], const float d[3], int i, const mtri_accel_t *mtris,
                                       const int tri_start, const int tri_end, const int obj_index,
                                       hit_data_t<S> &out_inter) {
    bool res = false;

    for (int j = tri_start / 8; j < (tri_end + 7) / 8; ++j) {
        res |= _IntersectTri(o, d, i, mtris[j], j * 8, out_inter);
    }

    if (res) {
        out_inter.obj_index[i] = obj_index;
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
    hit_data_t<S> inter = {Uninitialize};
    inter.mask = {0};
    inter.obj_index = {reinterpret_cast<const int &>(obj_index)};
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
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
bool Ray::NS::IntersectTris_AnyHit(const float o[3], const float d[3], int i, const mtri_accel_t *mtris,
                                   const tri_mat_data_t *materials, const uint32_t *indices, const int tri_start,
                                   const int tri_end, const int obj_index, hit_data_t<S> &out_inter) {
    bool res = false;

    for (int j = tri_start / 8; j < (tri_end + 7) / 8; j++) {
        res |= _IntersectTri(o, d, i, mtris[j], j * 8, out_inter);
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

    for (int ri = 0; ri < S; ri++) {
        if (!ray_mask[ri]) {
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
                alignas(S * 4) float res_dist[8];
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

                    if (!bbox_test(_inv_d, _inv_d_o, inter.t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, tr.inv_xform, tr_ro, tr_rd);

                    res |= Traverse_MicroTree_WithStack_ClosestHit(tr_ro, tr_rd, ri, nodes, m.node_index, mtris,
                                                                   tri_indices, int(mi_indices[j]), inter);
                }
            }
        }
    }

    // resolve primitive index indirection
    const simd_ivec<S> is_backfacing = (inter.prim_index < 0);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    inter.prim_index = gather(reinterpret_cast<const int *>(tri_indices), inter.prim_index);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return res;
}

template <int S>
bool Ray::NS::Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                  const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                  uint32_t node_index, const mesh_instance_t *mesh_instances,
                                                  const uint32_t *mi_indices, const mesh_t *meshes,
                                                  const transform_t *transforms, const tri_accel_t *tris,
                                                  const tri_mat_data_t *materials, const uint32_t *tri_indices,
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

                    const bool hit_found =
                        Traverse_MicroTree_WithStack_AnyHit(_ro, _rd, bbox_mask, nodes, m.node_index, tris, materials,
                                                            tri_indices, int(mi_indices[i]), inter);
                    res |= hit_found;
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
bool Ray::NS::Traverse_MacroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                  const simd_ivec<S> &ray_mask, const mbvh_node_t *nodes,
                                                  uint32_t node_index, const mesh_instance_t *mesh_instances,
                                                  const uint32_t *mi_indices, const mesh_t *meshes,
                                                  const transform_t *transforms, const mtri_accel_t *mtris,
                                                  const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                  hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3], inv_d_o[3];
    comp_aux_inv_values(ro, rd, inv_d, inv_d_o);

    for (int ri = 0; ri < S; ri++) {
        if (!ray_mask[ri]) {
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
                alignas(S * 4) float res_dist[8];
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

                    if (!bbox_test(_inv_d, _inv_d_o, inter.t[ri], mi.bbox_min, mi.bbox_max)) {
                        continue;
                    }

                    float tr_ro[3], tr_rd[3];
                    TransformRay(r_o, r_d, tr.inv_xform, tr_ro, tr_rd);

                    const bool hit_found =
                        Traverse_MicroTree_WithStack_AnyHit(tr_ro, tr_rd, ri, nodes, m.node_index, mtris, materials,
                                                            tri_indices, int(mi_indices[j]), inter);
                    res |= hit_found;
                }
            }
        }
    }

    // resolve primitive index indirection
    const simd_ivec<S> is_backfacing = (inter.prim_index < 0);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    inter.prim_index = gather(reinterpret_cast<const int *>(tri_indices), inter.prim_index);
    where(is_backfacing, inter.prim_index) = -inter.prim_index - 1;

    return res;
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
                const int tri_start = nodes[cur].prim_index & PRIM_INDEX_BITS,
                          tri_end = tri_start + nodes[cur].prim_count;
                res |= IntersectTris_ClosestHit(ro, rd, st.queue[st.index].mask, tris, tri_start, tri_end, obj_index,
                                                inter);
            }
        }
        st.index++;
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack_ClosestHit(const float ro[3], const float rd[3], int ri,
                                                      const mbvh_node_t *nodes, uint32_t node_index,
                                                      const mtri_accel_t *mtris, const uint32_t *tri_indices,
                                                      int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    float _inv_d[3], _inv_d_o[3];
    comp_aux_inv_values(ro, rd, _inv_d, _inv_d_o);

    TraversalStateStack_Single<MAX_STACK_SIZE> st;
    st.push(node_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t[ri]) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(S * 4) float res_dist[8];
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
            const int tri_start = nodes[cur.index].child[0] & PRIM_INDEX_BITS,
                      tri_end = tri_start + nodes[cur.index].child[1];
            res |= IntersectTris_ClosestHit(ro, rd, ri, mtris, tri_start, tri_end, obj_index, inter);
        }
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack_AnyHit(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3],
                                                  const simd_ivec<S> &ray_mask, const bvh_node_t *nodes,
                                                  uint32_t node_index, const tri_accel_t *tris,
                                                  const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                                  int obj_index, hit_data_t<S> &inter) {
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
                /*if (hit_found) {
                    const bool is_backfacing = inter.prim_index < 0;
                    const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                    if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                        (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                        return true;
                    }
                }*/
                res |= hit_found;
            }
        }
        st.index++;
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], int ri,
                                                  const mbvh_node_t *nodes, uint32_t node_index,
                                                  const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                                  const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    float _inv_d[3], _inv_d_o[3];
    comp_aux_inv_values(ro, rd, _inv_d, _inv_d_o);

    TraversalStateStack_Single<MAX_STACK_SIZE> st;
    st.push(node_index, 0.0f);

    while (!st.empty()) {
        stack_entry_t cur = st.pop();

        if (cur.dist > inter.t[ri]) {
            continue;
        }

    TRAVERSE:
        if (!is_leaf_node(nodes[cur.index])) {
            alignas(S * 4) float res_dist[8];
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
            const int tri_start = nodes[cur.index].child[0] & PRIM_INDEX_BITS,
                      tri_end = tri_start + nodes[cur.index].child[1];
            const bool hit_found =
                IntersectTris_AnyHit(ro, rd, ri, mtris, materials, tri_indices, tri_start, tri_end, obj_index, inter);
            /*if (hit_found) {
                const bool is_backfacing = inter.prim_index < 0;
                const uint32_t prim_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

                if ((!is_backfacing && (materials[tri_indices[prim_index]].front_mi & MATERIAL_SOLID_BIT)) ||
                    (is_backfacing && (materials[tri_indices[prim_index]].back_mi & MATERIAL_SOLID_BIT))) {
                    return true;
                }
            }*/
            res |= hit_found;
        }
    }

    return res;
}

template <int S>
Ray::NS::simd_fvec<S> Ray::NS::BRDF_PrincipledDiffuse(const simd_fvec<S> V[3], const simd_fvec<S> N[3],
                                                      const simd_fvec<S> L[3], const simd_fvec<S> H[3],
                                                      const simd_fvec<S> &roughness) {
    const simd_fvec<S> N_dot_L = dot(N, L);
    const simd_fvec<S> N_dot_V = dot(N, V);

    const simd_fvec<S> FL = schlick_weight(N_dot_L);
    const simd_fvec<S> FV = schlick_weight(N_dot_V);

    const simd_fvec<S> L_dot_H = dot(L, H);
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

    const simd_fvec<S> nl = max(dot(N, L), 0.0f);
    const simd_fvec<S> nv = max(dot(N, V), 0.0f);
    simd_fvec<S> t = dot(L, V) - nl * nv;

    where(t > 0.0f, t) /= (max(nl, nv) + FLT_MIN);

    const simd_fvec<S> is = nl * (a + b * t);

    ITERATE_3({ out_color[i] = is * base_color[i]; })
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

    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};
    Evaluate_OrenDiffuse_BSDF(_I, N, out_V, roughness, base_color, out_color);
}

template <int S>
void Ray::NS::Evaluate_PrincipledDiffuse_BSDF(const simd_fvec<S> V[3], const simd_fvec<S> N[3], const simd_fvec<S> L[3],
                                              const simd_fvec<S> &roughness, const simd_fvec<S> base_color[3],
                                              const simd_fvec<S> sheen_color[3], const bool uniform_sampling,
                                              simd_fvec<S> out_color[4]) {
    simd_fvec<S> weight, pdf;
    if (uniform_sampling) {
        weight = 2 * dot(N, L);
        pdf = 0.5f / PI;
    } else {
        weight = 1.0f;
        pdf = dot(N, L) / PI;
    }

    simd_fvec<S> H[3] = {L[0] + V[0], L[1] + V[1], L[2] + V[2]};
    normalize(H);

    const simd_fvec<S> dot_VH = dot(V, H);
    ITERATE_3({ where(dot_VH < 0.0f, H[i]) = -H[i]; })

    weight *= BRDF_PrincipledDiffuse(V, N, L, H, roughness);
    ITERATE_3({ out_color[i] = base_color[i] * weight; })

    const simd_fvec<S> FH = PI * schlick_weight(dot(L, H));
    ITERATE_3({ out_color[i] += FH * sheen_color[i]; })
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

    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};
    Evaluate_PrincipledDiffuse_BSDF(_I, N, out_V, roughness, base_color, sheen_color, uniform_sampling, out_color);
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
        (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0f - spec_F0);

    simd_fvec<S> F[3];
    ITERATE_3({ F[i] = mix(spec_col[i], simd_fvec<S>{1.0f}, FH); })

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2] * reflected_dir_ts[2]);
    ITERATE_3({
        F[i] *= (D * G / denom);
        where(denom == 0.0f, F[i]) = 0.0f;
    })

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf =
        D * G1(view_dir_ts, alpha_x, alpha_y) * max(dot(view_dir_ts, sampled_normal_ts), 0.0f) / abs(view_dir_ts[2]);
    where(abs(view_dir_ts[2]) == 0.0f, pdf) = 0.0f;

    const simd_fvec<S> div = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    where(div != 0.0f, pdf) = pdf / div;
#else
    const float pdf = D * sampled_normal_ts[2] / (4.0f * dot(view_dir_ts, sampled_normal_ts));
#endif

    ITERATE_3({ out_color[i] = F[i] * max(reflected_dir_ts[2], 0.0f); })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_GGXSpecular_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                      const simd_fvec<S> I[3], const simd_fvec<S> &roughness,
                                      const simd_fvec<S> &anisotropic, const simd_fvec<S> &spec_ior,
                                      const simd_fvec<S> &spec_F0, const simd_fvec<S> spec_col[3],
                                      const simd_fvec<S> &rand_u, const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                                      simd_fvec<S> out_color[4]) {
    const simd_fvec<S> roughness2 = roughness * roughness;
    const simd_fvec<S> aspect = sqrt(1.0f - 0.9f * anisotropic);

    const simd_fvec<S> alpha_x = roughness2 / aspect;
    const simd_fvec<S> alpha_y = roughness2 * aspect;

    const simd_ivec<S> is_mirror = simd_cast(alpha_x * alpha_y < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        reflect(I, N, dot(N, I), out_V);
        const simd_fvec<S> FH = (fresnel_dielectric_cos(dot(out_V, N), spec_ior) - spec_F0) / (1.0f - spec_F0);
        ITERATE_3({ out_color[i] = mix(spec_col[i], simd_fvec<S>{1.0f}, FH) * 1e6f; })
        out_color[3] = 1e6f;
    }

    const simd_ivec<S> is_glossy = ~is_mirror;
    if (is_glossy.all_zeros()) {
        return;
    }

    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, _I, view_dir_ts);
    normalize(view_dir_ts);

    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, alpha_x, alpha_y, rand_u, rand_v, sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif

    const simd_fvec<S> dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    simd_fvec<S> reflected_dir_ts[3];
    const simd_fvec<S> _view_dir_ts[3] = {-view_dir_ts[0], -view_dir_ts[1], -view_dir_ts[2]};
    reflect(_view_dir_ts, sampled_normal_ts, dot_N_V, reflected_dir_ts);
    normalize(reflected_dir_ts);

    simd_fvec<S> glossy_V[3], glossy_F[4];
    world_from_tangent(T, B, N, reflected_dir_ts, glossy_V);
    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha_x, alpha_y, spec_ior, spec_F0,
                              spec_col, glossy_F);

    ITERATE_3({ where(is_glossy, out_V[i]) = glossy_V[i]; })
    ITERATE_4({ where(is_glossy, out_color[i]) = glossy_F[i]; })
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

    const simd_fvec<S> denom = dot(refr_dir_ts, sampled_normal_ts) + dot(view_dir_ts, sampled_normal_ts) * eta;
    const simd_fvec<S> jacobian = max(-dot(refr_dir_ts, sampled_normal_ts), 0.0f) / (denom * denom);

    simd_fvec<S> F = D * G1i * G1o * max(dot(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian /
                     (/*-refr_dir_ts[2] */ view_dir_ts[2]);

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf = D * G1o * max(dot(view_dir_ts, sampled_normal_ts), 0.0f) * jacobian / view_dir_ts[2];
#else
    // const float pdf = D * std::max(sampled_normal_ts[2], 0.0f) * jacobian;
    const float pdf = D * sampled_normal_ts[2] * std::max(-dot(refr_dir_ts, sampled_normal_ts), 0.0f) / denom;
#endif

    const simd_fvec<S> is_valid = (refr_dir_ts[2]<0.0f & view_dir_ts[2]> 0.0f);

    ITERATE_3({
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
    const simd_fvec<S> roughness2 = (roughness * roughness);
    const simd_ivec<S> is_mirror = simd_cast(roughness2 * roughness2 < 1e-7f);
    if (is_mirror.not_all_zeros()) {
        const simd_fvec<S> cosi = -dot(I, N);
        const simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);

        const simd_fvec<S> m = eta * cosi - sqrt(cost2);
        ITERATE_3({ out_V[i] = eta * I[i] + m * N[i]; })
        normalize(out_V);

        out_V[3] = m;
        ITERATE_4({
            out_color[i] = 1e6f;
            where(cost2 < 0, out_color[i]) = 0.0f;
        })
    }

    const simd_ivec<S> is_glossy = ~is_mirror;
    if (is_glossy.all_zeros()) {
        return;
    }

    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, _I, view_dir_ts);
    normalize(view_dir_ts);

    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, roughness2, roughness2, rand_u, rand_v, sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(alpha_x, rand_u, rand_v);
#endif

    const simd_fvec<S> cosi = dot(view_dir_ts, sampled_normal_ts);
    const simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);

    ITERATE_4({ where(is_glossy, out_color[i]) = 0.0f; })

    const simd_ivec<S> cost2_positive = simd_cast(cost2 >= 0.0f);
    if (cost2_positive.not_all_zeros()) {
        const simd_fvec<S> m = eta * cosi - sqrt(cost2);
        simd_fvec<S> refr_dir_ts[3];
        ITERATE_3({ refr_dir_ts[i] = -eta * view_dir_ts[i] + m * sampled_normal_ts[i]; })
        normalize(refr_dir_ts);

        simd_fvec<S> glossy_V[3], glossy_F[4];
        world_from_tangent(T, B, N, refr_dir_ts, glossy_V);
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, roughness2, eta, refr_col, glossy_F);

        ITERATE_3({ where(is_glossy & cost2_positive, out_V[i]) = glossy_V[i]; })
        ITERATE_4({ where(is_glossy & cost2_positive, out_color[i]) = glossy_F[i]; })
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
        (fresnel_dielectric_cos(dot(reflected_dir_ts, sampled_normal_ts), clearcoat_ior) - clearcoat_F0) /
        (1.0f - clearcoat_F0);
    simd_fvec<S> F = mix(simd_fvec<S>{0.04f}, simd_fvec<S>{1.0f}, FH);

    const simd_fvec<S> denom = 4.0f * abs(view_dir_ts[2]) * abs(reflected_dir_ts[2]);
    F *= (D * G / denom);
    where(denom == 0.0f, F) = 0.0f;

#if USE_VNDF_GGX_SAMPLING == 1
    simd_fvec<S> pdf = D * G1(view_dir_ts, clearcoat_alpha, clearcoat_alpha) *
                       max(dot(view_dir_ts, sampled_normal_ts), 0.0f) / abs(view_dir_ts[2]);
    const simd_fvec<S> div = 4.0f * dot(view_dir_ts, sampled_normal_ts);
    where(div != 0.0f, pdf) = pdf / div;
#else
    float pdf = D * sampled_normal_ts[2] / (4.0f * dot(view_dir_ts, sampled_normal_ts));
#endif

    F *= max(reflected_dir_ts[2], 0.0f);

    ITERATE_3({ out_color[i] = F; })
    out_color[3] = pdf;
}

template <int S>
void Ray::NS::Sample_PrincipledClearcoat_BSDF(const simd_fvec<S> T[3], const simd_fvec<S> B[3], const simd_fvec<S> N[3],
                                              const simd_fvec<S> I[3], float clearcoat_roughness2, float clearcoat_ior,
                                              float clearcoat_F0, const simd_fvec<S> &rand_u,
                                              const simd_fvec<S> &rand_v, simd_fvec<S> out_V[3],
                                              simd_fvec<S> out_color[4]) {
    if (clearcoat_roughness2 * clearcoat_roughness2 < 1e-7f) {
        reflect(I, N, dot(N, I), out_V);

        const simd_fvec<S> FH =
            (fresnel_dielectric_cos(dot(out_V, N), simd_fvec<S>{clearcoat_ior}) - clearcoat_F0) / (1.0f - clearcoat_F0);
        const simd_fvec<S> F = mix(simd_fvec<S>{0.04f}, simd_fvec<S>{1.0f}, FH);

        ITERATE_3({ out_color[i] = F * 1e6f; })
        out_color[3] = 1e6f;
        return;
    }

    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};

    simd_fvec<S> view_dir_ts[3];
    tangent_from_world(T, B, N, _I, view_dir_ts);
    normalize(view_dir_ts);

    // NOTE: GTR1 distribution is not used for sampling because Cycles does it this way (???!)
    simd_fvec<S> sampled_normal_ts[3];
#if USE_VNDF_GGX_SAMPLING == 1
    SampleGGX_VNDF(view_dir_ts, simd_fvec<S>{clearcoat_roughness2}, simd_fvec<S>{clearcoat_roughness2}, rand_u, rand_v,
                   sampled_normal_ts);
#else
    const simd_fvec4 sampled_normal_ts = sample_GGX_NDF(clearcoat_roughness2, rand_u, rand_v);
#endif

    const simd_fvec<S> dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    simd_fvec<S> reflected_dir_ts[3];
    const simd_fvec<S> _view_dir_ts[3] = {-view_dir_ts[0], -view_dir_ts[1], -view_dir_ts[2]};
    reflect(_view_dir_ts, sampled_normal_ts, dot_N_V, reflected_dir_ts);
    normalize(reflected_dir_ts);

    world_from_tangent(T, B, N, reflected_dir_ts, out_V);
    Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts,
                                      simd_fvec<S>{clearcoat_roughness2}, simd_fvec<S>{clearcoat_ior},
                                      simd_fvec<S>{clearcoat_F0}, out_color);
}

template <int S>
force_inline void Ray::NS::TransformRay(const simd_fvec<S> ro[3], const simd_fvec<S> rd[3], const float *xform,
                                        simd_fvec<S> out_ro[3], simd_fvec<S> out_rd[3]) {
    out_ro[0] = ro[0] * xform[0] + ro[1] * xform[4] + ro[2] * xform[8] + xform[12];
    out_ro[1] = ro[0] * xform[1] + ro[1] * xform[5] + ro[2] * xform[9] + xform[13];
    out_ro[2] = ro[0] * xform[2] + ro[1] * xform[6] + ro[2] * xform[10] + xform[14];

    out_rd[0] = rd[0] * xform[0] + rd[1] * xform[4] + rd[2] * xform[8];
    out_rd[1] = rd[0] * xform[1] + rd[1] * xform[5] + rd[2] * xform[9];
    out_rd[2] = rd[0] * xform[2] + rd[1] * xform[6] + rd[2] * xform[10];
}

force_inline void Ray::NS::TransformRay(const float ro[3], const float rd[3], const float *xform, float out_ro[3],
                                        float out_rd[3]) {
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

template <int S>
void Ray::NS::TransformUVs(const simd_fvec<S> _uv[2], float sx, float sy, const texture_t &t,
                           const simd_ivec<S> &mip_level, simd_fvec<S> out_res[2]) {
    simd_fvec<S> pos[2];

    ITERATE(S, {
        pos[0][i] = float(t.pos[mip_level[i]][0]);
        pos[1][i] = float(t.pos[mip_level[i]][1]);
    });

    simd_fvec<S> size[2] = {float(t.width & TEXTURE_WIDTH_BITS), float(t.height & TEXTURE_HEIGHT_BITS)};
    if (t.height & TEXTURE_MIPS_BIT) {
        size[0] = simd_fvec<S>((t.width & TEXTURE_WIDTH_BITS) >> mip_level);
        size[1] = simd_fvec<S>((t.height & TEXTURE_HEIGHT_BITS) >> mip_level);
    }

    const simd_fvec<S> uv[2] = {_uv[0] - floor(_uv[0]), _uv[1] - floor(_uv[1])};
    const simd_fvec<S> res[2] = {pos[0] + uv[0] * size[0] + 1.0f, pos[1] + uv[1] * size[1] + 1.0f};

    out_res[0] = res[0] / sx;
    out_res[1] = res[1] / sy;
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
void Ray::NS::SampleNearest(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                            const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    const Ref::TextureAtlasBase &atlas = *atlases[t.atlas];
    simd_ivec<S> _lod = (simd_ivec<S>)lod;

    simd_fvec<S> _uvs[2];
    TransformUVs(uvs, atlas.size_x(), atlas.size_y(), t, _lod, _uvs);

    where(_lod > MAX_MIP_LEVEL, _lod) = MAX_MIP_LEVEL;

    for (int j = 0; j < S; j++) {
        if (!mask[j]) {
            continue;
        }

        const int page = t.page[_lod[j]];

        const auto &pix = atlas.Fetch(page, _uvs[0][j], _uvs[1][j]);

        ITERATE_4({ out_rgba[i][j] = static_cast<float>(pix.v[i]); });
    }

    const float k = 1.0f / 255.0f;
    ITERATE_4({ out_rgba[i] *= k; })
}

template <int S>
void Ray::NS::SampleBilinear(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                             const simd_ivec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    const Ref::TextureAtlasBase &atlas = *atlases[t.atlas];

    simd_fvec<S> _uvs[2];
    TransformUVs(uvs, atlas.size_x(), atlas.size_y(), t, lod, _uvs);

    _uvs[0] = _uvs[0] * atlas.size_x() - 0.5f;
    _uvs[1] = _uvs[1] * atlas.size_y() - 0.5f;

    const simd_fvec<S> k[2] = {_uvs[0] - floor(_uvs[0]), _uvs[1] - floor(_uvs[1])};

    simd_fvec<S> p0[4], p1[4];

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const int page = t.page[lod[i]];

        const auto &p00 = atlas.Fetch(page, int(_uvs[0][i]), int(_uvs[1][i]));
        const auto &p01 = atlas.Fetch(page, int(_uvs[0][i] + 1), int(_uvs[1][i]));
        const auto &p10 = atlas.Fetch(page, int(_uvs[0][i]), int(_uvs[1][i] + 1));
        const auto &p11 = atlas.Fetch(page, int(_uvs[0][i] + 1), int(_uvs[1][i] + 1));

        p0[0][i] = p01.v[0] * k[0][i] + p00.v[0] * (1 - k[0][i]);
        p0[1][i] = p01.v[1] * k[0][i] + p00.v[1] * (1 - k[0][i]);
        p0[2][i] = p01.v[2] * k[0][i] + p00.v[2] * (1 - k[0][i]);
        p0[3][i] = p01.v[3] * k[0][i] + p00.v[3] * (1 - k[0][i]);

        p1[0][i] = p11.v[0] * k[0][i] + p10.v[0] * (1 - k[0][i]);
        p1[1][i] = p11.v[1] * k[0][i] + p10.v[1] * (1 - k[0][i]);
        p1[2][i] = p11.v[2] * k[0][i] + p10.v[2] * (1 - k[0][i]);
        p1[3][i] = p11.v[3] * k[0][i] + p10.v[3] * (1 - k[0][i]);
    }

    where(mask, out_rgba[0]) = (p1[0] * k[1] + p0[0] * (1.0f - k[1]));
    where(mask, out_rgba[1]) = (p1[1] * k[1] + p0[1] * (1.0f - k[1]));
    where(mask, out_rgba[2]) = (p1[2] * k[1] + p0[2] * (1.0f - k[1]));
    where(mask, out_rgba[3]) = (p1[3] * k[1] + p0[3] * (1.0f - k[1]));
}

template <int S>
void Ray::NS::SampleBilinear(const Ref::TextureAtlasBase &atlas, const simd_fvec<S> uvs[2], const simd_ivec<S> &page,
                             const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> _p00[4], _p01[4], _p10[4], _p11[4];

    const simd_fvec<S> k[2] = {uvs[0] - floor(uvs[0]), uvs[1] - floor(uvs[1])};

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = atlas.Fetch(page[i], int(uvs[0][i] + 0), int(uvs[1][i] + 0));
        const auto &p01 = atlas.Fetch(page[i], int(uvs[0][i] + 1), int(uvs[1][i] + 0));
        const auto &p10 = atlas.Fetch(page[i], int(uvs[0][i] + 0), int(uvs[1][i] + 1));
        const auto &p11 = atlas.Fetch(page[i], int(uvs[0][i] + 1), int(uvs[1][i] + 1));

        _p00[0][i] = p00.v[0];
        _p00[1][i] = p00.v[1];
        _p00[2][i] = p00.v[2];
        _p00[3][i] = p00.v[3];

        _p01[0][i] = p01.v[0];
        _p01[1][i] = p01.v[1];
        _p01[2][i] = p01.v[2];
        _p01[3][i] = p01.v[3];

        _p10[0][i] = p10.v[0];
        _p10[1][i] = p10.v[1];
        _p10[2][i] = p10.v[2];
        _p10[3][i] = p10.v[3];

        _p11[0][i] = p11.v[0];
        _p11[1][i] = p11.v[1];
        _p11[2][i] = p11.v[2];
        _p11[3][i] = p11.v[3];
    }

    const simd_fvec<S> p0X[4] = {_p01[0] * k[0] + _p00[0] * (1 - k[0]), _p01[1] * k[0] + _p00[1] * (1 - k[0]),
                                 _p01[2] * k[0] + _p00[2] * (1 - k[0]), _p01[3] * k[0] + _p00[3] * (1 - k[0])};
    const simd_fvec<S> p1X[4] = {_p11[0] * k[0] + _p10[0] * (1 - k[0]), _p11[1] * k[0] + _p10[1] * (1 - k[0]),
                                 _p11[2] * k[0] + _p10[2] * (1 - k[0]), _p11[3] * k[0] + _p10[3] * (1 - k[0])};

    where(mask, out_rgba[0]) = p1X[0] * k[1] + p0X[0] * (1.0f - k[1]);
    where(mask, out_rgba[1]) = p1X[1] * k[1] + p0X[1] * (1.0f - k[1]);
    where(mask, out_rgba[2]) = p1X[2] * k[1] + p0X[2] * (1.0f - k[1]);
    where(mask, out_rgba[3]) = p1X[3] * k[1] + p0X[3] * (1.0f - k[1]);
}

template <int S>
void Ray::NS::SampleTrilinear(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                              const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> col1[4];
    SampleBilinear(atlases, t, uvs, (simd_ivec<S>)floor(lod), mask, col1);
    simd_fvec<S> col2[4];
    SampleBilinear(atlases, t, uvs, (simd_ivec<S>)ceil(lod), mask, col2);

    const simd_fvec<S> k = lod - floor(lod);

    ITERATE_4({ out_rgba[i] = col1[i] * (1.0f - k) + col2[i] * k; })
}

template <int S>
void Ray::NS::SampleAnisotropic(const Ref::TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec<S> uvs[2],
                                const simd_fvec<S> duv_dx[2], const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask,
                                simd_fvec<S> out_rgba[4]) {
    const Ref::TextureAtlasBase &atlas = *atlases[t.atlas];
    const int width = int(t.width & TEXTURE_WIDTH_BITS), height = int(t.height & TEXTURE_HEIGHT_BITS);

    const simd_fvec<S> _duv_dx[2] = {abs(duv_dx[0] * float(width)), abs(duv_dx[1] * float(height))};

    const simd_fvec<S> l1 = sqrt(_duv_dx[0] * _duv_dx[0] + _duv_dx[1] * _duv_dx[1]);

    const simd_fvec<S> _duv_dy[2] = {abs(duv_dy[0] * float(t.width & TEXTURE_WIDTH_BITS)),
                                     abs(duv_dy[1] * float(t.height & TEXTURE_HEIGHT_BITS))};

    const simd_fvec<S> l2 = sqrt(_duv_dy[0] * _duv_dy[0] + _duv_dy[1] * _duv_dy[1]);

    simd_fvec<S> lod, k = l2 / l1, step[2] = {duv_dx[0], duv_dx[1]};

    ITERATE(S, { lod[i] = fast_log2(std::min(_duv_dy[0][i], _duv_dy[1][i])); })

    const simd_fvec<S> _mask = l1 <= l2;
    where(_mask, k) = l1 / l2;
    where(_mask, step[0]) = duv_dy[0];
    where(_mask, step[1]) = duv_dy[1];

    ITERATE(S, {
        if (reinterpret_cast<const simd_ivec<S> &>(_mask)[i]) {
            lod[i] = fast_log2(std::min(_duv_dx[0][i], _duv_dx[1][i]));
        }
    })

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > float(MAX_MIP_LEVEL), lod) = float(MAX_MIP_LEVEL);

    const simd_ivec<S> imask = mask == 0;
    where(reinterpret_cast<const simd_fvec<S> &>(imask), lod) = 0.0f;

    simd_fvec<S> _uvs[2] = {uvs[0] - step[0] * 0.5f, uvs[1] - step[1] * 0.5f};

    simd_ivec<S> num = static_cast<simd_ivec<S>>(2.0f / k);
    where(num < 1, num) = 1;
    where(num > 2, num) = 4;

    step[0] /= (simd_fvec<S>)num;
    step[1] /= (simd_fvec<S>)num;

    ITERATE_4({ out_rgba[i] = 0.0f; })

    const auto lod1 = simd_ivec<S>{floor(lod)};
    const auto lod2 = simd_ivec<S>{ceil(lod)};

    simd_ivec<S> page1, page2;
    simd_fvec<S> pos1[2], pos2[2], size1[2], size2[2];

    ITERATE(S, {
        page1[i] = t.page[lod1[i]];
        page2[i] = t.page[lod2[i]];

        pos1[0][i] = t.pos[lod1[i]][0] + 0.5f;
        pos1[1][i] = t.pos[lod1[i]][1] + 0.5f;
        pos2[0][i] = t.pos[lod2[i]][0] + 0.5f;
        pos2[1][i] = t.pos[lod2[i]][1] + 0.5f;
        size1[0][i] = float(width >> lod1[i]);
        size1[1][i] = float(height >> lod1[i]);
        size2[0][i] = float(width >> lod2[i]);
        size2[1][i] = float(height >> lod2[i]);
    })

    const simd_fvec<S> kz = lod - floor(lod);

    const simd_fvec<S> kz_big_enough = kz > 0.0001f;
    const bool skip_z = reinterpret_cast<const simd_ivec<S> &>(kz_big_enough).all_zeros();

    for (int j = 0; j < 4; j++) {
        simd_ivec<S> new_imask = (num > j) & mask;
        if (new_imask.all_zeros()) {
            break;
        }

        const auto &fmask = reinterpret_cast<const simd_fvec<S> &>(new_imask);

        _uvs[0] = _uvs[0] - floor(_uvs[0]);
        _uvs[1] = _uvs[1] - floor(_uvs[1]);

        simd_fvec<S> col[4];

        const simd_fvec<S> _uvs1[2] = {pos1[0] + _uvs[0] * size1[0], pos1[1] + _uvs[1] * size1[1]};
        SampleBilinear(atlas, _uvs1, page1, new_imask, col);
        ITERATE_4({ where(fmask, out_rgba[i]) = out_rgba[i] + (1.0f - kz) * col[i]; })

        if (!skip_z) {
            const simd_fvec<S> _uvs2[2] = {pos2[0] + _uvs[0] * size2[0], pos2[1] + _uvs[1] * size2[1]};
            SampleBilinear(atlas, _uvs2, page2, new_imask, col);
            ITERATE_4({ where(fmask, out_rgba[i]) = out_rgba[i] + kz * col[i]; })
        }

        _uvs[0] = _uvs[0] + step[0];
        _uvs[1] = _uvs[1] + step[1];
    }

    const auto fnum = static_cast<simd_fvec<S>>(num);
    ITERATE_4({ out_rgba[i] /= fnum; })
}

template <int S>
void Ray::NS::SampleLatlong_RGBE(const Ref::TextureAtlasRGBA &atlas, const texture_t &t, const simd_fvec<S> dir[3],
                                 const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]) {
    const simd_fvec<S> r = sqrt(dir[0] * dir[0] + dir[2] * dir[2]);
    const simd_fvec<S> y = clamp(dir[1], -1.0f, 1.0f);
    simd_fvec<S> theta, u = 0.0f;

    where(r > FLT_EPS, u) = clamp(dir[0] / r, -1.0f, 1.0f);

    ITERATE(S, {
        theta[i] = std::acos(y[i]) / PI;
        u[i] = 0.5f * std::acos(u[i]) / PI;
    })

    where(dir[2] < 0.0f, u) = 1.0f - u;

    const simd_fvec<S> uvs[2] = {u * float(t.width & TEXTURE_WIDTH_BITS) + float(t.pos[0][0]) + 1.0f,
                                 theta * float(t.height & TEXTURE_HEIGHT_BITS) + float(t.pos[0][1]) + 1.0f};

    const simd_fvec<S> k[2] = {uvs[0] - floor(uvs[0]), uvs[1] - floor(uvs[1])};

    simd_fvec<S> _p00[3], _p01[3], _p10[3], _p11[3];

    for (int i = 0; i < S; i++) {
        if (!mask[i]) {
            continue;
        }

        const auto &p00 = atlas.Get(t.page[0], int(uvs[0][i] + 0), int(uvs[1][i] + 0));
        const auto &p01 = atlas.Get(t.page[0], int(uvs[0][i] + 1), int(uvs[1][i] + 0));
        const auto &p10 = atlas.Get(t.page[0], int(uvs[0][i] + 0), int(uvs[1][i] + 1));
        const auto &p11 = atlas.Get(t.page[0], int(uvs[0][i] + 1), int(uvs[1][i] + 1));

        float f = std::exp2(float(p00.v[3]) - 128.0f);
        _p00[0][i] = to_norm_float(p00.v[0]) * f;
        _p00[1][i] = to_norm_float(p00.v[1]) * f;
        _p00[2][i] = to_norm_float(p00.v[2]) * f;

        f = std::exp2(float(p01.v[3]) - 128.0f);
        _p01[0][i] = to_norm_float(p01.v[0]) * f;
        _p01[1][i] = to_norm_float(p01.v[1]) * f;
        _p01[2][i] = to_norm_float(p01.v[2]) * f;

        f = std::exp2(float(p10.v[3]) - 128.0f);
        _p10[0][i] = to_norm_float(p10.v[0]) * f;
        _p10[1][i] = to_norm_float(p10.v[1]) * f;
        _p10[2][i] = to_norm_float(p10.v[2]) * f;

        f = std::exp2(float(p11.v[3]) - 128.0f);
        _p11[0][i] = to_norm_float(p11.v[0]) * f;
        _p11[1][i] = to_norm_float(p11.v[1]) * f;
        _p11[2][i] = to_norm_float(p11.v[2]) * f;
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
Ray::NS::simd_fvec<S> Ray::NS::ComputeVisibility(const simd_fvec<S> p[3], const simd_fvec<S> d[3], simd_fvec<S> dist,
                                                 const simd_ivec<S> &mask, const float rand_val,
                                                 const simd_ivec<S> &rand_hash2, const scene_data_t &sc,
                                                 uint32_t node_index, const Ref::TextureAtlasBase *atlases[]) {
    simd_fvec<S> sh_ro[3];
    ITERATE_3({ sh_ro[i] = p[i]; })

    simd_fvec<S> visibility = 1.0f;

    simd_ivec<S> keep_going = simd_cast(dist > HIT_EPS) & mask;
    while (keep_going.not_all_zeros()) {
        hit_data_t<S> sh_inter;
        sh_inter.t = dist;

        if (sc.mnodes) {
            Traverse_MacroTree_WithStack_AnyHit(sh_ro, d, keep_going, sc.mnodes, node_index, sc.mesh_instances,
                                                sc.mi_indices, sc.meshes, sc.transforms, sc.mtris, sc.tri_materials,
                                                sc.tri_indices, sh_inter);
        } else {
            Traverse_MacroTree_WithStack_AnyHit(sh_ro, d, keep_going, sc.nodes, node_index, sc.mesh_instances,
                                                sc.mi_indices, sc.meshes, sc.transforms, sc.tris, sc.tri_materials,
                                                sc.tri_indices, sh_inter);
        }

        if (sh_inter.mask.all_zeros()) {
            break;
        }

        const simd_fvec<S> *I = d;
        const simd_fvec<S> w = 1.0f - sh_inter.u - sh_inter.v;

        simd_fvec<S> p1[3], p2[3], p3[3], n1[3], n2[3], n3[3], u1[2], u2[2], u3[2];

        simd_ivec<S> mat_index = {-1}, back_mat_index = {-1};

        simd_fvec<S> inv_xform1[3], inv_xform2[3], inv_xform3[3];
        
        for (int i = 0; i < S; i++) {
            if (!sh_inter.mask[i]) {
                continue;
            }

            const bool is_backfacing = (sh_inter.prim_index[i] < 0);
            const uint32_t tri_index = is_backfacing ? -sh_inter.prim_index[i] - 1 : sh_inter.prim_index[i];

            const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
            const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
            const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

            p1[0][i] = v1.p[0];
            p1[1][i] = v1.p[1];
            p1[2][i] = v1.p[2];
            p2[0][i] = v2.p[0];
            p2[1][i] = v2.p[1];
            p2[2][i] = v2.p[2];
            p3[0][i] = v3.p[0];
            p3[1][i] = v3.p[1];
            p3[2][i] = v3.p[2];

            n1[0][i] = v1.n[0];
            n1[1][i] = v1.n[1];
            n1[2][i] = v1.n[2];
            n2[0][i] = v2.n[0];
            n2[1][i] = v2.n[1];
            n2[2][i] = v2.n[2];
            n3[0][i] = v3.n[0];
            n3[1][i] = v3.n[1];
            n3[2][i] = v3.n[2];

            u1[0][i] = v1.t[0][0];
            u1[1][i] = v1.t[0][1];
            u2[0][i] = v2.t[0][0];
            u2[1][i] = v2.t[0][1];
            u3[0][i] = v3.t[0][0];
            u3[1][i] = v3.t[0][1];

            mat_index[i] = sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS;
            back_mat_index[i] = sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS;

            const transform_t *tr = &sc.transforms[sc.mesh_instances[sh_inter.obj_index[i]].tr_index];

            inv_xform1[0][i] = tr->inv_xform[0];
            inv_xform1[1][i] = tr->inv_xform[1];
            inv_xform1[2][i] = tr->inv_xform[2];
            inv_xform2[0][i] = tr->inv_xform[4];
            inv_xform2[1][i] = tr->inv_xform[5];
            inv_xform2[2][i] = tr->inv_xform[6];
            inv_xform3[0][i] = tr->inv_xform[8];
            inv_xform3[1][i] = tr->inv_xform[9];
            inv_xform3[2][i] = tr->inv_xform[10];
        }

        simd_fvec<S> sh_N[3];
        {
            simd_fvec<S> e21[3], e31[3];
            ITERATE_3({
                e21[i] = p2[i] - p1[i];
                e31[i] = p3[i] - p1[i];
            })
            cross(e21, e31, sh_N);
        }
        normalize(sh_N);

        const simd_fvec<S> sh_plane_N[3] = {dot(sh_N, inv_xform1), dot(sh_N, inv_xform2), dot(sh_N, inv_xform3)};

        const simd_ivec<S> backfacing = (sh_inter.prim_index < 0);
        where(backfacing, mat_index) = back_mat_index;

        sh_N[0] = n1[0] * w + n2[0] * sh_inter.u + n3[0] * sh_inter.v;
        sh_N[1] = n1[1] * w + n2[1] * sh_inter.u + n3[1] * sh_inter.v;
        sh_N[2] = n1[2] * w + n2[2] * sh_inter.u + n3[2] * sh_inter.v;

        // simd_fvec<S> _dot_I_N = dot(I, sh_N);

        const simd_fvec<S> sh_uvs[2] = {u1[0] * w + u2[0] * sh_inter.u + u3[0] * sh_inter.v,
                                        u1[1] * w + u2[1] * sh_inter.u + u3[1] * sh_inter.v};

        const simd_ivec<S> sh_rand_hash = hash(rand_hash2);
        const simd_fvec<S> sh_rand_offset = construct_float(sh_rand_hash);

        {
            simd_ivec<S> ray_queue[S];
            int index = 0;
            int num = 1;

            ray_queue[0] = sh_inter.mask;

            while (index != num) {
                uint32_t first_mi = 0xffffffff;

                for (int i = 0; i < S; i++) {
                    if (!ray_queue[index][i]) {
                        continue;
                    }

                    if (first_mi == 0xffffffff) {
                        first_mi = mat_index[i];
                    }
                }

                simd_ivec<S> same_mi = (mat_index == first_mi);
                simd_ivec<S> diff_mi = and_not(same_mi, ray_queue[index]);

                if (diff_mi.not_all_zeros()) {
                    ray_queue[num] = diff_mi;
                    num++;
                }

                if (first_mi != 0xffffffff) {
                    const material_t *mat = &sc.materials[first_mi];

                    while (mat->type == MixNode) {
                        simd_fvec<S> _mix_val = 1.0f;

                        if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
                            simd_fvec<S> mix[4];
                            SampleBilinear(atlases, sc.textures[mat->textures[BASE_TEXTURE]], sh_uvs, {0}, same_mi,
                                           mix);
                            _mix_val *= mix[0];
                        }
                        _mix_val *= mat->strength;

                        first_mi = 0xffffffff;

                        simd_fvec<S> sh_rand;

                        for (int i = 0; i < S; i++) {
                            float _unused;
                            sh_rand[i] = std::modf(rand_val + sh_rand_offset[i], &_unused);
                        }

                        for (int i = 0; i < S; i++) {
                            if (!same_mi[i]) {
                                continue;
                            }

                            float mix_val = _mix_val[i];

                            if (sh_rand[i] > mix_val) {
                                mat_index[i] = mat->textures[MIX_MAT1];
                                sh_rand[i] = (sh_rand[i] - mix_val) / (1.0f - mix_val);
                            } else {
                                mat_index[i] = mat->textures[MIX_MAT2];
                                sh_rand[i] = sh_rand[i] / mix_val;
                            }

                            if (first_mi == 0xffffffff) {
                                first_mi = mat_index[i];
                            }
                        }

                        const simd_ivec<S> _same_mi = (mat_index == first_mi);
                        diff_mi = and_not(_same_mi, same_mi);
                        same_mi = _same_mi;

                        if (diff_mi.not_all_zeros()) {
                            ray_queue[num] = diff_mi;
                            num++;
                        }

                        mat = &sc.materials[first_mi];
                    }

                    if (mat->type != TransparentNode) {
                        where(same_mi, visibility) = 0.0f;
                        index++;
                        continue;
                    }
                }

                simd_fvec<S> t = sh_inter.t + HIT_BIAS;
                ITERATE_3({ sh_ro[i] += d[i] * t; })
                dist -= t;

                index++;
            }
        }

        // update mask
        keep_going = simd_cast(dist > HIT_EPS) & mask & simd_cast(visibility > 0.0f);
    }

    return visibility;
}

template <int S>
void Ray::NS::ComputeDerivatives(const simd_fvec<S> I[3], const simd_fvec<S> &t, const simd_fvec<S> do_dx[3],
                                 const simd_fvec<S> do_dy[3], const simd_fvec<S> dd_dx[3], const simd_fvec<S> dd_dy[3],
                                 const simd_fvec<S> p1[3], const simd_fvec<S> p2[3], const simd_fvec<S> p3[3],
                                 const simd_fvec<S> n1[3], const simd_fvec<S> n2[3], const simd_fvec<S> n3[3],
                                 const simd_fvec<S> u1[2], const simd_fvec<S> u2[2], const simd_fvec<S> u3[2],
                                 const simd_fvec<S> plane_N[3], const simd_fvec<S> xform[16],
                                 derivatives_t<S> &out_der) {
    // From 'Tracing Ray Differentials' [1999]

    simd_fvec<S> temp[3];

    const simd_fvec<S> dot_I_N = -dot(I, plane_N);
    simd_fvec<S> inv_dot = 1.0f / dot_I_N;
    where(abs(dot_I_N) < FLT_EPS, inv_dot) = {0.0f};

    ITERATE_3({ temp[i] = do_dx[i] + t * dd_dx[i]; })

    const simd_fvec<S> dt_dx = dot(temp, plane_N) * inv_dot;
    ITERATE_3({ out_der.do_dx[i] = temp[i] + dt_dx * I[i]; })
    ITERATE_3({ out_der.dd_dx[i] = dd_dx[i]; })

    ITERATE_3({ temp[i] = do_dy[i] + t * dd_dy[i]; })

    const simd_fvec<S> dt_dy = dot(temp, plane_N) * inv_dot;
    ITERATE_3({ out_der.do_dy[i] = temp[i] + dt_dy * I[i]; })
    ITERATE_3({ out_der.dd_dy[i] = dd_dy[i]; })

    // From 'Physically Based Rendering: ...' book

    const simd_fvec<S> duv13[2] = {u1[0] - u3[0], u1[1] - u3[1]}, duv23[2] = {u2[0] - u3[0], u2[1] - u3[1]};
    simd_fvec<S> dp13[3] = {p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]},
                 dp23[3] = {p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]};

    TransformDirection(xform, dp13);
    TransformDirection(xform, dp23);

    const simd_fvec<S> det_uv = duv13[0] * duv23[1] - duv13[1] * duv23[0];
    simd_fvec<S> inv_det_uv = 1.0f / det_uv;
    where(abs(det_uv) < FLT_EPS, inv_det_uv) = 0.0f;

    const simd_fvec<S> dpdu[3] = {(duv23[1] * dp13[0] - duv13[1] * dp23[0]) * inv_det_uv,
                                  (duv23[1] * dp13[1] - duv13[1] * dp23[1]) * inv_det_uv,
                                  (duv23[1] * dp13[2] - duv13[1] * dp23[2]) * inv_det_uv};
    const simd_fvec<S> dpdv[3] = {(-duv23[0] * dp13[0] + duv13[0] * dp23[0]) * inv_det_uv,
                                  (-duv23[0] * dp13[1] + duv13[0] * dp23[1]) * inv_det_uv,
                                  (-duv23[0] * dp13[2] + duv13[0] * dp23[2]) * inv_det_uv};

    // System of equations
    simd_fvec<S> A[2][2] = {{dpdu[0], dpdv[0]}, {dpdu[1], dpdv[1]}};
    simd_fvec<S> Bx[2] = {out_der.do_dx[0], out_der.do_dx[1]};
    simd_fvec<S> By[2] = {out_der.do_dy[0], out_der.do_dy[1]};

    simd_fvec<S> mask1 = (abs(plane_N[0]) > abs(plane_N[1])) & (abs(plane_N[0]) > abs(plane_N[2]));
    where(mask1, A[0][0]) = dpdu[1];
    where(mask1, A[0][1]) = dpdv[1];
    where(mask1, A[1][0]) = dpdu[2];
    where(mask1, A[1][1]) = dpdv[2];
    where(mask1, Bx[0]) = out_der.do_dx[1];
    where(mask1, Bx[1]) = out_der.do_dx[2];
    where(mask1, By[0]) = out_der.do_dy[1];
    where(mask1, By[1]) = out_der.do_dy[2];

    simd_fvec<S> mask2 = (abs(plane_N[1]) > abs(plane_N[0])) & (abs(plane_N[1]) > abs(plane_N[2]));
    where(mask2, A[1][0]) = dpdu[2];
    where(mask2, A[1][1]) = dpdv[2];
    where(mask2, Bx[1]) = out_der.do_dx[2];
    where(mask2, By[1]) = out_der.do_dy[2];

    // Kramer's rule
    const simd_fvec<S> det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    simd_fvec<S> inv_det = 1.0f / det;
    where(abs(det) < FLT_EPS, inv_det) = {0.0f};

    out_der.duv_dx[0] = (A[1][1] * Bx[0] - A[0][1] * Bx[1]) * inv_det;
    out_der.duv_dx[1] = (A[0][0] * Bx[1] - A[1][0] * Bx[0]) * inv_det;

    out_der.duv_dy[0] = (A[1][1] * By[0] - A[0][1] * By[1]) * inv_det;
    out_der.duv_dy[1] = (A[0][0] * By[1] - A[1][0] * By[0]) * inv_det;

    // Derivative for normal
    const simd_fvec<S> dn1[3] = {n1[0] - n3[0], n1[1] - n3[1], n1[2] - n3[2]},
                       dn2[3] = {n2[0] - n3[0], n2[1] - n3[1], n2[2] - n3[2]};
    const simd_fvec<S> dndu[3] = {(duv23[1] * dn1[0] - duv13[1] * dn2[0]) * inv_det_uv,
                                  (duv23[1] * dn1[1] - duv13[1] * dn2[1]) * inv_det_uv,
                                  (duv23[1] * dn1[2] - duv13[1] * dn2[2]) * inv_det_uv};
    const simd_fvec<S> dndv[3] = {(-duv23[0] * dn1[0] + duv13[0] * dn2[0]) * inv_det_uv,
                                  (-duv23[0] * dn1[1] + duv13[0] * dn2[1]) * inv_det_uv,
                                  (-duv23[0] * dn1[2] + duv13[0] * dn2[2]) * inv_det_uv};

    ITERATE_3({ out_der.dndx[i] = dndu[i] * out_der.duv_dx[0] + dndv[i] * out_der.duv_dx[1]; })
    ITERATE_3({ out_der.dndy[i] = dndu[i] * out_der.duv_dy[0] + dndv[i] * out_der.duv_dy[1]; })

    out_der.ddn_dx = dot(out_der.dd_dx, plane_N) + dot(I, out_der.dndx);
    out_der.ddn_dy = dot(out_der.dd_dy, plane_N) + dot(I, out_der.dndy);
}

// Pick point on any light source for evaluation
template <int S>
void Ray::NS::SampleLightSource(const simd_fvec<S> P[3], const scene_data_t &sc,
                                const Ref::TextureAtlasBase *tex_atlases[], const float halton[],
                                const simd_fvec<S> sample_off[2], const simd_ivec<S> &ray_mask, light_sample_t<S> &ls) {
    const simd_fvec<S> u1 = fract(halton[RAND_DIM_LIGHT_PICK] + sample_off[0]);
    const simd_ivec<S> light_index = min(simd_ivec<S>{u1 * float(sc.li_indices.size())}, int(sc.li_indices.size() - 1));

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

        ITERATE_3({ where(ray_queue[index], ls.col[i]) = l.col[i] * float(sc.li_indices.size()); })

        if (l.type == LIGHT_TYPE_SPHERE) {
            const simd_fvec<S> r1 = fract(halton[RAND_DIM_LIGHT_U] + sample_off[0]);
            const simd_fvec<S> r2 = fract(halton[RAND_DIM_LIGHT_V] + sample_off[1]);

            simd_fvec<S> center_to_surface[3];
            ITERATE_3({ center_to_surface[i] = P[i] - l.sph.pos[i]; })

            const simd_fvec<S> dist_to_center = length(center_to_surface);
            ITERATE_3({ center_to_surface[i] /= dist_to_center; })

            // sample hemisphere
            const simd_fvec<S> r = sqrt(max(0.0f, 1.0f - r1 * r1));
            const simd_fvec<S> phi = 2.0f * PI * r2;

            const simd_fvec<S> sampled_dir[3] = {r * cos(phi), r * sin(phi), r1};

            simd_fvec<S> LT[3], LB[3];
            create_tbn(center_to_surface, LT, LB);

            simd_fvec<S> _sampled_dir[3];
            ITERATE_3({
                _sampled_dir[i] =
                    LT[i] * sampled_dir[0] + LB[i] * sampled_dir[1] + center_to_surface[i] * sampled_dir[2];
            })

            simd_fvec<S> light_surf_pos[3];
            ITERATE_3({ light_surf_pos[i] = l.sph.pos[i] + _sampled_dir[i] * l.sph.radius; })

            simd_fvec<S> L[3];
            ITERATE_3({ L[i] = light_surf_pos[i] - P[i]; })

            const simd_fvec<S> dist = length(L);

            where(ray_queue[index], ls.dist) = dist;
            ITERATE_3({ where(ray_queue[index], ls.L[i]) = L[i] / dist; })

            where(ray_queue[index], ls.area) = l.sph.area;

            simd_fvec<S> light_forward[3];
            ITERATE_3({ light_forward[i] = light_surf_pos[i] - l.sph.pos[i]; })
            normalize(light_forward);

            const simd_fvec<S> cos_theta = abs(dot(ls.L, light_forward));

            simd_fvec<S> pdf = (ls.dist * ls.dist) / (0.5f * ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;
        } else if (l.type == LIGHT_TYPE_DIR) {
            ITERATE_3({ where(ray_queue[index], ls.L[i]) = l.dir.dir[i]; })
            if (l.dir.angle != 0.0f) {
                const simd_fvec<S> r1 = fract(halton[RAND_DIM_LIGHT_U] + sample_off[0]);
                const simd_fvec<S> r2 = fract(halton[RAND_DIM_LIGHT_V] + sample_off[1]);

                const float radius = std::tan(l.dir.angle);

                simd_fvec<S> V[3];
                MapToCone(r1, r2, ls.L, radius, V);
                normalize(V);

                ITERATE_3({ where(ray_queue[index], ls.L[i]) = V[i]; })
            }

            where(ray_queue[index], ls.area) = 0.0f;
            where(ray_queue[index], ls.dist) = MAX_DIST;
            where(ray_queue[index], ls.pdf) = 1.0f;
        } else if (l.type == LIGHT_TYPE_RECT) {
            const simd_fvec<S> r1 = fract(halton[RAND_DIM_LIGHT_U] + sample_off[0]) - 0.5f;
            const simd_fvec<S> r2 = fract(halton[RAND_DIM_LIGHT_V] + sample_off[1]) - 0.5f;

            const simd_fvec<S> lp[3] = {l.rect.pos[0] + l.rect.u[0] * r1 + l.rect.v[0] * r2,
                                        l.rect.pos[1] + l.rect.u[1] * r1 + l.rect.v[1] * r2,
                                        l.rect.pos[2] + l.rect.u[2] * r1 + l.rect.v[2] * r2};

            simd_fvec<S> to_light[3];
            ITERATE_3({ to_light[i] = lp[i] - P[i]; })
            const simd_fvec<S> dist = length(to_light);

            where(ray_queue[index], ls.dist) = dist;
            ITERATE_3({ where(ray_queue[index], ls.L[i]) = to_light[i] / dist; })

            where(ray_queue[index], ls.area) = l.rect.area;

            float light_forward[3];
            cross(l.rect.u, l.rect.v, light_forward);
            normalize(light_forward);

            const simd_fvec<S> cos_theta =
                -ls.L[0] * light_forward[0] - ls.L[1] * light_forward[1] - ls.L[2] * light_forward[2];
            simd_fvec<S> pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sky_portal != 0) {
                simd_fvec<S> env_col[3] = {sc.env->env_col[0], sc.env->env_col[1], sc.env->env_col[2]};
                if (sc.env->env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Ref::TextureAtlasRGBA *>(tex_atlases[0]),
                                       sc.textures[sc.env->env_map], ls.L, ray_queue[index], tex_col);

                    if (sc.env->env_clamp > FLT_EPS) {
                        ITERATE_3({ env_col[i] = min(env_col[i] * tex_col[i], sc.env->env_clamp); })
                    }
                }
                ITERATE_3({ where(ray_queue[index], ls.col[i]) *= sc.env->env_col[i]; })
            }
        } else if (l.type == LIGHT_TYPE_DISK) {
            const simd_fvec<S> r1 = fract(halton[RAND_DIM_LIGHT_U] + sample_off[0]);
            const simd_fvec<S> r2 = fract(halton[RAND_DIM_LIGHT_V] + sample_off[1]);

            simd_fvec<S> offset[2] = {2.0f * r1 - 1.0f, 2.0f * r2 - 1.0f};
            const simd_ivec<S> mask = simd_cast(offset[0] != 0.0f & offset[1] != 0.0f);
            if (mask.not_all_zeros()) {
                simd_fvec<S> theta = 0.5f * PI - 0.25f * PI * (offset[0] / offset[1]), r = offset[1];

                where(abs(offset[0]) > abs(offset[1]), r) = offset[0];
                where(abs(offset[0]) > abs(offset[1]), theta) = 0.25f * PI * (offset[1] / offset[0]);

                where(mask, offset[0]) = 0.5f * r * cos(theta);
                where(mask, offset[1]) = 0.5f * r * sin(theta);
            }

            const simd_fvec<S> lp[3] = {l.disk.pos[0] + l.disk.u[0] * offset[0] + l.disk.v[0] * offset[1],
                                        l.disk.pos[1] + l.disk.u[1] * offset[0] + l.disk.v[1] * offset[1],
                                        l.disk.pos[2] + l.disk.u[2] * offset[0] + l.disk.v[2] * offset[1]};

            simd_fvec<S> to_light[3];
            ITERATE_3({ to_light[i] = lp[i] - P[i]; })
            const simd_fvec<S> dist = length(to_light);

            where(ray_queue[index], ls.dist) = dist;
            ITERATE_3({ where(ray_queue[index], ls.L[i]) = to_light[i] / dist; })

            where(ray_queue[index], ls.area) = l.disk.area;

            float light_forward[3];
            cross(l.disk.u, l.disk.v, light_forward);
            normalize(light_forward);

            const simd_fvec<S> cos_theta =
                -ls.L[0] * light_forward[0] - ls.L[1] * light_forward[1] - ls.L[2] * light_forward[2];
            simd_fvec<S> pdf = (ls.dist * ls.dist) / (ls.area * cos_theta);
            where(cos_theta <= 0.0f, pdf) = 0.0f;
            where(ray_queue[index], ls.pdf) = pdf;

            if (!l.visible) {
                where(ray_queue[index], ls.area) = 0.0f;
            }

            if (l.sky_portal != 0) {
                simd_fvec<S> env_col[3] = {sc.env->env_col[0], sc.env->env_col[1], sc.env->env_col[2]};
                if (sc.env->env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Ref::TextureAtlasRGBA *>(tex_atlases[0]),
                                       sc.textures[sc.env->env_map], ls.L, ray_queue[index], tex_col);

                    if (sc.env->env_clamp > FLT_EPS) {
                        ITERATE_3({ env_col[i] = min(env_col[i] * tex_col[i], sc.env->env_clamp); })
                    }
                }
                ITERATE_3({ where(ray_queue[index], ls.col[i]) *= sc.env->env_col[i]; })
            }
        } else if (l.type == LIGHT_TYPE_TRI) {
            const transform_t &ltr = sc.transforms[l.tri.xform_index];
            const uint32_t ltri_index = l.tri.tri_index;

            const vertex_t &v1 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 0]];
            const vertex_t &v2 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 1]];
            const vertex_t &v3 = sc.vertices[sc.vtx_indices[ltri_index * 3 + 2]];

            const simd_fvec<S> r1 = sqrt(fract(halton[RAND_DIM_LIGHT_U] + sample_off[0]));
            const simd_fvec<S> r2 = fract(halton[RAND_DIM_LIGHT_V] + sample_off[1]);

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
            ITERATE_3({ light_forward[i] /= light_fwd_len; })

            const simd_fvec<S> to_light[3] = {lp[0] - P[0], lp[1] - P[1], lp[2] - P[2]};
            where(ray_queue[index], ls.dist) = length(to_light);
            ITERATE_3({ where(ray_queue[index], ls.L[i]) = (to_light[i] / ls.dist); })

            const simd_fvec<S> cos_theta = abs(dot(ls.L, light_forward));
            where(simd_cast(cos_theta > 0.0f) & ray_queue[index], ls.pdf) = (ls.dist * ls.dist) / (ls.area * cos_theta);

            const material_t &lmat = sc.materials[sc.tri_materials[ltri_index].front_mi & MATERIAL_INDEX_BITS];
            if ((lmat.flags & MAT_FLAG_SKY_PORTAL) == 0) {
                if (lmat.textures[BASE_TEXTURE] != 0xffffffff) {
                    simd_fvec<S> tex_col[4];
                    SampleBilinear(tex_atlases, sc.textures[lmat.textures[BASE_TEXTURE]], luvs, simd_ivec<S>{0},
                                   ray_mask, tex_col);
                    ITERATE_3({ where(ray_queue[index], ls.col[i]) *= tex_col[i]; })
                }
            } else {
                simd_fvec<S> env_col[3] = {sc.env->env_col[0], sc.env->env_col[1], sc.env->env_col[2]};
                if (sc.env->env_map != 0xffffffff) {
                    simd_fvec<S> tex_col[3];
                    SampleLatlong_RGBE(*static_cast<const Ref::TextureAtlasRGBA *>(tex_atlases[0]),
                                       sc.textures[sc.env->env_map], ls.L, ray_queue[index], tex_col);
                    if (sc.env->env_clamp > FLT_EPS) {
                        ITERATE_3({ tex_col[i] = min(tex_col[i], sc.env->env_clamp); })
                    }
                    ITERATE_3({ env_col[i] *= tex_col[i]; })
                }
                ITERATE_3({ where(ray_queue[index], ls.col[i]) *= env_col[i]; })
            }
        }

        ++index;
    }
}

template <int S>
void Ray::NS::IntersectAreaLights(const ray_data_t<S> &r, const simd_ivec<S> &ray_mask, const light_t lights[],
                                  Span<const uint32_t> visible_lights, const transform_t transforms[],
                                  hit_data_t<S> &inout_inter) {
    for (uint32_t li = 0; li < uint32_t(visible_lights.size()); ++li) {
        const uint32_t light_index = visible_lights[li];
        const light_t &l = lights[light_index];
        if (l.type == LIGHT_TYPE_SPHERE) {
            const simd_fvec<S> op[3] = {l.sph.pos[0] - r.o[0], l.sph.pos[1] - r.o[1], l.sph.pos[2] - r.o[2]};
            const simd_fvec<S> b = dot(op, r.d);
            simd_fvec<S> det = b * b - dot(op, op) + l.sph.radius * l.sph.radius;

            simd_ivec<S> imask = simd_cast(det >= 0.0f) & ray_mask;
            if (imask.not_all_zeros()) {
                det = sqrt(det);
                const simd_fvec<S> t1 = b - det, t2 = b + det;

                const simd_fvec<S> mask1 = (t1 > HIT_EPS & t1 < inout_inter.t) & simd_cast(imask);
                const simd_fvec<S> mask2 = (t2 > HIT_EPS & t2 < inout_inter.t) & simd_cast(imask) & ~mask1;

                inout_inter.mask |= simd_cast(mask1 | mask2);

                where(mask1 | mask2, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                where(mask1, inout_inter.t) = t1;
                where(mask2, inout_inter.t) = t2;
            }
        } else if (l.type == LIGHT_TYPE_RECT) {
            float light_fwd[3];
            cross(l.rect.u, l.rect.v, light_fwd);
            normalize(light_fwd);

            const float plane_dist = dot(light_fwd, l.rect.pos);

            const simd_fvec<S> cos_theta = dot(r.d, light_fwd);
            const simd_fvec<S> t = (plane_dist - dot(light_fwd, r.o)) / cos_theta;

            const simd_ivec<S> imask = simd_cast(cos_theta<0.0f & t> HIT_EPS & t < inout_inter.t) & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot(l.rect.u, l.rect.u);
                const float dot_v = dot(l.rect.v, l.rect.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.rect.pos[0], p[1] - l.rect.pos[1], p[2] - l.rect.pos[2]};

                const simd_fvec<S> a1 = dot(l.rect.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot(l.rect.v, vi) / dot_v;

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

            const float plane_dist = dot(light_fwd, l.disk.pos);

            const simd_fvec<S> cos_theta = dot(r.d, light_fwd);
            const simd_fvec<S> t = (plane_dist - dot(light_fwd, r.o)) / cos_theta;

            const simd_ivec<S> imask = simd_cast(cos_theta<0.0f & t> HIT_EPS & t < inout_inter.t) & ray_mask;
            if (imask.not_all_zeros()) {
                const float dot_u = dot(l.disk.u, l.disk.u);
                const float dot_v = dot(l.disk.v, l.disk.v);

                const simd_fvec<S> p[3] = {fmadd(r.d[0], t, r.o[0]), fmadd(r.d[1], t, r.o[1]),
                                           fmadd(r.d[2], t, r.o[2])};
                const simd_fvec<S> vi[3] = {p[0] - l.disk.pos[0], p[1] - l.disk.pos[1], p[2] - l.disk.pos[2]};

                const simd_fvec<S> a1 = dot(l.disk.u, vi) / dot_u;
                const simd_fvec<S> a2 = dot(l.disk.v, vi) / dot_v;

                const simd_fvec<S> final_mask = (sqrt(a1 * a1 + a2 * a2) <= 0.5f) & simd_cast(imask);

                inout_inter.mask |= simd_cast(final_mask);
                where(final_mask, inout_inter.obj_index) = -simd_ivec<S>(light_index) - 1;
                where(final_mask, inout_inter.t) = t;
            }
        }
    }
}

template <int S>
void Ray::NS::ShadeSurface(const simd_ivec<S> &px_index, const pass_info_t &pi, const float *halton,
                           const hit_data_t<S> &inter, const ray_data_t<S> &ray, const scene_data_t &sc,
                           uint32_t node_index, const Ref::TextureAtlasBase *tex_atlases[], simd_fvec<S> out_rgba[4],
                           simd_ivec<S> out_secondary_masks[], ray_data_t<S> out_secondary_rays[],
                           int *out_secondary_rays_count, simd_ivec<S> out_shadow_masks[],
                           shadow_ray_t<S> out_shadow_rays[], int *out_shadow_rays_count) {
    out_rgba[0] = {0.0f};
    out_rgba[1] = {0.0f};
    out_rgba[2] = {0.0f};
    out_rgba[3] = {1.0f};

    const simd_ivec<S> ino_hit = ~inter.mask;
    if (ino_hit.not_all_zeros()) {
        simd_fvec<S> env_col[4] = {{1.0f}, {1.0f}, {1.0f}, {1.0f}};
        if (pi.should_add_environment()) {
            if (sc.env->env_map != 0xffffffff) {
                SampleLatlong_RGBE(*static_cast<const Ref::TextureAtlasRGBA *>(tex_atlases[0]),
                                   sc.textures[sc.env->env_map], ray.d, ino_hit, env_col);

                if (sc.env->env_clamp > FLT_EPS) {
                    ITERATE_3({ env_col[i] = min(env_col[i], simd_fvec<S>{sc.env->env_clamp}); })
                }
            }
            env_col[3] = 1.0f;
        }

        const simd_fvec<S> &fno_hit = simd_cast(ino_hit);

        where(fno_hit, out_rgba[0]) = ray.c[0] * env_col[0] * sc.env->env_col[0];
        where(fno_hit, out_rgba[1]) = ray.c[1] * env_col[1] * sc.env->env_col[1];
        where(fno_hit, out_rgba[2]) = ray.c[2] * env_col[2] * sc.env->env_col[2];
        where(fno_hit, out_rgba[3]) = env_col[3];
    }

    simd_ivec<S> is_active_lane = inter.mask;
    if (is_active_lane.all_zeros()) {
        return;
    }

    const simd_fvec<S> *I = ray.d;
    const simd_fvec<S> P[3] = {fmadd(inter.t, ray.d[0], ray.o[0]), fmadd(inter.t, ray.d[1], ray.o[1]),
                               fmadd(inter.t, ray.d[2], ray.o[2])};

    const simd_ivec<S> is_light_hit = is_active_lane & (inter.obj_index < 0); // Area light intersection
    if (is_light_hit.not_all_zeros()) {
        simd_ivec<S> ray_queue[S];
        ray_queue[0] = is_light_hit;

        int index = 0, num = 1;
        while (index != num) {
            const long mask = ray_queue[index].movemask();
            const uint32_t first_li = inter.obj_index[GetFirstBit(mask)];

            const simd_ivec<S> same_li = (inter.obj_index == first_li);
            const simd_ivec<S> diff_li = and_not(same_li, ray_queue[index]);

            if (diff_li.not_all_zeros()) {
                ray_queue[index] &= same_li;
                ray_queue[num++] = diff_li;
            }

            const light_t &l = sc.lights[-int(first_li) - 1];

            simd_fvec<S> lcol[3] = {l.col[0], l.col[1], l.col[2]};

            if (l.type == LIGHT_TYPE_SPHERE) {
                const simd_fvec<S> op[3] = {l.sph.pos[0] - ray.o[0], l.sph.pos[1] - ray.o[1], l.sph.pos[2] - ray.o[2]};
                const simd_fvec<S> b = dot(op, ray.d);
                const simd_fvec<S> det = sqrt(b * b - dot(op, op) + l.sph.radius * l.sph.radius);

                simd_fvec<S> dd[3] = {l.sph.pos[0] - P[0], l.sph.pos[1] - P[1], l.sph.pos[2] - P[2]};
                normalize(dd);

                const simd_fvec<S> cos_theta = dot(ray.d, dd);

                const simd_fvec<S> light_pdf = (inter.t * inter.t) / (0.5f * l.sph.area * cos_theta);
                const simd_fvec<S> bsdf_pdf = ray.pdf;

                const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                ITERATE_3({ lcol[i] *= mis_weight; });
            } else if (l.type == LIGHT_TYPE_RECT) {
                float light_fwd[3];
                cross(l.rect.u, l.rect.v, light_fwd);
                normalize(light_fwd);

                const float plane_dist = dot(light_fwd, l.rect.pos);
                const simd_fvec<S> cos_theta = dot(ray.d, light_fwd);

                const simd_fvec<S> light_pdf = (inter.t * inter.t) / (l.rect.area * cos_theta);
                const simd_fvec<S> bsdf_pdf = ray.pdf;

                const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                ITERATE_3({ lcol[i] *= mis_weight; });
            } else if (l.type == LIGHT_TYPE_DISK) {
                float light_fwd[3];
                cross(l.disk.u, l.disk.v, light_fwd);
                normalize(light_fwd);

                const float plane_dist = dot(light_fwd, l.disk.pos);
                const simd_fvec<S> cos_theta = dot(ray.d, light_fwd);

                const simd_fvec<S> light_pdf = (inter.t * inter.t) / (l.disk.area * cos_theta);
                const simd_fvec<S> bsdf_pdf = ray.pdf;

                const simd_fvec<S> mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                ITERATE_3({ lcol[i] *= mis_weight; });
            }

            where(ray_queue[index], out_rgba[0]) = ray.c[0] * lcol[0];
            where(ray_queue[index], out_rgba[1]) = ray.c[1] * lcol[1];
            where(ray_queue[index], out_rgba[2]) = ray.c[2] * lcol[2];
            where(ray_queue[index], out_rgba[3]) = 1.0f;

            ++index;
        }
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
    const int tr_indices_stride = sizeof(mesh_instance_t) / sizeof(int);

    const simd_ivec<S> tr_index = gather(tr_indices, obj_index * tr_indices_stride);

    const simd_ivec<S> vtx_indices[3] = {gather(reinterpret_cast<const int *>(sc.vtx_indices), tri_index * 3),
                                         gather(reinterpret_cast<const int *>(sc.vtx_indices), tri_index * 3 + 1),
                                         gather(reinterpret_cast<const int *>(sc.vtx_indices), tri_index * 3 + 2)};

    const simd_fvec<S> w = 1.0f - inter.u - inter.v;

    simd_fvec<S> p1[3], p2[3], p3[3], P_ls[3];
    { // Fetch vertex positions
        const float *vtx_positions = &sc.vertices[0].p[0];
        const int vtx_positions_stride = sizeof(vertex_t) / sizeof(float);

        ITERATE_3({ p1[i] = gather(vtx_positions, vtx_indices[0] * vtx_positions_stride + i); });
        ITERATE_3({ p2[i] = gather(vtx_positions, vtx_indices[1] * vtx_positions_stride + i); });
        ITERATE_3({ p3[i] = gather(vtx_positions, vtx_indices[2] * vtx_positions_stride + i); });

        ITERATE_3({ P_ls[i] = p1[i] * w + p2[i] * inter.u + p3[i] * inter.v; });
    }

    simd_fvec<S> n1[3], n2[3], n3[3], N[3];
    { // Fetch vertex normals
        const float *vtx_normals = &sc.vertices[0].n[0];
        const int vtx_normals_stride = sizeof(vertex_t) / sizeof(float);

        ITERATE_3({ n1[i] = gather(vtx_normals, vtx_indices[0] * vtx_normals_stride + i); });
        ITERATE_3({ n2[i] = gather(vtx_normals, vtx_indices[1] * vtx_normals_stride + i); });
        ITERATE_3({ n3[i] = gather(vtx_normals, vtx_indices[2] * vtx_normals_stride + i); });

        ITERATE_3({ N[i] = n1[i] * w + n2[i] * inter.u + n3[i] * inter.v; });
        normalize(N);
    }

    simd_fvec<S> u1[2], u2[2], u3[2], uvs[2];
    { // Fetch vertex uvs
        const float *vtx_uvs = &sc.vertices[0].t[0][0];
        const int vtx_normals_stride = sizeof(vertex_t) / sizeof(float);

        ITERATE_2({ u1[i] = gather(vtx_uvs, vtx_indices[0] * vtx_normals_stride + i); })
        ITERATE_2({ u2[i] = gather(vtx_uvs, vtx_indices[1] * vtx_normals_stride + i); })
        ITERATE_2({ u3[i] = gather(vtx_uvs, vtx_indices[2] * vtx_normals_stride + i); })

        ITERATE_2({ uvs[i] = u1[i] * w + u2[i] * inter.u + u3[i] * inter.v; })
    }

    simd_fvec<S> plane_N[3];
    { 
        simd_fvec<S> e21[3], e31[3];
        ITERATE_3({
            e21[i] = p2[i] - p1[i];
            e31[i] = p3[i] - p1[i];
        })
        cross(e21, e31, plane_N);
    }
    const simd_fvec<S> pa = length(plane_N);
    ITERATE_3({ plane_N[i] /= pa; })

    simd_fvec<S> B[3];
    { // Fetch vertex binormal
        const float *vtx_binormals = &sc.vertices[0].b[0];
        const int vtx_binormals_stride = sizeof(vertex_t) / sizeof(float);

        const simd_fvec<S> B1[3] = {gather(vtx_binormals, vtx_indices[0] * vtx_binormals_stride),
                                    gather(vtx_binormals, vtx_indices[0] * vtx_binormals_stride + 1),
                                    gather(vtx_binormals, vtx_indices[0] * vtx_binormals_stride + 2)};
        const simd_fvec<S> B2[3] = {gather(vtx_binormals, vtx_indices[1] * vtx_binormals_stride),
                                    gather(vtx_binormals, vtx_indices[1] * vtx_binormals_stride + 1),
                                    gather(vtx_binormals, vtx_indices[1] * vtx_binormals_stride + 2)};
        const simd_fvec<S> B3[3] = {gather(vtx_binormals, vtx_indices[2] * vtx_binormals_stride),
                                    gather(vtx_binormals, vtx_indices[2] * vtx_binormals_stride + 1),
                                    gather(vtx_binormals, vtx_indices[2] * vtx_binormals_stride + 2)};

        ITERATE_3({ B[i] = B1[i] * w + B2[i] * inter.u + B3[i] * inter.v; });
    }

    simd_fvec<S> T[3];
    cross(B, N, T);

    { // return black for non-existing backfacing material
        simd_ivec<S> no_back_mi = (mat_index >> 16) == 0xffff;
        no_back_mi &= is_backfacing & is_active_lane;
        where(no_back_mi, out_rgba[0]) = 0.0f;
        where(no_back_mi, out_rgba[1]) = 0.0f;
        where(no_back_mi, out_rgba[2]) = 0.0f;
        where(no_back_mi, out_rgba[3]) = 0.0f;
        is_active_lane &= ~no_back_mi;
    }

    if (is_active_lane.all_zeros()) {
        return;
    }

    where(~is_backfacing, mat_index) = mat_index & 0xffff; // use front material index
    where(is_backfacing, mat_index) = mat_index >> 16;     // use back material index

    ITERATE_3({ where(is_backfacing, plane_N[i]) = -plane_N[i]; });
    ITERATE_3({ where(is_backfacing, N[i]) = -N[i]; });
    ITERATE_3({ where(is_backfacing, B[i]) = -B[i]; });
    ITERATE_3({ where(is_backfacing, T[i]) = -T[i]; });

    simd_fvec<S> tangent[3] = {-P_ls[2], {0.0f}, P_ls[0]};

    simd_fvec<S> transform[16], inv_transform[16];

    { // fetch transformation matrices
        const float *transforms = &sc.transforms[0].xform[0];
        const float *inv_transforms = &sc.transforms[0].inv_xform[0];
        const int transforms_stride = sizeof(transform_t) / sizeof(float);

        ITERATE_16({ transform[i] = gather(transforms, tr_index * transforms_stride + i); })
        ITERATE_16({ inv_transform[i] = gather(inv_transforms, tr_index * transforms_stride + i); })
    }

    TransformNormal(inv_transform, plane_N);
    TransformNormal(inv_transform, N);
    TransformNormal(inv_transform, B);
    TransformNormal(inv_transform, T);

    TransformNormal(inv_transform, tangent);

    //////////////////////////////////

#ifdef USE_RAY_DIFFERENTIALS
    derivatives_t<S> surf_der;
    ComputeDerivatives(I, inter.t, ray.do_dx, ray.do_dy, ray.dd_dx, ray.dd_dy, p1, p2, p3, n1, n2, n3, u1, u2, u3,
                       plane_N, transform, surf_der);
#else
    const simd_fvec<S> ta = abs((u2[0] - u1[0]) * (u3[1] - u1[1]) - (u3[0] - u1[0]) * (u2[1] - u1[1]));

    const simd_fvec<S> cone_width = ray.cone_width + ray.cone_spread * inter.t;

    simd_fvec<S> lambda = 0.5f * fast_log2(ta / pa);
    lambda += fast_log2(cone_width);
    // lambda += 0.5 * fast_log2(tex_res.x * tex_res.y);
    // lambda -= fast_log2(abs(dot(I, plane_N)));
#endif

    //////////////////////////////////

    static const int MatDWORDStride = sizeof(material_t) / sizeof(float);

    // used to randomize halton sequence among pixels
    const simd_fvec<S> sample_off[2] = {construct_float(hash(px_index)), construct_float(hash(hash(px_index)))};

    const simd_ivec<S> diff_depth = ray.ray_depth & 0x000000ff;
    const simd_ivec<S> spec_depth = (ray.ray_depth >> 8) & 0x000000ff;
    const simd_ivec<S> refr_depth = (ray.ray_depth >> 16) & 0x000000ff;
    const simd_ivec<S> transp_depth = (ray.ray_depth >> 24) & 0x000000ff;
    const simd_ivec<S> total_depth = diff_depth + spec_depth + refr_depth + transp_depth;

    const simd_ivec<S> mat_type =
        gather(reinterpret_cast<const int *>(&sc.materials[0].type), mat_index * sizeof(material_t) / sizeof(int)) &
        0xff;

    simd_fvec<S> mix_rand = fract(halton[RAND_DIM_BSDF_PICK] + sample_off[0]);
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

                const texture_t &t = sc.textures[first_t];
#ifdef USE_RAY_DIFFERENTIALS
                const simd_fvec<S> base_lod = get_texture_lod(t, surf_der.duv_dx, surf_der.duv_dy, ray_queue[index]);
#else
                const simd_fvec<S> base_lod = get_texture_lod(t, lambda, ray_queue[index]);
#endif

                simd_fvec<S> tex_color[4];
                SampleBilinear(tex_atlases, t, uvs, simd_ivec<S>(base_lod), ray_queue[index], tex_color);

                where(ray_queue[index], mix_val) *= tex_color[0];

                ++index;
            }
        }

        const float *int_iors = &sc.materials[0].int_ior;
        const float *ext_iors = &sc.materials[0].ext_ior;

        const simd_fvec<S> int_ior = gather(int_iors, mat_index * MatDWORDStride);
        const simd_fvec<S> ext_ior = gather(ext_iors, mat_index * MatDWORDStride);

        simd_fvec<S> eta = ext_ior / int_ior;
        where(is_backfacing, eta) = int_ior / ext_ior;

        simd_fvec<S> RR = fresnel_dielectric_cos(dot(I, N), eta);
        where(int_ior == 0.0f, RR) = 1.0f;

        mix_val *= clamp(RR, 0.0f, 1.0f);

        const simd_ivec<S> use_mat1 = simd_cast(mix_rand > mix_val) & is_mix_mat;
        const simd_ivec<S> use_mat2 = ~use_mat1 & is_mix_mat;

        const int *all_mat_flags = reinterpret_cast<const int *>(&sc.materials[0].flags);
        const simd_ivec<S> is_add = (gather(all_mat_flags, mat_index * MatDWORDStride) & MAT_FLAG_MIX_ADD) != 0;

        const int *all_mat_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[0]);
        const simd_ivec<S> mat1_index = gather(&all_mat_textures[MIX_MAT1], mat_index * MatDWORDStride);
        const simd_ivec<S> mat2_index = gather(&all_mat_textures[MIX_MAT2], mat_index * MatDWORDStride);

        where(is_add & use_mat1, mix_weight) /= (1.0f - mix_val);
        where(use_mat1, mat_index) = mat1_index;
        where(use_mat1, mix_rand) = (mix_rand - mix_val) / (1.0f - mix_val);

        where(is_add & use_mat2, mix_weight) /= mix_val;
        where(use_mat2, mat_index) = mat2_index;
        where(use_mat2, mix_rand) /= mix_val;
    }

    { // apply normal map
        const int *norm_textures = reinterpret_cast<const int *>(&sc.materials[0].textures[NORMALS_TEXTURE]);
        const simd_ivec<S> normals_texture = gather(norm_textures, mat_index * MatDWORDStride);

        const simd_ivec<S> has_texture = (normals_texture != -1) & is_active_lane;
        if (has_texture.not_all_zeros()) {
            simd_ivec<S> ray_queue[S];
            ray_queue[0] = has_texture;

            simd_fvec<S> normals_tex[4] = {{0.0f}, {1.0f}, {0.0f}, {0.0f}};

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

                const texture_t &t = sc.textures[first_t];
                SampleBilinear(tex_atlases, t, uvs, simd_ivec<S>{0}, ray_queue[index], normals_tex);

                ++index;
            }

            ITERATE_3({ normals_tex[i] = normals_tex[i] * 2.0f - 1.0f; })

            simd_fvec<S> new_normal[3];
            ITERATE_3({ new_normal[i] = normals_tex[0] * T[i] + normals_tex[2] * N[i] + normals_tex[1] * B[i]; })
            normalize(new_normal);

            const int *normalmap_strengths = reinterpret_cast<const int *>(&sc.materials[0].normal_map_strength_unorm);
            const simd_ivec<S> normalmap_strength = gather(normalmap_strengths, mat_index * MatDWORDStride) & 0xffff;

            const simd_fvec<S> fstrength = conv_unorm_16(normalmap_strength);
            ITERATE_3({ new_normal[i] = N[i] + (new_normal[i] - N[i]) * fstrength; })
            normalize(new_normal);

            const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};
            ensure_valid_reflection(plane_N, _I, new_normal);

            where(has_texture, N[0]) = new_normal[0];
            where(has_texture, N[1]) = new_normal[1];
            where(has_texture, N[2]) = new_normal[2];
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
        rotate_around_axis(tangent, N, tangent_rotation, tangent);
    }

    cross(tangent, N, B);
    normalize(B);
    cross(N, B, T);
#endif

#if USE_NEE
    light_sample_t<S> ls;
    if (pi.should_add_direct_light() && !sc.li_indices.empty()) {
        SampleLightSource(P, sc, tex_atlases, halton, sample_off, is_active_lane, ls);
    }
    const simd_fvec<S> N_dot_L = dot(N, ls.L);
#endif

    simd_fvec<S> base_color[3];
    { // Fetch material base color
        const float *base_colors = &sc.materials[0].base_color[0];
        ITERATE_3({ base_color[i] = gather(base_colors + i, mat_index * MatDWORDStride); })

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

                const texture_t &t = sc.textures[first_t];
#ifdef USE_RAY_DIFFERENTIALS
                const simd_fvec<S> base_lod = get_texture_lod(t, surf_der.duv_dx, surf_der.duv_dy, ray_queue[index]);
#else
                const simd_fvec<S> base_lod = get_texture_lod(t, lambda, ray_queue[index]);
#endif

                simd_fvec<S> tex_color[4];
                SampleBilinear(tex_atlases, t, uvs, simd_ivec<S>(base_lod), ray_queue[index], tex_color);
                if (t.width & TEXTURE_SRGB_BIT) {
                    srgb_to_rgb(tex_color, tex_color);
                }

                ITERATE_3({ where(ray_queue[index], base_color[i]) *= tex_color[i]; })

                ++index;
            }
        }
    }

    simd_fvec<S> tint_color[3] = {{0.0f}, {0.0f}, {0.0f}};

    const simd_fvec<S> base_color_lum = lum(base_color);
    ITERATE_3({ where(base_color_lum > 0.0f, tint_color[i]) = base_color[i] / base_color_lum; })

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

                const texture_t &t = sc.textures[first_t];
#ifdef USE_RAY_DIFFERENTIALS
                const simd_fvec<S> roughness_lod =
                    get_texture_lod(t, surf_der.duv_dx, surf_der.duv_dy, ray_queue[index]);
#else
                const simd_fvec<S> roughness_lod = get_texture_lod(t, lambda, ray_queue[index]);
#endif

                simd_fvec<S> roughness_color[4];
                SampleBilinear(tex_atlases, t, uvs, simd_ivec<S>(roughness_lod), ray_queue[index], roughness_color);
                if (t.width & TEXTURE_SRGB_BIT) {
                    srgb_to_rgb(roughness_color, roughness_color);
                }
                where(ray_queue[index], roughness) *= roughness_color[0];

                ++index;
            }
        }
    }

    simd_fvec<S> col[3] = {0.0f, 0.0f, 0.0f};

    const simd_fvec<S> rand_u = fract(halton[RAND_DIM_BSDF_U] + sample_off[0]);
    const simd_fvec<S> rand_v = fract(halton[RAND_DIM_BSDF_V] + sample_off[1]);

    simd_ivec<S> secondary_mask = {0}, shadow_mask = {0};

    ray_data_t<S> new_ray;
#ifndef USE_RAY_DIFFERENTIALS
    new_ray.cone_width = cone_width;
    new_ray.cone_spread = ray.cone_spread;
#endif
    new_ray.xy = ray.xy;
    new_ray.pdf = 0.0f;

    shadow_ray_t<S> sh_r;
    sh_r.xy = ray.xy;

    { // Sample materials
        simd_ivec<S> ray_queue[S];
        ray_queue[0] = is_active_lane;

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

            const material_t *mat = &sc.materials[first_mi];
            if (mat->type == DiffuseNode) {
                simd_fvec<S> P_biased[3];
                offset_ray(P, plane_N, P_biased);
#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};

                    simd_fvec<S> diff_col[4];
                    Evaluate_OrenDiffuse_BSDF(_I, N, ls.L, roughness, base_color, diff_col);
                    const simd_fvec<S> &bsdf_pdf = diff_col[3];

                    simd_fvec<S> mis_weight = 1.0f;
                    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

                    ITERATE_3({ where(eval_light, sh_r.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(eval_light, sh_r.d[i]) = ls.L[i]; })
                    where(eval_light, sh_r.dist) = ls.dist - 10.0f * HIT_BIAS;
                    ITERATE_3({
                        where(eval_light, sh_r.c[i]) =
                            ray.c[i] * ls.col[i] * diff_col[i] * (mix_weight * mis_weight / ls.pdf);
                    })

                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= eval_light;
                }
#endif
                const simd_ivec<S> gen_ray = (diff_depth < pi.settings.max_diff_depth) &
                                             (total_depth < pi.settings.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    simd_fvec<S> V[3], F[4];
                    Sample_OrenDiffuse_BSDF(T, B, N, I, roughness, base_color, rand_u, rand_v, V, F);

                    where(gen_ray, new_ray.ray_depth) = ray.ray_depth + 0x00000001;

                    ITERATE_3({ where(gen_ray, new_ray.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(gen_ray, new_ray.pdf) = F[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(gen_ray, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dx[i]) = surf_der.dd_dx[i] - 2 * (dot(I, plane_N) * surf_der.dndx[i] +
                                                                                    surf_der.ddn_dx[i] * plane_N[i]);
                    })
                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dy[i]) = surf_der.dd_dy[i] - 2 * (dot(I, plane_N) * surf_der.dndy[i] +
                                                                                    surf_der.ddn_dy[i] * plane_N[i]);
                    })
#endif

                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == GlossyNode) {
                const float specular = 0.5f;
                const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
                const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);
                const simd_fvec<S> roughness2 = roughness * roughness;

                simd_fvec<S> P_biased[3];
                offset_ray(P, plane_N, P_biased);

#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(roughness2 * roughness2 >= 1e-7f) &
                                                simd_cast(N_dot_L > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};
                    simd_fvec<S> H[3] = {ls.L[0] - I[0], ls.L[1] - I[1], ls.L[2] - I[2]};
                    normalize(H);

                    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
                    tangent_from_world(T, B, N, _I, view_dir_ts);
                    tangent_from_world(T, B, N, ls.L, light_dir_ts);
                    tangent_from_world(T, B, N, H, sampled_normal_ts);

                    simd_fvec<S> spec_col[4];
                    Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2,
                                              simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0}, base_color, spec_col);
                    const simd_fvec<S> &bsdf_pdf = spec_col[3];

                    simd_fvec<S> mis_weight = 1.0f;
                    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

                    ITERATE_3({ where(eval_light, sh_r.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(eval_light, sh_r.d[i]) = ls.L[i]; })
                    where(eval_light, sh_r.dist) = ls.dist - 10.0f * HIT_BIAS;
                    ITERATE_3({
                        where(eval_light, sh_r.c[i]) =
                            ray.c[i] * ls.col[i] * spec_col[i] * (mix_weight * mis_weight / ls.pdf);
                    })

                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= eval_light;
                }
#endif

                const simd_ivec<S> gen_ray = (spec_depth < pi.settings.max_spec_depth) &
                                             (total_depth < pi.settings.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    simd_fvec<S> V[3], F[4];
                    Sample_GGXSpecular_BSDF(T, B, N, I, roughness, simd_fvec<S>{0.0f}, simd_fvec<S>{spec_ior},
                                            simd_fvec<S>{spec_F0}, base_color, rand_u, rand_v, V, F);

                    where(gen_ray, new_ray.ray_depth) = ray.ray_depth + 0x00000100;

                    ITERATE_3({ where(gen_ray, new_ray.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(gen_ray, new_ray.pdf) = F[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(gen_ray, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dx[i]) = surf_der.dd_dx[i] - 2 * (dot(I, plane_N) * surf_der.dndx[i] +
                                                                                    surf_der.ddn_dx[i] * plane_N[i]);
                    })
                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dy[i]) = surf_der.dd_dy[i] - 2 * (dot(I, plane_N) * surf_der.dndy[i] +
                                                                                    surf_der.ddn_dy[i] * plane_N[i]);
                    })
#endif

                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == RefractiveNode) {
                simd_fvec<S> eta = (mat->ext_ior / mat->int_ior);
                where(is_backfacing, eta) = (mat->int_ior / mat->ext_ior);
                const simd_fvec<S> roughness2 = roughness * roughness;

                simd_fvec<S> P_biased[3];
                const simd_fvec<S> _plane_N[3] = {-plane_N[0], -plane_N[1], -plane_N[2]};
                offset_ray(P, _plane_N, P_biased);

#if USE_NEE
                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & simd_cast(roughness2 * roughness2 >= 1e-7f) &
                                                simd_cast(N_dot_L < 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};
                    simd_fvec<S> H[3] = {ls.L[0] - I[0] * eta, ls.L[1] - I[1] * eta, ls.L[2] - I[2] * eta};
                    normalize(H);

                    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
                    tangent_from_world(T, B, N, _I, view_dir_ts);
                    tangent_from_world(T, B, N, ls.L, light_dir_ts);
                    tangent_from_world(T, B, N, H, sampled_normal_ts);

                    simd_fvec<S> refr_col[4];
                    Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2,
                                                simd_fvec<S>{eta}, base_color, refr_col);
                    const simd_fvec<S> &bsdf_pdf = refr_col[3];

                    simd_fvec<S> mis_weight = 1.0f;
                    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);

                    ITERATE_3({ where(eval_light, sh_r.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(eval_light, sh_r.d[i]) = ls.L[i]; })
                    where(eval_light, sh_r.dist) = ls.dist - 10.0f * HIT_BIAS;
                    ITERATE_3({
                        where(eval_light, sh_r.c[i]) =
                            ray.c[i] * ls.col[i] * refr_col[i] * (mix_weight * mis_weight / ls.pdf);
                    })

                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= eval_light;
                }
#endif

                const simd_ivec<S> gen_ray = (refr_depth < pi.settings.max_refr_depth) &
                                             (total_depth < pi.settings.max_total_depth) & ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    simd_fvec<S> V[4], F[4];
                    Sample_GGXRefraction_BSDF(T, B, N, I, roughness, eta, base_color, rand_u, rand_v, V, F);
                    const simd_fvec<S> m = V[3];

                    where(gen_ray, new_ray.ray_depth) = ray.ray_depth + 0x00010000;

                    ITERATE_3({ where(gen_ray, new_ray.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(gen_ray, new_ray.pdf) = F[3];

                    const simd_fvec<S> k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
                    
#ifdef USE_RAY_DIFFERENTIALS
                    const simd_fvec<S> dmdx = k * surf_der.ddn_dx;
                    const simd_fvec<S> dmdy = k * surf_der.ddn_dy;

                    ITERATE_3({ where(gen_ray, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dx[i]) =
                            eta * surf_der.dd_dx[i] - (m * surf_der.dndx[i] + dmdx * plane_N[i]);
                    })
                    ITERATE_3({
                        where(gen_ray, new_ray.dd_dy[i]) =
                            eta * surf_der.dd_dy[i] - (m * surf_der.dndy[i] + dmdx * plane_N[i]);
                    })
#endif

                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == EmissiveNode) {
                simd_fvec<S> mis_weight = 1.0f;
#if USE_NEE
                if (mat->flags & MAT_FLAG_SKY_PORTAL) {
                    simd_fvec<S> env_col[3] = {sc.env->env_col[0], sc.env->env_col[1], sc.env->env_col[2]};
                    if (sc.env->env_map != 0xffffffff) {
                        simd_fvec<S> tex_col[3];
                        SampleLatlong_RGBE(*static_cast<const Ref::TextureAtlasRGBA *>(tex_atlases[0]),
                                           sc.textures[sc.env->env_map], ls.L, ray_queue[index], tex_col);
                        if (sc.env->env_clamp > FLT_EPS) {
                            ITERATE_3({ tex_col[i] = min(tex_col[i], sc.env->env_clamp); })
                        }
                        ITERATE_3({ env_col[i] *= tex_col[i]; })
                    }
                    ITERATE_3({ where(ray_queue[index], base_color[i]) *= env_col[i]; })
                }

                if (pi.bounce > 0 && (mat->flags & (MAT_FLAG_MULT_IMPORTANCE | MAT_FLAG_SKY_PORTAL))) {
                    const simd_fvec<S> v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]},
                                       v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};

                    simd_fvec<S> light_forward[3];
                    cross(v1, v2, light_forward);
                    TransformDirection(transform, light_forward);

                    const simd_fvec<S> light_forward_len = length(light_forward);
                    ITERATE_3({ light_forward[i] /= light_forward_len; })
                    const simd_fvec<S> tri_area = 0.5f * light_forward_len;

                    const simd_fvec<S> cos_theta = abs(dot(I, light_forward));
                    const simd_fvec<S> light_pdf = (inter.t * inter.t) / (tri_area * cos_theta);
                    const simd_fvec<S> &bsdf_pdf = ray.pdf;

                    where(cos_theta > 0.0f, mis_weight) = power_heuristic(bsdf_pdf, light_pdf);
                }
#endif
                ITERATE_3(
                    { where(ray_queue[index], col[i]) += mix_weight * mis_weight * mat->strength * base_color[i]; })
            } else if (mat->type == TransparentNode) {
                const simd_ivec<S> gen_ray =
                    (transp_depth < pi.settings.max_transp_depth & total_depth < pi.settings.max_total_depth) &
                    ray_queue[index];
                if (gen_ray.not_all_zeros()) {
                    const simd_fvec<S> _plane_N[3] = {-plane_N[0], -plane_N[1], -plane_N[2]};

                    simd_fvec<S> new_p[3];
                    offset_ray(P, _plane_N, new_p);

                    where(gen_ray, new_ray.ray_depth) = ray.ray_depth + 0x01000000;

                    ITERATE_3({ where(gen_ray, new_ray.o[i]) = new_p[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.d[i]) = ray.d[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.c[i]) = ray.c[i]; })
                    where(gen_ray, new_ray.pdf) = ray.pdf;

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(gen_ray, new_ray.do_dx[i]) = ray.do_dx[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.do_dy[i]) = ray.do_dy[i]; })

                    ITERATE_3({ where(gen_ray, new_ray.dd_dx[i]) = ray.dd_dx[i]; })
                    ITERATE_3({ where(gen_ray, new_ray.dd_dy[i]) = ray.dd_dy[i]; })
#endif

                    where(gen_ray, new_ray.ray_depth) = ray.ray_depth + 0x01000000;

                    assert((secondary_mask & gen_ray).all_zeros());
                    secondary_mask |= gen_ray;
                }
            } else if (mat->type == PrincipledNode) {
                simd_fvec<S> metallic = unpack_unorm_16(mat->metallic_unorm);
                if (mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
                    const texture_t &t = sc.textures[mat->textures[METALLIC_TEXTURE]];
#ifdef USE_RAY_DIFFERENTIALS
                    const simd_fvec<S> metallic_lod =
                        get_texture_lod(t, surf_der.duv_dx, surf_der.duv_dy, ray_queue[index]);
#else
                    const simd_fvec<S> metallic_lod = get_texture_lod(t, lambda, ray_queue[index]);
#endif

                    simd_fvec<S> metallic_color[4];
                    SampleBilinear(tex_atlases, t, uvs, simd_ivec<S>(metallic_lod), ray_queue[index], metallic_color);

                    metallic *= metallic_color[0];
                }

                const float specular = unpack_unorm_16(mat->specular_unorm);
                const float transmission = unpack_unorm_16(mat->transmission_unorm);
                const float clearcoat = unpack_unorm_16(mat->clearcoat_unorm);
                const float clearcoat_roughness = unpack_unorm_16(mat->clearcoat_roughness_unorm);
                const float sheen = unpack_unorm_16(mat->sheen_unorm);
                const float sheen_tint = unpack_unorm_16(mat->sheen_tint_unorm);

                simd_fvec<S> spec_tmp_col[3];
                ITERATE_3({
                    spec_tmp_col[i] = mix(simd_fvec<S>{1.0f}, tint_color[i], unpack_unorm_16(mat->specular_tint_unorm));
                    spec_tmp_col[i] = mix(specular * 0.08f * spec_tmp_col[i], base_color[i], metallic);
                })

                const float spec_ior = (2.0f / (1.0f - std::sqrt(0.08f * specular))) - 1.0f;
                const float spec_F0 = fresnel_dielectric_cos(1.0f, spec_ior);

                // Approximation of FH (using shading normal)
                const simd_fvec<S> FN =
                    (fresnel_dielectric_cos(dot(I, N), simd_fvec<S>{spec_ior}) - spec_F0) / (1.0f - spec_F0);

                simd_fvec<S> approx_spec_col[3];
                ITERATE_3({ approx_spec_col[i] = mix(spec_tmp_col[i], simd_fvec<S>{1.0f}, FN); })

                const simd_fvec<S> spec_color_lum = lum(approx_spec_col);

                simd_fvec<S> diffuse_weight, specular_weight, clearcoat_weight, refraction_weight;
                get_lobe_weights(mix(base_color_lum, simd_fvec<S>{1.0f}, sheen), spec_color_lum, specular, metallic,
                                 transmission, clearcoat, &diffuse_weight, &specular_weight, &clearcoat_weight,
                                 &refraction_weight);

                simd_fvec<S> sheen_color[3];
                ITERATE_3({ sheen_color[i] = sheen * mix(simd_fvec<S>{1.0f}, tint_color[i], sheen_tint); })

                simd_fvec<S> eta = (mat->ext_ior / mat->int_ior);
                where(is_backfacing, eta) = (mat->int_ior / mat->ext_ior);

                const simd_fvec<S> fresnel = fresnel_dielectric_cos(dot(I, N), 1.0f / eta);

                const float clearcoat_ior = (2.0f / (1.0f - std::sqrt(0.08f * clearcoat))) - 1.0f;
                const float clearcoat_F0 = fresnel_dielectric_cos(1.0f, clearcoat_ior);
                const float clearcoat_roughness2 = clearcoat_roughness * clearcoat_roughness;

                const simd_fvec<S> transmission_roughness =
                    1.0f - (1.0f - roughness) * (1.0f - unpack_unorm_16(mat->transmission_roughness_unorm));
                const simd_fvec<S> transmission_roughness2 = transmission_roughness * transmission_roughness;

#if USE_NEE
                const simd_ivec<S> _is_backfacing = simd_cast(N_dot_L < 0.0f);
                const simd_ivec<S> _is_frontfacing = simd_cast(N_dot_L > 0.0f);

                const simd_ivec<S> eval_light = simd_cast(ls.pdf > 0.0f) & ray_queue[index];
                if (eval_light.not_all_zeros()) {
                    const simd_fvec<S> _I[3] = {-I[0], -I[1], -I[2]};

                    simd_fvec<S> lcol[3] = {0.0f, 0.0f, 0.0f};
                    simd_fvec<S> bsdf_pdf = 0.0f;

                    const simd_ivec<S> eval_diff_lobe = simd_cast(diffuse_weight > 0.0f) & _is_frontfacing & eval_light;
                    if (eval_diff_lobe.not_all_zeros()) {
                        simd_fvec<S> diff_col[4];
                        Evaluate_PrincipledDiffuse_BSDF(_I, N, ls.L, roughness, base_color, sheen_color,
                                                        pi.use_uniform_sampling(), diff_col);

                        where(eval_diff_lobe, bsdf_pdf) += diffuse_weight * diff_col[3];
                        ITERATE_3({ diff_col[i] *= (1.0f - metallic); })

                        ITERATE_3(
                            { where(eval_diff_lobe, lcol[i]) += ls.col[i] * N_dot_L * diff_col[i] / (PI * ls.pdf); })
                    }

                    simd_fvec<S> H[3];
                    ITERATE_3({
                        H[i] = ls.L[i] - I[i] * eta;
                        where(_is_frontfacing, H[i]) = ls.L[i] - I[i];
                    })
                    normalize(H);

                    const simd_fvec<S> roughness2 = roughness * roughness;
                    const simd_fvec<S> aspect = std::sqrt(1.0f - 0.9f * unpack_unorm_16(mat->anisotropic_unorm));

                    const simd_fvec<S> alpha_x = roughness2 / aspect;
                    const simd_fvec<S> alpha_y = roughness2 * aspect;

                    simd_fvec<S> view_dir_ts[3], light_dir_ts[3], sampled_normal_ts[3];
                    tangent_from_world(T, B, N, _I, view_dir_ts);
                    tangent_from_world(T, B, N, ls.L, light_dir_ts);
                    tangent_from_world(T, B, N, H, sampled_normal_ts);

                    const simd_ivec<S> eval_spec_lobe = simd_cast(specular_weight > 0.0f) &
                                                        simd_cast(alpha_x * alpha_y >= 1e-7f) & _is_frontfacing &
                                                        eval_light;
                    if (eval_spec_lobe.not_all_zeros()) {
                        simd_fvec<S> spec_col[4];
                        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, alpha_x, alpha_y,
                                                  simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0}, spec_tmp_col,
                                                  spec_col);

                        where(eval_spec_lobe, bsdf_pdf) += specular_weight * spec_col[3];

                        ITERATE_3({ where(eval_spec_lobe, lcol[i]) += ls.col[i] * spec_col[i] / ls.pdf; })
                    }

                    const simd_ivec<S> eval_coat_lobe =
                        simd_cast(clearcoat_weight > 0.0f) &
                        simd_ivec<S>{clearcoat_roughness2 * clearcoat_roughness2 >= 1e-7f ? -1 : 0} & _is_frontfacing &
                        eval_light;
                    if (eval_coat_lobe.not_all_zeros()) {
                        simd_fvec<S> clearcoat_col[4];
                        Evaluate_PrincipledClearcoat_BSDF(
                            view_dir_ts, sampled_normal_ts, light_dir_ts, simd_fvec<S>{clearcoat_roughness2},
                            simd_fvec<S>{clearcoat_ior}, simd_fvec<S>{clearcoat_F0}, clearcoat_col);

                        where(eval_coat_lobe, bsdf_pdf) += clearcoat_weight * clearcoat_col[3];

                        ITERATE_3({ where(eval_coat_lobe, lcol[i]) += 0.25f * ls.col[i] * clearcoat_col[i] / ls.pdf; })
                    }

                    const simd_ivec<S> eval_refr_spec_lobe =
                        simd_cast(fresnel != 0.0f) & simd_cast(refraction_weight > 0.0f) &
                        simd_cast(roughness2 * roughness2 >= 1e-7f) & _is_frontfacing & eval_light;
                    if (eval_refr_spec_lobe.not_all_zeros()) {
                        simd_fvec<S> spec_col[4], spec_temp_col[3] = {1.0f, 1.0f, 1.0f};
                        Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, roughness2, roughness2,
                                                  simd_fvec<S>{1.0f} /* ior */, simd_fvec<S>{0.0f} /* F0 */,
                                                  spec_temp_col, spec_col);
                        where(eval_refr_spec_lobe, bsdf_pdf) += refraction_weight * fresnel * spec_col[3];

                        ITERATE_3(
                            { where(eval_refr_spec_lobe, lcol[i]) += ls.col[i] * spec_col[i] * (fresnel / ls.pdf); })
                    }

                    const simd_ivec<S> eval_refr_trans_lobe =
                        simd_cast(fresnel != 1.0f) & simd_cast(refraction_weight > 0.0f) &
                        simd_cast(transmission_roughness2 * transmission_roughness2 >= 1e-7f) & _is_backfacing &
                        eval_light;
                    if (eval_refr_trans_lobe.not_all_zeros()) {
                        simd_fvec<S> refr_col[4];
                        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                                    transmission_roughness2, eta, base_color, refr_col);
                        where(eval_refr_trans_lobe, bsdf_pdf) += refraction_weight * (1.0f - fresnel) * refr_col[3];

                        ITERATE_3({
                            where(eval_refr_trans_lobe, lcol[i]) +=
                                ls.col[i] * refr_col[i] * ((1.0f - fresnel) / ls.pdf);
                        })
                    }

                    simd_fvec<S> mis_weight = 1.0f;
                    where(ls.area > 0.0f, mis_weight) = power_heuristic(ls.pdf, bsdf_pdf);
                    ITERATE_3({ where(eval_light, lcol[i]) *= mix_weight * mis_weight; })

                    ///
                    simd_fvec<S> P_biased[3];
                    offset_ray(P, plane_N, P_biased);

                    const simd_fvec<S> _plane_N[3] = {-plane_N[0], -plane_N[1], -plane_N[2]};
                    simd_fvec<S> _P_biased[3];
                    offset_ray(P, _plane_N, _P_biased);

                    ITERATE_3({ where(N_dot_L < 0.0f, P_biased[i]) = _P_biased[i]; })
                    ///

                    ITERATE_3({ where(eval_light, sh_r.o[i]) = P_biased[i]; })
                    ITERATE_3({ where(eval_light, sh_r.d[i]) = ls.L[i]; })
                    where(eval_light, sh_r.dist) = ls.dist - 10.0f * HIT_BIAS;
                    ITERATE_3({ where(eval_light, sh_r.c[i]) = ray.c[i] * lcol[i]; })

                    assert((shadow_mask & eval_light).all_zeros());
                    shadow_mask |= eval_light;
                }
#endif

                const simd_ivec<S> sample_diff_lobe = (diff_depth < pi.settings.max_diff_depth) &
                                                      (total_depth < pi.settings.max_total_depth) &
                                                      simd_cast(mix_rand < diffuse_weight) & ray_queue[index];
                if (sample_diff_lobe.not_all_zeros()) {
                    simd_fvec<S> V[3], diff_col[4];
                    Sample_PrincipledDiffuse_BSDF(T, B, N, I, roughness, base_color, sheen_color,
                                                  pi.use_uniform_sampling(), rand_u, rand_v, V, diff_col);

                    ITERATE_3({ diff_col[i] *= (1.0f - metallic); })

                    simd_fvec<S> new_p[3];
                    offset_ray(P, plane_N, new_p);

                    where(sample_diff_lobe, new_ray.ray_depth) = ray.ray_depth + 0x00000001;

                    ITERATE_3({ where(sample_diff_lobe, new_ray.o[i]) = new_p[i]; })
                    ITERATE_3({ where(sample_diff_lobe, new_ray.d[i]) = V[i]; })
                    ITERATE_3({
                        where(sample_diff_lobe, new_ray.c[i]) = ray.c[i] * diff_col[i] * mix_weight / diffuse_weight;
                    })
                    where(sample_diff_lobe, new_ray.pdf) = diff_col[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(sample_diff_lobe, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(sample_diff_lobe, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(sample_diff_lobe, new_ray.dd_dx[i]) =
                            surf_der.dd_dx[i] -
                            2 * (dot(I, plane_N) * surf_der.dndx[i] + surf_der.ddn_dx[i] * plane_N[i]);
                    })
                    ITERATE_3({
                        where(sample_diff_lobe, new_ray.dd_dy[i]) =
                            surf_der.dd_dy[i] -
                            2 * (dot(I, plane_N) * surf_der.dndy[i] + surf_der.ddn_dy[i] * plane_N[i]);
                    })
#endif

                    assert((secondary_mask & sample_diff_lobe).all_zeros());
                    secondary_mask |= sample_diff_lobe;
                }

                const simd_ivec<S> sample_spec_lobe =
                    (spec_depth < pi.settings.max_spec_depth) & (total_depth < pi.settings.max_total_depth) &
                    simd_cast(mix_rand >= diffuse_weight) & simd_cast(mix_rand < diffuse_weight + specular_weight) &
                    ray_queue[index];
                if (sample_spec_lobe.not_all_zeros()) {
                    simd_fvec<S> V[3], F[4];
                    Sample_GGXSpecular_BSDF(
                        T, B, N, I, roughness, simd_fvec<S>{unpack_unorm_16(mat->anisotropic_unorm)},
                        simd_fvec<S>{spec_ior}, simd_fvec<S>{spec_F0}, spec_tmp_col, rand_u, rand_v, V, F);
                    F[3] *= specular_weight;

                    simd_fvec<S> new_p[3];
                    offset_ray(P, plane_N, new_p);

                    where(sample_spec_lobe, new_ray.ray_depth) = ray.ray_depth + 0x00000100;

                    ITERATE_3({ where(sample_spec_lobe, new_ray.o[i]) = new_p[i]; })
                    ITERATE_3({ where(sample_spec_lobe, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(sample_spec_lobe, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(sample_spec_lobe, new_ray.pdf) = F[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(sample_spec_lobe, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(sample_spec_lobe, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(sample_spec_lobe, new_ray.dd_dx[i]) =
                            surf_der.dd_dx[i] -
                            2 * (dot(I, plane_N) * surf_der.dndx[i] + surf_der.ddn_dx[i] * plane_N[i]);
                    })
                    ITERATE_3({
                        where(sample_spec_lobe, new_ray.dd_dy[i]) =
                            surf_der.dd_dy[i] -
                            2 * (dot(I, plane_N) * surf_der.dndy[i] + surf_der.ddn_dy[i] * plane_N[i]);
                    })
#endif

                    assert(((sample_spec_lobe & ray_queue[index]) != sample_spec_lobe).all_zeros());
                    assert((secondary_mask & sample_spec_lobe).all_zeros());
                    secondary_mask |= sample_spec_lobe;
                }

                const simd_ivec<S> sample_coat_lobe =
                    (spec_depth < pi.settings.max_spec_depth) & (total_depth < pi.settings.max_total_depth) &
                    simd_cast(mix_rand >= diffuse_weight + specular_weight) &
                    simd_cast(mix_rand < diffuse_weight + specular_weight + clearcoat_weight) & ray_queue[index];
                if (sample_coat_lobe.not_all_zeros()) {
                    simd_fvec<S> V[3], F[4];
                    Sample_PrincipledClearcoat_BSDF(T, B, N, I, clearcoat_roughness2, clearcoat_ior, clearcoat_F0,
                                                    rand_u, rand_v, V, F);
                    F[3] *= clearcoat_weight;

                    simd_fvec<S> new_p[3];
                    offset_ray(P, plane_N, new_p);

                    where(sample_coat_lobe, new_ray.ray_depth) = ray.ray_depth + 0x00000100;

                    ITERATE_3({ where(sample_coat_lobe, new_ray.o[i]) = new_p[i]; })
                    ITERATE_3({ where(sample_coat_lobe, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(sample_coat_lobe, new_ray.c[i]) = 0.25f * ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(sample_coat_lobe, new_ray.pdf) = F[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(sample_coat_lobe, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(sample_coat_lobe, new_ray.do_dy[i]) = surf_der.do_dy[i]; })

                    ITERATE_3({
                        where(sample_coat_lobe, new_ray.dd_dx[i]) =
                            surf_der.dd_dx[i] -
                            2 * (dot(I, plane_N) * surf_der.dndx[i] + surf_der.ddn_dx[i] * plane_N[i]);
                    })
                    ITERATE_3({
                        where(sample_coat_lobe, new_ray.dd_dy[i]) =
                            surf_der.dd_dy[i] -
                            2 * (dot(I, plane_N) * surf_der.dndy[i] + surf_der.ddn_dy[i] * plane_N[i]);
                    })
#endif

                    assert((secondary_mask & sample_coat_lobe).all_zeros());
                    secondary_mask |= sample_coat_lobe;
                }

                const simd_ivec<S> sample_trans_lobe =
                    simd_cast(mix_rand >= diffuse_weight + specular_weight + clearcoat_weight) &
                    ((simd_cast(mix_rand >= fresnel) & (refr_depth < pi.settings.max_refr_depth)) |
                     (simd_cast(mix_rand < fresnel) & (spec_depth < pi.settings.max_spec_depth))) &
                    (total_depth < pi.settings.max_total_depth) & ray_queue[index];
                if (sample_trans_lobe.not_all_zeros()) {
                    where(sample_trans_lobe, mix_rand) -= diffuse_weight + specular_weight + clearcoat_weight;
                    where(sample_trans_lobe, mix_rand) /= refraction_weight;

                    simd_fvec<S> F[4], V[3];

                    const simd_ivec<S> sample_trans_spec_lobe = simd_cast(mix_rand < fresnel) & sample_trans_lobe;
                    if (sample_trans_spec_lobe.not_all_zeros()) {
                        const simd_fvec<S> spec_tmp_col[3] = {{1.0f}, {1.0f}, {1.0f}};
                        Sample_GGXSpecular_BSDF(T, B, N, I, roughness, simd_fvec<S>{0.0f} /* anisotropic */,
                                                simd_fvec<S>{1.0f} /* ior */, simd_fvec<S>{0.0f} /* F0 */, spec_tmp_col,
                                                rand_u, rand_v, V, F);

                        simd_fvec<S> new_p[3];
                        offset_ray(P, plane_N, new_p);

                        where(sample_trans_spec_lobe, new_ray.ray_depth) = ray.ray_depth + 0x00000100;

                        ITERATE_3({ where(sample_trans_spec_lobe, new_ray.o[i]) = new_p[i]; })

#ifdef USE_RAY_DIFFERENTIALS
                        ITERATE_3({
                            where(sample_trans_spec_lobe, new_ray.dd_dx[i]) =
                                surf_der.dd_dx[i] -
                                2 * (dot(I, plane_N) * surf_der.dndx[i] + surf_der.ddn_dx[i] * plane_N[i]);
                        })
                        ITERATE_3({
                            where(sample_trans_spec_lobe, new_ray.dd_dy[i]) =
                                surf_der.dd_dy[i] -
                                2 * (dot(I, plane_N) * surf_der.dndy[i] + surf_der.ddn_dy[i] * plane_N[i]);
                        })
#endif
                    }

                    const simd_ivec<S> sample_trans_refr_lobe = ~sample_trans_spec_lobe & sample_trans_lobe;
                    if (sample_trans_refr_lobe.not_all_zeros()) {
                        simd_fvec<S> _F[4], _V[4];
                        Sample_GGXRefraction_BSDF(T, B, N, I, transmission_roughness, eta, base_color, rand_u, rand_v,
                                                  _V, _F);
                        const simd_fvec<S> m = _V[3];

                        const simd_fvec<S> _plane_N[3] = {-plane_N[0], -plane_N[1], -plane_N[2]};
                        simd_fvec<S> new_p[3];
                        offset_ray(P, _plane_N, new_p);

                        where(sample_trans_refr_lobe, new_ray.ray_depth) = ray.ray_depth + 0x00010000;

                        ITERATE_4({ where(sample_trans_refr_lobe, F[i]) = _F[i]; })
                        ITERATE_3({ where(sample_trans_refr_lobe, V[i]) = _V[i]; })

                        ITERATE_3({ where(sample_trans_refr_lobe, new_ray.o[i]) = new_p[i]; })

#ifdef USE_RAY_DIFFERENTIALS
                        const simd_fvec<S> k = (eta - eta * eta * dot(I, plane_N) / dot(_V, plane_N));
                        const simd_fvec<S> dmdx = k * surf_der.ddn_dx;
                        const simd_fvec<S> dmdy = k * surf_der.ddn_dy;

                        ITERATE_3({
                            where(sample_trans_refr_lobe, new_ray.dd_dx[i]) =
                                eta * surf_der.dd_dx[i] - (m * surf_der.dndx[i] + dmdx * plane_N[i]);
                        })
                        ITERATE_3({
                            where(sample_trans_refr_lobe, new_ray.dd_dy[i]) =
                                eta * surf_der.dd_dy[i] - (m * surf_der.dndy[i] + dmdx * plane_N[i]);
                        })
#endif
                    }

                    F[3] *= refraction_weight;

                    ITERATE_3({ where(sample_trans_lobe, new_ray.d[i]) = V[i]; })
                    ITERATE_3({ where(sample_trans_lobe, new_ray.c[i]) = ray.c[i] * F[i] * mix_weight / F[3]; })
                    where(sample_trans_lobe, new_ray.pdf) = F[3];

#ifdef USE_RAY_DIFFERENTIALS
                    ITERATE_3({ where(sample_trans_lobe, new_ray.do_dx[i]) = surf_der.do_dx[i]; })
                    ITERATE_3({ where(sample_trans_lobe, new_ray.do_dy[i]) = surf_der.do_dy[i]; })
#endif

                    assert((sample_trans_spec_lobe & sample_trans_refr_lobe).all_zeros());
                    assert((secondary_mask & sample_trans_lobe).all_zeros());
                    secondary_mask |= sample_trans_lobe;
                }
            }

            ++index;
        }
    }

#if USE_PATH_TERMINATION
    const simd_ivec<S> can_terminate_path = total_depth >= int(pi.settings.termination_start_depth);
#else
    const simd_ivec<S> can_terminate_path = {0};
#endif

    const simd_fvec<S> lum = max(new_ray.c[0], max(new_ray.c[1], new_ray.c[2]));
    const simd_fvec<S> p = fract(halton[RAND_DIM_TERMINATE] + sample_off[0]);
    simd_fvec<S> q = {0.0f};
    where(can_terminate_path, q) = max(0.05f, 1.0f - lum);

    secondary_mask &= simd_cast(p >= q) & simd_cast(lum > 0.0f) & simd_cast(new_ray.pdf > 0.0f);
    if (secondary_mask.not_all_zeros()) {
        ITERATE_3({ new_ray.c[i] /= (1.0f - q); })

        const int index = (*out_secondary_rays_count)++;
        out_secondary_masks[index] = secondary_mask;
        out_secondary_rays[index] = new_ray;
    }

    if (shadow_mask.not_all_zeros()) {
        const int index = (*out_shadow_rays_count)++;
        out_shadow_masks[index] = shadow_mask;
        out_shadow_rays[index] = sh_r;
    }

    where(is_active_lane, out_rgba[0]) = ray.c[0] * col[0];
    where(is_active_lane, out_rgba[1]) = ray.c[1] * col[1];
    where(is_active_lane, out_rgba[2]) = ray.c[2] * col[2];
    where(is_active_lane, out_rgba[3]) = 1.0f;
}

#undef USE_VNDF_GGX_SAMPLING
#undef USE_NEE
#undef USE_PATH_TERMINATION
#undef FORCE_TEXTURE_LOD0

#pragma warning(pop)
