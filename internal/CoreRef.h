#pragma once

#include <vector>

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS Ref
#if defined(_M_AMD64) || defined(_M_X64) || (!defined(__ANDROID__) && defined(__x86_64__)) ||                          \
    (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE2
//#pragma message("Ray::Ref::simd_vec will use SSE2")
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#define USE_NEON
//#pragma message("Ray::Ref::simd_vec will use NEON")
#elif defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define USE_SSE2
//#pragma message("Ray::Ref::simd_vec will use SSE2")
#else
#pragma message("Ray::Ref::simd_vec will not use SIMD")
#endif

#include "simd/simd_vec.h"

#undef USE_SSE2
#undef USE_NEON
#undef NS

#pragma pop_macro("NS")

namespace Ray {
namespace Ref {
struct ray_packet_t {
    // origin and direction
    float o[3], d[3], pdf;
    // throughput color of ray
    float c[3];
    // derivatives
    float do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    int xy;
    // four 8-bit ray depth counters
    int ray_depth;
};
static_assert(sizeof(ray_packet_t) == 96, "!");

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

struct hit_data_t {
    int mask;
    int obj_index;
    int prim_index;
    float t, u, v;

    explicit hit_data_t(eUninitialize) {}
    hit_data_t();
};

struct derivatives_t {
    simd_fvec4 do_dx, do_dy, dd_dx, dd_dy;
    simd_fvec2 duv_dx, duv_dy;
    simd_fvec4 dndx, dndy;
    float ddn_dx, ddn_dy;
};

class TextureAtlasBase;
template <typename T, int N> class TextureAtlasLinear;
template <typename T, int N> class TextureAtlasTiled;
using TextureAtlasRGBA = TextureAtlasTiled<uint8_t, 4>;
using TextureAtlasRGB = TextureAtlasTiled<uint8_t, 3>;
using TextureAtlasRG = TextureAtlasTiled<uint8_t, 2>;
using TextureAtlasR = TextureAtlasTiled<uint8_t, 1>;

// Generation of rays
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton,
                         aligned_vector<ray_packet_t> &out_rays);
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr,
                              const uint32_t *vtx_indices, const vertex_t *vertices, const rect_t &r, int w, int h,
                              const float *halton, aligned_vector<ray_packet_t> &out_rays,
                              aligned_vector<hit_data_t> &out_inters);

// Sorting of rays
void SortRays_CPU(ray_packet_t *rays, size_t rays_count, const float root_min[3], const float cell_size[3],
                  uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
void SortRays_GPU(ray_packet_t *rays, size_t rays_count, const float root_min[3], const float cell_size[3],
                  uint32_t *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks,
                  ray_chunk_t *chunks_temp, uint32_t *skeleton);

// Intersect primitives
bool IntersectTris_ClosestHit(const ray_packet_t &r, const tri_accel_t *tris, int tri_start, int tri_end,
                              int obj_index, hit_data_t &out_inter);
bool IntersectTris_AnyHit(const ray_packet_t &r, const tri_accel_t *tris, const tri_mat_data_t *materials,
                          const uint32_t *indices, int tri_start, int tri_end, int obj_index, hit_data_t &out_inter);

// traditional bvh traversal with stack for outer nodes
bool Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                             const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                             const mesh_t *meshes, const transform_t *transforms,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const mbvh_node_t *oct_nodes, uint32_t root_index,
                                             const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                             const mesh_t *meshes, const transform_t *transforms,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                         const mesh_t *meshes, const transform_t *transforms, const tri_accel_t *tris,
                                         const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                         hit_data_t &inter);
bool Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const mbvh_node_t *nodes, uint32_t root_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                         const mesh_t *meshes, const transform_t *transforms, const tri_accel_t *tris,
                                         const tri_mat_data_t *materials, const uint32_t *tri_indices,
                                         hit_data_t &inter);
// traditional bvh traversal with stack for inner nodes
bool Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes,
                                             uint32_t root_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                             int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3], const mbvh_node_t *nodes,
                                             uint32_t root_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                             int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes,
                                         uint32_t root_index, const tri_accel_t *tris, const tri_mat_data_t *materials,
                                         const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3], const mbvh_node_t *nodes,
                                         uint32_t root_index, const tri_accel_t *tris, const tri_mat_data_t *materials,
                                         const uint32_t *tri_indices, int obj_index, hit_data_t &inter);

// BRDFs
float BRDF_PrincipledDiffuse(const simd_fvec4 &V, const simd_fvec4 &N, const simd_fvec4 &L, const simd_fvec4 &H,
                             float roughness);

simd_fvec4 Evaluate_OrenDiffuse_BSDF(const simd_fvec4 &V, const simd_fvec4 &N, const simd_fvec4 &L,
                                     const float roughness, const simd_fvec4 &base_color);
simd_fvec4 Sample_OrenDiffuse_BSDF(const simd_fvec4 &T, const simd_fvec4 &B, const simd_fvec4 &N, const simd_fvec4 &I,
                                   const float roughness, const simd_fvec4 &base_color, const float rand_u,
                                   const float rand_v, simd_fvec4 &out_V);

simd_fvec4 Evaluate_PrincipledDiffuse_BSDF(const simd_fvec4 &V, const simd_fvec4 &N, const simd_fvec4 &L,
                                           const float roughness, const simd_fvec4 &base_color,
                                           const simd_fvec4 &sheen_color, const bool uniform_sampling);
simd_fvec4 Sample_PrincipledDiffuse_BSDF(const simd_fvec4 &T, const simd_fvec4 &B, const simd_fvec4 &N,
                                         const simd_fvec4 &I, const float roughness, const simd_fvec4 &base_color,
                                         const simd_fvec4 &sheen_color, const bool uniform_sampling, const float rand_u,
                                         const float rand_v, simd_fvec4 &out_V);

simd_fvec4 Evaluate_GGXSpecular_BSDF(const simd_fvec4 &view_dir_ts, const simd_fvec4 &sampled_normal_ts,
                                     const simd_fvec4 &reflected_dir_ts, float alpha_x, float alpha_y, float spec_ior,
                                     float spec_F0, const simd_fvec4 &spec_col);
simd_fvec4 Sample_GGXSpecular_BSDF(const simd_fvec4 &T, const simd_fvec4 &B, const simd_fvec4 &N, const simd_fvec4 &I,
                                   float roughness, float anisotropic, float spec_ior, float spec_F0,
                                   const simd_fvec4 &spec_col, float rand_u, float rand_v, simd_fvec4 &out_V);

simd_fvec4 Evaluate_GGXRefraction_BSDF(const simd_fvec4 &view_dir_ts, const simd_fvec4 &sampled_normal_ts,
                                       const simd_fvec4 &refr_dir_ts, float roughness2, float eta,
                                       const simd_fvec4 &spec_col);
simd_fvec4 Sample_GGXRefraction_BSDF(const simd_fvec4 &T, const simd_fvec4 &B, const simd_fvec4 &N, const simd_fvec4 &I,
                                     float roughness, float eta, const simd_fvec4 &refr_col, float rand_u, float rand_v,
                                     simd_fvec4 &out_V);

simd_fvec4 Evaluate_PrincipledClearcoat_BSDF(const simd_fvec4 &view_dir_ts, const simd_fvec4 &sampled_normal_ts,
                                             const simd_fvec4 &reflected_dir_ts, float clearcoat_roughness2,
                                             float clearcoat_ior, float clearcoat_F0);
simd_fvec4 Sample_PrincipledClearcoat_BSDF(const simd_fvec4 &T, const simd_fvec4 &B, const simd_fvec4 &N,
                                           const simd_fvec4 &I, float clearcoat_roughness2, float clearcoat_ior,
                                           float clearcoat_F0, float rand_u, float rand_v, simd_fvec4 &out_V);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
simd_fvec4 TransformPoint(const simd_fvec4 &p, const float *xform);
simd_fvec4 TransformDirection(const simd_fvec4 &p, const float *xform);
simd_fvec4 TransformNormal(const simd_fvec4 &n, const float *inv_xform);
simd_fvec2 TransformUV(const simd_fvec2 &uv, const simd_fvec2 &tex_atlas_size, const texture_t &t, int mip_level);

// Sample Texture
simd_fvec4 SampleNearest(const TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleBilinear(const TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleBilinear(const TextureAtlasBase &atlas, const simd_fvec2 &uvs, int page);
simd_fvec4 SampleTrilinear(const TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleAnisotropic(const TextureAtlasBase *atlases[], const texture_t &t, const simd_fvec2 &uvs,
                             const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy);
simd_fvec4 SampleLatlong_RGBE(const TextureAtlasRGBA &atlas, const texture_t &t, const simd_fvec4 &dir);

// Get visibility between two points accounting for transparent materials
float ComputeVisibility(const simd_fvec4 &p, const simd_fvec4 &d, float dist, float rand_val, int rand_hash2,
                        const scene_data_t &sc, uint32_t node_index, const TextureAtlasBase *tex_atlases[]);

// Compute derivatives at hit point
void ComputeDerivatives(const simd_fvec4 &I, float t, const simd_fvec4 &do_dx, const simd_fvec4 &do_dy,
                        const simd_fvec4 &dd_dx, const simd_fvec4 &dd_dy, const vertex_t &v1, const vertex_t &v2,
                        const vertex_t &v3, const transform_t &tr, const simd_fvec4 &plane_N, derivatives_t &out_der);

// Evaluate direct light contribution
simd_fvec4 EvaluateDirectLights(const simd_fvec4 &I, const simd_fvec4 &P, const simd_fvec4 &N, const simd_fvec4 &T,
                                const simd_fvec4 &B, const simd_fvec4 &plane_N, const simd_fvec2 &uvs,
                                const bool is_backfacing, const material_t *mat, const derivatives_t &surf_der,
                                const pass_info_t &pi, const scene_data_t &sc, const TextureAtlasBase *tex_atlases[],
                                const uint32_t node_index, const float halton[], const float sample_off[2]);

// Shade
Ray::pixel_color_t ShadeSurface(const pass_info_t &pi, const hit_data_t &inter, const ray_packet_t &ray,
                                const float *halton, const scene_data_t &sc, uint32_t node_index,
                                const TextureAtlasBase *tex_atlases[], ray_packet_t *out_secondary_rays,
                                int *out_secondary_rays_count);
} // namespace Ref
} // namespace Ray
