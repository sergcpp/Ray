#pragma once

#include <vector>

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS Ref
#if defined(_M_AMD64) || defined(_M_X64) || (!defined(__ANDROID__) && defined(__x86_64__)) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE2
//#pragma message("Ray::Ref::simd_vec will use SSE2")
#elif defined(__ARM_NEON__) || defined(__aarch64__)
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
    float o[3], d[3];
    // color of Ray and ior of medium
    float c[3], ior;
    // derivatives
    float do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    int xy;
    // four 8-bit ray depth counters
    int ray_depth;
};

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = 1;

struct hit_data_t {
    int mask_values[RayPacketSize];
    int obj_indices[RayPacketSize];
    int prim_indices[RayPacketSize];
    float t, u, v;
    int xy;

    explicit hit_data_t(eUninitialize) {}
    hit_data_t();
};

struct derivatives_t {
    simd_fvec3 do_dx, do_dy, dd_dx, dd_dy;
    simd_fvec2 duv_dx, duv_dy;
    simd_fvec3 dndx, dndy;
    float ddn_dx, ddn_dy;
};

class TextureAtlasTiled;
using TextureAtlas = TextureAtlasTiled;

// Generation of rays
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays);
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                              const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays, aligned_vector<hit_data_t> &out_inters);

// Sorting of rays
void SortRays_CPU(ray_packet_t *rays, size_t rays_count, const float root_min[3], const float cell_size[3],
                  uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
void SortRays_GPU(ray_packet_t *rays, size_t rays_count, const float root_min[3], const float cell_size[3],
                  uint32_t *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

// Intersect primitives
bool IntersectTris_ClosestHit(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris_ClosestHit(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);

bool IntersectTris_AnyHit(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris_AnyHit(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);

#ifdef USE_STACKLESS_BVH_TRAVERSAL
// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_Stackless_CPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less gpu-style traversal of outer nodes
bool Traverse_MacroTree_Stackless_GPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less cpu-style traversal of inner nodes
bool Traverse_MicroTree_Stackless_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
// stack-less gpu-style traversal of inner nodes
bool Traverse_MicroTree_Stackless_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter);
#endif

// traditional bvh traversal with stack for outer nodes
bool Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                             const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_MacroTree_WithStack_ClosestHit(const ray_packet_t &r, const bvh_node8_t *oct_nodes, uint32_t root_index,
                                             const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                         const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
bool Traverse_MacroTree_WithStack_AnyHit(const ray_packet_t &r, const bvh_node8_t *nodes, uint32_t root_index,
                                         const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                         const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// traditional bvh traversal with stack for inner nodes
bool Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_ClosestHit(const ray_packet_t &r, const float inv_d[3], const bvh_node8_t *nodes, uint32_t root_index,
                                             const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                         const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
bool Traverse_MicroTree_WithStack_AnyHit(const ray_packet_t &r, const float inv_d[3], const bvh_node8_t *nodes, uint32_t root_index,
                                         const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);

// BRDFs
float BRDF_OrenNayar(const simd_fvec3 &L, const simd_fvec3 &I, const simd_fvec3 &N, const simd_fvec3 &B, float sigma);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
simd_fvec3 TransformPoint(const simd_fvec3 &p, const float *xform);
simd_fvec3 TransformNormal(const simd_fvec3 &n, const float *inv_xform);
simd_fvec2 TransformUV(const simd_fvec2 &uv, const simd_fvec2 &tex_atlas_size, const texture_t &t, int mip_level);

// Sample Texture
simd_fvec4 SampleNearest(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const simd_fvec2 &uvs, int page);
simd_fvec4 SampleTrilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleAnisotropic(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy);
simd_fvec4 SampleLatlong_RGBE(const TextureAtlas &atlas, const texture_t &t, const simd_fvec3 &dir);

// Get visibility between two points accounting for transparent materials
float ComputeVisibility(const simd_fvec3 &p1, const simd_fvec3 &p2, const float *halton, const int hi, int rand_hash2,
                        const scene_data_t &sc, uint32_t node_index, const TextureAtlas &tex_atlas);

// Compute punctual lights contribution
void AcumulateLightContribution(const light_t &l, const simd_fvec3 &I, const simd_fvec3 &P, const simd_fvec3 &N, const simd_fvec3 &B, const simd_fvec3 &plane_N,
                                const scene_data_t &sc, uint32_t node_index, const TextureAtlas &tex_atlas,
                                float sigma, const float *halton, const int hi, int rand_hash2, float rand_offset, float rand_offset2, simd_fvec3 &col);
simd_fvec3 ComputeDirectLighting(const simd_fvec3 &I, const simd_fvec3 &P, const simd_fvec3 &N, const simd_fvec3 &B, const simd_fvec3 &plane_N,
                                 float sigma, const float *halton, const int hi, int rand_hash, int rand_hash2, float rand_offset, float rand_offset2,
                                 const scene_data_t &sc, uint32_t node_index, uint32_t light_node_index, const TextureAtlas &tex_atlas);

// Compute derivatives at hit point
void ComputeDerivatives(const simd_fvec3 &I, float t, const simd_fvec3 &do_dx, const simd_fvec3 &do_dy, const simd_fvec3 &dd_dx, const simd_fvec3 &dd_dy,
                        const vertex_t &v1, const vertex_t &v2, const vertex_t &v3, const simd_fvec3 &plane_N, derivatives_t &out_der);

// Shade
Ray::pixel_color_t ShadeSurface(const pass_info_t &pi, const hit_data_t &inter, const ray_packet_t &ray, const float *halton,
                                const scene_data_t &sc, uint32_t node_index, uint32_t light_node_index, const TextureAtlas &tex_atlas,
                                ray_packet_t *out_secondary_rays, int *out_secondary_rays_count);
}
}