#pragma once

#include <vector>

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS ref
#if defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE
#pragma message("ray::ref::simd_vec will use SSE2")
#else
#pragma message("ray::ref::simd_vec will not use SIMD")
#endif

#include "simd/simd_vec.h"

#undef USE_SSE
#undef NS

#pragma pop_macro("NS")

namespace ray {
namespace ref {
struct ray_packet_t {
    rays_id_t id;
    // origin and direction
    float o[3], d[3];
    // color of ray
    float c[3];
    // derivatives
    float do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];

    // hint for aligned_vector
    static const size_t alignment = 1;
};

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = 1;

struct hit_data_t {
    int mask_values[RayPacketSize];
    int obj_indices[RayPacketSize];
    int prim_indices[RayPacketSize];
    float t, u, v;
    rays_id_t id;

    explicit hit_data_t(eUninitialize) {}
    hit_data_t();

    // hint for aligned_vector
    static const size_t alignment = 1;
};

struct environment_t {
    float sun_dir[3];
    float sun_col[3];
    float sky_col[3];
    float sun_softness;
};

class TextureAtlas;

// Generating rays
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_CPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less gpu-style traversal of outer nodes
bool Traverse_MacroTree_GPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less cpu-style traversal of inner nodes
bool Traverse_MicroTree_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);
// stack-less gpu-style traversal of inner nodes
bool Traverse_MicroTree_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
simd_fvec3 TransformNormal(const simd_fvec3 &n, const float *inv_xform);
simd_fvec2 TransformUV(const simd_fvec2 &uv, const simd_fvec2 &tex_atlas_size, const texture_t &t, int mip_level);

// Sample Texture
simd_fvec4 SampleNearest(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod);
simd_fvec4 SampleBilinear(const TextureAtlas &atlas, const simd_fvec2 &uvs, int page);
simd_fvec4 SampleTrilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod);
simd_fvec4 SampleAnisotropic(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy);

// Shade
ray::pixel_color_t ShadeSurface(const int index, const int iteration, const float *halton, const hit_data_t &inter, const ray_packet_t &ray, 
                                const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                const material_t *materials, const texture_t *textures, const TextureAtlas &tex_atlas, ray_packet_t *out_secondary_rays, int *out_secondary_rays_count);
}
}