#pragma once

#include <vector>

#include <math/common.hpp>

#include "Core.h"

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

    // hint for math::aligned_vector
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

    hit_data_t(eUninitialize) {}
    hit_data_t();

    // hint for math::aligned_vector
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
void GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, math::aligned_vector<ray_packet_t> &out_rays);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);
bool IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, int num_cones, hit_data_t &out_inter);
bool IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, int num_boxes, hit_data_t &out_inter);

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
math::vec3 TransformNormal(const math::vec3 &n, const float *inv_xform);
void TransformUVs(const float uvs[2], const float tex_atlas_size[2], const texture_t *t, int mip_level, float out_uvs[2]);

// Shade
ray::pixel_color_t ShadeSurface(const int index, const int iteration, const float *halton, const hit_data_t &inter, const ray_packet_t &ray, 
                                const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                const material_t *materials, const texture_t *textures, const TextureAtlas &tex_atlas, ray_packet_t *out_secondary_rays, int *out_secondary_rays_count);
}
}