#pragma once

#include <vector>

#include <math/common.hpp>

#include "Core.h"

namespace ray {
namespace ref {
struct ray_packet_t {
    rays_id_t id;
    float o[3], d[3];

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

// Generating rays
void ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r);
void GeneratePrimaryRays(const camera_t &cam, int w, int h, math::aligned_vector<ray_packet_t> &out_rays);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter);
bool IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, int num_cones, hit_data_t &out_inter);
bool IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, int num_boxes, hit_data_t &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less gpu-style traversal of outer nodes
bool Traverse_MacroTree_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
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
}
}