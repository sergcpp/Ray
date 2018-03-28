//#pragma once

#include <vector>

#include "Core.h"

#include "simd/simd_vec.h"

namespace ray {
namespace NS {

struct ray_packet_t {
    // directions of rays in packet
    simd_fvec16 d[3];
    // origins of rays in packet
    simd_fvec16 o[3];

    // hint for math::aligned_vector
    static const size_t alignment = alignof(simd_fvec16);
};

static_assert(sizeof(ray_packet_t) == 384, "!");
static_assert(alignof(ray_packet_t) == alignof(simd_fvec16), "!");

const int RayPacketDimX = 4;
const int RayPacketDimY = 4;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

struct hit_data_t {
    simd_ivec16 mask;
    simd_ivec16 obj_index;
    simd_ivec16 prim_index;
    simd_fvec16 t, u, v;

    hit_data_t(eUninitialize) {}
    hit_data_t();

    // hint for math::aligned_vector
    static const size_t alignment = alignof(simd_fvec16);
};

static_assert(sizeof(hit_data_t) == 384, "!");
static_assert(alignof(hit_data_t) == alignof(simd_fvec16), "!");

// Generating rays
void ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r);
void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const simd_ivec16 &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const simd_ivec16 &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t &out_inter);
bool IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, uint32_t num_cones, hit_data_t &out_inter);
bool IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, uint32_t num_boxes, hit_data_t &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_CPU(const ray_packet_t &r, const simd_ivec16 &ray_mask, const simd_fvec16 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less cpu-style traversal of inner nodes
bool Traverse_MicroTree_CPU(const ray_packet_t &r, const simd_ivec16 &ray_mask, const simd_fvec16 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
}
}