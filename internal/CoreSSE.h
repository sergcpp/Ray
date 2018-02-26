#pragma once

#include <vector>

#include <emmintrin.h>
#include <xmmintrin.h>

#include <math/common.hpp>

#include "Core.h"

namespace ray {
namespace sse {
struct ray_packet_t {
    // directions of rays in packet
    __m128 d[3];
    // origins of rays in packet
    __m128 o[3];
    // id of ray packet
    rays_id_t id;
    int pad[3];

    // hint for math::aligned_vector
    static const size_t alignment = alignof(__m128);
};

static_assert(sizeof(ray_packet_t) == 112, "!");
static_assert(alignof(ray_packet_t) == 16, "!");

const int RayPacketDimX = 2;
const int RayPacketDimY = 2;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

struct hit_data_t {
    __m128i mask;
    __m128i obj_index;
    __m128i prim_index;
    __m128 t, u, v;
    rays_id_t id;
    int pad[3];

    hit_data_t(eUninitialize) {}
    hit_data_t();

    // hint for math::aligned_vector
    static const size_t alignment = alignof(__m128);
};

static_assert(sizeof(hit_data_t) == 112, "!");
static_assert(alignof(hit_data_t) == alignof(__m128), "!");

// Generating rays
void ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r);
void GeneratePrimaryRays(const camera_t &cam, const region_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays);

// Intersect primitives
bool IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectTris(const ray_packet_t &r, const __m128i ray_mask, const tri_accel_t *tris, const uint32_t *indices, int num_tris, int obj_index, hit_data_t &out_inter);
bool IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, int num_cones, hit_data_t &out_inter);
bool IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, int num_boxes, hit_data_t &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
bool Traverse_MacroTree_CPU(const ray_packet_t &r, const __m128i ray_mask, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                            const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter);
// stack-less cpu-style traversal of inner nodes
bool Traverse_MicroTree_CPU(const ray_packet_t &r, const __m128i ray_mask, const __m128 inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                            const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter);

// Transform
ray_packet_t TransformRay(const ray_packet_t &r, const float *xform);
}
}