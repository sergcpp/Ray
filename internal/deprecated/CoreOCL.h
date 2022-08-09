#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include "Core.h"

namespace Ray {
namespace Ocl {
struct ray_data_t {
    // origin and direction (o.w and d.w are used for pixel coordinates)
    cl_float4 o, d;
    // color of ray, determines secondary Ray influence
    cl_float3 c;
    // derivatives
    cl_float4 do_dx, dd_dx, do_dy, dd_dy;
};
static_assert(sizeof(ray_data_t) == 112, "!");

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = 1;

const int CAM_USE_TENT_FILTER = 1;

struct camera_t {
    cl_float4 origin, fwd;
    cl_float4 side, up;
    cl_int flags;
    cl_int pad[3];

    camera_t() {}
    explicit camera_t(const Ray::camera_t &cam) {
        memcpy(&origin, &cam.origin[0], sizeof(float) * 3);
        origin.w = cam.fov;
        memcpy(&fwd, &cam.fwd[0], sizeof(float) * 3);
        fwd.w = cam.gamma;
        memcpy(&side, &cam.side[0], sizeof(float) * 3);
        side.w = cam.focus_distance;
        memcpy(&up, &cam.up[0], sizeof(float) * 3);
        up.w = cam.focus_factor;
        flags = 0;
        if (cam.filter == Tent) {
            flags |= CAM_USE_TENT_FILTER;
        }
    }
};

struct hit_data_t {
    cl_int mask, obj_index, prim_index;
    cl_float t, u, v;
    cl_float2 ray_id;
};
static_assert(sizeof(hit_data_t) == 32, "!");

struct environment_t {
    cl_float4 env_col_and_clamp;
    cl_uint env_map;
    cl_int pad[3];
};
static_assert(sizeof(environment_t) == 32, "!");
} // namespace Ocl
} // namespace Ray