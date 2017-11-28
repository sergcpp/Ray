#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include "Core.h"

namespace ray {
namespace ocl {
struct ray_packet_t {
    // origin and direction (o.w and d.w are used for pixel coordinates)
    cl_float4 o, d;
    // color of ray, determines secondary ray influence
    cl_float3 c;
    // derivatives
    cl_float3 do_dx, dd_dx, do_dy, dd_dy;
};
static_assert(sizeof(ray_packet_t) == 112, "!");

struct aabox_t {
    cl_float3 min, max;

    aabox_t() {}
    aabox_t(const ray::aabox_t &box) {
        memcpy(&min, &box.min[0], sizeof(float) * 3);
        memcpy(&max, &box.max[0], sizeof(float) * 3);
    }
};
static_assert(sizeof(aabox_t) == 32, "!");

struct cone_accel_t {
    cl_float3 o, v;
    float cos_phi_sqr;
    float cone_start, cone_end;

    cone_accel_t() {}
    cone_accel_t(const ray::cone_accel_t &acc) {
        memcpy(&o, &acc.o[0], sizeof(float) * 3);
        memcpy(&v, &acc.v[0], sizeof(float) * 3);
        cos_phi_sqr = acc.cos_phi_sqr;
        cone_start = acc.cone_start;
        cone_end = acc.cone_end;
    }
};

const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = 1;

struct camera_t {
    cl_float3 origin, fwd, side, up;

    camera_t() {}
    camera_t(const ray::camera_t &cam) {
        memcpy(&origin, &cam.origin[0], sizeof(float) * 3);
        memcpy(&fwd, &cam.fwd[0], sizeof(float) * 3);
        memcpy(&side, &cam.side[0], sizeof(float) * 3);
        memcpy(&up, &cam.up[0], sizeof(float) * 3);
    }
};

struct hit_data_t {
    cl_int mask, obj_index, prim_index;
    cl_float t, u, v;
    cl_float2 ray_id;
};
static_assert(sizeof(hit_data_t) == 32, "!");

struct environment_t {
    cl_float3 sun_dir;
    cl_float3 sun_col;
    cl_float3 sky_col; // TODO: replace with spherical garm.
    cl_float sun_softness;
    cl_float pad[3];
    // TODO: add environment texture
};
static_assert(sizeof(environment_t) == 64, "!");
}
}