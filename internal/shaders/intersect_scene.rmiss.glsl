#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "intersect_scene_interface.h"
#include "types.h"

layout(location = 0) rayPayloadInEXT hit_data_t g_pld;

void main() {
    g_pld.v = -1.0;
}
