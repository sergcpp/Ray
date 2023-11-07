#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "intersect_scene_interface.h"
#include "types.h"

layout(location = 0) rayPayloadInEXT hit_data_t g_pld;

hitAttributeEXT vec2 bary_coord;

void main() {
    g_pld.obj_index = gl_InstanceID;
    int prim_index = gl_InstanceCustomIndexEXT + gl_PrimitiveID;
    if (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT) {
        prim_index = -prim_index - 1;
    }
    g_pld.prim_index = prim_index;
    g_pld.t = gl_HitTEXT;
    g_pld.u = bary_coord.x;
    g_pld.v = bary_coord.y;
}


