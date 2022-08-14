#ifndef INTERSECT_GLSL
#define INTERSECT_GLSL

#include "types.glsl"

void IntersectTri(vec3 ro, vec3 rd, tri_accel_t tri, uint prim_index, inout hit_data_t inter) {
    float det = dot(rd, tri.n_plane.xyz);
    float dett = tri.n_plane.w - dot(ro, tri.n_plane.xyz);
    if (sign(dett) != sign(det * inter.t - dett)) {
        return;
    }

    vec3 p = det * ro + dett * rd;

    float detu = dot(p, tri.u_plane.xyz) + det * tri.u_plane.w;
    if (sign(detu) != sign(det - detu)) {
        return;
    }

    float detv = dot(p, tri.v_plane.xyz) + det * tri.v_plane.w;
    if (sign(detv) != sign(det - detu - detv)) {
        return;
    }

    float rdet = (1.0 / det);

    inter.mask = -1;
    inter.prim_index = (det < 0.0) ? int(prim_index) : -int(prim_index) - 1;
    inter.t = dett * rdet;
    inter.u = detu * rdet;
    inter.v = detv * rdet;
}

#ifdef FETCH_TRI
void IntersectTris_ClosestHit(vec3 ro, vec3 rd, int tri_start, int tri_end, int obj_index,
                              inout hit_data_t out_inter) {
    hit_data_t inter;
    inter.mask = 0;
    inter.obj_index = obj_index;
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, FETCH_TRI(i), i, inter);
    }

    out_inter.mask |= inter.mask;
    out_inter.obj_index = inter.mask != 0 ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.mask != 0 ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask != 0 ? inter.u : out_inter.u;
    out_inter.v = inter.mask != 0 ? inter.v : out_inter.v;
}

// TODO: make this actually anyhit
bool IntersectTris_AnyHit(vec3 ro, vec3 rd, int tri_start, int tri_end, int obj_index,
                          inout hit_data_t out_inter) {
    hit_data_t inter;
    inter.mask = 0;
    inter.obj_index = obj_index;
    inter.t = out_inter.t;

    for (int i = tri_start; i < tri_end; ++i) {
        IntersectTri(ro, rd, FETCH_TRI(i), i, inter);
    }

    out_inter.mask |= inter.mask;
    out_inter.obj_index = inter.mask != 0 ? inter.obj_index : out_inter.obj_index;
    out_inter.prim_index = inter.mask != 0 ? inter.prim_index : out_inter.prim_index;
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask != 0 ? inter.u : out_inter.u;
    out_inter.v = inter.mask != 0 ? inter.v : out_inter.v;

    return inter.mask != 0;
}
#endif

#endif // INTERSECT_GLSL