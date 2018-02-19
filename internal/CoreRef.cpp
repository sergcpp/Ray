#include "CoreRef.h"

#include <limits>

#include <math/math.hpp>

namespace ray {
namespace ref {
force_inline void _IntersectTri(const ray_packet_t &r, const tri_accel_t &tri, uint32_t i, hit_data_t &inter) {
    const int _next_u[] = { 1, 0, 0 },
                          _next_v[] = { 2, 2, 1 };

    int w = tri.ci & ray::W_BITS,
        u = _next_u[w],
        v = _next_v[w];

    float det = r.d[u] * tri.nu + r.d[v] * tri.nv + r.d[w];
    float dett = tri.np - (r.o[u] * tri.nu + r.o[v] * tri.nv + r.o[w]);
    float Du = r.d[u] * dett - (tri.pu - r.o[u]) * det;
    float Dv = r.d[v] * dett - (tri.pv - r.o[v]) * det;
    float detu = (tri.e1v * Du - tri.e1u * Dv);
    float detv = (tri.e0u * Dv - tri.e0v * Du);

    const float eps = std::numeric_limits<float>::epsilon();

    float tmpdet0 = det - detu - detv;
    //if ((tmpdet0 >= 0 && detu >= 0 && detv >= 0) || (tmpdet0 <= 0 && detu <= 0 && detv <= 0)) {
    if ((tmpdet0 > -eps && detu > -eps && detv > -eps) || (tmpdet0 < eps && detu < eps && detv < eps)) {
        float rdet = 1 / det;
        float t = dett * rdet;
        float u = detu * rdet;
        float v = detv * rdet;

        if (t > 0 && t < inter.t) {
            inter.mask_values[0] = 0xffffffff;
            inter.prim_indices[0] = i;
            inter.t = t;
            inter.u = u;
            inter.v = v;
        }
    }
}

force_inline uint32_t near_child(const ray_packet_t &r, const bvh_node_t &node) {
    return r.d[node.space_axis] < 0 ? node.right_child : node.left_child;
}

force_inline uint32_t far_child(const ray_packet_t &r, const bvh_node_t &node) {
    return r.d[node.space_axis] < 0 ? node.left_child : node.right_child;
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

bool bbox_test(const float o[3], const float inv_d[3], const float t, const float bbox_min[3], const float bbox_max[3]) {
    float low = inv_d[0] * (bbox_min[0] - o[0]);
    float high = inv_d[0] * (bbox_max[0] - o[0]);
    float tmin = math::min(low, high);
    float tmax = math::max(low, high);

    low = inv_d[1] * (bbox_min[1] - o[1]);
    high = inv_d[1] * (bbox_max[1] - o[1]);
    tmin = math::max(tmin, math::min(low, high));
    tmax = math::min(tmax, math::max(low, high));

    low = inv_d[2] * (bbox_min[2] - o[2]);
    high = inv_d[2] * (bbox_max[2] - o[2]);
    tmin = math::max(tmin, math::min(low, high));
    tmax = math::min(tmax, math::max(low, high));

    return tmin <= tmax && tmin <= t && tmax > 0;
}

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

enum eTraversalSource { FromParent, FromChild, FromSibling };
}
}

ray::ref::hit_data_t::hit_data_t() {
    mask_values[0] = 0;
    obj_indices[0] = -1;
    prim_indices[0] = -1;
    t = std::numeric_limits<float>::max();
}

void ray::ref::ConstructRayPacket(const float *o, const float *d, int size, ray_packet_t &out_r) {
    assert(size <= RayPacketSize);

    for (int i = 0; i < 3; i++) {
        out_r.o[i] = o[i];
        out_r.d[i] = d[i];
    }
}

void ray::ref::GeneratePrimaryRays(const camera_t &cam, const region_t &r, int w, int h, math::aligned_vector<ray_packet_t> &out_rays) {
    using namespace math;

    vec3 origin = make_vec3(cam.origin), fwd = make_vec3(cam.fwd), side = make_vec3(cam.side), up = make_vec3(cam.up);

    up *= float(h) / w;

    size_t i = 0;
    out_rays.resize(r.w * r.h);

    for (int y = r.y; y < r.y + r.h; y += RayPacketDimY) {
        for (int x = r.x; x < r.x + r.w; x += RayPacketDimX) {
            vec3 _d(float(x) / w - 0.5f, float(-y) / h + 0.5f, 1);
            _d = _d.x * side + _d.y * up + _d.z * fwd;
            _d = normalize(_d);

            ConstructRayPacket(value_ptr(origin), value_ptr(_d), RayPacketSize, out_rays[i]);
            out_rays[i].id.x = x;
            out_rays[i].id.y = y;
            i++;
        }
    }
}

bool ray::ref::IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_indices[0] = obj_index;
    inter.t = out_inter.t;

    for (int i = 0; i < num_tris; i++) {
        const tri_accel_t &tri = tris[i];

        _IntersectTri(r, tri, i, inter);
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = inter.mask_values[0] ? inter.obj_indices[0] : out_inter.obj_indices[0];
    out_inter.prim_indices[0] = inter.mask_values[0] ? inter.prim_indices[0] : out_inter.prim_indices[0];
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask_values[0] ? inter.u : out_inter.u;
    out_inter.v = inter.mask_values[0] ? inter.v : out_inter.v;

    return inter.mask_values[0] != 0;
}

bool ray::ref::IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, const uint32_t *indices, int num_indices, int obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_indices[0] = obj_index;
    inter.t = out_inter.t;

    for (int i = 0; i < num_indices; i++) {
        const tri_accel_t &tri = tris[indices[i]];

        _IntersectTri(r, tri, indices[i], inter);
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = inter.mask_values[0] ? inter.obj_indices[0] : out_inter.obj_indices[0];
    out_inter.prim_indices[0] = inter.mask_values[0] ? inter.prim_indices[0] : out_inter.prim_indices[0];
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask_values[0] ? inter.u : out_inter.u;
    out_inter.v = inter.mask_values[0] ? inter.v : out_inter.v;

    return inter.mask_values[0] != 0;
}

bool ray::ref::IntersectCones(const ray_packet_t &r, const cone_accel_t *cones, int num_cones, hit_data_t &out_inter) {
    using namespace math;

    hit_data_t inter;
    inter.t = out_inter.t;

    vec3 o = make_vec3(r.o);
    vec3 d = make_vec3(r.d);

    for (int i = 0; i < num_cones; i++) {
        const cone_accel_t &cone = cones[i];

        vec3 cone_o = make_vec3(cone.o);
        vec3 cone_v = make_vec3(cone.v);
        vec3 co = o - cone_o;

        float a = dot(d, cone_v);
        float c = dot(co, cone_v);
        float b = 2 * (a * c - dot(d, co) * cone.cos_phi_sqr);
        a = a * a - cone.cos_phi_sqr;
        c = c * c - dot(co, co) * cone.cos_phi_sqr;

        float D = b * b - 4 * a * c;
        if (D >= 0) {
            D = sqrtf(D);
            float t1 = (-b - D) / (2 * a), t2 = (-b + D) / (2 * a);

            if ((t1 > 0 && t1 < inter.t) || (t2 > 0 && t2 < inter.t)) {
                vec3 p1 = o + t1 * d, p2 = o + t2 * d;
                vec3 p1c = cone_o - p1, p2c = cone_o - p2;

                float dot1 = dot(p1c, cone_v), dot2 = dot(p2c, cone_v);

                if ((dot1 >= cone.cone_start && dot1 <= cone.cone_end) || (dot2 >= cone.cone_start && dot2 <= cone.cone_end)) {
                    inter.mask_values[0] = 0xffffffff;
                    inter.obj_indices[0] = i;
                    inter.t = t1 < t2 ? t1 : t2;
                }
            }
        }
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = (inter.obj_indices[0] & inter.mask_values[0]) + (out_inter.obj_indices[0] & ~inter.mask_values[0]);
    out_inter.t = inter.t; // already contains min value

    return inter.mask_values[0] != 0;
}

bool ray::ref::IntersectBoxes(const ray_packet_t &r, const aabox_t *boxes, int num_boxes, hit_data_t &out_inter) {
    using namespace math;

    hit_data_t inter;
    inter.t = out_inter.t;

    vec3 inv_d = 1.0f / make_vec3(r.d);

    for (int i = 0; i < num_boxes; i++) {
        const aabox_t &box = boxes[i];

        float low = inv_d[0] * (box.min[0] - r.o[0]);
        float high = inv_d[0] * (box.max[0] - r.o[0]);
        float tmin = min(low, high);
        float tmax = max(low, high);

        low = inv_d[1] * (box.min[1] - r.o[1]);
        high = inv_d[1] * (box.max[1] - r.o[1]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        low = inv_d[2] * (box.min[2] - r.o[2]);
        high = inv_d[2] * (box.max[2] - r.o[2]);
        tmin = max(tmin, min(low, high));
        tmax = min(tmax, max(low, high));

        if (tmin <= tmax && tmax > 0 && tmin < inter.t) {
            inter.mask_values[0] = 0xffffffff;
            inter.obj_indices[0] = i;
            inter.t = tmin;
        }
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = (inter.obj_indices[0] & inter.mask_values[0]) + (out_inter.obj_indices[0] & ~inter.mask_values[0]);
    out_inter.t = inter.t; // already contains min value

    return inter.mask_values[0] != 0;
}

bool ray::ref::Traverse_MacroTree_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = node_index;
    eTraversalSource src = FromSibling;

    if (!is_leaf_node(nodes[node_index])) {
        cur = near_child(r, nodes[node_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) return res;
            if (cur == near_child(r, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                    const auto &mi = mesh_instances[mi_indices[i]];
                    const auto &m = meshes[mi.mesh_index];
                    const auto &tr = transforms[mi.tr_index];

                    if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) continue;

                    ray_packet_t _r = TransformRay(r, tr.inv_xform);

                    float _inv_d[3] = { 1.0f / _r.d[0], 1.0f / _r.d[1], 1.0f / _r.d[2] };

                    if (Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter)) {
                        res = true;
                    }
                }

                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        case FromParent:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                    const auto &mi = mesh_instances[mi_indices[i]];
                    const auto &m = meshes[mi.mesh_index];
                    const auto &tr = transforms[mi.tr_index];

                    if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) continue;

                    ray_packet_t _r = TransformRay(r, tr.inv_xform);

                    float _inv_d[3] = { 1.0f / _r.d[0], 1.0f / _r.d[1], 1.0f / _r.d[2] };

                    if (Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter)) {
                        res = true;
                    }
                }

                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        }
    }

    return res;
}

bool ray::ref::Traverse_MacroTree_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = node_index;
    uint32_t last = node_index;

    if (!is_leaf_node(nodes[cur])) {
        cur = near_child(r, nodes[cur]);
    }

    while (true) {
        if (cur == 0xffffffff) return res;

        if (is_leaf_node(nodes[cur])) {
            for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                const auto &mi = mesh_instances[mi_indices[i]];
                const auto &m = meshes[mi.mesh_index];
                const auto &tr = transforms[mi.tr_index];

                if (!bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max)) continue;

                ray_packet_t _r = TransformRay(r, tr.inv_xform);

                float _inv_d[3] = { 1.0f / _r.d[0], 1.0f / _r.d[1], 1.0f / _r.d[2] };

                if (Traverse_MicroTree_GPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter)) {
                    res = true;
                }
            }
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t near = near_child(r, nodes[cur]);
        uint32_t far = far_child(r, nodes[cur]);

        if (last == far) {
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t try_child = (last == nodes[cur].parent) ? near : far;
        if (bbox_test(r.o, inv_d, inter.t, nodes[try_child])) {
            last = cur;
            cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur;
                cur = nodes[cur].parent;
            }
        }
    }

    return res;
}

bool ray::ref::Traverse_MicroTree_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = node_index;
    eTraversalSource src = FromSibling;

    // if not leaf node
    if (!is_leaf_node(nodes[node_index])) {
        cur = near_child(r, nodes[node_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == node_index || cur == 0xffffffff) return res;
            if (cur == near_child(r, nodes[nodes[cur].parent])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                if (IntersectTris(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter)) {
                    res = true;
                }

                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        case FromParent:
            if (!bbox_test(r.o, inv_d, inter.t, nodes[cur])) {
                cur = nodes[cur].sibling;
                src = FromSibling;
            } else if (is_leaf_node(nodes[cur])) {
                // process leaf
                if (IntersectTris(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter)) {
                    res = true;
                }

                cur = nodes[cur].sibling;
                src = FromSibling;
            } else {
                cur = near_child(r, nodes[cur]);
                src = FromParent;
            }
            break;
        }
    }

    return res;
}

bool ray::ref::Traverse_MicroTree_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t node_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = node_index;
    uint32_t last = node_index;

    if (!is_leaf_node(nodes[node_index])) {
        cur = near_child(r, nodes[node_index]);
        //last = cur;
    }

    while (true) {
        if (cur == 0xffffffff) return res;

        if (is_leaf_node(nodes[cur])) {
            if (IntersectTris(r, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter)) {
                res = true;
            }
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t near = near_child(r, nodes[cur]);
        uint32_t far = far_child(r, nodes[cur]);

        if (last == far) {
            last = cur;
            cur = nodes[cur].parent;
            continue;
        }

        uint32_t try_child = (last == nodes[cur].parent) ? near : far;
        if (bbox_test(r.o, inv_d, inter.t, nodes[try_child])) {
            last = cur;
            cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur;
                cur = nodes[cur].parent;
            }
        }
    }

    return res;
}

ray::ref::ray_packet_t ray::ref::TransformRay(const ray_packet_t &r, const float *xform) {
    math::vec3 _o = math::make_mat4(xform) * math::vec4(math::make_vec3(r.o), 1);
    math::vec3 _d = math::make_mat4(xform) * math::vec4(math::make_vec3(r.d), 0);

    math::vec3 inv_d = 1.0f / _d;

    ray_packet_t _r = r;
    memcpy(&_r.o[0], math::value_ptr(_o), 3 * sizeof(float));
    memcpy(&_r.d[0], math::value_ptr(_d), 3 * sizeof(float));

    return _r;
}