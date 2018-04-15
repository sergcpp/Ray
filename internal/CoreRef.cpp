#include "CoreRef.h"

#include <limits>

#include <math/math.hpp>

#include "TextureAtlasRef.h"

namespace ray {
namespace ref {
force_inline void _IntersectTri(const ray_packet_t &r, const tri_accel_t &tri, uint32_t i, hit_data_t &inter) {
    const int _next_u[] = { 1, 0, 0 },
                          _next_v[] = { 2, 2, 1 };

    int w = tri.ci & ray::TRI_W_BITS,
        u = _next_u[w],
        v = _next_v[w];

    float det = r.d[u] * tri.nu + r.d[v] * tri.nv + r.d[w];
    float dett = tri.np - (r.o[u] * tri.nu + r.o[v] * tri.nv + r.o[w]);
    float Du = r.d[u] * dett - (tri.pu - r.o[u]) * det;
    float Dv = r.d[v] * dett - (tri.pv - r.o[v]) * det;
    float detu = tri.e1v * Du - tri.e1u * Dv;
    float detv = tri.e0u * Dv - tri.e0v * Du;

    float tmpdet0 = det - detu - detv;
    //if ((tmpdet0 >= 0 && detu >= 0 && detv >= 0) || (tmpdet0 <= 0 && detu <= 0 && detv <= 0)) {
    if ((tmpdet0 > -HIT_EPS && detu > -HIT_EPS && detv > -HIT_EPS) || 
        (tmpdet0 < HIT_EPS && detu < HIT_EPS && detv < HIT_EPS)) {
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
    float tmin = std::min(low, high);
    float tmax = std::max(low, high);

    low = inv_d[1] * (bbox_min[1] - o[1]);
    high = inv_d[1] * (bbox_max[1] - o[1]);
    tmin = std::max(tmin, std::min(low, high));
    tmax = std::min(tmax, std::max(low, high));

    low = inv_d[2] * (bbox_min[2] - o[2]);
    high = inv_d[2] * (bbox_max[2] - o[2]);
    tmin = std::max(tmin, std::min(low, high));
    tmax = std::min(tmax, std::max(low, high));

    return tmin <= tmax && tmin <= t && tmax > 0;
}

force_inline bool bbox_test(const float o[3], const float inv_d[3], const float t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

force_inline int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

force_inline void safe_invert(const float v[3], float out_v[3]) {
    out_v[0] = 1.0f / v[0];
    out_v[1] = 1.0f / v[1];
    out_v[2] = 1.0f / v[2];

    if (v[0] <= FLT_EPS && v[0] >= 0) {
        out_v[0] = std::numeric_limits<float>::max();
    } else if (v[0] >= -FLT_EPS && v[0] < 0) {
        out_v[0] = -std::numeric_limits<float>::max();
    }

    if (v[1] <= FLT_EPS && v[1] >= 0) {
        out_v[1] = std::numeric_limits<float>::max();
    } else if (v[1] >= -FLT_EPS && v[1] < 0) {
        out_v[1] = -std::numeric_limits<float>::max();
    }

    if (v[2] <= FLT_EPS && v[2] >= 0) {
        out_v[2] = std::numeric_limits<float>::max();
    } else if (v[2] >= -FLT_EPS && v[2] < 0) {
        out_v[2] = -std::numeric_limits<float>::max();
    }
}

}
}

ray::ref::hit_data_t::hit_data_t() {
    mask_values[0] = 0;
    obj_indices[0] = -1;
    prim_indices[0] = -1;
    t = std::numeric_limits<float>::max();
}

void ray::ref::GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, math::aligned_vector<ray_packet_t> &out_rays) {
    using namespace math;

    vec3 origin = make_vec3(cam.origin), fwd = make_vec3(cam.fwd), side = make_vec3(cam.side), up = make_vec3(cam.up);

    up *= float(h) / w;

    auto get_pix_dir = [fwd, side, up, w, h](const float x, const float y) {
        vec3 _d(float(x) / w - 0.5f, float(-y) / h + 0.5f, 1);
        _d = _d.x * side + _d.y * up + _d.z * fwd;
        _d = normalize(_d);
        return _d;
    };

    size_t i = 0;
    out_rays.resize((size_t)r.w * r.h);

    for (int y = r.y; y < r.y + r.h; y += RayPacketDimY) {
        for (int x = r.x; x < r.x + r.w; x += RayPacketDimX) {
            auto &out_r = out_rays[i++];

            const int index = y * w + x;
            const int hi = (hash(index) + iteration) & (HaltonSeqLen - 1);

            float _x = (float)x + halton[hi * 2];
            float _y = (float)y + halton[hi * 2 + 1];

            vec3 _d = get_pix_dir(_x, _y);

            vec3 _dx = get_pix_dir(_x + 1, _y),
                 _dy = get_pix_dir(_x, _y + 1);

            for (int j = 0; j < 3; j++) {
                out_r.o[j] = origin[j];
                out_r.d[j] = _d[j];
                out_r.c[j] = 1.0f;

                out_r.do_dx[j] = 0;
                out_r.dd_dx[j] = _dx[j] - _d[j];
                out_r.do_dy[j] = 0;
                out_r.dd_dy[j] = _dy[j] - _d[j];
            }

            out_r.id.x = (uint16_t)x;
            out_r.id.y = (uint16_t)y;
        }
    }
}

bool ray::ref::IntersectTris(const ray_packet_t &r, const tri_accel_t *tris, int num_tris, int obj_index, hit_data_t &out_inter) {
    hit_data_t inter;
    inter.obj_indices[0] = obj_index;
    inter.t = out_inter.t;

    for (int i = 0; i < num_tris; i++) {
        _IntersectTri(r, tris[i], i, inter);
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
        uint32_t index = indices[i];
        _IntersectTri(r, tris[index], index, inter);
    }

    out_inter.mask_values[0] |= inter.mask_values[0];
    out_inter.obj_indices[0] = inter.mask_values[0] ? inter.obj_indices[0] : out_inter.obj_indices[0];
    out_inter.prim_indices[0] = inter.mask_values[0] ? inter.prim_indices[0] : out_inter.prim_indices[0];
    out_inter.t = inter.t; // already contains min value
    out_inter.u = inter.mask_values[0] ? inter.u : out_inter.u;
    out_inter.v = inter.mask_values[0] ? inter.v : out_inter.v;

    return inter.mask_values[0] != 0;
}

bool ray::ref::Traverse_MacroTree_CPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t cur = root_index;
    eTraversalSource src = FromSibling;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) return res;
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

                    float _inv_d[3];
                    safe_invert(_r.d, _inv_d);

                    res |= Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
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

                    float _inv_d[3];
                    safe_invert(_r.d, _inv_d);

                    res |= Traverse_MicroTree_CPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
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

bool ray::ref::Traverse_MacroTree_GPU(const ray_packet_t &r, const bvh_node_t *nodes, uint32_t root_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t &inter) {
    bool res = false;

    float inv_d[3];
    safe_invert(r.d, inv_d);

    uint32_t cur = root_index;
    uint32_t last = root_index;

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

                res |= Traverse_MicroTree_GPU(_r, _inv_d, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
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

bool ray::ref::Traverse_MicroTree_CPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = root_index;
    eTraversalSource src = FromSibling;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        src = FromParent;
    }

    while (true) {
        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) return res;
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
                res |= IntersectTris(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

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
                res |= IntersectTris(r, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

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

bool ray::ref::Traverse_MicroTree_GPU(const ray_packet_t &r, const float inv_d[3], const bvh_node_t *nodes, uint32_t root_index,
                                      const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t &inter) {
    bool res = false;

    uint32_t cur = root_index;
    uint32_t last = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        cur = near_child(r, nodes[root_index]);
        //last = cur;
    }

    while (true) {
        if (cur == 0xffffffff) return res;

        if (is_leaf_node(nodes[cur])) {
            res |= IntersectTris(r, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

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
    using namespace math;

    vec3 _o = make_mat4(xform) * vec4(make_vec3(r.o), 1);
    vec3 _d = make_mat4(xform) * vec4(make_vec3(r.d), 0);

    ray_packet_t _r = r;
    memcpy(&_r.o[0], value_ptr(_o), 3 * sizeof(float));
    memcpy(&_r.d[0], value_ptr(_d), 3 * sizeof(float));

    return _r;
}

math::vec3 ray::ref::TransformNormal(const math::vec3 &n, const float *inv_xform) {
    using namespace math;

    return vec3{ dot(make_vec3(&inv_xform[0]), n),
                 dot(make_vec3(&inv_xform[4]), n),
                 dot(make_vec3(&inv_xform[8]), n) };
}

void ray::ref::TransformUVs(const float _uvs[2], const float tex_atlas_size[2], const texture_t *t, int mip_level, float out_uvs[2]) {
    using namespace math;
    
    vec2 pos = { (float)t->pos[mip_level][0], (float)t->pos[mip_level][1] };
    vec2 size = { (float)(t->size[0] >> mip_level), (float)(t->size[1] >> mip_level) };
    vec2 uvs = make_vec2(_uvs);
    uvs = uvs - floor(uvs);
    vec2 res = pos + uvs * size + vec2{ 1.0f, 1.0f };
    res /= make_vec2(tex_atlas_size);
    
    out_uvs[0] = res.x;
    out_uvs[1] = res.y;
}

ray::pixel_color_t ray::ref::ShadeSurface(const int index, const int iteration, const float *halton, const hit_data_t &inter, const ray_packet_t &ray,
                                          const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                          const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                          const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                          const material_t *materials, const texture_t *textures, const TextureAtlas &tex_atlas, ray_packet_t *out_secondary_rays, int *out_secondary_rays_count) {
    using namespace math;

    if (!inter.mask_values[0]) {
        return ray::pixel_color_t{ ray.c[0] * env.sky_col[0], ray.c[1] * env.sky_col[1], ray.c[2] * env.sky_col[2], 1.0f };
    }

    const auto I = make_vec3(ray.d);
    const auto P = make_vec3(ray.o) + inter.t * I;

    const auto &tri = tris[inter.prim_indices[0]];

    const auto *mat = &materials[tri.mi];

    const auto &v1 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 0]];
    const auto &v2 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 1]];
    const auto &v3 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 2]];

    const vec3 n1 = make_vec3(v1.n);
    const vec3 n2 = make_vec3(v2.n);
    const vec3 n3 = make_vec3(v3.n);

    const vec2 u1 = make_vec2(v1.t0);
    const vec2 u2 = make_vec2(v2.t0);
    const vec2 u3 = make_vec2(v3.t0);

    float w = 1.0f - inter.u - inter.v;
    vec3 N = n1 * w + n2 * inter.u + n3 * inter.v;
    vec2 uvs = u1 * w + u2 * inter.u + u3 * inter.v;

    //////////////////////////////////////////

    const vec3 p1 = make_vec3(v1.p);
    const vec3 p2 = make_vec3(v2.p);
    const vec3 p3 = make_vec3(v3.p);

    // From 'Tracing Ray Differentials' [1999]

    float dt_dx = -dot(make_vec3(ray.do_dx) + inter.t * make_vec3(ray.dd_dx), N) / dot(I, N);
    float dt_dy = -dot(make_vec3(ray.do_dy) + inter.t * make_vec3(ray.dd_dy), N) / dot(I, N);

    const vec3 do_dx = (make_vec3(ray.do_dx) + inter.t * make_vec3(ray.dd_dx)) + dt_dx * I;
    const vec3 do_dy = (make_vec3(ray.do_dy) + inter.t * make_vec3(ray.dd_dy)) + dt_dy * I;
    const vec3 dd_dx = make_vec3(ray.dd_dx);
    const vec3 dd_dy = make_vec3(ray.dd_dy);

    //////////////////////////////////////////

    // From 'Physically Based Rendering: ...' book

    const vec2 duv13 = u1 - u3, duv23 = u2 - u3;
    const vec3 dp13 = p1 - p3, dp23 = p2 - p3;

    const float det_uv = duv13.x * duv23.y - duv13.y * duv23.x;
    const float inv_det_uv = abs(det_uv) < FLT_EPS ? 0 : 1.0f / det_uv;
    const vec3 dpdu = (duv23.y * dp13 - duv13.y * dp23) * inv_det_uv;
    const vec3 dpdv = (-duv23.x * dp13 + duv13.x * dp23) * inv_det_uv;

    vec2 A[2] = { { dpdu.x, dpdu.y }, { dpdv.x, dpdv.y } };
    vec2 Bx = { do_dx.x, do_dx.y };
    vec2 By = { do_dy.x, do_dy.y };

    if (abs(N.x) > abs(N.y) && abs(N.x) > abs(N.z)) {
        A[0] = { dpdu.y, dpdu.z };
        A[1] = { dpdv.y, dpdv.z };
        Bx = { do_dx.y, do_dx.z };
        By = { do_dy.y, do_dy.z };
    } else if (abs(N.y) > abs(N.z)) {
        A[0] = { dpdu.x, dpdu.z };
        A[1] = { dpdv.x, dpdv.z };
        Bx = { do_dx.x, do_dx.z };
        By = { do_dy.x, do_dy.z };
    }

    const float det = A[0].x * A[1].y - A[1].x * A[0].y;
    const float inv_det = abs(det) < FLT_EPS ? 0 : 1.0f / det;
    const vec2 duv_dx = vec2{ A[0].x * Bx.x - A[0].y * Bx.y, A[1].x * Bx.x - A[1].y * Bx.y } * inv_det;
    const vec2 duv_dy = vec2{ A[0].x * By.x - A[0].y * By.y, A[1].x * By.x - A[1].y * By.y } * inv_det;

    ////////////////////////////////////////////////////////

    const int hi = (hash(index) + iteration) & (HaltonSeqLen - 1);

    // resolve mix material
    while (mat->type == MixMaterial) {
        const auto mix = tex_atlas.SampleAnisotropic(textures[mat->textures[MAIN_TEXTURE]], uvs, duv_dx, duv_dy);
        const float r = halton[hi * 2];

        // shlick fresnel
        float RR = mat->fresnel + (1.0f - mat->fresnel) * pow(1.0f + dot(I, N), 5.0f);
        RR = clamp(RR, 0.0f, 1.0f);

        mat = (r * RR < mix.x) ? &materials[mat->textures[MIX_MAT1]] : &materials[mat->textures[MIX_MAT2]];
    }

    ////////////////////////////////////////////////////////

    // Derivative for normal

    const vec3 dn1 = n1 - n3, dn2 = n2 - n3;
    const vec3 dndu = (duv23.y * dn1 - duv13.y * dn2) * inv_det_uv;
    const vec3 dndv = (-duv23.x * dn1 + duv13.x * dn2) * inv_det_uv;

    const vec3 dndx = dndu * duv_dx.x + dndv * duv_dx.y;
    const vec3 dndy = dndu * duv_dy.x + dndv * duv_dy.y;

    const float ddn_dx = dot(dd_dx, N) + dot(I, dndx);
    const float ddn_dy = dot(dd_dy, N) + dot(I, dndy);

    ////////////////////////////////////////////////////////

    const vec3 b1 = make_vec3(v1.b);
    const vec3 b2 = make_vec3(v2.b);
    const vec3 b3 = make_vec3(v3.b);

    vec3 B = b1 * w + b2 * inter.u + b3 * inter.v;
    vec3 T = cross(B, N);

    auto normals = tex_atlas.SampleAnisotropic(textures[mat->textures[NORMALS_TEXTURE]], uvs, duv_dx, duv_dy);

    normals = normals * 2.0f - 1.0f;

    N = normals.x * B + normals.z * N + normals.y * T;

    //////////////////////////////////////////

    const auto *tr = &transforms[mesh_instances[inter.obj_indices[0]].tr_index];

    N = TransformNormal(N, tr->inv_xform);
    B = TransformNormal(B, tr->inv_xform);
    T = TransformNormal(T, tr->inv_xform);

    //////////////////////////////////////////

    auto albedo = tex_atlas.SampleAnisotropic(textures[mat->textures[MAIN_TEXTURE]], uvs, duv_dx, duv_dy);
    albedo.x *= mat->main_color[0];
    albedo.y *= mat->main_color[1];
    albedo.z *= mat->main_color[2];
    albedo = pow(albedo, vec4(2.2f));

    vec3 col;

    // generate secondary ray
    if (mat->type == DiffuseMaterial) {
        float k = dot(N, make_vec3(env.sun_dir));

        float v = 1;
        if (k > 0) {
            const float z = 1.0f - halton[hi * 2] * env.sun_softness;
            const float temp = sqrt(1.0f - z * z);

            const float phi = halton[hi * 2 + 1] * 2 * pi<float>();
            const float cos_phi = math::cos(phi);
            const float sin_phi = math::sin(phi);

            vec3 TT = cross(make_vec3(env.sun_dir), B);
            vec3 BB = cross(make_vec3(env.sun_dir), TT);
            vec3 V = temp * sin_phi * BB + z * make_vec3(env.sun_dir) + temp * cos_phi * TT;

            ray_packet_t r;

            memcpy(&r.o[0], value_ptr(P + HIT_BIAS * N), 3 * sizeof(float));
            memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));

            hit_data_t inter;
            Traverse_MacroTree_CPU(r, nodes, node_index, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, inter);
            if (inter.mask_values[0] == 0xffffffff) {
                v = 0;
            }
        }

        k = clamp(k, 0.0f, 1.0f);

        col = vec3(albedo) * make_vec3(env.sun_col) * v * k;

        const float z = halton[hi * 2];
        const float temp = sqrt(1.0f - z * z);

        const float phi = halton[((hash(hi) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * pi<float>();
        const float cos_phi = math::cos(phi);
        const float sin_phi = math::sin(phi);

        const vec3 V = temp * sin_phi * B + z * N + temp * cos_phi * T;

        ray_packet_t r;

        r.id = ray.id;

        memcpy(&r.o[0], value_ptr(P + HIT_BIAS * N), 3 * sizeof(float));
        memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));
        memcpy(&r.c[0], value_ptr(make_vec3(ray.c) * z * vec3(albedo)), 3 * sizeof(float));
        
        memcpy(&r.do_dx[0], value_ptr(do_dx), 3 * sizeof(float));
        memcpy(&r.do_dy[0], value_ptr(do_dy), 3 * sizeof(float));

        memcpy(&r.dd_dx[0], value_ptr(dd_dx - 2 * (dot(I, N) * dndx + ddn_dx * N)), 3 * sizeof(float));
        memcpy(&r.dd_dy[0], value_ptr(dd_dy - 2 * (dot(I, N) * dndy + ddn_dy * N)), 3 * sizeof(float));

        if ((r.c[0] * r.c[0] + r.c[1] * r.c[1] + r.c[2] * r.c[2]) > 0.005f) {
            const int index = (*out_secondary_rays_count)++;
            out_secondary_rays[index] = r;
        }
    } else {
        return pixel_color_t{ 0.0f, 1.0f, 1.0f, 1.0f };
    }

    return pixel_color_t{ ray.c[0] * col.r, ray.c[1] * col.g, ray.c[2] * col.b, 1.0f };
}