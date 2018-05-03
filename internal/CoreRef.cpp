#include "CoreRef.h"

#include <algorithm>
#include <limits>

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

force_inline float clamp(float val, float min, float max) {
    return val < min ? min : (val > max ? max : val);
}

force_inline simd_fvec3 cross(const simd_fvec3 &v1, const simd_fvec3 &v2) {
    return simd_fvec3{ v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0] };
}

force_inline simd_fvec3 reflect(const simd_fvec3 &I, const simd_fvec3 &N) {
    return I - 2 * dot(N, I) * N;
}

}
}

ray::ref::hit_data_t::hit_data_t() {
    mask_values[0] = 0;
    obj_indices[0] = -1;
    prim_indices[0] = -1;
    t = std::numeric_limits<float>::max();
}

void ray::ref::GeneratePrimaryRays(int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t> &out_rays) {
    simd_fvec3 origin = { cam.origin }, fwd = { cam.fwd }, side = { cam.side }, up = { cam.up };

    up *= float(h) / w;

    auto get_pix_dir = [fwd, side, up, w, h](const float x, const float y) {
        simd_fvec3 _d(float(x) / w - 0.5f, float(-y) / h + 0.5f, 1);
        _d = _d[0] * side + _d[1] * up + _d[2] * fwd;
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

            simd_fvec3 _d = get_pix_dir(_x, _y);

            simd_fvec3 _dx = get_pix_dir(_x + 1, _y),
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
    ray_packet_t _r = r;

    _r.o[0] = xform[0] * r.o[0] + xform[4] * r.o[1] + xform[8] * r.o[2] + xform[12];
    _r.o[1] = xform[1] * r.o[0] + xform[5] * r.o[1] + xform[9] * r.o[2] + xform[13];
    _r.o[2] = xform[2] * r.o[0] + xform[6] * r.o[1] + xform[10] * r.o[2] + xform[14];

    _r.d[0] = xform[0] * r.d[0] + xform[4] * r.d[1] + xform[8] * r.d[2];
    _r.d[1] = xform[1] * r.d[0] + xform[5] * r.d[1] + xform[9] * r.d[2];
    _r.d[2] = xform[2] * r.d[0] + xform[6] * r.d[1] + xform[10] * r.d[2];

    return _r;
}

ray::ref::simd_fvec3 ray::ref::TransformNormal(const simd_fvec3 &n, const float *inv_xform) {
    return simd_fvec3{ inv_xform[0] * n[0] + inv_xform[1] * n[1] + inv_xform[2] * n[2],
                       inv_xform[4] * n[0] + inv_xform[5] * n[1] + inv_xform[6] * n[2],
                       inv_xform[8] * n[0] + inv_xform[9] * n[1] + inv_xform[10] * n[2] };
}

ray::ref::simd_fvec2 ray::ref::TransformUV(const simd_fvec2 &_uv, const simd_fvec2 &tex_atlas_size, const texture_t &t, int mip_level) {
    simd_fvec2 pos = { (float)t.pos[mip_level][0], (float)t.pos[mip_level][1] };
    simd_fvec2 size = { (float)(t.size[0] >> mip_level), (float)(t.size[1] >> mip_level) };
    simd_fvec2 uv = _uv - floor(_uv);
    simd_fvec2 res = pos + uv * size + 1.0f;
    res /= tex_atlas_size;
    return res;
}

ray::ref::simd_fvec4 ray::ref::SampleNearest(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod) {
    int _lod = (int)lod;

    simd_fvec2 atlas_size = { atlas.size_x(), atlas.size_y() };
    simd_fvec2 _uvs = TransformUV(uvs, atlas_size, t, _lod);

    if (_lod > MAX_MIP_LEVEL) _lod = MAX_MIP_LEVEL;

    int page = t.page[_lod];

    const auto &pix = atlas.Get(page, _uvs[0], _uvs[1]);

    const float k = 1.0f / 255.0f;
    return simd_fvec4{ pix.r * k, pix.g * k, pix.b * k, pix.a * k };
}

ray::ref::simd_fvec4 ray::ref::SampleBilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, int lod) {
    simd_fvec2 atlas_size = { atlas.size_x(), atlas.size_y() };
    simd_fvec2 _uvs = TransformUV(uvs, atlas_size, t, lod);

    int page = t.page[lod];

    _uvs = _uvs * atlas_size - 0.5f;

    const auto &p00 = atlas.Get(page, int(_uvs[0]), int(_uvs[1]));
    const auto &p01 = atlas.Get(page, int(_uvs[0] + 1), int(_uvs[1]));
    const auto &p10 = atlas.Get(page, int(_uvs[0]), int(_uvs[1] + 1));
    const auto &p11 = atlas.Get(page, int(_uvs[0] + 1), int(_uvs[1] + 1));

    float kx = _uvs[0] - std::floor(_uvs[0]), ky = _uvs[1] - std::floor(_uvs[1]);

    const auto p0 = simd_fvec4{ p01.r * kx + p00.r * (1 - kx),
                                p01.g * kx + p00.g * (1 - kx),
                                p01.b * kx + p00.b * (1 - kx),
                                p01.a * kx + p00.a * (1 - kx) };

    const auto p1 = simd_fvec4{ p11.r * kx + p10.r * (1 - kx),
                                p11.g * kx + p10.g * (1 - kx),
                                p11.b * kx + p10.b * (1 - kx),
                                p11.a * kx + p10.a * (1 - kx) };

    const float k = 1.0f / 255.0f;
    return (p1 * ky + p0 * (1 - ky)) * k;
}

static float uint8_to_float_table[] = {
    0.000000000f, 0.003921569f, 0.007843138f, 0.011764706f, 0.015686275f, 0.019607844f, 0.023529412f, 0.027450981f, 0.031372551f, 0.035294119f, 0.039215688f, 0.043137256f, 0.047058824f, 0.050980393f, 0.054901961f, 0.058823530f,
    0.062745102f, 0.066666670f, 0.070588239f, 0.074509807f, 0.078431375f, 0.082352944f, 0.086274512f, 0.090196081f, 0.094117649f, 0.098039217f, 0.101960786f, 0.105882354f, 0.109803922f, 0.113725491f, 0.117647059f, 0.121568628f,
    0.125490203f, 0.129411772f, 0.133333340f, 0.137254909f, 0.141176477f, 0.145098045f, 0.149019614f, 0.152941182f, 0.156862751f, 0.160784319f, 0.164705887f, 0.168627456f, 0.172549024f, 0.176470593f, 0.180392161f, 0.184313729f,
    0.188235298f, 0.192156866f, 0.196078435f, 0.200000003f, 0.203921571f, 0.207843140f, 0.211764708f, 0.215686277f, 0.219607845f, 0.223529413f, 0.227450982f, 0.231372550f, 0.235294119f, 0.239215687f, 0.243137255f, 0.247058824f,
    0.250980407f, 0.254901975f, 0.258823544f, 0.262745112f, 0.266666681f, 0.270588249f, 0.274509817f, 0.278431386f, 0.282352954f, 0.286274523f, 0.290196091f, 0.294117659f, 0.298039228f, 0.301960796f, 0.305882365f, 0.309803933f,
    0.313725501f, 0.317647070f, 0.321568638f, 0.325490206f, 0.329411775f, 0.333333343f, 0.337254912f, 0.341176480f, 0.345098048f, 0.349019617f, 0.352941185f, 0.356862754f, 0.360784322f, 0.364705890f, 0.368627459f, 0.372549027f,
    0.376470596f, 0.380392164f, 0.384313732f, 0.388235301f, 0.392156869f, 0.396078438f, 0.400000006f, 0.403921574f, 0.407843143f, 0.411764711f, 0.415686280f, 0.419607848f, 0.423529416f, 0.427450985f, 0.431372553f, 0.435294122f,
    0.439215690f, 0.443137258f, 0.447058827f, 0.450980395f, 0.454901963f, 0.458823532f, 0.462745100f, 0.466666669f, 0.470588237f, 0.474509805f, 0.478431374f, 0.482352942f, 0.486274511f, 0.490196079f, 0.494117647f, 0.498039216f,
    0.501960814f, 0.505882382f, 0.509803951f, 0.513725519f, 0.517647088f, 0.521568656f, 0.525490224f, 0.529411793f, 0.533333361f, 0.537254930f, 0.541176498f, 0.545098066f, 0.549019635f, 0.552941203f, 0.556862772f, 0.560784340f,
    0.564705908f, 0.568627477f, 0.572549045f, 0.576470613f, 0.580392182f, 0.584313750f, 0.588235319f, 0.592156887f, 0.596078455f, 0.600000024f, 0.603921592f, 0.607843161f, 0.611764729f, 0.615686297f, 0.619607866f, 0.623529434f,
    0.627451003f, 0.631372571f, 0.635294139f, 0.639215708f, 0.643137276f, 0.647058845f, 0.650980413f, 0.654901981f, 0.658823550f, 0.662745118f, 0.666666687f, 0.670588255f, 0.674509823f, 0.678431392f, 0.682352960f, 0.686274529f,
    0.690196097f, 0.694117665f, 0.698039234f, 0.701960802f, 0.705882370f, 0.709803939f, 0.713725507f, 0.717647076f, 0.721568644f, 0.725490212f, 0.729411781f, 0.733333349f, 0.737254918f, 0.741176486f, 0.745098054f, 0.749019623f,
    0.752941191f, 0.756862760f, 0.760784328f, 0.764705896f, 0.768627465f, 0.772549033f, 0.776470602f, 0.780392170f, 0.784313738f, 0.788235307f, 0.792156875f, 0.796078444f, 0.800000012f, 0.803921580f, 0.807843149f, 0.811764717f,
    0.815686285f, 0.819607854f, 0.823529422f, 0.827450991f, 0.831372559f, 0.835294127f, 0.839215696f, 0.843137264f, 0.847058833f, 0.850980401f, 0.854901969f, 0.858823538f, 0.862745106f, 0.866666675f, 0.870588243f, 0.874509811f,
    0.878431380f, 0.882352948f, 0.886274517f, 0.890196085f, 0.894117653f, 0.898039222f, 0.901960790f, 0.905882359f, 0.909803927f, 0.913725495f, 0.917647064f, 0.921568632f, 0.925490201f, 0.929411769f, 0.933333337f, 0.937254906f,
    0.941176474f, 0.945098042f, 0.949019611f, 0.952941179f, 0.956862748f, 0.960784316f, 0.964705884f, 0.968627453f, 0.972549021f, 0.976470590f, 0.980392158f, 0.984313726f, 0.988235295f, 0.992156863f, 0.996078432f, 1.000000000f,
};

force_inline float to_norm_float(uint8_t v) {
    return uint8_to_float_table[v];
}

ray::ref::simd_fvec4 ray::ref::SampleBilinear(const TextureAtlas &atlas, const simd_fvec2 &uvs, int page) {
    const auto &p00 = atlas.Get(page, int(uvs[0] + 0), int(uvs[1] + 0));
    const auto &p01 = atlas.Get(page, int(uvs[0] + 1), int(uvs[1] + 0));
    const auto &p10 = atlas.Get(page, int(uvs[0] + 0), int(uvs[1] + 1));
    const auto &p11 = atlas.Get(page, int(uvs[0] + 1), int(uvs[1] + 1));

    simd_fvec2 k = uvs - floor(uvs);
    
    const auto _p00 = simd_fvec4{ to_norm_float(p00.r), to_norm_float(p00.g), to_norm_float(p00.b), to_norm_float(p00.a) };
    const auto _p01 = simd_fvec4{ to_norm_float(p01.r), to_norm_float(p01.g), to_norm_float(p01.b), to_norm_float(p01.a) };
    const auto _p10 = simd_fvec4{ to_norm_float(p10.r), to_norm_float(p10.g), to_norm_float(p10.b), to_norm_float(p10.a) };
    const auto _p11 = simd_fvec4{ to_norm_float(p11.r), to_norm_float(p11.g), to_norm_float(p11.b), to_norm_float(p11.a) };

    const auto p0X = _p01 * k[0] + _p00 * (1 - k[0]);
    const auto p1X = _p11 * k[0] + _p10 * (1 - k[0]);

    return (p1X * k[1] + p0X * (1 - k[1]));
}

ray::ref::simd_fvec4 ray::ref::SampleTrilinear(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, float lod) {
    auto col1 = SampleBilinear(atlas, t, uvs, (int)std::floor(lod));
    auto col2 = SampleBilinear(atlas, t, uvs, (int)std::ceil(lod));

    const float k = lod - std::floor(lod);
    return col1 * (1 - k) + col2 * k;
}

ray::ref::simd_fvec4 ray::ref::SampleAnisotropic(const TextureAtlas &atlas, const texture_t &t, const simd_fvec2 &uvs, const simd_fvec2 &duv_dx, const simd_fvec2 &duv_dy) {
    simd_fvec2 sz = { (float)t.size[0], (float)t.size[1] };

    float l1 = length(duv_dx * sz);
    float l2 = length(duv_dy * sz);

    float lod, k;
    simd_fvec2 step;

    if (l1 <= l2) {
        lod = log2(l1);
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = log2(l2);
        k = l2 / l1;
        step = duv_dx;
    }

    if (lod < 0.0f) lod = 0.0f;
    else if (lod >(float)MAX_MIP_LEVEL) lod = (float)MAX_MIP_LEVEL;

    simd_fvec2 _uvs = uvs - step * 0.5f;

    int num = (int)(2.0f / k);
    if (num < 1) num = 1;
    else if (num > 32) num = 32;

    step = step / float(num);

    auto res = simd_fvec4{ 0.0f };

    int lod1 = (int)std::floor(lod);
    int lod2 = (int)std::ceil(lod);

    int page1 = t.page[lod1];
    int page2 = t.page[lod2];

    simd_fvec2 pos1 = simd_fvec2{ (float)t.pos[lod1][0], (float)t.pos[lod1][1] } + 0.5f;
    simd_fvec2 size1 = { (float)(t.size[0] >> lod1), (float)(t.size[1] >> lod1) };

    simd_fvec2 pos2 = simd_fvec2{ (float)t.pos[lod2][0], (float)t.pos[lod2][1] } + 0.5f;
    simd_fvec2 size2 = { (float)(t.size[0] >> lod2), (float)(t.size[1] >> lod2) };

    const float kz = lod - std::floor(lod);

    for (int i = 0; i < num; i++) {
        _uvs = _uvs - floor(_uvs);

        simd_fvec2 _uvs1 = pos1 + _uvs * size1;
        res += (1 - kz) * SampleBilinear(atlas, _uvs1, page1);

        if (kz > 0.0001f) {
            simd_fvec2 _uvs2 = pos2 + _uvs * size2;
            res += kz * SampleBilinear(atlas, _uvs2, page2);
        }

        _uvs = _uvs + step;
    }

    return res / float(num);
}

ray::pixel_color_t ray::ref::ShadeSurface(const int index, const int iteration, const float *halton, const hit_data_t &inter, const ray_packet_t &ray,
                                          const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                          const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                          const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                          const material_t *materials, const texture_t *textures, const TextureAtlas &tex_atlas, ray_packet_t *out_secondary_rays, int *out_secondary_rays_count) {
    if (!inter.mask_values[0]) {
        return ray::pixel_color_t{ ray.c[0] * env.sky_col[0], ray.c[1] * env.sky_col[1], ray.c[2] * env.sky_col[2], 1.0f };
    }

    const auto I = simd_fvec3(ray.d);
    const auto P = simd_fvec3(ray.o) + inter.t * I;

    const auto &tri = tris[inter.prim_indices[0]];

    const auto *mat = &materials[tri.mi];

    const auto &v1 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 0]];
    const auto &v2 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 1]];
    const auto &v3 = vertices[vtx_indices[inter.prim_indices[0] * 3 + 2]];

    const auto n1 = simd_fvec3(v1.n);
    const auto n2 = simd_fvec3(v2.n);
    const auto n3 = simd_fvec3(v3.n);

    const auto u1 = simd_fvec2(v1.t0);
    const auto u2 = simd_fvec2(v2.t0);
    const auto u3 = simd_fvec2(v3.t0);

    float w = 1.0f - inter.u - inter.v;
    simd_fvec3 N = n1 * w + n2 * inter.u + n3 * inter.v;
    simd_fvec2 uvs = u1 * w + u2 * inter.u + u3 * inter.v;

    //////////////////////////////////////////

    const auto p1 = simd_fvec3(v1.p);
    const auto p2 = simd_fvec3(v2.p);
    const auto p3 = simd_fvec3(v3.p);

    // From 'Tracing Ray Differentials' [1999]

    float dot_I_N = dot(I, N);
    float inv_dot = std::abs(dot_I_N) < FLT_EPS ? 0.0f : 1.0f/dot_I_N;
    float dt_dx = -dot(simd_fvec3(ray.do_dx) + inter.t * simd_fvec3(ray.dd_dx), N) * inv_dot;
    float dt_dy = -dot(simd_fvec3(ray.do_dy) + inter.t * simd_fvec3(ray.dd_dy), N) * inv_dot;

    const auto do_dx = (simd_fvec3(ray.do_dx) + inter.t * simd_fvec3(ray.dd_dx)) + dt_dx * I;
    const auto do_dy = (simd_fvec3(ray.do_dy) + inter.t * simd_fvec3(ray.dd_dy)) + dt_dy * I;
    const auto dd_dx = simd_fvec3(ray.dd_dx);
    const auto dd_dy = simd_fvec3(ray.dd_dy);

    //////////////////////////////////////////

    // From 'Physically Based Rendering: ...' book

    const simd_fvec2 duv13 = u1 - u3, duv23 = u2 - u3;
    const simd_fvec3 dp13 = p1 - p3, dp23 = p2 - p3;

    const float det_uv = duv13[0] * duv23[1] - duv13[1] * duv23[0];
    const float inv_det_uv = std::abs(det_uv) < FLT_EPS ? 0 : 1.0f / det_uv;
    const simd_fvec3 dpdu = (duv23[1] * dp13 - duv13[1] * dp23) * inv_det_uv;
    const simd_fvec3 dpdv = (-duv23[0] * dp13 + duv13[0] * dp23) * inv_det_uv;

    simd_fvec2 A[2] = { { dpdu[0], dpdu[1] }, { dpdv[0], dpdv[1] } };
    simd_fvec2 Bx = { do_dx[0], do_dx[1] };
    simd_fvec2 By = { do_dy[0], do_dy[1] };

    if (std::abs(N[0]) > std::abs(N[1]) && std::abs(N[0]) > std::abs(N[2])) {
        A[0] = { dpdu[1], dpdu[2] };
        A[1] = { dpdv[1], dpdv[2] };
        Bx = { do_dx[1], do_dx[2] };
        By = { do_dy[1], do_dy[2] };
    } else if (std::abs(N[1]) > std::abs(N[2])) {
        A[0] = { dpdu[0], dpdu[2] };
        A[1] = { dpdv[0], dpdv[2] };
        Bx = { do_dx[0], do_dx[2] };
        By = { do_dy[0], do_dy[2] };
    }

    const float det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
    const float inv_det = std::abs(det) < FLT_EPS ? 0 : 1.0f / det;
    const auto duv_dx = simd_fvec2{ A[0][0] * Bx[0] - A[0][1] * Bx[1], A[1][0] * Bx[0] - A[1][1] * Bx[1] } * inv_det;
    const auto duv_dy = simd_fvec2{ A[0][0] * By[0] - A[0][1] * By[1], A[1][0] * By[0] - A[1][1] * By[1] } * inv_det;

    ////////////////////////////////////////////////////////

    const int hi = (hash(index) + iteration) & (HaltonSeqLen - 1);

    // resolve mix material
    while (mat->type == MixMaterial) {
        const auto mix = SampleAnisotropic(tex_atlas, textures[mat->textures[MAIN_TEXTURE]], uvs, duv_dx, duv_dy);
        const float r = halton[hi * 2];

        // shlick fresnel
        float RR = mat->fresnel + (1.0f - mat->fresnel) * std::pow(1.0f + dot(I, N), 5.0f);
        RR = clamp(RR, 0.0f, 1.0f);

        mat = (r * RR < mix[0]) ? &materials[mat->textures[MIX_MAT1]] : &materials[mat->textures[MIX_MAT2]];
    }

    ////////////////////////////////////////////////////////

    // Derivative for normal

    const auto dn1 = n1 - n3, dn2 = n2 - n3;
    const auto dndu = (duv23[1] * dn1 - duv13[1] * dn2) * inv_det_uv;
    const auto dndv = (-duv23[0] * dn1 + duv13[0] * dn2) * inv_det_uv;

    const auto dndx = dndu * duv_dx[0] + dndv * duv_dx[1];
    const auto dndy = dndu * duv_dy[0] + dndv * duv_dy[1];

    const float ddn_dx = dot(dd_dx, N) + dot(I, dndx);
    const float ddn_dy = dot(dd_dy, N) + dot(I, dndy);

    ////////////////////////////////////////////////////////

    const auto b1 = simd_fvec3(v1.b);
    const auto b2 = simd_fvec3(v2.b);
    const auto b3 = simd_fvec3(v3.b);

    simd_fvec3 B = b1 * w + b2 * inter.u + b3 * inter.v;
    simd_fvec3 T = cross(B, N);

    auto normals = SampleAnisotropic(tex_atlas, textures[mat->textures[NORMALS_TEXTURE]], uvs, duv_dx, duv_dy);

    normals = normals * 2.0f - 1.0f;

    N = normals[0] * B + normals[2] * N + normals[1] * T;

    //////////////////////////////////////////

    const auto *tr = &transforms[mesh_instances[inter.obj_indices[0]].tr_index];

    N = TransformNormal(N, tr->inv_xform);
    B = TransformNormal(B, tr->inv_xform);
    T = TransformNormal(T, tr->inv_xform);

    //////////////////////////////////////////

    auto albedo = SampleAnisotropic(tex_atlas, textures[mat->textures[MAIN_TEXTURE]], uvs, duv_dx, duv_dy);
    albedo[0] *= mat->main_color[0];
    albedo[1] *= mat->main_color[1];
    albedo[2] *= mat->main_color[2];
    albedo = pow(albedo, simd_fvec4(2.2f));

    simd_fvec3 col;

    // generate secondary ray
    if (mat->type == DiffuseMaterial) {
        float k = dot(N, simd_fvec3(env.sun_dir));

        float v = 1;
        if (k > 0) {
            const float z = 1.0f - halton[hi * 2] * env.sun_softness;
            const float temp = std::sqrt(1.0f - z * z);

            const float phi = halton[hi * 2 + 1] * 2 * PI;
            const float cos_phi = std::cos(phi);
            const float sin_phi = std::sin(phi);

            auto TT = cross(simd_fvec3(env.sun_dir), B);
            auto BB = cross(simd_fvec3(env.sun_dir), TT);
            auto V = temp * sin_phi * BB + z * simd_fvec3(env.sun_dir) + temp * cos_phi * TT;

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

        col = simd_fvec3(&albedo[0]) * simd_fvec3(env.sun_col) * v * k;

        const float z = halton[hi * 2];
        const float temp = std::sqrt(1.0f - z * z);

        const float phi = halton[((hash(hi) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * PI;
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);

        const auto V = temp * sin_phi * B + z * N + temp * cos_phi * T;

        ray_packet_t r;

        r.id = ray.id;

        memcpy(&r.o[0], value_ptr(P + HIT_BIAS * N), 3 * sizeof(float));
        memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));
        memcpy(&r.c[0], value_ptr(simd_fvec3(ray.c) * z * simd_fvec3(&albedo[0])), 3 * sizeof(float));

        memcpy(&r.do_dx[0], value_ptr(do_dx), 3 * sizeof(float));
        memcpy(&r.do_dy[0], value_ptr(do_dy), 3 * sizeof(float));

        memcpy(&r.dd_dx[0], value_ptr(dd_dx - 2 * (dot(I, N) * dndx + ddn_dx * N)), 3 * sizeof(float));
        memcpy(&r.dd_dy[0], value_ptr(dd_dy - 2 * (dot(I, N) * dndy + ddn_dy * N)), 3 * sizeof(float));

        if ((r.c[0] * r.c[0] + r.c[1] * r.c[1] + r.c[2] * r.c[2]) > 0.005f) {
            const int index = (*out_secondary_rays_count)++;
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == GlossyMaterial) {
        simd_fvec3 V = reflect(I, dot(I, N) > 0 ? N : -N);

        const float z = 1.0f - halton[hi * 2] * mat->roughness;
        const float temp = std::sqrt(1.0f - z * z);

        const float phi = halton[((hash(hi) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * PI;
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);

        auto TT = cross(V, B);
        auto BB = cross(V, TT);
        V = temp * sin_phi * BB + z * V + temp * cos_phi * TT;

        ray_packet_t r;

        r.id = ray.id;

        memcpy(&r.o[0], value_ptr(P + HIT_BIAS * N), 3 * sizeof(float));
        memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));
        memcpy(&r.c[0], value_ptr(simd_fvec3(ray.c) * z), 3 * sizeof(float));

        memcpy(&r.do_dx[0], value_ptr(do_dx), 3 * sizeof(float));
        memcpy(&r.do_dy[0], value_ptr(do_dy), 3 * sizeof(float));

        memcpy(&r.dd_dx[0], value_ptr(dd_dx - 2 * (dot(I, N) * dndx + ddn_dx * N)), 3 * sizeof(float));
        memcpy(&r.dd_dy[0], value_ptr(dd_dy - 2 * (dot(I, N) * dndy + ddn_dy * N)), 3 * sizeof(float));

        if ((r.c[0] * r.c[0] + r.c[1] * r.c[1] + r.c[2] * r.c[2]) > 0.005f) {
            const int index = (*out_secondary_rays_count)++;
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == RefractiveMaterial) {
        const auto _N = dot(I, N) > 0 ? -N : N;

        float eta = 1.0f / mat->ior;
        float cosi = dot(-I, _N);
        float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
        float m = eta * cosi - std::sqrt(std::abs(cost2));
        auto V = eta * I + m * _N;
        if (cost2 < 0) V = 0.0f;

        // ** REFACTOR THIS **

        const float z = 1.0f - halton[hi * 2] * mat->roughness;
        const float temp = std::sqrt(1.0f - z * z);

        const float phi = halton[((hash(hi) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * PI;
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);

        auto TT = normalize(cross(V, B));
        auto BB = normalize(cross(V, TT));
        V = temp * sin_phi * BB + z * V + temp * cos_phi * TT;

        //////////////////

        float k = (eta - eta * eta * dot(I, N) / dot(V, N));
        float dmdx = k * ddn_dx;
        float dmdy = k * ddn_dy;

        ray_packet_t r;

        r.id = ray.id;

        memcpy(&r.o[0], value_ptr(P + HIT_BIAS * I), 3 * sizeof(float));
        memcpy(&r.d[0], value_ptr(V), 3 * sizeof(float));
        memcpy(&r.c[0], value_ptr(simd_fvec3(ray.c) * z), 3 * sizeof(float));

        memcpy(&r.do_dx[0], value_ptr(do_dx), 3 * sizeof(float));
        memcpy(&r.do_dy[0], value_ptr(do_dy), 3 * sizeof(float));

        memcpy(&r.dd_dx[0], value_ptr(eta * dd_dx - (m * dndx + dmdx * N)), 3 * sizeof(float));
        memcpy(&r.dd_dy[0], value_ptr(eta * dd_dy - (m * dndy + dmdy * N)), 3 * sizeof(float));

        if ((r.c[0] * r.c[0] + r.c[1] * r.c[1] + r.c[2] * r.c[2]) > 0.005f) {
            const int index = (*out_secondary_rays_count)++;
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == EmissiveMaterial) {
        col = mat->strength * simd_fvec3(&albedo[0]);
    } else if (mat->type == TransparentMaterial) {
        ray_packet_t r;

        r.id = ray.id;

        memcpy(&r.o[0], value_ptr(P + HIT_BIAS * I), 3 * sizeof(float));
        memcpy(&r.d[0], &ray.d[0], 3 * sizeof(float));
        memcpy(&r.c[0], &ray.c[0], 3 * sizeof(float));

        memcpy(&r.do_dx[0], &ray.do_dx[0], 3 * sizeof(float));
        memcpy(&r.do_dy[0], &ray.do_dy[0], 3 * sizeof(float));

        memcpy(&r.dd_dx[0], &ray.dd_dx[0], 3 * sizeof(float));
        memcpy(&r.dd_dy[0], &ray.dd_dy[0], 3 * sizeof(float));

        if ((r.c[0] * r.c[0] + r.c[1] * r.c[1] + r.c[2] * r.c[2]) > 0.005f) {
            const int index = (*out_secondary_rays_count)++;
            out_secondary_rays[index] = r;
        }
    }

    return pixel_color_t{ ray.c[0] * col[0], ray.c[1] * col[1], ray.c[2] * col[2], 1.0f };
}