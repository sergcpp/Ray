#include "Core.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <math/math.hpp>
#include <vector>

#include "BVHSplit.h"

namespace ray {
const float axis_aligned_normal_eps = 0.000001f;
}

void ray::PreprocessTri(const float *p, int stride, tri_accel_t *acc) {
    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    if (!stride) stride = 3;

    // edges
    float e0[3] = { p[stride] - p[0], p[stride + 1] - p[1], p[stride + 2] - p[2] },
                  e1[3] = { p[stride * 2] - p[0], p[stride * 2 + 1] - p[1], p[stride * 2 + 2] - p[2] };

    float n[3] = { e0[1] * e1[2] - e0[2] * e1[1],
                   e0[2] * e1[0] - e0[0] * e1[2],
                   e0[0] * e1[1] - e0[1] * e1[0]
                 };

    int w, u, v;
    if (std::abs(n[0]) > std::abs(n[1]) && std::abs(n[0]) > std::abs(n[2])) {
        w = 0;
        u = 1;
        v = 2;
    } else if (std::abs(n[1]) > std::abs(n[0]) && std::abs(n[1]) > std::abs(n[2])) {
        w = 1;
        u = 0;
        v = 2;
    } else {
        w = 2;
        u = 0;
        v = 1;
    }

    acc->nu = n[u] / n[w];
    acc->nv = n[v] / n[w];
    acc->pu = p[u];
    acc->pv = p[v];
    acc->np = acc->nu * acc->pu + acc->nv * acc->pv + p[w];

    int sign = w == 1 ? -1 : 1;
    acc->e0u = sign * e0[u] / n[w];
    acc->e0v = sign * e0[v] / n[w];
    acc->e1u = sign * e1[u] / n[w];
    acc->e1v = sign * e1[v] / n[w];

    acc->ci = w;
    if (std::abs(acc->nu) < axis_aligned_normal_eps && std::abs(acc->nv) < axis_aligned_normal_eps) {
        acc->ci |= AXIS_ALIGNED_BIT;
    }
    assert((acc->ci & W_BITS) == w);
}

void ray::PreprocessCone(const float o[3], const float v[3], float phi, float cone_start, float cone_end, cone_accel_t *acc) {
    for (int i = 0; i < 3; i++) {
        acc->o[i] = o[i];
        acc->v[i] = -v[i];
    }
    float cos_phi = math::cos(phi);
    acc->cos_phi_sqr = cos_phi * cos_phi;
    acc->cone_start = cone_start;
    acc->cone_end = cone_end;
}

void ray::PreprocessBox(const float min[3], const float max[3], aabox_t *box) {
    for (int i = 0; i < 3; i++) {
        box->min[i] = min[i];
        box->max[i] = max[i];
    }
}

uint32_t ray::PreprocessMesh(const float *attrs, size_t attrs_count, const uint32_t *vtx_indices, size_t vtx_indices_count, eVertexLayout layout,
                             std::vector<bvh_node_t> &out_nodes, std::vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_tri_indices) {
    using namespace math;

    assert(vtx_indices_count && vtx_indices_count % 3 == 0);
    assert(layout == PxyzNxyzTuv);

    std::vector<prim_t> primitives;

    size_t tris_start = out_tris.size();
    size_t tris_count = vtx_indices_count / 3;
    out_tris.resize(tris_start + tris_count);

    for (size_t j = 0; j < vtx_indices_count; j += 3) {
        float p[9];

        if (layout == PxyzNxyzTuv) {
            memcpy(&p[0], &attrs[vtx_indices[j] * 8], 3 * sizeof(float));
            memcpy(&p[3], &attrs[vtx_indices[j + 1] * 8], 3 * sizeof(float));
            memcpy(&p[6], &attrs[vtx_indices[j + 2] * 8], 3 * sizeof(float));
        }

        PreprocessTri(&p[0], 0, &out_tris[tris_start + j / 3]);

        vec3 _min = min(make_vec3(&p[0]), min(make_vec3(&p[3]), make_vec3(&p[6]))),
             _max = max(make_vec3(&p[0]), max(make_vec3(&p[3]), make_vec3(&p[6])));

        primitives.push_back({ _min, _max });
    }

    size_t indices_start = out_tri_indices.size();
    uint32_t num_out_nodes = PreprocessPrims(&primitives[0], primitives.size(), out_nodes, out_tri_indices);

    for (size_t i = indices_start; i < out_tri_indices.size(); i++) {
        out_tri_indices[i] += tris_start;
    }

    return num_out_nodes;
}

uint32_t ray::PreprocessPrims(const prim_t *prims, size_t prims_count,
                              std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices) {
    using namespace math;

    struct prims_coll_t {
        std::vector<uint32_t> indices;
        vec3 min = vec3{ std::numeric_limits<float>::max() }, max = vec3{ std::numeric_limits<float>::lowest() };
        prims_coll_t() {}
        prims_coll_t(std::vector<uint32_t> &&_indices, const vec3 &_min, const vec3 &_max)
            : indices(std::move(_indices)), min(_min), max(_max) {
        }
    };

    std::vector<prims_coll_t> triangle_lists;
    triangle_lists.emplace_back();

    size_t num_nodes = out_nodes.size();
    int32_t root_node_index = (int32_t)num_nodes;

    for (size_t j = 0; j < prims_count; j++) {
        triangle_lists.back().indices.push_back(j);
        triangle_lists.back().min = min(triangle_lists.back().min, prims[j].bbox_min);
        triangle_lists.back().max = max(triangle_lists.back().max, prims[j].bbox_max);
    }

    while (!triangle_lists.empty()) {
        auto split_data = SplitPrimitives_SAH(prims, triangle_lists.back().indices, triangle_lists.back().min, triangle_lists.back().max);
        triangle_lists.pop_back();

        uint32_t leaf_index = (uint32_t)out_nodes.size(),
                 parent_index = 0xffffffff, sibling_index = 0;

        for (int32_t i = leaf_index - 1; i >= root_node_index; i--) {
            if (out_nodes[i].left_child == leaf_index) {
                parent_index = i;
                sibling_index = out_nodes[i].right_child;
                break;
            } else if (out_nodes[i].right_child == leaf_index) {
                parent_index = i;
                sibling_index = out_nodes[i].left_child;
                break;
            }
        }

        if (split_data.right_indices.empty()) {
            vec3 bbox_min = split_data.left_bounds[0],
                 bbox_max = split_data.left_bounds[1];

            out_nodes.push_back({ (uint32_t)out_indices.size(), (uint32_t)split_data.left_indices.size(), 0, 0, parent_index, sibling_index, 0,
                {   { bbox_min.x, bbox_min.y, bbox_min.z },
                    { bbox_max.x, bbox_max.y, bbox_max.z }
                }
            });
            out_indices.insert(out_indices.end(), split_data.left_indices.begin(), split_data.left_indices.end());
        } else {
            uint32_t index = (uint32_t)num_nodes;

            uint32_t space_axis = 0;
            vec3 c_left = (split_data.left_bounds[0] + split_data.left_bounds[1]) / 2,
                 c_right = (split_data.right_bounds[1] + split_data.right_bounds[1]) / 2;

            vec3 dist = abs(c_left - c_right);

            if (dist.x > dist.y && dist.x > dist.z) {
                space_axis = 0;
            } else if (dist.y > dist.x && dist.y > dist.z) {
                space_axis = 1;
            } else {
                space_axis = 2;
            }

            vec3 bbox_min = math::min(split_data.left_bounds[0], split_data.right_bounds[0]),
                 bbox_max = math::max(split_data.left_bounds[1], split_data.right_bounds[1]);

            out_nodes.push_back({ 0, 0, index + 1, index + 2, parent_index, sibling_index, space_axis,
                {   { bbox_min.x, bbox_min.y, bbox_min.z },
                    { bbox_max.x, bbox_max.y, bbox_max.z }
                }
            });

            // push_front
            triangle_lists.insert(triangle_lists.begin(), { std::move(split_data.left_indices), split_data.left_bounds[0], split_data.left_bounds[1] });
            triangle_lists.insert(triangle_lists.begin(), { std::move(split_data.right_indices), split_data.right_bounds[0], split_data.right_bounds[1] });

            num_nodes += 2;
        }
    }

    return (uint32_t)(out_nodes.size() - root_node_index);
}


bool ray::NaiivePluckerTest(const float p[9], const float o[3], const float d[3]) {
    // plucker coordinates for edges
    float e0[6] = { p[6] - p[0], p[7] - p[1], p[8] - p[2],
                    p[7] * p[2] - p[8] * p[1],
                    p[8] * p[0] - p[6] * p[2],
                    p[6] * p[1] - p[7] * p[0]
                  },
                  e1[6] = { p[3] - p[6], p[4] - p[7], p[5] - p[8],
                            p[4] * p[8] - p[5] * p[7],
                            p[5] * p[6] - p[3] * p[8],
                            p[3] * p[7] - p[4] * p[6]
                          },
                          e2[6] = { p[0] - p[3], p[1] - p[4], p[2] - p[5],
                                    p[1] * p[5] - p[2] * p[4],
                                    p[2] * p[3] - p[0] * p[5],
                                    p[0] * p[4] - p[1] * p[3]
                                  };

    // plucker coordinates for ray
    float R[6] = { d[1] * o[2] - d[2] * o[1],
                   d[2] * o[0] - d[0] * o[2],
                   d[0] * o[1] - d[1] * o[0],
                   d[0], d[1], d[2]
                 };

    float t0 = 0, t1 = 0, t2 = 0;
    for (int w = 0; w < 6; w++) {
        t0 += e0[w] * R[w];
        t1 += e1[w] * R[w];
        t2 += e2[w] * R[w];
    }

    return (t0 <= 0 && t1 <= 0 && t2 <= 0) || (t0 >= 0 && t1 >= 0 && t2 >= 0);
}

void ray::ConstructCamera(eCamType type, const float origin[3], const float fwd[3], float fov, camera_t *cam) {
    using namespace math;

    if (type == Persp) {
        vec3 o = make_vec3(origin);
        vec3 f = make_vec3(fwd);
        vec3 u = { 0, 1, 0 };

        vec3 s = normalize(cross(f, u));
        u = cross(s, f);

        cam->type = type;
        memcpy(&cam->origin[0], value_ptr(o), sizeof(vec3));
        memcpy(&cam->fwd[0], value_ptr(f), sizeof(vec3));
        memcpy(&cam->side[0], value_ptr(s), sizeof(vec3));
        memcpy(&cam->up[0], value_ptr(u), sizeof(vec3));
    } else if (type == Ortho) {
        // TODO!
    }
}

void ray::TransformBoundingBox(const float bbox[2][3], const float *xform, float out_bbox[2][3]) {
    using namespace math;

    out_bbox[0][0] = out_bbox[1][0] = xform[12];
    out_bbox[0][1] = out_bbox[1][1] = xform[13];
    out_bbox[0][2] = out_bbox[1][2] = xform[14];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            float a = xform[i * 4 + j] * bbox[0][i];
            float b = xform[i * 4 + j] * bbox[1][i];

            if (a < b) {
                out_bbox[0][j] += a;
                out_bbox[1][j] += b;
            } else {
                out_bbox[0][j] += b;
                out_bbox[1][j] += a;
            }
        }
    }
}