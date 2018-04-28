#include "Core.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <vector>

#include "BVHSplit.h"

namespace ray {
const float axis_aligned_normal_eps = 0.000001f;

force_inline ref::simd_fvec3 cross(const ref::simd_fvec3 &v1, const ref::simd_fvec3 &v2) {
    return { v1[1] * v2[2] - v1[2] * v2[1],
             v1[2] * v2[0] - v1[0] * v2[2],
             v1[0] * v2[1] - v1[1] * v2[0] };
}
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
        acc->ci |= TRI_AXIS_ALIGNED_BIT;
    }
    assert((acc->ci & TRI_W_BITS) == w);
}

void ray::PreprocessCone(const float o[3], const float v[3], float phi, float cone_start, float cone_end, cone_accel_t *acc) {
    for (int i = 0; i < 3; i++) {
        acc->o[i] = o[i];
        acc->v[i] = -v[i];
    }
    float cos_phi = std::cos(phi);
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

        ref::simd_fvec3 _min = min(ref::simd_fvec3{ &p[0] }, min(ref::simd_fvec3{ &p[3] }, ref::simd_fvec3{ &p[6] })),
                        _max = max(ref::simd_fvec3{ &p[0] }, max(ref::simd_fvec3{ &p[3] }, ref::simd_fvec3{ &p[6] }));

        primitives.push_back({ _min, _max });
    }

    size_t indices_start = out_tri_indices.size();
    uint32_t num_out_nodes = PreprocessPrims(&primitives[0], primitives.size(), out_nodes, out_tri_indices);

    for (size_t i = indices_start; i < out_tri_indices.size(); i++) {
        out_tri_indices[i] += (uint32_t)tris_start;
    }

    return num_out_nodes;
}

uint32_t ray::PreprocessPrims(const prim_t *prims, size_t prims_count,
                              std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices) {
    struct prims_coll_t {
        std::vector<uint32_t> indices;
        ref::simd_fvec3 min = { std::numeric_limits<float>::max() }, max = { std::numeric_limits<float>::lowest() };
        prims_coll_t() {}
        prims_coll_t(std::vector<uint32_t> &&_indices, const ref::simd_fvec3 &_min, const ref::simd_fvec3 &_max)
            : indices(std::move(_indices)), min(_min), max(_max) {
        }
    };

    std::vector<prims_coll_t> triangle_lists;
    triangle_lists.emplace_back();

    size_t num_nodes = out_nodes.size();
    int32_t root_node_index = (int32_t)num_nodes;

    for (size_t j = 0; j < prims_count; j++) {
        triangle_lists.back().indices.push_back((uint32_t)j);
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
            ref::simd_fvec3 bbox_min = split_data.left_bounds[0],
                            bbox_max = split_data.left_bounds[1];

            out_nodes.push_back({ (uint32_t)out_indices.size(), (uint32_t)split_data.left_indices.size(), 0, 0, parent_index, sibling_index, 0,
                {   { bbox_min[0], bbox_min[1], bbox_min[2] },
                    { bbox_max[0], bbox_max[1], bbox_max[2] }
                }
            });
            out_indices.insert(out_indices.end(), split_data.left_indices.begin(), split_data.left_indices.end());
        } else {
            uint32_t index = (uint32_t)num_nodes;

            uint32_t space_axis = 0;
            ref::simd_fvec3 c_left = (split_data.left_bounds[0] + split_data.left_bounds[1]) / 2,
                            c_right = (split_data.right_bounds[1] + split_data.right_bounds[1]) / 2;

            ref::simd_fvec3 dist = abs(c_left - c_right);

            if (dist[0] > dist[1] && dist[0] > dist[2]) {
                space_axis = 0;
            } else if (dist[1] > dist[0] && dist[1] > dist[2]) {
                space_axis = 1;
            } else {
                space_axis = 2;
            }

            ref::simd_fvec3 bbox_min = min(split_data.left_bounds[0], split_data.right_bounds[0]),
                            bbox_max = max(split_data.left_bounds[1], split_data.right_bounds[1]);

            out_nodes.push_back({ 0, 0, index + 1, index + 2, parent_index, sibling_index, space_axis,
                {   { bbox_min[0], bbox_min[1], bbox_min[2] },
                    { bbox_max[0], bbox_max[1], bbox_max[2] }
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
    if (type == Persp) {
        ref::simd_fvec3 o = { origin };
        ref::simd_fvec3 f = { fwd };
        ref::simd_fvec3 u = { 0, 1, 0 };

        ref::simd_fvec3 s = normalize(cross(f, u));
        u = cross(s, f);

        cam->type = type;
        memcpy(&cam->origin[0], &o[0], 3 * sizeof(float));
        memcpy(&cam->fwd[0], &f[0], 3 * sizeof(float));
        memcpy(&cam->side[0], &s[0], 3 * sizeof(float));
        memcpy(&cam->up[0], &u[0], 3 * sizeof(float));
    } else if (type == Ortho) {
        // TODO!
    }
}

void ray::TransformBoundingBox(const float bbox[2][3], const float *xform, float out_bbox[2][3]) {
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

void ray::InverseMatrix(const float mat[16], float out_mat[16]) {
    float A2323 = mat[10] * mat[15] - mat[11] * mat[14];
    float A1323 = mat[9] * mat[15] - mat[11] * mat[13];
    float A1223 = mat[9] * mat[14] - mat[10] * mat[13];
    float A0323 = mat[8] * mat[15] - mat[11] * mat[12];
    float A0223 = mat[8] * mat[14] - mat[10] * mat[12];
    float A0123 = mat[8] * mat[13] - mat[9] * mat[12];
    float A2313 = mat[6] * mat[15] - mat[7] * mat[14];
    float A1313 = mat[5] * mat[15] - mat[7] * mat[13];
    float A1213 = mat[5] * mat[14] - mat[6] * mat[13];
    float A2312 = mat[6] * mat[11] - mat[7] * mat[10];
    float A1312 = mat[5] * mat[11] - mat[7] * mat[9];
    float A1212 = mat[5] * mat[10] - mat[6] * mat[9];
    float A0313 = mat[4] * mat[15] - mat[7] * mat[12];
    float A0213 = mat[4] * mat[14] - mat[6] * mat[12];
    float A0312 = mat[4] * mat[11] - mat[7] * mat[8];
    float A0212 = mat[4] * mat[10] - mat[6] * mat[8];
    float A0113 = mat[4] * mat[13] - mat[5] * mat[12];
    float A0112 = mat[4] * mat[9] - mat[5] * mat[8];

    float inv_det = 1.0f / (mat[0] * (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223)
                            - mat[1] * (mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223)
                            + mat[2] * (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123)
                            - mat[3] * (mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123));

    out_mat[0] = inv_det *   (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223);
    out_mat[1] = inv_det * -(mat[1] * A2323 - mat[2] * A1323 + mat[3] * A1223);
    out_mat[2] = inv_det *   (mat[1] * A2313 - mat[2] * A1313 + mat[3] * A1213);
    out_mat[3] = inv_det * -(mat[1] * A2312 - mat[2] * A1312 + mat[3] * A1212);
    out_mat[4] = inv_det * -(mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223);
    out_mat[5] = inv_det *   (mat[0] * A2323 - mat[2] * A0323 + mat[3] * A0223);
    out_mat[6] = inv_det * -(mat[0] * A2313 - mat[2] * A0313 + mat[3] * A0213);
    out_mat[7] = inv_det *   (mat[0] * A2312 - mat[2] * A0312 + mat[3] * A0212);
    out_mat[8] = inv_det *   (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123);
    out_mat[9] = inv_det * -(mat[0] * A1323 - mat[1] * A0323 + mat[3] * A0123);
    out_mat[10] = inv_det *   (mat[0] * A1313 - mat[1] * A0313 + mat[3] * A0113);
    out_mat[11] = inv_det * -(mat[0] * A1312 - mat[1] * A0312 + mat[3] * A0112);
    out_mat[12] = inv_det * -(mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123);
    out_mat[13] = inv_det *   (mat[0] * A1223 - mat[1] * A0223 + mat[2] * A0123);
    out_mat[14] = inv_det * -(mat[0] * A1213 - mat[1] * A0213 + mat[2] * A0113);
    out_mat[15] = inv_det *   (mat[0] * A1212 - mat[1] * A0212 + mat[2] * A0112);
}