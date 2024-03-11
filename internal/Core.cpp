#include "Core.h"

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>

#include <deque>
#include <vector>

#include "BVHSplit.h"

namespace Ray {
#include "precomputed/__pmj02_samples.inl"

force_inline Ref::fvec3 cross(const Ref::fvec3 &v1, const Ref::fvec3 &v2) {
    return {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
}

// "Insert" two 0 bits after each of the 10 low bits of x
force_inline uint32_t Part1By2(uint32_t x) {
    x = x & 0b00000000000000000000001111111111;               // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0b00000011000000000000000011111111; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0b00000011000000001111000000001111;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0b00000011000011000011000011000011;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0b00001001001001001001001001001001;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

force_inline uint32_t EncodeMorton3(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

struct prim_chunk_t {
    uint32_t code, base, size;
};

const int BitsPerDim = 10;
const int BitsTotal = 3 * BitsPerDim;

void radix_sort_prim_chunks(prim_chunk_t *begin, prim_chunk_t *end, prim_chunk_t *begin1) {
    prim_chunk_t *end1 = begin1 + (end - begin);

    const int bits_per_pass = 6;
    const int bucket_size = 1 << bits_per_pass;
    const int bit_mask = bucket_size - 1;

    for (int shift = 0; shift < BitsTotal; shift += bits_per_pass) {
        size_t count[bucket_size] = {};
        for (prim_chunk_t *p = begin; p != end; p++) {
            count[(p->code >> shift) & bit_mask]++;
        }
        prim_chunk_t *bucket[bucket_size], *q = begin1;
        for (int i = 0; i < bucket_size; q += count[i++]) {
            bucket[i] = q;
        }
        for (prim_chunk_t *p = begin; p != end; p++) {
            *bucket[(p->code >> shift) & bit_mask]++ = *p;
        }
        std::swap(begin, begin1);
        std::swap(end, end1);
    }
}

void sort_mort_codes(uint32_t *morton_codes, size_t prims_count, uint32_t *out_indices) {
    std::vector<prim_chunk_t> run_chunks;
    run_chunks.reserve(prims_count);

    for (uint32_t start = 0, end = 1; end <= uint32_t(prims_count); end++) {
        if (end == uint32_t(prims_count) || (morton_codes[start] != morton_codes[end])) {

            run_chunks.push_back({morton_codes[start], start, end - start});

            start = end;
        }
    }

    std::vector<prim_chunk_t> run_chunks2(run_chunks.size());

    radix_sort_prim_chunks(&run_chunks[0], &run_chunks[0] + run_chunks.size(), &run_chunks2[0]);
    std::swap(run_chunks, run_chunks2);

    size_t counter = 0;
    for (const prim_chunk_t &ch : run_chunks) {
        for (uint32_t j = 0; j < ch.size; j++) {
            morton_codes[counter] = ch.code;
            out_indices[counter++] = ch.base + j;
        }
    }
}
} // namespace Ray

void Ray::CanonicalToDir(const float p[2], const float y_rotation, float out_d[3]) {
    const float cos_theta = 2 * p[0] - 1;
    float phi = 2 * PI * p[1] + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float sin_theta = sqrtf(1 - cos_theta * cos_theta);

    const float sin_phi = sinf(phi);
    const float cos_phi = cosf(phi);

    out_d[0] = sin_theta * cos_phi;
    out_d[1] = cos_theta;
    out_d[2] = -sin_theta * sin_phi;
}

void Ray::DirToCanonical(const float d[3], const float y_rotation, float out_p[2]) {
    const float cos_theta = fminf(fmaxf(d[1], -1.0f), 1.0f);

    float phi = -atan2f(d[2], d[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    out_p[0] = (cos_theta + 1.0f) / 2.0f;
    out_p[1] = phi / (2.0f * PI);
}

// Used to convert 16x16 sphere sector coordinates to single value
const uint8_t Ray::morton_table_16[] = {0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85};

// Used to convert 256x256x256 grid coordinates to single value, i think it leads to more uniform distribution than (z
// << 16) | (y << 8) | (x << 0)
const int Ray::morton_table_256[] = {
    0,       1,       8,       9,       64,      65,      72,      73,      512,     513,     520,     521,     576,
    577,     584,     585,     4096,    4097,    4104,    4105,    4160,    4161,    4168,    4169,    4608,    4609,
    4616,    4617,    4672,    4673,    4680,    4681,    32768,   32769,   32776,   32777,   32832,   32833,   32840,
    32841,   33280,   33281,   33288,   33289,   33344,   33345,   33352,   33353,   36864,   36865,   36872,   36873,
    36928,   36929,   36936,   36937,   37376,   37377,   37384,   37385,   37440,   37441,   37448,   37449,   262144,
    262145,  262152,  262153,  262208,  262209,  262216,  262217,  262656,  262657,  262664,  262665,  262720,  262721,
    262728,  262729,  266240,  266241,  266248,  266249,  266304,  266305,  266312,  266313,  266752,  266753,  266760,
    266761,  266816,  266817,  266824,  266825,  294912,  294913,  294920,  294921,  294976,  294977,  294984,  294985,
    295424,  295425,  295432,  295433,  295488,  295489,  295496,  295497,  299008,  299009,  299016,  299017,  299072,
    299073,  299080,  299081,  299520,  299521,  299528,  299529,  299584,  299585,  299592,  299593,  2097152, 2097153,
    2097160, 2097161, 2097216, 2097217, 2097224, 2097225, 2097664, 2097665, 2097672, 2097673, 2097728, 2097729, 2097736,
    2097737, 2101248, 2101249, 2101256, 2101257, 2101312, 2101313, 2101320, 2101321, 2101760, 2101761, 2101768, 2101769,
    2101824, 2101825, 2101832, 2101833, 2129920, 2129921, 2129928, 2129929, 2129984, 2129985, 2129992, 2129993, 2130432,
    2130433, 2130440, 2130441, 2130496, 2130497, 2130504, 2130505, 2134016, 2134017, 2134024, 2134025, 2134080, 2134081,
    2134088, 2134089, 2134528, 2134529, 2134536, 2134537, 2134592, 2134593, 2134600, 2134601, 2359296, 2359297, 2359304,
    2359305, 2359360, 2359361, 2359368, 2359369, 2359808, 2359809, 2359816, 2359817, 2359872, 2359873, 2359880, 2359881,
    2363392, 2363393, 2363400, 2363401, 2363456, 2363457, 2363464, 2363465, 2363904, 2363905, 2363912, 2363913, 2363968,
    2363969, 2363976, 2363977, 2392064, 2392065, 2392072, 2392073, 2392128, 2392129, 2392136, 2392137, 2392576, 2392577,
    2392584, 2392585, 2392640, 2392641, 2392648, 2392649, 2396160, 2396161, 2396168, 2396169, 2396224, 2396225, 2396232,
    2396233, 2396672, 2396673, 2396680, 2396681, 2396736, 2396737, 2396744, 2396745};

// Used to bind horizontal vector angle to sector on sphere
const float Ray::omega_step = 0.0625f;
const char Ray::omega_table[] = {15, 14, 13, 12, 12, 11, 11, 11, 10, 10, 9, 9, 9, 8, 8, 8, 8,
                                 7,  7,  7,  6,  6,  6,  5,  5,  4,  4,  4, 3, 3, 2, 1, 0};

// Used to bind vectical vector angle to sector on sphere
const float Ray::phi_step = 0.125f;
// clang-format off
const char Ray::phi_table[][17] = {{ 2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6  },
                                   { 1,  2,  2,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  5,  6,  6  },
                                   { 1,  1,  2,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6  },
                                   { 1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  6,  6,  6,  6  },
                                   { 1,  1,  1,  1,  2,  2,  2,  3,  4,  4,  5,  5,  6,  6,  6,  6,  6  },
                                   { 0,  1,  1,  1,  1,  2,  2,  3,  4,  4,  5,  6,  6,  6,  6,  6,  7  },
                                   { 0,  0,  0,  0,  1,  1,  2,  2,  4,  5,  6,  6,  6,  7,  7,  7,  7  },
                                   { 0,  0,  0,  0,  0,  0,  1,  2,  4,  6,  6,  7,  7,  7,  7,  7,  7  },
                                   { 15, 15, 15, 15, 15, 15, 15, 15, 8,  8,  8,  8,  8,  8,  8,  8,  8  },
                                   { 15, 15, 15, 15, 15, 15, 14, 14, 12, 10, 9,  8,  8,  8,  8,  8,  8  },
                                   { 15, 15, 15, 15, 14, 14, 14, 13, 12, 10, 10, 9,  9,  8,  8,  8,  8  },
                                   { 15, 14, 14, 14, 14, 14, 13, 12, 12, 11, 10, 10, 9,  9,  9,  9,  8  },
                                   { 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 10, 10, 9,  9,  9,  9  },
                                   { 14, 14, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 9,  9,  9  },
                                   { 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 10, 9,  9  },
                                   { 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 10, 10, 9  },
                                   { 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10 }};
// clang-format on

bool Ray::PreprocessTri(const float *p, int stride, tri_accel_t *out_acc) {
    if (!stride) {
        stride = 3;
    }

    // edges
    const float e0[3] = {p[stride] - p[0], p[stride + 1] - p[1], p[stride + 2] - p[2]},
                e1[3] = {p[stride * 2] - p[0], p[stride * 2 + 1] - p[1], p[stride * 2 + 2] - p[2]};

    float n[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};

    const float n_len_sqr = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (n_len_sqr == 0.0f) {
        // degenerate triangle
        return false;
    }

    const float u[3] = {(e1[1] * n[2] - e1[2] * n[1]) / n_len_sqr, (e1[2] * n[0] - e1[0] * n[2]) / n_len_sqr,
                        (e1[0] * n[1] - e1[1] * n[0]) / n_len_sqr};

    // edge planes
    out_acc->u_plane[0] = u[0];
    out_acc->u_plane[1] = u[1];
    out_acc->u_plane[2] = u[2];
    out_acc->u_plane[3] = -(u[0] * p[0] + u[1] * p[1] + u[2] * p[2]);

    const float v[3] = {(n[1] * e0[2] - n[2] * e0[1]) / n_len_sqr, (n[2] * e0[0] - n[0] * e0[2]) / n_len_sqr,
                        (n[0] * e0[1] - n[1] * e0[0]) / n_len_sqr};

    out_acc->v_plane[0] = v[0];
    out_acc->v_plane[1] = v[1];
    out_acc->v_plane[2] = v[2];
    out_acc->v_plane[3] = -(v[0] * p[0] + v[1] * p[1] + v[2] * p[2]);

    // normal plane
    const float l = sqrtf(n_len_sqr);
    n[0] /= l;
    n[1] /= l;
    n[2] /= l;

    out_acc->n_plane[0] = n[0];
    out_acc->n_plane[1] = n[1];
    out_acc->n_plane[2] = n[2];
    out_acc->n_plane[3] = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];

    return true;
}

uint32_t Ray::PreprocessMesh(const vtx_attribute_t &positions, Span<const uint32_t> vtx_indices, const int base_vertex,
                             const bvh_settings_t &s, std::vector<bvh_node_t> &out_nodes,
                             aligned_vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_tri_indices,
                             aligned_vector<mtri_accel_t> &out_tris2) {
    assert(!vtx_indices.empty() && vtx_indices.size() % 3 == 0);

    aligned_vector<prim_t> primitives;
    aligned_vector<tri_accel_t> triangles;
    std::vector<uint32_t> real_indices;

    primitives.reserve(vtx_indices.size() / 3);
    triangles.reserve(vtx_indices.size() / 3);
    real_indices.reserve(vtx_indices.size() / 3);

    for (int j = 0; j < int(vtx_indices.size()); j += 3) {
        Ref::fvec4 p[3] = {{0.0f}, {0.0f}, {0.0f}};

        const uint32_t i0 = vtx_indices[j + 0] + base_vertex, i1 = vtx_indices[j + 1] + base_vertex,
                       i2 = vtx_indices[j + 2] + base_vertex;

        memcpy(value_ptr(p[0]), &positions.data[positions.offset + i0 * positions.stride], 3 * sizeof(float));
        memcpy(value_ptr(p[1]), &positions.data[positions.offset + i1 * positions.stride], 3 * sizeof(float));
        memcpy(value_ptr(p[2]), &positions.data[positions.offset + i2 * positions.stride], 3 * sizeof(float));

        tri_accel_t tri = {};
        if (PreprocessTri(value_ptr(p[0]), 4, &tri)) {
            real_indices.push_back(uint32_t(j / 3));
            triangles.push_back(tri);
        } else {
            continue;
        }

        const Ref::fvec4 _min = min(p[0], min(p[1], p[2])), _max = max(p[0], max(p[1], p[2]));

        primitives.push_back({i0, i1, i2, _min, _max});
    }

    const size_t indices_start = out_tri_indices.size();
    uint32_t num_out_nodes;
    if (!s.use_fast_bvh_build) {
        num_out_nodes = PreprocessPrims_SAH(primitives, positions, s, out_nodes, out_tri_indices);
    } else {
        num_out_nodes = PreprocessPrims_HLBVH(primitives, out_nodes, out_tri_indices);
    }

    // make sure mesh occupies the whole 8-triangle slot
    while (out_tri_indices.size() % 8) {
        out_tri_indices.push_back(0);
    }

    // const uint32_t tris_start = out_tris2.size(), tris_count = out_tri_indices.size() - indices_start;
    out_tris.resize(out_tri_indices.size());
    out_tris2.resize(out_tri_indices.size() / 8);

    for (size_t i = indices_start; i < out_tri_indices.size(); i++) {
        const uint32_t j = out_tri_indices[i];

        out_tris[i] = triangles[j];

        for (int k = 0; k < 4; ++k) {
            out_tris2[i / 8].n_plane[k][i % 8] = triangles[j].n_plane[k];
            out_tris2[i / 8].u_plane[k][i % 8] = triangles[j].u_plane[k];
            out_tris2[i / 8].v_plane[k][i % 8] = triangles[j].v_plane[k];
        }

        out_tri_indices[i] = uint32_t(real_indices[j]);
    }

    return num_out_nodes;
}

uint32_t Ray::EmitLBVH_r(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes,
                         uint32_t prim_index, uint32_t prim_count, uint32_t index_offset, int bit_index,
                         std::vector<bvh_node_t> &out_nodes) {
    if (bit_index == -1 || prim_count < 8) {
        Ref::fvec4 bbox_min = {FLT_MAX}, bbox_max = {-FLT_MAX};

        for (uint32_t i = prim_index; i < prim_index + prim_count; i++) {
            bbox_min = min(bbox_min, prims[indices[i]].bbox_min);
            bbox_max = max(bbox_max, prims[indices[i]].bbox_max);
        }

        const auto node_index = uint32_t(out_nodes.size());

        out_nodes.emplace_back();
        bvh_node_t &node = out_nodes.back();

        node.prim_index = LEAF_NODE_BIT + prim_index + index_offset;
        node.prim_count = prim_count;

        memcpy(&node.bbox_min[0], value_ptr(bbox_min), 3 * sizeof(float));
        memcpy(&node.bbox_max[0], value_ptr(bbox_max), 3 * sizeof(float));

        return node_index;
    } else {
        const uint32_t mask = (1u << bit_index);

        if ((morton_codes[prim_index] & mask) == (morton_codes[prim_index + prim_count - 1] & mask)) {
            return EmitLBVH_r(prims, indices, morton_codes, prim_index, prim_count, index_offset, bit_index - 1,
                              out_nodes);
        }

        uint32_t search_start = prim_index, search_end = search_start + prim_count - 1;
        while (search_start + 1 != search_end) {
            const uint32_t mid = (search_start + search_end) / 2;
            if ((morton_codes[search_start] & mask) == (morton_codes[mid] & mask)) {
                search_start = mid;
            } else {
                search_end = mid;
            }
        }

        const uint32_t split_offset = search_end - prim_index;

        const auto node_index = uint32_t(out_nodes.size());
        out_nodes.emplace_back();

        uint32_t child0 =
            EmitLBVH_r(prims, indices, morton_codes, prim_index, split_offset, index_offset, bit_index - 1, out_nodes);
        uint32_t child1 = EmitLBVH_r(prims, indices, morton_codes, prim_index + split_offset, prim_count - split_offset,
                                     index_offset, bit_index - 1, out_nodes);

        uint32_t space_axis = bit_index % 3;
        if (out_nodes[child0].bbox_min[space_axis] > out_nodes[child1].bbox_min[space_axis]) {
            std::swap(child0, child1);
        }

        bvh_node_t &par_node = out_nodes[node_index];
        par_node.left_child = child0;
        par_node.right_child = (space_axis << 30) + child1;

        for (int i = 0; i < 3; i++) {
            par_node.bbox_min[i] = fminf(out_nodes[child0].bbox_min[i], out_nodes[child1].bbox_min[i]);
            par_node.bbox_max[i] = fmaxf(out_nodes[child0].bbox_max[i], out_nodes[child1].bbox_max[i]);
        }

        return node_index;
    }
}

uint32_t Ray::EmitLBVH(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes, uint32_t prim_index,
                       uint32_t prim_count, uint32_t index_offset, int bit_index, std::vector<bvh_node_t> &out_nodes) {
    struct proc_item_t {
        int bit_index;
        uint32_t prim_index, prim_count;
        uint32_t split_offset, node_index;
    };

    proc_item_t proc_stack[256];
    uint32_t stack_size = 0;

    const auto root_node_index = uint32_t(out_nodes.size());
    out_nodes.emplace_back();
    proc_stack[stack_size++] = {bit_index, prim_index, prim_count, 0xffffffff, root_node_index};

    while (stack_size) {
        proc_item_t &cur = proc_stack[stack_size - 1];

        if (cur.bit_index == -1 || cur.prim_count < 8) {
            Ref::fvec4 bbox_min = {FLT_MAX}, bbox_max = {-FLT_MAX};

            for (uint32_t i = cur.prim_index; i < cur.prim_index + cur.prim_count; i++) {
                bbox_min = min(bbox_min, prims[indices[i]].bbox_min);
                bbox_max = max(bbox_max, prims[indices[i]].bbox_max);
            }

            bvh_node_t &node = out_nodes[cur.node_index];

            node.prim_index = LEAF_NODE_BIT + cur.prim_index + index_offset;
            node.prim_count = cur.prim_count;

            memcpy(&node.bbox_min[0], value_ptr(bbox_min), 3 * sizeof(float));
            memcpy(&node.bbox_max[0], value_ptr(bbox_max), 3 * sizeof(float));
        } else {
            if (cur.split_offset == 0xffffffff) {
                const uint32_t mask = (1u << cur.bit_index);

                uint32_t search_start = cur.prim_index, search_end = search_start + cur.prim_count - 1;

                if ((morton_codes[search_start] & mask) == (morton_codes[search_end] & mask)) {
                    cur.bit_index--;
                    continue;
                }

                while (search_start + 1 != search_end) {
                    const uint32_t mid = (search_start + search_end) / 2;
                    if ((morton_codes[search_start] & mask) == (morton_codes[mid] & mask)) {
                        search_start = mid;
                    } else {
                        search_end = mid;
                    }
                }

                cur.split_offset = search_end - cur.prim_index;

                const auto child0 = uint32_t(out_nodes.size());
                out_nodes.emplace_back();

                const auto child1 = uint32_t(out_nodes.size());
                out_nodes.emplace_back();

                out_nodes[cur.node_index].left_child = child0;
                out_nodes[cur.node_index].right_child = child1;

                proc_stack[stack_size++] = {cur.bit_index - 1, cur.prim_index + cur.split_offset,
                                            cur.prim_count - cur.split_offset, 0xffffffff, child1};
                proc_stack[stack_size++] = {cur.bit_index - 1, cur.prim_index, cur.split_offset, 0xffffffff, child0};
                continue;
            } else {
                bvh_node_t &node = out_nodes[cur.node_index];

                for (int i = 0; i < 3; i++) {
                    node.bbox_min[i] =
                        fminf(out_nodes[node.left_child].bbox_min[i], out_nodes[node.right_child].bbox_min[i]);
                    node.bbox_max[i] =
                        fmaxf(out_nodes[node.left_child].bbox_max[i], out_nodes[node.right_child].bbox_max[i]);
                }

                const uint32_t space_axis = (cur.bit_index % 3);
                if (out_nodes[node.left_child].bbox_min[space_axis] >
                    out_nodes[node.right_child].bbox_min[space_axis]) {
                    std::swap(node.left_child, node.right_child);
                }
                node.right_child += (space_axis << 30);
            }
        }

        stack_size--;
    }

    return root_node_index;
}

uint32_t Ray::PreprocessPrims_SAH(Span<const prim_t> prims, const vtx_attribute_t &positions, const bvh_settings_t &s,
                                  std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices) {
    struct prims_coll_t {
        std::vector<uint32_t> indices;
        Ref::fvec4 min = {FLT_MAX, FLT_MAX, FLT_MAX, 0.0f}, max = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f};
        prims_coll_t() = default;
        prims_coll_t(std::vector<uint32_t> &&_indices, const Ref::fvec4 &_min, const Ref::fvec4 &_max)
            : indices(std::move(_indices)), min(_min), max(_max) {}
    };

    std::deque<prims_coll_t, aligned_allocator<prims_coll_t, alignof(prims_coll_t)>> prim_lists;
    prim_lists.emplace_back();

    size_t num_nodes = out_nodes.size();
    const auto root_node_index = uint32_t(num_nodes);

    for (uint32_t j = 0; j < uint32_t(prims.size()); j++) {
        prim_lists.back().indices.push_back(j);
        prim_lists.back().min = min(prim_lists.back().min, prims[j].bbox_min);
        prim_lists.back().max = max(prim_lists.back().max, prims[j].bbox_max);
    }

    Ref::fvec4 root_min = prim_lists.back().min, root_max = prim_lists.back().max;

    while (!prim_lists.empty()) {
        split_data_t split_data =
            SplitPrimitives_SAH(prims.data(), prim_lists.back().indices, positions, prim_lists.back().min,
                                prim_lists.back().max, root_min, root_max, s);
        prim_lists.pop_back();

        if (split_data.right_indices.empty()) {
            Ref::fvec4 bbox_min = split_data.left_bounds[0], bbox_max = split_data.left_bounds[1];

            out_nodes.emplace_back();
            bvh_node_t &n = out_nodes.back();

            n.prim_index = LEAF_NODE_BIT + uint32_t(out_indices.size());
            n.prim_count = uint32_t(split_data.left_indices.size());
            memcpy(&n.bbox_min[0], value_ptr(bbox_min), 3 * sizeof(float));
            memcpy(&n.bbox_max[0], value_ptr(bbox_max), 3 * sizeof(float));
            out_indices.insert(out_indices.end(), split_data.left_indices.begin(), split_data.left_indices.end());
        } else {
            const auto index = uint32_t(num_nodes);

            uint32_t space_axis = 0;
            const Ref::fvec4 c_left = (split_data.left_bounds[0] + split_data.left_bounds[1]) / 2.0f,
                                  c_right = (split_data.right_bounds[0] + split_data.right_bounds[1]) / 2.0f;

            const Ref::fvec4 dist = abs(c_left - c_right);

            if (dist.get<0>() > dist.get<1>() && dist.get<0>() > dist.get<2>()) {
                space_axis = 0;
            } else if (dist.get<1>() > dist.get<0>() && dist.get<1>() > dist.get<2>()) {
                space_axis = 1;
            } else {
                space_axis = 2;
            }

            const Ref::fvec4 bbox_min = min(split_data.left_bounds[0], split_data.right_bounds[0]),
                                  bbox_max = max(split_data.left_bounds[1], split_data.right_bounds[1]);

            out_nodes.emplace_back();
            bvh_node_t &n = out_nodes.back();
            n.left_child = index + 1;
            n.right_child = (space_axis << 30) + index + 2;
            memcpy(&n.bbox_min[0], value_ptr(bbox_min), 3 * sizeof(float));
            memcpy(&n.bbox_max[0], value_ptr(bbox_max), 3 * sizeof(float));
            prim_lists.emplace_front(std::move(split_data.left_indices), split_data.left_bounds[0],
                                     split_data.left_bounds[1]);
            prim_lists.emplace_front(std::move(split_data.right_indices), split_data.right_bounds[0],
                                     split_data.right_bounds[1]);

            num_nodes += 2;
        }
    }

    return uint32_t(out_nodes.size() - root_node_index);
}

uint32_t Ray::PreprocessPrims_HLBVH(Span<const prim_t> prims, std::vector<bvh_node_t> &out_nodes,
                                    std::vector<uint32_t> &out_indices) {
    std::vector<uint32_t> morton_codes(prims.size());

    Ref::fvec4 whole_min = {FLT_MAX, FLT_MAX, FLT_MAX, 0.0f}, whole_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f};

    const auto indices_start = uint32_t(out_indices.size());
    out_indices.reserve(out_indices.size() + prims.size());

    for (uint32_t j = 0; j < uint32_t(prims.size()); j++) {
        whole_min = min(whole_min, prims[j].bbox_min);
        whole_max = max(whole_max, prims[j].bbox_max);

        out_indices.push_back(j);
    }

    uint32_t *indices = &out_indices[indices_start];

    const Ref::fvec4 scale = float(1 << BitsPerDim) / (whole_max - whole_min + FLT_EPSILON);

    // compute morton codes
    for (int i = 0; i < int(prims.size()); i++) {
        const Ref::fvec4 center = 0.5f * (prims[i].bbox_min + prims[i].bbox_max);
        const Ref::fvec4 code = (center - whole_min) * scale;

        const auto x = uint32_t(code.get<0>()), y = uint32_t(code.get<1>()), z = uint32_t(code.get<2>());

        morton_codes[i] = EncodeMorton3(x, y, z);
    }

    sort_mort_codes(&morton_codes[0], morton_codes.size(), indices);

    struct treelet_t {
        uint32_t index, count;
        uint32_t node_index;
    };

    std::vector<treelet_t> treelets;
    treelets.reserve(1 << 12); // Top-level bvh can have up to 4096 items

    // Use upper 12 bits to extract treelets
    for (uint32_t start = 0, end = 1; end <= uint32_t(morton_codes.size()); end++) {
        uint32_t mask = 0b00111111111111000000000000000000;
        if (end == uint32_t(morton_codes.size()) || ((morton_codes[start] & mask) != (morton_codes[end] & mask))) {

            treelets.push_back({start, end - start});

            start = end;
        }
    }

    std::vector<bvh_node_t> bottom_nodes;

    // Build bottom-level hierarchy from each treelet using LBVH
    const int start_bit = 29 - 12;
    for (treelet_t &tr : treelets) {
        tr.node_index = EmitLBVH(prims.data(), indices, &morton_codes[0], tr.index, tr.count, indices_start, start_bit,
                                 bottom_nodes);
    }

    aligned_vector<prim_t> top_prims;
    for (const treelet_t &tr : treelets) {
        const bvh_node_t &node = bottom_nodes[tr.node_index];

        top_prims.emplace_back();
        prim_t &p = top_prims.back();
        memcpy(value_ptr(p.bbox_min), node.bbox_min, 3 * sizeof(float));
        memcpy(value_ptr(p.bbox_max), node.bbox_max, 3 * sizeof(float));
    }

    const auto top_nodes_start = uint32_t(out_nodes.size());

    std::vector<uint32_t> top_indices;

    // Force spliting until each primitive will be in separate leaf node
    bvh_settings_t s;
    s.oversplit_threshold = -1.0f;
    s.allow_spatial_splits = false;
    s.min_primitives_in_leaf = 1;

    // Build top level hierarchy using SAH
    const uint32_t new_nodes_count =
        PreprocessPrims_SAH({&top_prims[0], top_prims.size()}, {}, s, out_nodes, top_indices);
    unused(new_nodes_count);

    auto bottom_nodes_start = uint32_t(out_nodes.size());

    // Replace leaf nodes of top-level bvh with bottom level nodes
    for (uint32_t i = top_nodes_start; i < uint32_t(out_nodes.size()); i++) {
        bvh_node_t &n = out_nodes[i];
        if (!(n.prim_index & LEAF_NODE_BIT)) {
            bvh_node_t &left = out_nodes[n.left_child], &right = out_nodes[n.right_child & RIGHT_CHILD_BITS];

            if (left.prim_index & LEAF_NODE_BIT) {
                assert(left.prim_count == 1);
                const uint32_t index = (left.prim_index & PRIM_INDEX_BITS);

                const treelet_t &tr = treelets[top_indices[index]];
                n.left_child = bottom_nodes_start + tr.node_index;
            }

            if (right.prim_index & LEAF_NODE_BIT) {
                assert(right.prim_count == 1);
                const uint32_t index = (right.prim_index & PRIM_INDEX_BITS);

                const treelet_t &tr = treelets[top_indices[index]];
                n.right_child = (n.right_child & SEP_AXIS_BITS) + bottom_nodes_start + tr.node_index;
            }
        }
    }

    // Remove top-level leaf nodes
    for (auto it = out_nodes.begin() + top_nodes_start; it != out_nodes.end();) {
        if (it->prim_index & LEAF_NODE_BIT) {
            const auto index = uint32_t(std::distance(out_nodes.begin(), it));

            it = out_nodes.erase(it);
            bottom_nodes_start--;

            for (auto next_it = out_nodes.begin() + top_nodes_start; next_it != out_nodes.end(); ++next_it) {
                if (!(next_it->prim_index & LEAF_NODE_BIT)) {
                    if (next_it->left_child > index) {
                        next_it->left_child--;
                    }
                    if ((next_it->right_child & RIGHT_CHILD_BITS) > index) {
                        next_it->right_child--;
                    }
                }
            }
        } else {
            ++it;
        }
    }

    const uint32_t bottom_nodes_offset = bottom_nodes_start;

    // Offset nodes in bottom-level bvh
    for (bvh_node_t &n : bottom_nodes) {
        if (!(n.prim_index & LEAF_NODE_BIT)) {
            n.left_child += bottom_nodes_offset;
            n.right_child += bottom_nodes_offset;
        }
    }

    out_nodes.insert(out_nodes.end(), bottom_nodes.begin(), bottom_nodes.end());

    return uint32_t(out_nodes.size() - top_nodes_start);
}

uint32_t Ray::FlattenBVH_r(const bvh_node_t *nodes, const uint32_t node_index, const uint32_t parent_index,
                           aligned_vector<wbvh_node_t> &out_nodes) {
    const bvh_node_t &cur_node = nodes[node_index];

    // allocate new node
    const auto new_node_index = uint32_t(out_nodes.size());
    out_nodes.emplace_back();

    if (cur_node.prim_index & LEAF_NODE_BIT) {
        wbvh_node_t &new_node = out_nodes[new_node_index];

        new_node.bbox_min[0][0] = cur_node.bbox_min[0];
        new_node.bbox_min[1][0] = cur_node.bbox_min[1];
        new_node.bbox_min[2][0] = cur_node.bbox_min[2];

        new_node.bbox_max[0][0] = cur_node.bbox_max[0];
        new_node.bbox_max[1][0] = cur_node.bbox_max[1];
        new_node.bbox_max[2][0] = cur_node.bbox_max[2];

        new_node.child[0] = cur_node.prim_index;
        new_node.child[1] = cur_node.prim_count;

        return new_node_index;
    }

    // Gather children 2 levels deep

    uint32_t children[8];
    int children_count = 0;

    const bvh_node_t &child0 = nodes[cur_node.left_child];

    if (child0.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.left_child & LEFT_CHILD_BITS;
    } else {
        const bvh_node_t &child00 = nodes[child0.left_child];
        const bvh_node_t &child01 = nodes[child0.right_child & RIGHT_CHILD_BITS];

        if (child00.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child00.left_child;
            children[children_count++] = child00.right_child & RIGHT_CHILD_BITS;
        }

        if (child01.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child01.left_child;
            children[children_count++] = child01.right_child & RIGHT_CHILD_BITS;
        }
    }

    const bvh_node_t &child1 = nodes[cur_node.right_child & RIGHT_CHILD_BITS];

    if (child1.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.right_child & RIGHT_CHILD_BITS;
    } else {
        const bvh_node_t &child10 = nodes[child1.left_child];
        const bvh_node_t &child11 = nodes[child1.right_child & RIGHT_CHILD_BITS];

        if (child10.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child10.left_child;
            children[children_count++] = child10.right_child & RIGHT_CHILD_BITS;
        }

        if (child11.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child11.left_child;
            children[children_count++] = child11.right_child & RIGHT_CHILD_BITS;
        }
    }

    // Sort children in morton order
    Ref::fvec3 children_centers[8], whole_box_min = {FLT_MAX}, whole_box_max = {-FLT_MAX};
    for (int i = 0; i < children_count; i++) {
        children_centers[i] =
            0.5f * (Ref::fvec3{nodes[children[i]].bbox_min} + Ref::fvec3{nodes[children[i]].bbox_max});
        whole_box_min = min(whole_box_min, children_centers[i]);
        whole_box_max = max(whole_box_max, children_centers[i]);
    }

    whole_box_max += Ref::fvec3{0.001f};

    const Ref::fvec3 scale = 2.0f / (whole_box_max - whole_box_min);

    uint32_t sorted_children[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                   0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    for (int i = 0; i < children_count; i++) {
        Ref::fvec3 code = (children_centers[i] - whole_box_min) * scale;

        const auto x = uint32_t(code[0]), y = uint32_t(code[1]), z = uint32_t(code[2]);

        uint32_t mort = (z << 2) | (y << 1) | (x << 0);

        while (sorted_children[mort] != 0xffffffff) {
            mort = (mort + 1) % 8;
        }

        sorted_children[mort] = children[i];
    }

    uint32_t new_children[8];

    for (int i = 0; i < 8; i++) {
        if (sorted_children[i] != 0xffffffff) {
            new_children[i] = FlattenBVH_r(nodes, sorted_children[i], node_index, out_nodes);
        } else {
            new_children[i] = 0x7fffffff;
        }
    }

    wbvh_node_t &new_node = out_nodes[new_node_index];
    memcpy(new_node.child, new_children, sizeof(new_children));

    for (int i = 0; i < 8; i++) {
        if (new_children[i] != 0x7fffffff) {
            new_node.bbox_min[0][i] = nodes[sorted_children[i]].bbox_min[0];
            new_node.bbox_min[1][i] = nodes[sorted_children[i]].bbox_min[1];
            new_node.bbox_min[2][i] = nodes[sorted_children[i]].bbox_min[2];

            new_node.bbox_max[0][i] = nodes[sorted_children[i]].bbox_max[0];
            new_node.bbox_max[1][i] = nodes[sorted_children[i]].bbox_max[1];
            new_node.bbox_max[2][i] = nodes[sorted_children[i]].bbox_max[2];
        } else {
            // Init as invalid bounding box
            new_node.bbox_min[0][i] = new_node.bbox_min[1][i] = new_node.bbox_min[2][i] = 0.0f;
            new_node.bbox_max[0][i] = new_node.bbox_max[1][i] = new_node.bbox_max[2][i] = 0.0f;
        }
    }

    return new_node_index;
}

uint32_t Ray::FlattenBVH_r(const light_bvh_node_t *nodes, const uint32_t node_index, const uint32_t parent_index,
                           aligned_vector<light_wbvh_node_t> &out_nodes) {
    const light_bvh_node_t &cur_node = nodes[node_index];

    // allocate new node
    const auto new_node_index = uint32_t(out_nodes.size());
    out_nodes.emplace_back();

    if (cur_node.prim_index & LEAF_NODE_BIT) {
        light_wbvh_node_t &new_node = out_nodes[new_node_index];

        new_node.bbox_min[0][0] = cur_node.bbox_min[0];
        new_node.bbox_min[1][0] = cur_node.bbox_min[1];
        new_node.bbox_min[2][0] = cur_node.bbox_min[2];

        new_node.bbox_max[0][0] = cur_node.bbox_max[0];
        new_node.bbox_max[1][0] = cur_node.bbox_max[1];
        new_node.bbox_max[2][0] = cur_node.bbox_max[2];

        new_node.child[0] = cur_node.prim_index;
        new_node.child[1] = cur_node.prim_count;

        new_node.flux[0] = cur_node.flux;
        new_node.axis[0][0] = cur_node.axis[0];
        new_node.axis[1][0] = cur_node.axis[1];
        new_node.axis[2][0] = cur_node.axis[2];
        new_node.omega_n[0] = cur_node.omega_n;
        new_node.omega_e[0] = cur_node.omega_e;

        return new_node_index;
    }

    // Gather children 2 levels deep

    uint32_t children[8];
    int children_count = 0;

    const light_bvh_node_t &child0 = nodes[cur_node.left_child];

    if (child0.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.left_child & LEFT_CHILD_BITS;
    } else {
        const light_bvh_node_t &child00 = nodes[child0.left_child];
        const light_bvh_node_t &child01 = nodes[child0.right_child & RIGHT_CHILD_BITS];

        if (child00.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child00.left_child;
            children[children_count++] = child00.right_child & RIGHT_CHILD_BITS;
        }

        if (child01.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child01.left_child;
            children[children_count++] = child01.right_child & RIGHT_CHILD_BITS;
        }
    }

    const light_bvh_node_t &child1 = nodes[cur_node.right_child & RIGHT_CHILD_BITS];

    if (child1.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.right_child & RIGHT_CHILD_BITS;
    } else {
        const light_bvh_node_t &child10 = nodes[child1.left_child];
        const light_bvh_node_t &child11 = nodes[child1.right_child & RIGHT_CHILD_BITS];

        if (child10.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child10.left_child;
            children[children_count++] = child10.right_child & RIGHT_CHILD_BITS;
        }

        if (child11.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child11.left_child;
            children[children_count++] = child11.right_child & RIGHT_CHILD_BITS;
        }
    }

    // Sort children in morton order
    Ref::fvec3 children_centers[8], whole_box_min = {FLT_MAX}, whole_box_max = {-FLT_MAX};
    for (int i = 0; i < children_count; i++) {
        children_centers[i] =
            0.5f * (Ref::fvec3{nodes[children[i]].bbox_min} + Ref::fvec3{nodes[children[i]].bbox_max});
        whole_box_min = min(whole_box_min, children_centers[i]);
        whole_box_max = max(whole_box_max, children_centers[i]);
    }

    whole_box_max += Ref::fvec3{0.001f};

    const Ref::fvec3 scale = 2.0f / (whole_box_max - whole_box_min);

    uint32_t sorted_children[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                   0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    for (int i = 0; i < children_count; i++) {
        Ref::fvec3 code = (children_centers[i] - whole_box_min) * scale;

        const auto x = uint32_t(code[0]), y = uint32_t(code[1]), z = uint32_t(code[2]);

        uint32_t mort = (z << 2) | (y << 1) | (x << 0);

        while (sorted_children[mort] != 0xffffffff) {
            mort = (mort + 1) % 8;
        }

        sorted_children[mort] = children[i];
    }

    uint32_t new_children[8];

    for (int i = 0; i < 8; i++) {
        if (sorted_children[i] != 0xffffffff) {
            new_children[i] = FlattenBVH_r(nodes, sorted_children[i], node_index, out_nodes);
        } else {
            new_children[i] = 0x7fffffff;
        }
    }

    light_wbvh_node_t &new_node = out_nodes[new_node_index];
    memcpy(new_node.child, new_children, sizeof(new_children));

    for (int i = 0; i < 8; i++) {
        if (new_children[i] != 0x7fffffff) {
            new_node.bbox_min[0][i] = nodes[sorted_children[i]].bbox_min[0];
            new_node.bbox_min[1][i] = nodes[sorted_children[i]].bbox_min[1];
            new_node.bbox_min[2][i] = nodes[sorted_children[i]].bbox_min[2];

            new_node.bbox_max[0][i] = nodes[sorted_children[i]].bbox_max[0];
            new_node.bbox_max[1][i] = nodes[sorted_children[i]].bbox_max[1];
            new_node.bbox_max[2][i] = nodes[sorted_children[i]].bbox_max[2];

            new_node.flux[i] = nodes[sorted_children[i]].flux;
            new_node.axis[0][i] = nodes[sorted_children[i]].axis[0];
            new_node.axis[1][i] = nodes[sorted_children[i]].axis[1];
            new_node.axis[2][i] = nodes[sorted_children[i]].axis[2];
            new_node.omega_n[i] = nodes[sorted_children[i]].omega_n;
            new_node.omega_e[i] = nodes[sorted_children[i]].omega_e;
        } else {
            // Init as invalid bounding box
            new_node.bbox_min[0][i] = new_node.bbox_min[1][i] = new_node.bbox_min[2][i] = 0.0f;
            new_node.bbox_max[0][i] = new_node.bbox_max[1][i] = new_node.bbox_max[2][i] = 0.0f;
            // Init as zero light
            new_node.flux[i] = 0.0f;
            new_node.axis[0][i] = new_node.axis[1][i] = new_node.axis[2][i] = 0.0f;
            new_node.omega_n[i] = new_node.omega_e[i] = 0.0f;
        }
    }

    return new_node_index;
}

bool Ray::NaiivePluckerTest(const float p[9], const float o[3], const float d[3]) {
    // plucker coordinates for edges
    const float e0[6] = {p[6] - p[0],
                         p[7] - p[1],
                         p[8] - p[2],
                         p[7] * p[2] - p[8] * p[1],
                         p[8] * p[0] - p[6] * p[2],
                         p[6] * p[1] - p[7] * p[0]},
                e1[6] = {p[3] - p[6],
                         p[4] - p[7],
                         p[5] - p[8],
                         p[4] * p[8] - p[5] * p[7],
                         p[5] * p[6] - p[3] * p[8],
                         p[3] * p[7] - p[4] * p[6]},
                e2[6] = {p[0] - p[3],
                         p[1] - p[4],
                         p[2] - p[5],
                         p[1] * p[5] - p[2] * p[4],
                         p[2] * p[3] - p[0] * p[5],
                         p[0] * p[4] - p[1] * p[3]};

    // plucker coordinates for Ray
    const float R[6] = {
        d[1] * o[2] - d[2] * o[1], d[2] * o[0] - d[0] * o[2], d[0] * o[1] - d[1] * o[0], d[0], d[1], d[2]};

    float t0 = 0, t1 = 0, t2 = 0;
    for (int w = 0; w < 6; w++) {
        t0 += e0[w] * R[w];
        t1 += e1[w] * R[w];
        t2 += e2[w] * R[w];
    }

    return (t0 <= 0 && t1 <= 0 && t2 <= 0) || (t0 >= 0 && t1 >= 0 && t2 >= 0);
}

void Ray::ConstructCamera(const eCamType type, const ePixelFilter filter, const float filter_width,
                          const eViewTransform view_transform, const float origin[3], const float fwd[3],
                          const float up[3], const float shift[2], const float fov, const float sensor_height,
                          const float exposure, const float gamma, const float focus_distance, const float fstop,
                          const float lens_rotation, const float lens_ratio, const int lens_blades,
                          const float clip_start, const float clip_end, camera_t *cam) {
    if (type == eCamType::Persp) {
        auto o = Ref::fvec3{origin}, f = Ref::fvec3{fwd}, u = Ref::fvec3{up};

        if (u.length2() < FLT_EPS) {
            if (fabsf(f[1]) >= 0.999f) {
                u = {1.0f, 0.0f, 0.0f};
            } else {
                u = {0.0f, 1.0f, 0.0f};
            }
        }

        const Ref::fvec3 s = normalize(cross(f, u));
        u = cross(s, f);

        cam->type = type;
        cam->filter = filter;
        cam->filter_width = filter_width;
        cam->view_transform = view_transform;
        cam->ltype = eLensUnits::FOV;
        cam->fov = fov;
        cam->exposure = exposure;
        cam->gamma = gamma;
        cam->sensor_height = sensor_height;
        cam->focus_distance = fmaxf(focus_distance, 0.0f);
        cam->focal_length = 0.5f * sensor_height / tanf(0.5f * fov * PI / 180.0f);
        cam->fstop = fstop;
        cam->lens_rotation = lens_rotation;
        cam->lens_ratio = lens_ratio;
        cam->lens_blades = lens_blades;
        cam->clip_start = clip_start;
        cam->clip_end = clip_end;
        o.store_to(cam->origin);
        f.store_to(cam->fwd);
        s.store_to(cam->side);
        u.store_to(cam->up);
        memcpy(&cam->shift[0], shift, 2 * sizeof(float));
    } else if (type == eCamType::Ortho) {
        // TODO!
    }
}

void Ray::TransformBoundingBox(const float bbox_min[3], const float bbox_max[3], const float *xform,
                               float out_bbox_min[3], float out_bbox_max[3]) {
    out_bbox_min[0] = out_bbox_max[0] = xform[12];
    out_bbox_min[1] = out_bbox_max[1] = xform[13];
    out_bbox_min[2] = out_bbox_max[2] = xform[14];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            const float a = xform[i * 4 + j] * bbox_min[i];
            const float b = xform[i * 4 + j] * bbox_max[i];

            if (a < b) {
                out_bbox_min[j] += a;
                out_bbox_max[j] += b;
            } else {
                out_bbox_min[j] += b;
                out_bbox_max[j] += a;
            }
        }
    }
}

void Ray::InverseMatrix(const float mat[16], float out_mat[16]) {
    const float A2323 = mat[10] * mat[15] - mat[11] * mat[14];
    const float A1323 = mat[9] * mat[15] - mat[11] * mat[13];
    const float A1223 = mat[9] * mat[14] - mat[10] * mat[13];
    const float A0323 = mat[8] * mat[15] - mat[11] * mat[12];
    const float A0223 = mat[8] * mat[14] - mat[10] * mat[12];
    const float A0123 = mat[8] * mat[13] - mat[9] * mat[12];
    const float A2313 = mat[6] * mat[15] - mat[7] * mat[14];
    const float A1313 = mat[5] * mat[15] - mat[7] * mat[13];
    const float A1213 = mat[5] * mat[14] - mat[6] * mat[13];
    const float A2312 = mat[6] * mat[11] - mat[7] * mat[10];
    const float A1312 = mat[5] * mat[11] - mat[7] * mat[9];
    const float A1212 = mat[5] * mat[10] - mat[6] * mat[9];
    const float A0313 = mat[4] * mat[15] - mat[7] * mat[12];
    const float A0213 = mat[4] * mat[14] - mat[6] * mat[12];
    const float A0312 = mat[4] * mat[11] - mat[7] * mat[8];
    const float A0212 = mat[4] * mat[10] - mat[6] * mat[8];
    const float A0113 = mat[4] * mat[13] - mat[5] * mat[12];
    const float A0112 = mat[4] * mat[9] - mat[5] * mat[8];

    const float inv_det = 1.0f / (mat[0] * (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223) -
                                  mat[1] * (mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223) +
                                  mat[2] * (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123) -
                                  mat[3] * (mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123));

    out_mat[0] = inv_det * (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223);
    out_mat[1] = inv_det * -(mat[1] * A2323 - mat[2] * A1323 + mat[3] * A1223);
    out_mat[2] = inv_det * (mat[1] * A2313 - mat[2] * A1313 + mat[3] * A1213);
    out_mat[3] = inv_det * -(mat[1] * A2312 - mat[2] * A1312 + mat[3] * A1212);
    out_mat[4] = inv_det * -(mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223);
    out_mat[5] = inv_det * (mat[0] * A2323 - mat[2] * A0323 + mat[3] * A0223);
    out_mat[6] = inv_det * -(mat[0] * A2313 - mat[2] * A0313 + mat[3] * A0213);
    out_mat[7] = inv_det * (mat[0] * A2312 - mat[2] * A0312 + mat[3] * A0212);
    out_mat[8] = inv_det * (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123);
    out_mat[9] = inv_det * -(mat[0] * A1323 - mat[1] * A0323 + mat[3] * A0123);
    out_mat[10] = inv_det * (mat[0] * A1313 - mat[1] * A0313 + mat[3] * A0113);
    out_mat[11] = inv_det * -(mat[0] * A1312 - mat[1] * A0312 + mat[3] * A0112);
    out_mat[12] = inv_det * -(mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123);
    out_mat[13] = inv_det * (mat[0] * A1223 - mat[1] * A0223 + mat[2] * A0123);
    out_mat[14] = inv_det * -(mat[0] * A1213 - mat[1] * A0213 + mat[2] * A0113);
    out_mat[15] = inv_det * (mat[0] * A1212 - mat[1] * A0212 + mat[2] * A0112);
}
