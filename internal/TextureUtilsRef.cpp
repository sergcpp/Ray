#include "TextureUtilsRef.h"

#include "CoreRef.h"

#include <cassert>
#include <cmath>

#include <array>

void Ray::Ref::ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                                   std::vector<uint32_t> &new_vtx_indices, const uint32_t *indices,
                                   size_t indices_count) {
    auto cross = [](const simd_fvec3 &v1, const simd_fvec3 &v2) -> simd_fvec3 {
        return simd_fvec3{v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
    };

    std::vector<std::array<uint32_t, 3>> twin_verts(vertices.size(), {0, 0, 0});
    aligned_vector<simd_fvec3> binormals(vertices.size());
    for (size_t i = 0; i < indices_count; i += 3) {
        vertex_t *v0 = &vertices[indices[i + 0]];
        vertex_t *v1 = &vertices[indices[i + 1]];
        vertex_t *v2 = &vertices[indices[i + 2]];

        simd_fvec3 &b0 = binormals[indices[i + 0]];
        simd_fvec3 &b1 = binormals[indices[i + 1]];
        simd_fvec3 &b2 = binormals[indices[i + 2]];

        const simd_fvec3 dp1 = simd_fvec3(v1->p) - simd_fvec3(v0->p);
        const simd_fvec3 dp2 = simd_fvec3(v2->p) - simd_fvec3(v0->p);

        const simd_fvec2 dt1 = simd_fvec2(v1->t[0]) - simd_fvec2(v0->t[0]);
        const simd_fvec2 dt2 = simd_fvec2(v2->t[0]) - simd_fvec2(v0->t[0]);

        simd_fvec3 tangent, binormal;

        const float det = dt1[0] * dt2[1] - dt1[1] * dt2[0];
        if (std::abs(det) > FLT_EPS) {
            const float inv_det = 1.0f / det;
            tangent = (dp1 * dt2[1] - dp2 * dt1[1]) * inv_det;
            binormal = (dp2 * dt1[0] - dp1 * dt2[0]) * inv_det;
        } else {
            simd_fvec3 plane_N = cross(dp1, dp2);

            int w = 2;
            tangent = simd_fvec3{0.0f, 1.0f, 0.0f};
            if (std::abs(plane_N[0]) <= std::abs(plane_N[1]) && std::abs(plane_N[0]) <= std::abs(plane_N[2])) {
                tangent = simd_fvec3{1.0f, 0.0f, 0.0f};
                w = 1;
            } else if (std::abs(plane_N[2]) <= std::abs(plane_N[0]) && std::abs(plane_N[2]) <= std::abs(plane_N[1])) {
                tangent = simd_fvec3{0.0f, 0.0f, 1.0f};
                w = 0;
            }

            if (std::abs(plane_N[w]) > FLT_EPS) {
                binormal = normalize(cross(simd_fvec3(plane_N), tangent));
                tangent = normalize(cross(simd_fvec3(plane_N), binormal));

                // avoid floating-point underflow
                where(abs(binormal) < FLT_EPS, binormal) = 0.0f;
                where(abs(tangent) < FLT_EPS, tangent) = 0.0f;
            } else {
                binormal = {0.0f};
                tangent = {0.0f};
            }
        }

        int i1 = (v0->b[0] * tangent[0] + v0->b[1] * tangent[1] + v0->b[2] * tangent[2]) < 0;
        int i2 = 2 * (b0[0] * binormal[0] + b0[1] * binormal[1] + b0[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 0]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v0);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 0]][i1 + i2 - 1] = index;

                v1 = &vertices[indices[i + 1]];
                v2 = &vertices[indices[i + 2]];
            }
            new_vtx_indices[i] = index;
            v0 = &vertices[index - vtx_offset];
        } else {
            b0 = binormal;
        }

        v0->b[0] += tangent[0];
        v0->b[1] += tangent[1];
        v0->b[2] += tangent[2];

        i1 = v1->b[0] * tangent[0] + v1->b[1] * tangent[1] + v1->b[2] * tangent[2] < 0;
        i2 = 2 * (b1[0] * binormal[0] + b1[1] * binormal[1] + b1[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 1]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v1);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 1]][i1 + i2 - 1] = index;

                v0 = &vertices[indices[i + 0]];
                v2 = &vertices[indices[i + 2]];
            }
            new_vtx_indices[i + 1] = index;
            v1 = &vertices[index - vtx_offset];
        } else {
            b1 = binormal;
        }

        v1->b[0] += tangent[0];
        v1->b[1] += tangent[1];
        v1->b[2] += tangent[2];

        i1 = v2->b[0] * tangent[0] + v2->b[1] * tangent[1] + v2->b[2] * tangent[2] < 0;
        i2 = 2 * (b2[0] * binormal[0] + b2[1] * binormal[1] + b2[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 2]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v2);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 2]][i1 + i2 - 1] = index;

                v0 = &vertices[indices[i + 0]];
                v1 = &vertices[indices[i + 1]];
            }
            new_vtx_indices[i + 2] = index;
            v2 = &vertices[index - vtx_offset];
        } else {
            b2 = binormal;
        }

        v2->b[0] += tangent[0];
        v2->b[1] += tangent[1];
        v2->b[2] += tangent[2];
    }

    for (size_t i = vtx_start; i < vertices.size(); i++) {
        vertex_t &v = vertices[i];

        if (std::abs(v.b[0]) > FLT_EPS || std::abs(v.b[1]) > FLT_EPS || std::abs(v.b[2]) > FLT_EPS) {
            const auto tangent = simd_fvec3{v.b};
            simd_fvec3 binormal = cross(simd_fvec3(v.n), tangent);
            const float l = length(binormal);
            if (l > FLT_EPS) {
                binormal /= l;
                memcpy(&v.b[0], value_ptr(binormal), 3 * sizeof(float));
            }
        }
    }
}
