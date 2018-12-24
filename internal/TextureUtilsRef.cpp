#include "TextureUtilsRef.h"

#include "CoreRef.h"

#include <cmath>

#include <array>

std::vector<Ray::pixel_color8_t> Ray::Ref::DownsampleTexture(const std::vector<pixel_color8_t> &_tex, const int res[2]) {
    if (res[0] == 1 || res[1] == 1) return _tex;
    
    const pixel_color8_t *tex = &_tex[0];

    // TODO: properly downsample non-power-of-2 textures

    std::vector<pixel_color8_t> ret;
    for (int j = 0; j < res[1]; j += 2) {
        for (int i = 0; i < res[0]; i += 2) {
            int r = tex[(j + 0) * res[0] + i].r + tex[(j + 0) * res[0] + i + 1].r +
                    tex[(j + 1) * res[0] + i].r + tex[(j + 1) * res[0] + i + 1].r;
            int g = tex[(j + 0) * res[0] + i].g + tex[(j + 0) * res[0] + i + 1].g +
                    tex[(j + 1) * res[0] + i].g + tex[(j + 1) * res[0] + i + 1].g;
            int b = tex[(j + 0) * res[0] + i].b + tex[(j + 0) * res[0] + i + 1].b +
                    tex[(j + 1) * res[0] + i].b + tex[(j + 1) * res[0] + i + 1].b;
            int a = tex[(j + 0) * res[0] + i].a + tex[(j + 0) * res[0] + i + 1].a +
                    tex[(j + 1) * res[0] + i].a + tex[(j + 1) * res[0] + i + 1].a;

            ret.push_back({ (uint8_t)std::round(r * 0.25f), (uint8_t)std::round(g * 0.25f),
                (uint8_t)std::round(b * 0.25f), (uint8_t)std::round(a * 0.25f)
            });
        }
    }
    return ret;
}

void Ray::Ref::ComputeTextureBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices, std::vector<uint32_t> &new_vtx_indices,
                                   const uint32_t *indices, size_t indices_count) {

    std::vector<std::array<uint32_t, 3>> twin_verts(vertices.size(), { 0, 0, 0 });
    aligned_vector<simd_fvec3> binormals(vertices.size());
    for (size_t i = 0; i < indices_count; i += 3) {
        auto *v0 = &vertices[indices[i + 0]];
        auto *v1 = &vertices[indices[i + 1]];
        auto *v2 = &vertices[indices[i + 2]];

        auto &b0 = binormals[indices[i + 0]];
        auto &b1 = binormals[indices[i + 1]];
        auto &b2 = binormals[indices[i + 2]];

        simd_fvec3 dp1 = simd_fvec3(v1->p) - simd_fvec3(v0->p);
        simd_fvec3 dp2 = simd_fvec3(v2->p) - simd_fvec3(v0->p);

        simd_fvec2 dt1 = simd_fvec2(v1->t[0]) - simd_fvec2(v0->t[0]);
        simd_fvec2 dt2 = simd_fvec2(v2->t[0]) - simd_fvec2(v0->t[0]);

        simd_fvec3 tangent, binormal;

        float det = dt1[0] * dt2[1] - dt1[1] * dt2[0];
        if (std::abs(det) > FLT_EPS) {
            float inv_det = 1.0f / det;
            tangent = (dp1 * dt2[1] - dp2 * dt1[1]) * inv_det;
            binormal = (dp2 * dt1[0] - dp1 * dt2[0]) * inv_det;
        } else {
            simd_fvec3 plane_N = cross(dp1, dp2);
            tangent = simd_fvec3{ 0.0f, 1.0f, 0.0f };
            if (std::abs(plane_N[0]) <= std::abs(plane_N[1]) && std::abs(plane_N[0]) <= std::abs(plane_N[2])) {
                tangent = simd_fvec3{ 1.0f, 0.0f, 0.0f };
            } else if (std::abs(plane_N[2]) <= std::abs(plane_N[0]) && std::abs(plane_N[2]) <= std::abs(plane_N[1])) {
                tangent = simd_fvec3{ 0.0f, 0.0f, 1.0f };
            }

            binormal = normalize(cross(simd_fvec3(plane_N), tangent));
            tangent = normalize(cross(simd_fvec3(plane_N), binormal));
        }

        int i1 = (v0->b[0] * tangent[0] + v0->b[1] * tangent[1] + v0->b[2] * tangent[2]) < 0;
        int i2 = 2 * (b0[0] * binormal[0] + b0[1] * binormal[1] + b0[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 0]][i1 + i2 - 1];
            if (index == 0) {
                index = (uint32_t)(vtx_offset + vertices.size());
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
                index = (uint32_t)(vtx_offset + vertices.size());
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
                index = (uint32_t)(vtx_offset + vertices.size());
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

    auto cross = [](const simd_fvec3 &v1, const simd_fvec3 &v2) -> simd_fvec3 {
        return simd_fvec3{ v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0] };
    };

    for (size_t i = vtx_start; i < vertices.size(); i++) {
        auto &v = vertices[i];

        if (std::abs(v.b[0]) > FLT_EPS || std::abs(v.b[1]) > FLT_EPS || std::abs(v.b[2]) > FLT_EPS) {
            simd_fvec3 tangent = { v.b };
            simd_fvec3 binormal = cross(simd_fvec3(v.n), tangent);
            float l = length(binormal);
            if (l > FLT_EPS) {
                binormal /= l;
                memcpy(&v.b[0], &binormal[0], 3 * sizeof(float));
            }
        }
    }
}
