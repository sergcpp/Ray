#include "TextureUtilsRef.h"

#include <array>

std::vector<ray::pixel_color8_t> ray::ref::DownsampleTexture(const std::vector<pixel_color8_t> &_tex, const math::ivec2 &res) {
    const pixel_color8_t *tex = &_tex[0];

    // TODO: properly downsample non-power-of-2 textures

    std::vector<pixel_color8_t> ret;
    for (int j = 0; j < res.y; j += 2) {
        for (int i = 0; i < res.x; i += 2) {
            int r = tex[(j + 0) * res.x + i].r + tex[(j + 0) * res.x + i + 1].r +
                tex[(j + 1) * res.x + i].r + tex[(j + 1) * res.x + i + 1].r;
            int g = tex[(j + 0) * res.x + i].g + tex[(j + 0) * res.x + i + 1].g +
                tex[(j + 1) * res.x + i].g + tex[(j + 1) * res.x + i + 1].g;
            int b = tex[(j + 0) * res.x + i].b + tex[(j + 0) * res.x + i + 1].b +
                tex[(j + 1) * res.x + i].b + tex[(j + 1) * res.x + i + 1].b;
            int a = tex[(j + 0) * res.x + i].a + tex[(j + 0) * res.x + i + 1].a +
                tex[(j + 1) * res.x + i].a + tex[(j + 1) * res.x + i + 1].a;

            ret.push_back({ (uint8_t)std::round(r * 0.25f), (uint8_t)std::round(g * 0.25f),
                (uint8_t)std::round(b * 0.25f), (uint8_t)std::round(a * 0.25f)
            });
        }
    }
    return ret;
}

void ray::ref::ComputeTextureBasis(size_t vtx_offset, std::vector<vertex_t> &vertices, std::vector<uint32_t> &new_vtx_indices,
                                   const uint32_t *indices, size_t indices_count) {
    using namespace math;

    std::vector<std::array<uint32_t, 3>> twin_verts(vertices.size(), { 0, 0, 0 });
    aligned_vector<vec3> binormals(vertices.size());
    for (size_t i = 0; i < indices_count; i += 3) {
        auto *v0 = &vertices[indices[i + 0]];
        auto *v1 = &vertices[indices[i + 1]];
        auto *v2 = &vertices[indices[i + 2]];

        auto *b0 = &binormals[indices[i + 0]];
        auto *b1 = &binormals[indices[i + 1]];
        auto *b2 = &binormals[indices[i + 2]];

        vec3 dp1 = make_vec3(v1->p) - make_vec3(v0->p);
        vec3 dp2 = make_vec3(v2->p) - make_vec3(v0->p);

        vec2 dt1 = make_vec2(v1->t0) - make_vec2(v0->t0);
        vec2 dt2 = make_vec2(v2->t0) - make_vec2(v0->t0);

        float inv_det = 1.0f / (dt1.x * dt2.y - dt1.y * dt2.x);
        vec3 tangent = (dp1 * dt2.y - dp2 * dt1.y) * inv_det;
        vec3 binormal = (dp2 * dt1.x - dp1 * dt2.x) * inv_det;

        int i1 = v0->b[0] * tangent.x + v0->b[1] * tangent.y + v0->b[2] * tangent.z < 0;
        int i2 = 2 * (b0->x * binormal.x + b0->y * binormal.y + b0->z * binormal.z < 0);

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
            *b0 = binormal;
        }

        v0->b[0] += tangent.x;
        v0->b[1] += tangent.y;
        v0->b[2] += tangent.z;

        i1 = v1->b[0] * tangent.x + v1->b[1] * tangent.y + v1->b[2] * tangent.z < 0;
        i2 = 2 * (b1->x * binormal.x + b1->y * binormal.y + b1->z * binormal.z < 0);

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
            *b1 = binormal;
        }

        v1->b[0] += tangent.x;
        v1->b[1] += tangent.y;
        v1->b[2] += tangent.z;

        i1 = v2->b[0] * tangent.x + v2->b[1] * tangent.y + v2->b[2] * tangent.z < 0;
        i2 = 2 * (b2->x * binormal.x + b2->y * binormal.y + b2->z * binormal.z < 0);

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
            *b2 = binormal;
        }

        v2->b[0] += tangent.x;
        v2->b[1] += tangent.y;
        v2->b[2] += tangent.z;
    }

    for (auto &v : vertices) {
        vec3 tangent = make_vec3(v.b);
        vec3 binormal = normalize(cross(make_vec3(v.n), tangent));
        memcpy(&v.b[0], value_ptr(binormal), 3 * sizeof(float));

        if (std::isnan(binormal.x)) {
            //__debugbreak();
        }
    }
};