#include "RendererAVX.h"

#include "SceneRef.h"

#include <math/math.hpp>

namespace ray {
namespace avx {
__m256i bbox_test(const __m256 o[3], const __m256 inv_d[3], const __m256 t, const float _bbox_min[3], const float _bbox_max[3]);

extern const __m256i FF_MASK;
}
}

ray::avx::Renderer::Renderer(int w, int h) : framebuf_(w, h) {
    auto u_0_to_1 = []() {
        return float(rand()) / RAND_MAX;
    };

    for (int i = 0; i < 64; i++) {
        color_table_.push_back({ u_0_to_1(), u_0_to_1(), u_0_to_1(), 1 });
    }
}

std::shared_ptr<ray::SceneBase> ray::avx::Renderer::CreateScene() {
    return std::make_shared<ref::Scene>();
}

void ray::avx::Renderer::RenderScene(const std::shared_ptr<SceneBase> &_s, region_t region) {
    using namespace math;

    const auto s = std::dynamic_pointer_cast<ref::Scene>(_s);
    if (!s) return;

    const auto &cam = s->GetCamera(s->current_cam());

    const auto num_tris = (uint32_t)s->tris_.size();
    const auto *tris = num_tris ? &s->tris_[0] : nullptr;

    const auto num_indices = (uint32_t)s->tri_indices_.size();
    const auto *tri_indices = num_indices ? &s->tri_indices_[0] : nullptr;

    const auto num_nodes = (uint32_t)s->nodes_.size();
    const auto *nodes = num_nodes ? &s->nodes_[0] : nullptr;

    const auto macro_tree_root = (uint32_t)s->macro_nodes_start_;

    const auto num_meshes = (uint32_t)s->meshes_.size();
    const auto *meshes = num_meshes ? &s->meshes_[0] : nullptr;

    const auto num_transforms = (uint32_t)s->transforms_.size();
    const auto *transforms = num_transforms ? &s->transforms_[0] : nullptr;

    const auto num_mesh_instances = (uint32_t)s->mesh_instances_.size();
    const auto *mesh_instances = num_mesh_instances ? &s->mesh_instances_[0] : nullptr;

    const auto num_mi_indices = (uint32_t)s->mi_indices_.size();
    const auto *mi_indices = num_mi_indices ? &s->mi_indices_[0] : nullptr;

    const int w = framebuf_.w(), h = framebuf_.h();

    if (region.w == 0 || region.h == 0) {
        region = { 0, 0, w, h };
    }

    math::aligned_vector<ray_packet_t> primary_rays;
    
    GeneratePrimaryRays(cam, region, w, h, primary_rays);

    math::aligned_vector<hit_data_t> intersections;
    intersections.reserve(primary_rays.size());

    for (size_t i = 0; i < primary_rays.size(); i++) {
        hit_data_t inter;
        inter.id = primary_rays[i].id;

        const ray_packet_t &r = primary_rays[i];
        __m256 inv_d[3] = { _mm256_div_ps(ONE, r.d[0]), _mm256_div_ps(ONE, r.d[1]), _mm256_div_ps(ONE, r.d[2]) };

        if (Traverse_MacroTree_CPU(r, FF_MASK, inv_d, nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, inter)) {
            intersections.push_back(inter);
        }
    }

    const uint32_t col_table_mask = (uint32_t)color_table_.size() - 1;
    for (size_t i = 0; i < intersections.size(); i++) {
        const auto &ii = intersections[i];

        int x = ii.id.x;
        int y = ii.id.y;

        // maybe slow, but it is the only right way to do it
        int mask_values[RayPacketSize], prim_indices[RayPacketSize];
        memcpy(&mask_values[0], &ii.mask, sizeof(int) * RayPacketSize);
        memcpy(&prim_indices[0], &ii.prim_index, sizeof(int) * RayPacketSize);

        if (mask_values[0]) {
            const pixel_color_t col1 = color_table_[prim_indices[0] & col_table_mask];
            framebuf_.SetPixel(x, y, col1);
        }
        if (mask_values[1]) {
            const pixel_color_t col1 = color_table_[prim_indices[1] & col_table_mask];
            framebuf_.SetPixel(x + 1, y, col1);
        }
        if (mask_values[2]) {
            const pixel_color_t col1 = color_table_[prim_indices[2] & col_table_mask];
            framebuf_.SetPixel(x, y + 1, col1);
        }
        if (mask_values[3]) {
            const pixel_color_t col1 = color_table_[prim_indices[3] & col_table_mask];
            framebuf_.SetPixel(x + 1, y + 1, col1);
        }
        if (mask_values[4]) {
            const pixel_color_t col1 = color_table_[prim_indices[4] & col_table_mask];
            framebuf_.SetPixel(x + 2, y, col1);
        }
        if (mask_values[5]) {
            const pixel_color_t col1 = color_table_[prim_indices[5] & col_table_mask];
            framebuf_.SetPixel(x + 3, y, col1);
        }
        if (mask_values[6]) {
            const pixel_color_t col1 = color_table_[prim_indices[6] & col_table_mask];
            framebuf_.SetPixel(x + 2, y + 1, col1);
        }
        if (mask_values[7]) {
            const pixel_color_t col1 = color_table_[prim_indices[7] & col_table_mask];
            framebuf_.SetPixel(x + 3, y + 1, col1);
        }
    }
}