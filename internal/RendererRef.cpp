#include "RendererRef.h"

#include "SceneRef.h"

#include <math/math.hpp>

namespace ray {
namespace ref {
bool bbox_test(const float o[3], const float inv_d[3], const float t, const float bbox_min[3], const float bbox_max[3]);
}
}

ray::ref::Renderer::Renderer(int w, int h) : framebuf_(w, h) {
    auto u_0_to_1 = []() {
        return float(rand()) / RAND_MAX;
    };

    for (int i = 0; i < 64; i++) {
        color_table_.push_back({ u_0_to_1(), u_0_to_1(), u_0_to_1(), 1 });
    }
}

std::shared_ptr<ray::SceneBase> ray::ref::Renderer::CreateScene() {
    return std::make_shared<ref::Scene>();
}

void ray::ref::Renderer::RenderScene(const std::shared_ptr<SceneBase> &_s, const region_t &region) {
    using namespace math;

    auto s = std::dynamic_pointer_cast<ref::Scene>(_s);
    if (!s) return;

    const auto &cam = s->GetCamera(s->current_cam());

    int num_tris = (int)s->tris_.size();
    const auto *tris = num_tris ? &s->tris_[0] : nullptr;

    int num_indices = (int)s->tri_indices_.size();
    const auto *tri_indices = num_indices ? &s->tri_indices_[0] : nullptr;

    int num_nodes = (int)s->nodes_.size();
    const auto *nodes = num_nodes ? &s->nodes_[0] : nullptr;

    int macro_tree_root = (int)s->macro_nodes_start_;

    int num_meshes = (int)s->meshes_.size();
    const auto *meshes = num_meshes ? &s->meshes_[0] : nullptr;

    int num_transforms = (int)s->transforms_.size();
    const auto *transforms = num_transforms ? &s->transforms_[0] : nullptr;

    int num_mesh_instances = (int)s->mesh_instances_.size();
    const auto *mesh_instances = num_mesh_instances ? &s->mesh_instances_[0] : nullptr;

    int num_mi_indices = (int)s->mi_indices_.size();
    const auto *mi_indices = num_mi_indices ? &s->mi_indices_[0] : nullptr;

    int w = framebuf_.w(), h = framebuf_.h();

    math::aligned_vector<ray_packet_t> primary_rays;
    math::aligned_vector<hit_data_t> intersections;

    GeneratePrimaryRays(cam, region, w, h, primary_rays);

    intersections.reserve(primary_rays.size());

    for (size_t i = 0; i < primary_rays.size(); i++) {
        hit_data_t inter;
        inter.id = primary_rays[i].id;

        const ray_packet_t &r = primary_rays[i];
        float inv_d[3] = { 1.0f / r.d[0], 1.0f / r.d[1], 1.0f / r.d[2] };

        if (Traverse_MacroTree_CPU(r, inv_d, nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, inter)) {
            intersections.push_back(inter);
        }
    }

    for (size_t i = 0; i < intersections.size(); i++) {
        const auto &ii = intersections[i];

        int x = ii.id.x;
        int y = ii.id.y;

        const pixel_color_t col1 = color_table_[ii.prim_indices[0] % color_table_.size()];

        if (ii.mask_values[0]) {
            framebuf_.SetPixel(x, y, col1);
        }
    }
}
