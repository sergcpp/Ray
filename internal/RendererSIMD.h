
#include "CoreSIMD.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace ray {
namespace NS {
template <int DimX, int DimY>
class RendererSIMD : public RendererBase {
    ray::ref::Framebuffer framebuf_;

    std::vector<pixel_color_t> color_table_;
public:
    RendererSIMD(int w, int h);

    std::pair<int, int> size() const override {
        return std::make_pair(framebuf_.w(), framebuf_.h());
    }

    const pixel_color_t *get_pixels_ref() const override {
        return framebuf_.get_pixels_ref();
    }

    void Resize(int w, int h) override {
        framebuf_.Resize(w, h);
    }
    void Clear(const pixel_color_t &c) override {
        framebuf_.Clear(c);
    }

    std::shared_ptr<SceneBase> CreateScene() override;
    void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) override;

    virtual void GetStats(stats_t &st) override {
        
    }
};
}
}

////////////////////////////////////////////////////////////////////////////////////////////

#include "SceneRef.h"

#include <math/math.hpp>

template <int DimX, int DimY>
ray::NS::RendererSIMD<DimX, DimY>::RendererSIMD(int w, int h) : framebuf_(w, h) {
    auto u_0_to_1 = []() {
        return float(rand()) / RAND_MAX;
    };

    for (int i = 0; i < 64; i++) {
        color_table_.push_back({ u_0_to_1(), u_0_to_1(), u_0_to_1(), 1 });
    }
}

template <int DimX, int DimY>
std::shared_ptr<ray::SceneBase> ray::NS::RendererSIMD<DimX, DimY>::CreateScene() {
    return std::make_shared<ref::Scene>();
}

template <int DimX, int DimY>
void ray::NS::RendererSIMD<DimX, DimY>::RenderScene(const std::shared_ptr<SceneBase> &_s, RegionContext &region) {
    using namespace math;

    const int S = DimX * DimY;

    auto s = std::dynamic_pointer_cast<ref::Scene>(_s);
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

    auto rect = region.rect();
    if (rect.w == 0 || rect.h == 0) {
        rect = { 0, 0, w, h };
    }

    region.iteration++;

    math::aligned_vector<ray_packet_t<S>> primary_rays;
    math::aligned_vector<hit_data_t<S>> intersections;

    GeneratePrimaryRays<DimX, DimY>(cam, rect, w, h, primary_rays);

    intersections.reserve(primary_rays.size());

    for (size_t i = 0; i < primary_rays.size(); i++) {
        hit_data_t<S> inter;
        inter.x = primary_rays[i].x;
        inter.y = primary_rays[i].y;

        const auto &r = primary_rays[i];
        const simd_fvec<S> inv_d[3] = { { 1.0f / r.d[0] },{ 1.0f / r.d[1] },{ 1.0f / r.d[2] } };

        if (NS::Traverse_MacroTree_CPU(r, { -1 }, inv_d, nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, inter)) {
            intersections.push_back(inter);
        }
    }

    const uint32_t col_table_mask = (uint32_t)color_table_.size() - 1;
    for (size_t i = 0; i < intersections.size(); i++) {
        const auto &ii = intersections[i];

        int x = ii.x;
        int y = ii.y;

        const pixel_color_t col1 = { 0.65f, 0.65f, 0.65f, 1 };

        for (int j = 0; j < S; j++) {
            if (ii.mask[j]) {
                const pixel_color_t col1 = color_table_[ii.prim_index[j] & col_table_mask];
                framebuf_.SetPixel(x + NS::ray_packet_pattern_x[j], y + NS::ray_packet_pattern_y[j], col1);
            }
        }
    }
}