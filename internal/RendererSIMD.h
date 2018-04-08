
#include "CoreSIMD.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace ray {
namespace NS {
template <int DimX, int DimY>
class RendererSIMD : public RendererBase {
    ray::ref::Framebuffer clean_buf_, final_buf_, temp_buf_;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);
public:
    RendererSIMD(int w, int h);

    std::pair<int, int> size() const override {
        return std::make_pair(final_buf_.w(), final_buf_.h());
    }

    const pixel_color_t *get_pixels_ref() const override {
        return final_buf_.get_pixels_ref();
    }

    void Resize(int w, int h) override {
        clean_buf_.Resize(w, h);
        final_buf_.Resize(w, h);
        temp_buf_.Resize(w, h);
    }
    void Clear(const pixel_color_t &c) override {
        clean_buf_.Clear(c);
    }

    std::shared_ptr<SceneBase> CreateScene() override;
    void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) override;

    virtual void GetStats(stats_t &st) override {
        
    }
};
}
}

////////////////////////////////////////////////////////////////////////////////////////////

#include "SceneRef2.h"

#include <math/math.hpp>

template <int DimX, int DimY>
ray::NS::RendererSIMD<DimX, DimY>::RendererSIMD(int w, int h) : clean_buf_(w, h), final_buf_(w, h), temp_buf_(w, h) {
    auto rand_func = std::bind(std::uniform_int_distribution<int>(), std::mt19937(0));
    permutations_ = ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

template <int DimX, int DimY>
std::shared_ptr<ray::SceneBase> ray::NS::RendererSIMD<DimX, DimY>::CreateScene() {
    return std::make_shared<ref::Scene2>();
}

template <int DimX, int DimY>
void ray::NS::RendererSIMD<DimX, DimY>::RenderScene(const std::shared_ptr<SceneBase> &_s, RegionContext &region) {
    using namespace math;

    const int S = DimX * DimY;

    auto s = std::dynamic_pointer_cast<ref::Scene2>(_s);
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

    const auto num_vertices = (uint32_t)s->vertices_.size();
    const auto *vertices = num_vertices ? &s->vertices_[0] : nullptr;

    const auto num_vtx_indices = (uint32_t)s->vtx_indices_.size();
    const auto *vtx_indices = num_vtx_indices ? &s->vtx_indices_[0] : nullptr;

    const auto num_textures = (uint32_t)s->textures_.size();
    const auto *textures = num_textures ? &s->textures_[0] : nullptr;

    const auto num_materials = (uint32_t)s->materials_.size();
    const auto *materials = num_materials ? &s->materials_[0] : nullptr;

    const auto &tex_atlas = s->texture_atlas_;
    //const auto &env = s->env_;

    NS::environment_t env;
    memcpy(&env.sun_dir[0], &s->env_.sun_dir[0], 3 * sizeof(float));
    memcpy(&env.sun_col[0], &s->env_.sun_col[0], 3 * sizeof(float));
    memcpy(&env.sky_col[0], &s->env_.sky_col[0], 3 * sizeof(float));
    env.sun_softness = s->env_.sun_softness;

    const auto w = final_buf_.w(), h = final_buf_.h();

    auto rect = region.rect();
    if (rect.w == 0 || rect.h == 0) {
        rect = { 0, 0, w, h };
    }

    region.iteration++;
    if (!region.halton_seq || region.iteration % HaltonSeqLen == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    math::aligned_vector<ray_packet_t<S>> primary_rays;

    GeneratePrimaryRays<DimX, DimY>(cam, rect, w, h, primary_rays);

    math::aligned_vector<hit_data_t<S>> intersections(primary_rays.size());

    for (size_t i = 0; i < primary_rays.size(); i++) {
        const auto &r = primary_rays[i];

        intersections[i].x = r.x;
        intersections[i].y = r.y;
        NS::Traverse_MacroTree_CPU(r, { -1 }, nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, intersections[i]);
    }

    for (size_t i = 0; i < intersections.size(); i++) {
        const auto &r = primary_rays[i];
        const auto &inter = intersections[i];

        int x = inter.x;
        int y = inter.y;

        simd_ivec<S> index = { y * w + x };
        index += { ray_packet_layout_x };
        index += w * simd_ivec<S>{ ray_packet_layout_y };

        simd_fvec<S> out_rgba[4];
        NS::ShadeSurface<S>(index, region.iteration, &region.halton_seq[0], inter, r, env, mesh_instances,
                         mi_indices, meshes, transforms, vtx_indices, vertices, nodes, macro_tree_root,
                         tris, tri_indices, materials, textures, tex_atlas, out_rgba, nullptr, nullptr);

        for (int j = 0; j < S; j++) {
            clean_buf_.SetPixel(x + ray_packet_layout_x[j], y + ray_packet_layout_y[j], { out_rgba[0][j], out_rgba[1][j], out_rgba[2][j], out_rgba[3][j] });
        }
    }

    auto clamp_and_gamma_correct = [](const pixel_color_t &p) {
        auto c = make_vec4(&p.r);
        c = pow(c, vec4(1.0f / 2.2f));
        c = clamp(c, 0.0f, 1.0f);
        return pixel_color_t{ c.r, c.g, c.b, c.a };
    };

    final_buf_.CopyFrom(clean_buf_, rect, clamp_and_gamma_correct);
}

template <int DimX, int DimY>
void ray::NS::RendererSIMD<DimX, DimY>::UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HaltonSeqLen * 2]);
    }

    for (int i = 0; i < HaltonSeqLen; i++) {
        seq[i * 2 + 0] = ray::ScrambledRadicalInverse<29>(&permutations_[100], (uint64_t)(iteration + i));
        seq[i * 2 + 1] = ray::ScrambledRadicalInverse<31>(&permutations_[129], (uint64_t)(iteration + i));
    }
}