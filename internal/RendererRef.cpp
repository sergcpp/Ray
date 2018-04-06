#include "RendererRef.h"

#include <random>

#include "Halton.h"
#include "SceneRef.h"

#include <math/math.hpp>

ray::ref::Renderer::Renderer(int w, int h) : clean_buf_(w, h), final_buf_(w, h), temp_buf_(w, h) {
    auto u_0_to_1 = []() {
        return float(rand()) / RAND_MAX;
    };

    for (int i = 0; i < 64; i++) {
        color_table_.push_back({ u_0_to_1(), u_0_to_1(), u_0_to_1(), 1 });
    }

    auto rand_func = std::bind(std::uniform_int_distribution<int>(), std::mt19937(0));
    permutations_ = ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

std::shared_ptr<ray::SceneBase> ray::ref::Renderer::CreateScene() {
    return std::make_shared<ref::Scene>();
}

void ray::ref::Renderer::RenderScene(const std::shared_ptr<SceneBase> &_s, RegionContext &region) {
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

    const auto num_vertices = (uint32_t)s->vertices_.size();
    const auto *vertices = num_vertices ? &s->vertices_[0] : nullptr;

    const auto num_vtx_indices = (uint32_t)s->vtx_indices_.size();
    const auto *vtx_indices = num_vtx_indices ? &s->vtx_indices_[0] : nullptr;

    const auto num_textures = (uint32_t)s->textures_.size();
    const auto *textures = num_textures ? &s->textures_[0] : nullptr;

    const auto num_materials = (uint32_t)s->materials_.size();
    const auto *materials = num_materials ? &s->materials_[0] : nullptr;

    const auto &tex_atlas = s->texture_atlas_;
    const auto &env = s->env_;

    const auto w = final_buf_.w(), h = final_buf_.h();

    auto rect = region.rect();
    if (rect.w == 0 || rect.h == 0) {
        rect = { 0, 0, w, h };
    }

    region.iteration++;
    if (!region.halton_seq || region.iteration % HaltonSeqLen == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    aligned_vector<ray_packet_t> primary_rays;

    GeneratePrimaryRays(region.iteration, cam, rect, w, h, &region.halton_seq[0], primary_rays);

    aligned_vector<hit_data_t> intersections(primary_rays.size());

    for (size_t i = 0; i < primary_rays.size(); i++) {
        const ray_packet_t &r = primary_rays[i];
        auto inv_d = safe_invert(make_vec3(r.d));

        intersections[i].id = r.id;
        Traverse_MacroTree_CPU(r, value_ptr(inv_d), nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, intersections[i]);
    }

    aligned_vector<ray_packet_t> secondary_rays(intersections.size());
    int secondary_rays_count = 0;

    for (size_t i = 0; i < intersections.size(); i++) {
        const auto &r = primary_rays[i];
        const auto &inter = intersections[i];

        const int x = inter.id.x;
        const int y = inter.id.y;
        
        pixel_color_t col = ShadeSurface((y * w + x), region.iteration, &region.halton_seq[0], inter, r, env, mesh_instances, 
                                         mi_indices, meshes, transforms, vtx_indices, vertices, nodes, macro_tree_root,
                                         tris, tri_indices, materials, textures, tex_atlas, &secondary_rays[0], &secondary_rays_count);
        temp_buf_.SetPixel(x, y, col);
    }

    for (int bounce = 0; bounce < 4 && secondary_rays_count; bounce++) {
        for (int i = 0; i < secondary_rays_count; i++) {
            const auto &r = secondary_rays[i];
            auto inv_d = safe_invert(make_vec3(r.d));

            intersections[i] = {};
            intersections[i].id = r.id;
            Traverse_MacroTree_CPU(r, value_ptr(inv_d), nodes, macro_tree_root, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, intersections[i]);
        }

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        std::swap(primary_rays, secondary_rays);

        for (int i = 0; i < rays_count; i++) {
            const auto &r = primary_rays[i];
            const auto &inter = intersections[i];

            const int x = inter.id.x;
            const int y = inter.id.y;

            pixel_color_t col = ShadeSurface((y * w + x), region.iteration, &region.halton_seq[0], inter, r, env, mesh_instances,
                                             mi_indices, meshes, transforms, vtx_indices, vertices, nodes, macro_tree_root,
                                             tris, tri_indices, materials, textures, tex_atlas, &secondary_rays[0], &secondary_rays_count);

            temp_buf_.AddPixel(x, y, col);
        }
    }

    clean_buf_.MixIncremental(temp_buf_, rect, 1.0f / region.iteration);

    auto clamp_and_gamma_correct = [](const pixel_color_t &p) {
        auto c = make_vec4(&p.r);
        c = pow(c, vec4(1.0f / 2.2f));
        c = clamp(c, 0.0f, 1.0f);
        return pixel_color_t{ c.r, c.g, c.b, c.a };
    };

    final_buf_.CopyFrom(clean_buf_, rect, clamp_and_gamma_correct);
}

void ray::ref::Renderer::UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HaltonSeqLen * 2]);
    }

    for (int i = 0; i < HaltonSeqLen; i++) {
        seq[i * 2 + 0] = ray::ScrambledRadicalInverse<29>(&permutations_[100], (uint64_t)(iteration + i));
        seq[i * 2 + 1] = ray::ScrambledRadicalInverse<31>(&permutations_[129], (uint64_t)(iteration + i));
    }
}