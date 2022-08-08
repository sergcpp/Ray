
#include <chrono>
#include <functional>
#include <mutex>
#include <random>

#include "../RendererBase.h"
#include "CoreSIMD.h"
#include "FramebufferRef.h"
#include "Halton.h"

// TODO: remove this!!!
#include "CoreRef.h"

namespace Ray {
class ILog;
namespace NS {
template <int S> struct PassData {
    aligned_vector<ray_packet_t<S>> primary_rays;
    aligned_vector<simd_ivec<S>> primary_masks;
    aligned_vector<ray_packet_t<S>> secondary_rays;
    aligned_vector<simd_ivec<S>> secondary_masks;
    aligned_vector<hit_data_t<S>> intersections;

    aligned_vector<simd_ivec<S>> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;
    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;

    PassData() = default;

    PassData(const PassData &rhs) = delete;
    PassData(PassData &&rhs) { *this = std::move(rhs); }

    PassData &operator=(const PassData &rhs) = delete;
    PassData &operator=(PassData &&rhs) {
        primary_rays = std::move(rhs.primary_rays);
        primary_masks = std::move(rhs.primary_masks);
        secondary_rays = std::move(rhs.secondary_rays);
        intersections = std::move(rhs.intersections);
        hash_values = std::move(rhs.hash_values);
        head_flags = std::move(rhs.head_flags);
        scan_values = std::move(rhs.scan_values);
        chunks = std::move(rhs.chunks);
        chunks_temp = std::move(rhs.chunks_temp);
        skeleton = std::move(rhs.skeleton);
        return *this;
    }
};

template <int DimX, int DimY> class RendererSIMD : public RendererBase {
    ILog *log_;
    Ref::Framebuffer clean_buf_, final_buf_, temp_buf_;

    std::mutex pass_cache_mtx_;
    std::vector<PassData<DimX * DimY>> pass_cache_;

    bool use_wide_bvh_;
    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    RendererSIMD(const settings_t &s, ILog *log);

    std::pair<int, int> size() const override { return std::make_pair(final_buf_.w(), final_buf_.h()); }

    const pixel_color_t *get_pixels_ref() const override { return final_buf_.get_pixels_ref(); }

    const shl1_data_t *get_sh_data_ref() const override { return clean_buf_.get_sh_data_ref(); }

    void Resize(const int w, const int h) override {
        if (w_ != w || h_ != h) {
            clean_buf_.Resize(w, h, false);
            final_buf_.Resize(w, h, false);
            temp_buf_.Resize(w, h, false);

            w_ = w;
            h_ = h;
        }
    }
    void Clear(const pixel_color_t &c) override { clean_buf_.Clear(c); }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;

    virtual void GetStats(stats_t &st) override { st = stats_; }
    virtual void ResetStats() override { stats_ = {0}; }
};
} // namespace NS
} // namespace Ray

////////////////////////////////////////////////////////////////////////////////////////////

#include "SceneRef.h"
#include "UniformIntDistribution.h"

template <int DimX, int DimY>
Ray::NS::RendererSIMD<DimX, DimY>::RendererSIMD(const settings_t &s, ILog *log)
    : log_(log), clean_buf_(s.w, s.h), final_buf_(s.w, s.h), temp_buf_(s.w, s.h), use_wide_bvh_(s.use_wide_bvh) {
    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

template <int DimX, int DimY> Ray::SceneBase *Ray::NS::RendererSIMD<DimX, DimY>::CreateScene() {
    return new Ref::Scene(log_, use_wide_bvh_);
}

template <int DimX, int DimY>
void Ray::NS::RendererSIMD<DimX, DimY>::RenderScene(const SceneBase *_s, RegionContext &region) {
    const int S = DimX * DimY;

    const auto s = dynamic_cast<const Ref::Scene *>(_s);
    if (!s) {
        return;
    }

    const camera_t &cam = s->cams_[s->current_cam()].cam;

    scene_data_t sc_data;

    sc_data.env = &s->env_;
    sc_data.mesh_instances = s->mesh_instances_.empty() ? nullptr : &s->mesh_instances_[0];
    sc_data.mi_indices = s->mi_indices_.empty() ? nullptr : &s->mi_indices_[0];
    sc_data.meshes = s->meshes_.empty() ? nullptr : &s->meshes_[0];
    sc_data.transforms = s->transforms_.empty() ? nullptr : &s->transforms_[0];
    sc_data.vtx_indices = s->vtx_indices_.empty() ? nullptr : &s->vtx_indices_[0];
    sc_data.vertices = s->vertices_.empty() ? nullptr : &s->vertices_[0];
    sc_data.nodes = s->nodes_.empty() ? nullptr : &s->nodes_[0];
    sc_data.mnodes = s->mnodes_.empty() ? nullptr : &s->mnodes_[0];
    sc_data.tris = s->tris_.empty() ? nullptr : &s->tris_[0];
    sc_data.tri_indices = s->tri_indices_.empty() ? nullptr : &s->tri_indices_[0];
    sc_data.tri_materials = s->tri_materials_.empty() ? nullptr : &s->tri_materials_[0];
    sc_data.materials = s->materials_.empty() ? nullptr : &s->materials_[0];
    sc_data.textures = s->textures_.empty() ? nullptr : &s->textures_[0];
    sc_data.lights = s->lights_.empty() ? nullptr : &s->lights_[0];
    sc_data.li_indices = {s->li_indices_.data(), s->li_indices_.size()};
    sc_data.visible_lights = {s->visible_lights_.data(), s->visible_lights_.size()};

    const uint32_t macro_tree_root = s->macro_nodes_root_;

    const Ref::TextureAtlasBase *tex_atlases[] = {&s->tex_atlas_rgba_, &s->tex_atlas_rgb_, &s->tex_atlas_rg_,
                                                  &s->tex_atlas_r_};

    float root_min[3], cell_size[3];
    if (macro_tree_root != 0xffffffff) {
        float root_max[3];

        if (sc_data.mnodes) {
            const mbvh_node_t &root_node = sc_data.mnodes[macro_tree_root];

            root_min[0] = root_min[1] = root_min[2] = MAX_DIST;
            root_max[0] = root_max[1] = root_max[2] = -MAX_DIST;

            if (root_node.child[0] & LEAF_NODE_BIT) {
                ITERATE_3({ root_min[i] = root_node.bbox_min[i][0]; })
                ITERATE_3({ root_max[i] = root_node.bbox_max[i][0]; })
            } else {
                for (int j = 0; j < 8; j++) {
                    if (root_node.child[j] == 0x7fffffff) {
                        continue;
                    }

                    ITERATE_3({ root_min[i] = root_node.bbox_min[i][j]; })
                    ITERATE_3({ root_max[i] = root_node.bbox_max[i][j]; })
                }
            }
        } else {
            const bvh_node_t &root_node = sc_data.nodes[macro_tree_root];

            ITERATE_3({ root_min[i] = root_node.bbox_min[i]; })
            ITERATE_3({ root_max[i] = root_node.bbox_max[i]; })
        }

        ITERATE_3({ cell_size[i] = (root_max[i] - root_min[i]) / 255; })
    }

    const int w = final_buf_.w(), h = final_buf_.h();

    rect_t rect = region.rect();
    if (rect.w == 0 || rect.h == 0) {
        rect = {0, 0, w, h};
    }

    region.iteration++;
    if (!region.halton_seq || region.iteration % HALTON_SEQ_LEN == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    PassData<S> p;

    {
        std::lock_guard<std::mutex> _(pass_cache_mtx_);
        if (!pass_cache_.empty()) {
            p = std::move(pass_cache_.back());
            pass_cache_.pop_back();
        }

        // allocate sh data on demand
        if (cam.pass_settings.flags & OutputSH) {
            temp_buf_.Resize(w, h, true);
            clean_buf_.Resize(w, h, true);
        }
    }

    pass_info_t pass_info;

    pass_info.iteration = region.iteration;
    pass_info.bounce = 0;
    pass_info.settings = cam.pass_settings;
    pass_info.settings.max_total_depth = std::min(pass_info.settings.max_total_depth, (uint8_t)MAX_BOUNCES);

    const auto time_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> time_after_ray_gen;

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    if (cam.type != Geo) {
        GeneratePrimaryRays<DimX, DimY>(region.iteration, cam, rect, w, h, &region.halton_seq[hi], p.primary_rays,
                                        p.primary_masks);

        time_after_ray_gen = std::chrono::high_resolution_clock::now();

        p.intersections.resize(p.primary_rays.size());

        for (size_t i = 0; i < p.primary_rays.size(); i++) {
            const ray_packet_t<S> &r = p.primary_rays[i];
            hit_data_t<S> &inter = p.intersections[i];
            inter = {};

            if (macro_tree_root != 0xffffffff) {
                if (sc_data.mnodes) {
                    NS::Traverse_MacroTree_WithStack_ClosestHit(r, p.primary_masks[i], sc_data.mnodes, macro_tree_root,
                                                                sc_data.mesh_instances, sc_data.mi_indices,
                                                                sc_data.meshes, sc_data.transforms, sc_data.tris,
                                                                sc_data.tri_indices, inter);
                } else {
                    NS::Traverse_MacroTree_WithStack_ClosestHit(r, p.primary_masks[i], sc_data.nodes, macro_tree_root,
                                                                sc_data.mesh_instances, sc_data.mi_indices,
                                                                sc_data.meshes, sc_data.transforms, sc_data.tris,
                                                                sc_data.tri_indices, inter);
                }
            }
            // NS::IntersectAreaLights(r, {-1}, sc_data.lights, sc_data.visible_lights, sc_data.transforms, inter);
        }
    } else {
        const mesh_instance_t &mi = sc_data.mesh_instances[cam.mi_index];
        SampleMeshInTextureSpace<DimX, DimY>(region.iteration, cam.mi_index, cam.uv_index,
                                             sc_data.meshes[mi.mesh_index], sc_data.transforms[mi.tr_index],
                                             sc_data.vtx_indices, sc_data.vertices, rect, w, h, &region.halton_seq[hi],
                                             p.primary_rays, p.intersections);

        p.primary_masks.resize(p.primary_rays.size());

        time_after_ray_gen = std::chrono::high_resolution_clock::now();
    }

    const auto time_after_prim_trace = std::chrono::high_resolution_clock::now();

    p.secondary_rays.resize(p.intersections.size());
    p.secondary_masks.resize(p.intersections.size());
    int secondary_rays_count = 0;

    for (size_t i = 0; i < p.intersections.size(); i++) {
        const ray_packet_t<S> &r = p.primary_rays[i];
        const hit_data_t<S> &inter = p.intersections[i];

        const simd_ivec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;
        const simd_ivec<S> px_index = y * w + x;

        p.secondary_masks[i] = {0};

        simd_fvec<S> out_rgba[4] = {0.0f};
        NS::ShadeSurface(px_index, pass_info, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], inter, r, sc_data,
                         macro_tree_root, tex_atlases, out_rgba, p.secondary_masks.data(), p.secondary_rays.data(),
                         &secondary_rays_count);
        for (int j = 0; j < S; j++) {
            if (!p.primary_masks[i][j]) {
                continue;
            }
            temp_buf_.SetPixel(x[j], y[j], {out_rgba[0][j], out_rgba[1][j], out_rgba[2][j], out_rgba[3][j]});
        }
    }

    const auto time_after_prim_shade = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{};

    p.hash_values.resize(secondary_rays_count);
    // p.head_flags.resize(secondary_rays_count * S);
    p.scan_values.resize(secondary_rays_count * S);
    p.chunks.resize(secondary_rays_count * S);
    p.chunks_temp.resize(secondary_rays_count * S);
    // p.skeleton.resize(secondary_rays_count * S);

    if (cam.pass_settings.flags & OutputSH) {
        temp_buf_.ResetSampleData(rect);
        for (int i = 0; i < secondary_rays_count; i++) {
            const ray_packet_t<S> &r = p.secondary_rays[i];

            const simd_ivec<S> x = (r.xy >> 16), y = (r.xy & 0x0000FFFF);

            for (int j = 0; j < S; j++) {
                temp_buf_.SetSampleDir(x[j], y[j], r.d[0][j], r.d[1][j], r.d[2][j]);
                // sample weight for indirect lightmap has all r.c[0..2]`s set to same value
                temp_buf_.SetSampleWeight(x[j], y[j], r.c[0][j]);
            }
        }
    }

    for (int bounce = 1; bounce <= pass_info.settings.max_total_depth && secondary_rays_count &&
                         !(pass_info.settings.flags & SkipIndirectLight);
         bounce++) {
        auto time_secondary_sort_start = std::chrono::high_resolution_clock::now();

        SortRays_CPU(&p.secondary_rays[0], &p.secondary_masks[0], secondary_rays_count, root_min, cell_size,
                     &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

        auto time_secondary_trace_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < secondary_rays_count; i++) {
            const ray_packet_t<S> &r = p.secondary_rays[i];
            hit_data_t<S> &inter = p.intersections[i];
            inter = {};

            if (sc_data.mnodes) {
                NS::Traverse_MacroTree_WithStack_ClosestHit(
                    r, p.secondary_masks[i], sc_data.mnodes, macro_tree_root, sc_data.mesh_instances,
                    sc_data.mi_indices, sc_data.meshes, sc_data.transforms, sc_data.tris, sc_data.tri_indices, inter);
            } else {
                NS::Traverse_MacroTree_WithStack_ClosestHit(
                    r, p.secondary_masks[i], sc_data.nodes, macro_tree_root, sc_data.mesh_instances, sc_data.mi_indices,
                    sc_data.meshes, sc_data.transforms, sc_data.tris, sc_data.tri_indices, inter);
            }
            NS::IntersectAreaLights(r, p.secondary_masks[i], sc_data.lights, sc_data.visible_lights, sc_data.transforms,
                                    inter);
        }

        auto time_secondary_shade_start = std::chrono::high_resolution_clock::now();

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);
        std::swap(p.primary_masks, p.secondary_masks);

        pass_info.bounce = bounce;

        for (int i = 0; i < rays_count; i++) {
            const ray_packet_t<S> &r = p.primary_rays[i];
            const hit_data_t<S> &inter = p.intersections[i];

            const simd_ivec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;
            const simd_ivec<S> index = y * w + x;

            simd_fvec<S> out_rgba[4] = {0.0f};
            NS::ShadeSurface(index, pass_info,
                             &region.halton_seq[hi + RAND_DIM_BASE_COUNT + bounce * RAND_DIM_BOUNCE_COUNT], inter, r,
                             sc_data, macro_tree_root, tex_atlases, out_rgba, p.secondary_masks.data(),
                             p.secondary_rays.data(), &secondary_rays_count);
            out_rgba[3] = 0.0f;

            for (int j = 0; j < S; j++) {
                if (!p.primary_masks[i][j]) {
                    continue;
                }
                temp_buf_.AddPixel(x[j], y[j], {out_rgba[0][j], out_rgba[1][j], out_rgba[2][j], out_rgba[3][j]});
            }
        }

        auto time_secondary_shade_end = std::chrono::high_resolution_clock::now();
        secondary_sort_time +=
            std::chrono::duration<double, std::micro>{time_secondary_trace_start - time_secondary_sort_start};
        secondary_trace_time +=
            std::chrono::duration<double, std::micro>{time_secondary_shade_start - time_secondary_trace_start};
        secondary_shade_time +=
            std::chrono::duration<double, std::micro>{time_secondary_shade_end - time_secondary_shade_start};
    }

    {
        std::lock_guard<std::mutex> _(pass_cache_mtx_);
        pass_cache_.emplace_back(std::move(p));

        stats_.time_primary_ray_gen_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_ray_gen - time_start}.count();
        stats_.time_primary_trace_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_prim_trace - time_after_ray_gen}
                .count();
        stats_.time_primary_shade_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_prim_shade - time_after_prim_trace}
                .count();
        stats_.time_secondary_sort_us += (unsigned long long)secondary_sort_time.count();
        stats_.time_secondary_trace_us += (unsigned long long)secondary_trace_time.count();
        stats_.time_secondary_shade_us += (unsigned long long)secondary_shade_time.count();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / region.iteration;

    clean_buf_.MixWith(temp_buf_, rect, mix_factor);
    if (cam.pass_settings.flags & OutputSH) {
        temp_buf_.ComputeSHData(rect);
        clean_buf_.MixWith_SH(temp_buf_, rect, mix_factor);
    }

    auto clamp_and_gamma_correct = [&cam](const pixel_color_t &p) {
        auto c = simd_fvec4{&p.r};

        if (cam.dtype == SRGB) {
            ITERATE_3({
                if (c[i] < 0.0031308f) {
                    c[i] = 12.92f * c[i];
                } else {
                    c[i] = 1.055f * std::pow(c[i], (1.0f / 2.4f)) - 0.055f;
                }
            })
        }

        if (cam.gamma != 1.0f) {
            c = pow(c, simd_fvec4{1.0f / cam.gamma});
        }

        if (cam.pass_settings.flags & Clamp) {
            c = clamp(c, 0.0f, 1.0f);
        }
        return pixel_color_t{c[0], c[1], c[2], c[3]};
    };

    final_buf_.CopyFrom(clean_buf_, rect, clamp_and_gamma_correct);
}

template <int DimX, int DimY>
void Ray::NS::RendererSIMD<DimX, DimY>::UpdateHaltonSequence(const int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HALTON_COUNT * HALTON_SEQ_LEN]);
    }

    for (int i = 0; i < HALTON_SEQ_LEN; ++i) {
        uint32_t prime_sum = 0;
        for (int j = 0; j < HALTON_COUNT; ++j) {
            seq[i * HALTON_COUNT + j] =
                Ray::ScrambledRadicalInverse(g_primes[j], &permutations_[prime_sum], uint64_t(iteration + i));
            prime_sum += g_primes[j];
        }
    }
}
