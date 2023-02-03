
#include <chrono>
#include <functional>
#include <mutex>
#include <random>

#include "../RendererBase.h"
#include "CoreSIMD.h"
#include "FramebufferRef.h"
#include "Halton.h"

namespace Ray {
class ILog;
namespace NS {
template <int S> struct PassData {
    aligned_vector<ray_data_t<S>> primary_rays;
    aligned_vector<simd_ivec<S>> primary_masks;
    aligned_vector<ray_data_t<S>> secondary_rays;
    aligned_vector<simd_ivec<S>> secondary_masks;
    aligned_vector<shadow_ray_t<S>> shadow_rays;
    aligned_vector<simd_ivec<S>> shadow_masks;
    aligned_vector<hit_data_t<S>> intersections;

    aligned_vector<simd_ivec<S>> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;
    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;

    PassData() = default;

    PassData(const PassData &rhs) = delete;
    PassData(PassData &&rhs) noexcept { *this = std::move(rhs); }

    PassData &operator=(const PassData &rhs) = delete;
    PassData &operator=(PassData &&rhs) noexcept {
        primary_rays = std::move(rhs.primary_rays);
        primary_masks = std::move(rhs.primary_masks);
        secondary_rays = std::move(rhs.secondary_rays);
        secondary_masks = std::move(rhs.secondary_masks);
        shadow_rays = std::move(rhs.shadow_rays);
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

    std::mutex mtx_;

    bool use_wide_bvh_;
    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    RendererSIMD(const settings_t &s, ILog *log);

    std::pair<int, int> size() const override { return std::make_pair(final_buf_.w(), final_buf_.h()); }

    const char *device_name() const override { return "CPU"; }

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

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};

template <int S> Ray::NS::PassData<S> &get_per_thread_pass_data() {
    static thread_local Ray::NS::PassData<S> per_thread_pass_data;
    return per_thread_pass_data;
}

pixel_color_t clamp_and_gamma_correct(const pixel_color_t &p, const camera_t &cam) {
    auto c = simd_fvec4{&p.r};

    if (cam.dtype == SRGB) {
        UNROLLED_FOR(i, 3, {
            if (c.get<i>() < 0.0031308f) {
                c.set<i>(12.92f * c.get<i>());
            } else {
                c.set<i>(1.055f * std::pow(c.get<i>(), (1.0f / 2.4f)) - 0.055f);
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
void Ray::NS::RendererSIMD<DimX, DimY>::RenderScene(const SceneBase *scene, RegionContext &region) {
    const int S = DimX * DimY;

    const auto s = dynamic_cast<const Ref::Scene *>(scene);
    if (!s) {
        return;
    }

    const camera_t &cam = s->cams_[s->current_cam()].cam;

    const scene_data_t sc_data = {s->env_,
                                  s->mesh_instances_.data(),
                                  s->mi_indices_.empty() ? nullptr : &s->mi_indices_[0],
                                  s->meshes_.empty() ? nullptr : &s->meshes_[0],
                                  s->transforms_.empty() ? nullptr : &s->transforms_[0],
                                  s->vtx_indices_.empty() ? nullptr : &s->vtx_indices_[0],
                                  s->vertices_.empty() ? nullptr : &s->vertices_[0],
                                  s->nodes_.empty() ? nullptr : &s->nodes_[0],
                                  s->mnodes_.empty() ? nullptr : &s->mnodes_[0],
                                  s->tris_.empty() ? nullptr : &s->tris_[0],
                                  s->tri_indices_.empty() ? nullptr : &s->tri_indices_[0],
                                  s->mtris_.data(),
                                  s->tri_materials_.empty() ? nullptr : &s->tri_materials_[0],
                                  s->materials_.empty() ? nullptr : &s->materials_[0],
                                  s->lights_.empty() ? nullptr : &s->lights_[0],
                                  {s->li_indices_},
                                  {s->visible_lights_},
                                  {s->blocker_lights_}};

    const uint32_t macro_tree_root = s->macro_nodes_root_;

    float root_min[3], cell_size[3];
    if (macro_tree_root != 0xffffffff) {
        float root_max[3];

        if (sc_data.mnodes) {
            const mbvh_node_t &root_node = sc_data.mnodes[macro_tree_root];

            root_min[0] = root_min[1] = root_min[2] = MAX_DIST;
            root_max[0] = root_max[1] = root_max[2] = -MAX_DIST;

            if (root_node.child[0] & LEAF_NODE_BIT) {
                UNROLLED_FOR(i, 3, {
                    root_min[i] = root_node.bbox_min[i][0];
                    root_max[i] = root_node.bbox_max[i][0];
                })
            } else {
                for (int j = 0; j < 8; j++) {
                    if (root_node.child[j] == 0x7fffffff) {
                        continue;
                    }

                    UNROLLED_FOR(i, 3, {
                        root_min[i] = root_node.bbox_min[i][j];
                        root_max[i] = root_node.bbox_max[i][j];
                    })
                }
            }
        } else {
            const bvh_node_t &root_node = sc_data.nodes[macro_tree_root];

            UNROLLED_FOR(i, 3, {
                root_min[i] = root_node.bbox_min[i];
                root_max[i] = root_node.bbox_max[i];
            })
        }

        UNROLLED_FOR(i, 3, { cell_size[i] = (root_max[i] - root_min[i]) / 255; })
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

    PassData<S> &p = get_per_thread_pass_data<S>();

    // allocate sh data on demand
    if (cam.pass_settings.flags & OutputSH) {
        std::lock_guard<std::mutex> _(mtx_);

        temp_buf_.Resize(w, h, true);
        clean_buf_.Resize(w, h, true);
    }

    const auto time_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> time_after_ray_gen;

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    if (cam.type != Geo) {
        GeneratePrimaryRays<DimX, DimY>(region.iteration, cam, rect, w, h, &region.halton_seq[hi], p.primary_rays,
                                        p.primary_masks);

        time_after_ray_gen = std::chrono::high_resolution_clock::now();

        p.intersections.resize(p.primary_rays.size());

        for (size_t i = 0; i < p.primary_rays.size(); i++) {
            ray_data_t<S> &r = p.primary_rays[i];

            hit_data_t<S> &inter = p.intersections[i];
            inter = {};
            inter.t = cam.clip_end;

            if (macro_tree_root != 0xffffffff) {
                NS::IntersectScene(r, p.primary_masks[i], cam.pass_settings.min_transp_depth,
                                   cam.pass_settings.max_transp_depth, &region.halton_seq[hi + RAND_DIM_BASE_COUNT],
                                   sc_data, macro_tree_root, s->tex_storages_, inter);
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

    p.secondary_rays.resize(p.primary_rays.size());
    p.secondary_masks.resize(p.primary_rays.size());
    p.shadow_rays.resize(p.primary_rays.size());
    p.shadow_masks.resize(p.primary_rays.size());
    int secondary_rays_count = 0, shadow_rays_count = 0;

    for (size_t ri = 0; ri < p.intersections.size(); ri++) {
        const ray_data_t<S> &r = p.primary_rays[ri];
        const hit_data_t<S> &inter = p.intersections[ri];

        const simd_ivec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;

        p.secondary_masks[ri] = {0};

        simd_fvec<S> out_rgba[4] = {0.0f};
        NS::ShadeSurface(cam.pass_settings, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], inter, r, sc_data,
                         macro_tree_root, s->tex_storages_, out_rgba, p.secondary_masks.data(), p.secondary_rays.data(),
                         &secondary_rays_count, p.shadow_masks.data(), p.shadow_rays.data(), &shadow_rays_count);

        // TODO: vectorize this
        UNROLLED_FOR_S(i, S, {
            if (p.primary_masks[ri].template get<i>()) {
                temp_buf_.SetPixel(x.template get<i>(), y.template get<i>(),
                                   {out_rgba[0].template get<i>(), out_rgba[1].template get<i>(),
                                    out_rgba[2].template get<i>(), out_rgba[3].template get<i>()});
            }
        })
    }

    const auto time_after_prim_shade = std::chrono::high_resolution_clock::now();

    for (int ri = 0; ri < shadow_rays_count; ++ri) {
        const shadow_ray_t<S> &sh_r = p.shadow_rays[ri];

        const simd_ivec<S> x = sh_r.xy >> 16, y = sh_r.xy & 0x0000FFFF;

        simd_fvec<S> rc[3];
        NS::IntersectScene(sh_r, p.shadow_masks[ri], cam.pass_settings.max_transp_depth, sc_data, macro_tree_root,
                           s->tex_storages_, rc);
        const simd_fvec<S> k = NS::IntersectAreaLights(sh_r, p.shadow_masks[ri], sc_data.lights, sc_data.blocker_lights,
                                                       sc_data.transforms);
        UNROLLED_FOR(i, 3, { rc[i] *= k; })

        // TODO: vectorize this
        UNROLLED_FOR_S(i, S, {
            if (p.shadow_masks[ri].template get<i>()) {
                temp_buf_.AddPixel(x.template get<i>(), y.template get<i>(),
                                   {rc[0].template get<i>(), rc[1].template get<i>(), rc[2].template get<i>(), 0.0f});
            }
        })
    }

    const auto time_after_prim_shadow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{},
        secondary_shadow_time{};

    p.hash_values.resize(p.primary_rays.size());
    // p.head_flags.resize(p.primary_rays.size() * S);
    p.scan_values.resize(p.primary_rays.size() * S);
    p.chunks.resize(p.primary_rays.size() * S);
    p.chunks_temp.resize(p.primary_rays.size() * S);
    // p.skeleton.resize(p.primary_rays.size() * S);

    if (cam.pass_settings.flags & OutputSH) {
        temp_buf_.ResetSampleData(rect);
        for (int i = 0; i < secondary_rays_count; i++) {
            const ray_data_t<S> &r = p.secondary_rays[i];

            const simd_ivec<S> x = (r.xy >> 16), y = (r.xy & 0x0000FFFF);

            for (int j = 0; j < S; j++) {
                temp_buf_.SetSampleDir(x[j], y[j], r.d[0][j], r.d[1][j], r.d[2][j]);
                // sample weight for indirect lightmap has all r.c[0..2]`s set to same value
                temp_buf_.SetSampleWeight(x[j], y[j], r.c[0][j]);
            }
        }
    }

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; bounce++) {
        auto time_secondary_sort_start = std::chrono::high_resolution_clock::now();

        SortRays_CPU(&p.secondary_rays[0], &p.secondary_masks[0], secondary_rays_count, root_min, cell_size,
                     &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

        auto time_secondary_trace_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < secondary_rays_count; i++) {
            ray_data_t<S> &r = p.secondary_rays[i];

            hit_data_t<S> &inter = p.intersections[i];
            inter = {};

            NS::IntersectScene(r, p.secondary_masks[i], cam.pass_settings.min_transp_depth,
                               cam.pass_settings.max_transp_depth, &region.halton_seq[hi + RAND_DIM_BASE_COUNT],
                               sc_data, macro_tree_root, s->tex_storages_, inter);

            const simd_ivec<S> not_only_transparency_ray = (r.depth & 0x00ffffff) != 0;
            NS::IntersectAreaLights(r, p.secondary_masks[i] & not_only_transparency_ray, sc_data.lights,
                                    sc_data.visible_lights, sc_data.transforms, inter);
        }

        auto time_secondary_shade_start = std::chrono::high_resolution_clock::now();

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        shadow_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);
        std::swap(p.primary_masks, p.secondary_masks);

        for (int ri = 0; ri < rays_count; ri++) {
            const ray_data_t<S> &r = p.primary_rays[ri];
            const hit_data_t<S> &inter = p.intersections[ri];

            const simd_ivec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;

            simd_fvec<S> out_rgba[4] = {0.0f};
            NS::ShadeSurface(cam.pass_settings, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], inter, r, sc_data,
                             macro_tree_root, s->tex_storages_, out_rgba, p.secondary_masks.data(),
                             p.secondary_rays.data(), &secondary_rays_count, p.shadow_masks.data(),
                             p.shadow_rays.data(), &shadow_rays_count);

            // TODO: vectorize this
            UNROLLED_FOR_S(i, S, {
                if (p.primary_masks[ri].template get<i>()) {
                    temp_buf_.AddPixel(x.template get<i>(), y.template get<i>(),
                                       {out_rgba[0].template get<i>(), out_rgba[1].template get<i>(),
                                        out_rgba[2].template get<i>(), out_rgba[3].template get<i>()});
                }
            })
        }

        auto time_secondary_shadow_start = std::chrono::high_resolution_clock::now();

        for (int ri = 0; ri < shadow_rays_count; ++ri) {
            const shadow_ray_t<S> &sh_r = p.shadow_rays[ri];

            const simd_ivec<S> x = sh_r.xy >> 16, y = sh_r.xy & 0x0000FFFF;

            simd_fvec<S> rc[3];
            IntersectScene(sh_r, p.shadow_masks[ri], cam.pass_settings.max_transp_depth, sc_data, macro_tree_root,
                           s->tex_storages_, rc);
            const simd_fvec<S> k = NS::IntersectAreaLights(sh_r, p.shadow_masks[ri], sc_data.lights,
                                                           sc_data.blocker_lights, sc_data.transforms);
            UNROLLED_FOR(i, 3, { rc[i] *= k; })

            // TODO: vectorize this
            UNROLLED_FOR_S(i, S, {
                if (p.shadow_masks[ri].template get<i>()) {
                    temp_buf_.AddPixel(
                        x.template get<i>(), y.template get<i>(),
                        {rc[0].template get<i>(), rc[1].template get<i>(), rc[2].template get<i>(), 0.0f});
                }
            })
        }

        auto time_secondary_shadow_end = std::chrono::high_resolution_clock::now();
        secondary_sort_time +=
            std::chrono::duration<double, std::micro>{time_secondary_trace_start - time_secondary_sort_start};
        secondary_trace_time +=
            std::chrono::duration<double, std::micro>{time_secondary_shade_start - time_secondary_trace_start};
        secondary_shade_time +=
            std::chrono::duration<double, std::micro>{time_secondary_shadow_start - time_secondary_shade_start};
        secondary_shadow_time +=
            std::chrono::duration<double, std::micro>{time_secondary_shadow_end - time_secondary_shadow_start};
    }

    {
        std::lock_guard<std::mutex> _(mtx_);

        stats_.time_primary_ray_gen_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_ray_gen - time_start}.count();
        stats_.time_primary_trace_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_prim_trace - time_after_ray_gen}
                .count();
        stats_.time_primary_shade_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_prim_shade - time_after_prim_trace}
                .count();
        stats_.time_primary_shadow_us +=
            (unsigned long long)std::chrono::duration<double, std::micro>{time_after_prim_shadow -
                                                                          time_after_prim_shade}
                .count();
        stats_.time_secondary_sort_us += (unsigned long long)secondary_sort_time.count();
        stats_.time_secondary_trace_us += (unsigned long long)secondary_trace_time.count();
        stats_.time_secondary_shade_us += (unsigned long long)secondary_shade_time.count();
        stats_.time_secondary_shadow_us += (unsigned long long)secondary_shadow_time.count();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / float(region.iteration);

    clean_buf_.MixWith(temp_buf_, rect, mix_factor);
    if (cam.pass_settings.flags & OutputSH) {
        temp_buf_.ComputeSHData(rect);
        clean_buf_.MixWith_SH(temp_buf_, rect, mix_factor);
    }

    auto _clamp_and_gamma_correct = [&cam](const pixel_color_t &p) { return clamp_and_gamma_correct(p, cam); };

    final_buf_.CopyFrom(clean_buf_, rect, _clamp_and_gamma_correct);
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
