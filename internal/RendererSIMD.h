
#include <chrono>
#include <functional>
#include <mutex>
#include <random>

#include "../RendererBase.h"
#include "CoreSIMD.h"
#include "Halton.h"
#include "SceneRef.h"

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

    aligned_vector<color_rgba_t, 16> temp_final_buf;
    aligned_vector<color_rgba_t, 16> variance_buf;
    aligned_vector<color_rgba_t, 16> filtered_variance_buf;

    aligned_vector<simd_ivec<S>> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;
    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;
};

template <int DimX, int DimY> class RendererSIMD : public RendererBase {
    ILog *log_;
    aligned_vector<color_rgba_t, 16> dual_buf_[2], base_color_buf_, depth_normals_buf_, temp_buf_, final_buf_,
        raw_final_buf_, raw_filtered_buf_;

    std::mutex mtx_;

    bool use_wide_bvh_;
    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    Ref::tonemap_params_t tonemap_params_;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    RendererSIMD(const settings_t &s, ILog *log);

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    ILog *log() const override { return log_; }

    const char *device_name() const override { return "CPU"; }

    const color_rgba_t *get_pixels_ref() const override { return final_buf_.data(); }
    const color_rgba_t *get_raw_pixels_ref() const override { return raw_filtered_buf_.data(); }
    const color_rgba_t *get_aux_pixels_ref(const eAUXBuffer buf) const override {
        if (buf == eAUXBuffer::BaseColor) {
            return base_color_buf_.data();
        } else if (buf == eAUXBuffer::DepthNormals) {
            return depth_normals_buf_.data();
        }
        return nullptr;
    }

    const shl1_data_t *get_sh_data_ref() const override { return nullptr; }

    void Resize(const int w, const int h) override {
        if (w_ != w || h_ != h) {
            for (auto &buf : dual_buf_) {
                buf.assign(w * h, {});
                buf.shrink_to_fit();
            }
            temp_buf_.assign(w * h, {});
            temp_buf_.shrink_to_fit();
            final_buf_.assign(w * h, {});
            final_buf_.shrink_to_fit();
            raw_final_buf_.assign(w * h, {});
            raw_final_buf_.shrink_to_fit();
            raw_filtered_buf_.assign(w * h, {});
            raw_filtered_buf_.shrink_to_fit();

            w_ = w;
            h_ = h;
        }
    }
    void Clear(const color_rgba_t &c) override {
        for (auto &buf : dual_buf_) {
            buf.assign(w_ * h_, c);
        }
    }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};

template <int S> Ray::NS::PassData<S> &get_per_thread_pass_data() {
    static thread_local Ray::NS::PassData<S> per_thread_pass_data;
    return per_thread_pass_data;
}

} // namespace NS
} // namespace Ray

////////////////////////////////////////////////////////////////////////////////////////////

#include "UniformIntDistribution.h"

template <int DimX, int DimY>
Ray::NS::RendererSIMD<DimX, DimY>::RendererSIMD(const settings_t &s, ILog *log)
    : log_(log), use_wide_bvh_(s.use_wide_bvh) {
    auto mt = std::mt19937(0);
    auto dist = UniformIntDistribution<uint32_t>{};
    auto rand_func = [&]() { return dist(mt); };
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);

    Resize(s.w, s.h);
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

    const camera_t &cam = s->cams_[s->current_cam()._index].cam;

    std::shared_lock<std::shared_timed_mutex> scene_lock(s->mtx_);

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

    const rect_t &rect = region.rect();

    region.iteration++;
    if (!region.halton_seq || region.iteration % HALTON_SEQ_LEN == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    PassData<S> &p = get_per_thread_pass_data<S>();

    // allocate aux data on demand
    if (cam.pass_settings.flags & (Bitmask<ePassFlags>{ePassFlags::OutputBaseColor} | ePassFlags::OutputDepthNormals)) {
        // TODO: Skip locking here
        std::lock_guard<std::mutex> _(mtx_);

        if (cam.pass_settings.flags & ePassFlags::OutputBaseColor) {
            base_color_buf_.resize(w_ * h_, {});
        } else if (!base_color_buf_.empty()) {
            base_color_buf_ = {};
        }
        if (cam.pass_settings.flags & ePassFlags::OutputDepthNormals) {
            depth_normals_buf_.resize(w_ * h_, {});
        } else if (!depth_normals_buf_.empty()) {
            depth_normals_buf_ = {};
        }
    }

    using namespace std::chrono;

    const auto time_start = high_resolution_clock::now();
    time_point<high_resolution_clock> time_after_ray_gen;

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    if (cam.type != eCamType::Geo) {
        GeneratePrimaryRays<DimX, DimY>(region.iteration, cam, rect, w_, h_, &region.halton_seq[hi], p.primary_rays,
                                        p.primary_masks);

        time_after_ray_gen = high_resolution_clock::now();

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
                                             sc_data.vtx_indices, sc_data.vertices, rect, w_, h_,
                                             &region.halton_seq[hi], p.primary_rays, p.intersections);

        p.primary_masks.resize(p.primary_rays.size());

        time_after_ray_gen = high_resolution_clock::now();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / float(region.iteration);

    const auto time_after_prim_trace = high_resolution_clock::now();

    p.secondary_rays.resize(p.primary_rays.size());
    p.secondary_masks.resize(p.primary_rays.size());
    p.shadow_rays.resize(p.primary_rays.size());
    p.shadow_masks.resize(p.primary_rays.size());
    int secondary_rays_count = 0, shadow_rays_count = 0;

    simd_fvec<S> clamp_direct = cam.pass_settings.clamp_direct;
    where(clamp_direct == 0.0f, clamp_direct) = std::numeric_limits<float>::max();
    simd_fvec<S> clamp_indirect = cam.pass_settings.clamp_indirect;
    where(clamp_indirect == 0.0f, clamp_indirect) = std::numeric_limits<float>::max();

    for (size_t ri = 0; ri < p.intersections.size(); ri++) {
        const ray_data_t<S> &r = p.primary_rays[ri];
        const hit_data_t<S> &inter = p.intersections[ri];

        const simd_ivec<S> x = r.xy >> 16, y = r.xy & 0x0000FFFF;

        p.secondary_masks[ri] = {0};

        simd_fvec<S> out_rgba[4] = {0.0f}, out_base_color[4] = {0.0f}, out_depth_normal[4] = {0.0f};
        NS::ShadeSurface(cam.pass_settings, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], inter, r, sc_data,
                         macro_tree_root, s->tex_storages_, out_rgba, p.secondary_masks.data(), p.secondary_rays.data(),
                         &secondary_rays_count, p.shadow_masks.data(), p.shadow_rays.data(), &shadow_rays_count,
                         out_base_color, out_depth_normal);

        // TODO: match layouts!
        UNROLLED_FOR_S(i, S, {
            if (p.primary_masks[ri].template get<i>()) {
                UNROLLED_FOR(j, 3, { out_rgba[j] = min(out_rgba[j], clamp_direct); })
                UNROLLED_FOR(j, 4, {
                    temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v[j] = out_rgba[j].template get<i>();
                })
                if (cam.pass_settings.flags & ePassFlags::OutputBaseColor) {
                    auto old_val =
                        simd_fvec4(base_color_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                    old_val += (simd_fvec4{out_base_color[0].template get<i>(), out_base_color[1].template get<i>(),
                                           out_base_color[2].template get<i>(), 0.0f} -
                                old_val) *
                               mix_factor;
                    old_val.store_to(base_color_buf_[y.template get<i>() * w_ + x.template get<i>()].v,
                                     simd_mem_aligned);
                }
                if (cam.pass_settings.flags & ePassFlags::OutputDepthNormals) {
                    auto old_val = simd_fvec4(depth_normals_buf_[y.template get<i>() * w_ + x.template get<i>()].v,
                                              simd_mem_aligned);
                    old_val +=
                        (simd_fvec4{out_depth_normal[0].template get<i>(), out_depth_normal[1].template get<i>(),
                                    out_depth_normal[2].template get<i>(), out_depth_normal[3].template get<i>()} -
                         old_val) *
                        mix_factor;
                    old_val.store_to(depth_normals_buf_[y.template get<i>() * w_ + x.template get<i>()].v,
                                     simd_mem_aligned);
                }
            }
        })
    }

    const auto time_after_prim_shade = high_resolution_clock::now();

    for (int ri = 0; ri < shadow_rays_count; ++ri) {
        const shadow_ray_t<S> &sh_r = p.shadow_rays[ri];

        const simd_ivec<S> x = sh_r.xy >> 16, y = sh_r.xy & 0x0000FFFF;

        simd_fvec<S> rc[3];
        NS::IntersectScene(sh_r, p.shadow_masks[ri], cam.pass_settings.max_transp_depth, sc_data, macro_tree_root,
                           s->tex_storages_, rc);
        const simd_fvec<S> k = NS::IntersectAreaLights(sh_r, p.shadow_masks[ri], sc_data.lights, sc_data.blocker_lights,
                                                       sc_data.transforms);
        UNROLLED_FOR(i, 3, { rc[i] = min(rc[i] * k, clamp_direct); })

        // TODO: match layouts!
        UNROLLED_FOR_S(i, S, {
            if (p.shadow_masks[ri].template get<i>()) {
                auto old_val =
                    simd_fvec4(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                old_val += simd_fvec4(rc[0].template get<i>(), rc[1].template get<i>(), rc[2].template get<i>(), 0.0f);
                old_val.store_to(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
            }
        })
    }

    const auto time_after_prim_shadow = high_resolution_clock::now();
    duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{},
        secondary_shadow_time{};

    p.hash_values.resize(p.primary_rays.size());
    // p.head_flags.resize(p.primary_rays.size() * S);
    p.scan_values.resize(p.primary_rays.size() * S);
    p.chunks.resize(p.primary_rays.size() * S);
    p.chunks_temp.resize(p.primary_rays.size() * S);
    // p.skeleton.resize(p.primary_rays.size() * S);

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; bounce++) {
        auto time_secondary_sort_start = high_resolution_clock::now();

        SortRays_CPU(&p.secondary_rays[0], &p.secondary_masks[0], secondary_rays_count, root_min, cell_size,
                     &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

        auto time_secondary_trace_start = high_resolution_clock::now();

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

        auto time_secondary_shade_start = high_resolution_clock::now();

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
                             p.shadow_rays.data(), &shadow_rays_count, (simd_fvec<S> *)nullptr,
                             (simd_fvec<S> *)nullptr);
            UNROLLED_FOR(i, 3, { out_rgba[i] = min(out_rgba[i], clamp_indirect); })

            // TODO: match layouts!
            UNROLLED_FOR_S(i, S, {
                if (p.primary_masks[ri].template get<i>()) {
                    auto old_val =
                        simd_fvec4(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                    old_val += simd_fvec4(out_rgba[0].template get<i>(), out_rgba[1].template get<i>(),
                                          out_rgba[2].template get<i>(), out_rgba[3].template get<i>());
                    old_val.store_to(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                }
            })
        }

        auto time_secondary_shadow_start = high_resolution_clock::now();

        for (int ri = 0; ri < shadow_rays_count; ++ri) {
            const shadow_ray_t<S> &sh_r = p.shadow_rays[ri];

            const simd_ivec<S> x = sh_r.xy >> 16, y = sh_r.xy & 0x0000FFFF;

            simd_fvec<S> rc[3];
            IntersectScene(sh_r, p.shadow_masks[ri], cam.pass_settings.max_transp_depth, sc_data, macro_tree_root,
                           s->tex_storages_, rc);
            const simd_fvec<S> k = NS::IntersectAreaLights(sh_r, p.shadow_masks[ri], sc_data.lights,
                                                           sc_data.blocker_lights, sc_data.transforms);
            UNROLLED_FOR(i, 3, { rc[i] = min(rc[i] * k, clamp_indirect); })

            // TODO: vectorize this
            UNROLLED_FOR_S(i, S, {
                if (p.shadow_masks[ri].template get<i>()) {
                    auto old_val =
                        simd_fvec4(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                    old_val +=
                        simd_fvec4(rc[0].template get<i>(), rc[1].template get<i>(), rc[2].template get<i>(), 0.0f);
                    old_val.store_to(temp_buf_[y.template get<i>() * w_ + x.template get<i>()].v, simd_mem_aligned);
                }
            })
        }

        auto time_secondary_shadow_end = high_resolution_clock::now();
        secondary_sort_time += duration<double, std::micro>{time_secondary_trace_start - time_secondary_sort_start};
        secondary_trace_time += duration<double, std::micro>{time_secondary_shade_start - time_secondary_trace_start};
        secondary_shade_time += duration<double, std::micro>{time_secondary_shadow_start - time_secondary_shade_start};
        secondary_shadow_time += duration<double, std::micro>{time_secondary_shadow_end - time_secondary_shadow_start};
    }

    scene_lock.unlock();

    Ref::tonemap_params_t tonemap_params;
    tonemap_params.inv_gamma = (1.0f / cam.gamma);
    tonemap_params.srgb = (cam.dtype == eDeviceType::SRGB);

    Ref::simd_fvec4 exposure = std::pow(2.0f, cam.exposure);
    exposure.set<3>(1.0f);

    {
        std::lock_guard<std::mutex> _(mtx_);

        stats_.time_primary_ray_gen_us +=
            (unsigned long long)duration<double, std::micro>{time_after_ray_gen - time_start}.count();
        stats_.time_primary_trace_us +=
            (unsigned long long)duration<double, std::micro>{time_after_prim_trace - time_after_ray_gen}.count();
        stats_.time_primary_shade_us +=
            (unsigned long long)duration<double, std::micro>{time_after_prim_shade - time_after_prim_trace}.count();
        stats_.time_primary_shadow_us +=
            (unsigned long long)duration<double, std::micro>{time_after_prim_shadow - time_after_prim_shade}.count();
        stats_.time_secondary_sort_us += (unsigned long long)secondary_sort_time.count();
        stats_.time_secondary_trace_us += (unsigned long long)secondary_trace_time.count();
        stats_.time_secondary_shade_us += (unsigned long long)secondary_shade_time.count();
        stats_.time_secondary_shadow_us += (unsigned long long)secondary_shadow_time.count();

        tonemap_params_ = tonemap_params;
    }

    color_rgba_t *clean_buf = dual_buf_[(region.iteration - 1) % 2].data();

    const float half_mix_factor = 1.0f / float((region.iteration + 1) / 2);
    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            const simd_fvec4 new_val = {temp_buf_[y * w_ + x].v, simd_mem_aligned};

            simd_fvec4 cur_val = {clean_buf[y * w_ + x].v, simd_mem_aligned};
            cur_val += (new_val - cur_val) * half_mix_factor;
            cur_val.store_to(clean_buf[y * w_ + x].v, simd_mem_aligned);
        }
    }

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            auto p1 = Ref::simd_fvec4(dual_buf_[0][y * w_ + x].v, Ref::simd_mem_aligned) * exposure;
            auto p2 = Ref::simd_fvec4(dual_buf_[1][y * w_ + x].v, Ref::simd_mem_aligned) * exposure;

            const int p1_samples = (region.iteration + 1) / 2;
            const int p2_samples = (region.iteration) / 2;

            const float p1_weight = float(p1_samples) / float(region.iteration);
            const float p2_weight = float(p2_samples) / float(region.iteration);

            const Ref::simd_fvec4 untonemapped_res = p1_weight * p1 + p2_weight * p2;
            untonemapped_res.store_to(raw_final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            // Also store as denosed result until DenoiseImage method will be called
            untonemapped_res.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);

            const Ref::simd_fvec4 tonemapped_res = Ref::clamp_and_gamma_correct(tonemap_params, untonemapped_res);
            tonemapped_res.store_to(final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);

            p1 = Ref::reversible_tonemap(p1);
            p2 = Ref::reversible_tonemap(p2);

            const Ref::simd_fvec4 variance = 0.5f * (p1 - p2) * (p1 - p2);
            variance.store_to(temp_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
        }
    }
}

template <int DimX, int DimY> void Ray::NS::RendererSIMD<DimX, DimY>::DenoiseImage(const RegionContext &region) {
    const int S = DimX * DimY;

    using namespace std::chrono;
    const auto denoise_start = high_resolution_clock::now();

    const rect_t &rect = region.rect();

    const int EXT_RADIUS = 8;
    const rect_t rect_ext = {rect.x - EXT_RADIUS, rect.y - EXT_RADIUS, rect.w + 2 * EXT_RADIUS,
                             rect.h + 2 * EXT_RADIUS};

    PassData<S> &p = get_per_thread_pass_data<S>();

    p.temp_final_buf.resize(rect_ext.w * rect_ext.h);
    p.variance_buf.resize(rect_ext.w * rect_ext.h);
    p.filtered_variance_buf.resize(rect_ext.w * rect_ext.h);

#define FETCH_FINAL_BUF(_x, _y)                                                                                        \
    Ref::simd_fvec4(raw_final_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,      \
                    Ref::simd_mem_aligned)
#define FETCH_VARIANCE(_x, _y)                                                                                         \
    Ref::simd_fvec4(temp_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,           \
                    Ref::simd_mem_aligned)

    static const float GaussWeights[] = {0.2270270270f, 0.1945945946f, 0.1216216216f, 0.0540540541f, 0.0162162162f};

    for (int y = 0; y < rect_ext.h; ++y) {
        const int yy = rect_ext.y + y;
        for (int x = 0; x < rect_ext.w; ++x) {
            const int xx = rect_ext.x + x;
            const Ref::simd_fvec4 center_col = Ref::reversible_tonemap(FETCH_FINAL_BUF(xx, yy));
            center_col.store_to(p.temp_final_buf[y * rect_ext.w + x].v, Ref::simd_mem_aligned);

            const Ref::simd_fvec4 center_val = FETCH_VARIANCE(xx, yy);

            Ref::simd_fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res += FETCH_VARIANCE(xx - i + 1, yy) * GaussWeights[i + 1];
                res += FETCH_VARIANCE(xx + i + 1, yy) * GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.variance_buf[y * rect_ext.w + x].v, Ref::simd_mem_aligned);
        }
    }

#undef FETCH_VARIANCE
#undef FETCH_FINAL_BUF

    for (int y = 4; y < rect_ext.h - 4; ++y) {
        for (int x = 4; x < rect_ext.w - 4; ++x) {
            const simd_fvec4 center_val = {p.variance_buf[(y + 0) * rect_ext.w + x].v, simd_mem_aligned};

            simd_fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res +=
                    simd_fvec4(p.variance_buf[(y - i + 1) * rect_ext.w + x].v, simd_mem_aligned) * GaussWeights[i + 1];
                res +=
                    simd_fvec4(p.variance_buf[(y + i + 1) * rect_ext.w + x].v, simd_mem_aligned) * GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.filtered_variance_buf[y * rect_ext.w + x].v, simd_mem_aligned);
        }
    }

    const int NLM_WINDOW_SIZE = 7;
    const int NLM_NEIGHBORHOOD_SIZE = 3;

    static_assert(EXT_RADIUS >= (NLM_WINDOW_SIZE - 1) / 2 + (NLM_NEIGHBORHOOD_SIZE - 1) / 2, "!");

    Ref::NLMFilter<NLM_WINDOW_SIZE, NLM_NEIGHBORHOOD_SIZE>(
        p.temp_final_buf.data(), rect_t{EXT_RADIUS, EXT_RADIUS, rect.w, rect.h}, rect_ext.w, 1.0f, 0.45f,
        p.filtered_variance_buf.data(), rect, w_, raw_filtered_buf_.data());

    Ref::tonemap_params_t tonemap_params;

    {
        std::lock_guard<std::mutex> _(mtx_);
        tonemap_params = tonemap_params_;
    }

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            auto col = Ref::simd_fvec4(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            col = Ref::reversible_tonemap_invert(col);
            col.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            col = Ref::clamp_and_gamma_correct(tonemap_params, col);
            col.store_to(final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
        }
    }

    const auto denoise_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_denoise_us += (unsigned long long)duration<double, std::micro>{denoise_end - denoise_start}.count();
    }
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
