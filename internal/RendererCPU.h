#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <random>

#include "../RendererBase.h"
#include "CoreRef.h"
#include "Halton.h"
#include "SceneCPU.h"
#include "UniformIntDistribution.h"

#define DEBUG_ADAPTIVE_SAMPLING 0

namespace Ray {
class ILog;
namespace Ref {
class SIMDPolicy {
  public:
    using RayDataType = Ref::ray_data_t;
    using ShadowRayType = Ref::shadow_ray_t;
    using HitDataType = Ref::hit_data_t;
    using RayHashType = uint32_t;

  protected:
    static force_inline eRendererType type() { return eRendererType::Reference; }

    static force_inline void GeneratePrimaryRays(const camera_t &cam, const rect_t &r, int w, int h,
                                                 const float random_seq[], const int iteration,
                                                 const uint16_t required_samples[],
                                                 aligned_vector<Ref::ray_data_t> &out_rays) {
        Ref::GeneratePrimaryRays(cam, r, w, h, random_seq, iteration, required_samples, out_rays);
    }

    static force_inline void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                                      const transform_t &tr, const uint32_t *vtx_indices,
                                                      const vertex_t *vertices, const rect_t &r, int w, int h,
                                                      const float *random_seq, aligned_vector<ray_data_t> &out_rays,
                                                      aligned_vector<hit_data_t> &out_inters) {
        Ref::SampleMeshInTextureSpace(iteration, obj_index, uv_layer, mesh, tr, vtx_indices, vertices, r, w, h,
                                      random_seq, out_rays, out_inters);
    }

    static force_inline void IntersectScene(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth,
                                            const float *random_seq, const scene_data_t &sc, uint32_t root_index,
                                            const Cpu::TexStorageBase *const textures[], Span<hit_data_t> out_inter) {
        Ref::IntersectScene(rays, min_transp_depth, max_transp_depth, random_seq, sc, root_index, textures, out_inter);
    }

    static force_inline void IntersectAreaLights(Span<const ray_data_t> rays, const light_t lights[],
                                                 Span<const uint32_t> visible_lights, const transform_t transforms[],
                                                 Span<hit_data_t> inout_inters) {
        Ref::IntersectAreaLights(rays, lights, visible_lights, transforms, inout_inters);
    }

    static force_inline void TraceRays(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth,
                                       const scene_data_t &sc, uint32_t node_index, bool trace_lights,
                                       const Cpu::TexStorageBase *const textures[], const float random_seq[],
                                       Span<hit_data_t> out_inter) {
        Ref::TraceRays(rays, min_transp_depth, max_transp_depth, sc, node_index, trace_lights, textures, random_seq,
                       out_inter);
    }

    static force_inline void TraceShadowRays(Span<const shadow_ray_t> rays, int max_transp_depth, float _clamp_val,
                                             const scene_data_t &sc, uint32_t node_index, const float random_seq[],
                                             const Cpu::TexStorageBase *const textures[], int img_w,
                                             color_rgba_t *out_color) {
        Ref::TraceShadowRays(rays, max_transp_depth, _clamp_val, sc, node_index, random_seq, textures, img_w,
                             out_color);
    }

    static force_inline int SortRays_CPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3],
                                         uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                                         ray_chunk_t *chunks_temp) {
        return Ref::SortRays_CPU(rays, root_min, cell_size, hash_values, scan_values, chunks, chunks_temp);
    }

    static force_inline void ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t> inters,
                                          Span<const ray_data_t> rays, const float *random_seq, const scene_data_t &sc,
                                          uint32_t node_index, const Cpu::TexStorageBase *const textures[],
                                          ray_data_t *out_secondary_rays, int *out_secondary_rays_count,
                                          shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count, int img_w,
                                          float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                                          color_rgba_t *out_depth_normal) {
        Ref::ShadePrimary(ps, inters, rays, random_seq, sc, node_index, textures, out_secondary_rays,
                          out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count, img_w, mix_factor,
                          out_color, out_base_color, out_depth_normal);
    }

    static force_inline void ShadeSecondary(const pass_settings_t &ps, Span<const hit_data_t> inters,
                                            Span<const ray_data_t> rays, const float *random_seq,
                                            const scene_data_t &sc, uint32_t node_index,
                                            const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                                            int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                                            int *out_shadow_rays_count, int img_w, color_rgba_t *out_color) {
        Ref::ShadeSecondary(ps, inters, rays, random_seq, sc, node_index, textures, out_secondary_rays,
                            out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count, img_w, out_color);
    }
};
} // namespace Ref
namespace Cpu {
template <typename SIMDPolicy> class Renderer : public RendererBase, private SIMDPolicy {
    ILog *log_;

    bool use_wide_bvh_;
    aligned_vector<color_rgba_t, 16> dual_buf_[2], base_color_buf_, depth_normals_buf_, temp_buf_, final_buf_,
        raw_final_buf_, raw_filtered_buf_;
    std::vector<uint16_t> required_samples_;

    std::mutex mtx_;

    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    Ref::tonemap_params_t tonemap_params_;
    float variance_threshold_ = 0.0f;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    Renderer(const settings_t &s, ILog *log);

    eRendererType type() const override { return SIMDPolicy::type(); }

    ILog *log() const override { return log_; }

    const char *device_name() const override { return "CPU"; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    color_data_rgba_t get_pixels_ref() const override { return {final_buf_.data(), w_}; }
    color_data_rgba_t get_raw_pixels_ref() const override { return {raw_filtered_buf_.data(), w_}; }
    color_data_rgba_t get_aux_pixels_ref(const eAUXBuffer buf) const override {
        if (buf == eAUXBuffer::BaseColor) {
            return {base_color_buf_.data(), w_};
        } else if (buf == eAUXBuffer::DepthNormals) {
            return color_data_rgba_t{depth_normals_buf_.data(), w_};
        }
        return {};
    }

    const shl1_data_t *get_sh_data_ref() const override { return nullptr; }

    void Resize(const int w, const int h) override {
        if (w_ != w || h_ != h) {
            for (auto &buf : dual_buf_) {
                buf.assign(w * h, {});
                buf.shrink_to_fit();
            }
            required_samples_.assign(w * h, 0xffff);
            required_samples_.shrink_to_fit();
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
        required_samples_.assign(w_ * h_, 0xffff);
    }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};
} // namespace Cpu
namespace Ref {
using Renderer = Cpu::Renderer<Ref::SIMDPolicy>;
}
} // namespace Ray

namespace Ray {
namespace Cpu {
template <typename SIMDPolicy> struct PassData {
    aligned_vector<typename SIMDPolicy::RayDataType> primary_rays;
    aligned_vector<typename SIMDPolicy::RayDataType> secondary_rays;
    aligned_vector<typename SIMDPolicy::ShadowRayType> shadow_rays;
    aligned_vector<typename SIMDPolicy::HitDataType> intersections;

    aligned_vector<color_rgba_t, 16> temp_final_buf;
    aligned_vector<color_rgba_t, 16> variance_buf;
    aligned_vector<color_rgba_t, 16> filtered_variance_buf;
    aligned_vector<color_rgba_t, 16> feature_buf1, feature_buf2;

    aligned_vector<typename SIMDPolicy::RayHashType> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;

    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;
};

template <typename SIMDPolicy> PassData<SIMDPolicy> &get_per_thread_pass_data() {
    static thread_local PassData<SIMDPolicy> per_thread_pass_data;
    return per_thread_pass_data;
}
} // namespace Cpu
} // namespace Ray

template <typename SIMDPolicy>
Ray::Cpu::Renderer<SIMDPolicy>::Renderer(const settings_t &s, ILog *log) : log_(log), use_wide_bvh_(s.use_wide_bvh) {
    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);

    Resize(s.w, s.h);
}

template <typename SIMDPolicy> Ray::SceneBase *Ray::Cpu::Renderer<SIMDPolicy>::CreateScene() {
    return new Cpu::Scene(log_, use_wide_bvh_);
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::RenderScene(const SceneBase *scene, RegionContext &region) {
    const auto s = dynamic_cast<const Cpu::Scene *>(scene);
    if (!s) {
        return;
    }

    const camera_t &cam = s->cams_[s->current_cam()._index].cam;

    std::shared_lock<std::shared_timed_mutex> scene_lock(s->mtx_);

    const scene_data_t sc_data = {s->env_,
                                  s->mesh_instances_.empty() ? nullptr : &s->mesh_instances_[0],
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

    PassData<SIMDPolicy> &p = get_per_thread_pass_data<SIMDPolicy>();

    // allocate aux data on demand
    if (cam.pass_settings.flags & (Bitmask<ePassFlags>{ePassFlags::OutputBaseColor} | ePassFlags::OutputDepthNormals)) {
        // TODO: Skip locking here
        std::lock_guard<std::mutex> _(mtx_);

        if (cam.pass_settings.flags & ePassFlags::OutputBaseColor) {
            base_color_buf_.resize(w_ * h_);
        } else if (!base_color_buf_.empty()) {
            base_color_buf_ = {};
        }
        if (cam.pass_settings.flags & ePassFlags::OutputDepthNormals) {
            depth_normals_buf_.resize(w_ * h_);
        } else if (!depth_normals_buf_.empty()) {
            depth_normals_buf_ = {};
        }
    }

    using namespace std::chrono;

    const auto time_start = high_resolution_clock::now();
    time_point<high_resolution_clock> time_after_ray_gen;

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    if (cam.type != eCamType::Geo) {
        SIMDPolicy::GeneratePrimaryRays(cam, rect, w_, h_, &region.halton_seq[hi], region.iteration,
                                        required_samples_.data(), p.primary_rays);

        p.intersections.resize(p.primary_rays.size());
        for (auto &inter : p.intersections) {
            inter = {};
            inter.t = cam.clip_end - cam.clip_start;
        }

        time_after_ray_gen = high_resolution_clock::now();

        if (macro_tree_root != 0xffffffff) {
            SIMDPolicy::TraceRays(p.primary_rays, cam.pass_settings.min_transp_depth,
                                  cam.pass_settings.max_transp_depth, sc_data, macro_tree_root, false, s->tex_storages_,
                                  &region.halton_seq[hi + RAND_DIM_BASE_COUNT], p.intersections);
        }
    } else {
        const mesh_instance_t &mi = sc_data.mesh_instances[cam.mi_index];
        SIMDPolicy::SampleMeshInTextureSpace(region.iteration, int(cam.mi_index), int(cam.uv_index),
                                             sc_data.meshes[mi.mesh_index], sc_data.transforms[mi.tr_index],
                                             sc_data.vtx_indices, sc_data.vertices, rect, w_, h_,
                                             &region.halton_seq[hi], p.primary_rays, p.intersections);

        time_after_ray_gen = high_resolution_clock::now();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / float(region.iteration);

    const auto time_after_prim_trace = high_resolution_clock::now();

    p.secondary_rays.resize(p.primary_rays.size());
    p.shadow_rays.resize(p.primary_rays.size());

    int secondary_rays_count = 0, shadow_rays_count = 0;

    SIMDPolicy::ShadePrimary(cam.pass_settings, p.intersections, p.primary_rays,
                             &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data, macro_tree_root, s->tex_storages_,
                             &p.secondary_rays[0], &secondary_rays_count, &p.shadow_rays[0], &shadow_rays_count, w_,
                             mix_factor, temp_buf_.data(), base_color_buf_.data(), depth_normals_buf_.data());

    const auto time_after_prim_shade = high_resolution_clock::now();

    SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_direct, sc_data,
                                macro_tree_root, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], s->tex_storages_, w_,
                                temp_buf_.data());

    const auto time_after_prim_shadow = high_resolution_clock::now();
    duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{},
        secondary_shadow_time{};

    p.hash_values.resize(p.primary_rays.size());
    // p.head_flags.resize(rect.w * rect.h);
    p.scan_values.resize(rect.w * rect.h);
    p.chunks.resize(rect.w * rect.h);
    p.chunks_temp.resize(rect.w * rect.h);
    // p.skeleton.resize(rect.w * rect.h);

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; ++bounce) {
        const auto time_secondary_sort_start = high_resolution_clock::now();

        secondary_rays_count = SIMDPolicy::SortRays_CPU(
            Span<typename SIMDPolicy::RayDataType>{&p.secondary_rays[0], secondary_rays_count}, root_min, cell_size,
            &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

#if 0 // debug hash values
        static std::vector<simd_fvec3> color_table;
        if (color_table.empty()) {
            for (int i = 0; i < 1024; i++) {
                color_table.emplace_back(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX);
            }
        }

        for (int i = 0; i < secondary_rays_count; i++) {
            const auto &r = p.secondary_rays[i];

            const int x = r.id.x;
            const int y = r.id.y;

            const simd_fvec3 &c = color_table[hash(p.hash_values[i]) % 1024];

            color_rgba_t col = { c[0], c[1], c[2], 1.0f };
            temp_buf_.SetPixel(x, y, col);
        }
#endif

        const auto time_secondary_trace_start = high_resolution_clock::now();

        for (int i = 0; i < secondary_rays_count; i++) {
            p.intersections[i] = {};
        }

        SIMDPolicy::TraceRays(Span<typename SIMDPolicy::RayDataType>{p.secondary_rays.data(), secondary_rays_count},
                              cam.pass_settings.min_transp_depth, cam.pass_settings.max_transp_depth, sc_data,
                              macro_tree_root, true, s->tex_storages_, &region.halton_seq[hi + RAND_DIM_BASE_COUNT],
                              p.intersections);

        const auto time_secondary_shade_start = high_resolution_clock::now();

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        shadow_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);

        SIMDPolicy::ShadeSecondary(
            cam.pass_settings, Span<typename SIMDPolicy::HitDataType>{p.intersections.data(), rays_count},
            Span<typename SIMDPolicy::RayDataType>{p.primary_rays.data(), rays_count},
            &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data, macro_tree_root, s->tex_storages_,
            &p.secondary_rays[0], &secondary_rays_count, &p.shadow_rays[0], &shadow_rays_count, w_, temp_buf_.data());

        const auto time_secondary_shadow_start = high_resolution_clock::now();

        SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                    cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_indirect, sc_data,
                                    macro_tree_root, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], s->tex_storages_, w_,
                                    temp_buf_.data());

        const auto time_secondary_shadow_end = high_resolution_clock::now();
        secondary_sort_time += duration<double, std::micro>{time_secondary_trace_start - time_secondary_sort_start};
        secondary_trace_time += duration<double, std::micro>{time_secondary_shade_start - time_secondary_trace_start};
        secondary_shade_time += duration<double, std::micro>{time_secondary_shadow_start - time_secondary_shade_start};
        secondary_shadow_time += duration<double, std::micro>{time_secondary_shadow_end - time_secondary_shadow_start};
    }

    scene_lock.unlock();

    Ref::tonemap_params_t tonemap_params;
    tonemap_params.view_transform = cam.view_transform;
    tonemap_params.inv_gamma = (1.0f / cam.gamma);

    Ref::simd_fvec4 exposure = std::pow(2.0f, cam.exposure);
    exposure.set<3>(1.0f);

    float variance_threshold = region.iteration > cam.pass_settings.min_samples
                                   ? 0.5f * cam.pass_settings.variance_threshold * cam.pass_settings.variance_threshold
                                   : 0.0f;

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
        variance_threshold_ = variance_threshold;
    }

    color_rgba_t *clean_buf = dual_buf_[(region.iteration - 1) % 2].data();

    const float half_mix_factor = 1.0f / float((region.iteration + 1) / 2);
    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            if (required_samples_[y * w_ + x] < region.iteration) {
                continue;
            }

            const Ref::simd_fvec4 new_val = {temp_buf_[y * w_ + x].v, Ref::simd_mem_aligned};

            Ref::simd_fvec4 cur_val = {clean_buf[y * w_ + x].v, Ref::simd_mem_aligned};
            cur_val += (new_val - cur_val) * half_mix_factor;
            cur_val.store_to(clean_buf[y * w_ + x].v, Ref::simd_mem_aligned);
        }
    }

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            auto p1 = Ref::simd_fvec4{dual_buf_[0][y * w_ + x].v, Ref::simd_mem_aligned} * exposure;
            auto p2 = Ref::simd_fvec4{dual_buf_[1][y * w_ + x].v, Ref::simd_mem_aligned} * exposure;

            const int p1_samples = (region.iteration + 1) / 2;
            const int p2_samples = (region.iteration) / 2;

            const float p1_weight = float(p1_samples) / float(region.iteration);
            const float p2_weight = float(p2_samples) / float(region.iteration);

            const Ref::simd_fvec4 untonemapped_res = p1_weight * p1 + p2_weight * p2;
            untonemapped_res.store_to(raw_final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            // Also store as denosed result until DenoiseImage method will be called
            untonemapped_res.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);

            const Ref::simd_fvec4 tonemapped_res = Tonemap(tonemap_params, untonemapped_res);
            tonemapped_res.store_to(final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);

            p1 = reversible_tonemap(p1);
            p2 = reversible_tonemap(p2);

            const Ref::simd_fvec4 variance = 0.5f * (p1 - p2) * (p1 - p2);
            variance.store_to(temp_buf_[y * w_ + x].v, Ref::simd_mem_aligned);

#if DEBUG_ADAPTIVE_SAMPLING
            if (cam.pass_settings.variance_threshold != 0.0f && required_samples_[y * w_ + x] >= region.iteration &&
                (region.iteration % 2)) {
                final_buf_[y * w_ + x].v[0] = 1.0f;
                raw_final_buf_[y * w_ + x].v[0] = 1.0f;
            }
#endif

            if (simd_cast(variance >= variance_threshold_).not_all_zeros()) {
                required_samples_[y * w_ + x] = region.iteration + 1;
            }
        }
    }
}

template <typename SIMDPolicy> void Ray::Cpu::Renderer<SIMDPolicy>::DenoiseImage(const RegionContext &region) {
    using namespace std::chrono;
    const auto denoise_start = high_resolution_clock::now();

    const rect_t &rect = region.rect();

    // TODO: determine radius precisely!
    const int EXT_RADIUS = 8;
    const rect_t rect_ext = {rect.x - EXT_RADIUS, rect.y - EXT_RADIUS, rect.w + 2 * EXT_RADIUS,
                             rect.h + 2 * EXT_RADIUS};

    PassData<SIMDPolicy> &p = get_per_thread_pass_data<SIMDPolicy>();
    p.temp_final_buf.resize(rect_ext.w * rect_ext.h);
    p.variance_buf.resize(rect_ext.w * rect_ext.h);
    p.filtered_variance_buf.resize(rect_ext.w * rect_ext.h);
    if (!base_color_buf_.empty()) {
        p.feature_buf1.resize(rect_ext.w * rect_ext.h);
    } else {
        p.feature_buf1 = {};
    }
    if (!depth_normals_buf_.empty()) {
        p.feature_buf2.resize(rect_ext.w * rect_ext.h);
    } else {
        p.feature_buf2 = {};
    }

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

#define FETCH_BASE_COLOR(_x, _y)                                                                                       \
    base_color_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)]
#define FETCH_DEPTH_NORMALS(_x, _y)                                                                                    \
    depth_normals_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)]

    for (int y = 4; y < rect_ext.h - 4; ++y) {
        for (int x = 4; x < rect_ext.w - 4; ++x) {
            const Ref::simd_fvec4 center_val = {p.variance_buf[(y + 0) * rect_ext.w + x].v, Ref::simd_mem_aligned};

            Ref::simd_fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res += Ref::simd_fvec4(p.variance_buf[(y - i + 1) * rect_ext.w + x].v, Ref::simd_mem_aligned) *
                       GaussWeights[i + 1];
                res += Ref::simd_fvec4(p.variance_buf[(y + i + 1) * rect_ext.w + x].v, Ref::simd_mem_aligned) *
                       GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.filtered_variance_buf[y * rect_ext.w + x].v, Ref::simd_mem_aligned);

            if (!base_color_buf_.empty()) {
                p.feature_buf1[y * rect_ext.w + x] = FETCH_BASE_COLOR(rect_ext.x + x, rect_ext.y + y);
            }
            if (!depth_normals_buf_.empty()) {
                p.feature_buf2[y * rect_ext.w + x] = FETCH_DEPTH_NORMALS(rect_ext.x + x, rect_ext.y + y);
            }
        }
    }

#undef FETCH_BASE_COLOR
#undef FETCH_DEPTH_NORMALS

    Ref::tonemap_params_t tonemap_params;
    float variance_threshold;

    {
        std::lock_guard<std::mutex> _(mtx_);
        tonemap_params = tonemap_params_;
        variance_threshold = variance_threshold_;
    }

    for (int y = 0; y < rect.h; ++y) {
        const int yy = rect.y + y;
        for (int x = 0; x < rect.w; ++x) {
            const int xx = rect.x + x;

            const Ref::simd_fvec4 variance = {
                p.filtered_variance_buf[(y + EXT_RADIUS) * rect_ext.w + (x + EXT_RADIUS)].v, Ref::simd_mem_aligned};
            if (simd_cast(variance >= variance_threshold).not_all_zeros()) {
                required_samples_[yy * w_ + xx] = region.iteration + 1;
            }
        }
    }

    const int NLM_WINDOW_SIZE = 7;
    const int NLM_NEIGHBORHOOD_SIZE = 3;

    static_assert(EXT_RADIUS >= (NLM_WINDOW_SIZE - 1) / 2 + (NLM_NEIGHBORHOOD_SIZE - 1) / 2, "!");

    Ref::JointNLMFilter<NLM_WINDOW_SIZE, NLM_NEIGHBORHOOD_SIZE>(
        p.temp_final_buf.data(), rect_t{EXT_RADIUS, EXT_RADIUS, rect.w, rect.h}, rect_ext.w, 1.0f, 0.45f,
        p.filtered_variance_buf.data(), !p.feature_buf1.empty() ? p.feature_buf1.data() : nullptr, 64.0f,
        !p.feature_buf2.empty() ? p.feature_buf2.data() : nullptr, 32.0f, rect, w_, raw_filtered_buf_.data());

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            auto col = Ref::simd_fvec4(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            col = Ref::reversible_tonemap_invert(col);
            col.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
            col = Tonemap(tonemap_params, col);
            col.store_to(final_buf_[y * w_ + x].v, Ref::simd_mem_aligned);
        }
    }

    const auto denoise_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_denoise_us += (unsigned long long)duration<double, std::micro>{denoise_end - denoise_start}.count();
    }
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::UpdateHaltonSequence(const int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HALTON_COUNT * HALTON_SEQ_LEN]);
    }

    for (int i = 0; i < HALTON_SEQ_LEN; ++i) {
        uint32_t prime_sum = 0;
        for (int j = 0; j < HALTON_COUNT; ++j) {
            seq[i * HALTON_COUNT + j] =
                Ray::ScrambledRadicalInverse(g_primes[j], &permutations_[prime_sum], uint64_t(iteration) + i);
            prime_sum += g_primes[j];
        }
    }
}
