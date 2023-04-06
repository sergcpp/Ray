#include "RendererRef.h"

#include <chrono>
#include <functional>
#include <random>

#include "Halton.h"
#include "SceneRef.h"
#include "UniformIntDistribution.h"

namespace Ray {
namespace Ref {
struct PassData {
    aligned_vector<ray_data_t> primary_rays;
    aligned_vector<ray_data_t> secondary_rays;
    aligned_vector<shadow_ray_t> shadow_rays;
    aligned_vector<hit_data_t> intersections;

    aligned_vector<color_rgba_t, 16> temp_final_buf;
    aligned_vector<color_rgba_t, 16> variance_buf;
    aligned_vector<color_rgba_t, 16> filtered_variance_buf;

    std::vector<uint32_t> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;

    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;
};
} // namespace Ref

thread_local Ray::Ref::PassData g_per_thread_pass_data;
} // namespace Ray

Ray::Ref::Renderer::Renderer(const settings_t &s, ILog *log) : log_(log), use_wide_bvh_(s.use_wide_bvh) {
    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);

    Resize(s.w, s.h);
}

Ray::SceneBase *Ray::Ref::Renderer::CreateScene() { return new Ref::Scene(log_, use_wide_bvh_); }

void Ray::Ref::Renderer::RenderScene(const SceneBase *scene, RegionContext &region) {
    const auto s = dynamic_cast<const Ref::Scene *>(scene);
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

    PassData &p = g_per_thread_pass_data;

    // allocate aux data on demand
    if (cam.pass_settings.flags & (OutputBaseColor | OutputDepthNormals)) {
        // TODO: Skip locking here
        std::lock_guard<std::mutex> _(mtx_);

        if (cam.pass_settings.flags & OutputBaseColor) {
            base_color_buf_.resize(w_ * h_);
        } else if (!base_color_buf_.empty()) {
            base_color_buf_ = {};
        }
        if (cam.pass_settings.flags & OutputDepthNormals) {
            depth_normals_buf_.resize(w_ * h_);
        } else if (!depth_normals_buf_.empty()) {
            depth_normals_buf_ = {};
        }
    }

    using namespace std::chrono;

    const auto time_start = high_resolution_clock::now();
    time_point<high_resolution_clock> time_after_ray_gen;

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    if (cam.type != Geo) {
        GeneratePrimaryRays(cam, rect, w_, h_, &region.halton_seq[hi], p.primary_rays);

        time_after_ray_gen = high_resolution_clock::now();

        p.intersections.resize(p.primary_rays.size());

        for (size_t i = 0; i < p.primary_rays.size(); i++) {
            ray_data_t &r = p.primary_rays[i];

            hit_data_t &inter = p.intersections[i];
            inter = {};
            inter.t = cam.clip_end;

            if (macro_tree_root != 0xffffffff) {
                IntersectScene(r, cam.pass_settings.min_transp_depth, cam.pass_settings.max_transp_depth,
                               &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data, macro_tree_root, s->tex_storages_,
                               inter);
                // IntersectAreaLights(r, sc_data.lights, sc_data.visible_lights, sc_data.transforms, inter);
            }
        }
    } else {
        const mesh_instance_t &mi = sc_data.mesh_instances[cam.mi_index];
        SampleMeshInTextureSpace(region.iteration, int(cam.mi_index), int(cam.uv_index), sc_data.meshes[mi.mesh_index],
                                 sc_data.transforms[mi.tr_index], sc_data.vtx_indices, sc_data.vertices, rect, w_, h_,
                                 &region.halton_seq[hi], p.primary_rays, p.intersections);

        time_after_ray_gen = high_resolution_clock::now();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / float(region.iteration);

    const auto time_after_prim_trace = high_resolution_clock::now();

    p.secondary_rays.resize(rect.w * rect.h);
    p.shadow_rays.resize(rect.w * rect.h);

    int secondary_rays_count = 0, shadow_rays_count = 0;

    for (size_t i = 0; i < p.intersections.size(); i++) {
        const ray_data_t &r = p.primary_rays[i];
        const hit_data_t &inter = p.intersections[i];

        const int x = (r.xy >> 16) & 0x0000ffff;
        const int y = r.xy & 0x0000ffff;

        color_rgba_t base_color = {}, depth_normal = {};
        const color_rgba_t col =
            ShadeSurface(cam.pass_settings, inter, r, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data,
                         macro_tree_root, s->tex_storages_, &p.secondary_rays[0], &secondary_rays_count,
                         &p.shadow_rays[0], &shadow_rays_count, &base_color, &depth_normal);
        temp_buf_[y * w_ + x] = col;
        if (cam.pass_settings.flags & OutputBaseColor) {
            auto old_val = simd_fvec4{base_color_buf_[y * w_ + x].v, simd_mem_aligned};
            old_val += (simd_fvec4{base_color.v, simd_mem_aligned} - old_val) * mix_factor;
            old_val.store_to(base_color_buf_[y * w_ + x].v, simd_mem_aligned);
        }
        if (cam.pass_settings.flags & OutputDepthNormals) {
            auto old_val = simd_fvec4{depth_normals_buf_[y * w_ + x].v, simd_mem_aligned};
            old_val += (simd_fvec4{depth_normal.v, simd_mem_aligned} - old_val) * mix_factor;
            old_val.store_to(depth_normals_buf_[y * w_ + x].v, simd_mem_aligned);
        }
    }

    const auto time_after_prim_shade = high_resolution_clock::now();

    for (int i = 0; i < shadow_rays_count; ++i) {
        const shadow_ray_t &sh_r = p.shadow_rays[i];

        const int x = (sh_r.xy >> 16) & 0x0000ffff;
        const int y = sh_r.xy & 0x0000ffff;

        simd_fvec4 rc =
            IntersectScene(sh_r, cam.pass_settings.max_transp_depth, sc_data, macro_tree_root, s->tex_storages_);
        rc *= IntersectAreaLights(sh_r, sc_data.lights, sc_data.blocker_lights, sc_data.transforms);

        auto old_val = simd_fvec4{temp_buf_[y * w_ + x].v, simd_mem_aligned};
        old_val += rc;
        old_val.store_to(temp_buf_[y * w_ + x].v, simd_mem_aligned);
    }

    const auto time_after_prim_shadow = high_resolution_clock::now();
    duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{},
        secondary_shadow_time{};

    p.hash_values.resize(secondary_rays_count);
    // p.head_flags.resize(secondary_rays_count);
    p.scan_values.resize(secondary_rays_count);
    p.chunks.resize(secondary_rays_count);
    p.chunks_temp.resize(secondary_rays_count);
    // p.skeleton.resize(secondary_rays_count);

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; ++bounce) {
        const auto time_secondary_sort_start = high_resolution_clock::now();

        SortRays_CPU(&p.secondary_rays[0], size_t(secondary_rays_count), root_min, cell_size, &p.hash_values[0],
                     &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

#if 0 // debug hash values
        static std::vector<simd_fvec3> color_table;
        if (color_table.empty()) {
            for (int i = 0; i < 1024; i++) {
                color_table.emplace_back(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX);
            }
        }

        for (int i = 0; i < secondary_rays_count; i++) {
            const ray_data_t &r = p.secondary_rays[i];

            const int x = r.id.x;
            const int y = r.id.y;

            const simd_fvec3 &c = color_table[hash(p.hash_values[i]) % 1024];

            color_rgba_t col = { c[0], c[1], c[2], 1.0f };
            temp_buf_.SetPixel(x, y, col);
        }
#endif

        const auto time_secondary_trace_start = high_resolution_clock::now();

        for (int i = 0; i < secondary_rays_count; i++) {
            ray_data_t &r = p.secondary_rays[i];

            hit_data_t &inter = p.intersections[i];
            inter = {};

            IntersectScene(r, cam.pass_settings.min_transp_depth, cam.pass_settings.max_transp_depth,
                           &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data, macro_tree_root, s->tex_storages_,
                           inter);
            if (r.depth & 0x00ffffff) { // not only a transparency ray
                IntersectAreaLights(r, sc_data.lights, sc_data.visible_lights, sc_data.transforms, inter);
            }
        }

        const auto time_secondary_shade_start = high_resolution_clock::now();

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        shadow_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);

        for (int i = 0; i < rays_count; ++i) {
            const ray_data_t &r = p.primary_rays[i];
            const hit_data_t &inter = p.intersections[i];

            const int x = (r.xy >> 16) & 0x0000ffff;
            const int y = r.xy & 0x0000ffff;

            color_rgba_t col =
                ShadeSurface(cam.pass_settings, inter, r, &region.halton_seq[hi + RAND_DIM_BASE_COUNT], sc_data,
                             macro_tree_root, s->tex_storages_, &p.secondary_rays[0], &secondary_rays_count,
                             &p.shadow_rays[0], &shadow_rays_count, nullptr, nullptr);
            col.v[3] = 0.0f;

            auto old_val = simd_fvec4{temp_buf_[y * w_ + x].v, simd_mem_aligned};
            old_val += simd_fvec4{col.v};
            old_val.store_to(temp_buf_[y * w_ + x].v, simd_mem_aligned);
        }

        const auto time_secondary_shadow_start = high_resolution_clock::now();

        for (int i = 0; i < shadow_rays_count; ++i) {
            const shadow_ray_t &sh_r = p.shadow_rays[i];

            const int x = (sh_r.xy >> 16) & 0x0000ffff;
            const int y = sh_r.xy & 0x0000ffff;

            simd_fvec4 rc =
                IntersectScene(sh_r, cam.pass_settings.max_transp_depth, sc_data, macro_tree_root, s->tex_storages_);
            rc *= IntersectAreaLights(sh_r, sc_data.lights, sc_data.blocker_lights, sc_data.transforms);

            auto old_val = simd_fvec4{temp_buf_[y * w_ + x].v, simd_mem_aligned};
            old_val += rc;
            old_val.store_to(temp_buf_[y * w_ + x].v, simd_mem_aligned);
        }

        const auto time_secondary_shadow_end = high_resolution_clock::now();
        secondary_sort_time += duration<double, std::micro>{time_secondary_trace_start - time_secondary_sort_start};
        secondary_trace_time += duration<double, std::micro>{time_secondary_shade_start - time_secondary_trace_start};
        secondary_shade_time += duration<double, std::micro>{time_secondary_shadow_start - time_secondary_shade_start};
        secondary_shadow_time += duration<double, std::micro>{time_secondary_shadow_end - time_secondary_shadow_start};
    }

    scene_lock.unlock();

    tonemap_params_t tonemap_params;
    tonemap_params.exposure = std::pow(2.0f, cam.exposure);
    tonemap_params.inv_gamma = (1.0f / cam.gamma);
    tonemap_params.srgb = (cam.dtype == SRGB);
    tonemap_params.clamp = (cam.pass_settings.flags & Clamp) != 0;

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
            simd_fvec4 p1 = {dual_buf_[0][y * w_ + x].v, simd_mem_aligned};
            simd_fvec4 p2 = {dual_buf_[1][y * w_ + x].v, simd_mem_aligned};

            const int p1_samples = (region.iteration + 1) / 2;
            const int p2_samples = (region.iteration) / 2;

            const float p1_weight = float(p1_samples) / float(region.iteration);
            const float p2_weight = float(p2_samples) / float(region.iteration);

            const simd_fvec4 untonemapped_res = p1_weight * p1 + p2_weight * p2;
            untonemapped_res.store_to(raw_final_buf_[y * w_ + x].v, simd_mem_aligned);
            // Also store as denosed result until DenoiseImage method will be called
            untonemapped_res.store_to(raw_filtered_buf_[y * w_ + x].v, simd_mem_aligned);

            const simd_fvec4 tonemapped_res = clamp_and_gamma_correct(tonemap_params, untonemapped_res);
            tonemapped_res.store_to(final_buf_[y * w_ + x].v, simd_mem_aligned);

            p1 = reversible_tonemap(p1);
            p2 = reversible_tonemap(p2);

            const simd_fvec4 variance = 0.5f * (p1 - p2) * (p1 - p2);
            variance.store_to(temp_buf_[y * w_ + x].v, simd_mem_aligned);
        }
    }
}

void Ray::Ref::Renderer::DenoiseImage(const RegionContext &region) {
    using namespace std::chrono;
    const auto denoise_start = high_resolution_clock::now();

    const rect_t &rect = region.rect();

    // TODO: determine radius precisely!
    const int EXT_RADIUS = 8;
    const rect_t rect_ext = {rect.x - EXT_RADIUS, rect.y - EXT_RADIUS, rect.w + 2 * EXT_RADIUS,
                             rect.h + 2 * EXT_RADIUS};

    PassData &p = g_per_thread_pass_data;
    p.temp_final_buf.resize(rect_ext.w * rect_ext.h);
    p.variance_buf.resize(rect_ext.w * rect_ext.h);
    p.filtered_variance_buf.resize(rect_ext.w * rect_ext.h);

#define FETCH_FINAL_BUF(_x, _y)                                                                                        \
    simd_fvec4(raw_final_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,           \
               simd_mem_aligned)
#define FETCH_VARIANCE(_x, _y)                                                                                         \
    simd_fvec4(temp_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,                \
               simd_mem_aligned)

    static const float GaussWeights[] = {0.2270270270f, 0.1945945946f, 0.1216216216f, 0.0540540541f, 0.0162162162f};

    for (int y = 0; y < rect_ext.h; ++y) {
        const int yy = rect_ext.y + y;
        for (int x = 0; x < rect_ext.w; ++x) {
            const int xx = rect_ext.x + x;
            const simd_fvec4 center_col = reversible_tonemap(FETCH_FINAL_BUF(xx, yy));
            center_col.store_to(p.temp_final_buf[y * rect_ext.w + x].v, simd_mem_aligned);

            const simd_fvec4 center_val = FETCH_VARIANCE(xx, yy);

            simd_fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res += FETCH_VARIANCE(xx - i + 1, yy) * GaussWeights[i + 1];
                res += FETCH_VARIANCE(xx + i + 1, yy) * GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.variance_buf[y * rect_ext.w + x].v, simd_mem_aligned);
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

    NLMFilter<NLM_WINDOW_SIZE, NLM_NEIGHBORHOOD_SIZE>(
        p.temp_final_buf.data(), rect_t{EXT_RADIUS, EXT_RADIUS, rect.w, rect.h}, rect_ext.w, 1.0f, 0.45f,
        p.filtered_variance_buf.data(), rect, w_, raw_filtered_buf_.data());

    tonemap_params_t tonemap_params;

    {
        std::lock_guard<std::mutex> _(mtx_);
        tonemap_params = tonemap_params_;
    }

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            simd_fvec4 col = simd_fvec4(raw_filtered_buf_[y * w_ + x].v, simd_mem_aligned);
            col = reversible_tonemap_invert(col);
            col.store_to(raw_filtered_buf_[y * w_ + x].v, simd_mem_aligned);
            col = clamp_and_gamma_correct(tonemap_params, col);
            col.store_to(final_buf_[y * w_ + x].v, simd_mem_aligned);
        }
    }

    const auto denoise_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_denoise_us += (unsigned long long)duration<double, std::micro>{denoise_end - denoise_start}.count();
    }
}

void Ray::Ref::Renderer::UpdateHaltonSequence(const int iteration, std::unique_ptr<float[]> &seq) {
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
