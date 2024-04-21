#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <random>

#include "../Log.h"
#include "../RendererBase.h"
#include "CDFUtils.h"
#include "CoreRef.h"
#include "DenoiseRef.h"
#include "RadCacheRef.h"
#include "SceneCPU.h"
#include "ShadeRef.h"
#include "TonemapRef.h"
#include "UNetFilter.h"

#define DEBUG_ADAPTIVE_SAMPLING 0

namespace Ray {
int round_up(int v, int align);

void WritePFM(const char *base_name, const float values[], int w, int h, int channels);

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
                                                 const uint32_t rand_seq[], const uint32_t rand_seed,
                                                 const float filter_table[], const int iteration,
                                                 const uint16_t required_samples[],
                                                 aligned_vector<Ref::ray_data_t> &out_rays,
                                                 aligned_vector<Ref::hit_data_t> &out_inters) {
        Ref::GeneratePrimaryRays(cam, r, w, h, rand_seq, rand_seed, filter_table, iteration, required_samples, out_rays,
                                 out_inters);
    }

    static force_inline void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh,
                                                      const mesh_instance_t &mi, const uint32_t *vtx_indices,
                                                      const vertex_t *vertices, const rect_t &r, int w, int h,
                                                      const uint32_t rand_seq[], aligned_vector<ray_data_t> &out_rays,
                                                      aligned_vector<hit_data_t> &out_inters) {
        Ref::SampleMeshInTextureSpace(iteration, obj_index, uv_layer, mesh, mi, vtx_indices, vertices, r, w, h,
                                      rand_seq, out_rays, out_inters);
    }

    static force_inline void IntersectScene(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth,
                                            const uint32_t rand_seq[], const uint32_t random_seed, const int iteration,
                                            const scene_data_t &sc, uint32_t root_index,
                                            const Cpu::TexStorageBase *const textures[], Span<hit_data_t> out_inter) {
        Ref::IntersectScene(rays, min_transp_depth, max_transp_depth, rand_seq, random_seed, iteration, sc, root_index,
                            textures, out_inter);
    }

    static force_inline void TraceRays(Span<ray_data_t> rays, int min_transp_depth, int max_transp_depth,
                                       const scene_data_t &sc, uint32_t node_index, bool trace_lights,
                                       const Cpu::TexStorageBase *const textures[], const uint32_t rand_seq[],
                                       const uint32_t random_seed, const int iteration, Span<hit_data_t> out_inter) {
        Ref::TraceRays(rays, min_transp_depth, max_transp_depth, sc, node_index, trace_lights, textures, rand_seq,
                       random_seed, iteration, out_inter);
    }

    static force_inline void TraceShadowRays(Span<const shadow_ray_t> rays, int max_transp_depth, float clamp_val,
                                             const scene_data_t &sc, uint32_t node_index, const uint32_t rand_seq[],
                                             const uint32_t random_seed, const int iteration,
                                             const Cpu::TexStorageBase *const textures[], int img_w,
                                             color_rgba_t *out_color) {
        Ref::TraceShadowRays(rays, max_transp_depth, clamp_val, sc, node_index, rand_seq, random_seed, iteration,
                             textures, img_w, out_color);
    }

    static force_inline int SortRays_CPU(Span<ray_data_t> rays, const float root_min[3], const float cell_size[3],
                                         uint32_t *hash_values, uint32_t *scan_values, ray_chunk_t *chunks,
                                         ray_chunk_t *chunks_temp) {
        return Ref::SortRays_CPU(rays, root_min, cell_size, hash_values, scan_values, chunks, chunks_temp);
    }

    static force_inline void ShadePrimary(
        const pass_settings_t &ps, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
        const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration, const eSpatialCacheMode cache_mode,
        const scene_data_t &sc, const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
        int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count, int img_w,
        float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color, color_rgba_t *out_depth_normal) {
        Ref::ShadePrimary(ps, inters, rays, rand_seq, rand_seed, iteration, cache_mode, sc, textures,
                          out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count, img_w,
                          mix_factor, out_color, out_base_color, out_depth_normal);
    }

    static force_inline void ShadeSecondary(const pass_settings_t &ps, const float clamp_direct,
                                            Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                                            const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                                            const eSpatialCacheMode cache_mode, const scene_data_t &sc,
                                            const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                                            int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                                            int *out_shadow_rays_count, int img_w, color_rgba_t *out_color,
                                            color_rgba_t *out_base_color, color_rgba_t *out_depth_normal) {
        Ref::ShadeSecondary(ps, clamp_direct, inters, rays, rand_seq, rand_seed, iteration, cache_mode, sc, textures,
                            out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count, img_w,
                            out_color, out_base_color, out_depth_normal);
    }

    static force_inline void SpatialCacheUpdate(const cache_grid_params_t &params, Span<const hit_data_t> inters,
                                                Span<const ray_data_t> rays, Span<cache_data_t> cache_data,
                                                const color_rgba_t radiance[], const color_rgba_t depth_normals[],
                                                int img_w, Span<uint64_t> entries,
                                                Span<packed_cache_voxel_t> voxels_curr) {
        Ref::SpatialCacheUpdate(params, inters, rays, cache_data, radiance, depth_normals, img_w, entries, voxels_curr);
    }

    template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels,
              ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
              ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void Convolution3x3_GEMM(const float data1[], const float data2[], const float data3[],
                                                 const rect_t &rect, int in_w, int in_h, int w, int h, int stride,
                                                 const float weights[], const float biases[], float output[],
                                                 int output_stride) {
        Ref::Convolution3x3_GEMM<InChannels1, InChannels2, InChannels3, PxPitch, OutChannels, PreOp1, PreOp2, PreOp3,
                                 PostOp, Activation>(data1, data2, data3, rect, in_w, in_h, w, h, stride, weights,
                                                     biases, output, output_stride);
    }

    template <int InChannels, int OutChannels, int OutPxPitch = OutChannels, ePostOp PostOp = ePostOp::None,
              eActivation Activation = eActivation::ReLU>
    static force_inline void Convolution3x3_Direct(const float data[], const rect_t &rect, int w, int h, int stride,
                                                   const float weights[], const float biases[], float output[],
                                                   int output_stride) {
        Ref::Convolution3x3_Direct<InChannels, OutChannels, OutPxPitch, PostOp, Activation>(
            data, rect, w, h, stride, weights, biases, output, output_stride);
    }

    template <int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
              ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void ConvolutionConcat3x3_Direct(const float data1[], const float data2[], const rect_t &rect,
                                                         int w, int h, int stride1, int stride2, const float weights[],
                                                         const float biases[], float output[], int output_stride) {
        Ref::ConvolutionConcat3x3_Direct<InChannels1, InChannels2, OutChannels, PreOp1, PostOp, Activation>(
            data1, data2, rect, w, h, stride1, stride2, weights, biases, output, output_stride);
    }

    template <int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
              ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
              ePreOp PreOp4 = ePreOp::None, ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
    static force_inline void
    ConvolutionConcat3x3_1Direct_2GEMM(const float data1[], const float data2[], const float data3[],
                                       const float data4[], const rect_t &rect, int w, int h, int w2, int h2,
                                       int stride1, int stride2, const float weights[], const float biases[],
                                       float output[], int output_stride) {
        Ref::ConvolutionConcat3x3_1Direct_2GEMM<InChannels1, InChannels2, InChannels3, InChannels4, PxPitch2,
                                                OutChannels, PreOp1, PreOp2, PreOp3, PreOp4, PostOp, Activation>(
            data1, data2, data3, data4, rect, w, h, w2, h2, stride1, stride2, weights, biases, output, output_stride);
    }

    static force_inline void ClearBorders(const rect_t &rect, int w, int h, bool downscaled, int out_channels,
                                          float output[]) {
        Ref::ClearBorders(rect, w, h, downscaled, out_channels, output);
    }
};
} // namespace Ref
namespace Cpu {
template <typename SIMDPolicy> class Renderer : public RendererBase, private SIMDPolicy {
    ILog *log_;

    bool use_tex_compression_, use_spatial_cache_;
    aligned_vector<color_rgba_t, 16> full_buf_, half_buf_, base_color_buf_, depth_normals_buf_, temp_buf_, final_buf_,
        raw_filtered_buf_;
    std::vector<uint16_t> required_samples_;

    std::mutex mtx_;

    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    ePixelFilter filter_table_filter_ = ePixelFilter(-1);
    float filter_table_width_ = 0.0f;
    std::vector<float> filter_table_;
    void UpdateFilterTable(ePixelFilter filter, float filter_width);

    Ref::tonemap_params_t tonemap_params_;
    float variance_threshold_ = 0.0f;

    std::vector<cache_data_t> temp_cache_data_;

    aligned_vector<float, 64> unet_weights_;
    unet_weight_offsets_t unet_offsets_;
    bool unet_alias_memory_ = true;
    aligned_vector<float, 64> unet_tensors_heap_;
    struct {
        float *encConv0 = nullptr;
        float *pool1 = nullptr;
        float *pool2 = nullptr;
        float *pool3 = nullptr;
        float *pool4 = nullptr;
        float *enc_conv5a = nullptr;
        float *upsample4 = nullptr;
        float *dec_conv4a = nullptr;
        float *upsample3 = nullptr;
        float *dec_conv3a = nullptr;
        float *upsample2 = nullptr;
        float *dec_conv2a = nullptr;
        float *upsample1 = nullptr;
        float *dec_conv1a = nullptr;
        float *dec_conv1b = nullptr;
    } unet_tensors_;
    SmallVector<int, 2> unet_alias_dependencies_[UNetFilterPasses];
    void UpdateUNetFilterMemory();

  public:
    Renderer(const settings_t &s, ILog *log);

    eRendererType type() const override { return SIMDPolicy::type(); }

    ILog *log() const override { return log_; }

    const char *device_name() const override { return "CPU"; }

    bool is_spatial_caching_enabled() const override { return use_spatial_cache_; }

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
            full_buf_.assign(w * h, {});
            full_buf_.shrink_to_fit();
            half_buf_.assign(w * h, {});
            half_buf_.shrink_to_fit();
            base_color_buf_.assign(w * h, {});
            base_color_buf_.shrink_to_fit();
            depth_normals_buf_.assign(w * h, {});
            depth_normals_buf_.shrink_to_fit();
            required_samples_.assign(w * h, 0xffff);
            required_samples_.shrink_to_fit();
            temp_buf_.assign(w * h, {});
            temp_buf_.shrink_to_fit();
            final_buf_.assign(w * h, {});
            final_buf_.shrink_to_fit();
            raw_filtered_buf_.assign(w * h, {});
            raw_filtered_buf_.shrink_to_fit();

            if (use_spatial_cache_) {
                temp_cache_data_.assign((w / RAD_CACHE_DOWNSAMPLING_FACTOR) * (h / RAD_CACHE_DOWNSAMPLING_FACTOR), {});
                temp_cache_data_.shrink_to_fit();
            }

            w_ = w;
            h_ = h;

            UpdateUNetFilterMemory();
        }
    }

    void Clear(const color_rgba_t &c) override {
        full_buf_.assign(w_ * h_, c);
        half_buf_.assign(w_ * h_, c);
        required_samples_.assign(w_ * h_, 0xffff);
    }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase &scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;
    void DenoiseImage(int pass, const RegionContext &region) override;

    void UpdateSpatialCache(const SceneBase &scene, RegionContext &region) override;
    void ResolveSpatialCache(const SceneBase &scene,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }

    void InitUNetFilter(bool alias_memory, unet_filter_properties_t &out_props) override;
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
Ray::Cpu::Renderer<SIMDPolicy>::Renderer(const settings_t &s, ILog *log)
    : log_(log), use_tex_compression_(s.use_tex_compression), use_spatial_cache_(s.use_spatial_cache) {
    log->Info("===========================================");
    log->Info("Compression  is %s", use_tex_compression_ ? "enabled" : "disabled");
    log->Info("SpatialCache is %s", use_spatial_cache_ ? "enabled" : "disabled");
    log->Info("===========================================");

    Resize(s.w, s.h);
}

template <typename SIMDPolicy> Ray::SceneBase *Ray::Cpu::Renderer<SIMDPolicy>::CreateScene() {
    return new Cpu::Scene(log_, true /* use_wide_bvh */, use_tex_compression_, use_spatial_cache_);
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::RenderScene(const SceneBase &scene, RegionContext &region) {
    using namespace std::chrono;

    const auto &s = dynamic_cast<const Cpu::Scene &>(scene);

    std::shared_lock<std::shared_timed_mutex> scene_lock(s.mtx_);

    const camera_t &cam = s.cams_[s.current_cam()._index];

    ++region.iteration;

    cache_grid_params_t cache_grid_params;
    memcpy(cache_grid_params.cam_pos_curr, cam.origin, 3 * sizeof(float));
    cache_grid_params.exposure = std::pow(2.0f, cam.exposure);

    const scene_data_t sc_data = {s.env_,
                                  s.mesh_instances_.empty() ? nullptr : &s.mesh_instances_[0],
                                  s.mi_indices_.empty() ? nullptr : &s.mi_indices_[0],
                                  s.meshes_.empty() ? nullptr : &s.meshes_[0],
                                  s.vtx_indices_.empty() ? nullptr : &s.vtx_indices_[0],
                                  s.vertices_.empty() ? nullptr : &s.vertices_[0],
                                  s.nodes_.empty() ? nullptr : &s.nodes_[0],
                                  s.wnodes_.empty() ? nullptr : &s.wnodes_[0],
                                  s.tris_.empty() ? nullptr : &s.tris_[0],
                                  s.tri_indices_.empty() ? nullptr : &s.tri_indices_[0],
                                  s.mtris_.data(),
                                  s.tri_materials_.empty() ? nullptr : &s.tri_materials_[0],
                                  s.materials_.empty() ? nullptr : &s.materials_[0],
                                  {s.lights_.data(), s.lights_.capacity()},
                                  {s.li_indices_},
                                  s.visible_lights_count_,
                                  s.blocker_lights_count_,
                                  {s.light_nodes_},
                                  {s.light_wnodes_},
                                  cache_grid_params,
                                  {s.spatial_cache_entries_},
                                  {s.spatial_cache_voxels_prev_}};

    const uint32_t tlas_root = s.tlas_root_;

    float root_min[3], root_max[3], cell_size[3];
    s.GetBounds(root_min, root_max);
    UNROLLED_FOR(i, 3, { cell_size[i] = (root_max[i] - root_min[i]) / 255; })

    const rect_t &rect = region.rect();

    { // Check filter table
        // TODO: Skip locking here
        std::lock_guard<std::mutex> _(mtx_);
        if (cam.filter != filter_table_filter_ || cam.filter_width != filter_table_width_) {
            UpdateFilterTable(cam.filter, cam.filter_width);
            filter_table_filter_ = cam.filter;
            filter_table_width_ = cam.filter_width;
        }
    }

    PassData<SIMDPolicy> &p = get_per_thread_pass_data<SIMDPolicy>();

    // make sure we will not use stale values
    get_per_thread_BCCache<1>().Invalidate();
    get_per_thread_BCCache<2>().Invalidate();
    get_per_thread_BCCache<4>().Invalidate();

    const auto time_start = high_resolution_clock::now();
    time_point<high_resolution_clock> time_after_ray_gen;

    const uint32_t *rand_seq = __pmj02_samples;
    const uint32_t rand_seed = Ref::hash((region.iteration - 1) / RAND_SAMPLES_COUNT);

    if (cam.type != eCamType::Geo) {
        SIMDPolicy::GeneratePrimaryRays(cam, rect, w_, h_, rand_seq, rand_seed, filter_table_.data(), region.iteration,
                                        required_samples_.data(), p.primary_rays, p.intersections);

        time_after_ray_gen = high_resolution_clock::now();

        if (tlas_root != 0xffffffff) {
            SIMDPolicy::TraceRays(p.primary_rays, cam.pass_settings.min_transp_depth,
                                  cam.pass_settings.max_transp_depth, sc_data, tlas_root, false, s.tex_storages_,
                                  rand_seq, rand_seed, region.iteration, p.intersections);
        }
    } else {
        const mesh_instance_t &mi = sc_data.mesh_instances[cam.mi_index];
        SIMDPolicy::SampleMeshInTextureSpace(region.iteration, int(cam.mi_index), int(cam.uv_index),
                                             sc_data.meshes[mi.mesh_index], mi, sc_data.vtx_indices, sc_data.vertices,
                                             rect, w_, h_, rand_seq, p.primary_rays, p.intersections);

        time_after_ray_gen = high_resolution_clock::now();
    }

    // factor used to compute incremental average
    const float mix_factor = 1.0f / float(region.iteration);

    const auto time_after_prim_trace = high_resolution_clock::now();

    p.secondary_rays.resize(p.primary_rays.size());
    p.shadow_rays.resize(p.primary_rays.size());

    int secondary_rays_count = 0, shadow_rays_count = 0;

    const eSpatialCacheMode cache_mode = use_spatial_cache_ ? eSpatialCacheMode::Query : eSpatialCacheMode::None;
    SIMDPolicy::ShadePrimary(cam.pass_settings, p.intersections, p.primary_rays, rand_seq, rand_seed, region.iteration,
                             cache_mode, sc_data, s.tex_storages_, &p.secondary_rays[0], &secondary_rays_count,
                             &p.shadow_rays[0], &shadow_rays_count, w_, mix_factor, temp_buf_.data(),
                             base_color_buf_.data(), depth_normals_buf_.data());

    const auto time_after_prim_shade = high_resolution_clock::now();

    SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_direct, sc_data, tlas_root,
                                rand_seq, rand_seed, region.iteration, s.tex_storages_, w_, temp_buf_.data());

    const auto time_after_prim_shadow = high_resolution_clock::now();
    duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{},
        secondary_shadow_time{};

    p.hash_values.resize(p.primary_rays.size());
    p.scan_values.resize(round_up(rect.w, 4) * round_up(rect.h, 4));
    p.chunks.resize(round_up(rect.w, 4) * round_up(rect.h, 4));
    p.chunks_temp.resize(round_up(rect.w, 4) * round_up(rect.h, 4));

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; ++bounce) {
        const auto time_secondary_sort_start = high_resolution_clock::now();

        secondary_rays_count = SIMDPolicy::SortRays_CPU(
            Span<typename SIMDPolicy::RayDataType>{&p.secondary_rays[0], secondary_rays_count}, root_min, cell_size,
            &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

#if 0 // debug hash values
        static std::vector<fvec3> color_table;
        if (color_table.empty()) {
            for (int i = 0; i < 1024; i++) {
                color_table.emplace_back(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX);
            }
        }

        for (int i = 0; i < secondary_rays_count; i++) {
            const auto &r = p.secondary_rays[i];

            const int x = r.id.x;
            const int y = r.id.y;

            const fvec3 &c = color_table[hash(p.hash_values[i]) % 1024];

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
                              tlas_root, true, s.tex_storages_, rand_seq, rand_seed, region.iteration, p.intersections);

        const auto time_secondary_shade_start = high_resolution_clock::now();

        int rays_count = secondary_rays_count;
        secondary_rays_count = 0;
        shadow_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);

        // Use direct clamping value only for the first intersection with lightsource
        const float clamp_direct = (bounce == 1) ? cam.pass_settings.clamp_direct : cam.pass_settings.clamp_indirect;
        SIMDPolicy::ShadeSecondary(
            cam.pass_settings, clamp_direct, Span<typename SIMDPolicy::HitDataType>{p.intersections.data(), rays_count},
            Span<typename SIMDPolicy::RayDataType>{p.primary_rays.data(), rays_count}, rand_seq, rand_seed,
            region.iteration, cache_mode, sc_data, s.tex_storages_, &p.secondary_rays[0], &secondary_rays_count,
            &p.shadow_rays[0], &shadow_rays_count, w_, temp_buf_.data(), nullptr, nullptr);

        const auto time_secondary_shadow_start = high_resolution_clock::now();

        SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                    cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_indirect, sc_data,
                                    tlas_root, rand_seq, rand_seed, region.iteration, s.tex_storages_, w_,
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

    Ref::fvec4 exposure = std::pow(2.0f, cam.exposure);
    exposure.set<3>(1.0f);

    const float variance_threshold =
        region.iteration > cam.pass_settings.min_samples
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

    const bool is_class_a = popcount(uint32_t(region.iteration - 1) & 0xaaaaaaaa) & 1;
    const float half_mix_factor = 1.0f / float((region.iteration + 1) / 2);
    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            if (required_samples_[y * w_ + x] < region.iteration) {
                continue;
            }

            const auto new_val = Ref::fvec4{temp_buf_[y * w_ + x].v, Ref::vector_aligned} * exposure;
            // accumulate full buffer
            Ref::fvec4 cur_val_full = {full_buf_[y * w_ + x].v, Ref::vector_aligned};
            cur_val_full += (new_val - cur_val_full) * mix_factor;
            cur_val_full.store_to(full_buf_[y * w_ + x].v, Ref::vector_aligned);
            if (is_class_a) {
                // accumulate half buffer
                Ref::fvec4 cur_val_half = {half_buf_[y * w_ + x].v, Ref::vector_aligned};
                cur_val_half += (new_val - cur_val_half) * half_mix_factor;
                cur_val_half.store_to(half_buf_[y * w_ + x].v, Ref::vector_aligned);
            }
        }
    }

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            auto full_val = Ref::fvec4{full_buf_[y * w_ + x].v, Ref::vector_aligned};
            auto half_val = Ref::fvec4{half_buf_[y * w_ + x].v, Ref::vector_aligned};

            // Store as denosed result until DenoiseImage method will be called
            full_val.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::vector_aligned);

            const Ref::fvec4 tonemapped_res = Tonemap(tonemap_params, full_val);
            tonemapped_res.store_to(final_buf_[y * w_ + x].v, Ref::vector_aligned);

            const Ref::fvec4 p1 = reversible_tonemap(max(2.0f * full_val - half_val, 0.0f));
            const Ref::fvec4 p2 = reversible_tonemap(half_val);

            const Ref::fvec4 variance = 0.5f * (p1 - p2) * (p1 - p2);
            variance.store_to(temp_buf_[y * w_ + x].v, Ref::vector_aligned);

#if DEBUG_ADAPTIVE_SAMPLING
            if (cam.pass_settings.variance_threshold != 0.0f && required_samples_[y * w_ + x] >= region.iteration &&
                (region.iteration % 5) == 0) {
                final_buf_[y * w_ + x].v[0] = 1.0f;
                full_buf_[y * w_ + x].v[0] = 1.0f;
            }
#endif

            if (simd_cast(variance >= variance_threshold).not_all_zeros()) {
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
    p.feature_buf1.resize(rect_ext.w * rect_ext.h);
    p.feature_buf2.resize(rect_ext.w * rect_ext.h);

#define FETCH_FINAL_BUF(_x, _y)                                                                                        \
    Ref::fvec4(full_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,                \
               Ref::vector_aligned)
#define FETCH_VARIANCE(_x, _y)                                                                                         \
    Ref::fvec4(temp_buf_[std::min(std::max(_y, 0), h_ - 1) * w_ + std::min(std::max(_x, 0), w_ - 1)].v,                \
               Ref::vector_aligned)

    static const float GaussWeights[] = {0.2270270270f, 0.1945945946f, 0.1216216216f, 0.0540540541f, 0.0162162162f};

    for (int y = 0; y < rect_ext.h; ++y) {
        const int yy = rect_ext.y + y;
        for (int x = 0; x < rect_ext.w; ++x) {
            const int xx = rect_ext.x + x;
            const Ref::fvec4 center_col = reversible_tonemap(FETCH_FINAL_BUF(xx, yy));
            center_col.store_to(p.temp_final_buf[y * rect_ext.w + x].v, Ref::vector_aligned);

            const Ref::fvec4 center_val = FETCH_VARIANCE(xx, yy);

            Ref::fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res += FETCH_VARIANCE(xx - i + 1, yy) * GaussWeights[i + 1];
                res += FETCH_VARIANCE(xx + i + 1, yy) * GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.variance_buf[y * rect_ext.w + x].v, Ref::vector_aligned);
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
            const Ref::fvec4 center_val = {p.variance_buf[(y + 0) * rect_ext.w + x].v, Ref::vector_aligned};

            Ref::fvec4 res = center_val * GaussWeights[0];
            UNROLLED_FOR(i, 4, {
                res += Ref::fvec4(p.variance_buf[(y - i + 1) * rect_ext.w + x].v, Ref::vector_aligned) *
                       GaussWeights[i + 1];
                res += Ref::fvec4(p.variance_buf[(y + i + 1) * rect_ext.w + x].v, Ref::vector_aligned) *
                       GaussWeights[i + 1];
            })

            res = max(res, center_val);
            res.store_to(p.filtered_variance_buf[y * rect_ext.w + x].v, Ref::vector_aligned);

            p.feature_buf1[y * rect_ext.w + x] = FETCH_BASE_COLOR(rect_ext.x + x, rect_ext.y + y);
            p.feature_buf2[y * rect_ext.w + x] = FETCH_DEPTH_NORMALS(rect_ext.x + x, rect_ext.y + y);
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

            const Ref::fvec4 variance = {p.filtered_variance_buf[(y + EXT_RADIUS) * rect_ext.w + (x + EXT_RADIUS)].v,
                                         Ref::vector_aligned};
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
            auto col = Ref::fvec4(raw_filtered_buf_[y * w_ + x].v, Ref::vector_aligned);
            col = Ref::reversible_tonemap_invert(col);
            col.store_to(raw_filtered_buf_[y * w_ + x].v, Ref::vector_aligned);
            col = Tonemap(tonemap_params, col);
            col.store_to(final_buf_[y * w_ + x].v, Ref::vector_aligned);
        }
    }

    const auto denoise_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_denoise_us += (unsigned long long)duration<double, std::micro>{denoise_end - denoise_start}.count();
    }
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::DenoiseImage(const int pass, const RegionContext &region) {
    using namespace std::chrono;
    const auto denoise_start = high_resolution_clock::now();

    const int w_rounded = 16 * ((w_ + 15) / 16);
    const int h_rounded = 16 * ((h_ + 15) / 16);

    rect_t r = region.rect();
    if (pass < 15) {
        r.w = 16 * ((r.w + 15) / 16);
        r.h = 16 * ((r.h + 15) / 16);
    }

    const float *weights = unet_weights_.data();
    const unet_weight_offsets_t *offsets = &unet_offsets_;

    switch (pass) {
    case 0: {
        SIMDPolicy::template Convolution3x3_GEMM<3, 3, 3, 4, 32, ePreOp::HDRTransfer, ePreOp::None,
                                                 ePreOp::PositiveNormalize>(
            &full_buf_[0].v[0], &base_color_buf_[0].v[0], &depth_normals_buf_[0].v[0], r, w_, h_, w_rounded, h_rounded,
            w_, &weights[offsets->enc_conv0_weight], &weights[offsets->enc_conv0_bias],
            unet_tensors_.encConv0 + (w_rounded + 3) * 32, w_rounded + 2);
        SIMDPolicy::ClearBorders(r, w_rounded, h_rounded, false, 32, unet_tensors_.encConv0);
        break;
    }
    case 1: {
        SIMDPolicy::template Convolution3x3_Direct<32, 32, 32, Ray::ePostOp::Downscale>(
            unet_tensors_.encConv0 + (w_rounded + 3) * 32, r, w_rounded, h_rounded, w_rounded + 2,
            &weights[offsets->enc_conv1_weight], &weights[offsets->enc_conv1_bias],
            unet_tensors_.pool1 + (w_rounded / 2 + 3) * 32, w_rounded / 2 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded, h_rounded, true, 32, unet_tensors_.pool1);
        break;
    }
    case 2: {
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;
        SIMDPolicy::template Convolution3x3_Direct<32, 48, 48, Ray::ePostOp::Downscale>(
            unet_tensors_.pool1 + (w_rounded / 2 + 3) * 32, r, w_rounded / 2, h_rounded / 2, w_rounded / 2 + 2,
            &weights[offsets->enc_conv2_weight], &weights[offsets->enc_conv2_bias],
            unet_tensors_.pool2 + (w_rounded / 4 + 3) * 48, w_rounded / 4 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 2, h_rounded / 2, true, 48, unet_tensors_.pool2);
        break;
    }
    case 3: {
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;
        SIMDPolicy::template Convolution3x3_Direct<48, 64, 64, Ray::ePostOp::Downscale>(
            unet_tensors_.pool2 + (w_rounded / 4 + 3) * 48, r, w_rounded / 4, h_rounded / 4, w_rounded / 4 + 2,
            &weights[offsets->enc_conv3_weight], &weights[offsets->enc_conv3_bias],
            unet_tensors_.pool3 + (w_rounded / 8 + 3) * 64, w_rounded / 8 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 4, h_rounded / 4, true, 64, unet_tensors_.pool3);
        break;
    }
    case 4: {
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;
        SIMDPolicy::template Convolution3x3_Direct<64, 80, 80, Ray::ePostOp::Downscale>(
            unet_tensors_.pool3 + (w_rounded / 8 + 3) * 64, r, w_rounded / 8, h_rounded / 8, w_rounded / 8 + 2,
            &weights[offsets->enc_conv4_weight], &weights[offsets->enc_conv4_bias],
            unet_tensors_.pool4 + (w_rounded / 16 + 3) * 80, w_rounded / 16 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 8, h_rounded / 8, true, 80, unet_tensors_.pool4);
        break;
    }
    case 5: {
        r.x = r.x / 16;
        r.y = r.y / 16;
        r.w = (r.w + 15) / 16;
        r.h = (r.h + 15) / 16;
        SIMDPolicy::template Convolution3x3_Direct<80, 96>(
            unet_tensors_.pool4 + (w_rounded / 16 + 3) * 80, r, w_rounded / 16, h_rounded / 16, w_rounded / 16 + 2,
            &weights[offsets->enc_conv5a_weight], &weights[offsets->enc_conv5a_bias],
            unet_tensors_.enc_conv5a + (w_rounded / 16 + 3) * 96, w_rounded / 16 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 16, h_rounded / 16, false, 96, unet_tensors_.enc_conv5a);
        break;
    }
    case 6: {
        r.x = r.x / 16;
        r.y = r.y / 16;
        r.w = (r.w + 15) / 16;
        r.h = (r.h + 15) / 16;
        SIMDPolicy::template Convolution3x3_Direct<96, 96>(
            unet_tensors_.enc_conv5a + (w_rounded / 16 + 3) * 96, r, w_rounded / 16, h_rounded / 16, w_rounded / 16 + 2,
            &weights[offsets->enc_conv5b_weight], &weights[offsets->enc_conv5b_bias],
            unet_tensors_.upsample4 + (w_rounded / 16 + 3) * 96, w_rounded / 16 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 16, h_rounded / 16, false, 96, unet_tensors_.upsample4);
        break;
    }
    case 7: {
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;
        SIMDPolicy::template ConvolutionConcat3x3_Direct<96, 64, 112, Ray::ePreOp::Upscale>(
            unet_tensors_.upsample4 + (w_rounded / 16 + 3) * 96, unet_tensors_.pool3 + (w_rounded / 8 + 3) * 64, r,
            w_rounded / 8, h_rounded / 8, w_rounded / 16 + 2, w_rounded / 8 + 2, &weights[offsets->dec_conv4a_weight],
            &weights[offsets->dec_conv4a_bias], unet_tensors_.dec_conv4a + (w_rounded / 8 + 3) * 112,
            w_rounded / 8 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 8, h_rounded / 8, false, 112, unet_tensors_.dec_conv4a);
        break;
    }
    case 8: {
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;
        SIMDPolicy::template Convolution3x3_Direct<112, 112>(
            unet_tensors_.dec_conv4a + (w_rounded / 8 + 3) * 112, r, w_rounded / 8, h_rounded / 8, w_rounded / 8 + 2,
            &weights[offsets->dec_conv4b_weight], &weights[offsets->dec_conv4b_bias],
            unet_tensors_.upsample3 + (w_rounded / 8 + 3) * 112, w_rounded / 8 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 8, h_rounded / 8, false, 112, unet_tensors_.upsample3);
        break;
    }
    case 9: {
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;
        SIMDPolicy::template ConvolutionConcat3x3_Direct<112, 48, 96, Ray::ePreOp::Upscale>(
            unet_tensors_.upsample3 + (w_rounded / 8 + 3) * 112, unet_tensors_.pool2 + (w_rounded / 4 + 3) * 48, r,
            w_rounded / 4, h_rounded / 4, w_rounded / 8 + 2, w_rounded / 4 + 2, &weights[offsets->dec_conv3a_weight],
            &weights[offsets->dec_conv3a_bias], unet_tensors_.dec_conv3a + (w_rounded / 4 + 3) * 96, w_rounded / 4 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 4, h_rounded / 4, false, 96, unet_tensors_.dec_conv3a);
        break;
    }
    case 10: {
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;
        SIMDPolicy::template Convolution3x3_Direct<96, 96>(
            unet_tensors_.dec_conv3a + (w_rounded / 4 + 3) * 96, r, w_rounded / 4, h_rounded / 4, w_rounded / 4 + 2,
            &weights[offsets->dec_conv3b_weight], &weights[offsets->dec_conv3b_bias],
            unet_tensors_.upsample2 + (w_rounded / 4 + 3) * 96, w_rounded / 4 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 4, h_rounded / 4, false, 96, unet_tensors_.upsample2);
        break;
    }
    case 11: {
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;
        SIMDPolicy::template ConvolutionConcat3x3_Direct<96, 32, 64, Ray::ePreOp::Upscale>(
            unet_tensors_.upsample2 + (w_rounded / 4 + 3) * 96, unet_tensors_.pool1 + (w_rounded / 2 + 3) * 32, r,
            w_rounded / 2, h_rounded / 2, w_rounded / 4 + 2, w_rounded / 2 + 2, &weights[offsets->dec_conv2a_weight],
            &weights[offsets->dec_conv2a_bias], unet_tensors_.dec_conv2a + (w_rounded / 2 + 3) * 64, w_rounded / 2 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 2, h_rounded / 2, false, 64, unet_tensors_.dec_conv2a);
        break;
    }
    case 12: {
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;
        SIMDPolicy::template Convolution3x3_Direct<64, 64>(
            unet_tensors_.dec_conv2a + (w_rounded / 2 + 3) * 64, r, w_rounded / 2, h_rounded / 2, w_rounded / 2 + 2,
            &weights[offsets->dec_conv2b_weight], &weights[offsets->dec_conv2b_bias],
            unet_tensors_.upsample1 + (w_rounded / 2 + 3) * 64, w_rounded / 2 + 2);
        SIMDPolicy::ClearBorders(r, w_rounded / 2, h_rounded / 2, false, 64, unet_tensors_.upsample1);
        break;
    }
    case 13: {
        SIMDPolicy::template ConvolutionConcat3x3_1Direct_2GEMM<64, 3, 3, 3, 4, 64, Ray::ePreOp::Upscale,
                                                                Ray::ePreOp::HDRTransfer, Ray::ePreOp::None,
                                                                Ray::ePreOp::PositiveNormalize>(
            unet_tensors_.upsample1 + (w_rounded / 2 + 3) * 64, &full_buf_[0].v[0], &base_color_buf_[0].v[0],
            &depth_normals_buf_[0].v[0], r, w_rounded, h_rounded, w_, h_, w_rounded / 2 + 2, w_,
            &weights[offsets->dec_conv1a_weight], &weights[offsets->dec_conv1a_bias],
            unet_tensors_.dec_conv1a + (w_rounded + 3) * 64, w_rounded + 2);
        SIMDPolicy::ClearBorders(r, w_rounded, h_rounded, false, 64, unet_tensors_.dec_conv1a);
        break;
    }
    case 14: {
        SIMDPolicy::template Convolution3x3_Direct<64, 32>(
            unet_tensors_.dec_conv1a + (w_rounded + 3) * 64, r, w_rounded, h_rounded, w_rounded + 2,
            &weights[offsets->dec_conv1b_weight], &weights[offsets->dec_conv1b_bias],
            unet_tensors_.dec_conv1b + (w_rounded + 3) * 32, w_rounded + 2);
        SIMDPolicy::ClearBorders(r, w_rounded, h_rounded, false, 32, unet_tensors_.dec_conv1b);
        break;
    }
    case 15: {
        SIMDPolicy::template Convolution3x3_Direct<32, 3, 4, ePostOp::HDRTransfer>(
            unet_tensors_.dec_conv1b + (w_rounded + 3) * 32, r, w_, h_, w_rounded + 2,
            &weights[offsets->dec_conv0_weight], &weights[offsets->dec_conv0_bias], &raw_filtered_buf_[0].v[0], 0);

        Ref::tonemap_params_t tonemap_params;

        {
            std::lock_guard<std::mutex> _(mtx_);
            tonemap_params = tonemap_params_;
        }

        for (int y = r.y; y < r.y + r.h; ++y) {
            for (int x = r.x; x < r.x + r.w; ++x) {
                auto col = Ref::fvec4(raw_filtered_buf_[y * w_ + x].v, Ref::vector_aligned);
                col = Tonemap(tonemap_params, col);
                col.store_to(final_buf_[y * w_ + x].v, Ref::vector_aligned);
            }
        }

        break;
    }
    }

    const auto denoise_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_denoise_us += (unsigned long long)duration<double, std::micro>{denoise_end - denoise_start}.count();
    }
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::UpdateSpatialCache(const SceneBase &scene, RegionContext &region) {
    using namespace std::chrono;

    if (!use_spatial_cache_) {
        return;
    }

    const auto &s = dynamic_cast<const Cpu::Scene &>(scene);

    std::shared_lock<std::shared_timed_mutex> scene_lock(s.mtx_);

    camera_t cam = s.cams_[s.current_cam()._index];
    cam.fstop = 0.0f;
    cam.filter = ePixelFilter::Box;

    cache_grid_params_t cache_grid_params;
    memcpy(cache_grid_params.cam_pos_curr, cam.origin, 3 * sizeof(float));
    cache_grid_params.exposure = std::pow(2.0f, cam.exposure);

    const scene_data_t sc_data = {s.env_,
                                  s.mesh_instances_.empty() ? nullptr : &s.mesh_instances_[0],
                                  s.mi_indices_.empty() ? nullptr : &s.mi_indices_[0],
                                  s.meshes_.empty() ? nullptr : &s.meshes_[0],
                                  s.vtx_indices_.empty() ? nullptr : &s.vtx_indices_[0],
                                  s.vertices_.empty() ? nullptr : &s.vertices_[0],
                                  s.nodes_.empty() ? nullptr : &s.nodes_[0],
                                  s.wnodes_.empty() ? nullptr : &s.wnodes_[0],
                                  s.tris_.empty() ? nullptr : &s.tris_[0],
                                  s.tri_indices_.empty() ? nullptr : &s.tri_indices_[0],
                                  s.mtris_.data(),
                                  s.tri_materials_.empty() ? nullptr : &s.tri_materials_[0],
                                  s.materials_.empty() ? nullptr : &s.materials_[0],
                                  {s.lights_.data(), s.lights_.capacity()},
                                  {s.li_indices_},
                                  s.visible_lights_count_,
                                  s.blocker_lights_count_,
                                  {s.light_nodes_},
                                  {s.light_wnodes_},
                                  cache_grid_params,
                                  {},
                                  {}};

    const uint32_t tlas_root = s.tlas_root_;

    float root_min[3], root_max[3], cell_size[3];
    s.GetBounds(root_min, root_max);
    UNROLLED_FOR(i, 3, { cell_size[i] = (root_max[i] - root_min[i]) / 255; })

    rect_t rect = region.rect();
    rect.x /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.y /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.w /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.h /= RAD_CACHE_DOWNSAMPLING_FACTOR;

    PassData<SIMDPolicy> &p = get_per_thread_pass_data<SIMDPolicy>();

    // make sure we will not use stale values
    get_per_thread_BCCache<1>().Invalidate();
    get_per_thread_BCCache<2>().Invalidate();
    get_per_thread_BCCache<4>().Invalidate();

    const auto time_start = high_resolution_clock::now();

    const uint32_t *rand_seq = __pmj02_samples;
    const uint32_t rand_seed = Ref::hash(Ref::hash(region.iteration / RAND_SAMPLES_COUNT));

    if (cam.type != eCamType::Geo) {
        SIMDPolicy::GeneratePrimaryRays(cam, rect, (w_ / RAD_CACHE_DOWNSAMPLING_FACTOR),
                                        (h_ / RAD_CACHE_DOWNSAMPLING_FACTOR), rand_seq, rand_seed, nullptr,
                                        region.iteration + 1, nullptr, p.primary_rays, p.intersections);
        if (tlas_root != 0xffffffff) {
            SIMDPolicy::TraceRays(p.primary_rays, cam.pass_settings.min_transp_depth,
                                  cam.pass_settings.max_transp_depth, sc_data, tlas_root, false, s.tex_storages_,
                                  rand_seq, rand_seed, region.iteration + 1, p.intersections);
        }
    } else {
        assert(false && "Unsupported camera type!");
    }

    p.secondary_rays.resize(p.primary_rays.size());
    p.shadow_rays.resize(p.primary_rays.size());

    int secondary_rays_count = 0, shadow_rays_count = 0;

    SIMDPolicy::ShadePrimary(cam.pass_settings, p.intersections, p.primary_rays, rand_seq, rand_seed,
                             region.iteration + 1, eSpatialCacheMode::Update, sc_data, s.tex_storages_,
                             &p.secondary_rays[0], &secondary_rays_count, &p.shadow_rays[0], &shadow_rays_count, w_,
                             1.0f, temp_buf_.data(), nullptr, raw_filtered_buf_.data());

    SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_direct, sc_data, tlas_root,
                                rand_seq, rand_seed, region.iteration + 1, s.tex_storages_, w_, temp_buf_.data());

    rect_fill<cache_data_t>(temp_cache_data_, (w_ / RAD_CACHE_DOWNSAMPLING_FACTOR), rect, cache_data_t{});
    SIMDPolicy::SpatialCacheUpdate(cache_grid_params, p.intersections, p.primary_rays, temp_cache_data_,
                                   temp_buf_.data(), raw_filtered_buf_.data(), w_, s.spatial_cache_entries_,
                                   s.spatial_cache_voxels_curr_);

    p.hash_values.resize(p.primary_rays.size());
    p.scan_values.resize(round_up(rect.w, 4) * round_up(rect.h, 4));
    p.chunks.resize(round_up(rect.w, 4) * round_up(rect.h, 4));
    p.chunks_temp.resize(round_up(rect.w, 4) * round_up(rect.h, 4));

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth && secondary_rays_count; ++bounce) {
        secondary_rays_count = SIMDPolicy::SortRays_CPU(
            Span<typename SIMDPolicy::RayDataType>{&p.secondary_rays[0], secondary_rays_count}, root_min, cell_size,
            &p.hash_values[0], &p.scan_values[0], &p.chunks[0], &p.chunks_temp[0]);

        for (int i = 0; i < secondary_rays_count; i++) {
            p.intersections[i] = {};
        }

        auto rays = Span<typename SIMDPolicy::RayDataType>{p.secondary_rays.data(), secondary_rays_count};
        auto intersections = Span<typename SIMDPolicy::HitDataType>{p.intersections.data(), secondary_rays_count};

        SIMDPolicy::TraceRays(rays, cam.pass_settings.min_transp_depth, cam.pass_settings.max_transp_depth, sc_data,
                              tlas_root, true, s.tex_storages_, rand_seq, rand_seed, region.iteration + 1,
                              p.intersections);

        secondary_rays_count = 0;
        shadow_rays_count = 0;
        std::swap(p.primary_rays, p.secondary_rays);

        // Use direct clamping value only for the first intersection with lightsource
        const float clamp_direct = (bounce == 1) ? cam.pass_settings.clamp_direct : cam.pass_settings.clamp_indirect;
        SIMDPolicy::ShadeSecondary(cam.pass_settings, clamp_direct, intersections, rays, rand_seq, rand_seed,
                                   region.iteration + 1, eSpatialCacheMode::Update, sc_data, s.tex_storages_,
                                   &p.secondary_rays[0], &secondary_rays_count, &p.shadow_rays[0], &shadow_rays_count,
                                   w_, temp_buf_.data(), nullptr, raw_filtered_buf_.data());

        SIMDPolicy::TraceShadowRays(Span<typename SIMDPolicy::ShadowRayType>{p.shadow_rays.data(), shadow_rays_count},
                                    cam.pass_settings.max_transp_depth, cam.pass_settings.clamp_indirect, sc_data,
                                    tlas_root, rand_seq, rand_seed, region.iteration + 1, s.tex_storages_, w_,
                                    temp_buf_.data());

        SIMDPolicy::SpatialCacheUpdate(cache_grid_params, intersections, rays, temp_cache_data_, temp_buf_.data(),
                                       raw_filtered_buf_.data(), w_, s.spatial_cache_entries_,
                                       s.spatial_cache_voxels_curr_);
    }

    scene_lock.unlock();

    const auto time_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_cache_update_us += (unsigned long long)duration<double, std::micro>{time_end - time_start}.count();
    }
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::ResolveSpatialCache(
    const SceneBase &scene, const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    using namespace std::chrono;

    if (!use_spatial_cache_) {
        return;
    }

    const auto &s = dynamic_cast<const Cpu::Scene &>(scene);

    const auto time_start = high_resolution_clock::now();

    std::shared_lock<std::shared_timed_mutex> scene_lock(s.mtx_);

    const camera_t &cam = s.cams_[s.current_cam()._index];

    cache_grid_params_t params;
    memcpy(params.cam_pos_curr, cam.origin, 3 * sizeof(float));
    memcpy(params.cam_pos_prev, s.spatial_cache_cam_pos_prev_, 3 * sizeof(float));

    static const int ResolvePortion = 32768;
    assert((s.spatial_cache_entries_.size() % ResolvePortion) == 0);
    const int JobsCount = int(s.spatial_cache_entries_.size() / ResolvePortion);

    parallel_for(0, JobsCount, [&](const int i) {
        Ref::SpatialCacheResolve(params, s.spatial_cache_entries_, s.spatial_cache_voxels_curr_,
                                 s.spatial_cache_voxels_prev_, i * ResolvePortion, ResolvePortion);
    });

    std::swap(s.spatial_cache_voxels_prev_, s.spatial_cache_voxels_curr_);
    parallel_for(0, JobsCount, [&](const int i) {
        auto it = begin(s.spatial_cache_voxels_curr_) + i * ResolvePortion;
        std::fill(it, it + ResolvePortion, packed_cache_voxel_t{});
    });

    // Store previous camera position
    memcpy(s.spatial_cache_cam_pos_prev_, cam.origin, 3 * sizeof(float));

    scene_lock.unlock();

    const auto time_end = high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> _(mtx_);
        stats_.time_cache_resolve_us += (unsigned long long)duration<double, std::micro>{time_end - time_start}.count();
    }
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::UpdateFilterTable(ePixelFilter filter, float filter_width) {
    float (*filter_func)(float v, float width);

    switch (filter) {
    case ePixelFilter::Box:
        filter_func = filter_box;
        filter_width = 1.0f;
        break;
    case ePixelFilter::Gaussian:
        filter_func = filter_gaussian;
        filter_width *= 3.0f;
        break;
    case ePixelFilter::BlackmanHarris:
        filter_func = filter_blackman_harris;
        filter_width *= 2.0f;
        break;
    default:
        assert(false && "Unknown filter!");
    }

    filter_table_ =
        Ray::CDFInverted(FILTER_TABLE_SIZE, 0.0f, filter_width * 0.5f,
                         std::bind(filter_func, std::placeholders::_1, filter_width), true /* make_symmetric */);
}

template <typename SIMDPolicy>
void Ray::Cpu::Renderer<SIMDPolicy>::InitUNetFilter(const bool alias_memory, unet_filter_properties_t &out_props) {
    const int total_count = SetupUNetWeights<float>(true, 1, nullptr, nullptr);
    unet_weights_.resize(total_count);
    SetupUNetWeights(true, 1, &unet_offsets_, unet_weights_.data());

    unet_alias_memory_ = alias_memory;
    UpdateUNetFilterMemory();

    out_props.pass_count = UNetFilterPasses;
    for (int i = 0; i < UNetFilterPasses; ++i) {
        std::fill(&out_props.alias_dependencies[i][0], &out_props.alias_dependencies[i][0] + 4, -1);
        for (int j = 0; j < int(unet_alias_dependencies_[i].size()); ++j) {
            out_props.alias_dependencies[i][j] = unet_alias_dependencies_[i][j];
        }
    }
}

template <typename SIMDPolicy> void Ray::Cpu::Renderer<SIMDPolicy>::UpdateUNetFilterMemory() {
    unet_tensors_heap_ = {};
    if (unet_weights_.empty()) {
        return;
    }

    unet_filter_tensors_t tensors;
    const int required_memory = SetupUNetFilter(w_, h_, unet_alias_memory_, false, tensors, unet_alias_dependencies_);

#ifndef NDEBUG
    unet_tensors_heap_.resize(required_memory, NAN);
#else
    unet_tensors_heap_.resize(required_memory, 0.0f);
#endif

    unet_tensors_.encConv0 = unet_tensors_heap_.data() + tensors.enc_conv0_offset;
    unet_tensors_.pool1 = unet_tensors_heap_.data() + tensors.pool1_offset;
    unet_tensors_.pool2 = unet_tensors_heap_.data() + tensors.pool2_offset;
    unet_tensors_.pool3 = unet_tensors_heap_.data() + tensors.pool3_offset;
    unet_tensors_.pool4 = unet_tensors_heap_.data() + tensors.pool4_offset;
    unet_tensors_.enc_conv5a = unet_tensors_heap_.data() + tensors.enc_conv5a_offset;
    unet_tensors_.upsample4 = unet_tensors_heap_.data() + tensors.upsample4_offset;
    unet_tensors_.dec_conv4a = unet_tensors_heap_.data() + tensors.dec_conv4a_offset;
    unet_tensors_.upsample3 = unet_tensors_heap_.data() + tensors.upsample3_offset;
    unet_tensors_.dec_conv3a = unet_tensors_heap_.data() + tensors.dec_conv3a_offset;
    unet_tensors_.upsample2 = unet_tensors_heap_.data() + tensors.upsample2_offset;
    unet_tensors_.dec_conv2a = unet_tensors_heap_.data() + tensors.dec_conv2a_offset;
    unet_tensors_.upsample1 = unet_tensors_heap_.data() + tensors.upsample1_offset;
    unet_tensors_.dec_conv1a = unet_tensors_heap_.data() + tensors.dec_conv1a_offset;
    unet_tensors_.dec_conv1b = unet_tensors_heap_.data() + tensors.dec_conv1b_offset;
}
