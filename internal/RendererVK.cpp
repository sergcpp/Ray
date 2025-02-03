#include "RendererVK.h"

#include <functional>
#include <utility>

#include "CDFUtils.h"
#include "Core.h"
#include "SceneVK.h"
#include "UNetFilter.h"
#include "inflate/Inflate.h"

#include "Vk/BufferVK.h"
#include "Vk/DebugMarkerVK.h"
#include "Vk/DrawCallVK.h"
#include "Vk/PipelineVK.h"
#include "Vk/ProgramVK.h"
#include "Vk/SamplerVK.h"
#include "Vk/ShaderVK.h"
#include "Vk/TextureVK.h"

#include "../Log.h"

#include "shaders/sort_common.h"
#include "shaders/types.h"

#define DEBUG_HWRT 0
#define RUN_IN_LOCKSTEP 0
#define DISABLE_SORTING 0
#define ENABLE_RT_PIPELINE 0

static_assert(sizeof(Types::tri_accel_t) == sizeof(Ray::tri_accel_t), "!");
static_assert(sizeof(Types::bvh_node_t) == sizeof(Ray::bvh_node_t), "!");
static_assert(sizeof(Types::bvh2_node_t) == sizeof(Ray::bvh2_node_t), "!");
static_assert(sizeof(Types::light_bvh_node_t) == sizeof(Ray::light_bvh_node_t), "!");
static_assert(sizeof(Types::light_wbvh_node_t) == sizeof(Ray::light_wbvh_node_t), "!");
static_assert(sizeof(Types::light_cwbvh_node_t) == sizeof(Ray::light_cwbvh_node_t), "!");
static_assert(sizeof(Types::vertex_t) == sizeof(Ray::vertex_t), "!");
static_assert(sizeof(Types::mesh_t) == sizeof(Ray::mesh_t), "!");
static_assert(sizeof(Types::mesh_instance_t) == sizeof(Ray::mesh_instance_t), "!");
static_assert(sizeof(Types::light_t) == sizeof(Ray::light_t), "!");
static_assert(sizeof(Types::material_t) == sizeof(Ray::material_t), "!");
static_assert(sizeof(Types::atlas_texture_t) == sizeof(Ray::atlas_texture_t), "!");
static_assert(sizeof(Types::ray_chunk_t) == sizeof(Ray::ray_chunk_t), "!");
static_assert(sizeof(Types::ray_hash_t) == sizeof(Ray::ray_hash_t), "!");
static_assert(sizeof(Types::atmosphere_params_t) == sizeof(Ray::atmosphere_params_t), "!");

static_assert(Types::LIGHT_TYPE_SPHERE == Ray::LIGHT_TYPE_SPHERE, "!");
static_assert(Types::LIGHT_TYPE_DIR == Ray::LIGHT_TYPE_DIR, "!");
static_assert(Types::LIGHT_TYPE_LINE == Ray::LIGHT_TYPE_LINE, "!");
static_assert(Types::LIGHT_TYPE_RECT == Ray::LIGHT_TYPE_RECT, "!");
static_assert(Types::LIGHT_TYPE_DISK == Ray::LIGHT_TYPE_DISK, "!");
static_assert(Types::LIGHT_TYPE_TRI == Ray::LIGHT_TYPE_TRI, "!");
static_assert(Types::FILTER_BOX == int(Ray::ePixelFilter::Box), "!");
static_assert(Types::FILTER_GAUSSIAN == int(Ray::ePixelFilter::Gaussian), "!");
static_assert(Types::FILTER_BLACKMAN_HARRIS == int(Ray::ePixelFilter::BlackmanHarris), "!");
static_assert(Types::FILTER_TABLE_SIZE == Ray::FILTER_TABLE_SIZE, "!");

static_assert(int(Ray::eGPUResState::RenderTarget) == int(Ray::Vk::eResState::RenderTarget), "!");
static_assert(int(Ray::eGPUResState::UnorderedAccess) == int(Ray::Vk::eResState::UnorderedAccess), "!");
static_assert(int(Ray::eGPUResState::DepthRead) == int(Ray::Vk::eResState::DepthRead), "!");
static_assert(int(Ray::eGPUResState::DepthWrite) == int(Ray::Vk::eResState::DepthWrite), "!");
static_assert(int(Ray::eGPUResState::ShaderResource) == int(Ray::Vk::eResState::ShaderResource), "!");
static_assert(int(Ray::eGPUResState::CopyDst) == int(Ray::Vk::eResState::CopyDst), "!");
static_assert(int(Ray::eGPUResState::CopySrc) == int(Ray::Vk::eResState::CopySrc), "!");

namespace Ray {
extern const int LUT_DIMS;
extern const uint32_t *transform_luts[];
namespace Vk {
#include "shaders/output/convolution_112_112_coop.comp.spv.inl"
#include "shaders/output/convolution_112_112_fp16.comp.spv.inl"
#include "shaders/output/convolution_112_112_fp32.comp.spv.inl"
#include "shaders/output/convolution_32_32_Downsample_coop.comp.spv.inl"
#include "shaders/output/convolution_32_32_Downsample_fp16.comp.spv.inl"
#include "shaders/output/convolution_32_32_Downsample_fp32.comp.spv.inl"
#include "shaders/output/convolution_32_3_img_coop.comp.spv.inl"
#include "shaders/output/convolution_32_3_img_fp16.comp.spv.inl"
#include "shaders/output/convolution_32_3_img_fp32.comp.spv.inl"
#include "shaders/output/convolution_32_48_Downsample_coop.comp.spv.inl"
#include "shaders/output/convolution_32_48_Downsample_fp16.comp.spv.inl"
#include "shaders/output/convolution_32_48_Downsample_fp32.comp.spv.inl"
#include "shaders/output/convolution_48_64_Downsample_coop.comp.spv.inl"
#include "shaders/output/convolution_48_64_Downsample_fp16.comp.spv.inl"
#include "shaders/output/convolution_48_64_Downsample_fp32.comp.spv.inl"
#include "shaders/output/convolution_64_32_coop.comp.spv.inl"
#include "shaders/output/convolution_64_32_fp16.comp.spv.inl"
#include "shaders/output/convolution_64_32_fp32.comp.spv.inl"
#include "shaders/output/convolution_64_64_coop.comp.spv.inl"
#include "shaders/output/convolution_64_64_fp16.comp.spv.inl"
#include "shaders/output/convolution_64_64_fp32.comp.spv.inl"
#include "shaders/output/convolution_64_80_Downsample_coop.comp.spv.inl"
#include "shaders/output/convolution_64_80_Downsample_fp16.comp.spv.inl"
#include "shaders/output/convolution_64_80_Downsample_fp32.comp.spv.inl"
#include "shaders/output/convolution_80_96_coop.comp.spv.inl"
#include "shaders/output/convolution_80_96_fp16.comp.spv.inl"
#include "shaders/output/convolution_80_96_fp32.comp.spv.inl"
#include "shaders/output/convolution_96_96_coop.comp.spv.inl"
#include "shaders/output/convolution_96_96_fp16.comp.spv.inl"
#include "shaders/output/convolution_96_96_fp32.comp.spv.inl"
#include "shaders/output/convolution_Img_9_32_coop.comp.spv.inl"
#include "shaders/output/convolution_Img_9_32_fp16.comp.spv.inl"
#include "shaders/output/convolution_Img_9_32_fp32.comp.spv.inl"
#include "shaders/output/convolution_concat_112_48_96_coop.comp.spv.inl"
#include "shaders/output/convolution_concat_112_48_96_fp16.comp.spv.inl"
#include "shaders/output/convolution_concat_112_48_96_fp32.comp.spv.inl"
#include "shaders/output/convolution_concat_64_9_64_coop.comp.spv.inl"
#include "shaders/output/convolution_concat_64_9_64_fp16.comp.spv.inl"
#include "shaders/output/convolution_concat_64_9_64_fp32.comp.spv.inl"
#include "shaders/output/convolution_concat_96_32_64_coop.comp.spv.inl"
#include "shaders/output/convolution_concat_96_32_64_fp16.comp.spv.inl"
#include "shaders/output/convolution_concat_96_32_64_fp32.comp.spv.inl"
#include "shaders/output/convolution_concat_96_64_112_coop.comp.spv.inl"
#include "shaders/output/convolution_concat_96_64_112_fp16.comp.spv.inl"
#include "shaders/output/convolution_concat_96_64_112_fp32.comp.spv.inl"
#include "shaders/output/debug_rt.comp.spv.inl"
#include "shaders/output/filter_variance.comp.spv.inl"
#include "shaders/output/intersect_area_lights.comp.spv.inl"
#include "shaders/output/intersect_scene.rchit.spv.inl"
#include "shaders/output/intersect_scene.rgen.spv.inl"
#include "shaders/output/intersect_scene.rmiss.spv.inl"
#include "shaders/output/intersect_scene_hwrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_hwrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect.rgen.spv.inl"
#include "shaders/output/intersect_scene_indirect_hwrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect_hwrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect_swrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect_swrt_atlas_subgroup.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect_swrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_indirect_swrt_bindless_subgroup.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_swrt_atlas_subgroup.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_shadow_swrt_bindless_subgroup.comp.spv.inl"
#include "shaders/output/intersect_scene_swrt_atlas.comp.spv.inl"
#include "shaders/output/intersect_scene_swrt_atlas_subgroup.comp.spv.inl"
#include "shaders/output/intersect_scene_swrt_bindless.comp.spv.inl"
#include "shaders/output/intersect_scene_swrt_bindless_subgroup.comp.spv.inl"
#include "shaders/output/mix_incremental.comp.spv.inl"
#include "shaders/output/nlm_filter.comp.spv.inl"
#include "shaders/output/postprocess.comp.spv.inl"
#include "shaders/output/prepare_indir_args.comp.spv.inl"
#include "shaders/output/primary_ray_gen_adaptive.comp.spv.inl"
#include "shaders/output/primary_ray_gen_simple.comp.spv.inl"
#include "shaders/output/shade_primary_atlas.comp.spv.inl"
#include "shaders/output/shade_primary_atlas_cache_query.comp.spv.inl"
#include "shaders/output/shade_primary_atlas_cache_query_sky.comp.spv.inl"
#include "shaders/output/shade_primary_atlas_cache_update.comp.spv.inl"
#include "shaders/output/shade_primary_atlas_sky.comp.spv.inl"
#include "shaders/output/shade_primary_bindless.comp.spv.inl"
#include "shaders/output/shade_primary_bindless_cache_query.comp.spv.inl"
#include "shaders/output/shade_primary_bindless_cache_query_sky.comp.spv.inl"
#include "shaders/output/shade_primary_bindless_cache_update.comp.spv.inl"
#include "shaders/output/shade_primary_bindless_sky.comp.spv.inl"
#include "shaders/output/shade_secondary_atlas.comp.spv.inl"
#include "shaders/output/shade_secondary_atlas_cache_query.comp.spv.inl"
#include "shaders/output/shade_secondary_atlas_cache_query_sky.comp.spv.inl"
#include "shaders/output/shade_secondary_atlas_cache_update.comp.spv.inl"
#include "shaders/output/shade_secondary_atlas_sky.comp.spv.inl"
#include "shaders/output/shade_secondary_bindless.comp.spv.inl"
#include "shaders/output/shade_secondary_bindless_cache_query.comp.spv.inl"
#include "shaders/output/shade_secondary_bindless_cache_query_sky.comp.spv.inl"
#include "shaders/output/shade_secondary_bindless_cache_update.comp.spv.inl"
#include "shaders/output/shade_secondary_bindless_sky.comp.spv.inl"
#include "shaders/output/shade_sky.comp.spv.inl"
#include "shaders/output/sort_hash_rays.comp.spv.inl"
#include "shaders/output/sort_init_count_table.comp.spv.inl"
#include "shaders/output/sort_reduce.comp.spv.inl"
#include "shaders/output/sort_reorder_rays.comp.spv.inl"
#include "shaders/output/sort_scan.comp.spv.inl"
#include "shaders/output/sort_scan_add.comp.spv.inl"
#include "shaders/output/sort_scatter.comp.spv.inl"
#include "shaders/output/spatial_cache_resolve.comp.spv.inl"
#include "shaders/output/spatial_cache_update.comp.spv.inl"
#include "shaders/output/spatial_cache_update_compat.comp.spv.inl"
} // namespace Vk
void WritePFM(const char *base_name, const float values[], int w, int h, int channels);
} // namespace Ray

#if 0
#undef FAR
#include <zlib.h>

#pragma comment(lib, "zlibstatic.lib")

namespace Ray {
std::vector<uint8_t> deflate_data(const uint8_t *data, const int len) {
    z_stream stream = {};

    int ret = deflateInit(&stream, Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) {
        return {};
    }

    stream.next_in = (Bytef *)data;
    stream.avail_in = len;

    std::vector<uint8_t> out_data;
    uint8_t temp_buf[4096];

    do {
        stream.next_out = temp_buf;
        stream.avail_out = sizeof(temp_buf);

        ret = deflate(&stream, Z_FINISH);
        if (ret == Z_STREAM_ERROR) {
            return {};
        }

        const int count = sizeof(temp_buf) - stream.avail_out;
        out_data.insert(end(out_data), temp_buf, temp_buf + count);
    } while (stream.avail_out == 0);

    return out_data;
}
}
#endif

#define NS Vk
#include "RendererGPU.h"
#include "RendererGPU_kernels.h"
#undef NS

Ray::Vk::Renderer::Renderer(const settings_t &s, ILog *log,
                            const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    ctx_ = std::make_unique<Context>();
    const bool res = ctx_->Init(log, s.vk_device, s.vk_functions, s.preferred_device, s.validation_level);
    if (!res) {
        throw std::runtime_error("Error initializing vulkan context!");
    }

    assert(Types::RAND_SAMPLES_COUNT == Ray::RAND_SAMPLES_COUNT);
    assert(Types::RAND_DIMS_COUNT == Ray::RAND_DIMS_COUNT);

    use_hwrt_ = (s.use_hwrt && ctx_->ray_query_supported());
    use_bindless_ = s.use_bindless && ctx_->max_sampled_images() >= 16384u;
    use_tex_compression_ = s.use_tex_compression;
    use_fp16_ = ctx_->fp16_supported();
    use_subgroup_ = ctx_->subgroup_supported();
    use_coop_matrix_ = ctx_->coop_matrix_supported();
    use_spatial_cache_ = s.use_spatial_cache && ctx_->int64_supported();
    log->Info("HWRT         is %s", use_hwrt_ ? "enabled" : "disabled");
    log->Info("Bindless     is %s", use_bindless_ ? "enabled" : "disabled");
    log->Info("Compression  is %s", use_tex_compression_ ? "enabled" : "disabled");
    log->Info("Float16      is %s", use_fp16_ ? "enabled" : "disabled");
    log->Info("Subgroup     is %s", use_subgroup_ ? "enabled" : "disabled");
    log->Info("CoopMatrix   is %s", use_coop_matrix_ ? "enabled" : "disabled");
    log->Info("SpatialCache is %s", use_spatial_cache_ ? "enabled" : "disabled");
    log->Info("============================================================================");

    if (!InitPipelines(log, parallel_for)) {
        throw std::runtime_error("Error initializing vulkan pipelines!");
    }

    random_seq_buf_ = Buffer{"Random Seq", ctx_.get(), eBufType::Storage,
                             uint32_t(RAND_DIMS_COUNT * 2 * RAND_SAMPLES_COUNT * sizeof(uint32_t))};
    counters_buf_ = Buffer{"Counters", ctx_.get(), eBufType::Storage, sizeof(uint32_t) * 32};
    indir_args_buf_[0] =
        Buffer{"Indir Args (1/2)", ctx_.get(), eBufType::Indirect, 32 * sizeof(DispatchIndirectCommand)};
    indir_args_buf_[1] =
        Buffer{"Indir Args (2/2)", ctx_.get(), eBufType::Indirect, 32 * sizeof(DispatchIndirectCommand)};

    { // zero out counters, upload random sequence
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        const uint32_t zeros[32] = {};
        counters_buf_.UpdateImmediate(0, 32 * sizeof(uint32_t), zeros, cmd_buf);

        Buffer temp_upload_buf{"Temp upload buf", ctx_.get(), eBufType::Upload, random_seq_buf_.size()};
        { // update stage buffer
            uint8_t *mapped_ptr = temp_upload_buf.Map();
            memcpy(mapped_ptr, __pmj02_samples, RAND_DIMS_COUNT * 2 * RAND_SAMPLES_COUNT * sizeof(uint32_t));
            temp_upload_buf.Unmap();
        }

        CopyBufferToBuffer(temp_upload_buf, 0, random_seq_buf_, 0,
                           RAND_DIMS_COUNT * 2 * RAND_SAMPLES_COUNT * sizeof(uint32_t), cmd_buf);

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        temp_upload_buf.FreeImmediate();
    }

    { // create tonemap LUT texture
        Tex3DParams params = {};
        params.w = params.h = params.d = LUT_DIMS;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.format = eTexFormat::RGB10_A2;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::ClampToEdge;

        tonemap_lut_ = Texture3D{"Tonemap LUT", ctx_.get(), params, ctx_->default_mem_allocs(), ctx_->log()};
    }

    Renderer::Resize(s.w, s.h);
}

Ray::eRendererType Ray::Vk::Renderer::type() const { return eRendererType::Vulkan; }

const char *Ray::Vk::Renderer::device_name() const { return ctx_->device_properties().deviceName; }

void Ray::Vk::Renderer::Clear(const color_rgba_t &c) {
    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    } else {
        cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
    }

    const TransitionInfo img_transitions[] = {{&full_buf_, ResStateForClear},
                                              {&half_buf_, ResStateForClear},
                                              {&final_buf_, ResStateForClear},
                                              {&raw_filtered_buf_, ResStateForClear},
                                              {&base_color_buf_, ResStateForClear},
                                              {&depth_normals_buf_, ResStateForClear},
                                              {&required_samples_buf_, ResStateForClear}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, img_transitions);

    ClearColorImage(full_buf_, c.v, cmd_buf);
    ClearColorImage(half_buf_, c.v, cmd_buf);
    ClearColorImage(final_buf_, c.v, cmd_buf);
    ClearColorImage(raw_filtered_buf_, c.v, cmd_buf);

    static const float rgba_zero[] = {0.0f, 0.0f, 0.0f, 0.0f};
    ClearColorImage(base_color_buf_, rgba_zero, cmd_buf);
    ClearColorImage(depth_normals_buf_, rgba_zero, cmd_buf);

    { // Clear integer texture
        static const uint32_t rgba[4] = {0xffff, 0xffff, 0xffff, 0xffff};
        ClearColorImage(required_samples_buf_, rgba, cmd_buf);
    }

    if (!external_cmd_buf_.vk_cmd_buf) {
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }
}

void Ray::Vk::Renderer::RenderScene(const SceneBase &scene, RegionContext &region) {
    const auto &s = dynamic_cast<const Vk::Scene &>(scene);

    float root_min[3], cell_size[3];
    if (s.tlas_root_ != 0xffffffff) {
        const bvh_node_t &root_node = s.tlas_root_node_;
        for (int i = 0; i < 3; ++i) {
            root_min[i] = root_node.bbox_min[i];
            cell_size[i] = (root_node.bbox_max[i] - root_node.bbox_min[i]) / 255;
        }
    }

    ++region.iteration;

    const Ray::camera_t &cam = s.cams_[s.current_cam()._index];

    // TODO: Use common command buffer for all uploads
    if (cam.filter != filter_table_filter_ || cam.filter_width != filter_table_width_) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        UpdateFilterTable(cmd_buf, cam.filter, cam.filter_width);
        filter_table_filter_ = cam.filter;
        filter_table_width_ = cam.filter_width;

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    if (loaded_view_transform_ != cam.view_transform) {
        if (cam.view_transform != eViewTransform::Standard) {
            CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

            const uint32_t data_len = LUT_DIMS * LUT_DIMS * LUT_DIMS * sizeof(uint32_t);
            Buffer temp_upload_buf{"Temp tonemap LUT upload", ctx_.get(), eBufType::Upload, data_len};
            { // update stage buffer
                uint32_t *mapped_ptr = reinterpret_cast<uint32_t *>(temp_upload_buf.Map());
                const uint32_t *lut = transform_luts[int(cam.view_transform)];

                memcpy(mapped_ptr, lut, data_len);

                temp_upload_buf.Unmap();
            }

            const TransitionInfo res_transitions[] = {{&temp_upload_buf, eResState::CopySrc},
                                                      {&tonemap_lut_, eResState::CopyDst}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

            tonemap_lut_.SetSubImage(0, 0, 0, LUT_DIMS, LUT_DIMS, LUT_DIMS, eTexFormat::RGB10_A2, temp_upload_buf,
                                     cmd_buf, 0, data_len);

            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());

            temp_upload_buf.FreeImmediate();
        }
        loaded_view_transform_ = cam.view_transform;
    }

    cache_grid_params_t cache_grid_params;
    memcpy(cache_grid_params.cam_pos_curr, cam.origin, 3 * sizeof(float));
    cache_grid_params.exposure = std::pow(2.0f, cam.exposure);

    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    }

#if !RUN_IN_LOCKSTEP
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE,
                                    UINT64_MAX);
        ctx_->api().vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
    }
#endif

    const bool reset_queries = !ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP;
    if (!ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP) {
        ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
        ctx_->DestroyDeferredResources(ctx_->backend_frame);
        ctx_->default_descr_alloc()->Reset();
        ctx_->frame_cpu_synced[ctx_->backend_frame] = true;
    }

    stats_.time_primary_ray_gen_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_ray_gen[0], timestamps_[ctx_->backend_frame].primary_ray_gen[1]);
    stats_.time_primary_trace_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_trace[0], timestamps_[ctx_->backend_frame].primary_trace[1]);
    stats_.time_primary_shade_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_shade[0], timestamps_[ctx_->backend_frame].primary_shade[1]);
    stats_.time_primary_shadow_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_shadow[0], timestamps_[ctx_->backend_frame].primary_shadow[1]);

    stats_.time_secondary_sort_us = 0;
    for (int i = 0; i < int(timestamps_[ctx_->backend_frame].secondary_sort.size()); i += 2) {
        stats_.time_secondary_sort_us +=
            ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].secondary_sort[i + 0],
                                                 timestamps_[ctx_->backend_frame].secondary_sort[i + 1]);
    }

    stats_.time_secondary_trace_us = 0;
    for (int i = 0; i < int(timestamps_[ctx_->backend_frame].secondary_trace.size()); i += 2) {
        stats_.time_secondary_trace_us +=
            ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].secondary_trace[i + 0],
                                                 timestamps_[ctx_->backend_frame].secondary_trace[i + 1]);
    }

    stats_.time_secondary_shade_us = 0;
    for (int i = 0; i < int(timestamps_[ctx_->backend_frame].secondary_shade.size()); i += 2) {
        stats_.time_secondary_shade_us +=
            ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].secondary_shade[i + 0],
                                                 timestamps_[ctx_->backend_frame].secondary_shade[i + 1]);
    }

    stats_.time_secondary_shadow_us = 0;
    for (int i = 0; i < int(timestamps_[ctx_->backend_frame].secondary_shadow.size()); i += 2) {
        stats_.time_secondary_shadow_us +=
            ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].secondary_shadow[i + 0],
                                                 timestamps_[ctx_->backend_frame].secondary_shadow[i + 1]);
    }

#if RUN_IN_LOCKSTEP
    cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
    if (!cmd_buf) {
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
        ctx_->api().vkBeginCommandBuffer(cmd_buf, &begin_info);
    }
#endif

    DebugMarker render_scene_marker(ctx_.get(), cmd_buf, "Ray::RenderScene");

    if (reset_queries) {
        ctx_->api().vkCmdResetQueryPool(cmd_buf, ctx_->query_pool(ctx_->backend_frame), 0, MaxTimestampQueries);
    }

    //////////////////////////////////////////////////////////////////////////////////

    const scene_data_t sc_data = {s.env_,
                                  s.mesh_instances_.gpu_buf(),
                                  {s.meshes_.data(), s.meshes_.capacity()},
                                  s.vtx_indices_.gpu_buf(),
                                  s.vertices_.gpu_buf(),
                                  s.nodes_.gpu_buf(),
                                  s.tris_.gpu_buf(),
                                  s.tri_indices_.gpu_buf(),
                                  s.tri_materials_.gpu_buf(),
                                  s.materials_.gpu_buf(),
                                  s.tex_atlases_,
                                  s.atlas_textures_.gpu_buf(),
                                  s.lights_,
                                  s.li_indices_.buf(),
                                  int(s.li_indices_.size()),
                                  s.dir_lights_,
                                  s.visible_lights_count_,
                                  s.blocker_lights_count_,
                                  s.light_cwnodes_.buf(),
                                  s.rt_tlas_,
                                  s.env_map_qtree_.tex,
                                  int(s.env_map_qtree_.mips.size()),
                                  cache_grid_params,
                                  s.spatial_cache_entries_.buf(),
                                  s.spatial_cache_voxels_prev_.buf(),
                                  s.atmosphere_params_buf_,
                                  s.sky_transmittance_lut_tex_,
                                  s.sky_multiscatter_lut_tex_,
                                  s.sky_moon_tex_,
                                  s.sky_weather_tex_,
                                  s.sky_cirrus_tex_,
                                  s.sky_curl_tex_,
                                  s.sky_noise3d_tex_};
    TransitionSceneResources(cmd_buf, sc_data);

    VkMemoryBarrier mem_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    ctx_->api().vkCmdPipelineBarrier(
        cmd_buf,
        (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) &
            ctx_->supported_stages_mask(),
        (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) &
            ctx_->supported_stages_mask(),
        0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

    const rect_t rect = region.rect();
    const uint32_t rand_seed = Ref::hash((region.iteration - 1) / RAND_SAMPLES_COUNT);

    { // generate primary rays
        DebugMarker _(ctx_.get(), cmd_buf, "GeneratePrimaryRays");
        timestamps_[ctx_->backend_frame].primary_ray_gen[0] = ctx_->WriteTimestamp(cmd_buf, true);
        kernel_GeneratePrimaryRays(cmd_buf, cam, rand_seed, rect, w_, h_, random_seq_buf_, filter_table_,
                                   region.iteration, true, required_samples_buf_, counters_buf_, prim_rays_buf_);
        timestamps_[ctx_->backend_frame].primary_ray_gen[1] = ctx_->WriteTimestamp(cmd_buf, false);
    }

    const bool use_rt_pipeline = (use_hwrt_ && ENABLE_RT_PIPELINE);

    { // prepare indirect args
        DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[0]);
    }

#if DEBUG_HWRT
    { // debug
        DebugMarker _(cmd_buf, "Ray::Debug HWRT");
        kernel_DebugRT(cmd_buf, sc_data, macro_tree_root, prim_rays_buf_, temp_buf_);
    }
#else
    { // trace primary rays
        DebugMarker _(ctx_.get(), cmd_buf, "IntersectScenePrimary");
        timestamps_[ctx_->backend_frame].primary_trace[0] = ctx_->WriteTimestamp(cmd_buf, true);
        if (use_rt_pipeline) {
            kernel_IntersectScene_RTPipe(cmd_buf, indir_args_buf_[0], 1, cam.pass_settings, sc_data, random_seq_buf_,
                                         rand_seed, region.iteration, s.tlas_root_, cam.fwd,
                                         cam.clip_end - cam.clip_start, s.tex_atlases_, s.bindless_tex_data_,
                                         prim_rays_buf_, prim_hits_buf_);
        } else {
            kernel_IntersectScene(cmd_buf, indir_args_buf_[0], 0, counters_buf_, cam.pass_settings, sc_data,
                                  random_seq_buf_, rand_seed, region.iteration, s.tlas_root_, cam.fwd,
                                  cam.clip_end - cam.clip_start, s.tex_atlases_, s.bindless_tex_data_, prim_rays_buf_,
                                  prim_hits_buf_);
        }
        timestamps_[ctx_->backend_frame].primary_trace[1] = ctx_->WriteTimestamp(cmd_buf, false);
    }

    const eSpatialCacheMode cache_mode = use_spatial_cache_ ? eSpatialCacheMode::Query : eSpatialCacheMode::None;

    timestamps_[ctx_->backend_frame].primary_shade[0] = ctx_->WriteTimestamp(cmd_buf, true);

    { // shade primary hits
        DebugMarker _(ctx_.get(), cmd_buf, "ShadePrimaryHits");
        kernel_ShadePrimaryHits(cmd_buf, cam.pass_settings, cache_mode, s.env_, indir_args_buf_[0], 0, prim_hits_buf_,
                                prim_rays_buf_, sc_data, random_seq_buf_, rand_seed, region.iteration, rect,
                                s.tex_atlases_, s.bindless_tex_data_, temp_buf0_, secondary_rays_buf_, shadow_rays_buf_,
                                ray_hashes_bufs_[0], counters_buf_, temp_buf1_, temp_depth_normals_buf_);
    }

    { // prepare indirect args
        DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[0]);
    }

    if (sc_data.env.sky_map_spread_angle > 0.0f) {
        DebugMarker _(ctx_.get(), cmd_buf, "ShadeSkyPrimary");
        kernel_ShadeSkyPrimary(cmd_buf, cam.pass_settings, s.env_, indir_args_buf_[0], 4, prim_hits_buf_,
                               prim_rays_buf_, ray_hashes_bufs_[0], counters_buf_, sc_data, region.iteration,
                               temp_buf0_);
    }

    timestamps_[ctx_->backend_frame].primary_shade[1] = ctx_->WriteTimestamp(cmd_buf, false);

    { // trace shadow rays
        DebugMarker _(ctx_.get(), cmd_buf, "TraceShadow");
        timestamps_[ctx_->backend_frame].primary_shadow[0] = ctx_->WriteTimestamp(cmd_buf, true);
        kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_[0], 2, counters_buf_, sc_data,
                                    random_seq_buf_, rand_seed, region.iteration, s.tlas_root_,
                                    cam.pass_settings.clamp_direct, s.tex_atlases_, s.bindless_tex_data_,
                                    shadow_rays_buf_, temp_buf0_);
        timestamps_[ctx_->backend_frame].primary_shadow[1] = ctx_->WriteTimestamp(cmd_buf, false);
    }

    timestamps_[ctx_->backend_frame].secondary_sort.clear();
    timestamps_[ctx_->backend_frame].secondary_trace.clear();
    timestamps_[ctx_->backend_frame].secondary_shade.clear();
    timestamps_[ctx_->backend_frame].secondary_shadow.clear();

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth; ++bounce) {
#if !DISABLE_SORTING
        timestamps_[ctx_->backend_frame].secondary_sort.push_back(ctx_->WriteTimestamp(cmd_buf, true));

        if (!use_hwrt_ && use_subgroup_) {
            DebugMarker _(ctx_.get(), cmd_buf, "Sort Rays");

            kernel_SortHashRays(cmd_buf, indir_args_buf_[0], secondary_rays_buf_, counters_buf_, root_min, cell_size,
                                ray_hashes_bufs_[0]);
            RadixSort(cmd_buf, indir_args_buf_[0], ray_hashes_bufs_, count_table_buf_, counters_buf_,
                      reduce_table_buf_);
            kernel_SortReorderRays(cmd_buf, indir_args_buf_[0], 0, secondary_rays_buf_, ray_hashes_bufs_[0],
                                   counters_buf_, 1, prim_rays_buf_);

            std::swap(secondary_rays_buf_, prim_rays_buf_);
        }

        timestamps_[ctx_->backend_frame].secondary_sort.push_back(ctx_->WriteTimestamp(cmd_buf, false));
#endif // !DISABLE_SORTING

        timestamps_[ctx_->backend_frame].secondary_trace.push_back(ctx_->WriteTimestamp(cmd_buf, true));
        { // trace secondary rays
            DebugMarker _(ctx_.get(), cmd_buf, "IntersectSceneSecondary");
            if (use_rt_pipeline) {
                kernel_IntersectScene_RTPipe(cmd_buf, indir_args_buf_[0], 1, cam.pass_settings, sc_data,
                                             random_seq_buf_, rand_seed, region.iteration, s.tlas_root_, nullptr, -1.0f,
                                             s.tex_atlases_, s.bindless_tex_data_, secondary_rays_buf_, prim_hits_buf_);
            } else {
                kernel_IntersectScene(cmd_buf, indir_args_buf_[0], 0, counters_buf_, cam.pass_settings, sc_data,
                                      random_seq_buf_, rand_seed, region.iteration, s.tlas_root_, nullptr, -1.0f,
                                      s.tex_atlases_, s.bindless_tex_data_, secondary_rays_buf_, prim_hits_buf_);
            }
        }

        if (sc_data.visible_lights_count) {
            DebugMarker _(ctx_.get(), cmd_buf, "IntersectAreaLights");
            kernel_IntersectAreaLights(cmd_buf, sc_data, indir_args_buf_[0], counters_buf_, secondary_rays_buf_,
                                       prim_hits_buf_);
        }

        timestamps_[ctx_->backend_frame].secondary_trace.push_back(ctx_->WriteTimestamp(cmd_buf, false));

        timestamps_[ctx_->backend_frame].secondary_shade.push_back(ctx_->WriteTimestamp(cmd_buf, true));

        const float clamp_val = (bounce == 1) ? cam.pass_settings.clamp_direct : cam.pass_settings.clamp_indirect;
        { // shade secondary hits
            DebugMarker _(ctx_.get(), cmd_buf, "ShadeSecondaryHits");
            kernel_ShadeSecondaryHits(cmd_buf, cam.pass_settings, cache_mode, clamp_val, s.env_, indir_args_buf_[0], 0,
                                      prim_hits_buf_, secondary_rays_buf_, sc_data, random_seq_buf_, rand_seed,
                                      region.iteration, s.tex_atlases_, s.bindless_tex_data_, temp_buf0_,
                                      prim_rays_buf_, shadow_rays_buf_, ray_hashes_bufs_[0], counters_buf_,
                                      temp_depth_normals_buf_);
        }

        { // prepare indirect args
            DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
            kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[0]);
        }

        if (sc_data.env.sky_map_spread_angle > 0.0f) {
            DebugMarker _(ctx_.get(), cmd_buf, "ShadeSkySecondary");
            kernel_ShadeSkySecondary(cmd_buf, cam.pass_settings, clamp_val, s.env_, indir_args_buf_[0], 4,
                                     prim_hits_buf_, secondary_rays_buf_, ray_hashes_bufs_[0], counters_buf_, sc_data,
                                     region.iteration, temp_buf0_);
        }

        timestamps_[ctx_->backend_frame].secondary_shade.push_back(ctx_->WriteTimestamp(cmd_buf, false));

        { // trace shadow rays
            DebugMarker _(ctx_.get(), cmd_buf, "TraceShadow");
            timestamps_[ctx_->backend_frame].secondary_shadow.push_back(ctx_->WriteTimestamp(cmd_buf, true));
            kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_[0], 2, counters_buf_, sc_data,
                                        random_seq_buf_, rand_seed, region.iteration, s.tlas_root_,
                                        cam.pass_settings.clamp_indirect, s.tex_atlases_, s.bindless_tex_data_,
                                        shadow_rays_buf_, temp_buf0_);
            timestamps_[ctx_->backend_frame].secondary_shadow.push_back(ctx_->WriteTimestamp(cmd_buf, false));
        }

        std::swap(secondary_rays_buf_, prim_rays_buf_);
    }
#endif

    { // prepare result
        DebugMarker _(ctx_.get(), cmd_buf, "Prepare Result");

        const float exposure = std::pow(2.0f, cam.exposure);

        // factor used to compute incremental average
        const float mix_factor = 1.0f / float(region.iteration);
        const float half_mix_factor = 1.0f / float((region.iteration + 1) / 2);

        kernel_MixIncremental(cmd_buf, mix_factor, half_mix_factor, rect, region.iteration, exposure, temp_buf0_,
                              temp_buf1_, temp_depth_normals_buf_, required_samples_buf_, full_buf_, half_buf_,
                              base_color_buf_, depth_normals_buf_);
    }

    { // output final buffer, prepare variance
        DebugMarker _(ctx_.get(), cmd_buf, "Postprocess frame");

        tonemap_params_.view_transform = cam.view_transform;
        tonemap_params_.inv_gamma = (1.0f / cam.gamma);

        variance_threshold_ = region.iteration > cam.pass_settings.min_samples
                                  ? 0.5f * cam.pass_settings.variance_threshold * cam.pass_settings.variance_threshold
                                  : 0.0f;

        kernel_Postprocess(cmd_buf, full_buf_, half_buf_, tonemap_params_.inv_gamma, rect, variance_threshold_,
                           region.iteration, final_buf_, temp_buf0_, required_samples_buf_);
        // Also store as denosed result until Denoise method will be called
        const TransitionInfo img_transitions[] = {{&full_buf_, eResState::CopySrc},
                                                  {&raw_filtered_buf_, eResState::CopyDst}};
        TransitionResourceStates(cmd_buf, AllStages, AllStages, img_transitions);
        CopyImageToImage(cmd_buf, full_buf_, 0, rect.x, rect.y, raw_filtered_buf_, 0, rect.x, rect.y, rect.w, rect.h);
    }

    render_scene_marker.End();

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
        const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};

        if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = wait_semaphores;
            submit_info.pWaitDstStageMask = wait_stages;
        }

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

        const VkResult res = ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info,
                                                       ctx_->in_flight_fence(ctx_->backend_frame));
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to submit into a queue!");
        }

        ctx_->frame_cpu_synced[ctx_->backend_frame] = false;
        ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
        ctx_->render_finished_semaphore_is_set[prev_frame] = false;

        ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
    }
#endif
    frame_dirty_ = base_color_dirty_ = depth_normals_dirty_ = true;
}

void Ray::Vk::Renderer::DenoiseImage(const RegionContext &region) {
    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    }
#if !RUN_IN_LOCKSTEP
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE,
                                    UINT64_MAX);
        ctx_->api().vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
    }
#endif

    const bool reset_queries = !ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP;
    if (!ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP) {
        ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
        ctx_->DestroyDeferredResources(ctx_->backend_frame);
        ctx_->default_descr_alloc()->Reset();
    }

    stats_.time_denoise_us = ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].denoise[0],
                                                                  timestamps_[ctx_->backend_frame].denoise[1]);

#if RUN_IN_LOCKSTEP
    cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
    if (!cmd_buf) {
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        ctx_->api().vkBeginCommandBuffer(ctx_->draw_cmd_buf(ctx_->backend_frame), &begin_info);
        cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
    }
#endif

    DebugMarker denoise_marker(ctx_.get(), cmd_buf, "Ray::DenoiseImage");

    if (reset_queries) {
        ctx_->api().vkCmdResetQueryPool(cmd_buf, ctx_->query_pool(ctx_->backend_frame), 0, MaxTimestampQueries);
    }

    //////////////////////////////////////////////////////////////////////////////////

    timestamps_[ctx_->backend_frame].denoise[0] = ctx_->WriteTimestamp(cmd_buf, true);

    const rect_t &rect = region.rect();

    const auto &raw_variance = temp_buf0_;
    const auto &filtered_variance = temp_buf1_;

    { // Filter variance
        DebugMarker _(ctx_.get(), cmd_buf, "Filter Variance");
        kernel_FilterVariance(cmd_buf, raw_variance, rect, variance_threshold_, region.iteration, filtered_variance,
                              required_samples_buf_);
    }

    { // Apply NLM Filter
        DebugMarker _(ctx_.get(), cmd_buf, "NLM Filter");
        kernel_NLMFilter(cmd_buf, full_buf_, filtered_variance, 1.0f, 0.45f, base_color_buf_, 64.0f, depth_normals_buf_,
                         32.0f, raw_filtered_buf_, tonemap_params_.view_transform, tonemap_params_.inv_gamma, rect,
                         final_buf_);
    }

    timestamps_[ctx_->backend_frame].denoise[1] = ctx_->WriteTimestamp(cmd_buf, false);

    //////////////////////////////////////////////////////////////////////////////////

    denoise_marker.End();

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
        const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};

        if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = wait_semaphores;
            submit_info.pWaitDstStageMask = wait_stages;
        }

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

        const VkResult res = ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info,
                                                       ctx_->in_flight_fence(ctx_->backend_frame));
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to submit into a queue!");
        }

        ctx_->frame_cpu_synced[ctx_->backend_frame] = false;
        ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
        ctx_->render_finished_semaphore_is_set[prev_frame] = false;

        ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
    }
#endif
}

void Ray::Vk::Renderer::DenoiseImage(const int pass, const RegionContext &region) {
    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    }
    if (pass == 0) {
#if !RUN_IN_LOCKSTEP
        if (!external_cmd_buf_.vk_cmd_buf) {
            ctx_->api().vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE,
                                        UINT64_MAX);
            ctx_->api().vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
        }
#endif
        const bool reset_queries = !ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP;
        if (!ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP) {
            ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
            ctx_->DestroyDeferredResources(ctx_->backend_frame);
            ctx_->default_descr_alloc()->Reset();
            ctx_->frame_cpu_synced[ctx_->backend_frame] = true;
        }

        stats_.time_denoise_us = ctx_->GetTimestampIntervalDurationUs(timestamps_[ctx_->backend_frame].denoise[0],
                                                                      timestamps_[ctx_->backend_frame].denoise[1]);

#if RUN_IN_LOCKSTEP
        cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
        if (!external_cmd_buf_.vk_cmd_buf) {
            VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
            ctx_->api().vkBeginCommandBuffer(cmd_buf, &begin_info);
        }
#endif
        if (reset_queries) {
            ctx_->api().vkCmdResetQueryPool(cmd_buf, ctx_->query_pool(ctx_->backend_frame), 0, MaxTimestampQueries);
        }
#ifdef ENABLE_GPU_DEBUG
        VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = "Ray::DenoiseImage";
        label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
        ctx_->api().vkCmdBeginDebugUtilsLabelEXT(cmd_buf, &label);
#endif

        timestamps_[ctx_->backend_frame].denoise[0] = ctx_->WriteTimestamp(cmd_buf, true);
    } else {
#if RUN_IN_LOCKSTEP
        cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
        if (!cmd_buf) {
            cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
        }
#endif
    }

    //////////////////////////////////////////////////////////////////////////////////

    const int w_rounded = round_up(w_, 16);
    const int h_rounded = round_up(h_, 16);

    rect_t r = region.rect();
    if (pass < 15) {
        r.w = round_up(r.w, 16);
        r.h = round_up(r.h, 16);
    }

    Buffer *weights = &unet_weights_;
    const unet_weight_offsets_t *offsets = &unet_offsets_;

    // NOTE: timings captured for 513x513 resolution on 3080 nvidia

    switch (pass) {
    case 0: { // fp32 0.53ms, fp16 0.52ms, nv 0.33ms
        const int output_stride = round_up(w_rounded + 1, 16) + 1;
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 9 32");
        kernel_Convolution(cmd_buf, 9, 32, full_buf_, base_color_buf_, depth_normals_buf_, zero_border_sampler_, r,
                           w_rounded, h_rounded, *weights, offsets->enc_conv0_weight, offsets->enc_conv0_bias,
                           unet_tensors_heap_, unet_tensors_.enc_conv0_offset, output_stride);
    } break;
    case 1: { // fp32 2.44ms, fp16 1.96ms, nv 0.61ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 32 32 Downscale");
        const int input_stride = round_up(w_rounded + 1, 16) + 1, output_stride = round_up(w_rounded / 2 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 32, 32, unet_tensors_heap_, unet_tensors_.enc_conv0_offset, input_stride, r,
                           w_rounded, h_rounded, *weights, offsets->enc_conv1_weight, offsets->enc_conv1_bias,
                           unet_tensors_heap_, unet_tensors_.pool1_offset, output_stride, true);
    } break;
    case 2: { // fp32 1.17ms, fp16 0.76ms, nv 0.27ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 32 48 Downscale");
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;

        const int input_stride = round_up(w_rounded / 2 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 4 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 32, 48, unet_tensors_heap_, unet_tensors_.pool1_offset, input_stride, r,
                           w_rounded / 2, h_rounded / 2, *weights, offsets->enc_conv2_weight, offsets->enc_conv2_bias,
                           unet_tensors_heap_, unet_tensors_.pool2_offset, output_stride, true);
    } break;
    case 3: { // fp32 0.40ms, fp16 0.27ms, nv 0.09ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 48 64 Downscale");
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;

        const int input_stride = round_up(w_rounded / 4 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 8 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 48, 64, unet_tensors_heap_, unet_tensors_.pool2_offset, input_stride, r,
                           w_rounded / 4, h_rounded / 4, *weights, offsets->enc_conv3_weight, offsets->enc_conv3_bias,
                           unet_tensors_heap_, unet_tensors_.pool3_offset, output_stride, true);
    } break;
    case 4: { // fp32 0.24ms, fp16 0.24ms, nv 0.07ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 64 80 Downscale");
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;

        const int input_stride = round_up(w_rounded / 8 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 16 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 64, 80, unet_tensors_heap_, unet_tensors_.pool3_offset, input_stride, r,
                           w_rounded / 8, h_rounded / 8, *weights, offsets->enc_conv4_weight, offsets->enc_conv4_bias,
                           unet_tensors_heap_, unet_tensors_.pool4_offset, output_stride, true);
    } break;
    case 5: { // fp32 0.17ms, fp16 0.12ms, nv 0.05ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 80 96");
        r.x = r.x / 16;
        r.y = r.y / 16;
        r.w = (r.w + 15) / 16;
        r.h = (r.h + 15) / 16;

        const int input_stride = round_up(w_rounded / 16 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 16 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 80, 96, unet_tensors_heap_, unet_tensors_.pool4_offset, input_stride, r,
                           w_rounded / 16, h_rounded / 16, *weights, offsets->enc_conv5a_weight,
                           offsets->enc_conv5a_bias, unet_tensors_heap_, unet_tensors_.enc_conv5a_offset, output_stride,
                           false);
    } break;
    case 6: { // fp32 0.195ms, fp16 0.20ms, nv 0.06ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 96 96");
        r.x = r.x / 16;
        r.y = r.y / 16;
        r.w = (r.w + 15) / 16;
        r.h = (r.h + 15) / 16;

        const int input_stride = round_up(w_rounded / 16 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 16 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 96, 96, unet_tensors_heap_, unet_tensors_.enc_conv5a_offset, input_stride, r,
                           w_rounded / 16, h_rounded / 16, *weights, offsets->enc_conv5b_weight,
                           offsets->enc_conv5b_bias, unet_tensors_heap_, unet_tensors_.upsample4_offset, output_stride,
                           false);
    } break;
    case 7: { // fp32 0.71ms, fp16 0.55ms, nv 0.22ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution Concat 96 64 112");
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;

        const int input_stride1 = round_up(w_rounded / 16 + 1, 16) + 1,
                  input_stride2 = round_up(w_rounded / 8 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 8 + 1, 16) + 1;
        kernel_ConvolutionConcat(cmd_buf, 96, 64, 112, unet_tensors_heap_, unet_tensors_.upsample4_offset,
                                 input_stride1, true, unet_tensors_heap_, unet_tensors_.pool3_offset, input_stride2, r,
                                 w_rounded / 8, h_rounded / 8, *weights, offsets->dec_conv4a_weight,
                                 offsets->dec_conv4a_bias, unet_tensors_heap_, unet_tensors_.dec_conv4a_offset,
                                 output_stride);
    } break;
    case 8: { // fp32 0.47ms, fp16 0.36ms, nv 0.09ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 112 112");
        r.x = r.x / 8;
        r.y = r.y / 8;
        r.w = (r.w + 7) / 8;
        r.h = (r.h + 7) / 8;

        const int input_stride = round_up(w_rounded / 8 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 8 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 112, 112, unet_tensors_heap_, unet_tensors_.dec_conv4a_offset, input_stride, r,
                           w_rounded / 8, h_rounded / 8, *weights, offsets->dec_conv4b_weight, offsets->dec_conv4b_bias,
                           unet_tensors_heap_, unet_tensors_.upsample3_offset, output_stride, false);
    } break;
    case 9: { // fp32 1.57ms, fp16 1.11ms, nv 0.53ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution Concat 112 48 96");
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;

        const int input_stride1 = round_up(w_rounded / 8 + 1, 16) + 1,
                  input_stride2 = round_up(w_rounded / 4 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 4 + 1, 16) + 1;
        kernel_ConvolutionConcat(cmd_buf, 112, 48, 96, unet_tensors_heap_, unet_tensors_.upsample3_offset,
                                 input_stride1, true, unet_tensors_heap_, unet_tensors_.pool2_offset, input_stride2, r,
                                 w_rounded / 4, h_rounded / 4, *weights, offsets->dec_conv3a_weight,
                                 offsets->dec_conv3a_bias, unet_tensors_heap_, unet_tensors_.dec_conv3a_offset,
                                 output_stride);
    } break;
    case 10: { // fp32 1.33ms, fp16 1.32ms, nv 0.53ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 96 96");
        r.x = r.x / 4;
        r.y = r.y / 4;
        r.w = (r.w + 3) / 4;
        r.h = (r.h + 3) / 4;

        const int input_stride = round_up(w_rounded / 4 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 4 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 96, 96, unet_tensors_heap_, unet_tensors_.dec_conv3a_offset, input_stride, r,
                           w_rounded / 4, h_rounded / 4, *weights, offsets->dec_conv3b_weight, offsets->dec_conv3b_bias,
                           unet_tensors_heap_, unet_tensors_.upsample2_offset, output_stride, false);
    } break;
    case 11: { // fp32 4.37ms, fp16 2.97ms, nv 1.44ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution Concat 96 32 64");
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;

        const int input_stride1 = round_up(w_rounded / 4 + 1, 16) + 1,
                  input_stride2 = round_up(w_rounded / 2 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 2 + 1, 16) + 1;
        kernel_ConvolutionConcat(cmd_buf, 96, 32, 64, unet_tensors_heap_, unet_tensors_.upsample2_offset, input_stride1,
                                 true, unet_tensors_heap_, unet_tensors_.pool1_offset, input_stride2, r, w_rounded / 2,
                                 h_rounded / 2, *weights, offsets->dec_conv2a_weight, offsets->dec_conv2a_bias,
                                 unet_tensors_heap_, unet_tensors_.dec_conv2a_offset, output_stride);
    } break;
    case 12: { // fp32 3.83ms, fp16 2.74ms, nv 0.98ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 64 64");
        r.x = r.x / 2;
        r.y = r.y / 2;
        r.w = (r.w + 1) / 2;
        r.h = (r.h + 1) / 2;

        const int input_stride = round_up(w_rounded / 2 + 1, 16) + 1,
                  output_stride = round_up(w_rounded / 2 + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 64, 64, unet_tensors_heap_, unet_tensors_.dec_conv2a_offset, input_stride, r,
                           w_rounded / 2, h_rounded / 2, *weights, offsets->dec_conv2b_weight, offsets->dec_conv2b_bias,
                           unet_tensors_heap_, unet_tensors_.upsample1_offset, output_stride, false);
    } break;
    case 13: { // fp32 8.72ms, fp16 9.34ms, nv 3.46ms
        const int input_stride = round_up(w_rounded / 2 + 1, 16) + 1, output_stride = round_up(w_rounded + 1, 16) + 1;
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution Concat 64 9 64");
        kernel_ConvolutionConcat(cmd_buf, 64, 9, 64, unet_tensors_heap_, unet_tensors_.upsample1_offset, input_stride,
                                 true, full_buf_, base_color_buf_, depth_normals_buf_, zero_border_sampler_, r,
                                 w_rounded, h_rounded, *weights, offsets->dec_conv1a_weight, offsets->dec_conv1a_bias,
                                 unet_tensors_heap_, unet_tensors_.dec_conv1a_offset, output_stride);
    } break;
    case 14: { // fp32 7.88ms, fp16 5.73ms, nv 1.82ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 64 32");
        const int input_stride = round_up(w_rounded + 1, 16) + 1, output_stride = round_up(w_rounded + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 64, 32, unet_tensors_heap_, unet_tensors_.dec_conv1a_offset, input_stride, r,
                           w_rounded, h_rounded, *weights, offsets->dec_conv1b_weight, offsets->dec_conv1b_bias,
                           unet_tensors_heap_, unet_tensors_.dec_conv1b_offset, output_stride, false);
    } break;
    case 15: { // fp32 1.86ms, fp16 0.44ms, nv 0.13ms
        DebugMarker _(ctx_.get(), cmd_buf, "Convolution 32 3 Img ");
        const int input_stride = round_up(w_rounded + 1, 16) + 1;
        kernel_Convolution(cmd_buf, 32, 3, unet_tensors_heap_, unet_tensors_.dec_conv1b_offset, input_stride,
                           tonemap_params_.inv_gamma, r, w_, h_, *weights, offsets->dec_conv0_weight,
                           offsets->dec_conv0_bias, raw_filtered_buf_, final_buf_);
    } break;
    }

    //////////////////////////////////////////////////////////////////////////////////

    if (pass == 15) {
        timestamps_[ctx_->backend_frame].denoise[1] = ctx_->WriteTimestamp(cmd_buf, false);

#ifdef ENABLE_GPU_DEBUG
        ctx_->api().vkCmdEndDebugUtilsLabelEXT(cmd_buf);
#endif

#if RUN_IN_LOCKSTEP
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
        if (!external_cmd_buf_.vk_cmd_buf) {
            ctx_->api().vkEndCommandBuffer(cmd_buf);

            ///

            const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

            VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

            const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
            const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};

            if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
                submit_info.waitSemaphoreCount = 1;
                submit_info.pWaitSemaphores = wait_semaphores;
                submit_info.pWaitDstStageMask = wait_stages;
            }

            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

            const VkResult res = ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info,
                                                           ctx_->in_flight_fence(ctx_->backend_frame));
            if (res != VK_SUCCESS) {
                ctx_->log()->Error("Failed to submit into a queue!");
            }

            ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
            ctx_->render_finished_semaphore_is_set[prev_frame] = false;

            ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
        }
#endif
    } else {
#if RUN_IN_LOCKSTEP
        /*if (pass == 1) {
            Buffer debug_buf("Tensors Debug Buf", ctx_.get(), eBufType::Readback, unet_tensors_heap_.size());

            CopyBufferToBuffer(unet_tensors_heap_, 0, debug_buf, 0, unet_tensors_heap_.size(), cmd_buf);

            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());

            const uint16_t *_data = (const uint16_t *)debug_buf.Map();
            _data += unet_tensors_.pool1_offset / sizeof(uint16_t);

            const int stride = 369;
            const int channels = 32;

            std::vector<float> data1;
            for (int i = 0; i < stride * 500; ++i) {
                data1.push_back(f16_to_f32(_data[i * channels]));
            }
            WritePFM("test", data1.data(), stride, 500, 1);

            debug_buf.Unmap();
        }*/
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#endif
    }
}

void Ray::Vk::Renderer::UpdateSpatialCache(const SceneBase &scene, RegionContext &region) {
    if (!use_spatial_cache_) {
        return;
    }

    const auto &s = dynamic_cast<const Vk::Scene &>(scene);

    ++region.cache_iteration;

    float root_min[3], cell_size[3];
    if (s.tlas_root_ != 0xffffffff) {
        const bvh_node_t &root_node = s.tlas_root_node_;
        for (int i = 0; i < 3; ++i) {
            root_min[i] = root_node.bbox_min[i];
            cell_size[i] = (root_node.bbox_max[i] - root_node.bbox_min[i]) / 255;
        }
    }

    rect_t rect = region.rect();
    rect.x /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.y /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.w /= RAD_CACHE_DOWNSAMPLING_FACTOR;
    rect.h /= RAD_CACHE_DOWNSAMPLING_FACTOR;

    const camera_t &orig_cam = s.cams_[s.current_cam()._index];

    camera_t cam = s.cams_[s.current_cam()._index];
    cam.fstop = 0.0f;
    cam.filter = ePixelFilter::Box;

    // TODO: Filter table is unused, this can be removed
    if (orig_cam.filter != filter_table_filter_ || orig_cam.filter_width != filter_table_width_) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        UpdateFilterTable(cmd_buf, orig_cam.filter, orig_cam.filter_width);
        filter_table_filter_ = orig_cam.filter;
        filter_table_width_ = orig_cam.filter_width;

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    cache_grid_params_t cache_grid_params;
    memcpy(cache_grid_params.cam_pos_curr, cam.origin, 3 * sizeof(float));
    cache_grid_params.exposure = std::pow(2.0f, cam.exposure);

    const scene_data_t sc_data = {s.env_,
                                  s.mesh_instances_.gpu_buf(),
                                  {s.meshes_.data(), s.meshes_.capacity()},
                                  s.vtx_indices_.gpu_buf(),
                                  s.vertices_.gpu_buf(),
                                  s.nodes_.gpu_buf(),
                                  s.tris_.gpu_buf(),
                                  s.tri_indices_.gpu_buf(),
                                  s.tri_materials_.gpu_buf(),
                                  s.materials_.gpu_buf(),
                                  s.tex_atlases_,
                                  s.atlas_textures_.gpu_buf(),
                                  s.lights_,
                                  s.li_indices_.buf(),
                                  int(s.li_indices_.size()),
                                  s.dir_lights_,
                                  s.visible_lights_count_,
                                  s.blocker_lights_count_,
                                  s.light_cwnodes_.buf(),
                                  s.rt_tlas_,
                                  s.env_map_qtree_.tex,
                                  int(s.env_map_qtree_.mips.size()),
                                  cache_grid_params,
                                  s.spatial_cache_entries_.buf(),
                                  s.spatial_cache_voxels_curr_.buf(),
                                  s.atmosphere_params_buf_,
                                  {},
                                  {},
                                  {},
                                  {},
                                  {},
                                  {},
                                  {}};

    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    } else {
#if !RUN_IN_LOCKSTEP
        ctx_->api().vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE,
                                    UINT64_MAX);
        ctx_->api().vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
#endif
    }

    const bool reset_queries = !ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP;
    if (!ctx_->frame_cpu_synced[ctx_->backend_frame] || RUN_IN_LOCKSTEP) {
        ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
        ctx_->DestroyDeferredResources(ctx_->backend_frame);
        ctx_->default_descr_alloc()->Reset();
        ctx_->frame_cpu_synced[ctx_->backend_frame] = true;
    }

    stats_.time_cache_update_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].cache_update[0], timestamps_[ctx_->backend_frame].cache_update[1]);
    stats_.time_cache_resolve_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].cache_resolve[0], timestamps_[ctx_->backend_frame].cache_resolve[1]);

#if RUN_IN_LOCKSTEP
    cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
    if (!cmd_buf) {
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
        ctx_->api().vkBeginCommandBuffer(cmd_buf, &begin_info);
    }
#endif

    DebugMarker update_cache_marker(ctx_.get(), cmd_buf, "Ray::UpdateSpatialCache");

    if (reset_queries) {
        ctx_->api().vkCmdResetQueryPool(cmd_buf, ctx_->query_pool(ctx_->backend_frame), 0, MaxTimestampQueries);
    }

    //////////////////////////////////////////////////////////////////////////////////

    TransitionSceneResources(cmd_buf, sc_data);

    VkMemoryBarrier mem_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    ctx_->api().vkCmdPipelineBarrier(
        cmd_buf,
        (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) &
            ctx_->supported_stages_mask(),
        (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) &
            ctx_->supported_stages_mask(),
        0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

    const uint32_t rand_seed = Ref::hash(Ref::hash((region.cache_iteration - 1) / RAND_SAMPLES_COUNT));

    timestamps_[ctx_->backend_frame].cache_update[0] = ctx_->WriteTimestamp(cmd_buf, true);

    { // generate primary rays
        DebugMarker _(ctx_.get(), cmd_buf, "GeneratePrimaryRays");
        kernel_GeneratePrimaryRays(cmd_buf, cam, rand_seed, rect, (w_ / RAD_CACHE_DOWNSAMPLING_FACTOR),
                                   (h_ / RAD_CACHE_DOWNSAMPLING_FACTOR), random_seq_buf_, filter_table_,
                                   region.cache_iteration, false, required_samples_buf_, counters_buf_, prim_rays_buf_);
    }

    const bool use_rt_pipeline = (use_hwrt_ && ENABLE_RT_PIPELINE);

    { // prepare indirect args
        DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[0]);
    }

    { // trace primary rays
        DebugMarker _(ctx_.get(), cmd_buf, "IntersectScenePrimary");
        if (use_rt_pipeline) {
            kernel_IntersectScene_RTPipe(cmd_buf, indir_args_buf_[0], 1, cam.pass_settings, sc_data, random_seq_buf_,
                                         rand_seed, region.cache_iteration, s.tlas_root_, cam.fwd,
                                         cam.clip_end - cam.clip_start, s.tex_atlases_, s.bindless_tex_data_,
                                         prim_rays_buf_, prim_hits_buf_);
        } else {
            kernel_IntersectScene(cmd_buf, indir_args_buf_[0], 0, counters_buf_, cam.pass_settings, sc_data,
                                  random_seq_buf_, rand_seed, region.cache_iteration, s.tlas_root_, cam.fwd,
                                  cam.clip_end - cam.clip_start, s.tex_atlases_, s.bindless_tex_data_, prim_rays_buf_,
                                  prim_hits_buf_);
        }
    }

    { // shade primary hits
        DebugMarker _(ctx_.get(), cmd_buf, "ShadePrimaryHits");
        kernel_ShadePrimaryHits(cmd_buf, cam.pass_settings, eSpatialCacheMode::Update, s.env_, indir_args_buf_[0], 0,
                                prim_hits_buf_, prim_rays_buf_, sc_data, random_seq_buf_, rand_seed,
                                region.cache_iteration, rect, s.tex_atlases_, s.bindless_tex_data_, temp_buf0_,
                                secondary_rays_buf_, shadow_rays_buf_, {}, counters_buf_, temp_buf1_,
                                temp_depth_normals_buf_);
    }

    { // prepare indirect args
        DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[1]);
    }

    { // trace shadow rays
        DebugMarker _(ctx_.get(), cmd_buf, "Ray::TraceShadow");
        kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_[1], 2, counters_buf_, sc_data,
                                    random_seq_buf_, rand_seed, region.cache_iteration, s.tlas_root_,
                                    cam.pass_settings.clamp_direct, s.tex_atlases_, s.bindless_tex_data_,
                                    shadow_rays_buf_, temp_buf0_);
    }

    { // update spatial cache
        DebugMarker _(ctx_.get(), cmd_buf, "UpdateSpatialCache");
        temp_cache_data_buf_.Fill(0, temp_cache_data_buf_.size(), 0, cmd_buf);
        if (!ctx_->int64_atomics_supported()) {
            temp_lock_buf_.Fill(0, temp_lock_buf_.size(), 0, cmd_buf);
        }
        kernel_SpatialCacheUpdate(cmd_buf, cache_grid_params, indir_args_buf_[0], 0, counters_buf_, 0, prim_hits_buf_,
                                  prim_rays_buf_, temp_cache_data_buf_, temp_buf0_, temp_depth_normals_buf_,
                                  sc_data.spatial_cache_entries, sc_data.spatial_cache_voxels, temp_lock_buf_);
    }

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth; ++bounce) {
#if !DISABLE_SORTING
        if (!use_hwrt_ && use_subgroup_) {
            DebugMarker _(ctx_.get(), cmd_buf, "Sort Rays");

            kernel_SortHashRays(cmd_buf, indir_args_buf_[1], secondary_rays_buf_, counters_buf_, root_min, cell_size,
                                ray_hashes_bufs_[0]);
            RadixSort(cmd_buf, indir_args_buf_[1], ray_hashes_bufs_, count_table_buf_, counters_buf_,
                      reduce_table_buf_);
            kernel_SortReorderRays(cmd_buf, indir_args_buf_[1], 0, secondary_rays_buf_, ray_hashes_bufs_[0],
                                   counters_buf_, 1, prim_rays_buf_);

            std::swap(secondary_rays_buf_, prim_rays_buf_);
        }
#endif // !DISABLE_SORTING

        { // trace secondary rays
            DebugMarker _(ctx_.get(), cmd_buf, "IntersectSceneSecondary");
            if (use_rt_pipeline) {
                kernel_IntersectScene_RTPipe(cmd_buf, indir_args_buf_[1], 1, cam.pass_settings, sc_data,
                                             random_seq_buf_, rand_seed, region.cache_iteration, s.tlas_root_, nullptr,
                                             -1.0f, s.tex_atlases_, s.bindless_tex_data_, secondary_rays_buf_,
                                             prim_hits_buf_);
            } else {
                kernel_IntersectScene(cmd_buf, indir_args_buf_[1], 0, counters_buf_, cam.pass_settings, sc_data,
                                      random_seq_buf_, rand_seed, region.cache_iteration, s.tlas_root_, nullptr, -1.0f,
                                      s.tex_atlases_, s.bindless_tex_data_, secondary_rays_buf_, prim_hits_buf_);
            }
        }

        if (sc_data.visible_lights_count) {
            DebugMarker _(ctx_.get(), cmd_buf, "IntersectAreaLights");
            kernel_IntersectAreaLights(cmd_buf, sc_data, indir_args_buf_[1], counters_buf_, secondary_rays_buf_,
                                       prim_hits_buf_);
        }

        { // shade secondary hits
            DebugMarker _(ctx_.get(), cmd_buf, "ShadeSecondaryHits");
            const float clamp_val = (bounce == 1) ? cam.pass_settings.clamp_direct : cam.pass_settings.clamp_indirect;
            kernel_ShadeSecondaryHits(cmd_buf, cam.pass_settings, eSpatialCacheMode::Update, clamp_val, s.env_,
                                      indir_args_buf_[1], 0, prim_hits_buf_, secondary_rays_buf_, sc_data,
                                      random_seq_buf_, rand_seed, region.cache_iteration, s.tex_atlases_,
                                      s.bindless_tex_data_, temp_buf0_, prim_rays_buf_, shadow_rays_buf_, {},
                                      counters_buf_, temp_depth_normals_buf_);
        }

        { // prepare indirect args
            DebugMarker _(ctx_.get(), cmd_buf, "PrepareIndirArgs");
            kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_[0]);
        }

        { // trace shadow rays
            DebugMarker _(ctx_.get(), cmd_buf, "TraceShadow");
            kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_[0], 2, counters_buf_, sc_data,
                                        random_seq_buf_, rand_seed, region.cache_iteration, s.tlas_root_,
                                        cam.pass_settings.clamp_indirect, s.tex_atlases_, s.bindless_tex_data_,
                                        shadow_rays_buf_, temp_buf0_);
        }

        { // update spatial cache
            DebugMarker _(ctx_.get(), cmd_buf, "UpdateSpatialCache");
            kernel_SpatialCacheUpdate(cmd_buf, cache_grid_params, indir_args_buf_[1], 0, counters_buf_, 0,
                                      prim_hits_buf_, secondary_rays_buf_, temp_cache_data_buf_, temp_buf0_,
                                      temp_depth_normals_buf_, sc_data.spatial_cache_entries,
                                      sc_data.spatial_cache_voxels, temp_lock_buf_);
        }

        std::swap(secondary_rays_buf_, prim_rays_buf_);
        std::swap(indir_args_buf_[0], indir_args_buf_[1]);
    }

    timestamps_[ctx_->backend_frame].cache_update[1] = ctx_->WriteTimestamp(cmd_buf, false);

    update_cache_marker.End();

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#endif
}

void Ray::Vk::Renderer::ResolveSpatialCache(const SceneBase &scene,
                                            const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    if (!use_spatial_cache_) {
        return;
    }

    const auto &s = dynamic_cast<const Vk::Scene &>(scene);

    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    }
#if RUN_IN_LOCKSTEP
    cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
    if (!cmd_buf) {
        cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
    }
#endif

    DebugMarker resolve_cache_marker(ctx_.get(), cmd_buf, "Ray::ResolveSpatialCache");

    timestamps_[ctx_->backend_frame].cache_resolve[0] = ctx_->WriteTimestamp(cmd_buf, true);

    { // Resolve spatial cache
        DebugMarker _(ctx_.get(), cmd_buf, "ResolveSpatialCache");

        const camera_t &cam = s.cams_[s.current_cam()._index];

        cache_grid_params_t params;
        memcpy(params.cam_pos_curr, cam.origin, 3 * sizeof(float));
        memcpy(params.cam_pos_prev, s.spatial_cache_cam_pos_prev_, 3 * sizeof(float));

        kernel_SpatialCacheResolve(cmd_buf, params, s.spatial_cache_entries_.buf(), s.spatial_cache_voxels_curr_.buf(),
                                   s.spatial_cache_voxels_prev_.buf());

        std::swap(s.spatial_cache_voxels_prev_, s.spatial_cache_voxels_curr_);
        s.spatial_cache_voxels_curr_.buf().Fill(0, s.spatial_cache_voxels_curr_.buf().size(), 0, cmd_buf);

        // Store previous camera position
        memcpy(s.spatial_cache_cam_pos_prev_, cam.origin, 3 * sizeof(float));
    }

    timestamps_[ctx_->backend_frame].cache_resolve[1] = ctx_->WriteTimestamp(cmd_buf, false);

    resolve_cache_marker.End();

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
        const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};

        if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = wait_semaphores;
            submit_info.pWaitDstStageMask = wait_stages;
        }

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

        const VkResult res = ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info,
                                                       ctx_->in_flight_fence(ctx_->backend_frame));
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to submit into a queue!");
        }

        ctx_->frame_cpu_synced[ctx_->backend_frame] = false;
        ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
        ctx_->render_finished_semaphore_is_set[prev_frame] = false;

        ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
    }
#endif
}

void Ray::Vk::Renderer::ResetSpatialCache(const SceneBase &scene,
                                          const std::function<void(int, int, ParallelForFunction &&)> &) {
    if (!use_spatial_cache_) {
        return;
    }

    const auto &s = dynamic_cast<const Vk::Scene &>(scene);

    CommandBuffer cmd_buf = {};
    if (external_cmd_buf_.vk_cmd_buf) {
        ctx_->backend_frame = external_cmd_buf_.index;
        cmd_buf = external_cmd_buf_.vk_cmd_buf;
    }

#if !RUN_IN_LOCKSTEP
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE,
                                    UINT64_MAX);
        ctx_->api().vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
    }
#endif

    ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
    ctx_->DestroyDeferredResources(ctx_->backend_frame);
    ctx_->default_descr_alloc()->Reset();

#if RUN_IN_LOCKSTEP
    cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
#else
    if (!cmd_buf) {
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
        ctx_->api().vkBeginCommandBuffer(cmd_buf, &begin_info);
    }
#endif

    { // Reset spatial cache
        DebugMarker _(ctx_.get(), cmd_buf, "Ray::ResetSpatialCache");
        s.spatial_cache_voxels_prev_.buf().Fill(0, s.spatial_cache_voxels_prev_.buf().size(), 0, cmd_buf);
    }

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
    if (!external_cmd_buf_.vk_cmd_buf) {
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
        const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};

        if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = wait_semaphores;
            submit_info.pWaitDstStageMask = wait_stages;
        }

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

        const VkResult res = ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info,
                                                       ctx_->in_flight_fence(ctx_->backend_frame));
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to submit into a queue!");
        }

        ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
        ctx_->render_finished_semaphore_is_set[prev_frame] = false;

        ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
    }
#endif
}

Ray::color_data_rgba_t Ray::Vk::Renderer::get_pixels_ref(const bool tonemap) const {
    if (frame_dirty_ || pixel_readback_is_tonemapped_ != tonemap) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        { // download result
            DebugMarker _(ctx_.get(), cmd_buf, "Ray::Download Result");

            // TODO: fix this!
            const auto &buffer_to_use = tonemap ? final_buf_ : raw_filtered_buf_;

            const TransitionInfo res_transitions[] = {{&buffer_to_use, eResState::CopySrc},
                                                      {&pixel_readback_buf_, eResState::CopyDst}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

            CopyImageToBuffer(buffer_to_use, 0, 0, 0, w_, h_, pixel_readback_buf_, cmd_buf, 0);
        }

        InsertReadbackMemoryBarrier(ctx_->api(), cmd_buf);

#if RUN_IN_LOCKSTEP
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        // Wait for all in-flight frames to not leave semaphores in unwaited state
        SmallVector<VkSemaphore, MaxFramesInFlight> wait_semaphores;
        SmallVector<VkPipelineStageFlags, MaxFramesInFlight> wait_stages;
        for (int i = 0; i < MaxFramesInFlight; ++i) {
            const bool is_set = ctx_->render_finished_semaphore_is_set[i];
            if (is_set) {
                wait_semaphores.push_back(ctx_->render_finished_semaphore(i));
                wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        }

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        submit_info.waitSemaphoreCount = uint32_t(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf;

        ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
        ctx_->api().vkQueueWaitIdle(ctx_->graphics_queue());

        ctx_->api().vkFreeCommandBuffers(ctx_->device(), ctx_->temp_command_pool(), 1, &cmd_buf);
#endif
        // Can be reset after vkQueueWaitIdle
        for (bool &is_set : ctx_->render_finished_semaphore_is_set) {
            is_set = false;
        }

        frame_dirty_ = false;
        pixel_readback_is_tonemapped_ = tonemap;
    }

    return {frame_pixels_, w_};
}

Ray::color_data_rgba_t Ray::Vk::Renderer::get_aux_pixels_ref(const eAUXBuffer buf) const {
    bool &dirty_flag = (buf == eAUXBuffer::BaseColor) ? base_color_dirty_ : depth_normals_dirty_;

    const auto &buffer_to_use = (buf == eAUXBuffer::BaseColor) ? base_color_buf_ : depth_normals_buf_;
    const auto &readback_buffer_to_use =
        (buf == eAUXBuffer::BaseColor) ? base_color_readback_buf_ : depth_normals_readback_buf_;

    if (dirty_flag) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        { // download result
            DebugMarker _(ctx_.get(), cmd_buf, "Ray::Download Result");

            const TransitionInfo res_transitions[] = {{&buffer_to_use, eResState::CopySrc},
                                                      {&readback_buffer_to_use, eResState::CopyDst}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

            CopyImageToBuffer(buffer_to_use, 0, 0, 0, w_, h_, readback_buffer_to_use, cmd_buf, 0);
        }

        InsertReadbackMemoryBarrier(ctx_->api(), cmd_buf);

#if RUN_IN_LOCKSTEP
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
        ctx_->api().vkEndCommandBuffer(cmd_buf);

        // Wait for all in-flight frames to not leave semaphores in unwaited state
        SmallVector<VkSemaphore, MaxFramesInFlight> wait_semaphores;
        SmallVector<VkPipelineStageFlags, MaxFramesInFlight> wait_stages;
        for (int i = 0; i < MaxFramesInFlight; ++i) {
            const bool is_set = ctx_->render_finished_semaphore_is_set[i];
            if (is_set) {
                wait_semaphores.push_back(ctx_->render_finished_semaphore(i));
                wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        }

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        submit_info.waitSemaphoreCount = uint32_t(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf;

        ctx_->api().vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
        ctx_->api().vkQueueWaitIdle(ctx_->graphics_queue());

        ctx_->api().vkFreeCommandBuffers(ctx_->device(), ctx_->temp_command_pool(), 1, &cmd_buf);
#endif
        // Can be reset after vkQueueWaitIdle
        for (bool &is_set : ctx_->render_finished_semaphore_is_set) {
            is_set = false;
        }

        dirty_flag = false;
    }

    return {((buf == eAUXBuffer::BaseColor) ? base_color_pixels_ : depth_normals_pixels_), w_};
}

Ray::GpuImage Ray::Vk::Renderer::get_native_raw_pixels() const {
    return GpuImage{raw_filtered_buf_.handle().img, raw_filtered_buf_.handle().views[0],
                    eGPUResState(raw_filtered_buf_.resource_state)};
}

bool Ray::Vk::Renderer::InitUNetFilterPipelines(
    const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    ILog *log = ctx_->log();

    auto select_shader = [this](Span<const uint8_t> default_shader, Span<const uint8_t> fp16_shader,
                                Span<const uint8_t> coop_shader) {
        return use_fp16_ ? (use_coop_matrix_ ? coop_shader : fp16_shader) : default_shader;
    };

AGAIN:

    SmallVector<std::tuple<Shader &, Program &, Pipeline &, const char *, Span<const uint8_t>, eShaderType, bool>, 32>
        shaders_to_init = {
            {sh_.convolution_Img_9_32, prog_.convolution_Img_9_32, pi_.convolution_Img_9_32, "Convolution Img 9 32",
             select_shader(internal_shaders_output_convolution_Img_9_32_fp32_comp_spv,
                           internal_shaders_output_convolution_Img_9_32_fp16_comp_spv,
                           internal_shaders_output_convolution_Img_9_32_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_32_32_Downsample, prog_.convolution_32_32_Downsample, pi_.convolution_32_32_Downsample,
             "Convolution 32 32 Downsample",
             select_shader(internal_shaders_output_convolution_32_32_Downsample_fp32_comp_spv,
                           internal_shaders_output_convolution_32_32_Downsample_fp16_comp_spv,
                           internal_shaders_output_convolution_32_32_Downsample_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_32_48_Downsample, prog_.convolution_32_48_Downsample, pi_.convolution_32_48_Downsample,
             "Convolution 32 48 Downsample",
             select_shader(internal_shaders_output_convolution_32_48_Downsample_fp32_comp_spv,
                           internal_shaders_output_convolution_32_48_Downsample_fp16_comp_spv,
                           internal_shaders_output_convolution_32_48_Downsample_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_48_64_Downsample, prog_.convolution_48_64_Downsample, pi_.convolution_48_64_Downsample,
             "Convolution 48 64 Downsample",
             select_shader(internal_shaders_output_convolution_48_64_Downsample_fp32_comp_spv,
                           internal_shaders_output_convolution_48_64_Downsample_fp16_comp_spv,
                           internal_shaders_output_convolution_48_64_Downsample_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_64_80_Downsample, prog_.convolution_64_80_Downsample, pi_.convolution_64_80_Downsample,
             "Convolution 64 80 Downsample",
             select_shader(internal_shaders_output_convolution_64_80_Downsample_fp32_comp_spv,
                           internal_shaders_output_convolution_64_80_Downsample_fp16_comp_spv,
                           internal_shaders_output_convolution_64_80_Downsample_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_64_64, prog_.convolution_64_64, pi_.convolution_64_64, "Convolution 64 64",
             select_shader(internal_shaders_output_convolution_64_64_fp32_comp_spv,
                           internal_shaders_output_convolution_64_64_fp16_comp_spv,
                           internal_shaders_output_convolution_64_64_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_64_32, prog_.convolution_64_32, pi_.convolution_64_32, "Convolution 64 32",
             select_shader(internal_shaders_output_convolution_64_32_fp32_comp_spv,
                           internal_shaders_output_convolution_64_32_fp16_comp_spv,
                           internal_shaders_output_convolution_64_32_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_80_96, prog_.convolution_80_96, pi_.convolution_80_96, "Convolution 80 96",
             select_shader(internal_shaders_output_convolution_80_96_fp32_comp_spv,
                           internal_shaders_output_convolution_80_96_fp16_comp_spv,
                           internal_shaders_output_convolution_80_96_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_96_96, prog_.convolution_96_96, pi_.convolution_96_96, "Convolution 96 96",
             select_shader(internal_shaders_output_convolution_96_96_fp32_comp_spv,
                           internal_shaders_output_convolution_96_96_fp16_comp_spv,
                           internal_shaders_output_convolution_96_96_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_112_112, prog_.convolution_112_112, pi_.convolution_112_112, "Convolution 112 112",
             select_shader(internal_shaders_output_convolution_112_112_fp32_comp_spv,
                           internal_shaders_output_convolution_112_112_fp16_comp_spv,
                           internal_shaders_output_convolution_112_112_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_concat_96_64_112, prog_.convolution_concat_96_64_112, pi_.convolution_concat_96_64_112,
             "Convolution Concat 96 64 112",
             select_shader(internal_shaders_output_convolution_concat_96_64_112_fp32_comp_spv,
                           internal_shaders_output_convolution_concat_96_64_112_fp16_comp_spv,
                           internal_shaders_output_convolution_concat_96_64_112_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_concat_112_48_96, prog_.convolution_concat_112_48_96, pi_.convolution_concat_112_48_96,
             "Convolution Concat 112 48 96",
             select_shader(internal_shaders_output_convolution_concat_112_48_96_fp32_comp_spv,
                           internal_shaders_output_convolution_concat_112_48_96_fp16_comp_spv,
                           internal_shaders_output_convolution_concat_112_48_96_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_concat_96_32_64, prog_.convolution_concat_96_32_64, pi_.convolution_concat_96_32_64,
             "Convolution Concat 96 32 64",
             select_shader(internal_shaders_output_convolution_concat_96_32_64_fp32_comp_spv,
                           internal_shaders_output_convolution_concat_96_32_64_fp16_comp_spv,
                           internal_shaders_output_convolution_concat_96_32_64_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_concat_64_9_64, prog_.convolution_concat_64_9_64, pi_.convolution_concat_64_9_64,
             "Convolution Concat 64 9 64",
             select_shader(internal_shaders_output_convolution_concat_64_9_64_fp32_comp_spv,
                           internal_shaders_output_convolution_concat_64_9_64_fp16_comp_spv,
                           internal_shaders_output_convolution_concat_64_9_64_coop_comp_spv),
             eShaderType::Comp, false},
            {sh_.convolution_32_3_img, prog_.convolution_32_3_img, pi_.convolution_32_3_img, "Convolution 32 3 Img",
             select_shader(internal_shaders_output_convolution_32_3_img_fp32_comp_spv,
                           internal_shaders_output_convolution_32_3_img_fp16_comp_spv,
                           internal_shaders_output_convolution_32_3_img_coop_comp_spv),
             eShaderType::Comp, false}};
    parallel_for(0, int(shaders_to_init.size()), [&](const int i) {
        std::get<6>(shaders_to_init[i]) =
            std::get<0>(shaders_to_init[i])
                .Init(std::get<3>(shaders_to_init[i]), ctx_.get(), Inflate(std::get<4>(shaders_to_init[i])),
                      std::get<5>(shaders_to_init[i]), log);
        std::get<1>(shaders_to_init[i]) =
            Program{std::get<3>(shaders_to_init[i]), ctx_.get(), &std::get<0>(shaders_to_init[i]), log};
        std::get<2>(shaders_to_init[i]).Init(ctx_.get(), &std::get<1>(shaders_to_init[i]), log);
    });

    bool result = true;
    for (const auto &sh : shaders_to_init) {
        result &= std::get<6>(sh);
    }

    if (!result && use_coop_matrix_) {
        ctx_->log()->Warning("Failed to initialize UNet pipelines, retrying without cooperative matrix!");
        use_coop_matrix_ = false;
        goto AGAIN;
    }

    return result;
}

void Ray::Vk::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const pass_settings_t &ps,
                                              const scene_data_t &sc_data, const Buffer &rand_seq,
                                              const uint32_t rand_seed, const int iteration, const rect_t &rect,
                                              const uint32_t node_index, const float cam_fwd[3], const float clip_dist,
                                              Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
                                              const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, rand_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_SIZE_SLOT, bindless_tex.tex_sizes);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_.intersect_scene.layout(), 1, 1,
                                            &bindless_tex.descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRIS_BUF_SLOT, sc_data.tris);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRI_INDICES_BUF_SLOT, sc_data.tri_indices);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::NODES_BUF_SLOT, sc_data.nodes);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
    }

    IntersectScene::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.node_index = node_index;
    uniform_params.clip_dist = clip_dist;
    uniform_params.min_transp_depth = ps.min_transp_depth;
    uniform_params.max_transp_depth = ps.max_transp_depth;
    uniform_params.rand_seed = rand_seed;
    uniform_params.iteration = iteration;
    if (cam_fwd) {
        memcpy(&uniform_params.cam_fwd[0], &cam_fwd[0], 3 * sizeof(float));
    }

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + IntersectScene::LOCAL_GROUP_SIZE_X - 1) / IntersectScene::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + IntersectScene::LOCAL_GROUP_SIZE_Y - 1) / IntersectScene::LOCAL_GROUP_SIZE_Y), 1u};

    DispatchCompute(cmd_buf, pi_.intersect_scene, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const pass_settings_t &ps,
                                                     const scene_data_t &sc_data, const Buffer &rand_seq,
                                                     const uint32_t rand_seed, const int iteration, const rect_t &rect,
                                                     const uint32_t node_index, const float cam_fwd[3],
                                                     const float clip_dist, Span<const TextureAtlas> tex_atlases,
                                                     const BindlessTexData &bindless_tex, const Buffer &rays,
                                                     const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, rand_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    assert(bindless_tex.rt_descr_set);
    ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                        pi_.intersect_scene_rtpipe.layout(), 1, 1, &bindless_tex.rt_descr_set, 0,
                                        nullptr);

    IntersectScene::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.node_index = node_index;
    uniform_params.clip_dist = clip_dist;
    uniform_params.min_transp_depth = ps.min_transp_depth;
    uniform_params.max_transp_depth = ps.max_transp_depth;
    uniform_params.rand_seed = rand_seed;
    uniform_params.iteration = iteration;
    if (cam_fwd) {
        memcpy(&uniform_params.cam_fwd[0], &cam_fwd[0], 3 * sizeof(float));
    }

    const uint32_t dims[3] = {uint32_t(rect.w), uint32_t(rect.h), 1u};

    TraceRays(cmd_buf, pi_.intersect_scene_rtpipe, dims, bindings, &uniform_params, sizeof(uniform_params),
              ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const Buffer &indir_args,
                                              const int indir_args_index, const Buffer &counters,
                                              const pass_settings_t &ps, const scene_data_t &sc_data,
                                              const Buffer &rand_seq, const uint32_t rand_seed, const int iteration,
                                              uint32_t node_index, const float cam_fwd[3], const float clip_dist,
                                              Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
                                              const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, rand_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRO, IntersectScene::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_SIZE_SLOT, bindless_tex.tex_sizes);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            pi_.intersect_scene_indirect.layout(), 1, 1, &bindless_tex.descr_set, 0,
                                            nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRIS_BUF_SLOT, sc_data.tris);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRI_INDICES_BUF_SLOT, sc_data.tri_indices);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::NODES_BUF_SLOT, sc_data.nodes);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
    }

    IntersectScene::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.clip_dist = clip_dist;
    uniform_params.min_transp_depth = ps.min_transp_depth;
    uniform_params.max_transp_depth = ps.max_transp_depth;
    uniform_params.rand_seed = rand_seed;
    uniform_params.iteration = iteration;
    if (cam_fwd) {
        memcpy(&uniform_params.cam_fwd[0], &cam_fwd[0], 3 * sizeof(float));
    }

    DispatchComputeIndirect(cmd_buf, pi_.intersect_scene_indirect, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene_RTPipe(
    CommandBuffer cmd_buf, const Buffer &indir_args, const int indir_args_index, const pass_settings_t &ps,
    const scene_data_t &sc_data, const Buffer &rand_seq, const uint32_t rand_seed, const int iteration,
    const uint32_t node_index, const float cam_fwd[3], const float clip_dist, Span<const TextureAtlas> tex_atlases,
    const BindlessTexData &bindless_tex, const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, rand_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    assert(bindless_tex.rt_descr_set);
    ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                        pi_.intersect_scene_indirect_rtpipe.layout(), 1, 1, &bindless_tex.rt_descr_set,
                                        0, nullptr);

    IntersectScene::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.clip_dist = clip_dist;
    uniform_params.min_transp_depth = ps.min_transp_depth;
    uniform_params.max_transp_depth = ps.max_transp_depth;
    uniform_params.rand_seed = rand_seed;
    uniform_params.iteration = iteration;
    if (cam_fwd) {
        memcpy(&uniform_params.cam_fwd[0], &cam_fwd[0], 3 * sizeof(float));
    }

    TraceRaysIndirect(cmd_buf, pi_.intersect_scene_indirect_rtpipe, indir_args,
                      indir_args_index * sizeof(TraceRaysIndirectCommand), bindings, &uniform_params,
                      sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadePrimaryHits(
    CommandBuffer cmd_buf, const pass_settings_t &ps, const eSpatialCacheMode cache_usage, const environment_t &env,
    const Buffer &indir_args, const int indir_args_index, const Buffer &hits, const Buffer &rays,
    const scene_data_t &sc_data, const Buffer &rand_seq, const uint32_t rand_seed, const int iteration,
    const rect_t &rect, Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
    const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays, const Buffer &out_sky_rays,
    const Buffer &inout_counters, const Texture2D &out_base_color, const Texture2D &out_depth_normals) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hits, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&rand_seq, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess},
                                              {&out_rays, eResState::UnorderedAccess},
                                              {&out_sh_rays, eResState::UnorderedAccess},
                                              {&out_sky_rays, eResState::UnorderedAccess},
                                              {&inout_counters, eResState::UnorderedAccess},
                                              {&out_base_color, eResState::UnorderedAccess},
                                              {&out_depth_normals, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBufRO, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBufRO, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBufRO, Shade::LIGHTS_BUF_SLOT, sc_data.lights.gpu_buf()},
                                         {eBindTarget::SBufRO, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBufRO, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBufRO, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBufRO, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBufRO, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBufRO, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBufRO, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBufRO, Shade::RANDOM_SEQ_BUF_SLOT, rand_seq},
                                         {eBindTarget::SBufRO, Shade::LIGHT_CWNODES_BUF_SLOT, sc_data.light_cwnodes},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBufRW, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBufRW, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBufRW, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};
    if (sc_data.env.sky_map_spread_angle > 0.0f && cache_usage != eSpatialCacheMode::Update) {
        bindings.emplace_back(eBindTarget::SBufRW, Shade::OUT_SKY_RAYS_BUF_SLOT, out_sky_rays);
    }
    if (cache_usage == eSpatialCacheMode::Query) {
        bindings.emplace_back(eBindTarget::SBufRO, Shade::CACHE_ENTRIES_BUF_SLOT, sc_data.spatial_cache_entries);
        bindings.emplace_back(eBindTarget::SBufRO, Shade::CACHE_VOXELS_BUF_SLOT, sc_data.spatial_cache_voxels);
    }
    if (out_base_color.ready()) {
        bindings.emplace_back(eBindTarget::Image, Shade::OUT_BASE_COLOR_IMG_SLOT, out_base_color);
    }
    if (out_depth_normals.ready()) {
        bindings.emplace_back(eBindTarget::Image, Shade::OUT_DEPTH_NORMALS_IMG_SLOT, out_depth_normals);
    }

    Shade::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.iteration = iteration;
    uniform_params.li_count = sc_data.li_count;
    uniform_params.env_qtree_levels = sc_data.env_qtree_levels;
    uniform_params.regularize_alpha = ps.regularize_alpha;

    uniform_params.max_ray_depth =
        Ref::pack_ray_depth(ps.max_diff_depth, ps.max_spec_depth, ps.max_refr_depth, ps.max_transp_depth);
    uniform_params.max_total_depth = ps.max_total_depth;
    uniform_params.min_total_depth = ps.min_total_depth;

    uniform_params.rand_seed = rand_seed;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));
    memcpy(&uniform_params.back_col[0], env.back_col, 3 * sizeof(float));
    memcpy(&uniform_params.back_col[3], &env.back_map, sizeof(uint32_t));

    uniform_params.env_map_res = env.env_map_res;
    uniform_params.back_map_res = env.back_map_res;

    uniform_params.env_rotation = env.env_map_rotation;
    uniform_params.back_rotation = env.back_map_rotation;
    uniform_params.env_light_index = sc_data.env.light_index;

    uniform_params.sky_map_spread_angle = env.sky_map_spread_angle;

    uniform_params.limit_direct = (ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX;
    uniform_params.limit_indirect = (ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX;

    memcpy(&uniform_params.cam_pos_and_exposure[0], sc_data.spatial_cache_grid.cam_pos_curr, 3 * sizeof(float));
    uniform_params.cam_pos_and_exposure[3] = sc_data.spatial_cache_grid.exposure;

    Pipeline *pi = &pi_.shade_primary;
    if (cache_usage == eSpatialCacheMode::Update) {
        pi = &pi_.shade_primary_cache_update;
    } else if (cache_usage == eSpatialCacheMode::Query) {
        if (sc_data.env.sky_map_spread_angle > 0.0f) {
            pi = &pi_.shade_primary_cache_query_sky;
        } else {
            pi = &pi_.shade_primary_cache_query;
        }
    } else if (sc_data.env.sky_map_spread_angle > 0.0f) {
        pi = &pi_.shade_primary_sky;
    }

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_SIZE_SLOT, bindless_tex.tex_sizes);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi->layout(), 1, 1,
                                            &bindless_tex.descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, *pi, indir_args, indir_args_index * sizeof(DispatchIndirectCommand), bindings,
                            &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadeSecondaryHits(
    CommandBuffer cmd_buf, const pass_settings_t &ps, const eSpatialCacheMode cache_usage, float clamp_direct,
    const environment_t &env, const Buffer &indir_args, const int indir_args_index, const Buffer &hits,
    const Buffer &rays, const scene_data_t &sc_data, const Buffer &rand_seq, const uint32_t rand_seed,
    const int iteration, Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
    const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays, const Buffer &out_sky_rays,
    const Buffer &inout_counters, const Texture2D &out_depth_normals) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hits, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&rand_seq, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess},
                                              {&out_rays, eResState::UnorderedAccess},
                                              {&out_sh_rays, eResState::UnorderedAccess},
                                              {&out_sky_rays, eResState::UnorderedAccess},
                                              {&inout_counters, eResState::UnorderedAccess},
                                              {&out_depth_normals, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBufRO, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBufRO, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBufRO, Shade::LIGHTS_BUF_SLOT, sc_data.lights.gpu_buf()},
                                         {eBindTarget::SBufRO, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBufRO, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBufRO, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBufRO, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBufRO, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBufRO, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBufRO, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBufRO, Shade::RANDOM_SEQ_BUF_SLOT, rand_seq},
                                         {eBindTarget::SBufRO, Shade::LIGHT_CWNODES_BUF_SLOT, sc_data.light_cwnodes},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBufRW, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBufRW, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBufRW, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};
    if (sc_data.env.sky_map_spread_angle > 0.0f && cache_usage != eSpatialCacheMode::Update) {
        bindings.emplace_back(eBindTarget::SBufRW, Shade::OUT_SKY_RAYS_BUF_SLOT, out_sky_rays);
    }
    if (cache_usage == eSpatialCacheMode::Update) {
        bindings.emplace_back(eBindTarget::Image, Shade::OUT_DEPTH_NORMALS_IMG_SLOT, out_depth_normals);
    } else if (cache_usage == eSpatialCacheMode::Query) {
        bindings.emplace_back(eBindTarget::SBufRO, Shade::CACHE_ENTRIES_BUF_SLOT, sc_data.spatial_cache_entries);
        bindings.emplace_back(eBindTarget::SBufRO, Shade::CACHE_VOXELS_BUF_SLOT, sc_data.spatial_cache_voxels);
    }

    Shade::Params uniform_params = {};
    uniform_params.iteration = iteration;
    uniform_params.li_count = sc_data.li_count;
    uniform_params.env_qtree_levels = sc_data.env_qtree_levels;
    uniform_params.regularize_alpha = ps.regularize_alpha;

    uniform_params.max_ray_depth =
        Ref::pack_ray_depth(ps.max_diff_depth, ps.max_spec_depth, ps.max_refr_depth, ps.max_transp_depth);
    uniform_params.max_total_depth = ps.max_total_depth;
    uniform_params.min_total_depth = ps.min_total_depth;

    uniform_params.rand_seed = rand_seed;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));
    memcpy(&uniform_params.back_col[0], env.back_col, 3 * sizeof(float));
    memcpy(&uniform_params.back_col[3], &env.back_map, sizeof(uint32_t));

    uniform_params.env_map_res = env.env_map_res;
    uniform_params.back_map_res = env.back_map_res;

    uniform_params.env_rotation = env.env_map_rotation;
    uniform_params.back_rotation = env.back_map_rotation;
    uniform_params.env_light_index = sc_data.env.light_index;

    uniform_params.sky_map_spread_angle = env.sky_map_spread_angle;

    uniform_params.limit_direct = (clamp_direct != 0.0f) ? 3.0f * clamp_direct : FLT_MAX;
    uniform_params.limit_indirect = (ps.clamp_indirect != 0.0f) ? 3.0f * ps.clamp_indirect : FLT_MAX;

    memcpy(&uniform_params.cam_pos_and_exposure[0], sc_data.spatial_cache_grid.cam_pos_curr, 3 * sizeof(float));
    uniform_params.cam_pos_and_exposure[3] = sc_data.spatial_cache_grid.exposure;

    Pipeline *pi = &pi_.shade_secondary;
    if (cache_usage == eSpatialCacheMode::Update) {
        pi = &pi_.shade_secondary_cache_update;
    } else if (cache_usage == eSpatialCacheMode::Query) {
        if (sc_data.env.sky_map_spread_angle > 0.0f) {
            pi = &pi_.shade_secondary_cache_query_sky;
        } else {
            pi = &pi_.shade_secondary_cache_query;
        }
    } else if (sc_data.env.sky_map_spread_angle > 0.0f) {
        pi = &pi_.shade_secondary_sky;
    }

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_SIZE_SLOT, bindless_tex.tex_sizes);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi->layout(), 1, 1,
                                            &bindless_tex.descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, *pi, indir_args, indir_args_index * sizeof(DispatchIndirectCommand), bindings,
                            &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectSceneShadow(
    CommandBuffer cmd_buf, const pass_settings_t &ps, const Buffer &indir_args, const int indir_args_index,
    const Buffer &counters, const scene_data_t &sc_data, const Buffer &rand_seq, const uint32_t rand_seed,
    const int iteration, const uint32_t node_index, const float clamp_val, Span<const TextureAtlas> tex_atlases,
    const BindlessTexData &bindless_tex, const Buffer &sh_rays, const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&sh_rays, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectSceneShadow::TRIS_BUF_SLOT, sc_data.tris},
        {eBindTarget::SBufRO, IntersectSceneShadow::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
        {eBindTarget::SBufRO, IntersectSceneShadow::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectSceneShadow::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectSceneShadow::NODES_BUF_SLOT, sc_data.nodes},
        {eBindTarget::SBufRO, IntersectSceneShadow::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
        {eBindTarget::SBufRO, IntersectSceneShadow::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectSceneShadow::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectSceneShadow::SH_RAYS_BUF_SLOT, sh_rays},
        {eBindTarget::SBufRO, IntersectSceneShadow::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRO, IntersectSceneShadow::LIGHTS_BUF_SLOT, sc_data.lights.gpu_buf()},
        {eBindTarget::SBufRO, IntersectSceneShadow::LIGHT_CWNODES_BUF_SLOT, sc_data.light_cwnodes},
        {eBindTarget::SBufRO, IntersectSceneShadow::RANDOM_SEQ_BUF_SLOT, rand_seq},
        {eBindTarget::Image, IntersectSceneShadow::INOUT_IMG_SLOT, out_img}};

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectSceneShadow::TLAS_SLOT, sc_data.rt_tlas);
    }

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_SIZE_SLOT, bindless_tex.tex_sizes);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            pi_.intersect_scene_shadow.layout(), 1, 1, &bindless_tex.descr_set, 0,
                                            nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    IntersectSceneShadow::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.max_transp_depth = ps.max_transp_depth;
    uniform_params.lights_node_index = 0; // tree root
    uniform_params.blocker_lights_count = sc_data.blocker_lights_count;
    uniform_params.clamp_val = (clamp_val != 0.0f) ? 3.0f * clamp_val : FLT_MAX;
    uniform_params.rand_seed = rand_seed;
    uniform_params.iteration = iteration;

    DispatchComputeIndirect(cmd_buf, pi_.intersect_scene_shadow, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_DebugRT(CommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index,
                                       const Buffer &rays, const Texture2D &out_pixels) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_pixels, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, DebugRT::TRIS_BUF_SLOT, sc_data.tris},
                                {eBindTarget::SBufRO, DebugRT::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                {eBindTarget::SBufRO, DebugRT::NODES_BUF_SLOT, sc_data.nodes},
                                {eBindTarget::SBufRO, DebugRT::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                {eBindTarget::AccStruct, DebugRT::TLAS_SLOT, sc_data.rt_tlas},
                                {eBindTarget::SBufRO, DebugRT::RAYS_BUF_SLOT, rays},
                                {eBindTarget::Image, DebugRT::OUT_IMG_SLOT, out_pixels}};

    const uint32_t grp_count[3] = {uint32_t((w_ + DebugRT::LOCAL_GROUP_SIZE_X - 1) / DebugRT::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + DebugRT::LOCAL_GROUP_SIZE_Y - 1) / DebugRT::LOCAL_GROUP_SIZE_Y), 1u};

    DebugRT::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;

    DispatchCompute(cmd_buf, pi_.debug_rt, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

bool Ray::Vk::Renderer::InitPipelines(ILog *log,
                                      const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    auto select_shader = [this](Span<const uint8_t> bindless_shader, Span<const uint8_t> bindless_subgroup_shader,
                                Span<const uint8_t> atlas_shader, Span<const uint8_t> atlas_subgroup_shader) {
        return use_bindless_ ? (use_subgroup_ ? bindless_subgroup_shader : bindless_shader)
                             : (use_subgroup_ ? atlas_subgroup_shader : atlas_shader);
    };

    SmallVector<std::tuple<Shader &, const char *, Span<const uint8_t>, eShaderType, bool>, 32> shaders_to_init = {
        {sh_.prim_rays_gen_simple, "Primary Raygen Simple", internal_shaders_output_primary_ray_gen_simple_comp_spv,
         eShaderType::Comp, false},
        {sh_.prim_rays_gen_adaptive, "Primary Raygen Adaptive",
         internal_shaders_output_primary_ray_gen_adaptive_comp_spv, eShaderType::Comp, false},
        {sh_.intersect_area_lights, "Intersect Area Lights", internal_shaders_output_intersect_area_lights_comp_spv,
         eShaderType::Comp, false},
        {sh_.shade_primary, "Shade (Primary)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_primary_bindless_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_primary_atlas_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_primary_sky, "Shade (Primary) (Sky)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_primary_bindless_sky_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_primary_atlas_sky_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_primary_cache_update, "Shade (Primary) (Cache Update)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_primary_bindless_cache_update_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_primary_atlas_cache_update_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_primary_cache_query, "Shade (Primary) (Cache Query)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_primary_bindless_cache_query_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_primary_atlas_cache_query_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_primary_cache_query_sky, "Shade (Primary) (Cache Query) (Sky)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_primary_bindless_cache_query_sky_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_primary_atlas_cache_query_sky_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_secondary, "Shade (Secondary)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_secondary_bindless_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_secondary_atlas_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_secondary_sky, "Shade (Secondary) (Sky)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_secondary_bindless_sky_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_secondary_atlas_sky_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_secondary_cache_update, "Shade (Secondary) (Cache Update)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_secondary_bindless_cache_update_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_secondary_atlas_cache_update_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_secondary_cache_query, "Shade (Secondary) (Cache Query)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_secondary_bindless_cache_query_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_secondary_atlas_cache_query_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_secondary_cache_query_sky, "Shade (Secondary) (Cache Query) (Sky)",
         use_bindless_ ? Span<const uint8_t>{internal_shaders_output_shade_secondary_bindless_cache_query_sky_comp_spv}
                       : Span<const uint8_t>{internal_shaders_output_shade_secondary_atlas_cache_query_sky_comp_spv},
         eShaderType::Comp, false},
        {sh_.shade_sky, "Shade Sky", internal_shaders_output_shade_sky_comp_spv, eShaderType::Comp, false},
        {sh_.prepare_indir_args, "Prepare Indir Args", internal_shaders_output_prepare_indir_args_comp_spv,
         eShaderType::Comp, false},
        {sh_.mix_incremental, "Mix Incremental", internal_shaders_output_mix_incremental_comp_spv, eShaderType::Comp,
         false},
        {sh_.postprocess, "Postprocess", internal_shaders_output_postprocess_comp_spv, eShaderType::Comp, false},
        {sh_.filter_variance, "Filter Variance", internal_shaders_output_filter_variance_comp_spv, eShaderType::Comp,
         false},
        {sh_.nlm_filter, "NLM Filter", internal_shaders_output_nlm_filter_comp_spv, eShaderType::Comp, false},
        {sh_.sort_hash_rays, "Sort Hash Rays", internal_shaders_output_sort_hash_rays_comp_spv, eShaderType::Comp,
         false},
        {sh_.sort_init_count_table, "Sort Init Count Table", internal_shaders_output_sort_init_count_table_comp_spv,
         eShaderType::Comp, false},
        {sh_.sort_reduce, "Sort Reduce", internal_shaders_output_sort_reduce_comp_spv, eShaderType::Comp, false},
        {sh_.sort_scan, "Sort Scan", internal_shaders_output_sort_scan_comp_spv, eShaderType::Comp, false},
        {sh_.sort_scan_add, "Sort Scan Add", internal_shaders_output_sort_scan_add_comp_spv, eShaderType::Comp, false},
        {sh_.sort_scatter, "Sort Scatter", internal_shaders_output_sort_scatter_comp_spv, eShaderType::Comp, false},
        {sh_.sort_reorder_rays, "Sort Reorder Rays", internal_shaders_output_sort_reorder_rays_comp_spv,
         eShaderType::Comp, false},
        {sh_.spatial_cache_resolve, "Spatial Cache Resolve", internal_shaders_output_spatial_cache_resolve_comp_spv,
         eShaderType::Comp, false}};
    if (use_hwrt_) {
        shaders_to_init.emplace_back(
            sh_.intersect_scene, "Intersect Scene (Primary) (HWRT)",
            use_bindless_ ? Span<const uint8_t>{internal_shaders_output_intersect_scene_hwrt_bindless_comp_spv}
                          : Span<const uint8_t>{internal_shaders_output_intersect_scene_hwrt_atlas_comp_spv},
            eShaderType::Comp, false);
        shaders_to_init.emplace_back(
            sh_.intersect_scene_indirect, "Intersect Scene (Secondary) (HWRT)",
            use_bindless_ ? Span<const uint8_t>{internal_shaders_output_intersect_scene_indirect_hwrt_bindless_comp_spv}
                          : Span<const uint8_t>{internal_shaders_output_intersect_scene_indirect_hwrt_atlas_comp_spv},
            eShaderType::Comp, false);
        shaders_to_init.emplace_back(
            sh_.intersect_scene_shadow, "Intersect Scene (Shadow) (HWRT)",
            use_bindless_ ? Span<const uint8_t>{internal_shaders_output_intersect_scene_shadow_hwrt_bindless_comp_spv}
                          : Span<const uint8_t>{internal_shaders_output_intersect_scene_shadow_hwrt_atlas_comp_spv},
            eShaderType::Comp, false);
        shaders_to_init.emplace_back(sh_.intersect_scene_rgen, "Intersect Scene RGEN",
                                     internal_shaders_output_intersect_scene_rgen_spv, eShaderType::RayGen, false);
        shaders_to_init.emplace_back(sh_.intersect_scene_indirect_rgen, "Intersect Scene Indirect RGEN",
                                     internal_shaders_output_intersect_scene_indirect_rgen_spv, eShaderType::RayGen,
                                     false);
        shaders_to_init.emplace_back(sh_.intersect_scene_rchit, "Intersect Scene RCHIT",
                                     internal_shaders_output_intersect_scene_rchit_spv, eShaderType::ClosestHit, false);
        shaders_to_init.emplace_back(sh_.intersect_scene_rmiss, "Intersect Scene RMISS",
                                     internal_shaders_output_intersect_scene_rmiss_spv, eShaderType::Miss, false);
        shaders_to_init.emplace_back(sh_.debug_rt, "Debug RT", internal_shaders_output_debug_rt_comp_spv,
                                     eShaderType::Comp, false);
    } else {
        shaders_to_init.emplace_back(
            sh_.intersect_scene, "Intersect Scene (Primary) (SWRT)",
            select_shader(internal_shaders_output_intersect_scene_swrt_bindless_comp_spv,
                          internal_shaders_output_intersect_scene_swrt_bindless_subgroup_comp_spv,
                          internal_shaders_output_intersect_scene_swrt_atlas_comp_spv,
                          internal_shaders_output_intersect_scene_swrt_atlas_subgroup_comp_spv),
            eShaderType::Comp, false);
        shaders_to_init.emplace_back(
            sh_.intersect_scene_indirect, "Intersect Scene (Secondary) (SWRT)",
            select_shader(internal_shaders_output_intersect_scene_indirect_swrt_bindless_comp_spv,
                          internal_shaders_output_intersect_scene_indirect_swrt_bindless_subgroup_comp_spv,
                          internal_shaders_output_intersect_scene_indirect_swrt_atlas_comp_spv,
                          internal_shaders_output_intersect_scene_indirect_swrt_atlas_subgroup_comp_spv),
            eShaderType::Comp, false);
        shaders_to_init.emplace_back(
            sh_.intersect_scene_shadow, "Intersect Scene (Shadow) (SWRT)",
            select_shader(internal_shaders_output_intersect_scene_shadow_swrt_bindless_comp_spv,
                          internal_shaders_output_intersect_scene_shadow_swrt_bindless_subgroup_comp_spv,
                          internal_shaders_output_intersect_scene_shadow_swrt_atlas_comp_spv,
                          internal_shaders_output_intersect_scene_shadow_swrt_atlas_subgroup_comp_spv),
            eShaderType::Comp, false);
    }
    if (ctx_->int64_atomics_supported()) {
        shaders_to_init.emplace_back(sh_.spatial_cache_update, "Spatial Cache Update",
                                     internal_shaders_output_spatial_cache_update_comp_spv, eShaderType::Comp, false);
    } else {
        shaders_to_init.emplace_back(sh_.spatial_cache_update, "Spatial Cache Update (Compat)",
                                     internal_shaders_output_spatial_cache_update_compat_comp_spv, eShaderType::Comp,
                                     false);
    }
    parallel_for(0, int(shaders_to_init.size()), [&](const int i) {
        std::get<4>(shaders_to_init[i]) =
            std::get<0>(shaders_to_init[i])
                .Init(std::get<1>(shaders_to_init[i]), ctx_.get(), Inflate(std::get<2>(shaders_to_init[i])),
                      std::get<3>(shaders_to_init[i]), log);
    });

    bool result = true;
    for (const auto &sh : shaders_to_init) {
        result &= std::get<4>(sh);
    }

    prog_.prim_rays_gen_simple = Program{"Primary Raygen Simple", ctx_.get(), &sh_.prim_rays_gen_simple, log};
    prog_.prim_rays_gen_adaptive = Program{"Primary Raygen Adaptive", ctx_.get(), &sh_.prim_rays_gen_adaptive, log};
    prog_.intersect_scene = Program{"Intersect Scene (Primary)", ctx_.get(), &sh_.intersect_scene, log};
    prog_.intersect_scene_indirect =
        Program{"Intersect Scene (Secondary)", ctx_.get(), &sh_.intersect_scene_indirect, log};
    prog_.intersect_area_lights = Program{"Intersect Area Lights", ctx_.get(), &sh_.intersect_area_lights, log};
    prog_.shade_primary = Program{"Shade (Primary)", ctx_.get(), &sh_.shade_primary, log};
    prog_.shade_primary_sky = Program{"Shade (Primary) (Sky)", ctx_.get(), &sh_.shade_primary_sky, log};
    prog_.shade_primary_cache_update =
        Program{"Shade (Primary) (Cache Update)", ctx_.get(), &sh_.shade_primary_cache_update, log};
    prog_.shade_primary_cache_query =
        Program{"Shade (Primary) (Cache Query)", ctx_.get(), &sh_.shade_primary_cache_query, log};
    prog_.shade_primary_cache_query_sky =
        Program{"Shade (Primary) (Cache Query) (Sky)", ctx_.get(), &sh_.shade_primary_cache_query_sky, log};
    prog_.shade_secondary = Program{"Shade (Secondary)", ctx_.get(), &sh_.shade_secondary, log};
    prog_.shade_secondary_sky = Program{"Shade (Secondary) (Sky)", ctx_.get(), &sh_.shade_secondary_sky, log};
    prog_.shade_secondary_cache_update =
        Program{"Shade (Secondary) (Cache Update)", ctx_.get(), &sh_.shade_secondary_cache_update, log};
    prog_.shade_secondary_cache_query =
        Program{"Shade (Secondary) (Cache Query)", ctx_.get(), &sh_.shade_secondary_cache_query, log};
    prog_.shade_secondary_cache_query_sky =
        Program{"Shade (Secondary) (Cache Query) (Sky)", ctx_.get(), &sh_.shade_secondary_cache_query_sky, log};
    prog_.shade_sky = Program{"Shade Sky", ctx_.get(), &sh_.shade_sky, log};
    prog_.intersect_scene_shadow = Program{"Intersect Scene (Shadow)", ctx_.get(), &sh_.intersect_scene_shadow, log};
    prog_.prepare_indir_args = Program{"Prepare Indir Args", ctx_.get(), &sh_.prepare_indir_args, log};
    prog_.mix_incremental = Program{"Mix Incremental", ctx_.get(), &sh_.mix_incremental, log};
    prog_.postprocess = Program{"Postprocess", ctx_.get(), &sh_.postprocess, log};
    prog_.filter_variance = Program{"Filter Variance", ctx_.get(), &sh_.filter_variance, log};
    prog_.nlm_filter = Program{"NLM Filter", ctx_.get(), &sh_.nlm_filter, log};
    prog_.debug_rt = Program{"Debug RT", ctx_.get(), &sh_.debug_rt, log};
    prog_.sort_hash_rays = Program{"Hash Rays", ctx_.get(), &sh_.sort_hash_rays, log};
    prog_.sort_init_count_table = Program{"Init Count Table", ctx_.get(), &sh_.sort_init_count_table, log};
    prog_.sort_reduce = Program{"Sort Reduce", ctx_.get(), &sh_.sort_reduce, log};
    prog_.sort_scan = Program{"Sort Scan", ctx_.get(), &sh_.sort_scan, log};
    prog_.sort_scan_add = Program{"Sort Scan Add", ctx_.get(), &sh_.sort_scan_add, log};
    prog_.sort_scatter = Program{"Sort Scatter", ctx_.get(), &sh_.sort_scatter, log};
    prog_.sort_reorder_rays = Program{"Reorder Rays", ctx_.get(), &sh_.sort_reorder_rays, log};
    prog_.intersect_scene_rtpipe = Program{"Intersect Scene",
                                           ctx_.get(),
                                           &sh_.intersect_scene_rgen,
                                           &sh_.intersect_scene_rchit,
                                           nullptr,
                                           &sh_.intersect_scene_rmiss,
                                           nullptr,
                                           log};
    prog_.intersect_scene_indirect_rtpipe = Program{"Intersect Scene Indirect",
                                                    ctx_.get(),
                                                    &sh_.intersect_scene_indirect_rgen,
                                                    &sh_.intersect_scene_rchit,
                                                    nullptr,
                                                    &sh_.intersect_scene_rmiss,
                                                    nullptr,
                                                    log};
    prog_.spatial_cache_update = Program{"Spatial Cache Update", ctx_.get(), &sh_.spatial_cache_update, log};
    prog_.spatial_cache_resolve = Program{"Spatial Cache Resolve", ctx_.get(), &sh_.spatial_cache_resolve, log};

    SmallVector<std::tuple<Pipeline &, Program &, bool>, 32> pipelines_to_init = {
        {pi_.prim_rays_gen_simple, prog_.prim_rays_gen_simple, false},
        {pi_.prim_rays_gen_adaptive, prog_.prim_rays_gen_adaptive, false},
        {pi_.intersect_scene, prog_.intersect_scene, false},
        {pi_.intersect_scene_indirect, prog_.intersect_scene_indirect, false},
        {pi_.intersect_area_lights, prog_.intersect_area_lights, false},
        {pi_.shade_primary, prog_.shade_primary, false},
        {pi_.shade_primary_sky, prog_.shade_primary_sky, false},
        {pi_.shade_secondary, prog_.shade_secondary, false},
        {pi_.shade_secondary_sky, prog_.shade_secondary_sky, false},
        {pi_.shade_sky, prog_.shade_sky, false},
        {pi_.intersect_scene_shadow, prog_.intersect_scene_shadow, false},
        {pi_.prepare_indir_args, prog_.prepare_indir_args, false},
        {pi_.mix_incremental, prog_.mix_incremental, false},
        {pi_.postprocess, prog_.postprocess, false},
        {pi_.filter_variance, prog_.filter_variance, false},
        {pi_.nlm_filter, prog_.nlm_filter, false},
        {pi_.sort_hash_rays, prog_.sort_hash_rays, false},
        {pi_.sort_reorder_rays, prog_.sort_reorder_rays, false}};
    if (use_hwrt_) {
        if (ENABLE_RT_PIPELINE) {
            pipelines_to_init.emplace_back(pi_.intersect_scene_rtpipe, prog_.intersect_scene_rtpipe, false);
            pipelines_to_init.emplace_back(pi_.intersect_scene_indirect_rtpipe, prog_.intersect_scene_indirect_rtpipe,
                                           false);
        }
        pipelines_to_init.emplace_back(pi_.debug_rt, prog_.debug_rt, false);
    }
    if (use_subgroup_) {
        pipelines_to_init.emplace_back(pi_.sort_init_count_table, prog_.sort_init_count_table, false);
        pipelines_to_init.emplace_back(pi_.sort_reduce, prog_.sort_reduce, false);
        pipelines_to_init.emplace_back(pi_.sort_scan, prog_.sort_scan, false);
        pipelines_to_init.emplace_back(pi_.sort_scan_add, prog_.sort_scan_add, false);
        pipelines_to_init.emplace_back(pi_.sort_scatter, prog_.sort_scatter, false);
    }
    if (use_spatial_cache_) {
        pipelines_to_init.emplace_back(pi_.shade_primary_cache_update, prog_.shade_primary_cache_update, false);
        pipelines_to_init.emplace_back(pi_.shade_primary_cache_query, prog_.shade_primary_cache_query, false);
        pipelines_to_init.emplace_back(pi_.shade_primary_cache_query_sky, prog_.shade_primary_cache_query_sky, false);

        pipelines_to_init.emplace_back(pi_.shade_secondary_cache_update, prog_.shade_secondary_cache_update, false);
        pipelines_to_init.emplace_back(pi_.shade_secondary_cache_query, prog_.shade_secondary_cache_query, false);
        pipelines_to_init.emplace_back(pi_.shade_secondary_cache_query_sky, prog_.shade_secondary_cache_query_sky,
                                       false);

        pipelines_to_init.emplace_back(pi_.spatial_cache_update, prog_.spatial_cache_update, false);
        pipelines_to_init.emplace_back(pi_.spatial_cache_resolve, prog_.spatial_cache_resolve, false);
    }
    parallel_for(0, int(pipelines_to_init.size()), [&](const int i) {
        std::get<2>(pipelines_to_init[i]) =
            std::get<0>(pipelines_to_init[i]).Init(ctx_.get(), &std::get<1>(pipelines_to_init[i]), log);
    });

    // Release shader modules
    sh_ = {};

    for (const auto &pi : pipelines_to_init) {
        result &= std::get<2>(pi);
    }

    return result;
}

Ray::RendererBase *Ray::Vk::CreateRenderer(const settings_t &s, ILog *log,
                                           const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    return new Vk::Renderer(s, log, parallel_for);
}
