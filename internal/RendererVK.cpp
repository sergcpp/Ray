#include "RendererVK.h"

#include <functional>
#include <random>
#include <utility>

#include "Halton.h"
#include "SceneVK.h"
#include "UniformIntDistribution.h"

#include "Vk/DebugMarker.h"
#include "Vk/Shader.h"

#include "../Log.h"

#include "shaders/types.h"

#define DEBUG_HWRT 0
#define RUN_IN_LOCKSTEP 0

static_assert(sizeof(Types::tri_accel_t) == sizeof(Ray::tri_accel_t), "!");
static_assert(sizeof(Types::bvh_node_t) == sizeof(Ray::bvh_node_t), "!");
static_assert(sizeof(Types::vertex_t) == sizeof(Ray::vertex_t), "!");
static_assert(sizeof(Types::mesh_t) == sizeof(Ray::mesh_t), "!");
static_assert(sizeof(Types::transform_t) == sizeof(Ray::transform_t), "!");
static_assert(sizeof(Types::mesh_instance_t) == sizeof(Ray::mesh_instance_t), "!");
static_assert(sizeof(Types::light_t) == sizeof(Ray::light_t), "!");
static_assert(sizeof(Types::material_t) == sizeof(Ray::material_t), "!");
static_assert(sizeof(Types::atlas_texture_t) == sizeof(Ray::atlas_texture_t), "!");

static_assert(Types::LIGHT_TYPE_SPHERE == Ray::LIGHT_TYPE_SPHERE, "!");
static_assert(Types::LIGHT_TYPE_SPOT == Ray::LIGHT_TYPE_SPOT, "!");
static_assert(Types::LIGHT_TYPE_DIR == Ray::LIGHT_TYPE_DIR, "!");
static_assert(Types::LIGHT_TYPE_LINE == Ray::LIGHT_TYPE_LINE, "!");
static_assert(Types::LIGHT_TYPE_RECT == Ray::LIGHT_TYPE_RECT, "!");
static_assert(Types::LIGHT_TYPE_DISK == Ray::LIGHT_TYPE_DISK, "!");
static_assert(Types::LIGHT_TYPE_TRI == Ray::LIGHT_TYPE_TRI, "!");

namespace Ray {
namespace Vk {
#include "shaders/output/debug_rt.comp.inl"
#include "shaders/output/intersect_area_lights.comp.inl"
#include "shaders/output/intersect_scene_primary_hwrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_primary_hwrt_bindless.comp.inl"
#include "shaders/output/intersect_scene_primary_swrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_primary_swrt_bindless.comp.inl"
#include "shaders/output/intersect_scene_secondary_hwrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_secondary_hwrt_bindless.comp.inl"
#include "shaders/output/intersect_scene_secondary_swrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_secondary_swrt_bindless.comp.inl"
#include "shaders/output/intersect_scene_shadow_hwrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_shadow_hwrt_bindless.comp.inl"
#include "shaders/output/intersect_scene_shadow_swrt_atlas.comp.inl"
#include "shaders/output/intersect_scene_shadow_swrt_bindless.comp.inl"
#include "shaders/output/mix_incremental.comp.inl"
#include "shaders/output/postprocess.comp.inl"
#include "shaders/output/prepare_indir_args.comp.inl"
#include "shaders/output/primary_ray_gen.comp.inl"
#include "shaders/output/shade_primary_atlas.comp.inl"
#include "shaders/output/shade_primary_bindless.comp.inl"
#include "shaders/output/shade_secondary_atlas.comp.inl"
#include "shaders/output/shade_secondary_bindless.comp.inl"
} // namespace Vk
} // namespace Ray

Ray::Vk::Renderer::Renderer(const settings_t &s, ILog *log) : loaded_halton_(-1) {
    ctx_.reset(new Context);
    const bool res = ctx_->Init(log, s.preferred_device);
    if (!res) {
        throw std::runtime_error("Error initializings vulkan context!");
    }

    use_hwrt_ = (s.use_hwrt && ctx_->ray_query_supported());
    use_bindless_ = s.use_bindless && ctx_->max_combined_image_samplers() >= 16384u;
    use_tex_compression_ = s.use_tex_compression;
    log->Info("HWRT        is %s", use_hwrt_ ? "enabled" : "disabled");
    log->Info("Bindless    is %s", use_bindless_ ? "enabled" : "disabled");
    log->Info("Compression is %s", use_tex_compression_ ? "enabled" : "disabled");

    sh_prim_rays_gen_ = Shader{"Primary Raygen",
                               ctx_.get(),
                               internal_shaders_output_primary_ray_gen_comp_spv,
                               internal_shaders_output_primary_ray_gen_comp_spv_size,
                               eShaderType::Comp,
                               log};
    if (use_hwrt_) {
        sh_intersect_scene_primary_ =
            Shader{"Intersect Scene (Primary) (HWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_primary_hwrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_primary_hwrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_primary_hwrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_primary_hwrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    } else {
        sh_intersect_scene_primary_ =
            Shader{"Intersect Scene (Primary) (SWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_primary_swrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_primary_swrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_primary_swrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_primary_swrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    }

    if (use_hwrt_) {
        sh_intersect_scene_secondary_ =
            Shader{"Intersect Scene (Secondary) (HWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_secondary_hwrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_secondary_hwrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_secondary_hwrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_secondary_hwrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    } else {
        sh_intersect_scene_secondary_ =
            Shader{"Intersect Scene (Secondary) (SWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_secondary_swrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_secondary_swrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_secondary_swrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_secondary_swrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    }

    sh_intersect_area_lights_ = Shader{"Intersect Area Lights",
                                       ctx_.get(),
                                       internal_shaders_output_intersect_area_lights_comp_spv,
                                       internal_shaders_output_intersect_area_lights_comp_spv_size,
                                       eShaderType::Comp,
                                       log};
    sh_shade_primary_ = Shader{"Shade (Primary)",
                               ctx_.get(),
                               use_bindless_ ? internal_shaders_output_shade_primary_bindless_comp_spv
                                             : internal_shaders_output_shade_primary_atlas_comp_spv,
                               use_bindless_ ? int(internal_shaders_output_shade_primary_bindless_comp_spv_size)
                                             : int(internal_shaders_output_shade_primary_atlas_comp_spv_size),
                               eShaderType::Comp,
                               log};
    sh_shade_secondary_ = Shader{"Shade (Secondary)",
                                 ctx_.get(),
                                 use_bindless_ ? internal_shaders_output_shade_secondary_bindless_comp_spv
                                               : internal_shaders_output_shade_secondary_atlas_comp_spv,
                                 use_bindless_ ? int(internal_shaders_output_shade_secondary_bindless_comp_spv_size)
                                               : int(internal_shaders_output_shade_secondary_atlas_comp_spv_size),
                                 eShaderType::Comp,
                                 log};

    if (use_hwrt_) {
        sh_intersect_scene_shadow_ =
            Shader{"Intersect Scene (Shadow) (HWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_shadow_hwrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_shadow_hwrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_shadow_hwrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_shadow_hwrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    } else {
        sh_intersect_scene_shadow_ =
            Shader{"Intersect Scene (Shadow) (SWRT)",
                   ctx_.get(),
                   use_bindless_ ? internal_shaders_output_intersect_scene_shadow_swrt_bindless_comp_spv
                                 : internal_shaders_output_intersect_scene_shadow_swrt_atlas_comp_spv,
                   use_bindless_ ? int(internal_shaders_output_intersect_scene_shadow_swrt_bindless_comp_spv_size)
                                 : int(internal_shaders_output_intersect_scene_shadow_swrt_atlas_comp_spv_size),
                   eShaderType::Comp,
                   log};
    }
    sh_prepare_indir_args_ = Shader{"Prepare Indir Args",
                                    ctx_.get(),
                                    internal_shaders_output_prepare_indir_args_comp_spv,
                                    internal_shaders_output_prepare_indir_args_comp_spv_size,
                                    eShaderType::Comp,
                                    log};
    sh_mix_incremental_ = Shader{"Mix Incremental",
                                 ctx_.get(),
                                 internal_shaders_output_mix_incremental_comp_spv,
                                 internal_shaders_output_mix_incremental_comp_spv_size,
                                 eShaderType::Comp,
                                 log};
    sh_postprocess_ = Shader{"Postprocess",
                             ctx_.get(),
                             internal_shaders_output_postprocess_comp_spv,
                             internal_shaders_output_postprocess_comp_spv_size,
                             eShaderType::Comp,
                             log};
    if (use_hwrt_) {
        sh_debug_rt_ = Shader{"Debug RT",
                              ctx_.get(),
                              internal_shaders_output_debug_rt_comp_spv,
                              internal_shaders_output_debug_rt_comp_spv_size,
                              eShaderType::Comp,
                              log};
    }

    prog_prim_rays_gen_ = Program{"Primary Raygen", ctx_.get(), &sh_prim_rays_gen_, log};
    prog_intersect_scene_primary_ = Program{"Intersect Scene (Primary)", ctx_.get(), &sh_intersect_scene_primary_, log};
    prog_intersect_scene_secondary_ =
        Program{"Intersect Scene (Secondary)", ctx_.get(), &sh_intersect_scene_secondary_, log};
    prog_intersect_area_lights_ = Program{"Intersect Area Lights", ctx_.get(), &sh_intersect_area_lights_, log};
    prog_shade_primary_ = Program{"Shade (Primary)", ctx_.get(), &sh_shade_primary_, log};
    prog_shade_secondary_ = Program{"Shade (Secondary)", ctx_.get(), &sh_shade_secondary_, log};
    prog_intersect_scene_shadow_ = Program{"Intersect Scene (Shadow)", ctx_.get(), &sh_intersect_scene_shadow_, log};
    prog_prepare_indir_args_ = Program{"Prepare Indir Args", ctx_.get(), &sh_prepare_indir_args_, log};
    prog_mix_incremental_ = Program{"Mix Incremental", ctx_.get(), &sh_mix_incremental_, log};
    prog_postprocess_ = Program{"Postprocess", ctx_.get(), &sh_postprocess_, log};
    prog_debug_rt_ = Program{"Debug RT", ctx_.get(), &sh_debug_rt_, log};

    if (!pi_prim_rays_gen_.Init(ctx_.get(), &prog_prim_rays_gen_, log) ||
        !pi_intersect_scene_primary_.Init(ctx_.get(), &prog_intersect_scene_primary_, log) ||
        !pi_intersect_scene_secondary_.Init(ctx_.get(), &prog_intersect_scene_secondary_, log) ||
        !pi_intersect_area_lights_.Init(ctx_.get(), &prog_intersect_area_lights_, log) ||
        !pi_shade_primary_.Init(ctx_.get(), &prog_shade_primary_, log) ||
        !pi_shade_secondary_.Init(ctx_.get(), &prog_shade_secondary_, log) ||
        !pi_intersect_scene_shadow_.Init(ctx_.get(), &prog_intersect_scene_shadow_, log) ||
        !pi_prepare_indir_args_.Init(ctx_.get(), &prog_prepare_indir_args_, log) ||
        !pi_mix_incremental_.Init(ctx_.get(), &prog_mix_incremental_, log) ||
        !pi_postprocess_.Init(ctx_.get(), &prog_postprocess_, log) ||
        (use_hwrt_ && !pi_debug_rt_.Init(ctx_.get(), &prog_debug_rt_, log))) {
        throw std::runtime_error("Error initializing pipeline!");
    }

    halton_seq_buf_ =
        Buffer{"Halton Seq", ctx_.get(), eBufType::Storage, sizeof(float) * HALTON_COUNT * HALTON_SEQ_LEN};
    counters_buf_ = Buffer{"Counters", ctx_.get(), eBufType::Storage, sizeof(uint32_t) * 4};
    indir_args_buf_ = Buffer{"Indir Args", ctx_.get(), eBufType::Indirect, 2 * sizeof(DispatchIndirectCommand)};

    { // zero out counters
        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        const uint32_t zeros[4] = {};
        counters_buf_.UpdateImmediate(0, 4 * sizeof(uint32_t), zeros, cmd_buf);

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    Renderer::Resize(s.w, s.h);

    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

Ray::Vk::Renderer::~Renderer() { pixel_stage_buf_.Unmap(); }

const char *Ray::Vk::Renderer::device_name() const { return ctx_->device_properties().deviceName; }

void Ray::Vk::Renderer::Resize(const int w, const int h) {
    if (w_ == w && h_ == h) {
        return;
    }

    const int num_pixels = w * h;

    Tex2DParams params;
    params.w = w;
    params.h = h;
    params.format = eTexFormat::RawRGBA32F;
    params.usage = eTexUsageBits::Storage | eTexUsageBits::Transfer;

    temp_buf_ = Texture2D{"Temp Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    clean_buf_ = Texture2D{"Clean Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    final_buf_ = Texture2D{"Final Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};

    if (frame_pixels_) {
        pixel_stage_buf_.Unmap();
        frame_pixels_ = nullptr;
    }
    pixel_stage_buf_ = Buffer{"Px Stage Buf", ctx_.get(), eBufType::Stage, uint32_t(4 * w * h * sizeof(float))};
    frame_pixels_ = (const pixel_color_t *)pixel_stage_buf_.Map(BufMapRead, true /* persistent */);

    prim_rays_buf_ =
        Buffer{"Primary Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_data_t) * num_pixels)};
    secondary_rays_buf_ =
        Buffer{"Secondary Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_data_t) * num_pixels)};
    shadow_rays_buf_ =
        Buffer{"Shadow Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::shadow_ray_t) * num_pixels)};
    prim_hits_buf_ =
        Buffer{"Primary Hits", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::hit_data_t) * num_pixels)};

    w_ = w;
    h_ = h;
}

void Ray::Vk::Renderer::Clear(const pixel_color_t &c) {
    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

    const TransitionInfo img_transitions[] = {{&clean_buf_, eResState::CopyDst}, {&final_buf_, eResState::CopyDst}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, img_transitions);

    ClearColorImage(clean_buf_, &c.r, cmd_buf);
    ClearColorImage(final_buf_, &c.r, cmd_buf);

    EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
}

Ray::SceneBase *Ray::Vk::Renderer::CreateScene() {
    return new Vk::Scene(ctx_.get(), use_hwrt_, use_bindless_, use_tex_compression_);
}

void Ray::Vk::Renderer::RenderScene(const SceneBase *_s, RegionContext &region) {
    const auto s = dynamic_cast<const Vk::Scene *>(_s);
    if (!s) {
        return;
    }

    const uint32_t macro_tree_root = s->macro_nodes_start_;

    region.iteration++;
    if (!region.halton_seq || region.iteration % HALTON_SEQ_LEN == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    if (loaded_halton_ == -1 || (region.iteration / HALTON_SEQ_LEN) != (loaded_halton_ / HALTON_SEQ_LEN)) {
        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        Buffer temp_stage_buf{"Temp halton stage", ctx_.get(), eBufType::Stage, halton_seq_buf_.size()};
        { // update stage buffer
            uint8_t *mapped_ptr = temp_stage_buf.Map(BufMapWrite);
            memcpy(mapped_ptr, &region.halton_seq[0], sizeof(float) * HALTON_COUNT * HALTON_SEQ_LEN);
            temp_stage_buf.Unmap();
        }

        const TransitionInfo res_transitions[] = {{&temp_stage_buf, eResState::CopySrc},
                                                  {&halton_seq_buf_, eResState::CopyDst}};
        TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

        CopyBufferToBuffer(temp_stage_buf, 0, halton_seq_buf_, 0, sizeof(float) * HALTON_COUNT * HALTON_SEQ_LEN,
                           cmd_buf);

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        temp_stage_buf.FreeImmediate();
        loaded_halton_ = region.iteration;
    }

    const Ray::camera_t &cam = s->cams_[s->current_cam()].cam;

    scene_data_t sc_data = {&s->env_,
                            s->mesh_instances_.gpu_buf(),
                            s->mi_indices_.buf(),
                            s->meshes_.gpu_buf(),
                            s->transforms_.gpu_buf(),
                            s->vtx_indices_.buf(),
                            s->vertices_.buf(),
                            s->nodes_.buf(),
                            s->tris_.buf(),
                            s->tri_indices_.buf(),
                            s->tri_materials_.buf(),
                            s->materials_.gpu_buf(),
                            s->atlas_textures_.gpu_buf(),
                            s->lights_.gpu_buf(),
                            s->li_indices_.buf(),
                            int(s->li_indices_.size()),
                            s->visible_lights_.buf(),
                            int(s->visible_lights_.size()),
                            s->blocker_lights_.buf(),
                            int(s->blocker_lights_.size()),
                            s->rt_tlas_,
                            s->env_map_qtree_.tex,
                            int(s->env_map_qtree_.mips.size())};

#if !RUN_IN_LOCKSTEP
    vkWaitForFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame), VK_TRUE, UINT64_MAX);
    vkResetFences(ctx_->device(), 1, &ctx_->in_flight_fence(ctx_->backend_frame));
#endif

    ctx_->ReadbackTimestampQueries(ctx_->backend_frame);
    ctx_->DestroyDeferredResources(ctx_->backend_frame);
    ctx_->default_descr_alloc()->Reset();

    stats_.time_primary_ray_gen_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_ray_gen[0], timestamps_[ctx_->backend_frame].primary_ray_gen[1]);
    stats_.time_primary_trace_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_trace[0], timestamps_[ctx_->backend_frame].primary_trace[1]);
    stats_.time_primary_shade_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_shade[0], timestamps_[ctx_->backend_frame].primary_shade[1]);
    stats_.time_primary_shadow_us = ctx_->GetTimestampIntervalDurationUs(
        timestamps_[ctx_->backend_frame].primary_shadow[0], timestamps_[ctx_->backend_frame].primary_shadow[1]);

    stats_.time_secondary_sort_us = 0; // no sorting for now
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
    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());
#else
    VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(ctx_->draw_cmd_buf(ctx_->backend_frame), &begin_info);
    VkCommandBuffer cmd_buf = ctx_->draw_cmd_buf(ctx_->backend_frame);
#endif

    vkCmdResetQueryPool(cmd_buf, ctx_->query_pool(ctx_->backend_frame), 0, MaxTimestampQueries);

    //////////////////////////////////////////////////////////////////////////////////

    const int hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    { // transition resources
        SmallVector<TransitionInfo, 16> res_transitions;

        for (const auto &tex_atlas : s->tex_atlases_) {
            if (tex_atlas.resource_state != eResState::ShaderResource) {
                res_transitions.emplace_back(&tex_atlas, eResState::ShaderResource);
            }
        }

        if (sc_data.mi_indices && sc_data.mi_indices.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.mi_indices, eResState::ShaderResource);
        }
        if (sc_data.meshes && sc_data.meshes.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.meshes, eResState::ShaderResource);
        }
        if (sc_data.transforms && sc_data.transforms.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.transforms, eResState::ShaderResource);
        }
        if (sc_data.vtx_indices && sc_data.vtx_indices.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.vtx_indices, eResState::ShaderResource);
        }
        if (sc_data.vertices && sc_data.vertices.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.vertices, eResState::ShaderResource);
        }
        if (sc_data.nodes && sc_data.nodes.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.nodes, eResState::ShaderResource);
        }
        if (sc_data.tris && sc_data.tris.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.tris, eResState::ShaderResource);
        }
        if (sc_data.tri_indices && sc_data.tri_indices.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.tri_indices, eResState::ShaderResource);
        }
        if (sc_data.tri_materials && sc_data.tri_materials.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.tri_materials, eResState::ShaderResource);
        }
        if (sc_data.materials && sc_data.materials.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.materials, eResState::ShaderResource);
        }
        if (sc_data.atlas_textures && sc_data.atlas_textures.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.atlas_textures, eResState::ShaderResource);
        }
        if (sc_data.lights && sc_data.lights.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.lights, eResState::ShaderResource);
        }
        if (sc_data.li_indices && sc_data.li_indices.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.li_indices, eResState::ShaderResource);
        }
        if (sc_data.visible_lights && sc_data.visible_lights.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.visible_lights, eResState::ShaderResource);
        }
        if (sc_data.env_qtree.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&sc_data.env_qtree, eResState::ShaderResource);
        }

        TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);
    }

    { // generate primary rays
        DebugMarker _(cmd_buf, "GeneratePrimaryRays");
        timestamps_[ctx_->backend_frame].primary_ray_gen[0] = ctx_->WriteTimestamp(true);
        kernel_GeneratePrimaryRays(cmd_buf, cam, hi, region.rect(), halton_seq_buf_, prim_rays_buf_);
        timestamps_[ctx_->backend_frame].primary_ray_gen[1] = ctx_->WriteTimestamp(false);
    }

#if DEBUG_HWRT
    { // debug
        DebugMarker _(cmd_buf, "Debug HWRT");
        kernel_DebugRT(cmd_buf, sc_data, macro_tree_root, prim_rays_buf_, temp_buf_);
    }
#else
    { // trace primary rays
        DebugMarker _(cmd_buf, "IntersectScenePrimary");
        timestamps_[ctx_->backend_frame].primary_trace[0] = ctx_->WriteTimestamp(true);
        kernel_IntersectScenePrimary(cmd_buf, cam.pass_settings, sc_data, halton_seq_buf_, hi + RAND_DIM_BASE_COUNT,
                                     macro_tree_root, cam.clip_end, s->tex_atlases_, s->bindless_tex_data_.descr_set,
                                     prim_rays_buf_, prim_hits_buf_);
        timestamps_[ctx_->backend_frame].primary_trace[1] = ctx_->WriteTimestamp(false);
    }

    { // shade primary hits
        DebugMarker _(cmd_buf, "ShadePrimaryHits");
        timestamps_[ctx_->backend_frame].primary_shade[0] = ctx_->WriteTimestamp(true);
        kernel_ShadePrimaryHits(cmd_buf, cam.pass_settings, s->env_, prim_hits_buf_, prim_rays_buf_, sc_data,
                                halton_seq_buf_, hi + RAND_DIM_BASE_COUNT, s->tex_atlases_,
                                s->bindless_tex_data_.descr_set, temp_buf_, secondary_rays_buf_, shadow_rays_buf_,
                                counters_buf_);
        timestamps_[ctx_->backend_frame].primary_shade[1] = ctx_->WriteTimestamp(false);
    }

    { // prepare indirect args
        DebugMarker _(cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_);
    }

    { // trace shadow rays
        DebugMarker _(cmd_buf, "TraceShadow");
        timestamps_[ctx_->backend_frame].primary_shadow[0] = ctx_->WriteTimestamp(true);
        kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_, counters_buf_, sc_data,
                                    macro_tree_root, s->tex_atlases_, s->bindless_tex_data_.descr_set, shadow_rays_buf_,
                                    temp_buf_);
        timestamps_[ctx_->backend_frame].primary_shadow[1] = ctx_->WriteTimestamp(false);
    }

    timestamps_[ctx_->backend_frame].secondary_trace.clear();
    timestamps_[ctx_->backend_frame].secondary_shade.clear();
    timestamps_[ctx_->backend_frame].secondary_shadow.clear();

    for (int bounce = 1; bounce <= cam.pass_settings.max_total_depth; ++bounce) {
        timestamps_[ctx_->backend_frame].secondary_trace.push_back(ctx_->WriteTimestamp(true));
        { // trace secondary rays
            DebugMarker _(cmd_buf, "IntersectSceneSecondary");
            kernel_IntersectSceneSecondary(cmd_buf, indir_args_buf_, counters_buf_, cam.pass_settings, sc_data,
                                           halton_seq_buf_, hi + RAND_DIM_BASE_COUNT, macro_tree_root, s->tex_atlases_,
                                           s->bindless_tex_data_.descr_set, secondary_rays_buf_, prim_hits_buf_);
        }

        if (sc_data.visible_lights_count) {
            DebugMarker _(cmd_buf, "IntersectAreaLights");
            kernel_IntersectAreaLights(cmd_buf, sc_data, indir_args_buf_, counters_buf_, secondary_rays_buf_,
                                       prim_hits_buf_);
        }

        timestamps_[ctx_->backend_frame].secondary_trace.push_back(ctx_->WriteTimestamp(false));

        { // shade secondary hits
            DebugMarker _(cmd_buf, "ShadeSecondaryHits");
            timestamps_[ctx_->backend_frame].secondary_shade.push_back(ctx_->WriteTimestamp(true));
            kernel_ShadeSecondaryHits(cmd_buf, cam.pass_settings, s->env_, indir_args_buf_, prim_hits_buf_,
                                      secondary_rays_buf_, sc_data, halton_seq_buf_, hi + RAND_DIM_BASE_COUNT,
                                      s->tex_atlases_, s->bindless_tex_data_.descr_set, temp_buf_, prim_rays_buf_,
                                      shadow_rays_buf_, counters_buf_);
            timestamps_[ctx_->backend_frame].secondary_shade.push_back(ctx_->WriteTimestamp(false));
        }

        { // prepare indirect args
            DebugMarker _(cmd_buf, "PrepareIndirArgs");
            kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_);
        }

        { // trace shadow rays
            DebugMarker _(cmd_buf, "TraceShadow");
            timestamps_[ctx_->backend_frame].secondary_shadow.push_back(ctx_->WriteTimestamp(true));
            kernel_IntersectSceneShadow(cmd_buf, cam.pass_settings, indir_args_buf_, counters_buf_, sc_data,
                                        macro_tree_root, s->tex_atlases_, s->bindless_tex_data_.descr_set,
                                        shadow_rays_buf_, temp_buf_);
            timestamps_[ctx_->backend_frame].secondary_shadow.push_back(ctx_->WriteTimestamp(false));
        }

        // std::swap(final_buf_, temp_buf_);
        std::swap(secondary_rays_buf_, prim_rays_buf_);
    }
#endif
    { // prepare result
        DebugMarker _(cmd_buf, "Prepare Result");

        // factor used to compute incremental average
        const float mix_factor = 1.0f / float(region.iteration);

        kernel_MixIncremental(cmd_buf, clean_buf_, temp_buf_, mix_factor, final_buf_);
        std::swap(final_buf_, clean_buf_);

        postprocess_params_.clamp = (cam.pass_settings.flags & Clamp) ? 1 : 0;
        postprocess_params_.srgb = (cam.dtype == SRGB) ? 1 : 0;
        postprocess_params_.exposure = cam.exposure;
        postprocess_params_.gamma = (1.0f / cam.gamma);
    }

#if RUN_IN_LOCKSTEP
    EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
    vkEndCommandBuffer(cmd_buf);

    const int prev_frame = (ctx_->backend_frame + MaxFramesInFlight - 1) % MaxFramesInFlight;

    VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

    const VkSemaphore wait_semaphores[] = {ctx_->render_finished_semaphore(prev_frame)};
    const VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT};

    if (ctx_->render_finished_semaphore_is_set[prev_frame]) {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_semaphores;
        submit_info.pWaitDstStageMask = wait_stages;
    }

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &ctx_->draw_cmd_buf(ctx_->backend_frame);

    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &ctx_->render_finished_semaphore(ctx_->backend_frame);

    const VkResult res =
        vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, ctx_->in_flight_fence(ctx_->backend_frame));
    if (res != VK_SUCCESS) {
        ctx_->log()->Error("Failed to submit into a queue!");
    }

    ctx_->render_finished_semaphore_is_set[ctx_->backend_frame] = true;
    ctx_->render_finished_semaphore_is_set[prev_frame] = false;

    ctx_->backend_frame = (ctx_->backend_frame + 1) % MaxFramesInFlight;
#endif
    frame_dirty_ = true;
}

void Ray::Vk::Renderer::UpdateHaltonSequence(const int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HALTON_COUNT * HALTON_SEQ_LEN]);
    }

    for (int i = 0; i < HALTON_SEQ_LEN; ++i) {
        uint32_t prime_sum = 0;
        for (int j = 0; j < HALTON_COUNT; ++j) {
            seq[i * HALTON_COUNT + j] =
                ScrambledRadicalInverse(g_primes[j], &permutations_[prime_sum], uint64_t(iteration) + i);
            prime_sum += g_primes[j];
        }
    }
}

const Ray::pixel_color_t *Ray::Vk::Renderer::get_pixels_ref() const {
    if (frame_dirty_) {
        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        { // postprocess
            DebugMarker _(cmd_buf, "Postprocess frame");

            kernel_Postprocess(cmd_buf, clean_buf_, postprocess_params_.exposure, postprocess_params_.gamma,
                               postprocess_params_.clamp, postprocess_params_.srgb, final_buf_);
        }

        { // download result
            DebugMarker _(cmd_buf, "Download Result");

            const TransitionInfo res_transitions[] = {{&final_buf_, eResState::CopySrc},
                                                      {&pixel_stage_buf_, eResState::CopyDst}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

            CopyImageToBuffer(final_buf_, 0, 0, 0, w_, h_, pixel_stage_buf_, cmd_buf, 0);
        }

        VkMemoryBarrier mem_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mem_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mem_barrier, 0,
                             nullptr, 0, nullptr);

#if RUN_IN_LOCKSTEP
        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
#else
        vkEndCommandBuffer(cmd_buf);

        // Wait for all in-flight frames to not leave semaphores in unwaited state
        SmallVector<VkSemaphore, MaxFramesInFlight> wait_semaphores;
        SmallVector<VkPipelineStageFlags, MaxFramesInFlight> wait_stages;
        for (int i = 0; i < MaxFramesInFlight; ++i) {
            const bool is_set = ctx_->render_finished_semaphore_is_set[i];
            if (is_set) {
                wait_semaphores.push_back(ctx_->render_finished_semaphore(i));
                wait_stages.push_back(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
            }
        }

        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

        submit_info.waitSemaphoreCount = uint32_t(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf;

        vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx_->graphics_queue());

        vkFreeCommandBuffers(ctx_->device(), ctx_->temp_command_pool(), 1, &cmd_buf);
#endif
        // Can be reset after vkQueueWaitIdle
        for (bool &is_set : ctx_->render_finished_semaphore_is_set) {
            is_set = false;
        }

        pixel_stage_buf_.FlushMappedRange(0, pixel_stage_buf_.size());
        frame_dirty_ = false;
    }

    return frame_pixels_;
}
