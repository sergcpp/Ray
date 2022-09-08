#include "RendererVK.h"

#include <functional>
#include <random>
#include <string>
#include <utility>

#include "Halton.h"
#include "SceneVK.h"
#include "UniformIntDistribution.h"

#include "Vk/DebugMarker.h"
#include "Vk/DrawCall.h"
#include "Vk/Shader.h"

#include "shaders/types.glsl"

#define DEBUG_HWRT 0

static_assert(sizeof(Types::tri_accel_t) == sizeof(Ray::tri_accel_t), "!");
static_assert(sizeof(Types::bvh_node_t) == sizeof(Ray::bvh_node_t), "!");
static_assert(sizeof(Types::vertex_t) == sizeof(Ray::vertex_t), "!");
static_assert(sizeof(Types::mesh_t) == sizeof(Ray::mesh_t), "!");
static_assert(sizeof(Types::transform_t) == sizeof(Ray::transform_t), "!");
static_assert(sizeof(Types::mesh_instance_t) == sizeof(Ray::mesh_instance_t), "!");
static_assert(sizeof(Types::light_t) == sizeof(Ray::light_t), "!");
static_assert(sizeof(Types::material_t) == sizeof(Ray::material_t), "!");
static_assert(sizeof(Types::texture_t) == sizeof(Ray::texture_t), "!");

static_assert(Types::LIGHT_TYPE_SPHERE == Ray::LIGHT_TYPE_SPHERE, "!");
static_assert(Types::LIGHT_TYPE_SPOT == Ray::LIGHT_TYPE_SPOT, "!");
static_assert(Types::LIGHT_TYPE_DIR == Ray::LIGHT_TYPE_DIR, "!");
static_assert(Types::LIGHT_TYPE_LINE == Ray::LIGHT_TYPE_LINE, "!");
static_assert(Types::LIGHT_TYPE_RECT == Ray::LIGHT_TYPE_RECT, "!");
static_assert(Types::LIGHT_TYPE_DISK == Ray::LIGHT_TYPE_DISK, "!");
static_assert(Types::LIGHT_TYPE_TRI == Ray::LIGHT_TYPE_TRI, "!");

namespace Ray {
namespace Vk {
#include "shaders/debug_rt.comp.inl"
#include "shaders/intersect_area_lights.comp.inl"
#include "shaders/mix_incremental.comp.inl"
#include "shaders/postprocess.comp.inl"
#include "shaders/prepare_indir_args.comp.inl"
#include "shaders/primary_ray_gen.comp.inl"
#include "shaders/shade_primary_hits.comp.inl"
#include "shaders/shade_secondary_hits.comp.inl"
#include "shaders/trace_primary_rays_hwrt.comp.inl"
#include "shaders/trace_primary_rays_swrt.comp.inl"
#include "shaders/trace_secondary_rays_hwrt.comp.inl"
#include "shaders/trace_secondary_rays_swrt.comp.inl"
#include "shaders/trace_shadow_hwrt.comp.inl"
#include "shaders/trace_shadow_swrt.comp.inl"
} // namespace Vk
} // namespace Ray

Ray::Vk::Renderer::Renderer(const settings_t &s, ILog *log) : loaded_halton_(-1) {
    ctx_.reset(new Context);
    const bool res = ctx_->Init(log, s.preferred_device);
    if (!res) {
        throw std::runtime_error("Error initializing vulkan context!");
    }

    use_hwrt_ = (s.use_hwrt && ctx_->ray_query_supported());
    log->Info("HWRT is %s", use_hwrt_ ? "enabled" : "disabled");

    sh_prim_rays_gen_ = Shader{"Primary Raygen",
                               ctx_.get(),
                               src_Ray_internal_shaders_primary_ray_gen_comp_spv,
                               src_Ray_internal_shaders_primary_ray_gen_comp_spv_size,
                               eShaderType::Comp,
                               log};
    sh_trace_primary_rays_[0] = Shader{"Trace Primary Rays SWRT",
                                       ctx_.get(),
                                       src_Ray_internal_shaders_trace_primary_rays_swrt_comp_spv,
                                       src_Ray_internal_shaders_trace_primary_rays_swrt_comp_spv_size,
                                       eShaderType::Comp,
                                       log};
    if (use_hwrt_) {
        sh_trace_primary_rays_[1] = Shader{"Trace Primary Rays HWRT",
                                           ctx_.get(),
                                           src_Ray_internal_shaders_trace_primary_rays_hwrt_comp_spv,
                                           src_Ray_internal_shaders_trace_primary_rays_hwrt_comp_spv_size,
                                           eShaderType::Comp,
                                           log};
    }
    sh_trace_secondary_rays_[0] = Shader{"Trace Secondary Rays SWRT",
                                         ctx_.get(),
                                         src_Ray_internal_shaders_trace_secondary_rays_swrt_comp_spv,
                                         src_Ray_internal_shaders_trace_secondary_rays_swrt_comp_spv_size,
                                         eShaderType::Comp,
                                         log};
    if (use_hwrt_) {
        sh_trace_secondary_rays_[1] = Shader{"Trace Secondary Rays HWRT",
                                             ctx_.get(),
                                             src_Ray_internal_shaders_trace_secondary_rays_hwrt_comp_spv,
                                             src_Ray_internal_shaders_trace_secondary_rays_hwrt_comp_spv_size,
                                             eShaderType::Comp,
                                             log};
    }
    sh_intersect_area_lights_ = Shader{"Intersect Area Lights",
                                       ctx_.get(),
                                       src_Ray_internal_shaders_intersect_area_lights_comp_spv,
                                       src_Ray_internal_shaders_intersect_area_lights_comp_spv_size,
                                       eShaderType::Comp,
                                       log};
    sh_shade_primary_hits_ = Shader{"Shade Primary Hits",
                                    ctx_.get(),
                                    src_Ray_internal_shaders_shade_primary_hits_comp_spv,
                                    src_Ray_internal_shaders_shade_primary_hits_comp_spv_size,
                                    eShaderType::Comp,
                                    log};
    sh_shade_secondary_hits_ = Shader{"Shade Secondary Hits",
                                      ctx_.get(),
                                      src_Ray_internal_shaders_shade_secondary_hits_comp_spv,
                                      src_Ray_internal_shaders_shade_secondary_hits_comp_spv_size,
                                      eShaderType::Comp,
                                      log};
    sh_trace_shadow_[0] = Shader{"Trace Shadow SWRT",
                                 ctx_.get(),
                                 src_Ray_internal_shaders_trace_shadow_swrt_comp_spv,
                                 src_Ray_internal_shaders_trace_shadow_swrt_comp_spv_size,
                                 eShaderType::Comp,
                                 log};
    if (use_hwrt_) {
        sh_trace_shadow_[1] = Shader{"Trace Shadow HWRT",
                                     ctx_.get(),
                                     src_Ray_internal_shaders_trace_shadow_hwrt_comp_spv,
                                     src_Ray_internal_shaders_trace_shadow_hwrt_comp_spv_size,
                                     eShaderType::Comp,
                                     log};
    }
    sh_prepare_indir_args_ = Shader{"Prepare Indir Args",
                                    ctx_.get(),
                                    src_Ray_internal_shaders_prepare_indir_args_comp_spv,
                                    src_Ray_internal_shaders_prepare_indir_args_comp_spv_size,
                                    eShaderType::Comp,
                                    log};
    sh_mix_incremental_ = Shader{"Mix Incremental",
                                 ctx_.get(),
                                 src_Ray_internal_shaders_mix_incremental_comp_spv,
                                 src_Ray_internal_shaders_mix_incremental_comp_spv_size,
                                 eShaderType::Comp,
                                 log};
    sh_postprocess_ = Shader{"Postprocess",
                             ctx_.get(),
                             src_Ray_internal_shaders_postprocess_comp_spv,
                             src_Ray_internal_shaders_postprocess_comp_spv_size,
                             eShaderType::Comp,
                             log};
    if (use_hwrt_) {
        sh_debug_rt_ = Shader{"Debug RT",
                              ctx_.get(),
                              src_Ray_internal_shaders_debug_rt_comp_spv,
                              src_Ray_internal_shaders_debug_rt_comp_spv_size,
                              eShaderType::Comp,
                              log};
    }

    prog_prim_rays_gen_ = Program{"Primary Raygen", ctx_.get(), &sh_prim_rays_gen_, log};
    prog_trace_primary_rays_[0] = Program{"Trace Primary Rays SWRT", ctx_.get(), &sh_trace_primary_rays_[0], log};
    prog_trace_primary_rays_[1] = Program{"Trace Primary Rays HWRT", ctx_.get(), &sh_trace_primary_rays_[1], log};
    prog_trace_secondary_rays_[0] = Program{"Trace Secondary Rays SWRT", ctx_.get(), &sh_trace_secondary_rays_[0], log};
    prog_trace_secondary_rays_[1] = Program{"Trace Secondary Rays HWRT", ctx_.get(), &sh_trace_secondary_rays_[1], log};
    prog_intersect_area_lights_ = Program{"Intersect Area Lights", ctx_.get(), &sh_intersect_area_lights_, log};
    prog_shade_primary_hits_ = Program{"Shade Primary Hits", ctx_.get(), &sh_shade_primary_hits_, log};
    prog_shade_secondary_hits_ = Program{"Shade Secondary Hits", ctx_.get(), &sh_shade_secondary_hits_, log};
    prog_trace_shadow_[0] = Program{"Trace Shadow SWRT", ctx_.get(), &sh_trace_shadow_[0], log};
    prog_trace_shadow_[1] = Program{"Trace Shadow HWRT", ctx_.get(), &sh_trace_shadow_[1], log};
    prog_prepare_indir_args_ = Program{"Prepare Indir Args", ctx_.get(), &sh_prepare_indir_args_, log};
    prog_mix_incremental_ = Program{"Mix Incremental", ctx_.get(), &sh_mix_incremental_, log};
    prog_postprocess_ = Program{"Postprocess", ctx_.get(), &sh_postprocess_, log};
    prog_debug_rt_ = Program{"Debug RT", ctx_.get(), &sh_debug_rt_, log};

    if (!pi_prim_rays_gen_.Init(ctx_.get(), &prog_prim_rays_gen_, log) ||
        !pi_trace_primary_rays_[0].Init(ctx_.get(), &prog_trace_primary_rays_[0], log) ||
        (use_hwrt_ && !pi_trace_primary_rays_[1].Init(ctx_.get(), &prog_trace_primary_rays_[1], log)) ||
        !pi_trace_secondary_rays_[0].Init(ctx_.get(), &prog_trace_secondary_rays_[0], log) ||
        (use_hwrt_ && !pi_trace_secondary_rays_[1].Init(ctx_.get(), &prog_trace_secondary_rays_[1], log)) ||
        !pi_intersect_area_lights_.Init(ctx_.get(), &prog_intersect_area_lights_, log) ||
        !pi_shade_primary_hits_.Init(ctx_.get(), &prog_shade_primary_hits_, log) ||
        !pi_shade_secondary_hits_.Init(ctx_.get(), &prog_shade_secondary_hits_, log) ||
        !pi_trace_shadow_[0].Init(ctx_.get(), &prog_trace_shadow_[0], log) ||
        (use_hwrt_ && !pi_trace_shadow_[1].Init(ctx_.get(), &prog_trace_shadow_[1], log)) ||
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
        counters_buf_.UpdateImmediate(0, 4 * sizeof(uint32_t), &zeros, cmd_buf);

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    Resize(s.w, s.h);

    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);

    // throw std::runtime_error("Not implemented yet!");
}

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

    pixel_stage_buf_ = Buffer{"Px Stage Buf", ctx_.get(), eBufType::Stage, uint32_t(4 * w * h * sizeof(float))};

    frame_pixels_.resize(num_pixels);

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

Ray::SceneBase *Ray::Vk::Renderer::CreateScene() { return new Vk::Scene(ctx_.get()); }

void Ray::Vk::Renderer::RenderScene(const SceneBase *_s, RegionContext &region) {
    const auto s = dynamic_cast<const Vk::Scene *>(_s);
    if (!s) {
        return;
    }

    ctx_->DestroyDeferredResources(ctx_->backend_frame);

    const bool reset_result = ctx_->default_descr_alloc()->Reset();
    assert(reset_result);

    //

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
                            s->textures_.gpu_buf(),
                            s->lights_.gpu_buf(),
                            s->li_indices_.buf(),
                            int(s->li_indices_.size()),
                            s->visible_lights_.buf(),
                            int(s->visible_lights_.size()),
                            s->rt_tlas_};

    //

    pass_info_t pass_info;

    pass_info.iteration = region.iteration;
    pass_info.bounce = 0;
    pass_info.settings = cam.pass_settings;
    pass_info.settings.max_total_depth = std::min(pass_info.settings.max_total_depth, uint8_t(MAX_BOUNCES));

    const uint32_t hi = (region.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

    { // transition resources
        SmallVector<TransitionInfo, 16> res_transitions;

        for (int i = 0; i < 4; ++i) {
            if (s->tex_atlases_[i].resource_state != eResState::ShaderResource) {
                res_transitions.emplace_back(&s->tex_atlases_[i], eResState::ShaderResource);
            }
        }

        if (sc_data.mi_indices && sc_data.mi_indices.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.mi_indices, eResState::UnorderedAccess);
        }
        if (sc_data.meshes && sc_data.meshes.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.meshes, eResState::UnorderedAccess);
        }
        if (sc_data.transforms && sc_data.transforms.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.transforms, eResState::UnorderedAccess);
        }
        if (sc_data.vtx_indices && sc_data.vtx_indices.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.vtx_indices, eResState::UnorderedAccess);
        }
        if (sc_data.vertices && sc_data.vertices.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.vertices, eResState::UnorderedAccess);
        }
        if (sc_data.nodes && sc_data.nodes.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.nodes, eResState::UnorderedAccess);
        }
        if (sc_data.tris && sc_data.tris.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.tris, eResState::UnorderedAccess);
        }
        if (sc_data.tri_indices && sc_data.tri_indices.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.tri_indices, eResState::UnorderedAccess);
        }
        if (sc_data.tri_materials && sc_data.tri_materials.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.tri_materials, eResState::UnorderedAccess);
        }
        if (sc_data.materials && sc_data.materials.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.materials, eResState::UnorderedAccess);
        }
        if (sc_data.textures && sc_data.textures.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.textures, eResState::UnorderedAccess);
        }
        if (sc_data.lights && sc_data.lights.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.lights, eResState::UnorderedAccess);
        }
        if (sc_data.li_indices && sc_data.li_indices.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.li_indices, eResState::UnorderedAccess);
        }
        if (sc_data.visible_lights && sc_data.visible_lights.resource_state != eResState::UnorderedAccess) {
            res_transitions.emplace_back(&sc_data.visible_lights, eResState::UnorderedAccess);
        }

        TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);
    }

    { // generate primary rays
        DebugMarker _(cmd_buf, "GeneratePrimaryRays");
        kernel_GeneratePrimaryRays(cmd_buf, cam, hi, region.rect(), halton_seq_buf_, prim_rays_buf_);
    }

#if DEBUG_HWRT
    { // debug
        DebugMarker _(cmd_buf, "Debug HWRT");
        kernel_DebugRT(cmd_buf, sc_data, macro_tree_root, prim_rays_buf_, temp_buf_);
    }
#else
    { // trace primary rays
        DebugMarker _(cmd_buf, "TracePrimaryRays");
        kernel_TracePrimaryRays(cmd_buf, sc_data, macro_tree_root, prim_rays_buf_, prim_hits_buf_);
    }

    { // shade primary hits
        DebugMarker _(cmd_buf, "ShadePrimaryHits");
        kernel_ShadePrimaryHits(cmd_buf, pass_info.settings, s->env_, prim_hits_buf_, prim_rays_buf_, sc_data,
                                halton_seq_buf_, hi + RAND_DIM_BASE_COUNT, s->tex_atlases_, temp_buf_,
                                secondary_rays_buf_, shadow_rays_buf_, counters_buf_);
    }

    { // prepare indirect args
        DebugMarker _(cmd_buf, "PrepareIndirArgs");
        kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_);
    }

    { // trace shadow rays
        DebugMarker _(cmd_buf, "TraceShadow");
        kernel_TraceShadow(cmd_buf, indir_args_buf_, counters_buf_, sc_data, macro_tree_root,
                           region.halton_seq[hi + RAND_DIM_BASE_COUNT + RAND_DIM_BSDF_PICK], s->tex_atlases_,
                           shadow_rays_buf_, temp_buf_);
    }

    for (int bounce = 1;
         bounce <= pass_info.settings.max_total_depth && !(pass_info.settings.flags & SkipIndirectLight); ++bounce) {

        { // trace secondary rays
            DebugMarker _(cmd_buf, "TraceSecondaryRays");
            kernel_TraceSecondaryRays(cmd_buf, indir_args_buf_, counters_buf_, sc_data, macro_tree_root,
                                      secondary_rays_buf_, prim_hits_buf_);
        }

        if (sc_data.visible_lights_count) {
            DebugMarker _(cmd_buf, "IntersectAreaLights");
            kernel_IntersectAreaLights(cmd_buf, sc_data, indir_args_buf_, counters_buf_, secondary_rays_buf_,
                                       prim_hits_buf_);
        }

        { // shade secondary hits
            DebugMarker _(cmd_buf, "ShadeSecondaryHits");
            kernel_ShadeSecondaryHits(cmd_buf, pass_info.settings, s->env_, indir_args_buf_, prim_hits_buf_,
                                      secondary_rays_buf_, sc_data, halton_seq_buf_,
                                      hi + RAND_DIM_BASE_COUNT + bounce * RAND_DIM_BOUNCE_COUNT, s->tex_atlases_,
                                      temp_buf_, prim_rays_buf_, shadow_rays_buf_, counters_buf_);
        }

        { // prepare indirect args
            DebugMarker _(cmd_buf, "PrepareIndirArgs");
            kernel_PrepareIndirArgs(cmd_buf, counters_buf_, indir_args_buf_);
        }

        { // trace shadow rays
            DebugMarker _(cmd_buf, "TraceShadow");
            kernel_TraceShadow(
                cmd_buf, indir_args_buf_, counters_buf_, sc_data, macro_tree_root,
                region.halton_seq[hi + RAND_DIM_BASE_COUNT + bounce * RAND_DIM_BOUNCE_COUNT + RAND_DIM_BSDF_PICK],
                s->tex_atlases_, shadow_rays_buf_, temp_buf_);
        }

        // std::swap(final_buf_, temp_buf_);
        std::swap(secondary_rays_buf_, prim_rays_buf_);
    }
#endif
    { // prepare result
        DebugMarker _(cmd_buf, "Prepare Result");

        // factor used to compute incremental average
        const float mix_factor = 1.0f / region.iteration;

        kernel_MixIncremental(cmd_buf, clean_buf_, temp_buf_, mix_factor, final_buf_);
        std::swap(final_buf_, clean_buf_);

        const int _clamp = (cam.pass_settings.flags & Clamp) ? 1 : 0, _srgb = (cam.dtype == SRGB) ? 1 : 0;
        kernel_Postprocess(cmd_buf, clean_buf_, (1.0f / cam.gamma), _clamp, _srgb, final_buf_);
    }

    { // download result
        DebugMarker _(cmd_buf, "Download Result");

        const TransitionInfo res_transitions[] = {{&final_buf_, eResState::CopySrc},
                                                  {&pixel_stage_buf_, eResState::CopyDst}};
        TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

        final_buf_.CopyTextureData(pixel_stage_buf_, cmd_buf, 0);
    }

    EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    { // copy result
        const uint8_t *pixels = pixel_stage_buf_.Map(BufMapRead);
        memcpy(frame_pixels_.data(), pixels, frame_pixels_.size() * sizeof(pixel_color_t));
        pixel_stage_buf_.Unmap();
    }
}

void Ray::Vk::Renderer::UpdateHaltonSequence(const int iteration, std::unique_ptr<float[]> &seq) {
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
