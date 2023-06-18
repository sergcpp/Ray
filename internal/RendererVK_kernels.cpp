#include "RendererVK.h"

#include "SceneVK.h"
#include "Vk/DrawCallVK.h"

#include "shaders/debug_rt_interface.h"
#include "shaders/filter_variance_interface.h"
#include "shaders/intersect_area_lights_interface.h"
#include "shaders/intersect_scene_interface.h"
#include "shaders/intersect_scene_shadow_interface.h"
#include "shaders/mix_incremental_interface.h"
#include "shaders/nlm_filter_interface.h"
#include "shaders/postprocess_interface.h"
#include "shaders/prepare_indir_args_interface.h"
#include "shaders/primary_ray_gen_interface.h"
#include "shaders/shade_interface.h"
#include "shaders/sort_add_partial_sums_interface.h"
#include "shaders/sort_hash_rays_interface.h"
#include "shaders/sort_init_count_table_interface.h"
#include "shaders/sort_reorder_rays_interface.h"
#include "shaders/sort_scan_interface.h"
#include "shaders/sort_write_sorted_hashes_interface.h"

#include "shaders/types.h"

void Ray::Vk::Renderer::kernel_GeneratePrimaryRays(CommandBuffer cmd_buf, const camera_t &cam, const int hi,
                                                   const rect_t &rect, const Buffer &random_seq, const int iteration,
                                                   const Texture2D &req_samples_img, const Buffer &inout_counters,
                                                   const Buffer &out_rays) {
    const TransitionInfo res_transitions[] = {{&random_seq, eResState::ShaderResource},
                                              {&req_samples_img, eResState::ShaderResource},
                                              {&inout_counters, eResState::UnorderedAccess},
                                              {&out_rays, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, PrimaryRayGen::HALTON_SEQ_BUF_SLOT, random_seq},
                                {eBindTarget::Tex2D, PrimaryRayGen::REQUIRED_SAMPLES_IMG_SLOT, req_samples_img},
                                {eBindTarget::SBufRW, PrimaryRayGen::INOUT_COUNTERS_BUF_SLOT, inout_counters},
                                {eBindTarget::SBufRW, PrimaryRayGen::OUT_RAYS_BUF_SLOT, out_rays}};

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + PrimaryRayGen::LOCAL_GROUP_SIZE_X - 1) / PrimaryRayGen::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + PrimaryRayGen::LOCAL_GROUP_SIZE_Y - 1) / PrimaryRayGen::LOCAL_GROUP_SIZE_Y), 1u};

    PrimaryRayGen::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.hi = hi;

    const float temp = std::tan(0.5f * cam.fov * PI / 180.0f);
    uniform_params.spread_angle = std::atan(2.0f * temp / float(h_));

    memcpy(&uniform_params.cam_origin[0], cam.origin, 3 * sizeof(float));
    uniform_params.cam_origin[3] = temp;
    memcpy(&uniform_params.cam_fwd[0], cam.fwd, 3 * sizeof(float));
    uniform_params.cam_fwd[3] = cam.clip_start;
    memcpy(&uniform_params.cam_side[0], cam.side, 3 * sizeof(float));
    uniform_params.cam_side[3] = cam.focus_distance;
    memcpy(&uniform_params.cam_up[0], cam.up, 3 * sizeof(float));
    uniform_params.cam_up[3] = cam.sensor_height;
    uniform_params.cam_fstop = cam.fstop;
    uniform_params.cam_focal_length = cam.focal_length;
    uniform_params.cam_lens_rotation = cam.lens_rotation;
    uniform_params.cam_lens_ratio = cam.lens_ratio;
    uniform_params.cam_filter_and_lens_blades = (int(cam.filter) << 8) | cam.lens_blades;
    uniform_params.shift_x = cam.shift[0];
    uniform_params.shift_y = cam.shift[1];
    uniform_params.iteration = iteration;

    const bool adaptive = (iteration > cam.pass_settings.min_samples) && cam.pass_settings.min_samples != -1;
    DispatchCompute(cmd_buf, adaptive ? pi_prim_rays_gen_adaptive_ : pi_prim_rays_gen_simple_, grp_count, bindings,
                    &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const pass_settings_t &settings,
                                              const scene_data_t &sc_data, const Buffer &random_seq, const int hi,
                                              const rect_t &rect, const uint32_t node_index, const float inter_t,
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
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_.layout(), 1, 1,
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
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESHES_BUF_SLOT, sc_data.meshes);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MI_INDICES_BUF_SLOT, sc_data.mi_indices);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRANSFORMS_BUF_SLOT, sc_data.transforms);
    }

    IntersectScene::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.node_index = node_index;
    uniform_params.inter_t = inter_t;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + IntersectScene::LOCAL_GROUP_SIZE_X - 1) / IntersectScene::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + IntersectScene::LOCAL_GROUP_SIZE_Y - 1) / IntersectScene::LOCAL_GROUP_SIZE_Y), 1u};

    DispatchCompute(cmd_buf, pi_intersect_scene_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const pass_settings_t &settings,
                                                     const scene_data_t &sc_data, const Buffer &random_seq,
                                                     const int hi, const rect_t &rect, const uint32_t node_index,
                                                     const float inter_t, Span<const TextureAtlas> tex_atlases,
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
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    assert(bindless_tex.rt_descr_set);
    ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                        pi_intersect_scene_rtpipe_.layout(), 1, 1, &bindless_tex.rt_descr_set, 0,
                                        nullptr);

    IntersectScene::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.node_index = node_index;
    uniform_params.inter_t = inter_t;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    const uint32_t dims[3] = {uint32_t(rect.w), uint32_t(rect.h), 1u};

    TraceRays(cmd_buf, pi_intersect_scene_rtpipe_, dims, bindings, &uniform_params, sizeof(uniform_params),
              ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const Buffer &indir_args,
                                              const int indir_args_index, const Buffer &counters,
                                              const pass_settings_t &settings, const scene_data_t &sc_data,
                                              const Buffer &random_seq, const int hi, uint32_t node_index,
                                              const float inter_t, Span<const TextureAtlas> tex_atlases,
                                              const BindlessTexData &bindless_tex, const Buffer &rays,
                                              const Buffer &out_hits) {
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
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRO, IntersectScene::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            pi_intersect_scene_indirect_.layout(), 1, 1, &bindless_tex.descr_set, 0,
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
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESHES_BUF_SLOT, sc_data.meshes);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::MI_INDICES_BUF_SLOT, sc_data.mi_indices);
        bindings.emplace_back(eBindTarget::SBufRO, IntersectScene::TRANSFORMS_BUF_SLOT, sc_data.transforms);
    }

    IntersectScene::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.inter_t = inter_t;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    DispatchComputeIndirect(cmd_buf, pi_intersect_scene_indirect_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const Buffer &indir_args,
                                                     const int indir_args_index, const pass_settings_t &settings,
                                                     const scene_data_t &sc_data, const Buffer &random_seq,
                                                     const int hi, const uint32_t node_index, const float inter_t,
                                                     Span<const TextureAtlas> tex_atlases,
                                                     const BindlessTexData &bindless_tex, const Buffer &rays,
                                                     const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBufRO, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBufRO, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBufRO, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::SBufRW, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas},
        {eBindTarget::SBufRW, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    assert(bindless_tex.rt_descr_set);
    ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                        pi_intersect_scene_indirect_rtpipe_.layout(), 1, 1, &bindless_tex.rt_descr_set,
                                        0, nullptr);

    IntersectScene::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.inter_t = inter_t;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    TraceRaysIndirect(cmd_buf, pi_intersect_scene_indirect_rtpipe_, indir_args,
                      indir_args_index * sizeof(TraceRaysIndirectCommand), bindings, &uniform_params,
                      sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectAreaLights(CommandBuffer cmd_buf, const scene_data_t &sc_data,
                                                   const Buffer &indir_args, const Buffer &counters, const Buffer &rays,
                                                   const Buffer &inout_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&inout_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {
        {eBindTarget::SBufRO, IntersectAreaLights::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRO, IntersectAreaLights::LIGHTS_BUF_SLOT, sc_data.lights},
        {eBindTarget::SBufRO, IntersectAreaLights::VISIBLE_LIGHTS_BUF_SLOT, sc_data.visible_lights},
        {eBindTarget::SBufRO, IntersectAreaLights::TRANSFORMS_BUF_SLOT, sc_data.transforms},
        {eBindTarget::SBufRO, IntersectAreaLights::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRW, IntersectAreaLights::INOUT_HITS_BUF_SLOT, inout_hits}};

    IntersectAreaLights::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.visible_lights_count = sc_data.visible_lights_count;

    DispatchComputeIndirect(cmd_buf, pi_intersect_area_lights_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadePrimaryHits(
    CommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env, const Buffer &indir_args,
    const int indir_args_index, const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
    const Buffer &random_seq, const int hi, const rect_t &rect, Span<const TextureAtlas> tex_atlases,
    const BindlessTexData &bindless_tex, const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
    const Buffer &inout_counters, const Texture2D &out_base_color, const Texture2D &out_depth_normals) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hits, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&random_seq, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess},
                                              {&out_rays, eResState::UnorderedAccess},
                                              {&out_sh_rays, eResState::UnorderedAccess},
                                              {&inout_counters, eResState::UnorderedAccess},
                                              {&out_base_color, eResState::UnorderedAccess},
                                              {&out_depth_normals, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBufRO, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBufRO, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBufRO, Shade::LIGHTS_BUF_SLOT, sc_data.lights},
                                         {eBindTarget::SBufRO, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBufRO, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBufRO, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBufRO, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBufRO, Shade::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                         {eBindTarget::SBufRO, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBufRO, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBufRO, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBufRO, Shade::RANDOM_SEQ_BUF_SLOT, random_seq},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBufRW, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBufRW, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBufRW, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

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
    uniform_params.hi = hi;
    uniform_params.li_count = sc_data.li_count;
    uniform_params.env_qtree_levels = sc_data.env_qtree_levels;

    uniform_params.max_diff_depth = settings.max_diff_depth;
    uniform_params.max_spec_depth = settings.max_spec_depth;
    uniform_params.max_refr_depth = settings.max_refr_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.max_total_depth = settings.max_total_depth;
    uniform_params.min_total_depth = settings.min_total_depth;
    uniform_params.min_transp_depth = settings.min_transp_depth;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));
    memcpy(&uniform_params.back_col[0], env.back_col, 3 * sizeof(float));
    memcpy(&uniform_params.back_col[3], &env.back_map, sizeof(uint32_t));

    uniform_params.env_rotation = env.env_map_rotation;
    uniform_params.back_rotation = env.back_map_rotation;
    uniform_params.env_mult_importance = sc_data.env->multiple_importance ? 1 : 0;

    uniform_params.clamp_val =
        (settings.clamp_direct != 0.0f) ? settings.clamp_direct : std::numeric_limits<float>::max();

    Pipeline *pi = &pi_shade_primary_;
    if (out_base_color.ready()) {
        if (out_depth_normals.ready()) {
            pi = &pi_shade_primary_bn_;
        } else {
            pi = &pi_shade_primary_b_;
        }
    } else if (out_depth_normals.ready()) {
        pi = &pi_shade_primary_n_;
    }

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);

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
    CommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env, const Buffer &indir_args,
    const int indir_args_index, const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
    const Buffer &random_seq, const int hi, Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
    const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays, const Buffer &inout_counters) {
    const TransitionInfo res_transitions[] = {
        {&indir_args, eResState::IndirectArgument}, {&hits, eResState::ShaderResource},
        {&rays, eResState::ShaderResource},         {&random_seq, eResState::ShaderResource},
        {&out_img, eResState::UnorderedAccess},     {&out_rays, eResState::UnorderedAccess},
        {&out_sh_rays, eResState::UnorderedAccess}, {&inout_counters, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBufRO, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBufRO, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBufRO, Shade::LIGHTS_BUF_SLOT, sc_data.lights},
                                         {eBindTarget::SBufRO, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBufRO, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBufRO, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBufRO, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBufRO, Shade::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                         {eBindTarget::SBufRO, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBufRO, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBufRO, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBufRO, Shade::RANDOM_SEQ_BUF_SLOT, random_seq},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBufRW, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBufRW, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBufRW, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

    Shade::Params uniform_params = {};
    uniform_params.hi = hi;
    uniform_params.li_count = sc_data.li_count;
    uniform_params.env_qtree_levels = sc_data.env_qtree_levels;

    uniform_params.max_diff_depth = settings.max_diff_depth;
    uniform_params.max_spec_depth = settings.max_spec_depth;
    uniform_params.max_refr_depth = settings.max_refr_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.max_total_depth = settings.max_total_depth;
    uniform_params.min_total_depth = settings.min_total_depth;
    uniform_params.min_transp_depth = settings.min_transp_depth;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));
    memcpy(&uniform_params.back_col[0], env.back_col, 3 * sizeof(float));
    memcpy(&uniform_params.back_col[3], &env.back_map, sizeof(uint32_t));

    uniform_params.env_rotation = env.env_map_rotation;
    uniform_params.back_rotation = env.back_map_rotation;
    uniform_params.env_mult_importance = sc_data.env->multiple_importance ? 1 : 0;

    uniform_params.clamp_val =
        (settings.clamp_indirect != 0.0f) ? settings.clamp_indirect : std::numeric_limits<float>::max();

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_shade_secondary_.layout(), 1, 1,
                                            &bindless_tex.descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, pi_shade_secondary_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectSceneShadow(CommandBuffer cmd_buf, const pass_settings_t &settings,
                                                    const Buffer &indir_args, const int indir_args_index,
                                                    const Buffer &counters, const scene_data_t &sc_data,
                                                    const Buffer &random_seq, const int hi, const uint32_t node_index,
                                                    const float clamp_val, Span<const TextureAtlas> tex_atlases,
                                                    const BindlessTexData &bindless_tex, const Buffer &sh_rays,
                                                    const Texture2D &out_img) {
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
        {eBindTarget::SBufRO, IntersectSceneShadow::MESHES_BUF_SLOT, sc_data.meshes},
        {eBindTarget::SBufRO, IntersectSceneShadow::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
        {eBindTarget::SBufRO, IntersectSceneShadow::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
        {eBindTarget::SBufRO, IntersectSceneShadow::TRANSFORMS_BUF_SLOT, sc_data.transforms},
        {eBindTarget::SBufRO, IntersectSceneShadow::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBufRO, IntersectSceneShadow::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBufRO, IntersectSceneShadow::SH_RAYS_BUF_SLOT, sh_rays},
        {eBindTarget::SBufRO, IntersectSceneShadow::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRO, IntersectSceneShadow::LIGHTS_BUF_SLOT, sc_data.lights},
        {eBindTarget::SBufRO, IntersectSceneShadow::BLOCKER_LIGHTS_BUF_SLOT, sc_data.blocker_lights},
        {eBindTarget::SBufRO, IntersectSceneShadow::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::Image, IntersectSceneShadow::INOUT_IMG_SLOT, out_img}};

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectSceneShadow::TLAS_SLOT, sc_data.rt_tlas);
    }

    if (use_bindless_) {
        bindings.emplace_back(eBindTarget::Sampler, Types::TEXTURES_SAMPLER_SLOT, bindless_tex.shared_sampler);

        assert(bindless_tex.descr_set);
        ctx_->api().vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            pi_intersect_scene_shadow_.layout(), 1, 1, &bindless_tex.descr_set, 0,
                                            nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    IntersectSceneShadow::Params uniform_params = {};
    uniform_params.node_index = node_index;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.blocker_lights_count = sc_data.blocker_lights_count;
    uniform_params.clamp_val = (clamp_val != 0.0f) ? clamp_val : std::numeric_limits<float>::max();
    uniform_params.hi = hi;

    DispatchComputeIndirect(cmd_buf, pi_intersect_scene_shadow_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_PrepareIndirArgs(CommandBuffer cmd_buf, const Buffer &inout_counters,
                                                const Buffer &out_indir_args) {
    const TransitionInfo res_transitions[] = {{&inout_counters, eResState::UnorderedAccess},
                                              {&out_indir_args, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRW, PrepareIndirArgs::INOUT_COUNTERS_BUF_SLOT, inout_counters},
                                {eBindTarget::SBufRW, PrepareIndirArgs::OUT_INDIR_ARGS_SLOT, out_indir_args}};

    const uint32_t grp_count[3] = {1u, 1u, 1u};
    DispatchCompute(cmd_buf, pi_prepare_indir_args_, grp_count, bindings, nullptr, 0, ctx_->default_descr_alloc(),
                    ctx_->log());
}

void Ray::Vk::Renderer::kernel_MixIncremental(CommandBuffer cmd_buf, const float main_mix_factor,
                                              const float aux_mix_factor, const rect_t &rect, const int iteration,
                                              const Texture2D &temp_img, const Texture2D &temp_base_color,
                                              const Texture2D &temp_depth_normals, const Texture2D &req_samples,
                                              const Texture2D &out_img, const Texture2D &out_base_color,
                                              const Texture2D &out_depth_normals) {
    const TransitionInfo res_transitions[] = {
        {&temp_img, eResState::UnorderedAccess},         {&temp_base_color, eResState::UnorderedAccess},
        {&req_samples, eResState::UnorderedAccess},      {&out_img, eResState::UnorderedAccess},
        {&out_base_color, eResState::UnorderedAccess},   {&temp_depth_normals, eResState::UnorderedAccess},
        {&out_depth_normals, eResState::UnorderedAccess}};
    SmallVector<Binding, 16> bindings = {{eBindTarget::Image, MixIncremental::IN_TEMP_IMG_SLOT, temp_img},
                                         {eBindTarget::Image, MixIncremental::IN_REQ_SAMPLES_SLOT, req_samples},
                                         {eBindTarget::Image, MixIncremental::OUT_IMG_SLOT, out_img}};
    if (out_base_color.ready()) {
        bindings.emplace_back(eBindTarget::Image, MixIncremental::IN_TEMP_BASE_COLOR_SLOT, temp_base_color);
        bindings.emplace_back(eBindTarget::Image, MixIncremental::OUT_BASE_COLOR_IMG_SLOT, out_base_color);
    }
    if (out_depth_normals.ready()) {
        bindings.emplace_back(eBindTarget::Image, MixIncremental::IN_TEMP_DEPTH_NORMALS_SLOT, temp_depth_normals);
        bindings.emplace_back(eBindTarget::Image, MixIncremental::OUT_DEPTH_NORMALS_IMG_SLOT, out_depth_normals);
    }

    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + MixIncremental::LOCAL_GROUP_SIZE_X - 1) / MixIncremental::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + MixIncremental::LOCAL_GROUP_SIZE_Y - 1) / MixIncremental::LOCAL_GROUP_SIZE_Y), 1u};

    MixIncremental::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.main_mix_factor = main_mix_factor;
    uniform_params.aux_mix_factor = aux_mix_factor;
    uniform_params.iteration = iteration;

    Pipeline *pi = &pi_mix_incremental_;
    if (out_base_color.ready()) {
        if (out_depth_normals.ready()) {
            pi = &pi_mix_incremental_bn_;
        } else {
            pi = &pi_mix_incremental_b_;
        }
    } else if (out_depth_normals.ready()) {
        pi = &pi_mix_incremental_n_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_Postprocess(CommandBuffer cmd_buf, const Texture2D &img0_buf, const float img0_weight,
                                           const Texture2D &img1_buf, const float img1_weight, const float exposure,
                                           const float inv_gamma, const rect_t &rect, const float variance_threshold,
                                           const int iteration, const Texture2D &out_pixels,
                                           const Texture2D &out_raw_pixels, const Texture2D &out_variance,
                                           const Texture2D &out_req_samples) const {
    const TransitionInfo res_transitions[] = {
        {&img0_buf, eResState::UnorderedAccess},       {&img1_buf, eResState::UnorderedAccess},
        {&tonemap_lut_, eResState::ShaderResource},    {&out_pixels, eResState::UnorderedAccess},
        {&out_raw_pixels, eResState::UnorderedAccess}, {&out_variance, eResState::UnorderedAccess},
        {&out_req_samples, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Image, Postprocess::IN_IMG0_SLOT, img0_buf},
                                {eBindTarget::Image, Postprocess::IN_IMG1_SLOT, img1_buf},
                                {eBindTarget::Tex3D, Postprocess::TONEMAP_LUT_SLOT, tonemap_lut_},
                                {eBindTarget::Image, Postprocess::OUT_IMG_SLOT, out_pixels},
                                {eBindTarget::Image, Postprocess::OUT_RAW_IMG_SLOT, out_raw_pixels},
                                {eBindTarget::Image, Postprocess::OUT_VARIANCE_IMG_SLOT, out_variance},
                                {eBindTarget::Image, Postprocess::OUT_REQ_SAMPLES_IMG_SLOT, out_req_samples}};

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + Postprocess::LOCAL_GROUP_SIZE_X - 1) / Postprocess::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + Postprocess::LOCAL_GROUP_SIZE_Y - 1) / Postprocess::LOCAL_GROUP_SIZE_Y), 1u};

    Postprocess::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.exposure = exposure;
    uniform_params.inv_gamma = inv_gamma;
    uniform_params.img0_weight = img0_weight;
    uniform_params.img1_weight = img1_weight;
    uniform_params.tonemap_mode = (loaded_view_transform_ == eViewTransform::Standard) ? 0 : 1;
    uniform_params.variance_threshold = variance_threshold;
    uniform_params.iteration = iteration;

    DispatchCompute(cmd_buf, pi_postprocess_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_FilterVariance(CommandBuffer cmd_buf, const Texture2D &img_buf, const rect_t &rect,
                                              const float variance_threshold, const int iteration,
                                              const Texture2D &out_variance, const Texture2D &out_req_samples) {
    const TransitionInfo res_transitions[] = {{&img_buf, eResState::ShaderResource},
                                              {&out_variance, eResState::UnorderedAccess},
                                              {&out_req_samples, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Tex2DSampled, FilterVariance::IN_IMG_SLOT, img_buf},
                                {eBindTarget::Image, FilterVariance::OUT_IMG_SLOT, out_variance},
                                {eBindTarget::Image, FilterVariance::OUT_REQ_SAMPLES_IMG_SLOT, out_req_samples}};

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + FilterVariance::LOCAL_GROUP_SIZE_X - 1) / Postprocess::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + FilterVariance::LOCAL_GROUP_SIZE_Y - 1) / Postprocess::LOCAL_GROUP_SIZE_Y), 1u};

    FilterVariance::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.inv_img_size[0] = 1.0f / float(w_);
    uniform_params.inv_img_size[1] = 1.0f / float(h_);
    uniform_params.variance_threshold = variance_threshold;
    uniform_params.iteration = iteration;

    DispatchCompute(cmd_buf, pi_filter_variance_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_NLMFilter(CommandBuffer cmd_buf, const Texture2D &img_buf, const Texture2D &var_buf,
                                         const float alpha, const float damping, const Texture2D &base_color_img,
                                         const float base_color_weight, const Texture2D &depth_normals_img,
                                         const float depth_normals_weight, const Texture2D &out_raw_img,
                                         const eViewTransform view_transform, const float inv_gamma, const rect_t &rect,
                                         const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {
        {&img_buf, eResState::ShaderResource},           {&var_buf, eResState::ShaderResource},
        {&tonemap_lut_, eResState::ShaderResource},      {&base_color_img, eResState::ShaderResource},
        {&depth_normals_img, eResState::ShaderResource}, {&out_img, eResState::UnorderedAccess},
        {&out_raw_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Tex2DSampled, NLMFilter::IN_IMG_SLOT, img_buf},
                                {eBindTarget::Tex2DSampled, NLMFilter::VARIANCE_IMG_SLOT, var_buf},
                                {eBindTarget::Tex3D, NLMFilter::TONEMAP_LUT_SLOT, tonemap_lut_},
                                {eBindTarget::Tex2DSampled, NLMFilter::BASE_COLOR_IMG_SLOT, base_color_img},
                                {eBindTarget::Tex2DSampled, NLMFilter::DEPTH_NORMAL_IMG_SLOT, depth_normals_img},
                                {eBindTarget::Image, NLMFilter::OUT_IMG_SLOT, out_img},
                                {eBindTarget::Image, NLMFilter::OUT_RAW_IMG_SLOT, out_raw_img}};

    const uint32_t grp_count[3] = {
        uint32_t((rect.w + NLMFilter::LOCAL_GROUP_SIZE_X - 1) / NLMFilter::LOCAL_GROUP_SIZE_X),
        uint32_t((rect.h + NLMFilter::LOCAL_GROUP_SIZE_Y - 1) / NLMFilter::LOCAL_GROUP_SIZE_Y), 1u};

    NLMFilter::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.inv_img_size[0] = 1.0f / float(w_);
    uniform_params.inv_img_size[1] = 1.0f / float(h_);
    uniform_params.alpha = alpha;
    uniform_params.damping = damping;
    uniform_params.inv_gamma = inv_gamma;
    uniform_params.tonemap_mode = (loaded_view_transform_ == eViewTransform::Standard) ? 0 : 1;
    uniform_params.base_color_weight = base_color_weight;
    uniform_params.depth_normal_weight = depth_normals_weight;

    Pipeline *pi = &pi_nlm_filter_;
    if (base_color_img.ready() && depth_normals_img.ready()) {
        pi = &pi_nlm_filter_bn_;
    } else if (base_color_img.ready()) {
        pi = &pi_nlm_filter_b_;
    } else if (depth_normals_img.ready()) {
        pi = &pi_nlm_filter_n_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortHashRays(CommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &rays,
                                            const Buffer &counters, const float root_min[3], const float cell_size[3],
                                            const Buffer &out_hashes) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&rays, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_hashes, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortHashRays::RAYS_BUF_SLOT, rays},
                                {eBindTarget::SBufRO, SortHashRays::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortHashRays::OUT_HASHES_BUF_SLOT, out_hashes}};

    SortHashRays::Params uniform_params = {};
    memcpy(&uniform_params.root_min[0], root_min, 3 * sizeof(float));
    memcpy(&uniform_params.cell_size[0], cell_size, 3 * sizeof(float));

    DispatchComputeIndirect(cmd_buf, pi_sort_hash_rays_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortScan(CommandBuffer cmd_buf, const bool exclusive, const Buffer &indir_args,
                                        const int indir_args_index, const Buffer &input, const int input_offset,
                                        const int input_stride, const Buffer &out_scan_values,
                                        const Buffer &out_partial_sums) {
    static_assert(SortScan::SCAN_PORTION == SORT_SCAN_PORTION, "!");

    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&input, eResState::ShaderResource},
                                              {&out_scan_values, eResState::UnorderedAccess},
                                              {&out_partial_sums, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortScan::INPUT_BUF_SLOT, input},
                                {eBindTarget::SBufRW, SortScan::OUT_SCAN_VALUES_BUF_SLOT, out_scan_values},
                                {eBindTarget::SBufRW, SortScan::OUT_PARTIAL_SUMS_BUF_SLOT, out_partial_sums}};

    SortScan::Params uniform_params = {};
    uniform_params.offset = input_offset;
    uniform_params.stride = input_stride;

    DispatchComputeIndirect(cmd_buf, exclusive ? pi_sort_exclusive_scan_ : pi_sort_inclusive_scan_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortAddPartialSums(CommandBuffer cmd_buf, const Buffer &indir_args,
                                                  const int indir_args_index, const Buffer &partials_sums,
                                                  const Buffer &inout_values) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&partials_sums, eResState::ShaderResource},
                                              {&inout_values, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortAddPartialSums::PART_SUMS_BUF_SLOT, partials_sums},
                                {eBindTarget::SBufRW, SortAddPartialSums::INOUT_BUF_SLOT, inout_values}};

    DispatchComputeIndirect(cmd_buf, pi_sort_add_partial_sums_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, nullptr, 0,
                            ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortInitCountTable(CommandBuffer cmd_buf, const int shift, const Buffer &indir_args,
                                                  const int indir_args_index, const Buffer &hashes,
                                                  const Buffer &counters, const int counter_index,
                                                  const Buffer &out_count_table) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hashes, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_count_table, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortInitCountTable::HASHES_BUF_SLOT, hashes},
                                {eBindTarget::SBufRO, SortInitCountTable::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortInitCountTable::OUT_COUNT_TABLE_BUF_SLOT, out_count_table}};

    SortInitCountTable::Params uniform_params = {};
    uniform_params.counter = counter_index;
    uniform_params.shift = shift;

    DispatchComputeIndirect(cmd_buf, pi_sort_init_count_table_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortWriteSortedHashes(CommandBuffer cmd_buf, const int shift, const Buffer &indir_args,
                                                     const int indir_args_index, const Buffer &hashes,
                                                     const Buffer &offsets, const Buffer &counters, int counter_index,
                                                     int chunks_counter_index, const Buffer &out_chunks) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hashes, eResState::ShaderResource},
                                              {&offsets, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_chunks, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortWriteSortedChunks::HASHES_BUF_SLOT, hashes},
                                {eBindTarget::SBufRO, SortWriteSortedChunks::OFFSETS_BUF_SLOT, offsets},
                                {eBindTarget::SBufRO, SortWriteSortedChunks::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortWriteSortedChunks::OUT_HASHES_BUF_SLOT, out_chunks}};

    SortWriteSortedChunks::Params uniform_params = {};
    uniform_params.counter = counter_index;
    uniform_params.chunks_counter = chunks_counter_index;
    uniform_params.shift = shift;

    DispatchComputeIndirect(cmd_buf, pi_sort_write_sorted_hashes_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_SortReorderRays(CommandBuffer cmd_buf, const Buffer &indir_args,
                                               const int indir_args_index, const Buffer &in_rays, const Buffer &indices,
                                               const Buffer &counters, const int counter_index,
                                               const Buffer &out_rays) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&in_rays, eResState::ShaderResource},
                                              {&indices, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_rays, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortReorderRays::RAYS_BUF_SLOT, in_rays},
                                {eBindTarget::SBufRO, SortReorderRays::INDICES_BUF_SLOT, indices},
                                {eBindTarget::SBufRO, SortReorderRays::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortReorderRays::OUT_RAYS_BUF_SLOT, out_rays}};

    SortReorderRays::Params uniform_params = {};
    uniform_params.counter = counter_index;

    DispatchComputeIndirect(cmd_buf, pi_sort_reorder_rays_, indir_args,
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
                                {eBindTarget::SBufRO, DebugRT::MESHES_BUF_SLOT, sc_data.meshes},
                                {eBindTarget::SBufRO, DebugRT::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                {eBindTarget::SBufRO, DebugRT::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                {eBindTarget::SBufRO, DebugRT::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                {eBindTarget::AccStruct, DebugRT::TLAS_SLOT, sc_data.rt_tlas},
                                {eBindTarget::SBufRO, DebugRT::RAYS_BUF_SLOT, rays},
                                {eBindTarget::Image, DebugRT::OUT_IMG_SLOT, out_pixels}};

    const uint32_t grp_count[3] = {uint32_t((w_ + DebugRT::LOCAL_GROUP_SIZE_X - 1) / DebugRT::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + DebugRT::LOCAL_GROUP_SIZE_Y - 1) / DebugRT::LOCAL_GROUP_SIZE_Y), 1u};

    DebugRT::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;

    DispatchCompute(cmd_buf, pi_debug_rt_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}