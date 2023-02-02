#include "RendererVK.h"

#include "Vk/DrawCall.h"

#include "shaders/debug_rt_interface.h"
#include "shaders/intersect_area_lights_interface.h"
#include "shaders/intersect_scene_interface.h"
#include "shaders/intersect_scene_shadow_interface.h"
#include "shaders/mix_incremental_interface.h"
#include "shaders/postprocess_interface.h"
#include "shaders/prepare_indir_args_interface.h"
#include "shaders/primary_ray_gen_interface.h"
#include "shaders/shade_interface.h"

#include "shaders/types.h"

void Ray::Vk::Renderer::kernel_GeneratePrimaryRays(VkCommandBuffer cmd_buf, const camera_t &cam, const int hi,
                                                   const rect_t &rect, const Buffer &random_seq,
                                                   const Buffer &out_rays) {
    const TransitionInfo res_transitions[] = {{&random_seq, eResState::ShaderResource},
                                              {&prim_rays_buf_, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, PrimaryRayGen::HALTON_SEQ_BUF_SLOT, random_seq},
                                {eBindTarget::SBuf, PrimaryRayGen::OUT_RAYS_BUF_SLOT, prim_rays_buf_}};

    const uint32_t grp_count[3] = {
        uint32_t((w_ + PrimaryRayGen::LOCAL_GROUP_SIZE_X) / PrimaryRayGen::LOCAL_GROUP_SIZE_X),
        uint32_t((h_ + PrimaryRayGen::LOCAL_GROUP_SIZE_Y) / PrimaryRayGen::LOCAL_GROUP_SIZE_Y), 1u};

    PrimaryRayGen::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.hi = hi;

    const float temp = std::tan(0.5f * cam.fov * PI / 180.0f);
    uniform_params.spread_angle = std::atan(2.0f * temp / float(h_));

    memcpy(&uniform_params.cam_origin[0], cam.origin, 3 * sizeof(float));
    uniform_params.cam_origin[3] = temp;
    memcpy(&uniform_params.cam_fwd[0], cam.fwd, 3 * sizeof(float));
    memcpy(&uniform_params.cam_side[0], cam.side, 3 * sizeof(float));
    uniform_params.cam_side[3] = cam.focus_distance;
    memcpy(&uniform_params.cam_up[0], cam.up, 3 * sizeof(float));
    uniform_params.cam_up[3] = cam.sensor_height;
    uniform_params.cam_fstop = cam.fstop;
    uniform_params.cam_focal_length = cam.focal_length;
    uniform_params.cam_lens_rotation = cam.lens_rotation;
    uniform_params.cam_lens_ratio = cam.lens_ratio;
    uniform_params.cam_lens_blades = cam.lens_blades;
    uniform_params.cam_clip_start = cam.clip_start;
    uniform_params.cam_filter = cam.filter;
    uniform_params.shift_x = cam.shift[0];
    uniform_params.shift_y = cam.shift[1];

    DispatchCompute(cmd_buf, pi_prim_rays_gen_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectScenePrimary(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                     const scene_data_t &sc_data, const Buffer &random_seq,
                                                     const int hi, const uint32_t node_index, const float cam_clip_end,
                                                     Span<const TextureAtlas> tex_atlases,
                                                     VkDescriptorSet tex_descr_set, const Buffer &rays,
                                                     const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    IntersectScene::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;
    uniform_params.cam_clip_end = cam_clip_end;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBuf, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBuf, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBuf, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBuf, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBuf, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq}};

    if (use_bindless_) {
        assert(tex_descr_set);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_primary_.layout(), 1, 1,
                                &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArray, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::RAYS_BUF_SLOT, rays);
        bindings.emplace_back(eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::OUT_HITS_BUF_SLOT, out_hits);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRIS_BUF_SLOT, sc_data.tris);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRI_INDICES_BUF_SLOT, sc_data.tri_indices);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::NODES_BUF_SLOT, sc_data.nodes);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MESHES_BUF_SLOT, sc_data.meshes);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MI_INDICES_BUF_SLOT, sc_data.mi_indices);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRANSFORMS_BUF_SLOT, sc_data.transforms);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::RAYS_BUF_SLOT, rays);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::OUT_HITS_BUF_SLOT, out_hits);
    }

    const uint32_t grp_count[3] = {
        uint32_t((w_ + IntersectScene::LOCAL_GROUP_SIZE_X) / IntersectScene::LOCAL_GROUP_SIZE_X),
        uint32_t((h_ + IntersectScene::LOCAL_GROUP_SIZE_Y) / IntersectScene::LOCAL_GROUP_SIZE_Y), 1u};

    DispatchCompute(cmd_buf, pi_intersect_scene_primary_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectSceneSecondary(
    VkCommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &counters, const pass_settings_t &settings,
    const scene_data_t &sc_data, const Buffer &random_seq, const int hi, uint32_t node_index,
    Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set, const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    IntersectScene::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;
    uniform_params.min_transp_depth = settings.min_transp_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.hi = hi;

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBuf, IntersectScene::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBuf, IntersectScene::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBuf, IntersectScene::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBuf, IntersectScene::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBuf, IntersectScene::RANDOM_SEQ_BUF_SLOT, random_seq},
        {eBindTarget::SBuf, IntersectScene::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBuf, IntersectScene::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBuf, IntersectScene::OUT_HITS_BUF_SLOT, out_hits}};

    if (use_bindless_) {
        assert(tex_descr_set);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_secondary_.layout(), 1, 1,
                                &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArray, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectScene::TLAS_SLOT, sc_data.rt_tlas);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRIS_BUF_SLOT, sc_data.tris);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRI_INDICES_BUF_SLOT, sc_data.tri_indices);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::NODES_BUF_SLOT, sc_data.nodes);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MESHES_BUF_SLOT, sc_data.meshes);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::MI_INDICES_BUF_SLOT, sc_data.mi_indices);
        bindings.emplace_back(eBindTarget::SBuf, IntersectScene::TRANSFORMS_BUF_SLOT, sc_data.transforms);
    }

    DispatchComputeIndirect(cmd_buf, pi_intersect_scene_secondary_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectAreaLights(VkCommandBuffer cmd_buf, const scene_data_t &sc_data,
                                                   const Buffer &indir_args, const Buffer &counters, const Buffer &rays,
                                                   const Buffer &inout_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&inout_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {
        {eBindTarget::SBuf, IntersectAreaLights::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBuf, IntersectAreaLights::LIGHTS_BUF_SLOT, sc_data.lights},
        {eBindTarget::SBuf, IntersectAreaLights::VISIBLE_LIGHTS_BUF_SLOT, sc_data.visible_lights},
        {eBindTarget::SBuf, IntersectAreaLights::TRANSFORMS_BUF_SLOT, sc_data.transforms},
        {eBindTarget::SBuf, IntersectAreaLights::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBuf, IntersectAreaLights::INOUT_HITS_BUF_SLOT, inout_hits}};

    IntersectAreaLights::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.visible_lights_count = sc_data.visible_lights_count;

    DispatchComputeIndirect(cmd_buf, pi_intersect_area_lights_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadePrimaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                const environment_t &env, const Buffer &hits, const Buffer &rays,
                                                const scene_data_t &sc_data, const Buffer &random_seq, const int hi,
                                                Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                                const Texture2D &out_img, const Buffer &out_rays,
                                                const Buffer &out_sh_rays, const Buffer &inout_counters) {
    const TransitionInfo res_transitions[] = {
        {&hits, eResState::ShaderResource},           {&rays, eResState::ShaderResource},
        {&random_seq, eResState::ShaderResource},     {&out_img, eResState::UnorderedAccess},
        {&out_rays, eResState::UnorderedAccess},      {&out_sh_rays, eResState::UnorderedAccess},
        {&inout_counters, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBuf, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBuf, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBuf, Shade::LIGHTS_BUF_SLOT, sc_data.lights},
                                         {eBindTarget::SBuf, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBuf, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBuf, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBuf, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBuf, Shade::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                         {eBindTarget::SBuf, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBuf, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBuf, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBuf, Shade::RANDOM_SEQ_BUF_SLOT, random_seq},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBuf, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBuf, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBuf, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

    const uint32_t grp_count[3] = {uint32_t((w_ + Shade::LOCAL_GROUP_SIZE_X) / Shade::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + Shade::LOCAL_GROUP_SIZE_Y) / Shade::LOCAL_GROUP_SIZE_Y), 1u};

    Shade::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
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

    if (use_bindless_) {
        assert(tex_descr_set);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_shade_primary_.layout(), 1, 1,
                                &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArray, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchCompute(cmd_buf, pi_shade_primary_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadeSecondaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                  const environment_t &env, const Buffer &indir_args,
                                                  const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
                                                  const Buffer &random_seq, const int hi,
                                                  Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                                  const Texture2D &out_img, const Buffer &out_rays,
                                                  const Buffer &out_sh_rays, const Buffer &inout_counters) {
    const TransitionInfo res_transitions[] = {
        {&indir_args, eResState::IndirectArgument}, {&hits, eResState::ShaderResource},
        {&rays, eResState::ShaderResource},         {&random_seq, eResState::ShaderResource},
        {&out_img, eResState::UnorderedAccess},     {&out_rays, eResState::UnorderedAccess},
        {&out_sh_rays, eResState::UnorderedAccess}, {&inout_counters, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {{eBindTarget::SBuf, Shade::HITS_BUF_SLOT, hits},
                                         {eBindTarget::SBuf, Shade::RAYS_BUF_SLOT, rays},
                                         {eBindTarget::SBuf, Shade::LIGHTS_BUF_SLOT, sc_data.lights},
                                         {eBindTarget::SBuf, Shade::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                         {eBindTarget::SBuf, Shade::TRIS_BUF_SLOT, sc_data.tris},
                                         {eBindTarget::SBuf, Shade::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                         {eBindTarget::SBuf, Shade::MATERIALS_BUF_SLOT, sc_data.materials},
                                         {eBindTarget::SBuf, Shade::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                         {eBindTarget::SBuf, Shade::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                         {eBindTarget::SBuf, Shade::VERTICES_BUF_SLOT, sc_data.vertices},
                                         {eBindTarget::SBuf, Shade::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                         {eBindTarget::SBuf, Shade::RANDOM_SEQ_BUF_SLOT, random_seq},
                                         {eBindTarget::Tex2D, Shade::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
                                         {eBindTarget::Image, Shade::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::SBuf, Shade::OUT_RAYS_BUF_SLOT, out_rays},
                                         {eBindTarget::SBuf, Shade::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                         {eBindTarget::SBuf, Shade::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

    Shade::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
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

    if (use_bindless_) {
        assert(tex_descr_set);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_shade_secondary_.layout(), 1, 1,
                                &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArray, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, pi_shade_secondary_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_IntersectSceneShadow(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                    const Buffer &indir_args, const Buffer &counters,
                                                    const scene_data_t &sc_data, const uint32_t node_index,
                                                    Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                                    const Buffer &sh_rays, const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&sh_rays, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::SBuf, IntersectSceneShadow::TRIS_BUF_SLOT, sc_data.tris},
        {eBindTarget::SBuf, IntersectSceneShadow::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
        {eBindTarget::SBuf, IntersectSceneShadow::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
        {eBindTarget::SBuf, IntersectSceneShadow::MATERIALS_BUF_SLOT, sc_data.materials},
        {eBindTarget::SBuf, IntersectSceneShadow::NODES_BUF_SLOT, sc_data.nodes},
        {eBindTarget::SBuf, IntersectSceneShadow::MESHES_BUF_SLOT, sc_data.meshes},
        {eBindTarget::SBuf, IntersectSceneShadow::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
        {eBindTarget::SBuf, IntersectSceneShadow::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
        {eBindTarget::SBuf, IntersectSceneShadow::TRANSFORMS_BUF_SLOT, sc_data.transforms},
        {eBindTarget::SBuf, IntersectSceneShadow::VERTICES_BUF_SLOT, sc_data.vertices},
        {eBindTarget::SBuf, IntersectSceneShadow::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
        {eBindTarget::SBuf, IntersectSceneShadow::SH_RAYS_BUF_SLOT, sh_rays},
        {eBindTarget::SBuf, IntersectSceneShadow::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::Image, IntersectSceneShadow::OUT_IMG_SLOT, out_img}};

    if (use_hwrt_) {
        bindings.emplace_back(eBindTarget::AccStruct, IntersectSceneShadow::TLAS_SLOT, sc_data.rt_tlas);
    }

    if (use_bindless_) {
        assert(tex_descr_set);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_shadow_.layout(), 1, 1,
                                &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBuf, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArray, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    IntersectSceneShadow::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;
    uniform_params.max_transp_depth = settings.max_transp_depth;

    DispatchComputeIndirect(cmd_buf, pi_intersect_scene_shadow_, indir_args, sizeof(DispatchIndirectCommand), bindings,
                            &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_PrepareIndirArgs(VkCommandBuffer cmd_buf, const Buffer &inout_counters,
                                                const Buffer &out_indir_args) {
    const TransitionInfo res_transitions[] = {{&inout_counters, eResState::UnorderedAccess},
                                              {&out_indir_args, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, PrepareIndirectArgs::INOUT_COUNTERS_BUF_SLOT, inout_counters},
                                {eBindTarget::SBuf, PrepareIndirectArgs::OUT_INDIR_ARGS_SLOT, out_indir_args}};

    const uint32_t grp_count[3] = {1u, 1u, 1u};

    DispatchCompute(cmd_buf, pi_prepare_indir_args_, grp_count, bindings, nullptr, 0, ctx_->default_descr_alloc(),
                    ctx_->log());
}

void Ray::Vk::Renderer::kernel_MixIncremental(VkCommandBuffer cmd_buf, const Texture2D &fbuf1, const Texture2D &fbuf2,
                                              const float k, const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {{&fbuf1, eResState::UnorderedAccess},
                                              {&fbuf2, eResState::UnorderedAccess},
                                              {&out_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Image, MixIncremental::IN_IMG1_SLOT, fbuf1},
                                {eBindTarget::Image, MixIncremental::IN_IMG2_SLOT, fbuf2},
                                {eBindTarget::Image, MixIncremental::OUT_IMG_SLOT, out_img}};

    const uint32_t grp_count[3] = {
        uint32_t((w_ + MixIncremental::LOCAL_GROUP_SIZE_X) / MixIncremental::LOCAL_GROUP_SIZE_X),
        uint32_t((h_ + MixIncremental::LOCAL_GROUP_SIZE_Y) / MixIncremental::LOCAL_GROUP_SIZE_Y), 1u};

    MixIncremental::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.k = k;

    DispatchCompute(cmd_buf, pi_mix_incremental_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_Postprocess(VkCommandBuffer cmd_buf, const Texture2D &frame_buf,
                                           const float /*inv_gamma*/, const int clamp, const int srgb,
                                           const Texture2D &out_pixels) const {
    const TransitionInfo res_transitions[] = {{&frame_buf, eResState::UnorderedAccess},
                                              {&out_pixels, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Image, Postprocess::IN_IMG_SLOT, frame_buf},
                                {eBindTarget::Image, Postprocess::OUT_IMG_SLOT, out_pixels}};

    const uint32_t grp_count[3] = {uint32_t((w_ + Postprocess::LOCAL_GROUP_SIZE_X) / Postprocess::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + Postprocess::LOCAL_GROUP_SIZE_Y) / Postprocess::LOCAL_GROUP_SIZE_Y),
                                   1u};

    Postprocess::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.srgb = srgb;
    uniform_params._clamp = clamp;

    DispatchCompute(cmd_buf, pi_postprocess_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_DebugRT(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index,
                                       const Buffer &rays, const Texture2D &out_pixels) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_pixels, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, DebugRT::TRIS_BUF_SLOT, sc_data.tris},
                                {eBindTarget::SBuf, DebugRT::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                {eBindTarget::SBuf, DebugRT::NODES_BUF_SLOT, sc_data.nodes},
                                {eBindTarget::SBuf, DebugRT::MESHES_BUF_SLOT, sc_data.meshes},
                                {eBindTarget::SBuf, DebugRT::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                {eBindTarget::SBuf, DebugRT::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                {eBindTarget::SBuf, DebugRT::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                {eBindTarget::AccStruct, DebugRT::TLAS_SLOT, sc_data.rt_tlas},
                                {eBindTarget::SBuf, DebugRT::RAYS_BUF_SLOT, rays},
                                {eBindTarget::Image, DebugRT::OUT_IMG_SLOT, out_pixels}};

    const uint32_t grp_count[3] = {uint32_t((w_ + DebugRT::LOCAL_GROUP_SIZE_X) / DebugRT::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + DebugRT::LOCAL_GROUP_SIZE_Y) / DebugRT::LOCAL_GROUP_SIZE_Y), 1u};

    DebugRT::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;

    DispatchCompute(cmd_buf, pi_debug_rt_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}