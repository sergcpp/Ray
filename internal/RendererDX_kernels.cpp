#include "RendererDX.h"

#include "Dx/DrawCallDX.h"
#include "SceneDX.h"

#include "shaders/debug_rt_interface.h"
#include "shaders/intersect_scene_interface.h"
#include "shaders/intersect_scene_shadow_interface.h"
#include "shaders/shade_interface.h"

#include "shaders/types.h"

#define NS Dx
#include "RendererGPU_kernels.h"
#undef NS

void Ray::Dx::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const pass_settings_t &settings,
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

        bindings.emplace_back(eBindTarget::DescrTable, 2, bindless_tex.srv_descr_table);

        // assert(tex_descr_set);
        // vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_.layout(), 1, 1,
        //                         &tex_descr_set, 0, nullptr);
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

void Ray::Dx::Renderer::kernel_IntersectScene(CommandBuffer cmd_buf, const Buffer &indir_args,
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

        bindings.emplace_back(eBindTarget::DescrTable, 2, bindless_tex.srv_descr_table);

        // assert(tex_descr_set);
        // vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_indirect_.layout(), 1, 1,
        //                         &tex_descr_set, 0, nullptr);
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

void Ray::Dx::Renderer::kernel_ShadePrimaryHits(
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
        bindings.emplace_back(eBindTarget::DescrTable, 2, bindless_tex.srv_descr_table);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, *pi, indir_args, indir_args_index * sizeof(DispatchIndirectCommand), bindings,
                            &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Dx::Renderer::kernel_ShadeSecondaryHits(
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
        bindings.emplace_back(eBindTarget::DescrTable, 2, bindless_tex.srv_descr_table);

        // assert(tex_descr_set);
        // vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_shade_secondary_.layout(), 1, 1,
        //                         &tex_descr_set, 0, nullptr);
    } else {
        bindings.emplace_back(eBindTarget::SBufRO, Types::TEXTURES_BUF_SLOT, sc_data.atlas_textures);
        bindings.emplace_back(eBindTarget::Tex2DArraySampled, Types::TEXTURE_ATLASES_SLOT, tex_atlases);
    }

    DispatchComputeIndirect(cmd_buf, pi_shade_secondary_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Dx::Renderer::kernel_IntersectSceneShadow(CommandBuffer cmd_buf, const pass_settings_t &settings,
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
        bindings.emplace_back(eBindTarget::DescrTable, 2, bindless_tex.srv_descr_table);

        // assert(tex_descr_set);
        // vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pi_intersect_scene_shadow_.layout(), 1, 1,
        //                         &tex_descr_set, 0, nullptr);
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
