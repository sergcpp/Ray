#include "RendererVK.h"

#include "shaders/debug_rt_interface.glsl"
#include "shaders/intersect_area_lights_interface.glsl"
#include "shaders/mix_incremental_interface.glsl"
#include "shaders/postprocess_interface.glsl"
#include "shaders/prepare_indir_args_interface.glsl"
#include "shaders/primary_ray_gen_interface.glsl"
#include "shaders/shade_hits_interface.glsl"
#include "shaders/trace_rays_interface.glsl"
#include "shaders/trace_shadow_interface.glsl"

void Ray::Vk::Renderer::kernel_GeneratePrimaryRays(VkCommandBuffer cmd_buf, const camera_t &cam, const int hi,
                                                   const rect_t &rect, const Buffer &halton, const Buffer &out_rays) {
    const TransitionInfo res_transitions[] = {{&prim_rays_buf_, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, PrimaryRayGen::HALTON_SEQ_BUF_SLOT, halton_seq_buf_},
                                {eBindTarget::SBuf, PrimaryRayGen::OUT_RAYS_BUF_SLOT, prim_rays_buf_}};

    const uint32_t grp_count[3] = {
        uint32_t((w_ + PrimaryRayGen::LOCAL_GROUP_SIZE_X) / PrimaryRayGen::LOCAL_GROUP_SIZE_X),
        uint32_t((h_ + PrimaryRayGen::LOCAL_GROUP_SIZE_Y) / PrimaryRayGen::LOCAL_GROUP_SIZE_Y), 1u};

    PrimaryRayGen::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.hi = hi;

    const float temp = std::tan(0.5f * cam.fov * PI / 180.0f);
    uniform_params.spread_angle = std::atan(2.0f * temp / float(h_));

    memcpy(&uniform_params.cam_origin[0], cam.origin, 3 * sizeof(float));
    uniform_params.cam_origin[3] = cam.fov;
    memcpy(&uniform_params.cam_fwd[0], cam.fwd, 3 * sizeof(float));
    memcpy(&uniform_params.cam_side[0], cam.side, 3 * sizeof(float));
    uniform_params.cam_side[3] = cam.focus_distance;
    memcpy(&uniform_params.cam_up[0], cam.up, 3 * sizeof(float));
    uniform_params.cam_up[3] = cam.focus_factor;

    DispatchCompute(cmd_buf, pi_prim_rays_gen_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_TracePrimaryRays(VkCommandBuffer cmd_buf, const scene_data_t &sc_data,
                                                const uint32_t node_index, const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    TraceRays::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;

    if (use_hwrt_) {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceRays::RAYS_BUF_SLOT, rays},
                                    {eBindTarget::AccStruct, TraceRays::TLAS_SLOT, sc_data.rt_tlas},
                                    {eBindTarget::SBuf, TraceRays::OUT_HITS_BUF_SLOT, out_hits}};

        const uint32_t grp_count[3] = {uint32_t((w_ + TraceRays::LOCAL_GROUP_SIZE_X) / TraceRays::LOCAL_GROUP_SIZE_X),
                                       uint32_t((h_ + TraceRays::LOCAL_GROUP_SIZE_Y) / TraceRays::LOCAL_GROUP_SIZE_Y),
                                       1u};

        DispatchCompute(cmd_buf, pi_trace_primary_rays_[1], grp_count, bindings, &uniform_params,
                        sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
    } else {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceRays::TRIS_BUF_SLOT, sc_data.tris},
                                    {eBindTarget::SBuf, TraceRays::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                    {eBindTarget::SBuf, TraceRays::NODES_BUF_SLOT, sc_data.nodes},
                                    {eBindTarget::SBuf, TraceRays::MESHES_BUF_SLOT, sc_data.meshes},
                                    {eBindTarget::SBuf, TraceRays::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                    {eBindTarget::SBuf, TraceRays::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                    {eBindTarget::SBuf, TraceRays::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                    {eBindTarget::SBuf, TraceRays::RAYS_BUF_SLOT, rays},
                                    {eBindTarget::SBuf, TraceRays::OUT_HITS_BUF_SLOT, out_hits}};

        const uint32_t grp_count[3] = {uint32_t((w_ + TraceRays::LOCAL_GROUP_SIZE_X) / TraceRays::LOCAL_GROUP_SIZE_X),
                                       uint32_t((h_ + TraceRays::LOCAL_GROUP_SIZE_Y) / TraceRays::LOCAL_GROUP_SIZE_Y),
                                       1u};

        DispatchCompute(cmd_buf, pi_trace_primary_rays_[0], grp_count, bindings, &uniform_params,
                        sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
    }
}

void Ray::Vk::Renderer::kernel_TraceSecondaryRays(VkCommandBuffer cmd_buf, const Buffer &indir_args,
                                                  const Buffer &counters, const scene_data_t &sc_data,
                                                  uint32_t node_index, const Buffer &rays, const Buffer &out_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::UnorderedAccess},
                                              {&rays, eResState::UnorderedAccess},
                                              {&out_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    if (use_hwrt_) {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceRays::RAYS_BUF_SLOT, rays},
                                    {eBindTarget::AccStruct, TraceRays::TLAS_SLOT, sc_data.rt_tlas},
                                    {eBindTarget::SBuf, TraceRays::COUNTERS_BUF_SLOT, counters},
                                    {eBindTarget::SBuf, TraceRays::OUT_HITS_BUF_SLOT, out_hits}};

        DispatchComputeIndirect(cmd_buf, pi_trace_secondary_rays_[1], indir_args, 0, bindings, nullptr, 0,
                                ctx_->default_descr_alloc(), ctx_->log());
    } else {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceRays::TRIS_BUF_SLOT, sc_data.tris},
                                    {eBindTarget::SBuf, TraceRays::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                    {eBindTarget::SBuf, TraceRays::NODES_BUF_SLOT, sc_data.nodes},
                                    {eBindTarget::SBuf, TraceRays::MESHES_BUF_SLOT, sc_data.meshes},
                                    {eBindTarget::SBuf, TraceRays::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                    {eBindTarget::SBuf, TraceRays::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                    {eBindTarget::SBuf, TraceRays::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                    {eBindTarget::SBuf, TraceRays::RAYS_BUF_SLOT, rays},
                                    {eBindTarget::SBuf, TraceRays::COUNTERS_BUF_SLOT, counters},
                                    {eBindTarget::SBuf, TraceRays::OUT_HITS_BUF_SLOT, out_hits}};

        TraceRays::Params uniform_params;
        uniform_params.img_size[0] = w_;
        uniform_params.img_size[1] = h_;
        uniform_params.node_index = node_index;

        DispatchComputeIndirect(cmd_buf, pi_trace_secondary_rays_[0], indir_args, 0, bindings, &uniform_params,
                                sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
    }
}

void Ray::Vk::Renderer::kernel_IntersectAreaLights(VkCommandBuffer cmd_buf, const scene_data_t &sc_data,
                                                   const Buffer &indir_args, const Buffer &counters, const Buffer &rays,
                                                   const Buffer &inout_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::UnorderedAccess},
                                              {&rays, eResState::UnorderedAccess},
                                              {&inout_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {
        {eBindTarget::SBuf, IntersectAreaLights::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBuf, IntersectAreaLights::LIGHTS_BUF_SLOT, sc_data.lights},
        {eBindTarget::SBuf, IntersectAreaLights::VISIBLE_LIGHTS_BUF_SLOT, sc_data.visible_lights},
        {eBindTarget::SBuf, IntersectAreaLights::TRANSFORMS_BUF_SLOT, sc_data.transforms},
        {eBindTarget::SBuf, IntersectAreaLights::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBuf, IntersectAreaLights::INOUT_HITS_BUF_SLOT, inout_hits}};

    IntersectAreaLights::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.visible_lights_count = sc_data.visible_lights_count;

    DispatchComputeIndirect(cmd_buf, pi_intersect_area_lights_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadePrimaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                const environment_t &env, const Buffer &hits, const Buffer &rays,
                                                const scene_data_t &sc_data, const Buffer &halton, const int hi,
                                                const TextureAtlas tex_atlases[], const Texture2D &out_img,
                                                const Buffer &out_rays, const Buffer &out_sh_rays,
                                                const Buffer &inout_counters) {
    const TransitionInfo res_transitions[] = {
        {&hits, eResState::UnorderedAccess},          {&rays, eResState::UnorderedAccess},
        {&halton, eResState::UnorderedAccess},        {&out_img, eResState::UnorderedAccess},
        {&out_rays, eResState::UnorderedAccess},      {&out_sh_rays, eResState::UnorderedAccess},
        {&inout_counters, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, ShadeHits::HITS_BUF_SLOT, hits},
                                {eBindTarget::SBuf, ShadeHits::RAYS_BUF_SLOT, rays},
                                {eBindTarget::SBuf, ShadeHits::LIGHTS_BUF_SLOT, sc_data.lights},
                                {eBindTarget::SBuf, ShadeHits::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                {eBindTarget::SBuf, ShadeHits::TRIS_BUF_SLOT, sc_data.tris},
                                {eBindTarget::SBuf, ShadeHits::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                {eBindTarget::SBuf, ShadeHits::MATERIALS_BUF_SLOT, sc_data.materials},
                                {eBindTarget::SBuf, ShadeHits::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                {eBindTarget::SBuf, ShadeHits::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                {eBindTarget::SBuf, ShadeHits::VERTICES_BUF_SLOT, sc_data.vertices},
                                {eBindTarget::SBuf, ShadeHits::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                {eBindTarget::SBuf, ShadeHits::HALTON_SEQ_BUF_SLOT, halton},
                                {eBindTarget::SBuf, ShadeHits::TEXTURES_BUF_SLOT, sc_data.textures},
                                {eBindTarget::Tex2DArray, ShadeHits::TEXTURE_ATLASES_SLOT, {tex_atlases, 4}},
                                {eBindTarget::Image, ShadeHits::OUT_IMG_SLOT, out_img},
                                {eBindTarget::SBuf, ShadeHits::OUT_RAYS_BUF_SLOT, out_rays},
                                {eBindTarget::SBuf, ShadeHits::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                {eBindTarget::SBuf, ShadeHits::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

    const uint32_t grp_count[3] = {uint32_t((w_ + ShadeHits::LOCAL_GROUP_SIZE_X) / ShadeHits::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + ShadeHits::LOCAL_GROUP_SIZE_Y) / ShadeHits::LOCAL_GROUP_SIZE_Y), 1u};

    ShadeHits::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.hi = hi;
    uniform_params.li_count = sc_data.li_count;

    uniform_params.max_diff_depth = settings.max_diff_depth;
    uniform_params.max_spec_depth = settings.max_spec_depth;
    uniform_params.max_refr_depth = settings.max_refr_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.max_total_depth = settings.max_total_depth;

    uniform_params.termination_start_depth = settings.termination_start_depth;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));

    DispatchCompute(cmd_buf, pi_shade_primary_hits_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_ShadeSecondaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings,
                                                  const environment_t &env, const Buffer &indir_args,
                                                  const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
                                                  const Buffer &halton, const int hi, const TextureAtlas tex_atlases[],
                                                  const Texture2D &out_img, const Buffer &out_rays,
                                                  const Buffer &out_sh_rays, const Buffer &inout_counters) {
    const TransitionInfo res_transitions[] = {
        {&indir_args, eResState::IndirectArgument}, {&hits, eResState::UnorderedAccess},
        {&rays, eResState::UnorderedAccess},        {&halton, eResState::UnorderedAccess},
        {&out_img, eResState::UnorderedAccess},     {&out_rays, eResState::UnorderedAccess},
        {&out_sh_rays, eResState::UnorderedAccess}, {&inout_counters, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBuf, ShadeHits::HITS_BUF_SLOT, hits},
                                {eBindTarget::SBuf, ShadeHits::RAYS_BUF_SLOT, rays},
                                {eBindTarget::SBuf, ShadeHits::LIGHTS_BUF_SLOT, sc_data.lights},
                                {eBindTarget::SBuf, ShadeHits::LI_INDICES_BUF_SLOT, sc_data.li_indices},
                                {eBindTarget::SBuf, ShadeHits::TRIS_BUF_SLOT, sc_data.tris},
                                {eBindTarget::SBuf, ShadeHits::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                {eBindTarget::SBuf, ShadeHits::MATERIALS_BUF_SLOT, sc_data.materials},
                                {eBindTarget::SBuf, ShadeHits::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                {eBindTarget::SBuf, ShadeHits::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                {eBindTarget::SBuf, ShadeHits::VERTICES_BUF_SLOT, sc_data.vertices},
                                {eBindTarget::SBuf, ShadeHits::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                {eBindTarget::SBuf, ShadeHits::HALTON_SEQ_BUF_SLOT, halton},
                                {eBindTarget::SBuf, ShadeHits::TEXTURES_BUF_SLOT, sc_data.textures},
                                {eBindTarget::Tex2DArray, ShadeHits::TEXTURE_ATLASES_SLOT, {tex_atlases, 4}},
                                {eBindTarget::Image, ShadeHits::OUT_IMG_SLOT, out_img},
                                {eBindTarget::SBuf, ShadeHits::OUT_RAYS_BUF_SLOT, out_rays},
                                {eBindTarget::SBuf, ShadeHits::OUT_SH_RAYS_BUF_SLOT, out_sh_rays},
                                {eBindTarget::SBuf, ShadeHits::INOUT_COUNTERS_BUF_SLOT, inout_counters}};

    const uint32_t grp_count[3] = {uint32_t((w_ + ShadeHits::LOCAL_GROUP_SIZE_X) / ShadeHits::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + ShadeHits::LOCAL_GROUP_SIZE_Y) / ShadeHits::LOCAL_GROUP_SIZE_Y), 1u};

    ShadeHits::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.hi = hi;
    uniform_params.li_count = sc_data.li_count;

    uniform_params.max_diff_depth = settings.max_diff_depth;
    uniform_params.max_spec_depth = settings.max_spec_depth;
    uniform_params.max_refr_depth = settings.max_refr_depth;
    uniform_params.max_transp_depth = settings.max_transp_depth;
    uniform_params.max_total_depth = settings.max_total_depth;

    uniform_params.termination_start_depth = settings.termination_start_depth;

    memcpy(&uniform_params.env_col[0], env.env_col, 3 * sizeof(float));
    memcpy(&uniform_params.env_col[3], &env.env_map, sizeof(uint32_t));

    DispatchComputeIndirect(cmd_buf, pi_shade_secondary_hits_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_TraceShadow(VkCommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &counters,
                                           const scene_data_t &sc_data, uint32_t node_index, const float halton,
                                           const TextureAtlas tex_atlases[], const Buffer &sh_rays,
                                           const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::UnorderedAccess},
                                              {&sh_rays, eResState::UnorderedAccess},
                                              {&out_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    TraceShadow::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;
    uniform_params.halton = halton;

    if (use_hwrt_) {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceShadow::TRIS_BUF_SLOT, sc_data.tris},
                                    {eBindTarget::SBuf, TraceShadow::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                    {eBindTarget::SBuf, TraceShadow::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                    {eBindTarget::SBuf, TraceShadow::MATERIALS_BUF_SLOT, sc_data.materials},
                                    {eBindTarget::SBuf, TraceShadow::NODES_BUF_SLOT, sc_data.nodes},
                                    {eBindTarget::SBuf, TraceShadow::MESHES_BUF_SLOT, sc_data.meshes},
                                    {eBindTarget::SBuf, TraceShadow::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                    {eBindTarget::SBuf, TraceShadow::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                    {eBindTarget::SBuf, TraceShadow::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                    {eBindTarget::SBuf, TraceShadow::VERTICES_BUF_SLOT, sc_data.vertices},
                                    {eBindTarget::SBuf, TraceShadow::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                    {eBindTarget::SBuf, TraceShadow::TEXTURES_BUF_SLOT, sc_data.textures},
                                    {eBindTarget::Tex2DArray, TraceShadow::TEXTURE_ATLASES_SLOT, {tex_atlases, 4}},
                                    {eBindTarget::SBuf, TraceShadow::SH_RAYS_BUF_SLOT, sh_rays},
                                    {eBindTarget::SBuf, TraceShadow::COUNTERS_BUF_SLOT, counters},
                                    {eBindTarget::AccStruct, TraceShadow::TLAS_SLOT, sc_data.rt_tlas},
                                    {eBindTarget::Image, TraceShadow::OUT_IMG_SLOT, out_img}};

        DispatchComputeIndirect(cmd_buf, pi_trace_shadow_[1], indir_args, sizeof(DispatchIndirectCommand), bindings,
                                &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
    } else {
        const Binding bindings[] = {{eBindTarget::SBuf, TraceShadow::TRIS_BUF_SLOT, sc_data.tris},
                                    {eBindTarget::SBuf, TraceShadow::TRI_INDICES_BUF_SLOT, sc_data.tri_indices},
                                    {eBindTarget::SBuf, TraceShadow::TRI_MATERIALS_BUF_SLOT, sc_data.tri_materials},
                                    {eBindTarget::SBuf, TraceShadow::MATERIALS_BUF_SLOT, sc_data.materials},
                                    {eBindTarget::SBuf, TraceShadow::NODES_BUF_SLOT, sc_data.nodes},
                                    {eBindTarget::SBuf, TraceShadow::MESHES_BUF_SLOT, sc_data.meshes},
                                    {eBindTarget::SBuf, TraceShadow::MESH_INSTANCES_BUF_SLOT, sc_data.mesh_instances},
                                    {eBindTarget::SBuf, TraceShadow::MI_INDICES_BUF_SLOT, sc_data.mi_indices},
                                    {eBindTarget::SBuf, TraceShadow::TRANSFORMS_BUF_SLOT, sc_data.transforms},
                                    {eBindTarget::SBuf, TraceShadow::VERTICES_BUF_SLOT, sc_data.vertices},
                                    {eBindTarget::SBuf, TraceShadow::VTX_INDICES_BUF_SLOT, sc_data.vtx_indices},
                                    {eBindTarget::SBuf, TraceShadow::TEXTURES_BUF_SLOT, sc_data.textures},
                                    {eBindTarget::Tex2DArray, TraceShadow::TEXTURE_ATLASES_SLOT, {tex_atlases, 4}},
                                    {eBindTarget::SBuf, TraceShadow::SH_RAYS_BUF_SLOT, sh_rays},
                                    {eBindTarget::SBuf, TraceShadow::COUNTERS_BUF_SLOT, counters},
                                    {eBindTarget::Image, TraceShadow::OUT_IMG_SLOT, out_img}};

        DispatchComputeIndirect(cmd_buf, pi_trace_shadow_[0], indir_args, sizeof(DispatchIndirectCommand), bindings,
                                &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
    }
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

    MixIncremental::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.k = k;

    DispatchCompute(cmd_buf, pi_mix_incremental_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::Vk::Renderer::kernel_Postprocess(VkCommandBuffer cmd_buf, const Texture2D &frame_buf, const float inv_gamma,
                                           const int clamp, const int srgb, const Texture2D &out_pixels) {
    const TransitionInfo res_transitions[] = {{&frame_buf, eResState::UnorderedAccess},
                                              {&out_pixels, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Image, Postprocess::IN_IMG_SLOT, frame_buf},
                                {eBindTarget::Image, Postprocess::OUT_IMG_SLOT, out_pixels}};

    const uint32_t grp_count[3] = {uint32_t((w_ + Postprocess::LOCAL_GROUP_SIZE_X) / Postprocess::LOCAL_GROUP_SIZE_X),
                                   uint32_t((h_ + Postprocess::LOCAL_GROUP_SIZE_Y) / Postprocess::LOCAL_GROUP_SIZE_Y),
                                   1u};

    Postprocess::Params uniform_params;
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

    DebugRT::Params uniform_params;
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = node_index;

    DispatchCompute(cmd_buf, pi_debug_rt_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}