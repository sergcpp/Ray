
#include "shaders/convolution_interface.h"
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
#include "shaders/shade_sky_interface.h"
#include "shaders/sort_hash_rays_interface.h"
#include "shaders/sort_init_count_table_interface.h"
#include "shaders/sort_reduce_interface.h"
#include "shaders/sort_reorder_rays_interface.h"
#include "shaders/sort_scan_interface.h"
#include "shaders/sort_scatter_interface.h"
#include "shaders/spatial_cache_resolve_interface.h"
#include "shaders/spatial_cache_update_interface.h"

void Ray::NS::Renderer::kernel_GeneratePrimaryRays(CommandBuffer cmd_buf, const camera_t &cam, const uint32_t rand_seed,
                                                   const rect_t &rect, const int img_w, const int img_h,
                                                   const Buffer &rand_seq, const Buffer &filter_table,
                                                   const int iteration, const bool _adaptive,
                                                   const Texture2D &req_samples_img, const Buffer &inout_counters,
                                                   const Buffer &out_rays) {
    const TransitionInfo res_transitions[] = {{&rand_seq, eResState::ShaderResource},
                                              {&req_samples_img, eResState::ShaderResource},
                                              {&inout_counters, eResState::UnorderedAccess},
                                              {&out_rays, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, PrimaryRayGen::RANDOM_SEQ_BUF_SLOT, rand_seq},
                                {eBindTarget::SBufRO, PrimaryRayGen::FILTER_TABLE_BUF_SLOT, filter_table},
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
    uniform_params.img_size[0] = img_w;
    uniform_params.img_size[1] = img_h;
    uniform_params.rand_seed = rand_seed;

    const float temp = std::tan(0.5f * cam.fov * float(PI) / 180.0f);
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

    const bool adaptive =
        _adaptive && (iteration > cam.pass_settings.min_samples) && cam.pass_settings.min_samples != -1;
    DispatchCompute(cmd_buf, adaptive ? pi_prim_rays_gen_adaptive_ : pi_prim_rays_gen_simple_, grp_count, bindings,
                    &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_IntersectAreaLights(CommandBuffer cmd_buf, const scene_data_t &sc_data,
                                                   const Buffer &indir_args, const Buffer &counters, const Buffer &rays,
                                                   const Buffer &inout_hits) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&counters, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&inout_hits, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, IntersectAreaLights::RAYS_BUF_SLOT, rays},
                                {eBindTarget::SBufRO, IntersectAreaLights::LIGHTS_BUF_SLOT, sc_data.lights.gpu_buf()},
                                {eBindTarget::SBufRO, IntersectAreaLights::WNODES_BUF_SLOT, sc_data.light_wnodes},
                                {eBindTarget::SBufRO, IntersectAreaLights::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, IntersectAreaLights::INOUT_HITS_BUF_SLOT, inout_hits}};

    IntersectAreaLights::Params uniform_params = {};
    uniform_params.img_size[0] = w_;
    uniform_params.img_size[1] = h_;
    uniform_params.node_index = 0; // tree root

    DispatchComputeIndirect(cmd_buf, pi_intersect_area_lights_, indir_args, 0, bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_MixIncremental(CommandBuffer cmd_buf, const float mix_factor,
                                              const float half_mix_factor, const rect_t &rect, int iteration,
                                              const float exposure, const Texture2D &temp_img,
                                              const Texture2D &temp_base_color, const Texture2D &temp_depth_normals,
                                              const Texture2D &req_samples, const Texture2D &out_full_img,
                                              const Texture2D &out_half_img, const Texture2D &out_base_color,
                                              const Texture2D &out_depth_normals) {
    const TransitionInfo res_transitions[] = {
        {&temp_img, eResState::UnorderedAccess},           {&temp_base_color, eResState::UnorderedAccess},
        {&req_samples, eResState::UnorderedAccess},        {&out_full_img, eResState::UnorderedAccess},
        {&out_half_img, eResState::UnorderedAccess},       {&out_base_color, eResState::UnorderedAccess},
        {&temp_depth_normals, eResState::UnorderedAccess}, {&out_depth_normals, eResState::UnorderedAccess}};
    SmallVector<Binding, 16> bindings = {{eBindTarget::Image, MixIncremental::IN_TEMP_IMG_SLOT, temp_img},
                                         {eBindTarget::Image, MixIncremental::IN_REQ_SAMPLES_SLOT, req_samples},
                                         {eBindTarget::Image, MixIncremental::OUT_FULL_IMG_SLOT, out_full_img},
                                         {eBindTarget::Image, MixIncremental::OUT_HALF_IMG_SLOT, out_half_img}};
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

    const bool is_class_a = popcount(uint32_t(iteration - 1) & 0xaaaaaaaa) & 1;

    MixIncremental::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.mix_factor = mix_factor;
    uniform_params.half_mix_factor = half_mix_factor;
    uniform_params.iteration = iteration;
    uniform_params.accumulate_half_img = is_class_a ? 1.0f : 0.0f;
    uniform_params.exposure = exposure;

    DispatchCompute(cmd_buf, pi_mix_incremental_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_Postprocess(CommandBuffer cmd_buf, const Texture2D &full_buf, const Texture2D &half_buf,
                                           const float inv_gamma, const rect_t &rect, const float variance_threshold,
                                           const int iteration, const Texture2D &out_pixels,
                                           const Texture2D &out_variance, const Texture2D &out_req_samples) const {
    const TransitionInfo res_transitions[] = {
        {&full_buf, eResState::UnorderedAccess},     {&half_buf, eResState::UnorderedAccess},
        {&tonemap_lut_, eResState::ShaderResource},  {&out_pixels, eResState::UnorderedAccess},
        {&out_variance, eResState::UnorderedAccess}, {&out_req_samples, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::Image, Postprocess::IN_FULL_IMG_SLOT, full_buf},
                                {eBindTarget::Image, Postprocess::IN_HALF_IMG_SLOT, half_buf},
                                {eBindTarget::Tex3DSampled, Postprocess::TONEMAP_LUT_SLOT, tonemap_lut_},
                                {eBindTarget::Image, Postprocess::OUT_IMG_SLOT, out_pixels},
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
    uniform_params.inv_gamma = inv_gamma;
    uniform_params.tonemap_mode = (loaded_view_transform_ == eViewTransform::Standard) ? 0 : 1;
    uniform_params.variance_threshold = variance_threshold;
    uniform_params.iteration = iteration;

    DispatchCompute(cmd_buf, pi_postprocess_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_FilterVariance(CommandBuffer cmd_buf, const Texture2D &img_buf, const rect_t &rect,
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

void Ray::NS::Renderer::kernel_NLMFilter(CommandBuffer cmd_buf, const Texture2D &img_buf, const Texture2D &var_buf,
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

    SmallVector<Binding, 16> bindings = {{eBindTarget::Tex2DSampled, NLMFilter::IN_IMG_SLOT, img_buf},
                                         {eBindTarget::Tex2DSampled, NLMFilter::VARIANCE_IMG_SLOT, var_buf},
                                         {eBindTarget::Tex3DSampled, NLMFilter::TONEMAP_LUT_SLOT, tonemap_lut_},
                                         {eBindTarget::Image, NLMFilter::OUT_IMG_SLOT, out_img},
                                         {eBindTarget::Image, NLMFilter::OUT_RAW_IMG_SLOT, out_raw_img}};

    assert(base_color_img.ready());
    bindings.emplace_back(eBindTarget::Tex2DSampled, NLMFilter::BASE_COLOR_IMG_SLOT, base_color_img);
    assert(depth_normals_img.ready());
    bindings.emplace_back(eBindTarget::Tex2DSampled, NLMFilter::DEPTH_NORMAL_IMG_SLOT, depth_normals_img);

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

    DispatchCompute(cmd_buf, pi_nlm_filter_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels,
                                           const Texture2D &img_buf1, const Texture2D &img_buf2,
                                           const Texture2D &img_buf3, const Sampler &sampler, const rect_t &rect, int w,
                                           int h, const Buffer &weights, uint32_t weights_offset,
                                           uint32_t biases_offset, const Buffer &out_buf, uint32_t output_offset,
                                           int output_stride, const Texture2D &out_debug_img) {
    const TransitionInfo res_transitions[] = {{&img_buf1, eResState::ShaderResource},
                                              {&img_buf2, eResState::ShaderResource},
                                              {&img_buf3, eResState::ShaderResource},
                                              {&out_buf, eResState::UnorderedAccess},
                                              {&out_debug_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);
    const uint32_t weights_size = out_channels * 3 * round_up(3 * in_channels, 8) * el_sz;
    const uint32_t biases_size = out_channels * el_sz;
    const uint32_t output_size = output_stride * (h + 2) * out_channels * el_sz;

    SmallVector<Binding, 8> bindings = {
        {eBindTarget::Tex2D, Convolution::IN_IMG1_SLOT, img_buf1},
        {eBindTarget::Sampler, Convolution::IN_SAMPLER_SLOT, sampler},
        {eBindTarget::SBufRO, Convolution::WEIGHTS_BUF_SLOT, weights_offset, weights_size, weights},
        {eBindTarget::SBufRO, Convolution::BIASES_BUF_SLOT, biases_offset, biases_size, weights},
        {eBindTarget::SBufRW, Convolution::OUT_BUF_SLOT, output_offset, output_size, out_buf}};

    if (img_buf2.ready()) {
        bindings.emplace_back(eBindTarget::Tex2D, Convolution::IN_IMG2_SLOT, img_buf2);
    }
    if (img_buf3.ready()) {
        bindings.emplace_back(eBindTarget::Tex2D, Convolution::IN_IMG3_SLOT, img_buf3);
    }
    if (out_debug_img.ready()) {
        bindings.emplace_back(eBindTarget::Image, Convolution::OUT_DEBUG_IMG_SLOT, out_debug_img);
    }

    const uint32_t grp_count[3] = {uint32_t((rect.w + Convolution::TILE_M - 1) / Convolution::TILE_M),
                                   uint32_t((rect.h + 1) / 2),
                                   uint32_t((out_channels + Convolution::TILE_N - 1) / Convolution::TILE_N)};

    Convolution::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.inv_img_size[0] = 1.0f / float(w_);
    uniform_params.inv_img_size[1] = 1.0f / float(h_);
    uniform_params.output_stride = output_stride;
    uniform_params.in_dims[0] = w;
    uniform_params.in_dims[1] = h;
    uniform_params.out_dims[0] = w;
    uniform_params.out_dims[1] = h;

    Pipeline *pi = nullptr;
    if (in_channels == 9 && out_channels == 32) {
        assert(img_buf2.ready() && img_buf3.ready());
        pi = &pi_convolution_Img_9_32_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels,
                                           const Buffer &input_buf, uint32_t input_offset, int input_stride,
                                           const rect_t &rect, int w, int h, const Buffer &weights,
                                           uint32_t weights_offset, uint32_t biases_offset, const Buffer &out_buf,
                                           uint32_t output_offset, int output_stride, const bool downsample,
                                           const Texture2D &out_debug_img) {
    const TransitionInfo res_transitions[] = {{&out_buf, eResState::UnorderedAccess},
                                              {&out_debug_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);
    const uint32_t weights_size = out_channels * 3 * round_up(3 * in_channels, 8) * el_sz;
    const uint32_t biases_size = out_channels * el_sz;
    const uint32_t output_size = output_stride * (h + 2) * out_channels * el_sz;

    SmallVector<Binding, 16> bindings = {
        {eBindTarget::SBufRO, Convolution::IN_BUF1_SLOT, input_offset, input_buf},
        {eBindTarget::SBufRO, Convolution::WEIGHTS_BUF_SLOT, weights_offset, weights_size, weights},
        {eBindTarget::SBufRO, Convolution::BIASES_BUF_SLOT, biases_offset, biases_size, weights},
        {eBindTarget::SBufRW, Convolution::OUT_BUF_SLOT, output_offset, output_size, out_buf}};

    if (out_debug_img.ready()) {
        bindings.emplace_back(eBindTarget::Image, Convolution::OUT_DEBUG_IMG_SLOT, out_debug_img);
    }

    const uint32_t grp_count[3] = {uint32_t((rect.w + Convolution::TILE_M - 1) / Convolution::TILE_M),
                                   uint32_t((rect.h + 1) / 2),
                                   uint32_t((out_channels + Convolution::TILE_N - 1) / Convolution::TILE_N)};

    Convolution::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.input_stride1 = input_stride;
    uniform_params.output_stride = output_stride;
    uniform_params.in_dims[0] = w;
    uniform_params.in_dims[1] = h;
    if (downsample) {
        uniform_params.out_dims[0] = w / 2;
        uniform_params.out_dims[1] = h / 2;
    } else {
        uniform_params.out_dims[0] = w;
        uniform_params.out_dims[1] = h;
    }

    Pipeline *pi = nullptr;
    if (in_channels == 32 && out_channels == 32 && downsample) {
        pi = &pi_convolution_32_32_Downsample_;
    } else if (in_channels == 32 && out_channels == 48 && downsample) {
        pi = &pi_convolution_32_48_Downsample_;
    } else if (in_channels == 48 && out_channels == 64 && downsample) {
        pi = &pi_convolution_48_64_Downsample_;
    } else if (in_channels == 64 && out_channels == 80 && downsample) {
        pi = &pi_convolution_64_80_Downsample_;
    } else if (in_channels == 64 && out_channels == 64) {
        pi = &pi_convolution_64_64_;
    } else if (in_channels == 64 && out_channels == 32) {
        pi = &pi_convolution_64_32_;
    } else if (in_channels == 80 && out_channels == 96) {
        pi = &pi_convolution_80_96_;
    } else if (in_channels == 96 && out_channels == 96) {
        pi = &pi_convolution_96_96_;
    } else if (in_channels == 112 && out_channels == 112) {
        pi = &pi_convolution_112_112_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels,
                                           const Buffer &input_buf, uint32_t input_offset, int input_stride,
                                           const float inv_gamma, const rect_t &rect, int w, int h,
                                           const Buffer &weights, uint32_t weights_offset, uint32_t biases_offset,
                                           const Texture2D &out_img, const Texture2D &out_tonemapped_img) {
    const TransitionInfo res_transitions[] = {{&input_buf, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess},
                                              {&out_tonemapped_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);
    const uint32_t weights_size = out_channels * 3 * round_up(3 * in_channels, 8) * el_sz;
    const uint32_t biases_size = out_channels * el_sz;

    const Binding bindings[] = {
        {eBindTarget::SBufRO, Convolution::IN_BUF1_SLOT, input_offset, input_buf},
        {eBindTarget::SBufRO, Convolution::WEIGHTS_BUF_SLOT, weights_offset, weights_size, weights},
        {eBindTarget::SBufRO, Convolution::BIASES_BUF_SLOT, biases_offset, biases_size, weights},
        {eBindTarget::Tex3DSampled, Convolution::TONEMAP_LUT_SLOT, tonemap_lut_},
        {eBindTarget::Image, Convolution::OUT_IMG_SLOT, out_img},
        {eBindTarget::Image, Convolution::OUT_TONEMAPPED_IMG_SLOT, out_tonemapped_img}};

    const uint32_t grp_count[3] = {uint32_t((rect.w + Convolution::TILE_M - 1) / Convolution::TILE_M),
                                   uint32_t((rect.h + 1) / 2),
                                   uint32_t((out_channels + Convolution::TILE_N - 1) / Convolution::TILE_N)};

    Convolution::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.input_stride1 = input_stride;
    uniform_params.tonemap_mode = (loaded_view_transform_ == eViewTransform::Standard) ? 0 : 1;
    uniform_params.inv_gamma = inv_gamma;
    uniform_params.in_dims[0] = w;
    uniform_params.in_dims[1] = h;
    uniform_params.out_dims[0] = w;
    uniform_params.out_dims[1] = h;

    Pipeline *pi = nullptr;
    if (in_channels == 32 && out_channels == 3) {
        pi = &pi_convolution_32_3_img_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_ConvolutionConcat(CommandBuffer cmd_buf, int in_channels1, int in_channels2,
                                                 int out_channels, const Buffer &input_buf1, uint32_t input_offset1,
                                                 int input_stride1, bool upscale1, const Buffer &input_buf2,
                                                 uint32_t input_offset2, int input_stride2, const rect_t &rect, int w,
                                                 int h, const Buffer &weights, uint32_t weights_offset,
                                                 uint32_t biases_offset, const Buffer &out_buf, uint32_t output_offset,
                                                 int output_stride, const Texture2D &out_debug_img) {
    const TransitionInfo res_transitions[] = {{&out_buf, eResState::UnorderedAccess},
                                              {&out_debug_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);
    const uint32_t weights_size = out_channels * 3 * (3 * in_channels1 + round_up(3 * in_channels2, 8)) * el_sz;
    const uint32_t biases_size = out_channels * el_sz;
    const uint32_t output_size = output_stride * (h + 2) * out_channels * el_sz;

    SmallVector<Binding, 16> bindings = {
        {eBindTarget::SBufRO, Convolution::IN_BUF1_SLOT, input_offset1, input_buf1},
        {eBindTarget::SBufRO, Convolution::IN_BUF2_SLOT, input_offset2, input_buf2},
        {eBindTarget::SBufRO, Convolution::WEIGHTS_BUF_SLOT, weights_offset, weights_size, weights},
        {eBindTarget::SBufRO, Convolution::BIASES_BUF_SLOT, biases_offset, biases_size, weights},
        {eBindTarget::SBufRW, Convolution::OUT_BUF_SLOT, output_offset, output_size, out_buf}};

    if (out_debug_img.ready()) {
        bindings.emplace_back(eBindTarget::Image, Convolution::OUT_DEBUG_IMG_SLOT, out_debug_img);
    }

    const uint32_t grp_count[3] = {uint32_t((rect.w + Convolution::TILE_M - 1) / Convolution::TILE_M),
                                   uint32_t((rect.h + 1) / 2),
                                   uint32_t((out_channels + Convolution::TILE_N - 1) / Convolution::TILE_N)};

    Convolution::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.input_stride1 = input_stride1;
    uniform_params.input_stride2 = input_stride2;
    uniform_params.output_stride = output_stride;
    uniform_params.in_dims[0] = w;
    uniform_params.in_dims[1] = h;
    uniform_params.out_dims[0] = w;
    uniform_params.out_dims[1] = h;

    Pipeline *pi = nullptr;
    if (in_channels1 == 96 && in_channels2 == 64 && out_channels == 112 && upscale1) {
        pi = &pi_convolution_concat_96_64_112_;
    } else if (in_channels1 == 112 && in_channels2 == 48 && out_channels == 96 && upscale1) {
        pi = &pi_convolution_concat_112_48_96_;
    } else if (in_channels1 == 96 && in_channels2 == 32 && out_channels == 64 && upscale1) {
        pi = &pi_convolution_concat_96_32_64_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_ConvolutionConcat(CommandBuffer cmd_buf, int in_channels1, int in_channels2,
                                                 int out_channels, const Buffer &input_buf1, uint32_t input_offset1,
                                                 int input_stride1, bool upscale1, const Texture2D &img_buf1,
                                                 const Texture2D &img_buf2, const Texture2D &img_buf3,
                                                 const Sampler &sampler, const rect_t &rect, int w, int h,
                                                 const Buffer &weights, uint32_t weights_offset, uint32_t biases_offset,
                                                 const Buffer &out_buf, uint32_t output_offset, int output_stride,
                                                 const Texture2D &out_debug_img) {
    const TransitionInfo res_transitions[] = {{&img_buf1, eResState::ShaderResource},
                                              {&img_buf2, eResState::ShaderResource},
                                              {&out_buf, eResState::UnorderedAccess},
                                              {&out_debug_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);
    const uint32_t weights_size = out_channels * 3 * (3 * in_channels1 + round_up(3 * in_channels2, 8)) * el_sz;
    const uint32_t biases_size = out_channels * el_sz;
    const uint32_t output_size = output_stride * (h + 2) * out_channels * el_sz;

    SmallVector<Binding, 8> bindings = {
        {eBindTarget::SBufRO, Convolution::IN_BUF1_SLOT, input_offset1, input_buf1},
        {eBindTarget::Tex2D, Convolution::IN_IMG2_SLOT, img_buf1},
        {eBindTarget::Sampler, Convolution::IN_SAMPLER_SLOT, sampler},
        {eBindTarget::SBufRO, Convolution::WEIGHTS_BUF_SLOT, weights_offset, weights_size, weights},
        {eBindTarget::SBufRO, Convolution::BIASES_BUF_SLOT, biases_offset, biases_size, weights},
        {eBindTarget::SBufRW, Convolution::OUT_BUF_SLOT, output_offset, output_size, out_buf}};

    if (img_buf2.ready()) {
        bindings.emplace_back(eBindTarget::Tex2D, Convolution::IN_IMG3_SLOT, img_buf2);
    }
    if (img_buf3.ready()) {
        bindings.emplace_back(eBindTarget::Tex2D, Convolution::IN_IMG4_SLOT, img_buf3);
    }
    if (out_debug_img.ready()) {
        bindings.emplace_back(eBindTarget::Image, Convolution::OUT_DEBUG_IMG_SLOT, out_debug_img);
    }

    const uint32_t grp_count[3] = {uint32_t((rect.w + Convolution::TILE_M - 1) / Convolution::TILE_M),
                                   uint32_t((rect.h + 1) / 2),
                                   uint32_t((out_channels + Convolution::TILE_N - 1) / Convolution::TILE_N)};

    Convolution::Params uniform_params = {};
    uniform_params.rect[0] = rect.x;
    uniform_params.rect[1] = rect.y;
    uniform_params.rect[2] = rect.w;
    uniform_params.rect[3] = rect.h;
    uniform_params.inv_img_size[0] = 1.0f / float(w_);
    uniform_params.inv_img_size[1] = 1.0f / float(h_);
    uniform_params.input_stride1 = input_stride1;
    uniform_params.output_stride = output_stride;
    uniform_params.in_dims[0] = w;
    uniform_params.in_dims[1] = h;
    uniform_params.out_dims[0] = w;
    uniform_params.out_dims[1] = h;

    Pipeline *pi = nullptr;
    if (in_channels1 == 64 && in_channels2 == 3 && out_channels == 64 && upscale1) {
        pi = &pi_convolution_concat_64_3_64_;
    } else if (in_channels1 == 64 && in_channels2 == 6 && out_channels == 64 && upscale1) {
        assert(img_buf2.ready());
        pi = &pi_convolution_concat_64_6_64_;
    } else if (in_channels1 == 64 && in_channels2 == 9 && out_channels == 64 && upscale1) {
        assert(img_buf2.ready() && img_buf3.ready());
        pi = &pi_convolution_concat_64_9_64_;
    }

    DispatchCompute(cmd_buf, *pi, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_PrepareIndirArgs(CommandBuffer cmd_buf, const Buffer &inout_counters,
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

void Ray::NS::Renderer::kernel_SortHashRays(CommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &rays,
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

void Ray::NS::Renderer::kernel_SortReorderRays(CommandBuffer cmd_buf, const Buffer &indir_args,
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

void Ray::NS::Renderer::kernel_SortInitCountTable(CommandBuffer cmd_buf, const int shift, const Buffer &indir_args,
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

void Ray::NS::Renderer::kernel_SortReduce(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                          const Buffer &input, const Buffer &counters, int counter_index,
                                          const Buffer &out_reduce_table) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&input, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_reduce_table, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortReduce::INPUT_BUF_SLOT, input},
                                {eBindTarget::SBufRO, SortReduce::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortReduce::OUT_REDUCE_TABLE_BUF_SLOT, out_reduce_table}};

    SortReduce::Params uniform_params = {};
    uniform_params.counter = counter_index;

    DispatchComputeIndirect(cmd_buf, pi_sort_reduce_, indir_args, indir_args_index * sizeof(DispatchIndirectCommand),
                            bindings, &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(),
                            ctx_->log());
}

void Ray::NS::Renderer::kernel_SortScan(CommandBuffer cmd_buf, const Buffer &input, const Buffer &counters,
                                        int counter_index, const Buffer &out_scan_values) {
    SmallVector<TransitionInfo, 4> res_transitions = {{&counters, eResState::ShaderResource},
                                                      {&out_scan_values, eResState::UnorderedAccess}};
    if (&input != &out_scan_values) {
        res_transitions.emplace_back(&input, eResState::ShaderResource);
    }
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortScan::INPUT_BUF_SLOT, input},
                                {eBindTarget::SBufRO, SortScan::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortScan::OUT_SCAN_VALUES_BUF_SLOT, out_scan_values}};

    SortScan::Params uniform_params = {};
    uniform_params.counter = counter_index;

    const uint32_t grp_count[3] = {1, 1, 1};
    DispatchCompute(cmd_buf, pi_sort_scan_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_SortScanAdd(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                           const Buffer &input, const Buffer &scratch, const Buffer &counters,
                                           int counter_index, const Buffer &out_scan_values) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&input, eResState::ShaderResource},
                                              {&scratch, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_scan_values, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortScan::INPUT_BUF_SLOT, input},
                                {eBindTarget::SBufRO, SortScan::SCRATCH_BUF_SLOT, scratch},
                                {eBindTarget::SBufRO, SortScan::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortScan::OUT_SCAN_VALUES_BUF_SLOT, out_scan_values}};

    SortScan::Params uniform_params = {};
    uniform_params.counter = counter_index;

    DispatchComputeIndirect(cmd_buf, pi_sort_scan_add_, indir_args, indir_args_index * sizeof(DispatchIndirectCommand),
                            bindings, &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(),
                            ctx_->log());
}

void Ray::NS::Renderer::kernel_SortScatter(CommandBuffer cmd_buf, int shift, const Buffer &indir_args,
                                           const int indir_args_index, const Buffer &hashes, const Buffer &sum_table,
                                           const Buffer &counters, const int counter_index, const Buffer &out_chunks) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hashes, eResState::ShaderResource},
                                              {&sum_table, eResState::ShaderResource},
                                              {&counters, eResState::ShaderResource},
                                              {&out_chunks, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {{eBindTarget::SBufRO, SortScatter::HASHES_BUF_SLOT, hashes},
                                {eBindTarget::SBufRO, SortScatter::SUM_TABLE_BUF_SLOT, sum_table},
                                {eBindTarget::SBufRO, SortScatter::COUNTERS_BUF_SLOT, counters},
                                {eBindTarget::SBufRW, SortScatter::OUT_HASHES_BUF_SLOT, out_chunks}};

    SortScatter::Params uniform_params = {};
    uniform_params.counter = counter_index;
    uniform_params.shift = shift;

    DispatchComputeIndirect(cmd_buf, pi_sort_scatter_, indir_args, indir_args_index * sizeof(DispatchIndirectCommand),
                            bindings, &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(),
                            ctx_->log());
}

void Ray::NS::Renderer::kernel_SpatialCacheUpdate(CommandBuffer cmd_buf, const cache_grid_params_t &params,
                                                  const Buffer &indir_args, const int indir_args_index,
                                                  const Buffer &counters, const int counter_index, const Buffer &inters,
                                                  const Buffer &rays, const Buffer &cache_data,
                                                  const Texture2D &radiance_img, const Texture2D &depth_normals_img,
                                                  const Buffer &inout_entries, const Buffer &inout_voxels_curr,
                                                  const Buffer &inout_lock_buf) {
    SmallVector<TransitionInfo, 16> res_transitions = {{&indir_args, eResState::IndirectArgument},
                                                       {&counters, eResState::ShaderResource},
                                                       {&inters, eResState::ShaderResource},
                                                       {&rays, eResState::ShaderResource},
                                                       {&radiance_img, eResState::ShaderResource},
                                                       {&depth_normals_img, eResState::ShaderResource},
                                                       {&cache_data, eResState::UnorderedAccess},
                                                       {&inout_entries, eResState::UnorderedAccess},
                                                       {&inout_voxels_curr, eResState::UnorderedAccess}};
    if (!ctx_->int64_atomics_supported()) {
        res_transitions.emplace_back(&inout_lock_buf, eResState::UnorderedAccess);
    }
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 16> bindings = {
        {eBindTarget::SBufRO, CacheUpdate::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::SBufRO, CacheUpdate::HITS_BUF_SLOT, inters},
        {eBindTarget::SBufRO, CacheUpdate::RAYS_BUF_SLOT, rays},
        {eBindTarget::Tex2D, CacheUpdate::RADIANCE_TEX_SLOT, radiance_img},
        {eBindTarget::Tex2D, CacheUpdate::DEPTH_NORMAL_TEX_SLOT, depth_normals_img},
        {eBindTarget::SBufRW, CacheUpdate::INOUT_CACHE_DATA_BUF_SLOT, cache_data},
        {eBindTarget::SBufRW, CacheUpdate::INOUT_CACHE_ENTRIES_BUF_SLOT, inout_entries},
        {eBindTarget::SBufRW, CacheUpdate::INOUT_CACHE_VOXELS_BUF_SLOT, inout_voxels_curr}};
    if (!ctx_->int64_atomics_supported()) {
        bindings.emplace_back(eBindTarget::SBufRW, CacheUpdate::INOUT_CACHE_LOCK_BUF_SLOT, inout_lock_buf);
    }

    CacheUpdate::Params uniform_params = {};
    memcpy(&uniform_params.cam_pos_curr[0], params.cam_pos_curr, 3 * sizeof(float));
    uniform_params.cache_w = (w_ / RAD_CACHE_DOWNSAMPLING_FACTOR);
    uniform_params.entries_count = inout_entries.size() / sizeof(uint64_t);
    uniform_params.exposure = params.exposure;

    DispatchComputeIndirect(cmd_buf, pi_spatial_cache_update_, indir_args,
                            indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                            sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
}

void Ray::NS::Renderer::kernel_SpatialCacheResolve(CommandBuffer cmd_buf, const cache_grid_params_t &params,
                                                   const Buffer &inout_entries, const Buffer &inout_voxels_curr,
                                                   const Buffer &voxels_prev) {
    const TransitionInfo res_transitions[] = {{&voxels_prev, eResState::ShaderResource},
                                              {&inout_entries, eResState::UnorderedAccess},
                                              {&inout_voxels_curr, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    const Binding bindings[] = {
        {eBindTarget::SBufRO, CacheResolve::CACHE_VOXELS_PREV_BUF_SLOT, voxels_prev},
        {eBindTarget::SBufRW, CacheResolve::INOUT_CACHE_ENTRIES_BUF_SLOT, inout_entries},
        {eBindTarget::SBufRW, CacheResolve::INOUT_CACHE_VOXELS_CURR_BUF_SLOT, inout_voxels_curr}};

    CacheResolve::Params uniform_params = {};
    memcpy(&uniform_params.cam_pos_curr[0], params.cam_pos_curr, 3 * sizeof(float));
    memcpy(&uniform_params.cam_pos_prev[0], params.cam_pos_prev, 3 * sizeof(float));
    uniform_params.cache_w = (w_ / RAD_CACHE_DOWNSAMPLING_FACTOR);
    uniform_params.entries_count = inout_entries.size() / sizeof(uint64_t);
    const bool cam_moved =
        length2(Ref::make_fvec3(params.cam_pos_curr) - Ref::make_fvec3(params.cam_pos_prev)) > FLT_EPS;
    uniform_params.cam_moved = cam_moved ? 1.0f : 0.0f;

    assert((uniform_params.entries_count % CacheResolve::LOCAL_GROUP_SIZE_X) == 0);
    const uint32_t grp_count[3] = {uniform_params.entries_count / CacheResolve::LOCAL_GROUP_SIZE_X, 1, 1};
    DispatchCompute(cmd_buf, pi_spatial_cache_resolve_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                    ctx_->default_descr_alloc(), ctx_->log());
}

inline void Ray::NS::Renderer::kernel_ShadeSky(CommandBuffer cmd_buf, const pass_settings_t &ps, const float limit,
                                               const environment_t &env, const Buffer &indir_args,
                                               const int indir_args_index, const Buffer &hits, const Buffer &rays,
                                               const Buffer &ray_indices, const Buffer &counters,
                                               const scene_data_t &sc_data, const int iteration,
                                               const Texture2D &out_img) {
    const TransitionInfo res_transitions[] = {{&indir_args, eResState::IndirectArgument},
                                              {&hits, eResState::ShaderResource},
                                              {&rays, eResState::ShaderResource},
                                              {&ray_indices, eResState::ShaderResource},
                                              {&out_img, eResState::UnorderedAccess}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    SmallVector<Binding, 32> bindings = {
        {eBindTarget::UBuf, ShadeSky::ATMOSPHERE_PARAMS_BUF_SLOT, sc_data.atmosphere_params},
        {eBindTarget::SBufRO, ShadeSky::RAY_INDICES_BUF_SLOT, ray_indices},
        {eBindTarget::SBufRO, ShadeSky::HITS_BUF_SLOT, hits},
        {eBindTarget::SBufRO, ShadeSky::RAYS_BUF_SLOT, rays},
        {eBindTarget::SBufRO, ShadeSky::COUNTERS_BUF_SLOT, counters},
        {eBindTarget::Tex2D, ShadeSky::ENV_QTREE_TEX_SLOT, sc_data.env_qtree},
        {eBindTarget::Tex2DSampled, ShadeSky::TRANSMITTANCE_LUT_SLOT, sc_data.transmittance_lut},
        {eBindTarget::Tex2DSampled, ShadeSky::MULTISCATTER_LUT_SLOT, sc_data.multiscatter_lut},
        {eBindTarget::Tex2DSampled, ShadeSky::MOON_TEX_SLOT, sc_data.moon_tex},
        {eBindTarget::Tex2DSampled, ShadeSky::WEATHER_TEX_SLOT, sc_data.weather_tex},
        {eBindTarget::Tex2DSampled, ShadeSky::CIRRUS_TEX_SLOT, sc_data.cirrus_tex},
        {eBindTarget::Tex3DSampled, ShadeSky::NOISE3D_TEX_SLOT, sc_data.noise3d_tex},
        {eBindTarget::Image, ShadeSky::OUT_IMG_SLOT, out_img}};

    const uint32_t rand_seed = Ref::hash(iteration);

    ShadeSky::Params uniform_params = {};
    uniform_params.rand_seed = rand_seed;
    uniform_params.env_rotation = env.env_map_rotation;
    uniform_params.back_rotation = env.back_map_rotation;
    uniform_params.limit = limit;
    uniform_params.max_total_depth = ps.max_total_depth;
    uniform_params.li_count = sc_data.li_count;
    uniform_params.env_qtree_levels = sc_data.env_qtree_levels;
    uniform_params.env_light_index = sc_data.env.light_index;

    if (!sc_data.dir_lights.empty()) {
        for (const uint32_t li : sc_data.dir_lights) {
            const light_t &l = sc_data.lights.at(li);

            memcpy(uniform_params.light_dir, l.dir.dir, 3 * sizeof(float));
            uniform_params.light_dir[3] = l.dir.angle;
            memcpy(uniform_params.light_col, l.col, 3 * sizeof(float));
            if (l.dir.angle != 0.0f) {
                const float radius = tanf(l.dir.angle);
                uniform_params.light_col[0] *= (PI * radius * radius);
                uniform_params.light_col[1] *= (PI * radius * radius);
                uniform_params.light_col[2] *= (PI * radius * radius);
            }

            DispatchComputeIndirect(cmd_buf, pi_shade_sky_, indir_args,
                                    indir_args_index * sizeof(DispatchIndirectCommand), bindings, &uniform_params,
                                    sizeof(uniform_params), ctx_->default_descr_alloc(), ctx_->log());
        }
    } else if (sc_data.env.atmosphere.stars_brightness > 0.0f) {
        // Use fake lightsource (to light up the moon)
        const fvec4 light_dir = {0.0f, -1.0f, 0.0f, 0.0f},
                    light_col = {144809.866891f, 129443.618266f, 127098.894121f, 0.0f};

        memcpy(uniform_params.light_dir, value_ptr(light_dir), 3 * sizeof(float));
        uniform_params.light_dir[3] = 0.0f;
        memcpy(uniform_params.light_col, value_ptr(light_col), 3 * sizeof(float));

        DispatchComputeIndirect(cmd_buf, pi_shade_sky_, indir_args, indir_args_index * sizeof(DispatchIndirectCommand),
                                bindings, &uniform_params, sizeof(uniform_params), ctx_->default_descr_alloc(),
                                ctx_->log());
    }
}

inline void Ray::NS::Renderer::kernel_ShadeSkyPrimary(CommandBuffer cmd_buf, const pass_settings_t &ps,
                                                      const environment_t &env, const Buffer &indir_args,
                                                      const int indir_args_index, const Buffer &hits,
                                                      const Buffer &rays, const Buffer &ray_indices,
                                                      const Buffer &counters, const scene_data_t &sc_data,
                                                      const int iteration, const Texture2D &out_img) {
    const float limit = (ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX;
    kernel_ShadeSky(cmd_buf, ps, limit, env, indir_args, indir_args_index, hits, rays, ray_indices, counters, sc_data,
                    iteration, out_img);
}

inline void Ray::NS::Renderer::kernel_ShadeSkySecondary(
    CommandBuffer cmd_buf, const pass_settings_t &ps, const float clamp_direct, const environment_t &env,
    const Buffer &indir_args, int indir_args_index, const Buffer &hits, const Buffer &rays, const Buffer &ray_indices,
    const Buffer &counters, const scene_data_t &sc_data, int iteration, const Texture2D &out_img) {
    const float limit = (clamp_direct != 0.0f) ? 3.0f * clamp_direct : FLT_MAX;
    kernel_ShadeSky(cmd_buf, ps, limit, env, indir_args, indir_args_index, hits, rays, ray_indices, counters, sc_data,
                    iteration, out_img);
}