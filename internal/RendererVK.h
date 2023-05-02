#pragma once

#include "../RendererBase.h"
#include "Core.h"

#include "Vk/Buffer.h"
#include "Vk/Context.h"
#include "Vk/Pipeline.h"
#include "Vk/Program.h"
#include "Vk/Shader.h"
#include "Vk/Texture.h"

namespace Ray {
namespace Vk {
class TextureAtlas;
class AccStructure;

struct scene_data_t {
    const environment_t *env;
    const Buffer &mesh_instances;
    const Buffer &mi_indices;
    const Buffer &meshes;
    const Buffer &transforms;
    const Buffer &vtx_indices;
    const Buffer &vertices;
    const Buffer &nodes;
    const Buffer &tris;
    const Buffer &tri_indices;
    const Buffer &tri_materials;
    const Buffer &materials;
    const Buffer &atlas_textures;
    const Buffer &lights;
    const Buffer &li_indices;
    const int li_count;
    const Buffer &visible_lights;
    const int visible_lights_count;
    const Buffer &blocker_lights;
    const int blocker_lights_count;
    const AccStructure &rt_tlas;
    const Texture2D &env_qtree;
    int env_qtree_levels;
};

class Renderer : public RendererBase {
  protected:
    std::unique_ptr<Context> ctx_;

    Shader sh_prim_rays_gen_, sh_intersect_scene_, sh_intersect_scene_indirect_, sh_intersect_area_lights_,
        sh_shade_primary_, sh_shade_primary_b_, sh_shade_primary_n_, sh_shade_primary_bn_, sh_shade_secondary_,
        sh_intersect_scene_shadow_, sh_prepare_indir_args_, sh_mix_incremental_, sh_mix_incremental_b_,
        sh_mix_incremental_n_, sh_mix_incremental_bn_, sh_postprocess_, sh_filter_variance_, sh_nlm_filter_,
        sh_nlm_filter_b_, sh_nlm_filter_n_, sh_nlm_filter_bn_, sh_debug_rt_;

    Program prog_prim_rays_gen_, prog_intersect_scene_, prog_intersect_scene_indirect_, prog_intersect_area_lights_,
        prog_shade_primary_, prog_shade_primary_b_, prog_shade_primary_n_, prog_shade_primary_bn_,
        prog_shade_secondary_, prog_intersect_scene_shadow_, prog_prepare_indir_args_, prog_mix_incremental_,
        prog_mix_incremental_b_, prog_mix_incremental_n_, prog_mix_incremental_bn_, prog_postprocess_,
        prog_filter_variance_, prog_nlm_filter_, prog_nlm_filter_b_, prog_nlm_filter_n_, prog_nlm_filter_bn_,
        prog_debug_rt_;

    Pipeline pi_prim_rays_gen_, pi_intersect_scene_, pi_intersect_scene_indirect_, pi_intersect_area_lights_,
        pi_shade_primary_, pi_shade_primary_b_, pi_shade_primary_n_, pi_shade_primary_bn_, pi_shade_secondary_,
        pi_intersect_scene_shadow_, pi_prepare_indir_args_, pi_mix_incremental_, pi_mix_incremental_b_,
        pi_mix_incremental_n_, pi_mix_incremental_bn_, pi_postprocess_, pi_filter_variance_, pi_nlm_filter_,
        pi_nlm_filter_b_, pi_nlm_filter_n_, pi_nlm_filter_bn_, pi_debug_rt_;

    int w_ = 0, h_ = 0;
    bool use_hwrt_ = false, use_bindless_ = false, use_tex_compression_ = false;

    std::vector<uint16_t> permutations_;
    int loaded_halton_;

    // TODO: Optimize these!
    Texture2D temp_buf0_, dual_buf_[2], final_buf_, raw_final_buf_, raw_filtered_buf_;
    Texture2D temp_buf1_, base_color_buf_;
    Texture2D temp_depth_normals_buf_, depth_normals_buf_;
    Texture2D required_samples_buf_;

    Texture3D tonemap_lut_;
    eViewTransform loaded_view_transform_ = eViewTransform::Standard;

    Buffer halton_seq_buf_, prim_rays_buf_, secondary_rays_buf_, shadow_rays_buf_, prim_hits_buf_;
    Buffer counters_buf_, indir_args_buf_;

    Buffer pixel_stage_buf_, base_color_stage_buf_, depth_normals_stage_buf_;
    mutable bool pixel_stage_is_tonemapped_ = false;
    mutable bool frame_dirty_ = true, base_color_dirty_ = true, depth_normals_dirty_ = true;

    const color_rgba_t *frame_pixels_ = nullptr, *base_color_pixels_ = nullptr, *depth_normals_pixels_ = nullptr;
    std::vector<shl1_data_t> sh_data_host_;

    struct {
        eViewTransform view_transform;
        float inv_gamma;
    } tonemap_params_;
    float variance_threshold_ = 0.0f;

    struct {
        int primary_ray_gen[2];
        int primary_trace[2];
        int primary_shade[2];
        int primary_shadow[2];
        SmallVector<int, MAX_BOUNCES * 2> secondary_trace;
        SmallVector<int, MAX_BOUNCES * 2> secondary_shade;
        SmallVector<int, MAX_BOUNCES * 2> secondary_shadow;
        int denoise[2];
    } timestamps_[MaxFramesInFlight] = {};

    stats_t stats_ = {0};

    void kernel_GeneratePrimaryRays(VkCommandBuffer cmd_buf, const camera_t &cam, int hi, const rect_t &rect,
                                    const Buffer &random_seq, int iteration, const Texture2D &req_samples_img,
                                    const Buffer &inout_counters, const Buffer &out_rays);
    void kernel_IntersectScene(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const scene_data_t &sc_data,
                               const Buffer &random_seq, int hi, const rect_t &rect, uint32_t node_index, float inter_t,
                               Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set, const Buffer &rays,
                               const Buffer &out_hits);
    void kernel_IntersectScene(VkCommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &counters,
                               const pass_settings_t &settings, const scene_data_t &sc_data, const Buffer &random_seq,
                               int hi, uint32_t node_index, float inter_t, Span<const TextureAtlas> tex_atlases,
                               VkDescriptorSet tex_descr_set, const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectSceneShadow(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const Buffer &indir_args,
                                     const Buffer &counters, const scene_data_t &sc_data, uint32_t node_index,
                                     float clamp_val, Span<const TextureAtlas> tex_atlases,
                                     VkDescriptorSet tex_descr_set, const Buffer &sh_rays, const Texture2D &out_img);
    void kernel_IntersectAreaLights(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, const Buffer &indir_args,
                                    const Buffer &counters, const Buffer &rays, const Buffer &inout_hits);
    void kernel_ShadePrimaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                 const Buffer &indir_args, const Buffer &hits, const Buffer &rays,
                                 const scene_data_t &sc_data, const Buffer &random_seq, int hi, const rect_t &rect,
                                 Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                 const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                 const Buffer &inout_counters, const Texture2D &out_base_color,
                                 const Texture2D &out_depth_normals);
    void kernel_ShadeSecondaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                   const Buffer &indir_args, const Buffer &hits, const Buffer &rays,
                                   const scene_data_t &sc_data, const Buffer &random_seq, int hi,
                                   Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                   const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                   const Buffer &inout_counters);
    void kernel_PrepareIndirArgs(VkCommandBuffer cmd_buf, const Buffer &inout_counters, const Buffer &out_indir_args);
    void kernel_MixIncremental(VkCommandBuffer cmd_buf, float main_mix_factor, float aux_mix_factor, const rect_t &rect,
                               int iteration, const Texture2D &temp_img, const Texture2D &temp_base_color,
                               const Texture2D &temp_depth_normals, const Texture2D &req_samples,
                               const Texture2D &out_img, const Texture2D &out_base_color,
                               const Texture2D &out_depth_normals);
    void kernel_Postprocess(VkCommandBuffer cmd_buf, const Texture2D &img0_buf, float img0_weight,
                            const Texture2D &img1_buf, float img1_weight, float exposure, float inv_gamma,
                            const rect_t &rect, float variance_threshold, int iteration, const Texture2D &out_pixels,
                            const Texture2D &out_raw_pixels, const Texture2D &out_variance,
                            const Texture2D &out_req_samples) const;
    void kernel_FilterVariance(VkCommandBuffer cmd_buf, const Texture2D &img_buf, const rect_t &rect,
                               float variance_threshold, int iteration, const Texture2D &out_variance,
                               const Texture2D &out_req_samples);
    void kernel_NLMFilter(VkCommandBuffer cmd_buf, const Texture2D &img_buf, const Texture2D &var_buf, float alpha,
                          float damping, const Texture2D &base_color_img, float base_color_weight,
                          const Texture2D &depth_normals_img, float depth_normals_weight, const Texture2D &out_raw_img,
                          eViewTransform view_transform, float inv_gamma, const rect_t &rect, const Texture2D &out_img);
    void kernel_DebugRT(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index, const Buffer &rays,
                        const Texture2D &out_pixels);

    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

    const color_rgba_t *get_pixels_ref(bool tonemap) const;

  public:
    Renderer(const settings_t &s, ILog *log);
    ~Renderer() override;

    eRendererType type() const override { return eRendererType::Vulkan; }

    ILog *log() const override { return ctx_->log(); }

    const char *device_name() const override;

    bool is_hwrt() const override { return use_hwrt_; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    const color_rgba_t *get_pixels_ref() const override { return get_pixels_ref(true); }
    const color_rgba_t *get_raw_pixels_ref() const override { return get_pixels_ref(false); }
    const color_rgba_t *get_aux_pixels_ref(eAUXBuffer buf) const override;

    const shl1_data_t *get_sh_data_ref() const override { return &sh_data_host_[0]; }

    void Resize(int w, int h) override;
    void Clear(const color_rgba_t &c) override;

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};
} // namespace Vk
} // namespace Ray
