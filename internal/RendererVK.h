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
    const AccStructure &rt_tlas;
};

class Renderer : public RendererBase {
  protected:
    std::unique_ptr<Context> ctx_;

    Shader sh_prim_rays_gen_, sh_trace_primary_rays_[2], sh_trace_secondary_rays_[2], sh_intersect_area_lights_,
        sh_shade_primary_hits_[2], sh_shade_secondary_hits_[2], sh_trace_shadow_swrt_[2], sh_trace_shadow_hwrt_[2],
        sh_prepare_indir_args_, sh_mix_incremental_, sh_postprocess_, sh_debug_rt_;

    Program prog_prim_rays_gen_, prog_trace_primary_rays_[2], prog_trace_secondary_rays_[2],
        prog_intersect_area_lights_, prog_shade_primary_hits_[2], prog_shade_secondary_hits_[2],
        prog_trace_shadow_swrt_[2], prog_trace_shadow_hwrt_[2], prog_prepare_indir_args_, prog_mix_incremental_,
        prog_postprocess_, prog_debug_rt_;

    Pipeline pi_prim_rays_gen_, pi_trace_primary_rays_[2], pi_trace_secondary_rays_[2], pi_intersect_area_lights_,
        pi_shade_primary_hits_[2], pi_shade_secondary_hits_[2], pi_trace_shadow_swrt_[2], pi_trace_shadow_hwrt_[2],
        pi_prepare_indir_args_, pi_mix_incremental_, pi_postprocess_, pi_debug_rt_;

    int w_ = 0, h_ = 0;
    bool use_hwrt_ = false, use_bindless_ = false, use_tex_compression_ = false;

    std::vector<uint16_t> permutations_;
    int loaded_halton_;

    Texture2D temp_buf_, clean_buf_, final_buf_;

    Buffer halton_seq_buf_, prim_rays_buf_, secondary_rays_buf_, shadow_rays_buf_, prim_hits_buf_;
    Buffer counters_buf_, indir_args_buf_;

    Buffer pixel_stage_buf_;
    mutable bool frame_dirty_ = true;

    struct {
        int clamp, srgb;
        float gamma;
    } postprocess_params_ = {};

    const pixel_color_t *frame_pixels_ = nullptr;
    std::vector<shl1_data_t> sh_data_host_;

    stats_t stats_ = {0};

    void kernel_GeneratePrimaryRays(VkCommandBuffer cmd_buf, const camera_t &cam, int hi, const rect_t &rect,
                                    const Buffer &halton, const Buffer &out_rays);
    void kernel_TracePrimaryRays(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index,
                                 const Buffer &rays, const Buffer &out_hits);
    void kernel_TraceSecondaryRays(VkCommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &counters,
                                   const scene_data_t &sc_data, uint32_t node_index, const Buffer &rays,
                                   const Buffer &out_hits);
    void kernel_IntersectAreaLights(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, const Buffer &indir_args,
                                    const Buffer &counters, const Buffer &rays, const Buffer &inout_hits);
    void kernel_ShadePrimaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                 const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
                                 const Buffer &halton, int hi, Span<const TextureAtlas> tex_atlases,
                                 VkDescriptorSet tex_descr_set, const Texture2D &out_img, const Buffer &out_rays,
                                 const Buffer &out_sh_rays, const Buffer &inout_counters);
    void kernel_ShadeSecondaryHits(VkCommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                   const Buffer &indir_args, const Buffer &hits, const Buffer &rays,
                                   const scene_data_t &sc_data, const Buffer &halton, int hi,
                                   Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                   const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                   const Buffer &inout_counters);
    void kernel_TraceShadow(VkCommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &counters,
                            const scene_data_t &sc_data, uint32_t node_index, float halton,
                            Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set, const Buffer &sh_rays,
                            const Texture2D &out_img);
    void kernel_PrepareIndirArgs(VkCommandBuffer cmd_buf, const Buffer &inout_counters, const Buffer &out_indir_args);
    void kernel_MixIncremental(VkCommandBuffer cmd_buf, const Texture2D &fbuf1, const Texture2D &fbuf2, float k,
                               const Texture2D &out_img);
    void kernel_Postprocess(VkCommandBuffer cmd_buf, const Texture2D &frame_buf, float inv_gamma, int clamp, int srgb,
                            const Texture2D &out_pixels) const;
    void kernel_DebugRT(VkCommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index, const Buffer &rays,
                        const Texture2D &out_pixels);

    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    Renderer(const settings_t &s, ILog *log);
    ~Renderer() override;

    eRendererType type() const override { return RendererVK; }

    const char *device_name() const override;

    bool is_hwrt() const override { return use_hwrt_; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    const pixel_color_t *get_pixels_ref() const override;

    const shl1_data_t *get_sh_data_ref() const override { return &sh_data_host_[0]; }

    void Resize(int w, int h) override;
    void Clear(const pixel_color_t &c) override;

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }

    // static std::vector<Platform> QueryPlatforms();
};
} // namespace Vk
} // namespace Ray
