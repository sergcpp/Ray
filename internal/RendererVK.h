#pragma once

#include "../RendererBase.h"
#include "Core.h"

#include "Vk/BufferVK.h"
#include "Vk/ContextVK.h"
#include "Vk/PipelineVK.h"
#include "Vk/ProgramVK.h"
#include "Vk/ShaderVK.h"
#include "Vk/TextureVK.h"

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

    Shader sh_prim_rays_gen_simple_, sh_prim_rays_gen_adaptive_, sh_intersect_scene_, sh_intersect_scene_indirect_,
        sh_intersect_area_lights_, sh_shade_primary_, sh_shade_primary_b_, sh_shade_primary_n_, sh_shade_primary_bn_,
        sh_shade_secondary_, sh_intersect_scene_shadow_, sh_prepare_indir_args_, sh_mix_incremental_,
        sh_mix_incremental_b_, sh_mix_incremental_n_, sh_mix_incremental_bn_, sh_postprocess_, sh_filter_variance_,
        sh_nlm_filter_, sh_nlm_filter_b_, sh_nlm_filter_n_, sh_nlm_filter_bn_, sh_debug_rt_;
    Shader sh_sort_hash_rays_, sh_sort_exclusive_scan_, sh_sort_inclusive_scan_, sh_sort_add_partial_sums_,
        sh_sort_init_count_table_, sh_sort_write_sorted_hashes_, sh_sort_reorder_rays_;
    Shader sh_intersect_scene_rgen_, sh_intersect_scene_rchit_, sh_intersect_scene_rmiss_,
        sh_intersect_scene_indirect_rgen_;

    Program prog_prim_rays_gen_simple_, prog_prim_rays_gen_adaptive_, prog_intersect_scene_,
        prog_intersect_scene_indirect_, prog_intersect_area_lights_, prog_shade_primary_, prog_shade_primary_b_,
        prog_shade_primary_n_, prog_shade_primary_bn_, prog_shade_secondary_, prog_intersect_scene_shadow_,
        prog_prepare_indir_args_, prog_mix_incremental_, prog_mix_incremental_b_, prog_mix_incremental_n_,
        prog_mix_incremental_bn_, prog_postprocess_, prog_filter_variance_, prog_nlm_filter_, prog_nlm_filter_b_,
        prog_nlm_filter_n_, prog_nlm_filter_bn_, prog_debug_rt_;
    Program prog_sort_hash_rays_, prog_sort_exclusive_scan_, prog_sort_inclusive_scan_, prog_sort_add_partial_sums_,
        prog_sort_init_count_table_, prog_sort_write_sorted_hashes_, prog_sort_reorder_rays_;
    Program prog_intersect_scene_rtpipe_, prog_intersect_scene_indirect_rtpipe_;

    Pipeline pi_prim_rays_gen_simple_, pi_prim_rays_gen_adaptive_, pi_intersect_scene_, pi_intersect_scene_indirect_,
        pi_intersect_area_lights_, pi_shade_primary_, pi_shade_primary_b_, pi_shade_primary_n_, pi_shade_primary_bn_,
        pi_shade_secondary_, pi_intersect_scene_shadow_, pi_prepare_indir_args_, pi_mix_incremental_,
        pi_mix_incremental_b_, pi_mix_incremental_n_, pi_mix_incremental_bn_, pi_postprocess_, pi_filter_variance_,
        pi_nlm_filter_, pi_nlm_filter_b_, pi_nlm_filter_n_, pi_nlm_filter_bn_, pi_debug_rt_;
    Pipeline pi_sort_hash_rays_, pi_sort_exclusive_scan_, pi_sort_inclusive_scan_, pi_sort_add_partial_sums_,
        pi_sort_init_count_table_, pi_sort_write_sorted_hashes_, pi_sort_reorder_rays_, pi_intersect_scene_rtpipe_,
        pi_intersect_scene_indirect_rtpipe_;

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

    Buffer halton_seq_buf_, prim_rays_buf_, secondary_rays_buf_, shadow_rays_buf_, prim_hits_buf_, ray_hashes_bufs_[2],
        scan_values_bufs_[4], partial_sums_bufs_[4], count_table_buf_;
    Buffer counters_buf_, indir_args_buf_;

    static const int SORT_SCAN_PORTION = 256;

    Buffer pixel_readback_buf_, base_color_readback_buf_, depth_normals_readback_buf_;
    mutable bool pixel_readback_is_tonemapped_ = false;
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
        SmallVector<int, MAX_BOUNCES * 2> secondary_sort;
        SmallVector<int, MAX_BOUNCES * 2> secondary_trace;
        SmallVector<int, MAX_BOUNCES * 2> secondary_shade;
        SmallVector<int, MAX_BOUNCES * 2> secondary_shadow;
        int denoise[2];
    } timestamps_[MaxFramesInFlight] = {};

    stats_t stats_ = {0};

    void kernel_GeneratePrimaryRays(CommandBuffer cmd_buf, const camera_t &cam, int hi, const rect_t &rect,
                                    const Buffer &random_seq, int iteration, const Texture2D &req_samples_img,
                                    const Buffer &inout_counters, const Buffer &out_rays);
    void kernel_IntersectScene(CommandBuffer cmd_buf, const pass_settings_t &settings, const scene_data_t &sc_data,
                               const Buffer &random_seq, int hi, const rect_t &rect, uint32_t node_index, float inter_t,
                               Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set, const Buffer &rays,
                               const Buffer &out_hits);
    void kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const pass_settings_t &settings,
                                      const scene_data_t &sc_data, const Buffer &random_seq, int hi, const rect_t &rect,
                                      uint32_t node_index, float inter_t, Span<const TextureAtlas> tex_atlases,
                                      VkDescriptorSet tex_descr_set, const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectScene(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                               const Buffer &counters, const pass_settings_t &settings, const scene_data_t &sc_data,
                               const Buffer &random_seq, int hi, uint32_t node_index, float inter_t,
                               Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set, const Buffer &rays,
                               const Buffer &out_hits);
    void kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                      const pass_settings_t &settings, const scene_data_t &sc_data,
                                      const Buffer &random_seq, int hi, uint32_t node_index, float inter_t,
                                      Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                      const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectSceneShadow(CommandBuffer cmd_buf, const pass_settings_t &settings, const Buffer &indir_args,
                                     int indir_args_index, const Buffer &counters, const scene_data_t &sc_data,
                                     const Buffer &random_seq, int hi, uint32_t node_index, float clamp_val,
                                     Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                     const Buffer &sh_rays, const Texture2D &out_img);
    void kernel_IntersectAreaLights(CommandBuffer cmd_buf, const scene_data_t &sc_data, const Buffer &indir_args,
                                    const Buffer &counters, const Buffer &rays, const Buffer &inout_hits);
    void kernel_ShadePrimaryHits(CommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                 const Buffer &indir_args, int indir_args_index, const Buffer &hits, const Buffer &rays,
                                 const scene_data_t &sc_data, const Buffer &random_seq, int hi, const rect_t &rect,
                                 Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                 const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                 const Buffer &inout_counters, const Texture2D &out_base_color,
                                 const Texture2D &out_depth_normals);
    void kernel_ShadeSecondaryHits(CommandBuffer cmd_buf, const pass_settings_t &settings, const environment_t &env,
                                   const Buffer &indir_args, int indir_args_index, const Buffer &hits,
                                   const Buffer &rays, const scene_data_t &sc_data, const Buffer &random_seq, int hi,
                                   Span<const TextureAtlas> tex_atlases, VkDescriptorSet tex_descr_set,
                                   const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                   const Buffer &inout_counters);
    void kernel_PrepareIndirArgs(CommandBuffer cmd_buf, const Buffer &inout_counters, const Buffer &out_indir_args);
    void kernel_MixIncremental(CommandBuffer cmd_buf, float main_mix_factor, float aux_mix_factor, const rect_t &rect,
                               int iteration, const Texture2D &temp_img, const Texture2D &temp_base_color,
                               const Texture2D &temp_depth_normals, const Texture2D &req_samples,
                               const Texture2D &out_img, const Texture2D &out_base_color,
                               const Texture2D &out_depth_normals);
    void kernel_Postprocess(CommandBuffer cmd_buf, const Texture2D &img0_buf, float img0_weight,
                            const Texture2D &img1_buf, float img1_weight, float exposure, float inv_gamma,
                            const rect_t &rect, float variance_threshold, int iteration, const Texture2D &out_pixels,
                            const Texture2D &out_raw_pixels, const Texture2D &out_variance,
                            const Texture2D &out_req_samples) const;
    void kernel_FilterVariance(CommandBuffer cmd_buf, const Texture2D &img_buf, const rect_t &rect,
                               float variance_threshold, int iteration, const Texture2D &out_variance,
                               const Texture2D &out_req_samples);
    void kernel_NLMFilter(CommandBuffer cmd_buf, const Texture2D &img_buf, const Texture2D &var_buf, float alpha,
                          float damping, const Texture2D &base_color_img, float base_color_weight,
                          const Texture2D &depth_normals_img, float depth_normals_weight, const Texture2D &out_raw_img,
                          eViewTransform view_transform, float inv_gamma, const rect_t &rect, const Texture2D &out_img);
    void kernel_SortHashRays(CommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &rays,
                             const Buffer &counters, const float root_min[3], const float cell_size[3],
                             const Buffer &out_hashes);
    void kernel_SortScan(CommandBuffer cmd_buf, bool exclusive, const Buffer &indir_args, int indir_args_index,
                         const Buffer &input, int input_offset, int input_stride, const Buffer &out_scan_values,
                         const Buffer &out_partial_sums);
    void kernel_SortExclusiveScan(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                  const Buffer &input, int input_offset, int input_stride,
                                  const Buffer &out_scan_values, const Buffer &out_partial_sums) {
        kernel_SortScan(cmd_buf, true, indir_args, indir_args_index, input, input_offset, input_stride, out_scan_values,
                        out_partial_sums);
    }
    void kernel_SortInclusiveScan(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                  const Buffer &input, int input_offset, int input_stride,
                                  const Buffer &out_scan_values, const Buffer &out_partial_sums) {
        kernel_SortScan(cmd_buf, false, indir_args, indir_args_index, input, input_offset, input_stride,
                        out_scan_values, out_partial_sums);
    }
    void kernel_SortAddPartialSums(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                   const Buffer &partials_sums, const Buffer &inout_values);
    void kernel_SortInitCountTable(CommandBuffer cmd_buf, int shift, const Buffer &indir_args, int indir_args_index,
                                   const Buffer &hashes, const Buffer &counters, int counter_index,
                                   const Buffer &out_count_table);
    void kernel_SortWriteSortedHashes(CommandBuffer cmd_buf, int shift, const Buffer &indir_args,
                                      int indir_args_index, const Buffer &hashes, const Buffer &offsets,
                                      const Buffer &counters, int counter_index, int chunks_counter_index,
                                      const Buffer &out_chunks);
    void kernel_SortReorderRays(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                const Buffer &in_rays, const Buffer &indices, const Buffer &counters, int counter_index,
                                const Buffer &out_rays);
    void kernel_DebugRT(CommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index, const Buffer &rays,
                        const Texture2D &out_pixels);

    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

    void RadixSort(CommandBuffer cmd_buf, const Buffer &indir_args, Buffer hashes[2], Buffer &count_table,
                   const Buffer &counters, Buffer partial_sums[], Buffer scan_values[]);

    void ExclusiveScan(CommandBuffer cmd_buf, const Buffer &indir_args, const int indir_args_indices[],
                       const Buffer &input, const uint32_t offset, const uint32_t stride, const Buffer partial_sums[],
                       const Buffer scan_values[]);

    color_data_rgba_t get_pixels_ref(bool tonemap) const;

  public:
    Renderer(const settings_t &s, ILog *log);
    ~Renderer() override;

    eRendererType type() const override { return eRendererType::Vulkan; }

    ILog *log() const override { return ctx_->log(); }

    const char *device_name() const override;

    bool is_hwrt() const override { return use_hwrt_; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    color_data_rgba_t get_pixels_ref() const override { return get_pixels_ref(true); }
    color_data_rgba_t get_raw_pixels_ref() const override { return get_pixels_ref(false); }
    color_data_rgba_t get_aux_pixels_ref(eAUXBuffer buf) const override;

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
