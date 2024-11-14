

namespace Ray {
namespace NS {
class TextureAtlas;
class AccStructure;
struct BindlessTexData;

struct scene_data_t {
    const environment_t &env;
    const Buffer &mesh_instances;
    Span<const mesh_t> meshes;
    const Buffer &vtx_indices;
    const Buffer &vertices;
    const Buffer &nodes;
    const Buffer &tris;
    const Buffer &tri_indices;
    const Buffer &tri_materials;
    const Buffer &materials;
    Span<const TextureAtlas> tex_atlases;
    const Buffer &atlas_textures;
    const SparseStorage<light_t> &lights;
    const Buffer &li_indices;
    const int li_count;
    Span<const uint32_t> dir_lights;
    const uint32_t visible_lights_count;
    const uint32_t blocker_lights_count;
    const Buffer &light_cwnodes;
    const AccStructure &rt_tlas;
    const Texture2D &env_qtree;
    int env_qtree_levels;
    const cache_grid_params_t &spatial_cache_grid;
    const Buffer &spatial_cache_entries;
    const Buffer &spatial_cache_voxels;
    const Buffer &atmosphere_params;
    const Texture2D &transmittance_lut;
    const Texture2D &multiscatter_lut;
    const Texture2D &moon_tex;
    const Texture2D &weather_tex;
    const Texture2D &cirrus_tex;
    const Texture2D &curl_tex;
    const Texture3D &noise3d_tex;
};

class Renderer : public RendererBase {
  protected:
    std::unique_ptr<Context> ctx_;
    GpuCommandBuffer external_cmd_buf_;

    Shader sh_prim_rays_gen_simple_, sh_prim_rays_gen_adaptive_;
    Shader sh_intersect_scene_, sh_intersect_scene_indirect_, sh_intersect_area_lights_;
    Shader sh_shade_primary_, sh_shade_primary_sky_, sh_shade_primary_cache_update_, sh_shade_primary_cache_query_,
        sh_shade_primary_cache_query_sky_, sh_shade_secondary_, sh_shade_secondary_sky_,
        sh_shade_secondary_cache_update_, sh_shade_secondary_cache_query_, sh_shade_secondary_cache_query_sky_;
    Shader sh_shade_sky_;
    Shader sh_intersect_scene_shadow_, sh_prepare_indir_args_, sh_mix_incremental_, sh_postprocess_,
        sh_filter_variance_, sh_nlm_filter_, sh_debug_rt_;
    Shader sh_sort_hash_rays_, sh_sort_init_count_table_, sh_sort_reduce_, sh_sort_scan_, sh_sort_scan_add_,
        sh_sort_scatter_, sh_sort_reorder_rays_;
    Shader sh_intersect_scene_rgen_, sh_intersect_scene_rchit_, sh_intersect_scene_rmiss_,
        sh_intersect_scene_indirect_rgen_;
    Shader sh_convolution_Img_9_32_, sh_convolution_32_32_Downsample_, sh_convolution_32_48_Downsample_,
        sh_convolution_48_64_Downsample_, sh_convolution_64_80_Downsample_, sh_convolution_64_64_,
        sh_convolution_64_32_, sh_convolution_80_96_, sh_convolution_96_96_, sh_convolution_112_112_,
        sh_convolution_concat_96_64_112_, sh_convolution_concat_112_48_96_, sh_convolution_concat_96_32_64_,
        sh_convolution_concat_64_3_64_, sh_convolution_concat_64_6_64_, sh_convolution_concat_64_9_64_,
        sh_convolution_32_3_img_;
    Shader sh_spatial_cache_update_, sh_spatial_cache_resolve_;

    Program prog_prim_rays_gen_simple_, prog_prim_rays_gen_adaptive_;
    Program prog_intersect_scene_, prog_intersect_scene_indirect_, prog_intersect_area_lights_;
    Program prog_shade_primary_, prog_shade_primary_sky_, prog_shade_primary_cache_update_,
        prog_shade_primary_cache_query_, prog_shade_primary_cache_query_sky_, prog_shade_secondary_,
        prog_shade_secondary_sky_, prog_shade_secondary_cache_update_, prog_shade_secondary_cache_query_,
        prog_shade_secondary_cache_query_sky_;
    Program prog_shade_sky_;
    Program prog_intersect_scene_shadow_, prog_prepare_indir_args_, prog_mix_incremental_, prog_postprocess_,
        prog_filter_variance_, prog_nlm_filter_, prog_debug_rt_;
    Program prog_sort_hash_rays_, prog_sort_init_count_table_, prog_sort_reduce_, prog_sort_scan_, prog_sort_scan_add_,
        prog_sort_scatter_, prog_sort_reorder_rays_;
    Program prog_intersect_scene_rtpipe_, prog_intersect_scene_indirect_rtpipe_;
    Program prog_convolution_Img_9_32_, prog_convolution_32_32_Downsample_, prog_convolution_32_48_Downsample_,
        prog_convolution_48_64_Downsample_, prog_convolution_64_80_Downsample_, prog_convolution_64_64_,
        prog_convolution_64_32_, prog_convolution_80_96_, prog_convolution_96_96_, prog_convolution_112_112_,
        prog_convolution_concat_96_64_112_, prog_convolution_concat_112_48_96_, prog_convolution_concat_96_32_64_,
        prog_convolution_concat_64_3_64_, prog_convolution_concat_64_6_64_, prog_convolution_concat_64_9_64_,
        prog_convolution_32_3_img_;
    Program prog_spatial_cache_update_, prog_spatial_cache_resolve_;

    Pipeline pi_prim_rays_gen_simple_, pi_prim_rays_gen_adaptive_;
    Pipeline pi_intersect_scene_, pi_intersect_scene_indirect_, pi_intersect_area_lights_;
    Pipeline pi_shade_primary_, pi_shade_primary_sky_, pi_shade_primary_cache_update_, pi_shade_primary_cache_query_,
        pi_shade_primary_cache_query_sky_, pi_shade_secondary_, pi_shade_secondary_sky_,
        pi_shade_secondary_cache_update_, pi_shade_secondary_cache_query_, pi_shade_secondary_cache_query_sky_;
    Pipeline pi_shade_sky_;
    Pipeline pi_intersect_scene_shadow_, pi_prepare_indir_args_, pi_mix_incremental_, pi_postprocess_,
        pi_filter_variance_, pi_nlm_filter_, pi_debug_rt_;
    Pipeline pi_sort_hash_rays_, pi_sort_init_count_table_, pi_sort_reduce_, pi_sort_scan_, pi_sort_scan_add_,
        pi_sort_scatter_, pi_sort_reorder_rays_, pi_intersect_scene_rtpipe_, pi_intersect_scene_indirect_rtpipe_;
    Pipeline pi_convolution_Img_9_32_, pi_convolution_32_32_Downsample_, pi_convolution_32_48_Downsample_,
        pi_convolution_48_64_Downsample_, pi_convolution_64_80_Downsample_, pi_convolution_64_64_,
        pi_convolution_64_32_, pi_convolution_80_96_, pi_convolution_96_96_, pi_convolution_112_112_,
        pi_convolution_concat_96_64_112_, pi_convolution_concat_112_48_96_, pi_convolution_concat_96_32_64_,
        pi_convolution_concat_64_3_64_, pi_convolution_concat_64_6_64_, pi_convolution_concat_64_9_64_,
        pi_convolution_32_3_img_;
    Pipeline pi_spatial_cache_update_, pi_spatial_cache_resolve_;

    bool InitShaders(ILog *log);

    int w_ = 0, h_ = 0;
    bool use_hwrt_ = false, use_bindless_ = false, use_tex_compression_ = false, use_fp16_ = false,
         use_coop_matrix_ = false, use_subgroup_ = false, use_spatial_cache_ = false;

    ePixelFilter filter_table_filter_ = ePixelFilter(-1);
    float filter_table_width_ = 0.0f;
    Buffer filter_table_;
    void UpdateFilterTable(CommandBuffer cmd_buf, ePixelFilter filter, float filter_width);

    // TODO: Optimize these!
    Texture2D temp_buf0_, full_buf_, half_buf_, final_buf_, raw_filtered_buf_;
    Texture2D temp_buf1_, base_color_buf_;
    Texture2D temp_depth_normals_buf_, depth_normals_buf_;
    Texture2D required_samples_buf_;

    Sampler zero_border_sampler_;

    Texture3D tonemap_lut_;
    eViewTransform loaded_view_transform_ = eViewTransform::Standard;

    Buffer random_seq_buf_, prim_rays_buf_, secondary_rays_buf_, shadow_rays_buf_, prim_hits_buf_, ray_hashes_bufs_[2],
        count_table_buf_, reduce_table_buf_;
    Buffer counters_buf_, indir_args_buf_[2];

    Buffer temp_cache_data_buf_, temp_lock_buf_;

    Buffer pixel_readback_buf_, base_color_readback_buf_, depth_normals_readback_buf_;
    mutable bool pixel_readback_is_tonemapped_ = false;
    mutable bool frame_dirty_ = true, base_color_dirty_ = true, depth_normals_dirty_ = true;

    const color_rgba_t *frame_pixels_ = nullptr, *base_color_pixels_ = nullptr, *depth_normals_pixels_ = nullptr;
    std::vector<shl1_data_t> sh_data_host_;

    Buffer unet_weights_;
    unet_weight_offsets_t unet_offsets_;
    bool unet_alias_memory_ = true;
    Buffer unet_tensors_heap_;
    unet_filter_tensors_t unet_tensors_ = {};
    SmallVector<int, 2> unet_alias_dependencies_[UNetFilterPasses];
    bool InitUNetFilterPipelines();
    void UpdateUNetFilterMemory(CommandBuffer cmd_buf);

    struct {
        eViewTransform view_transform;
        float inv_gamma;
    } tonemap_params_;
    float variance_threshold_ = 0.0f;

    struct {
        int cache_update[2];
        int cache_resolve[2];
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

    void kernel_GeneratePrimaryRays(CommandBuffer cmd_buf, const camera_t &cam, uint32_t rand_seed, const rect_t &rect,
                                    int img_w, int img_h, const Buffer &rand_seq, const Buffer &filter_table,
                                    int iteration, bool adaptive, const Texture2D &req_samples_img,
                                    const Buffer &inout_counters, const Buffer &out_rays);
    void kernel_IntersectScene(CommandBuffer cmd_buf, const pass_settings_t &ps, const scene_data_t &sc_data,
                               const Buffer &rand_seq, uint32_t rand_seed, int iteration, const rect_t &rect,
                               uint32_t node_index, const float cam_fwd[3], float clip_dist,
                               Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
                               const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const pass_settings_t &ps, const scene_data_t &sc_data,
                                      const Buffer &rand_seq, uint32_t rand_seed, int iteration, const rect_t &rect,
                                      uint32_t node_index, const float cam_fwd[3], float clip_dist,
                                      Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
                                      const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectScene(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                               const Buffer &counters, const pass_settings_t &ps, const scene_data_t &sc_data,
                               const Buffer &rand_seq, uint32_t rand_seed, int iteration, uint32_t node_index,
                               const float cam_fwd[3], float clip_dist, Span<const TextureAtlas> tex_atlases,
                               const BindlessTexData &bindless_tex, const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectScene_RTPipe(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                      const pass_settings_t &ps, const scene_data_t &sc_data, const Buffer &rand_seq,
                                      uint32_t rand_seed, int iteration, uint32_t node_index, const float cam_fwd[3],
                                      float clip_dist, Span<const TextureAtlas> tex_atlases,
                                      const BindlessTexData &bindless_tex, const Buffer &rays, const Buffer &out_hits);
    void kernel_IntersectSceneShadow(CommandBuffer cmd_buf, const pass_settings_t &ps, const Buffer &indir_args,
                                     int indir_args_index, const Buffer &counters, const scene_data_t &sc_data,
                                     const Buffer &rand_seq, uint32_t rand_seed, int iteration, uint32_t node_index,
                                     float clamp_val, Span<const TextureAtlas> tex_atlases,
                                     const BindlessTexData &bindless_tex, const Buffer &sh_rays,
                                     const Texture2D &out_img);
    void kernel_IntersectAreaLights(CommandBuffer cmd_buf, const scene_data_t &sc_data, const Buffer &indir_args,
                                    const Buffer &counters, const Buffer &rays, const Buffer &inout_hits);
    void kernel_ShadePrimaryHits(CommandBuffer cmd_buf, const pass_settings_t &ps, eSpatialCacheMode cache,
                                 const environment_t &env, const Buffer &indir_args, int indir_args_index,
                                 const Buffer &hits, const Buffer &rays, const scene_data_t &sc_data,
                                 const Buffer &rand_seq, uint32_t rand_seed, int iteration, const rect_t &rect,
                                 Span<const TextureAtlas> tex_atlases, const BindlessTexData &bindless_tex,
                                 const Texture2D &out_img, const Buffer &out_rays, const Buffer &out_sh_rays,
                                 const Buffer &out_sky_rays, const Buffer &inout_counters,
                                 const Texture2D &out_base_color, const Texture2D &out_depth_normals);
    void kernel_ShadeSecondaryHits(CommandBuffer cmd_buf, const pass_settings_t &ps, eSpatialCacheMode cache_usage,
                                   float clamp_direct, const environment_t &env, const Buffer &indir_args,
                                   int indir_args_index, const Buffer &hits, const Buffer &rays,
                                   const scene_data_t &sc_data, const Buffer &rand_seq, uint32_t rand_seed,
                                   int iteration, Span<const TextureAtlas> tex_atlases,
                                   const BindlessTexData &bindless_tex, const Texture2D &out_img,
                                   const Buffer &out_rays, const Buffer &out_sh_rays, const Buffer &out_sky_rays,
                                   const Buffer &inout_counters, const Texture2D &out_depth_normals);
    void kernel_ShadeSky(CommandBuffer cmd_buf, const pass_settings_t &ps, float limit, const environment_t &env,
                         const Buffer &indir_args, int indir_args_index, const Buffer &hits, const Buffer &rays,
                         const Buffer &ray_indices, const Buffer &counters, const scene_data_t &sc_data, int iteration,
                         const Texture2D &out_img);
    void kernel_ShadeSkyPrimary(CommandBuffer cmd_buf, const pass_settings_t &ps, const environment_t &env,
                                const Buffer &indir_args, int indir_args_index, const Buffer &hits, const Buffer &rays,
                                const Buffer &ray_indices, const Buffer &counters, const scene_data_t &sc_data,
                                int iteration, const Texture2D &out_img);
    void kernel_ShadeSkySecondary(CommandBuffer cmd_buf, const pass_settings_t &ps, float clamp_direct,
                                  const environment_t &env, const Buffer &indir_args, int indir_args_index,
                                  const Buffer &hits, const Buffer &rays, const Buffer &ray_indices,
                                  const Buffer &counters, const scene_data_t &sc_data, int iteration,
                                  const Texture2D &out_img);
    void kernel_PrepareIndirArgs(CommandBuffer cmd_buf, const Buffer &inout_counters, const Buffer &out_indir_args);
    void kernel_MixIncremental(CommandBuffer cmd_buf, float mix_factor, float half_mix_factor, const rect_t &rect,
                               int iteration, float exposure, const Texture2D &temp_img,
                               const Texture2D &temp_base_color, const Texture2D &temp_depth_normals,
                               const Texture2D &req_samples, const Texture2D &out_full_img,
                               const Texture2D &out_half_img, const Texture2D &out_base_color,
                               const Texture2D &out_depth_normals);
    void kernel_Postprocess(CommandBuffer cmd_buf, const Texture2D &full_buf, const Texture2D &half_buf,
                            float inv_gamma, const rect_t &rect, float variance_threshold, int iteration,
                            const Texture2D &out_pixels, const Texture2D &out_variance,
                            const Texture2D &out_req_samples) const;
    void kernel_FilterVariance(CommandBuffer cmd_buf, const Texture2D &img_buf, const rect_t &rect,
                               float variance_threshold, int iteration, const Texture2D &out_variance,
                               const Texture2D &out_req_samples);
    void kernel_NLMFilter(CommandBuffer cmd_buf, const Texture2D &img_buf, const Texture2D &var_buf, float alpha,
                          float damping, const Texture2D &base_color_img, float base_color_weight,
                          const Texture2D &depth_normals_img, float depth_normals_weight, const Texture2D &out_raw_img,
                          eViewTransform view_transform, float inv_gamma, const rect_t &rect, const Texture2D &out_img);
    void kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels, const Texture2D &img_buf1,
                            const Texture2D &img_buf2, const Texture2D &img_buf3, const Sampler &sampler,
                            const rect_t &rect, int w, int h, const Buffer &weights, uint32_t weights_offset,
                            uint32_t biases_offset, const Buffer &out_buf, uint32_t output_offset, int output_stride,
                            const Texture2D &out_debug_img = {});
    void kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels, const Buffer &input_buf,
                            uint32_t input_offset, int input_stride, const rect_t &rect, int w, int h,
                            const Buffer &weights, uint32_t weights_offset, uint32_t biases_offset,
                            const Buffer &out_buf, uint32_t output_offset, int output_stride, bool downsample,
                            const Texture2D &out_debug_img = {});
    void kernel_Convolution(CommandBuffer cmd_buf, int in_channels, int out_channels, const Buffer &input_buf,
                            uint32_t input_offset, int input_stride, float inv_gamma, const rect_t &rect, int w, int h,
                            const Buffer &weights, uint32_t weights_offset, uint32_t biases_offset,
                            const Texture2D &out_img, const Texture2D &out_tonemapped_img);
    void kernel_ConvolutionConcat(CommandBuffer cmd_buf, int in_channels1, int in_channels2, int out_channels,
                                  const Buffer &input_buf1, uint32_t input_offset1, int input_stride1, bool upscale1,
                                  const Buffer &input_buf2, uint32_t input_offset2, int input_stride2,
                                  const rect_t &rect, int w, int h, const Buffer &weights, uint32_t weights_offset,
                                  uint32_t biases_offset, const Buffer &out_buf, uint32_t output_offset,
                                  int output_stride, const Texture2D &out_debug_img = {});
    void kernel_ConvolutionConcat(CommandBuffer cmd_buf, int in_channels1, int in_channels2, int out_channels,
                                  const Buffer &input_buf1, uint32_t input_offset1, int input_stride1, bool upscale1,
                                  const Texture2D &img_buf1, const Texture2D &img_buf2, const Texture2D &img_buf3,
                                  const Sampler &sampler, const rect_t &rect, int w, int h, const Buffer &weights,
                                  uint32_t weights_offset, uint32_t biases_offset, const Buffer &out_buf,
                                  uint32_t output_offset, int output_stride, const Texture2D &out_debug_img = {});
    void kernel_SortHashRays(CommandBuffer cmd_buf, const Buffer &indir_args, const Buffer &rays,
                             const Buffer &counters, const float root_min[3], const float cell_size[3],
                             const Buffer &out_hashes);
    void kernel_SortInitCountTable(CommandBuffer cmd_buf, int shift, const Buffer &indir_args, int indir_args_index,
                                   const Buffer &hashes, const Buffer &counters, int counter_index,
                                   const Buffer &out_count_table);
    void kernel_SortReduce(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index, const Buffer &input,
                           const Buffer &counters, int counter_index, const Buffer &out_reduce_table);
    void kernel_SortScan(CommandBuffer cmd_buf, const Buffer &input, const Buffer &counters, int counter_index,
                         const Buffer &out_scan_values);
    void kernel_SortScanAdd(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index, const Buffer &input,
                            const Buffer &scratch, const Buffer &counters, int counter_index,
                            const Buffer &out_scan_values);
    void kernel_SortScatter(CommandBuffer cmd_buf, int shift, const Buffer &indir_args, int indir_args_index,
                            const Buffer &hashes, const Buffer &sum_table, const Buffer &counters, int counter_index,
                            const Buffer &out_chunks);
    void kernel_SortReorderRays(CommandBuffer cmd_buf, const Buffer &indir_args, int indir_args_index,
                                const Buffer &in_rays, const Buffer &indices, const Buffer &counters, int counter_index,
                                const Buffer &out_rays);
    void kernel_DebugRT(CommandBuffer cmd_buf, const scene_data_t &sc_data, uint32_t node_index, const Buffer &rays,
                        const Texture2D &out_pixels);
    void kernel_SpatialCacheUpdate(CommandBuffer cmd_buf, const cache_grid_params_t &params, const Buffer &indir_args,
                                   const int indir_args_index, const Buffer &counters, const int counter_index,
                                   const Buffer &inters, const Buffer &rays, const Buffer &cache_data,
                                   const Texture2D &radiance_img, const Texture2D &depth_normals_img,
                                   const Buffer &inout_entries, const Buffer &inout_voxels_curr,
                                   const Buffer &inout_lock_buf);
    void kernel_SpatialCacheResolve(CommandBuffer cmd_buf, const cache_grid_params_t &params,
                                    const Buffer &inout_entries, const Buffer &inout_voxels_curr,
                                    const Buffer &voxels_prev);

    void TransitionSceneResources(CommandBuffer cmd_buf, const scene_data_t &sc_data) const;
    void RadixSort(CommandBuffer cmd_buf, const Buffer &indir_args, Buffer hashes[2], Buffer &count_table,
                   const Buffer &counters, const Buffer &reduce_table);

    color_data_rgba_t get_pixels_ref(bool tonemap) const;

  public:
    Renderer(const settings_t &s, ILog *log);
    ~Renderer() override;

    eRendererType type() const override;

    ILog *log() const override { return ctx_->log(); }

    const char *device_name() const override;

    bool is_hwrt() const override { return use_hwrt_; }

    bool is_spatial_caching_enabled() const override { return use_spatial_cache_; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    color_data_rgba_t get_pixels_ref() const override { return get_pixels_ref(true); }
    color_data_rgba_t get_raw_pixels_ref() const override { return get_pixels_ref(false); }
    color_data_rgba_t get_aux_pixels_ref(eAUXBuffer buf) const override;

    const shl1_data_t *get_sh_data_ref() const override { return &sh_data_host_[0]; }

    GpuImage get_native_raw_pixels() const override;

    void set_native_raw_pixels_state(const eGPUResState state) override {
        raw_filtered_buf_.resource_state = eResState(state);
    }

    void set_command_buffer(const GpuCommandBuffer cmd_buf) {
        external_cmd_buf_ = cmd_buf;
        ctx_->frame_cpu_synced[cmd_buf.index] = false;
    }

    void Resize(int w, int h) override;
    void Clear(const color_rgba_t &c) override;

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase &scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;
    void DenoiseImage(int pass, const RegionContext &region) override;

    void UpdateSpatialCache(const SceneBase &scene, RegionContext &region) override;
    void ResolveSpatialCache(const SceneBase &scene,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) override;
    void ResetSpatialCache(const SceneBase &scene,
                           const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }

    void InitUNetFilter(bool alias_memory, unet_filter_properties_t &out_props) override;
};
} // namespace NS
} // namespace Ray

inline Ray::NS::Renderer::~Renderer() {
    pixel_readback_buf_.Unmap();
    if (base_color_readback_buf_) {
        base_color_readback_buf_.Unmap();
    }
    if (depth_normals_readback_buf_) {
        depth_normals_readback_buf_.Unmap();
    }
}

inline Ray::SceneBase *Ray::NS::Renderer::CreateScene() {
    return new NS::Scene(ctx_.get(), use_hwrt_, use_bindless_, use_tex_compression_, use_spatial_cache_);
}

inline void Ray::NS::Renderer::Resize(const int w, const int h) {
    if (w_ == w && h_ == h) {
        return;
    }

    const int num_pixels = w * h;

    Tex2DParams params;
    params.w = w;
    params.h = h;
    params.format = eTexFormat::RawRGBA32F;
    params.usage = eTexUsageBits::Sampled | eTexUsageBits::Storage | eTexUsageBits::Transfer;
    params.sampling.wrap = eTexWrap::ClampToEdge;

    temp_buf0_ = Texture2D{"Temp Image 0", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    temp_buf1_ = Texture2D{"Temp Image 1", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    full_buf_ = Texture2D{"Full Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    half_buf_ = Texture2D{"Half Image [1]", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    base_color_buf_ = Texture2D{"Base Color Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    temp_depth_normals_buf_ =
        Texture2D{"Temp Depth-Normals Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    depth_normals_buf_ =
        Texture2D{"Depth-Normals Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    final_buf_ = Texture2D{"Final Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    raw_filtered_buf_ =
        Texture2D{"Raw Filtered Final Image", ctx_.get(), params, ctx_->default_memory_allocs(), ctx_->log()};
    { // Texture that holds required sample count per pixel
        Tex2DParams uparams = params;
        uparams.format = eTexFormat::RawR16UI;
        required_samples_buf_ =
            Texture2D{"Required samples Image", ctx_.get(), uparams, ctx_->default_memory_allocs(), ctx_->log()};
    }

    { // Sampler with black border
        SamplingParams params;
        params.wrap = eTexWrap::ClampToBorder;
        zero_border_sampler_ = Sampler{ctx_.get(), params};
    }

    if (frame_pixels_) {
        pixel_readback_buf_.Unmap();
        frame_pixels_ = nullptr;
    }
    pixel_readback_buf_ = Buffer{"Px Readback Buf", ctx_.get(), eBufType::Readback,
                                 uint32_t(round_up(4 * w * sizeof(float), TextureDataPitchAlignment) * h)};
    frame_pixels_ = (const color_rgba_t *)pixel_readback_buf_.Map(true /* persistent */);

    if (base_color_readback_buf_) {
        base_color_readback_buf_.Unmap();
        base_color_pixels_ = nullptr;
    }
    base_color_readback_buf_ = Buffer{"Base Color Stage Buf", ctx_.get(), eBufType::Readback,
                                      uint32_t(round_up(4 * w * sizeof(float), TextureDataPitchAlignment) * h)};
    base_color_pixels_ = (const color_rgba_t *)base_color_readback_buf_.Map(true /* persistent */);
    if (depth_normals_readback_buf_) {
        depth_normals_readback_buf_.Unmap();
        depth_normals_pixels_ = nullptr;
    }
    depth_normals_readback_buf_ = Buffer{"Depth Normals Stage Buf", ctx_.get(), eBufType::Readback,
                                         uint32_t(round_up(4 * w * sizeof(float), TextureDataPitchAlignment) * h)};
    depth_normals_pixels_ = (const color_rgba_t *)depth_normals_readback_buf_.Map(true /* persistent */);

    prim_rays_buf_ =
        Buffer{"Primary Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_data_t) * num_pixels)};
    secondary_rays_buf_ =
        Buffer{"Secondary Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_data_t) * num_pixels)};
    shadow_rays_buf_ =
        Buffer{"Shadow Rays", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::shadow_ray_t) * num_pixels)};
    prim_hits_buf_ =
        Buffer{"Primary Hits", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::hit_data_t) * num_pixels)};

    ray_hashes_bufs_[0] =
        Buffer{"Ray Hashes #0", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_hash_t) * num_pixels)};
    ray_hashes_bufs_[1] =
        Buffer{"Ray Hashes #1", ctx_.get(), eBufType::Storage, uint32_t(sizeof(Types::ray_hash_t) * num_pixels)};

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int blocks_count = (num_pixels + BlockSize - 1) / BlockSize;

    count_table_buf_ = Buffer{"Count Table", ctx_.get(), eBufType::Storage,
                              uint32_t(sizeof(uint32_t) * SORT_BINS_COUNT * blocks_count)};

    const int reduce_blocks_count = (blocks_count + BlockSize - 1) / BlockSize;
    reduce_table_buf_ = Buffer{"Reduce Table", ctx_.get(), eBufType::Storage,
                               uint32_t(sizeof(uint32_t) * SORT_BINS_COUNT * reduce_blocks_count)};

    if (use_spatial_cache_) {
        const int cache_w = w / RAD_CACHE_DOWNSAMPLING_FACTOR, cache_h = h / RAD_CACHE_DOWNSAMPLING_FACTOR;
        temp_cache_data_buf_ = Buffer{"Temp Cache Data", ctx_.get(), eBufType::Storage,
                                      uint32_t(sizeof(cache_data_t) * cache_w * cache_h)};
        if (!ctx_->int64_atomics_supported()) {
            temp_lock_buf_ = Buffer{"Temp Lock Buffer", ctx_.get(), eBufType::Storage,
                                    uint32_t(sizeof(uint32_t) * HASH_GRID_CACHE_ENTRIES_COUNT)};
        }
    }

    w_ = w;
    h_ = h;

    if (unet_tensors_heap_) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        UpdateUNetFilterMemory(cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    Clear(color_rgba_t{});
}

inline void Ray::NS::Renderer::UpdateFilterTable(CommandBuffer cmd_buf, const ePixelFilter filter, float filter_width) {
    float (*filter_func)(float v, float width);
    switch (filter) {
    case ePixelFilter::Box:
        filter_func = filter_box;
        filter_width = 1.0f;
        break;
    case ePixelFilter::Gaussian:
        filter_func = filter_gaussian;
        filter_width *= 3.0f;
        break;
    case ePixelFilter::BlackmanHarris:
        filter_func = filter_blackman_harris;
        filter_width *= 2.0f;
        break;
    default:
        assert(false && "Unknown filter!");
    }

    // TODO: Avoid unnecessary copy
    const std::vector<float> filter_table =
        Ray::CDFInverted(FILTER_TABLE_SIZE, 0.0f, filter_width * 0.5f,
                         std::bind(filter_func, std::placeholders::_1, filter_width), true /* make_symmetric */);

    Buffer stage_buf("Filter Table Stage", ctx_.get(), eBufType::Upload, FILTER_TABLE_SIZE * sizeof(float));
    { // Update stage buffer
        uint8_t *stage_data = stage_buf.Map();
        memcpy(stage_data, filter_table.data(), FILTER_TABLE_SIZE * sizeof(float));
        stage_buf.Unmap();
    }

    filter_table_ = Buffer{"Filter Table", ctx_.get(), eBufType::Storage, FILTER_TABLE_SIZE * sizeof(float)};

    CopyBufferToBuffer(stage_buf, 0, filter_table_, 0, FILTER_TABLE_SIZE * sizeof(float), cmd_buf);
}

inline void Ray::NS::Renderer::InitUNetFilter(const bool alias_memory, unet_filter_properties_t &out_props) {
    CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    Buffer temp_upload_buf;

    if (use_fp16_) {
        const int total_count = SetupUNetWeights<uint16_t>(false, 8, nullptr, nullptr);

        temp_upload_buf =
            Buffer{"UNet Weights CBN Upload", ctx_.get(), eBufType::Upload, uint32_t(total_count * sizeof(uint16_t))};
        unet_weights_ =
            Buffer{"UNet Weights CBN", ctx_.get(), eBufType::Storage, uint32_t(total_count * sizeof(uint16_t))};

        uint16_t *out_weights = (uint16_t *)temp_upload_buf.Map();
        SetupUNetWeights(false, 8, &unet_offsets_, out_weights);
        temp_upload_buf.Unmap();

        CopyBufferToBuffer(temp_upload_buf, 0, unet_weights_, 0, sizeof(uint16_t) * total_count, cmd_buf);
    } else {
        const int total_count = SetupUNetWeights<float>(false, 8, nullptr, nullptr);

        temp_upload_buf =
            Buffer{"UNet Weights CBN Upload", ctx_.get(), eBufType::Upload, uint32_t(total_count * sizeof(float))};
        unet_weights_ =
            Buffer{"UNet Weights CBN", ctx_.get(), eBufType::Storage, uint32_t(total_count * sizeof(float))};

        float *out_weights = (float *)temp_upload_buf.Map();
        SetupUNetWeights(false, 8, &unet_offsets_, out_weights);
        temp_upload_buf.Unmap();

        CopyBufferToBuffer(temp_upload_buf, 0, unet_weights_, 0, sizeof(float) * total_count, cmd_buf);
    }

    const TransitionInfo res_transitions[] = {{&unet_weights_, eResState::ShaderResource}};
    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

    unet_alias_memory_ = alias_memory;
    UpdateUNetFilterMemory(cmd_buf);

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);

    temp_upload_buf.FreeImmediate();

    unet_offsets_.enc_conv0_weight *= el_sz;
    unet_offsets_.enc_conv0_bias *= el_sz;
    unet_offsets_.enc_conv1_weight *= el_sz;
    unet_offsets_.enc_conv1_bias *= el_sz;
    unet_offsets_.enc_conv2_weight *= el_sz;
    unet_offsets_.enc_conv2_bias *= el_sz;
    unet_offsets_.enc_conv3_weight *= el_sz;
    unet_offsets_.enc_conv3_bias *= el_sz;
    unet_offsets_.enc_conv4_weight *= el_sz;
    unet_offsets_.enc_conv4_bias *= el_sz;
    unet_offsets_.enc_conv5a_weight *= el_sz;
    unet_offsets_.enc_conv5a_bias *= el_sz;
    unet_offsets_.enc_conv5b_weight *= el_sz;
    unet_offsets_.enc_conv5b_bias *= el_sz;
    unet_offsets_.dec_conv4a_weight *= el_sz;
    unet_offsets_.dec_conv4a_bias *= el_sz;
    unet_offsets_.dec_conv4b_weight *= el_sz;
    unet_offsets_.dec_conv4b_bias *= el_sz;
    unet_offsets_.dec_conv3a_weight *= el_sz;
    unet_offsets_.dec_conv3a_bias *= el_sz;
    unet_offsets_.dec_conv3b_weight *= el_sz;
    unet_offsets_.dec_conv3b_bias *= el_sz;
    unet_offsets_.dec_conv2a_weight *= el_sz;
    unet_offsets_.dec_conv2a_bias *= el_sz;
    unet_offsets_.dec_conv2b_weight *= el_sz;
    unet_offsets_.dec_conv2b_bias *= el_sz;
    unet_offsets_.dec_conv1a_weight *= el_sz;
    unet_offsets_.dec_conv1a_bias *= el_sz;
    unet_offsets_.dec_conv1b_weight *= el_sz;
    unet_offsets_.dec_conv1b_bias *= el_sz;
    unet_offsets_.dec_conv0_weight *= el_sz;
    unet_offsets_.dec_conv0_bias *= el_sz;

    out_props.pass_count = UNetFilterPasses;
    for (int i = 0; i < UNetFilterPasses; ++i) {
        std::fill(&out_props.alias_dependencies[i][0], &out_props.alias_dependencies[i][0] + 4, -1);
        for (int j = 0; j < int(unet_alias_dependencies_[i].size()); ++j) {
            out_props.alias_dependencies[i][j] = unet_alias_dependencies_[i][j];
        }
    }

    if (!pi_convolution_Img_9_32_) {
        if (!InitUNetFilterPipelines()) {
            throw std::runtime_error("Error initializing pipeline!");
        }
    }
}

inline void Ray::NS::Renderer::UpdateUNetFilterMemory(CommandBuffer cmd_buf) {
    unet_tensors_heap_ = {};
    if (!unet_weights_) {
        return;
    }

    const int el_sz = use_fp16_ ? sizeof(uint16_t) : sizeof(float);

    const int required_memory =
        SetupUNetFilter(w_, h_, unet_alias_memory_, true, unet_tensors_, unet_alias_dependencies_);
    unet_tensors_heap_ = Buffer{"UNet Tensors", ctx_.get(), eBufType::Storage, uint32_t(required_memory * el_sz)};

    if (use_fp16_) {
#ifndef NDEBUG
        const uint32_t fill_val = (f32_to_f16(NAN) << 16) | f32_to_f16(NAN);
#else
        const uint32_t fill_val = 0;
#endif
        unet_tensors_heap_.Fill(0, required_memory * el_sz, fill_val, cmd_buf);
    } else {
#ifndef NDEBUG
        const float fill_val = NAN;
#else
        const float fill_val = 0.0f;
#endif
        unet_tensors_heap_.Fill(0, required_memory * el_sz, reinterpret_cast<const uint32_t &>(fill_val), cmd_buf);
    }

    unet_tensors_.enc_conv0_offset *= el_sz;
    unet_tensors_.enc_conv0_size *= el_sz;
    unet_tensors_.pool1_offset *= el_sz;
    unet_tensors_.pool1_size *= el_sz;
    unet_tensors_.pool2_offset *= el_sz;
    unet_tensors_.pool2_size *= el_sz;
    unet_tensors_.pool3_offset *= el_sz;
    unet_tensors_.pool3_size *= el_sz;
    unet_tensors_.pool4_offset *= el_sz;
    unet_tensors_.pool4_size *= el_sz;
    unet_tensors_.enc_conv5a_offset *= el_sz;
    unet_tensors_.enc_conv5a_size *= el_sz;
    unet_tensors_.upsample4_offset *= el_sz;
    unet_tensors_.upsample4_size *= el_sz;
    unet_tensors_.dec_conv4a_offset *= el_sz;
    unet_tensors_.dec_conv4a_size *= el_sz;
    unet_tensors_.upsample3_offset *= el_sz;
    unet_tensors_.upsample3_size *= el_sz;
    unet_tensors_.dec_conv3a_offset *= el_sz;
    unet_tensors_.dec_conv3a_size *= el_sz;
    unet_tensors_.upsample2_offset *= el_sz;
    unet_tensors_.upsample2_size *= el_sz;
    unet_tensors_.dec_conv2a_offset *= el_sz;
    unet_tensors_.dec_conv2a_size *= el_sz;
    unet_tensors_.upsample1_offset *= el_sz;
    unet_tensors_.upsample1_size *= el_sz;
    unet_tensors_.dec_conv1a_offset *= el_sz;
    unet_tensors_.dec_conv1a_size *= el_sz;
    unet_tensors_.dec_conv1b_offset *= el_sz;
    unet_tensors_.dec_conv1b_size *= el_sz;
}

inline void Ray::NS::Renderer::TransitionSceneResources(CommandBuffer cmd_buf, const scene_data_t &sc_data) const {
    SmallVector<TransitionInfo, 16> res_transitions;

    for (const auto &tex_atlas : sc_data.tex_atlases) {
        if (tex_atlas.resource_state != eResState::ShaderResource) {
            res_transitions.emplace_back(&tex_atlas, eResState::ShaderResource);
        }
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
    if (!sc_data.lights.empty() && sc_data.lights.gpu_buf().resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.lights.gpu_buf(), eResState::ShaderResource);
    }
    if (sc_data.li_indices && sc_data.li_indices.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.li_indices, eResState::ShaderResource);
    }
    if (sc_data.env_qtree.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.env_qtree, eResState::ShaderResource);
    }
    if (sc_data.atmosphere_params.resource_state != eResState::UniformBuffer) {
        res_transitions.emplace_back(&sc_data.atmosphere_params, eResState::UniformBuffer);
    }
    if (sc_data.transmittance_lut.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.transmittance_lut, eResState::ShaderResource);
    }
    if (sc_data.multiscatter_lut.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.multiscatter_lut, eResState::ShaderResource);
    }
    if (sc_data.moon_tex.ready() && sc_data.moon_tex.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.moon_tex, eResState::ShaderResource);
    }
    if (sc_data.weather_tex.ready() && sc_data.weather_tex.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.weather_tex, eResState::ShaderResource);
    }
    if (sc_data.cirrus_tex.ready() && sc_data.cirrus_tex.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.cirrus_tex, eResState::ShaderResource);
    }
    if (sc_data.curl_tex.ready() && sc_data.curl_tex.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.curl_tex, eResState::ShaderResource);
    }
    if (sc_data.noise3d_tex.handle() && sc_data.noise3d_tex.resource_state != eResState::ShaderResource) {
        res_transitions.emplace_back(&sc_data.noise3d_tex, eResState::ShaderResource);
    }

    TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);
}

inline void Ray::NS::Renderer::RadixSort(CommandBuffer cmd_buf, const Buffer &indir_args, Buffer _hashes[2],
                                         Buffer &count_table, const Buffer &counters, const Buffer &reduce_table) {
    DebugMarker _(ctx_.get(), cmd_buf, "Radix Sort");

    static const char *MarkerStrings[] = {"Radix Sort Iter #0 [Bits   0-4]", "Radix Sort Iter #1 [Bits   4-8]",
                                          "Radix Sort Iter #2 [Bits  8-12]", "Radix Sort Iter #3 [Bits 12-16]",
                                          "Radix Sort Iter #4 [Bits 16-20]", "Radix Sort Iter #5 [Bits 20-24]",
                                          "Radix Sort Iter #6 [Bits 24-28]", "Radix Sort Iter #7 [Bits 28-32]"};

    Buffer *hashes[] = {&_hashes[0], &_hashes[1]};
    for (int shift = 0; shift < 32; shift += 4) {
        DebugMarker _(ctx_.get(), cmd_buf, MarkerStrings[shift / 4]);

        kernel_SortInitCountTable(cmd_buf, shift, indir_args, 6, *hashes[0], counters, 6, count_table);

        kernel_SortReduce(cmd_buf, indir_args, 7, count_table, counters, 7, reduce_table);

        kernel_SortScan(cmd_buf, reduce_table, counters, 7, reduce_table);

        kernel_SortScanAdd(cmd_buf, indir_args, 7, count_table, reduce_table, counters, 7, count_table);

        kernel_SortScatter(cmd_buf, shift, indir_args, 6, *hashes[0], count_table, counters, 6, *hashes[1]);

        std::swap(hashes[0], hashes[1]);
    }
    assert(hashes[0] == &_hashes[0]);
}
