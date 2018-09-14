#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include "Core.h"
#include "../RendererBase.h"

namespace Ray {
namespace Ocl {
struct camera_t;
struct environment_t;

class TextureAtlas;

class Renderer : public RendererBase {
protected:
    cl::Platform platform_;
    cl::Device device_;

    cl_uint max_compute_units_, max_clock_;
    cl_ulong mem_size_;
    size_t max_work_group_size_;
    size_t scan_portion_, seg_scan_portion_;
    size_t trace_group_size_x_, trace_group_size_y_;
    size_t max_image_buffer_size_;
    size_t tri_rast_x_, tri_rast_y_;
    size_t tri_bin_size_;

    cl::Context context_;
    cl::Program program_;

    cl::CommandQueue queue_;

    cl::Kernel prim_rays_gen_kernel_, sample_mesh_reset_bins_kernel_, sample_mesh_bin_stage_kernel_, sample_mesh_raster_stage_kernel_, 
    texture_debug_page_kernel_,
    shade_primary_kernel_, shade_secondary_kernel_, trace_primary_rays_kernel_, trace_primary_rays_img_kernel_,
    compute_ray_hashes_kernel_, set_head_flags_kernel_, excl_scan_kernel_,
    incl_scan_kernel_, add_partial_sums_kernel_, init_chunk_hash_and_base_kernel_,
    init_chunk_size_kernel_, init_skel_and_head_flags_kernel_, init_count_table_kernel_,
    write_sorted_chunks_kernel_, excl_seg_scan_kernel_, incl_seg_scan_kernel_, add_seg_partial_sums_kernel_,
    reorder_rays_kernel_, trace_secondary_rays_kernel_, trace_secondary_rays_img_kernel_, mix_incremental_kernel_, post_process_kernel_;

    cl::Buffer prim_rays_buf_, prim_inters_buf_, color_table_buf_,
    secondary_rays_buf_, secondary_rays_count_buf_;

    int w_, h_;

    std::vector<uint16_t> permutations_;
    int loaded_halton_;

    cl::Buffer halton_seq_buf_, ray_hashes_buf_, head_flags_buf_, scan_values_buf_, scan_values2_buf_,
               scan_values3_buf_, scan_values4_buf_, partial_sums_buf_, partial_sums2_buf_,
               partial_sums3_buf_, partial_sums4_buf_, chunks_buf_,
               chunks2_buf_, skeleton_buf_, counters_buf_, partial_flags_buf_, partial_flags2_buf_, partial_flags3_buf_, partial_flags4_buf_,
               tri_bin_buf_;

    cl::Image2D temp_buf_, clean_buf_, final_buf_;

    std::vector<float> frame_pixels_;

    stats_t stats_ = { 0 };

    bool kernel_GeneratePrimaryRays(cl_int iteration, const Ray::Ocl::camera_t &cam, const Ray::rect_t &rect, cl_int w, cl_int h, const cl::Buffer &halton, const cl::Buffer &out_rays);
    bool kernel_SampleMesh_ResetBins(cl_int w, cl_int h, const cl::Buffer &tri_bin_buf);
    bool kernel_SampleMesh_BinStage(cl_int uv_layer, uint32_t tris_index, uint32_t tris_count, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
                                    cl_int w, cl_int h, const cl::Buffer &tri_bin_buf);
    bool kernel_SampleMesh_RasterStage(cl_int uv_layer, cl_int iteration, cl_uint tr_index, const cl::Buffer &transforms,
                                       const cl::Buffer &vtx_indices, const cl::Buffer &vertices, cl_int w, cl_int h,
                                       const cl::Buffer &halton_seq, const cl::Buffer &tri_bin_buf,
                                       const cl::Buffer &out_rays, const cl::Buffer &out_inters);
    bool kernel_TextureDebugPage(const cl::Image2DArray &textures, cl_int page, const cl::Image2D &frame_buf);
    bool kernel_ShadePrimary(const pass_info_t pi, const cl::Buffer &halton, const Ray::rect_t &rect, cl_int w,
                             const cl::Buffer &intersections, const cl::Buffer &rays,
                             const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
                             const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
                             const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices,
                             const environment_t &env, const cl::Buffer &materials, const cl::Buffer &textures, 
                             const cl::Image2DArray &texture_atlas, const cl::Buffer &lights, const cl::Buffer &li_incies, cl_uint light_node_index,
                             const cl::Image2D &frame_buf, const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count);
    bool kernel_ShadeSecondary(const pass_info_t pi, const cl::Buffer &halton,
                               const cl::Buffer &intersections, const cl::Buffer &rays,
                               int rays_count, int w, int h,
                               const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
                               const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
                               const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices,
                               const environment_t &env, const cl::Buffer &materials, const cl::Buffer &textures, const cl::Image2DArray &texture_atlas,
                               const cl::Buffer &lights, const cl::Buffer &li_incies, cl_uint light_node_index,
                               const cl::Image2D &frame_buf, const cl::Image2D &frame_buf2,
                               const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count);
    bool kernel_TracePrimaryRays(const cl::Buffer &rays, const Ray::rect_t &rect, cl_int w,
                                 const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                 const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_TracePrimaryRaysImg(const cl::Buffer &rays, const Ray::rect_t &rect, cl_int w,
                                    const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                    const cl::Image1DBuffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_TraceSecondaryRays(const cl::Buffer &rays, cl_int rays_count,
                                   const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                   const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_TraceSecondaryRaysImg(const cl::Buffer &rays, cl_int rays_count,
                                      const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                      const cl::Image1DBuffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_ComputeRayHashes(const cl::Buffer &rays, cl_int rays_count, cl_float3 root_min, cl_float3 cell_size, const cl::Buffer &out_hashes);
    bool kernel_SetHeadFlags(const cl::Buffer &hashes, cl_int hashes_count, const cl::Buffer &out_head_flags);
    bool kernel_ExclusiveScan(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride, const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums);
    bool kernel_InclusiveScan(const cl::Buffer &values, cl_int count, const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums);
    bool kernel_ExclusiveSegScan(const cl::Buffer &values, const cl::Buffer &flags, cl_int count, const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums);
    bool kernel_InclusiveSegScan(const cl::Buffer &values, const cl::Buffer &flags, cl_int count, const cl::Buffer &out_scan_values,
                                 const cl::Buffer &out_partial_sums, const cl::Buffer &out_partial_flags);
    bool kernel_AddPartialSums(const cl::Buffer &values, cl_int count, const cl::Buffer &partial_sums);
    bool kernel_AddSegPartialSums(const cl::Buffer &flags, const cl::Buffer &values, const cl::Buffer &partial_sums, cl_int count, cl_int group_size);
    bool kernel_InitChunkHashAndBase(const cl::Buffer &chunks, cl_int count, const cl::Buffer &hash_values, const cl::Buffer &head_flags, const cl::Buffer &scan_values);
    bool kernel_InitChunkSize(const cl::Buffer &chunks, cl_int count, cl_int ray_count);
    bool kernel_InitSkeletonAndHeadFlags(const cl::Buffer &scan_values, const cl::Buffer &chunks, cl_int count, const cl::Buffer &skeleton, const cl::Buffer &head_flags);
    bool kernel_InitCountTable(const cl::Buffer &chunks, cl_int count, cl_int group_size, cl_int shift, const cl::Buffer &counters);
    bool kernel_WriteSortedChunks(const cl::Buffer &chunks_in, const cl::Buffer &offsets, const cl::Buffer &counts, cl_int count, cl_int shift, cl_int group_size, const cl::Buffer &chunks_out);
    bool kernel_ReorderRays(const cl::Buffer &in_rays, const cl::Buffer &in_indices, cl_int count, const cl::Buffer &out_rays);
    bool kernel_MixIncremental(const cl::Image2D &fbuf1, const cl::Image2D &fbuf2, cl_float k, const cl::Image2D &res);
    bool kernel_Postprocess(const cl::Image2D &frame_buf, cl_int w, cl_int h, cl_float gamma, const cl::Image2D &out_pixels);

    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

    bool ExclusiveScan_CPU(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride, const cl::Buffer &out_scan_values);
    bool ExclusiveScan_GPU(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride,
                           const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &scan_values2,
                           const cl::Buffer &scan_values3, const cl::Buffer &scan_values4, const cl::Buffer &out_scan_values);

    bool InclusiveSegScan_CPU(const cl::Buffer &flags, const cl::Buffer &values, cl_int count, const cl::Buffer &out_scan_values);
    bool InclusiveSegScan_GPU(const cl::Buffer &flags, const cl::Buffer &values, cl_int count, const cl::Buffer &partial_sums, 
                              const cl::Buffer &partial_sums2, const cl::Buffer &partial_sums3, const cl::Buffer &partial_sums4,
                              const cl::Buffer &partial_flags, const cl::Buffer &partial_flags2, const cl::Buffer &partial_flags3, 
                              const cl::Buffer &partial_flags4, const cl::Buffer &scan_values2, const cl::Buffer &scan_values3,
                              const cl::Buffer &scan_values4, const cl::Buffer &out_scan_values);

    bool ReorderRays_CPU(const cl::Buffer &scan_values, const cl::Buffer &rays, cl_int count);

    bool PerformRadixSort_CPU(const cl::Buffer &chunks, cl_int count);
    bool PerformRadixSort_GPU(const cl::Buffer &chunks, const cl::Buffer &chunks2, cl_int count, const cl::Buffer &counters,
                              const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &scan_values,
                              const cl::Buffer &scan_values2, const cl::Buffer &scan_values3, const cl::Buffer &scan_values4);

    bool SortRays(const cl::Buffer &in_rays, cl_int rays_count, cl_float3 root_min, cl_float3 cell_size, const cl::Buffer &ray_hashes,
                  const cl::Buffer &head_flags, const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &partial_sums3, const cl::Buffer &partial_sums4,
                  const cl::Buffer &partial_flags, const cl::Buffer &partial_flags2, const cl::Buffer &partial_flags3, const cl::Buffer &partial_flags4,
                  const cl::Buffer &scan_values, const cl::Buffer &scan_values2, const cl::Buffer &scan_values3, const cl::Buffer &scan_values4,
                  const cl::Buffer &chunks, const cl::Buffer &chunks2, const cl::Buffer &counters, const cl::Buffer &skeleton, const cl::Buffer &out_rays);
public:
    Renderer(int w, int h, int platform_index = -1, int device_index = -1);
    ~Renderer() override = default;

    eRendererType type() const override { return RendererOCL; }

    std::pair<int, int> size() const override {
        return std::make_pair(w_, h_);
    }

    const pixel_color_t *get_pixels_ref() const override {
        return (const pixel_color_t *)&frame_pixels_[0];
    }

    void Resize(int w, int h) override;
    void Clear(const pixel_color_t &c) override;

    std::shared_ptr<SceneBase> CreateScene() override;
    void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = { 0 }; }

    static std::vector<Platform> QueryPlatforms();
};
}
}
