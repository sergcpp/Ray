#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include "../RendererBase.h"

namespace ray {
namespace ocl {
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

    cl::Context context_;
    cl::Program program_;

    cl::CommandQueue queue_;

    cl::Kernel prim_rays_gen_kernel_,
    intersect_tris_kernel_, intersect_boxes_kernel_, intersect_cones_kernel_,
    texture_debug_page_kernel_,
    shade_primary_kernel_, shade_secondary_kernel_, trace_primary_rays_kernel_,
    trace_secondary_rays_kernel_, mix_incremental_kernel_, post_process_kernel_;

    cl::Buffer prim_rays_buf_, prim_inters_buf_, color_table_buf_,
    secondary_rays_buf_, secondary_rays_count_buf_;

    int w_, h_;

    int iteration_;
    std::vector<uint16_t> permutations_;

    cl::Buffer halton_seq_buf_;

    cl::Image2D temp_buf_, clean_buf_, final_buf_;

    std::vector<float> frame_pixels_;

    bool kernel_GeneratePrimaryRays(cl_int iteration, const ray::ocl::camera_t &cam, const cl::Buffer &halton, cl_int w, cl_int h, const cl::Buffer &out_rays);
    bool kernel_IntersectTris(const cl::Buffer &rays, cl_int rays_count, const cl::Buffer &tris, cl_int tris_count, const cl::Buffer &intersections, const cl::Buffer &intersections_counter);
    bool kernel_IntersectCones(const cl::Buffer &rays, cl_int rays_count, const cl::Buffer &cones, cl_int cones_count, const cl::Buffer &intersections, const cl::Buffer &intersections_counter);
    bool kernel_IntersectBoxes(const cl::Buffer &rays, cl_int rays_count, const cl::Buffer &boxes, cl_int boxes_count, const cl::Buffer &intersections, const cl::Buffer &intersections_counter);
    bool kernel_TextureDebugPage(const cl::Image2DArray &textures, cl_int page, const cl::Image2D &frame_buf);
    bool kernel_ShadePrimary(cl_int iteration, const cl::Buffer &halton,
                             const cl::Buffer &intersections, const cl::Buffer &rays,
                             int w, int h,
                             const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
                             const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
                             const cl::Buffer &nodes, cl_uint node_index,
                             const cl::Buffer &tris, const cl::Buffer &tri_indices,
                             const environment_t &env, const cl::Buffer &materials,
                             const cl::Buffer &textures, const cl::Image2DArray &texture_atlas, const cl::Image2D &frame_buf,
                             const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count);
    bool kernel_ShadeSecondary(cl_int iteration, const cl::Buffer &halton,
                               const cl::Buffer &intersections, const cl::Buffer &rays,
                               cl_int rays_count, int w, int h,
                               const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
                               const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
                               const cl::Buffer &nodes, cl_uint node_index,
                               const cl::Buffer &tris, const cl::Buffer &tri_indices,
                               const environment_t &env, const cl::Buffer &materials,
                               const cl::Buffer &textures, const cl::Image2DArray &texture_atlas, const cl::Image2D &frame_buf, const cl::Image2D &frame_buf2,
                               const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count);
    bool kernel_TracePrimaryRays(const cl::Buffer &rays, cl_int w, cl_int h,
                                 const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                 const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_TraceSecondaryRays(const cl::Buffer &rays, cl_int rays_count,
                                   const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                   const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections);
    bool kernel_MixIncremental(const cl::Image2D &fbuf1, const cl::Image2D &fbuf2, cl_float k, const cl::Image2D &res);
    bool kernel_Postprocess(const cl::Image2D &frame_buf, cl_int w, cl_int h, const cl::Image2D &out_pixels);

    bool UpdateHaltonSequence();
public:
    Renderer(int w, int h);
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

    void GetStats(stats_t &st) override;
};
}
}
