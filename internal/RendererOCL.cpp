#include "RendererOCL.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <random>
#include <string>
#include <utility>

#include "CoreOCL.h"
#include "Halton.h"
#include "SceneOCL.h"
#include "TextureAtlasOCL.h"

namespace ray {
namespace ocl {
const char *cl_src_types =
#include "kernels/types.cl"
    ;
const char *cl_src_primary_ray_gen =
#include "kernels/primary_ray_gen.cl"
    ;
const char *cl_src_intersect =
#include "kernels/intersect.cl"
    ;
const char *cl_src_traverse =
#include "kernels/traverse_bvh.cl"
    ;
const char *cl_src_trace =
#include "kernels/trace.cl"
    ;
const char *cl_src_texturing =
#include "kernels/texture.cl"
    ;
const char *cl_src_shade =
#include "kernels/shade.cl"
    ;
const char *cl_src_postprocess =
#include "kernels/postprocess.cl"
    ;
const char *cl_src_transform =
#include "kernels/transform.cl"
    ;
}
}

ray::ocl::Renderer::Renderer(int w, int h) : w_(w), h_(h), loaded_halton_(-1) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) throw std::runtime_error("Cannot create OpenCL renderer!");

    size_t platform_index = 0;
    for (size_t i = 0; i < platforms.size(); i++) {
        auto s = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        if (s.find("NVIDIA") != std::string::npos || s.find("AMD") != std::string::npos) {
            platform_index = i;
            break;
        }
    }

    platform_ = cl::Platform::setDefault(platforms[platform_index]);

    std::vector<cl::Device> devices;
    platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) throw std::runtime_error("Cannot create OpenCL renderer!");

    device_ = cl::Device::setDefault(devices[0]);
    if (device_ != devices[0]) throw std::runtime_error("Cannot create OpenCL renderer!");

    // get properties
    max_compute_units_ = device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    max_clock_ = device_.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    mem_size_ = device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    max_work_group_size_ = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    {
        // create context
        cl_int error = CL_SUCCESS;
        context_ = cl::Context(devices, nullptr, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        queue_ = cl::CommandQueue(context_, device_, cl::QueueProperties::None, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
    }

    {
        // load kernels

        std::string cl_src_defines;
        cl_src_defines += "#define TRI_W_BITS " + std::to_string(TRI_W_BITS) + "\n";
        cl_src_defines += "#define TRI_AXIS_ALIGNED_BIT " + std::to_string(TRI_AXIS_ALIGNED_BIT) + "\n";
        cl_src_defines += "#define HIT_BIAS " + std::to_string(HIT_BIAS) + "f\n";
        cl_src_defines += "#define HIT_EPS " + std::to_string(HIT_EPS) + "f\n";
        cl_src_defines += "#define FLT_EPS " + std::to_string(FLT_EPS) + "f\n";
        cl_src_defines += "#define PI " + std::to_string(PI) + "f\n";
        cl_src_defines += "#define HaltonSeqLen " + std::to_string(HaltonSeqLen) + "\n";
        cl_src_defines += "#define MAX_MIP_LEVEL " + std::to_string(MAX_MIP_LEVEL) + "\n";
        cl_src_defines += "#define NUM_MIP_LEVELS " + std::to_string(NUM_MIP_LEVELS) + "\n";
        cl_src_defines += "#define MAX_TEXTURE_SIZE " + std::to_string(MAX_TEXTURE_SIZE) + "\n";
        cl_src_defines += "#define MAX_MATERIAL_TEXTURES " + std::to_string(MAX_MATERIAL_TEXTURES) + "\n";
        cl_src_defines += "#define DiffuseMaterial " + std::to_string(DiffuseMaterial) + "\n";
        cl_src_defines += "#define GlossyMaterial " + std::to_string(GlossyMaterial) + "\n";
        cl_src_defines += "#define RefractiveMaterial " + std::to_string(RefractiveMaterial) + "\n";
        cl_src_defines += "#define EmissiveMaterial " + std::to_string(EmissiveMaterial) + "\n";
        cl_src_defines += "#define MixMaterial " + std::to_string(MixMaterial) + "\n";
        cl_src_defines += "#define TransparentMaterial " + std::to_string(TransparentMaterial) + "\n";
        cl_src_defines += "#define MAIN_TEXTURE " + std::to_string(MAIN_TEXTURE) + "\n";
        cl_src_defines += "#define NORMALS_TEXTURE " + std::to_string(NORMALS_TEXTURE) + "\n";
        cl_src_defines += "#define MIX_MAT1 " + std::to_string(MIX_MAT1) + "\n";
        cl_src_defines += "#define MIX_MAT2 " + std::to_string(MIX_MAT2) + "\n";

        cl_int error = CL_SUCCESS;
        cl::Program::Sources srcs = {
            cl_src_defines,
            cl_src_types, cl_src_transform, cl_src_primary_ray_gen,
            cl_src_intersect, cl_src_traverse, cl_src_trace,
            cl_src_texturing, cl_src_shade, cl_src_postprocess
        };

        program_ = cl::Program(context_, srcs, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        std::string build_opts = "-Werror -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math ";// = "-cl-opt-disable ";

        struct stat info = { 0 };
        if (stat("./.dumps", &info) == 0 && info.st_mode & S_IFDIR) {
            build_opts += "-save-temps=./.dumps/ ";
        }

        error = program_.build(build_opts.c_str());
        if (error == CL_INVALID_BUILD_OPTIONS) {
            // -cl-strict-aliasing not supported sometimes, try to build without it
            build_opts = "-Werror -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math ";
            program_ = cl::Program(context_, srcs, &error);
            if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
            error = program_.build(build_opts.c_str());
        }

        if (error != CL_SUCCESS) {
            auto build_log = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
#if defined(_MSC_VER)
            __debugbreak();
#endif
            throw std::runtime_error("Cannot create OpenCL renderer!");
        }

        prim_rays_gen_kernel_ = cl::Kernel(program_, "GeneratePrimaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        texture_debug_page_kernel_ = cl::Kernel(program_, "TextureDebugPage", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        shade_primary_kernel_ = cl::Kernel(program_, "ShadePrimary", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        shade_secondary_kernel_ = cl::Kernel(program_, "ShadeSecondary", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        trace_primary_rays_kernel_ = cl::Kernel(program_, "TracePrimaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        trace_secondary_rays_kernel_ = cl::Kernel(program_, "TraceSecondaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        mix_incremental_kernel_ = cl::Kernel(program_, "MixIncremental", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        post_process_kernel_ = cl::Kernel(program_, "PostProcess", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

#if !defined(NDEBUG)
        cl::Kernel types_check = cl::Kernel(program_, "TypesCheck", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        char buf[512];
        int argc = 0;
        if (types_check.setArg(argc++, sizeof(ray_packet_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(ocl::camera_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(tri_accel_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(aabox_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(hit_data_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(bvh_node_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(vertex_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(mesh_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(mesh_instance_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(transform_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(texture_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(material_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(environment_t), buf) != CL_SUCCESS) {
#if defined(_MSC_VER)
            __debugbreak();
#endif
            throw std::runtime_error("Cannot create OpenCL renderer!");
        }
#endif
    }

    {
        // create buffers
        cl_int error = CL_SUCCESS;
        Resize(w, h);

        //secondary_inters_count_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        //if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        secondary_rays_count_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        std::vector<pixel_color_t> color_table;
        /*for (int i = 0; i < 256; i++) {
            color_table.push_back({ ray::U_0_p1(), ray::U_0_p1(), ray::U_0_p1(), 1 });
        }*/
        for (int i = 0; i < 64; i++) {
            color_table.push_back( { float(i) / 63, float(i) / 63, float(i) / 63, 1 });
        }

        color_table_buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(pixel_color_t) * color_table.size(), nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        error = queue_.enqueueWriteBuffer(color_table_buf_, CL_TRUE, 0, sizeof(pixel_color_t) * color_table.size(), &color_table[0]);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        halton_seq_buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(float) * HaltonSeqLen * 2, nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
    }

    auto rand_func = std::bind(std::uniform_int_distribution<int>(), std::mt19937(0));
    permutations_ = ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

void ray::ocl::Renderer::Resize(int w, int h) {
    cl_int error = CL_SUCCESS;
    prim_rays_buf_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS, sizeof(ray_packet_t) * w * h, nullptr, &error);
    secondary_rays_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(ray_packet_t) * w * h, nullptr, &error);
    prim_inters_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(hit_data_t) * w * h, nullptr, &error);
    temp_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    clean_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    final_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

    frame_pixels_.resize((size_t)4 * w * h);

    w_ = w;
    h_ = h;
}

void ray::ocl::Renderer::Clear(const pixel_color_t &c) {
    static_assert(sizeof(pixel_color_t) == sizeof(cl_float4), "!");
    queue_.enqueueFillImage(clean_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
    queue_.enqueueFillImage(final_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
}

std::shared_ptr<ray::SceneBase> ray::ocl::Renderer::CreateScene() {
    return std::make_shared<ocl::Scene>(context_, queue_);
}

void ray::ocl::Renderer::RenderScene(const std::shared_ptr<SceneBase> &_s, RegionContext &region) {
    auto s = std::dynamic_pointer_cast<ocl::Scene>(_s);
    if (!s) return;

    region.iteration++;
    if (!region.halton_seq || region.iteration % HaltonSeqLen == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    if (region.iteration != loaded_halton_) {
        if (CL_SUCCESS != queue_.enqueueWriteBuffer(halton_seq_buf_, CL_TRUE, 0, sizeof(float) * HaltonSeqLen * 2, &region.halton_seq[0])) {
            return;
        }
        loaded_halton_ = region.iteration;
    }

    const auto &cam = s->GetCamera(s->current_cam());

    ray::ocl::camera_t cl_cam = { cam };

    cl_cam.up.x *= float(h_) / w_;
    cl_cam.up.y *= float(h_) / w_;
    cl_cam.up.z *= float(h_) / w_;

    if (!kernel_GeneratePrimaryRays((cl_int)region.iteration, cl_cam, halton_seq_buf_, w_, h_, prim_rays_buf_)) return;

    {
        cl_int error = CL_SUCCESS;

        if (!kernel_TracePrimaryRays(prim_rays_buf_, w_, h_,
                                     s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                                     s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;

        cl_int secondary_rays_count = 0;
        if (queue_.enqueueWriteBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                      &secondary_rays_count) != CL_SUCCESS) return;

        if (!kernel_ShadePrimary((cl_int)region.iteration, halton_seq_buf_,
                                 prim_inters_buf_, prim_rays_buf_, w_, h_,
                                 s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(),
                                 s->transforms_.buf(), s->vtx_indices_.buf(), s->vertices_.buf(),
                                 s->nodes_.buf(), (cl_uint)s->macro_nodes_start_,
                                 s->tris_.buf(), s->tri_indices_.buf(),
                                 s->env_, s->materials_.buf(), s->textures_.buf(), s->texture_atlas_.atlas(), temp_buf_,
                                 secondary_rays_buf_, secondary_rays_count_buf_)) return;

        if (queue_.enqueueReadBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                     &secondary_rays_count) != CL_SUCCESS) return;

        for (int depth = 0; depth < MAX_BOUNCES && secondary_rays_count; depth++) {
            if (!kernel_TraceSecondaryRays(secondary_rays_buf_, secondary_rays_count,
                                           s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                                           s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;

            cl_int new_secondary_rays_count = 0;
            if (queue_.enqueueWriteBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                          &new_secondary_rays_count) != CL_SUCCESS) return;

#if 0
            pixel_color_t c = { 0, 0, 0, 0 };
            queue_.enqueueFillImage(temp_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
#endif

            if (queue_.enqueueCopyImage(temp_buf_, final_buf_, { 0, 0, 0 }, { 0, 0, 0 },
        { (size_t)w_, (size_t)h_, 1 }) != CL_SUCCESS) return;

            if (!kernel_ShadeSecondary((cl_int)region.iteration, halton_seq_buf_,
                                       prim_inters_buf_, secondary_rays_buf_, secondary_rays_count, w_, h_,
                                       s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(),
                                       s->transforms_.buf(), s->vtx_indices_.buf(), s->vertices_.buf(),
                                       s->nodes_.buf(), (cl_uint)s->macro_nodes_start_,
                                       s->tris_.buf(), s->tri_indices_.buf(),
                                       s->env_, s->materials_.buf(), s->textures_.buf(), s->texture_atlas_.atlas(), final_buf_, temp_buf_,
                                       prim_rays_buf_, secondary_rays_count_buf_)) return;

            if (queue_.enqueueReadBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                         &secondary_rays_count) != CL_SUCCESS) return;

            std::swap(final_buf_, temp_buf_);
            std::swap(secondary_rays_buf_, prim_rays_buf_);
        }

        float k = 1.0f / region.iteration;

        if (!kernel_MixIncremental(clean_buf_, temp_buf_, (cl_float)k, final_buf_)) return;
        std::swap(final_buf_, clean_buf_);

        if (!kernel_Postprocess(clean_buf_, w_, h_, final_buf_)) return;

        error = queue_.enqueueReadImage(final_buf_, CL_TRUE, {}, { (size_t)w_, (size_t)h_, 1 }, 0, 0, &frame_pixels_[0]);
    }
}

void ray::ocl::Renderer::GetStats(stats_t &st) {
    
}

bool ray::ocl::Renderer::kernel_GeneratePrimaryRays(const cl_int iteration, const ray::ocl::camera_t &cam, const cl::Buffer &halton, cl_int w, cl_int h, const cl::Buffer &out_rays) {
    cl_uint argc = 0;
    if (prim_rays_gen_kernel_.setArg(argc++, iteration) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, cam) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, out_rays) != CL_SUCCESS) {
        return false;
    }
    return CL_SUCCESS == queue_.enqueueNDRangeKernel(prim_rays_gen_kernel_, cl::NullRange, cl::NDRange { (size_t)w, (size_t)h });
}

bool ray::ocl::Renderer::kernel_TextureDebugPage(const cl::Image2DArray &textures, cl_int page, const cl::Image2D &frame_buf) {
    cl_uint argc = 0;
    if (texture_debug_page_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            texture_debug_page_kernel_.setArg(argc++, page) != CL_SUCCESS ||
            texture_debug_page_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS) {
        return false;
    }

    auto w = frame_buf.getImageInfo<CL_IMAGE_WIDTH>(),
         h = frame_buf.getImageInfo<CL_IMAGE_HEIGHT>();

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(texture_debug_page_kernel_, cl::NullRange, cl::NDRange { (size_t)w, (size_t)h });
}

bool ray::ocl::Renderer::kernel_ShadePrimary(const cl_int iteration, const cl::Buffer &halton,
        const cl::Buffer &intersections, const cl::Buffer &rays,
        int w, int h,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
        const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
        const cl::Buffer &nodes, cl_uint node_index,
        const cl::Buffer &tris, const cl::Buffer &tri_indices,
        const environment_t &env, const cl::Buffer &materials,
        const cl::Buffer &textures, const cl::Image2DArray &texture_atlas, const cl::Image2D &frame_buf,
        const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count) {
    cl_uint argc = 0;
    if (shade_primary_kernel_.setArg(argc++, iteration) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, intersections) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, env) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, materials) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, texture_atlas) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, secondary_rays) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, secondary_rays_count) != CL_SUCCESS) {
        return false;
    }

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(shade_primary_kernel_, cl::NullRange, cl::NDRange { (size_t)w, (size_t)h });
}

bool ray::ocl::Renderer::kernel_ShadeSecondary(const cl_int iteration, const cl::Buffer &halton,
        const cl::Buffer &intersections, const cl::Buffer &rays,
        cl_int rays_count, int w, int h,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
        const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
        const cl::Buffer &nodes, cl_uint node_index,
        const cl::Buffer &tris, const cl::Buffer &tri_indices,
        const environment_t &env, const cl::Buffer &materials,
        const cl::Buffer &textures, const cl::Image2DArray &texture_atlas, const cl::Image2D &frame_buf, const cl::Image2D &frame_buf2,
        const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count) {
    if (rays_count == 0) return true;

    cl_uint argc = 0;
    if (shade_secondary_kernel_.setArg(argc++, iteration) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, intersections) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, env) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, materials) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, texture_atlas) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, frame_buf2) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, secondary_rays) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, secondary_rays_count) != CL_SUCCESS) {
        return false;
    }

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(shade_secondary_kernel_, cl::NullRange, cl::NDRange { (size_t)rays_count });
}

bool ray::ocl::Renderer::kernel_TracePrimaryRays(const cl::Buffer &rays, cl_int w, cl_int h, const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_primary_rays_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    // local group size 8x8 seems to be optimal for traversing BVH in most scenes

    int border_x = w % 8, border_y = h % 8;

    cl::NDRange global = { (size_t)(w - border_x), (size_t)(h - border_y) };
    cl::NDRange local = { (size_t)8, std::min((size_t)8, max_work_group_size_ / 8) };

    if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, cl::NullRange, global, local) != CL_SUCCESS) {
        return false;
    }

    if (border_x) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { (size_t)(w - border_x), 0 }, { (size_t)(border_x), (size_t)(h - border_y) }) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_y) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { 0, (size_t)(h - border_y) }, { (size_t)(w), (size_t)(border_y) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool ray::ocl::Renderer::kernel_TraceSecondaryRays(const cl::Buffer &rays, cl_int rays_count,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_secondary_rays_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            trace_secondary_rays_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(rays_count) };
    cl::NDRange local = cl::NullRange;

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(trace_secondary_rays_kernel_, cl::NullRange, global, local);
}

bool ray::ocl::Renderer::kernel_MixIncremental(const cl::Image2D &fbuf1, const cl::Image2D &fbuf2, cl_float k, const cl::Image2D &res) {
    cl_uint argc = 0;
    if (mix_incremental_kernel_.setArg(argc++, fbuf1) != CL_SUCCESS ||
            mix_incremental_kernel_.setArg(argc++, fbuf2) != CL_SUCCESS ||
            mix_incremental_kernel_.setArg(argc++, k) != CL_SUCCESS ||
            mix_incremental_kernel_.setArg(argc++, res) != CL_SUCCESS) {
        return false;
    }

    const auto w = fbuf1.getImageInfo<CL_IMAGE_WIDTH>(),
               h = fbuf1.getImageInfo<CL_IMAGE_HEIGHT>();

    cl::NDRange global = { (size_t)w, (size_t)h };
    cl::NDRange local = cl::NullRange;

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(mix_incremental_kernel_, cl::NullRange, global, local);
}

bool ray::ocl::Renderer::kernel_Postprocess(const cl::Image2D &frame_buf, cl_int w, cl_int h, const cl::Image2D &out_pixels) {
    cl_uint argc = 0;
    if (post_process_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, h) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, out_pixels) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)w, (size_t)h };
    cl::NDRange local = cl::NullRange;//{ (size_t)8, std::min((size_t)8, max_work_group_size_ / 8) };

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(post_process_kernel_, cl::NullRange, global, local);
}

void ray::ocl::Renderer::UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HaltonSeqLen * 2]);
    }

    for (int i = 0; i < HaltonSeqLen; i++) {
        seq[i * 2 + 0] = ray::ScrambledRadicalInverse<29>(&permutations_[100], (uint64_t)(iteration + i));
        seq[i * 2 + 1] = ray::ScrambledRadicalInverse<31>(&permutations_[129], (uint64_t)(iteration + i));
    }
}