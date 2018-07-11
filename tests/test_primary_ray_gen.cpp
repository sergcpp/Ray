#include "test_common.h"

#include <iostream>
#include <string>

#include "../internal/CoreRef.h"
#if !defined(__ANDROID__)
#include "../internal/RendererSSE.h"
#include "../internal/RendererAVX.h"
#if !defined(DISABLE_OCL)
#include "../internal/CoreOCL.h"
#include "../internal/RendererOCL.h"
#endif
#endif

#include "../internal/simd/detect.h"

void test_primary_ray_gen() {
    extern std::vector<float> primary_ray_gen_test_data;

    auto features = ray::GetCpuFeatures();

    ray::camera_t cam;

    const float o[] = { 0, 0, 4 },
                      d[] = { 0, 0, -1 };

    ray::ConstructCamera(ray::Persp, o, d, 53.13f, 2.2f, &cam);

    std::vector<float> dummy_halton(ray::HaltonSeqLen * 2);

    {
        // test reference
        ray::aligned_vector<ray::ref::ray_packet_t> rays;
        ray::ref::GeneratePrimaryRays(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 16);
        for (int i = 0; i < 16; i++) {
            require(rays[i].id.id == primary_ray_gen_test_data[i * 7]);
            require(rays[i].o[0] == Approx(primary_ray_gen_test_data[i * 7 + 1]));
            require(rays[i].o[1] == Approx(primary_ray_gen_test_data[i * 7 + 2]));
            require(rays[i].o[2] == Approx(primary_ray_gen_test_data[i * 7 + 3]));
            require(rays[i].d[0] == Approx(primary_ray_gen_test_data[i * 7 + 4]));
            require(rays[i].d[1] == Approx(primary_ray_gen_test_data[i * 7 + 5]));
            require(rays[i].d[2] == Approx(primary_ray_gen_test_data[i * 7 + 6]));
        }
    }
    
    if (features.sse2_supported) {
#if !defined(__ANDROID__)
        // test sse
        ray::aligned_vector<ray::sse::ray_packet_t<ray::sse::RayPacketSize>> rays;
        ray::sse::GeneratePrimaryRays<ray::sse::RayPacketDimX, ray::sse::RayPacketDimY>(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 4);

        int i = 0;
        for (int y = 0; y < 4; y += ray::sse::RayPacketDimY) {
            for (int x = 0; x < 4; x += ray::sse::RayPacketDimX) {
                float r1[ray::sse::RayPacketSize];
                memcpy(&r1[0], &rays[i].o[0], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 1]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 1]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 1]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 1]));
                memcpy(&r1[0], &rays[i].o[1], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 2]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 2]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 2]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 2]));
                memcpy(&r1[0], &rays[i].o[2], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 3]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 3]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 3]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 3]));

                memcpy(&r1[0], &rays[i].d[0], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 4]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 4]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 4]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 4]));
                memcpy(&r1[0], &rays[i].d[1], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 5]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 5]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 5]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 5]));
                memcpy(&r1[0], &rays[i].d[2], sizeof(float) * ray::sse::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 6]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 6]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 6]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 6]));

                i++;
            }
        }
#endif
    } else {
        std::cout << "Cannot test SSE" << std::endl;
    }
    
    if (features.avx_supported) {
#if !defined(__ANDROID__)
        // test avx
        ray::aligned_vector<ray::avx::ray_packet_t<ray::avx::RayPacketSize>> rays;
        ray::avx::GeneratePrimaryRays<ray::avx::RayPacketDimX, ray::avx::RayPacketDimY>(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 2);

        int i = 0;
        for (int y = 0; y < 4; y += ray::avx::RayPacketDimY) {
            for (int x = 0; x < 4; x += ray::avx::RayPacketDimX) {
                float r1[ray::avx::RayPacketSize];
                memcpy(&r1[0], &rays[i].o[0], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 1]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 1]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 1]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 1]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 1]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 1]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 1]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 1]));
                memcpy(&r1[0], &rays[i].o[1], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 2]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 2]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 2]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 2]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 2]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 2]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 2]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 2]));
                memcpy(&r1[0], &rays[i].o[2], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 3]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 3]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 3]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 3]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 3]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 3]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 3]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 3]));

                memcpy(&r1[0], &rays[i].d[0], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 4]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 4]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 4]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 4]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 4]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 4]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 4]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 4]));
                memcpy(&r1[0], &rays[i].d[1], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 5]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 5]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 5]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 5]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 5]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 5]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 5]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 5]));
                memcpy(&r1[0], &rays[i].d[2], sizeof(float) * ray::avx::RayPacketSize);
                require(r1[0] == Approx(primary_ray_gen_test_data[(y * 4 + x) * 7 + 6]));
                require(r1[1] == Approx(primary_ray_gen_test_data[(y * 4 + x + 1) * 7 + 6]));
                require(r1[2] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x) * 7 + 6]));
                require(r1[3] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 7 + 6]));
                require(r1[4] == Approx(primary_ray_gen_test_data[(y * 4 + x + 2) * 7 + 6]));
                require(r1[5] == Approx(primary_ray_gen_test_data[(y * 4 + x + 3) * 7 + 6]));
                require(r1[6] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 7 + 6]));
                require(r1[7] == Approx(primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 7 + 6]));

                i++;
            }
        }
#endif
    } else {
        std::cout << "Cannot test AVX" << std::endl;
    }
    
    {
#if defined(__ANDROID__) || defined(DISABLE_OCL)
         std::cout << "Skipping OpenCL test" << std::endl;
#else
        // test OpenCL
        class TestRenderer : public ray::ocl::Renderer {
        public:
            TestRenderer() : ray::ocl::Renderer(4, 4) {
                std::vector<float> dummy_halton(ray::HaltonSeqLen * 2);
                cl_int error = queue_.enqueueWriteBuffer(halton_seq_buf_, CL_TRUE, 0, sizeof(float) * ray::HaltonSeqLen * 2, &dummy_halton[0]);
                require(error == CL_SUCCESS);

                // override host_no_access with host_read_only to check results
                prim_rays_buf_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(ray::ocl::ray_packet_t) * w_ * h_, nullptr, &error);
            }

            void Test(const ray::camera_t &cam, const std::vector<float> &test_data) {
                ray::ocl::camera_t cl_cam = { cam };

                require(kernel_GeneratePrimaryRays(0, cl_cam, { 0, 0, w_, h_ }, w_, h_, halton_seq_buf_, prim_rays_buf_));

                std::vector<ray::ocl::ray_packet_t> rays(w_ * h_);
                cl_int error = queue_.enqueueReadBuffer(prim_rays_buf_, CL_TRUE, 0, rays.size() * sizeof(ray::ocl::ray_packet_t), &rays[0]);
                require(error == CL_SUCCESS);

                require(rays.size() == 16);
                for (int i = 0; i < 16; i++) {
                    require(rays[i].o.x == Approx(primary_ray_gen_test_data[i * 7 + 1]));
                    require(rays[i].o.y == Approx(primary_ray_gen_test_data[i * 7 + 2]));
                    require(rays[i].o.z == Approx(primary_ray_gen_test_data[i * 7 + 3]));
                    require(rays[i].d.x == Approx(primary_ray_gen_test_data[i * 7 + 4]));
                    require(rays[i].d.y == Approx(primary_ray_gen_test_data[i * 7 + 5]));
                    require(rays[i].d.z == Approx(primary_ray_gen_test_data[i * 7 + 6]));
                }
            }
        };

        try {
            TestRenderer r;
            r.Test(cam, primary_ray_gen_test_data);
        } catch (std::runtime_error &e) {
            std::cout << "Failed to test OpenCL: " << e.what() << std::endl;
        }
#endif
    }
}