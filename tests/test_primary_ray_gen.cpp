#include "test_common.h"

#include <iostream>
#include <string>

#include "../internal/CoreRef.h"
#if !defined(__ANDROID__)
#include "../internal/RendererSSE2.h"
#include "../internal/RendererAVX.h"
#include "../internal/RendererAVX2.h"
#if !defined(DISABLE_OCL)
#include "../internal/CoreOCL.h"
#include "../internal/RendererOCL.h"
#endif
#endif

#include "../internal/simd/detect.h"

void test_primary_ray_gen() {
    extern float g_primary_ray_gen_test_data[];

    auto features = Ray::GetCpuFeatures();

    Ray::camera_t cam;

    const float o[] = { 0, 0, 4 },
                      d[] = { 0, 0, -1 };

    Ray::ConstructCamera(Ray::Persp, Ray::Box, o, d, 53.13f, 2.2f, 1.0f, 0.0f, &cam);

    std::vector<float> dummy_halton(Ray::HALTON_SEQ_LEN * 2, 0.0f);

    {
        // test reference
        Ray::aligned_vector<Ray::Ref::ray_packet_t> rays;
        Ray::Ref::GeneratePrimaryRays(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 16);
        for (int i = 0; i < 16; i++) {
            require(rays[i].xy == (int)g_primary_ray_gen_test_data[i * 23]);
            require(rays[i].o[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 1]));
            require(rays[i].o[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 2]));
            require(rays[i].o[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 3]));
            require(rays[i].d[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 4]));
            require(rays[i].d[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 5]));
            require(rays[i].d[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 6]));
            require(rays[i].c[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 7]));
            require(rays[i].c[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 8]));
            require(rays[i].c[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 9]));
            require(rays[i].ior == Approx(g_primary_ray_gen_test_data[i * 23 + 10]));
            require(rays[i].do_dx[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 11]));
            require(rays[i].do_dx[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 12]));
            require(rays[i].do_dx[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 13]));
            require(rays[i].dd_dx[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 14]));
            require(rays[i].dd_dx[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 15]));
            require(rays[i].dd_dx[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 16]));
            require(rays[i].do_dy[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 17]));
            require(rays[i].do_dy[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 18]));
            require(rays[i].do_dy[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 19]));
            require(rays[i].dd_dy[0] == Approx(g_primary_ray_gen_test_data[i * 23 + 20]));
            require(rays[i].dd_dy[1] == Approx(g_primary_ray_gen_test_data[i * 23 + 21]));
            require(rays[i].dd_dy[2] == Approx(g_primary_ray_gen_test_data[i * 23 + 22]));
        }
    }
    
    if (features.sse2_supported) {
#if !defined(__ANDROID__)
        // test Sse
        Ray::aligned_vector<Ray::Sse2::ray_packet_t<Ray::Sse2::RayPacketSize>> rays;
        Ray::Sse2::GeneratePrimaryRays<Ray::Sse2::RayPacketDimX, Ray::Sse2::RayPacketDimY>(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 4);

        int i = 0;
        for (int y = 0; y < 4; y += Ray::Sse2::RayPacketDimY) {
            for (int x = 0; x < 4; x += Ray::Sse2::RayPacketDimX) {
                int i1[Ray::Sse2::RayPacketSize];
                memcpy(&i1[0], &rays[i].xy[0], sizeof(int) * Ray::Sse2::RayPacketSize);
                require(i1[0] == (int)g_primary_ray_gen_test_data[(y * 4 + x) * 23 + 0]);
                require(i1[1] == (int)g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + 0]);
                require(i1[2] == (int)g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + 0]);
                require(i1[3] == (int)g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + 0]);

#define CHECK_r1_4(off) \
    require(r1[0] == Approx(g_primary_ray_gen_test_data[(y * 4 + x) * 23 + off]));              \
    require(r1[1] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + off]));          \
    require(r1[2] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + off]));        \
    require(r1[3] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + off]))

                float r1[Ray::Sse2::RayPacketSize];

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].o[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(1 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].d[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(4 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].c[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(7 + j);
                }

                memcpy(&r1[0], &rays[i].ior, sizeof(float) * Ray::Sse2::RayPacketSize);
                CHECK_r1_4(10);

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dx[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(11 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dx[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(14 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dy[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(17 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dy[j], sizeof(float) * Ray::Sse2::RayPacketSize);
                    CHECK_r1_4(20 + j);
                }

#undef CHECK_r1_4

                i++;
            }
        }
#endif
    } else {
        std::cout << "Cannot test SSE" << std::endl;
    }
    
    if (features.avx_supported) {
#if !defined(__ANDROID__)
        // test Avx
        Ray::aligned_vector<Ray::Avx::ray_packet_t<Ray::Avx::RayPacketSize>> rays;
        Ray::Avx::GeneratePrimaryRays<Ray::Avx::RayPacketDimX, Ray::Avx::RayPacketDimY>(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 2);

        int i = 0;
        for (int y = 0; y < 4; y += Ray::Avx::RayPacketDimY) {
            for (int x = 0; x < 4; x += Ray::Avx::RayPacketDimX) {
                int i1[Ray::Avx::RayPacketSize];
                memcpy(&i1[0], &rays[i].xy[0], sizeof(int) * Ray::Avx::RayPacketSize);
                require(i1[0] == int(g_primary_ray_gen_test_data[(y * 4 + x) * 23 + 0]));
                require(i1[1] == int(g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + 0]));
                require(i1[2] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + 0]));
                require(i1[3] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + 0]));
                require(i1[4] == int(g_primary_ray_gen_test_data[(y * 4 + x + 2) * 23 + 0]));
                require(i1[5] == int(g_primary_ray_gen_test_data[(y * 4 + x + 3) * 23 + 0]));
                require(i1[6] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 23 + 0]));
                require(i1[7] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 23 + 0]));

#define CHECK_r1_8(off) \
    require(r1[0] == Approx(g_primary_ray_gen_test_data[(y * 4 + x) * 23 + off]));            \
    require(r1[1] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + off]));        \
    require(r1[2] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + off]));      \
    require(r1[3] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + off]));  \
    require(r1[4] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 2) * 23 + off]));        \
    require(r1[5] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 3) * 23 + off]));        \
    require(r1[6] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 23 + off]));  \
    require(r1[7] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 23 + off]))

                float r1[Ray::Avx::RayPacketSize];

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].o[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(1 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].d[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(4 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].c[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(7 + j);
                }

                memcpy(&r1[0], &rays[i].ior, sizeof(float) * Ray::Avx::RayPacketSize);
                CHECK_r1_8(10);

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dx[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(11 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dx[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(14 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dy[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(17 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dy[j], sizeof(float) * Ray::Avx::RayPacketSize);
                    CHECK_r1_8(20 + j);
                }

#undef CHECK_r1_8

                i++;
            }
        }
#endif
    } else {
        std::cout << "Cannot test AVX" << std::endl;
    }

    if (features.avx2_supported) {
#if !defined(__ANDROID__)
        // test Avx
        Ray::aligned_vector<Ray::Avx2::ray_packet_t<Ray::Avx2::RayPacketSize>> rays;
        Ray::Avx2::GeneratePrimaryRays<Ray::Avx2::RayPacketDimX, Ray::Avx2::RayPacketDimY>(0, cam, { 0, 0, 4, 4 }, 4, 4, &dummy_halton[0], rays);

        require(rays.size() == 2);

        int i = 0;
        for (int y = 0; y < 4; y += Ray::Avx2::RayPacketDimY) {
            for (int x = 0; x < 4; x += Ray::Avx2::RayPacketDimX) {
                int i1[Ray::Avx2::RayPacketSize];
                memcpy(&i1[0], &rays[i].xy[0], sizeof(int) * Ray::Avx2::RayPacketSize);
                require(i1[0] == int(g_primary_ray_gen_test_data[(y * 4 + x) * 23 + 0]));
                require(i1[1] == int(g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + 0]));
                require(i1[2] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + 0]));
                require(i1[3] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + 0]));
                require(i1[4] == int(g_primary_ray_gen_test_data[(y * 4 + x + 2) * 23 + 0]));
                require(i1[5] == int(g_primary_ray_gen_test_data[(y * 4 + x + 3) * 23 + 0]));
                require(i1[6] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 23 + 0]));
                require(i1[7] == int(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 23 + 0]));

#define CHECK_r1_8(off) \
    require(r1[0] == Approx(g_primary_ray_gen_test_data[(y * 4 + x) * 23 + off]));            \
    require(r1[1] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 1) * 23 + off]));        \
    require(r1[2] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x) * 23 + off]));      \
    require(r1[3] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 1) * 23 + off]));  \
    require(r1[4] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 2) * 23 + off]));        \
    require(r1[5] == Approx(g_primary_ray_gen_test_data[(y * 4 + x + 3) * 23 + off]));        \
    require(r1[6] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 2) * 23 + off]));  \
    require(r1[7] == Approx(g_primary_ray_gen_test_data[((y + 1) * 4 + x + 3) * 23 + off]))

                float r1[Ray::Avx2::RayPacketSize];

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].o[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(1 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].d[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(4 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].c[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(7 + j);
                }

                memcpy(&r1[0], &rays[i].ior, sizeof(float) * Ray::Avx2::RayPacketSize);
                CHECK_r1_8(10);

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dx[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(11 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dx[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(14 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].do_dy[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(17 + j);
                }

                for (int j = 0; j < 3; j++) {
                    memcpy(&r1[0], &rays[i].dd_dy[j], sizeof(float) * Ray::Avx2::RayPacketSize);
                    CHECK_r1_8(20 + j);
                }

#undef CHECK_r1_8

                i++;
            }
        }
#endif
    } else {
        std::cout << "Cannot test AVX2" << std::endl;
    }
    
    {
#if defined(__ANDROID__) || defined(DISABLE_OCL)
         std::cout << "Skipping OpenCL test" << std::endl;
#else
        // test OpenCL
        class TestRenderer : public Ray::Ocl::Renderer {
        public:
            TestRenderer() : Ray::Ocl::Renderer(4, 4) {
                std::vector<float> dummy_halton(Ray::HALTON_SEQ_LEN * 2);
                cl_int error = queue_.enqueueWriteBuffer(halton_seq_buf_, CL_TRUE, 0, sizeof(float) * Ray::HALTON_SEQ_LEN * 2, &dummy_halton[0]);
                require(error == CL_SUCCESS);

                // override host_no_access with host_read_only to check results
                prim_rays_buf_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(Ray::Ocl::ray_packet_t) * w_ * h_, nullptr, &error);
            }

            void Test(const Ray::camera_t &cam, const float test_data[]) {
                auto cl_cam = Ray::Ocl::camera_t{ cam };

                require(kernel_GeneratePrimaryRays(0, cl_cam, { 0, 0, w_, h_ }, w_, h_, halton_seq_buf_, prim_rays_buf_));

                std::vector<Ray::Ocl::ray_packet_t> rays(w_ * h_);
                cl_int error = queue_.enqueueReadBuffer(prim_rays_buf_, CL_TRUE, 0, rays.size() * sizeof(Ray::Ocl::ray_packet_t), &rays[0]);
                require(error == CL_SUCCESS);

                require(rays.size() == 16);
                for (int i = 0; i < 16; i++) {
                    int x = int(rays[i].o.w), y = int(rays[i].d.w);

                    int xy = int(test_data[i * 23 + 0]);
                    require(x == ((xy >> 16) & 0x0000ffff));
                    require(y == (xy & 0x0000ffff));

                    require(rays[i].o.x == Approx(test_data[i * 23 + 1]));
                    require(rays[i].o.y == Approx(test_data[i * 23 + 2]));
                    require(rays[i].o.z == Approx(test_data[i * 23 + 3]));
                    require(rays[i].d.x == Approx(test_data[i * 23 + 4]));
                    require(rays[i].d.y == Approx(test_data[i * 23 + 5]));
                    require(rays[i].d.z == Approx(test_data[i * 23 + 6]));
                    require(rays[i].c.x == Approx(test_data[i * 23 + 7]));
                    require(rays[i].c.y == Approx(test_data[i * 23 + 8]));
                    require(rays[i].c.z == Approx(test_data[i * 23 + 9]));
                    require(rays[i].c.w == Approx(test_data[i * 23 + 10]));

                    require(rays[i].do_dx.x == Approx(test_data[i * 23 + 11]));
                    require(rays[i].do_dx.y == Approx(test_data[i * 23 + 12]));
                    require(rays[i].do_dx.z == Approx(test_data[i * 23 + 13]));
                    require(rays[i].dd_dx.x == Approx(test_data[i * 23 + 14]));
                    require(rays[i].dd_dx.y == Approx(test_data[i * 23 + 15]));
                    require(rays[i].dd_dx.z == Approx(test_data[i * 23 + 16]));
                    require(rays[i].do_dy.x == Approx(test_data[i * 23 + 17]));
                    require(rays[i].do_dy.y == Approx(test_data[i * 23 + 18]));
                    require(rays[i].do_dy.z == Approx(test_data[i * 23 + 19]));
                    require(rays[i].dd_dy.x == Approx(test_data[i * 23 + 20]));
                    require(rays[i].dd_dy.y == Approx(test_data[i * 23 + 21]));
                    require(rays[i].dd_dy.z == Approx(test_data[i * 23 + 22]));
                }
            }
        };

        try {
            TestRenderer r;
            r.Test(cam, g_primary_ray_gen_test_data);
        } catch (std::runtime_error &e) {
            std::cout << "Failed to test OpenCL: " << e.what() << std::endl;
        }
#endif
    }
}