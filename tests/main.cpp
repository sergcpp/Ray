
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <chrono>
#include <future>

#include "../Ray.h"
#include "thread_pool.h"

void test_simd();
void test_hashmap();
void test_hashset();
void test_huffman();
void test_inflate();
void test_scope_exit();
void test_freelist_alloc();
void test_small_vector();
void test_span();
void test_sparse_storage();
void test_tex_storage();

void test_aux_channels(const char *arch_list[], const char *preferred_device);
void test_ray_flags(const char *arch_list[], const char *preferred_device);
void test_two_sided_mat(const char *arch_list[], const char *preferred_device);
void test_oren_mat0(const char *arch_list[], const char *preferred_device);
void test_oren_mat1(const char *arch_list[], const char *preferred_device);
void test_oren_mat2(const char *arch_list[], const char *preferred_device);
void test_diff_mat0(const char *arch_list[], const char *preferred_device);
void test_diff_mat1(const char *arch_list[], const char *preferred_device);
void test_diff_mat2(const char *arch_list[], const char *preferred_device);
void test_sheen_mat0(const char *arch_list[], const char *preferred_device);
void test_sheen_mat1(const char *arch_list[], const char *preferred_device);
void test_sheen_mat2(const char *arch_list[], const char *preferred_device);
void test_sheen_mat3(const char *arch_list[], const char *preferred_device);
void test_glossy_mat0(const char *arch_list[], const char *preferred_device);
void test_glossy_mat1(const char *arch_list[], const char *preferred_device);
void test_glossy_mat2(const char *arch_list[], const char *preferred_device);
void test_spec_mat0(const char *arch_list[], const char *preferred_device);
void test_spec_mat1(const char *arch_list[], const char *preferred_device);
void test_spec_mat2(const char *arch_list[], const char *preferred_device);
void test_aniso_mat0(const char *arch_list[], const char *preferred_device);
void test_aniso_mat1(const char *arch_list[], const char *preferred_device);
void test_aniso_mat2(const char *arch_list[], const char *preferred_device);
void test_aniso_mat3(const char *arch_list[], const char *preferred_device);
void test_aniso_mat4(const char *arch_list[], const char *preferred_device);
void test_aniso_mat5(const char *arch_list[], const char *preferred_device);
void test_aniso_mat6(const char *arch_list[], const char *preferred_device);
void test_aniso_mat7(const char *arch_list[], const char *preferred_device);
void test_tint_mat0(const char *arch_list[], const char *preferred_device);
void test_tint_mat1(const char *arch_list[], const char *preferred_device);
void test_tint_mat2(const char *arch_list[], const char *preferred_device);
void test_plastic_mat0(const char *arch_list[], const char *preferred_device);
void test_plastic_mat1(const char *arch_list[], const char *preferred_device);
void test_plastic_mat2(const char *arch_list[], const char *preferred_device);
void test_metal_mat0(const char *arch_list[], const char *preferred_device);
void test_metal_mat1(const char *arch_list[], const char *preferred_device);
void test_metal_mat2(const char *arch_list[], const char *preferred_device);
void test_emit_mat0(const char *arch_list[], const char *preferred_device);
void test_emit_mat1(const char *arch_list[], const char *preferred_device);
void test_coat_mat0(const char *arch_list[], const char *preferred_device);
void test_coat_mat1(const char *arch_list[], const char *preferred_device);
void test_coat_mat2(const char *arch_list[], const char *preferred_device);
void test_refr_mis0(const char *arch_list[], const char *preferred_device);
void test_refr_mis1(const char *arch_list[], const char *preferred_device);
void test_refr_mis2(const char *arch_list[], const char *preferred_device);
void test_refr_mat0(const char *arch_list[], const char *preferred_device);
void test_refr_mat1(const char *arch_list[], const char *preferred_device);
void test_refr_mat2(const char *arch_list[], const char *preferred_device);
void test_refr_mat3(const char *arch_list[], const char *preferred_device);
void test_trans_mat0(const char *arch_list[], const char *preferred_device);
void test_trans_mat1(const char *arch_list[], const char *preferred_device);
void test_trans_mat2(const char *arch_list[], const char *preferred_device);
void test_trans_mat3(const char *arch_list[], const char *preferred_device);
void test_trans_mat4(const char *arch_list[], const char *preferred_device);
void test_trans_mat5(const char *arch_list[], const char *preferred_device);
void test_alpha_mat0(const char *arch_list[], const char *preferred_device);
void test_alpha_mat1(const char *arch_list[], const char *preferred_device);
void test_alpha_mat2(const char *arch_list[], const char *preferred_device);
void test_alpha_mat3(const char *arch_list[], const char *preferred_device);
void test_alpha_mat4(const char *arch_list[], const char *preferred_device);
void test_complex_mat0(const char *arch_list[], const char *preferred_device);
void test_complex_mat1(const char *arch_list[], const char *preferred_device);
void test_complex_mat2(const char *arch_list[], const char *preferred_device);
void test_complex_mat3(const char *arch_list[], const char *preferred_device);
void test_complex_mat4(const char *arch_list[], const char *preferred_device);
void test_complex_mat5(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_caching(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_clipped(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_adaptive(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_regions(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_nlm_filter(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_unet_filter(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_dof(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_mesh_lights(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_sphere_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_inside_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_spot_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_dir_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_sun_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_moon_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_hdri_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_nlm_filter(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_unet_filter(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_dof(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_mesh_lights(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_sphere_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_spot_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_dir_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_hdri_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat7_refractive(const char *arch_list[], const char *preferred_device);
void test_complex_mat7_principled(const char *arch_list[], const char *preferred_device);
void assemble_material_test_images(const char *arch_list[]);

bool g_stop_on_fail = false;
std::atomic_bool g_tests_success{true};
std::atomic_bool g_log_contains_errors{false};
bool g_catch_flt_exceptions = false;
bool g_determine_sample_count = false;
bool g_minimal_output = false;
bool g_nohwrt = false;
bool g_nodx = false;
int g_validation_level = 1;

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

bool InitAndDestroyFakeGLContext();
#endif

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; ++i) {
        printf("%s ", argv[i]);
    }
    printf("\n");

    printf("Ray Version: %s\n", Ray::Version());
    puts(" ---------------");

    using namespace std::chrono;

    const auto t1 = high_resolution_clock::now();

    bool full_tests = false, nogpu = false, nocpu = false, run_detail_tests_on_fail = false;
    const char *device_name = nullptr;
    const char *preferred_arch[] = {nullptr, nullptr};
    double time_limit_m = DBL_MAX;
    int threads_count = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nogpu") == 0) {
            nogpu = true;
        } else if (strcmp(argv[i], "--nocpu") == 0) {
            nocpu = true;
        } else if (strcmp(argv[i], "--nodx") == 0) {
            g_nodx = true;
        } else if (strcmp(argv[i], "--full") == 0) {
            full_tests = true;
        } else if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) && (++i != argc)) {
            device_name = argv[i];
        } else if (strcmp(argv[i], "--detail_on_fail") == 0) {
            run_detail_tests_on_fail = true;
        } else if (strcmp(argv[i], "--arch") == 0 && (++i != argc)) {
            preferred_arch[0] = argv[i];
        } else if (strcmp(argv[i], "-j") == 0 && (++i != argc)) {
            threads_count = atoi(argv[i]);
        } else if (strncmp(argv[i], "-j", 2) == 0) {
            threads_count = atoi(&argv[i][2]);
        } else if (strcmp(argv[i], "--time_limit") == 0 && (++i != argc)) {
            time_limit_m = atof(argv[i]);
#ifdef _WIN32
            SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
        } else if ((strcmp(argv[i], "--validation_level") == 0 || strcmp(argv[i], "-vl") == 0) && (++i != argc)) {
            g_validation_level = atoi(argv[i]);
        } else if (strcmp(argv[i], "--nohwrt") == 0) {
            g_nohwrt = true;
        }
    }

    if (g_determine_sample_count) {
        threads_count = 1;
    }

    g_minimal_output = true;

#if defined(_WIN32) && !defined(__clang__)
    const bool enable_fp_exceptions = !nocpu || full_tests;
    if (enable_fp_exceptions) {
        unsigned old_value;
        _controlfp_s(&old_value, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);
        g_catch_flt_exceptions = true;
    }
#endif

    test_simd();
    puts(" ---------------");
    test_freelist_alloc();
    test_hashset();
    test_hashmap();
    test_huffman();
    test_inflate();
    test_scope_exit();
    test_small_vector();
    test_span();
    test_sparse_storage();
    test_tex_storage();
    puts(" ---------------");

#ifdef _WIN32
    if (!nogpu) {
        // Stupid workaround that should not exist.
        // Make sure vulkan will be able to use discrete Intel GPU when dual Xe/Arc GPUs are available.
        InitAndDestroyFakeGLContext();
    }
#endif

    static const char *ArchListFull[] = {"REF", "SSE2", "SSE41", "AVX", "AVX2", "AVX512", "NEON", "VK", "DX", nullptr};
    static const char *ArchListFullNoGPU[] = {"REF", "SSE2", "SSE41", "AVX", "AVX2", "AVX512", "NEON", nullptr};
    static const char *ArchListDefault[] = {"AVX2", "NEON", "VK", "DX", nullptr};
#ifndef __APPLE__
    static const char *ArchListDefaultNoGPU[] = {"AVX2", "NEON", nullptr};
#else
    // NOTE: Rosetta doesn't support AVX, so we test SSE4.1 here
    static const char *ArchListDefaultNoGPU[] = {"SSE41", "NEON", nullptr};
#endif
    static const char *ArchListGPUOnly[] = {"DX", "VK", nullptr};

    bool detailed_material_tests_needed = full_tests;
    bool tests_success_final = g_tests_success;

    const char **arch_list = ArchListDefault;
    if (preferred_arch[0]) {
        arch_list = preferred_arch;
    } else if (nocpu) {
        arch_list = ArchListGPUOnly;
    } else if (full_tests) {
        if (!nogpu) {
            arch_list = ArchListFull;
        } else {
            arch_list = ArchListFullNoGPU;
        }
    } else if (nogpu) {
        arch_list = ArchListDefaultNoGPU;
    }

    ThreadPool mt_run_pool(threads_count);

    if (g_tests_success) {
        const auto t2 = high_resolution_clock::now();
        std::vector<std::future<void>> futures;

        futures.push_back(mt_run_pool.Enqueue(test_aux_channels, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_ray_flags, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_two_sided_mat, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat0, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat1, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat2, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat3, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat4, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_caching, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_clipped, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_adaptive, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_regions, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_nlm_filter, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_unet_filter, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_dof, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_mesh_lights, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_sphere_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_inside_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_spot_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_dir_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_sun_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_moon_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat5_hdri_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_nlm_filter, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_unet_filter, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_dof, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_mesh_lights, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_sphere_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_spot_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_dir_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat6_hdri_light, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat7_refractive, arch_list, device_name));
        futures.push_back(mt_run_pool.Enqueue(test_complex_mat7_principled, arch_list, device_name));

        for (auto &f : futures) {
            f.wait();
        }
        printf("Finished complex_mat tests in %.2f minutes\n",
               duration<double>(high_resolution_clock::now() - t2).count() / 60.0);

        // schedule detailed material tests if complex tests failed (to find out the reason)
        if (run_detail_tests_on_fail) {
            detailed_material_tests_needed |= !g_tests_success;
        }
        tests_success_final &= g_tests_success;
        g_tests_success = true;
    }

    if (detailed_material_tests_needed) {
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_oren_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_oren_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_oren_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished oren_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_diff_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_diff_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_diff_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished diff_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_sheen_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_sheen_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_sheen_mat2, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_sheen_mat3, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished sheen_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_glossy_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_glossy_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_glossy_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished glossy_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_spec_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_spec_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_spec_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished spec_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat2, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat3, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat4, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat5, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat6, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_aniso_mat7, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished aniso_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_metal_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_metal_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_metal_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished metal_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_plastic_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_plastic_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_plastic_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished plastic_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_tint_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_tint_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_tint_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished tint_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_emit_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_emit_mat1, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished emit_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_coat_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_coat_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_coat_mat2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished coat_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_refr_mis0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_refr_mis1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_refr_mis2, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished refr_mis tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_refr_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_refr_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_refr_mat2, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_refr_mat3, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished refr_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_trans_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_trans_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_trans_mat2, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_trans_mat3, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_trans_mat4, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_trans_mat5, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished trans_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts(" ---------------");
            std::vector<std::future<void>> futures;

            futures.push_back(mt_run_pool.Enqueue(test_alpha_mat0, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_alpha_mat1, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_alpha_mat2, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_alpha_mat3, arch_list, device_name));
            futures.push_back(mt_run_pool.Enqueue(test_alpha_mat4, arch_list, device_name));

            for (auto &f : futures) {
                f.wait();
            }
            printf("Finished alpha_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
    }

    const double test_duration_m = duration<double>(high_resolution_clock::now() - t1).count() / 60.0;

    assemble_material_test_images(arch_list);

    printf("FINISHED ALL TESTS in %.2f minutes\n", test_duration_m);

    if (g_log_contains_errors) {
        printf("LOG CONTAINS ERRORS!\n");
    }

    tests_success_final &= !g_log_contains_errors;
    tests_success_final &= g_tests_success;
    tests_success_final &= (test_duration_m <= time_limit_m);
    if (tests_success_final) {
        puts("SUCCESS");
    } else {
        puts("FAILED");
    }
    return tests_success_final ? 0 : -1;
}

//
// Dirty workaround for Intel discrete GPU
//
#ifdef _WIN32
extern "C" {
// Enable High Performance Graphics while using Integrated Graphics
__declspec(dllexport) int32_t NvOptimusEnablement = 1;                  // Nvidia
__declspec(dllexport) int32_t AmdPowerXpressRequestHighPerformance = 1; // AMD
}

bool InitAndDestroyFakeGLContext() {
    HWND fake_window = ::CreateWindowEx(NULL, NULL, "FakeWindow", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                        256, 256, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

    HDC fake_dc = GetDC(fake_window);

    PIXELFORMATDESCRIPTOR pixel_format = {};
    pixel_format.nSize = sizeof(pixel_format);
    pixel_format.nVersion = 1;
    pixel_format.dwFlags = PFD_SUPPORT_OPENGL;
    pixel_format.iPixelType = PFD_TYPE_RGBA;
    pixel_format.cColorBits = 24;
    pixel_format.cAlphaBits = 8;
    pixel_format.cDepthBits = 0;

    int pix_format_id = ChoosePixelFormat(fake_dc, &pixel_format);
    if (pix_format_id == 0) {
        printf("ChoosePixelFormat() failed\n");
        return false;
    }

    if (!SetPixelFormat(fake_dc, pix_format_id, &pixel_format)) {
        // printf("SetPixelFormat() failed (0x%08x)\n", GetLastError());
        return false;
    }

    HGLRC fake_rc = wglCreateContext(fake_dc);

    wglDeleteContext(fake_rc);
    ReleaseDC(fake_window, fake_dc);
    DestroyWindow(fake_window);

    return true;
}
#endif
