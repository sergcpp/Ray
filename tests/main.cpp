
#include <cfloat>
#include <cstdio>
#include <cstring>

#include <atomic>
#include <chrono>

void test_tex_storage();
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
void test_complex_mat5_dof(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_mesh_lights(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_sphere_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_spot_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_sun_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat5_hdr_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_dof(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_mesh_lights(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_sphere_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_spot_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_sun_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat6_hdr_light(const char *arch_list[], const char *preferred_device);
void test_complex_mat7_refractive(const char *arch_list[], const char *preferred_device);
void test_complex_mat7_principled(const char *arch_list[], const char *preferred_device);
void assemble_material_test_images(const char *arch_list[]);
void test_simd();
void test_mesh_lights();
void test_texture();

bool g_stop_on_fail = false;
bool g_tests_success = true;
std::atomic_bool g_log_contains_errors{false};
bool g_catch_flt_exceptions = false;
bool g_determine_sample_count = false;

#ifdef _WIN32
bool InitAndDestroyFakeGLContext();
#endif

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; ++i) {
        printf("%s ", argv[i]);
    }
    printf("\n");

    using namespace std::chrono;

    const auto t1 = high_resolution_clock::now();

    bool full_tests = false, nogpu = false, nocpu = false, run_detail_tests_on_fail = false;
    const char *device_name = nullptr;
    const char *preferred_arch[] = {nullptr, nullptr};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nogpu") == 0) {
            nogpu = true;
        } else if (strcmp(argv[i], "--nocpu") == 0) {
            nocpu = true;
        } else if (strcmp(argv[i], "--full") == 0) {
            full_tests = true;
        } else if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) && (++i != argc)) {
            device_name = argv[i];
        } else if (strcmp(argv[i], "--detail_on_fail") == 0) {
            run_detail_tests_on_fail = true;
        } else if (strcmp(argv[i], "--arch") == 0 && (++i != argc)) {
            preferred_arch[0] = argv[i];
        }
    }

#if defined(_WIN32) && !defined(__clang__)
    const bool enable_fp_exceptions = !nocpu || full_tests;
    if (enable_fp_exceptions) {
        _controlfp(_EM_INEXACT, _MCW_EM);
        g_catch_flt_exceptions = true;
    }
#endif

    test_simd();
    test_tex_storage();
    // test_mesh_lights();

#ifdef _WIN32
    // Stupid workaround that should not exist.
    // Make sure vulkan will be able to use discrete Intel GPU when dual Xe/Arc GPUs are available.
    InitAndDestroyFakeGLContext();
#endif

    static const char *ArchListFull[] = {"REF", "SSE2", "SSE41", "AVX", "AVX2", "AVX512", "NEON", "VK", nullptr};
    static const char *ArchListFullNoGPU[] = {"REF", "SSE2", "SSE41", "AVX", "AVX2", "AVX512", "NEON", nullptr};
    static const char *ArchListDefault[] = {"AVX2", "NEON", "VK", nullptr};
    static const char *ArchListDefaultNoGPU[] = {"AVX2", "NEON", nullptr};
    static const char *ArchListGPUOnly[] = {"VK", nullptr};

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

    if (g_tests_success) {
        const auto t2 = high_resolution_clock::now();
        puts("---------------");
        test_complex_mat0(arch_list, device_name);
        test_complex_mat1(arch_list, device_name);
        test_complex_mat2(arch_list, device_name);
        test_complex_mat3(arch_list, device_name);
        test_complex_mat4(arch_list, device_name);
        test_complex_mat5(arch_list, device_name);
        test_complex_mat5_dof(arch_list, device_name);
        test_complex_mat5_mesh_lights(arch_list, device_name);
        test_complex_mat5_sphere_light(arch_list, device_name);
        test_complex_mat5_spot_light(arch_list, device_name);
        test_complex_mat5_sun_light(arch_list, device_name);
        test_complex_mat5_hdr_light(arch_list, device_name);
        test_complex_mat6(arch_list, device_name);
        test_complex_mat6_dof(arch_list, device_name);
        test_complex_mat6_mesh_lights(arch_list, device_name);
        test_complex_mat6_sphere_light(arch_list, device_name);
        test_complex_mat6_spot_light(arch_list, device_name);
        test_complex_mat6_sun_light(arch_list, device_name);
        test_complex_mat6_hdr_light(arch_list, device_name);
        test_complex_mat7_refractive(arch_list, device_name);
        test_complex_mat7_principled(arch_list, device_name);
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
            puts("---------------");
            test_oren_mat0(arch_list, device_name);
            test_oren_mat1(arch_list, device_name);
            test_oren_mat2(arch_list, device_name);
            printf("Finished oren_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_diff_mat0(arch_list, device_name);
            test_diff_mat1(arch_list, device_name);
            test_diff_mat2(arch_list, device_name);
            printf("Finished diff_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_sheen_mat0(arch_list, device_name);
            test_sheen_mat1(arch_list, device_name);
            test_sheen_mat2(arch_list, device_name);
            test_sheen_mat3(arch_list, device_name);
            printf("Finished sheen_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_glossy_mat0(arch_list, device_name);
            test_glossy_mat1(arch_list, device_name);
            test_glossy_mat2(arch_list, device_name);
            printf("Finished glossy_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_spec_mat0(arch_list, device_name);
            test_spec_mat1(arch_list, device_name);
            test_spec_mat2(arch_list, device_name);
            printf("Finished spec_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_aniso_mat0(arch_list, device_name);
            test_aniso_mat1(arch_list, device_name);
            test_aniso_mat2(arch_list, device_name);
            test_aniso_mat3(arch_list, device_name);
            test_aniso_mat4(arch_list, device_name);
            test_aniso_mat5(arch_list, device_name);
            test_aniso_mat6(arch_list, device_name);
            test_aniso_mat7(arch_list, device_name);
            printf("Finished aniso_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_metal_mat0(arch_list, device_name);
            test_metal_mat1(arch_list, device_name);
            test_metal_mat2(arch_list, device_name);
            printf("Finished metal_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_plastic_mat0(arch_list, device_name);
            test_plastic_mat1(arch_list, device_name);
            test_plastic_mat2(arch_list, device_name);
            printf("Finished plastic_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_tint_mat0(arch_list, device_name);
            test_tint_mat1(arch_list, device_name);
            test_tint_mat2(arch_list, device_name);
            printf("Finished tint_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_emit_mat0(arch_list, device_name);
            test_emit_mat1(arch_list, device_name);
            printf("Finished emit_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_coat_mat0(arch_list, device_name);
            test_coat_mat1(arch_list, device_name);
            test_coat_mat2(arch_list, device_name);
            printf("Finished coat_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mis0(arch_list, device_name);
            test_refr_mis1(arch_list, device_name);
            test_refr_mis2(arch_list, device_name);
            printf("Finished refr_mis tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mat0(arch_list, device_name);
            test_refr_mat1(arch_list, device_name);
            test_refr_mat2(arch_list, device_name);
            test_refr_mat3(arch_list, device_name);
            printf("Finished refr_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_trans_mat0(arch_list, device_name);
            test_trans_mat1(arch_list, device_name);
            test_trans_mat2(arch_list, device_name);
            test_trans_mat3(arch_list, device_name);
            test_trans_mat4(arch_list, device_name);
            test_trans_mat5(arch_list, device_name);
            printf("Finished trans_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_alpha_mat0(arch_list, device_name);
            test_alpha_mat1(arch_list, device_name);
            test_alpha_mat2(arch_list, device_name);
            test_alpha_mat3(arch_list, device_name);
            test_alpha_mat4(arch_list, device_name);
            printf("Finished alpha_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
    }
    assemble_material_test_images(arch_list);
    // test_texture();

    printf("FINISHED ALL TESTS in %.2f minutes\n", duration<double>(high_resolution_clock::now() - t1).count() / 60.0);

    if (g_log_contains_errors) {
        printf("LOG CONTAINS ERRORS!\n");
    }

    tests_success_final &= !g_log_contains_errors;
    tests_success_final &= g_tests_success;
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
#include <Windows.h>

extern "C" {
// Enable High Performance Graphics while using Integrated Graphics
__declspec(dllexport) int32_t NvOptimusEnablement = 0x00000001;     // Nvidia
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1; // AMD
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
        printf("SetPixelFormat() failed\n");
        return false;
    }

    HGLRC fake_rc = wglCreateContext(fake_dc);

    wglDeleteContext(fake_rc);
    ReleaseDC(fake_window, fake_dc);
    DestroyWindow(fake_window);

    return true;
}
#endif