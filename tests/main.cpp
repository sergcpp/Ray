
#include <cstdio>
#include <cstring>

#include <chrono>

void test_atlas();
void test_oren_mat0(const char *arch_list[]);
void test_oren_mat1(const char *arch_list[]);
void test_oren_mat2(const char *arch_list[]);
void test_oren_mat3(const char *arch_list[]);
void test_oren_mat4(const char *arch_list[]);
void test_diff_mat0(const char *arch_list[]);
void test_diff_mat1(const char *arch_list[]);
void test_diff_mat2(const char *arch_list[]);
void test_diff_mat3(const char *arch_list[]);
void test_diff_mat4(const char *arch_list[]);
void test_sheen_mat0(const char *arch_list[]);
void test_sheen_mat1(const char *arch_list[]);
void test_sheen_mat2(const char *arch_list[]);
void test_sheen_mat3(const char *arch_list[]);
void test_glossy_mat0(const char *arch_list[]);
void test_glossy_mat1(const char *arch_list[]);
void test_glossy_mat2(const char *arch_list[]);
void test_glossy_mat3(const char *arch_list[]);
void test_glossy_mat4(const char *arch_list[]);
void test_spec_mat0(const char *arch_list[]);
void test_spec_mat1(const char *arch_list[]);
void test_spec_mat2(const char *arch_list[]);
void test_spec_mat3(const char *arch_list[]);
void test_spec_mat4(const char *arch_list[]);
void test_aniso_mat0(const char *arch_list[]);
void test_aniso_mat1(const char *arch_list[]);
void test_aniso_mat2(const char *arch_list[]);
void test_aniso_mat3(const char *arch_list[]);
void test_aniso_mat4(const char *arch_list[]);
void test_aniso_mat5(const char *arch_list[]);
void test_aniso_mat6(const char *arch_list[]);
void test_aniso_mat7(const char *arch_list[]);
void test_tint_mat0(const char *arch_list[]);
void test_tint_mat1(const char *arch_list[]);
void test_tint_mat2(const char *arch_list[]);
void test_tint_mat3(const char *arch_list[]);
void test_tint_mat4(const char *arch_list[]);
void test_plastic_mat0(const char *arch_list[]);
void test_plastic_mat1(const char *arch_list[]);
void test_plastic_mat2(const char *arch_list[]);
void test_plastic_mat3(const char *arch_list[]);
void test_plastic_mat4(const char *arch_list[]);
void test_metal_mat0(const char *arch_list[]);
void test_metal_mat1(const char *arch_list[]);
void test_metal_mat2(const char *arch_list[]);
void test_metal_mat3(const char *arch_list[]);
void test_metal_mat4(const char *arch_list[]);
void test_emit_mat0(const char *arch_list[]);
void test_emit_mat1(const char *arch_list[]);
void test_coat_mat0(const char *arch_list[]);
void test_coat_mat1(const char *arch_list[]);
void test_coat_mat2(const char *arch_list[]);
void test_coat_mat3(const char *arch_list[]);
void test_coat_mat4(const char *arch_list[]);
void test_refr_mis0(const char *arch_list[]);
void test_refr_mis1(const char *arch_list[]);
void test_refr_mis2(const char *arch_list[]);
void test_refr_mis3(const char *arch_list[]);
void test_refr_mis4(const char *arch_list[]);
void test_refr_mat0(const char *arch_list[]);
void test_refr_mat1(const char *arch_list[]);
void test_refr_mat2(const char *arch_list[]);
void test_refr_mat3(const char *arch_list[]);
void test_refr_mat4(const char *arch_list[]);
void test_refr_mat5(const char *arch_list[]);
void test_trans_mat0(const char *arch_list[]);
void test_trans_mat1(const char *arch_list[]);
void test_trans_mat2(const char *arch_list[]);
void test_trans_mat3(const char *arch_list[]);
void test_trans_mat4(const char *arch_list[]);
void test_trans_mat5(const char *arch_list[]);
void test_trans_mat6(const char *arch_list[]);
void test_trans_mat7(const char *arch_list[]);
void test_trans_mat8(const char *arch_list[]);
void test_trans_mat9(const char *arch_list[]);
void test_alpha_mat0(const char *arch_list[]);
void test_alpha_mat1(const char *arch_list[]);
void test_alpha_mat2(const char *arch_list[]);
void test_alpha_mat3(const char *arch_list[]);
void test_complex_mat0(const char *arch_list[]);
void test_complex_mat1(const char *arch_list[]);
void test_complex_mat2(const char *arch_list[]);
void test_complex_mat3(const char *arch_list[]);
void test_complex_mat4(const char *arch_list[]);
void test_complex_mat4_mesh_lights(const char *arch_list[]);
void test_complex_mat4_sphere_light(const char *arch_list[]);
void test_complex_mat4_sun_light(const char *arch_list[]);
void test_complex_mat5(const char *arch_list[]);
void test_complex_mat5_mesh_lights(const char *arch_list[]);
void test_complex_mat5_sphere_light(const char *arch_list[]);
void test_complex_mat5_sun_light(const char *arch_list[]);
void test_complex_mat6(const char *arch_list[]);
void assemble_material_test_images(const char *arch_list[]);
void test_simd();
void test_mesh_lights();
void test_texture();

bool g_stop_on_fail = false;
bool g_tests_success = true;

int main(int argc, char *argv[]) {
    using namespace std::chrono;

    const auto t1 = high_resolution_clock::now();

    const bool full_tests = (argc > 1 && strcmp(argv[1], "--full") == 0);

    test_atlas();
    test_simd();
    // test_mesh_lights();

    static const char *ArchListFull[] = {"ref", "sse2", "sse41", "avx", "avx2", "neon", nullptr};
    static const char *ArchListDefault[] = {"ref", "avx2", "neon", nullptr};

    bool detailed_material_tests_needed = full_tests;
    bool tests_success_final = g_tests_success;

    const char **arch_list = full_tests ? ArchListFull : ArchListDefault;

#if 1
    if (g_tests_success) {
        const auto t2 = high_resolution_clock::now();
        puts("---------------");
        test_complex_mat0(arch_list);
        test_complex_mat1(arch_list);
        test_complex_mat2(arch_list);
        test_complex_mat3(arch_list);
        test_complex_mat4(arch_list);
        test_complex_mat4_mesh_lights(arch_list);
        test_complex_mat4_sphere_light(arch_list);
        test_complex_mat4_sun_light(arch_list);
        test_complex_mat5(arch_list);
        test_complex_mat5_mesh_lights(arch_list);
        test_complex_mat5_sphere_light(arch_list);
        test_complex_mat5_sun_light(arch_list);
        test_complex_mat6(arch_list);
        printf("Finished complex_mat tests in %.2f minutes\n",
               duration<double>(high_resolution_clock::now() - t2).count() / 60.0);

        // schedule detailed material tests if complex tests failed (to find out the reason)
        detailed_material_tests_needed |= !g_tests_success;
        tests_success_final &= g_tests_success;
        g_tests_success = true;
    }
#endif

    if (detailed_material_tests_needed) {
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_oren_mat0(arch_list);
            test_oren_mat1(arch_list);
            test_oren_mat2(arch_list);
            test_oren_mat3(arch_list);
            test_oren_mat4(arch_list);
            printf("Finished oren_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_diff_mat0(arch_list);
            test_diff_mat1(arch_list);
            test_diff_mat2(arch_list);
            test_diff_mat3(arch_list);
            test_diff_mat4(arch_list);
            printf("Finished diff_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_sheen_mat0(arch_list);
            test_sheen_mat1(arch_list);
            test_sheen_mat2(arch_list);
            test_sheen_mat3(arch_list);
            printf("Finished sheen_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_glossy_mat0(arch_list);
            test_glossy_mat1(arch_list);
            test_glossy_mat2(arch_list);
            test_glossy_mat3(arch_list);
            test_glossy_mat4(arch_list);
            printf("Finished glossy_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_spec_mat0(arch_list);
            test_spec_mat1(arch_list);
            test_spec_mat2(arch_list);
            test_spec_mat3(arch_list);
            test_spec_mat4(arch_list);
            printf("Finished spec_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_aniso_mat0(arch_list);
            test_aniso_mat1(arch_list);
            test_aniso_mat2(arch_list);
            test_aniso_mat3(arch_list);
            test_aniso_mat4(arch_list);
            test_aniso_mat5(arch_list);
            test_aniso_mat6(arch_list);
            test_aniso_mat7(arch_list);
            printf("Finished aniso_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_metal_mat0(arch_list);
            test_metal_mat1(arch_list);
            test_metal_mat2(arch_list);
            test_metal_mat3(arch_list);
            test_metal_mat4(arch_list);
            printf("Finished metal_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_plastic_mat0(arch_list);
            test_plastic_mat1(arch_list);
            test_plastic_mat2(arch_list);
            test_plastic_mat3(arch_list);
            test_plastic_mat4(arch_list);
            printf("Finished plastic_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_tint_mat0(arch_list);
            test_tint_mat1(arch_list);
            test_tint_mat2(arch_list);
            test_tint_mat3(arch_list);
            test_tint_mat4(arch_list);
            printf("Finished tint_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_emit_mat0(arch_list);
            test_emit_mat1(arch_list);
            printf("Finished emit_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_coat_mat0(arch_list);
            test_coat_mat1(arch_list);
            test_coat_mat2(arch_list);
            test_coat_mat3(arch_list);
            test_coat_mat4(arch_list);
            printf("Finished coat_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mis0(arch_list);
            test_refr_mis1(arch_list);
            test_refr_mis2(arch_list);
            test_refr_mis3(arch_list);
            test_refr_mis4(arch_list);
            printf("Finished refr_mis tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mat0(arch_list);
            test_refr_mat1(arch_list);
            test_refr_mat2(arch_list);
            test_refr_mat3(arch_list);
            test_refr_mat4(arch_list);
            test_refr_mat5(arch_list);
            printf("Finished refr_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_trans_mat0(arch_list);
            test_trans_mat1(arch_list);
            test_trans_mat2(arch_list);
            test_trans_mat3(arch_list);
            test_trans_mat4(arch_list);
            test_trans_mat5(arch_list);
            test_trans_mat6(arch_list);
            test_trans_mat7(arch_list);
            test_trans_mat8(arch_list);
            test_trans_mat9(arch_list);
            printf("Finished trans_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success || full_tests) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_alpha_mat0(arch_list);
            test_alpha_mat1(arch_list);
            test_alpha_mat2(arch_list);
            test_alpha_mat3(arch_list);
            printf("Finished alpha_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
    }
    assemble_material_test_images(arch_list);
    // test_texture();

    printf("FINISHED ALL TESTS in %.2f minutes\n", duration<double>(high_resolution_clock::now() - t1).count() / 60.0);

    tests_success_final &= g_tests_success;
    if (tests_success_final) {
        puts("SUCCESS");
    } else {
        puts("FAILED");
    }
    return tests_success_final ? 0 : -1;
}
