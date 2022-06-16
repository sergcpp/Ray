
#include <cstdio>
#include <cstring>

#include <chrono>

void test_atlas();
void test_oren_mat0();
void test_oren_mat1();
void test_oren_mat2();
void test_oren_mat3();
void test_oren_mat4();
void test_diff_mat0();
void test_diff_mat1();
void test_diff_mat2();
void test_diff_mat3();
void test_diff_mat4();
void test_sheen_mat0();
void test_sheen_mat1();
void test_sheen_mat2();
void test_sheen_mat3();
void test_glossy_mat0();
void test_glossy_mat1();
void test_glossy_mat2();
void test_glossy_mat3();
void test_glossy_mat4();
void test_spec_mat0();
void test_spec_mat1();
void test_spec_mat2();
void test_spec_mat3();
void test_spec_mat4();
void test_aniso_mat0();
void test_aniso_mat1();
void test_aniso_mat2();
void test_aniso_mat3();
void test_aniso_mat4();
void test_aniso_mat5();
void test_aniso_mat6();
void test_aniso_mat7();
void test_tint_mat0();
void test_tint_mat1();
void test_tint_mat2();
void test_tint_mat3();
void test_tint_mat4();
void test_plastic_mat0();
void test_plastic_mat1();
void test_plastic_mat2();
void test_plastic_mat3();
void test_plastic_mat4();
void test_metal_mat0();
void test_metal_mat1();
void test_metal_mat2();
void test_metal_mat3();
void test_metal_mat4();
void test_emit_mat0();
void test_emit_mat1();
void test_coat_mat0();
void test_coat_mat1();
void test_coat_mat2();
void test_coat_mat3();
void test_coat_mat4();
void test_refr_mis0();
void test_refr_mis1();
void test_refr_mis2();
void test_refr_mis3();
void test_refr_mis4();
void test_refr_mat0();
void test_refr_mat1();
void test_refr_mat2();
void test_refr_mat3();
void test_refr_mat4();
void test_refr_mat5();
void test_trans_mat0();
void test_trans_mat1();
void test_trans_mat2();
void test_trans_mat3();
void test_trans_mat4();
void test_trans_mat5();
void test_trans_mat6();
void test_trans_mat7();
void test_trans_mat8();
void test_trans_mat9();
void test_alpha_mat0();
void test_alpha_mat1();
void test_alpha_mat2();
void test_alpha_mat3();
void test_complex_mat0();
void test_complex_mat1();
void test_complex_mat2();
void test_complex_mat3();
void test_complex_mat4();
void test_complex_mat5();
void test_complex_mat6();
void assemble_material_test_images();
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

    bool detailed_material_tests_needed = full_tests;
    bool tests_success_final = g_tests_success;
#if 1
    if (g_tests_success) {
        const auto t2 = high_resolution_clock::now();
        puts("---------------");
        test_complex_mat0();
        test_complex_mat1();
        test_complex_mat2();
        test_complex_mat3();
        test_complex_mat4();
        test_complex_mat5();
        test_complex_mat6();
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
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_oren_mat0();
            test_oren_mat1();
            test_oren_mat2();
            test_oren_mat3();
            test_oren_mat4();
            printf("Finished oren_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_diff_mat0();
            test_diff_mat1();
            test_diff_mat2();
            test_diff_mat3();
            test_diff_mat4();
            printf("Finished diff_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_sheen_mat0();
            test_sheen_mat1();
            test_sheen_mat2();
            test_sheen_mat3();
            printf("Finished sheen_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_glossy_mat0();
            test_glossy_mat1();
            test_glossy_mat2();
            test_glossy_mat3();
            test_glossy_mat4();
            printf("Finished glossy_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_spec_mat0();
            test_spec_mat1();
            test_spec_mat2();
            test_spec_mat3();
            test_spec_mat4();
            printf("Finished spec_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_aniso_mat0();
            test_aniso_mat1();
            test_aniso_mat2();
            test_aniso_mat3();
            test_aniso_mat4();
            test_aniso_mat5();
            test_aniso_mat6();
            test_aniso_mat7();
            printf("Finished aniso_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_metal_mat0();
            test_metal_mat1();
            test_metal_mat2();
            test_metal_mat3();
            test_metal_mat4();
            printf("Finished metal_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_plastic_mat0();
            test_plastic_mat1();
            test_plastic_mat2();
            test_plastic_mat3();
            test_plastic_mat4();
            printf("Finished plastic_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_tint_mat0();
            test_tint_mat1();
            test_tint_mat2();
            test_tint_mat3();
            test_tint_mat4();
            printf("Finished tint_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_emit_mat0();
            test_emit_mat1();
            printf("Finished emit_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_coat_mat0();
            test_coat_mat1();
            test_coat_mat2();
            test_coat_mat3();
            test_coat_mat4();
            printf("Finished coat_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mis0();
            test_refr_mis1();
            test_refr_mis2();
            test_refr_mis3();
            test_refr_mis4();
            printf("Finished refr_mis tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_refr_mat0();
            test_refr_mat1();
            test_refr_mat2();
            test_refr_mat3();
            test_refr_mat4();
            test_refr_mat5();
            printf("Finished refr_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_trans_mat0();
            test_trans_mat1();
            test_trans_mat2();
            test_trans_mat3();
            test_trans_mat4();
            test_trans_mat5();
            test_trans_mat6();
            test_trans_mat7();
            test_trans_mat8();
            test_trans_mat9();
            printf("Finished trans_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
#if 1
        if (g_tests_success) {
            const auto t2 = high_resolution_clock::now();
            puts("---------------");
            test_alpha_mat0();
            test_alpha_mat1();
            test_alpha_mat2();
            test_alpha_mat3();
            printf("Finished alpha_mat tests in %.2f minutes\n",
                   duration<double>(high_resolution_clock::now() - t2).count() / 60.0);
        }
#endif
    }
    assemble_material_test_images();
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
