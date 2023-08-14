#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>

#include "../Ray.h"

#include "test_scene.h"
#include "thread_pool.h"
#include "utils.h"

extern bool g_determine_sample_count;
extern bool g_minimal_output;
std::mutex g_stdout_mtx;

template <typename MatDesc>
void run_material_test(const char *arch_list[], const char *preferred_device, const char *test_name,
                       const MatDesc &mat_desc, const int sample_count, const double min_psnr, const int pix_thres,
                       const eDenoiseMethod denoise = eDenoiseMethod::None, const bool partial = false,
                       const char *textures[] = nullptr, const eTestScene test_scene = eTestScene::Standard) {
    run_material_test(arch_list, preferred_device, test_name, mat_desc, sample_count, sample_count, 0.0f, min_psnr,
                      pix_thres, denoise, partial, textures, test_scene);
}

template <typename MatDesc>
void run_material_test(const char *arch_list[], const char *preferred_device, const char *test_name,
                       const MatDesc &mat_desc, const int min_sample_count, const int max_sample_count,
                       const float variance_threshold, const double min_psnr, const int pix_thres,
                       const eDenoiseMethod denoise = eDenoiseMethod::None, const bool partial = false,
                       const char *textures[] = nullptr, const eTestScene test_scene = eTestScene::Standard) {
    using namespace std::chrono;

    char name_buf[1024];
    snprintf(name_buf, sizeof(name_buf), "test_data/%s/ref.tga", test_name);

    int test_img_w, test_img_h;
    const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
    require_return(!test_img.empty());

    Ray::settings_t s;
    s.w = test_img_w;
    s.h = test_img_h;
    s.preferred_device = preferred_device;
    s.use_wide_bvh = true;

    const int DiffThres = 32;

    for (const char **arch = arch_list; *arch; ++arch) {
        const auto rt = Ray::RendererTypeFromName(*arch);

        for (const bool use_hwrt : {false, true}) {
            s.use_hwrt = use_hwrt;

            int current_sample_count = max_sample_count;
            int failed_count = -1, succeeded_count = 4096;
            bool images_match = false, searching = false;
            do {
                const auto start_time = high_resolution_clock::now();

                auto renderer = std::unique_ptr<Ray::RendererBase>(Ray::CreateRenderer(s, &g_log_err, rt));
                if (!renderer || renderer->type() != rt || renderer->is_hwrt() != use_hwrt) {
                    // skip unsupported (we fell back to some other renderer)
                    break;
                }
                if (preferred_device) {
                    // make sure we use requested device
                    if (!require(Ray::MatchDeviceNames(renderer->device_name(), preferred_device))) {
                        std::lock_guard<std::mutex> _(g_stdout_mtx);
                        printf("Wrong device: %s (%s was requested)\n", renderer->device_name(), preferred_device);
                        return;
                    }
                }

                auto scene = std::unique_ptr<Ray::SceneBase>(renderer->CreateScene());

                const bool output_base_color =
                    (denoise == eDenoiseMethod::NLM_b || denoise == eDenoiseMethod::NLM_bn ||
                     denoise == eDenoiseMethod::UNet_b || denoise == eDenoiseMethod::UNet_bn);
                const bool output_normals = (denoise == eDenoiseMethod::NLM_bn || denoise == eDenoiseMethod::UNet_bn);
                setup_test_scene(*scene, output_base_color, output_normals, min_sample_count, variance_threshold,
                                 mat_desc, textures, test_scene);

                snprintf(name_buf, sizeof(name_buf), "Test %s", test_name);
                schedule_render_jobs(*renderer, scene.get(), s, current_sample_count, denoise, partial, name_buf);

                const Ray::color_data_rgba_t pixels = renderer->get_pixels_ref();

                std::unique_ptr<uint8_t[]> img_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                std::unique_ptr<uint8_t[]> diff_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                std::unique_ptr<uint8_t[]> mask_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                memset(&mask_data_u8[0], 0, test_img_w * test_img_h * 3);

                double mse = 0.0;

                int error_pixels = 0;
                for (int j = 0; j < test_img_h; j++) {
                    for (int i = 0; i < test_img_w; i++) {
                        const Ray::color_rgba_t &p = pixels.ptr[j * pixels.pitch + i];

                        const auto r = uint8_t(p.v[0] * 255);
                        const auto g = uint8_t(p.v[1] * 255);
                        const auto b = uint8_t(p.v[2] * 255);

                        img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 0] = r;
                        img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 1] = g;
                        img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 2] = b;

                        const uint8_t diff_r = std::abs(r - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 0]);
                        const uint8_t diff_g = std::abs(g - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 1]);
                        const uint8_t diff_b = std::abs(b - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 2]);

                        diff_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 0] = diff_r;
                        diff_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 1] = diff_g;
                        diff_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 2] = diff_b;

                        if (diff_r > DiffThres || diff_g > DiffThres || diff_b > DiffThres) {
                            mask_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 0] = 255;
                            ++error_pixels;
                        }

                        mse += diff_r * diff_r;
                        mse += diff_g * diff_g;
                        mse += diff_b * diff_b;
                    }
                }

                mse /= 3.0;
                mse /= (test_img_w * test_img_h);

                double psnr = -10.0 * std::log10(mse / (255.0 * 255.0));
                psnr = std::floor(psnr * 100.0) / 100.0;

                const double test_duration_m =
                    duration<double>(high_resolution_clock::now() - start_time).count() / 60.0;

                {
                    std::lock_guard<std::mutex> _(g_stdout_mtx);
                    if (g_minimal_output) {
                        printf("\r%s (%6s, %s): %.1f%% ", name_buf, Ray::RendererTypeName(rt),
                               s.use_hwrt ? "HWRT" : "SWRT", 100.0);
                    }
                    printf("(PSNR: %.2f/%.2f dB, Fireflies: %i/%i, Time: %.2fm)\n", psnr, min_psnr, error_pixels,
                           pix_thres, test_duration_m);
                    fflush(stdout);
                }

                std::string type = Ray::RendererTypeName(rt);
                if (use_hwrt) {
                    type += "_HWRT";
                }

                snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_out.tga", test_name, type.c_str());
                WriteTGA(&img_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_diff.tga", test_name, type.c_str());
                WriteTGA(&diff_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_mask.tga", test_name, type.c_str());
                WriteTGA(&mask_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                images_match = (psnr >= min_psnr) && (error_pixels <= pix_thres);
                require(images_match || searching);

                if (!images_match) {
                    failed_count = std::max(failed_count, current_sample_count);
                    if (succeeded_count != 4096) {
                        current_sample_count = (failed_count + succeeded_count) / 2;
                    } else {
                        current_sample_count *= 2;
                    }
                } else {
                    succeeded_count = std::min(succeeded_count, current_sample_count);
                    current_sample_count = (failed_count + succeeded_count) / 2;
                }
                if (searching) {
                    std::lock_guard<std::mutex> _(g_stdout_mtx);
                    printf("Current_sample_count = %i (%i - %i)\n", current_sample_count, failed_count,
                           succeeded_count);
                }
                searching |= !images_match;
            } while (g_determine_sample_count && searching && (succeeded_count - failed_count) > 1);
            if (g_determine_sample_count && searching && succeeded_count != max_sample_count) {
                std::lock_guard<std::mutex> _(g_stdout_mtx);
                printf("Required sample count for %s: %i\n", test_name, succeeded_count);
            }
        }
    }
}

void assemble_material_test_images(const char *arch_list[]) {
    static const int ImgCountW = 5;
    static const char *test_names[][ImgCountW] = {
        {"oren_mat0", "oren_mat1", "oren_mat2", "diff_mat0", "diff_mat1"},
        {"diff_mat2", "sheen_mat0", "sheen_mat1", "sheen_mat2", "sheen_mat3"},
        {"glossy_mat0", "glossy_mat1", "glossy_mat2", "spec_mat0", "spec_mat1"},
        {"spec_mat2", "aniso_mat0", "aniso_mat1", "aniso_mat2", "aniso_mat3"},
        {"aniso_mat4", "aniso_mat5", "aniso_mat6", "aniso_mat7", "metal_mat0"},
        {"metal_mat1", "metal_mat2", "plastic_mat0", "plastic_mat1", "plastic_mat2"},
        {"tint_mat0", "tint_mat1", "tint_mat2", "emit_mat0", "emit_mat1"},
        {"coat_mat0", "coat_mat1", "coat_mat2", "refr_mis0", "refr_mis1"},
        {"refr_mat0", "refr_mat1", "refr_mat2", "refr_mat3", "trans_mat5"},
        {"trans_mat0", "trans_mat1", "trans_mat2", "trans_mat3", "trans_mat4"},
        {"alpha_mat0", "alpha_mat1", "alpha_mat2", "alpha_mat3", "alpha_mat4"},
        {"complex_mat0", "complex_mat1", "complex_mat2", "complex_mat3", "complex_mat4"},
        {"complex_mat5", "complex_mat5_mesh_lights", "complex_mat5_sphere_light", "complex_mat5_sun_light",
         "complex_mat5_hdr_light"},
        {"complex_mat6", "complex_mat6_mesh_lights", "complex_mat6_sphere_light", "complex_mat6_sun_light",
         "complex_mat6_hdr_light"},
        {"complex_mat5_regions", "complex_mat5_dof", "complex_mat5_spot_light", "complex_mat6_dof",
         "complex_mat6_spot_light"},
        {"refr_mis2", "complex_mat5_nlm_filter_bn", "complex_mat5_adaptive", "complex_mat5_clipped",
         "complex_mat6_nlm_filter_bn"},
        {"complex_mat5_nlm_filter", "complex_mat5_nlm_filter_b", "complex_mat5_unet_filter",
         "complex_mat5_unet_filter_b", "complex_mat5_unet_filter_bn"},
        {"complex_mat6_nlm_filter", "complex_mat6_nlm_filter_b", "complex_mat6_unet_filter_b",
         "complex_mat6_unet_filter_bn"}};
    const int ImgCountH = sizeof(test_names) / sizeof(test_names[0]);

    const int OutImageW = 256 * ImgCountW;
    const int OutImageH = 256 * ImgCountH;

    std::unique_ptr<uint8_t[]> material_refs(new uint8_t[OutImageH * OutImageW * 4]);
    std::unique_ptr<uint8_t[]> material_imgs(new uint8_t[OutImageH * OutImageW * 4]);
    std::unique_ptr<uint8_t[]> material_masks(new uint8_t[OutImageH * OutImageW * 4]);
    memset(material_refs.get(), 0, OutImageH * OutImageW * 4);
    memset(material_imgs.get(), 0, OutImageH * OutImageW * 4);
    memset(material_masks.get(), 0, OutImageH * OutImageW * 4);

    int font_img_w, font_img_h;
    const auto font_img = LoadTGA("test_data/font.tga", font_img_w, font_img_h);
    auto blit_chars_to_alpha = [&](uint8_t out_img[], const int x, const int y, const char *str) {
        const int GlyphH = font_img_h;
        const int GlyphW = (GlyphH / 2);

        int cur_offset_x = x;
        while (*str) {
            const int glyph_index = int(*str) - 32;

            for (int j = 0; j < GlyphH; ++j) {
                for (int i = 0; i < GlyphW; ++i) {
                    const uint8_t val = font_img[4 * (j * font_img_w + i + glyph_index * GlyphW) + 0];
                    out_img[4 * ((y + j) * OutImageW + cur_offset_x + i) + 3] = val;
                }
            }

            cur_offset_x += GlyphW;
            ++str;
        }
    };

    char name_buf[1024];

    for (const char **arch = arch_list; *arch; ++arch) {
        std::string type = *arch;
        for (const bool hwrt : {false, true}) {
            const auto rt = Ray::RendererTypeFromName(type.c_str());
            if (hwrt) {
                if (!Ray::RendererSupportsHWRT(rt)) {
                    continue;
                }
                type += "_HWRT";
            }

            bool found_at_least_one_image = false;
            for (int j = 0; j < ImgCountH; ++j) {
                const int top_down_j = ImgCountH - j - 1;

                for (int i = 0; i < ImgCountW && test_names[j][i]; ++i) {
                    { // reference image
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/ref.tga", test_names[j][i]);

                        int test_img_w, test_img_h;
                        const auto img_ref = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!img_ref.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_refs[((top_down_j * 256 + k) * OutImageW + i * 256) * 4],
                                       &img_ref[k * test_img_w * 4], test_img_w * 4);
                            }
                        }

                        blit_chars_to_alpha(material_refs.get(), i * 256, top_down_j * 256, test_names[j][i]);
                    }

                    { // test output
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_out.tga", test_names[j][i], type.c_str());

                        int test_img_w, test_img_h;
                        const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!test_img.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_imgs[((top_down_j * 256 + k) * OutImageW + i * 256) * 4],
                                       &test_img[k * test_img_w * 4], test_img_w * 4);
                            }
                            found_at_least_one_image = true;
                        }

                        blit_chars_to_alpha(material_imgs.get(), i * 256, top_down_j * 256, test_names[j][i]);
                    }

                    { // error mask
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_mask.tga", test_names[j][i],
                                 type.c_str());

                        int test_img_w, test_img_h;
                        const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!test_img.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_masks[((top_down_j * 256 + k) * OutImageW + i * 256) * 4],
                                       &test_img[k * test_img_w * 4], test_img_w * 4);
                            }
                        }

                        blit_chars_to_alpha(material_masks.get(), i * 256, top_down_j * 256, test_names[j][i]);
                    }
                }
            }

            if (found_at_least_one_image) {
                snprintf(name_buf, sizeof(name_buf), "test_data/material_%s_imgs.tga", type.c_str());
                WriteTGA(material_imgs.get(), OutImageW, OutImageH, 4, name_buf);
                snprintf(name_buf, sizeof(name_buf), "test_data/material_%s_masks.tga", type.c_str());
                WriteTGA(material_masks.get(), OutImageW, OutImageH, 4, name_buf);
            }
        }
    }

    WriteTGA(material_refs.get(), OutImageW, OutImageH, 4, "test_data/material_refs.tga");
}

const double DefaultMinPSNR = 30.0;
const double FastMinPSNR = 28.0;
const double VeryFastMinPSNR = 25.0;
const int DefaultPixThres = 1;

//
// Oren-nayar material tests
//

void test_oren_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 310;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;

    run_material_test(arch_list, preferred_device, "oren_mat0", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

void test_oren_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 130;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "oren_mat1", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

void test_oren_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 310;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "oren_mat2", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

//
// Diffuse material tests
//

void test_diff_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 120;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.0f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat0", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

void test_diff_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 120;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat1", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

void test_diff_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 120;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat2", desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

//
// Sheen material tests
//

void test_sheen_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 260;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 0.5f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat0", mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_sheen_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 290;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat1", mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_sheen_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 140;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat2", mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_sheen_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 140;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 1.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat3", mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Glossy material tests
//

void test_glossy_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1680;
    const int PixThres = 100;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "glossy_mat0", node_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_glossy_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 400;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "glossy_mat1", node_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_glossy_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 170;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "glossy_mat2", node_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Specular material tests
//

void test_spec_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1640;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat0", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_spec_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 400;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.5f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat1", spec_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_spec_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 170;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 1.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat2", spec_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Anisotropic material tests
//

void test_aniso_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 910;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.25f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test(arch_list, preferred_device, "aniso_mat0", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 920;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.5f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test(arch_list, preferred_device, "aniso_mat1", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 951;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.75f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test(arch_list, preferred_device, "aniso_mat2", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 950;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test(arch_list, preferred_device, "aniso_mat3", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat4(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 960;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.125f;

    run_material_test(arch_list, preferred_device, "aniso_mat4", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat5(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 920;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.25f;

    run_material_test(arch_list, preferred_device, "aniso_mat5", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat6(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1110;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.375f;

    run_material_test(arch_list, preferred_device, "aniso_mat6", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_aniso_mat7(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 955;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.5f;

    run_material_test(arch_list, preferred_device, "aniso_mat7", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

//
// Metal material tests
//

void test_metal_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 870;
    const int PixThres = 100;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.0f;
    metal_mat_desc.base_color[1] = 0.5f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat0", metal_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_metal_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 180;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.5f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat1", metal_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_metal_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 160;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat2", metal_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Plastic material tests
//

void test_plastic_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 370;
    const int PixThres = 100;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.0f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "plastic_mat0", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      PixThres);
}

void test_plastic_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 330;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.0f;
    plastic_mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "plastic_mat1", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_plastic_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 280;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "plastic_mat2", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Tint material tests
//

void test_tint_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 680;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.0f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "tint_mat0", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_tint_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 2210;
    const int PixThres = 100;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.0f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "tint_mat1", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_tint_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 540;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "tint_mat2", spec_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Emissive material tests
//

void test_emit_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 330;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 0.5f;

    run_material_test(arch_list, preferred_device, "emit_mat0", mat_desc, SampleCount, DefaultMinPSNR, DefaultPixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_NoLight);
}

void test_emit_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 620;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 1.0f;

    run_material_test(arch_list, preferred_device, "emit_mat1", mat_desc, SampleCount, DefaultMinPSNR, DefaultPixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_NoLight);
}

//
// Clear coat material tests
//

void test_coat_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 200;
    const int PixThres = 10;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "coat_mat0", mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_coat_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 290;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "coat_mat1", mat_desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

void test_coat_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 210;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "coat_mat2", mat_desc, SampleCount, DefaultMinPSNR, DefaultPixThres);
}

//
// Refractive material tests
//

void test_refr_mis0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1330;
    const int PixThres = 10;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "refr_mis0", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

void test_refr_mis1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 670;
    const int PixThres = 10;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "refr_mis1", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

void test_refr_mis2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 620;
    const int PixThres = 10;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mis2", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

///

void test_refr_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1060;
    const double MinPSNR = 24.97;
    const int PixThres = 3846;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.001f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mat0", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1060;
    const double MinPSNR = 26.99;
    const int PixThres = 2384;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "refr_mat1", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1080;
    const double MinPSNR = 31.66;
    const int PixThres = 1521;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "refr_mat2", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1080;
    const double MinPSNR = 34.36;
    const int PixThres = 40;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mat3", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

//
// Transmissive material tests
//

void test_trans_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1060;
    const double MinPSNR = 24.96;
    const int PixThres = 3840;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.001f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "trans_mat0", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1030;
    const double MinPSNR = 25.43;
    const int PixThres = 2689;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "trans_mat1", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1050;
    const double MinPSNR = 27.86;
    const int PixThres = 1192;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "trans_mat2", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1030;
    const double MinPSNR = 31.24;
    const int PixThres = 100;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "trans_mat3", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat4(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1060;
    const double MinPSNR = 27.11;
    const int PixThres = 1482;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "trans_mat4", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat5(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1240;
    const int PixThres = 10;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "trans_mat5", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

//
// Transparent material tests
//

void test_alpha_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 680;
    const int PixThres = 100;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.75f;

    run_material_test(arch_list, preferred_device, "alpha_mat0", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 1530;
    const int PixThres = 100;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.5f;

    run_material_test(arch_list, preferred_device, "alpha_mat1", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 880;
    const int PixThres = 100;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.25f;

    run_material_test(arch_list, preferred_device, "alpha_mat2", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 190;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.0f;

    run_material_test(arch_list, preferred_device, "alpha_mat3", alpha_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

void test_alpha_mat4(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 190;

    Ray::shading_node_desc_t alpha_mat_desc;
    alpha_mat_desc.type = Ray::eShadingNode::Transparent;
    alpha_mat_desc.base_color[0] = 0.75f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.0f;

    run_material_test(arch_list, preferred_device, "alpha_mat4", alpha_mat_desc, SampleCount, DefaultMinPSNR,
                      DefaultPixThres);
}

//
// Complex material tests
//

void test_complex_mat0(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 17;
    const int PixThres = 762;

    Ray::principled_mat_desc_t wood_mat_desc;
    wood_mat_desc.base_texture = Ray::TextureHandle{0};
    wood_mat_desc.roughness = 1.0f;
    wood_mat_desc.roughness_texture = Ray::TextureHandle{2};
    wood_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/older-wood-flooring_albedo_2045.tga",
        "test_data/textures/older-wood-flooring_normal-ogl_2045.tga",
        "test_data/textures/older-wood-flooring_roughness_2045.tga",
    };

    run_material_test(arch_list, preferred_device, "complex_mat0", wood_mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures);
}

void test_complex_mat1(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 20;
    const int PixThres = 777;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/streaky-metal1_albedo.tga",
        "test_data/textures/streaky-metal1_normal-ogl_rgba.tga",
        "test_data/textures/streaky-metal1_roughness.tga",
    };

    run_material_test(arch_list, preferred_device, "complex_mat1", metal_mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures);
}

void test_complex_mat2(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 17;
    const int PixThres = 630;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/rusting-lined-metal_albedo.tga", "test_data/textures/rusting-lined-metal_normal-ogl.tga",
        "test_data/textures/rusting-lined-metal_roughness.tga", "test_data/textures/rusting-lined-metal_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat2", metal_mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures);
}

void test_complex_mat3(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 13;
    const int PixThres = 426;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};
    metal_mat_desc.normal_map_intensity = 0.3f;

    const char *textures[] = {
        "test_data/textures/stone_trims_02_BaseColor.tga", "test_data/textures/stone_trims_02_Normal.tga",
        "test_data/textures/stone_trims_02_Roughness.tga", "test_data/textures/stone_trims_02_Metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat3", metal_mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures);
}

void test_complex_mat4(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 2398;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};
    metal_mat_desc.alpha_texture = Ray::TextureHandle{4};

    const char *textures[] = {
        "test_data/textures/Fence007A_2K_Color.tga", "test_data/textures/Fence007A_2K_NormalGL.tga",
        "test_data/textures/Fence007A_2K_Roughness.tga", "test_data/textures/Fence007A_2K_Metalness.tga",
        "test_data/textures/Fence007A_2K_Opacity.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat4", metal_mat_desc, SampleCount, VeryFastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures);
}

void test_complex_mat5(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 35;
    const int PixThres = 4771;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5", metal_mat_desc, SampleCount, VeryFastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures);
}

void test_complex_mat5_clipped(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 34;
    const int PixThres = 4740;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_clipped", metal_mat_desc, SampleCount, VeryFastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_Clipped);
}

void test_complex_mat5_adaptive(const char *arch_list[], const char *preferred_device) {
    const int MinSampleCount = 8;
    const int MaxSampleCount = 21;
    const float VarianceThreshold = 0.004f;
    const int PixThres = 2072;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_adaptive", metal_mat_desc, MinSampleCount,
                      MaxSampleCount, VarianceThreshold, FastMinPSNR, PixThres, eDenoiseMethod::NLM_bn, false,
                      textures);
}

void test_complex_mat5_regions(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 4038;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_regions", metal_mat_desc, SampleCount, VeryFastMinPSNR,
                      PixThres, eDenoiseMethod::None, true, textures);
}

void test_complex_mat5_nlm_filter(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 29;
    const int PixThres = 2068;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_nlm_filter", metal_mat_desc, SampleCount, FastMinPSNR,
                      PixThres, eDenoiseMethod::NLM, false, textures);
}

void test_complex_mat5_nlm_filter_b(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 22;
    const int PixThres = 2085;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_nlm_filter_b", metal_mat_desc, SampleCount,
                      FastMinPSNR, PixThres, eDenoiseMethod::NLM_b, false, textures);
}

void test_complex_mat5_nlm_filter_bn(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 19;
    const int PixThres = 2089;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_nlm_filter_bn", metal_mat_desc, SampleCount,
                      FastMinPSNR, PixThres, eDenoiseMethod::NLM_bn, false, textures);
}

void test_complex_mat5_unet_filter(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 35;
    const int PixThres = 1162;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_unet_filter", metal_mat_desc, SampleCount,
                      DefaultMinPSNR, PixThres, eDenoiseMethod::UNet, false, textures);
}

void test_complex_mat5_unet_filter_b(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 26;
    const int PixThres = 1117;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_unet_filter_b", metal_mat_desc, SampleCount,
                      DefaultMinPSNR, PixThres, eDenoiseMethod::UNet_b, false, textures);
}

void test_complex_mat5_unet_filter_bn(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 27;
    const int PixThres = 1098;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_unet_filter_bn", metal_mat_desc, SampleCount,
                      DefaultMinPSNR, PixThres, eDenoiseMethod::UNet_bn, false, textures);
}

void test_complex_mat5_dof(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 30;
    const double MinPSNR = 21.0;
    const int PixThres = 9976;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_dof", metal_mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures, eTestScene::Standard_DOF0);
}

void test_complex_mat5_mesh_lights(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 25;
    const double MinPSNR = 23.0;
    const int PixThres = 7273;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_mesh_lights", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_MeshLights);
}

void test_complex_mat5_sphere_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 24;
    const double MinPSNR = 23.0;
    const int PixThres = 5105;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_sphere_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_SphereLight);
}

void test_complex_mat5_spot_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 5;
    const int PixThres = 729;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_spot_light", metal_mat_desc, SampleCount, FastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_SpotLight);
}

void test_complex_mat5_sun_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 19;
    const double MinPSNR = 23.0;
    const int PixThres = 6063;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_sun_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_SunLight);
}

void test_complex_mat5_hdr_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 19;
    const double MinPSNR = 23.0;
    const int PixThres = 6236;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_hdr_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_HDRLight);
}

void test_complex_mat6(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 45;
    const double MinPSNR = 23.0;
    const int PixThres = 4245;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres);
}

void test_complex_mat6_nlm_filter(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 18;
    const int PixThres = 1929;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_nlm_filter", olive_mat_desc, SampleCount,
                      VeryFastMinPSNR, PixThres, eDenoiseMethod::NLM);
}

void test_complex_mat6_nlm_filter_b(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 14;
    const int PixThres = 1756;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_nlm_filter_b", olive_mat_desc, SampleCount,
                      VeryFastMinPSNR, PixThres, eDenoiseMethod::NLM_b);
}

void test_complex_mat6_nlm_filter_bn(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 1619;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_nlm_filter_bn", olive_mat_desc, SampleCount,
                      VeryFastMinPSNR, PixThres, eDenoiseMethod::NLM_bn);
}

void test_complex_mat6_unet_filter(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 38;
    const int PixThres = 1007;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_unet_filter", olive_mat_desc, SampleCount,
                      FastMinPSNR, PixThres, eDenoiseMethod::UNet);
}

void test_complex_mat6_unet_filter_b(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 28;
    const int PixThres = 924;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_unet_filter_b", olive_mat_desc, SampleCount,
                      FastMinPSNR, PixThres, eDenoiseMethod::UNet_b);
}

void test_complex_mat6_unet_filter_bn(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 29;
    const int PixThres = 914;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_unet_filter_bn", olive_mat_desc, SampleCount,
                      FastMinPSNR, PixThres, eDenoiseMethod::UNet_bn);
}

void test_complex_mat6_dof(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 27;
    const double MinPSNR = 22.0;
    const int PixThres = 5198;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_dof", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_DOF1);
}

void test_complex_mat6_mesh_lights(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 30;
    const double MinPSNR = 21.0;
    const int PixThres = 7453;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_mesh_lights", olive_mat_desc, SampleCount,
                      MinPSNR, PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_complex_mat6_sphere_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 22;
    const double MinPSNR = 21.0;
    const int PixThres = 8463;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_sphere_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_SphereLight);
}

void test_complex_mat6_spot_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 6;
    const int PixThres = 273;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_spot_light", olive_mat_desc, SampleCount, FastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_SpotLight);
}

void test_complex_mat6_sun_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 49;
    const double MinPSNR = 19.0;
    const int PixThres = 10374;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_sun_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_SunLight);
}

void test_complex_mat6_hdr_light(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 64;
    const double MinPSNR = 21.0;
    const int PixThres = 6060;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_hdr_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_HDRLight);
}

void test_complex_mat7_refractive(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 8;
    const double MinPSNR = 21.0;
    const int PixThres = 7395;

    Ray::principled_mat_desc_t unused;
    run_material_test(arch_list, preferred_device, "complex_mat7_refractive", unused, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_GlassBall0);
}

void test_complex_mat7_principled(const char *arch_list[], const char *preferred_device) {
    const int SampleCount = 24;
    const double MinPSNR = 21.0;
    const int PixThres = 8193;

    Ray::principled_mat_desc_t unused;
    run_material_test(arch_list, preferred_device, "complex_mat7_principled", unused, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_GlassBall1);
}
