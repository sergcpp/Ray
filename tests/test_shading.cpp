#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>

#include "../Ray.h"
#include "../internal/TextureUtils.h"

#include "test_scene.h"
#include "thread_pool.h"
#include "utils.h"

extern bool g_determine_sample_count;
extern bool g_minimal_output;
extern bool g_nohwrt;
extern bool g_nodx;
std::mutex g_stdout_mtx;
extern int g_validation_level;

template <typename MatDesc>
void run_material_test(const char *arch_list[], std::string_view preferred_device, const char *test_name,
                       const MatDesc &mat_desc, const int sample_count, const double min_psnr, const int pix_thres,
                       const eDenoiseMethod denoise = eDenoiseMethod::None, const bool partial = false,
                       const char *textures[] = nullptr, const eTestScene test_scene = eTestScene::Standard) {
    run_material_test(arch_list, preferred_device, test_name, mat_desc, sample_count, sample_count, 0.0f, min_psnr,
                      pix_thres, denoise, partial, false, textures, test_scene);
}

template <typename MatDesc>
void run_material_test(const char *arch_list[], std::string_view preferred_device, const char *test_name,
                       const MatDesc &mat_desc, const int min_sample_count, const int max_sample_count,
                       const float variance_threshold, const double min_psnr, const int pix_thres,
                       const eDenoiseMethod denoise = eDenoiseMethod::None, const bool partial = false,
                       const bool caching = false, const char *textures[] = nullptr,
                       const eTestScene test_scene = eTestScene::Standard) {
    using namespace std::chrono;
    using namespace Ray;

    char name_buf[1024];
    snprintf(name_buf, sizeof(name_buf), "test_data/%s/ref.tga", test_name);

    int test_img_w, test_img_h;
    const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
    require_return(!test_img.empty());

    settings_t s;
    s.w = test_img_w;
    s.h = test_img_h;
    s.preferred_device = preferred_device;
    s.validation_level = g_validation_level;
    s.use_spatial_cache = caching;

    ThreadPool threads(std::thread::hardware_concurrency());

    const int DiffThres = 32;

    for (const char **arch = arch_list; *arch; ++arch) {
        const auto rt = RendererTypeFromName(*arch);
        if (g_nodx && rt == eRendererType::DirectX12) {
            continue;
        }

        for (const bool use_hwrt : {false, true}) {
            if (use_hwrt && g_nohwrt) {
                continue;
            }

            s.use_hwrt = use_hwrt;

            int current_sample_count = max_sample_count;
            int failed_count = -1, succeeded_count = 4096;
            bool images_match = false, searching = false;
            do {
                const auto start_time = high_resolution_clock::now();

                using namespace std::placeholders;
                auto parallel_for =
                    std::bind(&ThreadPool::ParallelFor<ParallelForFunction>, std::ref(threads), _1, _2, _3);

                auto renderer = std::unique_ptr<RendererBase>(CreateRenderer(s, &g_log_err, parallel_for, rt));
                if (!renderer || renderer->type() != rt || renderer->is_hwrt() != use_hwrt ||
                    renderer->is_spatial_caching_enabled() != caching) {
                    // skip unsupported (we fell back to some other renderer)
                    break;
                }
                if (!preferred_device.empty()) {
                    // make sure we use requested device
                    if (!require(MatchDeviceNames(renderer->device_name(), preferred_device))) {
                        std::lock_guard<std::mutex> _(g_stdout_mtx);
                        printf("Wrong device: %s (%s was requested)\n", renderer->device_name().data(),
                               preferred_device.data());
                        return;
                    }
                }

                auto scene = std::unique_ptr<SceneBase>(renderer->CreateScene());

                setup_test_scene(threads, *scene, min_sample_count, variance_threshold, mat_desc, textures, test_scene);

                { // test Resize robustness
                    renderer->Resize(test_img_w / 2, test_img_h / 2);
                    renderer->Resize(test_img_w, test_img_h);
                }

                snprintf(name_buf, sizeof(name_buf), "Test %-25s", test_name);
                schedule_render_jobs(threads, *renderer, scene.get(), s, current_sample_count, denoise, partial,
                                     name_buf);

                const color_data_rgba_t pixels = renderer->get_pixels_ref();

                auto img_data_u8 = std::make_unique<uint8_t[]>(test_img_w * test_img_h * 3);
                auto diff_data_u8 = std::make_unique<uint8_t[]>(test_img_w * test_img_h * 3);
                auto mask_data_u8 = std::make_unique<uint8_t[]>(test_img_w * test_img_h * 3);
                memset(&mask_data_u8[0], 0, test_img_w * test_img_h * 3);

                double mse = 0.0;

                int error_pixels = 0;
                for (int j = 0; j < test_img_h; j++) {
                    for (int i = 0; i < test_img_w; i++) {
                        const color_rgba_t &p = pixels.ptr[j * pixels.pitch + i];

                        const auto r = uint8_t(p.v[0] * 255);
                        const auto g = uint8_t(p.v[1] * 255);
                        const auto b = uint8_t(p.v[2] * 255);

                        img_data_u8[3 * (j * test_img_w + i) + 0] = r;
                        img_data_u8[3 * (j * test_img_w + i) + 1] = g;
                        img_data_u8[3 * (j * test_img_w + i) + 2] = b;

                        const uint8_t diff_r = std::abs(r - test_img[4 * (j * test_img_w + i) + 0]);
                        const uint8_t diff_g = std::abs(g - test_img[4 * (j * test_img_w + i) + 1]);
                        const uint8_t diff_b = std::abs(b - test_img[4 * (j * test_img_w + i) + 2]);

                        diff_data_u8[3 * (j * test_img_w + i) + 0] = diff_r;
                        diff_data_u8[3 * (j * test_img_w + i) + 1] = diff_g;
                        diff_data_u8[3 * (j * test_img_w + i) + 2] = diff_b;

                        if (diff_r > DiffThres || diff_g > DiffThres || diff_b > DiffThres) {
                            mask_data_u8[3 * (j * test_img_w + i) + 0] = 255;
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
                        printf("\r%s (%6s, %s): %.1f%% ", name_buf, RendererTypeName(rt).data(),
                               s.use_hwrt ? "HWRT" : "SWRT", 100.0);
                    }
                    printf("(PSNR: %.2f/%.2f dB, Fireflies: %i/%i, Time: %.2fm)\n", psnr, min_psnr, error_pixels,
                           pix_thres, test_duration_m);
                    fflush(stdout);
                }

                std::string type(RendererTypeName(rt));
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
    using namespace Ray;

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
        {"complex_mat6", "complex_mat6_mesh_lights", "complex_mat6_sphere_light", "complex_mat6_hdr_light"},
        {"complex_mat5_regions", "complex_mat5_dof", "complex_mat5_spot_light", "complex_mat6_dof",
         "complex_mat6_spot_light"},
        {"refr_mis2", "complex_mat5_nlm_filter", "complex_mat5_adaptive", "complex_mat5_clipped",
         "complex_mat6_nlm_filter"},
        {"complex_mat5_unet_filter", "complex_mat6_unet_filter", "complex_mat5_dir_light", "complex_mat6_dir_light",
         "complex_mat5_moon_light"}};
    const int ImgCountH = sizeof(test_names) / sizeof(test_names[0]);

    const int OutImageW = 256 * ImgCountW;
    const int OutImageH = 256 * ImgCountH;

    auto material_refs = std::make_unique<uint8_t[]>(OutImageH * OutImageW * 4);
    auto material_imgs = std::make_unique<uint8_t[]>(OutImageH * OutImageW * 4);
    auto material_masks = std::make_unique<uint8_t[]>(OutImageH * OutImageW * 4);
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
            const auto rt = RendererTypeFromName(type.c_str());
            if (hwrt) {
                if (!RendererSupportsHWRT(rt)) {
                    continue;
                }
                type += "_HWRT";
            }

            bool found_at_least_one_image = false;
            for (int j = 0; j < ImgCountH; ++j) {
                for (int i = 0; i < ImgCountW && test_names[j][i]; ++i) {
                    { // reference image
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/ref.tga", test_names[j][i]);

                        int test_img_w, test_img_h;
                        const auto img_ref = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!img_ref.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_refs[((j * 256 + k) * OutImageW + i * 256) * 4],
                                       &img_ref[k * test_img_w * 4], test_img_w * 4);
                            }
                        }

                        blit_chars_to_alpha(material_refs.get(), i * 256, j * 256, test_names[j][i]);
                    }

                    { // test output
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_out.tga", test_names[j][i], type.c_str());

                        int test_img_w, test_img_h;
                        const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!test_img.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_imgs[((j * 256 + k) * OutImageW + i * 256) * 4],
                                       &test_img[k * test_img_w * 4], test_img_w * 4);
                            }
                            found_at_least_one_image = true;
                        }

                        blit_chars_to_alpha(material_imgs.get(), i * 256, j * 256, test_names[j][i]);
                    }

                    { // error mask
                        snprintf(name_buf, sizeof(name_buf), "test_data/%s/%s_mask.tga", test_names[j][i],
                                 type.c_str());

                        int test_img_w, test_img_h;
                        const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
                        if (!test_img.empty()) {
                            for (int k = 0; k < test_img_h; ++k) {
                                memcpy(&material_masks[((j * 256 + k) * OutImageW + i * 256) * 4],
                                       &test_img[k * test_img_w * 4], test_img_w * 4);
                            }
                        }

                        blit_chars_to_alpha(material_masks.get(), i * 256, j * 256, test_names[j][i]);
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

//
// Oren-nayar material tests
//

void test_oren_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 11;
    const int PixThres = 394;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;

    run_material_test(arch_list, preferred_device, "oren_mat0", desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_oren_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 308;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "oren_mat1", desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_oren_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const double MinPSNR = 30.7;
    const int PixThres = 385;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::eShadingNode::Diffuse;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "oren_mat2", desc, SampleCount, MinPSNR, PixThres);
}

//
// Diffuse material tests
//

void test_diff_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 328;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.0f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat0", desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_diff_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 212;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat1", desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_diff_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const double MinPSNR = 30.9;
    const int PixThres = 240;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;
    desc.specular = 0.0f;

    run_material_test(arch_list, preferred_device, "diff_mat2", desc, SampleCount, MinPSNR, PixThres);
}

//
// Sheen material tests
//

void test_sheen_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 215;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 0.5f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat0", mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_sheen_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 155;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat1", mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_sheen_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 265;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat2", mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_sheen_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const double MinPSNR = 30.5;
    const int PixThres = 270;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.roughness = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 1.0f;

    run_material_test(arch_list, preferred_device, "sheen_mat3", mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Glossy material tests
//

void test_glossy_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 440;
    const int PixThres = 375;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "glossy_mat0", node_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_glossy_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 54;
    const int PixThres = 350;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "glossy_mat1", node_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_glossy_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 130;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::eShadingNode::Glossy;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "glossy_mat2", node_desc, SampleCount, DefaultMinPSNR, PixThres);
}

//
// Specular material tests
//

void test_spec_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 440;
    const int PixThres = 375;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat0", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_spec_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 54;
    const int PixThres = 350;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.5f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat1", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_spec_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 130;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 1.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "spec_mat2", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

//
// Anisotropic material tests
//

void test_aniso_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 326;
    const int PixThres = 490;

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

void test_aniso_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 330;
    const int PixThres = 465;

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

void test_aniso_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 336;
    const int PixThres = 455;

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

void test_aniso_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 352;
    const int PixThres = 475;

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

void test_aniso_mat4(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 350;
    const int PixThres = 470;

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

void test_aniso_mat5(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 310;
    const int PixThres = 540;

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

void test_aniso_mat6(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 336;
    const int PixThres = 495;

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

void test_aniso_mat7(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 352;
    const int PixThres = 505;

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

void test_metal_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 86;
    const int PixThres = 1110;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.0f;
    metal_mat_desc.base_color[1] = 0.5f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat0", metal_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_metal_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 16;
    const int PixThres = 350;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.5f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat1", metal_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_metal_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const double MinPSNR = 30.6;
    const int PixThres = 245;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test(arch_list, preferred_device, "metal_mat2", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Plastic material tests
//

void test_plastic_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 30;
    const int PixThres = 890;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.0f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "plastic_mat0", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      PixThres);
}

void test_plastic_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 19;
    const int PixThres = 210;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.0f;
    plastic_mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "plastic_mat1", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      PixThres);
}

void test_plastic_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 14;
    const int PixThres = 275;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "plastic_mat2", plastic_mat_desc, SampleCount, DefaultMinPSNR,
                      PixThres);
}

//
// Tint material tests
//

void test_tint_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 26;
    const int PixThres = 1070;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.0f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "tint_mat0", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_tint_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 27;
    const int PixThres = 1630;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.0f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "tint_mat1", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_tint_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 16;
    const int PixThres = 410;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "tint_mat2", spec_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

//
// Emissive material tests
//

void test_emit_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 90;
    const int PixThres = 390;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 0.5f;

    run_material_test(arch_list, preferred_device, "emit_mat0", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_NoLight);
}

void test_emit_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 158;
    const int PixThres = 475;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 1.0f;

    run_material_test(arch_list, preferred_device, "emit_mat1", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_NoLight);
}

//
// Clear coat material tests
//

void test_coat_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const double MinPSNR = 30.55;
    const int PixThres = 275;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "coat_mat0", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const double MinPSNR = 30.80;
    const int PixThres = 165;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "coat_mat1", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 155;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "coat_mat2", mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

//
// Refractive material tests
//

void test_refr_mis0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 9;
    const double MinPSNR = 30.90;
    const int PixThres = 335;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "refr_mis0", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

void test_refr_mis1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 13;
    const double MinPSNR = 30.65;
    const int PixThres = 265;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "refr_mis1", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

void test_refr_mis2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const double MinPSNR = 30.60;
    const int PixThres = 225;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mis2", mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Refraction_Plane);
}

///

void test_refr_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 295;
    const int PixThres = 1140;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.001f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mat0", mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 382;
    const int PixThres = 765;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "refr_mat1", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 70;
    const int PixThres = 2175;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "refr_mat2", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_refr_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 35;
    const int PixThres = 435;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::eShadingNode::Refractive;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "refr_mat3", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

//
// Transmissive material tests
//

void test_trans_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 309;
    const int PixThres = 1125;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.001f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "trans_mat0", mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 850;
    const int PixThres = 940;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "trans_mat1", mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 254;
    const int PixThres = 930;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.5f;

    run_material_test(arch_list, preferred_device, "trans_mat2", mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 221;
    const int PixThres = 425;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test(arch_list, preferred_device, "trans_mat3", mat_desc, SampleCount, DefaultMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat4(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 264;
    const int PixThres = 810;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test(arch_list, preferred_device, "trans_mat4", mat_desc, SampleCount, FastMinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_trans_mat5(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 64;
    const int PixThres = 200;

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

void test_alpha_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 46;
    const int PixThres = 650;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.75f;

    run_material_test(arch_list, preferred_device, "alpha_mat0", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 52;
    const int PixThres = 535;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.5f;

    run_material_test(arch_list, preferred_device, "alpha_mat1", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 42;
    const int PixThres = 415;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.25f;

    run_material_test(arch_list, preferred_device, "alpha_mat2", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 18;
    const int PixThres = 120;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.0f;

    run_material_test(arch_list, preferred_device, "alpha_mat3", alpha_mat_desc, SampleCount, DefaultMinPSNR, PixThres);
}

void test_alpha_mat4(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 13;
    const double MinPSNR = 30.50;
    const int PixThres = 405;

    Ray::shading_node_desc_t alpha_mat_desc;
    alpha_mat_desc.type = Ray::eShadingNode::Transparent;
    alpha_mat_desc.base_color[0] = 0.75f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.0f;

    run_material_test(arch_list, preferred_device, "alpha_mat4", alpha_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Complex material tests
//

void test_two_sided_mat(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const double MinPSNR = 28.7;
    const int PixThres = 700;

    Ray::principled_mat_desc_t front_mat_desc;
    front_mat_desc.base_color[0] = 0.5f;
    front_mat_desc.base_color[1] = 0.0f;
    front_mat_desc.base_color[2] = 0.0f;
    front_mat_desc.metallic = 1.0f;
    front_mat_desc.roughness = 0.0f;
    front_mat_desc.alpha_texture = Ray::TextureHandle{0};

    const char *textures[] = {"test_data/textures/Fence007A_2K_Opacity.dds"};

    run_material_test(arch_list, preferred_device, "two_sided_mat", front_mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, textures, eTestScene::Two_Sided);
}

void test_complex_mat0(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 11;
    const int PixThres = 760;

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

void test_complex_mat1(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 725;

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

void test_complex_mat2(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 625;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat3(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 8;
    const int PixThres = 400;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat4(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 10;
    const int PixThres = 2165;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};
    metal_mat_desc.alpha_texture = Ray::TextureHandle{4};

    const char *textures[] = {
        "test_data/textures/Fence007A_2K_Color.dds", "test_data/textures/Fence007A_2K_NormalDX.dds",
        "test_data/textures/Fence007A_2K_Roughness.dds", "test_data/textures/Fence007A_2K_Metalness.dds",
        "test_data/textures/Fence007A_2K_Opacity.dds"};

    run_material_test(arch_list, preferred_device, "complex_mat4", metal_mat_desc, SampleCount, VeryFastMinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures);
}

void test_complex_mat5(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 31;
    const int PixThres = 4695;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_clipped(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 37;
    const int PixThres = 5215;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_caching(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 31;
    const int PixThres = 4695;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_caching", metal_mat_desc, SampleCount, SampleCount,
                      0.0f, VeryFastMinPSNR, PixThres, eDenoiseMethod::None, false, true, textures,
                      eTestScene::Standard);
}

void test_complex_mat5_adaptive(const char *arch_list[], std::string_view preferred_device) {
    const int MinSampleCount = 8;
    const int MaxSampleCount = 18;
    const float VarianceThreshold = 0.004f;
    const int PixThres = 2065;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_adaptive", metal_mat_desc, MinSampleCount,
                      MaxSampleCount, VarianceThreshold, FastMinPSNR, PixThres, eDenoiseMethod::NLM, false, false,
                      textures);
}

void test_complex_mat5_regions(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 7;
    const double MinPSNR = 25.50;
    const int PixThres = 3420;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_regions", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, true, textures);
}

void test_complex_mat5_nlm_filter(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 16;
    const int PixThres = 2115;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_unet_filter(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 22;
    const int PixThres = 1150;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_dof(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 24;
    const double MinPSNR = 21.0;
    const int PixThres = 9755;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_mesh_lights(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 29;
    const int PixThres = 4765;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_mesh_lights", metal_mat_desc, SampleCount,
                      VeryFastMinPSNR, PixThres, eDenoiseMethod::None, false, textures,
                      eTestScene::Standard_MeshLights);
}

void test_complex_mat5_sphere_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 40;
    const double MinPSNR = 24.0;
    const int PixThres = 1465;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_inside_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 48;
    const double MinPSNR = 25.0;
    const int PixThres = 2720;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_inside_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_InsideLight);
}

void test_complex_mat5_spot_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 5;
    const double MinPSNR = 31.30;
    const int PixThres = 565;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_spot_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_SpotLight);
}

void test_complex_mat5_dir_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 26;
    const double MinPSNR = 23.0;
    const int PixThres = 5145;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_dir_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_DirLight);
}

void test_complex_mat5_sun_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 15;
    const double MinPSNR = 24.25;
    const int PixThres = 4890;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
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

void test_complex_mat5_moon_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 430;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_moon_light", metal_mat_desc, SampleCount,
                      DefaultMinPSNR, PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_MoonLight);
}

void test_complex_mat5_hdri_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 18;
    const double MinPSNR = 23.0;
    const int PixThres = 6270;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = Ray::TextureHandle{0};
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = Ray::TextureHandle{2};
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = Ray::TextureHandle{3};
    metal_mat_desc.normal_map = Ray::TextureHandle{1};

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test(arch_list, preferred_device, "complex_mat5_hdri_light", metal_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, textures, eTestScene::Standard_HDRLight);
}

void test_complex_mat6(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 43;
    const double MinPSNR = 23.0;
    const int PixThres = 4375;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6", olive_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_complex_mat6_nlm_filter(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const int PixThres = 1610;

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

void test_complex_mat6_unet_filter(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 23;
    const int PixThres = 900;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_unet_filter", olive_mat_desc, SampleCount, FastMinPSNR,
                      PixThres, eDenoiseMethod::UNet);
}

void test_complex_mat6_dof(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 25;
    const double MinPSNR = 22.0;
    const int PixThres = 5015;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_dof", olive_mat_desc, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_DOF1);
}

void test_complex_mat6_mesh_lights(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 41;
    const double MinPSNR = 23.0;
    const int PixThres = 4375;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_mesh_lights", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_MeshLights);
}

void test_complex_mat6_sphere_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 86;
    const double MinPSNR = 23.0;
    const int PixThres = 1175;

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

void test_complex_mat6_spot_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 4;
    const double MinPSNR = 31.15;
    const int PixThres = 195;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_spot_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_SpotLight);
}

void test_complex_mat6_dir_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 86;
    const double MinPSNR = 18.0;
    const int PixThres = 9450;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_dir_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_DirLight);
}

void test_complex_mat6_hdri_light(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 55;
    const double MinPSNR = 21.0;
    const int PixThres = 6300;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test(arch_list, preferred_device, "complex_mat6_hdri_light", olive_mat_desc, SampleCount, MinPSNR,
                      PixThres, eDenoiseMethod::None, false, nullptr, eTestScene::Standard_HDRLight);
}

void test_complex_mat7_refractive(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 5;
    const double MinPSNR = 21.0;
    const int PixThres = 8015;

    Ray::principled_mat_desc_t unused;
    run_material_test(arch_list, preferred_device, "complex_mat7_refractive", unused, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_GlassBall0);
}

void test_complex_mat7_principled(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 12;
    const double MinPSNR = 21.0;
    const int PixThres = 8225;

    Ray::principled_mat_desc_t unused;
    run_material_test(arch_list, preferred_device, "complex_mat7_principled", unused, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::None, false, nullptr, eTestScene::Standard_GlassBall1);
}

void test_ray_flags(const char *arch_list[], std::string_view preferred_device) {
    const int SampleCount = 30;
    const double MinPSNR = 29.55;
    const int PixThres = 2120;

    Ray::principled_mat_desc_t unused;
    run_material_test(arch_list, preferred_device, "ray_flags", unused, SampleCount, MinPSNR, PixThres,
                      eDenoiseMethod::UNet, false, nullptr, eTestScene::Ray_Flags);
}