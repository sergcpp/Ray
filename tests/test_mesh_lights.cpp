#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <fstream>

#include "../RendererFactory.h"

#include "utils.h"

#include "test_img1.h"
#include "test_scene1.h"

void test_mesh_lights() {
    const int NUM_SAMPLES = 4096;
    const int DIFF_THRES = 4;
    const int PIX_THRES = 54;

    const Ray::color_rgba8_t white = {255, 255, 255, 255};

    const float view_origin[] = {2.0f, 2.0f, 0.0f};
    const float view_dir[] = {-1.0f, 0.0f, 0.0f};
    const float view_up[] = {0.0f, 1.0f, 0.0f};

    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;

    Ray::tex_desc_t tex_desc1;
    tex_desc1.w = 1;
    tex_desc1.h = 1;
    tex_desc1.generate_mipmaps = true;
    tex_desc1.data = &white;

    Ray::shading_node_desc_t mat_desc1;
    mat_desc1.type = Ray::EmissiveNode;
    mat_desc1.strength = 10.0f;
    mat_desc1.multiple_importance = true;
    mat_desc1.base_color[0] = 1.0f;
    mat_desc1.base_color[1] = 0.0f;
    mat_desc1.base_color[2] = 0.0f;

    Ray::shading_node_desc_t mat_desc2;
    mat_desc2.type = Ray::EmissiveNode;
    mat_desc2.strength = 10.0f;
    mat_desc2.multiple_importance = true;
    mat_desc2.base_color[0] = 0.0f;
    mat_desc2.base_color[1] = 1.0f;
    mat_desc2.base_color[2] = 0.0f;

    Ray::shading_node_desc_t mat_desc3;
    mat_desc3.type = Ray::EmissiveNode;
    mat_desc3.strength = 10.0f;
    mat_desc3.multiple_importance = true;
    mat_desc3.base_color[0] = 0.0f;
    mat_desc3.base_color[1] = 0.0f;
    mat_desc3.base_color[2] = 1.0f;

    Ray::shading_node_desc_t mat_desc4;
    mat_desc4.type = Ray::DiffuseNode;
    mat_desc4.base_color[0] = 1.0f;
    mat_desc4.base_color[1] = 1.0f;
    mat_desc4.base_color[2] = 1.0f;

    Ray::mesh_desc_t mesh_desc;
    mesh_desc.prim_type = Ray::TriangleList;
    mesh_desc.layout = Ray::PxyzNxyzTuv;
    mesh_desc.vtx_attrs = &attrs[0];
    mesh_desc.vtx_attrs_count = attrs_count / 8;
    mesh_desc.vtx_indices = &indices[0];
    mesh_desc.vtx_indices_count = indices_count;

    {
        Ray::settings_t s;
        s.w = 64;
        s.h = 64;

        const Ray::eRendererType renderer_types[] = {
            Ray::RendererRef, /*Ray::RendererSSE2, Ray::RendererAVX, Ray::RendererAVX2,
#if defined(__ANDROID__)
Ray::RendererNEON,
#elif !defined(DISABLE_OCL)
Ray::RendererOCL
#endif*/
        };

        for (const bool use_wide_bvh : {false, true}) {
            s.use_wide_bvh = use_wide_bvh;
            for (const bool output_sh : {false, true}) {
                for (const Ray::eRendererType rt : renderer_types) {
                    auto renderer = std::unique_ptr<Ray::RendererBase>(Ray::CreateRenderer(s, &Ray::g_null_log, rt));
                    auto scene = std::unique_ptr<Ray::SceneBase>(renderer->CreateScene());

                    Ray::camera_desc_t cam_desc;
                    cam_desc.type = Ray::Persp;
                    cam_desc.filter = Ray::Box;
                    cam_desc.dtype = Ray::None;
                    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
                    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
                    memcpy(&cam_desc.up[0], &view_up[0], 3 * sizeof(float));
                    cam_desc.fov = 45.0f;
                    cam_desc.gamma = 1.0f;
                    cam_desc.focus_distance = 1.0f;
                    cam_desc.output_sh = output_sh;
                    cam_desc.clamp = true;

                    const Ray::Camera cam = scene->AddCamera(cam_desc);
                    scene->set_current_cam(cam);

                    scene->SetEnvironment(env_desc);

                    const Ray::Texture t1 = scene->AddTexture(tex_desc1);

                    mat_desc1.base_texture = t1;
                    const Ray::Material m1 = scene->AddMaterial(mat_desc1);

                    mat_desc2.base_texture = t1;
                    const Ray::Material m2 = scene->AddMaterial(mat_desc2);

                    mat_desc3.base_texture = t1;
                    const Ray::Material m3 = scene->AddMaterial(mat_desc3);

                    mat_desc4.base_texture = t1;
                    const Ray::Material m4 = scene->AddMaterial(mat_desc4);

                    mesh_desc.shapes.emplace_back(m1, m1, groups[0], groups[1]);
                    mesh_desc.shapes.emplace_back(m2, m2, groups[2], groups[3]);
                    mesh_desc.shapes.emplace_back(m3, m3, groups[4], groups[5]);
                    mesh_desc.shapes.emplace_back(m4, m4, groups[6], groups[7]);

                    const Ray::Mesh mesh = scene->AddMesh(mesh_desc);

                    const float xform[16] = {1.0f, 0.0f, 0.0f, 0.0f, // NOLINT
                                             0.0f, 1.0f, 0.0f, 0.0f, // NOLINT
                                             0.0f, 0.0f, 1.0f, 0.0f, // NOLINT
                                             0.0f, 0.0f, 0.0f, 1.0f};

                    const Ray::MeshInstance mesh_instance = scene->AddMeshInstance(mesh, xform);
                    (void)mesh_instance;

                    renderer->Clear({0, 0, 0, 0});

                    auto reg = Ray::RegionContext{{0, 0, 64, 64}};
                    for (int i = 0; i < NUM_SAMPLES; ++i) {
                        renderer->RenderScene(scene.get(), reg);
                        const float prog = 100.0f * float(i + 1) / NUM_SAMPLES;
                        printf("\rTest mesh lights (%s, %c, %s): %.1f%% ", Ray::RendererTypeName(rt),
                               use_wide_bvh ? 'w' : 'n', output_sh ? "sh" : "co", prog);
                        fflush(stdout);
                    }

                    // printf("\rTest mesh lights (%s): 100.0%% ", Ray::RendererTypeName(rt));

                    const Ray::pixel_color_t *pixels = renderer->get_pixels_ref();

                    uint8_t img_data_u8[img_w * img_h * 3];
                    uint8_t diff_data_u8[img_w * img_h * 3];
                    int error_pixels = 0;

                    for (int j = 0; j < img_h; j++) {
                        for (int i = 0; i < img_w; i++) {
                            const Ray::pixel_color_t &p = pixels[i];

                            const auto r = uint8_t(p.r * 255);
                            const auto g = uint8_t(p.g * 255);
                            const auto b = uint8_t(p.b * 255);

                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 0] = r;
                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 1] = g;
                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 2] = b;

                            const uint8_t diff_r = std::abs(r - img_data[4 * ((img_h - j - 1) * img_h + i) + 0]);
                            const uint8_t diff_g = std::abs(g - img_data[4 * ((img_h - j - 1) * img_h + i) + 1]);
                            const uint8_t diff_b = std::abs(b - img_data[4 * ((img_h - j - 1) * img_h + i) + 2]);

                            diff_data_u8[3 * ((img_h - j - 1) * img_w + i) + 0] = diff_r;
                            diff_data_u8[3 * ((img_h - j - 1) * img_w + i) + 1] = diff_g;
                            diff_data_u8[3 * ((img_h - j - 1) * img_w + i) + 2] = diff_b;

                            if (diff_r > DIFF_THRES || diff_g > DIFF_THRES || diff_b > DIFF_THRES) {
                                ++error_pixels;
                            }
                        }
                        pixels += img_h;
                    }

                    printf("(error pixels: %i)\n", error_pixels);

                    WriteTGA(&img_data_u8[0], img_w, img_h, 3, "test_data/mesh_lights/out.tga");
                    WriteTGA(&diff_data_u8[0], img_w, img_h, 3, "test_data/mesh_lights/diff.tga");
                    WriteTGA(&img_data[0], img_w, img_h, 4, "test_data/mesh_lights/ref.tga");
                    require((error_pixels <= PIX_THRES) && "Images do not match!");
                }
            }
        }
    }
}
