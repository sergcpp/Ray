#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <fstream>

#include "../RendererFactory.h"

#include "test_scene2.h"
#include "utils.h"

void test_texture() {
    const int NUM_SAMPLES = 256;
    const int DIFF_THRES = 6;
    const int PIX_THRES = 50;

    // Setup camera
    const float view_origin[] = {0.0f, 10.0f, 0.0f};
    const float view_dir[] = {0.0f, -1.0f, 0.0f};
    const float view_up[] = {0.0f, 0.0f, -1.0f};

    Ray::camera_desc_t cam_desc;
    cam_desc.type = Ray::Persp;
    cam_desc.filter = Ray::Box;
    cam_desc.dtype = Ray::None;
    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
    memcpy(&cam_desc.up[0], &view_up[0], 3 * sizeof(float));
    cam_desc.fov = 90.0f;
    cam_desc.gamma = 1.0f;
    cam_desc.focus_distance = 1.0f;
    cam_desc.focus_factor = 0.0f;

    // Setup environment
    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.5f;

    // Create texture
    const int TexRes = 512;
    std::unique_ptr<Ray::color_rgba8_t[]> checker_texture(new Ray::color_rgba8_t[TexRes * TexRes]);

    for (int j = 0; j < TexRes; ++j) {
        for (int i = 0; i < TexRes; ++i) {
            Ray::color_rgba8_t &p = checker_texture[j * TexRes + i];

            p.v[0] = p.v[1] = p.v[2] = ((i + j) % 2) ? 255 : 0;
            p.v[3] = 255;
        }
    }

    Ray::tex_desc_t tex_desc;
    tex_desc.w = tex_desc.h = TexRes;
    tex_desc.generate_mipmaps = true;
    tex_desc.is_srgb = false;
    tex_desc.data = checker_texture.get();

    // Create material
    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::EmissiveNode;

    // Create mesh
    Ray::mesh_desc_t mesh_desc;
    mesh_desc.prim_type = Ray::TriangleList;
    mesh_desc.layout = Ray::PxyzNxyzTuv;
    mesh_desc.vtx_attrs = &attrs[0];
    mesh_desc.vtx_attrs_count = attrs_count / 8;
    mesh_desc.vtx_indices = &indices[0];
    mesh_desc.vtx_indices_count = indices_count;

    const float xform[16] = {1.0f, 0.0f, 0.0f, 0.0f, // NOLINT
                             0.0f, 1.0f, 0.0f, 0.0f, // NOLINT
                             0.0f, 0.0f, 1.0f, 0.0f, // NOLINT
                             0.0f, 0.0f, 0.0f, 1.0f};

    {
        const int ImgRes = 256;

        Ray::settings_t s;
        s.w = s.h = ImgRes;

        const Ray::eRendererType renderer_types[] = {
            Ray::RendererRef,
            /*Ray::RendererSSE2,
            Ray::RendererAVX,
            Ray::RendererAVX2,
#if defined(__ANDROID__)
            Ray::RendererNEON,
#elif !defined(DISABLE_OCL)
            Ray::RendererOCL
#endif*/
        };

        for (const bool use_wide_bvh : {false, true}) {
            s.use_wide_bvh = use_wide_bvh;
            for (const Ray::eRendererType rt : renderer_types) {
                auto renderer = std::unique_ptr<Ray::RendererBase>(Ray::CreateRenderer(s, &Ray::g_null_log, rt));
                auto scene = std::unique_ptr<Ray::SceneBase>(renderer->CreateScene());

                const uint32_t cam = scene->AddCamera(cam_desc);
                scene->set_current_cam(cam);

                scene->SetEnvironment(env_desc);

                const uint32_t tex_id = scene->AddTexture(tex_desc);

                mat_desc.base_texture = tex_id;
                const uint32_t mat_id = scene->AddMaterial(mat_desc);

                mesh_desc.shapes.push_back({mat_id, mat_id, 0, indices_count});

                const uint32_t mesh_id = scene->AddMesh(mesh_desc);

                scene->AddMeshInstance(mesh_id, xform);

                renderer->Clear();

                auto reg = Ray::RegionContext{{0, 0, ImgRes, ImgRes}};
                for (int i = 0; i < NUM_SAMPLES; ++i) {
                    renderer->RenderScene(scene.get(), reg);
                    const float prog = 100.0f * float(i + 1) / NUM_SAMPLES;
                    printf("\rTest texture (%s, %c): %.1f%% ", Ray::RendererTypeName(rt), use_wide_bvh ? 'w' : 'n',
                           prog);
                    fflush(stdout);
                }

                const Ray::pixel_color_t *pixels = renderer->get_pixels_ref();

                uint8_t img_data_u8[ImgRes * ImgRes * 3];
                uint8_t diff_data_u8[ImgRes * ImgRes * 3];
                int error_pixels = 0;

                for (int j = 0; j < ImgRes; j++) {
                    for (int i = 0; i < ImgRes; i++) {
                        const Ray::pixel_color_t &p = pixels[i];

                        const uint8_t r = uint8_t(p.r * 255);
                        const uint8_t g = uint8_t(p.g * 255);
                        const uint8_t b = uint8_t(p.b * 255);

                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 0] = r;
                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 1] = g;
                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 2] = b;

                        const uint8_t diff_r = std::abs(r - 127);
                        const uint8_t diff_g = std::abs(g - 127);
                        const uint8_t diff_b = std::abs(b - 127);

                        diff_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 0] = diff_r;
                        diff_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 1] = diff_g;
                        diff_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 2] = diff_b;

                        if (diff_r > DIFF_THRES || diff_g > DIFF_THRES || diff_b > DIFF_THRES) {
                            ++error_pixels;
                        }
                    }
                    pixels += ImgRes;
                }

                printf("(error pixels: %i)\n", error_pixels);

                WriteTGA(&img_data_u8[0], ImgRes, ImgRes, 3, "test_data/texture/out.tga");
                WriteTGA(&diff_data_u8[0], ImgRes, ImgRes, 3, "test_data/texture/diff.tga");
                require((error_pixels <= PIX_THRES) && "Images do not match!");
            }
        }
    }
}
