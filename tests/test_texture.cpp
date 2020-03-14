#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <fstream>

#include "../RendererFactory.h"

#include "test_scene2.h"

void WriteTGA(const uint8_t *data, int w, int h, int bpp, const std::string &name);

void test_texture() {
    const int NUM_SAMPLES = 256;

    // Setup camera
    const float view_origin[] = { 0.0f, 10.0f, 0.0f };
    const float view_dir[] = { 0.0f, -1.0f, 0.0f };
    const float view_up[] = { 0.0f, 0.0f, -1.0f };

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
    const int ImgRes = 512;
    std::unique_ptr<Ray::pixel_color8_t[]> checker_texture(new Ray::pixel_color8_t[ImgRes * ImgRes]);

    for (int j = 0; j < ImgRes; j++) {
        for (int i = 0; i < ImgRes; i++) {
            Ray::pixel_color8_t &p = checker_texture[j * ImgRes + i];

            p.r = p.g = p.b = ((i + j) % 2) ? 255 : 0;
            p.a = 255;
        }
    }

    Ray::tex_desc_t tex_desc;
    tex_desc.w = tex_desc.h = ImgRes;
    tex_desc.generate_mipmaps = true;
    tex_desc.is_srgb = false;
    tex_desc.data = checker_texture.get();

    // Create material
    Ray::mat_desc_t mat_desc;
    mat_desc.type = Ray::EmissiveMaterial;

    // Create mesh
    Ray::mesh_desc_t mesh_desc;
    mesh_desc.prim_type = Ray::TriangleList;
    mesh_desc.layout = Ray::PxyzNxyzTuv;
    mesh_desc.vtx_attrs = &attrs[0];
    mesh_desc.vtx_attrs_count = attrs_count / 8;
    mesh_desc.vtx_indices = &indices[0];
    mesh_desc.vtx_indices_count = indices_count;

    const float xform[16] = { 1.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 1.0f };

    {
        const int ImgRes = 256;

        Ray::settings_t s;
        s.w = s.h = ImgRes;

        std::shared_ptr<Ray::RendererBase> renderer;

        Ray::eRendererType renderer_types[] = { Ray::RendererRef, Ray::RendererSSE2, Ray::RendererAVX, Ray::RendererAVX2,
#if defined(__ANDROID__)
            Ray::RendererNEON,
#elif !defined(DISABLE_OCL)
			Ray::RendererOCL
#endif
			};
       
        // bvh_type 0 - shallow binary bvh
        // bvh_type 1 - wide oct bvh

        for (int bvh_type = 0; bvh_type < 2; bvh_type++) {
            // mode 0 - default
            // mode 1 - with additional SH generation, different sampling method is used, so should be tested separately

            for (Ray::eRendererType rt : renderer_types) {
                s.use_wide_bvh = bvh_type == 0 ? false : true;

                renderer = Ray::CreateRenderer(s, rt);

                auto scene = renderer->CreateScene();

                uint32_t cam = scene->AddCamera(cam_desc);
                scene->set_current_cam(cam);

                scene->SetEnvironment(env_desc);

                uint32_t tex_id = scene->AddTexture(tex_desc);

                mat_desc.main_texture = tex_id;
                uint32_t mat_id = scene->AddMaterial(mat_desc);

                mesh_desc.shapes.push_back({ mat_id, 0, indices_count });

                uint32_t mesh_id = scene->AddMesh(mesh_desc);

                scene->AddMeshInstance(mesh_id, xform);

                renderer->Clear();

                int prog = 0;

                printf("0%%\n");

                auto reg = Ray::RegionContext{ { 0, 0, ImgRes, ImgRes } };
                for (int i = 0; i < NUM_SAMPLES; i++) {
                    renderer->RenderScene(scene, reg);
                    float new_prog = float(100 * i) / NUM_SAMPLES;
                    if (new_prog - prog > 10) {
                        prog = (int)new_prog;
                        printf("%i%%\n", prog);
                    }
                }

                printf("100%%\n");

                const Ray::pixel_color_t *pixels = renderer->get_pixels_ref();

                float diff = 0;

                uint8_t img_data_u8[ImgRes * ImgRes * 3];

                for (int j = 0; j < ImgRes; j++) {
                    for (int i = 0; i < ImgRes; i++) {
                        const Ray::pixel_color_t &p = pixels[j * ImgRes + i];

                        uint8_t r = uint8_t(p.r * 255);
                        uint8_t g = uint8_t(p.g * 255);
                        uint8_t b = uint8_t(p.b * 255);
                        //uint8_t a = uint8_t(p.a * 255);

                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 0] = r;
                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 1] = g;
                        img_data_u8[3 * ((ImgRes - j - 1) * ImgRes + i) + 2] = b;

                        diff += std::abs(p.r - 0.5f);
                        diff += std::abs(p.g - 0.5f);
                        diff += std::abs(p.b - 0.5f);
                    }
                }

                double d = double(diff) / (ImgRes * ImgRes * 3);

                printf("Error: %f\n", d);

                if (d > 0.0015) {
                    WriteTGA(&img_data_u8[0], ImgRes, ImgRes, 3, "test_texture_fail.tga");
                    require(false && "image is wrong");
                }
            }
        }
    }
}
