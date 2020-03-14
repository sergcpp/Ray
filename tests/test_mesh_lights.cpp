#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <fstream>

#include "../RendererFactory.h"

#include "test_scene1.h"
#include "test_img1.h"

void WriteTGA(const uint8_t *data, int w, int h, int bpp, const std::string &name) {
    std::ofstream file(name, std::ios::binary);

    unsigned char header[18] = { 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    header[12] = w & 0xFF;
    header[13] = (w >> 8) & 0xFF;
    header[14] = (h) & 0xFF;
    header[15] = (h >> 8) & 0xFF;
    header[16] = bpp * 8;

    file.write((char *)&header[0], sizeof(unsigned char) * 18);

    auto out_data = std::unique_ptr<uint8_t[]>{ new uint8_t[w * h * bpp] };
    if (bpp == 3) {
        for (int i = 0; i < w * h; i++) {
            out_data[i * 3 + 0] = data[i * 3 + 2];
            out_data[i * 3 + 1] = data[i * 3 + 1];
            out_data[i * 3 + 2] = data[i * 3 + 0];
        }
    } else {
        for (int i = 0; i < w * h; i++) {
            out_data[i * 4 + 0] = data[i * 4 + 2];
            out_data[i * 4 + 1] = data[i * 4 + 1];
            out_data[i * 4 + 2] = data[i * 4 + 0];
            out_data[i * 4 + 3] = data[i * 4 + 3];
        }
    }

    file.write((const char *)&out_data[0], w * h * bpp);

    static const char footer[26] = "\0\0\0\0" // no extension area
        "\0\0\0\0"// no developer directory
        "TRUEVISION-XFILE"// yep, this is a TGA file
        ".";
    file.write((const char *)&footer, sizeof(footer));
}

void test_mesh_lights() {
    const int NUM_SAMPLES = 2048;

    const Ray::pixel_color8_t white = { 255, 255, 255, 255 };

    const float view_origin[] = { 2.0f, 2.0f, 0.0f };
    const float view_dir[] = { -1.0f, 0.0f, 0.0f };

    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;

    Ray::tex_desc_t tex_desc1;
    tex_desc1.w = 1;
    tex_desc1.h = 1;
    tex_desc1.generate_mipmaps = true;
    tex_desc1.data = &white;

    Ray::mat_desc_t mat_desc1;
    mat_desc1.type = Ray::EmissiveMaterial;
    mat_desc1.strength = 10.0f;
    mat_desc1.main_color[0] = 1.0f;
    mat_desc1.main_color[1] = 0.0f;
    mat_desc1.main_color[2] = 0.0f;

    Ray::mat_desc_t mat_desc2;
    mat_desc2.type = Ray::EmissiveMaterial;
    mat_desc2.strength = 10.0f;
    mat_desc2.main_color[0] = 0.0f;
    mat_desc2.main_color[1] = 1.0f;
    mat_desc2.main_color[2] = 0.0f;

    Ray::mat_desc_t mat_desc3;
    mat_desc3.type = Ray::EmissiveMaterial;
    mat_desc3.strength = 10.0f;
    mat_desc3.main_color[0] = 0.0f;
    mat_desc3.main_color[1] = 0.0f;
    mat_desc3.main_color[2] = 1.0f;

    Ray::mat_desc_t mat_desc4;
    mat_desc4.type = Ray::DiffuseMaterial;
    mat_desc4.main_color[0] = 1.0f;
    mat_desc4.main_color[1] = 1.0f;
    mat_desc4.main_color[2] = 1.0f;

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

            for (int mode = 0; mode < 2; mode++) {
                for (Ray::eRendererType rt : renderer_types) {
                    s.use_wide_bvh = bvh_type == 0 ? false : true;

                    renderer = Ray::CreateRenderer(s, rt);

                    std::shared_ptr<Ray::SceneBase> scene = renderer->CreateScene();

                    Ray::camera_desc_t cam_desc;
                    cam_desc.type = Ray::Persp;
                    cam_desc.filter = Ray::Box;
                    cam_desc.dtype = Ray::None;
                    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
                    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
                    cam_desc.fov = 45.0f;
                    cam_desc.gamma = 1.0f;
                    cam_desc.focus_distance = 1.0f;
                    cam_desc.focus_factor = 0.0f;
                    if (mode == 0) {
                        cam_desc.output_sh = false;
                    } else if (mode == 1) {
                        cam_desc.output_sh = true;
                    }

                    const uint32_t cam = scene->AddCamera(cam_desc);
                    scene->set_current_cam(cam);

                    scene->SetEnvironment(env_desc);

                    const uint32_t t1 = scene->AddTexture(tex_desc1);

                    mat_desc1.main_texture = t1;
                    const uint32_t m1 = scene->AddMaterial(mat_desc1);

                    mat_desc2.main_texture = t1;
                    const uint32_t m2 = scene->AddMaterial(mat_desc2);

                    mat_desc3.main_texture = t1;
                    const uint32_t m3 = scene->AddMaterial(mat_desc3);

                    mat_desc4.main_texture = t1;
                    const uint32_t m4 = scene->AddMaterial(mat_desc4);

                    mesh_desc.shapes.push_back({ m1, groups[0], groups[1] });
                    mesh_desc.shapes.push_back({ m2, groups[2], groups[3] });
                    mesh_desc.shapes.push_back({ m3, groups[4], groups[5] });
                    mesh_desc.shapes.push_back({ m4, groups[6], groups[7] });

                    const uint32_t mesh = scene->AddMesh(mesh_desc);

                    const float xform[16] = {
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f
                    };

                    const uint32_t mesh_instance = scene->AddMeshInstance(mesh, xform);
                    (void)mesh_instance;

                    renderer->Clear();

                    int prog = 0;

                    printf("0%%\n");

                    auto reg = Ray::RegionContext{ { 0, 0, 64, 64 } };
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

                    uint64_t diff = 0;

                    uint8_t img_data_u8[img_w * img_h * 3];

                    for (int j = 0; j < img_h; j++) {
                        for (int i = 0; i < img_w; i++) {
                            Ray::pixel_color_t p = pixels[j * img_w + i];

                            if (p.r > 1.0f) p.r = 1.0f;
                            if (p.g > 1.0f) p.g = 1.0f;
                            if (p.b > 1.0f) p.b = 1.0f;

                            uint8_t r = uint8_t(p.r * 255);
                            uint8_t g = uint8_t(p.g * 255);
                            uint8_t b = uint8_t(p.b * 255);
                            //uint8_t a = uint8_t(p.a * 255);

                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 0] = r;
                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 1] = g;
                            img_data_u8[3 * ((img_h - j - 1) * img_w + i) + 2] = b;

                            diff += std::abs(r - img_data[4 * ((img_h - j - 1) * img_h + i) + 0]);
                            diff += std::abs(g - img_data[4 * ((img_h - j - 1) * img_h + i) + 1]);
                            diff += std::abs(b - img_data[4 * ((img_h - j - 1) * img_h + i) + 2]);
                        }
                    }

                    const double d = double(diff) / (img_w * img_h * 3);

                    printf("Error: %f\n", d);

                    if (d > 1.0) {
                        WriteTGA(&img_data_u8[0], img_w, img_h, 3, "test_mesh_lights_fail.tga");
                        WriteTGA(&img_data[0], img_w, img_h, 4, "test_mesh_lights_true.tga");
                        require(false && "images do not match");
                    }
                }
            }
        }
    }
}
