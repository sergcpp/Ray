#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <memory>
#include <vector>

#include "../../Ray.h"

std::vector<uint8_t> GenerateCheckerboard(int res, int square_size);
void WriteTGA(const Ray::color_rgba_t *data, int pitch, int w, int h, int bpp, const char *name);

int main() {
    const int IMG_W = 256, IMG_H = 256;
    const int SAMPLE_COUNT = 256;

    // Initial frame resolution, can be changed later
    Ray::settings_t s;
    s.w = IMG_W;
    s.h = IMG_H;

    // Additional Ray::eRendererType parameter can be passed (Vulkan GPU renderer created by default)
    Ray::RendererBase *renderer = Ray::CreateRenderer(s, &Ray::g_stdout_log);

    // Each renderer has its own storage implementation (RAM, GPU-RAM),
    // so renderer itself should create scene object
    Ray::SceneBase *scene = renderer->CreateScene();

    // Setup environment
    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;
    scene->SetEnvironment(env_desc);

    // Add checker texture
    const std::vector<uint8_t> tex_data = GenerateCheckerboard(128, 16);
    Ray::tex_desc_t tex_desc;
    tex_desc.format = Ray::eTextureFormat::RGBA8888;
    tex_desc.w = 128;
    tex_desc.h = 128;
    tex_desc.generate_mipmaps = false;
    tex_desc.data = tex_data;

    Ray::TextureHandle checker_tex = scene->AddTexture(tex_desc);

    // Add principled material
    Ray::principled_mat_desc_t mat_desc1;
    mat_desc1.base_texture = checker_tex;
    mat_desc1.roughness = 0.25f;
    mat_desc1.roughness_texture = checker_tex;
    const Ray::MaterialHandle mat0 = scene->AddMaterial(mat_desc1);

    Ray::shading_node_desc_t mat_desc2;
    mat_desc2.type = Ray::eShadingNode::Diffuse;
    mat_desc2.base_color[0] = 0.5f;
    mat_desc2.base_color[1] = 0.5f;
    mat_desc2.base_color[2] = 0.5f;
    const Ray::MaterialHandle mat1 = scene->AddMaterial(mat_desc2);

    mat_desc2.base_color[0] = 0.5f;
    mat_desc2.base_color[1] = 0.0f;
    mat_desc2.base_color[2] = 0.0f;
    const Ray::MaterialHandle mat2 = scene->AddMaterial(mat_desc2);

    mat_desc2.base_color[0] = 0.0f;
    mat_desc2.base_color[1] = 0.5f;
    mat_desc2.base_color[2] = 0.0f;
    const Ray::MaterialHandle mat3 = scene->AddMaterial(mat_desc2);

    // Add emissive material
    Ray::shading_node_desc_t mat_desc3;
    mat_desc3.type = Ray::eShadingNode::Emissive;
    mat_desc3.strength = 100.0f;
    mat_desc3.importance_sample = true; // Use NEE for this lightsource
    const Ray::MaterialHandle mat4 = scene->AddMaterial(mat_desc3);

    // Setup test mesh
    // position(3 floats), normal(3 floats), tex_coord(2 floats)
    // clang-format off
    const float attrs[] = { // floor
                            0.0f, 0.0f, -0.5592f,           0.0f, 1.0f, 0.0f,                   1.0f, 1.0f,
                            0.0f, 0.0f, 0.0f,               0.0f, 1.0f, 0.0f,                   1.0f, 0.0f,
                            -0.5528f, 0.0f, 0.0f,           0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.5496f, 0.0f, -0.5592f,       0.0f, 1.0f, 0.0f,                   0.0f, 1.0f,
                            // back wall
                            0.0f, 0.0f, -0.5592f,           0.0f, 0.0f, 1.0f,                   0.0f, 0.0f,
                            -0.5496f, 0.0f, -0.5592f,       0.0f, 0.0f, 1.0f,                   0.0f, 0.0f,
                            -0.556f, 0.5488f, -0.5592f,     0.0f, 0.0f, 1.0f,                   0.0f, 0.0f,
                            0.0f, 0.5488f, -0.5592f,        0.0f, 0.0f, 1.0f,                   0.0f, 0.0f,
                            // ceiling
                            -0.556f, 0.5488f, -0.5592f,     0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            0.0f, 0.5488f, -0.5592f,        0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            0.0f, 0.5488f, 0.0f,            0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            -0.556f, 0.5488f, 0.0f,         0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            // left wall
                            -0.5528f, 0.0f, 0.0f,           1.0f, 0.0f, 0.0f,                   0.0f, 0.0f,
                            -0.5496f, 0.0f, -0.5592f,       1.0f, 0.0f, 0.0f,                   0.0f, 0.0f,
                            -0.556f, 0.5488f, 0.0f,         1.0f, 0.0f, 0.0f,                   0.0f, 0.0f,
                            -0.556f, 0.5488f, -0.5592f,     1.0f, 0.0f, 0.0f,                   0.0f, 0.0f,
                            // right wall
                            0.0f, 0.0f, -0.5592f,           -1.0f, 0.0f, 0.0f,                  0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f,               -1.0f, 0.0f, 0.0f,                  0.0f, 0.0f,
                            0.0f, 0.5488f, -0.5592f,        -1.0f, 0.0f, 0.0f,                  0.0f, 0.0f,
                            0.0f, 0.5488f, 0.0f,            -1.0f, 0.0f, 0.0f,                  0.0f, 0.0f,
                            // light
                            -0.213f, 0.5478f, -0.227f,      0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            -0.343f, 0.5478f, -0.227f,      0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            -0.343f, 0.5478f, -0.332f,      0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            -0.213f, 0.5478f, -0.332f,      0.0f, -1.0f, 0.0f,                  0.0f, 0.0f,
                            // short block
                            -0.240464f, 0.0f, -0.271646f,   0.285951942f, 0.0f, -0.958243966f,  0.0f, 0.0f,
                            -0.240464f, 0.165f, -0.271646f, 0.285951942f, 0.0f, -0.958243966f,  0.0f, 0.0f,
                            -0.082354f, 0.165f, -0.224464f, 0.285951942f, 0.0f, -0.958243966f,  0.0f, 0.0f,
                            -0.082354f, 0.0f, -0.224464f,   0.285951942f, 0.0f, -0.958243966f,  0.0f, 0.0f,
                            -0.240464f, 0.0, -0.271646f,    -0.958243966f, 0.0f, -0.285951942f, 0.0f, 0.0f,
                            -0.240464f, 0.165f, -0.271646f, -0.958243966f, 0.0f, -0.285951942f, 0.0f, 0.0f,
                            -0.287646f, 0.165f, -0.113536f, -0.958243966f, 0.0f, -0.285951942f, 0.0f, 0.0f,
                            -0.287646f, 0.0f, -0.113536f,   -0.958243966f, 0.0f, -0.285951942f, 0.0f, 0.0f,
                            -0.082354f, 0.0f, -0.224464f,   0.958243966f, 0.0f, 0.285951942f,   0.0f, 0.0f,
                            -0.082354f, 0.165f, -0.224464f, 0.958243966f, 0.0f, 0.285951942f,   0.0f, 0.0f,
                            -0.129536f, 0.165f, -0.066354f, 0.958243966f, 0.0f, 0.285951942f,   0.0f, 0.0f,
                            -0.129536f, 0.0f, -0.066354f,   0.958243966f, 0.0f, 0.285951942f,   0.0f, 0.0f,
                            -0.287646f, 0.0f, -0.113536f,   -0.285951942f, 0.0f, 0.958243966f,  0.0f, 0.0f,
                            -0.287646f, 0.165f, -0.113536f, -0.285951942f, 0.0f, 0.958243966f,  0.0f, 0.0f,
                            -0.129536f, 0.165f, -0.066354f, -0.285951942f, 0.0f, 0.958243966f,  0.0f, 0.0f,
                            -0.129536f, 0.0f, -0.066354f,   -0.285951942f, 0.0f, 0.958243966f,  0.0f, 0.0f,
                            -0.240464f, 0.165f, -0.271646f, 0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.082354f, 0.165f, -0.224464f, 0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.129536f, 0.165f, -0.066354f, 0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.287646f, 0.165f, -0.113536f, 0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            // tall block
                            -0.471239f, 0.0f, -0.405353f,   -0.296278358f, 0.0f, -0.955101609f, 0.0f, 0.0f,
                            -0.471239f, 0.33f, -0.405353f,  -0.296278358f, 0.0f, -0.955101609f, 0.0f, 0.0f,
                            -0.313647f, 0.33f, -0.454239f,  -0.296278358f, 0.0f, -0.955101609f, 0.0f, 0.0f,
                            -0.313647f, 0.0f, -0.454239f,   -0.296278358f, 0.0f, -0.955101609f, 0.0f, 0.0f,
                            -0.264761f, 0.0f, -0.296647f,   0.955101609f, 0.0f, -0.296278358f,  0.0f, 0.0f,
                            -0.264761f, 0.33f, -0.296647f,  0.955101609f, 0.0f, -0.296278358f,  0.0f, 0.0f,
                            -0.313647f, 0.33f, -0.454239f,  0.955101609f, 0.0f, -0.296278358f,  0.0f, 0.0f,
                            -0.313647f, 0.0f, -0.454239f,   0.955101609f, 0.0f, -0.296278358f,  0.0f, 0.0f,
                            -0.471239f, 0.0f, -0.405353f,   -0.955101609f, 0.0f, 0.296278358f,  0.0f, 0.0f,
                            -0.471239f, 0.33f, -0.405353f,  -0.955101609f, 0.0f, 0.296278358f,  0.0f, 0.0f,
                            -0.422353f, 0.33f, -0.247761f,  -0.955101609f, 0.0f, 0.296278358f,  0.0f, 0.0f,
                            -0.422353f, 0.0f, -0.247761f,   -0.955101609f, 0.0f, 0.296278358f,  0.0f, 0.0f,
                            -0.422353f, 0.0f, -0.247761f,   0.296278358f, 0.0f, 0.955101609f,   0.0f, 0.0f,
                            -0.422353f, 0.33f, -0.247761f,  0.296278358f, 0.0f, 0.955101609f,   0.0f, 0.0f,
                            -0.264761f, 0.33f, -0.296647f,  0.296278358f, 0.0f, 0.955101609f,   0.0f, 0.0f,
                            -0.264761f, 0.0f, -0.296647f,   0.296278358f, 0.0f, 0.955101609f,   0.0f, 0.0f,
                            -0.471239f, 0.33f, -0.405353f,  0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.313647f, 0.33f, -0.454239f,  0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.264761f, 0.33f, -0.296647f,  0.0f, 1.0f, 0.0f,                   0.0f, 0.0f,
                            -0.422353f, 0.33f, -0.247761f,  0.0f, 1.0f, 0.0f,                   0.0f, 0.0f};
    const uint32_t indices[] = { 0, 2, 1, 0, 3, 2,
                                 4, 6, 5, 4, 7, 6,
                                 8, 9, 10, 8, 10, 11,
                                 12, 13, 14, 13, 15, 14,
                                 16, 17, 18, 18, 17, 19,
                                 20, 21, 22, 20, 22, 23,
                                 24, 25, 26, 24, 26, 27,
                                 28, 30, 29, 28, 31, 30,
                                 32, 33, 34, 32, 34, 35,
                                 36, 38, 37, 36, 39, 38,
                                 40, 42, 41, 40, 43, 42,
                                 44, 45, 46, 44, 46, 47,
                                 48, 50, 49, 48, 51, 50,
                                 52, 54, 53, 52, 55, 54,
                                 56, 58, 57, 56, 59, 58,
                                 60, 62, 61, 60, 63, 62};
    // clang-format on

    Ray::mesh_desc_t mesh_desc;
    mesh_desc.prim_type = Ray::ePrimType::TriangleList;
    mesh_desc.vtx_positions = {attrs, 0, 8};
    mesh_desc.vtx_normals = {attrs, 3, 8};
    mesh_desc.vtx_uvs = {attrs, 6, 8};
    mesh_desc.vtx_indices = indices;

    // Setup material groups
    const Ray::mat_group_desc_t groups[] = {
        {mat0, 0, 6},  {mat1, 6, 12}, {mat2, 19, 6}, {mat3, 25, 6}, {mat4, Ray::InvalidMaterialHandle, 31, 6},
        {mat1, 37, 60}};
    mesh_desc.groups = groups;

    Ray::MeshHandle mesh1 = scene->AddMesh(mesh_desc);

    // Instantiate mesh
    const float xform[] = {1.0f, 0.0f, 0.0f, 0.0f, //
                           0.0f, 1.0f, 0.0f, 0.0f, //
                           0.0f, 0.0f, 1.0f, 0.0f, //
                           0.0f, 0.0f, 0.0f, 1.0f};
    scene->AddMeshInstance(mesh1, xform);

    // Add camera
    const float view_origin[] = {-0.278f, 0.273f, 0.8f};
    const float view_dir[] = {0.0f, 0.0f, -1.0f};

    Ray::camera_desc_t cam_desc;
    cam_desc.type = Ray::eCamType::Persp;
    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
    cam_desc.fov = 39.1463f;

    const Ray::CameraHandle cam = scene->AddCamera(cam_desc);
    scene->set_current_cam(cam);

    scene->Finalize();

    // Render image
    if (Ray::RendererSupportsMultithreading(renderer->type())) {
        // Split image into 4 regions
        Ray::RegionContext regions[] = {Ray::RegionContext{{0, 0, IMG_W / 2, IMG_H / 2}},
                                        Ray::RegionContext{{IMG_W / 2, 0, IMG_W / 2, IMG_H / 2}},
                                        Ray::RegionContext{{0, IMG_H / 2, IMG_W / 2, IMG_H / 2}},
                                        Ray::RegionContext{{IMG_W / 2, IMG_H / 2, IMG_W / 2, IMG_H / 2}}};
#pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < SAMPLE_COUNT; j++) {
                renderer->RenderScene(*scene, regions[i]);
            }
        }
    } else {
        // Create region contex for frame, setup to use whole frame
        auto region = Ray::RegionContext{{0, 0, IMG_W, IMG_H}};
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            // Each call performs one iteration, blocks until finished
            renderer->RenderScene(*scene, region);
            printf("Renderered %i samples\n", i);
        }
    }
    printf("Done\n");

    // Get rendered image pixels in 32-bit floating point RGBA format
    const Ray::color_data_rgba_t pixels = renderer->get_pixels_ref();

    // Save image
    WriteTGA(pixels.ptr, pixels.pitch, IMG_W, IMG_H, 3, "03_principled.tga");
    printf("Image saved as samples/03_principled.tga\n");

    delete scene;
    delete renderer;
}

#define float_to_byte(val)                                                                                             \
    (((val) <= 0.0f) ? 0 : (((val) > (1.0f - 0.5f / 255.0f)) ? 255 : uint8_t((255.0f * (val)) + 0.5f)))

void WriteTGA(const Ray::color_rgba_t *data, int pitch, const int w, const int h, const int bpp, const char *name) {
    if (!pitch) {
        pitch = w;
    }

    std::ofstream file(name, std::ios::binary);

    unsigned char header[18] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    header[12] = w & 0xFF;
    header[13] = (w >> 8) & 0xFF;
    header[14] = (h)&0xFF;
    header[15] = (h >> 8) & 0xFF;
    header[16] = bpp * 8;
    header[17] |= (1 << 5); // set origin to upper left corner

    file.write((char *)&header[0], sizeof(header));

    auto out_data = std::unique_ptr<uint8_t[]>{new uint8_t[size_t(w) * h * bpp]};
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            out_data[(j * w + i) * bpp + 0] = float_to_byte(data[j * pitch + i].v[2]);
            out_data[(j * w + i) * bpp + 1] = float_to_byte(data[j * pitch + i].v[1]);
            out_data[(j * w + i) * bpp + 2] = float_to_byte(data[j * pitch + i].v[0]);
            if (bpp == 4) {
                out_data[i * 4 + 3] = float_to_byte(data[j * pitch + i].v[3]);
            }
        }
    }

    file.write((const char *)&out_data[0], size_t(w) * h * bpp);

    static const char footer[26] = "\0\0\0\0"         // no extension area
                                   "\0\0\0\0"         // no developer directory
                                   "TRUEVISION-XFILE" // yep, this is a TGA file
                                   ".";
    file.write(footer, sizeof(footer));
}

std::vector<uint8_t> GenerateCheckerboard(const int res, const int square_size) {
    std::vector<uint8_t> ret(4 * res * res);

    for (int i = 0; i < res; i++) {
        for (int j = 0; j < res; j++) {
            const int index = i * res + j;
            const int square_x = j / square_size;
            const int square_y = i / square_size;

            if ((square_x + square_y) % 2 == 0) {
                ret[4 * index + 0] = 10;
                ret[4 * index + 1] = 10;
                ret[4 * index + 2] = 10;
                ret[4 * index + 3] = 255;
            } else {
                ret[4 * index + 0] = 250;
                ret[4 * index + 1] = 250;
                ret[4 * index + 2] = 250;
                ret[4 * index + 3] = 255;
            }
        }
    }

    return ret;
}