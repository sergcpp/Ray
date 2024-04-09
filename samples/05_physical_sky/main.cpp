#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <memory>

#include "../../Ray.h"

void WriteTGA(const Ray::color_rgba_t *data, int pitch, int w, int h, int bpp, const char *name);

int main() {
    const int IMG_W = 256, IMG_H = 256;
    const int SAMPLE_COUNT = 64;

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
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 1.0f;
    env_desc.back_col[0] = env_desc.back_col[1] = env_desc.back_col[2] = 1.0f;
    env_desc.env_map = env_desc.back_map = Ray::PhysicalSkyTexture;
    scene->SetEnvironment(env_desc);

    // Add directional light
    Ray::directional_light_desc_t dir_light_desc;
    dir_light_desc.color[0] = dir_light_desc.color[1] = dir_light_desc.color[2] = 12.0f;
    dir_light_desc.direction[0] = 0.0f;
    dir_light_desc.direction[1] = -0.0871558040f;
    dir_light_desc.direction[2] = 0.996194720f;
    dir_light_desc.angle = 4.0f;
    scene->AddLight(dir_light_desc);

    // Add camera
    const float view_origin[] = {0.0f, 0.0f, 0.0f};
    const float view_dir[] = {0.0f, 0.0f, -1.0f};

    Ray::camera_desc_t cam_desc;
    cam_desc.type = Ray::eCamType::Persp;
    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
    cam_desc.fov = 45.0f;
    cam_desc.exposure = -1.0f;

    const Ray::CameraHandle cam = scene->AddCamera(cam_desc);
    scene->set_current_cam(cam);

    scene->Finalize();

    // Create region contex for frame, setup to use whole frame
    auto region = Ray::RegionContext{{0, 0, IMG_W, IMG_H}};

    // Render image
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        // Each call performs one iteration, blocks until finished
        renderer->RenderScene(*scene, region);
        printf("Renderered %i samples\n", i);
    }
    printf("Done\n");

    // Get rendered image pixels in 32-bit floating point RGBA format
    const Ray::color_data_rgba_t pixels = renderer->get_pixels_ref();

    // Save image
    WriteTGA(pixels.ptr, pixels.pitch, IMG_W, IMG_H, 3, "05_physical_sky.tga");
    printf("Image saved as samples/05_physical_sky.tga\n");

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
