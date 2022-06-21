#include "test_common.h"

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <fstream>

#include "../RendererFactory.h"

#include "thread_pool.h"
#include "utils.h"

template <typename MatDesc> void load_needed_textures(Ray::SceneBase &scene, MatDesc &mat_desc, const char *textures[]);

template <>
void load_needed_textures(Ray::SceneBase &scene, Ray::shading_node_desc_t &mat_desc, const char *textures[]) {
    if (!textures) {
        return;
    }

    if (mat_desc.base_texture != 0xffffffff && textures[0]) {
        int img_w, img_h;
        const auto img_data = LoadTGA(textures[0], img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data.data();
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = true;

        mat_desc.base_texture = scene.AddTexture(tex_desc);
    }
}

template <>
void load_needed_textures(Ray::SceneBase &scene, Ray::principled_mat_desc_t &mat_desc, const char *textures[]) {
    if (!textures) {
        return;
    }

    if (mat_desc.base_texture != 0xffffffff && textures[mat_desc.base_texture]) {
        int img_w, img_h;
        const auto img_data = LoadTGA(textures[mat_desc.base_texture], img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data.data();
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = true;

        mat_desc.base_texture = scene.AddTexture(tex_desc);
    }

    if (mat_desc.normal_map != 0xffffffff && textures[mat_desc.normal_map]) {
        int img_w, img_h;
        const auto img_data = LoadTGA(textures[mat_desc.normal_map], img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data.data();
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = false;
        tex_desc.is_srgb = false;

        mat_desc.normal_map = scene.AddTexture(tex_desc);
    }

    if (mat_desc.roughness_texture != 0xffffffff && textures[mat_desc.roughness_texture]) {
        int img_w, img_h;
        const auto img_data = LoadTGA(textures[mat_desc.roughness_texture], img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data.data();
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = false;

        mat_desc.roughness_texture = scene.AddTexture(tex_desc);
    }

    if (mat_desc.metallic_texture != 0xffffffff && textures[mat_desc.metallic_texture]) {
        int img_w, img_h;
        const auto img_data = LoadTGA(textures[mat_desc.metallic_texture], img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data.data();
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = false;

        mat_desc.metallic_texture = scene.AddTexture(tex_desc);
    }
}

namespace {
const int STANDARD_SCENE = 0;
const int STANDARD_SCENE_MESH_LIGHTS = 1;
const int STANDARD_SCENE_NO_LIGHTS = 2;
const int REFR_PLANE_SCENE = 3;
} // namespace

template <typename MatDesc>
void setup_material_scene(Ray::SceneBase &scene, const bool output_sh, const MatDesc &main_mat_desc,
                          const char *textures[], const int scene_index) {
    { // setup camera
        const float view_origin_standard[] = {0.16149f, 0.294997f, 0.332965f};
        const float view_dir_standard[] = {-0.364128768f, -0.555621922f, -0.747458696f};
        const float view_origin_refr[] = {-0.074711f, 0.099348f, -0.049506f};
        const float view_dir_refr[] = {0.725718915f, 0.492017448f, 0.480885535f};
        const float view_up[] = {0.0f, 1.0f, 0.0f};

        Ray::camera_desc_t cam_desc;
        cam_desc.type = Ray::Persp;
        cam_desc.filter = Ray::Box;
        cam_desc.dtype = Ray::SRGB;
        if (scene_index == STANDARD_SCENE || scene_index == STANDARD_SCENE_MESH_LIGHTS ||
            scene_index == STANDARD_SCENE_NO_LIGHTS) {
            memcpy(&cam_desc.origin[0], &view_origin_standard[0], 3 * sizeof(float));
            memcpy(&cam_desc.fwd[0], &view_dir_standard[0], 3 * sizeof(float));
            cam_desc.fov = 18.1806f;
        } else if (scene_index == REFR_PLANE_SCENE) {
            memcpy(&cam_desc.origin[0], &view_origin_refr[0], 3 * sizeof(float));
            memcpy(&cam_desc.fwd[0], &view_dir_refr[0], 3 * sizeof(float));
            cam_desc.fov = 45.1806f;
        }
        memcpy(&cam_desc.up[0], &view_up[0], 3 * sizeof(float));
        cam_desc.clamp = true;
        cam_desc.output_sh = output_sh;

        const uint32_t cam = scene.AddCamera(cam_desc);
        scene.set_current_cam(cam);
    }

    { // setup environment
        Ray::environment_desc_t env_desc;
        env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;
        scene.SetEnvironment(env_desc);
    }

    MatDesc main_mat_desc_copy = main_mat_desc;
    load_needed_textures(scene, main_mat_desc_copy, textures);
    const uint32_t main_mat = scene.AddMaterial(main_mat_desc_copy);

    uint32_t floor_mat;
    {
        Ray::principled_mat_desc_t floor_mat_desc;
        floor_mat_desc.base_color[0] = 0.75f;
        floor_mat_desc.base_color[1] = 0.75f;
        floor_mat_desc.base_color[2] = 0.75f;
        floor_mat_desc.specular = 0.0f;
        floor_mat = scene.AddMaterial(floor_mat_desc);
    }

    uint32_t walls_mat;
    {
        Ray::principled_mat_desc_t walls_mat_desc;
        walls_mat_desc.base_color[0] = 0.5f;
        walls_mat_desc.base_color[1] = 0.5f;
        walls_mat_desc.base_color[2] = 0.5f;
        walls_mat_desc.specular = 0.0f;
        walls_mat = scene.AddMaterial(walls_mat_desc);
    }

    uint32_t white_mat;
    {
        Ray::principled_mat_desc_t white_mat_desc;
        white_mat_desc.base_color[0] = 0.64f;
        white_mat_desc.base_color[1] = 0.64f;
        white_mat_desc.base_color[2] = 0.64f;
        white_mat_desc.specular = 0.0f;
        white_mat = scene.AddMaterial(white_mat_desc);
    }

    uint32_t light_grey_mat;
    {
        Ray::principled_mat_desc_t light_grey_mat_desc;
        light_grey_mat_desc.base_color[0] = 0.32f;
        light_grey_mat_desc.base_color[1] = 0.32f;
        light_grey_mat_desc.base_color[2] = 0.32f;
        light_grey_mat_desc.specular = 0.0f;
        light_grey_mat = scene.AddMaterial(light_grey_mat_desc);
    }

    uint32_t mid_grey_mat;
    {
        Ray::principled_mat_desc_t mid_grey_mat_desc;
        mid_grey_mat_desc.base_color[0] = 0.16f;
        mid_grey_mat_desc.base_color[1] = 0.16f;
        mid_grey_mat_desc.base_color[2] = 0.16f;
        mid_grey_mat_desc.specular = 0.0f;
        mid_grey_mat = scene.AddMaterial(mid_grey_mat_desc);
    }

    uint32_t dark_grey_mat;
    {
        Ray::principled_mat_desc_t dark_grey_mat_desc;
        dark_grey_mat_desc.base_color[0] = 0.08f;
        dark_grey_mat_desc.base_color[1] = 0.08f;
        dark_grey_mat_desc.base_color[2] = 0.08f;
        dark_grey_mat_desc.specular = 0.0f;
        dark_grey_mat = scene.AddMaterial(dark_grey_mat_desc);
    }

    uint32_t square_light_mat;
    {
        Ray::shading_node_desc_t square_light_mat_desc;
        square_light_mat_desc.type = Ray::EmissiveNode;
        square_light_mat_desc.strength = 20.3718f;
        square_light_mat_desc.multiple_importance = true;
        square_light_mat_desc.base_color[0] = 1.0f;
        square_light_mat_desc.base_color[1] = 1.0f;
        square_light_mat_desc.base_color[2] = 1.0f;
        square_light_mat = scene.AddMaterial(square_light_mat_desc);
    }

    uint32_t disc_light_mat;
    {
        Ray::shading_node_desc_t disc_light_mat_desc;
        disc_light_mat_desc.type = Ray::EmissiveNode;
        disc_light_mat_desc.strength = 81.4873f;
        disc_light_mat_desc.multiple_importance = true;
        disc_light_mat_desc.base_color[0] = 1.0f;
        disc_light_mat_desc.base_color[1] = 1.0f;
        disc_light_mat_desc.base_color[2] = 1.0f;
        disc_light_mat = scene.AddMaterial(disc_light_mat_desc);
    }

    uint32_t base_mesh;
    {
        std::vector<float> base_attrs;
        std::vector<uint32_t> base_indices, base_groups;
        std::tie(base_attrs, base_indices, base_groups) = LoadBIN("test_data/meshes/mat_test/base.bin");

        Ray::mesh_desc_t base_mesh_desc;
        base_mesh_desc.prim_type = Ray::TriangleList;
        base_mesh_desc.layout = Ray::PxyzNxyzTuv;
        base_mesh_desc.vtx_attrs = &base_attrs[0];
        base_mesh_desc.vtx_attrs_count = uint32_t(base_attrs.size()) / 8;
        base_mesh_desc.vtx_indices = &base_indices[0];
        base_mesh_desc.vtx_indices_count = uint32_t(base_indices.size());
        base_mesh_desc.shapes.push_back({mid_grey_mat, mid_grey_mat, base_groups[0], base_groups[1]});
        base_mesh = scene.AddMesh(base_mesh_desc);
    }

    uint32_t model_mesh;
    {
        std::vector<float> model_attrs;
        std::vector<uint32_t> model_indices, model_groups;
        if (scene_index == STANDARD_SCENE || scene_index == STANDARD_SCENE_MESH_LIGHTS ||
            scene_index == STANDARD_SCENE_NO_LIGHTS) {
            std::tie(model_attrs, model_indices, model_groups) = LoadBIN("test_data/meshes/mat_test/model.bin");
        } else if (scene_index == REFR_PLANE_SCENE) {
            std::tie(model_attrs, model_indices, model_groups) = LoadBIN("test_data/meshes/mat_test/refr_plane.bin");
        }

        Ray::mesh_desc_t model_mesh_desc;
        model_mesh_desc.prim_type = Ray::TriangleList;
        model_mesh_desc.layout = Ray::PxyzNxyzTuv;
        model_mesh_desc.vtx_attrs = &model_attrs[0];
        model_mesh_desc.vtx_attrs_count = uint32_t(model_attrs.size()) / 8;
        model_mesh_desc.vtx_indices = &model_indices[0];
        model_mesh_desc.vtx_indices_count = uint32_t(model_indices.size());
        model_mesh_desc.shapes.push_back({main_mat, main_mat, model_groups[0], model_groups[1]});
        model_mesh = scene.AddMesh(model_mesh_desc);
    }

    uint32_t core_mesh;
    {
        std::vector<float> core_attrs;
        std::vector<uint32_t> core_indices, core_groups;
        std::tie(core_attrs, core_indices, core_groups) = LoadBIN("test_data/meshes/mat_test/core.bin");

        Ray::mesh_desc_t core_mesh_desc;
        core_mesh_desc.prim_type = Ray::TriangleList;
        core_mesh_desc.layout = Ray::PxyzNxyzTuv;
        core_mesh_desc.vtx_attrs = &core_attrs[0];
        core_mesh_desc.vtx_attrs_count = uint32_t(core_attrs.size()) / 8;
        core_mesh_desc.vtx_indices = &core_indices[0];
        core_mesh_desc.vtx_indices_count = uint32_t(core_indices.size());
        core_mesh_desc.shapes.push_back({mid_grey_mat, mid_grey_mat, core_groups[0], core_groups[1]});
        core_mesh = scene.AddMesh(core_mesh_desc);
    }

    uint32_t subsurf_bar_mesh;
    {
        std::vector<float> subsurf_bar_attrs;
        std::vector<uint32_t> subsurf_bar_indices, subsurf_bar_groups;
        std::tie(subsurf_bar_attrs, subsurf_bar_indices, subsurf_bar_groups) =
            LoadBIN("test_data/meshes/mat_test/subsurf_bar.bin");

        Ray::mesh_desc_t subsurf_bar_mesh_desc;
        subsurf_bar_mesh_desc.prim_type = Ray::TriangleList;
        subsurf_bar_mesh_desc.layout = Ray::PxyzNxyzTuv;
        subsurf_bar_mesh_desc.vtx_attrs = &subsurf_bar_attrs[0];
        subsurf_bar_mesh_desc.vtx_attrs_count = uint32_t(subsurf_bar_attrs.size()) / 8;
        subsurf_bar_mesh_desc.vtx_indices = &subsurf_bar_indices[0];
        subsurf_bar_mesh_desc.vtx_indices_count = uint32_t(subsurf_bar_indices.size());
        subsurf_bar_mesh_desc.shapes.push_back({white_mat, white_mat, subsurf_bar_groups[0], subsurf_bar_groups[1]});
        subsurf_bar_mesh_desc.shapes.push_back(
            {dark_grey_mat, dark_grey_mat, subsurf_bar_groups[2], subsurf_bar_groups[3]});
        subsurf_bar_mesh = scene.AddMesh(subsurf_bar_mesh_desc);
    }

    uint32_t text_mesh;
    {
        std::vector<float> text_attrs;
        std::vector<uint32_t> text_indices, text_groups;
        std::tie(text_attrs, text_indices, text_groups) = LoadBIN("test_data/meshes/mat_test/text.bin");

        Ray::mesh_desc_t text_mesh_desc;
        text_mesh_desc.prim_type = Ray::TriangleList;
        text_mesh_desc.layout = Ray::PxyzNxyzTuv;
        text_mesh_desc.vtx_attrs = &text_attrs[0];
        text_mesh_desc.vtx_attrs_count = uint32_t(text_attrs.size()) / 8;
        text_mesh_desc.vtx_indices = &text_indices[0];
        text_mesh_desc.vtx_indices_count = uint32_t(text_indices.size());
        text_mesh_desc.shapes.push_back({white_mat, white_mat, text_groups[0], text_groups[1]});
        text_mesh = scene.AddMesh(text_mesh_desc);
    }

    uint32_t env_mesh;
    {
        std::vector<float> env_attrs;
        std::vector<uint32_t> env_indices, env_groups;
        std::tie(env_attrs, env_indices, env_groups) = LoadBIN("test_data/meshes/mat_test/env.bin");

        Ray::mesh_desc_t env_mesh_desc;
        env_mesh_desc.prim_type = Ray::TriangleList;
        env_mesh_desc.layout = Ray::PxyzNxyzTuv;
        env_mesh_desc.vtx_attrs = &env_attrs[0];
        env_mesh_desc.vtx_attrs_count = uint32_t(env_attrs.size()) / 8;
        env_mesh_desc.vtx_indices = &env_indices[0];
        env_mesh_desc.vtx_indices_count = uint32_t(env_indices.size());
        env_mesh_desc.shapes.push_back({floor_mat, floor_mat, env_groups[0], env_groups[1]});
        env_mesh_desc.shapes.push_back({walls_mat, walls_mat, env_groups[2], env_groups[3]});
        env_mesh_desc.shapes.push_back({dark_grey_mat, dark_grey_mat, env_groups[4], env_groups[5]});
        env_mesh_desc.shapes.push_back({light_grey_mat, light_grey_mat, env_groups[6], env_groups[7]});
        env_mesh_desc.shapes.push_back({mid_grey_mat, mid_grey_mat, env_groups[8], env_groups[9]});
        env_mesh = scene.AddMesh(env_mesh_desc);
    }

    uint32_t square_light_mesh;
    {
        std::vector<float> square_light_attrs;
        std::vector<uint32_t> square_light_indices, square_light_groups;
        std::tie(square_light_attrs, square_light_indices, square_light_groups) =
            LoadBIN("test_data/meshes/mat_test/square_light.bin");

        Ray::mesh_desc_t square_light_mesh_desc;
        square_light_mesh_desc.prim_type = Ray::TriangleList;
        square_light_mesh_desc.layout = Ray::PxyzNxyzTuv;
        square_light_mesh_desc.vtx_attrs = &square_light_attrs[0];
        square_light_mesh_desc.vtx_attrs_count = uint32_t(square_light_attrs.size()) / 8;
        square_light_mesh_desc.vtx_indices = &square_light_indices[0];
        square_light_mesh_desc.vtx_indices_count = uint32_t(square_light_indices.size());
        square_light_mesh_desc.shapes.push_back(
            {square_light_mat, square_light_mat, square_light_groups[0], square_light_groups[1]});
        square_light_mesh_desc.shapes.push_back(
            {dark_grey_mat, dark_grey_mat, square_light_groups[2], square_light_groups[3]});
        square_light_mesh = scene.AddMesh(square_light_mesh_desc);
    }

    uint32_t disc_light_mesh;
    {
        std::vector<float> disc_light_attrs;
        std::vector<uint32_t> disc_light_indices, disc_light_groups;
        std::tie(disc_light_attrs, disc_light_indices, disc_light_groups) =
            LoadBIN("test_data/meshes/mat_test/disc_light.bin");

        Ray::mesh_desc_t disc_light_mesh_desc;
        disc_light_mesh_desc.prim_type = Ray::TriangleList;
        disc_light_mesh_desc.layout = Ray::PxyzNxyzTuv;
        disc_light_mesh_desc.vtx_attrs = &disc_light_attrs[0];
        disc_light_mesh_desc.vtx_attrs_count = uint32_t(disc_light_attrs.size()) / 8;
        disc_light_mesh_desc.vtx_indices = &disc_light_indices[0];
        disc_light_mesh_desc.vtx_indices_count = uint32_t(disc_light_indices.size());
        disc_light_mesh_desc.shapes.push_back(
            {disc_light_mat, disc_light_mat, disc_light_groups[0], disc_light_groups[1]});
        disc_light_mesh_desc.shapes.push_back(
            {dark_grey_mat, dark_grey_mat, disc_light_groups[2], disc_light_groups[3]});
        disc_light_mesh = scene.AddMesh(disc_light_mesh_desc);
    }

    static const float identity[16] = {1.0f, 0.0f, 0.0f, 0.0f, // NOLINT
                                       0.0f, 1.0f, 0.0f, 0.0f, // NOLINT
                                       0.0f, 0.0f, 1.0f, 0.0f, // NOLINT
                                       0.0f, 0.0f, 0.0f, 1.0f};

    static const float model_xform[16] = {0.707106769f,  0.0f,   0.707106769f, 0.0f, // NOLINT
                                          0.0f,          1.0f,   0.0f,         0.0f, // NOLINT
                                          -0.707106769f, 0.0f,   0.707106769f, 0.0f, // NOLINT
                                          0.0f,          0.062f, 0.0f,         1.0f};

    if (scene_index == STANDARD_SCENE || scene_index == STANDARD_SCENE_MESH_LIGHTS ||
        scene_index == STANDARD_SCENE_NO_LIGHTS) {
        scene.AddMeshInstance(model_mesh, model_xform);
        scene.AddMeshInstance(base_mesh, identity);
        scene.AddMeshInstance(core_mesh, identity);
        scene.AddMeshInstance(subsurf_bar_mesh, identity);
        scene.AddMeshInstance(text_mesh, identity);
    } else if (scene_index == REFR_PLANE_SCENE) {
        scene.AddMeshInstance(model_mesh, identity);
    }
    scene.AddMeshInstance(env_mesh, identity);
    if (scene_index == STANDARD_SCENE_MESH_LIGHTS || scene_index == REFR_PLANE_SCENE) {
        //
        // Use mesh lights
        //
        if (scene_index != REFR_PLANE_SCENE) {
            scene.AddMeshInstance(square_light_mesh, identity);
        }
        scene.AddMeshInstance(disc_light_mesh, identity);
    } else if (scene_index == STANDARD_SCENE) {
        //
        // Use explicit lights sources
        //
        { // rect light
            static const float xform[16] = {-0.425036609f, 2.24262476e-06f, -0.905176163f, 0.00000000f,
                                            -0.876228273f, 0.250873595f,    0.411444396f,  0.00000000f,
                                            0.227085724f,  0.968019843f,    -0.106628500f, 0.00000000f,
                                            -0.436484009f, 0.187178999f,    0.204932004f,  1.00000000f};

            Ray::rect_light_desc_t new_light;

            new_light.color[0] = 20.3718f;
            new_light.color[1] = 20.3718f;
            new_light.color[2] = 20.3718f;

            new_light.width = 0.162f;
            new_light.height = 0.162f;

            new_light.visible = true;
            new_light.sky_portal = false;

            scene.AddLight(new_light, xform);
        }
        { // disk light
            static const float xform[16] = {0.813511789f,  -0.536388099f, -0.224691749f, 0.00000000f,
                                            0.538244009f,  0.548162937f,  0.640164733f,  0.00000000f,
                                            -0.220209062f, -0.641720533f, 0.734644651f,  0.00000000f,
                                            0.360500991f,  0.461762011f,  0.431780994f,  1.00000000f};

            Ray::disk_light_desc_t new_light;

            new_light.color[0] = 81.4873f;
            new_light.color[1] = 81.4873f;
            new_light.color[2] = 81.4873f;

            new_light.size_x = 0.1296f;
            new_light.size_y = 0.1296f;

            new_light.visible = true;
            new_light.sky_portal = false;

            scene.AddLight(new_light, xform);
        }
    } else if (scene_index == STANDARD_SCENE_NO_LIGHTS) {
        // nothing
    }

    scene.Finalize();
}

void schedule_render_jobs(Ray::RendererBase &renderer, const Ray::SceneBase *scene, const Ray::settings_t &settings,
                          const bool output_sh, const int samples, const char *log_str) {
    const auto rt = renderer.type();
    const auto sz = renderer.size();

    if (rt == Ray::RendererRef /*|| rt == Ray::RendererSSE2 || rt == Ray::RendererAVX || rt == Ray::RendererAVX2*/) {
        const int BucketSize = 16;

        std::vector<Ray::RegionContext> region_contexts;
        for (int y = 0; y < sz.second; y += BucketSize) {
            for (int x = 0; x < sz.first; x += BucketSize) {
                const auto rect =
                    Ray::rect_t{x, y, std::min(sz.first - x, BucketSize), std::min(sz.second - y, BucketSize)};
                region_contexts.emplace_back(rect);
            }
        }

        ThreadPool threads(std::thread::hardware_concurrency());

        auto render_job = [&](int j, int portion) {
            for (int i = 0; i < portion; ++i) {
                renderer.RenderScene(scene, region_contexts[j]);
            }
        };

        const int SamplePortion = 16;
        for (int i = 0; i < samples; i += std::min(SamplePortion, samples - i)) {
            std::vector<std::future<void>> job_res;
            for (int j = 0; j < int(region_contexts.size()); ++j) {
                job_res.push_back(threads.Enqueue(render_job, j, std::min(SamplePortion, samples - i)));
            }
            for (auto &res : job_res) {
                res.wait();
            }

            // report progress percentage
            const float prog = 100.0f * float(i + std::min(SamplePortion, samples - i)) / samples;
            printf("\r%s (%s, %c, %s): %.1f%% ", log_str, Ray::RendererTypeName(rt), settings.use_wide_bvh ? 'w' : 'n',
                   output_sh ? "sh" : "co", prog);
            fflush(stdout);
        }
    } else {
        for (int i = 0; i < samples; ++i) {
            auto region = Ray::RegionContext{{0, 0, sz.first, sz.second}};
            renderer.RenderScene(scene, region);
        }
    }
}

template <typename MatDesc>
void run_material_test(const char *test_name, const MatDesc &mat_desc, const int sample_count, const double min_psnr,
                       const int pix_thres, const char *textures[] = nullptr, const int scene_index = 0) {
    char name_buf[1024];
    snprintf(name_buf, sizeof(name_buf), "test_data/%s/ref.tga", test_name);

    int test_img_w, test_img_h;
    const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
    require_skip(!test_img.empty());

    {
        Ray::settings_t s;
        s.w = test_img_w;
        s.h = test_img_h;

        static const Ray::eRendererType renderer_types[] = {
            Ray::RendererRef, /*Ray::RendererSSE2, Ray::RendererAVX, Ray::RendererAVX2,
#if defined(__ANDROID__)
Ray::RendererNEON,
#elif !defined(DISABLE_OCL)
Ray::RendererOCL
#endif*/
        };

        const int DiffThres = 32;

        for (const bool use_wide_bvh : {true}) {
            s.use_wide_bvh = use_wide_bvh;
            for (const bool output_sh : {false}) {
                for (const Ray::eRendererType rt : renderer_types) {
                    auto renderer = std::unique_ptr<Ray::RendererBase>(Ray::CreateRenderer(s, &Ray::g_null_log, rt));
                    auto scene = std::unique_ptr<Ray::SceneBase>(renderer->CreateScene());

                    setup_material_scene(*scene, output_sh, mat_desc, textures, scene_index);

                    snprintf(name_buf, sizeof(name_buf), "Test %s", test_name);
                    schedule_render_jobs(*renderer, scene.get(), s, output_sh, sample_count, name_buf);

                    const Ray::pixel_color_t *pixels = renderer->get_pixels_ref();

                    std::unique_ptr<uint8_t[]> img_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                    std::unique_ptr<uint8_t[]> diff_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                    std::unique_ptr<uint8_t[]> mask_data_u8(new uint8_t[test_img_w * test_img_h * 3]);
                    memset(&mask_data_u8[0], 0, test_img_w * test_img_h * 3);

                    double mse = 0.0f;

                    int error_pixels = 0;
                    for (int j = 0; j < test_img_h; j++) {
                        for (int i = 0; i < test_img_w; i++) {
                            const Ray::pixel_color_t &p = pixels[j * test_img_w + i];

                            const uint8_t r = uint8_t(p.r * 255);
                            const uint8_t g = uint8_t(p.g * 255);
                            const uint8_t b = uint8_t(p.b * 255);

                            img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 0] = r;
                            img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 1] = g;
                            img_data_u8[3 * ((test_img_h - j - 1) * test_img_w + i) + 2] = b;

                            const uint8_t diff_r =
                                std::abs(r - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 0]);
                            const uint8_t diff_g =
                                std::abs(g - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 1]);
                            const uint8_t diff_b =
                                std::abs(b - test_img[4 * ((test_img_h - j - 1) * test_img_w + i) + 2]);

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

                    double psnr = -10.0 * std::log10(mse / (255.0f * 255.0f));
                    psnr = std::floor(psnr * 100.0) / 100.0;

                    printf("(PSNR: %.3f/%.3f dB, Fireflies: %i/%i)\n", psnr, min_psnr, error_pixels, pix_thres);

                    snprintf(name_buf, sizeof(name_buf), "test_data/%s/out.tga", test_name);
                    WriteTGA(&img_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                    snprintf(name_buf, sizeof(name_buf), "test_data/%s/diff.tga", test_name);
                    WriteTGA(&diff_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                    snprintf(name_buf, sizeof(name_buf), "test_data/%s/mask.tga", test_name);
                    WriteTGA(&mask_data_u8[0], test_img_w, test_img_h, 3, name_buf);
                    require((psnr >= min_psnr) && (error_pixels <= pix_thres) && "Images do not match!");
                }
            }
        }
    }
}

void assemble_material_test_images() {
    const int ImgCountW = 5;
    const char *test_names[][ImgCountW] = {
        {"oren_mat0", "oren_mat1", "oren_mat2", "oren_mat3", "oren_mat4"},
        {"diff_mat0", "diff_mat1", "diff_mat2", "diff_mat3", "diff_mat4"},
        {"sheen_mat0", "sheen_mat1", "sheen_mat2", "sheen_mat3"},
        {"spec_mat0", "spec_mat1", "spec_mat2", "spec_mat3", "spec_mat4"},
        {"aniso_mat0", "aniso_mat1", "aniso_mat2", "aniso_mat3", "aniso_mat4"},
        {"aniso_mat5", "aniso_mat6", "aniso_mat7"},
        {"metal_mat0", "metal_mat1", "metal_mat2", "metal_mat3", "metal_mat4"},
        {"plastic_mat0", "plastic_mat1", "plastic_mat2", "plastic_mat3", "plastic_mat4"},
        {"tint_mat0", "tint_mat1", "tint_mat2", "tint_mat3", "tint_mat4"},
        {"emit_mat0", "emit_mat1"},
        {"coat_mat0", "coat_mat1", "coat_mat2", "coat_mat3", "coat_mat4"},
        {"refr_mis0", "refr_mis1", "refr_mis2", "refr_mis3", "refr_mis4"},
        {"refr_mat0", "refr_mat1", "refr_mat2", "refr_mat3", "refr_mat4"},
        {"refr_mat5"},
        {"trans_mat0", "trans_mat1", "trans_mat2", "trans_mat3", "trans_mat4"},
        {"trans_mat5", "trans_mat6", "trans_mat7", "trans_mat8", "trans_mat9"},
        {"alpha_mat0", "alpha_mat1", "alpha_mat2", "alpha_mat3"},
        {"complex_mat0", "complex_mat1", "complex_mat2", "complex_mat3", "complex_mat4"},
        {"complex_mat5", "complex_mat6"}};
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
                snprintf(name_buf, sizeof(name_buf), "test_data/%s/out.tga", test_names[j][i]);

                int test_img_w, test_img_h;
                const auto test_img = LoadTGA(name_buf, test_img_w, test_img_h);
                if (!test_img.empty()) {
                    for (int k = 0; k < test_img_h; ++k) {
                        memcpy(&material_imgs[((top_down_j * 256 + k) * OutImageW + i * 256) * 4],
                               &test_img[k * test_img_w * 4], test_img_w * 4);
                    }
                }

                blit_chars_to_alpha(material_imgs.get(), i * 256, top_down_j * 256, test_names[j][i]);
            }

            { // error mask
                snprintf(name_buf, sizeof(name_buf), "test_data/%s/mask.tga", test_names[j][i]);

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

    WriteTGA(material_refs.get(), OutImageW, OutImageH, 4, "test_data/material_refs.tga");
    WriteTGA(material_imgs.get(), OutImageW, OutImageH, 4, "test_data/material_imgs.tga");
    WriteTGA(material_masks.get(), OutImageW, OutImageH, 4, "test_data/material_masks.tga");
}

//
// Oren-nayar material tests
//

void test_oren_mat0() {
    const int SampleCount = 256;
    const double MinPSNR = 41.29;
    const int PixThres = 0;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::DiffuseNode;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;

    run_material_test("oren_mat0", desc, SampleCount, MinPSNR, PixThres);
}

void test_oren_mat1() {
    const int SampleCount = 256;
    const double MinPSNR = 41.39;
    const int PixThres = 0;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::DiffuseNode;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.25f;

    run_material_test("oren_mat1", desc, SampleCount, MinPSNR, PixThres);
}

void test_oren_mat2() {
    const int SampleCount = 256;
    const double MinPSNR = 40.54;
    const int PixThres = 0;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::DiffuseNode;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;

    run_material_test("oren_mat2", desc, SampleCount, MinPSNR, PixThres);
}

void test_oren_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 40.57;
    const int PixThres = 0;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::DiffuseNode;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.75f;

    run_material_test("oren_mat3", desc, SampleCount, MinPSNR, PixThres);
}

void test_oren_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 41.45;
    const int PixThres = 0;

    Ray::shading_node_desc_t desc;
    desc.type = Ray::DiffuseNode;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;

    run_material_test("oren_mat4", desc, SampleCount, MinPSNR, PixThres);
}

//
// Diffuse material tests
//

void test_diff_mat0() {
    const int SampleCount = 256;
    const double MinPSNR = 41.52;
    const int PixThres = 0;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.0f;
    desc.specular = 0.0f;

    run_material_test("diff_mat0", desc, SampleCount, MinPSNR, PixThres);
}

void test_diff_mat1() {
    const int SampleCount = 256;
    const double MinPSNR = 41.5;
    const int PixThres = 0;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.25f;
    desc.specular = 0.0f;

    run_material_test("diff_mat1", desc, SampleCount, MinPSNR, PixThres);
}

void test_diff_mat2() {
    const int SampleCount = 256;
    const double MinPSNR = 40.61;
    const int PixThres = 0;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 0.5f;
    desc.specular = 0.0f;

    run_material_test("diff_mat2", desc, SampleCount, MinPSNR, PixThres);
}

void test_diff_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 40.56;
    const int PixThres = 0;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.5f;
    desc.base_color[1] = 0.5f;
    desc.base_color[2] = 0.0f;
    desc.roughness = 0.75f;
    desc.specular = 0.0f;

    run_material_test("diff_mat3", desc, SampleCount, MinPSNR, PixThres);
}

void test_diff_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 41.41;
    const int PixThres = 0;

    Ray::principled_mat_desc_t desc;
    desc.base_color[0] = 0.0f;
    desc.base_color[1] = 0.0f;
    desc.base_color[2] = 0.5f;
    desc.roughness = 1.0f;
    desc.specular = 0.0f;

    run_material_test("diff_mat4", desc, SampleCount, MinPSNR, PixThres);
}

//
// Sheen material tests
//

void test_sheen_mat0() {
    const int SampleCount = 256;
    const double MinPSNR = 41.83;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 0.5f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test("sheen_mat0", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_sheen_mat1() {
    const int SampleCount = 256;
    const double MinPSNR = 41.33;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test("sheen_mat1", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_sheen_mat2() {
    const int SampleCount = 256;
    const double MinPSNR = 41.48;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 0.0f;

    run_material_test("sheen_mat2", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_sheen_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 41.34;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.1f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.1f;
    mat_desc.specular = 0.0f;
    mat_desc.sheen = 1.0f;
    mat_desc.sheen_tint = 1.0f;

    run_material_test("sheen_mat3", mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Glossy material tests
//

void test_glossy_mat0() {
    const int SampleCount = 2048;
    const double MinPSNR = 35.06;
    const int PixThres = 54;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::GlossyNode;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.0f;

    run_material_test("glossy_mat0", node_desc, SampleCount, MinPSNR, PixThres);
}

void test_glossy_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.63;
    const int PixThres = 55;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::GlossyNode;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.25f;

    run_material_test("glossy_mat1", node_desc, SampleCount, MinPSNR, PixThres);
}

void test_glossy_mat2() {
    const int SampleCount = 512;
    const double MinPSNR = 38.05;
    const int PixThres = 0;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::GlossyNode;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.5f;

    run_material_test("glossy_mat2", node_desc, SampleCount, MinPSNR, PixThres);
}

void test_glossy_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 39.35;
    const int PixThres = 0;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::GlossyNode;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 0.75f;

    run_material_test("glossy_mat3", node_desc, SampleCount, MinPSNR, PixThres);
}

void test_glossy_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 40.35;
    const int PixThres = 0;

    Ray::shading_node_desc_t node_desc;
    node_desc.type = Ray::GlossyNode;
    node_desc.base_color[0] = 1.0f;
    node_desc.base_color[1] = 1.0f;
    node_desc.base_color[2] = 1.0f;
    node_desc.roughness = 1.0f;

    run_material_test("glossy_mat4", node_desc, SampleCount, MinPSNR, PixThres);
}

//
// Specular material tests
//

void test_spec_mat0() {
    const int SampleCount = 2048;
    const double MinPSNR = 35.06;
    const int PixThres = 54;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test("spec_mat0", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_spec_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.63;
    const int PixThres = 55;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test("spec_mat1", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_spec_mat2() {
    const int SampleCount = 512;
    const double MinPSNR = 38.05;
    const int PixThres = 0;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.5f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test("spec_mat2", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_spec_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 39.35;
    const int PixThres = 0;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.75f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test("spec_mat3", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_spec_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 40.35;
    const int PixThres = 0;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 1.0f;
    spec_mat_desc.metallic = 1.0f;

    run_material_test("spec_mat4", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Anisotropic material tests
//

void test_aniso_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.6;
    const int PixThres = 70;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.25f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test("aniso_mat0", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.59;
    const int PixThres = 56;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.5f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test("aniso_mat1", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.56;
    const int PixThres = 54;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 0.75f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test("aniso_mat2", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat3() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.39;
    const int PixThres = 75;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.0f;

    run_material_test("aniso_mat3", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat4() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.33;
    const int PixThres = 73;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.125f;

    run_material_test("aniso_mat4", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat5() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.85;
    const int PixThres = 67;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.25f;

    run_material_test("aniso_mat5", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat6() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.5;
    const int PixThres = 85;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.375f;

    run_material_test("aniso_mat6", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_aniso_mat7() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.39;
    const int PixThres = 72;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 1.0f;
    spec_mat_desc.base_color[1] = 1.0f;
    spec_mat_desc.base_color[2] = 1.0f;
    spec_mat_desc.roughness = 0.25f;
    spec_mat_desc.metallic = 1.0f;
    spec_mat_desc.anisotropic = 1.0f;
    spec_mat_desc.anisotropic_rotation = 0.5f;

    run_material_test("aniso_mat7", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Plastic material tests
//

void test_plastic_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 39.75;
    const int PixThres = 18;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.0f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 0.0f;

    run_material_test("plastic_mat0", plastic_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_plastic_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 41.29;
    const int PixThres = 1;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.5f;
    plastic_mat_desc.base_color[1] = 0.0f;
    plastic_mat_desc.base_color[2] = 0.0f;
    plastic_mat_desc.roughness = 0.25f;

    run_material_test("plastic_mat1", plastic_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_plastic_mat2() {
    const int SampleCount = 512;
    const double MinPSNR = 41.39;
    const int PixThres = 0;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.0f;
    plastic_mat_desc.roughness = 0.5f;

    run_material_test("plastic_mat2", plastic_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_plastic_mat3() {
    const int SampleCount = 512;
    const double MinPSNR = 40.99;
    const int PixThres = 0;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.5f;
    plastic_mat_desc.base_color[1] = 0.0f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 0.75f;

    run_material_test("plastic_mat3", plastic_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_plastic_mat4() {
    const int SampleCount = 512;
    const double MinPSNR = 41.16;
    const int PixThres = 0;

    Ray::principled_mat_desc_t plastic_mat_desc;
    plastic_mat_desc.base_color[0] = 0.0f;
    plastic_mat_desc.base_color[1] = 0.5f;
    plastic_mat_desc.base_color[2] = 0.5f;
    plastic_mat_desc.roughness = 1.0f;

    run_material_test("plastic_mat4", plastic_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Tint material tests
//

void test_tint_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 39.38;
    const int PixThres = 20;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.0f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.0f;

    run_material_test("tint_mat0", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_tint_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 40.82;
    const int PixThres = 0;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.0f;
    spec_mat_desc.base_color[1] = 0.5f;
    spec_mat_desc.base_color[2] = 0.0f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.25f;

    run_material_test("tint_mat1", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_tint_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 34.56;
    const int PixThres = 187;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.0f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.5f;

    run_material_test("tint_mat2", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_tint_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 39.66;
    const int PixThres = 0;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.5f;
    spec_mat_desc.base_color[2] = 0.0f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 0.75f;

    run_material_test("tint_mat3", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_tint_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 40.03;
    const int PixThres = 4;

    Ray::principled_mat_desc_t spec_mat_desc;
    spec_mat_desc.base_color[0] = 0.5f;
    spec_mat_desc.base_color[1] = 0.0f;
    spec_mat_desc.base_color[2] = 0.5f;
    spec_mat_desc.specular_tint = 1.0f;
    spec_mat_desc.roughness = 1.0f;

    run_material_test("tint_mat4", spec_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Metal material tests
//

void test_metal_mat0() {
    const int SampleCount = 512;
    const double MinPSNR = 36.04;
    const int PixThres = 29;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.0f;
    metal_mat_desc.base_color[1] = 0.5f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test("metal_mat0", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_metal_mat1() {
    const int SampleCount = 512;
    const double MinPSNR = 37.21;
    const int PixThres = 9;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.5f;
    metal_mat_desc.base_color[2] = 0.0f;
    metal_mat_desc.roughness = 0.25f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test("metal_mat1", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_metal_mat2() {
    const int SampleCount = 256;
    const double MinPSNR = 39.92;
    const int PixThres = 0;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.5f;
    metal_mat_desc.roughness = 0.5f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test("metal_mat2", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_metal_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 41.62;
    const int PixThres = 0;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.0f;
    metal_mat_desc.base_color[1] = 0.5f;
    metal_mat_desc.base_color[2] = 0.0f;
    metal_mat_desc.roughness = 0.75f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test("metal_mat3", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_metal_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 41.86;
    const int PixThres = 0;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_color[0] = 0.5f;
    metal_mat_desc.base_color[1] = 0.0f;
    metal_mat_desc.base_color[2] = 0.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.metallic = 1.0f;

    run_material_test("metal_mat4", metal_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Emissive material tests
//

void test_emit_mat0() {
    const int SampleCount = 512;
    const double MinPSNR = 36.92;
    const int PixThres = 10;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 0.5f;

    run_material_test("emit_mat0", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_NO_LIGHTS);
}

void test_emit_mat1() {
    const int SampleCount = 512;
    const double MinPSNR = 35.13;
    const int PixThres = 9;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;

    mat_desc.emission_color[0] = 1.0f;
    mat_desc.emission_color[1] = 1.0f;
    mat_desc.emission_color[2] = 1.0f;
    mat_desc.emission_strength = 1.0f;

    run_material_test("emit_mat1", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_NO_LIGHTS);
}

//
// Clear coat material tests
//

void test_coat_mat0() {
    const int SampleCount = 512;
    const double MinPSNR = 42.95;
    const int PixThres = 4;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.0f;

    run_material_test("coat_mat0", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat1() {
    const int SampleCount = 256;
    const double MinPSNR = 40.5;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.25f;

    run_material_test("coat_mat1", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat2() {
    const int SampleCount = 256;
    const double MinPSNR = 40.91;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.5f;

    run_material_test("coat_mat2", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat3() {
    const int SampleCount = 256;
    const double MinPSNR = 40.55;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 0.75f;

    run_material_test("coat_mat3", mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_coat_mat4() {
    const int SampleCount = 256;
    const double MinPSNR = 39.79;
    const int PixThres = 0;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.specular = 0.0f;
    mat_desc.clearcoat = 1.0f;
    mat_desc.clearcoat_roughness = 1.0f;

    run_material_test("coat_mat4", mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Refractive material tests
//

void test_refr_mis0() {
    const int SampleCount = 256;
    const double MinPSNR = 39.76;
    const int PixThres = 54;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test("refr_mis0", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, REFR_PLANE_SCENE);
}

void test_refr_mis1() {
    const int SampleCount = 256;
    const double MinPSNR = 39.75;
    const int PixThres = 24;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.25f;

    run_material_test("refr_mis1", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, REFR_PLANE_SCENE);
}

void test_refr_mis2() {
    const int SampleCount = 256;
    const double MinPSNR = 42.07;
    const int PixThres = 0;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test("refr_mis2", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, REFR_PLANE_SCENE);
}

void test_refr_mis3() {
    const int SampleCount = 256;
    const double MinPSNR = 43.51;
    const int PixThres = 0;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.75f;

    run_material_test("refr_mis3", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, REFR_PLANE_SCENE);
}

void test_refr_mis4() {
    const int SampleCount = 512;
    const double MinPSNR = 46.30;
    const int PixThres = 0;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test("refr_mis4", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, REFR_PLANE_SCENE);
}

///

void test_refr_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 31.61;
    const int PixThres = 296;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.001f;
    mat_desc.roughness = 1.0f;

    run_material_test("refr_mat0", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_refr_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.48;
    const int PixThres = 186;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.0f;

    run_material_test("refr_mat1", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_refr_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 31.7;
    const int PixThres = 1593;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.25f;

    run_material_test("refr_mat2", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_refr_mat3() {
    const int SampleCount = 1024;
    const double MinPSNR = 35.92;
    const int PixThres = 56;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 0.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.5f;

    run_material_test("refr_mat3", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_refr_mat4() {
    const int SampleCount = 1024;
    const double MinPSNR = 38.11;
    const int PixThres = 0;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 0.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 0.75f;

    run_material_test("refr_mat4", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_refr_mat5() {
    const int SampleCount = 1024;
    const double MinPSNR = 37.25;
    const int PixThres = 0;

    Ray::shading_node_desc_t mat_desc;
    mat_desc.type = Ray::RefractiveNode;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 0.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.int_ior = 1.45f;
    mat_desc.roughness = 1.0f;

    run_material_test("refr_mat5", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

//
// Transmissive material tests
//

void test_trans_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 31.60;
    const int PixThres = 301;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.001f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test("trans_mat0", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 28.8;
    const int PixThres = 587;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test("trans_mat1", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 26.38;
    const int PixThres = 1941;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.25f;

    run_material_test("trans_mat2", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat3() {
    const int SampleCount = 1024;
    const double MinPSNR = 30.12;
    const int PixThres = 396;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.5f;

    run_material_test("trans_mat3", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat4() {
    const int SampleCount = 1024;
    const double MinPSNR = 32.22;
    const int PixThres = 77;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.75f;

    run_material_test("trans_mat4", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat5() {
    const int SampleCount = 1024;
    const double MinPSNR = 33.17;
    const int PixThres = 40;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 1.0f;

    run_material_test("trans_mat5", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat6() {
    const int SampleCount = 1024;
    const double MinPSNR = 25.33;
    const int PixThres = 2584;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.25f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test("trans_mat6", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat7() {
    const int SampleCount = 1024;
    const double MinPSNR = 28.84;
    const int PixThres = 665;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.5f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test("trans_mat7", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat8() {
    const int SampleCount = 1024;
    const double MinPSNR = 32.21;
    const int PixThres = 48;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 0.75f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test("trans_mat8", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

void test_trans_mat9() {
    const int SampleCount = 1024;
    const double MinPSNR = 35.34;
    const int PixThres = 1;

    Ray::principled_mat_desc_t mat_desc;
    mat_desc.base_color[0] = 1.0f;
    mat_desc.base_color[1] = 1.0f;
    mat_desc.base_color[2] = 1.0f;
    mat_desc.specular = 0.0f;
    mat_desc.ior = 1.45f;
    mat_desc.roughness = 1.0f;
    mat_desc.transmission = 1.0f;
    mat_desc.transmission_roughness = 0.0f;

    run_material_test("trans_mat9", mat_desc, SampleCount, MinPSNR, PixThres, nullptr, STANDARD_SCENE_MESH_LIGHTS);
}

//
// Transparent material tests
//

void test_alpha_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 38.74;
    const int PixThres = 16;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.75f;

    run_material_test("alpha_mat0", alpha_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_alpha_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 35.54;
    const int PixThres = 8;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.5f;

    run_material_test("alpha_mat1", alpha_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_alpha_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 35.92;
    const int PixThres = 5;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.25f;

    run_material_test("alpha_mat2", alpha_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_alpha_mat3() {
    const int SampleCount = 1024;
    const double MinPSNR = 38.2;
    const int PixThres = 24;

    Ray::principled_mat_desc_t alpha_mat_desc;
    alpha_mat_desc.base_color[0] = 0.0f;
    alpha_mat_desc.base_color[1] = 0.0f;
    alpha_mat_desc.base_color[2] = 0.5f;
    alpha_mat_desc.roughness = 0.0f;
    alpha_mat_desc.alpha = 0.0f;

    run_material_test("alpha_mat3", alpha_mat_desc, SampleCount, MinPSNR, PixThres);
}

//
// Complex material tests
//

void test_complex_mat0() {
    const int SampleCount = 1024;
    const double MinPSNR = 38.7;
    const int PixThres = 5;

    Ray::principled_mat_desc_t wood_mat_desc;
    wood_mat_desc.base_texture = 0;
    wood_mat_desc.roughness = 1.0f;
    wood_mat_desc.roughness_texture = 2;
    wood_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/bamboo-wood-semigloss-albedo.tga",
        "test_data/textures/bamboo-wood-semigloss-normal.tga",
        "test_data/textures/bamboo-wood-semigloss-roughness.tga",
    };

    run_material_test("complex_mat0", wood_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}

void test_complex_mat1() {
    const int SampleCount = 1024;
    const double MinPSNR = 40.44;
    const int PixThres = 1;

    Ray::principled_mat_desc_t wood_mat_desc;
    wood_mat_desc.base_texture = 0;
    wood_mat_desc.roughness = 1.0f;
    wood_mat_desc.roughness_texture = 2;
    wood_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/older-wood-flooring_albedo.tga",
        "test_data/textures/older-wood-flooring_normal-ogl.tga",
        "test_data/textures/older-wood-flooring_roughness.tga",
    };

    run_material_test("complex_mat1", wood_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}

void test_complex_mat2() {
    const int SampleCount = 1024;
    const double MinPSNR = 40.26;
    const int PixThres = 3;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = 0;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = 2;
    metal_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/streaky-metal1_albedo.tga",
        "test_data/textures/streaky-metal1_normal-ogl.tga",
        "test_data/textures/streaky-metal1_roughness.tga",
    };

    run_material_test("complex_mat2", metal_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}

void test_complex_mat3() {
    const int SampleCount = 1024;
    const double MinPSNR = 40.58;
    const int PixThres = 2;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = 0;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = 2;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = 3;
    metal_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/rusting-lined-metal_albedo.tga", "test_data/textures/rusting-lined-metal_normal-ogl.tga",
        "test_data/textures/rusting-lined-metal_roughness.tga", "test_data/textures/rusting-lined-metal_metallic.tga"};

    run_material_test("complex_mat3", metal_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}

void test_complex_mat4() {
    const int SampleCount = 1024;
    const double MinPSNR = 34.88;
    const int PixThres = 121;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = 0;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = 2;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = 3;
    metal_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test("complex_mat4", metal_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}

void test_complex_mat4_mesh_lights() {
    const int SampleCount = 1024;
    const double MinPSNR = 32.36;
    const int PixThres = 888;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = 0;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = 2;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = 3;
    metal_mat_desc.normal_map = 1;

    const char *textures[] = {
        "test_data/textures/gold-scuffed_basecolor-boosted.tga", "test_data/textures/gold-scuffed_normal.tga",
        "test_data/textures/gold-scuffed_roughness.tga", "test_data/textures/gold-scuffed_metallic.tga"};

    run_material_test("complex_mat4_mesh_lights", metal_mat_desc, SampleCount, MinPSNR, PixThres, textures,
                      STANDARD_SCENE_MESH_LIGHTS);
}

void test_complex_mat5() {
    const int SampleCount = 1024;
    const double MinPSNR = 31.65;
    const int PixThres = 241;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test("complex_mat5", olive_mat_desc, SampleCount, MinPSNR, PixThres);
}

void test_complex_mat5_mesh_lights() {
    const int SampleCount = 1024;
    const double MinPSNR = 30.80;
    const int PixThres = 486;

    Ray::principled_mat_desc_t olive_mat_desc;
    olive_mat_desc.base_color[0] = 0.836164f;
    olive_mat_desc.base_color[1] = 0.836164f;
    olive_mat_desc.base_color[2] = 0.656603f;
    olive_mat_desc.roughness = 0.041667f;
    olive_mat_desc.transmission = 1.0f;
    olive_mat_desc.ior = 2.3f;

    run_material_test("complex_mat5_mesh_lights", olive_mat_desc, SampleCount, MinPSNR, PixThres, nullptr,
                      STANDARD_SCENE_MESH_LIGHTS);
}

void test_complex_mat6() {
    const int SampleCount = 256;
    const double MinPSNR = 39.74;
    const int PixThres = 0;

    Ray::principled_mat_desc_t metal_mat_desc;
    metal_mat_desc.base_texture = 0;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.roughness = 1.0f;
    metal_mat_desc.roughness_texture = 2;
    metal_mat_desc.metallic = 1.0f;
    metal_mat_desc.metallic_texture = 3;
    metal_mat_desc.normal_map = 1;
    metal_mat_desc.normal_map_intensity = 0.3f;

    const char *textures[] = {
        "test_data/textures/stone_trims_02_BaseColor.tga", "test_data/textures/stone_trims_02_Normal.tga",
        "test_data/textures/stone_trims_02_Roughness.tga", "test_data/textures/stone_trims_02_Metallic.tga"};

    run_material_test("complex_mat6", metal_mat_desc, SampleCount, MinPSNR, PixThres, textures);
}
