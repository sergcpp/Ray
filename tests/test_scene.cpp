#include "test_scene.h"

#include <cstring>

#include "test_common.h"

#include "../Ray.h"

#include "thread_pool.h"
#include "utils.h"

extern bool g_minimal_output;
extern std::mutex g_stdout_mtx;

LogErr g_log_err;

void load_needed_textures(Ray::SceneBase &scene, Ray::shading_node_desc_t &mat_desc, const char *textures[]) {
    if (!textures) {
        return;
    }

    if (mat_desc.base_texture != Ray::InvalidTextureHandle && textures[0]) {
        int img_w, img_h;
        auto img_data = LoadTGA(textures[0], true /* flip_y */, img_w, img_h);
        require(!img_data.empty());

        // drop alpha channel
        for (int i = 0; i < img_w * img_h; ++i) {
            img_data[3 * i + 0] = img_data[4 * i + 0];
            img_data[3 * i + 1] = img_data[4 * i + 1];
            img_data[3 * i + 2] = img_data[4 * i + 2];
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = true;

        mat_desc.base_texture = scene.AddTexture(tex_desc);
    }
}

void load_needed_textures(Ray::SceneBase &scene, Ray::principled_mat_desc_t &mat_desc, const char *textures[]) {
    if (!textures) {
        return;
    }

    if (mat_desc.base_texture != Ray::InvalidTextureHandle && textures[mat_desc.base_texture._index]) {
        int img_w = 0, img_h = 0, mips = 1;
        Ray::eTextureFormat format = Ray::eTextureFormat::RGB888;
        Ray::eTextureConvention convention = Ray::eTextureConvention::OGL;
        std::vector<uint8_t> img_data;

        if (strstr(textures[mat_desc.base_texture._index], ".tga")) {
            img_data = LoadTGA(textures[mat_desc.base_texture._index], true /* flip_y */, img_w, img_h);
            require(!img_data.empty());

            // drop alpha channel
            for (int i = 0; i < img_w * img_h; ++i) {
                img_data[3 * i + 0] = img_data[4 * i + 0];
                img_data[3 * i + 1] = img_data[4 * i + 1];
                img_data[3 * i + 2] = img_data[4 * i + 2];
            }
        } else if (strstr(textures[mat_desc.base_texture._index], ".dds")) {
            int channels = 0;
            img_data = LoadDDS(textures[mat_desc.base_texture._index], img_w, img_h, mips, channels);
            require_fatal(channels == 3);
            format = Ray::eTextureFormat::BC1;
            convention = Ray::eTextureConvention::DX;
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = format;
        tex_desc.convention = convention;
        tex_desc.mips_count = mips;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = true;

        mat_desc.base_texture = scene.AddTexture(tex_desc);
    }

    if (mat_desc.normal_map != Ray::InvalidTextureHandle && textures[mat_desc.normal_map._index]) {
        int img_w = 0, img_h = 0, mips = 1;
        Ray::eTextureFormat format = Ray::eTextureFormat::RGB888;
        Ray::eTextureConvention convention = Ray::eTextureConvention::OGL;
        std::vector<uint8_t> img_data;

        if (strstr(textures[mat_desc.normal_map._index], ".tga")) {
            img_data = LoadTGA(textures[mat_desc.normal_map._index], true /* flip_y */, img_w, img_h);
            require(!img_data.empty());

            // drop alpha channel
            for (int i = 0; i < img_w * img_h; ++i) {
                img_data[3 * i + 0] = img_data[4 * i + 0];
                img_data[3 * i + 1] = img_data[4 * i + 1];
                img_data[3 * i + 2] = img_data[4 * i + 2];
            }
        } else if (strstr(textures[mat_desc.normal_map._index], ".dds")) {
            int channels = 0;
            img_data = LoadDDS(textures[mat_desc.normal_map._index], img_w, img_h, mips, channels);
            require_fatal(channels == 2);
            format = Ray::eTextureFormat::BC5;
            convention = Ray::eTextureConvention::DX;
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = format;
        tex_desc.convention = convention;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.is_normalmap = true;
        tex_desc.generate_mipmaps = false;
        tex_desc.is_srgb = false;

        mat_desc.normal_map = scene.AddTexture(tex_desc);
    }

    if (mat_desc.roughness_texture != Ray::InvalidTextureHandle && textures[mat_desc.roughness_texture._index]) {
        int img_w = 0, img_h = 0, mips = 1;
        Ray::eTextureFormat format = Ray::eTextureFormat::R8;
        Ray::eTextureConvention convention = Ray::eTextureConvention::OGL;
        std::vector<uint8_t> img_data;

        if (strstr(textures[mat_desc.roughness_texture._index], ".tga")) {
            img_data = LoadTGA(textures[mat_desc.roughness_texture._index], true /* flip_y */, img_w, img_h);
            require(!img_data.empty());

            // use only red channel
            for (int i = 0; i < img_w * img_h; ++i) {
                img_data[i] = img_data[4 * i + 0];
            }
        } else if (strstr(textures[mat_desc.roughness_texture._index], ".dds")) {
            int channels = 0;
            img_data = LoadDDS(textures[mat_desc.roughness_texture._index], img_w, img_h, mips, channels);
            require_fatal(channels == 1);
            format = Ray::eTextureFormat::BC4;
            convention = Ray::eTextureConvention::DX;
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = format;
        tex_desc.convention = convention;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = false;

        mat_desc.roughness_texture = scene.AddTexture(tex_desc);
    }

    if (mat_desc.metallic_texture != Ray::InvalidTextureHandle && textures[mat_desc.metallic_texture._index]) {
        int img_w = 0, img_h = 0, mips = 1;
        Ray::eTextureFormat format = Ray::eTextureFormat::R8;
        Ray::eTextureConvention convention = Ray::eTextureConvention::OGL;
        std::vector<uint8_t> img_data;

        if (strstr(textures[mat_desc.metallic_texture._index], ".tga")) {
            img_data = LoadTGA(textures[mat_desc.metallic_texture._index], true /* flip_y */, img_w, img_h);
            require(!img_data.empty());

            // use only red channel
            for (int i = 0; i < img_w * img_h; ++i) {
                img_data[i] = img_data[4 * i + 0];
            }
        } else if (strstr(textures[mat_desc.metallic_texture._index], ".dds")) {
            int channels = 0;
            img_data = LoadDDS(textures[mat_desc.metallic_texture._index], img_w, img_h, mips, channels);
            require_fatal(channels == 1);
            format = Ray::eTextureFormat::BC4;
            convention = Ray::eTextureConvention::DX;
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = format;
        tex_desc.convention = convention;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = true;
        tex_desc.is_srgb = false;

        mat_desc.metallic_texture = scene.AddTexture(tex_desc);
    }

    if (mat_desc.alpha_texture != Ray::InvalidTextureHandle && textures[mat_desc.alpha_texture._index]) {
        int img_w = 0, img_h = 0, mips = 1;
        Ray::eTextureFormat format = Ray::eTextureFormat::R8;
        Ray::eTextureConvention convention = Ray::eTextureConvention::OGL;
        std::vector<uint8_t> img_data;

        if (strstr(textures[mat_desc.alpha_texture._index], ".tga")) {
            img_data = LoadTGA(textures[mat_desc.alpha_texture._index], true /* flip_y */, img_w, img_h);
            require(!img_data.empty());

            // use only red channel
            for (int i = 0; i < img_w * img_h; ++i) {
                img_data[i] = img_data[4 * i + 0];
            }
        } else if (strstr(textures[mat_desc.alpha_texture._index], ".dds")) {
            int channels = 0;
            img_data = LoadDDS(textures[mat_desc.alpha_texture._index], img_w, img_h, mips, channels);
            require_fatal(channels == 1);
            format = Ray::eTextureFormat::BC4;
            convention = Ray::eTextureConvention::DX;
        }

        Ray::tex_desc_t tex_desc;
        tex_desc.format = format;
        tex_desc.convention = convention;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = false;
        tex_desc.is_srgb = false;

        mat_desc.alpha_texture = scene.AddTexture(tex_desc);
    }
}

template <typename MatDesc>
void setup_test_scene(ThreadPool &threads, Ray::SceneBase &scene, const int min_samples, const float variance_threshold,
                      const MatDesc &main_mat_desc, const char *textures[], const eTestScene test_scene) {
    { // setup camera
        static const float view_origin_standard[] = {0.16149f, 0.294997f, 0.332965f};
        static const float view_dir_standard[] = {-0.364128768f, -0.555621922f, -0.747458696f};
        static const float view_origin_refr[] = {-0.074711f, 0.099348f, -0.049506f};
        static const float view_dir_refr[] = {0.725718915f, 0.492017448f, 0.480885535f};
        static const float view_up[] = {0.0f, 1.0f, 0.0f};

        Ray::camera_desc_t cam_desc;
        cam_desc.type = Ray::eCamType::Persp;
        cam_desc.filter = Ray::ePixelFilter::Box;
        if (test_scene == eTestScene::Standard_SunLight || test_scene == eTestScene::Standard_MoonLight) {
            cam_desc.view_transform = Ray::eViewTransform::AgX;
        } else if (test_scene == eTestScene::Standard_DirLight) {
            cam_desc.view_transform = Ray::eViewTransform::Filmic_HighContrast;
        } else {
            cam_desc.view_transform = Ray::eViewTransform::Standard;
        }
        if (test_scene == eTestScene::Refraction_Plane) {
            memcpy(&cam_desc.origin[0], &view_origin_refr[0], 3 * sizeof(float));
            memcpy(&cam_desc.fwd[0], &view_dir_refr[0], 3 * sizeof(float));
            cam_desc.fov = 45.1806f;
        } else {
            memcpy(&cam_desc.origin[0], &view_origin_standard[0], 3 * sizeof(float));
            memcpy(&cam_desc.fwd[0], &view_dir_standard[0], 3 * sizeof(float));
            cam_desc.fov = 18.1806f;
        }
        if (test_scene == eTestScene::Standard_Clipped) {
            cam_desc.clip_start = 0.4f;
            cam_desc.clip_end = 0.5f;
        }
        memcpy(&cam_desc.up[0], &view_up[0], 3 * sizeof(float));

        cam_desc.regularize_alpha = 0.0f; // disabled

        if (test_scene == eTestScene::Standard_DOF0) {
            cam_desc.sensor_height = 0.018f;
            cam_desc.focus_distance = 0.1f;
            cam_desc.fstop = 0.1f;
            cam_desc.lens_blades = 6;
            cam_desc.lens_rotation = 30.0f * 3.141592653589f / 180.0f;
            cam_desc.lens_ratio = 2.0f;
        } else if (test_scene == eTestScene::Standard_DOF1) {
            cam_desc.sensor_height = 0.018f;
            cam_desc.focus_distance = 0.4f;
            cam_desc.fstop = 0.1f;
            cam_desc.lens_blades = 0;
            cam_desc.lens_rotation = 30.0f * 3.141592653589f / 180.0f;
            cam_desc.lens_ratio = 2.0f;
        } else if (test_scene == eTestScene::Standard_GlassBall0 || test_scene == eTestScene::Standard_GlassBall1) {
            cam_desc.max_diff_depth = 8;
            cam_desc.max_spec_depth = 8;
            cam_desc.max_refr_depth = 8;
            cam_desc.max_total_depth = 9;
        } else if (test_scene == eTestScene::Ray_Flags) {
            cam_desc.regularize_alpha = 0.1f;
        } else if (test_scene == eTestScene::Standard_SunLight) {
            cam_desc.exposure = -14.0f;
        } else if (test_scene == eTestScene::Standard_MoonLight) {
            cam_desc.exposure = 8.0f;
        }

        cam_desc.min_total_depth = 4;

        cam_desc.min_samples = min_samples;
        cam_desc.variance_threshold = variance_threshold;

        const Ray::CameraHandle cam = scene.AddCamera(cam_desc);
        scene.set_current_cam(cam);
    }

    MatDesc main_mat_desc_copy = main_mat_desc;
    load_needed_textures(scene, main_mat_desc_copy, textures);
    const Ray::MaterialHandle main_mat = scene.AddMaterial(main_mat_desc_copy);

    Ray::MaterialHandle floor_mat;
    {
        Ray::principled_mat_desc_t floor_mat_desc;
        floor_mat_desc.base_color[0] = 0.75f;
        floor_mat_desc.base_color[1] = 0.75f;
        floor_mat_desc.base_color[2] = 0.75f;
        floor_mat_desc.roughness = 0.0f;
        floor_mat_desc.specular = 0.0f;
        floor_mat = scene.AddMaterial(floor_mat_desc);
    }

    Ray::MaterialHandle walls_mat;
    {
        Ray::principled_mat_desc_t walls_mat_desc;
        walls_mat_desc.base_color[0] = 0.5f;
        walls_mat_desc.base_color[1] = 0.5f;
        walls_mat_desc.base_color[2] = 0.5f;
        walls_mat_desc.roughness = 0.0f;
        walls_mat_desc.specular = 0.0f;
        walls_mat = scene.AddMaterial(walls_mat_desc);
    }

    Ray::MaterialHandle white_mat;
    {
        Ray::principled_mat_desc_t white_mat_desc;
        white_mat_desc.base_color[0] = 0.64f;
        white_mat_desc.base_color[1] = 0.64f;
        white_mat_desc.base_color[2] = 0.64f;
        white_mat_desc.roughness = 0.0f;
        white_mat_desc.specular = 0.0f;
        white_mat = scene.AddMaterial(white_mat_desc);
    }

    Ray::MaterialHandle light_grey_mat;
    {
        Ray::principled_mat_desc_t light_grey_mat_desc;
        light_grey_mat_desc.base_color[0] = 0.32f;
        light_grey_mat_desc.base_color[1] = 0.32f;
        light_grey_mat_desc.base_color[2] = 0.32f;
        light_grey_mat_desc.roughness = 0.0f;
        light_grey_mat_desc.specular = 0.0f;
        light_grey_mat = scene.AddMaterial(light_grey_mat_desc);
    }

    Ray::MaterialHandle mid_grey_mat;
    {
        Ray::principled_mat_desc_t mid_grey_mat_desc;
        mid_grey_mat_desc.base_color[0] = 0.16f;
        mid_grey_mat_desc.base_color[1] = 0.16f;
        mid_grey_mat_desc.base_color[2] = 0.16f;
        mid_grey_mat_desc.roughness = 0.0f;
        mid_grey_mat_desc.specular = 0.0f;
        mid_grey_mat = scene.AddMaterial(mid_grey_mat_desc);
    }

    Ray::MaterialHandle dark_grey_mat;
    {
        Ray::principled_mat_desc_t dark_grey_mat_desc;
        dark_grey_mat_desc.base_color[0] = 0.08f;
        dark_grey_mat_desc.base_color[1] = 0.08f;
        dark_grey_mat_desc.base_color[2] = 0.08f;
        dark_grey_mat_desc.roughness = 0.0f;
        dark_grey_mat_desc.specular = 0.0f;
        dark_grey_mat = scene.AddMaterial(dark_grey_mat_desc);
    }

    Ray::MaterialHandle square_light_mat;
    {
        Ray::shading_node_desc_t square_light_mat_desc;
        square_light_mat_desc.type = Ray::eShadingNode::Emissive;
        square_light_mat_desc.strength = 20.3718f;
        square_light_mat_desc.multiple_importance = true;
        square_light_mat_desc.base_color[0] = 1.0f;
        square_light_mat_desc.base_color[1] = 1.0f;
        square_light_mat_desc.base_color[2] = 1.0f;
        square_light_mat = scene.AddMaterial(square_light_mat_desc);
    }

    Ray::MaterialHandle disc_light_mat;
    {
        Ray::shading_node_desc_t disc_light_mat_desc;
        disc_light_mat_desc.type = Ray::eShadingNode::Emissive;
        disc_light_mat_desc.strength = 81.4873f;
        disc_light_mat_desc.multiple_importance = true;
        disc_light_mat_desc.base_color[0] = 1.0f;
        disc_light_mat_desc.base_color[1] = 1.0f;
        disc_light_mat_desc.base_color[2] = 1.0f;
        disc_light_mat = scene.AddMaterial(disc_light_mat_desc);
    }

    Ray::MaterialHandle glossy_red, glossy_green;
    {
        Ray::shading_node_desc_t glossy_mat_desc;
        glossy_mat_desc.type = Ray::eShadingNode::Glossy;
        glossy_mat_desc.base_color[0] = 1.0f;
        glossy_mat_desc.base_color[1] = glossy_mat_desc.base_color[2] = 0.0f;

        glossy_red = scene.AddMaterial(glossy_mat_desc);

        glossy_mat_desc.base_color[1] = 1.0f;
        glossy_mat_desc.base_color[0] = glossy_mat_desc.base_color[2] = 0.0f;

        glossy_green = scene.AddMaterial(glossy_mat_desc);
    }

    Ray::MaterialHandle refr_mat_flags;
    {
        Ray::principled_mat_desc_t refr_mat_flags_desc;
        refr_mat_flags_desc.roughness = 0.0f;
        refr_mat_flags_desc.transmission = 1.0f;
        refr_mat_flags_desc.ior = 2.3f;
        refr_mat_flags = scene.AddMaterial(refr_mat_flags_desc);
    }

    Ray::MaterialHandle glassball_mat0;
    if (test_scene == eTestScene::Standard_GlassBall0) {
        Ray::shading_node_desc_t glassball_mat0_desc;
        glassball_mat0_desc.type = Ray::eShadingNode::Refractive;
        glassball_mat0_desc.base_color[0] = 1.0f;
        glassball_mat0_desc.base_color[1] = 1.0f;
        glassball_mat0_desc.base_color[2] = 1.0f;
        glassball_mat0_desc.roughness = 0.0f;
        glassball_mat0_desc.ior = 1.45f;
        glassball_mat0 = scene.AddMaterial(glassball_mat0_desc);
    } else {
        Ray::principled_mat_desc_t glassball_mat0_desc;
        glassball_mat0_desc.base_color[0] = 1.0f;
        glassball_mat0_desc.base_color[1] = 1.0f;
        glassball_mat0_desc.base_color[2] = 1.0f;
        glassball_mat0_desc.roughness = 0.0f;
        glassball_mat0_desc.ior = 1.45f;
        glassball_mat0_desc.transmission = 1.0f;
        glassball_mat0 = scene.AddMaterial(glassball_mat0_desc);
    }

    Ray::MaterialHandle glassball_mat1;
    if (test_scene == eTestScene::Standard_GlassBall0) {
        Ray::shading_node_desc_t glassball_mat1_desc;
        glassball_mat1_desc.type = Ray::eShadingNode::Refractive;
        glassball_mat1_desc.base_color[0] = 1.0f;
        glassball_mat1_desc.base_color[1] = 1.0f;
        glassball_mat1_desc.base_color[2] = 1.0f;
        glassball_mat1_desc.roughness = 0.0f;
        glassball_mat1_desc.ior = 1.0f;
        glassball_mat1 = scene.AddMaterial(glassball_mat1_desc);
    } else {
        Ray::principled_mat_desc_t glassball_mat1_desc;
        glassball_mat1_desc.base_color[0] = 1.0f;
        glassball_mat1_desc.base_color[1] = 1.0f;
        glassball_mat1_desc.base_color[2] = 1.0f;
        glassball_mat1_desc.roughness = 0.0f;
        glassball_mat1_desc.ior = 1.0f;
        glassball_mat1_desc.transmission = 1.0f;
        glassball_mat1 = scene.AddMaterial(glassball_mat1_desc);
    }

    Ray::MaterialHandle two_sided_back;
    {
        Ray::principled_mat_desc_t back_mat_desc;
        back_mat_desc.base_color[0] = 0.0f;
        back_mat_desc.base_color[1] = 0.0f;
        back_mat_desc.base_color[2] = 0.5f;
        back_mat_desc.roughness = 0.0f;
        two_sided_back = scene.AddMaterial(back_mat_desc);
    }

    std::vector<Ray::MeshHandle> meshes_to_delete;

    Ray::MeshHandle base_mesh;
    {
        std::vector<float> base_attrs;
        std::vector<uint32_t> base_indices, base_groups;
        std::tie(base_attrs, base_indices, base_groups) = LoadBIN("test_data/meshes/mat_test/base.bin");

        Ray::mesh_desc_t base_mesh_desc;
        base_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        base_mesh_desc.vtx_positions = {base_attrs, 0, 8};
        base_mesh_desc.vtx_normals = {base_attrs, 3, 8};
        base_mesh_desc.vtx_uvs = {base_attrs, 6, 8};
        base_mesh_desc.vtx_indices = base_indices;

        const Ray::mat_group_desc_t groups[] = {{mid_grey_mat, base_groups[0], base_groups[1]}};
        base_mesh_desc.groups = groups;

        base_mesh = scene.AddMesh(base_mesh_desc);
    }

    Ray::MeshHandle model_mesh;
    {
        std::vector<float> model_attrs;
        std::vector<uint32_t> model_indices, model_groups;
        if (test_scene == eTestScene::Refraction_Plane) {
            std::tie(model_attrs, model_indices, model_groups) = LoadBIN("test_data/meshes/mat_test/refr_plane.bin");
        } else {
            std::tie(model_attrs, model_indices, model_groups) = LoadBIN("test_data/meshes/mat_test/model.bin");
        }

        Ray::mesh_desc_t model_mesh_desc;
        model_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        model_mesh_desc.vtx_positions = {model_attrs, 0, 8};
        model_mesh_desc.vtx_normals = {model_attrs, 3, 8};
        model_mesh_desc.vtx_uvs = {model_attrs, 6, 8};
        model_mesh_desc.vtx_indices = model_indices;

        const Ray::mat_group_desc_t groups[] = {{main_mat, model_groups[0], model_groups[1]}};
        model_mesh_desc.groups = groups;

        model_mesh = scene.AddMesh(model_mesh_desc);
    }

    Ray::MeshHandle core_mesh;
    {
        std::vector<float> core_attrs;
        std::vector<uint32_t> core_indices, core_groups;
        std::tie(core_attrs, core_indices, core_groups) = LoadBIN("test_data/meshes/mat_test/core.bin");

        Ray::mesh_desc_t core_mesh_desc;
        core_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        core_mesh_desc.vtx_positions = {core_attrs, 0, 8};
        core_mesh_desc.vtx_normals = {core_attrs, 3, 8};
        core_mesh_desc.vtx_uvs = {core_attrs, 6, 8};
        core_mesh_desc.vtx_indices = core_indices;

        const Ray::mat_group_desc_t groups[] = {{mid_grey_mat, core_groups[0], core_groups[1]}};
        core_mesh_desc.groups = groups;

        core_mesh = scene.AddMesh(core_mesh_desc);
    }

    Ray::MeshHandle subsurf_bar_mesh;
    {
        std::vector<float> subsurf_bar_attrs;
        std::vector<uint32_t> subsurf_bar_indices, subsurf_bar_groups;
        std::tie(subsurf_bar_attrs, subsurf_bar_indices, subsurf_bar_groups) =
            LoadBIN("test_data/meshes/mat_test/subsurf_bar.bin");

        Ray::mesh_desc_t subsurf_bar_mesh_desc;
        subsurf_bar_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        subsurf_bar_mesh_desc.vtx_positions = {subsurf_bar_attrs, 0, 8};
        subsurf_bar_mesh_desc.vtx_normals = {subsurf_bar_attrs, 3, 8};
        subsurf_bar_mesh_desc.vtx_uvs = {subsurf_bar_attrs, 6, 8};
        subsurf_bar_mesh_desc.vtx_indices = subsurf_bar_indices;

        const Ray::mat_group_desc_t groups[] = {{white_mat, subsurf_bar_groups[0], subsurf_bar_groups[1]},
                                                {dark_grey_mat, subsurf_bar_groups[2], subsurf_bar_groups[3]}};
        subsurf_bar_mesh_desc.groups = groups;

        subsurf_bar_mesh = scene.AddMesh(subsurf_bar_mesh_desc);
    }

    Ray::MeshHandle text_mesh;
    {
        std::vector<float> text_attrs;
        std::vector<uint32_t> text_indices, text_groups;
        std::tie(text_attrs, text_indices, text_groups) = LoadBIN("test_data/meshes/mat_test/text.bin");

        Ray::mesh_desc_t text_mesh_desc;
        text_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        text_mesh_desc.vtx_positions = {text_attrs, 0, 8};
        text_mesh_desc.vtx_normals = {text_attrs, 3, 8};
        text_mesh_desc.vtx_uvs = {text_attrs, 6, 8};
        text_mesh_desc.vtx_indices = text_indices;

        const Ray::mat_group_desc_t groups[] = {{white_mat, text_groups[0], text_groups[1]}};
        text_mesh_desc.groups = groups;

        text_mesh = scene.AddMesh(text_mesh_desc);

        // Add mesh one more time to test compaction later
        meshes_to_delete.push_back(text_mesh);
        text_mesh = scene.AddMesh(text_mesh_desc);
    }

    Ray::MeshHandle two_sided_mesh;
    {
        std::vector<float> text_attrs;
        std::vector<uint32_t> text_indices, text_groups;
        std::tie(text_attrs, text_indices, text_groups) = LoadBIN("test_data/meshes/mat_test/two_sided.bin");

        Ray::mesh_desc_t mesh_desc;
        mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        mesh_desc.vtx_positions = {text_attrs, 0, 8};
        mesh_desc.vtx_normals = {text_attrs, 3, 8};
        mesh_desc.vtx_uvs = {text_attrs, 6, 8};
        mesh_desc.vtx_indices = text_indices;

        const Ray::mat_group_desc_t groups[] = {{main_mat, two_sided_back, text_groups[0], text_groups[1]}};
        mesh_desc.groups = groups;

        two_sided_mesh = scene.AddMesh(mesh_desc);
    }

    Ray::MeshHandle env_mesh;
    {
        std::vector<float> env_attrs;
        std::vector<uint32_t> env_indices, env_groups;
        if (test_scene == eTestScene::Standard_DirLight || test_scene == eTestScene::Standard_SunLight ||
            test_scene == eTestScene::Standard_MoonLight || test_scene == eTestScene::Standard_HDRLight) {
            std::tie(env_attrs, env_indices, env_groups) = LoadBIN("test_data/meshes/mat_test/env_floor.bin");
        } else {
            std::tie(env_attrs, env_indices, env_groups) = LoadBIN("test_data/meshes/mat_test/env.bin");
        }

        Ray::mesh_desc_t env_mesh_desc;
        env_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        env_mesh_desc.vtx_positions = {env_attrs, 0, 8};
        env_mesh_desc.vtx_normals = {env_attrs, 3, 8};
        env_mesh_desc.vtx_uvs = {env_attrs, 6, 8};
        env_mesh_desc.vtx_indices = env_indices;

        std::vector<Ray::mat_group_desc_t> groups;
        if (test_scene == eTestScene::Standard_DirLight || test_scene == eTestScene::Standard_SunLight ||
            test_scene == eTestScene::Standard_MoonLight || test_scene == eTestScene::Standard_HDRLight) {
            groups.emplace_back(floor_mat, floor_mat, env_groups[0], env_groups[1]);
            groups.emplace_back(dark_grey_mat, dark_grey_mat, env_groups[2], env_groups[3]);
            groups.emplace_back(mid_grey_mat, mid_grey_mat, env_groups[4], env_groups[5]);
        } else {
            groups.emplace_back(floor_mat, floor_mat, env_groups[0], env_groups[1]);
            groups.emplace_back(walls_mat, walls_mat, env_groups[2], env_groups[3]);
            groups.emplace_back(dark_grey_mat, dark_grey_mat, env_groups[4], env_groups[5]);
            groups.emplace_back(light_grey_mat, light_grey_mat, env_groups[6], env_groups[7]);
            groups.emplace_back(mid_grey_mat, mid_grey_mat, env_groups[8], env_groups[9]);
        }
        env_mesh_desc.groups = groups;

        env_mesh = scene.AddMesh(env_mesh_desc);

        // Add mesh one more time to test compaction later
        meshes_to_delete.push_back(env_mesh);
        env_mesh = scene.AddMesh(env_mesh_desc);
    }

    Ray::MeshHandle square_light_mesh;
    {
        std::vector<float> square_light_attrs;
        std::vector<uint32_t> square_light_indices, square_light_groups;
        std::tie(square_light_attrs, square_light_indices, square_light_groups) =
            LoadBIN("test_data/meshes/mat_test/square_light.bin");

        Ray::mesh_desc_t square_light_mesh_desc;
        square_light_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        square_light_mesh_desc.vtx_positions = {square_light_attrs, 0, 8};
        square_light_mesh_desc.vtx_normals = {square_light_attrs, 3, 8};
        square_light_mesh_desc.vtx_uvs = {square_light_attrs, 6, 8};
        square_light_mesh_desc.vtx_indices = square_light_indices;

        const Ray::mat_group_desc_t groups[] = {{square_light_mat, square_light_groups[0], square_light_groups[1]},
                                                {dark_grey_mat, square_light_groups[2], square_light_groups[3]}};
        square_light_mesh_desc.groups = groups;

        square_light_mesh = scene.AddMesh(square_light_mesh_desc);

        // Add mesh one more time to test compaction later
        meshes_to_delete.push_back(square_light_mesh);
        square_light_mesh = scene.AddMesh(square_light_mesh_desc);
    }

    Ray::MeshHandle disc_light_mesh;
    {
        std::vector<float> disc_light_attrs;
        std::vector<uint32_t> disc_light_indices, disc_light_groups;
        std::tie(disc_light_attrs, disc_light_indices, disc_light_groups) =
            LoadBIN("test_data/meshes/mat_test/disc_light.bin");

        Ray::mesh_desc_t disc_light_mesh_desc;
        disc_light_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        disc_light_mesh_desc.vtx_positions = {disc_light_attrs, 0, 8};
        disc_light_mesh_desc.vtx_normals = {disc_light_attrs, 3, 8};
        disc_light_mesh_desc.vtx_uvs = {disc_light_attrs, 6, 8};
        disc_light_mesh_desc.vtx_indices = disc_light_indices;

        const Ray::mat_group_desc_t groups[] = {{disc_light_mat, disc_light_groups[0], disc_light_groups[1]},
                                                {dark_grey_mat, disc_light_groups[2], disc_light_groups[3]}};
        disc_light_mesh_desc.groups = groups;

        disc_light_mesh = scene.AddMesh(disc_light_mesh_desc);
    }

    Ray::MeshHandle glassball_mesh;
    {
        std::vector<float> glassball_attrs;
        std::vector<uint32_t> glassball_indices, glassball_groups;
        std::tie(glassball_attrs, glassball_indices, glassball_groups) =
            LoadBIN("test_data/meshes/mat_test/glassball.bin");

        Ray::mesh_desc_t glassball_mesh_desc;
        glassball_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        glassball_mesh_desc.vtx_positions = {glassball_attrs, 0, 8};
        glassball_mesh_desc.vtx_normals = {glassball_attrs, 3, 8};
        glassball_mesh_desc.vtx_uvs = {glassball_attrs, 6, 8};
        glassball_mesh_desc.vtx_indices = glassball_indices;

        const Ray::mat_group_desc_t groups[] = {{glassball_mat0, glassball_groups[0], glassball_groups[1]},
                                                {glassball_mat1, glassball_groups[2], glassball_groups[3]}};
        glassball_mesh_desc.groups = groups;

        glassball_mesh = scene.AddMesh(glassball_mesh_desc);
    }

    Ray::MeshHandle box_mesh, box2_mesh, box3_mesh;
    {
        std::vector<float> box_attrs;
        std::vector<uint32_t> box_indices, box_groups;
        std::tie(box_attrs, box_indices, box_groups) = LoadBIN("test_data/meshes/mat_test/box.bin");

        Ray::mesh_desc_t box_mesh_desc;
        box_mesh_desc.prim_type = Ray::ePrimType::TriangleList;
        box_mesh_desc.vtx_positions = {box_attrs, 0, 8};
        box_mesh_desc.vtx_normals = {box_attrs, 3, 8};
        box_mesh_desc.vtx_uvs = {box_attrs, 6, 8};
        box_mesh_desc.vtx_indices = box_indices;

        const Ray::mat_group_desc_t groups[] = {{glossy_red, box_groups[0], box_groups[1]}};
        box_mesh_desc.groups = groups;
        box_mesh = scene.AddMesh(box_mesh_desc);

        const Ray::mat_group_desc_t groups2[] = {{refr_mat_flags, box_groups[0], box_groups[1]}};
        box_mesh_desc.groups = groups2;
        box2_mesh = scene.AddMesh(box_mesh_desc);

        const Ray::mat_group_desc_t groups3[] = {{glossy_green, box_groups[0], box_groups[1]}};
        box_mesh_desc.groups = groups3;
        box3_mesh = scene.AddMesh(box_mesh_desc);
    }

    static const float identity[16] = {1.0f, 0.0f, 0.0f, 0.0f, // NOLINT
                                       0.0f, 1.0f, 0.0f, 0.0f, // NOLINT
                                       0.0f, 0.0f, 1.0f, 0.0f, // NOLINT
                                       0.0f, 0.0f, 0.0f, 1.0f};

    static const float model_xform[16] = {0.707106769f,  0.0f,   0.707106769f, 0.0f, // NOLINT
                                          0.0f,          1.0f,   0.0f,         0.0f, // NOLINT
                                          -0.707106769f, 0.0f,   0.707106769f, 0.0f, // NOLINT
                                          0.0f,          0.062f, 0.0f,         1.0f};

    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;
    env_desc.back_col[0] = env_desc.back_col[1] = env_desc.back_col[2] = 0.0f;

    if (test_scene == eTestScene::Refraction_Plane) {
        scene.AddMeshInstance(model_mesh, identity);
    } else if (test_scene == eTestScene::Standard_GlassBall0 || test_scene == eTestScene::Standard_GlassBall1) {
        static const float glassball_xform[16] = {1.0f, 0.0f,  0.0f, 0.0f, // NOLINT
                                                  0.0f, 1.0f,  0.0f, 0.0f, // NOLINT
                                                  0.0f, 0.0f,  1.0f, 0.0f, // NOLINT
                                                  0.0f, 0.05f, 0.0f, 1.0f};

        scene.AddMeshInstance(glassball_mesh, glassball_xform);
    } else if (test_scene == eTestScene::Ray_Flags) {
        float box_xform[16] = {0.01f,  0.0f,  0.0f,  0.0f, // NOLINT
                               0.0f,   0.05f, 0.0f,  0.0f, // NOLINT
                               0.0f,   0.0f,  0.01f, 0.0f, // NOLINT
                               -0.05f, 0.05f, 0.0f,  1.0f};
        Ray::mesh_instance_desc_t mi;
        mi.xform = box_xform;
        mi.mesh = box_mesh;

        mi.shadow_visibility = false;
        scene.AddMeshInstance(mi);
        mi.shadow_visibility = true;

        box_xform[12] = 0.0f;
        box_xform[13] = 0.051f;
        mi.specular_visibility = false;
        mi.mesh = box2_mesh;
        scene.AddMeshInstance(mi);
        mi.mesh = box_mesh;
        mi.specular_visibility = true;

        box_xform[12] = 0.05f;
        box_xform[13] = 0.05f;
        mi.diffuse_visibility = false;
        scene.AddMeshInstance(mi);
        mi.diffuse_visibility = true;

        // second row
        mi.mesh = box3_mesh;

        box_xform[12] = -0.05f;
        box_xform[14] = -0.05f;
        mi.camera_visibility = false;
        scene.AddMeshInstance(mi);
        mi.camera_visibility = true;

        box_xform[12] = 0.0f;
        mi.refraction_visibility = false;
        scene.AddMeshInstance(mi);
        mi.refraction_visibility = true;

        box_xform[12] = 0.05f;
        scene.AddMeshInstance(mi);
    } else if (test_scene == eTestScene::Two_Sided) {
        const float two_sided_xform[16] = {1.0f, 0.0f,  0.0f, 0.0f, // NOLINT
                                           0.0f, 1.0f,  0.0f, 0.0f, // NOLINT
                                           0.0f, 0.0f,  1.0f, 0.0f, // NOLINT
                                           0.0f, 0.04f, 0.0f, 1.0f};

        scene.AddMeshInstance(two_sided_mesh, two_sided_xform);
        scene.AddMeshInstance(base_mesh, identity);
        scene.AddMeshInstance(text_mesh, identity);
    } else {
        scene.AddMeshInstance(model_mesh, model_xform);
        scene.AddMeshInstance(base_mesh, identity);
        scene.AddMeshInstance(core_mesh, identity);
        scene.AddMeshInstance(subsurf_bar_mesh, identity);
        scene.AddMeshInstance(text_mesh, identity);
    }
    scene.AddMeshInstance(env_mesh, identity);
    if (test_scene == eTestScene::Standard_MeshLights || test_scene == eTestScene::Refraction_Plane) {
        //
        // Use mesh lights
        //
        if (test_scene != eTestScene::Refraction_Plane) {
            scene.AddMeshInstance(square_light_mesh, identity);
        }
        scene.AddMeshInstance(disc_light_mesh, identity);
    } else if (test_scene == eTestScene::Standard || test_scene == eTestScene::Standard_SphereLight ||
               test_scene == eTestScene::Standard_SpotLight || test_scene == eTestScene::Standard_DOF0 ||
               test_scene == eTestScene::Standard_DOF1 || test_scene == eTestScene::Standard_GlassBall0 ||
               test_scene == eTestScene::Standard_GlassBall1 || test_scene == eTestScene::Standard_Clipped ||
               test_scene == eTestScene::Two_Sided) {
        //
        // Use explicit lights sources
        //
        if (test_scene == eTestScene::Standard || test_scene == eTestScene::Standard_DOF0 ||
            test_scene == eTestScene::Standard_DOF1 || test_scene == eTestScene::Standard_GlassBall0 ||
            test_scene == eTestScene::Standard_GlassBall1 || test_scene == eTestScene::Standard_Clipped ||
            test_scene == eTestScene::Two_Sided) {
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
        } else if (test_scene == eTestScene::Standard_SphereLight) {
            { // sphere light
                Ray::sphere_light_desc_t new_light;

                new_light.color[0] = 7.95775f;
                new_light.color[1] = 7.95775f;
                new_light.color[2] = 7.95775f;

                new_light.position[0] = -0.436484f;
                new_light.position[1] = 0.187179f;
                new_light.position[2] = 0.204932f;

                new_light.radius = 0.05f;

                new_light.visible = true;

                scene.AddLight(new_light);
            }
            { // line light
                static const float xform[16] = {0.813511789f,  -0.536388099f, -0.224691749f, 0.00000000f,
                                                0.538244009f,  0.548162937f,  0.640164733f,  0.00000000f,
                                                -0.220209062f, -0.641720533f, 0.734644651f,  0.00000000f,
                                                0.0f,          0.461762f,     0.0f,          1.00000000f};

                Ray::line_light_desc_t new_light;

                new_light.color[0] = 80.0f;
                new_light.color[1] = 80.0f;
                new_light.color[2] = 80.0f;

                new_light.radius = 0.005f;
                new_light.height = 0.2592f;

                new_light.visible = true;
                new_light.sky_portal = false;

                scene.AddLight(new_light, xform);
            }
        } else if (test_scene == eTestScene::Standard_SpotLight) {
            { // spot light
                Ray::spot_light_desc_t new_light;

                new_light.color[0] = 10.1321182f;
                new_light.color[1] = 10.1321182f;
                new_light.color[2] = 10.1321182f;

                new_light.position[0] = -0.436484f;
                new_light.position[1] = 0.187179f;
                new_light.position[2] = 0.204932f;

                new_light.direction[0] = 0.699538708f;
                new_light.direction[1] = -0.130918920f;
                new_light.direction[2] = -0.702499688f;

                new_light.radius = 0.05f;
                new_light.spot_size = 45.0f;
                new_light.spot_blend = 0.15f;

                new_light.visible = true;

                scene.AddLight(new_light);
            }
        }
    } else if (test_scene == eTestScene::Standard_DirLight) {
        Ray::directional_light_desc_t sun_desc;

        sun_desc.direction[0] = 0.541675210f;
        sun_desc.direction[1] = -0.541675210f;
        sun_desc.direction[2] = -0.642787635f;

        sun_desc.color[0] = sun_desc.color[1] = sun_desc.color[2] = 12.0f;
        sun_desc.angle = 10.0f;

        scene.AddLight(sun_desc);
    } else if (test_scene == eTestScene::Standard_SunLight) {
        Ray::directional_light_desc_t sun_desc;

        sun_desc.direction[0] = 0.454519480f;
        sun_desc.direction[1] = -0.454519480f;
        sun_desc.direction[2] = -0.766044438f;

        sun_desc.color[0] = 144809.866891f;
        sun_desc.color[1] = 129443.618266f;
        sun_desc.color[2] = 127098.894121f;

        sun_desc.angle = 4.0f;

        scene.AddLight(sun_desc);
    } else if (test_scene == eTestScene::Ray_Flags) {
        Ray::sphere_light_desc_t new_light;

        new_light.color[0] = 0.0253302939f;
        new_light.color[1] = 0.0253302939f;
        new_light.color[2] = 0.0253302939f;

        new_light.position[0] = -0.05f;
        new_light.position[1] = 0.2f;
        new_light.position[2] = 0.075f;

        new_light.radius = 0.0f;

        new_light.visible = true;

        scene.AddLight(new_light);
    } else if (test_scene == eTestScene::Standard_MoonLight || test_scene == eTestScene::Standard_NoLight) {
        // nothing
    }

    if (test_scene == eTestScene::Standard_HDRLight || test_scene == eTestScene::Standard_Clipped) {
        int img_w, img_h;
        auto img_data = LoadHDR("test_data/textures/studio_small_03_2k.hdr", img_w, img_h);
        require(!img_data.empty());

        Ray::tex_desc_t tex_desc;
        tex_desc.format = Ray::eTextureFormat::RGBA8888;
        tex_desc.data = img_data;
        tex_desc.w = img_w;
        tex_desc.h = img_h;
        tex_desc.generate_mipmaps = false;
        tex_desc.is_srgb = false;
        tex_desc.force_no_compression = true;

        env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.25f;
        env_desc.back_col[0] = env_desc.back_col[1] = env_desc.back_col[2] = 0.25f;

        env_desc.env_map = env_desc.back_map = scene.AddTexture(tex_desc);
        if (test_scene == eTestScene::Standard_HDRLight) {
            env_desc.env_map_rotation = env_desc.back_map_rotation = 2.35619449019f;
        }
    } else if (test_scene == eTestScene::Standard_SunLight) {
        env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 1.0f;
        env_desc.back_col[0] = env_desc.back_col[1] = env_desc.back_col[2] = 1.0f;

        env_desc.env_map = env_desc.back_map = Ray::PhysicalSkyTexture;
    } else if (test_scene == eTestScene::Standard_MoonLight) {
        env_desc.atmosphere.clouds_density = 0.4f;

        env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 1.0f;
        env_desc.back_col[0] = env_desc.back_col[1] = env_desc.back_col[2] = 1.0f;

        env_desc.env_map = env_desc.back_map = Ray::PhysicalSkyTexture;
    }

    scene.SetEnvironment(env_desc);

    for (const Ray::MeshHandle mesh : meshes_to_delete) {
        scene.RemoveMesh(mesh);
    }

    using namespace std::placeholders;
    scene.Finalize(std::bind(&ThreadPool::ParallelFor<Ray::ParallelForFunction>, &threads, _1, _2, _3));
}

template void setup_test_scene(ThreadPool &threads, Ray::SceneBase &scene, int min_samples, float variance_threshold,
                               const Ray::shading_node_desc_t &main_mat_desc, const char *textures[],
                               eTestScene test_scene);
template void setup_test_scene(ThreadPool &threads, Ray::SceneBase &scene, int min_samples, float variance_threshold,
                               const Ray::principled_mat_desc_t &main_mat_desc, const char *textures[],
                               eTestScene test_scene);

void schedule_render_jobs(ThreadPool &threads, Ray::RendererBase &renderer, const Ray::SceneBase *scene,
                          const Ray::settings_t &settings, const int max_samples, const eDenoiseMethod denoise,
                          const bool partial, const char *log_str) {
    const auto rt = renderer.type();
    const auto sz = renderer.size();

    static const int BucketSize = 16;
    static const int SamplePortion = 16;

    if (Ray::RendererSupportsMultithreading(rt)) {
        bool skip_tile = false;
        std::vector<Ray::RegionContext> region_contexts;
        for (int y = 0; y < sz.second; y += BucketSize) {
            skip_tile = !skip_tile;
            for (int x = 0; x < sz.first; x += BucketSize) {
                skip_tile = !skip_tile;
                if (partial && skip_tile) {
                    continue;
                }

                const auto rect =
                    Ray::rect_t{x, y, std::min(sz.first - x, BucketSize), std::min(sz.second - y, BucketSize)};
                region_contexts.emplace_back(rect);
            }
        }

        auto render_job = [&](const int j, const int portion) {
#if defined(_WIN32)
            if (g_catch_flt_exceptions) {
                unsigned old_value;
                _controlfp_s(&old_value, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);
            }
#endif
            for (int i = 0; i < portion; ++i) {
                renderer.RenderScene(scene, region_contexts[j]);
            }
        };

        auto denoise_job_nlm = [&](const int j) {
#if defined(_WIN32)
            if (g_catch_flt_exceptions) {
                unsigned old_value;
                //_controlfp_s(&old_value, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);
                _controlfp_s(&old_value, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW | _EM_INVALID, _MCW_EM);
            }
#endif
            renderer.DenoiseImage(region_contexts[j]);
        };

        auto denoise_job_unet = [&](const int pass, const int j) {
#if defined(_WIN32)
            if (g_catch_flt_exceptions) {
                unsigned old_value;
                _controlfp_s(&old_value, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);
            }
#endif
            renderer.DenoiseImage(pass, region_contexts[j]);
        };

        for (int i = 0; i < max_samples; i += std::min(SamplePortion, max_samples - i)) {
            std::vector<std::future<void>> job_res;
            for (int j = 0; j < int(region_contexts.size()); ++j) {
                job_res.push_back(threads.Enqueue(render_job, j, std::min(SamplePortion, max_samples - i)));
            }
            for (auto &res : job_res) {
                res.wait();
            }
            job_res.clear();

            if (i + std::min(SamplePortion, max_samples - i) == max_samples && denoise != eDenoiseMethod::None) {
                if (denoise == eDenoiseMethod::NLM) {
                    for (int j = 0; j < int(region_contexts.size()); ++j) {
                        job_res.push_back(threads.Enqueue(denoise_job_nlm, j));
                    }
                    for (auto &res : job_res) {
                        res.wait();
                    }
                    job_res.clear();
                } else if (denoise == eDenoiseMethod::UNet) {
                    Ray::unet_filter_properties_t props;
                    renderer.InitUNetFilter(true, props);

                    for (int pass = 0; pass < props.pass_count; ++pass) {
                        for (int j = 0; j < int(region_contexts.size()); ++j) {
                            job_res.push_back(threads.Enqueue(denoise_job_unet, pass, j));
                        }
                        for (auto &res : job_res) {
                            res.wait();
                        }
                        job_res.clear();
                    }
                }
            }

            // report progress percentage
            if (!g_minimal_output) {
                const float prog = 100.0f * float(i + std::min(SamplePortion, max_samples - i)) / float(max_samples);
                std::lock_guard<std::mutex> _(g_stdout_mtx);
                printf("\r%s (%6s, %s): %.1f%% ", log_str, Ray::RendererTypeName(rt),
                       settings.use_hwrt ? "HWRT" : "SWRT", prog);
                fflush(stdout);
            }
        }
    } else {
        std::vector<Ray::RegionContext> region_contexts;
        if (partial) {
            bool skip_tile = false;
            for (int y = 0; y < sz.second; y += BucketSize) {
                skip_tile = !skip_tile;
                for (int x = 0; x < sz.first; x += BucketSize) {
                    skip_tile = !skip_tile;
                    if (partial && skip_tile) {
                        continue;
                    }

                    const auto rect =
                        Ray::rect_t{x, y, std::min(sz.first - x, BucketSize), std::min(sz.second - y, BucketSize)};
                    region_contexts.emplace_back(rect);
                }
            }
        } else {
            region_contexts.emplace_back(Ray::rect_t{0, 0, sz.first, sz.second});
        }

        for (int i = 0; i < max_samples; ++i) {
            for (auto &region : region_contexts) {
                renderer.RenderScene(scene, region);
            }

            if (((i % SamplePortion) == 0 || i == max_samples - 1) && !g_minimal_output) {
                // report progress percentage
                const float prog = 100.0f * float(i + 1) / float(max_samples);
                std::lock_guard<std::mutex> _(g_stdout_mtx);
                printf("\r%s (%6s, %s): %.1f%% ", log_str, Ray::RendererTypeName(rt),
                       settings.use_hwrt ? "HWRT" : "SWRT", prog);
                fflush(stdout);
            }
        }
        if (denoise == eDenoiseMethod::NLM) {
            for (auto &region : region_contexts) {
                renderer.DenoiseImage(region);
            }
        } else if (denoise == eDenoiseMethod::UNet) {
            Ray::unet_filter_properties_t props;
            renderer.InitUNetFilter(true, props);

            for (int pass = 0; pass < props.pass_count; ++pass) {
                for (auto &region : region_contexts) {
                    renderer.DenoiseImage(pass, region);
                }
            }
        }
    }
}
