#pragma once

#include "../SceneBase.h"
//#include "TextureAtlasOCL.h"
#include "VectorVK.h"

namespace Ray {
namespace Vk {
class Context;
class Renderer;

class Scene : public SceneBase {
  protected:
    friend class Vk::Renderer;

    Context *ctx_;
    /*const cl::CommandQueue &queue_;
    size_t max_img_buf_size_;*/

    Vector<bvh_node_t> nodes_;
    Vector<tri_accel_t> tris_;
    Vector<uint32_t> tri_indices_;
    Vector<tri_mat_data_t> tri_materials_;
    std::vector<tri_mat_data_t> tri_materials_cpu_;
    Vector<transform_t> transforms_;
    Vector<mesh_t> meshes_;
    std::vector<mesh_t> meshes_cpu_;

    Vector<mesh_instance_t> mesh_instances_;
    Vector<uint32_t> mi_indices_;
    Vector<vertex_t> vertices_;
    Vector<uint32_t> vtx_indices_;

    Vector<material_t> materials_;
    std::vector<material_t> materials_cpu_;
    Vector<texture_t> textures_;
    std::vector<texture_t> textures_cpu_;

    TextureAtlas tex_atlases_[4];

    // TODO: use sparse storage!!
    Vector<light_t> lights_;
    Vector<uint32_t> li_indices_;
    Vector<uint32_t> visible_lights_;

    environment_t env_;

    uint32_t macro_nodes_start_ = 0xffffffff, macro_nodes_count_ = 0;
    /*uint32_t light_nodes_start_ = 0xffffffff, light_nodes_count_ = 0;

    uint32_t default_env_texture_;
    uint32_t default_normals_texture_;*/

    void RemoveNodes(uint32_t node_index, uint32_t node_count);
    void RebuildTLAS();
    //void RebuildLightBVH();

    void GenerateTextureMips();

  public:
    Scene(Context *ctx);

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    uint32_t AddTexture(const tex_desc_t &t) override;
    void RemoveTexture(uint32_t) override {}

    uint32_t AddMaterial(const shading_node_desc_t &m) override;
    uint32_t AddMaterial(const principled_mat_desc_t &m) override;
    void RemoveMaterial(uint32_t) override {}

    uint32_t AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(uint32_t) override;

    uint32_t AddLight(const directional_light_desc_t &l) override;
    uint32_t AddLight(const sphere_light_desc_t &l) override;
    uint32_t AddLight(const rect_light_desc_t &l, const float *xform) override;
    uint32_t AddLight(const disk_light_desc_t &l, const float *xform) override;
    void RemoveLight(uint32_t i) override {}

    uint32_t AddMeshInstance(uint32_t m_index, const float *xform) override;
    void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) override;
    void RemoveMeshInstance(uint32_t) override;

    void Finalize() override { GenerateTextureMips(); }

    uint32_t triangle_count() override { return (uint32_t)0; }
    uint32_t node_count() override { return (uint32_t)0; }
};
} // namespace Ocl
} // namespace Ray
