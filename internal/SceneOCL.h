#pragma once

#include "TextureAtlasOCL.h"
#include "VectorOCL.h"
#include "../SceneBase.h"

namespace ray {
namespace ocl {
class Renderer;

class Scene : public SceneBase {
protected:
    friend class ocl::Renderer;

    const cl::Context &context_;
    const cl::CommandQueue &queue_;
    size_t max_img_buf_size_;

    ocl::Vector<bvh_node_t> nodes_;
    ocl::Vector<tri_accel_t> tris_;
    ocl::Vector<uint32_t> tri_indices_;
    ocl::Vector<transform_t> transforms_;
    ocl::Vector<mesh_t> meshes_;
    ocl::Vector<mesh_instance_t> mesh_instances_;
    ocl::Vector<uint32_t> mi_indices_;
    ocl::Vector<vertex_t> vertices_;
    ocl::Vector<uint32_t> vtx_indices_;

    ocl::Vector<material_t> materials_;
    ocl::Vector<texture_t> textures_;
    ocl::TextureAtlas texture_atlas_;

    ocl::environment_t env_;

    uint32_t macro_nodes_start_ = 0, macro_nodes_count_ = 0;

    uint32_t default_normals_texture_;

    void RemoveNodes(uint32_t node_index, uint32_t node_count);
    void RebuildMacroBVH();
public:
    Scene(const cl::Context &context, const cl::CommandQueue &queue, size_t max_img_buffer_size);

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    uint32_t AddTexture(const tex_desc_t &t) override;
    void RemoveTexture(uint32_t) override {}

    uint32_t AddMaterial(const mat_desc_t &m) override;
    void RemoveMaterial(uint32_t) override {}

    uint32_t AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(uint32_t) override;

    uint32_t AddMeshInstance(uint32_t m_index, const float *xform) override;
    void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) override;
    void RemoveMeshInstance(uint32_t) override;

    uint32_t triangle_count() override {
        return (uint32_t)tris_.size();
    }
    uint32_t node_count() override {
        return (uint32_t)nodes_.size();
    }
};
}
}