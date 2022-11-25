#pragma once

#include "../SceneBase.h"
#include "SparseStorage.h"
#include "SparseStorageVK.h"
#include "VectorVK.h"
#include "Vk/AccStructure.h"
#include "Vk/TextureAtlas.h"

namespace Ray {
namespace Vk {
class Context;
class Renderer;

class Scene : public SceneBase {
  protected:
    friend class Vk::Renderer;

    Context *ctx_;
    bool use_hwrt_ = false, use_bindless_ = false, use_tex_compression_ = false;

    Vector<bvh_node_t> nodes_;
    Vector<tri_accel_t> tris_;
    Vector<uint32_t> tri_indices_;
    Vector<tri_mat_data_t> tri_materials_;
    std::vector<tri_mat_data_t> tri_materials_cpu_;
    SparseStorage<transform_t> transforms_;
    SparseStorage<mesh_t> meshes_;
    SparseStorage<mesh_instance_t> mesh_instances_;
    Vector<uint32_t> mi_indices_;
    Vector<vertex_t> vertices_;
    Vector<uint32_t> vtx_indices_;

    SparseStorage<material_t> materials_;
    SparseStorage<atlas_texture_t> atlas_textures_;
    Ref::SparseStorage<Texture2D> bindless_textures_;

    struct BindlessTexData {
        DescrPool descr_pool;
        VkDescriptorSetLayout descr_layout = {};
        VkDescriptorSet descr_set = {};

        BindlessTexData(Context *ctx) : descr_pool(ctx) {}
    };

    BindlessTexData bindless_tex_data_;

    SmallVector<VkDescriptorSet, 1024> textures_descr_sets_;

    TextureAtlas tex_atlases_[7];

    SparseStorage<light_t> lights_;
    Vector<uint32_t> li_indices_;
    Vector<uint32_t> visible_lights_;

    environment_t env_;

    uint32_t macro_nodes_start_ = 0xffffffff, macro_nodes_count_ = 0;

    Buffer rt_blas_buf_, rt_geo_data_buf_, rt_instance_buf_, rt_tlas_buf_;

    struct MeshBlas {
        AccStructure acc;
        uint32_t geo_index, geo_count;
    };
    std::vector<MeshBlas> rt_mesh_blases_;
    AccStructure rt_tlas_;

    void RemoveNodes(uint32_t node_index, uint32_t node_count);
    void RebuildTLAS();
    // void RebuildLightBVH();

    void GenerateTextureMips();
    void PrepareBindlessTextures();
    void RebuildHWAccStructures();

    uint32_t AddAtlasTexture(const tex_desc_t &t);
    uint32_t AddBindlessTexture(const tex_desc_t &t);

    template <typename T, int N>
    void WriteTextureMips(const color_t<T, N> data[], const int _res[2], int mip_count, bool compress,
                          uint8_t out_data[], uint32_t out_size[16]);

  public:
    Scene(Context *ctx, bool use_hwrt, bool use_bindless, bool use_tex_compression);
    ~Scene() override;

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    uint32_t AddTexture(const tex_desc_t &t) override {
        if (use_bindless_) {
            return AddBindlessTexture(t);
        } else {
            return AddAtlasTexture(t);
        }
    }
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
    uint32_t AddLight(const line_light_desc_t &l, const float *xform) override;
    void RemoveLight(uint32_t i) override;

    uint32_t AddMeshInstance(uint32_t m_index, const float *xform) override;
    void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) override;
    void RemoveMeshInstance(uint32_t) override;

    void Finalize() override {
        GenerateTextureMips();
        PrepareBindlessTextures();
        RebuildHWAccStructures();
    }

    uint32_t triangle_count() override { return (uint32_t)0; }
    uint32_t node_count() override { return (uint32_t)0; }
};
} // namespace Vk
} // namespace Ray
