#pragma once

#include "../SceneBase.h"
#include "SparseStorage.h"
#include "SparseStorageVK.h"
#include "VectorVK.h"
#include "Vk/AccStructure.h"
#include "Vk/DescriptorPool.h"
#include "Vk/Texture.h"
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

        explicit BindlessTexData(Context *ctx) : descr_pool(ctx) {}
    };

    BindlessTexData bindless_tex_data_;

    SmallVector<VkDescriptorSet, 1024> textures_descr_sets_;

    TextureAtlas tex_atlases_[7];

    SparseStorage<light_t> lights_;
    Vector<uint32_t> li_indices_;
    Vector<uint32_t> visible_lights_;
    Vector<uint32_t> blocker_lights_;

    environment_t env_;
    LightHandle env_map_light_ = InvalidLightHandle;
    struct {
        int res = -1;
        SmallVector<aligned_vector<simd_fvec4>, 16> mips;
        Texture2D tex;
    } env_map_qtree_;

    uint32_t macro_nodes_start_ = 0xffffffff, macro_nodes_count_ = 0;

    Buffer rt_blas_buf_, rt_geo_data_buf_, rt_instance_buf_, rt_tlas_buf_;

    struct MeshBlas {
        AccStructure acc;
        uint32_t geo_index, geo_count;
    };
    std::vector<MeshBlas> rt_mesh_blases_;
    AccStructure rt_tlas_;

    MaterialHandle AddMaterial_nolock(const shading_node_desc_t &m);
    void SetMeshInstanceTransform_nolock(MeshInstanceHandle mi_handle, const float *xform);

    void RemoveLight_nolock(LightHandle i);
    void RemoveNodes_nolock(uint32_t node_index, uint32_t node_count);
    void RebuildTLAS_nolock();
    // void RebuildLightBVH();

    void PrepareEnvMapQTree_nolock();
    void GenerateTextureMips_nolock();
    void PrepareBindlessTextures_nolock();
    void RebuildHWAccStructures_nolock();

    TextureHandle AddAtlasTexture_nolock(const tex_desc_t &t);
    TextureHandle AddBindlessTexture_nolock(const tex_desc_t &t);

    template <typename T, int N>
    static void WriteTextureMips(const color_t<T, N> data[], const int _res[2], int mip_count, bool compress,
                                 uint8_t out_data[], uint32_t out_size[16]);

  public:
    Scene(Context *ctx, bool use_hwrt, bool use_bindless, bool use_tex_compression);
    ~Scene() override;

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    TextureHandle AddTexture(const tex_desc_t &t) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        if (use_bindless_) {
            return AddBindlessTexture_nolock(t);
        } else {
            return AddAtlasTexture_nolock(t);
        }
    }
    void RemoveTexture(const TextureHandle t) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        if (use_bindless_) {
            bindless_textures_.erase(t._index >> 24);
        } else {
            atlas_textures_.erase(t._index);
        }
    }

    MaterialHandle AddMaterial(const shading_node_desc_t &m) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        return AddMaterial_nolock(m);
    }
    MaterialHandle AddMaterial(const principled_mat_desc_t &m) override;
    void RemoveMaterial(const MaterialHandle m) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        materials_.erase(m._index);
    }

    MeshHandle AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(MeshHandle) override;

    LightHandle AddLight(const directional_light_desc_t &l) override;
    LightHandle AddLight(const sphere_light_desc_t &l) override;
    LightHandle AddLight(const spot_light_desc_t &l) override;
    LightHandle AddLight(const rect_light_desc_t &l, const float *xform) override;
    LightHandle AddLight(const disk_light_desc_t &l, const float *xform) override;
    LightHandle AddLight(const line_light_desc_t &l, const float *xform) override;
    void RemoveLight(const LightHandle i) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        RemoveLight_nolock(i);
    }

    MeshInstanceHandle AddMeshInstance(MeshHandle mesh, const float *xform) override;
    void SetMeshInstanceTransform(MeshInstanceHandle mi_handle, const float* xform) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        SetMeshInstanceTransform_nolock(mi_handle, xform);
    }
    void RemoveMeshInstance(MeshInstanceHandle) override;

    void Finalize() override;

    uint32_t triangle_count() const override { return 0; }
    uint32_t node_count() const override { return 0; }
};
} // namespace Vk
} // namespace Ray
