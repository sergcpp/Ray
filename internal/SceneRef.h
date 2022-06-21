#pragma once

#include <vector>

#include "../SceneBase.h"
#include "BVHSplit.h"
#include "CoreRef.h"
#include "SparseStorage.h"
#include "TextureAtlasRef.h"

namespace Ray {
namespace ref2 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Sse2 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Avx {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Avx2 {
template <int DimX, int DimY> class RendererSIMD;
}

namespace Neon {
template <int DimX, int DimY> class RendererSIMD;
}

namespace Ref {
class Renderer;

class Scene : public SceneBase {
  protected:
    friend class Ref::Renderer;
    template <int DimX, int DimY> friend class Sse2::RendererSIMD;
    template <int DimX, int DimY> friend class Avx::RendererSIMD;
    template <int DimX, int DimY> friend class Avx2::RendererSIMD;
    template <int DimX, int DimY> friend class Neon::RendererSIMD;

    ILog *log_;

    bool use_wide_bvh_;
    std::vector<bvh_node_t> nodes_;
    aligned_vector<mbvh_node_t> mnodes_;
    std::vector<tri_accel_t> tris_;
    std::vector<tri_accel2_t> tris2_;
    std::vector<uint32_t> tri_indices_;
    std::vector<tri_mat_data_t> tri_materials_;
    SparseStorage<transform_t> transforms_;
    SparseStorage<mesh_t> meshes_;
    SparseStorage<mesh_instance_t> mesh_instances_;
    std::vector<uint32_t> mi_indices_;
    std::vector<vertex_t> vertices_;
    std::vector<uint32_t> vtx_indices_;

    SparseStorage<material_t> materials_;
    SparseStorage<texture_t> textures_;

    TextureAtlasRGBA tex_atlas_rgba_;
    TextureAtlasRGB tex_atlas_rgb_;
    TextureAtlasRG tex_atlas_rg_;
    TextureAtlasR tex_atlas_r_;

    std::vector<light_t> lights_;
    std::vector<uint32_t> li_indices_;

    SparseStorage<light2_t> lights2_;
    std::vector<uint32_t> visible_lights_; // compacted list of all visible lights

    environment_t env_;

    uint32_t macro_nodes_root_ = 0xffffffff, macro_nodes_count_ = 0;
    uint32_t light_nodes_root_ = 0xffffffff, light_nodes_count_ = 0;

    void RemoveTris(uint32_t tris_index, uint32_t tris_count);
    void RemoveNodes(uint32_t node_index, uint32_t node_count);
    void RebuildTLAS();
    void RebuildLightBVH();

    void GenerateTextureMipmaps();

  public:
    Scene(ILog *log, bool use_wide_bvh);
    ~Scene() override;

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    uint32_t AddTexture(const tex_desc_t &t) override;
    void RemoveTexture(const uint32_t i) override { textures_.erase(i); }

    uint32_t AddMaterial(const shading_node_desc_t &m) override;
    uint32_t AddMaterial(const principled_mat_desc_t &m) override;
    void RemoveMaterial(const uint32_t i) override { materials_.erase(i); }

    uint32_t AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(uint32_t) override;

    uint32_t AddLight(const directional_light_desc_t &l) override;
    uint32_t AddLight(const sphere_light_desc_t &l) override;
    uint32_t AddLight(const rect_light_desc_t &l, const float *xform) override;
    uint32_t AddLight(const disk_light_desc_t &l, const float *xform) override;
    void RemoveLight(uint32_t i) override;

    uint32_t AddMeshInstance(uint32_t m_index, const float *xform) override;
    void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) override;
    void RemoveMeshInstance(uint32_t) override;

    void Finalize() override;

    uint32_t triangle_count() override { return uint32_t(tris_.size()); }
    uint32_t node_count() override { return uint32_t(use_wide_bvh_ ? mnodes_.size() : nodes_.size()); }
};
} // namespace Ref
} // namespace Ray