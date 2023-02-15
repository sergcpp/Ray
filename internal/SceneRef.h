#pragma once

#include <vector>

#include "../SceneBase.h"
#include "CoreRef.h"
#include "SmallVector.h"
#include "SparseStorage.h"
#include "TextureStorageRef.h"

namespace Ray {
namespace Sse2 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Sse41 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Avx {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Avx2 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Avx512 {
template <int DimX, int DimY> class RendererSIMD;
}
namespace Neon {
template <int DimX, int DimY> class RendererSIMD;
}

class ILog;

namespace Ref {
class Renderer;

class Scene : public SceneBase {
  protected:
    friend class Ref::Renderer;
    template <int DimX, int DimY> friend class Sse2::RendererSIMD;
    template <int DimX, int DimY> friend class Sse41::RendererSIMD;
    template <int DimX, int DimY> friend class Avx::RendererSIMD;
    template <int DimX, int DimY> friend class Avx2::RendererSIMD;
    template <int DimX, int DimY> friend class Avx512::RendererSIMD;
    template <int DimX, int DimY> friend class Neon::RendererSIMD;

    ILog *log_;

    bool use_wide_bvh_;
    std::vector<bvh_node_t> nodes_;
    aligned_vector<mbvh_node_t> mnodes_;
    std::vector<tri_accel_t> tris_;
    std::vector<uint32_t> tri_indices_;
    aligned_vector<mtri_accel_t> mtris_;
    std::vector<tri_mat_data_t> tri_materials_;
    SparseStorage<transform_t> transforms_;
    SparseStorage<mesh_t> meshes_;
    SparseStorage<mesh_instance_t> mesh_instances_;
    std::vector<uint32_t> mi_indices_;
    std::vector<vertex_t> vertices_;
    std::vector<uint32_t> vtx_indices_;

    SparseStorage<material_t> materials_;

    TexStorageRGBA tex_storage_rgba_;
    TexStorageRGB tex_storage_rgb_;
    TexStorageRG tex_storage_rg_;
    TexStorageR tex_storage_r_;

    TexStorageBase *tex_storages_[4] = {&tex_storage_rgba_, &tex_storage_rgb_, &tex_storage_rg_, &tex_storage_r_};

    SparseStorage<light_t> lights_;
    std::vector<uint32_t> li_indices_;     // compacted list of all lights
    std::vector<uint32_t> visible_lights_; // compacted list of all visible lights
    std::vector<uint32_t> blocker_lights_; // compacted list of all light blocker lights

    environment_t env_;
    Light env_map_light_ = InvalidLight;
    struct {
        int res = -1;
        SmallVector<aligned_vector<float, 16>, 16> mips;
    } env_map_qtree_;

    uint32_t macro_nodes_root_ = 0xffffffff, macro_nodes_count_ = 0;

    void RemoveTris(uint32_t tris_index, uint32_t tris_count);
    void RemoveNodes(uint32_t node_index, uint32_t node_count);
    void RebuildTLAS();

    void PrepareEnvMapQTree();

  public:
    Scene(ILog *log, bool use_wide_bvh);
    ~Scene() override;

    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

    Texture AddTexture(const tex_desc_t &t) override;
    void RemoveTexture(const Texture t) override { tex_storages_[t._index >> 24]->Free(t._index & 0x00ffffff); }

    Material AddMaterial(const shading_node_desc_t &m) override;
    Material AddMaterial(const principled_mat_desc_t &m) override;
    void RemoveMaterial(const Material m) override { materials_.erase(m._index); }

    Mesh AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(Mesh m) override;

    Light AddLight(const directional_light_desc_t &l) override;
    Light AddLight(const sphere_light_desc_t &l) override;
    Light AddLight(const spot_light_desc_t &l) override;
    Light AddLight(const rect_light_desc_t &l, const float *xform) override;
    Light AddLight(const disk_light_desc_t &l, const float *xform) override;
    Light AddLight(const line_light_desc_t &l, const float *xform) override;
    void RemoveLight(Light l) override;

    MeshInstance AddMeshInstance(Mesh mesh, const float *xform) override;
    void SetMeshInstanceTransform(MeshInstance mi, const float *xform) override;
    void RemoveMeshInstance(MeshInstance) override;

    void Finalize() override;

    uint32_t triangle_count() const override { return uint32_t(tris_.size()); }
    uint32_t node_count() const override { return use_wide_bvh_ ? uint32_t(mnodes_.size()) : uint32_t(nodes_.size()); }
};
} // namespace Ref
} // namespace Ray