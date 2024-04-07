#pragma once

#include <vector>

#include "CoreRef.h"
#include "SceneCommon.h"
#include "SmallVector.h"
#include "SparseStorageCPU.h"
#include "TextureStorageCPU.h"

namespace Ray {
namespace Cpu {
template <typename SIMDPolicy> class Renderer;
}
namespace Ref {
class SIMDPolicy;
}
namespace Sse2 {
class SIMDPolicy;
}
namespace Sse41 {
class SIMDPolicy;
}
namespace Avx {
class SIMDPolicy;
}
namespace Avx2 {
class SIMDPolicy;
}
namespace Avx512 {
class SIMDPolicy;
}
namespace Neon {
class SIMDPolicy;
}

class ILog;

namespace Cpu {
class Scene : public SceneCommon {
  protected:
    friend class Cpu::Renderer<Ref::SIMDPolicy>;
    friend class Cpu::Renderer<Sse2::SIMDPolicy>;
    friend class Cpu::Renderer<Sse41::SIMDPolicy>;
    friend class Cpu::Renderer<Avx::SIMDPolicy>;
    friend class Cpu::Renderer<Avx2::SIMDPolicy>;
    friend class Cpu::Renderer<Avx512::SIMDPolicy>;
    friend class Cpu::Renderer<Neon::SIMDPolicy>;

    bool use_wide_bvh_, use_tex_compression_;

    SparseStorage<bvh_node_t> nodes_;
    SparseStorage<wbvh_node_t> wnodes_;
    SparseStorage<tri_accel_t> tris_;
    SparseStorage<uint32_t> tri_indices_;
    SparseStorage<mtri_accel_t> mtris_;
    SparseStorage<tri_mat_data_t> tri_materials_;
    SparseStorage<mesh_t> meshes_;
    SparseStorage<mesh_instance_t> mesh_instances_;
    std::vector<uint32_t> mi_indices_;
    SparseStorage<vertex_t> vertices_;
    SparseStorage<uint32_t> vtx_indices_;

    SparseStorage<material_t> materials_;

    TexStorageRGBA tex_storage_rgba_;
    TexStorageRGB tex_storage_rgb_;
    TexStorageRG tex_storage_rg_;
    TexStorageR tex_storage_r_;
    TexStorageBCn<3> tex_storage_bc1_;
    TexStorageBCn<4> tex_storage_bc3_;
    TexStorageBCn<1> tex_storage_bc4_;
    TexStorageBCn<2> tex_storage_bc5_;

    TexStorageBase *tex_storages_[8] = {&tex_storage_rgba_, &tex_storage_rgb_, &tex_storage_rg_,  &tex_storage_r_,
                                        &tex_storage_bc1_,  &tex_storage_bc3_, &tex_storage_bc4_, &tex_storage_bc5_};

    SparseStorage<light_t> lights_;
    std::vector<uint32_t> li_indices_; // compacted list of all lights
    uint32_t visible_lights_count_ = 0, blocker_lights_count_ = 0;

    std::vector<light_bvh_node_t> light_nodes_;
    aligned_vector<light_wbvh_node_t> light_wnodes_;

    LightHandle env_map_light_ = InvalidLightHandle;
    TextureHandle physical_sky_texture_ = InvalidTextureHandle;
    struct {
        int res = -1;
        float medium_lum = 0.0f;
        SmallVector<aligned_vector<float, 16>, 16> mips;
    } env_map_qtree_;

    mutable std::vector<uint64_t> spatial_cache_entries_;
    mutable aligned_vector<packed_cache_voxel_t, 16> spatial_cache_voxels_curr_, spatial_cache_voxels_prev_;
    mutable float spatial_cache_cam_pos_prev_[3] = {};

    uint32_t tlas_root_ = 0xffffffff, tlas_block_ = 0xffffffff;

    void RemoveMesh_nolock(MeshHandle m);
    void RemoveMeshInstance_nolock(MeshInstanceHandle i);
    void RebuildTLAS_nolock();
    void RebuildLightTree_nolock();

    void PrepareSkyEnvMap_nolock(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
    void PrepareEnvMapQTree_nolock();

    MaterialHandle AddMaterial_nolock(const shading_node_desc_t &m);
    void SetMeshInstanceTransform_nolock(MeshInstanceHandle mi, const float *xform);

  public:
    Scene(ILog *log, bool use_wide_bvh, bool use_tex_compression, bool use_spatial_cache);
    ~Scene() override;

    TextureHandle AddTexture(const tex_desc_t &t) override;
    void RemoveTexture(const TextureHandle t) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        tex_storages_[t._index >> 24]->Free(t._index & 0x00ffffff);
    }

    MaterialHandle AddMaterial(const shading_node_desc_t &m) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        return AddMaterial_nolock(m);
    }
    MaterialHandle AddMaterial(const principled_mat_desc_t &m) override;
    void RemoveMaterial(const MaterialHandle m) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        materials_.Erase(m._block);
    }

    MeshHandle AddMesh(const mesh_desc_t &m) override;
    void RemoveMesh(MeshHandle m) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        RemoveMesh_nolock(m);
    }

    LightHandle AddLight(const directional_light_desc_t &l) override;
    LightHandle AddLight(const sphere_light_desc_t &l) override;
    LightHandle AddLight(const spot_light_desc_t &l) override;
    LightHandle AddLight(const rect_light_desc_t &l, const float *xform) override;
    LightHandle AddLight(const disk_light_desc_t &l, const float *xform) override;
    LightHandle AddLight(const line_light_desc_t &l, const float *xform) override;
    void RemoveLight(LightHandle l) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        lights_.Erase(l._block);
    }

    MeshInstanceHandle AddMeshInstance(const mesh_instance_desc_t &mi) override;
    void SetMeshInstanceTransform(MeshInstanceHandle mi, const float *xform) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        SetMeshInstanceTransform_nolock(mi, xform);
    }
    void RemoveMeshInstance(MeshInstanceHandle mi) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        RemoveMeshInstance_nolock(mi);
    }

    void Finalize(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) override;

    void GetBounds(float bbox_min[3], float bbox_max[3]) const;

    uint32_t triangle_count() const override {
        std::shared_lock<std::shared_timed_mutex> lock(mtx_);
        return uint32_t(tris_.size());
    }
    uint32_t node_count() const override {
        std::shared_lock<std::shared_timed_mutex> lock(mtx_);
        return use_wide_bvh_ ? uint32_t(wnodes_.size()) : uint32_t(nodes_.size());
    }
};
} // namespace Cpu
} // namespace Ray