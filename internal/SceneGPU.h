#pragma once

#include <cfloat>
#include <climits>

#include "../Log.h"
#include "BVHSplit.h"
#include "SceneCommon.h"
#include "SparseStorageCPU.h"
#include "TextureUtils.h"
#include "Time_.h"

#include "shaders/shade_sky_interface.h"

namespace Ray {
extern const int MOON_TEX_W;
extern const int MOON_TEX_H;
extern const uint8_t __moon_tex[];
extern const int WEATHER_TEX_RES;
extern const uint8_t __weather_tex[];
extern const int NOISE_3D_RES;
extern const uint8_t __3d_noise_tex[];
extern const int CIRRUS_TEX_RES;
extern const uint8_t __cirrus_tex[];
extern const int CURL_TEX_RES;
extern const uint8_t __curl_tex[];

Ref::fvec4 rgb_to_rgbe(const Ref::fvec4 &rgb);
namespace NS {
class Context;
class Renderer;

template <class T> force_inline T clamp(const T &val, const T &min_val, const T &max_val) {
    return std::min(std::max(val, min_val), max_val);
}

inline Ref::fvec4 cross(const Ref::fvec4 &v1, const Ref::fvec4 &v2) {
    return Ref::fvec4{v1.get<1>() * v2.get<2>() - v1.get<2>() * v2.get<1>(),
                      v1.get<2>() * v2.get<0>() - v1.get<0>() * v2.get<2>(),
                      v1.get<0>() * v2.get<1>() - v1.get<1>() * v2.get<0>(), 0.0f};
}

const eTexFormat g_to_internal_format[] = {
    eTexFormat::Undefined, // Undefined
    eTexFormat::RGBA8,     // RGBA8888
    eTexFormat::RGB8,      // RGB888
    eTexFormat::RG8,       // RG88
    eTexFormat::R8,        // R8
    eTexFormat::BC1,       // BC1
    eTexFormat::BC3,       // BC3
    eTexFormat::BC4,       // BC4
    eTexFormat::BC5        // BC5
};

class Scene : public SceneCommon {
  protected:
    friend class NS::Renderer;

    Context *ctx_;
    bool use_hwrt_ = false, use_bindless_ = false, use_tex_compression_ = false;

    SparseStorage<bvh2_node_t, false /* Replicate */> nodes_;
    SparseStorage<tri_accel_t, false /* Replicate */> tris_;
    SparseStorage<uint32_t, false /* Replicate */> tri_indices_;
    SparseStorage<tri_mat_data_t> tri_materials_;
    Cpu::SparseStorage<mesh_t> meshes_;
    SparseStorage<mesh_instance_t> mesh_instances_;
    SparseStorage<vertex_t> vertices_;
    SparseStorage<uint32_t> vtx_indices_;

    SparseStorage<material_t> materials_;
    SparseStorage<atlas_texture_t> atlas_textures_;
    Cpu::SparseStorage<Texture> bindless_textures_;

    BindlessTexData bindless_tex_data_;

    TextureAtlas tex_atlases_[8];

    SparseStorage<light_t> lights_;
    Vector<uint32_t> li_indices_;
    uint32_t visible_lights_count_ = 0, blocker_lights_count_ = 0;
    Vector<light_cwbvh_node_t> light_cwnodes_;
    std::vector<uint32_t> dir_lights_; // compacted list of all directional lights

    Texture sky_transmittance_lut_tex_, sky_multiscatter_lut_tex_;
    Texture sky_moon_tex_, sky_weather_tex_, sky_cirrus_tex_, sky_curl_tex_;
    Texture sky_noise3d_tex_;

    LightHandle env_map_light_ = InvalidLightHandle;
    TextureHandle physical_sky_texture_ = InvalidTextureHandle;
    struct {
        int res = -1;
        float medium_lum = 0.0f;
        SmallVector<aligned_vector<fvec4>, 16> mips;
        Texture tex;
    } env_map_qtree_;

    mutable Vector<uint64_t> spatial_cache_entries_;
    mutable Vector<packed_cache_voxel_t> spatial_cache_voxels_curr_, spatial_cache_voxels_prev_;
    mutable float spatial_cache_cam_pos_prev_[3] = {};

    uint32_t tlas_root_ = 0xffffffff, tlas_block_ = 0xffffffff;

    bvh_node_t tlas_root_node_ = {};

    Buffer rt_geo_data_buf_, rt_instance_buf_, rt_tlas_buf_;

    struct MeshBlas {
        AccStructure acc;
        uint32_t geo_index, geo_count;
        FreelistAlloc::Allocation mem_alloc;
    };
    Cpu::SparseStorage<MeshBlas> rt_mesh_blases_;

    const uint32_t RtBLASChunkSize = 16 * 1024 * 1024;
    FreelistAlloc rt_blas_mem_alloc_;
    std::vector<Buffer> rt_blas_buffers_;
    AccStructure rt_tlas_;

    Buffer atmosphere_params_buf_;

    Shader sh_bake_sky_;

    Program prog_bake_sky_;

    Pipeline pi_bake_sky_;

    bool InitPipelines();

    MaterialHandle AddMaterial_nolock(const shading_node_desc_t &m);
    void SetMeshInstanceTransform_nolock(MeshInstanceHandle mi_handle, const float *xform);

    void RemoveMesh_nolock(MeshHandle m);
    void RemoveMeshInstance_nolock(MeshInstanceHandle i);
    void Rebuild_SWRT_TLAS_nolock();
    void RebuildLightTree_nolock();

    std::vector<Ray::color_rgba8_t> CalcSkyEnvTexture(const atmosphere_params_t &params, const int res[2],
                                                      const light_t lights[], Span<const uint32_t> dir_lights);
    void PrepareSkyEnvMap_nolock(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
    void PrepareEnvMapQTree_nolock();
    void GenerateTextureMips_nolock();
    void PrepareBindlessTextures_nolock();
    std::pair<uint32_t, uint32_t> Build_HWRT_BLAS_nolock(uint32_t vert_index, uint32_t vert_count);
    void Rebuild_HWRT_TLAS_nolock();

    TextureHandle AddAtlasTexture_nolock(const tex_desc_t &t);
    TextureHandle AddBindlessTexture_nolock(const tex_desc_t &t);

    // Workaround for Intel Arc issues
    void _insert_mem_barrier(void *cmdbuf);

    template <typename T, int N>
    static void WriteTextureMips(const color_t<T, N> data[], const int _res[2], int mip_count, bool compress,
                                 uint8_t out_data[], uint32_t out_size[16]);

  public:
    Scene(Context *ctx, bool use_hwrt, bool use_bindless, bool use_tex_compression, bool use_spatial_cache);
    ~Scene() override;

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
            bindless_textures_.Erase(t._block);
        } else {
            atlas_textures_.Erase(t._block);
        }
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
    void RemoveLight(const LightHandle i) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        lights_.Erase(i._block);
    }

    MeshInstanceHandle AddMeshInstance(const mesh_instance_desc_t &mi) override;
    void SetMeshInstanceTransform(MeshInstanceHandle mi_handle, const float *xform) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        SetMeshInstanceTransform_nolock(mi_handle, xform);
    }
    void RemoveMeshInstance(MeshInstanceHandle mi) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        RemoveMeshInstance_nolock(mi);
    }

    void Finalize(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) override;

    uint32_t triangle_count() const override {
        std::shared_lock<std::shared_timed_mutex> lock(mtx_);
        return uint32_t(vtx_indices_.size() / 3);
    }
    uint32_t node_count() const override {
        std::shared_lock<std::shared_timed_mutex> lock(mtx_);
        return uint32_t(nodes_.size());
    }
};
} // namespace NS
} // namespace Ray

namespace Ray {
int round_up(int v, int align);
}

inline Ray::NS::Scene::Scene(Context *ctx, const bool use_hwrt, const bool use_bindless, const bool use_tex_compression,
                             const bool use_spatial_cache)
    : ctx_(ctx), use_hwrt_(use_hwrt), use_bindless_(use_bindless), use_tex_compression_(use_tex_compression),
      nodes_(ctx, "Nodes"), tris_(ctx, "Tris"), tri_indices_(ctx, "Tri Indices"), tri_materials_(ctx, "Tri Materials"),
      mesh_instances_(ctx, "Mesh Instances"), vertices_(ctx, "Vertices"), vtx_indices_(ctx, "Vtx Indices"),
      materials_(ctx, "Materials"), atlas_textures_(ctx, "Atlas Textures"), bindless_tex_data_{ctx},
      tex_atlases_{{ctx, "Atlas RGBA", eTexFormat::RGBA8, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas RGB", eTexFormat::RGB8, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas RG", eTexFormat::RG8, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas R", eTexFormat::R8, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas BC1", eTexFormat::BC1, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas BC3", eTexFormat::BC3, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas BC4", eTexFormat::BC4, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, "Atlas BC5", eTexFormat::BC5, eTexFilter::Nearest, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE}},
      lights_(ctx, "Lights"), li_indices_(ctx, "LI Indices"), light_cwnodes_(ctx, "Light CWNodes"),
      spatial_cache_entries_(ctx, "Spatial Cache Entries"),
      spatial_cache_voxels_curr_(ctx, "Spatial Cache Voxels (1/2)"),
      spatial_cache_voxels_prev_(ctx, "Spatial Cache Voxels (2/2)") {
    SceneBase::log_ = ctx->log();
    InitPipelines();
    SetEnvironment({});
    if (use_spatial_cache) {
        spatial_cache_entries_.Resize(HASH_GRID_CACHE_ENTRIES_COUNT);
        spatial_cache_entries_.Fill(0, 0, HASH_GRID_CACHE_ENTRIES_COUNT);
        spatial_cache_voxels_curr_.Resize(HASH_GRID_CACHE_ENTRIES_COUNT);
        spatial_cache_voxels_curr_.Fill({}, 0, HASH_GRID_CACHE_ENTRIES_COUNT);
        spatial_cache_voxels_prev_.Resize(HASH_GRID_CACHE_ENTRIES_COUNT);
        spatial_cache_voxels_prev_.Fill({}, 0, HASH_GRID_CACHE_ENTRIES_COUNT);
    }
}

inline Ray::TextureHandle Ray::NS::Scene::AddAtlasTexture_nolock(const tex_desc_t &_t) {
    atlas_texture_t t;
    t.width = uint16_t(_t.w);
    t.height = uint16_t(_t.h);

    if (_t.is_srgb) {
        t.width |= ATLAS_TEX_SRGB_BIT;
    }
    if (_t.is_YCoCg) {
        t.height |= ATLAS_TEX_YCOCG_BIT;
    }

    if (((_t.generate_mipmaps && !IsCompressedFormat(_t.format)) || _t.mips_count > 1) &&
        _t.w > MIN_ATLAS_TEXTURE_SIZE && _t.h > MIN_ATLAS_TEXTURE_SIZE) {
        t.height |= ATLAS_TEX_MIPS_BIT;
    }

    int res[2] = {_t.w, _t.h};

    const bool use_compression = use_tex_compression_ && !_t.force_no_compression;

    std::unique_ptr<color_rg8_t[]> repacked_normalmap;
    bool reconstruct_z = _t.reconstruct_z;

    const void *tex_data = _t.data.data();

    if (_t.format == eTextureFormat::RGBA8888) {
        if (!_t.is_normalmap) {
            t.atlas = 0;
        } else {
            // TODO: get rid of this allocation
            repacked_normalmap = std::make_unique<color_rg8_t[]>(res[0] * res[1]);
            const bool invert_y = (_t.convention == eTextureConvention::DX);
            const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data.data());
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_normalmap[i].v[0] = rgba_data[i].v[0];
                repacked_normalmap[i].v[1] = invert_y ? (255 - rgba_data[i].v[1]) : rgba_data[i].v[1];
                reconstruct_z |= (rgba_data[i].v[2] < 250);
            }

            tex_data = repacked_normalmap.get();
            t.atlas = use_compression ? 7 : 2;
        }
    } else if (_t.format == eTextureFormat::RGB888) {
        if (!_t.is_normalmap) {
            t.atlas = use_compression ? 5 : 1;
            t.height |= ATLAS_TEX_YCOCG_BIT;
        } else {
            // TODO: get rid of this allocation
            repacked_normalmap = std::make_unique<color_rg8_t[]>(res[0] * res[1]);
            const bool invert_y = (_t.convention == eTextureConvention::DX);
            const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data.data());
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_normalmap[i].v[0] = rgb_data[i].v[0];
                repacked_normalmap[i].v[1] = invert_y ? (255 - rgb_data[i].v[1]) : rgb_data[i].v[1];
                reconstruct_z |= (rgb_data[i].v[2] < 250);
            }

            tex_data = repacked_normalmap.get();
            t.atlas = use_compression ? 7 : 2;
        }
    } else if (_t.format == eTextureFormat::RG88) {
        t.atlas = use_compression ? 7 : 2;
        reconstruct_z = _t.is_normalmap;

        const bool invert_y = _t.is_normalmap && (_t.convention == eTextureConvention::DX);
        if (invert_y) {
            // TODO: get rid of this allocation
            repacked_normalmap = std::make_unique<color_rg8_t[]>(res[0] * res[1]);
            const auto *rg_data = reinterpret_cast<const color_rg8_t *>(_t.data.data());
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_normalmap[i].v[0] = rg_data[i].v[0];
                repacked_normalmap[i].v[1] = 255 - rg_data[i].v[1];
            }
            tex_data = repacked_normalmap.get();
        }
    } else if (_t.format == eTextureFormat::R8) {
        t.atlas = use_compression ? 6 : 3;
    } else {
        const bool flip_vertical = (_t.convention == eTextureConvention::DX);
        const bool invert_green = (_t.convention == eTextureConvention::DX) && _t.is_normalmap;
        reconstruct_z = _t.is_normalmap && (_t.format == eTextureFormat::BC5);

        int read_offset = 0;
        int _res[2] = {_t.w, _t.h};
        // TODO: Get rid of allocation
        std::vector<uint8_t> temp_data;
        for (int i = 0; i < std::min(_t.mips_count, NUM_MIP_LEVELS) && _res[0] >= 4 && _res[1] >= 4; ++i) {
            if (_t.format == eTextureFormat::BC1) {
                t.atlas = 4;
                temp_data.resize(GetRequiredMemory_BCn<3>(_res[0], _res[1], 1));
                Preprocess_BCn<3>(&_t.data[read_offset], (_res[0] + 3) / 4, (_res[1] + 3) / 4, flip_vertical,
                                  invert_green, temp_data.data());
            } else if (_t.format == eTextureFormat::BC3) {
                t.atlas = 5;
                temp_data.resize(GetRequiredMemory_BCn<4>(_res[0], _res[1], 1));
                Preprocess_BCn<4>(&_t.data[read_offset], (_res[0] + 3) / 4, (_res[1] + 3) / 4, flip_vertical,
                                  invert_green, temp_data.data());
            } else if (_t.format == eTextureFormat::BC4) {
                t.atlas = 6;
                temp_data.resize(GetRequiredMemory_BCn<1>(_res[0], _res[1], 1));
                Preprocess_BCn<1>(&_t.data[read_offset], (_res[0] + 3) / 4, (_res[1] + 3) / 4, flip_vertical,
                                  invert_green, temp_data.data());
            } else if (_t.format == eTextureFormat::BC5) {
                t.atlas = 7;
                temp_data.resize(GetRequiredMemory_BCn<2>(_res[0], _res[1], 1));
                Preprocess_BCn<2>(&_t.data[read_offset], (_res[0] + 3) / 4, (_res[1] + 3) / 4, flip_vertical,
                                  invert_green, temp_data.data());
            }

            int pos[2] = {};
            const int page = tex_atlases_[t.atlas].AllocateRaw(temp_data.data(), _res, pos);

            t.page[i] = uint8_t(page);
            t.pos[i][0] = uint16_t(pos[0]);
            t.pos[i][1] = uint16_t(pos[1]);

            read_offset += int(temp_data.size());

            _res[0] /= 2;
            _res[1] /= 2;
        }

        // Fill remaining mip levels
        for (int i = _t.mips_count; i < NUM_MIP_LEVELS; i++) {
            t.page[i] = t.page[_t.mips_count - 1];
            t.pos[i][0] = t.pos[_t.mips_count - 1][0];
            t.pos[i][1] = t.pos[_t.mips_count - 1][1];
        }
    }

    if (reconstruct_z) {
        t.width |= uint32_t(ATLAS_TEX_RECONSTRUCT_Z_BIT);
    }

    if (!IsCompressedFormat(_t.format)) { // Allocate initial mip level
        int page = -1, pos[2] = {};
        if (t.atlas == 0) {
            page = tex_atlases_[0].Allocate<uint8_t, 4>(reinterpret_cast<const color_rgba8_t *>(tex_data), res, pos);
        } else if (t.atlas == 1 || t.atlas == 5) {
            page =
                tex_atlases_[t.atlas].Allocate<uint8_t, 3>(reinterpret_cast<const color_rgb8_t *>(tex_data), res, pos);
        } else if (t.atlas == 2 || t.atlas == 7) {
            page =
                tex_atlases_[t.atlas].Allocate<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(tex_data), res, pos);
        } else if (t.atlas == 3 || t.atlas == 6) {
            page = tex_atlases_[t.atlas].Allocate<uint8_t, 1>(reinterpret_cast<const color_r8_t *>(tex_data), res, pos);
        }

        if (page == -1) {
            return InvalidTextureHandle;
        }

        t.page[0] = uint8_t(page);
        t.pos[0][0] = uint16_t(pos[0]);
        t.pos[0][1] = uint16_t(pos[1]);
    }

    // Temporarily fill remaining mip levels with the last one (mips will be added later)
    for (int i = 1; i < NUM_MIP_LEVELS && !IsCompressedFormat(_t.format); i++) {
        t.page[i] = t.page[0];
        t.pos[i][0] = t.pos[0][0];
        t.pos[i][1] = t.pos[0][1];
    }

    if (_t.generate_mipmaps && (use_compression || !ctx_->image_blit_supported()) && !IsCompressedFormat(_t.format)) {
        // We have to generate mips here as uncompressed data will be lost

        int pages[16], positions[16][2];
        if (_t.format == eTextureFormat::RGBA8888) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 4>(reinterpret_cast<const color_rgba8_t *>(_t.data.data()), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else if (_t.format == eTextureFormat::RGB888) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 3>(reinterpret_cast<const color_rgb8_t *>(_t.data.data()), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else if (_t.format == eTextureFormat::RG88) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(_t.data.data()), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else if (_t.format == eTextureFormat::R8) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 1>(reinterpret_cast<const color_r8_t *>(_t.data.data()), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else {
            return InvalidTextureHandle;
        }

        for (int i = 1; i < NUM_MIP_LEVELS; i++) {
            t.page[i] = uint8_t(pages[i - 1]);
            t.pos[i][0] = uint16_t(positions[i - 1][0]);
            t.pos[i][1] = uint16_t(positions[i - 1][1]);
        }
    }

    log_->Info("Ray: Texture '%s' loaded (atlas = %i, %ix%i)", _t.name.data(), int(t.atlas), _t.w, _t.h);
    log_->Info("Ray: Atlasses are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC1[%i], BC3[%i], BC4[%i], BC5[%i])",
               tex_atlases_[0].page_count(), tex_atlases_[1].page_count(), tex_atlases_[2].page_count(),
               tex_atlases_[3].page_count(), tex_atlases_[4].page_count(), tex_atlases_[5].page_count(),
               tex_atlases_[6].page_count(), tex_atlases_[7].page_count());

    const std::pair<uint32_t, uint32_t> at = atlas_textures_.push(t);
    return TextureHandle{at.first, at.second};
}

inline Ray::TextureHandle Ray::NS::Scene::AddBindlessTexture_nolock(const tex_desc_t &_t) {
    eTexFormat src_fmt = eTexFormat::Undefined, fmt = eTexFormat::Undefined;

    const int expected_mip_count = CalcMipCount(_t.w, _t.h, 4);
    const int mip_count = (_t.generate_mipmaps && !Ray::IsCompressedFormat(_t.format))
                              ? expected_mip_count
                              : std::min(_t.mips_count, expected_mip_count);

    Buffer temp_stage_buf("Temp stage buf", ctx_, eBufType::Upload,
                          3 * _t.w * _t.h * 4 + 4096 * mip_count); // allocate for worst case
    uint8_t *stage_data = temp_stage_buf.Map();

    bool use_compression = use_tex_compression_ && !_t.force_no_compression;
    use_compression &= CanBeBlockCompressed(_t.w, _t.h, mip_count);

    uint32_t data_size[16] = {};

    std::unique_ptr<uint8_t[]> repacked_data;
    bool reconstruct_z = _t.reconstruct_z, is_YCoCg = _t.is_YCoCg;

    if (_t.format == eTextureFormat::RGBA8888) {
        if (!_t.is_normalmap) {
            src_fmt = fmt = eTexFormat::RGBA8;
            data_size[0] = round_up(_t.w * 4, TextureDataPitchAlignment) * _t.h;

            const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data.data());

            int j = 0;
            for (int y = 0; y < _t.h; ++y) {
                memcpy(&stage_data[j], &rgba_data[y * _t.w], _t.w * 4);
                j += round_up(_t.w * 4, TextureDataPitchAlignment);
            }
        } else {
            // TODO: get rid of this allocation
            repacked_data = std::make_unique<uint8_t[]>(2 * _t.w * _t.h);

            const bool invert_y = (_t.convention == Ray::eTextureConvention::DX);
            const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data.data());
            for (int i = 0; i < _t.w * _t.h; ++i) {
                repacked_data[i * 2 + 0] = rgba_data[i].v[0];
                repacked_data[i * 2 + 1] = invert_y ? (255 - rgba_data[i].v[1]) : rgba_data[i].v[1];
                reconstruct_z |= (rgba_data[i].v[2] < 250);
            }

            if (use_compression) {
                src_fmt = eTexFormat::RG8;
                fmt = eTexFormat::BC5;
                data_size[0] = GetRequiredMemory_BC5(_t.w, _t.h, TextureDataPitchAlignment);
                CompressImage_BC5<2>(&repacked_data[0], _t.w, _t.h, stage_data,
                                     GetRequiredMemory_BC5(_t.w, 1, TextureDataPitchAlignment));
            } else {
                src_fmt = fmt = eTexFormat::RG8;
                data_size[0] = round_up(_t.w * 2, TextureDataPitchAlignment) * _t.h;

                int j = 0;
                for (int y = 0; y < _t.h; ++y) {
                    memcpy(&stage_data[j], &repacked_data[y * _t.w * 2], _t.w * 2);
                    j += round_up(_t.w * 2, TextureDataPitchAlignment);
                }
            }
        }
    } else if (_t.format == eTextureFormat::RGB888) {
        if (!_t.is_normalmap) {
            if (use_compression) {
                auto temp_YCoCg = ConvertRGB_to_CoCgxY(_t.data.data(), _t.w, _t.h);
                is_YCoCg = true;
                src_fmt = eTexFormat::RGB8;
                fmt = eTexFormat::BC3;
                data_size[0] = GetRequiredMemory_BC3(_t.w, _t.h, TextureDataPitchAlignment);
                CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), _t.w, _t.h, stage_data,
                                                       GetRequiredMemory_BC3(_t.w, 1, TextureDataPitchAlignment));
            } else if (ctx_->rgb8_unorm_is_supported()) {
                src_fmt = fmt = eTexFormat::RGB8;
                data_size[0] = round_up(_t.w * 3, TextureDataPitchAlignment) * _t.h;

                const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data.data());

                int j = 0;
                for (int y = 0; y < _t.h; ++y) {
                    memcpy(&stage_data[j], &rgb_data[y * _t.w], _t.w * 3);
                    j += round_up(_t.w * 3, TextureDataPitchAlignment);
                }
            } else {
                // Fallback to 4-component texture
                src_fmt = fmt = eTexFormat::RGBA8;
                data_size[0] = round_up(_t.w * 4, TextureDataPitchAlignment) * _t.h;

                // TODO: get rid of this allocation
                repacked_data = std::make_unique<uint8_t[]>(4 * _t.w * _t.h);

                const auto *rgb_data = _t.data.data();

                for (int i = 0; i < _t.w * _t.h; ++i) {
                    repacked_data[i * 4 + 0] = rgb_data[i * 3 + 0];
                    repacked_data[i * 4 + 1] = rgb_data[i * 3 + 1];
                    repacked_data[i * 4 + 2] = rgb_data[i * 3 + 2];
                    repacked_data[i * 4 + 3] = 255;
                }

                int j = 0;
                for (int y = 0; y < _t.h; ++y) {
                    memcpy(&stage_data[j], &repacked_data[y * _t.w * 4], _t.w * 4);
                    j += round_up(_t.w * 4, TextureDataPitchAlignment);
                }
            }
        } else {
            // TODO: get rid of this allocation
            repacked_data = std::make_unique<uint8_t[]>(2 * _t.w * _t.h);

            const bool invert_y = (_t.convention == Ray::eTextureConvention::DX);
            const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data.data());
            for (int i = 0; i < _t.w * _t.h; ++i) {
                repacked_data[i * 2 + 0] = rgb_data[i].v[0];
                repacked_data[i * 2 + 1] = invert_y ? (255 - rgb_data[i].v[1]) : rgb_data[i].v[1];
                reconstruct_z |= (rgb_data[i].v[2] < 250);
            }

            if (use_compression) {
                src_fmt = eTexFormat::RG8;
                fmt = eTexFormat::BC5;
                data_size[0] = GetRequiredMemory_BC5(_t.w, _t.h, TextureDataPitchAlignment);
                CompressImage_BC5<2>(&repacked_data[0], _t.w, _t.h, stage_data,
                                     GetRequiredMemory_BC5(_t.w, 1, TextureDataPitchAlignment));
            } else {
                src_fmt = fmt = eTexFormat::RG8;
                data_size[0] = round_up(_t.w * 2, TextureDataPitchAlignment) * _t.h;

                int j = 0;
                for (int y = 0; y < _t.h; ++y) {
                    memcpy(&stage_data[j], &repacked_data[y * _t.w * 2], _t.w * 2);
                    j += round_up(_t.w * 2, TextureDataPitchAlignment);
                }
            }
        }
    } else if (_t.format == eTextureFormat::RG88) {
        src_fmt = fmt = eTexFormat::RG8;
        data_size[0] = round_up(_t.w * 2, TextureDataPitchAlignment) * _t.h;

        const bool invert_y = _t.is_normalmap && (_t.convention == Ray::eTextureConvention::DX);
        const auto *rg_data = reinterpret_cast<const color_rg8_t *>(_t.data.data());

        int j = 0;
        for (int y = 0; y < _t.h; ++y) {
            auto *dst = reinterpret_cast<color_rg8_t *>(&stage_data[j]);
            for (int x = 0; x < _t.w; ++x) {
                dst[x].v[0] = rg_data[y * _t.w + x].v[0];
                dst[x].v[1] = invert_y ? (255 - rg_data[y * _t.w + x].v[1]) : rg_data[y * _t.w + x].v[1];
            }
            j += round_up(_t.w * 2, TextureDataPitchAlignment);
        }

        reconstruct_z = _t.is_normalmap;
    } else if (_t.format == eTextureFormat::R8) {
        if (use_compression) {
            src_fmt = eTexFormat::R8;
            fmt = eTexFormat::BC4;
            data_size[0] = GetRequiredMemory_BC4(_t.w, _t.h, TextureDataPitchAlignment);
            CompressImage_BC4<1>(_t.data.data(), _t.w, _t.h, stage_data,
                                 GetRequiredMemory_BC4(_t.w, 1, TextureDataPitchAlignment));
        } else {
            src_fmt = fmt = eTexFormat::R8;
            data_size[0] = round_up(_t.w, TextureDataPitchAlignment) * _t.h;

            const auto *r_data = reinterpret_cast<const color_r8_t *>(_t.data.data());

            int j = 0;
            for (int y = 0; y < _t.h; ++y) {
                memcpy(&stage_data[j], &r_data[y * _t.w], _t.w);
                j += round_up(_t.w, TextureDataPitchAlignment);
            }
        }
    } else {
        //
        // Compressed formats
        //
        src_fmt = fmt = g_to_internal_format[int(_t.format)];

        const bool flip_vertical = (_t.convention == eTextureConvention::DX);
        const bool invert_green = (_t.convention == eTextureConvention::DX) && _t.is_normalmap;
        reconstruct_z = _t.is_normalmap && (_t.format == eTextureFormat::BC5);

        int read_offset = 0, write_offset = 0;
        int w = _t.w, h = _t.h;
        for (int i = 0; i < mip_count; ++i) {
            if (_t.format == eTextureFormat::BC1) {
                data_size[i] = Preprocess_BCn<3>(&_t.data[read_offset], (w + 3) / 4, (h + 3) / 4, flip_vertical,
                                                 invert_green, &stage_data[write_offset],
                                                 GetRequiredMemory_BC1(w, 1, TextureDataPitchAlignment));
            } else if (_t.format == eTextureFormat::BC3) {
                data_size[i] = Preprocess_BCn<4>(&_t.data[read_offset], (w + 3) / 4, (h + 3) / 4, flip_vertical,
                                                 invert_green, &stage_data[write_offset],
                                                 GetRequiredMemory_BC3(w, 1, TextureDataPitchAlignment));
            } else if (_t.format == eTextureFormat::BC4) {
                data_size[i] = Preprocess_BCn<1>(&_t.data[read_offset], (w + 3) / 4, (h + 3) / 4, flip_vertical,
                                                 invert_green, &stage_data[write_offset],
                                                 GetRequiredMemory_BC4(w, 1, TextureDataPitchAlignment));
            } else if (_t.format == eTextureFormat::BC5) {
                data_size[i] = Preprocess_BCn<2>(&_t.data[read_offset], (w + 3) / 4, (h + 3) / 4, flip_vertical,
                                                 invert_green, &stage_data[write_offset],
                                                 GetRequiredMemory_BC5(w, 1, TextureDataPitchAlignment));
            }

            read_offset += data_size[i];
            write_offset += round_up(data_size[i], 4096);

            w /= 2;
            h /= 2;
        }
    }

    if (_t.generate_mipmaps && !IsCompressedFormat(src_fmt)) {
        const int res[2] = {_t.w, _t.h};
        if (src_fmt == eTexFormat::RGBA8) {
            const auto *rgba_data =
                reinterpret_cast<const color_rgba8_t *>(repacked_data ? repacked_data.get() : _t.data.data());
            WriteTextureMips(rgba_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::RGB8) {
            const auto *rgb_data =
                reinterpret_cast<const color_rgb8_t *>(repacked_data ? repacked_data.get() : _t.data.data());
            WriteTextureMips(rgb_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::RG8) {
            const auto *rg_data =
                reinterpret_cast<const color_rg8_t *>(repacked_data ? repacked_data.get() : _t.data.data());
            WriteTextureMips(rg_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::R8) {
            const auto *r_data =
                reinterpret_cast<const color_r8_t *>(repacked_data ? repacked_data.get() : _t.data.data());
            WriteTextureMips(r_data, res, mip_count, use_compression, stage_data, data_size);
        }
    }

    temp_stage_buf.Unmap();

    TexParams p = {};
    p.w = _t.w;
    p.h = _t.h;
    p.format = fmt;
    if (_t.is_srgb && !is_YCoCg && !RequiresManualSRGBConversion(fmt)) {
        p.format = ToSRGBFormat(p.format);
    }
    p.mip_count = mip_count;
    p.usage = Bitmask<eTexUsage>(eTexUsage::Transfer) | eTexUsage::Sampled;
    p.sampling.filter = eTexFilter::Nearest;

    std::pair<uint32_t, uint32_t> ret = bindless_textures_.emplace(!_t.name.empty() ? _t.name.data() : "Bindless Tex",
                                                                   ctx_, p, ctx_->default_mem_allocs(), log_);

    { // Submit GPU commands
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        int res[2] = {_t.w, _t.h};
        uint32_t data_offset = 0;
        for (int i = 0; i < p.mip_count; ++i) {
            bindless_textures_[ret.first].SetSubImage(i, 0, 0, 0, res[0], res[1], 1, p.format, temp_stage_buf, cmd_buf,
                                                      data_offset, data_size[i]);
            res[0] = std::max(res[0] / 2, 1);
            res[1] = std::max(res[1] / 2, 1);
            data_offset += round_up(data_size[i], 4096);
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    temp_stage_buf.FreeImmediate();

    log_->Info("Ray: Texture '%s' loaded (%ix%i)", _t.name.data(), _t.w, _t.h);

    assert(ret.first <= 0x00ffffff);

    if (_t.is_srgb && (is_YCoCg || RequiresManualSRGBConversion(fmt))) {
        ret.first |= TEX_SRGB_BIT;
    }
    if (reconstruct_z) {
        ret.first |= TEX_RECONSTRUCT_Z_BIT;
    }
    if (is_YCoCg) {
        ret.first |= TEX_YCOCG_BIT;
    }

    return TextureHandle{ret.first, ret.second};
}

template <typename T, int N>
void Ray::NS::Scene::WriteTextureMips(const color_t<T, N> data[], const int _res[2], const int mip_count,
                                      const bool compress, uint8_t out_data[], uint32_t out_size[16]) {
    int src_res[2] = {_res[0], _res[1]};

    // TODO: try to get rid of these allocations
    std::vector<color_t<T, N>> _src_data, dst_data;
    for (int i = 1; i < mip_count; ++i) {
        const int dst_res[2] = {std::max(src_res[0] / 2, 1), std::max(src_res[1] / 2, 1)};

        dst_data.clear();
        dst_data.reserve(dst_res[0] * dst_res[1]);

        const color_t<T, N> *src_data = (i == 1) ? data : _src_data.data();

        for (int y = 0; y < dst_res[1]; ++y) {
            for (int x = 0; x < dst_res[0]; ++x) {
                const color_t<T, N> c00 = src_data[(2 * y + 0) * src_res[0] + (2 * x + 0)];
                const color_t<T, N> c10 = src_data[(2 * y + 0) * src_res[0] + std::min(2 * x + 1, src_res[0] - 1)];
                const color_t<T, N> c11 =
                    src_data[std::min(2 * y + 1, src_res[1] - 1) * src_res[0] + std::min(2 * x + 1, src_res[0] - 1)];
                const color_t<T, N> c01 = src_data[std::min(2 * y + 1, src_res[1] - 1) * src_res[0] + (2 * x + 0)];

                color_t<T, N> res;
                for (int j = 0; j < N; ++j) {
                    res.v[j] = (c00.v[j] + c10.v[j] + c11.v[j] + c01.v[j]) / 4;
                }

                dst_data.push_back(res);
            }
        }

        assert(dst_data.size() == (dst_res[0] * dst_res[1]));

        out_data += round_up(out_size[i - 1], 4096);
        if (compress && N <= 3) {
            if (N == 3) {
                auto temp_YCoCg = ConvertRGB_to_CoCgxY(&dst_data[0].v[0], dst_res[0], dst_res[1]);

                out_size[i] = GetRequiredMemory_BC3(dst_res[0], dst_res[1], TextureDataPitchAlignment);
                CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), dst_res[0], dst_res[1], out_data,
                                                       GetRequiredMemory_BC3(dst_res[0], 1, TextureDataPitchAlignment));
            } else if (N == 1) {
                out_size[i] = GetRequiredMemory_BC4(dst_res[0], dst_res[1], TextureDataPitchAlignment);
                CompressImage_BC4<N>(&dst_data[0].v[0], dst_res[0], dst_res[1], out_data,
                                     GetRequiredMemory_BC4(dst_res[0], 1, TextureDataPitchAlignment));
            } else if (N == 2) {
                out_size[i] = GetRequiredMemory_BC5(dst_res[0], dst_res[1], TextureDataPitchAlignment);
                CompressImage_BC5<2>(&dst_data[0].v[0], dst_res[0], dst_res[1], out_data,
                                     GetRequiredMemory_BC5(dst_res[0], 1, TextureDataPitchAlignment));
            }
        } else {
            out_size[i] = int(dst_res[1] * round_up(dst_res[0] * sizeof(color_t<T, N>), TextureDataPitchAlignment));
            int j = 0;
            for (int y = 0; y < dst_res[1]; ++y) {
                memcpy(&out_data[j], &dst_data[y * dst_res[0]], dst_res[0] * sizeof(color_t<T, N>));
                j += round_up(dst_res[0] * sizeof(color_t<T, N>), TextureDataPitchAlignment);
            }
        }

        src_res[0] = dst_res[0];
        src_res[1] = dst_res[1];
        std::swap(_src_data, dst_data);
    }
}

inline Ray::MaterialHandle Ray::NS::Scene::AddMaterial_nolock(const shading_node_desc_t &m) {
    material_t mat = {};

    mat.type = m.type;
    mat.textures[BASE_TEXTURE] = m.base_texture._index;
    mat.roughness_unorm = pack_unorm_16(m.roughness);
    mat.textures[ROUGH_TEXTURE] = m.roughness_texture._index;
    memcpy(&mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    mat.ior = m.ior;
    mat.tangent_rotation = 0.0f;
    mat.flags = 0;

    if (m.type == eShadingNode::Diffuse) {
        mat.sheen_unorm = pack_unorm_16(clamp(0.5f * m.sheen, 0.0f, 1.0f));
        mat.sheen_tint_unorm = pack_unorm_16(clamp(m.tint, 0.0f, 1.0f));
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
    } else if (m.type == eShadingNode::Glossy) {
        mat.tangent_rotation = 2.0f * PI * m.anisotropic_rotation;
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
        mat.tint_unorm = pack_unorm_16(clamp(m.tint, 0.0f, 1.0f));
    } else if (m.type == eShadingNode::Refractive) {
    } else if (m.type == eShadingNode::Emissive) {
        mat.strength = m.strength;
        if (m.importance_sample) {
            mat.flags |= MAT_FLAG_IMP_SAMPLE;
        }
    } else if (m.type == eShadingNode::Mix) {
        mat.strength = m.strength;
        mat.textures[MIX_MAT1] = m.mix_materials[0]._index;
        mat.textures[MIX_MAT2] = m.mix_materials[1]._index;
        if (m.mix_add) {
            mat.flags |= MAT_FLAG_MIX_ADD;
        }
    } else if (m.type == eShadingNode::Transparent) {
    }

    mat.textures[NORMALS_TEXTURE] = m.normal_map._index;
    mat.normal_map_strength_unorm = pack_unorm_16(clamp(m.normal_map_intensity, 0.0f, 1.0f));

    const std::pair<uint32_t, uint32_t> mi = materials_.push(mat);
    return MaterialHandle{mi.first, mi.second};
}

inline Ray::MaterialHandle Ray::NS::Scene::AddMaterial(const principled_mat_desc_t &m) {
    material_t main_mat = {};

    main_mat.type = eShadingNode::Principled;
    main_mat.textures[BASE_TEXTURE] = m.base_texture._index;
    memcpy(&main_mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    main_mat.sheen_unorm = pack_unorm_16(clamp(0.5f * m.sheen, 0.0f, 1.0f));
    main_mat.sheen_tint_unorm = pack_unorm_16(clamp(m.sheen_tint, 0.0f, 1.0f));
    main_mat.roughness_unorm = pack_unorm_16(clamp(m.roughness, 0.0f, 1.0f));
    main_mat.tangent_rotation = 2.0f * PI * clamp(m.anisotropic_rotation, 0.0f, 1.0f);
    main_mat.textures[ROUGH_TEXTURE] = m.roughness_texture._index;
    main_mat.metallic_unorm = pack_unorm_16(clamp(m.metallic, 0.0f, 1.0f));
    main_mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
    main_mat.ior = m.ior;
    main_mat.flags = 0;
    main_mat.transmission_unorm = pack_unorm_16(clamp(m.transmission, 0.0f, 1.0f));
    main_mat.transmission_roughness_unorm = pack_unorm_16(clamp(m.transmission_roughness, 0.0f, 1.0f));
    main_mat.textures[NORMALS_TEXTURE] = m.normal_map._index;
    main_mat.normal_map_strength_unorm = pack_unorm_16(clamp(m.normal_map_intensity, 0.0f, 1.0f));
    main_mat.anisotropic_unorm = pack_unorm_16(clamp(m.anisotropic, 0.0f, 1.0f));
    main_mat.specular_unorm = pack_unorm_16(clamp(m.specular, 0.0f, 1.0f));
    main_mat.textures[SPECULAR_TEXTURE] = m.specular_texture._index;
    main_mat.specular_tint_unorm = pack_unorm_16(clamp(m.specular_tint, 0.0f, 1.0f));
    main_mat.clearcoat_unorm = pack_unorm_16(clamp(m.clearcoat, 0.0f, 1.0f));
    main_mat.clearcoat_roughness_unorm = pack_unorm_16(clamp(m.clearcoat_roughness, 0.0f, 1.0f));

    const std::pair<uint32_t, uint32_t> mi = materials_.push(main_mat);
    auto root_node = MaterialHandle{mi.first, mi.second};
    MaterialHandle emissive_node = InvalidMaterialHandle, transparent_node = InvalidMaterialHandle;

    if (m.emission_strength > 0.0f &&
        (m.emission_color[0] > 0.0f || m.emission_color[1] > 0.0f || m.emission_color[2] > 0.0f)) {
        shading_node_desc_t emissive_desc;
        emissive_desc.type = eShadingNode::Emissive;

        memcpy(emissive_desc.base_color, m.emission_color, 3 * sizeof(float));
        emissive_desc.base_texture = m.emission_texture;
        emissive_desc.strength = m.emission_strength;
        emissive_desc.importance_sample = m.importance_sample;

        emissive_node = AddMaterial(emissive_desc);
    }

    if (m.alpha != 1.0f || m.alpha_texture != InvalidTextureHandle) {
        shading_node_desc_t transparent_desc;
        transparent_desc.type = eShadingNode::Transparent;

        transparent_node = AddMaterial(transparent_desc);
    }

    if (emissive_node != InvalidMaterialHandle) {
        if (root_node == InvalidMaterialHandle) {
            root_node = emissive_node;
        } else {
            shading_node_desc_t mix_node;
            mix_node.type = eShadingNode::Mix;
            mix_node.base_texture = InvalidTextureHandle;
            mix_node.strength = 0.5f;
            mix_node.ior = 0.0f;
            mix_node.mix_add = true;

            mix_node.mix_materials[0] = root_node;
            mix_node.mix_materials[1] = emissive_node;

            root_node = AddMaterial(mix_node);
        }
    }

    if (transparent_node != InvalidMaterialHandle) {
        if (root_node == InvalidMaterialHandle || m.alpha == 0.0f) {
            root_node = transparent_node;
        } else {
            shading_node_desc_t mix_node;
            mix_node.type = eShadingNode::Mix;
            mix_node.base_texture = m.alpha_texture;
            mix_node.strength = m.alpha;
            mix_node.ior = 0.0f;

            mix_node.mix_materials[0] = transparent_node;
            mix_node.mix_materials[1] = root_node;

            root_node = AddMaterial(mix_node);
        }
    }

    return root_node;
}

inline Ray::MeshHandle Ray::NS::Scene::AddMesh(const mesh_desc_t &_m) {
    std::vector<bvh_node_t> new_nodes;
    aligned_vector<tri_accel_t> new_tris;
    std::vector<uint32_t> new_tri_indices;
    std::vector<uint32_t> new_vtx_indices;

    bvh_settings_t s;
    s.oversplit_threshold = -1.0f;
    s.allow_spatial_splits = _m.allow_spatial_splits;
    s.use_fast_bvh_build = _m.use_fast_bvh_build;
    s.min_primitives_in_leaf = 8;
    s.primitive_alignment = 2;

    fvec4 bbox_min{FLT_MAX}, bbox_max{-FLT_MAX};

    if (use_hwrt_) {
        for (int j = 0; j < int(_m.vtx_indices.size()); j += 3) {
            fvec4 p[3];

            const uint32_t i0 = _m.vtx_indices[j + 0], i1 = _m.vtx_indices[j + 1], i2 = _m.vtx_indices[j + 2];

            memcpy(value_ptr(p[0]), &_m.vtx_positions.data[_m.vtx_positions.offset + i0 * _m.vtx_positions.stride],
                   3 * sizeof(float));
            memcpy(value_ptr(p[1]), &_m.vtx_positions.data[_m.vtx_positions.offset + i1 * _m.vtx_positions.stride],
                   3 * sizeof(float));
            memcpy(value_ptr(p[2]), &_m.vtx_positions.data[_m.vtx_positions.offset + i2 * _m.vtx_positions.stride],
                   3 * sizeof(float));

            bbox_min = min(bbox_min, min(p[0], min(p[1], p[2])));
            bbox_max = max(bbox_max, max(p[0], max(p[1], p[2])));
        }
    } else {
        aligned_vector<mtri_accel_t> _unused;
        PreprocessMesh(_m.vtx_positions, _m.vtx_indices, _m.base_vertex, s, new_nodes, new_tris, new_tri_indices,
                       _unused);

        memcpy(value_ptr(bbox_min), new_nodes[0].bbox_min, 3 * sizeof(float));
        memcpy(value_ptr(bbox_max), new_nodes[0].bbox_max, 3 * sizeof(float));
    }

    std::vector<tri_mat_data_t> new_tri_materials(_m.vtx_indices.size() / 3, {0xffff, 0xffff});

    // init triangle materials
    for (const mat_group_desc_t &grp : _m.groups) {
        bool is_front_solid = true, is_back_solid = true;

        uint32_t material_stack[32];
        material_stack[0] = grp.front_mat._index;
        uint32_t material_count = 1;

        while (material_count && is_front_solid) {
            const material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == eShadingNode::Mix) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == eShadingNode::Transparent) {
                is_front_solid = false;
            }
        }

        if (grp.back_mat != InvalidMaterialHandle) {
            if (grp.back_mat != grp.front_mat) {
                material_stack[0] = grp.back_mat._index;
                material_count = 1;
            } else {
                is_back_solid = is_front_solid;
            }
        }

        while (material_count && is_back_solid) {
            const material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == eShadingNode::Mix) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == eShadingNode::Transparent) {
                is_back_solid = false;
            }
        }

        for (size_t i = grp.vtx_start; i < grp.vtx_start + grp.vtx_count; i += 3) {
            tri_mat_data_t &tri_mat = new_tri_materials[i / 3];

            assert(grp.front_mat._index < (1 << 14) && "Not enough bits to reference material!");
            tri_mat.front_mi = uint16_t(grp.front_mat._index);
            if (is_front_solid) {
                tri_mat.front_mi |= MATERIAL_SOLID_BIT;
            }
            if (grp.back_mat != InvalidMaterialHandle) {
                assert(grp.back_mat._index < (1 << 14) && "Not enough bits to reference material!");
                tri_mat.back_mi = uint16_t(grp.back_mat._index);
                if (is_back_solid) {
                    tri_mat.back_mi |= MATERIAL_SOLID_BIT;
                }
            }
        }
    }

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    for (int i = 0; i < _m.vtx_indices.size(); ++i) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + _m.base_vertex);
    }

    const std::pair<uint32_t, uint32_t> trimat_index =
        tri_materials_.Allocate(new_tri_materials.data(), uint32_t(new_tri_materials.size()));
    // offset triangle indices
    for (uint32_t &i : new_tri_indices) {
        i += trimat_index.first;
    }

    // add attributes
    std::vector<vertex_t> new_vertices(_m.vtx_positions.data.size() / _m.vtx_positions.stride);
    for (int i = 0; i < int(new_vertices.size()); ++i) {
        vertex_t &v = new_vertices[i];

        memcpy(&v.p[0], &_m.vtx_positions.data[_m.vtx_positions.offset + i * _m.vtx_positions.stride],
               3 * sizeof(float));
        memcpy(&v.n[0], &_m.vtx_normals.data[_m.vtx_normals.offset + i * _m.vtx_normals.stride], 3 * sizeof(float));
        memcpy(&v.t[0], &_m.vtx_uvs.data[_m.vtx_uvs.offset + i * _m.vtx_uvs.stride], 2 * sizeof(float));

        if (!_m.vtx_binormals.data.empty()) {
            memcpy(&v.b[0], &_m.vtx_binormals.data[_m.vtx_binormals.offset + i * _m.vtx_binormals.stride],
                   3 * sizeof(float));
        }
    }

    if (_m.vtx_binormals.data.empty()) {
        ComputeTangentBasis(0, 0, new_vertices, new_vtx_indices, _m.vtx_indices);
    }

    const std::pair<uint32_t, uint32_t> vtx_index =
        vertices_.Allocate(new_vertices.data(), uint32_t(new_vertices.size()));

    const std::pair<uint32_t, uint32_t> vtx_indices_index =
        vtx_indices_.Allocate(nullptr, uint32_t(new_vtx_indices.size()));
    assert(trimat_index.second == vtx_indices_index.second);
    for (uint32_t &i : new_vtx_indices) {
        i += vtx_index.first;
    }
    vtx_indices_.Set(vtx_indices_index.first, uint32_t(new_vtx_indices.size()), new_vtx_indices.data());

    std::pair<uint32_t, uint32_t> tris_index = {}, tri_indices_index = {}, nodes_index = {0xffffffff, 0xffffffff};
    if (use_hwrt_) {
        tris_index = trimat_index;
        nodes_index = Build_HWRT_BLAS_nolock(vtx_indices_index.first, uint32_t(new_vtx_indices.size()));
    } else {
        tris_index = tris_.Allocate(&new_tris[0], uint32_t(new_tris.size()));
        tri_indices_index = tri_indices_.Allocate(&new_tri_indices[0], uint32_t(new_tri_indices.size()));
        assert(tri_indices_index.first == tris_index.first);
        assert(tri_indices_index.second == tris_index.second);
        assert(trimat_index.second == tris_index.second);

        std::vector<bvh2_node_t> new_bvh2_nodes;
        ConvertToBVH2(new_nodes, new_bvh2_nodes);

        nodes_index = nodes_.Allocate(nullptr, uint32_t(new_bvh2_nodes.size()));

        // offset nodes and primitives
        for (bvh2_node_t &n : new_bvh2_nodes) {
            if ((n.left_child & BVH2_PRIM_COUNT_BITS) == 0) {
                assert(n.left_child < new_bvh2_nodes.size());
                n.left_child += nodes_index.first;
            } else {
                n.left_child += tri_indices_index.first;
            }
            if ((n.right_child & BVH2_PRIM_COUNT_BITS) == 0) {
                assert(n.right_child < new_bvh2_nodes.size());
                n.right_child += nodes_index.first;
            } else {
                n.right_child += tri_indices_index.first;
            }
        }
        nodes_.Set(nodes_index.first, uint32_t(new_bvh2_nodes.size()), new_bvh2_nodes.data());
    }

    // add mesh
    mesh_t m = {};
    memcpy(m.bbox_min, value_ptr(bbox_min), 3 * sizeof(float));
    memcpy(m.bbox_max, value_ptr(bbox_max), 3 * sizeof(float));
    m.node_index = nodes_index.first;
    m.node_block = nodes_index.second;
    m.tris_index = tris_index.first;
    m.tris_block = tris_index.second;
    m.tris_count = uint32_t(new_tris.size());
    m.vert_index = vtx_indices_index.first;
    m.vert_block = vtx_indices_index.second;
    m.vert_count = uint32_t(new_vtx_indices.size());

    m.vert_data_index = vtx_index.first;
    m.vert_data_block = vtx_index.second;

    const std::pair<uint32_t, uint32_t> mesh_index = meshes_.push(m);
    return MeshHandle{mesh_index.first, mesh_index.second};
}

inline void Ray::NS::Scene::RemoveMesh_nolock(const MeshHandle i) {
    const mesh_t &m = meshes_[i._index];

    const uint32_t node_index = m.node_index, node_block = m.node_block;
    const uint32_t tris_block = m.tris_block;
    const uint32_t vert_block = m.vert_block, vert_data_block = m.vert_data_block;

    meshes_.Erase(i._block);

    [[maybe_unused]] bool rebuild_required = false;
    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end();) {
        mesh_instance_t &mi = *it;
        if (mi.mesh_index == i._index) {
            it = mesh_instances_.erase(it);
            rebuild_required = true;
        } else {
            ++it;
        }
    }

    if (use_hwrt_) {
        { // release struct memory
            const MeshBlas &blas = rt_mesh_blases_[node_index];
            rt_blas_mem_alloc_.Free(blas.mem_alloc.block);
        }
        rt_mesh_blases_.Erase(node_block);
    } else {
        tris_.Erase(tris_block);
        tri_indices_.Erase(tris_block);
        nodes_.Erase(node_block);
    }
    tri_materials_.Erase(tris_block);
    vertices_.Erase(vert_data_block);
    vtx_indices_.Erase(vert_block);
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const directional_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_DIR;
    l.visible = _l.multiple_importance;
    l.cast_shadow = _l.cast_shadow;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    l.dir.dir[0] = -_l.direction[0];
    l.dir.dir[1] = -_l.direction[1];
    l.dir.dir[2] = -_l.direction[2];
    l.dir.angle = _l.angle * PI / 360.0f;
    if (l.dir.angle > 0.0f) {
        const float radius = std::tan(l.dir.angle);
        const float mul = 1.0f / (PI * radius * radius);
        l.col[0] *= mul;
        l.col[1] *= mul;
        l.col[2] *= mul;
    }

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const sphere_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.visible = _l.multiple_importance && (_l.radius > 0.0f);
    l.cast_shadow = _l.cast_shadow;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;
    l.sph.spot = l.sph.blend = -1.0f;

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const spot_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.visible = _l.multiple_importance && (_l.radius > 0.0f);
    l.cast_shadow = _l.cast_shadow;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));
    memcpy(&l.sph.dir[0], &_l.direction[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;
    l.sph.spot = 0.5f * PI * _l.spot_size / 180.0f;
    l.sph.blend = _l.spot_blend * _l.spot_blend;

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const rect_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_RECT;
    l.doublesided = _l.doublesided;
    l.visible = _l.multiple_importance;
    l.cast_shadow = _l.cast_shadow;
    l.sky_portal = _l.sky_portal;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);
    if (_l.sky_portal) {
        l.ray_visibility |= (1u << RAY_TYPE_SHADOW);
    }

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.rect.pos[0] = xform[12];
    l.rect.pos[1] = xform[13];
    l.rect.pos[2] = xform[14];

    l.rect.area = _l.width * _l.height;

    const Ref::fvec4 uvec = _l.width * TransformDirection(Ref::fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::fvec4 vvec = _l.height * TransformDirection(Ref::fvec4{0.0f, 0.0f, 1.0f, 0.0f}, xform);

    memcpy(l.rect.u, value_ptr(uvec), 3 * sizeof(float));
    memcpy(l.rect.v, value_ptr(vvec), 3 * sizeof(float));

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const disk_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_DISK;
    l.doublesided = _l.doublesided;
    l.visible = _l.multiple_importance;
    l.cast_shadow = _l.cast_shadow;
    l.sky_portal = _l.sky_portal;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);
    if (_l.sky_portal) {
        l.ray_visibility |= (1u << RAY_TYPE_SHADOW);
    }

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.disk.pos[0] = xform[12];
    l.disk.pos[1] = xform[13];
    l.disk.pos[2] = xform[14];

    l.disk.area = 0.25f * PI * _l.size_x * _l.size_y;

    const Ref::fvec4 uvec = _l.size_x * TransformDirection(Ref::fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::fvec4 vvec = _l.size_y * TransformDirection(Ref::fvec4{0.0f, 0.0f, 1.0f, 0.0f}, xform);

    memcpy(l.disk.u, value_ptr(uvec), 3 * sizeof(float));
    memcpy(l.disk.v, value_ptr(vvec), 3 * sizeof(float));

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::LightHandle Ray::NS::Scene::AddLight(const line_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_LINE;
    l.visible = _l.multiple_importance;
    l.cast_shadow = _l.cast_shadow;
    l.sky_portal = _l.sky_portal;
    l.ray_visibility |= (_l.diffuse_visibility << RAY_TYPE_DIFFUSE);
    l.ray_visibility |= (_l.specular_visibility << RAY_TYPE_SPECULAR);
    l.ray_visibility |= (_l.refraction_visibility << RAY_TYPE_REFR);

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.line.pos[0] = xform[12];
    l.line.pos[1] = xform[13];
    l.line.pos[2] = xform[14];

    l.line.area = 2.0f * PI * _l.radius * _l.height;

    const Ref::fvec4 uvec = TransformDirection(Ref::fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::fvec4 vvec = TransformDirection(Ref::fvec4{0.0f, 1.0f, 0.0f, 0.0f}, xform);

    memcpy(l.line.u, value_ptr(uvec), 3 * sizeof(float));
    l.line.radius = _l.radius;
    memcpy(l.line.v, value_ptr(vvec), 3 * sizeof(float));
    l.line.height = _l.height;

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

inline Ray::MeshInstanceHandle Ray::NS::Scene::AddMeshInstance(const mesh_instance_desc_t &mi_desc) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const mesh_t &m = meshes_[mi_desc.mesh._index];

    mesh_instance_t mi = {};
    mi.mesh_index = mi_desc.mesh._index;
    mi.node_index = m.node_index;
    mi.lights_index = 0xffffffff;

    mi.ray_visibility = 0;
    mi.ray_visibility |= (mi_desc.camera_visibility << RAY_TYPE_CAMERA);
    mi.ray_visibility |= (mi_desc.diffuse_visibility << RAY_TYPE_DIFFUSE);
    mi.ray_visibility |= (mi_desc.specular_visibility << RAY_TYPE_SPECULAR);
    mi.ray_visibility |= (mi_desc.refraction_visibility << RAY_TYPE_REFR);
    mi.ray_visibility |= (mi_desc.shadow_visibility << RAY_TYPE_SHADOW);

    const std::pair<uint32_t, uint32_t> mi_index = mesh_instances_.emplace();

    { // find emissive triangles and add them as light emitters
        std::vector<light_t> new_lights;

        for (uint32_t tri = (m.vert_index / 3); tri < (m.vert_index + m.vert_count) / 3; ++tri) {
            const tri_mat_data_t &tri_mat = tri_materials_[tri];

            SmallVector<uint16_t, 64> mat_indices;
            mat_indices.push_back(tri_mat.front_mi & MATERIAL_INDEX_BITS);

            uint16_t front_emissive = 0xffff;
            for (int i = 0; i < int(mat_indices.size()); ++i) {
                const material_t &mat = materials_[mat_indices[i]];
                if (mat.type == eShadingNode::Emissive && (mat.flags & MAT_FLAG_IMP_SAMPLE)) {
                    front_emissive = mat_indices[i];
                    break;
                } else if (mat.type == eShadingNode::Mix) {
                    mat_indices.push_back(mat.textures[MIX_MAT1]);
                    mat_indices.push_back(mat.textures[MIX_MAT2]);
                }
            }

            mat_indices.clear();
            if (tri_mat.back_mi != 0xffff) {
                mat_indices.push_back(tri_mat.back_mi & MATERIAL_INDEX_BITS);
            }

            uint16_t back_emissive = 0xffff;
            for (int i = 0; i < int(mat_indices.size()); ++i) {
                const material_t &mat = materials_[mat_indices[i]];
                if (mat.type == eShadingNode::Emissive && (mat.flags & MAT_FLAG_IMP_SAMPLE)) {
                    back_emissive = mat_indices[i];
                    break;
                } else if (mat.type == eShadingNode::Mix) {
                    mat_indices.push_back(mat.textures[MIX_MAT1]);
                    mat_indices.push_back(mat.textures[MIX_MAT2]);
                }
            }

            if (front_emissive != 0xffff) {
                const material_t &mat = materials_[front_emissive];

                new_lights.emplace_back();
                light_t &new_light = new_lights.back();
                new_light.type = LIGHT_TYPE_TRI;
                new_light.doublesided = (back_emissive != 0xffff) ? 1 : 0;
                new_light.cast_shadow = 1;
                new_light.visible = 0;
                new_light.sky_portal = 0;
                new_light.ray_visibility = mi.ray_visibility;
                new_light.ray_visibility &= ~RAY_TYPE_CAMERA_BIT;
                new_light.ray_visibility &= ~RAY_TYPE_SHADOW_BIT;
                new_light.tri.tri_index = tri;
                new_light.tri.mi_index = mi_index.first;
                new_light.tri.tex_index = mat.textures[BASE_TEXTURE];
                new_light.col[0] = mat.base_color[0] * mat.strength;
                new_light.col[1] = mat.base_color[1] * mat.strength;
                new_light.col[2] = mat.base_color[2] * mat.strength;
            }
        }

        if (!new_lights.empty()) {
            const std::pair<uint32_t, uint32_t> lights_index =
                lights_.Allocate(new_lights.data(), uint32_t(new_lights.size()));
            mi.lights_index = lights_index.first;
            assert(lights_index.second <= 0xffffff);
            mi.ray_visibility |= (lights_index.second << 8);
        }
    }

    mesh_instances_.Set(mi_index.first, mi);

    auto ret = MeshInstanceHandle{mi_index.first, mi_index.second};

    SetMeshInstanceTransform_nolock(ret, mi_desc.xform);

    return ret;
}

inline void Ray::NS::Scene::SetMeshInstanceTransform_nolock(const MeshInstanceHandle mi_handle, const float *xform) {
    mesh_instance_t mi = mesh_instances_[mi_handle._index];

    memcpy(mi.xform, xform, 16 * sizeof(float));
    InverseMatrix(mi.xform, mi.inv_xform);

    mesh_instances_.Set(mi_handle._index, mi);
}

inline void Ray::NS::Scene::RemoveMeshInstance_nolock(const MeshInstanceHandle i) {
    const mesh_instance_t &mi = mesh_instances_[i._index];

    if (mi.lights_index != 0xffffffff) {
        const uint32_t light_block = (mi.ray_visibility >> 8);
        lights_.Erase(light_block);
    }
    mesh_instances_.Erase(i._block);
}

inline void Ray::NS::Scene::Finalize(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    if (env_map_light_ != InvalidLightHandle) {
        lights_.Erase(env_map_light_._block);
    }
    env_map_qtree_ = {};
    env_.qtree_levels = 0;
    env_.light_index = 0xffffffff;
    env_.sky_map_spread_angle = 0.0f;

    if (env_.env_map != InvalidTextureHandle._index &&
        (env_.env_map == PhysicalSkyTexture._index || env_.env_map == physical_sky_texture_._index)) {
        env_.sky_map_spread_angle = 2 * PI / float(env_.envmap_resolution);
        if (!atmosphere_params_buf_) {
            atmosphere_params_buf_ = Buffer{"Atmosphere Params", ctx_, eBufType::Uniform, sizeof(atmosphere_params_t)};
        }
        { // Update atmosphere parameters
            Buffer temp_upload_buf{"Temp atmosphere params upload", ctx_, eBufType::Upload,
                                   sizeof(atmosphere_params_t)};
            { // update stage buffer
                uint8_t *mapped_ptr = temp_upload_buf.Map();
                memcpy(mapped_ptr, &env_.atmosphere, sizeof(atmosphere_params_t));
                temp_upload_buf.Unmap();
            }
            CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
            CopyBufferToBuffer(temp_upload_buf, 0, atmosphere_params_buf_, 0, sizeof(atmosphere_params_t), cmd_buf);
            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());

            temp_upload_buf.FreeImmediate();
        }
        PrepareSkyEnvMap_nolock(parallel_for);
    }

    if (env_.importance_sample && env_.env_col[0] > 0.0f && env_.env_col[1] > 0.0f && env_.env_col[2] > 0.0f) {
        if (env_.env_map != InvalidTextureHandle._index) {
            PrepareEnvMapQTree_nolock();
        } else {
            // Dummy
            TexParams p;
            p.w = p.h = 1;
            p.format = eTexFormat::RGBA32F;
            p.mip_count = 1;
            p.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;

            env_map_qtree_.tex = Texture("Env map qtree", ctx_, p, ctx_->default_mem_allocs(), log_);
        }
        { // add env light source
            light_t l = {};

            l.type = LIGHT_TYPE_ENV;
            l.visible = 1;
            l.cast_shadow = 1;
            l.col[0] = l.col[1] = l.col[2] = 1.0f;
            l.ray_visibility |= RAY_TYPE_DIFFUSE_BIT;
            l.ray_visibility |= RAY_TYPE_SPECULAR_BIT;
            l.ray_visibility |= RAY_TYPE_REFR_BIT;

            const std::pair<uint32_t, uint32_t> li = lights_.push(l);
            env_map_light_ = LightHandle{li.first, li.second};
            env_.light_index = env_map_light_._index;
        }
    } else {
        // Dummy
        TexParams p;
        p.w = p.h = 1;
        p.format = eTexFormat::RGBA32F;
        p.mip_count = 1;
        p.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;

        env_map_qtree_.tex = Texture("Env map qtree", ctx_, p, ctx_->default_mem_allocs(), log_);
    }

    if (use_bindless_ && env_.env_map != InvalidTextureHandle._index) {
        const auto &env_map_params = bindless_textures_[env_.env_map].params;
        env_.env_map_res = (env_map_params.w << 16) | env_map_params.h;
    } else {
        env_.env_map_res = 0;
    }

    if (use_bindless_ && env_.back_map != InvalidTextureHandle._index) {
        const auto &back_map_params = bindless_textures_[env_.back_map].params;
        env_.back_map_res = (back_map_params.w << 16) | back_map_params.h;
    } else {
        env_.back_map_res = 0;
    }

    GenerateTextureMips_nolock();
    PrepareBindlessTextures_nolock();
    if (use_hwrt_) {
        Rebuild_HWRT_TLAS_nolock();
    } else {
        Rebuild_SWRT_TLAS_nolock();
    }
    RebuildLightTree_nolock();
}

inline void Ray::NS::Scene::Rebuild_SWRT_TLAS_nolock() {
    if (tlas_root_ != 0xffffffff) {
        nodes_.Erase(tlas_block_);
        tlas_root_ = tlas_block_ = 0xffffffff;
    }

    const size_t mi_count = mesh_instances_.size();
    if (!mi_count) {
        return;
    }

    aligned_vector<prim_t> primitives;
    primitives.reserve(mi_count);

    for (auto it = mesh_instances_.cbegin(); it != mesh_instances_.cend(); ++it) {
        const mesh_t &m = meshes_[it->mesh_index];

        Ref::fvec4 mi_bbox_min = 0.0f, mi_bbox_max = 0.0f;
        TransformBoundingBox(m.bbox_min, m.bbox_max, it->xform, value_ptr(mi_bbox_min), value_ptr(mi_bbox_max));

        primitives.push_back({0, 0, 0, mi_bbox_min, mi_bbox_max});
    }

    bvh_settings_t s = {};
    s.oversplit_threshold = -1.0f;
    s.min_primitives_in_leaf = 1;

    std::vector<bvh_node_t> bvh_nodes;
    std::vector<uint32_t> mi_indices;
    PreprocessPrims_SAH(primitives, {}, s, bvh_nodes, mi_indices);

    std::vector<bvh2_node_t> bvh2_nodes;
    ConvertToBVH2(bvh_nodes, bvh2_nodes);

    const std::pair<uint32_t, uint32_t> nodes_index = nodes_.Allocate(nullptr, uint32_t(bvh2_nodes.size()));
    // offset nodes
    for (bvh2_node_t &n : bvh2_nodes) {
        if ((n.left_child & BVH2_PRIM_COUNT_BITS) == 0) {
            n.left_child += nodes_index.first;
        } else {
            n.left_child = (n.left_child & BVH2_PRIM_COUNT_BITS) | mi_indices[n.left_child & BVH2_PRIM_INDEX_BITS];
        }
        if ((n.right_child & BVH2_PRIM_COUNT_BITS) == 0) {
            n.right_child += nodes_index.first;
        } else {
            n.right_child = (n.right_child & BVH2_PRIM_COUNT_BITS) | mi_indices[n.right_child & BVH2_PRIM_INDEX_BITS];
        }
    }
    nodes_.Set(nodes_index.first, uint32_t(bvh2_nodes.size()), bvh2_nodes.data());

    tlas_root_ = nodes_index.first;
    tlas_block_ = nodes_index.second;

    // store root node
    tlas_root_node_ = bvh_nodes[0];
}

// #define DUMP_SKY_ENV
#ifdef DUMP_SKY_ENV
extern "C" {
int SaveEXR(const float *data, int width, int height, int components, const int save_as_fp16,
            std::string_view outfilename, const char **err);
}
#endif

inline std::vector<Ray::color_rgba8_t> Ray::NS::Scene::CalcSkyEnvTexture(const atmosphere_params_t &params,
                                                                         const int res[2], const light_t lights[],
                                                                         Span<const uint32_t> dir_lights) {
    TexParams p;
    p.w = res[0];
    p.h = res[1];
    p.format = eTexFormat::RGBA32F;
    p.usage = Bitmask<eTexUsage>(eTexUsage::Storage) | eTexUsage::Transfer;
    auto temp_img = Texture{"Temp Sky Tex", ctx_, p, ctx_->default_mem_allocs(), log_};

    { // Write sky image
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        const TransitionInfo res_transition = {&temp_img, ResStateForClear};
        TransitionResourceStates(cmd_buf, AllStages, AllStages, {&res_transition, 1});

        static const float rgba[4] = {};
        ClearColorImage(temp_img, rgba, cmd_buf);

        const TransitionInfo res_transitions[] = {{&atmosphere_params_buf_, eResState::UniformBuffer},
                                                  {&sky_transmittance_lut_tex_, eResState::ShaderResource},
                                                  {&sky_multiscatter_lut_tex_, eResState::ShaderResource},
                                                  {&sky_moon_tex_, eResState::ShaderResource},
                                                  {&sky_weather_tex_, eResState::ShaderResource},
                                                  {&sky_cirrus_tex_, eResState::ShaderResource},
                                                  {&sky_curl_tex_, eResState::ShaderResource},
                                                  {&sky_noise3d_tex_, eResState::ShaderResource},
                                                  {&temp_img, eResState::UnorderedAccess}};
        TransitionResourceStates(cmd_buf, AllStages, AllStages, res_transitions);

        const Binding bindings[] = {
            {eBindTarget::UBuf, ShadeSky::ATMOSPHERE_PARAMS_BUF_SLOT, atmosphere_params_buf_},
            {eBindTarget::TexSampled, ShadeSky::TRANSMITTANCE_LUT_SLOT, sky_transmittance_lut_tex_},
            {eBindTarget::TexSampled, ShadeSky::MULTISCATTER_LUT_SLOT, sky_multiscatter_lut_tex_},
            {eBindTarget::TexSampled, ShadeSky::MOON_TEX_SLOT, sky_moon_tex_},
            {eBindTarget::TexSampled, ShadeSky::WEATHER_TEX_SLOT, sky_weather_tex_},
            {eBindTarget::TexSampled, ShadeSky::CIRRUS_TEX_SLOT, sky_cirrus_tex_},
            {eBindTarget::TexSampled, ShadeSky::CURL_TEX_SLOT, sky_curl_tex_},
            {eBindTarget::TexSampled, ShadeSky::NOISE3D_TEX_SLOT, sky_noise3d_tex_},
            {eBindTarget::Image, ShadeSky::OUT_IMG_SLOT, temp_img}};

        ShadeSky::Params uniform_params = {};
        uniform_params.res[0] = res[0];
        uniform_params.res[1] = res[1];

        const uint32_t grp_count[3] = {uint32_t(res[0]) / ShadeSky::LOCAL_GROUP_SIZE_X,
                                       uint32_t(res[1]) / ShadeSky::LOCAL_GROUP_SIZE_Y, 1};

        if (!dir_lights.empty()) {
            for (const uint32_t li : dir_lights) {
                const light_t &l = lights[li];

                memcpy(uniform_params.light_dir, l.dir.dir, 3 * sizeof(float));
                uniform_params.light_dir[3] = l.dir.angle;
                memcpy(uniform_params.light_col, l.col, 3 * sizeof(float));
                memcpy(uniform_params.light_col_point, l.col, 3 * sizeof(float));
                uniform_params.light_col[3] = cosf(l.dir.angle);
                if (l.dir.angle != 0.0f) {
                    const float radius = tanf(l.dir.angle);
                    uniform_params.light_col_point[0] *= (PI * radius * radius);
                    uniform_params.light_col_point[1] *= (PI * radius * radius);
                    uniform_params.light_col_point[2] *= (PI * radius * radius);
                }

                DispatchCompute(cmd_buf, pi_bake_sky_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                                ctx_->default_descr_alloc(), ctx_->log());
            }
        } else if (env_.atmosphere.stars_brightness > 0.0f) {
            // Use fake lightsource (to light up the moon)
            const fvec4 light_dir = {0.0f, -1.0f, 0.0f, 0.0f}, light_col = {144809.859f, 129443.617f, 127098.89f, 0.0f};

            memcpy(uniform_params.light_dir, value_ptr(light_dir), 3 * sizeof(float));
            uniform_params.light_dir[3] = 0.0f;
            memcpy(uniform_params.light_col, value_ptr(light_col), 3 * sizeof(float));
            memcpy(uniform_params.light_col_point, value_ptr(light_col), 3 * sizeof(float));
            uniform_params.light_col[3] = 0.0f;

            DispatchCompute(cmd_buf, pi_bake_sky_, grp_count, bindings, &uniform_params, sizeof(uniform_params),
                            ctx_->default_descr_alloc(), ctx_->log());
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    Buffer temp_readback_buf("Temp Sky Readback", ctx_, eBufType::Readback,
                             round_up(4 * res[0] * sizeof(float), TextureDataPitchAlignment) * res[1]);

    { // Readback texture data
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        CopyImageToBuffer(temp_img, 0, 0, 0, res[0], res[1], temp_readback_buf, cmd_buf, 0);
        InsertReadbackMemoryBarrier(ctx_->api(), cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    std::vector<color_rgba8_t> rgbe_pixels(res[0] * res[1]);

    const float *f32_data = reinterpret_cast<const float *>(temp_readback_buf.Map());
#ifdef DUMP_SKY_ENV
    const char *err = nullptr;
    SaveEXR(f32_data, res[0], res[1], 4, 0, "sky.exr", &err);
#endif
    for (int i = 0; i < res[0] * res[1]; ++i) {
        Ref::fvec4 color(&f32_data[4 * i]);
        color = rgb_to_rgbe(color);

        rgbe_pixels[i].v[0] = uint8_t(color.get<0>());
        rgbe_pixels[i].v[1] = uint8_t(color.get<1>());
        rgbe_pixels[i].v[2] = uint8_t(color.get<2>());
        rgbe_pixels[i].v[3] = uint8_t(color.get<3>());
    }
    temp_readback_buf.Unmap();
    temp_readback_buf.FreeImmediate();

    return rgbe_pixels;
}

inline void
Ray::NS::Scene::PrepareSkyEnvMap_nolock(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    const uint64_t t1 = Ray::GetTimeMs();

    if (physical_sky_texture_ != InvalidTextureHandle) {
        if (use_bindless_) {
            bindless_textures_.Erase(physical_sky_texture_._block);
        } else {
            atlas_textures_.Erase(physical_sky_texture_._block);
        }
    }

    // Find directional light sources
    dir_lights_.clear();
    for (auto it = lights_.cbegin(); it != lights_.cend(); ++it) {
        if (it->type == LIGHT_TYPE_DIR) {
            dir_lights_.push_back(it.index());
        }
    }

    // if (dir_lights.empty()) {
    //     env_.env_map = InvalidTextureHandle._index;
    //     if (env_.back_map == PhysicalSkyTexture._index) {
    //         env_.back_map = InvalidTextureHandle._index;
    //     }
    //     return;
    // }

    if (!sky_moon_tex_) {
        TexParams params;
        params.w = MOON_TEX_W;
        params.h = MOON_TEX_H;
        params.format = eTexFormat::RGBA8_srgb;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::ClampToEdge;

        sky_moon_tex_ = Texture{"Moon Tex", ctx_, params, ctx_->default_mem_allocs(), log_};

        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload, 4 * MOON_TEX_W * MOON_TEX_H);
        uint8_t *mapped_ptr = stage_buf.Map();
        for (int i = 0; i < MOON_TEX_W * MOON_TEX_H; ++i) {
            mapped_ptr[4 * i + 0] = __moon_tex[3 * i + 0];
            mapped_ptr[4 * i + 1] = __moon_tex[3 * i + 1];
            mapped_ptr[4 * i + 2] = __moon_tex[3 * i + 2];
            mapped_ptr[4 * i + 3] = 255;
        }
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_moon_tex_.SetSubImage(0, 0, 0, 0, MOON_TEX_W, MOON_TEX_H, 1, eTexFormat::RGBA8_srgb, stage_buf, cmd_buf, 0,
                                  stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }

    if (!sky_weather_tex_) {
        TexParams params;
        params.w = params.h = WEATHER_TEX_RES;
        params.format = eTexFormat::RGBA8;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::Repeat;

        sky_weather_tex_ = Texture{"Weather Tex", ctx_, params, ctx_->default_mem_allocs(), log_};

        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload, 4 * WEATHER_TEX_RES * WEATHER_TEX_RES);
        uint8_t *mapped_ptr = stage_buf.Map();
        for (int i = 0; i < WEATHER_TEX_RES * WEATHER_TEX_RES; ++i) {
            mapped_ptr[4 * i + 0] = __weather_tex[3 * i + 0];
            mapped_ptr[4 * i + 1] = __weather_tex[3 * i + 1];
            mapped_ptr[4 * i + 2] = __weather_tex[3 * i + 2];
            mapped_ptr[4 * i + 3] = 255;
        }
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_weather_tex_.SetSubImage(0, 0, 0, 0, WEATHER_TEX_RES, WEATHER_TEX_RES, 1, eTexFormat::RGBA8, stage_buf,
                                     cmd_buf, 0, stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }

    if (!sky_cirrus_tex_) {
        TexParams params;
        params.w = params.h = CIRRUS_TEX_RES;
        params.format = eTexFormat::RG8;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::Repeat;

        sky_cirrus_tex_ = Texture{"Cirrus Tex", ctx_, params, ctx_->default_mem_allocs(), log_};

        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload, 2 * CIRRUS_TEX_RES * CIRRUS_TEX_RES);
        uint8_t *mapped_ptr = stage_buf.Map();
        memcpy(mapped_ptr, __cirrus_tex, 2 * CIRRUS_TEX_RES * CIRRUS_TEX_RES);
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_cirrus_tex_.SetSubImage(0, 0, 0, 0, CIRRUS_TEX_RES, CIRRUS_TEX_RES, 1, eTexFormat::RG8, stage_buf, cmd_buf,
                                    0, stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }

    if (!sky_curl_tex_) {
        TexParams params;
        params.w = params.h = CURL_TEX_RES;
        params.format = eTexFormat::RGBA8_srgb;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::Repeat;

        sky_curl_tex_ = Texture{"Curl Tex", ctx_, params, ctx_->default_mem_allocs(), log_};

        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload, 4 * CURL_TEX_RES * CURL_TEX_RES);
        uint8_t *mapped_ptr = stage_buf.Map();
        for (int i = 0; i < CURL_TEX_RES * CURL_TEX_RES; ++i) {
            mapped_ptr[4 * i + 0] = __curl_tex[3 * i + 0];
            mapped_ptr[4 * i + 1] = __curl_tex[3 * i + 1];
            mapped_ptr[4 * i + 2] = __curl_tex[3 * i + 2];
            mapped_ptr[4 * i + 3] = 255;
        }
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_curl_tex_.SetSubImage(0, 0, 0, 0, CURL_TEX_RES, CURL_TEX_RES, 1, eTexFormat::RGBA8_srgb, stage_buf, cmd_buf,
                                  0, stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }

    if (!sky_noise3d_tex_.handle()) {
        TexParams params;
        params.w = params.h = params.d = NOISE_3D_RES;
        params.format = eTexFormat::R8;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;
        params.sampling.filter = eTexFilter::Bilinear;
        params.sampling.wrap = eTexWrap::Repeat;

        sky_noise3d_tex_ = Texture{"Noise 3d Tex", ctx_, params, ctx_->default_mem_allocs(), log_};

        const uint32_t data_len = NOISE_3D_RES * NOISE_3D_RES * round_up(NOISE_3D_RES, TextureDataPitchAlignment);
        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload, data_len);
        uint8_t *mapped_ptr = stage_buf.Map();

        int i = 0;
        for (int yz = 0; yz < NOISE_3D_RES * NOISE_3D_RES; ++yz) {
            memcpy(&mapped_ptr[i], &__3d_noise_tex[yz * NOISE_3D_RES], NOISE_3D_RES);
            i += round_up(NOISE_3D_RES, TextureDataPitchAlignment);
        }

        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_noise3d_tex_.SetSubImage(0, 0, 0, 0, NOISE_3D_RES, NOISE_3D_RES, NOISE_3D_RES, eTexFormat::R8, stage_buf,
                                     cmd_buf, 0, NOISE_3D_RES * NOISE_3D_RES * NOISE_3D_RES);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }

    const int SkyEnvRes[] = {env_.envmap_resolution, env_.envmap_resolution / 2};
    std::vector<color_rgba8_t> rgbe_pixels;
    if (pi_bake_sky_) {
        // Use GPU
        rgbe_pixels = CalcSkyEnvTexture(env_.atmosphere, SkyEnvRes, lights_.data(), dir_lights_);
    } else {
        // Use CPU
        rgbe_pixels =
            SceneCommon::CalcSkyEnvTexture(env_.atmosphere, SkyEnvRes, lights_.data(), dir_lights_, parallel_for);
    }

    tex_desc_t desc = {};
    desc.format = eTextureFormat::RGBA8888;
    desc.name = "Physical Sky Texture";
    desc.data = Span<const uint8_t>{&rgbe_pixels[0].v[0], 4 * SkyEnvRes[0] * SkyEnvRes[1]};
    desc.w = SkyEnvRes[0];
    desc.h = SkyEnvRes[1];
    desc.is_srgb = false;
    desc.force_no_compression = true;

    if (use_bindless_) {
        physical_sky_texture_ = AddBindlessTexture_nolock(desc);
    } else {
        physical_sky_texture_ = AddAtlasTexture_nolock(desc);
    }

    env_.env_map = physical_sky_texture_._index;
    if (env_.back_map == PhysicalSkyTexture._index) {
        env_.back_map = physical_sky_texture_._index;
    }

    log_->Info("PrepareSkyEnvMap (%ix%i) done in %lldms", SkyEnvRes[0], SkyEnvRes[1], (long long)(GetTimeMs() - t1));
}

inline void Ray::NS::Scene::PrepareEnvMapQTree_nolock() {
    const int tex = int(env_.env_map & 0x00ffffff);

    Buffer temp_stage_buf;
    ivec2 size;
    int pitch = 0;

    if (use_bindless_) {
        const Texture &t = bindless_textures_[tex];
        size.template set<0>(t.params.w);
        size.template set<1>(t.params.h);

        assert(t.params.format == eTexFormat::RGBA8);
        pitch = round_up(t.params.w * GetPerPixelDataLen(eTexFormat::RGBA8), TextureDataPitchAlignment);
        const uint32_t data_size = pitch * t.params.h;

        temp_stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Readback, data_size);

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        _insert_mem_barrier(cmd_buf);

        CopyImageToBuffer(t, 0, 0, 0, t.params.w, t.params.h, temp_stage_buf, cmd_buf, 0);

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    } else {
        const atlas_texture_t &t = atlas_textures_[tex];
        size.template set<0>(t.width & ATLAS_TEX_WIDTH_BITS);
        size.template set<1>(t.height & ATLAS_TEX_HEIGHT_BITS);

        const TextureAtlas &atlas = tex_atlases_[t.atlas];

        assert(atlas.format() == eTexFormat::RGBA8);
        pitch = round_up(size.get<0>() * GetPerPixelDataLen(atlas.real_format()), TextureDataPitchAlignment);
        const uint32_t data_size = pitch * size.get<1>();

        temp_stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Readback, data_size);

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        _insert_mem_barrier(cmd_buf);

        atlas.CopyRegionTo(t.page[0], t.pos[0][0], t.pos[0][1], (t.width & ATLAS_TEX_WIDTH_BITS),
                           (t.height & ATLAS_TEX_HEIGHT_BITS), temp_stage_buf, cmd_buf, 0);

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    pitch /= 4;

    const uint8_t *rgbe_data = temp_stage_buf.Map();

    const int lowest_dim = std::min(size[0], size[1]);

    env_map_qtree_.res = 1;
    while (2 * env_map_qtree_.res < lowest_dim) {
        env_map_qtree_.res *= 2;
    }

    assert(env_map_qtree_.mips.empty());

    int cur_res = env_map_qtree_.res;
    float total_lum = 0.0f;

    { // initialize the first quadtree level
        env_map_qtree_.mips.emplace_back(cur_res * cur_res / 4, 0.0f);

        static const float FilterWeights[][5] = {{1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},    //
                                                 {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f}, //
                                                 {7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f}, //
                                                 {4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f}, //
                                                 {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}};
        static const float FilterSize = 0.5f;

        for (int qy = 0; qy < cur_res; ++qy) {
            for (int qx = 0; qx < cur_res; ++qx) {
                for (int jj = -2; jj <= 2; ++jj) {
                    for (int ii = -2; ii <= 2; ++ii) {
                        const Ref::fvec2 q = {Ref::fract(1.0f + (float(qx) + 0.5f + ii * FilterSize) / cur_res),
                                              Ref::fract(1.0f + (float(qy) + 0.5f + jj * FilterSize) / cur_res)};
                        fvec4 dir;
                        CanonicalToDir(value_ptr(q), 0.0f, value_ptr(dir));

                        const float theta = acosf(clamp(dir.get<1>(), -1.0f, 1.0f)) / PI;
                        float phi = atan2f(dir.get<2>(), dir.get<0>());
                        if (phi < 0) {
                            phi += 2 * PI;
                        }
                        if (phi > 2 * PI) {
                            phi -= 2 * PI;
                        }

                        const float u = Ref::fract(0.5f * phi / PI);

                        const fvec2 uvs = fvec2{u, theta} * fvec2(size);
                        const ivec2 iuvs = clamp(ivec2(uvs), ivec2(0), size - 1);

                        const uint8_t *col_rgbe = &rgbe_data[4 * (iuvs.get<1>() * pitch + iuvs.get<0>())];
                        fvec4 col_rgb;
                        rgbe_to_rgb(col_rgbe, value_ptr(col_rgb));
                        const float cur_lum = (col_rgb.get<0>() + col_rgb.get<1>() + col_rgb.get<2>());

                        int index = 0;
                        index |= (qx & 1) << 0;
                        index |= (qy & 1) << 1;

                        const int _qx = (qx / 2);
                        const int _qy = (qy / 2);

                        auto &qvec = env_map_qtree_.mips[0][_qy * cur_res / 2 + _qx];
                        qvec.set(index, qvec[index] + cur_lum * FilterWeights[ii + 2][jj + 2]);
                    }
                }
            }
        }

        for (const fvec4 &v : env_map_qtree_.mips[0]) {
            total_lum += hsum(v);
        }

        cur_res /= 2;
    }

    env_map_qtree_.medium_lum = total_lum / float(cur_res * cur_res);

    temp_stage_buf.Unmap();
    temp_stage_buf.FreeImmediate();

    while (cur_res > 1) {
        env_map_qtree_.mips.emplace_back(cur_res * cur_res / 4, 0.0f);
        const auto &prev_mip = env_map_qtree_.mips[env_map_qtree_.mips.size() - 2];

        for (int y = 0; y < cur_res; ++y) {
            for (int x = 0; x < cur_res; ++x) {
                const float res_lum = prev_mip[y * cur_res + x][0] + prev_mip[y * cur_res + x][1] +
                                      prev_mip[y * cur_res + x][2] + prev_mip[y * cur_res + x][3];

                int index = 0;
                index |= (x & 1) << 0;
                index |= (y & 1) << 1;

                const int qx = (x / 2);
                const int qy = (y / 2);

                env_map_qtree_.mips.back()[qy * cur_res / 2 + qx].set(index, res_lum);
            }
        }

        cur_res /= 2;
    }

    //
    // Determine how many levels was actually required
    //

    const float LumFractThreshold = 0.005f;

    cur_res = 2;
    int the_last_required_lod = 0;
    for (int lod = int(env_map_qtree_.mips.size()) - 1; lod >= 0; --lod) {
        the_last_required_lod = lod;
        const auto &cur_mip = env_map_qtree_.mips[lod];

        bool subdivision_required = false;
        for (int y = 0; y < (cur_res / 2) && !subdivision_required; ++y) {
            for (int x = 0; x < (cur_res / 2) && !subdivision_required; ++x) {
                const ivec4 mask = simd_cast(cur_mip[y * cur_res / 2 + x] > LumFractThreshold * total_lum);
                subdivision_required |= mask.not_all_zeros();
            }
        }

        if (!subdivision_required) {
            break;
        }

        cur_res *= 2;
    }

    //
    // Drop not needed levels
    //

    while (the_last_required_lod != 0) {
        for (int i = 1; i < int(env_map_qtree_.mips.size()); ++i) {
            env_map_qtree_.mips[i - 1] = std::move(env_map_qtree_.mips[i]);
        }
        env_map_qtree_.res /= 2;
        env_map_qtree_.mips.pop_back();
        --the_last_required_lod;
    }

    env_.qtree_levels = int(env_map_qtree_.mips.size());
    for (int i = 0; i < env_.qtree_levels; ++i) {
        env_.qtree_mips[i] = value_ptr(env_map_qtree_.mips[i][0]);
    }
    for (int i = env_.qtree_levels; i < std::size(env_.qtree_mips); ++i) {
        env_.qtree_mips[i] = nullptr;
    }

    //
    // Upload texture
    //

    int req_size = 0, mip_offsets[16] = {};
    for (int i = 0; i < env_.qtree_levels; ++i) {
        mip_offsets[i] = req_size;
        req_size += 4096 * int((env_map_qtree_.mips[i].size() * sizeof(fvec4) + 4096 - 1) / 4096);
    }

    temp_stage_buf = Buffer("Temp upload buf", ctx_, eBufType::Upload, req_size);
    uint8_t *stage_data = temp_stage_buf.Map();

    for (int i = 0; i < env_.qtree_levels; ++i) {
        const int res = (env_map_qtree_.res >> i) / 2;
        assert(res * res == env_map_qtree_.mips[i].size());

        int j = mip_offsets[i];
        for (int y = 0; y < res; ++y) {
            memcpy(&stage_data[j], &env_map_qtree_.mips[i][y * res], res * sizeof(fvec4));
            j += round_up(res * sizeof(fvec4), TextureDataPitchAlignment);
        }
    }
    temp_stage_buf.Unmap();

    TexParams p;
    p.w = p.h = (env_map_qtree_.res / 2);
    p.format = eTexFormat::RGBA32F;
    p.mip_count = env_.qtree_levels;
    p.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;

    env_map_qtree_.tex = Texture("Env map qtree", ctx_, p, ctx_->default_mem_allocs(), log_);

    CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    for (int i = 0; i < env_.qtree_levels; ++i) {
        env_map_qtree_.tex.SetSubImage(i, 0, 0, 0, (env_map_qtree_.res >> i) / 2, (env_map_qtree_.res >> i) / 2, 1,
                                       eTexFormat::RGBA32F, temp_stage_buf, cmd_buf, mip_offsets[i],
                                       int(env_map_qtree_.mips[i].size() * sizeof(fvec4)));
    }

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    temp_stage_buf.FreeImmediate();

    log_->Info("Env map qtree res is %i", env_map_qtree_.res);
}

inline void Ray::NS::Scene::RebuildLightTree_nolock() {
    aligned_vector<prim_t> primitives;
    primitives.reserve(lights_.size());

    struct additional_data_t {
        Ref::fvec4 axis;
        float flux, omega_n, omega_e;
    };
    aligned_vector<additional_data_t> additional_data;
    additional_data.reserve(lights_.size());

    visible_lights_count_ = blocker_lights_count_ = 0;
    li_indices_.Clear();
    std::vector<uint32_t> new_li_indices;
    new_li_indices.reserve(lights_.size());

    for (auto it = lights_.cbegin(); it != lights_.cend(); ++it) {
        const light_t &l = *it;
        if (l.type == LIGHT_TYPE_DIR && physical_sky_texture_ != InvalidTextureHandle) {
            // Directional lights are already 'baked' into sky texture
            continue;
        }

        Ref::fvec4 bbox_min = 0.0f, bbox_max = 0.0f, axis = {0.0f, 1.0f, 0.0f, 0.0f};
        float area = 1.0f, omega_n = 0.0f, omega_e = 0.0f;
        float lum = l.col[0] + l.col[1] + l.col[2];

        new_li_indices.push_back(it.index());
        if (l.visible) {
            ++visible_lights_count_;
        }
        if ((l.ray_visibility & RAY_TYPE_SHADOW_BIT) != 0) {
            ++blocker_lights_count_;
        }

        switch (l.type) {
        case LIGHT_TYPE_SPHERE: {
            const auto pos = Ref::fvec4{l.sph.pos[0], l.sph.pos[1], l.sph.pos[2], 0.0f};

            bbox_min = pos - Ref::fvec4{l.sph.radius, l.sph.radius, l.sph.radius, 0.0f};
            bbox_max = pos + Ref::fvec4{l.sph.radius, l.sph.radius, l.sph.radius, 0.0f};
            if (l.sph.area != 0.0f) {
                area = l.sph.area;
            }
            omega_n = PI; // normals in all directions
            omega_e = PI / 2.0f;
        } break;
        case LIGHT_TYPE_DIR: {
            bbox_min = Ref::fvec4{-MAX_DIST, -MAX_DIST, -MAX_DIST, 0.0f};
            bbox_max = Ref::fvec4{MAX_DIST, MAX_DIST, MAX_DIST, 0.0f};
            axis = Ref::fvec4{l.dir.dir[0], l.dir.dir[1], l.dir.dir[2], 0.0f};
            omega_n = 0.0f; // single normal
            omega_e = l.dir.angle;
            if (l.dir.angle != 0.0f) {
                const float radius = tanf(l.dir.angle);
                area = (PI * radius * radius);
            }
        } break;
        case LIGHT_TYPE_LINE: {
            const auto pos = Ref::fvec4{l.line.pos[0], l.line.pos[1], l.line.pos[2], 0.0f};
            auto light_u = Ref::fvec4{l.line.u[0], l.line.u[1], l.line.u[2], 0.0f},
                 light_dir = Ref::fvec4{l.line.v[0], l.line.v[1], l.line.v[2], 0.0f};
            Ref::fvec4 light_v = NS::cross(light_u, light_dir);

            light_u *= l.line.radius;
            light_v *= l.line.radius;
            light_dir *= 0.5f * l.line.height;

            const Ref::fvec4 p0 = pos + light_dir + light_u + light_v, p1 = pos + light_dir + light_u - light_v,
                             p2 = pos + light_dir - light_u + light_v, p3 = pos + light_dir - light_u - light_v,
                             p4 = pos - light_dir + light_u + light_v, p5 = pos - light_dir + light_u - light_v,
                             p6 = pos - light_dir - light_u + light_v, p7 = pos - light_dir - light_u - light_v;

            bbox_min = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
            bbox_max = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));
            area = l.line.area;
            omega_n = PI; // normals in all directions
            omega_e = PI / 2.0f;
        } break;
        case LIGHT_TYPE_RECT: {
            const auto pos = Ref::fvec4{l.rect.pos[0], l.rect.pos[1], l.rect.pos[2], 0.0f};
            const auto u = 0.5f * Ref::fvec4{l.rect.u[0], l.rect.u[1], l.rect.u[2], 0.0f};
            const auto v = 0.5f * Ref::fvec4{l.rect.v[0], l.rect.v[1], l.rect.v[2], 0.0f};

            const Ref::fvec4 p0 = pos + u + v, p1 = pos + u - v, p2 = pos - u + v, p3 = pos - u - v;
            bbox_min = min(min(p0, p1), min(p2, p3));
            bbox_max = max(max(p0, p1), max(p2, p3));
            area = l.rect.area;

            axis = normalize(NS::cross(u, v));
            omega_n = l.doublesided ? PI : 0.0f;
            omega_e = PI / 2.0f;
        } break;
        case LIGHT_TYPE_DISK: {
            const auto pos = Ref::fvec4{l.disk.pos[0], l.disk.pos[1], l.disk.pos[2], 0.0f};
            const auto u = 0.5f * Ref::fvec4{l.disk.u[0], l.disk.u[1], l.disk.u[2], 0.0f};
            const auto v = 0.5f * Ref::fvec4{l.disk.v[0], l.disk.v[1], l.disk.v[2], 0.0f};

            const Ref::fvec4 p0 = pos + u + v, p1 = pos + u - v, p2 = pos - u + v, p3 = pos - u - v;
            bbox_min = min(min(p0, p1), min(p2, p3));
            bbox_max = max(max(p0, p1), max(p2, p3));
            area = l.disk.area;

            axis = normalize(NS::cross(u, v));
            omega_n = l.doublesided ? PI : 0.0f;
            omega_e = PI / 2.0f;
        } break;
        case LIGHT_TYPE_TRI: {
            const mesh_instance_t &lmi = mesh_instances_[l.tri.mi_index];
            const uint32_t ltri_index = l.tri.tri_index;

            const vertex_t &v1 = vertices_[vtx_indices_[ltri_index * 3 + 0]];
            const vertex_t &v2 = vertices_[vtx_indices_[ltri_index * 3 + 1]];
            const vertex_t &v3 = vertices_[vtx_indices_[ltri_index * 3 + 2]];

            auto p1 = Ref::fvec4(v1.p[0], v1.p[1], v1.p[2], 0.0f), p2 = Ref::fvec4(v2.p[0], v2.p[1], v2.p[2], 0.0f),
                 p3 = Ref::fvec4(v3.p[0], v3.p[1], v3.p[2], 0.0f);

            p1 = TransformPoint(p1, lmi.xform);
            p2 = TransformPoint(p2, lmi.xform);
            p3 = TransformPoint(p3, lmi.xform);

            bbox_min = min(p1, min(p2, p3));
            bbox_max = max(p1, max(p2, p3));

            Ref::fvec4 light_forward = NS::cross(p2 - p1, p3 - p1);
            area = 0.5f * length(light_forward);

            axis = normalize(light_forward);
            omega_n = l.doublesided ? PI : 0.0f;
            omega_e = PI / 2.0f;
        } break;
        case LIGHT_TYPE_ENV: {
            lum = (lum / 3.0f) * env_map_qtree_.medium_lum;
            bbox_min = Ref::fvec4{-MAX_DIST, -MAX_DIST, -MAX_DIST, 0.0f};
            bbox_max = Ref::fvec4{MAX_DIST, MAX_DIST, MAX_DIST, 0.0f};
            omega_n = PI; // normals in all directions
            omega_e = PI / 2.0f;
        } break;
        default:
            continue;
        }

        primitives.push_back({0, 0, 0, bbox_min, bbox_max});

        const float flux = lum * area;
        additional_data.push_back({axis, flux, omega_n, omega_e});
    }

    li_indices_.Append(new_li_indices.data(), new_li_indices.size());

    light_cwnodes_.Clear();

    if (primitives.empty()) {
        return;
    }

    std::vector<bvh_node_t> temp_nodes;

    std::vector<uint32_t> prim_indices;
    prim_indices.reserve(primitives.size());

    bvh_settings_t s;
    s.oversplit_threshold = -1.0f;
    s.allow_spatial_splits = false;
    s.min_primitives_in_leaf = 1;
    PreprocessPrims_SAH(primitives, {}, s, temp_nodes, prim_indices);

    std::vector<light_bvh_node_t> temp_lnodes(temp_nodes.size(), light_bvh_node_t{});
    for (uint32_t i = 0; i < temp_nodes.size(); ++i) {
        static_cast<bvh_node_t &>(temp_lnodes[i]) = temp_nodes[i];
        if ((temp_nodes[i].prim_index & LEAF_NODE_BIT) != 0) {
            const uint32_t prim_index = prim_indices[temp_nodes[i].prim_index & PRIM_INDEX_BITS];
            memcpy(temp_lnodes[i].axis, value_ptr(additional_data[prim_index].axis), 3 * sizeof(float));
            temp_lnodes[i].flux = additional_data[prim_index].flux;
            temp_lnodes[i].omega_n = additional_data[prim_index].omega_n;
            temp_lnodes[i].omega_e = additional_data[prim_index].omega_e;
        }
    }

    std::vector<uint32_t> parent_indices(temp_lnodes.size());
    parent_indices[0] = 0xffffffff; // root node has no parent

    std::vector<uint32_t> leaf_indices;
    leaf_indices.reserve(primitives.size());

    SmallVector<uint32_t, 128> stack;
    stack.push_back(0);
    while (!stack.empty()) {
        const uint32_t i = stack.back();
        stack.pop_back();

        if ((temp_lnodes[i].prim_index & LEAF_NODE_BIT) == 0) {
            const uint32_t left_child = temp_lnodes[i].left_child,
                           right_child = (temp_lnodes[i].right_child & RIGHT_CHILD_BITS);
            parent_indices[left_child] = parent_indices[right_child] = i;

            stack.push_back(left_child);
            stack.push_back(right_child);
        } else {
            leaf_indices.push_back(i);
        }
    }

    // Propagate flux and cone up the hierarchy
    std::vector<uint32_t> to_process;
    to_process.reserve(temp_lnodes.size());
    to_process.insert(end(to_process), begin(leaf_indices), end(leaf_indices));
    for (uint32_t i = 0; i < uint32_t(to_process.size()); ++i) {
        const uint32_t n = to_process[i];
        const uint32_t parent = parent_indices[n];
        if (parent == 0xffffffff) {
            continue;
        }

        temp_lnodes[parent].flux += temp_lnodes[n].flux;
        if (temp_lnodes[parent].axis[0] == 0.0f && temp_lnodes[parent].axis[1] == 0.0f &&
            temp_lnodes[parent].axis[2] == 0.0f) {
            memcpy(temp_lnodes[parent].axis, temp_lnodes[n].axis, 3 * sizeof(float));
            temp_lnodes[parent].omega_n = temp_lnodes[n].omega_n;
        } else {
            auto axis1 = Ref::fvec4{temp_lnodes[parent].axis}, axis2 = Ref::fvec4{temp_lnodes[n].axis};
            axis1.set<3>(0.0f);
            axis2.set<3>(0.0f);

            const float angle_between = acosf(clamp(dot(axis1, axis2), -1.0f, 1.0f));

            axis1 += axis2;
            const float axis_length = length(axis1);
            if (axis_length != 0.0f) {
                axis1 /= axis_length;
            } else {
                axis1 = Ref::fvec4{0.0f, 1.0f, 0.0f, 0.0f};
            }

            memcpy(temp_lnodes[parent].axis, value_ptr(axis1), 3 * sizeof(float));

            temp_lnodes[parent].omega_n =
                fminf(0.5f * (temp_lnodes[parent].omega_n +
                              fmaxf(temp_lnodes[parent].omega_n, angle_between + temp_lnodes[n].omega_n)),
                      PI);
        }
        temp_lnodes[parent].omega_e = fmaxf(temp_lnodes[parent].omega_e, temp_lnodes[n].omega_e);
        if ((temp_lnodes[parent].left_child & LEFT_CHILD_BITS) == n) {
            to_process.push_back(parent);
        }
    }

    // Remove indices indirection
    for (uint32_t i = 0; i < leaf_indices.size(); ++i) {
        light_bvh_node_t &n = temp_lnodes[leaf_indices[i]];
        assert((n.prim_index & LEAF_NODE_BIT) != 0);
        const uint32_t li_index = new_li_indices[prim_indices[n.prim_index & PRIM_INDEX_BITS]];
        n.prim_index &= ~PRIM_INDEX_BITS;
        n.prim_index |= li_index;
    }

    aligned_vector<light_cwbvh_node_t> temp_light_cwnodes;
    [[maybe_unused]] const uint32_t root_node = FlattenLightBVH_r(temp_lnodes, 0, temp_light_cwnodes);
    assert(root_node == 0);

    // Collapse leaf level (all leafs have only 1 light)
    if ((temp_light_cwnodes[0].child[0] & LEAF_NODE_BIT) != 0) {
        for (int j = 1; j < 8; ++j) {
            temp_light_cwnodes[0].child[j] = 0x7fffffff;
        }
    }
    std::vector<bool> should_remove(temp_light_cwnodes.size(), false);
    for (uint32_t i = 0; i < temp_light_cwnodes.size(); ++i) {
        if ((temp_light_cwnodes[i].child[0] & LEAF_NODE_BIT) == 0) {
            for (int j = 0; j < 8; ++j) {
                if (temp_light_cwnodes[i].child[j] == 0x7fffffff) {
                    continue;
                }
                if ((temp_light_cwnodes[temp_light_cwnodes[i].child[j]].child[0] & LEAF_NODE_BIT) != 0) {
                    assert(temp_light_cwnodes[temp_light_cwnodes[i].child[j]].child[1] == 1);
                    should_remove[temp_light_cwnodes[i].child[j]] = true;
                    temp_light_cwnodes[i].child[j] = temp_light_cwnodes[temp_light_cwnodes[i].child[j]].child[0];
                }
            }
        }
    }
    std::vector<uint32_t> compacted_indices;
    compacted_indices.reserve(should_remove.size());
    uint32_t compacted_count = 0;
    for (const bool b : should_remove) {
        compacted_indices.push_back(compacted_count);
        if (!b) {
            ++compacted_count;
        }
    }
    for (int i = 0; i < int(temp_light_cwnodes.size()); ++i) {
        if (should_remove[i]) {
            continue;
        }
        light_cwbvh_node_t &n = temp_light_cwnodes[i];
        for (int j = 0; j < 8; ++j) {
            if (n.child[j] == 0x7fffffff) {
                continue;
            }
            if ((n.child[j] & LEAF_NODE_BIT) == 0) {
                n.child[j] = compacted_indices[n.child[j] & PRIM_INDEX_BITS];
            }
        }
        temp_light_cwnodes[compacted_indices[i]] = n;
    }
    temp_light_cwnodes.resize(compacted_count);
    light_cwnodes_.Append(temp_light_cwnodes.data(), temp_light_cwnodes.size());
}

inline void Ray::NS::Scene::SetEnvironment(const environment_desc_t &env) {
    SceneCommon::SetEnvironment(env);

    if (!sky_transmittance_lut_tex_) {
        TexParams params;
        params.w = SKY_TRANSMITTANCE_LUT_W;
        params.h = SKY_TRANSMITTANCE_LUT_H;
        params.format = eTexFormat::RGBA32F;
        params.sampling.wrap = eTexWrap::ClampToEdge;
        params.sampling.filter = eTexFilter::Bilinear;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;

        sky_transmittance_lut_tex_ = Texture{"Sky Transmittance LUT", ctx_, params, ctx_->default_mem_allocs(), log_};
    }
    if (!sky_multiscatter_lut_tex_) {
        TexParams params;
        params.w = params.h = SKY_MULTISCATTER_LUT_RES;
        params.format = eTexFormat::RGBA32F;
        params.sampling.wrap = eTexWrap::ClampToEdge;
        params.sampling.filter = eTexFilter::Bilinear;
        params.usage = Bitmask<eTexUsage>(eTexUsage::Sampled) | eTexUsage::Transfer;

        sky_multiscatter_lut_tex_ = Texture{"Sky Multiscatter LUT", ctx_, params, ctx_->default_mem_allocs(), log_};
    }

    // Upload textures
    if (!sky_transmittance_lut_.empty()) {
        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload,
                                  4 * SKY_TRANSMITTANCE_LUT_W * SKY_TRANSMITTANCE_LUT_H * sizeof(float));
        uint8_t *mapped_ptr = stage_buf.Map();
        memcpy(mapped_ptr, sky_transmittance_lut_.data(),
               4 * SKY_TRANSMITTANCE_LUT_W * SKY_TRANSMITTANCE_LUT_H * sizeof(float));
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_transmittance_lut_tex_.SetSubImage(0, 0, 0, 0, SKY_TRANSMITTANCE_LUT_W, SKY_TRANSMITTANCE_LUT_H, 1,
                                               eTexFormat::RGBA32F, stage_buf, cmd_buf, 0, stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }
    if (!sky_multiscatter_lut_.empty()) {
        Buffer stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Upload,
                                  4 * SKY_MULTISCATTER_LUT_RES * SKY_MULTISCATTER_LUT_RES * sizeof(float));
        uint8_t *mapped_ptr = stage_buf.Map();
        memcpy(mapped_ptr, sky_multiscatter_lut_.data(),
               4 * SKY_MULTISCATTER_LUT_RES * SKY_MULTISCATTER_LUT_RES * sizeof(float));
        stage_buf.Unmap();

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        sky_multiscatter_lut_tex_.SetSubImage(0, 0, 0, 0, SKY_MULTISCATTER_LUT_RES, SKY_MULTISCATTER_LUT_RES, 1,
                                              eTexFormat::RGBA32F, stage_buf, cmd_buf, 0, stage_buf.size());
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        stage_buf.FreeImmediate();
    }
}