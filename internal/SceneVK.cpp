#include "SceneVK.h"

#include <cassert>

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#include "../Log.h"
#include "BVHSplit.h"
#include "TextureUtilsRef.h"
#include "Utils.h"
#include "Vk/Context.h"
#include "Vk/TextureParams.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) < (y) ? (y) : (x))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define CLAMP(x, lo, hi) (MIN(MAX((x), (lo)), (hi)))

namespace Ray {
uint32_t next_power_of_two(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void to_khr_xform(const float xform[16], float matrix[3][4]) {
    // transpose
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix[i][j] = xform[4 * j + i];
        }
    }
}

namespace Vk {
VkDeviceSize align_up(const VkDeviceSize size, const VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}
} // namespace Vk
} // namespace Ray

Ray::Vk::Scene::Scene(Context *ctx, const bool use_hwrt, const bool use_bindless, const bool use_tex_compression)
    : ctx_(ctx), use_hwrt_(use_hwrt), use_bindless_(use_bindless), use_tex_compression_(use_tex_compression),
      nodes_(ctx), tris_(ctx), tri_indices_(ctx), tri_materials_(ctx), transforms_(ctx, "Transforms"),
      meshes_(ctx, "Meshes"), mesh_instances_(ctx, "Mesh Instances"), mi_indices_(ctx), vertices_(ctx),
      vtx_indices_(ctx), materials_(ctx, "Materials"), atlas_textures_(ctx, "Atlas Textures"), bindless_tex_data_{ctx},
      tex_atlases_{{ctx, eTexFormat::RawRGBA8888, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::RawRGB888, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::RawRG88, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::RawR8, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::BC3, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::BC4, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                   {ctx, eTexFormat::BC5, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE}},
      lights_(ctx, "Lights"), li_indices_(ctx), visible_lights_(ctx), blocker_lights_(ctx) {}

Ray::Vk::Scene::~Scene() {
    bindless_textures_.clear();
    vkDestroyDescriptorSetLayout(ctx_->device(), bindless_tex_data_.descr_layout, nullptr);
}

void Ray::Vk::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(env.env_col, env_.env_col, 3 * sizeof(float));
    env.env_map = TextureHandle{env_.env_map};
    memcpy(env.back_col, env_.back_col, 3 * sizeof(float));
    env.back_map = TextureHandle{env_.back_map};
    env.env_map_rotation = env_.env_map_rotation;
    env.back_map_rotation = env_.back_map_rotation;
    env.multiple_importance = env_.multiple_importance;
}

void Ray::Vk::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(env_.env_col, env.env_col, 3 * sizeof(float));
    env_.env_map = env.env_map._index;
    memcpy(env_.back_col, env.back_col, 3 * sizeof(float));
    env_.back_map = env.back_map._index;
    env_.env_map_rotation = env.env_map_rotation;
    env_.back_map_rotation = env.back_map_rotation;
    env_.multiple_importance = env.multiple_importance;
}

Ray::TextureHandle Ray::Vk::Scene::AddAtlasTexture(const tex_desc_t &_t) {
    atlas_texture_t t;
    t.width = uint16_t(_t.w);
    t.height = uint16_t(_t.h);

    if (_t.is_srgb) {
        t.width |= ATLAS_TEX_SRGB_BIT;
    }

    if (_t.generate_mipmaps) {
        t.height |= ATLAS_TEX_MIPS_BIT;
    }

    int res[2] = {_t.w, _t.h};

    const bool use_compression = use_tex_compression_ && !_t.force_no_compression;

    std::unique_ptr<color_rg8_t[]> repacked_normalmap(new color_rg8_t[res[0] * res[1]]);
    bool recostruct_z = false;

    const void *tex_data = _t.data;

    if (_t.format == eTextureFormat::RGBA8888) {
        if (!_t.is_normalmap) {
            t.atlas = 0;
        } else {
            // TODO: get rid of this allocation
            repacked_normalmap.reset(new color_rg8_t[res[0] * res[1]]);

            const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data);
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_normalmap[i].v[0] = rgba_data[i].v[0];
                repacked_normalmap[i].v[1] = rgba_data[i].v[1];
                recostruct_z |= (rgba_data[i].v[2] < 250);
            }

            tex_data = repacked_normalmap.get();
            t.atlas = use_compression ? 6 : 2;
        }
    } else if (_t.format == eTextureFormat::RGB888) {
        if (!_t.is_normalmap) {
            t.atlas = use_compression ? 4 : 1;
        } else {
            // TODO: get rid of this allocation
            repacked_normalmap.reset(new color_rg8_t[res[0] * res[1]]);

            const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data);
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_normalmap[i].v[0] = rgb_data[i].v[0];
                repacked_normalmap[i].v[1] = rgb_data[i].v[1];
                recostruct_z |= (rgb_data[i].v[2] < 250);
            }

            tex_data = repacked_normalmap.get();
            t.atlas = use_compression ? 6 : 2;
        }
    } else if (_t.format == eTextureFormat::RG88) {
        t.atlas = use_compression ? 6 : 2;
    } else if (_t.format == eTextureFormat::R8) {
        t.atlas = use_compression ? 5 : 3;
    }

    if (recostruct_z) {
        t.width |= uint32_t(ATLAS_TEX_RECONSTRUCT_Z_BIT);
    }

    { // Allocate initial mip level
        int page = -1, pos[2];
        if (t.atlas == 0) {
            page = tex_atlases_[0].Allocate<uint8_t, 4>(reinterpret_cast<const color_rgba8_t *>(tex_data), res, pos);
        } else if (t.atlas == 1 || t.atlas == 4) {
            page =
                tex_atlases_[t.atlas].Allocate<uint8_t, 3>(reinterpret_cast<const color_rgb8_t *>(tex_data), res, pos);
        } else if (t.atlas == 2 || t.atlas == 6) {
            page =
                tex_atlases_[t.atlas].Allocate<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(tex_data), res, pos);
            page = tex_atlases_[2].Allocate<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(tex_data), res, pos);
        } else if (t.atlas == 3 || t.atlas == 5) {
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
    for (int i = 1; i < NUM_MIP_LEVELS; i++) {
        t.page[i] = t.page[0];
        t.pos[i][0] = t.pos[0][0];
        t.pos[i][1] = t.pos[0][1];
    }

    if (_t.generate_mipmaps && use_compression) {
        // We have to generate mips here as uncompressed data will be lost

        int pages[16], positions[16][2];
        if (_t.format == eTextureFormat::RGB888) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 3>(reinterpret_cast<const color_rgb8_t *>(_t.data), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else if (_t.format == eTextureFormat::RG88) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(_t.data), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else if (_t.format == eTextureFormat::R8) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 1>(reinterpret_cast<const color_r8_t *>(_t.data), res,
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

    ctx_->log()->Info("Ray: Texture loaded (atlas = %i, %ix%i)", int(t.atlas), _t.w, _t.h);
    ctx_->log()->Info("Ray: Atlasses are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC3[%i], BC4[%i], BC5[%i])",
                      tex_atlases_[0].page_count(), tex_atlases_[1].page_count(), tex_atlases_[2].page_count(),
                      tex_atlases_[3].page_count(), tex_atlases_[4].page_count(), tex_atlases_[5].page_count(),
                      tex_atlases_[6].page_count());

    return TextureHandle{atlas_textures_.push(t)};
}

Ray::TextureHandle Ray::Vk::Scene::AddBindlessTexture(const tex_desc_t &_t) {
    eTexFormat src_fmt = eTexFormat::Undefined, fmt = eTexFormat::Undefined;

    Buffer temp_stage_buf("Temp stage buf", ctx_, eBufType::Stage, 2 * _t.w * _t.h * 4,
                          4096); // allocate for worst case
    uint8_t *stage_data = temp_stage_buf.Map(BufMapWrite);

    const bool use_compression = use_tex_compression_ && !_t.force_no_compression;

    uint32_t data_size[16] = {};

    std::unique_ptr<uint8_t[]> repacked_data;
    bool recostruct_z = false, is_YCoCg = false;

    if (_t.format == eTextureFormat::RGBA8888) {
        if (!_t.is_normalmap) {
            src_fmt = fmt = eTexFormat::RawRGBA8888;
            data_size[0] = _t.w * _t.h * 4;
            memcpy(stage_data, _t.data, data_size[0]);
        } else {
            // TODO: get rid of this allocation
            repacked_data.reset(new uint8_t[2 * _t.w * _t.h]);

            const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data);
            for (int i = 0; i < _t.w * _t.h; ++i) {
                repacked_data[i * 2 + 0] = rgba_data[i].v[0];
                repacked_data[i * 2 + 1] = rgba_data[i].v[1];
                recostruct_z |= (rgba_data[i].v[2] < 250);
            }

            if (use_compression) {
                src_fmt = eTexFormat::RawRG88;
                fmt = eTexFormat::BC5;
                data_size[0] = GetRequiredMemory_BC5(_t.w, _t.h);
                CompressImage_BC5<2>(&repacked_data[0], _t.w, _t.h, stage_data);
            } else {
                src_fmt = fmt = eTexFormat::RawRG88;
                data_size[0] = _t.w * _t.h * 2;
                memcpy(stage_data, _t.data, data_size[0]);
            }
        }
    } else if (_t.format == eTextureFormat::RGB888) {
        if (!_t.is_normalmap) {
            if (use_compression) {
                auto temp_YCoCg = ConvertRGB_to_CoCgxY(reinterpret_cast<const uint8_t *>(_t.data), _t.w, _t.h);
                is_YCoCg = true;
                src_fmt = eTexFormat::RawRGB888;
                fmt = eTexFormat::BC3;
                data_size[0] = GetRequiredMemory_BC3(_t.w, _t.h);
                CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), _t.w, _t.h, stage_data);
            } else if (ctx_->rgb8_unorm_is_supported()) {
                src_fmt = fmt = eTexFormat::RawRGB888;
                data_size[0] = _t.w * _t.h * 3;
                memcpy(stage_data, _t.data, data_size[0]);
            } else {
                // Fallback to 4-component texture
                src_fmt = fmt = eTexFormat::RawRGBA8888;
                data_size[0] = _t.w * _t.h * 4;

                // TODO: get rid of this allocation
                repacked_data.reset(new uint8_t[4 * _t.w * _t.h]);

                const auto *rgb_data = reinterpret_cast<const uint8_t *>(_t.data);
                for (int i = 0; i < _t.w * _t.h; ++i) {
                    repacked_data[i * 4 + 0] = rgb_data[i * 3 + 0];
                    repacked_data[i * 4 + 1] = rgb_data[i * 3 + 1];
                    repacked_data[i * 4 + 2] = rgb_data[i * 3 + 2];
                    repacked_data[i * 4 + 3] = 255;
                }

                memcpy(stage_data, repacked_data.get(), data_size[0]);
            }
        } else {
            // TODO: get rid of this allocation
            repacked_data.reset(new uint8_t[2 * _t.w * _t.h]);

            const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data);
            for (int i = 0; i < _t.w * _t.h; ++i) {
                repacked_data[i * 2 + 0] = rgb_data[i].v[0];
                repacked_data[i * 2 + 1] = rgb_data[i].v[1];
                recostruct_z |= (rgb_data[i].v[2] < 250);
            }

            if (use_compression) {
                src_fmt = eTexFormat::RawRG88;
                fmt = eTexFormat::BC5;
                data_size[0] = GetRequiredMemory_BC5(_t.w, _t.h);
                CompressImage_BC5<2>(&repacked_data[0], _t.w, _t.h, stage_data);
            } else {
                src_fmt = fmt = eTexFormat::RawRG88;
                data_size[0] = _t.w * _t.h * 2;
                memcpy(stage_data, repacked_data.get(), data_size[0]);
            }
        }
    } else if (_t.format == eTextureFormat::RG88) {
        src_fmt = fmt = eTexFormat::RawRG88;
        data_size[0] = _t.w * _t.h * 2;
        memcpy(stage_data, _t.data, data_size[0]);
    } else if (_t.format == eTextureFormat::R8) {
        if (use_compression) {
            src_fmt = eTexFormat::RawR8;
            fmt = eTexFormat::BC4;
            data_size[0] = GetRequiredMemory_BC4(_t.w, _t.h);
            CompressImage_BC4<1>(reinterpret_cast<const uint8_t *>(_t.data), _t.w, _t.h, stage_data);
        } else {
            src_fmt = fmt = eTexFormat::RawR8;
            data_size[0] = _t.w * _t.h;
            memcpy(stage_data, _t.data, data_size[0]);
        }
    }

    int mip_count = 1;
    if (_t.generate_mipmaps) {
        mip_count = CalcMipCount(_t.w, _t.h, 1, eTexFilter::Bilinear);

        const int res[2] = {_t.w, _t.h};
        if (src_fmt == eTexFormat::RawRGBA8888) {
            const auto *rgba_data =
                reinterpret_cast<const color_rgba8_t *>(repacked_data ? repacked_data.get() : _t.data);
            WriteTextureMips(rgba_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::RawRGB888) {
            const auto *rgb_data =
                reinterpret_cast<const color_rgb8_t *>(repacked_data ? repacked_data.get() : _t.data);
            WriteTextureMips(rgb_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::RawRG88) {
            const auto *rg_data = reinterpret_cast<const color_rg8_t *>(repacked_data ? repacked_data.get() : _t.data);
            WriteTextureMips(rg_data, res, mip_count, use_compression, stage_data, data_size);
        } else if (src_fmt == eTexFormat::RawR8) {
            const auto *r_data = reinterpret_cast<const color_r8_t *>(repacked_data ? repacked_data.get() : _t.data);
            WriteTextureMips(r_data, res, mip_count, use_compression, stage_data, data_size);
        }
    }

    temp_stage_buf.FlushMappedRange(0, temp_stage_buf.size(), true);
    temp_stage_buf.Unmap();

    Tex2DParams p = {};
    p.w = _t.w;
    p.h = _t.h;
    if (_t.is_srgb && !is_YCoCg && fmt != eTexFormat::BC4) {
        p.flags |= eTexFlagBits::SRGB;
    }
    p.mip_count = mip_count;
    p.usage = eTexUsageBits::Transfer | eTexUsageBits::Sampled;
    p.format = fmt;
    p.sampling.filter = eTexFilter::Bilinear;

    uint32_t ret = bindless_textures_.emplace(_t.name ? _t.name : "Bindless Tex", ctx_, p,
                                              ctx_->default_memory_allocs(), ctx_->log());

    { // Submit GPU commands
        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        int res[2] = {_t.w, _t.h};
        uint32_t data_offset = 0;
        for (int i = 0; i < p.mip_count; ++i) {
            bindless_textures_[ret].SetSubImage(i, 0, 0, res[0], res[1], fmt, temp_stage_buf, cmd_buf, data_offset,
                                                data_size[i]);
            res[0] = MAX(res[0] / 2, 1);
            res[1] = MAX(res[1] / 2, 1);
            data_offset += 4096 * ((data_size[i] + 4095) / 4096);
        }

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    temp_stage_buf.FreeImmediate();

    ctx_->log()->Info("Ray: Texture loaded (%ix%i)", _t.w, _t.h);

    assert(ret <= 0x00ffffffff);

    if (_t.is_srgb && (is_YCoCg || fmt == eTexFormat::BC4)) {
        ret |= TEX_SRGB_BIT;
    }
    if (recostruct_z) {
        ret |= TEX_RECONSTRUCT_Z_BIT;
    }
    if (is_YCoCg) {
        ret |= TEX_YCOCG_BIT;
    }

    return TextureHandle{ret};
}

template <typename T, int N>
void Ray::Vk::Scene::WriteTextureMips(const color_t<T, N> data[], const int _res[2], const int mip_count,
                                      const bool compress, uint8_t out_data[], uint32_t out_size[16]) {
    int src_res[2] = {_res[0], _res[1]};

    // TODO: try to get rid of these allocations
    std::vector<color_t<T, N>> _src_data, dst_data;
    for (int i = 1; i < mip_count; ++i) {
        const int dst_res[2] = {MAX(src_res[0] / 2, 1), MAX(src_res[1] / 2, 1)};

        dst_data.clear();
        dst_data.reserve(dst_res[0] * dst_res[1]);

        const color_t<T, N> *src_data = (i == 1) ? data : _src_data.data();

        for (int y = 0; y < dst_res[1]; ++y) {
            for (int x = 0; x < dst_res[0]; ++x) {
                const color_t<T, N> c00 = src_data[(2 * y + 0) * src_res[0] + (2 * x + 0)];
                const color_t<T, N> c10 = src_data[(2 * y + 0) * src_res[0] + MIN(2 * x + 1, src_res[0] - 1)];
                const color_t<T, N> c11 =
                    src_data[MIN(2 * y + 1, src_res[1] - 1) * src_res[0] + MIN(2 * x + 1, src_res[0] - 1)];
                const color_t<T, N> c01 = src_data[MIN(2 * y + 1, src_res[1] - 1) * src_res[0] + (2 * x + 0)];

                color_t<T, N> res;
                for (int j = 0; j < N; ++j) {
                    res.v[j] = (c00.v[j] + c10.v[j] + c11.v[j] + c01.v[j]) / 4;
                }

                dst_data.push_back(res);
            }
        }

        assert(dst_data.size() == (dst_res[0] * dst_res[1]));

        out_data += 4096 * ((out_size[i - 1] + 4095) / 4096);
        if (compress) {
            if (N == 3) {
                auto temp_YCoCg = ConvertRGB_to_CoCgxY(&dst_data[0].v[0], dst_res[0], dst_res[1]);

                out_size[i] = GetRequiredMemory_BC3(dst_res[0], dst_res[1]);
                CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), dst_res[0], dst_res[1], out_data);
            } else if (N == 1) {
                out_size[i] = GetRequiredMemory_BC4(dst_res[0], dst_res[1]);
                CompressImage_BC4<N>(&dst_data[0].v[0], dst_res[0], dst_res[1], out_data);
            } else if (N == 2) {
                out_size[i] = GetRequiredMemory_BC5(dst_res[0], dst_res[1]);
                CompressImage_BC5<2>(&dst_data[0].v[0], dst_res[0], dst_res[1], out_data);
            }
        } else {
            out_size[i] = int(dst_data.size() * sizeof(color_t<T, N>));
            memcpy(out_data, dst_data.data(), out_size[i]);
        }

        src_res[0] = dst_res[0];
        src_res[1] = dst_res[1];
        std::swap(_src_data, dst_data);
    }
}

template void Ray::Vk::Scene::WriteTextureMips<uint8_t, 1>(const color_t<uint8_t, 1> data[], const int _res[2],
                                                           int mip_count, bool compress, uint8_t out_data[],
                                                           uint32_t out_size[16]);
template void Ray::Vk::Scene::WriteTextureMips<uint8_t, 2>(const color_t<uint8_t, 2> data[], const int _res[2],
                                                           int mip_count, bool compress, uint8_t out_data[],
                                                           uint32_t out_size[16]);
template void Ray::Vk::Scene::WriteTextureMips<uint8_t, 3>(const color_t<uint8_t, 3> data[], const int _res[2],
                                                           int mip_count, bool compress, uint8_t out_data[],
                                                           uint32_t out_size[16]);
template void Ray::Vk::Scene::WriteTextureMips<uint8_t, 4>(const color_t<uint8_t, 4> data[], const int _res[2],
                                                           int mip_count, bool compress, uint8_t out_data[],
                                                           uint32_t out_size[16]);

Ray::MaterialHandle Ray::Vk::Scene::AddMaterial(const shading_node_desc_t &m) {
    material_t mat;

    mat.type = m.type;
    mat.textures[BASE_TEXTURE] = m.base_texture._index;
    mat.roughness_unorm = pack_unorm_16(m.roughness);
    mat.textures[ROUGH_TEXTURE] = m.roughness_texture._index;
    memcpy(&mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    mat.ior = m.ior;
    mat.tangent_rotation = 0.0f;
    mat.flags = 0;

    if (m.type == DiffuseNode) {
        mat.sheen_unorm = pack_unorm_16(CLAMP(0.5f * m.sheen, 0.0f, 1.0f));
        mat.sheen_tint_unorm = pack_unorm_16(CLAMP(m.tint, 0.0f, 1.0f));
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
    } else if (m.type == GlossyNode) {
        mat.tangent_rotation = 2.0f * PI * m.anisotropic_rotation;
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
        mat.tint_unorm = pack_unorm_16(CLAMP(m.tint, 0.0f, 1.0f));
    } else if (m.type == RefractiveNode) {
    } else if (m.type == EmissiveNode) {
        mat.strength = m.strength;
        if (m.multiple_importance) {
            mat.flags |= MAT_FLAG_MULT_IMPORTANCE;
        }
    } else if (m.type == MixNode) {
        mat.strength = m.strength;
        mat.textures[MIX_MAT1] = m.mix_materials[0]._index;
        mat.textures[MIX_MAT2] = m.mix_materials[1]._index;
        if (m.mix_add) {
            mat.flags |= MAT_FLAG_MIX_ADD;
        }
    } else if (m.type == TransparentNode) {
    }

    mat.textures[NORMALS_TEXTURE] = m.normal_map._index;
    mat.normal_map_strength_unorm = pack_unorm_16(CLAMP(m.normal_map_intensity, 0.0f, 1.0f));

    return MaterialHandle{materials_.push(mat)};
}

Ray::MaterialHandle Ray::Vk::Scene::AddMaterial(const principled_mat_desc_t &m) {
    material_t main_mat;

    main_mat.type = PrincipledNode;
    main_mat.textures[BASE_TEXTURE] = m.base_texture._index;
    memcpy(&main_mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    main_mat.sheen_unorm = pack_unorm_16(CLAMP(0.5f * m.sheen, 0.0f, 1.0f));
    main_mat.sheen_tint_unorm = pack_unorm_16(CLAMP(m.sheen_tint, 0.0f, 1.0f));
    main_mat.roughness_unorm = pack_unorm_16(CLAMP(m.roughness, 0.0f, 1.0f));
    main_mat.tangent_rotation = 2.0f * PI * CLAMP(m.anisotropic_rotation, 0.0f, 1.0f);
    main_mat.textures[ROUGH_TEXTURE] = m.roughness_texture._index;
    main_mat.metallic_unorm = pack_unorm_16(CLAMP(m.metallic, 0.0f, 1.0f));
    main_mat.textures[METALLIC_TEXTURE] = m.metallic_texture._index;
    main_mat.ior = m.ior;
    main_mat.flags = 0;
    main_mat.transmission_unorm = pack_unorm_16(CLAMP(m.transmission, 0.0f, 1.0f));
    main_mat.transmission_roughness_unorm = pack_unorm_16(CLAMP(m.transmission_roughness, 0.0f, 1.0f));
    main_mat.textures[NORMALS_TEXTURE] = m.normal_map._index;
    main_mat.normal_map_strength_unorm = pack_unorm_16(CLAMP(m.normal_map_intensity, 0.0f, 1.0f));
    main_mat.anisotropic_unorm = pack_unorm_16(CLAMP(m.anisotropic, 0.0f, 1.0f));
    main_mat.specular_unorm = pack_unorm_16(CLAMP(m.specular, 0.0f, 1.0f));
    main_mat.textures[SPECULAR_TEXTURE] = m.specular_texture._index;
    main_mat.specular_tint_unorm = pack_unorm_16(CLAMP(m.specular_tint, 0.0f, 1.0f));
    main_mat.clearcoat_unorm = pack_unorm_16(CLAMP(m.clearcoat, 0.0f, 1.0f));
    main_mat.clearcoat_roughness_unorm = pack_unorm_16(CLAMP(m.clearcoat_roughness, 0.0f, 1.0f));

    auto root_node = MaterialHandle{materials_.push(main_mat)};
    MaterialHandle emissive_node = InvalidMaterialHandle, transparent_node = InvalidMaterialHandle;

    if (m.emission_strength > 0.0f &&
        (m.emission_color[0] > 0.0f || m.emission_color[1] > 0.0f || m.emission_color[2] > 0.0f)) {
        shading_node_desc_t emissive_desc;
        emissive_desc.type = EmissiveNode;

        memcpy(emissive_desc.base_color, m.emission_color, 3 * sizeof(float));
        emissive_desc.base_texture = m.emission_texture;
        emissive_desc.strength = m.emission_strength;

        emissive_node = AddMaterial(emissive_desc);
    }

    if (m.alpha != 1.0f || m.alpha_texture != InvalidTextureHandle) {
        shading_node_desc_t transparent_desc;
        transparent_desc.type = TransparentNode;

        transparent_node = AddMaterial(transparent_desc);
    }

    if (emissive_node != InvalidMaterialHandle) {
        if (root_node == InvalidMaterialHandle) {
            root_node = emissive_node;
        } else {
            shading_node_desc_t mix_node;
            mix_node.type = MixNode;
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
            mix_node.type = MixNode;
            mix_node.base_texture = m.alpha_texture;
            mix_node.strength = m.alpha;
            mix_node.ior = 0.0f;

            mix_node.mix_materials[0] = transparent_node;
            mix_node.mix_materials[1] = root_node;

            root_node = AddMaterial(mix_node);
        }
    }

    return MaterialHandle{root_node};
}

Ray::MeshHandle Ray::Vk::Scene::AddMesh(const mesh_desc_t &_m) {
    std::vector<bvh_node_t> new_nodes;
    std::vector<tri_accel_t> new_tris;
    std::vector<uint32_t> new_tri_indices;
    std::vector<uint32_t> new_vtx_indices;

    bvh_settings_t s;
    s.allow_spatial_splits = _m.allow_spatial_splits;
    s.use_fast_bvh_build = _m.use_fast_bvh_build;

    Ref::simd_fvec4 bbox_min{std::numeric_limits<float>::max()}, bbox_max{std::numeric_limits<float>::lowest()};

    const size_t attr_stride = AttrStrides[_m.layout];
    if (use_hwrt_) {
        for (int j = 0; j < int(_m.vtx_indices_count); j += 3) {
            Ref::simd_fvec4 p[3];

            const uint32_t i0 = _m.vtx_indices[j + 0], i1 = _m.vtx_indices[j + 1], i2 = _m.vtx_indices[j + 2];

            memcpy(value_ptr(p[0]), &_m.vtx_attrs[i0 * attr_stride], 3 * sizeof(float));
            memcpy(value_ptr(p[1]), &_m.vtx_attrs[i1 * attr_stride], 3 * sizeof(float));
            memcpy(value_ptr(p[2]), &_m.vtx_attrs[i2 * attr_stride], 3 * sizeof(float));

            bbox_min = min(bbox_min, min(p[0], min(p[1], p[2])));
            bbox_max = max(bbox_max, max(p[0], max(p[1], p[2])));
        }
    } else {
        aligned_vector<mtri_accel_t> _unused;
        PreprocessMesh(_m.vtx_attrs, {_m.vtx_indices, _m.vtx_indices_count}, _m.layout, _m.base_vertex,
                       0 /* temp value */, s, new_nodes, new_tris, new_tri_indices, _unused);

        memcpy(value_ptr(bbox_min), new_nodes[0].bbox_min, 3 * sizeof(float));
        memcpy(value_ptr(bbox_max), new_nodes[0].bbox_max, 3 * sizeof(float));
    }

    std::vector<tri_mat_data_t> new_tri_materials(_m.vtx_indices_count / 3);

    // init triangle materials
    for (const shape_desc_t &sh : _m.shapes) {
        bool is_front_solid = true, is_back_solid = true;

        uint32_t material_stack[32];
        material_stack[0] = sh.front_mat._index;
        uint32_t material_count = 1;

        while (material_count) {
            const material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == MixNode) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == TransparentNode) {
                is_front_solid = false;
                break;
            }
        }

        material_stack[0] = sh.back_mat._index;
        material_count = 1;

        while (material_count) {
            const material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == MixNode) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == TransparentNode) {
                is_back_solid = false;
                break;
            }
        }

        for (size_t i = sh.vtx_start; i < sh.vtx_start + sh.vtx_count; i += 3) {
            tri_mat_data_t &tri_mat = new_tri_materials[i / 3];

            assert(sh.front_mat._index < (1 << 14) && "Not enough bits to reference material!");
            assert(sh.back_mat._index < (1 << 14) && "Not enough bits to reference material!");

            tri_mat.front_mi = uint16_t(sh.front_mat._index);
            if (is_front_solid) {
                tri_mat.front_mi |= MATERIAL_SOLID_BIT;
            }

            tri_mat.back_mi = uint16_t(sh.back_mat._index);
            if (is_back_solid) {
                tri_mat.back_mi |= MATERIAL_SOLID_BIT;
            }
        }
    }

    for (size_t i = 0; i < _m.vtx_indices_count; i++) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + _m.base_vertex + uint32_t(vertices_.size()));
    }

    // offset nodes and primitives
    for (bvh_node_t &n : new_nodes) {
        if (n.prim_index & LEAF_NODE_BIT) {
            n.prim_index += uint32_t(tri_indices_.size());
        } else {
            n.left_child += uint32_t(nodes_.size());
            n.right_child += uint32_t(nodes_.size());
        }
    }

    // offset triangle indices
    for (uint32_t &i : new_tri_indices) {
        i += uint32_t(tri_materials_.size());
    }

    tri_materials_.Append(&new_tri_materials[0], new_tri_materials.size());
    tri_materials_cpu_.insert(tri_materials_cpu_.end(), &new_tri_materials[0],
                              &new_tri_materials[0] + new_tri_materials.size());

    // add mesh
    mesh_t m = {};
    memcpy(m.bbox_min, value_ptr(bbox_min), 3 * sizeof(float));
    memcpy(m.bbox_max, value_ptr(bbox_max), 3 * sizeof(float));
    m.node_index = uint32_t(nodes_.size());
    m.node_count = uint32_t(new_nodes.size());
    m.tris_index = uint32_t(tris_.size());
    m.tris_count = uint32_t(new_tris.size());
    m.vert_index = uint32_t(vtx_indices_.size());
    m.vert_count = uint32_t(new_vtx_indices.size());

    const uint32_t mesh_index = meshes_.push(m);

    if (!use_hwrt_) {
        // add nodes
        nodes_.Append(&new_nodes[0], new_nodes.size());
    }

    const size_t stride = AttrStrides[_m.layout];

    // add attributes
    std::vector<vertex_t> new_vertices(_m.vtx_attrs_count);
    for (size_t i = 0; i < _m.vtx_attrs_count; ++i) {
        vertex_t &v = new_vertices[i];

        memcpy(&v.p[0], (_m.vtx_attrs + i * stride), 3 * sizeof(float));
        memcpy(&v.n[0], (_m.vtx_attrs + i * stride + 3), 3 * sizeof(float));

        if (_m.layout == PxyzNxyzTuv) {
            memcpy(&v.t[0][0], (_m.vtx_attrs + i * stride + 6), 2 * sizeof(float));
            v.t[1][0] = v.t[1][1] = 0.0f;
            v.b[0] = v.b[1] = v.b[2] = 0.0f;
        } else if (_m.layout == PxyzNxyzTuvTuv) {
            memcpy(&v.t[0][0], (_m.vtx_attrs + i * stride + 6), 2 * sizeof(float));
            memcpy(&v.t[1][0], (_m.vtx_attrs + i * stride + 8), 2 * sizeof(float));
            v.b[0] = v.b[1] = v.b[2] = 0.0f;
        } else if (_m.layout == PxyzNxyzBxyzTuv) {
            memcpy(&v.b[0], (_m.vtx_attrs + i * stride + 6), 3 * sizeof(float));
            memcpy(&v.t[0][0], (_m.vtx_attrs + i * stride + 9), 2 * sizeof(float));
            v.t[1][0] = v.t[1][1] = 0.0f;
        } else if (_m.layout == PxyzNxyzBxyzTuvTuv) {
            memcpy(&v.b[0], (_m.vtx_attrs + i * stride + 6), 3 * sizeof(float));
            memcpy(&v.t[0][0], (_m.vtx_attrs + i * stride + 9), 2 * sizeof(float));
            memcpy(&v.t[1][0], (_m.vtx_attrs + i * stride + 11), 2 * sizeof(float));
        }
    }

    if (_m.layout == PxyzNxyzTuv || _m.layout == PxyzNxyzTuvTuv) {
        Ref::ComputeTangentBasis(vertices_.size(), 0, new_vertices, new_vtx_indices, _m.vtx_indices,
                                 _m.vtx_indices_count);
    }

    vertices_.Append(&new_vertices[0], new_vertices.size());

    // add vertex indices
    vtx_indices_.Append(&new_vtx_indices[0], new_vtx_indices.size());

    if (!use_hwrt_) {
        // add triangles
        tris_.Append(&new_tris[0], new_tris.size());
        // add triangle indices
        tri_indices_.Append(&new_tri_indices[0], new_tri_indices.size());
    }

    return MeshHandle{mesh_index};
}

void Ray::Vk::Scene::RemoveMesh(MeshHandle) {
    // TODO!!!
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const directional_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_DIR;
    l.cast_shadow = _l.cast_shadow;
    l.visible = false;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    l.dir.dir[0] = -_l.direction[0];
    l.dir.dir[1] = -_l.direction[1];
    l.dir.dir[2] = -_l.direction[2];
    l.dir.angle = _l.angle * PI / 360.0f;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    return LightHandle{light_index};
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const sphere_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;
    l.sph.spot = l.sph.blend = -1.0f;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);

    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    return LightHandle{light_index};
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const spot_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));
    memcpy(&l.sph.dir[0], &_l.direction[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;
    l.sph.spot = 0.5f * PI * _l.spot_size / 180.0f;
    l.sph.blend = _l.spot_blend * _l.spot_blend;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    return LightHandle{light_index};
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const rect_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_RECT;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.rect.pos[0] = xform[12];
    l.rect.pos[1] = xform[13];
    l.rect.pos[2] = xform[14];

    l.rect.area = _l.width * _l.height;

    const Ref::simd_fvec4 uvec = _l.width * Ref::TransformDirection(Ref::simd_fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::simd_fvec4 vvec = _l.height * Ref::TransformDirection(Ref::simd_fvec4{0.0f, 0.0f, 1.0f, 0.0f}, xform);

    memcpy(l.rect.u, value_ptr(uvec), 3 * sizeof(float));
    memcpy(l.rect.v, value_ptr(vvec), 3 * sizeof(float));

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    if (_l.sky_portal) {
        blocker_lights_.PushBack(light_index);
    }
    return LightHandle{light_index};
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const disk_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_DISK;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.disk.pos[0] = xform[12];
    l.disk.pos[1] = xform[13];
    l.disk.pos[2] = xform[14];

    l.disk.area = 0.25f * PI * _l.size_x * _l.size_y;

    const Ref::simd_fvec4 uvec = _l.size_x * Ref::TransformDirection(Ref::simd_fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::simd_fvec4 vvec = _l.size_y * Ref::TransformDirection(Ref::simd_fvec4{0.0f, 0.0f, 1.0f, 0.0f}, xform);

    memcpy(l.disk.u, value_ptr(uvec), 3 * sizeof(float));
    memcpy(l.disk.v, value_ptr(vvec), 3 * sizeof(float));

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    if (_l.sky_portal) {
        blocker_lights_.PushBack(light_index);
    }
    return LightHandle{light_index};
}

Ray::LightHandle Ray::Vk::Scene::AddLight(const line_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_LINE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));

    l.line.pos[0] = xform[12];
    l.line.pos[1] = xform[13];
    l.line.pos[2] = xform[14];

    l.line.area = 2.0f * PI * _l.radius * _l.height;

    const Ref::simd_fvec4 uvec = TransformDirection(Ref::simd_fvec4{1.0f, 0.0f, 0.0f, 0.0f}, xform);
    const Ref::simd_fvec4 vvec = TransformDirection(Ref::simd_fvec4{0.0f, 1.0f, 0.0f, 0.0f}, xform);

    memcpy(l.line.u, value_ptr(uvec), 3 * sizeof(float));
    l.line.radius = _l.radius;
    memcpy(l.line.v, value_ptr(vvec), 3 * sizeof(float));
    l.line.height = _l.height;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    return LightHandle{light_index};
}

void Ray::Vk::Scene::RemoveLight(const LightHandle i) {
    if (!lights_.exists(i._index)) {
        return;
    }

    //{ // remove from compacted list
    //    auto it = find(begin(li_indices_), end(li_indices_), i);
    //    assert(it != end(li_indices_));
    //    li_indices_.erase(it);
    //}

    // if (lights_[i].visible) {
    //     auto it = find(begin(visible_lights_), end(visible_lights_), i);
    //     assert(it != end(visible_lights_));
    //     visible_lights_.erase(it);
    // }

    // if (lights_[i].sky_portal) {
    //     auto it = find(begin(blocker_lights_), end(blocker_lights_), i);
    //     assert(it != end(blocker_lights_));
    //     blocker_lights_.erase(it);
    // }

    lights_.erase(i._index);
}

Ray::MeshInstanceHandle Ray::Vk::Scene::AddMeshInstance(const MeshHandle mesh, const float *xform) {
    mesh_instance_t mi = {};
    mi.mesh_index = mesh._index;
    mi.tr_index = transforms_.emplace();

    const uint32_t mi_index = mesh_instances_.push(mi);

    { // find emissive triangles and add them as emitters
        const mesh_t &m = meshes_[mesh._index];
        for (uint32_t tri = (m.vert_index / 3); tri < (m.vert_index + m.vert_count) / 3; ++tri) {
            const tri_mat_data_t &tri_mat = tri_materials_cpu_[tri];

            const material_t &front_mat = materials_[tri_mat.front_mi & MATERIAL_INDEX_BITS];
            if (front_mat.type == EmissiveNode && (front_mat.flags & MAT_FLAG_MULT_IMPORTANCE)) {
                light_t new_light;
                new_light.type = LIGHT_TYPE_TRI;
                new_light.cast_shadow = 1;
                new_light.visible = 0;
                new_light.sky_portal = 0;
                new_light.tri.tri_index = tri;
                new_light.tri.xform_index = mi.tr_index;
                new_light.col[0] = front_mat.base_color[0] * front_mat.strength;
                new_light.col[1] = front_mat.base_color[1] * front_mat.strength;
                new_light.col[2] = front_mat.base_color[2] * front_mat.strength;
                const uint32_t index = lights_.push(new_light);
                li_indices_.PushBack(index);
            }
        }
    }

    SetMeshInstanceTransform(MeshInstanceHandle{mi_index}, xform);

    return MeshInstanceHandle{mi_index};
}

void Ray::Vk::Scene::SetMeshInstanceTransform(const MeshInstanceHandle mi_handle, const float *xform) {
    transform_t tr = {};

    memcpy(tr.xform, xform, 16 * sizeof(float));
    InverseMatrix(tr.xform, tr.inv_xform);

    mesh_instance_t mi = mesh_instances_[mi_handle._index];

    const mesh_t &m = meshes_[mi.mesh_index];
    TransformBoundingBox(m.bbox_min, m.bbox_max, xform, mi.bbox_min, mi.bbox_max);

    mesh_instances_.Set(mi_handle._index, mi);
    transforms_.Set(mi.tr_index, tr);

    RebuildTLAS();
}

void Ray::Vk::Scene::RemoveMeshInstance(MeshInstanceHandle) {
    // TODO!!
}

void Ray::Vk::Scene::Finalize() {
    if (env_map_light_ != InvalidLightHandle) {
        RemoveLight(env_map_light_);
    }
    env_map_qtree_ = {};
    env_.qtree_levels = 0;

    if (env_.multiple_importance && env_.env_col[0] > 0.0f && env_.env_col[1] > 0.0f && env_.env_col[2] > 0.0f) {
        if (env_.env_map != 0xffffffff) {
            PrepareEnvMapQTree();
        } else {
            // Dummy
            Tex2DParams p;
            p.w = p.h = 1;
            p.format = eTexFormat::RawRGBA32F;
            p.mip_count = 1;
            p.usage = eTexUsageBits::Sampled | eTexUsageBits::Transfer;

            env_map_qtree_.tex = Texture2D("Env map qtree", ctx_, p, ctx_->default_memory_allocs(), ctx_->log());
        }
        { // add env light source
            light_t l = {};

            l.type = LIGHT_TYPE_ENV;
            l.cast_shadow = 1;
            l.col[0] = l.col[1] = l.col[2] = 1.0f;

            env_map_light_ = LightHandle{lights_.push(l)};
            li_indices_.PushBack(env_map_light_._index);
        }
    } else {
        // Dummy
        Tex2DParams p;
        p.w = p.h = 1;
        p.format = eTexFormat::RawRGBA32F;
        p.mip_count = 1;
        p.usage = eTexUsageBits::Sampled | eTexUsageBits::Transfer;

        env_map_qtree_.tex = Texture2D("Env map qtree", ctx_, p, ctx_->default_memory_allocs(), ctx_->log());
    }

    GenerateTextureMips();
    PrepareBindlessTextures();
    RebuildHWAccStructures();
}

void Ray::Vk::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
    if (!node_count) {
        return;
    }

    /*nodes_.Erase(node_index, node_count);

    if (node_index != nodes_.size()) {
        size_t meshes_count = meshes_.size();
        std::vector<mesh_t> meshes(meshes_count);
        meshes_.Get(&meshes[0], 0, meshes_.size());

        for (mesh_t &m : meshes) {
            if (m.node_index > node_index) {
                m.node_index -= node_count;
            }
        }
        meshes_.Set(&meshes[0], 0, meshes_count);

        size_t nodes_count = nodes_.size();
        std::vector<bvh_node_t> nodes(nodes_count);
        nodes_.Get(&nodes[0], 0, nodes_count);

        for (uint32_t i = node_index; i < nodes.size(); i++) {
            bvh_node_t &n = nodes[i];
            if ((n.prim_index & LEAF_NODE_BIT) == 0) {
                if (n.left_child > node_index) {
                    n.left_child -= node_count;
                }
                if ((n.right_child & RIGHT_CHILD_BITS) > node_index) {
                    n.right_child -= node_count;
                }
            }
        }
        nodes_.Set(&nodes[0], 0, nodes_count);

        if (macro_nodes_start_ > node_index) {
            macro_nodes_start_ -= node_count;
        }
    }*/
}

void Ray::Vk::Scene::RebuildTLAS() {
    RemoveNodes(macro_nodes_start_, macro_nodes_count_);
    mi_indices_.Clear();

    const size_t mi_count = mesh_instances_.size();

    std::vector<prim_t> primitives;
    primitives.reserve(mi_count);

    for (const mesh_instance_t &mi : mesh_instances_) {
        primitives.push_back({0, 0, 0, Ref::simd_fvec4{mi.bbox_min[0], mi.bbox_min[1], mi.bbox_min[2], 0.0f},
                              Ref::simd_fvec4{mi.bbox_max[0], mi.bbox_max[1], mi.bbox_max[2], 0.0f}});
    }

    std::vector<bvh_node_t> bvh_nodes;
    std::vector<uint32_t> mi_indices;

    macro_nodes_start_ = uint32_t(nodes_.size());
    macro_nodes_count_ = PreprocessPrims_SAH(primitives, nullptr, 0, {}, bvh_nodes, mi_indices);

    // offset nodes
    for (bvh_node_t &n : bvh_nodes) {
        if ((n.prim_index & LEAF_NODE_BIT) == 0) {
            n.left_child += uint32_t(nodes_.size());
            n.right_child += uint32_t(nodes_.size());
        }
    }

    nodes_.Append(&bvh_nodes[0], bvh_nodes.size());
    mi_indices_.Append(&mi_indices[0], mi_indices.size());
}

void Ray::Vk::Scene::PrepareEnvMapQTree() {
    const int tex = int(env_.env_map & 0x00ffffff);

    Buffer temp_stage_buf;
    std::unique_ptr<uint8_t[]> temp_atlas_data;

    simd_ivec2 size;

    if (use_bindless_) {
        const Texture2D &t = bindless_textures_[tex];
        size.template set<0>(t.params.w);
        size.template set<1>(t.params.h);

        assert(t.params.format == eTexFormat::RawRGBA8888);
        const uint32_t data_size = t.params.w * t.params.h * GetPerPixelDataLen(eTexFormat::RawRGBA8888);

        temp_stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Stage, data_size);

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        CopyImageToBuffer(t, 0, 0, 0, t.params.w, t.params.h, temp_stage_buf, cmd_buf, 0);

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    } else {
        const atlas_texture_t &t = atlas_textures_[tex];
        size.template set<0>(t.width & ATLAS_TEX_WIDTH_BITS);
        size.template set<1>(t.height & ATLAS_TEX_HEIGHT_BITS);

        const TextureAtlas &atlas = tex_atlases_[t.atlas];

        assert(atlas.format() == eTexFormat::RawRGBA8888);
        const uint32_t data_size = atlas.res_x() * atlas.res_y() * GetPerPixelDataLen(atlas.format());

        temp_stage_buf = Buffer("Temp stage buf", ctx_, eBufType::Stage, data_size);

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

        atlas.CopyRegionTo(t.page[0], t.pos[0][0] + 1, t.pos[0][1] + 1, (t.width & ATLAS_TEX_WIDTH_BITS),
                           (t.height & ATLAS_TEX_HEIGHT_BITS), temp_stage_buf, cmd_buf, 0);

        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    const uint8_t *rgbe_data = temp_stage_buf.Map(BufMapRead);

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

        for (int y = 0; y < size[1]; ++y) {
            const float theta = PI * float(y) / float(size[1]);
            for (int x = 0; x < size[0]; ++x) {
                const float phi = 2.0f * PI * float(x) / float(size[0]);

                const uint8_t *col_rgbe = &rgbe_data[4 * (y * size[0] + x)];
                simd_fvec4 col_rgb;
                rgbe_to_rgb(col_rgbe, value_ptr(col_rgb));

                const float cur_lum = (col_rgb[0] + col_rgb[1] + col_rgb[2]);

                auto dir =
                    simd_fvec4{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi), 0.0f};

                simd_fvec2 q;
                DirToCanonical(value_ptr(dir), 0.0f, value_ptr(q));

                int qx = CLAMP(int(cur_res * q[0]), 0, cur_res - 1);
                int qy = CLAMP(int(cur_res * q[1]), 0, cur_res - 1);

                int index = 0;
                index |= (qx & 1) << 0;
                index |= (qy & 1) << 1;

                qx /= 2;
                qy /= 2;

                simd_fvec4 &qvec = env_map_qtree_.mips[0][qy * cur_res / 2 + qx];
                qvec.set(index, std::max(qvec[index], cur_lum));
            }
        }

        for (const simd_fvec4 &v : env_map_qtree_.mips[0]) {
            total_lum += (v[0] + v[1] + v[2] + v[3]);
        }

        cur_res /= 2;
    }

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

    const float LumFractThreshold = 0.01f;

    cur_res = 2;
    int the_last_required_lod;
    for (int lod = int(env_map_qtree_.mips.size()) - 1; lod >= 0; --lod) {
        the_last_required_lod = lod;
        const auto &cur_mip = env_map_qtree_.mips[lod];

        bool subdivision_required = false;
        for (int y = 0; y < (cur_res / 2) && !subdivision_required; ++y) {
            for (int x = 0; x < (cur_res / 2) && !subdivision_required; ++x) {
                const simd_ivec4 mask = simd_cast(cur_mip[y * cur_res / 2 + x] > LumFractThreshold * total_lum);
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
    for (int i = env_.qtree_levels; i < countof(env_.qtree_mips); ++i) {
        env_.qtree_mips[i] = nullptr;
    }

    //
    // Upload texture
    //

    int req_size = 0, mip_offsets[16] = {};
    for (int i = 0; i < env_.qtree_levels; ++i) {
        mip_offsets[i] = req_size;
        req_size += 4096 * int((env_map_qtree_.mips[i].size() * sizeof(simd_fvec4) + 4096 - 1) / 4096);
    }

    temp_stage_buf = Buffer("Temp upload buf", ctx_, eBufType::Stage, req_size);
    uint8_t *stage_data = temp_stage_buf.Map(BufMapWrite);

    for (int i = 0; i < env_.qtree_levels; ++i) {
        memcpy(&stage_data[mip_offsets[i]], env_map_qtree_.mips[i].data(),
               env_map_qtree_.mips[i].size() * sizeof(simd_fvec4));
    }

    Tex2DParams p;
    p.w = p.h = (env_map_qtree_.res / 2);
    p.format = eTexFormat::RawRGBA32F;
    p.mip_count = env_.qtree_levels;
    p.usage = eTexUsageBits::Sampled | eTexUsageBits::Transfer;

    env_map_qtree_.tex = Texture2D("Env map qtree", ctx_, p, ctx_->default_memory_allocs(), ctx_->log());

    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

    for (int i = 0; i < env_.qtree_levels; ++i) {
        env_map_qtree_.tex.SetSubImage(i, 0, 0, (env_map_qtree_.res >> i) / 2, (env_map_qtree_.res >> i) / 2,
                                       eTexFormat::RawRGBA32F, temp_stage_buf, cmd_buf, mip_offsets[i],
                                       int(env_map_qtree_.mips[i].size() * sizeof(simd_fvec4)));
    }

    EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    temp_stage_buf.Unmap();
    temp_stage_buf.FreeImmediate();

    ctx_->log()->Info("Env map qtree res is %i", env_map_qtree_.res);
}

void Ray::Vk::Scene::GenerateTextureMips() {
    struct mip_gen_info {
        uint32_t texture_index;
        uint16_t size; // used for sorting
        uint8_t dst_mip;
        uint8_t atlas_index; // used for sorting
    };

    std::vector<mip_gen_info> mips_to_generate;
    mips_to_generate.reserve(atlas_textures_.size());

    for (uint32_t i = 0; i < uint32_t(atlas_textures_.size()); ++i) {
        const atlas_texture_t &t = atlas_textures_[i];
        if ((t.height & ATLAS_TEX_MIPS_BIT) == 0) {
            continue;
        }

        int mip = 0;
        int res[2] = {(t.width & ATLAS_TEX_WIDTH_BITS), (t.height & ATLAS_TEX_HEIGHT_BITS)};

        res[0] /= 2;
        res[1] /= 2;
        ++mip;

        while (res[0] >= 1 && res[1] >= 1) {
            const bool requires_generation =
                t.page[mip] == t.page[0] && t.pos[mip][0] == t.pos[0][0] && t.pos[mip][1] == t.pos[0][1];
            if (requires_generation) {
                mips_to_generate.emplace_back();
                auto &m = mips_to_generate.back();
                m.texture_index = i;
                m.size = std::max(res[0], res[1]);
                m.dst_mip = mip;
                m.atlas_index = t.atlas;
            }

            res[0] /= 2;
            res[1] /= 2;
            ++mip;
        }
    }

    // Sort for more optimal allocation
    sort(begin(mips_to_generate), end(mips_to_generate), [](const mip_gen_info &lhs, const mip_gen_info &rhs) {
        if (lhs.atlas_index == rhs.atlas_index) {
            return lhs.size > rhs.size;
        }
        return lhs.atlas_index < rhs.atlas_index;
    });

    for (const mip_gen_info &info : mips_to_generate) {
        atlas_texture_t t = atlas_textures_[info.texture_index];

        const int dst_mip = info.dst_mip;
        const int src_mip = dst_mip - 1;
        const int src_res[2] = {(t.width & ATLAS_TEX_WIDTH_BITS) >> src_mip,
                                (t.height & ATLAS_TEX_HEIGHT_BITS) >> src_mip};
        assert(src_res[0] != 0 && src_res[1] != 0);

        const int src_pos[2] = {t.pos[src_mip][0] + 1, t.pos[src_mip][1] + 1};

        int pos[2];
        const int page = tex_atlases_[t.atlas].DownsampleRegion(t.page[src_mip], src_pos, src_res, pos);
        if (page == -1) {
            ctx_->log()->Error("Failed to allocate texture!");
            break;
        }

        t.page[dst_mip] = uint8_t(page);
        t.pos[dst_mip][0] = uint16_t(pos[0]);
        t.pos[dst_mip][1] = uint16_t(pos[1]);

        if (src_res[0] == 1 || src_res[1] == 1) {
            // fill remaining mip levels with the last one
            for (int i = dst_mip + 1; i < NUM_MIP_LEVELS; i++) {
                t.page[i] = t.page[dst_mip];
                t.pos[i][0] = t.pos[dst_mip][0];
                t.pos[i][1] = t.pos[dst_mip][1];
            }
        }

        atlas_textures_.Set(info.texture_index, t);
    }

    ctx_->log()->Info("Ray: Atlasses are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC3[%i], BC4[%i], BC5[%i])",
                      tex_atlases_[0].page_count(), tex_atlases_[1].page_count(), tex_atlases_[2].page_count(),
                      tex_atlases_[3].page_count(), tex_atlases_[4].page_count(), tex_atlases_[5].page_count(),
                      tex_atlases_[6].page_count());
}

void Ray::Vk::Scene::PrepareBindlessTextures() {
    assert(bindless_textures_.capacity() <= ctx_->max_combined_image_samplers());

    DescrSizes descr_sizes;
    descr_sizes.img_sampler_count = ctx_->max_combined_image_samplers();

    const bool bres = bindless_tex_data_.descr_pool.Init(descr_sizes, 1 /* sets_count */);
    if (!bres) {
        ctx_->log()->Error("Failed to init descriptor pool!");
    }

    if (!bindless_tex_data_.descr_layout) {
        VkDescriptorSetLayoutBinding textures_binding = {};
        textures_binding.binding = 0;
        textures_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textures_binding.descriptorCount = ctx_->max_combined_image_samplers();
        textures_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 1;
        layout_info.pBindings = &textures_binding;

        VkDescriptorBindingFlagsEXT bind_flag = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT};
        extended_info.bindingCount = 1u;
        extended_info.pBindingFlags = &bind_flag;
        layout_info.pNext = &extended_info;

        const VkResult res =
            vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &bindless_tex_data_.descr_layout);
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to create descriptor set layout!");
        }
    }

    bindless_tex_data_.descr_pool.Reset();
    bindless_tex_data_.descr_set = bindless_tex_data_.descr_pool.Alloc(bindless_tex_data_.descr_layout);

    { // Transition resources
        std::vector<TransitionInfo> img_transitions;
        img_transitions.reserve(bindless_textures_.size());

        for (const auto &tex : bindless_textures_) {
            img_transitions.emplace_back(&tex, eResState::ShaderResource);
        }

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());
        TransitionResourceStates(cmd_buf, AllStages, AllStages, img_transitions);
        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    for (uint32_t i = 0; i < bindless_textures_.capacity(); ++i) {
        if (!bindless_textures_.exists(i)) {
            continue;
        }

        VkDescriptorImageInfo img_info = bindless_textures_[i].vk_desc_image_info();

        VkWriteDescriptorSet descr_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        descr_write.dstSet = bindless_tex_data_.descr_set;
        descr_write.dstBinding = 0;
        descr_write.dstArrayElement = i;
        descr_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descr_write.descriptorCount = 1;
        descr_write.pBufferInfo = nullptr;
        descr_write.pImageInfo = &img_info;
        descr_write.pTexelBufferView = nullptr;
        descr_write.pNext = nullptr;

        vkUpdateDescriptorSets(ctx_->device(), 1, &descr_write, 0, nullptr);
    }
}

void Ray::Vk::Scene::RebuildHWAccStructures() {
    if (!use_hwrt_) {
        return;
    }

    static const VkDeviceSize AccStructAlignment = 256;

    struct Blas {
        SmallVector<VkAccelerationStructureGeometryKHR, 16> geometries;
        SmallVector<VkAccelerationStructureBuildRangeInfoKHR, 16> build_ranges;
        SmallVector<uint32_t, 16> prim_counts;
        VkAccelerationStructureBuildSizesInfoKHR size_info = {};
        VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    };
    std::vector<Blas> all_blases;

    uint32_t needed_build_scratch_size = 0;
    uint32_t needed_total_acc_struct_size = 0;

    for (const mesh_t &mesh : meshes_) {
        VkAccelerationStructureGeometryTrianglesDataKHR tri_data = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        tri_data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri_data.vertexData.deviceAddress = vertices_.buf().vk_device_address();
        tri_data.vertexStride = sizeof(vertex_t);
        tri_data.indexType = VK_INDEX_TYPE_UINT32;
        tri_data.indexData.deviceAddress = vtx_indices_.buf().vk_device_address();
        // TODO: fix this!
        tri_data.maxVertex = mesh.vert_index + mesh.vert_count;

        //
        // Gather geometries
        //
        all_blases.emplace_back();
        Blas &new_blas = all_blases.back();

        {
            auto &new_geo = new_blas.geometries.emplace_back();
            new_geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            new_geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            new_geo.flags = 0;
            // if ((mat_flags & uint32_t(Ren::eMatFlags::AlphaTest)) == 0) {
            //     new_geo.flags |= VK_GEOMETRY_OPAQUE_BIT_KHR;
            // }
            new_geo.geometry.triangles = tri_data;

            auto &new_range = new_blas.build_ranges.emplace_back();
            new_range.firstVertex = 0; // mesh.vert_index;
            new_range.primitiveCount = mesh.vert_count / 3;
            new_range.primitiveOffset = mesh.vert_index * sizeof(uint32_t);
            new_range.transformOffset = 0;

            new_blas.prim_counts.push_back(new_range.primitiveCount);
        }

        //
        // Query needed memory
        //
        new_blas.build_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        new_blas.build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        new_blas.build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        new_blas.build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        new_blas.build_info.geometryCount = uint32_t(new_blas.geometries.size());
        new_blas.build_info.pGeometries = new_blas.geometries.cdata();

        new_blas.size_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(ctx_->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &new_blas.build_info, new_blas.prim_counts.cdata(),
                                                &new_blas.size_info);

        // make sure we will not use this potentially stale pointer
        new_blas.build_info.pGeometries = nullptr;

        needed_build_scratch_size = std::max(needed_build_scratch_size, uint32_t(new_blas.size_info.buildScratchSize));
        needed_total_acc_struct_size +=
            uint32_t(align_up(new_blas.size_info.accelerationStructureSize, AccStructAlignment));

        rt_mesh_blases_.emplace_back();
    }

    if (!all_blases.empty()) {
        //
        // Allocate memory
        //
        Buffer scratch_buf("BLAS Scratch Buf", ctx_, eBufType::Storage, next_power_of_two(needed_build_scratch_size));
        VkDeviceAddress scratch_addr = scratch_buf.vk_device_address();

        Buffer acc_structs_buf("BLAS Before-Compaction Buf", ctx_, eBufType::AccStructure,
                               needed_total_acc_struct_size);

        //

        VkQueryPoolCreateInfo query_pool_create_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_pool_create_info.queryCount = uint32_t(all_blases.size());
        query_pool_create_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;

        VkQueryPool query_pool;
        VkResult res = vkCreateQueryPool(ctx_->device(), &query_pool_create_info, nullptr, &query_pool);
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("Failed to create query pool!");
        }

        std::vector<AccStructure> blases_before_compaction;
        blases_before_compaction.resize(all_blases.size());

        { // Submit build commands
            VkDeviceSize acc_buf_offset = 0;
            VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

            vkCmdResetQueryPool(cmd_buf, query_pool, 0, uint32_t(all_blases.size()));

            for (int i = 0; i < int(all_blases.size()); ++i) {
                VkAccelerationStructureCreateInfoKHR acc_create_info = {
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
                acc_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                acc_create_info.buffer = acc_structs_buf.vk_handle();
                acc_create_info.offset = acc_buf_offset;
                acc_create_info.size = all_blases[i].size_info.accelerationStructureSize;
                acc_buf_offset += align_up(acc_create_info.size, AccStructAlignment);

                VkAccelerationStructureKHR acc_struct;
                VkResult res = vkCreateAccelerationStructureKHR(ctx_->device(), &acc_create_info, nullptr, &acc_struct);
                if (res != VK_SUCCESS) {
                    ctx_->log()->Error("Failed to create acceleration structure!");
                }

                auto &acc = blases_before_compaction[i];
                if (!acc.Init(ctx_, acc_struct)) {
                    ctx_->log()->Error("Failed to init BLAS!");
                }

                all_blases[i].build_info.pGeometries = all_blases[i].geometries.cdata();

                all_blases[i].build_info.dstAccelerationStructure = acc_struct;
                all_blases[i].build_info.scratchData.deviceAddress = scratch_addr;

                const VkAccelerationStructureBuildRangeInfoKHR *build_ranges = all_blases[i].build_ranges.cdata();
                vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &all_blases[i].build_info, &build_ranges);

                { // Place barrier
                    VkMemoryBarrier scr_buf_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    scr_buf_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    scr_buf_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &scr_buf_barrier,
                                         0, nullptr, 0, nullptr);
                }

                vkCmdWriteAccelerationStructuresPropertiesKHR(
                    cmd_buf, 1, &all_blases[i].build_info.dstAccelerationStructure,
                    VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, query_pool, i);
            }

            EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
        }

        std::vector<VkDeviceSize> compact_sizes(all_blases.size());
        res = vkGetQueryPoolResults(ctx_->device(), query_pool, 0, uint32_t(all_blases.size()),
                                    all_blases.size() * sizeof(VkDeviceSize), compact_sizes.data(),
                                    sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);
        assert(res == VK_SUCCESS);

        vkDestroyQueryPool(ctx_->device(), query_pool, nullptr);

        VkDeviceSize total_compacted_size = 0;
        for (int i = 0; i < int(compact_sizes.size()); ++i) {
            total_compacted_size += align_up(compact_sizes[i], AccStructAlignment);
        }

        rt_blas_buf_ =
            Buffer{"BLAS After-Compaction Buf", ctx_, eBufType::AccStructure, uint32_t(total_compacted_size)};

        { // Submit compaction commands
            VkDeviceSize compact_acc_buf_offset = 0;
            VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

            for (int i = 0; i < int(all_blases.size()); ++i) {
                VkAccelerationStructureCreateInfoKHR acc_create_info = {
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
                acc_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                acc_create_info.buffer = rt_blas_buf_.vk_handle();
                acc_create_info.offset = compact_acc_buf_offset;
                acc_create_info.size = compact_sizes[i];
                assert(compact_acc_buf_offset + compact_sizes[i] <= total_compacted_size);
                compact_acc_buf_offset += align_up(acc_create_info.size, AccStructAlignment);

                VkAccelerationStructureKHR compact_acc_struct;
                const VkResult res =
                    vkCreateAccelerationStructureKHR(ctx_->device(), &acc_create_info, nullptr, &compact_acc_struct);
                if (res != VK_SUCCESS) {
                    ctx_->log()->Error("Failed to create acceleration structure!");
                }

                VkCopyAccelerationStructureInfoKHR copy_info = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
                copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                copy_info.src = blases_before_compaction[i].vk_handle();
                copy_info.dst = compact_acc_struct;

                vkCmdCopyAccelerationStructureKHR(cmd_buf, &copy_info);

                auto &vk_blas = rt_mesh_blases_[i].acc;
                if (!vk_blas.Init(ctx_, compact_acc_struct)) {
                    ctx_->log()->Error("Blas compaction failed!");
                }
            }

            EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

            for (auto &b : blases_before_compaction) {
                b.FreeImmediate();
            }
            acc_structs_buf.FreeImmediate();
            scratch_buf.FreeImmediate();
        }
    }

    //
    // Build TLAS
    //

    struct RTGeoInstance {
        uint32_t indices_start;
        uint32_t vertices_start;
        uint32_t material_index;
        uint32_t flags;
    };
    static_assert(sizeof(RTGeoInstance) == 16, "!");

    std::vector<RTGeoInstance> geo_instances;
    std::vector<VkAccelerationStructureInstanceKHR> tlas_instances;

    for (const mesh_instance_t &instance : mesh_instances_) {
        auto &blas = rt_mesh_blases_[instance.mesh_index];
        blas.geo_index = uint32_t(geo_instances.size());
        blas.geo_count = 0;

        auto &vk_blas = blas.acc;

        tlas_instances.emplace_back();
        auto &new_instance = tlas_instances.back();
        to_khr_xform(transforms_[instance.tr_index].xform, new_instance.transform.matrix);
        new_instance.instanceCustomIndex = meshes_[instance.mesh_index].vert_index / 3;
        // blas.geo_index;
        new_instance.mask = 0xff;
        new_instance.instanceShaderBindingTableRecordOffset = 0;
        new_instance.flags = 0;
        // VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR; //
        // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        new_instance.accelerationStructureReference = static_cast<uint64_t>(vk_blas.vk_device_address());

        // const mesh_t &mesh = meshes_[instance.mesh_index];
        {
            ++blas.geo_count;

            geo_instances.emplace_back();
            auto &geo = geo_instances.back();
            geo.indices_start = 0;  // mesh.
            geo.vertices_start = 0; // acc.mesh->attribs_buf1().offset / 16;
            geo.material_index = 0; // grp.mat.index();
            geo.flags = 0;
        }
    }

    if (geo_instances.empty()) {
        geo_instances.emplace_back();
        auto &dummy_geo = geo_instances.back();
        dummy_geo = {};

        tlas_instances.emplace_back();
        auto &dummy_instance = tlas_instances.back();
        dummy_instance = {};
    }

    rt_geo_data_buf_ =
        Buffer{"RT Geo Data Buf", ctx_, eBufType::Storage, uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
    Buffer geo_data_stage_buf{"RT Geo Data Stage Buf", ctx_, eBufType::Stage,
                              uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
    {
        uint8_t *geo_data_stage = geo_data_stage_buf.Map(BufMapWrite);
        memcpy(geo_data_stage, geo_instances.data(), geo_instances.size() * sizeof(RTGeoInstance));
        geo_data_stage_buf.Unmap();
    }

    rt_instance_buf_ = Buffer{"RT Instance Buf", ctx_, eBufType::Storage,
                              uint32_t(tlas_instances.size() * sizeof(VkAccelerationStructureInstanceKHR))};
    Buffer instance_stage_buf{"RT Instance Stage Buf", ctx_, eBufType::Stage,
                              uint32_t(tlas_instances.size() * sizeof(VkAccelerationStructureInstanceKHR))};
    {
        uint8_t *instance_stage = instance_stage_buf.Map(BufMapWrite);
        memcpy(instance_stage, tlas_instances.data(),
               tlas_instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
        instance_stage_buf.Unmap();
    }

    VkDeviceAddress instance_buf_addr = rt_instance_buf_.vk_device_address();

    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());

    CopyBufferToBuffer(geo_data_stage_buf, 0, rt_geo_data_buf_, 0, geo_data_stage_buf.size(), cmd_buf);
    CopyBufferToBuffer(instance_stage_buf, 0, rt_instance_buf_, 0, instance_stage_buf.size(), cmd_buf);

    { // Make sure compaction copying of BLASes has finished
        VkMemoryBarrier mem_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mem_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &mem_barrier, 0, nullptr, 0,
                             nullptr);
    }

    Buffer tlas_scratch_buf;

    { //
        VkAccelerationStructureGeometryInstancesDataKHR instances_data = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        instances_data.data.deviceAddress = instance_buf_addr;

        VkAccelerationStructureGeometryKHR tlas_geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        tlas_geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlas_geo.geometry.instances = instances_data;

        VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        tlas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV;
        tlas_build_info.geometryCount = 1;
        tlas_build_info.pGeometries = &tlas_geo;
        tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;

        const auto instance_count = uint32_t(tlas_instances.size());
        const uint32_t max_instance_count = instance_count;

        VkAccelerationStructureBuildSizesInfoKHR size_info = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(ctx_->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &tlas_build_info, &max_instance_count, &size_info);

        rt_tlas_buf_ = Buffer{"TLAS Buf", ctx_, eBufType::AccStructure, uint32_t(size_info.accelerationStructureSize)};

        VkAccelerationStructureCreateInfoKHR create_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        create_info.buffer = rt_tlas_buf_.vk_handle();
        create_info.offset = 0;
        create_info.size = size_info.accelerationStructureSize;

        VkAccelerationStructureKHR tlas_handle;
        VkResult res = vkCreateAccelerationStructureKHR(ctx_->device(), &create_info, nullptr, &tlas_handle);
        if (res != VK_SUCCESS) {
            ctx_->log()->Error("[SceneManager::InitHWAccStructures]: Failed to create acceleration structure!");
        }

        tlas_scratch_buf = Buffer{"TLAS Scratch Buf", ctx_, eBufType::Storage, uint32_t(size_info.buildScratchSize)};
        VkDeviceAddress tlas_scratch_buf_addr = tlas_scratch_buf.vk_device_address();

        tlas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
        tlas_build_info.dstAccelerationStructure = tlas_handle;
        tlas_build_info.scratchData.deviceAddress = tlas_scratch_buf_addr;

        VkAccelerationStructureBuildRangeInfoKHR range_info = {};
        range_info.primitiveOffset = 0;
        range_info.primitiveCount = instance_count;
        range_info.firstVertex = 0;
        range_info.transformOffset = 0;

        const VkAccelerationStructureBuildRangeInfoKHR *build_range = &range_info;
        vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &tlas_build_info, &build_range);

        if (!rt_tlas_.Init(ctx_, tlas_handle)) {
            ctx_->log()->Error("[SceneManager::InitHWAccStructures]: Failed to init TLAS!");
        }
    }

    EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    tlas_scratch_buf.FreeImmediate();
    instance_stage_buf.FreeImmediate();
    geo_data_stage_buf.FreeImmediate();
}

#undef _MIN
#undef _MAX
#undef _ABS
#undef _CLAMP
