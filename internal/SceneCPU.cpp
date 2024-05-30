#include "SceneCPU.h"

#include <cassert>
#include <cstring>

#include <functional>

#include "../Log.h"
#include "BVHSplit.h"
#include "CoreRef.h"
#include "TextureUtils.h"
#include "Time_.h"

namespace Ray {
namespace Cpu {
template <typename T> T clamp(T val, T min, T max) { return (val < min ? min : (val > max ? max : val)); }

Ref::fvec4 cross(const Ref::fvec4 &v1, const Ref::fvec4 &v2) {
    return Ref::fvec4{v1.get<1>() * v2.get<2>() - v1.get<2>() * v2.get<1>(),
                      v1.get<2>() * v2.get<0>() - v1.get<0>() * v2.get<2>(),
                      v1.get<0>() * v2.get<1>() - v1.get<1>() * v2.get<0>(), 0.0f};
}
} // namespace Cpu
} // namespace Ray

Ray::Cpu::Scene::Scene(ILog *log, const bool use_wide_bvh, const bool use_tex_compression, const bool use_spatial_cache)
    : use_wide_bvh_(use_wide_bvh), use_tex_compression_(use_tex_compression) {
    SceneBase::log_ = log;
    SetEnvironment({});
    if (use_spatial_cache) {
        spatial_cache_entries_.resize(HASH_GRID_CACHE_ENTRIES_COUNT, 0);
        spatial_cache_voxels_curr_.resize(HASH_GRID_CACHE_ENTRIES_COUNT, {});
        spatial_cache_voxels_prev_.resize(HASH_GRID_CACHE_ENTRIES_COUNT, {});
    }
}

Ray::Cpu::Scene::~Scene() {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end();) {
        MeshInstanceHandle to_delete = {it.index(), it.block()};
        ++it;
        Scene::RemoveMeshInstance_nolock(to_delete);
    }
    for (auto it = meshes_.begin(); it != meshes_.end();) {
        MeshHandle to_delete = {it.index(), it.block()};
        ++it;
        Scene::RemoveMesh_nolock(to_delete);
    }

    if (tlas_root_ != 0xffffffff) {
        if (use_wide_bvh_) {
            wnodes_.Erase(tlas_block_);
        } else {
            nodes_.Erase(tlas_block_);
        }
        tlas_root_ = tlas_block_ = 0xffffffff;
    }
}

Ray::TextureHandle Ray::Cpu::Scene::AddTexture(const tex_desc_t &_t) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const int res[2] = {_t.w, _t.h};

    bool use_compression = use_tex_compression_ && !_t.force_no_compression;
    bool reconstruct_z = _t.reconstruct_z, is_YCoCg = _t.is_YCoCg;

    int storage = -1, index = -1;
    if (_t.format == eTextureFormat::RGBA8888) {
        const auto *rgba_data = reinterpret_cast<const color_rgba8_t *>(_t.data.data());
        if (!_t.is_normalmap) {
            storage = 0;
            index = tex_storage_rgba_.Allocate(Span<const color_rgba8_t>(rgba_data, res[0] * res[1]), res,
                                               _t.generate_mipmaps);
        } else {
            // TODO: get rid of this allocation
            std::vector<color_rg8_t> repacked_data(res[0] * res[1]);
            const bool invert_y = (_t.convention == Ray::eTextureConvention::DX);
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_data[i].v[0] = rgba_data[i].v[0];
                repacked_data[i].v[1] = invert_y ? (255 - rgba_data[i].v[1]) : rgba_data[i].v[1];
                reconstruct_z |= (rgba_data[i].v[2] < 250);
            }
            if (use_compression) {
                storage = 7;
                index = tex_storage_bc5_.Allocate(repacked_data, res, _t.generate_mipmaps);
            } else {
                storage = 2;
                index = tex_storage_rg_.Allocate(repacked_data, res, _t.generate_mipmaps);
            }
        }
    } else if (_t.format == eTextureFormat::RGB888) {
        const auto *rgb_data = reinterpret_cast<const color_rgb8_t *>(_t.data.data());
        if (!_t.is_normalmap) {
            if (use_compression) {
                is_YCoCg = true;
                storage = 5;
                index = tex_storage_bc3_.Allocate(Span<const color_rgb8_t>(rgb_data, res[0] * res[1]), res,
                                                  _t.generate_mipmaps);
            } else {
                storage = 1;
                index = tex_storage_rgb_.Allocate(Span<const color_rgb8_t>(rgb_data, res[0] * res[1]), res,
                                                  _t.generate_mipmaps);
            }
        } else {
            // TODO: get rid of this allocation
            const bool invert_y = (_t.convention == Ray::eTextureConvention::DX);
            std::vector<color_rg8_t> repacked_data(res[0] * res[1]);
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_data[i].v[0] = rgb_data[i].v[0];
                repacked_data[i].v[1] = invert_y ? (255 - rgb_data[i].v[1]) : rgb_data[i].v[1];
                reconstruct_z |= (rgb_data[i].v[2] < 250);
            }

            if (use_compression) {
                storage = 7;
                index = tex_storage_bc5_.Allocate(repacked_data, res, _t.generate_mipmaps);
            } else {
                storage = 2;
                index = tex_storage_rg_.Allocate(repacked_data, res, _t.generate_mipmaps);
            }
        }
    } else if (_t.format == eTextureFormat::RG88) {
        const auto *data_to_use = reinterpret_cast<const color_rg8_t *>(_t.data.data());
        // TODO: get rid of this allocation
        std::vector<color_rg8_t> repacked_data;
        const bool invert_y = (_t.convention == Ray::eTextureConvention::DX);
        if (_t.is_normalmap && invert_y) {
            repacked_data.resize(res[0] * res[1]);
            for (int i = 0; i < res[0] * res[1]; ++i) {
                repacked_data[i].v[0] = data_to_use[i].v[0];
                repacked_data[i].v[1] = invert_y ? (255 - data_to_use[i].v[1]) : data_to_use[i].v[1];
            }
            data_to_use = repacked_data.data();
        }

        if (use_compression) {
            storage = 7;
            index = tex_storage_bc5_.Allocate(Span<const color_rg8_t>(data_to_use, res[0] * res[1]), res,
                                              _t.generate_mipmaps);
        } else {
            storage = 2;
            index = tex_storage_rg_.Allocate(Span<const color_rg8_t>(data_to_use, res[0] * res[1]), res,
                                             _t.generate_mipmaps);
        }
        reconstruct_z = _t.is_normalmap;
    } else if (_t.format == eTextureFormat::R8) {
        if (use_compression) {
            storage = 6;
            index = tex_storage_bc4_.Allocate(
                Span<const color_r8_t>(reinterpret_cast<const color_r8_t *>(_t.data.data()), res[0] * res[1]), res,
                _t.generate_mipmaps);
        } else {
            storage = 3;
            index = tex_storage_r_.Allocate(
                Span<const color_r8_t>(reinterpret_cast<const color_r8_t *>(_t.data.data()), res[0] * res[1]), res,
                _t.generate_mipmaps);
        }
    } else if (_t.format == eTextureFormat::BC1) {
        storage = 4;
        index = tex_storage_bc1_.AllocateRaw(_t.data, res, _t.mips_count, (_t.convention == eTextureConvention::DX),
                                             false /* invert_green */);

    } else if (_t.format == eTextureFormat::BC3) {
        storage = 5;
        index = tex_storage_bc3_.AllocateRaw(_t.data, res, _t.mips_count, (_t.convention == eTextureConvention::DX),
                                             false /* invert_green */);
    } else if (_t.format == eTextureFormat::BC4) {
        storage = 6;
        index = tex_storage_bc4_.AllocateRaw(_t.data, res, _t.mips_count, (_t.convention == eTextureConvention::DX),
                                             false /* invert_green */);
    } else if (_t.format == eTextureFormat::BC5) {
        storage = 7;
        const bool flip_vertical = (_t.convention == eTextureConvention::DX);
        const bool invert_green = (_t.convention == eTextureConvention::DX) && _t.is_normalmap;
        index = tex_storage_bc5_.AllocateRaw(_t.data, res, _t.mips_count, flip_vertical, invert_green);
        reconstruct_z = _t.is_normalmap;
    }

    if (storage == -1) {
        return InvalidTextureHandle;
    }

    log_->Info("Ray: Texture '%s' loaded (storage = %i, %ix%i)", _t.name ? _t.name : "(unknown)", storage, _t.w, _t.h);
    log_->Info("Ray: Storages are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC1[%i], BC3[%i], BC4[%i], BC5[%i])",
               tex_storage_rgba_.img_count(), tex_storage_rgb_.img_count(), tex_storage_rg_.img_count(),
               tex_storage_r_.img_count(), tex_storage_bc1_.img_count(), tex_storage_bc3_.img_count(),
               tex_storage_bc4_.img_count(), tex_storage_bc5_.img_count());

    uint32_t ret = 0;

    ret |= uint32_t(storage) << 28;
    if (_t.is_srgb) {
        ret |= TEX_SRGB_BIT;
    }
    if (reconstruct_z) {
        ret |= TEX_RECONSTRUCT_Z_BIT;
    }
    if (is_YCoCg) {
        ret |= TEX_YCOCG_BIT;
    }
    ret |= index;

    return TextureHandle{ret};
}

Ray::MaterialHandle Ray::Cpu::Scene::AddMaterial_nolock(const shading_node_desc_t &m) {
    material_t mat = {};

    mat.type = m.type;
    mat.textures[BASE_TEXTURE] = m.base_texture._index;
    mat.roughness_unorm = pack_unorm_16(clamp(m.roughness, 0.0f, 1.0f));
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
        if (m.multiple_importance) {
            mat.flags |= MAT_FLAG_MULT_IMPORTANCE;
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

    const std::pair<uint32_t, uint32_t> ret = materials_.push(mat);
    return MaterialHandle{ret.first, ret.second};
}

Ray::MaterialHandle Ray::Cpu::Scene::AddMaterial(const principled_mat_desc_t &m) {
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

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> rn = materials_.push(main_mat);
    auto root_node = MaterialHandle{rn.first, rn.second};
    MaterialHandle emissive_node = InvalidMaterialHandle, transparent_node = InvalidMaterialHandle;

    if (m.emission_strength > 0.0f &&
        (m.emission_color[0] > 0.0f || m.emission_color[1] > 0.0f || m.emission_color[2] > 0.0f)) {
        shading_node_desc_t emissive_desc;
        emissive_desc.type = eShadingNode::Emissive;

        memcpy(emissive_desc.base_color, m.emission_color, 3 * sizeof(float));
        emissive_desc.base_texture = m.emission_texture;
        emissive_desc.strength = m.emission_strength;
        emissive_desc.multiple_importance = m.multiple_importance;

        emissive_node = AddMaterial_nolock(emissive_desc);
    }

    if (m.alpha != 1.0f || m.alpha_texture != InvalidTextureHandle) {
        shading_node_desc_t transparent_desc;
        transparent_desc.type = eShadingNode::Transparent;

        transparent_node = AddMaterial_nolock(transparent_desc);
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

            root_node = AddMaterial_nolock(mix_node);
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

            root_node = AddMaterial_nolock(mix_node);
        }
    }

    return MaterialHandle{root_node};
}

Ray::MeshHandle Ray::Cpu::Scene::AddMesh(const mesh_desc_t &_m) {
    bvh_settings_t s;
    s.oversplit_threshold = 0.95f;
    s.allow_spatial_splits = _m.allow_spatial_splits;
    s.use_fast_bvh_build = _m.use_fast_bvh_build;

    const uint64_t t1 = Ray::GetTimeMs();

    std::vector<bvh_node_t> temp_nodes;
    aligned_vector<tri_accel_t> temp_tris;
    aligned_vector<mtri_accel_t> temp_mtris;
    std::vector<uint32_t> temp_tri_indices;

    PreprocessMesh(_m.vtx_positions, _m.vtx_indices, _m.base_vertex, s, temp_nodes, temp_tris, temp_tri_indices,
                   temp_mtris);

    log_->Info("Ray: Mesh \'%s\' preprocessed in %lldms", _m.name ? _m.name : "(unknown)", (Ray::GetTimeMs() - t1));

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> tris_index = tris_.Allocate(uint32_t(temp_tris.size()));

    mesh_t m = {};
    m.tris_index = tris_index.first;
    m.tris_block = tris_index.second;
    m.tris_count = uint32_t(temp_tris.size());
    for (uint32_t i = 0; i < uint32_t(temp_tris.size()); ++i) {
        tris_[m.tris_index + i] = temp_tris[i];
    }

    const std::pair<uint32_t, uint32_t> mtris_index = mtris_.Allocate(uint32_t(temp_mtris.size()));
    assert(mtris_index.first == m.tris_index / 8);
    for (uint32_t i = 0; i < uint32_t(temp_mtris.size()); ++i) {
        mtris_[mtris_index.first + i] = temp_mtris[i];
    }

    const std::pair<uint32_t, uint32_t> trimat_index =
        tri_materials_.Allocate(uint32_t(_m.vtx_indices.size() / 3), tri_mat_data_t{0xffff, 0xffff});
    const std::pair<uint32_t, uint32_t> tri_indices_index = tri_indices_.Allocate(uint32_t(temp_tri_indices.size()));
    assert(tri_indices_index.first == tris_index.first);
    assert(tri_indices_index.second == tris_index.second);
    for (uint32_t i = 0; i < uint32_t(temp_tri_indices.size()); ++i) {
        tri_indices_[tri_indices_index.first + i] = trimat_index.first + temp_tri_indices[i];
    }

    memcpy(m.bbox_min, temp_nodes[0].bbox_min, 3 * sizeof(float));
    memcpy(m.bbox_max, temp_nodes[0].bbox_max, 3 * sizeof(float));

    if (use_wide_bvh_) {
        const uint64_t t2 = Ray::GetTimeMs();

        aligned_vector<wbvh_node_t> temp_wnodes;
        temp_wnodes.reserve(temp_nodes.size() / 8);

        FlattenBVH_r(temp_nodes.data(), 0, 0xffffffff, temp_wnodes);
        const std::pair<uint32_t, uint32_t> wnodes_index = wnodes_.Allocate(uint32_t(temp_wnodes.size()));

        for (uint32_t i = 0; i < uint32_t(temp_wnodes.size()); ++i) {
            const wbvh_node_t &in_n = temp_wnodes[i];
            wbvh_node_t &out_n = wnodes_[wnodes_index.first + i];

            out_n = in_n;
            if (out_n.child[0] & LEAF_NODE_BIT) {
                out_n.child[0] += tri_indices_index.first;
            } else {
                for (int j = 0; j < 8; ++j) {
                    if (out_n.child[j] != 0x7fffffff) {
                        out_n.child[j] += wnodes_index.first;
                    }
                }
            }
        }

        m.node_index = wnodes_index.first;
        m.node_block = wnodes_index.second;

        log_->Info("Ray: Mesh \'%s\' BVH flattened in %lldms", _m.name ? _m.name : "(unknown)",
                   (Ray::GetTimeMs() - t2));
    } else {
        const std::pair<uint32_t, uint32_t> nodes_index = nodes_.Allocate(uint32_t(temp_nodes.size()));

        for (uint32_t i = 0; i < uint32_t(temp_nodes.size()); ++i) {
            const bvh_node_t &in_n = temp_nodes[i];
            bvh_node_t &out_n = nodes_[nodes_index.first + i];

            out_n = in_n;
            if (out_n.prim_index & LEAF_NODE_BIT) {
                out_n.prim_index += tri_indices_index.first;
            } else {
                out_n.left_child += nodes_index.first;
                out_n.right_child += nodes_index.first;
            }
        }

        m.node_index = nodes_index.first;
        m.node_block = nodes_index.second;
    }

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
            tri_mat_data_t &tri_mat = tri_materials_[trimat_index.first + uint32_t(i / 3)];

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

    std::vector<vertex_t> new_vertices(_m.vtx_positions.data.size() / _m.vtx_positions.stride);
    std::vector<uint32_t> new_vtx_indices(_m.vtx_indices.size());
    for (int i = 0; i < _m.vtx_indices.size(); ++i) {
        new_vtx_indices[i] = _m.vtx_indices[i] + _m.base_vertex;
    }

    // add attributes
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
        // NOTE: this may add some vertices and indices
        ComputeTangentBasis(0, 0, new_vertices, new_vtx_indices, new_vtx_indices);
    }

    const std::pair<uint32_t, uint32_t> vtx_index = vertices_.Allocate(uint32_t(new_vertices.size()));
    memcpy(&vertices_[vtx_index.first], new_vertices.data(), new_vertices.size() * sizeof(vertex_t));

    const std::pair<uint32_t, uint32_t> vtx_indices_index = vtx_indices_.Allocate(uint32_t(new_vtx_indices.size()));
    assert(trimat_index.second == vtx_indices_index.second);
    for (uint32_t i = 0; i < uint32_t(new_vtx_indices.size()); ++i) {
        vtx_indices_[vtx_indices_index.first + i] = vtx_index.first + new_vtx_indices[i];
    }

    m.vert_index = vtx_indices_index.first;
    m.vert_block = vtx_indices_index.second;
    m.vert_count = uint32_t(new_vtx_indices.size());

    m.vert_data_index = vtx_index.first;
    m.vert_data_block = vtx_index.second;

    const std::pair<uint32_t, uint32_t> ret = meshes_.emplace(m);
    return MeshHandle{ret.first, ret.second};
}

void Ray::Cpu::Scene::RemoveMesh_nolock(const MeshHandle i) {
    const mesh_t &m = meshes_[i._index];

    const uint32_t node_block = m.node_block;
    const uint32_t tris_block = m.tris_block;
    const uint32_t vert_block = m.vert_block, vert_data_block = m.vert_data_block;

    meshes_.Erase(i._block);

    bool rebuild_required = false;
    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end();) {
        mesh_instance_t &mi = *it;
        if (mi.mesh_index == i._index) {
            it = mesh_instances_.erase(it);
            rebuild_required = true;
        } else {
            ++it;
        }
    }
    (void)rebuild_required;

    tris_.Erase(tris_block);
    mtris_.Erase(tris_block);
    tri_indices_.Erase(tris_block);
    tri_materials_.Erase(tris_block);
    vertices_.Erase(vert_data_block);
    vtx_indices_.Erase(vert_block);
    if (use_wide_bvh_) {
        wnodes_.Erase(node_block);
    } else {
        nodes_.Erase(node_block);
    }
}

Ray::LightHandle Ray::Cpu::Scene::AddLight(const directional_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_DIR;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.blocking = false;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    l.dir.dir[0] = -_l.direction[0];
    l.dir.dir[1] = -_l.direction[1];
    l.dir.dir[2] = -_l.direction[2];
    l.dir.angle = _l.angle * PI / 360.0f;
    if (l.dir.angle != 0.0f) {
        const float radius = tanf(l.dir.angle);
        const float mul = 1.0f / (PI * radius * radius);
        l.col[0] *= mul;
        l.col[1] *= mul;
        l.col[2] *= mul;
    }

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

Ray::LightHandle Ray::Cpu::Scene::AddLight(const sphere_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible && (_l.radius > 0.0f);
    l.blocking = false;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;
    l.sph.spot = l.sph.blend = -1.0f;

    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> light_index = lights_.push(l);
    return LightHandle{light_index.first, light_index.second};
}

Ray::LightHandle Ray::Cpu::Scene::AddLight(const spot_light_desc_t &_l) {
    light_t l = {};

    l.type = LIGHT_TYPE_SPHERE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.blocking = false;

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

Ray::LightHandle Ray::Cpu::Scene::AddLight(const rect_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_RECT;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;
    l.blocking = _l.sky_portal;

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

Ray::LightHandle Ray::Cpu::Scene::AddLight(const disk_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_DISK;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;
    l.blocking = _l.sky_portal;

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

Ray::LightHandle Ray::Cpu::Scene::AddLight(const line_light_desc_t &_l, const float *xform) {
    light_t l = {};

    l.type = LIGHT_TYPE_LINE;
    l.cast_shadow = _l.cast_shadow;
    l.visible = _l.visible;
    l.sky_portal = _l.sky_portal;
    l.blocking = false;

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

Ray::MeshInstanceHandle Ray::Cpu::Scene::AddMeshInstance(const mesh_instance_desc_t &mi_desc) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> mi_index = mesh_instances_.emplace();

    mesh_instance_t &mi = mesh_instances_.at(mi_index.first);
    mi.mesh_index = mi_desc.mesh._index;
    mi.mesh_block = mi_desc.mesh._block;
    mi.lights_index = 0xffffffff;
    mi.ray_visibility = 0x000000ff;

    if (!mi_desc.camera_visibility) {
        mi.ray_visibility &= ~(1u << RAY_TYPE_CAMERA);
    }
    if (!mi_desc.diffuse_visibility) {
        mi.ray_visibility &= ~(1u << RAY_TYPE_DIFFUSE);
    }
    if (!mi_desc.specular_visibility) {
        mi.ray_visibility &= ~(1u << RAY_TYPE_SPECULAR);
    }
    if (!mi_desc.refraction_visibility) {
        mi.ray_visibility &= ~(1u << RAY_TYPE_REFR);
    }
    if (!mi_desc.shadow_visibility) {
        mi.ray_visibility &= ~(1u << RAY_TYPE_SHADOW);
    }

    { // find emissive triangles and add them as light emitters
        std::vector<light_t> new_lights;

        const mesh_t &m = meshes_[mi_desc.mesh._index];
        for (uint32_t tri = (m.vert_index / 3); tri < (m.vert_index + m.vert_count) / 3; ++tri) {
            const tri_mat_data_t &tri_mat = tri_materials_[tri];

            SmallVector<uint16_t, 64> mat_indices;
            mat_indices.push_back(tri_mat.front_mi & MATERIAL_INDEX_BITS);

            uint16_t front_emissive = 0xffff;
            for (int i = 0; i < int(mat_indices.size()); ++i) {
                const material_t &mat = materials_[mat_indices[i]];
                if (mat.type == eShadingNode::Emissive && (mat.flags & MAT_FLAG_MULT_IMPORTANCE)) {
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
                if (mat.type == eShadingNode::Emissive && (mat.flags & MAT_FLAG_MULT_IMPORTANCE)) {
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
                new_light.cast_shadow = 1;
                new_light.type = LIGHT_TYPE_TRI;
                new_light.doublesided = (back_emissive != 0xffff) ? 1 : 0;
                new_light.cast_shadow = 1;
                new_light.visible = 0;
                new_light.sky_portal = 0;
                new_light.blocking = 0;
                new_light.tri.tri_index = tri;
                new_light.tri.mi_index = mi_index.first;
                new_light.tri.tex_index = mat.textures[BASE_TEXTURE];
                new_light.col[0] = mat.base_color[0] * mat.strength;
                new_light.col[1] = mat.base_color[1] * mat.strength;
                new_light.col[2] = mat.base_color[2] * mat.strength;
            }
        }

        if (!new_lights.empty()) {
            const std::pair<uint32_t, uint32_t> lights_index = lights_.Allocate(uint32_t(new_lights.size()));
            for (uint32_t i = 0; i < uint32_t(new_lights.size()); ++i) {
                lights_[lights_index.first + i] = new_lights[i];
            }

            mi.lights_index = lights_index.first;
            assert(lights_index.second <= 0xffffff);
            mi.ray_visibility |= (lights_index.second << 8);
        }
    }

    const MeshInstanceHandle ret = {mi_index.first, mi_index.second};
    SetMeshInstanceTransform_nolock(ret, mi_desc.xform);

    return ret;
}

void Ray::Cpu::Scene::SetMeshInstanceTransform_nolock(const MeshInstanceHandle mi_handle, const float *xform) {
    mesh_instance_t &mi = mesh_instances_[mi_handle._index];

    memcpy(mi.xform, xform, 16 * sizeof(float));
    InverseMatrix(mi.xform, mi.inv_xform);

    const mesh_t &m = meshes_[mi.mesh_index];
    TransformBoundingBox(m.bbox_min, m.bbox_max, xform, mi.bbox_min, mi.bbox_max);
}

void Ray::Cpu::Scene::RemoveMeshInstance_nolock(const MeshInstanceHandle i) {
    mesh_instance_t &mi = mesh_instances_[i._index];

    if (mi.lights_index != 0xffffffff) {
        const uint32_t light_block = (mi.ray_visibility >> 8);
        lights_.Erase(light_block);
    }
    mesh_instances_.Erase(i._block);
}

void Ray::Cpu::Scene::Finalize(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
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
        PrepareSkyEnvMap_nolock(parallel_for);
        env_.sky_map_spread_angle = 2 * PI / float(env_.envmap_resolution);
    }

    if (env_.multiple_importance && env_.env_col[0] > 0.0f && env_.env_col[1] > 0.0f && env_.env_col[2] > 0.0f) {
        if (env_.env_map != InvalidTextureHandle._index) {
            PrepareEnvMapQTree_nolock();
        }
        { // add env light source
            light_t l = {};

            l.type = LIGHT_TYPE_ENV;
            l.visible = 1;
            l.cast_shadow = 1;
            l.col[0] = l.col[1] = l.col[2] = 1.0f;

            const std::pair<uint32_t, uint32_t> li = lights_.push(l);
            env_map_light_ = LightHandle{li.first, li.second};
            env_.light_index = env_map_light_._index;
        }
    }

    RebuildTLAS_nolock();
    RebuildLightTree_nolock();
}

void Ray::Cpu::Scene::RebuildTLAS_nolock() {
    if (tlas_root_ != 0xffffffff) {
        if (use_wide_bvh_) {
            wnodes_.Erase(tlas_block_);
        } else {
            nodes_.Erase(tlas_block_);
        }
        tlas_root_ = tlas_block_ = 0xffffffff;
    }
    mi_indices_.clear();

    if (mesh_instances_.empty()) {
        return;
    }

    aligned_vector<prim_t> primitives;
    primitives.reserve(mesh_instances_.size());

    for (const mesh_instance_t &mi : mesh_instances_) {
        primitives.push_back({0, 0, 0, Ref::fvec4{mi.bbox_min[0], mi.bbox_min[1], mi.bbox_min[2], 0.0f},
                              Ref::fvec4{mi.bbox_max[0], mi.bbox_max[1], mi.bbox_max[2], 0.0f}});
    }

    std::vector<bvh_node_t> temp_nodes;
    PreprocessPrims_SAH(primitives, {}, {}, temp_nodes, mi_indices_);

    if (use_wide_bvh_) {
        aligned_vector<wbvh_node_t> temp_wnodes;
        temp_wnodes.reserve(temp_nodes.size() / 8);

        FlattenBVH_r(temp_nodes.data(), 0, 0xffffffff, temp_wnodes);
        const std::pair<uint32_t, uint32_t> wnodes_index = wnodes_.Allocate(uint32_t(temp_wnodes.size()));

        for (uint32_t i = 0; i < uint32_t(temp_wnodes.size()); ++i) {
            const wbvh_node_t &in_n = temp_wnodes[i];
            wbvh_node_t &out_n = wnodes_[wnodes_index.first + i];

            out_n = in_n;
            if ((out_n.child[0] & LEAF_NODE_BIT) == 0) {
                for (int j = 0; j < 8; ++j) {
                    if (out_n.child[j] != 0x7fffffff) {
                        out_n.child[j] += wnodes_index.first;
                    }
                }
            }
        }

        tlas_root_ = wnodes_index.first;
        tlas_block_ = wnodes_index.second;
    } else {
        const std::pair<uint32_t, uint32_t> nodes_index = nodes_.Allocate(uint32_t(temp_nodes.size()));

        for (uint32_t i = 0; i < uint32_t(temp_nodes.size()); ++i) {
            const bvh_node_t &in_n = temp_nodes[i];
            bvh_node_t &out_n = nodes_[nodes_index.first + i];

            out_n = in_n;
            if ((out_n.prim_index & LEAF_NODE_BIT) == 0) {
                out_n.left_child += nodes_index.first;
                out_n.right_child += nodes_index.first;
            }
        }

        tlas_root_ = nodes_index.first;
        tlas_block_ = nodes_index.second;
    }
}

void Ray::Cpu::Scene::PrepareSkyEnvMap_nolock(
    const std::function<void(int, int, ParallelForFunction &&)> &parallel_for) {
    const uint64_t t1 = Ray::GetTimeMs();

    if (physical_sky_texture_ != InvalidTextureHandle) {
        tex_storages_[physical_sky_texture_._index >> 24]->Free(physical_sky_texture_._index & 0x00ffffff);
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

    const int SkyEnvRes[] = {env_.envmap_resolution, env_.envmap_resolution / 2};
    const std::vector<color_rgba8_t> rgbe_pixels =
        CalcSkyEnvTexture(env_.atmosphere, SkyEnvRes, lights_.data(), dir_lights_, parallel_for);

    const int storage = 0;
    const int index = tex_storage_rgba_.Allocate(rgbe_pixels, SkyEnvRes, false);

    physical_sky_texture_._index = (uint32_t(storage) << 28) | index;

    env_.env_map = physical_sky_texture_._index;
    if (env_.back_map == PhysicalSkyTexture._index) {
        env_.back_map = physical_sky_texture_._index;
    }

    log_->Info("PrepareSkyEnvMap (%ix%i) done in %lldms", SkyEnvRes[0], SkyEnvRes[1], GetTimeMs() - t1);
}

void Ray::Cpu::Scene::PrepareEnvMapQTree_nolock() {
    const int tex = int(env_.env_map & 0x00ffffff);
    Ref::ivec2 size;
    tex_storage_rgba_.GetIRes(tex, 0, value_ptr(size));

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
                        Ref::fvec4 dir;
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

                        const Ref::fvec2 uvs = Ref::fvec2{u, theta} * Ref::fvec2(size);
                        const Ref::ivec2 iuvs = clamp(Ref::ivec2(uvs), Ref::ivec2(0), size - 1);

                        const color_rgba8_t col_rgbe = tex_storage_rgba_.Get(tex, iuvs.get<0>(), iuvs.get<1>(), 0);
                        const Ref::fvec4 col_rgb = Ref::rgbe_to_rgb(col_rgbe);
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

        for (const Ref::fvec4 &v : env_map_qtree_.mips[0]) {
            total_lum += hsum(v);
        }

        cur_res /= 2;
    }

    env_map_qtree_.medium_lum = total_lum / float(cur_res * cur_res);

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

    static const float LumFractThreshold = 0.005f;

    cur_res = 2;
    int the_last_required_lod = 0;
    for (int lod = int(env_map_qtree_.mips.size()) - 1; lod >= 0; --lod) {
        the_last_required_lod = lod;
        const auto &cur_mip = env_map_qtree_.mips[lod];

        bool subdivision_required = false;
        for (int y = 0; y < (cur_res / 2) && !subdivision_required; ++y) {
            for (int x = 0; x < (cur_res / 2) && !subdivision_required; ++x) {
                const Ref::ivec4 mask = simd_cast(cur_mip[y * cur_res / 2 + x] > LumFractThreshold * total_lum);
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

    log_->Info("Env map qtree res is %i", env_map_qtree_.res);
}

void Ray::Cpu::Scene::RebuildLightTree_nolock() {
    aligned_vector<prim_t> primitives;
    primitives.reserve(lights_.size());

    struct additional_data_t {
        Ref::fvec4 axis;
        float flux, omega_n, omega_e;
    };
    aligned_vector<additional_data_t> additional_data;
    additional_data.reserve(lights_.size());

    visible_lights_count_ = blocker_lights_count_ = 0;
    li_indices_.clear();

    for (auto it = lights_.cbegin(); it != lights_.cend(); ++it) {
        const light_t &l = *it;
        if (l.type == LIGHT_TYPE_DIR && physical_sky_texture_ != InvalidTextureHandle) {
            // Directional lights are already 'baked' into sky texture
            continue;
        }

        Ref::fvec4 bbox_min = 0.0f, bbox_max = 0.0f, axis = {0.0f, 1.0f, 0.0f, 0.0f};
        float area = 1.0f, omega_n = 0.0f, omega_e = 0.0f;
        float lum = l.col[0] + l.col[1] + l.col[2];

        li_indices_.push_back(it.index());
        if (l.visible) {
            ++visible_lights_count_;
        }
        if (l.blocking) {
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
            Ref::fvec4 light_v = Ray::Cpu::cross(light_u, light_dir);

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

            axis = normalize(Ray::Cpu::cross(u, v));
            omega_n = 0.0f; // single normal
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

            axis = normalize(Ray::Cpu::cross(u, v));
            omega_n = 0.0f; // single normal
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

            Ref::fvec4 light_forward = Ray::Cpu::cross(p2 - p1, p3 - p1);
            area = 0.5f * length(light_forward);

            axis = normalize(light_forward);
            omega_n = PI; // normals in all directions (triangle lights are double-sided)
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

    light_nodes_.clear();
    light_wnodes_.clear();

    if (primitives.empty()) {
        return;
    }

    std::vector<uint32_t> prim_indices;
    prim_indices.reserve(primitives.size());

    std::vector<bvh_node_t> temp_nodes;

    bvh_settings_t s;
    s.oversplit_threshold = -1.0f;
    s.allow_spatial_splits = false;
    s.min_primitives_in_leaf = 1;
    PreprocessPrims_SAH(primitives, {}, s, temp_nodes, prim_indices);

    light_nodes_.resize(temp_nodes.size(), light_bvh_node_t{});
    for (uint32_t i = 0; i < temp_nodes.size(); ++i) {
        static_cast<bvh_node_t &>(light_nodes_[i]) = temp_nodes[i];
        if ((temp_nodes[i].prim_index & LEAF_NODE_BIT) != 0) {
            const uint32_t prim_index = prim_indices[temp_nodes[i].prim_index & PRIM_INDEX_BITS];
            memcpy(light_nodes_[i].axis, value_ptr(additional_data[prim_index].axis), 3 * sizeof(float));
            light_nodes_[i].flux = additional_data[prim_index].flux;
            light_nodes_[i].omega_n = additional_data[prim_index].omega_n;
            light_nodes_[i].omega_e = additional_data[prim_index].omega_e;
        }
    }

    std::vector<uint32_t> parent_indices(light_nodes_.size());
    parent_indices[0] = 0xffffffff; // root node has no parent

    std::vector<uint32_t> leaf_indices;
    leaf_indices.reserve(primitives.size());

    SmallVector<uint32_t, 128> stack;
    stack.push_back(0);
    while (!stack.empty()) {
        const uint32_t i = stack.back();
        stack.pop_back();

        if ((light_nodes_[i].prim_index & LEAF_NODE_BIT) == 0) {
            const uint32_t left_child = (light_nodes_[i].left_child & LEFT_CHILD_BITS),
                           right_child = (light_nodes_[i].right_child & RIGHT_CHILD_BITS);
            parent_indices[left_child] = parent_indices[right_child] = i;

            stack.push_back(left_child);
            stack.push_back(right_child);
        } else {
            leaf_indices.push_back(i);
        }
    }

    // Propagate flux and cone up the hierarchy
    std::vector<uint32_t> to_process;
    to_process.reserve(light_nodes_.size());
    to_process.insert(end(to_process), begin(leaf_indices), end(leaf_indices));
    for (uint32_t i = 0; i < uint32_t(to_process.size()); ++i) {
        const uint32_t n = to_process[i];
        const uint32_t parent = parent_indices[n];
        if (parent == 0xffffffff) {
            continue;
        }

        light_nodes_[parent].flux += light_nodes_[n].flux;
        if (light_nodes_[parent].axis[0] == 0.0f && light_nodes_[parent].axis[1] == 0.0f &&
            light_nodes_[parent].axis[2] == 0.0f) {
            memcpy(light_nodes_[parent].axis, light_nodes_[n].axis, 3 * sizeof(float));
            light_nodes_[parent].omega_n = light_nodes_[n].omega_n;
        } else {
            auto axis1 = Ref::fvec4{light_nodes_[parent].axis}, axis2 = Ref::fvec4{light_nodes_[n].axis};
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

            memcpy(light_nodes_[parent].axis, value_ptr(axis1), 3 * sizeof(float));

            light_nodes_[parent].omega_n =
                fminf(0.5f * (light_nodes_[parent].omega_n +
                              fmaxf(light_nodes_[parent].omega_n, angle_between + light_nodes_[n].omega_n)),
                      PI);
        }
        light_nodes_[parent].omega_e = fmaxf(light_nodes_[parent].omega_e, light_nodes_[n].omega_e);
        if ((light_nodes_[parent].left_child & LEFT_CHILD_BITS) == n) {
            to_process.push_back(parent);
        }
    }

    // Remove indices indirection
    for (uint32_t i = 0; i < leaf_indices.size(); ++i) {
        light_bvh_node_t &n = light_nodes_[leaf_indices[i]];
        assert((n.prim_index & LEAF_NODE_BIT) != 0);
        const uint32_t li_index = li_indices_[prim_indices[n.prim_index & PRIM_INDEX_BITS]];
        n.prim_index &= ~PRIM_INDEX_BITS;
        n.prim_index |= li_index;
    }

    if (use_wide_bvh_) {
        const uint32_t root_node = FlattenBVH_r(light_nodes_.data(), 0, 0xffffffff, light_wnodes_);
        assert(root_node == 0);
        (void)root_node;
        light_nodes_.clear();
    }
}

void Ray::Cpu::Scene::GetBounds(float bbox_min[3], float bbox_max[3]) const {
    bbox_min[0] = bbox_min[1] = bbox_min[2] = MAX_DIST;
    bbox_max[0] = bbox_max[1] = bbox_max[2] = -MAX_DIST;
    if (tlas_root_ != 0xffffffff) {
        if (use_wide_bvh_) {
            const wbvh_node_t &root_node = wnodes_[tlas_root_];
            if (root_node.child[0] & LEAF_NODE_BIT) {
                for (int i = 0; i < 3; ++i) {
                    bbox_min[i] = root_node.bbox_min[i][0];
                    bbox_max[i] = root_node.bbox_max[i][0];
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    if (root_node.child[j] == 0x7fffffff) {
                        continue;
                    }
                    for (int i = 0; i < 3; ++i) {
                        bbox_min[i] = fminf(bbox_min[i], root_node.bbox_min[i][j]);
                        bbox_max[i] = fmaxf(bbox_max[i], root_node.bbox_max[i][j]);
                    }
                }
            }
        } else {
            const bvh_node_t &root_node = nodes_[tlas_root_];
            for (int i = 0; i < 3; ++i) {
                bbox_min[i] = root_node.bbox_min[i];
                bbox_max[i] = root_node.bbox_max[i];
            }
        }
    }
    if (!light_wnodes_.empty()) {
        if (use_wide_bvh_) {
            const wbvh_node_t &root_node = light_wnodes_[0];
            if (root_node.child[0] & LEAF_NODE_BIT) {
                for (int i = 0; i < 3; ++i) {
                    bbox_min[i] = fminf(bbox_min[i], root_node.bbox_min[i][0]);
                    bbox_max[i] = fmaxf(bbox_max[i], root_node.bbox_max[i][0]);
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    if (root_node.child[j] == 0x7fffffff) {
                        continue;
                    }
                    for (int i = 0; i < 3; ++i) {
                        bbox_min[i] = fminf(bbox_min[i], root_node.bbox_min[i][j]);
                        bbox_max[i] = fmaxf(bbox_max[i], root_node.bbox_max[i][j]);
                    }
                }
            }
        } else {
            const bvh_node_t &root_node = light_nodes_[0];
            for (int i = 0; i < 3; ++i) {
                bbox_min[i] = fminf(bbox_min[i], root_node.bbox_min[i]);
                bbox_max[i] = fmaxf(bbox_max[i], root_node.bbox_max[i]);
            }
        }
    }
}
