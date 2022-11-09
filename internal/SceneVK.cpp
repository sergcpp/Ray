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

#define _MIN(x, y) ((x) < (y) ? (x) : (y))
#define _MAX(x, y) ((x) < (y) ? (y) : (x))
#define _ABS(x) ((x) < 0 ? -(x) : (x))
#define _CLAMP(x, lo, hi) (_MIN(_MAX((x), (lo)), (hi)))

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

Ray::Vk::Scene::Scene(Context *ctx, const bool use_hwrt, const bool use_tex_compression)
    : ctx_(ctx), use_hwrt_(use_hwrt), use_tex_compression_(use_tex_compression), nodes_(ctx), tris_(ctx),
      tri_indices_(ctx), tri_materials_(ctx), transforms_(ctx, "Transforms"), meshes_(ctx, "Meshes"),
      mesh_instances_(ctx, "Mesh Instances"), mi_indices_(ctx), vertices_(ctx), vtx_indices_(ctx),
      materials_(ctx, "Materials"),
      textures_(ctx, "Textures"), tex_atlases_{{ctx, eTexFormat::RawRGBA8888, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                                               {ctx, eTexFormat::RawRGB888, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                                               {ctx, eTexFormat::RawRG88, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                                               {ctx, eTexFormat::RawR8, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                                               {ctx, eTexFormat::BC3, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE},
                                               {ctx, eTexFormat::BC4, TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE}},
      lights_(ctx, "Lights"), li_indices_(ctx), visible_lights_(ctx) {}

void Ray::Vk::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(&env.env_col[0], &env_.env_col, 3 * sizeof(float));
    env.env_clamp = env_.env_clamp;
    env.env_map = env_.env_map;
}

void Ray::Vk::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(&env_.env_col, &env.env_col[0], 3 * sizeof(float));
    env_.env_clamp = env.env_clamp;
    env_.env_map = env.env_map;
}

uint32_t Ray::Vk::Scene::AddTexture(const tex_desc_t &_t) {
    texture_t t;
    t.width = uint16_t(_t.w);
    t.height = uint16_t(_t.h);

    if (_t.is_srgb) {
        t.width |= TEXTURE_SRGB_BIT;
    }

    if (_t.generate_mipmaps) {
        t.height |= TEXTURE_MIPS_BIT;
    }

    int res[2] = {_t.w, _t.h};

    std::vector<color_rgba8_t> tex_data_rgba8;
    std::vector<color_rgb8_t> tex_data_rgb8;
    std::vector<color_rg8_t> tex_data_rg8;
    std::vector<color_r8_t> tex_data_r8;

    const bool use_compression = use_tex_compression_ && !_t.force_no_compression;

    if (_t.format == eTextureFormat::RGBA8888) {
        t.atlas = 0;
    } else if (_t.format == eTextureFormat::RGB888) {
        if (use_compression && _t.is_srgb) {
            t.atlas = 4;
        } else {
            t.atlas = 1;
        }
    } else if (_t.format == eTextureFormat::RG88) {
        t.atlas = 2;
    } else if (_t.format == eTextureFormat::R8) {
        if (use_compression) {
            t.atlas = 5;
        } else {
            t.atlas = 3;
        }
    }

    { // Allocate initial mip level
        int page = -1, pos[2];

        if (_t.format == eTextureFormat::RGBA8888) {
            page = tex_atlases_[0].Allocate<uint8_t, 4>(reinterpret_cast<const color_rgba8_t *>(_t.data), res, pos);
        } else if (_t.format == eTextureFormat::RGB888) {
            page =
                tex_atlases_[t.atlas].Allocate<uint8_t, 3>(reinterpret_cast<const color_rgb8_t *>(_t.data), res, pos);
        } else if (_t.format == eTextureFormat::RG88) {
            page = tex_atlases_[2].Allocate<uint8_t, 2>(reinterpret_cast<const color_rg8_t *>(_t.data), res, pos);
        } else if (_t.format == eTextureFormat::R8) {
            page = tex_atlases_[t.atlas].Allocate<uint8_t, 1>(reinterpret_cast<const color_r8_t *>(_t.data), res, pos);
        }

        if (page == -1) {
            return 0xffffffff;
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
        } else if (_t.format == eTextureFormat::R8) {
            tex_atlases_[t.atlas].AllocateMips<uint8_t, 1>(reinterpret_cast<const color_r8_t *>(_t.data), res,
                                                           NUM_MIP_LEVELS - 1, pages, positions);
        } else {
            return 0xffffffff;
        }

        for (int i = 1; i < NUM_MIP_LEVELS; i++) {
            t.page[i] = uint8_t(pages[i - 1]);
            t.pos[i][0] = uint16_t(positions[i - 1][0]);
            t.pos[i][1] = uint16_t(positions[i - 1][1]);
        }
    }

    ctx_->log()->Info("Ray: Texture loaded (atlas = %i, %ix%i)", int(t.atlas), _t.w, _t.h);
    ctx_->log()->Info("Ray: Atlasses are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC3[%i], BC4[%i])",
                      tex_atlases_[0].page_count(), tex_atlases_[1].page_count(), tex_atlases_[2].page_count(),
                      tex_atlases_[3].page_count(), tex_atlases_[4].page_count(), tex_atlases_[5].page_count());

    return textures_.push(t);
}

uint32_t Ray::Vk::Scene::AddMaterial(const shading_node_desc_t &m) {
    material_t mat;

    mat.type = m.type;
    mat.textures[BASE_TEXTURE] = m.base_texture;
    mat.roughness_unorm = pack_unorm_16(m.roughness);
    mat.textures[ROUGH_TEXTURE] = m.roughness_texture;
    memcpy(&mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    mat.int_ior = m.int_ior;
    mat.ext_ior = m.ext_ior;
    mat.tangent_rotation = 0.0f;
    mat.flags = 0;

    if (m.type == DiffuseNode) {
        mat.sheen_unorm = pack_unorm_16(_CLAMP(m.sheen, 0.0f, 1.0f));
        mat.sheen_tint_unorm = pack_unorm_16(_CLAMP(m.tint, 0.0f, 1.0f));
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
    } else if (m.type == GlossyNode) {
        mat.tangent_rotation = 2.0f * PI * m.anisotropic_rotation;
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
        mat.tint_unorm = pack_unorm_16(_CLAMP(m.tint, 0.0f, 1.0f));
    } else if (m.type == RefractiveNode) {
    } else if (m.type == EmissiveNode) {
        mat.strength = m.strength;
        if (m.multiple_importance) {
            mat.flags |= MAT_FLAG_MULT_IMPORTANCE;
        }
        if (m.sky_portal) {
            mat.flags |= MAT_FLAG_SKY_PORTAL;
        }
    } else if (m.type == MixNode) {
        mat.strength = m.strength;
        mat.textures[MIX_MAT1] = m.mix_materials[0];
        mat.textures[MIX_MAT2] = m.mix_materials[1];
        if (m.mix_add) {
            mat.flags |= MAT_FLAG_MIX_ADD;
        }
    } else if (m.type == TransparentNode) {
    }

    mat.textures[NORMALS_TEXTURE] = m.normal_map;
    mat.normal_map_strength_unorm = pack_unorm_16(_CLAMP(m.normal_map_intensity, 0.0f, 1.0f));

    return materials_.push(mat);
}

uint32_t Ray::Vk::Scene::AddMaterial(const principled_mat_desc_t &m) {
    material_t main_mat;

    main_mat.type = PrincipledNode;
    main_mat.textures[BASE_TEXTURE] = m.base_texture;
    memcpy(&main_mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    main_mat.sheen_unorm = pack_unorm_16(_CLAMP(m.sheen, 0.0f, 1.0f));
    main_mat.sheen_tint_unorm = pack_unorm_16(_CLAMP(m.sheen_tint, 0.0f, 1.0f));
    main_mat.roughness_unorm = pack_unorm_16(_CLAMP(m.roughness, 0.0f, 1.0f));
    main_mat.tangent_rotation = 2.0f * PI * _CLAMP(m.anisotropic_rotation, 0.0f, 1.0f);
    main_mat.textures[ROUGH_TEXTURE] = m.roughness_texture;
    main_mat.metallic_unorm = pack_unorm_16(_CLAMP(m.metallic, 0.0f, 1.0f));
    main_mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
    main_mat.int_ior = m.ior;
    main_mat.ext_ior = 1.0f;
    main_mat.flags = 0;
    main_mat.transmission_unorm = pack_unorm_16(_CLAMP(m.transmission, 0.0f, 1.0f));
    main_mat.transmission_roughness_unorm = pack_unorm_16(_CLAMP(m.transmission_roughness, 0.0f, 1.0f));
    main_mat.textures[NORMALS_TEXTURE] = m.normal_map;
    main_mat.normal_map_strength_unorm = pack_unorm_16(_CLAMP(m.normal_map_intensity, 0.0f, 1.0f));
    main_mat.anisotropic_unorm = pack_unorm_16(_CLAMP(m.anisotropic, 0.0f, 1.0f));
    main_mat.specular_unorm = pack_unorm_16(_CLAMP(m.specular, 0.0f, 1.0f));
    main_mat.specular_tint_unorm = pack_unorm_16(_CLAMP(m.specular_tint, 0.0f, 1.0f));
    main_mat.clearcoat_unorm = pack_unorm_16(_CLAMP(m.clearcoat, 0.0f, 1.0f));
    main_mat.clearcoat_roughness_unorm = pack_unorm_16(_CLAMP(m.clearcoat_roughness, 0.0f, 1.0f));

    uint32_t root_node = materials_.push(main_mat);
    uint32_t emissive_node = 0xffffffff, transparent_node = 0xffffffff;

    if (m.emission_strength > 0.0f &&
        (m.emission_color[0] > 0.0f || m.emission_color[1] > 0.0f || m.emission_color[2] > 0.0f)) {
        shading_node_desc_t emissive_desc;
        emissive_desc.type = EmissiveNode;

        memcpy(emissive_desc.base_color, m.emission_color, 3 * sizeof(float));
        emissive_desc.base_texture = m.emission_texture;
        emissive_desc.strength = m.emission_strength;

        emissive_node = AddMaterial(emissive_desc);
    }

    if (m.alpha != 1.0f || m.alpha_texture != 0xffffffff) {
        shading_node_desc_t transparent_desc;
        transparent_desc.type = TransparentNode;

        transparent_node = AddMaterial(transparent_desc);
    }

    if (emissive_node != 0xffffffff) {
        if (root_node == 0xffffffff) {
            root_node = emissive_node;
        } else {
            shading_node_desc_t mix_node;
            mix_node.type = MixNode;
            mix_node.base_texture = 0xffffffff;
            mix_node.strength = 0.5f;
            mix_node.int_ior = mix_node.ext_ior = 0.0f;
            mix_node.mix_add = true;

            mix_node.mix_materials[0] = root_node;
            mix_node.mix_materials[1] = emissive_node;

            root_node = AddMaterial(mix_node);
        }
    }

    if (transparent_node != 0xffffffff) {
        if (root_node == 0xffffffff || m.alpha == 0.0f) {
            root_node = transparent_node;
        } else {
            shading_node_desc_t mix_node;
            mix_node.type = MixNode;
            mix_node.base_texture = m.alpha_texture;
            mix_node.strength = m.alpha;
            mix_node.int_ior = mix_node.ext_ior = 0.0f;

            mix_node.mix_materials[0] = transparent_node;
            mix_node.mix_materials[1] = root_node;

            root_node = AddMaterial(mix_node);
        }
    }

    return root_node;
}

uint32_t Ray::Vk::Scene::AddMesh(const mesh_desc_t &_m) {
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
        for (int j = 0; j < _m.vtx_indices_count; j += 3) {
            Ref::simd_fvec4 p[3];

            const uint32_t i0 = _m.vtx_indices[j + 0], i1 = _m.vtx_indices[j + 1], i2 = _m.vtx_indices[j + 2];

            memcpy(&p[0][0], &_m.vtx_attrs[i0 * attr_stride], 3 * sizeof(float));
            memcpy(&p[1][0], &_m.vtx_attrs[i1 * attr_stride], 3 * sizeof(float));
            memcpy(&p[2][0], &_m.vtx_attrs[i2 * attr_stride], 3 * sizeof(float));

            bbox_min = min(bbox_min, min(p[0], min(p[1], p[2])));
            bbox_max = max(bbox_max, max(p[0], max(p[1], p[2])));
        }
    } else {
        aligned_vector<mtri_accel_t> _unused;
        PreprocessMesh(_m.vtx_attrs, {_m.vtx_indices, _m.vtx_indices_count}, _m.layout, _m.base_vertex,
                       0 /* temp value */, s, new_nodes, new_tris, new_tri_indices, _unused);

        memcpy(&bbox_min[0], new_nodes[0].bbox_min, 3 * sizeof(float));
        memcpy(&bbox_max[0], new_nodes[0].bbox_max, 3 * sizeof(float));
    }

    std::vector<tri_mat_data_t> new_tri_materials(_m.vtx_indices_count / 3);

    // init triangle materials
    for (const shape_desc_t &s : _m.shapes) {
        bool is_front_solid = true, is_back_solid = true;

        uint32_t material_stack[32];
        material_stack[0] = s.mat_index;
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

        material_stack[0] = s.back_mat_index;
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

        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            tri_mat_data_t &tri_mat = new_tri_materials[i / 3];

            assert(s.mat_index < (1 << 14) && "Not enough bits to reference material!");
            assert(s.back_mat_index < (1 << 14) && "Not enough bits to reference material!");

            tri_mat.front_mi = uint16_t(s.mat_index);
            if (is_front_solid) {
                tri_mat.front_mi |= MATERIAL_SOLID_BIT;
            }

            tri_mat.back_mi = uint16_t(s.back_mat_index);
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
    mesh_t m;
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

    return mesh_index;
}

void Ray::Vk::Scene::RemoveMesh(uint32_t) {
    // TODO!!!
}

uint32_t Ray::Vk::Scene::AddLight(const directional_light_desc_t &_l) {
    light_t l;

    l.type = LIGHT_TYPE_DIR;
    l.visible = false;
    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    l.dir.dir[0] = -_l.direction[0];
    l.dir.dir[1] = -_l.direction[1];
    l.dir.dir[2] = -_l.direction[2];
    l.dir.angle = _l.angle * PI / 360.0f;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);
    return light_index;
}

uint32_t Ray::Vk::Scene::AddLight(const sphere_light_desc_t &_l) {
    light_t l;

    l.type = LIGHT_TYPE_SPHERE;
    l.visible = _l.visible;

    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    memcpy(&l.sph.pos[0], &_l.position[0], 3 * sizeof(float));

    l.sph.area = 4.0f * PI * _l.radius * _l.radius;
    l.sph.radius = _l.radius;

    const uint32_t light_index = lights_.push(l);
    li_indices_.PushBack(light_index);

    if (_l.visible) {
        visible_lights_.PushBack(light_index);
    }
    return light_index;
}

uint32_t Ray::Vk::Scene::AddLight(const rect_light_desc_t &_l, const float *xform) {
    light_t l;

    l.type = LIGHT_TYPE_RECT;
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
    return light_index;
}

uint32_t Ray::Vk::Scene::AddLight(const disk_light_desc_t &_l, const float *xform) {
    light_t l;

    l.type = LIGHT_TYPE_DISK;
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
    return light_index;
}

void Ray::Vk::Scene::RemoveLight(const uint32_t i) {
    if (!lights_.exists(i)) {
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

    lights_.erase(i);
}

uint32_t Ray::Vk::Scene::AddMeshInstance(const uint32_t mesh_index, const float *xform) {
    mesh_instance_t mi;
    mi.mesh_index = mesh_index;
    mi.tr_index = transforms_.emplace();

    const uint32_t mi_index = mesh_instances_.push(mi);

    { // find emissive triangles and add them as emitters
        const mesh_t &m = meshes_[mesh_index];
        for (uint32_t tri = (m.vert_index / 3); tri < (m.vert_index + m.vert_count) / 3; ++tri) {
            const tri_mat_data_t &tri_mat = tri_materials_cpu_[tri];

            const material_t &front_mat = materials_[tri_mat.front_mi & MATERIAL_INDEX_BITS];
            if (front_mat.type == EmissiveNode &&
                (front_mat.flags & (MAT_FLAG_MULT_IMPORTANCE | MAT_FLAG_SKY_PORTAL))) {
                light_t new_light;
                new_light.type = LIGHT_TYPE_TRI;
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

    SetMeshInstanceTransform(mi_index, xform);

    return mi_index;
}

void Ray::Vk::Scene::SetMeshInstanceTransform(const uint32_t mi_index, const float *xform) {
    transform_t tr;

    memcpy(tr.xform, xform, 16 * sizeof(float));
    InverseMatrix(tr.xform, tr.inv_xform);

    mesh_instance_t mi = mesh_instances_[mi_index];

    const mesh_t &m = meshes_[mi.mesh_index];
    TransformBoundingBox(m.bbox_min, m.bbox_max, xform, mi.bbox_min, mi.bbox_max);

    mesh_instances_.Set(mi_index, mi);
    transforms_.Set(mi.tr_index, tr);

    RebuildTLAS();
}

void Ray::Vk::Scene::RemoveMeshInstance(uint32_t) {
    // TODO!!
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
        primitives.push_back({0, 0, 0, Ref::simd_fvec4{mi.bbox_min}, Ref::simd_fvec4{mi.bbox_max}});
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

void Ray::Vk::Scene::GenerateTextureMips() {
    struct mip_gen_info {
        uint32_t texture_index;
        uint16_t size; // used for sorting
        uint8_t dst_mip;
        uint8_t atlas_index; // used for sorting
    };

    std::vector<mip_gen_info> mips_to_generate;
    mips_to_generate.reserve(textures_.size());

    for (uint32_t i = 0; i < uint32_t(textures_.size()); ++i) {
        const texture_t &t = textures_[i];
        if ((t.height & TEXTURE_MIPS_BIT) == 0) {
            continue;
        }

        int mip = 0;
        int res[2] = {(t.width & TEXTURE_WIDTH_BITS), (t.height & TEXTURE_HEIGHT_BITS)};

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
        texture_t t = textures_[info.texture_index];

        const int dst_mip = info.dst_mip;
        const int src_mip = dst_mip - 1;
        const int src_res[2] = {(t.width & TEXTURE_WIDTH_BITS) >> src_mip, (t.height & TEXTURE_HEIGHT_BITS) >> src_mip};
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

        textures_.Set(info.texture_index, t);
    }

    ctx_->log()->Info("Ray: Atlasses are (RGBA[%i], RGB[%i], RG[%i], R[%i], BC3[%i], BC4[%i])",
                      tex_atlases_[0].page_count(), tex_atlases_[1].page_count(), tex_atlases_[2].page_count(),
                      tex_atlases_[3].page_count(), tex_atlases_[4].page_count(), tex_atlases_[5].page_count());
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
        VkAccelerationStructureBuildSizesInfoKHR size_info;
        VkAccelerationStructureBuildGeometryInfoKHR build_info;
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
        assert(res == VK_SUCCESS);

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

        const mesh_t &mesh = meshes_[instance.mesh_index];
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

        const uint32_t instance_count = uint32_t(tlas_instances.size());
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

        tlas_scratch_buf =
            Buffer{"TLAS Scratch Buf", ctx_, eBufType::Storage, uint32_t(size_info.buildScratchSize)};
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
