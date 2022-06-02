#include "SceneRef.h"

#include <cassert>
#include <cstring>

#include "TextureUtilsRef.h"
#include "Time_.h"

#define _CLAMP(val, min, max) (val < min ? min : (val > max ? max : val))

Ray::Ref::Scene::Scene(ILog *log, const bool use_wide_bvh)
    : log_(log), use_wide_bvh_(use_wide_bvh), texture_atlas_(TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE) {}

Ray::Ref::Scene::~Scene() {
    for (uint32_t i = 0; i < uint32_t(mesh_instances_.size()); ++i) {
        RemoveMeshInstance(i);
    }
    for (uint32_t i = 0; i < uint32_t(meshes_.size()); ++i) {
        RemoveMesh(i);
    }
    materials_.clear();
    textures_.clear();
}

void Ray::Ref::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(&env.env_col[0], &env_.env_col, 3 * sizeof(float));
    env.env_clamp = env_.env_clamp;
    env.env_map = env_.env_map;
}

void Ray::Ref::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(&env_.env_col, &env.env_col[0], 3 * sizeof(float));
    env_.env_clamp = env.env_clamp;
    env_.env_map = env.env_map;
}

uint32_t Ray::Ref::Scene::AddTexture(const tex_desc_t &_t) {
    const auto tex_index = uint32_t(textures_.size());

    texture_t t;
    t.width = uint16_t(_t.w);
    t.height = uint16_t(_t.h);

    if (_t.is_srgb) {
        t.width |= TEXTURE_SRGB_BIT;
    }

    if (_t.generate_mipmaps) {
        t.height |= TEXTURE_MIPS_BIT;
    }

    int mip = 0;
    int res[2] = {_t.w, _t.h};

    std::vector<pixel_color8_t> tex_data(_t.data, _t.data + _t.w * _t.h);

    while (res[0] >= 1 && res[1] >= 1) {
        int pos[2];
        const int page = texture_atlas_.Allocate(&tex_data[0], res, pos);
        if (page == -1) {
            // release allocated mip levels on fail
            for (int i = mip; i >= 0; i--) {
                const int _pos[2] = {t.pos[i][0], t.pos[i][1]};
                texture_atlas_.Free(t.page[i], _pos);
            }
            return 0xffffffff;
        }

        t.page[mip] = uint8_t(page);
        t.pos[mip][0] = uint16_t(pos[0]);
        t.pos[mip][1] = uint16_t(pos[1]);

        mip++;

        if (_t.generate_mipmaps) {
            tex_data = Ref::DownsampleTexture(tex_data.data(), res);

            res[0] /= 2;
            res[1] /= 2;
        } else {
            break;
        }
    }

    // fill remaining mip levels with the last one
    for (int i = mip; i < NUM_MIP_LEVELS; i++) {
        t.page[i] = t.page[mip - 1];
        t.pos[i][0] = t.pos[mip - 1][0];
        t.pos[i][1] = t.pos[mip - 1][1];
    }

    return textures_.push(t);
}

uint32_t Ray::Ref::Scene::AddMaterial(const shading_node_desc_t &m) {
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
        mat.metallic = m.metallic;
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
    } else if (m.type == GlossyNode) {
        mat.tangent_rotation = 2.0f * PI * m.anisotropic_rotation;
        mat.metallic = m.metallic;
        mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
        mat.tint = m.tint;
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

    return materials_.push(mat);
}

uint32_t Ray::Ref::Scene::AddMaterial(const principled_mat_desc_t &m) {
    material_t main_mat;

    main_mat.type = PrincipledNode;
    main_mat.textures[BASE_TEXTURE] = m.base_texture;
    memcpy(&main_mat.base_color[0], &m.base_color[0], 3 * sizeof(float));
    main_mat.sheen_unorm = pack_unorm_16(_CLAMP(m.sheen, 0.0f, 1.0f));
    main_mat.sheen_tint_unorm = pack_unorm_16(_CLAMP(m.sheen_tint, 0.0f, 1.0f));
    main_mat.roughness_unorm = pack_unorm_16(_CLAMP(m.roughness, 0.0f, 1.0f));
    main_mat.tangent_rotation = 2.0f * PI * _CLAMP(m.anisotropic_rotation, 0.0f, 1.0f);
    main_mat.textures[ROUGH_TEXTURE] = m.roughness_texture;
    main_mat.metallic = _CLAMP(m.metallic, 0.0f, 1.0f);
    main_mat.textures[METALLIC_TEXTURE] = m.metallic_texture;
    main_mat.int_ior = m.ior;
    main_mat.ext_ior = 1.0f;
    main_mat.flags = 0;
    main_mat.transmission = m.transmission;
    main_mat.transmission_roughness = m.transmission_roughness;
    main_mat.textures[NORMALS_TEXTURE] = m.normal_map;
    main_mat.anisotropic_unorm = pack_unorm_16(_CLAMP(m.anisotropic, 0.0f, 1.0f));
    main_mat.specular = _CLAMP(m.specular, 0.0f, 1.0f);
    main_mat.specular_tint = _CLAMP(m.specular_tint, 0.0f, 1.0f);
    main_mat.clearcoat = _CLAMP(m.clearcoat, 0.0f, 1.0f);
    main_mat.clearcoat_roughness = _CLAMP(m.clearcoat_roughness, 0.0f, 1.0f);
    main_mat.flags = 0;

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

    if (m.alpha != 1.0f) {
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
            mix_node.base_texture = 0xffffffff;
            mix_node.strength = 1.0f - m.alpha;
            mix_node.int_ior = mix_node.ext_ior = 0.0f;

            mix_node.mix_materials[0] = root_node;
            mix_node.mix_materials[1] = transparent_node;

            root_node = AddMaterial(mix_node);
        }
    }

    return root_node;
}

uint32_t Ray::Ref::Scene::AddMesh(const mesh_desc_t &_m) {
    meshes_.emplace_back();
    mesh_t &m = meshes_.back();

    const auto tris_start = uint32_t(tris2_.size());
    // const auto tri_index_start = uint32_t(tri_indices_.size());

    bvh_settings_t s;
    s.node_traversal_cost = 0.025f;
    s.oversplit_threshold = 0.95f;
    s.allow_spatial_splits = _m.allow_spatial_splits;
    s.use_fast_bvh_build = _m.use_fast_bvh_build;

    const uint64_t t1 = Ray::GetTimeMs();

    m.node_index = uint32_t(nodes_.size());
    m.node_count = PreprocessMesh(_m.vtx_attrs, _m.vtx_indices, _m.vtx_indices_count, _m.layout, _m.base_vertex,
                                  uint32_t(tri_materials_.size()), s, nodes_, tris_, tris2_, tri_indices_);

    log_->Info("Ray: Mesh preprocessed in %lldms", (Ray::GetTimeMs() - t1));

    if (use_wide_bvh_) {
        const uint64_t t2 = Ray::GetTimeMs();

        const auto before_count = uint32_t(mnodes_.size());
        const uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), m.node_index, 0xffffffff, mnodes_);

        m.node_index = new_root;
        m.node_count = uint32_t(mnodes_.size() - before_count);

        // nodes_ variable is treated as temporary storage
        nodes_.clear();

        log_->Info("Ray: BVH flattened in %lldms", (Ray::GetTimeMs() - t2));
    }

    const auto tri_materials_start = uint32_t(tri_materials_.size());
    tri_materials_.resize(tri_materials_start + (_m.vtx_indices_count / 3));

    // init triangle materials
    for (const shape_desc_t &s : _m.shapes) {
        bool is_front_solid = true, is_back_solid = true;

        uint32_t material_stack[32];
        material_stack[0] = s.mat_index;
        uint32_t material_count = 1;

        while (material_count) {
            material_t &mat = materials_[material_stack[--material_count]];

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
            material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == MixNode) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == TransparentNode) {
                is_back_solid = false;
                break;
            }
        }

        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            tri_mat_data_t &tri_mat = tri_materials_[tri_materials_start + (i / 3)];

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

    m.tris_index = tris_start;
    m.tris_count = uint32_t(tris2_.size() - tris_start);

    std::vector<uint32_t> new_vtx_indices;
    new_vtx_indices.reserve(_m.vtx_indices_count);
    for (size_t i = 0; i < _m.vtx_indices_count; i++) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + _m.base_vertex + uint32_t(vertices_.size()));
    }

    const size_t stride = AttrStrides[_m.layout];

    // add attributes
    const size_t new_vertices_start = vertices_.size();
    vertices_.resize(new_vertices_start + _m.vtx_attrs_count);
    for (size_t i = 0; i < _m.vtx_attrs_count; i++) {
        vertex_t &v = vertices_[new_vertices_start + i];

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
        ComputeTangentBasis(0, new_vertices_start, vertices_, new_vtx_indices, &new_vtx_indices[0],
                            new_vtx_indices.size());
    }

    m.vert_index = uint32_t(vtx_indices_.size());
    m.vert_count = uint32_t(new_vtx_indices.size());

    vtx_indices_.insert(vtx_indices_.end(), new_vtx_indices.begin(), new_vtx_indices.end());

    return uint32_t(meshes_.size() - 1);
}

void Ray::Ref::Scene::RemoveMesh(const uint32_t i) {
    const mesh_t &m = meshes_[i];

    const uint32_t node_index = m.node_index, node_count = m.node_count;
    const uint32_t tris_index = m.tris_index, tris_count = m.tris_count;

    auto last_mesh_index = uint32_t(meshes_.size() - 1);

    std::swap(meshes_[i], meshes_[last_mesh_index]);

    meshes_.pop_back();

    bool rebuild_needed = false;

    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end();) {
        mesh_instance_t &mi = *it;

        if (mi.mesh_index == last_mesh_index) {
            mi.mesh_index = i;
        }

        if (mi.mesh_index == i) {
            it = mesh_instances_.erase(it);
            rebuild_needed = true;
        } else {
            ++it;
        }
    }

    RemoveTris(tris_index, tris_count);
    RemoveNodes(node_index, node_count);

    if (rebuild_needed) {
        RebuildTLAS();
    }
}

uint32_t Ray::Ref::Scene::AddLight(const light_desc_t &_l) {
    light_t l;
    memcpy(&l.pos[0], &_l.position[0], 3 * sizeof(float));
    l.radius = _l.radius;
    memcpy(&l.col[0], &_l.color[0], 3 * sizeof(float));
    if (_l.type == SpotLight) {
        l.dir[0] = -_l.direction[0];
        l.dir[1] = -_l.direction[1];
        l.dir[2] = -_l.direction[2];
        l.spot = std::cos(_l.angle * PI / 180.0f);
    } else if (_l.type == PointLight) {
        l.dir[0] = l.dir[2] = 0.0f;
        l.dir[1] = 1.0f;
        l.spot = -1.0f;
    } else if (_l.type == DirectionalLight) {
        float dist = 9999999.0f;

        l.dir[0] = -_l.direction[0];
        l.dir[1] = -_l.direction[1];
        l.dir[2] = -_l.direction[2];

        l.pos[0] = l.dir[0] * dist;
        l.pos[1] = l.dir[1] * dist;
        l.pos[2] = l.dir[2] * dist;

        l.radius = dist * std::tan(_l.angle * PI / 180.0f) + 1.0f;

        float k = 1.0f + dist / l.radius;
        k *= k;

        l.col[0] = _l.color[0] * k;
        l.col[1] = _l.color[1] * k;
        l.col[2] = _l.color[2] * k;

        l.spot = 0.0f;
    }

    l.brightness = std::max(l.col[0], std::max(l.col[1], l.col[2]));

    lights_.push_back(l);

    RebuildLightBVH();

    return uint32_t(lights_.size() - 1);
}

void Ray::Ref::Scene::RemoveLight(const uint32_t i) {
    // TODO!!!
    unused(i);
}

uint32_t Ray::Ref::Scene::AddMeshInstance(const uint32_t mesh_index, const float *xform) {
    const auto mi_index = uint32_t(mesh_instances_.size());
    const auto tr_index = uint32_t(transforms_.size());

    mesh_instances_.emplace_back();
    mesh_instance_t &mi = mesh_instances_.back();
    mi.mesh_index = mesh_index;
    mi.tr_index = tr_index;

    transforms_.emplace_back();

    { // find emissive triangles and add them as emitters
        const mesh_t &m = meshes_[mesh_index];
        for (uint32_t tri = (m.vert_index / 3); tri < (m.vert_index + m.vert_count) / 3; ++tri) {
            const tri_mat_data_t &tri_mat = tri_materials_[tri];

            const material_t &front_mat = materials_[tri_mat.front_mi & MATERIAL_INDEX_BITS];
            if (front_mat.type == EmissiveNode &&
                (front_mat.flags & (MAT_FLAG_MULT_IMPORTANCE | MAT_FLAG_SKY_PORTAL))) {
                lights2_.emplace_back();
                light2_t &new_light = lights2_.back();
                new_light.type = LIGHT_TYPE_TRI;
                new_light.xform = tr_index;
                new_light.tri.index = tri;
                new_light.col[0] = front_mat.base_color[0] * front_mat.strength;
                new_light.col[1] = front_mat.base_color[1] * front_mat.strength;
                new_light.col[2] = front_mat.base_color[2] * front_mat.strength;
            }
        }
    }

    SetMeshInstanceTransform(mi_index, xform);

    return mi_index;
}

void Ray::Ref::Scene::SetMeshInstanceTransform(uint32_t mi_index, const float *xform) {
    mesh_instance_t &mi = mesh_instances_[mi_index];
    transform_t &tr = transforms_[mi.tr_index];

    memcpy(tr.xform, xform, 16 * sizeof(float));
    InverseMatrix(tr.xform, tr.inv_xform);

    const mesh_t &m = meshes_[mi.mesh_index];

    if (!use_wide_bvh_) {
        const bvh_node_t &n = nodes_[m.node_index];
        TransformBoundingBox(n.bbox_min, n.bbox_max, xform, mi.bbox_min, mi.bbox_max);
    } else {
        const mbvh_node_t &n = mnodes_[m.node_index];

        float bbox_min[3] = {MAX_DIST, MAX_DIST, MAX_DIST}, bbox_max[3] = {-MAX_DIST, -MAX_DIST, -MAX_DIST};

        if (n.child[0] & LEAF_NODE_BIT) {
            bbox_min[0] = n.bbox_min[0][0];
            bbox_min[1] = n.bbox_min[1][0];
            bbox_min[2] = n.bbox_min[2][0];

            bbox_max[0] = n.bbox_max[0][0];
            bbox_max[1] = n.bbox_max[1][0];
            bbox_max[2] = n.bbox_max[2][0];
        } else {
            for (int i = 0; i < 8; i++) {
                if (n.child[i] == 0x7fffffff) {
                    continue;
                }

                bbox_min[0] = std::min(bbox_min[0], n.bbox_min[0][i]);
                bbox_min[1] = std::min(bbox_min[1], n.bbox_min[1][i]);
                bbox_min[2] = std::min(bbox_min[2], n.bbox_min[2][i]);

                bbox_max[0] = std::max(bbox_max[0], n.bbox_max[0][i]);
                bbox_max[1] = std::max(bbox_max[1], n.bbox_max[1][i]);
                bbox_max[2] = std::max(bbox_max[2], n.bbox_max[2][i]);
            }
        }

        TransformBoundingBox(bbox_min, bbox_max, xform, mi.bbox_min, mi.bbox_max);
    }

    RebuildTLAS();
}

void Ray::Ref::Scene::RemoveMeshInstance(uint32_t i) {
    mesh_instances_.erase(mesh_instances_.begin() + i);

    RebuildTLAS();
}

void Ray::Ref::Scene::RemoveTris(uint32_t tris_index, uint32_t tris_count) {
    if (!tris_count) {
        return;
    }

    tris2_.erase(std::next(tris2_.begin(), tris_index), std::next(tris2_.begin(), tris_index + tris_count));

    if (tris_index != tris2_.size()) {
        for (mesh_t &m : meshes_) {
            if (m.tris_index > tris_index) {
                m.tris_index -= tris_count;
            }
        }
    }
}

void Ray::Ref::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
    if (!node_count) {
        return;
    }

    if (!use_wide_bvh_) {
        nodes_.erase(std::next(nodes_.begin(), node_index), std::next(nodes_.begin(), node_index + node_count));
    } else {
        mnodes_.erase(std::next(mnodes_.begin(), node_index), std::next(mnodes_.begin(), node_index + node_count));
    }

    if ((!use_wide_bvh_ && node_index != nodes_.size()) || (use_wide_bvh_ && node_index != mnodes_.size())) {
        for (mesh_t &m : meshes_) {
            if (m.node_index > node_index) {
                m.node_index -= node_count;
            }
        }

        for (uint32_t i = node_index; i < nodes_.size(); i++) {
            bvh_node_t &n = nodes_[i];

#ifdef USE_STACKLESS_BVH_TRAVERSAL
            if (n.parent != 0xffffffff && n.parent > node_index) {
                n.parent -= node_count;
            }
#endif
            if ((n.prim_index & LEAF_NODE_BIT) == 0) {
                if (n.left_child > node_index) {
                    n.left_child -= node_count;
                }
                if ((n.right_child & RIGHT_CHILD_BITS) > node_index) {
                    n.right_child -= node_count;
                }
            }
        }

        for (uint32_t i = node_index; i < mnodes_.size(); i++) {
            mbvh_node_t &n = mnodes_[i];

            if ((n.child[0] & LEAF_NODE_BIT) == 0) {
                if (n.child[0] > node_index) {
                    n.child[0] -= node_count;
                }
                if (n.child[1] > node_index) {
                    n.child[1] -= node_count;
                }
                if (n.child[2] > node_index) {
                    n.child[2] -= node_count;
                }
                if (n.child[3] > node_index) {
                    n.child[3] -= node_count;
                }
                if (n.child[4] > node_index) {
                    n.child[4] -= node_count;
                }
                if (n.child[5] > node_index) {
                    n.child[5] -= node_count;
                }
                if (n.child[6] > node_index) {
                    n.child[6] -= node_count;
                }
                if (n.child[7] > node_index) {
                    n.child[7] -= node_count;
                }
            }
        }

        if (macro_nodes_root_ > node_index) {
            macro_nodes_root_ -= node_count;
        }

        if (light_nodes_root_ > node_index) {
            light_nodes_root_ -= node_count;
        }
    }
}

void Ray::Ref::Scene::RebuildTLAS() {
    RemoveNodes(macro_nodes_root_, macro_nodes_count_);
    mi_indices_.clear();

    macro_nodes_root_ = 0xffffffff;
    macro_nodes_count_ = 0;

    if (mesh_instances_.empty()) {
        return;
    }

    std::vector<prim_t> primitives;
    primitives.reserve(mesh_instances_.size());

    for (const mesh_instance_t &mi : mesh_instances_) {
        primitives.push_back({0, 0, 0, Ref::simd_fvec4{mi.bbox_min}, Ref::simd_fvec4{mi.bbox_max}});
    }

    macro_nodes_root_ = uint32_t(nodes_.size());
    macro_nodes_count_ = PreprocessPrims_SAH(&primitives[0], primitives.size(), nullptr, 0, {}, nodes_, mi_indices_);

    if (use_wide_bvh_) {
        const auto before_count = uint32_t(mnodes_.size());
        const uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), macro_nodes_root_, 0xffffffff, mnodes_);

        macro_nodes_root_ = new_root;
        macro_nodes_count_ = static_cast<uint32_t>(mnodes_.size() - before_count);

        // nodes_ is temporary storage when wide BVH is used
        nodes_.clear();
    }
}

void Ray::Ref::Scene::RebuildLightBVH() {
    RemoveNodes(light_nodes_root_, light_nodes_count_);
    li_indices_.clear();

    std::vector<prim_t> primitives;
    primitives.reserve(lights_.size());

    for (const light_t &l : lights_) {
        const float influence = l.radius * (std::sqrt(l.brightness / LIGHT_ATTEN_CUTOFF) - 1.0f);

        simd_fvec3 bbox_min = {0.0f}, bbox_max = {0.0f};

        const simd_fvec3 p1 = {-l.dir[0] * influence, -l.dir[1] * influence, -l.dir[2] * influence};

        bbox_min = min(bbox_min, p1);
        bbox_max = max(bbox_max, p1);

        const simd_fvec3 p2 = {-l.dir[0] * l.spot * influence, -l.dir[1] * l.spot * influence,
                               -l.dir[2] * l.spot * influence};

        const float d = std::sqrt(1.0f - l.spot * l.spot) * influence;

        bbox_min = min(bbox_min, p2 - simd_fvec3{d, 0.0f, d});
        bbox_max = max(bbox_max, p2 + simd_fvec3{d, 0.0f, d});

        if (l.spot < 0.0f) {
            bbox_min = min(bbox_min, p1 - simd_fvec3{influence, 0.0f, influence});
            bbox_max = max(bbox_max, p1 + simd_fvec3{influence, 0.0f, influence});
        }

        simd_fvec3 up = {1.0f, 0.0f, 0.0f};
        if (std::abs(l.dir[1]) < std::abs(l.dir[2]) && std::abs(l.dir[1]) < std::abs(l.dir[0])) {
            up = {0.0f, 1.0f, 0.0f};
        } else if (std::abs(l.dir[2]) < std::abs(l.dir[0]) && std::abs(l.dir[2]) < std::abs(l.dir[1])) {
            up = {0.0f, 0.0f, 1.0f};
        }

        const simd_fvec3 side = {-l.dir[1] * up[2] + l.dir[2] * up[1], -l.dir[2] * up[0] + l.dir[0] * up[2],
                                 -l.dir[0] * up[1] + l.dir[1] * up[0]};

        const float xform[16] = {side[0], l.dir[0], up[0], 0.0f, side[1],  l.dir[1], up[1],    0.0f,
                                 side[2], l.dir[2], up[2], 0.0f, l.pos[0], l.pos[1], l.pos[2], 1.0f};

        primitives.emplace_back();
        prim_t &prim = primitives.back();

        prim.i0 = prim.i1 = prim.i2 = 0;
        TransformBoundingBox(&bbox_min[0], &bbox_max[0], xform, &prim.bbox_min[0], &prim.bbox_max[0]);
    }

    light_nodes_root_ = uint32_t(nodes_.size());
    light_nodes_count_ = PreprocessPrims_SAH(&primitives[0], primitives.size(), nullptr, 0, {}, nodes_, li_indices_);

    if (use_wide_bvh_) {
        const auto before_count = uint32_t(mnodes_.size());
        const uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), light_nodes_root_, 0xffffffff, mnodes_);

        light_nodes_root_ = new_root;
        light_nodes_count_ = uint32_t(mnodes_.size() - before_count);

        // nodes_ is temporary storage when wide BVH is used
        nodes_.clear();
    }
}

#undef _CLAMP
