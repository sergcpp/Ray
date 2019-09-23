#include "SceneRef.h"

#include <cassert>
#include <cstring>

#include "TextureUtilsRef.h"

Ray::Ref::Scene::Scene(bool use_wide_bvh) : use_wide_bvh_(use_wide_bvh), texture_atlas_(TEXTURE_ATLAS_SIZE, TEXTURE_ATLAS_SIZE) {
    {   // add default environment map (white)
        static const pixel_color8_t default_env_map = { 255, 255, 255, 128 };

        tex_desc_t t;
        t.data = &default_env_map;
        t.w = 1;
        t.h = 1;
        t.generate_mipmaps = false;

        default_env_texture_ = AddTexture(t);

        if (default_env_texture_ == 0xffffffff) {
            throw std::runtime_error("Cannot allocate 1px default env map!");
        }

        Ray::environment_desc_t desc;
        desc.env_col[0] = desc.env_col[1] = desc.env_col[2] = 0.0f;
        desc.env_map = default_env_texture_;
        SetEnvironment(desc);
    }

    {   // add default normal map (flat)
        static const pixel_color8_t default_normalmap = { 127, 127, 255 };

        tex_desc_t t;
        t.data = &default_normalmap;
        t.w = 1;
        t.h = 1;
        t.generate_mipmaps = false;

        default_normals_texture_ = AddTexture(t);

        if (default_normals_texture_ == 0xffffffff) {
            throw std::runtime_error("Cannot allocate 1px default normal map!");
        }
    }
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
    if (env_.env_map == 0xffffffff) {
        env_.env_map = default_env_texture_;
    }
}

uint32_t Ray::Ref::Scene::AddTexture(const tex_desc_t &_t) {
    uint32_t tex_index = (uint32_t)textures_.size();

    texture_t t;
    t.width = (uint16_t)_t.w;
    t.height = (uint16_t)_t.h;

    if (_t.is_srgb) {
        t.width |= TEXTURE_SRGB_BIT;
    }

    int mip = 0;
    int res[2] = { _t.w, _t.h };

    std::vector<pixel_color8_t> tex_data(_t.data, _t.data + _t.w * _t.h);

    while (res[0] >= 1 && res[1] >= 1) {
        int pos[2];
        int page = texture_atlas_.Allocate(&tex_data[0], res, pos);
        if (page == -1) {
            // release allocated mip levels on fail
            for (int i = mip; i >= 0; i--) {
                int _pos[2] = { t.pos[i][0], t.pos[i][1] };
                texture_atlas_.Free(t.page[i], _pos);
            }
            return 0xffffffff;
        }

        t.page[mip] = (uint8_t)page;
        t.pos[mip][0] = (uint16_t)pos[0];
        t.pos[mip][1] = (uint16_t)pos[1];

        mip++;

        if (_t.generate_mipmaps) {
            tex_data = Ref::DownsampleTexture(tex_data, res);

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

    textures_.push_back(t);

    return tex_index;
}

uint32_t Ray::Ref::Scene::AddMaterial(const mat_desc_t &m) {
    material_t mat;

    mat.type = m.type;
    mat.textures[MAIN_TEXTURE] = m.main_texture;
    memcpy(&mat.main_color[0], &m.main_color[0], 3 * sizeof(float));
    mat.int_ior = m.int_ior;
    mat.ext_ior = m.ext_ior;

    if (m.type == DiffuseMaterial) {
        mat.roughness = m.roughness;
    } else if (m.type == GlossyMaterial) {
        mat.roughness = m.roughness;
    } else if (m.type == RefractiveMaterial) {
        mat.roughness = m.roughness;
    } else if (m.type == EmissiveMaterial) {
        mat.strength = m.strength;
    } else if (m.type == MixMaterial) {
        mat.strength = m.strength;
        mat.textures[MIX_MAT1] = m.mix_materials[0];
        mat.textures[MIX_MAT2] = m.mix_materials[1];
    } else if (m.type == TransparentMaterial) {

    }

    if (m.normal_map != 0xffffffff) {
        mat.textures[NORMALS_TEXTURE] = m.normal_map;
    } else {
        mat.textures[NORMALS_TEXTURE] = default_normals_texture_;
    }

    uint32_t mat_index = (uint32_t)materials_.size();

    materials_.push_back(mat);

    return mat_index;
}

uint32_t Ray::Ref::Scene::AddMesh(const mesh_desc_t &_m) {
    meshes_.emplace_back();
    mesh_t &m = meshes_.back();
    
    uint32_t tris_start = (uint32_t)tris_.size(),
             tri_index_start = (uint32_t)tri_indices_.size();

    bvh_settings_t s;
    s.node_traversal_cost = 0.025f;
    s.oversplit_threshold = 0.95f;
    s.allow_spatial_splits = _m.allow_spatial_splits;
    s.use_fast_bvh_build = _m.use_fast_bvh_build;

    m.node_index = (uint32_t)nodes_.size();
    m.node_count = PreprocessMesh(_m.vtx_attrs, _m.vtx_indices, _m.vtx_indices_count, _m.layout, _m.base_vertex, s, nodes_, tris_, tri_indices_);

    if (use_wide_bvh_) {
        uint32_t before_count = (uint32_t)mnodes_.size();
        uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), m.node_index, 0xffffffff, mnodes_);

        m.node_index = new_root;
        m.node_count = (uint32_t)(mnodes_.size() - before_count);

        // nodes_ is treated as temporary storage
        nodes_.clear();
    }

    // init triangle materials
    for (const shape_desc_t &s : _m.shapes) {
        bool is_solid = true;

        uint32_t material_stack[32];
        material_stack[0] = s.mat_index;
        uint32_t material_count = 1;

        while (material_count) {
            material_t &mat = materials_[material_stack[--material_count]];

            if (mat.type == MixMaterial) {
                material_stack[material_count++] = mat.textures[MIX_MAT1];
                material_stack[material_count++] = mat.textures[MIX_MAT2];
            } else if (mat.type == TransparentMaterial) {
                is_solid = false;
                break;
            }
        }

        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            tri_accel_t &tri = tris_[tris_start + i / 3];

            if (is_solid) {
                tri.ci = (tri.ci | uint32_t(TRI_SOLID_BIT));
            } else {
                tri.ci = (tri.ci & ~uint32_t(TRI_SOLID_BIT));
            }

            tri.mi = s.mat_index;
            tri.back_mi = s.back_mat_index;
        }
    }

    m.tris_index = tris_start;
    m.tris_count = (uint32_t)tris_.size() - tris_start;

    std::vector<uint32_t> new_vtx_indices;
    new_vtx_indices.reserve(_m.vtx_indices_count);
    for (size_t i = 0; i < _m.vtx_indices_count; i++) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + _m.base_vertex + (uint32_t)vertices_.size());
    }

    size_t stride = AttrStrides[_m.layout];

    // add attributes
    size_t new_vertices_start = vertices_.size();
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
        ComputeTangentBasis(0, new_vertices_start, vertices_, new_vtx_indices, &new_vtx_indices[0], new_vtx_indices.size());
    }

    vtx_indices_.insert(vtx_indices_.end(), new_vtx_indices.begin(), new_vtx_indices.end());

    return (uint32_t)(meshes_.size() - 1);
}

void Ray::Ref::Scene::RemoveMesh(uint32_t i) {
    const mesh_t &m = meshes_[i];

    uint32_t node_index = m.node_index,
             node_count = m.node_count;

    uint32_t tris_index = m.tris_index,
             tris_count = m.tris_count;

    auto last_mesh_index = static_cast<uint32_t>(meshes_.size() - 1);

    std::swap(meshes_[i], meshes_[last_mesh_index]);

    meshes_.pop_back();

    bool rebuild_needed = false;

    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end(); ) {
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
        RebuildMacroBVH();
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

    return (uint32_t)(lights_.size() - 1);
}

void Ray::Ref::Scene::RemoveLight(uint32_t i) {
    // TODO!!!
    unused(i);
}

uint32_t Ray::Ref::Scene::AddMeshInstance(uint32_t mesh_index, const float *xform) {
    auto mi_index = static_cast<uint32_t>(mesh_instances_.size());

    mesh_instances_.emplace_back();
    mesh_instance_t &mi = mesh_instances_.back();
    mi.mesh_index = mesh_index;
    mi.tr_index = (uint32_t)transforms_.size();
    transforms_.emplace_back();

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

        float bbox_min[3] = { MAX_DIST, MAX_DIST, MAX_DIST },
              bbox_max[3] = { -MAX_DIST, -MAX_DIST, -MAX_DIST };

        if (n.child[0] & LEAF_NODE_BIT) {
            bbox_min[0] = n.bbox_min[0][0];
            bbox_min[1] = n.bbox_min[1][0];
            bbox_min[2] = n.bbox_min[2][0];

            bbox_max[0] = n.bbox_max[0][0];
            bbox_max[1] = n.bbox_max[1][0];
            bbox_max[2] = n.bbox_max[2][0];
        } else {
            for (int i = 0; i < 8; i++) {
                if (n.child[i] == 0x7fffffff) continue;

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

    RebuildMacroBVH();
}

void Ray::Ref::Scene::RemoveMeshInstance(uint32_t i) {
    mesh_instances_.erase(mesh_instances_.begin() + i);

    RebuildMacroBVH();
}

void Ray::Ref::Scene::RemoveTris(uint32_t tris_index, uint32_t tris_count) {
    if (!tris_count) return;

    tris_.erase(std::next(tris_.begin(), tris_index),
                std::next(tris_.begin(), tris_index + tris_count));

    if (tris_index != tris_.size()) {
        for (mesh_t &m : meshes_) {
            if (m.tris_index > tris_index) {
                m.tris_index -= tris_count;
            }
        }
    }
}

void Ray::Ref::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
    if (!node_count) return;

    if (!use_wide_bvh_) {
        nodes_.erase(std::next(nodes_.begin(), node_index),
                     std::next(nodes_.begin(), node_index + node_count));
    } else {
        mnodes_.erase(std::next(mnodes_.begin(), node_index),
                         std::next(mnodes_.begin(), node_index + node_count));
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
            if (n.parent != 0xffffffff && n.parent > node_index) n.parent -= node_count;
#endif
            if ((n.prim_index & LEAF_NODE_BIT) == 0) {
                if (n.left_child > node_index) n.left_child -= node_count;
                if ((n.right_child & RIGHT_CHILD_BITS) > node_index) n.right_child -= node_count;
            }
        }

        for (uint32_t i = node_index; i < mnodes_.size(); i++) {
            mbvh_node_t &n = mnodes_[i];

            if ((n.child[0] & LEAF_NODE_BIT) == 0) {
                if (n.child[0] > node_index) n.child[0] -= node_count;
                if (n.child[1] > node_index) n.child[1] -= node_count;
                if (n.child[2] > node_index) n.child[2] -= node_count;
                if (n.child[3] > node_index) n.child[3] -= node_count;
                if (n.child[4] > node_index) n.child[4] -= node_count;
                if (n.child[5] > node_index) n.child[5] -= node_count;
                if (n.child[6] > node_index) n.child[6] -= node_count;
                if (n.child[7] > node_index) n.child[7] -= node_count;
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

void Ray::Ref::Scene::RebuildMacroBVH() {
    RemoveNodes(macro_nodes_root_, macro_nodes_count_);
    mi_indices_.clear();

    std::vector<prim_t> primitives;
    primitives.reserve(mesh_instances_.size());

    for (const mesh_instance_t &mi : mesh_instances_) {
        primitives.push_back({ 0, 0, 0, Ref::simd_fvec3{ mi.bbox_min }, Ref::simd_fvec3{ mi.bbox_max } });
    }

    macro_nodes_root_ = static_cast<uint32_t>(nodes_.size());
    macro_nodes_count_ = PreprocessPrims_SAH(&primitives[0], primitives.size(), nullptr, 0, {}, nodes_, mi_indices_);

    if (use_wide_bvh_) {
        uint32_t before_count = static_cast<uint32_t>(mnodes_.size());
        uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), macro_nodes_root_, 0xffffffff, mnodes_);

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
        float influence = l.radius * (std::sqrt(l.brightness / LIGHT_ATTEN_CUTOFF) - 1.0f);

        simd_fvec3 bbox_min = { 0.0f }, bbox_max = { 0.0f };

        simd_fvec3 p1 = { -l.dir[0] * influence,
                          -l.dir[1] * influence,
                          -l.dir[2] * influence };

        bbox_min = min(bbox_min, p1);
        bbox_max = max(bbox_max, p1);

        simd_fvec3 p2 = { -l.dir[0] * l.spot * influence,
                          -l.dir[1] * l.spot * influence,
                          -l.dir[2] * l.spot * influence };

        float d = std::sqrt(1.0f - l.spot * l.spot) * influence;

        bbox_min = min(bbox_min, p2 - simd_fvec3{ d, 0.0f, d });
        bbox_max = max(bbox_max, p2 + simd_fvec3{ d, 0.0f, d });

        if (l.spot < 0.0f) {
            bbox_min = min(bbox_min, p1 - simd_fvec3{ influence, 0.0f, influence });
            bbox_max = max(bbox_max, p1 + simd_fvec3{ influence, 0.0f, influence });
        }

        simd_fvec3 up = { 1.0f, 0.0f, 0.0f };
        if (std::abs(l.dir[1]) < std::abs(l.dir[2]) && std::abs(l.dir[1]) < std::abs(l.dir[0])) {
            up = { 0.0f, 1.0f, 0.0f };
        } else if (std::abs(l.dir[2]) < std::abs(l.dir[0]) && std::abs(l.dir[2]) < std::abs(l.dir[1])) {
            up = { 0.0f, 0.0f, 1.0f };
        }

        simd_fvec3 side = { -l.dir[1] * up[2] + l.dir[2] * up[1],
                            -l.dir[2] * up[0] + l.dir[0] * up[2],
                            -l.dir[0] * up[1] + l.dir[1] * up[0] };

        float xform[16] = { side[0],  l.dir[0], up[0],    0.0f,
                            side[1],  l.dir[1], up[1],    0.0f,
                            side[2],  l.dir[2], up[2],    0.0f,
                            l.pos[0], l.pos[1], l.pos[2], 1.0f };

        primitives.emplace_back();
        prim_t &prim = primitives.back();

        prim.i0 = prim.i1 = prim.i2 = 0;
        TransformBoundingBox(&bbox_min[0], &bbox_max[0], xform, &prim.bbox_min[0], &prim.bbox_max[0]);
    }

    light_nodes_root_ = static_cast<uint32_t>(nodes_.size());
    light_nodes_count_ = PreprocessPrims_SAH(&primitives[0], primitives.size(), nullptr, 0, {}, nodes_, li_indices_);

    if (use_wide_bvh_) {
        uint32_t before_count = static_cast<uint32_t>(mnodes_.size());
        uint32_t new_root = FlattenBVH_Recursive(nodes_.data(), light_nodes_root_, 0xffffffff, mnodes_);

        light_nodes_root_ = new_root;
        light_nodes_count_ = static_cast<uint32_t>(mnodes_.size() - before_count);

        // nodes_ is temporary storage when wide BVH is used
        nodes_.clear();
    }
}