#include "SceneRef.h"

#include "TextureUtilsRef.h"

#include <cstring>

ray::ref::Scene::Scene() : texture_atlas_({ MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE }) {
    pixel_color8_t default_normalmap = { 127, 127, 255 };

    tex_desc_t t;
    t.data = &default_normalmap;
    t.w = 1;
    t.h = 1;

    default_normals_texture_ = AddTexture(t);

    if (default_normals_texture_ == 0xffffffff) {
        throw std::runtime_error("Cannot allocate 1px default normal map!");
    }
}

void ray::ref::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(&env.sun_dir[0], &env_.sun_dir, 3 * sizeof(float));
    memcpy(&env.sun_col[0], &env_.sun_col, 3 * sizeof(float));
    memcpy(&env.sky_col[0], &env_.sky_col, 3 * sizeof(float));
    env.sun_softness = env_.sun_softness;
}

void ray::ref::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(&env_.sun_dir, &env.sun_dir[0], 3 * sizeof(float));
    memcpy(&env_.sun_col, &env.sun_col[0], 3 * sizeof(float));
    memcpy(&env_.sky_col, &env.sky_col[0], 3 * sizeof(float));
    env_.sun_softness = env.sun_softness;
}

uint32_t ray::ref::Scene::AddTexture(const tex_desc_t &_t) {
    uint32_t tex_index = (uint32_t)textures_.size();

    texture_t t;
    t.size[0] = (uint16_t)_t.w;
    t.size[1] = (uint16_t)_t.h;

    int mip = 0;
    math::ivec2 res = { _t.w, _t.h };

    std::vector<pixel_color8_t> tex_data(_t.data, _t.data + _t.w * _t.h);

    while (res.x >= 1 && res.y >= 1) {
        math::ivec2 pos;
        int page = texture_atlas_.Allocate(&tex_data[0], res, pos);
        if (page == -1) {
            // release allocated mip levels on fail
            for (int i = mip; i >= 0; i--) {
                texture_atlas_.Free(t.page[i], { t.pos[i][0], t.pos[i][1] });
            }
            return 0xffffffff;
        }

        t.page[mip] = (uint32_t)page;
        t.pos[mip][0] = (uint16_t)pos.x;
        t.pos[mip][1] = (uint16_t)pos.y;

        tex_data = ref::DownsampleTexture(tex_data, res);

        res.x /= 2;
        res.y /= 2;
        mip++;
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

uint32_t ray::ref::Scene::AddMaterial(const mat_desc_t &m) {
    material_t mat;

    mat.type = m.type;
    mat.textures[MAIN_TEXTURE] = m.main_texture;
    memcpy(&mat.main_color[0], &m.main_color[0], 3 * sizeof(float));
    mat.fresnel = m.fresnel;

    if (m.type == DiffuseMaterial) {

    } else if (m.type == GlossyMaterial) {
        mat.roughness = m.roughness;
    } else if (m.type == RefractiveMaterial) {
        mat.roughness = m.roughness;
        mat.ior = m.ior;
    } else if (m.type == EmissiveMaterial) {
        mat.strength = m.strength;
    } else if (m.type == MixMaterial) {
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

uint32_t ray::ref::Scene::AddMesh(const mesh_desc_t &_m) {
    meshes_.emplace_back();
    auto &m = meshes_.back();
    m.node_index = (uint32_t)nodes_.size();
    m.node_count = 0;

    size_t tris_start = tris_.size();
    m.node_count += PreprocessMesh(_m.vtx_attrs, _m.vtx_attrs_count, _m.vtx_indices, _m.vtx_indices_count, _m.layout, nodes_, tris_, tri_indices_);

    for (const auto &s : _m.shapes) {
        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            tris_[tris_start + i / 3].mi = s.material_index;
        }
    }

    std::vector<uint32_t> new_vtx_indices;
    new_vtx_indices.reserve(_m.vtx_indices_count);
    for (size_t i = 0; i < _m.vtx_indices_count; i++) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + (uint32_t)vertices_.size());
    }

    // add attributes
    assert(_m.layout == PxyzNxyzTuv);
    size_t new_vertices_start = vertices_.size();
    vertices_.resize(new_vertices_start + _m.vtx_attrs_count);
    for (size_t i = 0; i < _m.vtx_attrs_count; i++) {
        auto &v = vertices_[new_vertices_start + i];

        memcpy(&v.p[0], (_m.vtx_attrs + i * 8), 3 * sizeof(float));
        memcpy(&v.n[0], (_m.vtx_attrs + i * 8 + 3), 3 * sizeof(float));
        memcpy(&v.t0[0], (_m.vtx_attrs + i * 8 + 6), 2 * sizeof(float));

        memset(&v.b[0], 0, 3 * sizeof(float));
    }

    ComputeTextureBasis(0, vertices_, new_vtx_indices, _m.vtx_indices, _m.vtx_indices_count);

    vtx_indices_.insert(vtx_indices_.end(), new_vtx_indices.begin(), new_vtx_indices.end());

    return (uint32_t)(meshes_.size() - 1);
}

void ray::ref::Scene::RemoveMesh(uint32_t i) {
    const auto &m = meshes_[i];

    uint32_t node_index = m.node_index,
             node_count = m.node_count;

    uint32_t last_mesh_index = (uint32_t)(meshes_.size() - 1);

    std::swap(meshes_[i], meshes_[last_mesh_index]);

    meshes_.pop_back();

    bool rebuild_needed = false;

    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end(); ) {
        auto &mi = *it;

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

    RemoveNodes(node_index, node_count);

    if (rebuild_needed) {
        RebuildMacroBVH();
    }
}

uint32_t ray::ref::Scene::AddMeshInstance(uint32_t mesh_index, const float *xform) {
    uint32_t mi_index = (uint32_t)mesh_instances_.size();

    mesh_instances_.emplace_back();
    auto &mi = mesh_instances_.back();
    mi.mesh_index = mesh_index;
    mi.tr_index = (uint32_t)transforms_.size();
    transforms_.emplace_back();

    SetMeshInstanceTransform(mi_index, xform);

    return mi_index;
}

void ray::ref::Scene::SetMeshInstanceTransform(uint32_t mi_index, const float *xform) {
    using namespace math;

    auto &mi = mesh_instances_[mi_index];
    auto &tr = transforms_[mi.tr_index];

    math::mat4 inv_mat = math::inverse(math::make_mat4(xform));
    memcpy(tr.xform, xform, 16 * sizeof(float));
    memcpy(tr.inv_xform, math::value_ptr(inv_mat), 16 * sizeof(float));

    const auto &m = meshes_[mi.mesh_index];
    const auto &n = nodes_[m.node_index];

    float transformed_bbox[2][3];
    TransformBoundingBox(n.bbox, xform, transformed_bbox);

    memcpy(mi.bbox_min, transformed_bbox[0], sizeof(float) * 3);
    memcpy(mi.bbox_max, transformed_bbox[1], sizeof(float) * 3);

    RebuildMacroBVH();
}

void ray::ref::Scene::RemoveMeshInstance(uint32_t i) {
    mesh_instances_.erase(mesh_instances_.begin() + i);

    RebuildMacroBVH();
}

void ray::ref::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
    if (!node_count) return;

    nodes_.erase(std::next(nodes_.begin(), node_index),
                 std::next(nodes_.begin(), node_index + node_count));

    if (node_index != nodes_.size()) {
        for (auto &m : meshes_) {
            if (m.node_index > node_index) {
                m.node_index -= node_count;
            }
        }

        for (uint32_t i = node_index; i < nodes_.size(); i++) {
            auto &n = nodes_[i];

            if (n.parent != 0xffffffff && n.parent > node_index) n.parent -= node_count;
            if (n.sibling && n.sibling > node_index) n.sibling -= node_count;
            if (!n.prim_count) {
                if (n.left_child > node_index) n.left_child -= node_count;
                if (n.right_child > node_index) n.right_child -= node_count;
            }
        }

        if (macro_nodes_start_ > node_index) {
            macro_nodes_start_ -= node_count;
        }
    }
}

void ray::ref::Scene::RebuildMacroBVH() {
    using namespace math;

    RemoveNodes(macro_nodes_start_, macro_nodes_count_);
    mi_indices_.clear();

    std::vector<prim_t> primitives;
    primitives.reserve(mesh_instances_.size());

    for (const auto &mi : mesh_instances_) {
        primitives.push_back({ make_vec3(mi.bbox_min), make_vec3(mi.bbox_max) });
    }

    macro_nodes_start_ = (uint32_t)nodes_.size();
    macro_nodes_count_ = PreprocessPrims(&primitives[0], primitives.size(), nodes_, mi_indices_);
}
