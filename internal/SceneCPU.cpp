#include "SceneCPU.h"

#include <cstring>

ray::ref::Scene::Scene() {}

uint32_t ray::ref::Scene::AddMesh(const mesh_desc_t &desc) {
    meshes_.emplace_back();
    auto &m = meshes_.back();
    m.node_index = (uint32_t)nodes_.size();
    m.node_count = 0;

    size_t tris_start = tris_.size();
    m.node_count += PreprocessMesh(desc.vtx_attrs, desc.vtx_attrs_count, desc.vtx_indices, desc.vtx_indices_count, desc.layout, nodes_, tris_, tri_indices_);

    for (const auto &s : desc.shapes) {
        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            tris_[tris_start + i / 3].mi = s.material_index;
        }
    }

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
