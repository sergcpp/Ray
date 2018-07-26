#include "SceneOCL.h"

#include <cassert>

#include "BVHSplit.h"
#include "TextureUtilsRef.h"

ray::ocl::Scene::Scene(const cl::Context &context, const cl::CommandQueue &queue)
    : context_(context), queue_(queue),
      nodes_(context, queue, CL_MEM_READ_ONLY),
      tris_(context, queue, CL_MEM_READ_ONLY),
      tri_indices_(context, queue, CL_MEM_READ_ONLY),
      transforms_(context, queue, CL_MEM_READ_ONLY),
      meshes_(context, queue, CL_MEM_READ_ONLY),
      mesh_instances_(context, queue, CL_MEM_READ_ONLY),
      mi_indices_(context, queue, CL_MEM_READ_ONLY),
      vertices_(context, queue, CL_MEM_READ_ONLY),
      vtx_indices_(context, queue, CL_MEM_READ_ONLY),
      materials_(context, queue, CL_MEM_READ_ONLY),
      textures_(context, queue, CL_MEM_READ_ONLY),
    texture_atlas_(context_, queue_, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE) {
    SetEnvironment( { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });

    pixel_color8_t default_normalmap = { 127, 127, 255 };

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

void ray::ocl::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(&env.sun_dir[0], &env_.sun_dir, 3 * sizeof(float));
    memcpy(&env.sun_col[0], &env_.sun_col, 3 * sizeof(float));
    memcpy(&env.sky_col[0], &env_.sky_col, 3 * sizeof(float));
    env.sun_softness = env_.sun_softness;
}

void ray::ocl::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(&env_.sun_dir, &env.sun_dir[0], 3 * sizeof(float));
    memcpy(&env_.sun_col, &env.sun_col[0], 3 * sizeof(float));
    memcpy(&env_.sky_col, &env.sky_col[0], 3 * sizeof(float));
    env_.sun_softness = env.sun_softness;
}

uint32_t ray::ocl::Scene::AddTexture(const tex_desc_t &_t) {
    uint32_t tex_index = (uint32_t)textures_.size();

    texture_t t;
    t.size[0] = (uint16_t)_t.w;
    t.size[1] = (uint16_t)_t.h;

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

        t.page[mip] = (uint32_t)page;
        t.pos[mip][0] = (uint16_t)pos[0];
        t.pos[mip][1] = (uint16_t)pos[1];

        mip++;

        if (_t.generate_mipmaps) {
            tex_data = ref::DownsampleTexture(tex_data, res);

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

    textures_.PushBack(t);

    return tex_index;
}

uint32_t ray::ocl::Scene::AddMaterial(const mat_desc_t &m) {
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

    materials_.PushBack(mat);

    return mat_index;
}

uint32_t ray::ocl::Scene::AddMesh(const mesh_desc_t &_m) {
    std::vector<bvh_node_t> new_nodes;
    std::vector<tri_accel_t> new_tris;
    std::vector<uint32_t> new_tri_indices;
    std::vector<uint32_t> new_vtx_indices;

    PreprocessMesh(_m.vtx_attrs, _m.vtx_attrs_count, _m.vtx_indices, _m.vtx_indices_count, _m.layout, new_nodes, new_tris, new_tri_indices);
    for (size_t i = 0; i < _m.vtx_indices_count; i++) {
        new_vtx_indices.push_back(_m.vtx_indices[i] + (uint32_t)vertices_.size());
    }

    // set material index for triangles
    for (const auto &s : _m.shapes) {
        for (size_t i = s.vtx_start; i < s.vtx_start + s.vtx_count; i += 3) {
            new_tris[i / 3].mi = s.material_index;
        }
    }

    // offset nodes and primitives
    for (auto &n : new_nodes) {
        if (n.parent != 0xffffffff) n.parent += (uint32_t)nodes_.size();
        if (n.sibling) n.sibling += (uint32_t)nodes_.size();
        if (n.prim_count) {
            n.prim_index += (uint32_t)tri_indices_.size();
        } else {
            n.left_child += (uint32_t)nodes_.size();
            n.right_child += (uint32_t)nodes_.size();
        }
    }

    // offset triangle indices
    for (auto &i : new_tri_indices) {
        i += (uint32_t)tris_.size();
    }

    // add mesh
    mesh_t m;
    m.node_index = (uint32_t)nodes_.size();
    m.node_count = (uint32_t)new_nodes.size();

    uint32_t mesh_index = (uint32_t)meshes_.size();
    meshes_.PushBack(m);

    // add nodes
    nodes_.Append(&new_nodes[0], new_nodes.size());

    // add attributes
    assert(_m.layout == PxyzNxyzTuv);
    std::vector<vertex_t> new_vertices(_m.vtx_attrs_count);
    for (size_t i = 0; i < _m.vtx_attrs_count; i++) {
        auto &v = new_vertices[i];

        memcpy(&v.p[0], (_m.vtx_attrs + i * 8), 3 * sizeof(float));
        memcpy(&v.n[0], (_m.vtx_attrs + i * 8 + 3), 3 * sizeof(float));
        memcpy(&v.t0[0], (_m.vtx_attrs + i * 8 + 6), 2 * sizeof(float));

        memset(&v.b[0], 0, 3 * sizeof(float));
    }

    ref::ComputeTextureBasis(vertices_.size(), 0, new_vertices, new_vtx_indices, _m.vtx_indices, _m.vtx_indices_count);

    vertices_.Append(&new_vertices[0], new_vertices.size());

    // add vertex indices
    vtx_indices_.Append(&new_vtx_indices[0], new_vtx_indices.size());

    // add triangles
    tris_.Append(&new_tris[0], new_tris.size());

    // add triangle indices
    tri_indices_.Append(&new_tri_indices[0], new_tri_indices.size());

    return mesh_index;
}

void ray::ocl::Scene::RemoveMesh(uint32_t) {
    // TODO!!!
}

uint32_t ray::ocl::Scene::AddMeshInstance(uint32_t mesh_index, const float *xform) {
    uint32_t mi_index = (uint32_t)mesh_instances_.size();

    mesh_instance_t mi;
    mi.mesh_index = mesh_index;
    mi.tr_index = (uint32_t)transforms_.size();
    mesh_instances_.PushBack(mi);

    transform_t tr;

    memcpy(tr.xform, xform, 16 * sizeof(float));
    InverseMatrix(tr.xform, tr.inv_xform);

    transforms_.PushBack(tr);

    SetMeshInstanceTransform(mi_index, xform);

    return mi_index;
}

void ray::ocl::Scene::SetMeshInstanceTransform(uint32_t mi_index, const float *xform) {
    transform_t tr;

    memcpy(tr.xform, xform, 16 * sizeof(float));
    InverseMatrix(tr.xform, tr.inv_xform);

    mesh_instance_t mi;
    mesh_instances_.Get(mi_index, mi);

    mesh_t m;
    meshes_.Get(mi.mesh_index, m);

    bvh_node_t n;
    nodes_.Get(m.node_index, n);

    float transformed_bbox[2][3];
    TransformBoundingBox(n.bbox, xform, transformed_bbox);

    memcpy(mi.bbox_min, transformed_bbox[0], sizeof(float) * 3);
    memcpy(mi.bbox_max, transformed_bbox[1], sizeof(float) * 3);

    mesh_instances_.Set(mi_index, mi);
    transforms_.Set(mi.tr_index, tr);

    RebuildMacroBVH();
}

void ray::ocl::Scene::RemoveMeshInstance(uint32_t) {
    // TODO!!
}

void ray::ocl::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
    if (!node_count) return;

    nodes_.Erase(node_index, node_count);

    if (node_index != nodes_.size()) {
        size_t meshes_count = meshes_.size();
        std::vector<mesh_t> meshes(meshes_count);
        meshes_.Get(&meshes[0], 0, meshes_.size());

        for (auto &m : meshes) {
            if (m.node_index > node_index) {
                m.node_index -= node_count;
            }
        }
        meshes_.Set(&meshes[0], 0, meshes_count);

        size_t nodes_count = nodes_.size();
        std::vector<bvh_node_t> nodes(nodes_count);
        nodes_.Get(&nodes[0], 0, nodes_count);

        for (uint32_t i = node_index; i < nodes.size(); i++) {
            auto &n = nodes[i];

            if (n.parent != 0xffffffff && n.parent > node_index) n.parent -= node_count;
            if (n.sibling && n.sibling > node_index) n.sibling -= node_count;
            if (!n.prim_count) {
                if (n.left_child > node_index) n.left_child -= node_count;
                if (n.right_child > node_index) n.right_child -= node_count;
            }
        }
        nodes_.Set(&nodes[0], 0, nodes_count);

        if (macro_nodes_start_ > node_index) {
            macro_nodes_start_ -= node_count;
        }
    }
}

void ray::ocl::Scene::RebuildMacroBVH() {
    RemoveNodes(macro_nodes_start_, macro_nodes_count_);
    mi_indices_.Clear();

    size_t mi_count = mesh_instances_.size();

    std::vector<prim_t> primitives;
    primitives.reserve(mi_count);

    std::vector<mesh_instance_t> mesh_instances(mi_count);
    mesh_instances_.Get(&mesh_instances[0], 0, mi_count);

    for (const auto &mi : mesh_instances) {
        primitives.push_back({ ref::simd_fvec3{ mi.bbox_min }, ref::simd_fvec3{ mi.bbox_max } });
    }

    std::vector<bvh_node_t> bvh_nodes;
    std::vector<uint32_t> mi_indices;

    macro_nodes_start_ = (uint32_t)nodes_.size();
    macro_nodes_count_ = PreprocessPrims(&primitives[0], primitives.size(), bvh_nodes, mi_indices);

    // offset nodes
    for (auto &n : bvh_nodes) {
        if (n.parent != 0xffffffff) n.parent += (uint32_t)nodes_.size();
        if (n.sibling) n.sibling += (uint32_t)nodes_.size();
        if (!n.prim_count) {
            n.left_child += (uint32_t)nodes_.size();
            n.right_child += (uint32_t)nodes_.size();
        }
    }

    nodes_.Append(&bvh_nodes[0], bvh_nodes.size());
    mi_indices_.Append(&mi_indices[0], mi_indices.size());
}
