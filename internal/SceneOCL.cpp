#include "SceneOCL.h"

#include <cassert>

#include "BVHSplit.h"
#include "TextureUtilsRef.h"

Ray::Ocl::Scene::Scene(const cl::Context &context, const cl::CommandQueue &queue, size_t max_img_buf_size)
    : context_(context), queue_(queue), max_img_buf_size_(max_img_buf_size),
    nodes_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    tris_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    tri_indices_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    transforms_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    meshes_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    mesh_instances_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    mi_indices_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    vertices_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    vtx_indices_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    materials_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    textures_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    texture_atlas_(context_, queue_, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE),
    lights_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_),
    li_indices_(context, queue, CL_MEM_READ_ONLY, 16, max_img_buf_size_) {
    {   // add default environment map (white)
        pixel_color8_t default_env_map = { 255, 255, 255, 128 };

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
}

void Ray::Ocl::Scene::GetEnvironment(environment_desc_t &env) {
    memcpy(&env.env_col[0], &env_.env_col_and_clamp, 3 * sizeof(float));
    env.env_clamp = env_.env_col_and_clamp.w;
    env.env_map = env_.env_map;
}

void Ray::Ocl::Scene::SetEnvironment(const environment_desc_t &env) {
    memcpy(&env_.env_col_and_clamp, &env.env_col[0], 3 * sizeof(float));
    env_.env_col_and_clamp.w = env.env_clamp;
    env_.env_map = env.env_map;
    if (env_.env_map == 0xffffffff) {
        env_.env_map = default_env_texture_;
    }
}

uint32_t Ray::Ocl::Scene::AddTexture(const tex_desc_t &_t) {
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

    textures_.PushBack(t);

    return tex_index;
}

uint32_t Ray::Ocl::Scene::AddMaterial(const mat_desc_t &m) {
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

uint32_t Ray::Ocl::Scene::AddMesh(const mesh_desc_t &_m) {
    std::vector<bvh_node_t> new_nodes;
    std::vector<tri_accel_t> new_tris;
    std::vector<uint32_t> new_tri_indices;
    std::vector<uint32_t> new_vtx_indices;

    PreprocessMesh(_m.vtx_attrs, _m.vtx_attrs_count, _m.vtx_indices, _m.vtx_indices_count, _m.layout, _m.allow_spatial_splits, new_nodes, new_tris, new_tri_indices);
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

    int stride = _m.layout == PxyzNxyzTuvTuv ? 10 : 8;

    // add attributes
    std::vector<vertex_t> new_vertices(_m.vtx_attrs_count);
    for (size_t i = 0; i < _m.vtx_attrs_count; i++) {
        auto &v = new_vertices[i];

        memcpy(&v.p[0], (_m.vtx_attrs + i * 8), 3 * sizeof(float));
        memcpy(&v.n[0], (_m.vtx_attrs + i * 8 + 3), 3 * sizeof(float));
        memcpy(&v.t0[0], (_m.vtx_attrs + i * 8 + 6), 2 * sizeof(float));
        if (_m.layout == PxyzNxyzTuv) {
            v.t1[0] = v.t1[2] = 0.0f;
        } else if (_m.layout == PxyzNxyzTuvTuv) {
            memcpy(&v.t1[0], (_m.vtx_attrs + i * stride + 8), 2 * sizeof(float));
        }

        v.b[0] = v.b[1] = v.b[2] = 0.0f;
    }

    Ref::ComputeTextureBasis(vertices_.size(), 0, new_vertices, new_vtx_indices, _m.vtx_indices, _m.vtx_indices_count);

    vertices_.Append(&new_vertices[0], new_vertices.size());

    // add vertex indices
    vtx_indices_.Append(&new_vtx_indices[0], new_vtx_indices.size());

    // add triangles
    tris_.Append(&new_tris[0], new_tris.size());

    // add triangle indices
    tri_indices_.Append(&new_tri_indices[0], new_tri_indices.size());

    return mesh_index;
}

void Ray::Ocl::Scene::RemoveMesh(uint32_t) {
    // TODO!!!
}

uint32_t Ray::Ocl::Scene::AddLight(const light_desc_t &_l) {
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
        float dist = 99999999.0f;

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

    lights_.PushBack(l);

    RebuildLightBVH();

    return (uint32_t)(lights_.size() - 1);
}

void Ray::Ocl::Scene::RemoveLight(uint32_t i) {
    // TODO!!!
}

uint32_t Ray::Ocl::Scene::AddMeshInstance(uint32_t mesh_index, const float *xform) {
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

void Ray::Ocl::Scene::SetMeshInstanceTransform(uint32_t mi_index, const float *xform) {
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

void Ray::Ocl::Scene::RemoveMeshInstance(uint32_t) {
    // TODO!!
}

void Ray::Ocl::Scene::RemoveNodes(uint32_t node_index, uint32_t node_count) {
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
            if (!n.prim_count) {
                if (n.left_child > node_index) n.left_child -= node_count;
                if (n.right_child > node_index) n.right_child -= node_count;
            }
        }
        nodes_.Set(&nodes[0], 0, nodes_count);

        if (macro_nodes_start_ > node_index) {
            macro_nodes_start_ -= node_count;
        }

        if (light_nodes_start_ > node_index) {
            light_nodes_start_ -= node_count;
        }
    }
}

void Ray::Ocl::Scene::RebuildMacroBVH() {
    RemoveNodes(macro_nodes_start_, macro_nodes_count_);
    mi_indices_.Clear();

    size_t mi_count = mesh_instances_.size();

    std::vector<prim_t> primitives;
    primitives.reserve(mi_count);

    std::vector<mesh_instance_t> mesh_instances(mi_count);
    mesh_instances_.Get(&mesh_instances[0], 0, mi_count);

    for (const auto &mi : mesh_instances) {
        primitives.push_back({ 0, 0, 0, Ref::simd_fvec3{ mi.bbox_min }, Ref::simd_fvec3{ mi.bbox_max } });
    }

    std::vector<bvh_node_t> bvh_nodes;
    std::vector<uint32_t> mi_indices;

    macro_nodes_start_ = (uint32_t)nodes_.size();
    macro_nodes_count_ = PreprocessPrims(&primitives[0], primitives.size(), nullptr, 0, false, bvh_nodes, mi_indices);

    // offset nodes
    for (auto &n : bvh_nodes) {
        if (n.parent != 0xffffffff) n.parent += (uint32_t)nodes_.size();
        if (!n.prim_count) {
            n.left_child += (uint32_t)nodes_.size();
            n.right_child += (uint32_t)nodes_.size();
        }
    }

    nodes_.Append(&bvh_nodes[0], bvh_nodes.size());
    mi_indices_.Append(&mi_indices[0], mi_indices.size());
}

void Ray::Ocl::Scene::RebuildLightBVH() {
    RemoveNodes(light_nodes_start_, light_nodes_count_);
    li_indices_.Clear();

    std::vector<prim_t> primitives;
    primitives.reserve(lights_.size());

    std::vector<light_t> lights(lights_.size());
    lights_.Get(&lights[0], 0, lights_.size());

    for (const auto &l : lights) {
        float influence = l.radius * (std::sqrt(l.brightness / LIGHT_ATTEN_CUTOFF) - 1.0f);

        Ref::simd_fvec3 bbox_min = { 0.0f }, bbox_max = { 0.0f };

        Ref::simd_fvec3 p1 = { -l.dir[0] * influence,
                               -l.dir[1] * influence,
                               -l.dir[2] * influence };

        bbox_min = min(bbox_min, p1);
        bbox_max = max(bbox_max, p1);

        Ref::simd_fvec3 p2 = { -l.dir[0] * l.spot * influence,
                               -l.dir[1] * l.spot * influence,
                               -l.dir[2] * l.spot * influence };

        float d = std::sqrt(1.0f - l.spot * l.spot) * influence;

        bbox_min = min(bbox_min, p2 - Ref::simd_fvec3{ d, 0.0f, d });
        bbox_max = max(bbox_max, p2 + Ref::simd_fvec3{ d, 0.0f, d });

        if (l.spot < 0.0f) {
            bbox_min = min(bbox_min, p1 - Ref::simd_fvec3{ influence, 0.0f, influence });
            bbox_max = max(bbox_max, p1 + Ref::simd_fvec3{ influence, 0.0f, influence });
        }

        Ref::simd_fvec3 up = { 1.0f, 0.0f, 0.0f };
        if (std::abs(l.dir[1]) < std::abs(l.dir[2]) && std::abs(l.dir[1]) < std::abs(l.dir[0])) {
            up = { 0.0f, 1.0f, 0.0f };
        } else if (std::abs(l.dir[2]) < std::abs(l.dir[0]) && std::abs(l.dir[2]) < std::abs(l.dir[1])) {
            up = { 0.0f, 0.0f, 1.0f };
        }

        Ref::simd_fvec3 side = { -l.dir[1] * up[2] + l.dir[2] * up[1],
                                 -l.dir[2] * up[0] + l.dir[0] * up[2],
                                 -l.dir[0] * up[1] + l.dir[1] * up[0] };

        float xform[16] = { side[0],  l.dir[0], up[0],    0.0f,
                            side[1],  l.dir[1], up[1],    0.0f,
                            side[2],  l.dir[2], up[2],    0.0f,
                            l.pos[0], l.pos[1], l.pos[2], 1.0f };

        float bbox[2][3] = { { bbox_min[0], bbox_min[1], bbox_min[2] },
                             { bbox_max[0], bbox_max[1], bbox_max[2] } };
        float tr_bbox[2][3];

        TransformBoundingBox(bbox, xform, tr_bbox);

        primitives.push_back({ 0, 0, 0, Ref::simd_fvec3{ tr_bbox[0] }, Ref::simd_fvec3{ tr_bbox[1] } });
    }

    std::vector<bvh_node_t> bvh_nodes;
    std::vector<uint32_t> li_indices;

    light_nodes_start_ = (uint32_t)nodes_.size();
    light_nodes_count_ = PreprocessPrims(&primitives[0], primitives.size(), nullptr, 0, false, bvh_nodes, li_indices);

    // offset nodes
    for (auto &n : bvh_nodes) {
        if (n.parent != 0xffffffff) n.parent += (uint32_t)nodes_.size();
        if (!n.prim_count) {
            n.left_child += (uint32_t)nodes_.size();
            n.right_child += (uint32_t)nodes_.size();
        }
    }

    nodes_.Append(&bvh_nodes[0], bvh_nodes.size());
    li_indices_.Append(&li_indices[0], li_indices.size());
}
