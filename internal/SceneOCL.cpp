#include "SceneOCL.h"

#include "BVHSplit.h"

#include <math/math.hpp>

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
    texture_atlas_(context_, queue_, {
    MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE
}) {
    SetEnvironment( { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });

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
    math::ivec2 res = { _t.w, _t.h };

    // TODO: add downsampling for non-power-of-2 textures
    auto downsample = [](std::vector<pixel_color8_t> &_tex, const math::ivec2 &res) {
        const pixel_color8_t *tex = &_tex[0];

        std::vector<pixel_color8_t> ret;
        for (int j = 0; j < res.y; j += 2) {
            for (int i = 0; i < res.x; i += 2) {
                int r = tex[(j + 0) * res.x + i].r + tex[(j + 0) * res.x + i + 1].r +
                        tex[(j + 1) * res.x + i].r + tex[(j + 1) * res.x + i + 1].r;
                int g = tex[(j + 0) * res.x + i].g + tex[(j + 0) * res.x + i + 1].g +
                        tex[(j + 1) * res.x + i].g + tex[(j + 1) * res.x + i + 1].g;
                int b = tex[(j + 0) * res.x + i].b + tex[(j + 0) * res.x + i + 1].b +
                        tex[(j + 1) * res.x + i].b + tex[(j + 1) * res.x + i + 1].b;
                int a = tex[(j + 0) * res.x + i].a + tex[(j + 0) * res.x + i + 1].a +
                        tex[(j + 1) * res.x + i].a + tex[(j + 1) * res.x + i + 1].a;

                ret.push_back( { (uint8_t)std::round(r * 0.25f), (uint8_t)std::round(g * 0.25f),
                                 (uint8_t)std::round(b * 0.25f), (uint8_t)std::round(a * 0.25f)
                               });
            }
        }
        return ret;
    };

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

        tex_data = downsample(tex_data, res);

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
    using namespace math;

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

    //const auto &t = textures_[0];

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

    std::vector<std::array<uint32_t, 3>> twin_verts(new_vertices.size(), { 0, 0, 0 });
    aligned_vector<vec3> binormals(new_vertices.size());
    for (size_t i = 0; i < _m.vtx_indices_count; i += 3) {
        auto *v0 = &new_vertices[_m.vtx_indices[i + 0]];
        auto *v1 = &new_vertices[_m.vtx_indices[i + 1]];
        auto *v2 = &new_vertices[_m.vtx_indices[i + 2]];

        auto *b0 = &binormals[_m.vtx_indices[i + 0]];
        auto *b1 = &binormals[_m.vtx_indices[i + 1]];
        auto *b2 = &binormals[_m.vtx_indices[i + 2]];

        vec3 dp1 = make_vec3(v1->p) - make_vec3(v0->p);
        vec3 dp2 = make_vec3(v2->p) - make_vec3(v0->p);

        vec2 dt1 = make_vec2(v1->t0) - make_vec2(v0->t0);
        vec2 dt2 = make_vec2(v2->t0) - make_vec2(v0->t0);

        float inv_det = 1.0f / (dt1.x * dt2.y - dt1.y * dt2.x);
        vec3 tangent = (dp1 * dt2.y - dp2 * dt1.y) * inv_det;
        vec3 binormal = (dp2 * dt1.x - dp1 * dt2.x) * inv_det;

        int i1 = v0->b[0] * tangent.x + v0->b[1] * tangent.y + v0->b[2] * tangent.z < 0;
        int i2 = 2 * (b0->x * binormal.x + b0->y * binormal.y + b0->z * binormal.z < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[_m.vtx_indices[i + 0]][i1 + i2 - 1];
            if (index == 0) {
                index = (uint32_t)(vertices_.size() + new_vertices.size());
                new_vertices.push_back(*v0);
                memset(&new_vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[_m.vtx_indices[i + 0]][i1 + i2 - 1] = index;

                v1 = &new_vertices[_m.vtx_indices[i + 1]];
                v2 = &new_vertices[_m.vtx_indices[i + 2]];
            }
            new_vtx_indices[i] = index;
            v0 = &new_vertices[index - vertices_.size()];
        } else {
            *b0 = binormal;
        }

        v0->b[0] += tangent.x;
        v0->b[1] += tangent.y;
        v0->b[2] += tangent.z;

        i1 = v1->b[0] * tangent.x + v1->b[1] * tangent.y + v1->b[2] * tangent.z < 0;
        i2 = 2 * (b1->x * binormal.x + b1->y * binormal.y + b1->z * binormal.z < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[_m.vtx_indices[i + 1]][i1 + i2 - 1];
            if (index == 0) {
                index = (uint32_t)(vertices_.size() + new_vertices.size());
                new_vertices.push_back(*v1);
                memset(&new_vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[_m.vtx_indices[i + 1]][i1 + i2 - 1] = index;

                v0 = &new_vertices[_m.vtx_indices[i + 0]];
                v2 = &new_vertices[_m.vtx_indices[i + 2]];
            }
            new_vtx_indices[i + 1] = index;
            v1 = &new_vertices[index - vertices_.size()];
        } else {
            *b1 = binormal;
        }

        v1->b[0] += tangent.x;
        v1->b[1] += tangent.y;
        v1->b[2] += tangent.z;

        i1 = v2->b[0] * tangent.x + v2->b[1] * tangent.y + v2->b[2] * tangent.z < 0;
        i2 = 2 * (b2->x * binormal.x + b2->y * binormal.y + b2->z * binormal.z < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[_m.vtx_indices[i + 2]][i1 + i2 - 1];
            if (index == 0) {
                index = (uint32_t)(vertices_.size() + new_vertices.size());
                new_vertices.push_back(*v2);
                memset(&new_vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[_m.vtx_indices[i + 2]][i1 + i2 - 1] = index;

                v0 = &new_vertices[_m.vtx_indices[i + 0]];
                v1 = &new_vertices[_m.vtx_indices[i + 1]];
            }
            new_vtx_indices[i + 2] = index;
            v2 = &new_vertices[index - vertices_.size()];
        } else {
            *b2 = binormal;
        }

        v2->b[0] += tangent.x;
        v2->b[1] += tangent.y;
        v2->b[2] += tangent.z;
    }

    for (auto &v : new_vertices) {
        vec3 tangent = make_vec3(v.b);
        vec3 binormal = normalize(cross(make_vec3(v.n), tangent));
        memcpy(&v.b[0], value_ptr(binormal), 3 * sizeof(float));
    }

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

    math::mat4 inv_mat = math::inverse(math::make_mat4(xform));
    memcpy(tr.xform, xform, 16 * sizeof(float));
    memcpy(tr.inv_xform, math::value_ptr(inv_mat), 16 * sizeof(float));
    transforms_.PushBack(tr);

    SetMeshInstanceTransform(mi_index, xform);

    return mi_index;
}

void ray::ocl::Scene::SetMeshInstanceTransform(uint32_t mi_index, const float *xform) {
    transform_t tr;

    math::mat4 inv_mat = math::inverse(math::make_mat4(xform));
    memcpy(tr.xform, xform, 16 * sizeof(float));
    memcpy(tr.inv_xform, math::value_ptr(inv_mat), 16 * sizeof(float));

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
    using namespace math;

    RemoveNodes(macro_nodes_start_, macro_nodes_count_);
    mi_indices_.Clear();

    size_t mi_count = mesh_instances_.size();

    std::vector<prim_t> primitives;
    primitives.reserve(mi_count);

    std::vector<mesh_instance_t> mesh_instances(mi_count);
    mesh_instances_.Get(&mesh_instances[0], 0, mi_count);

    for (const auto &mi : mesh_instances) {
        primitives.push_back( { make_vec3(mi.bbox_min), make_vec3(mi.bbox_max) });
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
