#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>

#include "Types.h"

namespace ray {
enum ePrimType {
    // indexed triangle list
    TriangleList,
};
enum eVertexLayout {
    // P - vertex position
    // N - vertex normal
    // T - vertex texture coordinates
    PxyzNxyzTuv, // [ P.x, P.y, P.z, N.x, N.y, N.z, T.x, Ty ]
};

enum eMaterialType {
    DiffuseMaterial,
    GlossyMaterial,
    RefractiveMaterial,
    EmissiveMaterial,
    MixMaterial,
    TransparentMaterial,
};

struct mat_desc_t {
    eMaterialType type;
    float main_color[3] = { 1, 1, 1 };
    uint32_t main_texture;
    uint32_t normal_map = 0xffffffff;
    uint32_t mix_materials[2] = { 0xffffffff };
    float roughness = 0;
    float strength = 1;
    float fresnel = 1;
    float ior = 1;
};

struct shape_desc_t {
    uint32_t material_index;
    size_t vtx_start;
    size_t vtx_count;
};

struct mesh_desc_t {
    ePrimType prim_type;
    eVertexLayout layout;
    const float *vtx_attrs;
    size_t vtx_attrs_count;
    const uint32_t *vtx_indices;
    size_t vtx_indices_count;
    std::vector<shape_desc_t> shapes;
};

struct tex_desc_t {
    const pixel_color8_t *data;
    int w, h;
};

struct environment_desc_t {
    float sun_dir[3];
    float sun_col[3];
    float sky_col[3];
    float sun_softness;
};

class SceneBase {
protected:
    union cam_storage_t {
        camera_t cam;
        uint32_t next_free;
    };

    std::vector<cam_storage_t> cams_;
    uint32_t cam_first_free_ = 0xffffffff;

    uint32_t current_cam_ = 0xffffffff;
public:
    virtual ~SceneBase() = default;

    virtual void GetEnvironment(environment_desc_t &env) = 0;
    virtual void SetEnvironment(const environment_desc_t &env) = 0;

    virtual uint32_t AddTexture(const tex_desc_t &t) = 0;
    virtual void RemoveTexture(uint32_t) = 0;

    virtual uint32_t AddMaterial(const mat_desc_t &m) = 0;
    virtual void RemoveMaterial(uint32_t) = 0;

    virtual uint32_t AddMesh(const mesh_desc_t &m) = 0;
    virtual void RemoveMesh(uint32_t) = 0;

    virtual uint32_t AddMeshInstance(uint32_t m_index, const float *xform) = 0;
    virtual void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) = 0;
    virtual void RemoveMeshInstance(uint32_t) = 0;

    uint32_t AddCamera(eCamType type) {
        const float o[3] = { 0, 0, 0 }, fwd[3] = { 0, 0, -1 }, fov = 60;
        return AddCamera(type, o, fwd, fov);
    }
    uint32_t AddCamera(eCamType type, const float origin[3], const float fwd[3], float fov);
    const camera_t &GetCamera(uint32_t i) const {
        return cams_[i].cam;
    }
    void SetCamera(uint32_t i, eCamType type, const float origin[3], const float fwd[3], float fov);
    void RemoveCamera(uint32_t i);

    uint32_t current_cam() {
        return current_cam_;
    }
    void set_current_cam(uint32_t i) {
        current_cam_ = i;
    }

    virtual uint32_t triangle_count() = 0;
    virtual uint32_t node_count() = 0;
};
}