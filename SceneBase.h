#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>

#include "Types.h"

/**
  @file
*/

namespace ray {
/// Mesh primitive type
enum ePrimType {
    TriangleList,   ///< indexed triangle list
};

/** Vertex attribute layout.
    P - vertex position
    N - vertex normal
    T - vertex texture coordinates
*/
enum eVertexLayout {
    PxyzNxyzTuv, ///< [ P.x, P.y, P.z, N.x, N.y, N.z, T.x, Ty ]
};

/// Mesh region material type
enum eMaterialType {
    DiffuseMaterial,        ///< Pure Lambert diffuse
    GlossyMaterial,         ///< Pure reflective material
    RefractiveMaterial,     ///< Pure refractive material
    EmissiveMaterial,       ///< Pure emissive material
    MixMaterial,            ///< Mix of two materials
    TransparentMaterial,    ///< Transparent material
};

/// Material descriptor struct
struct mat_desc_t {
    eMaterialType type;                         ///< Material type
    float main_color[3] = { 1, 1, 1 };          ///< Main color
    uint32_t main_texture;                      ///< Main texture index
    uint32_t normal_map = 0xffffffff;           ///< Normal map index
    uint32_t mix_materials[2] = { 0xffffffff }; ///< Indices for two materials for mixing
    float roughness = 0;                        ///< Roughness of reflective or refractive material
    float strength = 1;                         ///< Strength of emissive material
    float fresnel = 1;                          ///< Fresnel factor of mix material
    float ior = 1;                              ///< IOR for reflective or refractive material
};

/// Defines mesh region with specific material
struct shape_desc_t {
    uint32_t material_index;    ///< Index of material
    size_t vtx_start;           ///< Vertex start index
    size_t vtx_count;           ///< Vertex count
};

/// Mesh description
struct mesh_desc_t {
    ePrimType prim_type;                ///< Primitive type
    eVertexLayout layout;               ///< Vertex attribute layout
    const float *vtx_attrs;             ///< Pointer to vertex attribute
    size_t vtx_attrs_count;             ///< Vertex attribute count (number of vertices)
    const uint32_t *vtx_indices;        ///< Pointer to vertex indices, defining primitive
    size_t vtx_indices_count;           ///< Primitive indices count
    std::vector<shape_desc_t> shapes;   ///< Vector of shapes in mesh
};

/// Texture description
struct tex_desc_t {
    const pixel_color8_t *data;     ///< Single byte RGBA pixel data
    int w,                          ///< Texture width
        h;                          ///< Texture height
};

/// Environment description
struct environment_desc_t {
    float sun_dir[3];               ///< Sun direction unit vector
    float sun_col[3];               ///< Sun color
    float sky_col[3];               ///< Sky color
    float sun_softness;             ///< defines shadow softness (0 - had shadow)
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