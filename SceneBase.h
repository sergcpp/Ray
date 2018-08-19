#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>

#include "Types.h"

/**
  @file SceneBase.h
  @brief Contains common scene interface
*/

namespace Ray {
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
    bool allow_spatial_splits = false;  ///< Better BVH, worse load times and memory consumption
};

/// Texture description
struct tex_desc_t {
    const pixel_color8_t *data;     ///< Single byte RGBA pixel data
    int w,                          ///< Texture width
        h;                          ///< Texture height
    bool generate_mipmaps;
};

enum eLightType {
    PointLight,
    SpotLight,
    DirectionalLight,
};

// Light description
struct light_desc_t {
    eLightType type;
    float position[3], radius;
    float color[3];
    float direction[3], angle;
};

/// Environment description
struct environment_desc_t {
    float sky_col[3];               ///< Sky color
};

/** Base Scene class,
    cpu and gpu backends have different implementation of SceneBase
*/
class SceneBase {
protected:
    union cam_storage_t {
        camera_t cam;
        uint32_t next_free;
    };

    std::vector<cam_storage_t> cams_;       ///< scene cameras
    uint32_t cam_first_free_ = 0xffffffff;  ///< index to first free cam in cams_ array

    uint32_t current_cam_ = 0xffffffff;     ///< index of current camera
public:
    virtual ~SceneBase() = default;

    /// Get current environment description
    virtual void GetEnvironment(environment_desc_t &env) = 0;

    /// Set environment from description
    virtual void SetEnvironment(const environment_desc_t &env) = 0;

    /** @brief Adds texture to scene
        @param t texture description
        @return New texture index
    */
    virtual uint32_t AddTexture(const tex_desc_t &t) = 0;

    /** @brief Removes texture with specific index from scene
        @param i texture index
    */
    virtual void RemoveTexture(uint32_t i) = 0;

    /** @brief Adds material to scene
        @param m material description
        @return New material index
    */
    virtual uint32_t AddMaterial(const mat_desc_t &m) = 0;

    /** @brief Removes material with specific index from scene
        @param i material index
    */
    virtual void RemoveMaterial(uint32_t i) = 0;

    /** @brief Adds mesh to scene
        @param m mesh description
        @return New mesh index
    */
    virtual uint32_t AddMesh(const mesh_desc_t &m) = 0;

    /** @brief Removes mesh with specific index from scene
        @param i mesh index
    */
    virtual void RemoveMesh(uint32_t i) = 0;

    /** @brief Adds light to scene
        @param l light description
        @return New light index
    */
    virtual uint32_t AddLight(const light_desc_t &l) = 0;

    /** @brief Removes light with specific index from scene
        @param i light index
    */
    virtual void RemoveLight(uint32_t i) = 0;

    /** @brief Adds mesh instance to a scene
        @param m_index mesh index
        @param xform array of 16 floats holding transformation matrix
        @return New mesh instance index
    */
    virtual uint32_t AddMeshInstance(uint32_t m_index, const float *xform) = 0;

    /** @brief Sets mesh instance transformation
        @param mi_index mesh instance index
        @param xform array of 16 floats holding transformation matrix
    */
    virtual void SetMeshInstanceTransform(uint32_t mi_index, const float *xform) = 0;

    /** @brief Removes mesh instance from scene
        @param mi_index mesh instance index
        
        Removes mesh instance from scene. Associated mesh remains loaded in scene even if
        there is no instances of this mesh left.
    */
    virtual void RemoveMeshInstance(uint32_t mi_index) = 0;

    /** @brief Adds camera to a scene
        @param type camera projection type
        @return New camera index
    */
    uint32_t AddCamera(eCamType type) {
        const float o[3] = { 0, 0, 0 }, fwd[3] = { 0, 0, -1 }, fov = 60;
        return AddCamera(type, Tent, o, fwd, fov, 1.0f, 1.0f, 0.001f);
    }

    /** @brief Adds camera to a scene
        @param type camera projection type
        @param filter image filter type
        @param origin camera origin point
        @param fwd camera forward unit vector
        @param fov camera field of view in grad.
        @param gamma
        @param focus distance
        @param
        @return New camera index
    */
    uint32_t AddCamera(eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor);
    
    /** @brief Get const reference to a camera with specific index
        @param i camera index
        @return Camera const reference
    */
    const camera_t &GetCamera(uint32_t i) const {
        return cams_[i].cam;
    }

    /** @brief Sets camera properties
        @param i camera index
        @param type camera projection type
        @param origin camera origin point
        @param fwd camera forward unit vector
        @param fov camera field of view in grad.
    */
    void SetCamera(uint32_t i, eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor);
    
    /** @brief Removes camera with specific index from scene
        @param i camera index

        Removes camera with specific index from scene. Other cameras indices remain valid.
    */
    void RemoveCamera(uint32_t i);

    /** @brief Get const reference to a camera with specific index
        @return Current camera index
    */
    uint32_t current_cam() {
        return current_cam_;
    }

    /** @brief Sets camera with specific index to be current
        @param i camera index
    */
    void set_current_cam(uint32_t i) {
        current_cam_ = i;
    }

    /// Overall triangle count in scene
    virtual uint32_t triangle_count() = 0;

    /// Overall BVH node count in scene
    virtual uint32_t node_count() = 0;
};
}