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
    TriangleList, ///< indexed triangle list
};

/** Vertex attribute layout.
    P - vertex position
    N - vertex normal
    B - vertex binormal (oriented to vertical texture axis)
    T - vertex texture coordinates
*/
enum eVertexLayout {
    PxyzNxyzTuv = 0,    ///< [ P.x, P.y, P.z, N.x, N.y, N.z, T.x, T.y ]
    PxyzNxyzTuvTuv,     ///< [ P.x, P.y, P.z, N.x, N.y, N.z, T.x, T.y, T.x, T.y ]
    PxyzNxyzBxyzTuv,    ///< [ P.x, P.y, P.z, N.x, N.y, N.z, B.x, B.y, B.z, T.x, T.y ]
    PxyzNxyzBxyzTuvTuv, ///< [ P.x, P.y, P.z, N.x, N.y, N.z, B.x, B.y, B.z, T.x, T.y, T.x, T.y ]
};

/** Vertex attribute stride value.
    Represented in number of floats
*/
const size_t AttrStrides[] = {
    8,  ///< PxyzNxyzTuv
    10, ///< PxyzNxyzTuvTuv
    11, ///< PxyzNxyzBxyzTuv
    13, ///< PxyzNxyzBxyzTuvTuv
};

/// Mesh region material type
enum eShadingNode {
    DiffuseNode,
    GlossyNode,
    RefractiveNode,
    EmissiveNode,
    MixNode,
    TransparentNode,
    PrincipledNode,
};

/// Shading node descriptor struct
struct shading_node_desc_t {
    eShadingNode type;                        ///< Material type
    float base_color[3] = {1, 1, 1};          ///< Base color
    uint32_t base_texture = 0xffffffff;       ///< Base texture index
    uint32_t normal_map = 0xffffffff;         ///< Normal map index
    float normal_map_intensity = 1.0f;        ///< Normal map intensity
    uint32_t mix_materials[2] = {0xffffffff}; ///< Indices for two materials for mixing
    float roughness = 0;
    uint32_t roughness_texture = 0xffffffff;
    float anisotropic = 0;                    ///< 
    float anisotropic_rotation = 0;           ///<
    float sheen = 0;
    float specular = 0;
    float strength = 1;                       ///< Strength of emissive material
    float fresnel = 1;                        ///< Fresnel factor of mix material
    float int_ior = 1;                        ///< interior IOR for reflective or refractive material
    float ext_ior = 1;                        ///< exterior IOR for reflective or refractive material
    float tint = 0;
    uint32_t metallic_texture = 0xffffffff;
    bool multiple_importance = false;         ///< Enable explicit emissive geometry sampling
    bool sky_portal = false;
    bool mix_add = false;
};

/// Printcipled material descriptor struct (metallicness workflow)
struct principled_mat_desc_t {
    float base_color[3] = {1, 1, 1};
    uint32_t base_texture = 0xffffffff;
    float metallic = 0;
    uint32_t metallic_texture = 0xffffffff;
    float specular = 0.5f;
    uint32_t specular_texture = 0xffffffff;
    float specular_tint = 0;
    float roughness = 0.5f;
    uint32_t roughness_texture = 0xffffffff;
    float anisotropic = 0;
    float anisotropic_rotation = 0;
    float sheen = 0;
    float sheen_tint = 0.5f;
    float clearcoat = 0.0f;
    float clearcoat_roughness = 0.0f;
    float ior = 1.45f;
    float transmission = 0.0f;
    float transmission_roughness = 0.0f;
    float emission_color[3] = {0, 0, 0};
    uint32_t emission_texture = 0xffffffff;
    float emission_strength = 0;
    float alpha = 1.0f;
    uint32_t alpha_texture = 0xffffffff;
    uint32_t normal_map = 0xffffffff;
    float normal_map_intensity = 1.0f;
};

/// Defines mesh region with specific material
struct shape_desc_t {
    uint32_t mat_index;      ///< Index of material
    uint32_t back_mat_index; ///< Index of material applied for back faces
    size_t vtx_start;        ///< Vertex start index
    size_t vtx_count;        ///< Vertex count

    shape_desc_t(uint32_t _material_index, uint32_t _back_material_index, size_t _vtx_start, size_t _vtx_count)
        : mat_index(_material_index), back_mat_index(_back_material_index), vtx_start(_vtx_start),
          vtx_count(_vtx_count) {}

    shape_desc_t(uint32_t _material_index, size_t _vtx_start, size_t _vtx_count)
        : mat_index(_material_index), back_mat_index(0xffffffff), vtx_start(_vtx_start), vtx_count(_vtx_count) {}
};

/// Mesh description
struct mesh_desc_t {
    ePrimType prim_type;               ///< Primitive type
    eVertexLayout layout;              ///< Vertex attribute layout
    const float *vtx_attrs;            ///< Pointer to vertex attribute
    size_t vtx_attrs_count;            ///< Vertex attribute count (number of vertices)
    const uint32_t *vtx_indices;       ///< Pointer to vertex indices, defining primitive
    size_t vtx_indices_count;          ///< Primitive indices count
    int base_vertex = 0;               ///< Shift applied to indices
    std::vector<shape_desc_t> shapes;  ///< Vector of shapes in mesh
    bool allow_spatial_splits = false; ///< Better BVH, worse load times and memory consumption
    bool use_fast_bvh_build = false;   ///< Use faster BVH construction with less tree quality
};

enum eTextureFormat {
    RGBA8888,
    RGB888,
    RG88,
    R8
};

/// Texture description
struct tex_desc_t {
    eTextureFormat format;
    const char *name = nullptr;         ///< Debug name
    const void *data;
    int w,                              ///< Texture width
        h;                              ///< Texture height
    bool is_srgb = true;
    bool is_normalmap = false;
    bool force_no_compression = false;  ///< Make sure texture will have the best quality
    bool generate_mipmaps = false;
};

enum eLightType {
    SphereLight,
    SpotLight,
    DirectionalLight,
    LineLight,
    RectLight
};

// Light description
struct directional_light_desc_t {
    float color[3];
    float direction[3], angle;
    bool cast_shadow = true;
};

struct sphere_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float position[3] = {0.0f, 0.0f, 0.0f};
    float radius = 1.0f;
    bool visible = true; // visibility for secondary bounces
    bool cast_shadow = true;
};

struct rect_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float width = 1.0f, height = 1.0f;
    bool sky_portal = false;
    bool visible = true; // visibility for secondary bounces
    bool cast_shadow = true;
};

struct disk_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float size_x = 1.0f, size_y = 1.0f;
    bool sky_portal = false;
    bool visible = true; // visibility for secondary bounces
    bool cast_shadow = true;
};

struct line_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float radius = 1.0f, height = 1.0f;
    bool sky_portal = false;
    bool visible = true; // visibility for secondary bounces
    bool cast_shadow = true;
};

// Camera description
struct camera_desc_t {
    eCamType type = Persp;               ///< Type of projection
    eFilterType filter = Tent;           ///< Reconstruction filter
    eDeviceType dtype = SRGB;            ///< Device type
    eLensUnits ltype = FOV;              ///< Lens units type
    float origin[3];                     ///< Camera origin
    float fwd[3] = {};                   ///< Camera forward unit vector
    float up[3];                         ///< Camera up vector (optional)
    float fov, gamma = 1.0f;             ///< Field of view in degrees, gamma
    float sensor_height = 0.036f;        ///< Camera sensor height
    float focus_distance = 1.0f;         ///< Distance to focus point
    float focal_length = 0.0f;           ///< Focal length
    float fstop = 0.0f;                  ///< Focal fstop
    float lens_rotation = 0.0f;          ///< Bokeh rotation
    float lens_ratio = 1.0f;             ///< Bokeh distortion
    int lens_blades = 0;                 ///< Bokeh shape
    float clip_start = 0;                ///< Clip start
    float clip_end = 3.402823466e+30F;   ///< Clip end
    uint32_t mi_index, uv_index = 0;     ///< Index of mesh instance and uv layer used by geometry cam
    bool lighting_only = false;          ///< Render lightmap only
    bool skip_direct_lighting = false;   ///< Render indirect light contribution only
    bool skip_indirect_lighting = false; ///< Render direct light contribution only
    bool no_background = false;          ///< Do not render background
    bool clamp = false;                  ///< Clamp color values to [0..1] range
    bool output_sh = false;              ///< Output 2-band (4 coeff) spherical harmonics data
    uint8_t max_diff_depth = 4;          ///< Maximum tracing depth of diffuse rays
    uint8_t max_spec_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_refr_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_transp_depth = 8;        ///< Maximum tracing depth of transparency rays
    uint8_t max_total_depth = 8;         ///< Maximum tracing depth of all rays
    uint8_t termination_start_depth = 3; ///< Depth at which random rays termination starts
};

/// Environment description
struct environment_desc_t {
    float env_col[3] = {0.0f};          ///< Environment color
    uint32_t env_map = 0xffffffff;      ///< Environment texture
    bool multiple_importance = true;    ///< Enable explicit env map sampling
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

    std::vector<cam_storage_t> cams_;      ///< scene cameras
    uint32_t cam_first_free_ = 0xffffffff; ///< index to first free cam in cams_ array

    uint32_t current_cam_ = 0xffffffff; ///< index of current camera
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
        @param m root shading node description
        @return New material index
    */
    virtual uint32_t AddMaterial(const shading_node_desc_t &m) = 0;

    /** @brief Adds material to scene
        @param m principled material description
        @return New material index
    */
    virtual uint32_t AddMaterial(const principled_mat_desc_t &m) = 0;

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
    virtual uint32_t AddLight(const directional_light_desc_t &l) = 0;
    virtual uint32_t AddLight(const sphere_light_desc_t &l) = 0;
    virtual uint32_t AddLight(const rect_light_desc_t &l, const float *xform) = 0;
    virtual uint32_t AddLight(const disk_light_desc_t &l, const float *xform) = 0;
    virtual uint32_t AddLight(const line_light_desc_t &l, const float *xform) = 0;

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

    virtual void Finalize() = 0;

    /** @brief Adds camera to a scene
        @param c camera description
        @return New camera index
    */
    uint32_t AddCamera(const camera_desc_t &c);

    /** @brief Get camera description
        @param i camera index
    */
    void GetCamera(uint32_t i, camera_desc_t &c) const;
    ;

    /** @brief Sets camera properties
        @param i camera index
        @param c camera description
    */
    void SetCamera(uint32_t i, const camera_desc_t &c);

    /** @brief Removes camera with specific index from scene
        @param i camera index

        Removes camera with specific index from scene. Other cameras indices remain valid.
    */
    void RemoveCamera(uint32_t i);

    /** @brief Get const reference to a camera with specific index
        @return Current camera index
    */
    uint32_t current_cam() const { return current_cam_; }

    /** @brief Sets camera with specific index to be current
        @param i camera index
    */
    void set_current_cam(uint32_t i) { current_cam_ = i; }

    /// Overall triangle count in scene
    virtual uint32_t triangle_count() = 0;

    /// Overall BVH node count in scene
    virtual uint32_t node_count() = 0;
};
} // namespace Ray
