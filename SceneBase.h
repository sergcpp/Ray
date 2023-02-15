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
#define DEFINE_HANDLE(object)                                                                                           \
    using object = struct object##_T {                                                                                  \
        uint32_t _index;                                                                                                \
    };                                                                                                                  \
    inline bool operator==(const object lhs, const object rhs) { return lhs._index == rhs._index; }                     \
    inline bool operator!=(const object lhs, const object rhs) { return lhs._index != rhs._index; }                     \
    static const object Invalid##object = object{0xffffffff};

DEFINE_HANDLE(Camera)
DEFINE_HANDLE(Texture)
DEFINE_HANDLE(Material)
DEFINE_HANDLE(Mesh)
DEFINE_HANDLE(MeshInstance)
DEFINE_HANDLE(Light)

#undef DEFINE_HANDLE

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
    eShadingNode type;                             ///< Material type
    float base_color[3] = {1, 1, 1};               ///< Base color
    Texture base_texture = InvalidTexture;         ///< Base texture index
    Texture normal_map = InvalidTexture;           ///< Normal map index
    float normal_map_intensity = 1.0f;             ///< Normal map intensity
    Material mix_materials[2] = {InvalidMaterial}; ///< Indices for two materials for mixing
    float roughness = 0;
    Texture roughness_texture = InvalidTexture;
    float anisotropic = 0;          ///<
    float anisotropic_rotation = 0; ///<
    float sheen = 0;
    float specular = 0;
    float strength = 1; ///< Strength of emissive material
    float fresnel = 1;  ///< Fresnel factor of mix material
    float ior = 1;      ///< IOR for reflective or refractive material
    float tint = 0;
    Texture metallic_texture = InvalidTexture;
    bool multiple_importance = false; ///< Enable explicit emissive geometry sampling
    bool mix_add = false;
};

/// Printcipled material descriptor struct (metallicness workflow)
struct principled_mat_desc_t {
    float base_color[3] = {1, 1, 1};
    Texture base_texture = InvalidTexture;
    float metallic = 0;
    Texture metallic_texture = InvalidTexture;
    float specular = 0.5f;
    Texture specular_texture = InvalidTexture;
    float specular_tint = 0;
    float roughness = 0.5f;
    Texture roughness_texture = InvalidTexture;
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
    Texture emission_texture = InvalidTexture;
    float emission_strength = 0;
    float alpha = 1.0f;
    Texture alpha_texture = InvalidTexture;
    Texture normal_map = InvalidTexture;
    float normal_map_intensity = 1.0f;
};

/// Defines mesh region with specific material
struct shape_desc_t {
    Material front_mat; ///< Index of material
    Material back_mat;  ///< Index of material applied for back faces
    size_t vtx_start;   ///< Vertex start index
    size_t vtx_count;   ///< Vertex count

    shape_desc_t(const Material _front_material, const Material _back_material, size_t _vtx_start, size_t _vtx_count)
        : front_mat(_front_material), back_mat(_back_material), vtx_start(_vtx_start), vtx_count(_vtx_count) {}

    shape_desc_t(const Material _front_material, size_t _vtx_start, size_t _vtx_count)
        : front_mat(_front_material), back_mat(InvalidMaterial), vtx_start(_vtx_start), vtx_count(_vtx_count) {}
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

enum eTextureFormat { RGBA8888, RGB888, RG88, R8 };

/// Texture description
struct tex_desc_t {
    eTextureFormat format;
    const char *name = nullptr; ///< Debug name
    const void *data;
    int w, ///< Texture width
        h; ///< Texture height
    bool is_srgb = true;
    bool is_normalmap = false;
    bool force_no_compression = false; ///< Make sure texture will have the best quality
    bool generate_mipmaps = false;
};

enum eLightType { SphereLight, SpotLight, DirectionalLight, LineLight, RectLight };

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

struct spot_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float position[3] = {0.0f, 0.0f, 0.0f};
    float direction[3] = {0.0f, -1.0f, 0.0f};
    float spot_size = 45.0f;
    float spot_blend = 0.15f;
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
    float origin[3] = {};                ///< Camera origin
    float fwd[3] = {};                   ///< Camera forward unit vector
    float up[3] = {};                    ///< Camera up vector (optional)
    float shift[2] = {};                 ///< Camera shift
    float exposure = 0.0f;               ///< Camera exposure in stops (output = value * (2 ^ exposure))
    float fov = 45.0f, gamma = 1.0f;     ///< Field of view in degrees, gamma
    float sensor_height = 0.036f;        ///< Camera sensor height
    float focus_distance = 1.0f;         ///< Distance to focus point
    float focal_length = 0.0f;           ///< Focal length
    float fstop = 0.0f;                  ///< Focal fstop
    float lens_rotation = 0.0f;          ///< Bokeh rotation
    float lens_ratio = 1.0f;             ///< Bokeh distortion
    int lens_blades = 0;                 ///< Bokeh shape
    float clip_start = 0;                ///< Clip start
    float clip_end = 3.402823466e+30F;   ///< Clip end
    uint32_t mi_index = 0xffffffff,      ///< Index of mesh instance
        uv_index = 0;                    ///< UV layer used by geometry cam
    bool lighting_only = false;          ///< Render lightmap only
    bool skip_direct_lighting = false;   ///< Render indirect light contribution only
    bool skip_indirect_lighting = false; ///< Render direct light contribution only
    bool no_background = false;          ///< Do not render background
    bool clamp = false;                  ///< Clamp color values to [0..1] range
    bool output_sh = false;              ///< Output 2-band (4 coeff) spherical harmonics data
    uint8_t max_diff_depth = 4;          ///< Maximum tracing depth of diffuse rays
    uint8_t max_spec_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_refr_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_transp_depth = 8; ///< Maximum tracing depth of transparency rays (note: does not obey total depth)
    uint8_t max_total_depth = 8;  ///< Maximum tracing depth of all rays (except transparency)
    uint8_t min_total_depth = 2;  ///< Depth after which random rays termination starts
    uint8_t min_transp_depth = 2; ///< Depth after which random rays termination starts
};

/// Environment description
struct environment_desc_t {
    float env_col[3] = {};             ///< Environment color
    Texture env_map = InvalidTexture;  ///< Environment texture
    float back_col[3] = {};            ///< Background color
    Texture back_map = InvalidTexture; ///< Background texture
    float env_map_rotation = 0.0f;
    float back_map_rotation = 0.0f;
    bool multiple_importance = true; ///< Enable explicit env map sampling
};

/** Base Scene class,
    cpu and gpu backends have different implementation of SceneBase
*/
class SceneBase {
  protected:
    union cam_storage_t {
        camera_t cam;
        Camera next_free;
    };

    std::vector<cam_storage_t> cams_;      ///< scene cameras
    Camera cam_first_free_ = InvalidCamera; ///< index to first free cam in cams_ array

    Camera current_cam_ = InvalidCamera; ///< index of current camera
  public:
    virtual ~SceneBase() = default;

    /// Get current environment description
    virtual void GetEnvironment(environment_desc_t &env) = 0;

    /// Set environment from description
    virtual void SetEnvironment(const environment_desc_t &env) = 0;

    /** @brief Adds texture to scene
        @param t texture description
        @return New texture handle
    */
    virtual Texture AddTexture(const tex_desc_t &t) = 0;

    /** @brief Removes texture with specific index from scene
        @param t texture handle
    */
    virtual void RemoveTexture(Texture t) = 0;

    /** @brief Adds material to scene
        @param m root shading node description
        @return New material handle
    */
    virtual Material AddMaterial(const shading_node_desc_t &m) = 0;

    /** @brief Adds material to scene
        @param m principled material description
        @return New material handle
    */
    virtual Material AddMaterial(const principled_mat_desc_t &m) = 0;

    /** @brief Removes material with specific index from scene
        @param m material handle
    */
    virtual void RemoveMaterial(Material m) = 0;

    /** @brief Adds mesh to scene
        @param m mesh description
        @return New mesh index
    */
    virtual Mesh AddMesh(const mesh_desc_t &m) = 0;

    /** @brief Removes mesh with specific index from scene
        @param i mesh index
    */
    virtual void RemoveMesh(Mesh m) = 0;

    /** @brief Adds light to scene
        @param l light description
        @return New light index
    */
    virtual Light AddLight(const directional_light_desc_t &l) = 0;
    virtual Light AddLight(const sphere_light_desc_t &l) = 0;
    virtual Light AddLight(const spot_light_desc_t &l) = 0;
    virtual Light AddLight(const rect_light_desc_t &l, const float *xform) = 0;
    virtual Light AddLight(const disk_light_desc_t &l, const float *xform) = 0;
    virtual Light AddLight(const line_light_desc_t &l, const float *xform) = 0;

    /** @brief Removes light with specific index from scene
        @param l light handle
    */
    virtual void RemoveLight(Light l) = 0;

    /** @brief Adds mesh instance to a scene
        @param mesh mesh handle
        @param xform array of 16 floats holding transformation matrix
        @return New mesh instance handle
    */
    virtual MeshInstance AddMeshInstance(Mesh mesh, const float *xform) = 0;

    /** @brief Sets mesh instance transformation
        @param mi mesh instance handle
        @param xform array of 16 floats holding transformation matrix
    */
    virtual void SetMeshInstanceTransform(MeshInstance mi, const float *xform) = 0;

    /** @brief Removes mesh instance from scene
        @param mi mesh instance handle

        Removes mesh instance from scene. Associated mesh remains loaded in scene even if
        there is no instances of this mesh left.
    */
    virtual void RemoveMeshInstance(MeshInstance mi) = 0;

    virtual void Finalize() = 0;

    /** @brief Adds camera to a scene
        @param c camera description
        @return New camera handle
    */
    Camera AddCamera(const camera_desc_t &c);

    /** @brief Get camera description
        @param i camera handle
    */
    void GetCamera(Camera i, camera_desc_t &c) const;
    ;

    /** @brief Sets camera properties
        @param i camera handle
        @param c camera description
    */
    void SetCamera(Camera i, const camera_desc_t &c);

    /** @brief Removes camera with specific index from scene
        @param i camera handle

        Removes camera with specific index from scene. Other cameras indices remain valid.
    */
    void RemoveCamera(Camera i);

    /** @brief Get const reference to a camera with specific index
        @return Current camera index
    */
    Camera current_cam() const { return current_cam_; }

    /** @brief Sets camera with specific index to be current
        @param i camera index
    */
    void set_current_cam(Camera i) { current_cam_ = i; }

    /// Overall triangle count in scene
    virtual uint32_t triangle_count() const = 0;

    /// Overall BVH node count in scene
    virtual uint32_t node_count() const = 0;
};
} // namespace Ray
