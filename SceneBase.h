#pragma once

#include <cstddef>
#include <cstdint>

#include <functional>

#include "Span.h"
#include "Types.h"

/**
  @file SceneBase.h
  @brief Contains common scene interface
*/

namespace Ray {
#define DEFINE_HANDLE(object)                                                                                          \
    using object = struct object##_T {                                                                                 \
        uint32_t _index;                                                                                               \
        uint32_t _block;                                                                                               \
    };                                                                                                                 \
    inline bool operator==(const object lhs, const object rhs) { return lhs._index == rhs._index; }                    \
    inline bool operator!=(const object lhs, const object rhs) { return lhs._index != rhs._index; }                    \
    static const object Invalid##object = object{0xffffffff};

DEFINE_HANDLE(CameraHandle)
DEFINE_HANDLE(LightHandle)
DEFINE_HANDLE(MaterialHandle)
DEFINE_HANDLE(MeshHandle)
DEFINE_HANDLE(MeshInstanceHandle)
DEFINE_HANDLE(TextureHandle)

#undef DEFINE_HANDLE

const TextureHandle PhysicalSkyTexture = {0xfffffffe};

/// Mesh primitive type
enum class ePrimType {
    TriangleList, ///< indexed triangle list
};

/// Mesh region material type
enum class eShadingNode : uint32_t { Diffuse, Glossy, Refractive, Emissive, Mix, Transparent, Principled };

/// Shading node descriptor struct
struct shading_node_desc_t {
    eShadingNode type;                                         ///< Material type
    float base_color[3] = {1, 1, 1};                           ///< Base color
    TextureHandle base_texture = InvalidTextureHandle;         ///< Base texture index
    TextureHandle normal_map = InvalidTextureHandle;           ///< Normal map index
    float normal_map_intensity = 1.0f;                         ///< Normal map intensity
    MaterialHandle mix_materials[2] = {InvalidMaterialHandle}; ///< Indices for two materials for mixing
    float roughness = 0;                                       ///< Roughness
    TextureHandle roughness_texture = InvalidTextureHandle;    ///< Roughness texture
    float anisotropic = 0;                                     ///< Amount of anisotropy [0; 1]
    float anisotropic_rotation = 0;                            ///< Anisotropy rotation [-PI; +PI]
    float sheen = 0;                                           ///< Sheen
    float specular = 0;                                        ///< Specular
    float strength = 1;                                        ///< Strength of emissive material
    float fresnel = 1;                                         ///< Fresnel factor of mix material
    float ior = 1;                                             ///< IOR for reflective or refractive material
    float tint = 0;                                            ///< Specular tint
    TextureHandle metallic_texture = InvalidTextureHandle;     ///< Metalness texture
    bool importance_sample = false;                            ///< Enable explicit emissive geometry sampling
    bool mix_add = false;                                      ///< Enable additive mixing
};

/// Printcipled material descriptor struct (metallicness workflow)
struct principled_mat_desc_t {
    float base_color[3] = {1, 1, 1};                        ///< Base color
    TextureHandle base_texture = InvalidTextureHandle;      ///< Base color texture
    float metallic = 0;                                     ///< Metalness value
    TextureHandle metallic_texture = InvalidTextureHandle;  ///< Metalness texture
    float specular = 0.5f;                                  ///< Specular value [0; 1]
    TextureHandle specular_texture = InvalidTextureHandle;  ///< Specular texture
    float specular_tint = 0;                                ///< Specular tint
    float roughness = 0.5f;                                 ///< Roughness value
    TextureHandle roughness_texture = InvalidTextureHandle; ///< Roughness texture
    float anisotropic = 0;                                  ///< Amount of anisotropy [0; 1]
    float anisotropic_rotation = 0;                         ///< Anisotropy rotation [-PI; +PI]
    float sheen = 0;                                        ///< Sheen
    float sheen_tint = 0.5f;                                ///< Sheen tint
    float clearcoat = 0;                                    ///< Weight of clearcoat layer
    float clearcoat_roughness = 0;                          ///< Clearcoat layer roughness
    float ior = 1.45f;                                      ///< IOR
    float transmission = 0;                                 ///< Transmission amount
    float transmission_roughness = 0;                       ///< Transmission roughness
    float emission_color[3] = {0, 0, 0};                    ///< Emissive color
    TextureHandle emission_texture = InvalidTextureHandle;  ///< Emissive texture
    float emission_strength = 1;                            ///< Emission strength
    float alpha = 1;                                        ///< Material transparency (alpha blending)
    TextureHandle alpha_texture = InvalidTextureHandle;     ///< Transparency texture
    TextureHandle normal_map = InvalidTextureHandle;        ///< Material normalmap
    float normal_map_intensity = 1;                         ///< Normalmap intensity
    bool importance_sample = false;                         ///< Enable explicit emissive geometry sampling
};

/// Defines mesh region with specific material
struct mat_group_desc_t {
    MaterialHandle front_mat; ///< Index of material
    MaterialHandle back_mat;  ///< Index of material applied for back faces
    size_t vtx_start;         ///< Vertex start index
    size_t vtx_count;         ///< Vertex count

    mat_group_desc_t(const MaterialHandle _front_material, const MaterialHandle _back_material, size_t _vtx_start,
                     size_t _vtx_count)
        : front_mat(_front_material), back_mat(_back_material), vtx_start(_vtx_start), vtx_count(_vtx_count) {}

    mat_group_desc_t(const MaterialHandle _front_material, size_t _vtx_start, size_t _vtx_count)
        : front_mat(_front_material), back_mat(_front_material), vtx_start(_vtx_start), vtx_count(_vtx_count) {}
};

struct vtx_attribute_t {
    Span<const float> data; ///< Float array of data
    int offset = 0;         ///< Offset to attribute expressed in sizeof(float)
    int stride = 0;         ///< Stride between vertices expressed in sizeof(float)
};

/// Mesh description
struct mesh_desc_t {
    const char *name = nullptr;          ///< Mesh name (for debugging)
    ePrimType prim_type;                 ///< Primitive type
    vtx_attribute_t vtx_positions;       ///< Vertex positions
    vtx_attribute_t vtx_normals;         ///< Vertex normals
    vtx_attribute_t vtx_binormals;       ///< Vertex binormals (optional)
    vtx_attribute_t vtx_uvs;             ///< Vertex texture coordinates
    Span<const uint32_t> vtx_indices;    ///< Vertex indices, defining primitives
    int base_vertex = 0;                 ///< Shift applied to indices
    Span<const mat_group_desc_t> groups; ///< Shapes of a mesh
    bool allow_spatial_splits = false;   ///< Better BVH, worse load times and memory consumption
    bool use_fast_bvh_build = false;     ///< Use faster BVH construction with less tree quality
};

/// Mesh instance description
struct mesh_instance_desc_t {
    const float *xform = nullptr;        ///< 4x4 transformation matrix (16 floats)
    MeshHandle mesh = InvalidMeshHandle; ///< Mesh handle
    bool camera_visibility = true;       ///< Instance visibility to camera rays
    bool diffuse_visibility = true;      ///< Instance visibility to diffuse rays
    bool specular_visibility = true;     ///< Instance visibility to specular rays
    bool refraction_visibility = true;   ///< Instance visibility to refraction (transmission) rays
    bool shadow_visibility = true;       ///< Instance visibility to shadow rays
};

enum class eTextureFormat { Undefined, RGBA8888, RGB888, RG88, R8, BC1, BC3, BC4, BC5 };

enum class eTextureConvention {
    OGL, // OpenGL, default
    DX   // DirectX, invert y for normalmaps + flip BC-compressed textures vertically
};

const int TexFormatChannelCount[] = {
    0, // Undefined
    4, // RGBA8888
    3, // RGB888
    2, // RG88
    1, // R8
    3, // BC1
    4, // BC3
    1, // BC4
    2  // BC5
};

inline bool IsCompressedFormat(const eTextureFormat format) {
    switch (format) {
    case eTextureFormat::BC1:
    case eTextureFormat::BC3:
    case eTextureFormat::BC4:
    case eTextureFormat::BC5:
        return true;
    default:
        return false;
    }
}

/// Texture description
struct tex_desc_t {
    eTextureFormat format; ///< Texture data format
    eTextureConvention convention =
        eTextureConvention::OGL;       ///< Texture convention (affects normalmaps and BC-compressed textures)
    const char *name = nullptr;        ///< Debug name
    Span<const uint8_t> data;          ///< Texture data
    int w,                             ///< Texture width
        h;                             ///< Texture height
    int mips_count = 1;                ///< Count of mips provided in data
    bool is_srgb = true;               ///< Treat this texture as SRGB
    bool is_normalmap = false;         ///< Is this a normalmap
    bool is_YCoCg = false;             ///< Texture is in YCoCg format
    bool force_no_compression = false; ///< Disable compression (guarantee the best quality)
    bool generate_mipmaps = false;     ///< Generate mipmaps for this texture
    bool reconstruct_z = false;        ///< Reconstruct normalmap z component (instead of setting it to 1)
};

/// Directional lightsource description
struct directional_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float direction[3] = {0.0f, -1.0f, 0.0f}, angle = 0.0f;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Spherical light source description
struct sphere_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float position[3] = {0.0f, 0.0f, 0.0f};
    float radius = 1.0f;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Spotlight description
struct spot_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float position[3] = {0.0f, 0.0f, 0.0f};
    float direction[3] = {0.0f, -1.0f, 0.0f};
    float spot_size = 45.0f;
    float spot_blend = 0.15f;
    float radius = 1.0f;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Rectangular lightsource description
struct rect_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float width = 1.0f, height = 1.0f;
    bool sky_portal = false;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Disk lightsource description
struct disk_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float size_x = 1.0f, size_y = 1.0f;
    bool sky_portal = false;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Line light description
struct line_light_desc_t {
    float color[3] = {1.0f, 1.0f, 1.0f};
    float radius = 1.0f, height = 1.0f;
    bool sky_portal = false;
    bool visible = true;
    bool cast_shadow = true;
    bool diffuse_visibility = true;    ///< Light visibility to diffuse rays
    bool specular_visibility = true;   ///< Light visibility to specular rays
    bool refraction_visibility = true; ///< Light visibility to refraction (transmission) rays
};

/// Camera description
struct camera_desc_t {
    eCamType type = eCamType::Persp;                          ///< Type of projection
    ePixelFilter filter = ePixelFilter::BlackmanHarris;       ///< Reconstruction filter
    eViewTransform view_transform = eViewTransform::Standard; ///< View transform
    eLensUnits ltype = eLensUnits::FOV;                       ///< Lens units type
    float filter_width = 1.5f;                                ///< Width of the reconstruction filter
    float origin[3] = {};                                     ///< Camera origin
    float fwd[3] = {};                                        ///< Camera forward unit vector
    float up[3] = {};                                         ///< Camera up vector (optional)
    float shift[2] = {};                                      ///< Camera shift
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
    bool output_sh = false;              ///< Output 2-band (4 coeff) spherical harmonics data
    uint8_t max_diff_depth = 4;          ///< Maximum tracing depth of diffuse rays
    uint8_t max_spec_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_refr_depth = 8;          ///< Maximum tracing depth of glossy rays
    uint8_t max_transp_depth = 8;    ///< Maximum tracing depth of transparency rays (note: does not obey total depth)
    uint8_t max_total_depth = 8;     ///< Maximum tracing depth of all rays (except transparency)
    uint8_t min_total_depth = 2;     ///< Depth after which random rays termination starts
    uint8_t min_transp_depth = 2;    ///< Depth after which random rays termination starts
    float clamp_direct = 0.0f;       ///< Clamp direct lighting (0.0 - no clamp)
    float clamp_indirect = 0.0f;     ///< Clamp indirect lighting (0.0 - no clamp)
    int min_samples = 128;           ///< Minimal number of samples will be taken regardless of variance
    float variance_threshold = 0.0f; ///< Variance below which rendering should stop
    float regularize_alpha = 0.03f;  ///< Maximum squared material roughness to apply path regularization
};

/// Atmosphere description
struct atmosphere_params_t {
    float planet_radius = 6371000.0f;                  ///< Planet radius (default is Earth)
    float viewpoint_height = 700.0f;                   ///< Height of the viewpoint to bake environment from
    float atmosphere_height = 100000.0f;               ///< Height of the atmosphere
    float rayleigh_height = atmosphere_height * 0.08f; ///< Rayleigh layer height
    float mie_height = atmosphere_height * 0.012f;     ///< MIE layer height
    float clouds_height_beg = 2000.0f;                 ///< Height where clouds start
    float clouds_height_end = 2500.0f;                 ///< Height where clouds end
    float clouds_variety = 0.5f;                       ///< Clouds variety
    float clouds_density = 0.5f;                       ///< Clouds density
    float clouds_offset_x = 0.0f;                      ///< Clouds offset by x axis
    float clouds_offset_z = 0.0f;                      ///< Clouds offset by z axis
    float cirrus_clouds_amount = 0.5f;                 ///< Amount of the distant clouds
    float cirrus_clouds_height = 6000.0f;              ///< Height of the distant clouds
    float ozone_height_center = 25000.0f;              ///< Height of the ozone layer (center of tent function)
    float ozone_half_width = 15000.0f;                 ///< Half of the width of the ozone layer
    float atmosphere_density = 1.0f;                   ///< Atmosphere density multiplier
    float stars_brightness = 1.0f;                     ///< Brightness of the stars in the sky (set to 0.0 to disable)
    float moon_radius = 1737400.0f;                    ///< Moon radius (set to 0.0 to disable)
    float moon_distance = 100000000.0f;                // 363100000.0f; ///< Distance from Earth to the Moon
    alignas(16) float moon_dir[4] = {0.707f, 0.707f, 0.0f, 0.0f};
    alignas(16) float rayleigh_scattering[4] = {5.802f * 1e-6f, 13.558f * 1e-6f, 33.100f * 1e-6f, 0.0f};
    alignas(16) float mie_scattering[4] = {3.996f * 1e-6f, 3.996f * 1e-6f, 3.996f * 1e-6f, 0.0f};
    alignas(16) float mie_extinction[4] = {4.440f * 1e-6f, 4.440f * 1e-6f, 4.440f * 1e-6f, 0.0f};
    alignas(16) float mie_absorption[4] = {0.444f * 1e-6f, 0.444f * 1e-6f, 0.444f * 1e-6f, 0.0f};
    alignas(16) float ozone_absorbtion[4] = {0.650f * 1e-6f, 1.881f * 1e-6f, 0.085f * 1e-6f, 0.0f};
    alignas(16) float ground_albedo[4] = {0.05f, 0.05f, 0.05f, 0.0f};
};

/// Environment description
struct environment_desc_t {
    float env_col[3] = {};                         ///< Environment color
    TextureHandle env_map = InvalidTextureHandle;  ///< Environment texture
    float back_col[3] = {};                        ///< Background color
    TextureHandle back_map = InvalidTextureHandle; ///< Background texture
    float env_map_rotation = 0.0f;                 ///< Environment map rotation in radians
    float back_map_rotation = 0.0f;                ///< Background map rotation in radians
    int envmap_resolution = 1024;                  ///< Resolution of the generated env texture
    bool importance_sample = true;                 ///< Enable explicit env map sampling
    atmosphere_params_t atmosphere;                ///< Atmosphere parameters
};

class ILog;

using ParallelForFunction = std::function<void(int)>;

inline void parallel_for_serial(const int from, const int to, ParallelForFunction &&f) {
    for (int i = from; i < to; ++i) {
        f(i);
    }
}

/** Base Scene class,
    cpu and gpu backends have different implementation of SceneBase
*/
class SceneBase {
  protected:
    ILog *log_ = nullptr;

  public:
    virtual ~SceneBase() = default;

    /// Log
    ILog *log() const { return log_; }

    /// Get current environment description
    virtual void GetEnvironment(environment_desc_t &env) = 0;

    /// Set environment from description
    virtual void SetEnvironment(const environment_desc_t &env) = 0;

    /** @brief Adds texture to scene
        @param t texture description
        @return New texture handle
    */
    virtual TextureHandle AddTexture(const tex_desc_t &t) = 0;

    /** @brief Removes texture with specific index from scene
        @param t texture handle
    */
    virtual void RemoveTexture(TextureHandle t) = 0;

    /** @brief Adds material to scene
        @param m root shading node description
        @return New material handle
    */
    virtual MaterialHandle AddMaterial(const shading_node_desc_t &m) = 0;

    /** @brief Adds material to scene
        @param m principled material description
        @return New material handle
    */
    virtual MaterialHandle AddMaterial(const principled_mat_desc_t &m) = 0;

    /** @brief Removes material with specific index from scene
        @param m material handle
    */
    virtual void RemoveMaterial(MaterialHandle m) = 0;

    /** @brief Adds mesh to scene
        @param m mesh description
        @return New mesh index
    */
    virtual MeshHandle AddMesh(const mesh_desc_t &m) = 0;

    /** @brief Removes mesh with specific index from scene
        @param i mesh index
    */
    virtual void RemoveMesh(MeshHandle m) = 0;

    /** @brief Adds light to scene
        @param l light description
        @return New light index
    */
    virtual LightHandle AddLight(const directional_light_desc_t &l) = 0;
    virtual LightHandle AddLight(const sphere_light_desc_t &l) = 0;
    virtual LightHandle AddLight(const spot_light_desc_t &l) = 0;
    virtual LightHandle AddLight(const rect_light_desc_t &l, const float *xform) = 0;
    virtual LightHandle AddLight(const disk_light_desc_t &l, const float *xform) = 0;
    virtual LightHandle AddLight(const line_light_desc_t &l, const float *xform) = 0;

    /** @brief Removes light with specific index from scene
        @param l light handle
    */
    virtual void RemoveLight(LightHandle l) = 0;

    /** @brief Adds mesh instance to a scene
        @param mesh mesh handle
        @param xform array of 16 floats holding transformation matrix
        @return New mesh instance handle
    */
    MeshInstanceHandle AddMeshInstance(MeshHandle mesh, const float *xform) {
        mesh_instance_desc_t mi;
        mi.xform = xform;
        mi.mesh = mesh;
        return AddMeshInstance(mi);
    }

    /** @brief Adds mesh instance to a scene
        @param mi mesh instance description
        @return New mesh instance handle
    */
    virtual MeshInstanceHandle AddMeshInstance(const mesh_instance_desc_t &mi) = 0;

    /** @brief Sets mesh instance transformation
        @param mi mesh instance handle
        @param xform array of 16 floats holding transformation matrix
    */
    virtual void SetMeshInstanceTransform(MeshInstanceHandle mi, const float *xform) = 0;

    /** @brief Removes mesh instance from scene
        @param mi mesh instance handle

        Removes mesh instance from scene. Associated mesh remains loaded in scene even if
        there is no instances of this mesh left.
    */
    virtual void RemoveMeshInstance(MeshInstanceHandle mi) = 0;

    virtual void
    Finalize(const std::function<void(int, int, ParallelForFunction &&)> &parallel_for = parallel_for_serial) = 0;

    /** @brief Adds camera to a scene
        @param c camera description
        @return New camera handle
    */
    virtual CameraHandle AddCamera(const camera_desc_t &c) = 0;

    /** @brief Get camera description
        @param i camera handle
    */
    virtual void GetCamera(CameraHandle i, camera_desc_t &c) const = 0;

    /** @brief Sets camera properties
        @param i camera handle
        @param c camera description
    */
    virtual void SetCamera(CameraHandle i, const camera_desc_t &c) = 0;

    /** @brief Removes camera with specific index from scene
        @param i camera handle

        Removes camera with specific index from scene. Other cameras indices remain valid.
    */
    virtual void RemoveCamera(CameraHandle i) = 0;

    /** @brief Get const reference to a camera with specific index
        @return Current camera index
    */
    virtual CameraHandle current_cam() const = 0;

    /** @brief Sets camera with specific index to be current
        @param i camera index
    */
    virtual void set_current_cam(CameraHandle i) = 0;

    /// Overall triangle count in scene
    virtual uint32_t triangle_count() const = 0;

    /// Overall BVH node count in scene
    virtual uint32_t node_count() const = 0;
};
} // namespace Ray
