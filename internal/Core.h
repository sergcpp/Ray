#pragma once

#include <cmath>
#include <cstdint>

#include "../SceneBase.h"
#include "../Span.h"
#include "../Types.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#define assume_aligned(ptr, sz) (__builtin_assume_aligned((const void *)ptr, sz))
#define vectorcall
#define restrict __restrict__
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#define vectorcall __vectorcall
#define assume_aligned(ptr, sz) (__assume((((const char *)ptr) - ((const char *)0)) % (sz) == 0), (ptr))
#define restrict __restrict

#include <intrin.h>

#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_bittestandcomplement)
#pragma intrinsic(_InterlockedExchangeAdd)
#pragma intrinsic(_InterlockedCompareExchange)
#pragma intrinsic(_InterlockedCompareExchange64)

#define Ray_InterlockedExchangeAdd(x, y) _InterlockedExchangeAdd((long *)(x), y)
#define Ray_InterlockedCompareExchange(x, y, z) _InterlockedCompareExchange((long *)(x), y, z)
#define Ray_InterlockedCompareExchange64(x, y, z) _InterlockedCompareExchange64((long long *)(x), y, z)

static_assert(sizeof(long) == 4, "!");
static_assert(sizeof(long long) == 8, "!");

#ifdef _M_IX86
// Win32 doesn't have _BitScanForward64 so emulate it with two 32 bit calls
force_inline unsigned char _BitScanForward64(unsigned long *Index, unsigned __int64 Mask) {
    // Scan the Low Word
    if (_BitScanForward(Index, static_cast<unsigned long>(Mask))) {
        return 1;
    }
    // Scan the High Word
    if (_BitScanForward(Index, static_cast<unsigned long>(Mask >> 32))) {
        *Index += 32; // Create a bit offset from the LSB
        return 1;
    }
    return 0;
}
#endif
#else

#define Ray_InterlockedExchangeAdd __sync_fetch_and_add
#define Ray_InterlockedCompareExchange(dst, exch, comp) __sync_val_compare_and_swap(dst, comp, exch)
#define Ray_InterlockedCompareExchange64(dst, exch, comp) __sync_val_compare_and_swap(dst, comp, exch)

#endif

#define unused(x) ((void)x)
#define countof(x) (sizeof(x) / sizeof(x[0]))

#include "simd/aligned_allocator.h"

#define pack_unorm_16(x) uint16_t(x * 65535.0f)
#define unpack_unorm_16(x) (float(x) / 65535.0f)

namespace Ray {
using uint = uint32_t;
#include "Constants.inl"

enum eUninitialize { Uninitialize };

struct alignas(16) tri_accel_t {
    float n_plane[4];
    float u_plane[4];
    float v_plane[4];
};
static_assert(sizeof(tri_accel_t) == 48, "!");

struct alignas(32) mtri_accel_t {
    float n_plane[4][8];
    float u_plane[4][8];
    float v_plane[4][8];
};
static_assert(sizeof(mtri_accel_t) == 384, "!");

struct bvh_node_t {
    float bbox_min[3];
    union {
        uint32_t prim_index; // First bit is used to identify leaf node
        uint32_t left_child;
    };
    float bbox_max[3];
    union {
        uint32_t prim_count; // First two bits are used for separation axis (0, 1 or 2 - x, y or z)
        uint32_t right_child;
    };
};
static_assert(sizeof(bvh_node_t) == 32, "!");

struct light_bvh_node_t : public bvh_node_t {
    float flux;
    float axis[3];
    float omega_n; // cone angle enclosing light normals
    float omega_e; // emission angle around each normal
};
static_assert(sizeof(light_bvh_node_t) == 56, "!");

struct alignas(32) wbvh_node_t {
    float bbox_min[3][8];
    float bbox_max[3][8];
    uint32_t child[8];
};
static_assert(sizeof(wbvh_node_t) == 224, "!");

struct light_wbvh_node_t : public wbvh_node_t {
    float flux[8];
    uint32_t axis[8];
    uint32_t cos_omega_ne[8];
};
static_assert(sizeof(light_wbvh_node_t) == 320, "!");

struct alignas(16) cwbvh_node_t {
    float bbox_min[3];
    float _unused0;
    float bbox_max[3];
    float _unused1;
    uint8_t ch_bbox_min[3][8];
    uint8_t ch_bbox_max[3][8];
    uint32_t child[8];
};
static_assert(sizeof(cwbvh_node_t) == 112, "!");

struct light_cwbvh_node_t : public cwbvh_node_t {
    float flux[8];
    uint32_t axis[8];
    uint32_t cos_omega_ne[8];
};
static_assert(sizeof(light_cwbvh_node_t) == 208, "!");

struct atlas_texture_t {
    uint16_t width;
    uint16_t height;
    uint32_t atlas;
    uint8_t page[NUM_MIP_LEVELS];
    uint16_t pos[NUM_MIP_LEVELS][2];
};
static_assert(sizeof(atlas_texture_t) == 68, "!");

const uint32_t TEX_SRGB_BIT = (0b00000001u << 24);
const uint32_t TEX_RECONSTRUCT_Z_BIT = (0b00000010u << 24);
const uint32_t TEX_YCOCG_BIT = (0b00000100u << 24);

struct tri_mat_data_t {
    uint16_t front_mi, back_mi;
};

struct material_t {
    uint32_t textures[MAX_MATERIAL_TEXTURES];
    float base_color[3];
    uint32_t flags;
    eShadingNode type;
    union {
        float tangent_rotation;
        float strength;
    };
    uint16_t roughness_unorm;
    uint16_t anisotropic_unorm;
    float ior;
    uint16_t sheen_unorm;
    uint16_t sheen_tint_unorm;
    uint16_t tint_unorm;
    uint16_t metallic_unorm;
    uint16_t transmission_unorm;
    uint16_t transmission_roughness_unorm;
    uint16_t specular_unorm;
    uint16_t specular_tint_unorm;
    uint16_t clearcoat_unorm;
    uint16_t clearcoat_roughness_unorm;
    uint16_t normal_map_strength_unorm;
    uint16_t _pad;
};
static_assert(sizeof(material_t) == 76, "!");

struct light_t {
    uint32_t type : 3;
    uint32_t doublesided : 1;
    uint32_t cast_shadow : 1;
    uint32_t visible : 1;
    uint32_t sky_portal : 1;
    uint32_t ray_visibility : 8;
    uint32_t _unused0 : 17;
    float col[3];
    union {
        struct {
            float pos[3], area;
            float dir[3], radius;
            float spot, blend, _unused[2];
        } sph;
        struct {
            float pos[3], area;
            float u[3], _unused0;
            float v[3], _unused1;
        } rect;
        struct {
            float pos[3], area;
            float u[3], _unused0;
            float v[3], _unused1;
        } disk;
        struct {
            float pos[3], area;
            float u[3], radius;
            float v[3], height;
        } line;
        struct {
            uint32_t tri_index;
            uint32_t mi_index;
            uint32_t tex_index;
            float _unused[9];
        } tri;
        struct {
            float dir[3], angle;
            float _unused[8];
        } dir;
    };
};
static_assert(sizeof(light_t) == 64, "!");

struct prim_t;

struct bvh_settings_t {
    float oversplit_threshold = 0.95f;
    bool allow_spatial_splits = false;
    bool use_fast_bvh_build = false;
    int min_primitives_in_leaf = 8;
};

template <typename T, size_t Alignment = alignof(T)>
using aligned_vector = std::vector<T, aligned_allocator<T, Alignment>>;

// bit scan forward
force_inline long GetFirstBit(long mask) {
#ifdef _MSC_VER
    unsigned long ret;
    _BitScanForward(&ret, (unsigned long)mask);
    return long(ret);
#else
    return long(__builtin_ffsl(mask) - 1);
#endif
}

force_inline bool GetFirstBit(const uint64_t mask, unsigned long *bit_index) {
#ifdef _MSC_VER
    return _BitScanForward64(bit_index, mask);
#else
    const int ret = __builtin_ffsll((long long)mask);
    (*bit_index) = ret - 1;
    return ret != 0;
#endif
}

force_inline int CountTrailingZeroes(const uint64_t mask) {
#ifdef _MSC_VER
    // return int(_tzcnt_u64(mask));
    if (mask == 0) {
        return 64;
    }
    unsigned long r = 0;
    _BitScanForward64(&r, mask);
    return r;
#else
    return (mask == 0) ? 64 : __builtin_ctzll(mask);
#endif
}

// bit test and complement
force_inline long ClearBit(long mask, long index) {
#ifdef _MSC_VER
    _bittestandcomplement(&mask, index);
    return mask;
#else
    return (mask & ~(1 << index));
#endif
}

force_inline int popcount(unsigned x) {
    int c = 0;
    for (; x != 0; x &= x - 1) {
        c++;
    }
    return c;
}

// Creates struct of precomputed triangle data for faster Plucker intersection test
bool PreprocessTri(const float *p, int stride, tri_accel_t *out_acc);

// Builds BVH for mesh and precomputes triangle data
uint32_t PreprocessMesh(const vtx_attribute_t &positions, Span<const uint32_t> vtx_indices, int base_vertex,
                        const bvh_settings_t &s, std::vector<bvh_node_t> &out_nodes,
                        aligned_vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_indices,
                        aligned_vector<mtri_accel_t> &out_tris2);

// Recursively builds linear bvh for a set of primitives
uint32_t EmitLBVH_r(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes, uint32_t prim_index,
                    uint32_t prim_count, uint32_t index_offset, int bit_index, std::vector<bvh_node_t> &out_nodes);
// Iteratively builds linear bvh for a set of primitives
uint32_t EmitLBVH(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes, uint32_t prim_index,
                  uint32_t prim_count, uint32_t index_offset, int bit_index, std::vector<bvh_node_t> &out_nodes);

// Builds SAH-based BVH for a set of primitives, slow
uint32_t PreprocessPrims_SAH(Span<const prim_t> prims, const vtx_attribute_t &positions, const bvh_settings_t &s,
                             std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices);

// Builds linear BVH for a set of primitives, fast
uint32_t PreprocessPrims_HLBVH(Span<const prim_t> prims, std::vector<bvh_node_t> &out_nodes,
                               std::vector<uint32_t> &out_indices);

uint32_t FlattenBVH_r(const bvh_node_t *nodes, uint32_t node_index, uint32_t parent_index,
                      aligned_vector<wbvh_node_t> &out_nodes);
uint32_t FlattenLightBVH_r(const light_bvh_node_t *nodes, uint32_t node_index, uint32_t parent_index,
                           aligned_vector<light_wbvh_node_t> &out_nodes);
uint32_t FlattenLightBVH_r(const light_bvh_node_t *nodes, uint32_t node_index, uint32_t parent_index,
                           aligned_vector<light_cwbvh_node_t> &out_nodes);

bool NaiivePluckerTest(const float p[9], const float o[3], const float d[3]);

const int FILTER_TABLE_SIZE = 1024;

inline float filter_box(float /*v*/, float /*width*/) { return 1.0f; }
inline float filter_gaussian(float v, float width) {
    v *= 6.0f / width;
    return expf(-2.0f * v * v);
}
inline float filter_blackman_harris(float v, float width) {
    v = 2.0f * PI * (v / width + 0.5f);
    return 0.35875f - 0.48829f * cosf(v) + 0.14128f * cosf(2.0f * v) - 0.01168f * cosf(3.0f * v);
}

void ConstructCamera(eCamType type, ePixelFilter filter, float filter_width, eViewTransform view_transform,
                     const float origin[3], const float fwd[3], const float up[3], const float shift[2], float fov,
                     float sensor_height, float exposure, float gamma, float focus_distance, float fstop,
                     float lens_rotation, float lens_ratio, int lens_blades, float clip_start, float clip_end,
                     camera_t *cam);

// Applies 4x4 matrix matrix transform to bounding box
void TransformBoundingBox(const float bbox_min[3], const float bbox_max[3], const float *xform, float out_bbox_min[3],
                          float out_bbox_max[3]);

void InverseMatrix(const float mat[16], float out_mat[16]);

extern const int __pmj02_sample_count;
extern const int __pmj02_dims_count;
extern const uint32_t __pmj02_samples[];

const int RAND_SAMPLES_COUNT = __pmj02_sample_count;
const int RAND_DIMS_COUNT = __pmj02_dims_count;

struct vertex_t {
    float p[3], n[3], b[3], t[2];
};
static_assert(sizeof(vertex_t) == 44, "!");

struct mesh_t {
    float bbox_min[3], bbox_max[3];
    uint32_t node_index, node_block;
    uint32_t tris_index, tris_block, tris_count;
    uint32_t vert_index, vert_block, vert_count;
    uint32_t vert_data_index, vert_data_block;
};
static_assert(sizeof(mesh_t) == 64, "!");

struct mesh_instance_t {
    float bbox_min[3];
    uint32_t _unused;
    float bbox_max[3];
    uint32_t mesh_index;
    uint32_t _unused2;
    uint32_t mesh_block;
    uint32_t lights_index;
    uint32_t ray_visibility; // upper 24 bits identify lights_block
    float xform[16], inv_xform[16];
};
static_assert(sizeof(mesh_instance_t) == 176, "!");

struct environment_t {
    float env_col[3];
    uint32_t env_map;
    float back_col[3];
    uint32_t back_map;
    float env_map_rotation;
    float back_map_rotation;
    const float *qtree_mips[16];
    int qtree_levels;
    bool importance_sample;
    uint32_t light_index;
    uint32_t env_map_res;  // 16-bit
    uint32_t back_map_res; // 16-bit
    float sky_map_spread_angle;
    int envmap_resolution;
    atmosphere_params_t atmosphere;
};

force_inline float to_norm_float(uint8_t v) {
    uint32_t val = 0x3f800000 + v * 0x8080 + (v + 1) / 2;
    union {
        uint32_t i;
        float f;
    } ret = {val};
    return ret.f - 1.0f;
}

force_inline void rgbe_to_rgb(const uint8_t rgbe[4], float out_rgb[3]) {
    const float f = std::exp2(float(rgbe[3]) - 128.0f);
    out_rgb[0] = to_norm_float(rgbe[0]) * f;
    out_rgb[1] = to_norm_float(rgbe[1]) * f;
    out_rgb[2] = to_norm_float(rgbe[2]) * f;
}

void CanonicalToDir(const float p[2], float y_rotation, float out_d[3]);
void DirToCanonical(const float d[3], float y_rotation, float out_p[2]);

uint32_t EncodeOctDir(const float d[3]);

extern const uint8_t morton_table_16[];
extern const int morton_table_256[];

extern const float omega_step;
extern const char omega_table[];

extern const float phi_step;
extern const char phi_table[][17];

struct ray_chunk_t {
    uint32_t hash, base, size;
};

struct ray_hash_t {
    uint32_t hash, index;
};

enum class eActivation { ReLU };
enum class ePostOp { None, Downscale, HDRTransfer, PositiveNormalize };
enum class ePreOp { None, Upscale, HDRTransfer, PositiveNormalize };

/*struct pass_info_t {
    int iteration, bounce;
    pass_settings_t settings;

    force_inline bool should_add_direct_light() const {
        // skip for primary bounce if we want only indirect light contribution
        return !(settings.flags & SkipDirectLight) || bounce > 2;
    }

    force_inline bool should_add_environment() const { return !(settings.flags & NoBackground) || bounce > 2; }

    force_inline bool should_consider_albedo() const {
        // do not use albedo in lightmap mode for primary bounce
        return !(settings.flags & LightingOnly) || bounce > 2;
    }

    force_inline bool use_uniform_sampling() const {
        // do not use diffuse-specific sampling
        return ((settings.flags & OutputSH) && bounce <= 2);
    }
};
static_assert(sizeof(pass_info_t) == 20, "!");*/

struct cache_voxel_t {
    float radiance[3] = {};
    uint32_t sample_count = 0;
    uint32_t frame_count = 0;
};

struct packed_cache_voxel_t {
    uint32_t v[4] = {};
};

inline cache_voxel_t unpack_voxel_data(const packed_cache_voxel_t &v) {
    cache_voxel_t ret;
    ret.radiance[0] = float(v.v[0]) / RAD_CACHE_RADIANCE_SCALE;
    ret.radiance[1] = float(v.v[1]) / RAD_CACHE_RADIANCE_SCALE;
    ret.radiance[2] = float(v.v[2]) / RAD_CACHE_RADIANCE_SCALE;
    ret.sample_count = (v.v[3] >> 0) & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;
    ret.frame_count = (v.v[3] >> RAD_CACHE_SAMPLE_COUNTER_BIT_NUM) & RAD_CACHE_FRAME_COUNTER_BIT_MASK;
    return ret;
}

struct cache_grid_params_t {
    float cam_pos_curr[3] = {0.0f, 0.0f, 0.0f}, cam_pos_prev[3] = {0.0f, 0.0f, 0.0f};
    float log_base = RAD_CACHE_GRID_LOGARITHM_BASE;
    float scale = RAD_CACHE_GRID_SCALE;
    float exposure = 1.0f;
};

struct cache_data_t {
    uint32_t cache_entries[RAD_CACHE_PROPAGATION_DEPTH];
    float sample_weight[RAD_CACHE_PROPAGATION_DEPTH][3];
    int path_len;
};

enum eSpatialCacheMode { None, Update, Query };

struct scene_data_t {
    const environment_t &env;
    const mesh_instance_t *mesh_instances;
    const uint32_t *mi_indices;
    const mesh_t *meshes;
    const uint32_t *vtx_indices;
    const vertex_t *vertices;
    const bvh_node_t *nodes;
    const wbvh_node_t *wnodes;
    const tri_accel_t *tris;
    const uint32_t *tri_indices;
    const mtri_accel_t *mtris;
    const tri_mat_data_t *tri_materials;
    const material_t *materials;
    Span<const light_t> lights;
    Span<const uint32_t> li_indices;
    Span<const uint32_t> dir_lights;
    uint32_t visible_lights_count;
    uint32_t blocker_lights_count;
    Span<const light_bvh_node_t> light_nodes;
    Span<const light_cwbvh_node_t> light_cwnodes;
    Span<const float> sky_transmittance_lut, sky_multiscatter_lut;
    const cache_grid_params_t &spatial_cache_grid;
    Span<const uint64_t> spatial_cache_entries;
    Span<const packed_cache_voxel_t> spatial_cache_voxels;
};

force_inline float clamp(const float val, const float min, const float max) {
    return val < min ? min : (val > max ? max : val);
}
force_inline float saturate(const float val) { return clamp(val, 0.0f, 1.0f); }

force_inline float sqr(const float x) { return x * x; }

force_inline float mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }

force_inline float log_base(const float x, const float base) { return logf(x) / logf(base); }

force_inline float linstep(const float smin, const float smax, const float x) {
    return saturate((x - smin) / (smax - smin));
}

force_inline float smoothstep(const float edge0, const float edge1, const float x) {
    const float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

force_inline float from_unit_to_sub_uvs(const float u, const float resolution) {
    return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f));
}
force_inline float from_sub_uvs_to_unit(const float u, const float resolution) {
    return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f));
}

force_inline float calc_voxel_size(const uint32_t grid_level, const cache_grid_params_t &params) {
    return powf(params.log_base, float(grid_level)) / (params.scale * powf(params.log_base, HASH_GRID_LEVEL_BIAS));
}

template <typename T> void rect_fill(Span<T> data, const int stride, const rect_t &rect, T &&val) {
    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            data[y * stride + x] = val;
        }
    }
}

} // namespace Ray
