#pragma once

#include <cstdint>

#include "../SceneBase.h"
#include "../Types.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#define unused(x) ((void)x)

#include "simd/aligned_allocator.h"

namespace Ray {
enum eUninitialize { Uninitialize };

struct tri_accel_t {
    float nu, nv;
    float np;
    float pu, pv;
    int32_t ci;
    float e0u, e0v;
    float e1u, e1v;
    uint32_t mi, back_mi;
};
static_assert(sizeof(tri_accel_t) == 48, "!");

const uint8_t TRI_W_BITS = 0b00000011;
const uint8_t TRI_AXIS_ALIGNED_BIT = 0b00000100;
const uint8_t TRI_INV_NORMAL_BIT = 0b00001000;
const uint8_t TRI_SOLID_BIT = 0b00010000;

const float HIT_BIAS = 0.001f;
const float HIT_EPS = 0.000001f;
const float FLT_EPS = 0.0000001f;

const float PI = 3.141592653589793238463f;

const float MAX_DIST = 3.402823466e+38F;

const int MAX_BOUNCES = 10;

const float RAY_TERM_THRES = 0.01f;

const float LIGHT_ATTEN_CUTOFF = 0.001f;

const uint32_t LEAF_NODE_BIT   = (1u << 31);
const uint32_t PRIM_INDEX_BITS = ~LEAF_NODE_BIT;
const uint32_t LEFT_CHILD_BITS = ~LEAF_NODE_BIT;

const uint32_t SEP_AXIS_BITS    = (0b11u << 30);
const uint32_t PRIM_COUNT_BITS  = ~SEP_AXIS_BITS;
const uint32_t RIGHT_CHILD_BITS = ~SEP_AXIS_BITS;

struct bvh_node_t {
    float bbox_min[3];
    union {
        uint32_t prim_index;    // First bit is used to identify leaf node
        uint32_t left_child;
    };
    float bbox_max[3];
    union {
        uint32_t prim_count;    // First two bits are used for separation axis (0, 1 or 2 - x, y or z)
        uint32_t right_child;
    };
#ifdef USE_STACKLESS_BVH_TRAVERSAL
    uint32_t parent;
#endif
};

#ifdef USE_STACKLESS_BVH_TRAVERSAL
static_assert(sizeof(bvh_node_t) == 36, "!");
#else
static_assert(sizeof(bvh_node_t) == 32, "!");
#endif

const int MAX_MIP_LEVEL = 11;
const int NUM_MIP_LEVELS = MAX_MIP_LEVEL + 1;
const int MAX_TEXTURE_SIZE = (1 << NUM_MIP_LEVELS);

struct texture_t {
    uint16_t size[2];
    uint8_t page[NUM_MIP_LEVELS];
    uint16_t pos[NUM_MIP_LEVELS][2];
};
static_assert(sizeof(texture_t) == 64, "!");

const int MAX_MATERIAL_TEXTURES = 4;

const int NORMALS_TEXTURE = 0;
const int MAIN_TEXTURE = 1;

const int MIX_MAT1 = 2;
const int MIX_MAT2 = 3;

const int MAX_STACK_SIZE = 32;

struct material_t {
    uint32_t textures[MAX_MATERIAL_TEXTURES];
    float main_color[3], pad;
    uint32_t type;
    union {
        float roughness;
        float strength;
    };
    float int_ior, ext_ior;
};
static_assert(sizeof(material_t) == 48, "!");

struct light_t {
    float pos[3], radius;
    float col[3], brightness;
    float dir[3], spot;
};
static_assert(sizeof(light_t) == 48, "!");

struct prim_t;
struct split_settings_t;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, alignof(T)>>;

// Creates struct of precomputed triangle data for faster Plucker intersection test
bool PreprocessTri(const float *p, int stride, tri_accel_t *acc);
// Extructs planar triangle normal from precomputed triangle data
void ExtractPlaneNormal(const tri_accel_t &tri, float *out_normal);

// Builds BVH for mesh and precomputes triangle data
uint32_t PreprocessMesh(const float *attrs, const uint32_t *indices, size_t indices_count, eVertexLayout layout, int base_vertex,
                        bool allow_spatial_splits, bool use_fast_bvh_build,
                        std::vector<bvh_node_t> &out_nodes, std::vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_indices);

// Recursively builds linear bvh for a set of primitives
uint32_t EmitLBVH_Recursive(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes, uint32_t prim_index, uint32_t prim_count, uint32_t index_offset, int bit_index, std::vector<bvh_node_t> &out_nodes);
// Iteratively builds linear bvh for a set of primitives
uint32_t EmitLBVH_NonRecursive(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes, uint32_t prim_index, uint32_t prim_count, uint32_t index_offset, int bit_index, std::vector<bvh_node_t> &out_nodes);

// Builds SAH-based BVH for a set of primitives, slow
uint32_t PreprocessPrims_SAH(const prim_t *prims, size_t prims_count, const float *positions, size_t stride,
                             const split_settings_t &s, std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices);

// Builds linear BVH for a set of primitives, fast
uint32_t PreprocessPrims_HLBVH(const prim_t *prims, size_t prims_count, std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices);

bool NaiivePluckerTest(const float p[9], const float o[3], const float d[3]);

void ConstructCamera(eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor, camera_t *cam);

// Applies 4x4 matrix matrix transform to bounding box
void TransformBoundingBox(const float bbox_min[3], const float bbox_max[3], const float *xform, float out_bbox_min[3], float out_bbox_max[3]);

void InverseMatrix(const float mat[16], float out_mat[16]);

// Arrays of prime numbers, used to generate halton sequence for sampling
const int PrimesCount = 24;
const int g_primes[PrimesCount] =     { 2, 3, 5, 7,  11, 13, 17, 19, 23, 29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89  };
const int g_prime_sums[PrimesCount] = { 0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639, 712, 791, 874 };

const int HALTON_COUNT = PrimesCount;
const int HALTON_2D_COUNT = PrimesCount / 2;
const int HALTON_SEQ_LEN = 256;

// Sampling stages must be independent from each other (otherwise it may lead to artifacts), so different halton sequences at each ray bounce must be used.
// This leads to limited number of bounces. Can be easily fixed by generating huge primes table above, but i do not need it for now.
static_assert(MAX_BOUNCES + 2 <= HALTON_2D_COUNT, "!");

struct vertex_t {
    float p[3], n[3], b[3], t[2][2];
};
static_assert(sizeof(vertex_t) == 52, "!");

struct mesh_t {
    uint32_t node_index, node_count;
    uint32_t tris_index, tris_count;
};
static_assert(sizeof(mesh_t) == 16, "!");

struct transform_t {
    float xform[16], inv_xform[16];
};
static_assert(sizeof(transform_t) == 128, "!");

struct mesh_instance_t {
    float bbox_min[3];
    uint32_t tr_index;
    float bbox_max[3];
    uint32_t mesh_index;
};
static_assert(sizeof(mesh_instance_t) == 32, "!");

struct environment_t {
    float env_col[3];
    float env_clamp;
    uint32_t env_map;
};

extern const float uint8_to_float_table[];

force_inline float to_norm_float(uint8_t v) {
    return uint8_to_float_table[v];
}

extern const uint8_t morton_table_16[];
extern const int morton_table_256[];

extern const float omega_step;
extern const char omega_table[];

extern const float phi_step;
extern const char phi_table[][17];

extern const int sampling_pattern[];

struct ray_chunk_t {
    uint32_t hash, base, size;
};

struct pass_info_t {
    int index, rand_index;
    int iteration, bounce;
    pass_settings_t settings;

    force_inline bool should_add_direct_light() const {
        // skip for primary bounce if we want only indirect light contribution
        return !(settings.flags & SkipDirectLight) || bounce > 2;
    }

    force_inline bool should_add_environment() const {
        return !(settings.flags & NoBackground) || bounce > 2;
    }

    force_inline bool should_consider_albedo() const {
        // do not use albedo in lightmap mode for primary bounce
        return !(settings.flags & LightingOnly) || bounce > 2;
    }

    force_inline bool use_uniform_sampling() const {
        // do not use diffuse-specific sampling
        return ((settings.flags & OutputSH) && bounce <= 2);
    }
};
static_assert(sizeof(pass_info_t) == 28, "!");

struct scene_data_t {
    const environment_t *env;
    const mesh_instance_t *mesh_instances;
    const uint32_t *mi_indices;
    const mesh_t *meshes;
    const transform_t *transforms;
    const uint32_t *vtx_indices;
    const vertex_t *vertices;
    const bvh_node_t *nodes;
    const tri_accel_t *tris;
    const uint32_t *tri_indices;
    const material_t *materials;
    const texture_t *textures;
    const light_t *lights;
    const uint32_t *li_indices;
};

}
