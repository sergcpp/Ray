#pragma once

#include <cstdint>

#include "../SceneBase.h"
#include "../Types.h"

#ifdef __GNUC__
#if !defined(__ANDROID__) && !defined(__EMSCRIPTEN__)
#define force_inline __attribute__((always_inline))
#else
#define force_inline __inline__
#endif
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

namespace ray {
enum eUninitialize { Uninitialize };

union rays_id_t {
    uint32_t id;
    struct {
        uint16_t x, y;
    };
};

struct tri_accel_t {
    float nu, nv;
    float np;
    float pu, pv;
    int32_t ci;
    float e0u, e0v;
    float e1u, e1v;
    uint32_t mi;
    uint32_t pad;
};
static_assert(sizeof(tri_accel_t) == 48, "!");

const uint8_t W_BITS = 0b00000011;
const uint8_t AXIS_ALIGNED_BIT = 0b00000100;

struct cone_accel_t {
    float o[3], v[3];
    float cos_phi_sqr;
    float cone_start, cone_end;
};
static_assert(sizeof(cone_accel_t) == 36, "!");

struct aabox_t {
    float min[3], max[3];
};
static_assert(sizeof(aabox_t) == 24, "!");

struct bvh_node_t {
    uint32_t prim_index, prim_count,
             left_child, right_child, parent, sibling,
             space_axis; // axis with maximal child's centroids distance
    float bbox[2][3];
};
static_assert(sizeof(bvh_node_t) == 52, "!");

const int MAX_MIP_LEVEL = 11;
const int NUM_MIP_LEVELS = MAX_MIP_LEVEL + 1;
const int MAX_TEXTURE_SIZE = (1 << NUM_MIP_LEVELS);

struct texture_t {
    uint16_t size[2];
    uint8_t page[NUM_MIP_LEVELS];
    uint16_t pos[NUM_MIP_LEVELS][2];
};
static_assert(sizeof(texture_t) == 64, "!");

const int MAX_MATERIAL_TEXTURES = 5;

const int NORMALS_TEXTURE = 0;
const int MAIN_TEXTURE = 1;

const int MIX_MAT1 = 2;
const int MIX_MAT2 = 3;

struct material_t {
    uint32_t type;
    uint32_t textures[MAX_MATERIAL_TEXTURES];
    union {
        float roughness;
        float strength;
    };
    union {
        float fresnel;
        float ior;
    };
};
static_assert(sizeof(material_t) == 32, "!");

struct prim_t;

void PreprocessTri(const float *p, int stride, tri_accel_t *acc);
void PreprocessCone(const float o[3], const float v[3], float phi, float cone_start, float cone_end, cone_accel_t *acc);
void PreprocessBox(const float min[3], const float max[3], aabox_t *box);

uint32_t PreprocessMesh(const float *attrs, size_t attrs_count, const uint32_t *indices, size_t indices_count, eVertexLayout layout,
                        std::vector<bvh_node_t> &out_nodes, std::vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_indices);

uint32_t PreprocessPrims(const prim_t *prims, size_t prims_count,
                         std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices);

bool NaiivePluckerTest(const float p[9], const float o[3], const float d[3]);

void ConstructCamera(eCamType type, const float origin[3], const float fwd[3], float fov, camera_t *cam);

void TransformBoundingBox(const float bbox[2][3], const float *xform, float out_bbox[2][3]);

const int PrimesCount = 11;
const int g_primes[PrimesCount] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 };
const int g_prime_sums[PrimesCount] = { 0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129 };

const int HaltonSeqLen = 512;

struct vertex_t {
    float p[3], n[3], b[3], t0[2];
};
static_assert(sizeof(vertex_t) == 44, "!");

struct mesh_t {
    uint32_t node_index, node_count;
};
static_assert(sizeof(mesh_t) == 8, "!");

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
}
