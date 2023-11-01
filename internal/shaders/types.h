#ifndef TYPES_H
#define TYPES_H

#include "_interface_common.h"

INTERFACE_START(Types)

#include "constants.h"

const int RAND_SAMPLES_COUNT = 4096;
const int RAND_DIMS_COUNT = 32;

const int MAX_STACK_SIZE = 48;

const UINT_TYPE LEAF_NODE_BIT = (1u << 31);
const UINT_TYPE PRIM_INDEX_BITS = ~LEAF_NODE_BIT;
const UINT_TYPE LEFT_CHILD_BITS = ~LEAF_NODE_BIT;

const UINT_TYPE SEP_AXIS_BITS = (3u << 30); // 0b11u
const UINT_TYPE PRIM_COUNT_BITS = ~SEP_AXIS_BITS;
const UINT_TYPE RIGHT_CHILD_BITS = ~SEP_AXIS_BITS;

const int MAX_MATERIAL_TEXTURES = 5;

const int NORMALS_TEXTURE = 0;
const int BASE_TEXTURE = 1;
const int ROUGH_TEXTURE = 2;
const int METALLIC_TEXTURE = 3;
const int SPECULAR_TEXTURE = 4;

const int MIX_MAT1 = 3;
const int MIX_MAT2 = 4;

const int MATERIAL_SOLID_BIT = 32768; // 0b1000000000000000
const int MATERIAL_INDEX_BITS = 16383; // 0b0011111111111111

#define MAX_DIST 3.402823466e+30

#define HIT_EPS 0.000001

#define FLT_EPS 0.0000001
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#ifndef HIT_BIAS
#define HIT_BIAS 0.00001
#endif
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38
#endif
#ifndef PI
#define PI 3.141592653589793238463
#endif

#define SPHERICAL_AREA_THRESHOLD 0.00005 // non-zero minimal spherical area to sample using spherical method (avoids precision issues)

const int DiffuseNode = 0;
const int GlossyNode = 1;
const int RefractiveNode = 2;
const int EmissiveNode = 3;
const int MixNode = 4;
const int TransparentNode = 5;
const int PrincipledNode = 6;

const UINT_TYPE MAT_FLAG_MULT_IMPORTANCE = (1u << 0u);
const UINT_TYPE MAT_FLAG_MIX_ADD = (1u << 1u);

const int NUM_MIP_LEVELS = 12;
const int MAX_MIP_LEVEL = NUM_MIP_LEVELS - 1;
const int MIN_ATLAS_TEXTURE_SIZE = 4;
const int MAX_ATLAS_TEXTURE_SIZE = (MIN_ATLAS_TEXTURE_SIZE << MAX_MIP_LEVEL);

const int TEXTURE_ATLAS_SIZE = 8192;

const int ATLAS_TEX_SRGB_BIT            = 32768; // 0b1000000000000000
const int ATLAS_TEX_RECONSTRUCT_Z_BIT   = 16384; // 0b0100000000000000
const int ATLAS_TEX_WIDTH_BITS          = 16383; // 0b0011111111111111
const int ATLAS_TEX_MIPS_BIT            = 32768; // 0b1000000000000000
const int ATLAS_TEX_YCOCG_BIT           = 16384; // 0b0100000000000000
const int ATLAS_TEX_HEIGHT_BITS         = 16383; // 0b0011111111111111

const UINT_TYPE TEX_SRGB_BIT          = (1u << 24); // 0b00000001
const UINT_TYPE TEX_RECONSTRUCT_Z_BIT = (2u << 24); // 0b00000010
const UINT_TYPE TEX_YCOCG_BIT         = (4u << 24); // 0b00000100

const UINT_TYPE TEXTURES_SAMPLER_SLOT = 20;
const UINT_TYPE TEXTURES_SIZE_SLOT = 21;
const UINT_TYPE TEXTURES_BUF_SLOT = 22;
const UINT_TYPE TEXTURE_ATLASES_SLOT = 23;

const int FILTER_BOX = 0;
const int FILTER_GAUSSIAN = 1;
const int FILTER_BLACKMAN_HARRIS = 2;

const int FILTER_TABLE_SIZE = 1024;

struct ray_data_t {
	float o[3], d[3], pdf;
	float c[3];
    float ior[4];
	float cone_width, cone_spread;
	UINT_TYPE xy;
	UINT_TYPE depth;
};

struct shadow_ray_t {
    // origin
    float o[3];
    // four 8-bit ray depth counters
    UINT_TYPE depth;
    // direction and distance
    float d[3], dist;
    // throughput color of ray
    float c[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    UINT_TYPE xy;
};

struct tri_accel_t {
    VEC4_TYPE n_plane;
    VEC4_TYPE u_plane;
    VEC4_TYPE v_plane;
};

struct hit_data_t {
    int mask;
    int obj_index;
    int prim_index;
    float t, u, v;
};

struct bvh_node_t {
    VEC4_TYPE bbox_min; // w is prim_index/left_child
    VEC4_TYPE bbox_max; // w is prim_count/right_child
};

struct light_bvh_node_t {
    float bbox_min[3];
    UINT_TYPE left_child;
    float bbox_max[3];
    UINT_TYPE right_child;
    float flux;
    float axis[3];
    float omega_n; // cone angle enclosing light normals
    float omega_e; // emission angle around each normal
};

struct light_wbvh_node_t {
    float bbox_min[3][8];
    float bbox_max[3][8];
    UINT_TYPE child[8];
    float flux[8];
    float axis[3][8];
    float omega_n[8];
    float omega_e[8];
};

struct vertex_t {
    float p[3], n[3], b[3], t[2];
};

struct mesh_t {
    float bbox_min[3], bbox_max[3];
    UINT_TYPE node_index, node_block;
    UINT_TYPE tris_index, tris_block, tris_count;
    UINT_TYPE vert_index, vert_block, vert_count;
    UINT_TYPE vert_data_index, vert_data_block;
};

struct transform_t {
    MAT4_TYPE xform, inv_xform;
};

struct mesh_instance_t {
    VEC4_TYPE bbox_min; // w is tr_index
    VEC4_TYPE bbox_max; // w is mesh_index
    UVEC4_TYPE block_ndx; // xy - indexes of transform and mesh blocks, z - lights index, w - ray_visibility
};

struct light_t {
    UVEC4_TYPE type_and_param0;
    VEC4_TYPE param1;
    VEC4_TYPE param2;
    VEC4_TYPE param3;
};

#define SPH_POS param1.xyz
#define SPH_AREA param1.w
#define SPH_DIR param2.xyz
#define SPH_RADIUS param2.w
#define SPH_SPOT param3.x
#define SPH_BLEND param3.y

#define RECT_POS param1.xyz
#define RECT_AREA param1.w
#define RECT_U param2.xyz
#define RECT_V param3.xyz

#define DISK_POS param1.xyz
#define DISK_AREA param1.w
#define DISK_U param2.xyz
#define DISK_V param3.xyz

#define LINE_POS param1.xyz
#define LINE_AREA param1.w
#define LINE_U param2.xyz
#define LINE_RADIUS param2.w
#define LINE_V param3.xyz
#define LINE_HEIGHT param3.w

#define TRI_TRI_INDEX param1.x
#define TRI_XFORM_INDEX param1.y
#define TRI_TEX_INDEX param1.z

#define DIR_DIR param1.xyz
#define DIR_ANGLE param1.w

struct material_t {
    UINT_TYPE textures[MAX_MATERIAL_TEXTURES];
    float base_color[3];
    UINT_TYPE flags;
    UINT_TYPE type;
    float tangent_rotation_or_strength;
    UINT_TYPE roughness_and_anisotropic;
    float ior;
    UINT_TYPE sheen_and_sheen_tint;
    UINT_TYPE tint_and_metallic;
    UINT_TYPE transmission_and_transmission_roughness;
    UINT_TYPE specular_and_specular_tint;
    UINT_TYPE clearcoat_and_clearcoat_roughness;
    UINT_TYPE normal_map_strength_unorm;
};

struct atlas_texture_t {
    UINT_TYPE size;
    UINT_TYPE atlas;
    UINT_TYPE page[(NUM_MIP_LEVELS + 3) / 4];
    UINT_TYPE pos[NUM_MIP_LEVELS];
};

struct ray_chunk_t {
    UINT_TYPE hash, base, size;
};

struct ray_hash_t {
    UINT_TYPE hash, index;
};

INTERFACE_END

#endif // TYPES_H