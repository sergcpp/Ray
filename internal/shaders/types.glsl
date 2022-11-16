#ifndef TYPES_GLSL
#define TYPES_GLSL

#include "_interface_common.glsl"

INTERFACE_START(Types)

const int RAND_DIM_FILTER_U = 0;
const int RAND_DIM_FILTER_V = 1;
const int RAND_DIM_LENS_U = 2;
const int RAND_DIM_LENS_V = 3;
const int RAND_DIM_BASE_COUNT = 4; // independent from bounce count

const int RAND_DIM_BSDF_PICK = 0;
const int RAND_DIM_BSDF_U = 1;
const int RAND_DIM_BSDF_V = 2;
const int RAND_DIM_LIGHT_PICK = 3;
const int RAND_DIM_LIGHT_U = 4;
const int RAND_DIM_LIGHT_V = 5;
const int RAND_DIM_TERMINATE = 6;
const int RAND_DIM_BOUNCE_COUNT = 7; // separate for each bounce

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
#define HIT_BIAS 0.00001f
#endif
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38f
#endif
#ifndef PI
#define PI 3.141592653589793238463f
#endif

const int LIGHT_TYPE_SPHERE = 0;
const int LIGHT_TYPE_SPOT = 1;
const int LIGHT_TYPE_DIR = 2;
const int LIGHT_TYPE_LINE = 3;
const int LIGHT_TYPE_RECT = 4;
const int LIGHT_TYPE_DISK = 5;
const int LIGHT_TYPE_TRI = 6;

const int DiffuseNode = 0;
const int GlossyNode = 1;
const int RefractiveNode = 2;
const int EmissiveNode = 3;
const int MixNode = 4;
const int TransparentNode = 5;
const int PrincipledNode = 6;

const UINT_TYPE MAT_FLAG_MULT_IMPORTANCE = (1u << 0u);
const UINT_TYPE MAT_FLAG_MIX_ADD = (1u << 1u);
const UINT_TYPE MAT_FLAG_SKY_PORTAL = (1u << 2u);

const int NUM_MIP_LEVELS = 14;
const int MAX_MIP_LEVEL = NUM_MIP_LEVELS - 1;
const int MAX_TEXTURE_SIZE = (1 << MAX_MIP_LEVEL);

const int TEXTURE_ATLAS_SIZE = 8192 + 256; // small margin to account for borders

const int ATLAS_TEX_SRGB_BIT            = 32768; // 0b1000000000000000
const int ATLAS_TEX_RECONSTRUCT_Z_BIT   = 16384; // 0b0100000000000000
const int ATLAS_TEX_WIDTH_BITS          = 16383; // 0b0011111111111111
const int ATLAS_TEX_MIPS_BIT            = 32768; // 0b1000000000000000
const int ATLAS_TEX_HEIGHT_BITS         = 16383; // 0b0011111111111111

const UINT_TYPE TEX_SRGB_BIT          = (1u << 24); // 0b00000001
const UINT_TYPE TEX_RECONSTRUCT_Z_BIT = (2u << 24); // 0b00000010
const UINT_TYPE TEX_YCOCG_BIT         = (4u << 24); // 0b00000100

struct ray_data_t {
	float o[3], d[3], pdf;
	float c[3];
#ifdef USE_RAY_DIFFERENTIALS

#else
	float cone_width, cone_spread;
#endif
	int xy;
	int ray_depth;
};

struct shadow_ray_t {
    // origin and direction
    float o[3], d[3], dist;
    // throughput color of ray
    float c[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    int xy;
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

struct vertex_t {
    float p[3], n[3], b[3], t[2][2];
};

struct mesh_t {
    float bbox_min[3], bbox_max[3];
    UINT_TYPE node_index, node_count;
    UINT_TYPE tris_index, tris_count;
    UINT_TYPE vert_index, vert_count;
};

struct transform_t {
    MAT4_TYPE xform, inv_xform;
};

struct mesh_instance_t {
    VEC4_TYPE bbox_min; // w is tr_index
    VEC4_TYPE bbox_max; // w is mesh_index
};

struct light_t {
    UVEC4_TYPE type_and_param0;
    VEC4_TYPE param1;
    VEC4_TYPE param2;
    VEC4_TYPE param3;
};

#define SPH_POS param1.xyz
#define SPH_AREA param1.w
#define SPH_RADIUS param2.x

#define RECT_POS param1.xyz
#define RECT_AREA param1.w
#define RECT_U param2.xyz
#define RECT_V param3.xyz

#define DISK_POS param1.xyz
#define DISK_AREA param1.w
#define DISK_U param2.xyz
#define DISK_V param3.xyz

#define TRI_TRI_INDEX param1.x
#define TRI_XFORM_INDEX param1.y

#define DIR_DIR param1.xyz
#define DIR_ANGLE param1.w

struct material_t {
    UINT_TYPE textures[MAX_MATERIAL_TEXTURES];
    float base_color[3];
    UINT_TYPE flags;
    UINT_TYPE type;
    float tangent_rotation_or_strength;
    UINT_TYPE roughness_and_anisotropic;
    float int_ior;
    float ext_ior;
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

INTERFACE_END

#endif // TYPES_GLSL