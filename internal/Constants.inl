//
// Global constants
//
const int MAX_STACK_SIZE = 48;
const int MAX_BOUNCES = 128;

const float HIT_BIAS = 0.00001f;
const float HIT_EPS = 0.000001f;
const float FLT_EPS = 0.0000001f;

const float MAX_DIST = 3.402823466e+30F; // 3.402823466e+38F

const float SPHERICAL_AREA_THRESHOLD =
    0.00005f; // non-zero minimal spherical area to sample using spherical method (avoids precision issues)

const uint LEAF_NODE_BIT = (1u << 31);
const uint PRIM_INDEX_BITS = ~LEAF_NODE_BIT;
const uint LEFT_CHILD_BITS = ~LEAF_NODE_BIT;

const uint SEP_AXIS_BITS = (3u << 30); // 0b11u
const uint PRIM_COUNT_BITS = ~SEP_AXIS_BITS;
const uint RIGHT_CHILD_BITS = ~SEP_AXIS_BITS;

const float PI = 3.141592653589793238463f;

//
// Random sequence constants
//
const int RAND_DIM_FILTER = 0;
const int RAND_DIM_LENS = 1;
const int RAND_DIM_BASE_COUNT = 2; // independent from bounce count

const int RAND_DIM_BSDF_PICK = 0;
const int RAND_DIM_BSDF = 1;
const int RAND_DIM_LIGHT_PICK = 2;
const int RAND_DIM_LIGHT = 3;
const int RAND_DIM_TEX = 4;
const int RAND_DIM_BOUNCE_COUNT = 5; // separate for each bounce

//
// Light constants
//
const int LIGHT_TYPE_SPHERE = 0;
const int LIGHT_TYPE_DIR = 1;
const int LIGHT_TYPE_LINE = 2;
const int LIGHT_TYPE_RECT = 3;
const int LIGHT_TYPE_DISK = 4;
const int LIGHT_TYPE_TRI = 5;
const int LIGHT_TYPE_ENV = 6;

//
// Ray constants
//
const int RAY_TYPE_CAMERA = 0;
const int RAY_TYPE_DIFFUSE = 1;
const int RAY_TYPE_SPECULAR = 2;
const int RAY_TYPE_REFR = 3;
const int RAY_TYPE_SHADOW = 4;

//
// Material constants
//
const int MAX_MATERIAL_TEXTURES = 5;

const int NORMALS_TEXTURE = 0;
const int BASE_TEXTURE = 1;
const int ROUGH_TEXTURE = 2;
const int METALLIC_TEXTURE = 3;
const int SPECULAR_TEXTURE = 4;

const int MIX_MAT1 = 3;
const int MIX_MAT2 = 4;

const int MATERIAL_SOLID_BIT = 32768;  // 0b1000000000000000
const int MATERIAL_INDEX_BITS = 16383; // 0b0011111111111111

const uint MAT_FLAG_MULT_IMPORTANCE = (1u << 0u);
const uint MAT_FLAG_MIX_ADD = (1u << 1u);

const int NUM_MIP_LEVELS = 12;
const int MAX_MIP_LEVEL = NUM_MIP_LEVELS - 1;
const int MIN_ATLAS_TEXTURE_SIZE = 4;
const int MAX_ATLAS_TEXTURE_SIZE = (MIN_ATLAS_TEXTURE_SIZE << MAX_MIP_LEVEL);

const int TEXTURE_ATLAS_SIZE = 8192;

const int ATLAS_TEX_SRGB_BIT = 32768;          // 0b1000000000000000
const int ATLAS_TEX_RECONSTRUCT_Z_BIT = 16384; // 0b0100000000000000
const int ATLAS_TEX_WIDTH_BITS = 16383;        // 0b0011111111111111
const int ATLAS_TEX_MIPS_BIT = 32768;          // 0b1000000000000000
const int ATLAS_TEX_YCOCG_BIT = 16384;         // 0b0100000000000000
const int ATLAS_TEX_HEIGHT_BITS = 16383;       // 0b0011111111111111