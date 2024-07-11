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
const int RAND_DIM_CACHE = 5;
// 6 and 7 reserved for the future use
const int RAND_DIM_BOUNCE_COUNT = 8; // separate for each bounce

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

//
// Light transport
//
const float MAX_CONE_SPREAD_INCREMENT = 0.05f;

//
// Spatial hashing
//
const uint HASH_GRID_CACHE_ENTRIES_COUNT = (1u << 22);
const uint HASH_GRID_POSITION_BIT_NUM = 17u;
const uint HASH_GRID_POSITION_BIT_MASK = (1u << HASH_GRID_POSITION_BIT_NUM) - 1;
const uint HASH_GRID_LEVEL_BIT_NUM = 10u;
const uint HASH_GRID_LEVEL_BIT_MASK = (1u << HASH_GRID_LEVEL_BIT_NUM) - 1;
const uint HASH_GRID_NORMAL_BIT_NUM = 3u;
const uint HASH_GRID_NORMAL_BIT_MASK = (1u << HASH_GRID_NORMAL_BIT_NUM) - 1;
const uint HASH_GRID_HASH_MAP_BUCKET_SIZE = 32u;
const uint HASH_GRID_INVALID_CACHE_ENTRY = 0xFFFFFFFFu;
const uint HASH_GRID_LEVEL_BIAS = 2u; // positive bias adds extra levels with content magnification
const uint HASH_GRID_INVALID_HASH_KEY = 0u;
const bool HASH_GRID_USE_NORMALS = true;
const bool HASH_GRID_ALLOW_COMPACTION = (HASH_GRID_HASH_MAP_BUCKET_SIZE == 32u);

//
// Radiance caching
//
const int RAD_CACHE_SAMPLE_COUNT_MAX = 128;
const int RAD_CACHE_SAMPLE_COUNT_MIN = 8;
const float RAD_CACHE_RADIANCE_SCALE = 1e4f;
const int RAD_CACHE_SAMPLE_COUNTER_BIT_NUM = 20;
const uint RAD_CACHE_SAMPLE_COUNTER_BIT_MASK = ((1u << RAD_CACHE_SAMPLE_COUNTER_BIT_NUM) - 1);
const uint RAD_CACHE_FRAME_COUNTER_BIT_NUM = (32 - RAD_CACHE_SAMPLE_COUNTER_BIT_NUM);
const uint RAD_CACHE_FRAME_COUNTER_BIT_MASK = ((1u << RAD_CACHE_FRAME_COUNTER_BIT_NUM) - 1);
const float RAD_CACHE_GRID_LOGARITHM_BASE = 2.0f;
const int RAD_CACHE_STALE_FRAME_NUM_MAX = 128;
const int RAD_CACHE_DOWNSAMPLING_FACTOR = 4;
const bool RAD_CACHE_ENABLE_COMPACTION = true;
const bool RAD_CACHE_FILTER_ADJACENT_LEVELS = true;
const int RAD_CACHE_PROPAGATION_DEPTH = 4;
const float RAD_CACHE_GRID_SCALE = 50.0f;
const float RAD_CACHE_MIN_ROUGHNESS = 0.4f;

//
// Atmosphere
//
const int SKY_TRANSMITTANCE_LUT_W = 256;
const int SKY_TRANSMITTANCE_LUT_H = 64;
const int SKY_MULTISCATTER_LUT_RES = 32;
const int SKY_PRE_ATMOSPHERE_SAMPLE_COUNT = 4;
const int SKY_MAIN_ATMOSPHERE_SAMPLE_COUNT = 12;
const int SKY_CLOUDS_SAMPLE_COUNT = 48;
const float SKY_CLOUDS_HORIZON_CUTOFF = 0.005f;
const float SKY_MOON_SUN_RELATION = 0.0000001f;
const float SKY_STARS_THRESHOLD = 14.0f;
const float SKY_SUN_BLEND_VAL = 0.000005f;
