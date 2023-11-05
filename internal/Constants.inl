const int MAX_STACK_SIZE = 48;

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