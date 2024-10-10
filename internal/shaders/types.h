#ifndef TYPES_H
#define TYPES_H

#include "_interface_common.h"

INTERFACE_START(Types)

#include "../Constants.inl"

const int RAND_SAMPLES_COUNT = 4096;
const int RAND_DIMS_COUNT = 32;

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38
#endif

const int DiffuseNode = 0;
const int GlossyNode = 1;
const int RefractiveNode = 2;
const int EmissiveNode = 3;
const int MixNode = 4;
const int TransparentNode = 5;
const int PrincipledNode = 6;

const uint TEX_SRGB_BIT = (1u << 24);           // 0b00000001
const uint TEX_RECONSTRUCT_Z_BIT = (2u << 24);  // 0b00000010
const uint TEX_YCOCG_BIT = (4u << 24);          // 0b00000100

const uint TEXTURES_SAMPLER_SLOT = 22;
const uint TEXTURES_SIZE_SLOT = 23;
const uint TEXTURES_BUF_SLOT = 24;
const uint TEXTURE_ATLASES_SLOT = 25;

const int FILTER_BOX = 0;
const int FILTER_GAUSSIAN = 1;
const int FILTER_BLACKMAN_HARRIS = 2;

const int FILTER_TABLE_SIZE = 1024;

struct ray_data_t {
	float o[3], d[3], pdf;
	float c[3];
    float ior[4];
	float cone_width, cone_spread;
	uint xy;
    uint depth;
};

struct shadow_ray_t {
    // origin
    float o[3];
    // four 8-bit ray depth counters
    uint depth;
    // direction and distance
    float d[3], dist;
    // throughput color of ray
    float c[3];
    // 16-bit pixel coordinates of ray ((x << 16) | y)
    uint xy;
};

struct tri_accel_t {
    vec4 n_plane;
    vec4 u_plane;
    vec4 v_plane;
};

struct hit_data_t {
    int obj_index;
    int prim_index;
    float t, u, v;
};

struct bvh_node_t {
    vec4 bbox_min; // w is prim_index/left_child
    vec4 bbox_max; // w is prim_count/right_child
};

struct light_bvh_node_t {
    float bbox_min[3];
    uint left_child;
    float bbox_max[3];
    uint right_child;
    float flux;
    float axis[3];
    float omega_n; // cone angle enclosing light normals
    float omega_e; // emission angle around each normal
};

struct light_wbvh_node_t {
    float bbox_min[3][8];
    float bbox_max[3][8];
    uint child[8];
    float flux[8];
    uint axis[8];
    uint cos_omega_ne[8];
};

struct light_cwbvh_node_t {
    float bbox_min[3];
    float _unused0;
    float bbox_max[3];
    float _unused1;
    uint ch_bbox_min[3][2];
    uint ch_bbox_max[3][2];
    uint child[8];
    float flux[8];
    uint axis[8];
    uint cos_omega_ne[8];
};

struct vertex_t {
    float p[3], n[3], b[3], t[2];
};

struct mesh_t {
    float bbox_min[3], bbox_max[3];
    uint node_index, node_block;
    uint tris_index, tris_block, tris_count;
    uint vert_index, vert_block, vert_count;
    uint vert_data_index, vert_data_block;
};

struct mesh_instance_t {
    vec4 bbox_min;      // w is tr_index
    vec4 bbox_max;      // w is mesh_index
    uvec4 block_ndx;    // xy - indexes of transform and mesh blocks, z - lights index, w - ray_visibility
    mat4 xform, inv_xform;
};

struct light_t {
    uvec4 type_and_param0;
    vec4 param1;
    vec4 param2;
    vec4 param3;
};

#define LIGHT_TYPE(l) (l.type_and_param0.x & 0x7)
#define LIGHT_DOUBLE_SIDED(l) ((l.type_and_param0.x & (1 << 3)) != 0)
#define LIGHT_CAST_SHADOW(l) ((l.type_and_param0.x & (1 << 4)) != 0)
#define LIGHT_VISIBLE(l) ((l.type_and_param0.x & (1 << 5)) != 0)
#define LIGHT_SKY_PORTAL(l) ((l.type_and_param0.x & (1 << 6)) != 0)
#define LIGHT_RAY_VISIBILITY(l) ((l.type_and_param0.x >> 7u) & 0xffu)

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
#define TRI_MI_INDEX param1.y
#define TRI_TEX_INDEX param1.z

#define DIR_DIR param1.xyz
#define DIR_ANGLE param1.w

struct material_t {
    uint textures[MAX_MATERIAL_TEXTURES];
    float base_color[3];
    uint flags;
    uint type;
    float tangent_rotation_or_strength;
    uint roughness_and_anisotropic;
    float ior;
    uint sheen_and_sheen_tint;
    uint tint_and_metallic;
    uint transmission_and_transmission_roughness;
    uint specular_and_specular_tint;
    uint clearcoat_and_clearcoat_roughness;
    uint normal_map_strength_unorm;
};

struct atlas_texture_t {
    uint size;
    uint atlas;
    uint page[(NUM_MIP_LEVELS + 3) / 4];
    uint pos[NUM_MIP_LEVELS];
};

struct ray_chunk_t {
    uint hash, base, size;
};

struct ray_hash_t {
    uint hash, index;
};

struct cache_voxel_t {
    vec3 radiance;
    uint sample_count;
    uint frame_count;
};

struct cache_grid_params_t {
    vec3 cam_pos_curr, cam_pos_prev;
    float log_base;
    float scale;
    float exposure;
};

struct cache_data_t {
    uint cache_entries[RAD_CACHE_PROPAGATION_DEPTH];
    float sample_weight[RAD_CACHE_PROPAGATION_DEPTH][3];
    int path_len;
};

struct atmosphere_params_t {
    float planet_radius;
    float viewpoint_height;
    float atmosphere_height;
    float rayleigh_height;
    //
    float mie_height;
    float clouds_height_beg;
    float clouds_height_end;
    float clouds_variety;
    //
    float clouds_density;
    float clouds_offset_x;
    float clouds_offset_z;
    float clouds_flutter_x;
    //
    float clouds_flutter_z;
    float cirrus_clouds_amount;
    float cirrus_clouds_height;
    float ozone_height_center;
    //
    float ozone_half_width;
    float atmosphere_density;
    float stars_brightness;
    float moon_radius;
    //
    float moon_distance;
    float _unused0;
    float _unused1;
    float _unused2;
    //
    vec4 moon_dir;
    vec4 rayleigh_scattering;
    vec4 mie_scattering;
    vec4 mie_extinction;
    vec4 mie_absorption;
    vec4 ozone_absorbtion;
    vec4 ground_albedo;
};

INTERFACE_END

#endif // TYPES_H