#ifndef SHADE_INTERFACE_H
#define SHADE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Shade)

struct Params {
    uvec4 rect;
    vec4 env_col;
    vec4 back_col;
    //
    int iteration;
    int li_count;
    uint max_ray_depth;
    float regularize_alpha;
    //
    float limit_direct;
    float sky_map_spread_angle;
    int max_total_depth;
    int min_total_depth;
    //
    uint rand_seed;
    int env_qtree_levels;
    float env_rotation;
    float back_rotation;
    //
    int env_light_index;
    float limit_indirect;
    uint env_map_res;
    uint back_map_res;
    //
    vec4 cam_pos_and_exposure;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int HITS_BUF_SLOT = 7;
const int RAYS_BUF_SLOT = 8;
const int LIGHTS_BUF_SLOT = 9;
const int LI_INDICES_BUF_SLOT = 10;
const int TRIS_BUF_SLOT = 11;
const int TRI_MATERIALS_BUF_SLOT = 12;
const int MATERIALS_BUF_SLOT = 13;
const int MESH_INSTANCES_BUF_SLOT = 14;
const int VERTICES_BUF_SLOT = 15;
const int VTX_INDICES_BUF_SLOT = 16;
const int RANDOM_SEQ_BUF_SLOT = 17;
const int LIGHT_WNODES_BUF_SLOT = 18;
const int ENV_QTREE_TEX_SLOT = 19;
const int CACHE_ENTRIES_BUF_SLOT = 20;
const int CACHE_VOXELS_BUF_SLOT = 21;

const int OUT_IMG_SLOT = 0;
const int OUT_RAYS_BUF_SLOT = 1;
const int OUT_SH_RAYS_BUF_SLOT = 2;
const int OUT_SKY_RAYS_BUF_SLOT = 6;
const int INOUT_COUNTERS_BUF_SLOT = 3;

const int OUT_BASE_COLOR_IMG_SLOT = 4;
const int OUT_DEPTH_NORMALS_IMG_SLOT = 5;

INTERFACE_END

#endif // SHADE_INTERFACE_H
