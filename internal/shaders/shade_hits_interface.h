#ifndef SHADE_HITS_INTERFACE_H
#define SHADE_HITS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(ShadeHits)

struct Params {
    UVEC2_TYPE img_size;
    int hi;
    int li_count;
    int max_diff_depth;
    int max_spec_depth;
    int max_refr_depth;
    int max_transp_depth;
    int max_total_depth;
    int termination_start_depth;
    float env_rotation;
    int env_qtree_levels;
    VEC4_TYPE env_col;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int HITS_BUF_SLOT = 4;
const int RAYS_BUF_SLOT = 5;
const int LIGHTS_BUF_SLOT = 6;
const int LI_INDICES_BUF_SLOT = 7;
const int TRIS_BUF_SLOT = 8;
const int TRI_MATERIALS_BUF_SLOT = 9;
const int MATERIALS_BUF_SLOT = 10;
const int TRANSFORMS_BUF_SLOT = 11;
const int MESH_INSTANCES_BUF_SLOT = 12;
const int VERTICES_BUF_SLOT = 13;
const int VTX_INDICES_BUF_SLOT = 14;
const int HALTON_SEQ_BUF_SLOT = 15;
const int ENV_QTREE_TEX_SLOT = 16;

const int OUT_IMG_SLOT = 0;
const int OUT_RAYS_BUF_SLOT = 1;
const int OUT_SH_RAYS_BUF_SLOT = 2;
const int INOUT_COUNTERS_BUF_SLOT = 3;

INTERFACE_END

#endif // SHADE_HITS_INTERFACE_H
