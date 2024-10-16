#ifndef INTERSECT_SCENE_SHADOW_INTERFACE_H
#define INTERSECT_SCENE_SHADOW_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(IntersectSceneShadow)

struct Params {
    uint node_index;
    int max_transp_depth;
    uint lights_node_index;
    int blocker_lights_count;
    float clamp_val;
    uint rand_seed;
    int iteration;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int TRIS_BUF_SLOT = 1;
const int TRI_INDICES_BUF_SLOT = 2;
const int TRI_MATERIALS_BUF_SLOT = 3;
const int MATERIALS_BUF_SLOT = 4;
const int NODES_BUF_SLOT = 5;
const int MESH_INSTANCES_BUF_SLOT = 6;
const int VERTICES_BUF_SLOT = 7;
const int VTX_INDICES_BUF_SLOT = 8;
const int SH_RAYS_BUF_SLOT = 9;
const int COUNTERS_BUF_SLOT = 10;
const int LIGHTS_BUF_SLOT = 11;
const int LIGHT_CWNODES_BUF_SLOT = 12;
const int RANDOM_SEQ_BUF_SLOT = 13;
const int TLAS_SLOT = 14;

const int INOUT_IMG_SLOT = 0;

INTERFACE_END

#endif // INTERSECT_SCENE_SHADOW_INTERFACE_H