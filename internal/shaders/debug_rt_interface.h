#ifndef DEBUG_RT_INTERFACE_H
#define DEBUG_RT_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(DebugRT)

struct Params {
    UVEC2_TYPE img_size;
    UINT_TYPE node_index;
    float halton;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int TRIS_BUF_SLOT = 1;
const int TRI_INDICES_BUF_SLOT = 2;
const int NODES_BUF_SLOT = 3;
const int MESHES_BUF_SLOT = 4;
const int MESH_INSTANCES_BUF_SLOT = 5;
const int MI_INDICES_BUF_SLOT = 6;
const int TRANSFORMS_BUF_SLOT = 7;
const int RAYS_BUF_SLOT = 8;
const int TLAS_SLOT = 9;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // DEBUG_RT_INTERFACE_H