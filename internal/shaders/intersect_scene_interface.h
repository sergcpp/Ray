#ifndef INTERSECT_SCENE_INTERFACE_H
#define INTERSECT_SCENE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(IntersectScene)

struct Params {
    uvec4 rect;
    uint node_index;
    float clip_dist;
    int min_transp_depth;
    int max_transp_depth;
    uint rand_seed;
    int iteration;
    int _pad1;
    int _pad2;
    vec4 cam_fwd;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int VERTICES_BUF_SLOT = 1;
const int VTX_INDICES_BUF_SLOT = 2;
const int TRIS_BUF_SLOT = 3;
const int TRI_INDICES_BUF_SLOT = 4;
const int TRI_MATERIALS_BUF_SLOT = 5;
const int MATERIALS_BUF_SLOT = 6;
const int NODES_BUF_SLOT = 7;
const int MESHES_BUF_SLOT = 8;
const int MESH_INSTANCES_BUF_SLOT = 9;
const int MI_INDICES_BUF_SLOT = 10;
const int RAYS_BUF_SLOT = 11;
const int TLAS_SLOT = 12;
const int COUNTERS_BUF_SLOT = 13;
const int RANDOM_SEQ_BUF_SLOT = 14;

const int OUT_HITS_BUF_SLOT = 0;

INTERFACE_END

#endif // INTERSECT_SCENE_INTERFACE_H