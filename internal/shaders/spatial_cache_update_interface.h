#ifndef SPATIAL_CACHE_UPDATE_INTERFACE_H
#define SPATIAL_CACHE_UPDATE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(CacheUpdate)

struct Params {
    vec4 cam_pos_curr;
    uint cache_w;
    uint entries_count;
    float exposure;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int HITS_BUF_SLOT = 4;
const int RAYS_BUF_SLOT = 5;
const int COUNTERS_BUF_SLOT = 6;
const int RADIANCE_TEX_SLOT = 7;
const int DEPTH_NORMAL_TEX_SLOT = 8;

const int INOUT_CACHE_DATA_BUF_SLOT = 0;
const int INOUT_CACHE_ENTRIES_BUF_SLOT = 1;
const int INOUT_CACHE_VOXELS_BUF_SLOT = 2;
const int INOUT_CACHE_LOCK_BUF_SLOT = 3;

INTERFACE_END

#endif // SPATIAL_CACHE_UPDATE_INTERFACE_H