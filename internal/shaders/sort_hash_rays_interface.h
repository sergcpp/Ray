#ifndef SORT_HASH_RAYS_INTERFACE_H
#define SORT_HASH_RAYS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortHashRays)

struct Params {
    vec4 root_min;
    vec4 cell_size;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int RAYS_BUF_SLOT = 1;
const int COUNTERS_BUF_SLOT = 2;

const int OUT_HASHES_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_HASH_RAYS_INTERFACE_H