#ifndef SORT_REORDER_RAYS_INTERFACE_H
#define SORT_REORDER_RAYS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortReorderRays)

struct Params {
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int RAYS_BUF_SLOT = 1;
const int INDICES_BUF_SLOT = 2;
const int COUNTERS_BUF_SLOT = 3;

const int OUT_RAYS_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_REORDER_RAYS_INTERFACE_H