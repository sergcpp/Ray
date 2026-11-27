#ifndef SORT_LOCAL_INTERFACE_H
#define SORT_LOCAL_INTERFACE_H

#include "_interface_common.h"
#include "sort_common.h"

INTERFACE_START(SortLocal)

struct Params {
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
    vec4 root_min;
    vec4 cell_size;
};

const int COUNTERS_BUF_SLOT = 2;
const int RAYS_BUF_SLOT = 3;
const int OUT_HASHES_BUF_SLOT = 4;

INTERFACE_END

#endif // SORT_LOCAL_INTERFACE_H
