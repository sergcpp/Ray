#ifndef SORT_INIT_CHUNKS_INTERFACE_H
#define SORT_INIT_CHUNKS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortInitChunks)

struct Params {
    int chunks_counter;
    int rays_counter;
    int _pad0;
    int _pad1;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int HASH_VALUES_BUF_SLOT = 1;
const int HEAD_FLAGS_BUF_SLOT = 2;
const int SCAN_VALUES_BUF_SLOT = 3;
const int COUNTERS_BUF_SLOT = 4;

const int INOUT_CHUNKS_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_INIT_CHUNKS_INTERFACE_H