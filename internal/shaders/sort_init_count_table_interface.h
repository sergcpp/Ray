#ifndef SORT_INIT_COUNT_TABLE_INTERFACE_H
#define SORT_INIT_COUNT_TABLE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortInitCountTable)

#include "sort_common.h"

struct Params {
    int shift;
    int counter;
    int hash_input;
    int _pad0;
    vec4 root_min;
    vec4 cell_size;
};

const int HASHES_BUF_SLOT = 1;
const int COUNTERS_BUF_SLOT = 2;
const int RAYS_BUF_SLOT = 3;
const int OUT_HASHES_BUF_SLOT = 4;

const int OUT_COUNT_TABLE_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_INIT_COUNT_TABLE_INTERFACE_H