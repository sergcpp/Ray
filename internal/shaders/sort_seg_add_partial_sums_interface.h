#ifndef SORT_SEG_ADD_PARTIAL_SUMS_INTERFACE_H
#define SORT_SEG_ADD_PARTIAL_SUMS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortSegAddPartialSums)

struct Params {
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int PART_SUMS_BUF_SLOT = 1;
const int PART_FLAGS_BUF_SLOT = 2;
const int COUNTERS_BUF_SLOT = 3;

const int INOUT_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_SEG_ADD_PARTIAL_SUMS_INTERFACE_H