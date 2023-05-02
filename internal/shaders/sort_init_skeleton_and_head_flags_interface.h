#ifndef SORT_INIT_SKELETON_AND_HEAD_FLAGS_INTERFACE_H
#define SORT_INIT_SKELETON_AND_HEAD_FLAGS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortInitSkeletonAndHeadFlags)

struct Params {
    int counter;
    int _pad0;
    int _pad1;
    int _pad2;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int COUNTERS_BUF_SLOT = 2;
const int SCAN_VALUES_BUF_SLOT = 3;
const int CHUNKS_BUF_SLOT = 4;

const int OUT_SKELETON_BUF_SLOT = 0;
const int OUT_HEAD_FLAGS_BUF_SLOT = 1;

INTERFACE_END

#endif // SORT_INIT_SKELETON_AND_HEAD_FLAGS_INTERFACE_H