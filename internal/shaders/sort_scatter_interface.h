#ifndef SORT_SCATTER_INTERFACE_H
#define SORT_SCATTER_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortScatter)

struct Params {
    int shift;
    int counter;
    int _pad0;
    int _pad1;
};

const int LOCAL_GROUP_SIZE_X = 256;
const int SCAN_PORTION = 256;

const int HASHES_BUF_SLOT = 1;
const int COUNTERS_BUF_SLOT = 2;
const int SUM_TABLE_BUF_SLOT = 3;

const int OUT_HASHES_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_SCATTER_INTERFACE_H