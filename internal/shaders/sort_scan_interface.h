#ifndef SORT_SCAN_INTERFACE_H
#define SORT_SCAN_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortScan)

struct Params {
    int offset;
    int stride;
    int _pad0;
    int _pad1;
};

const int LOCAL_GROUP_SIZE_X = 256;
const int SCAN_PORTION = 256;

const int INPUT_BUF_SLOT = 2;

const int OUT_SCAN_VALUES_BUF_SLOT = 0;
const int OUT_PARTIAL_SUMS_BUF_SLOT = 1;

INTERFACE_END

#endif // SORT_SCAN_INTERFACE_H