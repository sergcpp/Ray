#ifndef SORT_SEG_SCAN_INTERFACE_H
#define SORT_SEG_SCAN_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortSegScan)

const int LOCAL_GROUP_SIZE_X = 64;
const int SEG_SCAN_PORTION = 64;

const int VALUES_BUF_SLOT = 3;
const int FLAGS_BUF_SLOT = 4;

const int OUT_SCAN_VALUES_BUF_SLOT = 0;
const int OUT_PARTIAL_SUMS_BUF_SLOT = 1;
const int OUT_PARTIAL_FLAGS_BUF_SLOT = 2;

INTERFACE_END

#endif // SORT_SEG_SCAN_INTERFACE_H