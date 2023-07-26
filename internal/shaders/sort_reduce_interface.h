#ifndef SORT_REDUCE_INTERFACE_H
#define SORT_REDUCE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortReduce)

struct Params {
    int offset;
    int stride;
    int counter;
    int _pad0;
};

const int INPUT_BUF_SLOT = 1;
const int COUNTERS_BUF_SLOT = 2;

const int OUT_REDUCE_TABLE_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_REDUCE_INTERFACE_H