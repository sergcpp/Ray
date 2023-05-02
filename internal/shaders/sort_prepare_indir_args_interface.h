#ifndef SORT_PREPARE_INDIR_ARGS_INTERFACE_H
#define SORT_PREPARE_INDIR_ARGS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortPrepareIndirArgs)

struct Params {
    int in_counter;
    int out_counter;
    int indir_args_index;
    int _pad0;
};

const int HEAD_FLAGS_BUF_SLOT = 2;
const int SCAN_VALUES_BUF_SLOT = 3;

const int INOUT_COUNTERS_BUF_SLOT = 0;
const int OUT_INDIR_ARGS_SLOT = 1;

INTERFACE_END

#endif // SORT_PREPARE_INDIR_ARGS_INTERFACE_H