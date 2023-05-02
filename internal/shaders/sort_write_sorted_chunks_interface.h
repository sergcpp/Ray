#ifndef SORT_WRITE_SORTED_CHUNKS_INTERFACE_H
#define SORT_WRITE_SORTED_CHUNKS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(SortWriteSortedChunks)

struct Params {
    int shift;
    int counter;
    int chunks_counter;
    int _pad0;
};

const int LOCAL_GROUP_SIZE_X = 64;

const int CHUNKS_BUF_SLOT = 1;
const int OFFSETS_BUF_SLOT = 2;
const int COUNTERS_BUF_SLOT = 3;

const int OUT_CHUNKS_BUF_SLOT = 0;

INTERFACE_END

#endif // SORT_WRITE_SORTED_CHUNKS_INTERFACE_H