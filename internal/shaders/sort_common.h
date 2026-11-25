#ifndef SORT_COMMON_H
#define SORT_COMMON_H

const int SORT_BITS_PER_PASS = 8;
const int SORT_BINS_COUNT = (1 << SORT_BITS_PER_PASS);
const int SORT_ELEMENTS_PER_THREAD = 4;
const int SORT_THREADGROUP_SIZE = 128;
// Histogram rows per block: a thread partition (row = li / 32), correct for any subgroup size >= 32.
const int SORT_HIST_ROWS = SORT_THREADGROUP_SIZE / 32;

#endif // SORT_COMMON_H
