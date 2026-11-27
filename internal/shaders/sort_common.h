#ifndef SORT_COMMON_H
#define SORT_COMMON_H

const int SORT_BITS_PER_PASS = 8;
const int SORT_BINS_COUNT = (1 << SORT_BITS_PER_PASS);
const int SORT_THREADGROUP_SIZE = 128;

// Block-local sort: each workgroup sorts SORT_LOCAL_BLOCK consecutive rays entirely in
// shared memory. The local slot index rides in the packed record's low bits.
const int SORT_LOCAL_BLOCK = 2048;
const int SORT_LOCAL_IDX_BITS = 11; // log2(SORT_LOCAL_BLOCK)
// Single LSD digit: the coarse origin cell. Scatter ranking is atomic-cursor based,
// which is not stable, so a minor digit (direction) cannot be added.
const int SORT_LOCAL_SHIFT = 16;

#ifdef __cplusplus
// sort_local's structure hard-codes these invariants; breaking one corrupts the perm silently.
static_assert(SORT_BINS_COUNT == 2 * SORT_THREADGROUP_SIZE); // bin-pair scan: thread li owns bins 2*li, 2*li+1
static_assert(SORT_LOCAL_BLOCK % SORT_THREADGROUP_SIZE == 0); // tile loop
static_assert((1 << SORT_LOCAL_IDX_BITS) == SORT_LOCAL_BLOCK); // index field packing
static_assert(SORT_THREADGROUP_SIZE == 128); // shader Kogge-Stone scan runs 7 steps (2^7 = TS)
#endif

#endif // SORT_COMMON_H
