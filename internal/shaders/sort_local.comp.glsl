#version 450
#extension GL_GOOGLE_include_directive : require

// Block-local LSD radix sort: one workgroup per SORT_LOCAL_BLOCK consecutive rays, all
// coordination in shared memory - no cross-block atomics, one dispatch per bounce.
// Packed record: hash bits 11..31 stay in place, low 11 bits carry the local slot index.
// Scatter ranks with a per-bin atomic cursor: wave-size agnostic (Intel's VK driver picks
// 8-lane subgroups freely and its DX driver wave8 - DX has no wave-size requirement), but
// single-digit only: atomic order is not stable, so a minor digit would be scrambled.

#include "sort_local_interface.h"
#include "common.glsl"
#include "sort_common.h"
#include "sort_hash.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(local_size_x = SORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

layout(std430, binding = OUT_HASHES_BUF_SLOT) writeonly buffer OutHashes {
    uint g_out_indices[];
};

const int SORT_EPT = SORT_LOCAL_BLOCK / SORT_THREADGROUP_SIZE;

shared uint s_buf[2][SORT_LOCAL_BLOCK];
// Per-bin counts, converted in place to exclusive bases, then used as scatter cursors
shared uint s_cnt[SORT_BINS_COUNT];
shared uint s_scan[SORT_THREADGROUP_SIZE];

void main() {
    const int li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const uint data_count = g_counters[g_params.counter];
    const int block_start = wi * SORT_LOCAL_BLOCK;

    for (int i = 0; i < SORT_EPT; ++i) {
        const int idx = block_start + i * SORT_THREADGROUP_SIZE + li;
        if (idx < int(data_count)) {
            const uint h = get_ray_hash(g_rays[idx], g_params.root_min.xyz, g_params.cell_size.xyz);
            // Bit 23 is cleared too: a far-corner origin can set bits 16..23, and the 0xffffffff
            // sentinel must win the shift-16 digit tie or data gets pushed past data_count.
            s_buf[0][i * SORT_THREADGROUP_SIZE + li] =
                (h & ~(0x800000u | ((1u << SORT_LOCAL_IDX_BITS) - 1u))) | uint(idx - block_start);
        } else {
            // Padding key (0xffffffff) sorts past data_count, like the sentinel in the global sort
            s_buf[0][i * SORT_THREADGROUP_SIZE + li] = 0xffffffffu;
        }
    }
    groupMemoryBarrier(); barrier();

    for (int k = li; k < SORT_BINS_COUNT; k += SORT_THREADGROUP_SIZE) {
        s_cnt[k] = 0;
    }
    groupMemoryBarrier(); barrier();

    for (int i = 0; i < SORT_EPT; ++i) {
        const uint bin = (s_buf[0][i * SORT_THREADGROUP_SIZE + li] >> SORT_LOCAL_SHIFT) & (SORT_BINS_COUNT - 1);
        atomicAdd(s_cnt[bin], 1u);
    }
    groupMemoryBarrier(); barrier();

    // Per-bin totals (thread li owns bin pair 2*li, 2*li+1), then Kogge-Stone scan of pair sums
    const uint tot_lo = s_cnt[2 * li];
    const uint tot_hi = s_cnt[2 * li + 1];
    s_scan[li] = tot_lo + tot_hi;
    groupMemoryBarrier(); barrier();
    for (int k = 0; k < 7; ++k) {
        const uint add = (li >= (1 << k)) ? s_scan[li - (1 << k)] : 0u;
        groupMemoryBarrier(); barrier();
        s_scan[li] += add;
        groupMemoryBarrier(); barrier();
    }
    // Counts -> exclusive bases; the scatter cursors end at the next bin's base
    const uint excl = s_scan[li] - (tot_lo + tot_hi);
    s_cnt[2 * li] = excl;
    s_cnt[2 * li + 1] = excl + tot_lo;
    groupMemoryBarrier(); barrier();

    // The cursor hands out unique slots in the bin's range; order within a bin is atomic
    // order - a valid perm, pixel-neutral, but not run-to-run stable
    for (int i = 0; i < SORT_EPT; ++i) {
        const uint w = s_buf[0][i * SORT_THREADGROUP_SIZE + li];
        const uint bin = (w >> SORT_LOCAL_SHIFT) & (SORT_BINS_COUNT - 1);
        s_buf[1][atomicAdd(s_cnt[bin], 1u)] = w;
    }
    groupMemoryBarrier(); barrier();

    for (int i = 0; i < SORT_EPT; ++i) {
        const int pos = i * SORT_THREADGROUP_SIZE + li;
        const int out_index = block_start + pos;
        // Sentinels land past the block's valid count, so the guard drops them
        if (out_index < int(data_count)) {
            const uint w = s_buf[1][pos];
            g_out_indices[out_index] = uint(block_start) + (w & ((1u << SORT_LOCAL_IDX_BITS) - 1u));
        }
    }
}
