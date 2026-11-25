#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

// Taken from FFX_ParallelSort.h
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "sort_scatter_interface.h"
#include "common.glsl"
#include "sort_common.h"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HASHES_BUF_SLOT) readonly buffer RayChunks {
    ray_hash_t g_hashes[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = SUM_TABLE_BUF_SLOT) readonly buffer SumTableBuf {
    uint g_sum_table[];
};

layout(std430, binding = OUT_HASHES_BUF_SLOT) writeonly buffer OutHashes {
    ray_hash_t g_out_hashes[];
};

layout (local_size_x = SORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// Subgroups of at least 16 lanes (a threadgroup holds at most 8 of them)
const int SORT_MAX_SUBGROUPS = SORT_THREADGROUP_SIZE / 16;

shared uint g_shared_bin_offsets[SORT_BINS_COUNT];
// Per-subgroup key counts, double-buffered by tile parity so zeroing never races with the previous tile's reads
shared uint g_shared_sg_counts[2][SORT_MAX_SUBGROUPS * SORT_BINS_COUNT];

uint popcount4(uvec4 v) {
    return bitCount(v.x) + bitCount(v.y) + bitCount(v.z) + bitCount(v.w);
}

void main() {
    const int gi = int(gl_GlobalInvocationID.x), li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const uint data_count = g_counters[g_params.counter];

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int blocks_count = int(data_count + BlockSize - 1) / BlockSize;

    const int subgroup_id = li / int(gl_SubgroupSize);
    const int sg_table_size = (SORT_THREADGROUP_SIZE / int(gl_SubgroupSize)) * SORT_BINS_COUNT;

    // SORT_BINS_COUNT == 2 * SORT_THREADGROUP_SIZE: thread li owns the bin pair (2*li, 2*li + 1)
    g_shared_bin_offsets[2 * li] = g_sum_table[2 * li * blocks_count + wi];
    g_shared_bin_offsets[2 * li + 1] = g_sum_table[(2 * li + 1) * blocks_count + wi];

    const int block_start = BlockSize * wi;

    int data_index = block_start + li;

    ray_hash_t src_keys[SORT_ELEMENTS_PER_THREAD];
    [[unroll]] for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        src_keys[i] = g_hashes[data_index + i * SORT_THREADGROUP_SIZE];
    }

    // Previous tile's per-bin counts, added to the cached offsets after the next tile's first barrier
    uint pending_lo = 0, pending_hi = 0;

    for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        ray_hash_t local_key = ray_hash_t(0xffffffff, 0xffffffff);
        if (data_index < data_count) {
            local_key = src_keys[i];
        }

        const uint key_index = (local_key.hash >> g_params.shift) & (SORT_BINS_COUNT - 1);

        // Subgroup lanes sharing this key; the rank among them is stable (lane order)
        uvec4 match = subgroupBallot(true);
        [[unroll]] for (int b = 0; b < SORT_BITS_PER_PASS; ++b) {
            const bool bit = ((key_index >> b) & 1u) != 0u;
            const uvec4 bit_mask = subgroupBallot(bit);
            match &= bit ? bit_mask : ~bit_mask;
        }
        const uint rank_in_sg = popcount4(match & gl_SubgroupLtMask);

        const int buf = i & 1;
        for (int k = li; k < sg_table_size; k += SORT_THREADGROUP_SIZE) {
            g_shared_sg_counts[buf][k] = 0;
        }

        groupMemoryBarrier(); barrier();

        // All reads of the previous tile are done, safe to advance the cached offsets
        g_shared_bin_offsets[2 * li] += pending_lo;
        g_shared_bin_offsets[2 * li + 1] += pending_hi;

        if (rank_in_sg == 0) {
            g_shared_sg_counts[buf][subgroup_id * SORT_BINS_COUNT + key_index] = popcount4(match);
        }

        groupMemoryBarrier(); barrier();

        // Stable rank within the bin: keys of earlier subgroups first, then earlier lanes of this one
        uint rank_in_bin = rank_in_sg;
        for (int s = 0; s < subgroup_id; ++s) {
            rank_in_bin += g_shared_sg_counts[buf][s * SORT_BINS_COUNT + key_index];
        }

        pending_lo = 0;
        pending_hi = 0;
        for (int s = 0; s < SORT_THREADGROUP_SIZE / int(gl_SubgroupSize); ++s) {
            pending_lo += g_shared_sg_counts[buf][s * SORT_BINS_COUNT + 2 * li];
            pending_hi += g_shared_sg_counts[buf][s * SORT_BINS_COUNT + 2 * li + 1];
        }

        const uint total_offset = g_shared_bin_offsets[key_index] + rank_in_bin;

        // Padding keys (0xffffffff) land in the last bin after all real keys, so they fall past data_count
        if (total_offset < data_count) {
            g_out_hashes[total_offset] = local_key;
        }

        data_index += SORT_THREADGROUP_SIZE;
    }
}
