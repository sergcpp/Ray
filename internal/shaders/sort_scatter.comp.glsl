#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_arithmetic : require
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

shared uint g_shared_sums[SORT_THREADGROUP_SIZE];

uint ThreadgroupExclusiveScan(uint local_sum, uint li) {
    uint subgroup_sum = subgroupExclusiveAdd(local_sum);

    uint subgroup_id = li / gl_SubgroupSize;
    uint lane_id = gl_SubgroupInvocationID;

    if (lane_id == gl_SubgroupSize - 1) {
        g_shared_sums[subgroup_id] = subgroup_sum + local_sum;
    }

    groupMemoryBarrier(); barrier();

    if (subgroup_id == 0) {
        g_shared_sums[li] = subgroupExclusiveAdd(g_shared_sums[li]);
    }

    groupMemoryBarrier(); barrier();

    subgroup_sum += g_shared_sums[subgroup_id];

    return subgroup_sum;
}

shared ray_hash_t g_shared_hashes[SORT_THREADGROUP_SIZE];

shared uint g_shared_bin_offsets[SORT_THREADGROUP_SIZE];
shared uint g_shared_local_counters[SORT_BINS_COUNT];
shared uint g_shared_scratch[SORT_THREADGROUP_SIZE];

void main() {
    const int gi = int(gl_GlobalInvocationID.x), li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const uint data_count = g_counters[g_params.counter];

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int blocks_count = int(data_count + BlockSize - 1) / BlockSize;

    if (li < SORT_BINS_COUNT) {
        g_shared_bin_offsets[li] = g_sum_table[li * blocks_count + wi];
    }

    groupMemoryBarrier(); barrier();

    const int block_start = BlockSize * wi;

    int data_index = block_start + li;

    ray_hash_t src_keys[SORT_ELEMENTS_PER_THREAD];
    [[unroll]] for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        src_keys[i] = g_hashes[data_index + i * SORT_THREADGROUP_SIZE];
    }

    for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        if (li < SORT_BINS_COUNT) {
            g_shared_local_counters[li] = 0;
        }

        ray_hash_t local_key = ray_hash_t(0xffffffff, 0xffffffff);
        if (data_index < data_count) {
            local_key = src_keys[i];
        }

        for (uint bit_shift = 0; bit_shift < SORT_BITS_PER_PASS; bit_shift += 2) {
            uint key_index = (local_key.hash >> g_params.shift) & 0xf;
            uint bit_key = (key_index >> bit_shift) & 0x3;

            uint packed_counter = 1u << (bit_key * 8);
            uint local_sum = ThreadgroupExclusiveScan(packed_counter, li);

            if (li == SORT_THREADGROUP_SIZE - 1) {
                g_shared_scratch[0] = local_sum + packed_counter;
            }

            groupMemoryBarrier(); barrier();

            packed_counter = g_shared_scratch[0];

            // Add prefix offsets for all 4 bit "keys" (packedHistogram = 0xsum2_1_0|sum1_0|sum0|0)
            packed_counter = (packed_counter << 8) + (packed_counter << 16) + (packed_counter << 24);

            // Calculate the proper offset for this thread's value
            local_sum += packed_counter;

            // Calculate target offset
            uint key_offset = (local_sum >> (bit_key * 8)) & 0xff;

            // Re-arrange the keys (store, sync, load)
            g_shared_hashes[key_offset] = local_key;
            groupMemoryBarrier(); barrier();
            local_key = g_shared_hashes[li];

            groupMemoryBarrier(); barrier();
        }

        // Need to recalculate the keyIndex on this thread now that values have been copied around the thread group
        uint key_index = (local_key.hash >> g_params.shift) & 0xf;

        // Reconstruct histogram
        atomicAdd(g_shared_local_counters[key_index], 1);

        groupMemoryBarrier(); barrier();

        // Prefix histogram
        uint histogram_prefix_sum = subgroupExclusiveAdd(li < SORT_BINS_COUNT ? g_shared_local_counters[li] : 0);

        // Broadcast prefix-sum
        if (li < SORT_BINS_COUNT) {
            g_shared_scratch[li] = histogram_prefix_sum;
        }

        // Get the global offset for this key out of the cache
        uint global_offset = g_shared_bin_offsets[key_index];

        groupMemoryBarrier(); barrier();

        // Get the local offset (at this point the keys are all in increasing order from 0 -> num bins in localID 0 -> thread group size)
        uint local_offset = li - g_shared_scratch[key_index];

        // Write to destination
        uint total_offset = global_offset + local_offset;

        if (total_offset < data_count) {
            g_out_hashes[total_offset] = local_key;
        }

        groupMemoryBarrier(); barrier();

        // Update the cached histogram for the next set of entries
        if (li < SORT_BINS_COUNT) {
            g_shared_bin_offsets[li] += g_shared_local_counters[li];
        }

        data_index += SORT_THREADGROUP_SIZE;
    }
}
