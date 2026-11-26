#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

#include "sort_scan_interface.h"
#include "common.glsl"
#include "sort_common.h"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = INPUT_BUF_SLOT) readonly buffer InputBuf {
    uint g_input[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = SCRATCH_BUF_SLOT) readonly buffer ScratchBuf {
    uint g_scan_scratch[];
};

layout(std430, binding = OUT_SCAN_VALUES_BUF_SLOT) writeonly buffer OutScanValues {
    uint g_out_scan_values[];
};

layout (local_size_x = SORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

shared uint g_shared_sums[SORT_THREADGROUP_SIZE];

uint ThreadgroupReduce(uint local_sum, uint li) {
    uint subgroup_reduced = subgroupAdd(local_sum);

    uint subgroup_id = li / gl_SubgroupSize;
    if (subgroupElect()) {
        g_shared_sums[subgroup_id] = subgroup_reduced;
    }

    groupMemoryBarrier(); barrier();

    if (subgroup_id == 0) {
        subgroup_reduced = subgroupAdd(li < (SORT_THREADGROUP_SIZE / gl_SubgroupSize) ? g_shared_sums[li] : 0);
    }

    return subgroup_reduced;
}

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

shared uint g_shared_values[SORT_ELEMENTS_PER_THREAD][SORT_THREADGROUP_SIZE];
shared uint g_chunk_total;
shared uint g_prefix_total;

void main() {
    const int li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int blocks_count = int(g_counters[g_params.counter]);

    const int scan_count = blocks_count;
    const int groups_per_bin = (blocks_count + BlockSize - 1) / BlockSize;

    const int bin_id = wi / groups_per_bin;
    const int bin_offset = bin_id * blocks_count;

    const int chunk_begin = (wi % groups_per_bin) * BlockSize;
    const int chunk_end = chunk_begin + BlockSize;

    // Inline exclusive prefix of the unscanned reduce table: the old single-workgroup
    // sort_scan dispatch is fused here (the table is 256 bins x a few chunks at most).
    uint prefix = 0;
    for (int i = li; i < wi; i += SORT_THREADGROUP_SIZE) {
        prefix += g_scan_scratch[i];
    }
    prefix = ThreadgroupReduce(prefix, li);

    groupMemoryBarrier(); barrier();
    if (li == 0) {
        g_prefix_total = prefix;
    }
    groupMemoryBarrier(); barrier();

    uint partial_sum = g_prefix_total;

    for (int base_index = chunk_begin; base_index < chunk_end; base_index += BlockSize) {
        for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
            const int data_index = base_index + i * SORT_THREADGROUP_SIZE + li;

            const int col = ((i * SORT_THREADGROUP_SIZE) + li) / SORT_ELEMENTS_PER_THREAD;
            const int row = ((i * SORT_THREADGROUP_SIZE) + li) % SORT_ELEMENTS_PER_THREAD;
            g_shared_values[row][col] = (data_index < scan_count) ? g_input[bin_offset + data_index] : 0;
        }

        groupMemoryBarrier(); barrier();

        uint threadgroup_sum = 0;
        for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
            uint tmp = g_shared_values[i][li];
            g_shared_values[i][li] = threadgroup_sum;
            threadgroup_sum += tmp;
        }
        const uint local_total = threadgroup_sum;

        threadgroup_sum = ThreadgroupExclusiveScan(threadgroup_sum, li);

        groupMemoryBarrier(); barrier();
        if (li == SORT_THREADGROUP_SIZE - 1) {
            g_chunk_total = threadgroup_sum + local_total;
        }
        groupMemoryBarrier(); barrier();

        for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
            g_shared_values[i][li] += threadgroup_sum + partial_sum;
        }

        groupMemoryBarrier(); barrier();

        for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
            const int data_index = base_index + i * SORT_THREADGROUP_SIZE + li;

            const int col = ((i * SORT_THREADGROUP_SIZE) + li) / SORT_ELEMENTS_PER_THREAD;
            const int row = ((i * SORT_THREADGROUP_SIZE) + li) % SORT_ELEMENTS_PER_THREAD;

            if (data_index < scan_count) {
                g_out_scan_values[bin_offset + data_index] = g_shared_values[row][col];
            }
        }

        partial_sum += g_chunk_total;

        groupMemoryBarrier(); barrier();
    }
}
