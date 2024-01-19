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

void main() {
    const int gi = int(gl_GlobalInvocationID.x), li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int blocks_count = int(g_counters[g_params.counter]);

#if ADD_PARTIAL_SUMS
    const int scan_count = blocks_count;
    const int groups_per_bin = (blocks_count + BlockSize - 1) / BlockSize;

    const int bin_id = wi / groups_per_bin;
    const int bin_offset = bin_id * blocks_count;

    const int base_index = (wi % groups_per_bin) * SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
#else
    const int scan_count = SORT_BINS_COUNT * ((blocks_count + BlockSize - 1) / BlockSize);

    const int bin_offset = 0;
    const int base_index = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE * wi;
#endif

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

    threadgroup_sum = ThreadgroupExclusiveScan(threadgroup_sum, li);

#if ADD_PARTIAL_SUMS
    uint partial_sum = g_scan_scratch[wi];
#else
    uint partial_sum = 0;
#endif

    for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        g_shared_values[i][li] += threadgroup_sum;
    }

    groupMemoryBarrier(); barrier();

    for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        const int data_index = base_index + i * SORT_THREADGROUP_SIZE + li;

        const int col = ((i * SORT_THREADGROUP_SIZE) + li) / SORT_ELEMENTS_PER_THREAD;
        const int row = ((i * SORT_THREADGROUP_SIZE) + li) % SORT_ELEMENTS_PER_THREAD;

        if (data_index < scan_count) {
            g_out_scan_values[bin_offset + data_index] = g_shared_values[row][col] + partial_sum;
        }
    }
}
