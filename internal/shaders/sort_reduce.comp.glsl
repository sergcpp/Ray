#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

#include "sort_reduce_interface.h"
#include "common.glsl"
#include "sort_common.h"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = INPUT_BUF_SLOT) readonly buffer InputBuf {
    uint g_input[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_REDUCE_TABLE_BUF_SLOT) writeonly buffer OutReduceTable {
    uint g_out_reduce_table[];
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

void main() {
    const int li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int block_count = int(g_counters[g_params.counter]);
    const int groups_per_bin = (block_count + BlockSize - 1) / BlockSize;

    const int bin_id = wi / groups_per_bin;
    const int bin_offset = bin_id * block_count;

    const int base_index = (wi % groups_per_bin) * BlockSize;

    uint threadgroup_sum = 0;
    for (uint i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        uint data_index = base_index + i * SORT_THREADGROUP_SIZE + li;
        threadgroup_sum += (data_index < block_count) ? g_input[bin_offset + data_index] : 0;
    }

    threadgroup_sum = ThreadgroupReduce(threadgroup_sum, li);

    if (li == 0) {
        g_out_reduce_table[wi] = threadgroup_sum;
    }
}
