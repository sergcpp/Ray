#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_init_count_table_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HASHES_BUF_SLOT) readonly buffer RayChunks {
    ray_hash_t g_hashes[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_COUNT_TABLE_BUF_SLOT) writeonly buffer OutBuf {
    uint g_count_table[];
};

shared uint g_shared_counters[0x10 * SORT_THREADGROUP_SIZE];

layout (local_size_x = SORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);
    for (int i = 0; i < 0x10; ++i) {
        g_shared_counters[i * SORT_THREADGROUP_SIZE + li] = 0;
    }
    groupMemoryBarrier(); barrier();

    const uint data_count = g_counters[g_params.counter];

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int block_count = int(data_count + BlockSize - 1) / BlockSize;
    const int block_start = BlockSize * wi;

    int data_index = block_start + li;

    uint hashes[SORT_ELEMENTS_PER_THREAD];
    [[unroll]] for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        if (data_index + i * SORT_THREADGROUP_SIZE < data_count) {
            hashes[i] = g_hashes[data_index + i * SORT_THREADGROUP_SIZE].hash;
        }
    }

    for (int i = 0; i < SORT_ELEMENTS_PER_THREAD; ++i) {
        if (data_index < data_count) {
            uint local_key = (hashes[i] >> g_params.shift) & 0xf;
            atomicAdd(g_shared_counters[local_key * SORT_THREADGROUP_SIZE + li], 1u);
            data_index += SORT_THREADGROUP_SIZE;
        }
    }

    groupMemoryBarrier(); barrier();

    if (li < 0x10) {
        uint sum = 0;
        for (int i = 0; i < SORT_THREADGROUP_SIZE; ++i) {
            sum += g_shared_counters[li * SORT_THREADGROUP_SIZE + i];
        }
        g_count_table[li * block_count + wi] = sum;
    }
}
