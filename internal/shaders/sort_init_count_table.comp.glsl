#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_init_count_table_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = CHUNKS_BUF_SLOT) readonly buffer RayChunks {
    ray_chunk_t g_chunks[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_COUNT_TABLE_BUF_SLOT) writeonly buffer OutBuf {
    uint g_count_table[];
};

shared uint g_shared_counters[0x10];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    const int li = int(gl_LocalInvocationID.x);
    for (int i = li; i < 0x10; i += LOCAL_GROUP_SIZE_X) {
        g_shared_counters[i] = 0;
    }
    groupMemoryBarrier(); barrier();

    if (gi < g_counters[g_params.counter]) {
        atomicAdd(g_shared_counters[(g_chunks[gi].hash >> g_params.shift) & 0xF], 1u);
    }
    groupMemoryBarrier(); barrier();

    for (int i = li; i < 0x10; i += LOCAL_GROUP_SIZE_X) {
        g_count_table[i * gl_NumWorkGroups.x + gl_WorkGroupID.x] = g_shared_counters[i];
    }
}
