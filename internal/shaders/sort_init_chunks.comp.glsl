#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_init_chunks_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

#if HASH_AND_BASE
layout(std430, binding = HASH_VALUES_BUF_SLOT) readonly buffer HashValues {
    uint g_hash_values[];
};

layout(std430, binding = HEAD_FLAGS_BUF_SLOT) readonly buffer HeadFlags {
    uint g_head_flags[];
};

layout(std430, binding = SCAN_VALUES_BUF_SLOT) readonly buffer ScanValues {
    uint g_scan_values[];
};
#endif

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = INOUT_CHUNKS_BUF_SLOT) buffer OutChunks {
    ray_chunk_t g_inout_chunks[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint gi = gl_GlobalInvocationID.x;
    if (gi >= g_counters[g_params.chunks_counter]) {
        return;
    }
#if HASH_AND_BASE
    if (g_head_flags[gi] != 0) {
        g_inout_chunks[g_scan_values[gi]].hash = g_hash_values[gi];
        g_inout_chunks[g_scan_values[gi]].base = gi;
    }
#else // count
    if (gi == g_counters[g_params.chunks_counter] - 1) {
        g_inout_chunks[gi].size = g_counters[g_params.rays_counter] - g_inout_chunks[gi].base;
    } else {
        g_inout_chunks[gi].size = g_inout_chunks[gi + 1].base - g_inout_chunks[gi].base;
    }
#endif
}
