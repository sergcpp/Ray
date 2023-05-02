#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_init_skeleton_and_head_flags_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = SCAN_VALUES_BUF_SLOT) readonly buffer ScanValues {
    uint g_scan_values[];
};

layout(std430, binding = CHUNKS_BUF_SLOT) readonly buffer Chunks {
    ray_chunk_t g_chunks[];
};

layout(std430, binding = OUT_SKELETON_BUF_SLOT) writeonly buffer OutSkeletonBuf {
    uint g_skeleton[];
};

layout(std430, binding = OUT_HEAD_FLAGS_BUF_SLOT) writeonly buffer OutHeadFlagsBuf {
    uint g_head_flags[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= g_counters[g_params.counter]) {
        return;
    }

    uint index = g_scan_values[gi];
    g_skeleton[index] = g_chunks[gi].base;
    g_head_flags[index] = 1;
}
