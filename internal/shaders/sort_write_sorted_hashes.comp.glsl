#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_write_sorted_hashes_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = OFFSETS_BUF_SLOT) readonly buffer Offsets {
    uint g_offsets[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = HASHES_BUF_SLOT) readonly buffer Chunks {
    ray_hash_t g_hashes[];
};

layout(std430, binding = OUT_HASHES_BUF_SLOT) writeonly buffer OutChunks {
    ray_hash_t g_out_hashes[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= g_counters[g_params.counter]) {
        return;
    }

    uint local_offsets[0x10];
    for (int i = 0; i < 0x10; i++) {
        local_offsets[i] = g_offsets[i * g_counters[g_params.counter] + gi];
    }

    for (int i = 0; i < LOCAL_GROUP_SIZE_X; ++i) {
        if ((gi * LOCAL_GROUP_SIZE_X + i) < g_counters[g_params.chunks_counter]) {
            uint index = (g_hashes[gi * LOCAL_GROUP_SIZE_X + i].hash >> g_params.shift) & 0xF;
            uint local_index = local_offsets[index]++;

            g_out_hashes[local_index] = g_hashes[gi * LOCAL_GROUP_SIZE_X + i];
        }
    }
}
