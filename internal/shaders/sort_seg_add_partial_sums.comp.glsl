#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_seg_add_partial_sums_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = PART_SUMS_BUF_SLOT) readonly buffer PartSums {
    uint g_part_sums[];
};

layout(std430, binding = PART_FLAGS_BUF_SLOT) readonly buffer PartFlags {
    uint g_part_flags[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = INOUT_BUF_SLOT) buffer InOutBuf {
    uint g_inout_values[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= g_counters[g_params.counter]) {
        return;
    }

    uint flag = 0;
    for (int i = 0; i < LOCAL_GROUP_SIZE_X && gi != 0; ++i) {
        flag |= g_part_flags[gi * LOCAL_GROUP_SIZE_X + i];
        g_inout_values[gi * LOCAL_GROUP_SIZE_X + i] =
            flag != 0 ? g_inout_values[gi * LOCAL_GROUP_SIZE_X + i]
                      : (g_inout_values[gi * LOCAL_GROUP_SIZE_X + i] + g_part_sums[gi - 1]);
    }
}
