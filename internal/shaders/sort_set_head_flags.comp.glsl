#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_set_head_flags_interface.h"
#include "common.glsl"

layout(std430, binding = INPUT_BUF_SLOT) readonly buffer InputBuf {
    uint g_input[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_HEAD_FLAGS_BUF_SLOT) writeonly buffer OutHeadFlags {
    uint g_out_head_flags[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= g_counters[1]) {
        return;
    }

    if (gi == 0) {
        g_out_head_flags[gi] = 1;
    } else {
        g_out_head_flags[gi] = (g_input[gi] != g_input[gi - 1]) ? 1 : 0;
    }
}
