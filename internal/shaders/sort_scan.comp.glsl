#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_scan_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = INPUT_BUF_SLOT) readonly buffer InputBuf {
    uint g_input[];
};

layout(std430, binding = OUT_SCAN_VALUES_BUF_SLOT) writeonly buffer OutScanValues {
    uint g_out_scan_values[];
};

layout(std430, binding = OUT_PARTIAL_SUMS_BUF_SLOT) writeonly buffer OutPartialSums {
    uint g_out_partial_sums[];
};

shared uint g_temp[2][SCAN_PORTION];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    const int li = int(gl_LocalInvocationID.x);

    int pout = 0, pin = 1;

#if EXCLUSIVE_SCAN
    g_temp[pout][li] = (li == 0) ? 0 : g_input[g_params.stride * (gi - 1) + g_params.offset];
#else
    g_temp[pout][li] = g_input[g_params.stride * gi + g_params.offset];
#endif
    g_temp[pin][li] = 0;

    groupMemoryBarrier(); barrier();

    for (int offset = 1; offset < SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            g_temp[pout][li] = g_temp[pin][li] + g_temp[pin][li - offset];
        } else {
            g_temp[pout][li] = g_temp[pin][li];
        }

        groupMemoryBarrier(); barrier();
    }

    g_out_scan_values[gi] = g_temp[pout][li];

    if (li == LOCAL_GROUP_SIZE_X - 1) {
#if EXCLUSIVE_SCAN
        g_out_partial_sums[gl_WorkGroupID.x] = g_temp[pout][li] + g_input[g_params.stride * gi + g_params.offset];
#else
        g_out_partial_sums[gl_WorkGroupID.x] = g_temp[pout][li];
#endif
    }
}
