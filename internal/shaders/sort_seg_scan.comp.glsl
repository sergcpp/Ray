#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_seg_scan_interface.h"
#include "common.glsl"

layout(std430, binding = VALUES_BUF_SLOT) readonly buffer Values {
    uint g_values[];
};

layout(std430, binding = FLAGS_BUF_SLOT) readonly buffer Flags {
    uint g_flags[];
};

layout(std430, binding = OUT_SCAN_VALUES_BUF_SLOT) writeonly buffer OutScanValues {
    uint g_out_scan_values[];
};

layout(std430, binding = OUT_PARTIAL_SUMS_BUF_SLOT) writeonly buffer OutPartialSums {
    uint g_out_partial_sums[];
};

layout(std430, binding = OUT_PARTIAL_FLAGS_BUF_SLOT) writeonly buffer OutPartialFlags {
    uint g_out_partial_flags[];
};

shared uint g_temp[2][SEG_SCAN_PORTION];
shared uint g_temp_flags[2][SEG_SCAN_PORTION];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    const int li = int(gl_LocalInvocationID.x);

    int pout = 0, pin = 1;

#if EXCLUSIVE_SCAN
    g_temp[pout][li] = (li == 0) ? 0 : g_values[gi - 1];
#else
    g_temp[pout][li] = g_values[gi];
#endif
    g_temp_flags[pout][li] = g_flags[gi];
    g_temp[pin][li] = 0;
    g_temp_flags[pin][li] = 0;

    groupMemoryBarrier(); barrier();

    for (int offset = 1; offset < SEG_SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            g_temp_flags[pout][li] = g_temp_flags[pin][li] | g_temp_flags[pin][li - offset];
            g_temp[pout][li] = g_temp_flags[pin][li] != 0 ? g_temp[pin][li] : (g_temp[pin][li] + g_temp[pin][li - offset]);
        } else {
            g_temp_flags[pout][li] = g_temp_flags[pin][li];
            g_temp[pout][li] = g_temp[pin][li];
        }

        groupMemoryBarrier(); barrier();
    }

    g_out_scan_values[gi] = g_temp[pout][li];

    if (li == LOCAL_GROUP_SIZE_X - 1) {
#if EXCLUSIVE_SCAN
        g_out_partial_sums[gl_WorkGroupID.x] = g_temp[pout][li] + g_values[gi];
#else
        g_out_partial_sums[gl_WorkGroupID.x] = g_temp[pout][li];
#endif
        g_out_partial_flags[gl_WorkGroupID.x] = g_temp_flags[pout][li];
    }
}
