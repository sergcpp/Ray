#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_add_partial_sums_interface.h"
#include "common.glsl"

layout(std430, binding = PART_SUMS_BUF_SLOT) readonly buffer PartSums {
    uint g_part_sums[];
};

layout(std430, binding = INOUT_BUF_SLOT) writeonly buffer InOutBuf {
    uint g_inout_values[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int gi = int(gl_GlobalInvocationID.x);
    const int grp = int(gl_WorkGroupID.x);
    if (grp != 0) {
        g_inout_values[gi] += g_part_sums[grp - 1];
    }
}
