#version 450
#extension GL_GOOGLE_include_directive : require

#if defined(GL_ES) || defined(VULKAN)
    precision highp int;
    precision highp float;
#endif

#include "sort_prepare_indir_args_interface.h"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HEAD_FLAGS_BUF_SLOT) readonly buffer HeadFlags {
    uint g_head_flags[];
};

layout(std430, binding = SCAN_VALUES_BUF_SLOT) readonly buffer ScanValues {
    uint g_scan_values[];
};

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer Counters {
    uint g_counters[];
};
layout(std430, binding = OUT_INDIR_ARGS_SLOT) writeonly buffer IndirArgs {
    uint g_out_indir_args[];
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint chunks_count = g_scan_values[g_counters[g_params.in_counter] - 1] + g_head_flags[g_counters[g_params.in_counter] - 1];

    uint group_count = (chunks_count + 63) / 64;
    g_out_indir_args[3 * (g_params.indir_args_index + 0) + 0] = group_count;
    g_out_indir_args[3 * (g_params.indir_args_index + 0) + 1] = 1;
    g_out_indir_args[3 * (g_params.indir_args_index + 0) + 2] = 1;

    g_counters[g_params.out_counter] = chunks_count;

    uint chunks_scan_count = group_count;
    for (int i = 0; i < 3; ++i) {
        g_counters[g_params.out_counter + 1 + i] = chunks_scan_count;
        g_out_indir_args[3 * (g_params.indir_args_index + 1 + i) + 0] = (chunks_scan_count + 63) / 64;
        g_out_indir_args[3 * (g_params.indir_args_index + 1 + i) + 1] = 1;
        g_out_indir_args[3 * (g_params.indir_args_index + 1 + i) + 2] = 1;
        chunks_scan_count = (chunks_scan_count + 63) / 64;
    }

    uint counters_count = group_count * 0x10;
    g_out_indir_args[3 * (g_params.indir_args_index + 4) + 0] = (counters_count + 63) / 64;
    g_out_indir_args[3 * (g_params.indir_args_index + 4) + 1] = 1;
    g_out_indir_args[3 * (g_params.indir_args_index + 4) + 2] = 1;

    g_counters[g_params.out_counter + 4] = counters_count;

    uint scan_count = (counters_count + 63) / 64;
    for (int i = 0; i < 4; ++i) {
        g_counters[g_params.out_counter + 5 + i] = scan_count;
        g_out_indir_args[3 * (g_params.indir_args_index + 5 + i) + 0] = (scan_count + 63) / 64;
        g_out_indir_args[3 * (g_params.indir_args_index + 5 + i) + 1] = 1;
        g_out_indir_args[3 * (g_params.indir_args_index + 5 + i) + 2] = 1;
        scan_count = (scan_count + 63) / 64;
    }
}
