#version 450
#extension GL_GOOGLE_include_directive : require

#if defined(GL_ES) || defined(VULKAN)
    precision highp int;
    precision highp float;
#endif

#include "prepare_indir_args_interface.h"

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer Counters {
    uint g_counters[];
};
layout(std430, binding = OUT_INDIR_ARGS_SLOT) writeonly buffer IndirArgs {
    uint g_out_indir_args[];
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    { // secondary rays
        uint ray_count = g_counters[0];

        g_out_indir_args[0] = (ray_count + 63) / 64;
        g_out_indir_args[1] = 1;
        g_out_indir_args[2] = 1;

        g_counters[0] = 0;
        g_counters[1] = ray_count;
    }

    { // shadow rays
        uint sh_ray_count = g_counters[2];

        g_out_indir_args[3] = (sh_ray_count + 63) / 64;
        g_out_indir_args[4] = 1;
        g_out_indir_args[5] = 1;

        g_counters[2] = 0;
        g_counters[3] = sh_ray_count;
    }
}
