#version 450
#extension GL_GOOGLE_include_directive : require

#include "prepare_indir_args_interface.h"
#include "sort_common.h"

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_INDIR_ARGS_SLOT) writeonly buffer IndirArgs {
    uint g_out_indir_args[];
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    // store previous value
    g_counters[8] = g_counters[1];
    { // secondary rays
        uint ray_count = g_counters[0];

        g_out_indir_args[0] = (ray_count + 63) / 64;
        g_out_indir_args[1] = 1;
        g_out_indir_args[2] = 1;

        g_out_indir_args[3] = ray_count;
        g_out_indir_args[4] = 1;
        g_out_indir_args[5] = 1;

        { // arguments for sorting
            const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
            uint blocks_count = (ray_count + BlockSize - 1) / BlockSize;
            g_counters[6] = ray_count;
            g_out_indir_args[18] = blocks_count;
            g_out_indir_args[19] = 1;
            g_out_indir_args[20] = 1;

            g_counters[7] = blocks_count;
            g_out_indir_args[21] = SORT_BINS_COUNT * ((blocks_count + BlockSize - 1) / BlockSize);
            g_out_indir_args[22] = 1;
            g_out_indir_args[23] = 1;
        }

        g_counters[0] = 0;
        g_counters[1] = ray_count;
    }
    { // shadow rays
        uint sh_ray_count = g_counters[2];

        g_out_indir_args[6] = (sh_ray_count + 63) / 64;
        g_out_indir_args[7] = 1;
        g_out_indir_args[8] = 1;

        g_out_indir_args[9] = sh_ray_count;
        g_out_indir_args[10] = 1;
        g_out_indir_args[11] = 1;

        g_counters[2] = 0;
        g_counters[3] = sh_ray_count;
    }
    { // sky rays
        uint sky_ray_count = g_counters[4];

        g_out_indir_args[12] = (sky_ray_count + 63) / 64;
        g_out_indir_args[13] = 1;
        g_out_indir_args[14] = 1;

        // unused
        g_out_indir_args[15] = sky_ray_count;
        g_out_indir_args[16] = 1;
        g_out_indir_args[17] = 1;

        g_counters[4] = 0;
        g_counters[5] = sky_ray_count;
    }
}
