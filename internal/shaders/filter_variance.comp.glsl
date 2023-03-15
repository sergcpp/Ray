#version 450
#extension GL_GOOGLE_include_directive : require

#include "filter_variance_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG_SLOT) uniform sampler2D g_in_img;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;

shared vec4 g_temp_variance0[16][16], g_temp_variance1[16][8];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    ivec2 gi = ivec2(gl_GlobalInvocationID.xy), li = ivec2(gl_LocalInvocationID.xy);

    //
    // Load variance into shared memory (16x16 region)
    //
    vec2 tex_coord = (vec2(gi) + vec2(0.5)) / vec2(g_params.img_size);
    g_temp_variance0[0 + li.y][0 + li.x] = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, -4));
    g_temp_variance0[0 + li.y][8 + li.x] = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, -4));
    g_temp_variance0[8 + li.y][0 + li.x] = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, +4));
    g_temp_variance0[8 + li.y][8 + li.x] = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, +4));

    groupMemoryBarrier(); barrier();

    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    //
    // Filter variance horizontally
    //
    const float GaussWeights[] = {0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162};

    [[unroll]] for (int j = 0; j < 16; j += 8) {
        vec4 center_val = g_temp_variance0[li.y + j][li.x + 4];
        vec4 res = center_val * GaussWeights[0];
        [[unroll]] for (int i = 1; i < 5; ++i) {
            res += g_temp_variance0[li.y + j][li.x + 4 + i] * GaussWeights[i];
            res += g_temp_variance0[li.y + j][li.x + 4 - i] * GaussWeights[i];
        }
        res = max(res, center_val);
        g_temp_variance1[li.y + j][li.x] = res;
    }

    groupMemoryBarrier(); barrier();

    //
    // Filter variance vertically
    //
    vec4 center_val = g_temp_variance1[li.y + 4][li.x];
    vec4 res_variance = center_val * GaussWeights[0];
    [[unroll]] for (int i = 1; i < 5; ++i) {
        res_variance += g_temp_variance1[li.y + 4 + i][li.x] * GaussWeights[i];
        res_variance += g_temp_variance1[li.y + 4 - i][li.x] * GaussWeights[i];
    }

    res_variance = max(res_variance, center_val);
    imageStore(g_out_img, gi, res_variance);
}
