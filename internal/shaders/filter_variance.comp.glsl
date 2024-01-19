#version 450
#extension GL_GOOGLE_include_directive : require

#include "filter_variance_interface.h"
#include "common.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG_SLOT) uniform sampler2D g_in_img;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
layout(binding = OUT_REQ_SAMPLES_IMG_SLOT, r16ui) uniform uimage2D g_out_req_samples_img;

shared uint g_temp_variance0_0[16][16], g_temp_variance0_1[16][16];
shared uint g_temp_variance1_0[16][8], g_temp_variance1_1[16][8];

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    ivec2 gi = ivec2(g_params.rect.xy + gl_GlobalInvocationID.xy), li = ivec2(gl_LocalInvocationID.xy);

    //
    // Load variance into shared memory (16x16 region)
    //
    vec2 tex_coord = (vec2(gi) + vec2(0.5)) * g_params.inv_img_size;
    vec4 v00 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, -4)));
    g_temp_variance0_0[0 + li.y][0 + li.x] = packHalf2x16(v00.xy);
    g_temp_variance0_1[0 + li.y][0 + li.x] = packHalf2x16(v00.zw);
    vec4 v01 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, -4)));
    g_temp_variance0_0[0 + li.y][8 + li.x] = packHalf2x16(v01.xy);
    g_temp_variance0_1[0 + li.y][8 + li.x] = packHalf2x16(v01.zw);
    vec4 v10 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, +4)));
    g_temp_variance0_0[8 + li.y][0 + li.x] = packHalf2x16(v10.xy);
    g_temp_variance0_1[8 + li.y][0 + li.x] = packHalf2x16(v10.zw);
    vec4 v11 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, +4)));
    g_temp_variance0_0[8 + li.y][8 + li.x] = packHalf2x16(v11.xy);
    g_temp_variance0_1[8 + li.y][8 + li.x] = packHalf2x16(v11.zw);

    groupMemoryBarrier(); barrier();

    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    //
    // Filter variance horizontally
    //
    const float GaussWeights[] = {0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162};

    [[unroll]] for (int j = 0; j < 16; j += 8) {
        vec4 center_val = vec4(unpackHalf2x16(g_temp_variance0_0[li.y + j][li.x + 4]),
                               unpackHalf2x16(g_temp_variance0_1[li.y + j][li.x + 4]));
        vec4 res = center_val * GaussWeights[0];
        [[unroll]] for (int i = 1; i < 5; ++i) {
            res += vec4(unpackHalf2x16(g_temp_variance0_0[li.y + j][li.x + 4 + i]),
                        unpackHalf2x16(g_temp_variance0_1[li.y + j][li.x + 4 + i])) * GaussWeights[i];
            res += vec4(unpackHalf2x16(g_temp_variance0_0[li.y + j][li.x + 4 - i]),
                        unpackHalf2x16(g_temp_variance0_1[li.y + j][li.x + 4 - i])) * GaussWeights[i];
        }
        res = max(res, center_val);
        g_temp_variance1_0[li.y + j][li.x] = packHalf2x16(res.xy);
        g_temp_variance1_1[li.y + j][li.x] = packHalf2x16(res.zw);
    }

    groupMemoryBarrier(); barrier();

    //
    // Filter variance vertically
    //
    vec4 center_val = vec4(unpackHalf2x16(g_temp_variance1_0[li.y + 4][li.x]),
                           unpackHalf2x16(g_temp_variance1_1[li.y + 4][li.x]));
    vec4 res_variance = center_val * GaussWeights[0];
    [[unroll]] for (int i = 1; i < 5; ++i) {
        res_variance += vec4(unpackHalf2x16(g_temp_variance1_0[li.y + 4 + i][li.x]),
                             unpackHalf2x16(g_temp_variance1_1[li.y + 4 + i][li.x])) * GaussWeights[i];
        res_variance += vec4(unpackHalf2x16(g_temp_variance1_0[li.y + 4 - i][li.x]),
                             unpackHalf2x16(g_temp_variance1_1[li.y + 4 - i][li.x])) * GaussWeights[i];
    }

    res_variance = max(res_variance, center_val);
    imageStore(g_out_img, gi, res_variance);

    if (any(greaterThanEqual(res_variance, vec4(g_params.variance_threshold)))) {
        imageStore(g_out_req_samples_img, gi, uvec4(g_params.iteration + 1));
    }
}
