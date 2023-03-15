#version 450
#extension GL_GOOGLE_include_directive : require

#include "postprocess_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG0_SLOT, rgba32f) uniform readonly image2D g_in_img0;
layout(binding = IN_IMG1_SLOT, rgba32f) uniform readonly image2D g_in_img1;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
layout(binding = OUT_RAW_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_raw_img;
layout(binding = OUT_VARIANCE_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_variance_img;

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

vec4 clamp_and_gamma_correct(vec4 col) {
    col *= g_params.exposure;

    [[unroll]] for (int i = 0; i < 3 && g_params.srgb != 0; ++i) {
        if (col[i] < 0.0031308) {
            col[i] = 12.92 * col[i];
        } else {
            col[i] = 1.055 * pow(col[i], (1.0 / 2.4)) - 0.055;
        }
    }

    if (g_params._clamp != 0) {
        col.xyz = clamp(col.xyz, vec3(0.0), vec3(1.0));
    }

    return col;
}

void main() {
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    ivec2 gi = ivec2(gl_GlobalInvocationID.xy);

    // Mix two half-buffers together
    vec4 img0 = imageLoad(g_in_img0, gi);
    vec4 img1 = imageLoad(g_in_img1, gi);

    vec4 untonemapped_res = (g_params.img0_weight * img0) + (g_params.img1_weight * img1);
    imageStore(g_out_raw_img, gi, untonemapped_res);

    vec4 tonemapped_res = clamp_and_gamma_correct(untonemapped_res);
    imageStore(g_out_img, gi, tonemapped_res);

    img0 = clamp_and_gamma_correct(img0);
    img1 = clamp_and_gamma_correct(img1);

    vec4 variance = 0.5 * (img0 - img1) * (img0 - img1);
    imageStore(g_out_variance_img, gi, variance);
}
