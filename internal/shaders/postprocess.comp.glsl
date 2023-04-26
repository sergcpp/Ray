#version 450
#extension GL_GOOGLE_include_directive : require

#include "postprocess_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

#define DEBUG_ADAPTIVE_SAMPLING 0

vec3 heatmap(float u) {
    float level = u * PI / 2.0;
    return vec3(sin(level), sin(2.0 * level), cos(level));
}

layout(binding = IN_IMG0_SLOT, rgba32f) uniform readonly image2D g_in_img0;
layout(binding = IN_IMG1_SLOT, rgba32f) uniform readonly image2D g_in_img1;
layout(binding = TONEMAP_LUT_SLOT) uniform sampler3D g_tonemap_lut;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
layout(binding = OUT_RAW_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_raw_img;
layout(binding = OUT_VARIANCE_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_variance_img;
layout(binding = OUT_REQ_SAMPLES_IMG_SLOT, r16ui) uniform uimage2D g_out_req_samples_img;

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    ivec2 gi = ivec2(g_params.rect.xy + gl_GlobalInvocationID.xy);

    // Mix two half-buffers together
    vec4 img0 = imageLoad(g_in_img0, gi);
    vec4 img1 = imageLoad(g_in_img1, gi);

    img0.xyz *= g_params.exposure;
    img1.xyz *= g_params.exposure;

    vec4 untonemapped_res = (g_params.img0_weight * img0) + (g_params.img1_weight * img1);
#if DEBUG_ADAPTIVE_SAMPLING
    uint req_samples = imageLoad(g_out_req_samples_img, gi).r;
    vec3 debug_color = heatmap(float(req_samples) / g_params.iteration);
    if ((g_params.iteration % 2) != 0 /*&& req_samples >= g_params.iteration*/) {
        untonemapped_res.rgb = mix(untonemapped_res.rgb, debug_color, vec3(0.5));
    }
#endif
    imageStore(g_out_raw_img, gi, untonemapped_res);

    vec4 tonemapped_res;
    [[dont_flatten]] if (g_params.tonemap_mode == 0) {
        tonemapped_res = TonemapStandard(g_params.inv_gamma, untonemapped_res);
    } else {
        tonemapped_res = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, untonemapped_res);
    }
#if DEBUG_ADAPTIVE_SAMPLING
    if ((g_params.iteration % 2) != 0 /*&& req_samples >= g_params.iteration*/) {
        tonemapped_res.rgb = mix(tonemapped_res.rgb, debug_color, vec3(0.5));
    }
#endif
    imageStore(g_out_img, gi, tonemapped_res);

    img0 = reversible_tonemap(img0);
    img1 = reversible_tonemap(img1);

    vec4 variance = 0.5 * (img0 - img1) * (img0 - img1);
    imageStore(g_out_variance_img, gi, variance);

    if (any(greaterThanEqual(variance, vec4(g_params.variance_threshold)))) {
        imageStore(g_out_req_samples_img, gi, uvec4(g_params.iteration + 1));
    }
}
