#version 450
#extension GL_GOOGLE_include_directive : require

#include "mix_incremental_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_TEMP_IMG_SLOT, rgba32f) uniform readonly image2D g_temp_img;
layout(binding = IN_REQ_SAMPLES_SLOT, r16ui) uniform readonly uimage2D g_req_samples_img;

layout(binding = OUT_FULL_IMG_SLOT, rgba32f) uniform image2D g_out_full_img;
layout(binding = OUT_HALF_IMG_SLOT, rgba32f) uniform image2D g_out_half_img;
layout(binding = IN_TEMP_BASE_COLOR_SLOT, rgba32f) uniform readonly image2D g_temp_base_color;
layout(binding = OUT_BASE_COLOR_IMG_SLOT, rgba32f) uniform image2D g_out_base_color_img;
layout(binding = IN_TEMP_DEPTH_NORMALS_SLOT, rgba32f) uniform readonly image2D g_temp_depth_normals_img;
layout(binding = OUT_DEPTH_NORMALS_IMG_SLOT, rgba32f) uniform image2D g_out_depth_normals_img;

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    ivec2 icoord = ivec2(g_params.rect.xy + gl_GlobalInvocationID.xy);

    uint req_samples = imageLoad(g_req_samples_img, icoord).r;
    [[dont_flatten]] if (req_samples < g_params.iteration) {
        return;
    }

    vec3 new_val = imageLoad(g_temp_img, icoord).rgb * g_params.exposure;

    { // accumulate full buffer
        vec3 old_val = imageLoad(g_out_full_img, icoord).rgb;
        vec3 diff = new_val - old_val;
        imageStore(g_out_full_img, icoord, vec4(old_val + diff * g_params.mix_factor, 1.0));
    }

    [[dont_flatten]] if (g_params.accumulate_half_img > 0.5) {
        vec3 old_val = imageLoad(g_out_half_img, icoord).rgb;
        vec3 diff = new_val - old_val;
        imageStore(g_out_half_img, icoord, vec4(old_val + diff * g_params.half_mix_factor, 1.0));
    }

    { // accumulate base color
        vec3 base_color1 = imageLoad(g_out_base_color_img, icoord).rgb;
        vec3 base_color2 = imageLoad(g_temp_base_color, icoord).rgb;
        vec3 base_color_diff = base_color2 - base_color1;
        imageStore(g_out_base_color_img, icoord, vec4(base_color1 + base_color_diff * g_params.mix_factor, 1.0));
    }
    { // accumulate depth-normals
        vec4 depth_normals1 = imageLoad(g_out_depth_normals_img, icoord);
        vec4 depth_normals2 = imageLoad(g_temp_depth_normals_img, icoord);
        depth_normals2.xyz = clamp(depth_normals2.xyz, vec3(-1.0), vec3(1.0));
        vec4 depth_normals_diff = depth_normals2 - depth_normals1;
        imageStore(g_out_depth_normals_img, icoord, vec4(depth_normals1 + depth_normals_diff * g_params.mix_factor));
    }
}
