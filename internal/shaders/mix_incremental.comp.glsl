#version 450
#extension GL_GOOGLE_include_directive : require

#include "mix_incremental_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG1_SLOT, rgba32f) uniform readonly image2D g_in_img1;
layout(binding = IN_IMG2_SLOT, rgba32f) uniform readonly image2D g_in_img2;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
#if OUTPUT_BASE_COLOR
    layout(binding = IN_TEMP_BASE_COLOR_SLOT, rgba32f) uniform readonly image2D g_temp_base_color;
    layout(binding = OUT_BASE_COLOR_IMG_SLOT, rgba32f) uniform image2D g_out_base_color_img;
#endif
#if OUTPUT_DEPTH_NORMALS
    layout(binding = IN_TEMP_DEPTH_NORMALS_SLOT, rgba32f) uniform readonly image2D g_temp_depth_normals_img;
    layout(binding = OUT_DEPTH_NORMALS_IMG_SLOT, rgba32f) uniform image2D g_out_depth_normals_img;
#endif

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    ivec2 icoord = ivec2(gl_GlobalInvocationID.xy);

    vec3 col1 = imageLoad(g_in_img1, icoord).rgb;
    vec3 col2 = imageLoad(g_in_img2, icoord).rgb;

    vec3 diff = col2 - col1;
    imageStore(g_out_img, icoord, vec4(col1 + diff * g_params.k, 1.0));

#if OUTPUT_BASE_COLOR
    vec3 base_color1 = imageLoad(g_out_base_color_img, icoord).rgb;
    vec3 base_color2 = imageLoad(g_temp_base_color, icoord).rgb;
    vec3 base_color_diff = base_color2 - base_color1;
    imageStore(g_out_base_color_img, icoord, vec4(base_color1 + base_color_diff * g_params.k, 1.0));
#endif
#if OUTPUT_DEPTH_NORMALS
    vec4 depth_normals1 = imageLoad(g_out_depth_normals_img, icoord);
    vec4 depth_normals2 = imageLoad(g_temp_depth_normals_img, icoord);
    depth_normals2.xyz = clamp(depth_normals2.xyz, vec3(-1.0), vec3(1.0));
    vec4 depth_normals_diff = depth_normals2 - depth_normals1;
    imageStore(g_out_depth_normals_img, icoord, vec4(depth_normals1 + depth_normals_diff * g_params.k));
#endif
}
