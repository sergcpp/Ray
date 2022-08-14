#version 450
#extension GL_GOOGLE_include_directive : require

#include "mix_incremental_interface.glsl"
#include "types.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG1_SLOT, rgba32f) uniform readonly image2D g_in_img1;
layout(binding = IN_IMG2_SLOT, rgba32f) uniform readonly image2D g_in_img2;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;

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
}
