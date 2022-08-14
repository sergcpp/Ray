#version 450
#extension GL_GOOGLE_include_directive : require

#include "postprocess_interface.glsl"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG_SLOT, rgba32f) uniform readonly image2D g_in_img;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    ivec2 icoord = ivec2(gl_GlobalInvocationID.xy);

    vec4 col = imageLoad(g_in_img, icoord);

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

    imageStore(g_out_img, icoord, col);
}
