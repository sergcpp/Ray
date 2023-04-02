#version 450
#extension GL_GOOGLE_include_directive : require

#include "nlm_filter_interface.h"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG_SLOT) uniform sampler2D g_in_img;
layout(binding = VARIANCE_IMG_SLOT) uniform sampler2D g_variance_img;

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;

#define USE_SHARED_MEMORY 1
shared uint g_temp_color0[16][16], g_temp_color1[16][16];
shared uint g_temp_variance0[16][16], g_temp_variance1[16][16];

const int WINDOW_SIZE = 7;
const int NEIGHBORHOOD_SIZE = 3;

const int WindowRadius = (WINDOW_SIZE - 1) / 2;
const float PatchDistanceNormFactor = NEIGHBORHOOD_SIZE * NEIGHBORHOOD_SIZE;
const int NeighborRadius = (NEIGHBORHOOD_SIZE - 1) / 2;

layout(local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    ivec2 gi = ivec2(gl_GlobalInvocationID.xy), li = ivec2(gl_LocalInvocationID.xy);
    vec2 tex_coord = (vec2(gi) + vec2(0.5)) / vec2(g_params.img_size);

#if USE_SHARED_MEMORY
    //
    // Load color and variance into shared memory (16x16 region)
    //
    vec4 c00 = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, -4));
    g_temp_color0[0 + li.y][0 + li.x] = packHalf2x16(c00.xy);
    g_temp_color1[0 + li.y][0 + li.x] = packHalf2x16(c00.zw);
    vec4 c01 = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, -4));
    g_temp_color0[0 + li.y][8 + li.x] = packHalf2x16(c01.xy);
    g_temp_color1[0 + li.y][8 + li.x] = packHalf2x16(c01.zw);
    vec4 c10 = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, +4));
    g_temp_color0[8 + li.y][0 + li.x] = packHalf2x16(c10.xy);
    g_temp_color1[8 + li.y][0 + li.x] = packHalf2x16(c10.zw);
    vec4 c11 = textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, +4));
    g_temp_color0[8 + li.y][8 + li.x] = packHalf2x16(c11.xy);
    g_temp_color1[8 + li.y][8 + li.x] = packHalf2x16(c11.zw);

    vec4 v00 = textureLodOffset(g_variance_img, tex_coord, 0.0, ivec2(-4, -4));
    g_temp_variance0[0 + li.y][0 + li.x] = packHalf2x16(v00.xy);
    g_temp_variance1[0 + li.y][0 + li.x] = packHalf2x16(v00.zw);
    vec4 v01 = textureLodOffset(g_variance_img, tex_coord, 0.0, ivec2(+4, -4));
    g_temp_variance0[0 + li.y][8 + li.x] = packHalf2x16(v01.xy);
    g_temp_variance1[0 + li.y][8 + li.x] = packHalf2x16(v01.zw);
    vec4 v10 = textureLodOffset(g_variance_img, tex_coord, 0.0, ivec2(-4, +4));
    g_temp_variance0[8 + li.y][0 + li.x] = packHalf2x16(v10.xy);
    g_temp_variance1[8 + li.y][0 + li.x] = packHalf2x16(v10.zw);
    vec4 v11 = textureLodOffset(g_variance_img, tex_coord, 0.0, ivec2(+4, +4));
    g_temp_variance0[8 + li.y][8 + li.x] = packHalf2x16(v11.xy);
    g_temp_variance1[8 + li.y][8 + li.x] = packHalf2x16(v11.zw);

    groupMemoryBarrier(); barrier();

    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    vec4 sum_output = vec4(0.0);
    float sum_weight = 0.0;

    [[unroll]] for (int k = -WindowRadius; k <= WindowRadius; ++k) {
        [[unroll]] for (int l = -WindowRadius; l <= WindowRadius; ++l) {
            vec4 distance = vec4(0.0);

            [[unroll]] for (int q = -NeighborRadius; q <= NeighborRadius; ++q) {
                [[unroll]] for (int p = -NeighborRadius; p <= NeighborRadius; ++p) {
                    vec4 ipx = vec4(unpackHalf2x16(g_temp_color0[li.y + 4 + q][li.x + 4 + p]),
                                    unpackHalf2x16(g_temp_color1[li.y + 4 + q][li.x + 4 + p]));
                    vec4 jpx = vec4(unpackHalf2x16(g_temp_color0[li.y + 4 + k + q][li.x + 4 + l + p]),
                                    unpackHalf2x16(g_temp_color1[li.y + 4 + k + q][li.x + 4 + l + p]));

                    vec4 ivar = vec4(unpackHalf2x16(g_temp_variance0[li.y + 4 + q][li.x + 4 + p]),
                                     unpackHalf2x16(g_temp_variance1[li.y + 4 + q][li.x + 4 + p]));
                    vec4 jvar = vec4(unpackHalf2x16(g_temp_variance0[li.y + 4 + k + q][li.x + 4 + l + p]),
                                     unpackHalf2x16(g_temp_variance1[li.y + 4 + k + q][li.x + 4 + l + p]));
                    vec4 min_var = min(ivar, jvar);

                    distance += ((ipx - jpx) * (ipx - jpx) - g_params.alpha * (ivar + min_var)) /
                                (0.0001 + g_params.damping * g_params.damping * (ivar + jvar));
                }
            }

            float patch_distance =
                0.25 * PatchDistanceNormFactor * (distance.x + distance.y + distance.z + distance.w);

            float weight = exp(-max(0.0, patch_distance));

            sum_output += vec4(unpackHalf2x16(g_temp_color0[li.y + 4 + k][li.x + 4 + l]),
                               unpackHalf2x16(g_temp_color1[li.y + 4 + k][li.x + 4 + l])) * weight;
            sum_weight += weight;
        }
    }

    [[flatten]] if (sum_weight != 0.0) {
        sum_output /= sum_weight;
    }
#else
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    int ix = gi.x, iy = gi.y;
    vec2 inv_res = vec2(1.0) / vec2(g_params.img_size);

    vec4 sum_output = vec4(0.0);
    float sum_weight = 0.0;

    [[unroll]] for (int k = -WindowRadius; k <= WindowRadius; ++k) {
        const int jy = iy + k;

        [[unroll]] for (int l = -WindowRadius; l <= WindowRadius; ++l) {
            const int jx = ix + l;

            vec4 distance = vec4(0.0);

            [[unroll]] for (int q = -NeighborRadius; q <= NeighborRadius; ++q) {
                [[unroll]] for (int p = -NeighborRadius; p <= NeighborRadius; ++p) {
                    vec4 ipx = textureLod(g_in_img, tex_coord + vec2(p, q) * inv_res, 0.0);
                    vec4 jpx = textureLod(g_in_img, tex_coord + vec2(l + p, k + q) * inv_res, 0.0);

                    vec4 ivar = textureLod(g_variance_img, tex_coord + vec2(p, q) * inv_res, 0.0);
                    vec4 jvar = textureLod(g_variance_img, tex_coord + vec2(l + p, k + q) * inv_res, 0.0);
                    vec4 min_var = min(ivar, jvar);

                    distance += ((ipx - jpx) * (ipx - jpx) - g_params.alpha * (ivar + min_var)) /
                                (0.0001 + g_params.damping * g_params.damping * (ivar + jvar));
                }
            }

            float patch_distance =
                0.25 * PatchDistanceNormFactor * (distance.x + distance.y + distance.z + distance.w);

            float weight = exp(-max(0.0, patch_distance));

            sum_output += textureLod(g_in_img, tex_coord + vec2(l, k) * inv_res, 0.0) * weight;
            sum_weight += weight;
        }
    }

    [[flatten]] if (sum_weight != 0.0) {
        sum_output /= sum_weight;
    }
#endif

    imageStore(g_out_img, gi, sum_output);
}
