#version 450
#extension GL_GOOGLE_include_directive : require

#include "nlm_filter_interface.h"
#include "common.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(binding = IN_IMG_SLOT) uniform sampler2D g_in_img;
layout(binding = VARIANCE_IMG_SLOT) uniform sampler2D g_variance_img;
layout(binding = TONEMAP_LUT_SLOT) uniform sampler3D g_tonemap_lut;

#if USE_BASE_COLOR
    layout(binding = BASE_COLOR_IMG_SLOT) uniform sampler2D g_base_color_img;
#endif
#if USE_DEPTH_NORMAL
    layout(binding = DEPTH_NORMAL_IMG_SLOT) uniform sampler2D g_depth_normal_img;
#endif

layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
layout(binding = OUT_RAW_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_raw_img;

#define USE_SHARED_MEMORY 1
shared uint g_temp_color0[16][16], g_temp_color1[16][16];
shared uint g_temp_variance0[16][16], g_temp_variance1[16][16];
#if USE_BASE_COLOR
    shared uint g_temp_base_color[16][16];
#endif
#if USE_DEPTH_NORMAL
    shared uint g_temp_depth_normal[16][16];
#endif

const int WINDOW_SIZE = 7;
const int NEIGHBORHOOD_SIZE = 3;

const int WindowRadius = (WINDOW_SIZE - 1) / 2;
const float PatchDistanceNormFactor = NEIGHBORHOOD_SIZE * NEIGHBORHOOD_SIZE;
const int NeighborRadius = (NEIGHBORHOOD_SIZE - 1) / 2;

layout(local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    ivec2 gi = ivec2(g_params.rect.xy + gl_GlobalInvocationID.xy), li = ivec2(gl_LocalInvocationID.xy);
    vec2 tex_coord = (vec2(gi) + vec2(0.5)) * g_params.inv_img_size;

#if USE_SHARED_MEMORY
    //
    // Load color and variance into shared memory (16x16 region)
    //
    vec4 c00 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, -4)));
    g_temp_color0[0 + li.y][0 + li.x] = packHalf2x16(c00.xy);
    g_temp_color1[0 + li.y][0 + li.x] = packHalf2x16(c00.zw);
    vec4 c01 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, -4)));
    g_temp_color0[0 + li.y][8 + li.x] = packHalf2x16(c01.xy);
    g_temp_color1[0 + li.y][8 + li.x] = packHalf2x16(c01.zw);
    vec4 c10 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(-4, +4)));
    g_temp_color0[8 + li.y][0 + li.x] = packHalf2x16(c10.xy);
    g_temp_color1[8 + li.y][0 + li.x] = packHalf2x16(c10.zw);
    vec4 c11 = reversible_tonemap(textureLodOffset(g_in_img, tex_coord, 0.0, ivec2(+4, +4)));
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

#if USE_BASE_COLOR
    vec4 b00 = textureLodOffset(g_base_color_img, tex_coord, 0.0, ivec2(-4, -4));
    g_temp_base_color[0 + li.y][0 + li.x] = packUnorm4x8(b00);
    vec4 b01 = textureLodOffset(g_base_color_img, tex_coord, 0.0, ivec2(+4, -4));
    g_temp_base_color[0 + li.y][8 + li.x] = packUnorm4x8(b01);
    vec4 b10 = textureLodOffset(g_base_color_img, tex_coord, 0.0, ivec2(-4, +4));
    g_temp_base_color[8 + li.y][0 + li.x] = packUnorm4x8(b10);
    vec4 b11 = textureLodOffset(g_base_color_img, tex_coord, 0.0, ivec2(+4, +4));
    g_temp_base_color[8 + li.y][8 + li.x] = packUnorm4x8(b11);
#endif

#if USE_DEPTH_NORMAL
    vec4 n00 = textureLodOffset(g_depth_normal_img, tex_coord, 0.0, ivec2(-4, -4));
    g_temp_depth_normal[0 + li.y][0 + li.x] = packUnorm4x8(n00 * vec4(0.5, 0.5, 0.5, 0.0625) + vec4(0.5, 0.5, 0.5, 0.0));
    vec4 n01 = textureLodOffset(g_depth_normal_img, tex_coord, 0.0, ivec2(+4, -4));
    g_temp_depth_normal[0 + li.y][8 + li.x] = packUnorm4x8(n01 * vec4(0.5, 0.5, 0.5, 0.0625) + vec4(0.5, 0.5, 0.5, 0.0));
    vec4 n10 = textureLodOffset(g_depth_normal_img, tex_coord, 0.0, ivec2(-4, +4));
    g_temp_depth_normal[8 + li.y][0 + li.x] = packUnorm4x8(n10 * vec4(0.5, 0.5, 0.5, 0.0625) + vec4(0.5, 0.5, 0.5, 0.0));
    vec4 n11 = textureLodOffset(g_depth_normal_img, tex_coord, 0.0, ivec2(+4, +4));
    g_temp_depth_normal[8 + li.y][8 + li.x] = packUnorm4x8(n11 * vec4(0.5, 0.5, 0.5, 0.0625) + vec4(0.5, 0.5, 0.5, 0.0));
#endif

    groupMemoryBarrier(); barrier();

    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    vec4 sum_output = vec4(0.0);
    float sum_weight = 0.0;

    for (int k = -WindowRadius; k <= WindowRadius; ++k) {
        for (int l = -WindowRadius; l <= WindowRadius; ++l) {
            vec4 color_distance = vec4(0.0);

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

                    color_distance += ((ipx - jpx) * (ipx - jpx) - g_params.alpha * (ivar + min_var)) /
                                      (0.0001 + g_params.damping * g_params.damping * (ivar + jvar));
                }
            }

            float patch_distance =
                0.25 * PatchDistanceNormFactor * (color_distance.x + color_distance.y + color_distance.z + color_distance.w);
            float weight = exp(-max(0.0, patch_distance));

#if USE_BASE_COLOR || USE_DEPTH_NORMAL
            vec4 feature_distance = vec4(0.0);
#if USE_BASE_COLOR
            { // calc base color distance
                vec4 ipx = unpackUnorm4x8(g_temp_base_color[li.y + 4][li.x + 4]);
                vec4 jpx = unpackUnorm4x8(g_temp_base_color[li.y + 4 + k][li.x + 4 + l]);

                feature_distance = g_params.base_color_weight * (ipx - jpx) * (ipx - jpx);
            }
#endif // USE_BASE_COLOR
#if USE_DEPTH_NORMAL
            { // calc feature1 distance
                vec4 ipx = unpackUnorm4x8(g_temp_depth_normal[li.y + 4][li.x + 4]) * vec4(2.0, 2.0, 2.0, 16.0) - vec4(1.0, 1.0, 1.0, 0.0);
                vec4 jpx = unpackUnorm4x8(g_temp_depth_normal[li.y + 4 + k][li.x + 4 + l]) * vec4(2.0, 2.0, 2.0, 16.0) - vec4(1.0, 1.0, 1.0, 0.0);

                feature_distance = max(feature_distance,
                                       g_params.depth_normal_weight * (ipx - jpx) * (ipx - jpx));
            }
#endif // USE_DEPTH_NORMAL
            float feature_patch_distance =
                0.25 * (feature_distance.x + feature_distance.y + feature_distance.z + feature_distance.w);
            float feature_weight = exp(-max(0.0, feature_patch_distance));

            weight = min(weight, feature_weight);
#endif // USE_BASE_COLOR || USE_DEPTH_NORMAL

            sum_output += vec4(unpackHalf2x16(g_temp_color0[li.y + 4 + k][li.x + 4 + l]),
                               unpackHalf2x16(g_temp_color1[li.y + 4 + k][li.x + 4 + l])) * weight;
            sum_weight += weight;
        }
    }

    [[flatten]] if (sum_weight != 0.0) {
        sum_output /= sum_weight;
    }
#else
    [[dont_flatten]] if (gl_GlobalInvocationID.x >= g_params.rect_size.x || gl_GlobalInvocationID.y >= g_params.rect_size.y) {
        return;
    }

    int ix = gi.x, iy = gi.y;

    vec4 sum_output = vec4(0.0);
    float sum_weight = 0.0;

    for (int k = -WindowRadius; k <= WindowRadius; ++k) {
        const int jy = iy + k;

        for (int l = -WindowRadius; l <= WindowRadius; ++l) {
            const int jx = ix + l;

            vec4 distance = vec4(0.0);

            [[unroll]] for (int q = -NeighborRadius; q <= NeighborRadius; ++q) {
                [[unroll]] for (int p = -NeighborRadius; p <= NeighborRadius; ++p) {
                    vec4 ipx = textureLod(g_in_img, tex_coord + vec2(p, q) * g_params.inv_img_size, 0.0);
                    vec4 jpx = textureLod(g_in_img, tex_coord + vec2(l + p, k + q) * g_params.inv_img_size, 0.0);

                    vec4 ivar = textureLod(g_variance_img, tex_coord + vec2(p, q) * g_params.inv_img_size, 0.0);
                    vec4 jvar = textureLod(g_variance_img, tex_coord + vec2(l + p, k + q) * g_params.inv_img_size, 0.0);
                    vec4 min_var = min(ivar, jvar);

                    distance += ((ipx - jpx) * (ipx - jpx) - g_params.alpha * (ivar + min_var)) /
                                (0.0001 + g_params.damping * g_params.damping * (ivar + jvar));
                }
            }

            float patch_distance =
                0.25 * PatchDistanceNormFactor * (distance.x + distance.y + distance.z + distance.w);

            float weight = exp(-max(0.0, patch_distance));

            sum_output += textureLod(g_in_img, tex_coord + vec2(l, k) * g_params.inv_img_size, 0.0) * weight;
            sum_weight += weight;
        }
    }

    [[flatten]] if (sum_weight != 0.0) {
        sum_output /= sum_weight;
    }
#endif

    sum_output = reversible_tonemap_invert(sum_output);

    imageStore(g_out_raw_img, gi, sum_output);
    [[dont_flatten]] if (g_params.tonemap_mode == 0) {
        sum_output = TonemapStandard(g_params.inv_gamma, sum_output);
    } else {
        sum_output = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, sum_output);
    }
    imageStore(g_out_img, gi, sum_output);
}
