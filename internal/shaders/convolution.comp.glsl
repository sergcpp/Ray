#version 450
#extension GL_GOOGLE_include_directive : require
#if USE_FP16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#if USE_NV_COOP_MATRIX
#extension GL_KHR_memory_scope_semantics : require
#extension GL_NV_cooperative_matrix : require
#endif

#include "convolution_interface.h"
#include "common.glsl"

#if !USE_FP16
    #define float16_t float
    #define f16vec3 vec3
    #define f16vec4 vec4
#endif

layout(push_constant) uniform UniformParams {
    Params g_params;
};

#if IMG_INPUT1
layout(binding = IN_IMG1_SLOT) uniform texture2D g_in_img1;
#elif BUF_INPUT1
layout(std430, binding = IN_BUF1_SLOT) readonly restrict buffer Input1Buf {
    float16_t g_input1[];
};
#endif

#if BUF_INPUT2
layout(std430, binding = IN_BUF2_SLOT) readonly restrict buffer Input2Buf {
    float16_t g_input2[];
};
#elif IMG_INPUT2
layout(binding = IN_IMG2_SLOT) uniform texture2D g_in_img2;
#endif

#if IMG_INPUT3
layout(binding = IN_IMG3_SLOT) uniform texture2D g_in_img3;
#endif

#if IMG_INPUT4
layout(binding = IN_IMG4_SLOT) uniform texture2D g_in_img4;
#endif

#if IMG_INPUT1 || IMG_INPUT2
layout(binding = IN_SAMPLER_SLOT) uniform sampler g_sampler;
#endif

layout(std430, binding = WEIGHTS_BUF_SLOT) readonly restrict buffer WeightsBuf {
    float16_t g_weights[];
};

layout(std430, binding = BIASES_BUF_SLOT) readonly restrict buffer BiasesBuf {
    float16_t g_biases[];
};

#if OUT_IMG
layout(binding = TONEMAP_LUT_SLOT) uniform sampler3D g_tonemap_lut;
layout(binding = OUT_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_img;
layout(binding = OUT_TONEMAPPED_IMG_SLOT, rgba32f) uniform writeonly image2D g_out_tonemapped_img;
#else
layout(std430, binding = OUT_BUF_SLOT) writeonly restrict buffer OutBuf {
    float16_t g_out_buf[];
};
#if DEBUG_IMG
layout(binding = OUT_DEBUG_IMG_SLOT, rgba16f) uniform writeonly image2D g_out_debug_img;
#endif
#endif

const float a = 1.41283765e+03;
const float b = 1.64593172e+00;
const float c = 4.31384981e-01;
const float d = -2.94139609e-03;
const float e = 1.92653254e-01;
const float f = 6.26026094e-03;
const float g = 9.98620152e-01;
const float y0 = 1.57945760e-06;
const float y1 = 3.22087631e-02;
const float x0 = 2.23151711e-03;
const float x1 = 3.70974749e-01;

float transfer_input1(float val) {
    const float norm_scale = 0.318967164;

#if HDR_TRANSFER1
    if (val <= y0) {
        return a * val * norm_scale;
    } else if (val <= y1) {
        return (b * pow(val, c) + d) * norm_scale;
    } else {
        return (e * log(val + f) + g) * norm_scale;
    }
#elif POS_NORMALIZE1
    return 0.5 * val + 0.5;
#else
    return val;
#endif
}

vec3 transfer_input1(vec3 val) {
    return vec3(transfer_input1(val.x), transfer_input1(val.y), transfer_input1(val.z));
}

float transfer_input2(float val) {
    const float norm_scale = 0.318967164;

#if HDR_TRANSFER2
    if (val <= y0) {
        return a * val * norm_scale;
    } else if (val <= y1) {
        return (b * pow(val, c) + d) * norm_scale;
    } else {
        return (e * log(val + f) + g) * norm_scale;
    }
#elif POS_NORMALIZE2
    return 0.5 * val + 0.5;
#else
    return val;
#endif
}

vec3 transfer_input2(vec3 val) {
    return vec3(transfer_input2(val.x), transfer_input2(val.y), transfer_input2(val.z));
}

float transfer_input3(float val) {
    const float norm_scale = 0.318967164;

#if HDR_TRANSFER3
    if (val <= y0) {
        return a * val * norm_scale;
    } else if (val <= y1) {
        return (b * pow(val, c) + d) * norm_scale;
    } else {
        return (e * log(val + f) + g) * norm_scale;
    }
#elif POS_NORMALIZE3
    return 0.5 * val + 0.5;
#else
    return val;
#endif
}

vec3 transfer_input3(vec3 val) {
    return vec3(transfer_input3(val.x), transfer_input3(val.y), transfer_input3(val.z));
}

float transfer_input4(float val) {
    const float norm_scale = 0.318967164;

#if HDR_TRANSFER4
    if (val <= y0) {
        return a * val * norm_scale;
    } else if (val <= y1) {
        return (b * pow(val, c) + d) * norm_scale;
    } else {
        return (e * log(val + f) + g) * norm_scale;
    }
#elif POS_NORMALIZE4
    return 0.5 * val + 0.5;
#else
    return val;
#endif
}

vec3 transfer_input4(vec3 val) {
    return vec3(transfer_input4(val.x), transfer_input4(val.y), transfer_input4(val.z));
}

float transfer_output(float val) {
    const float norm_scale = 3.13511896;

#if HDR_TRANSFER1
    val *= norm_scale;
    if (val <= x0) {
        return val / a;
    } else if (val <= x1) {
        return pow((val - d) / b, 1.0 / c);
    } else {
        return exp((val - g) / e) - f;
    }
#elif POS_NORMALIZE1
    return 2.0 * val - 1.0;
#else
    return val;
#endif
}

vec3 transfer_output(vec3 val) {
    return vec3(transfer_output(val.x), transfer_output(val.y), transfer_output(val.z));
}

#if BUF_INPUT1
int in_offset1(int x, int y, int c) {
    return IN_CHANNELS1 * (y * g_params.input_stride1 + x) + c;
}
#endif

#if BUF_INPUT2
int in_offset2(int x, int y, int c) {
    return IN_CHANNELS2 * (y * g_params.input_stride2 + x) + c;
}
#endif

#if !OUT_IMG
int out_offset(int x, int y, int c) {
    return OUT_CHANNELS * (y * g_params.output_stride + x) + c;
}
#endif

#ifndef IN_CHANNELS2
    #define IN_CHANNELS2 0
#endif

#if USE_NV_COOP_MATRIX && 1
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = 16, local_size_y = 1, local_size_z = 8) in;
#endif

const int C_ROWS = TILE_M / 16;
const int C_COLS = TILE_N / 8;

shared float16_t g_mat_staging0[16 * 8];
shared float16_t g_mat_staging1[16 * 8];
shared float16_t g_mat_staging2[16 * 8];
shared float16_t g_mat_staging3[16 * 8];

void main() {
    ivec3 tile_id = ivec3(gl_WorkGroupID), li = ivec3(gl_LocalInvocationID);

#if USE_NV_COOP_MATRIX && 1
    int x = TILE_M * tile_id.x;
    int y = tile_id.y * 2;
    int c = tile_id.z * TILE_N;

    //if (x >= int(g_params.out_dims[0]) || y >= int(g_params.out_dims[1]) || c >= OUT_CHANNELS) {
    //    return;
    //}

    if (x >= int(g_params.in_dims[0]) || y >= int(g_params.in_dims[1]) || c >= OUT_CHANNELS) {
        return;
    }

    fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
    for (int i = 0; i < C_COLS; ++i) {
        const int ii = int(gl_LocalInvocationIndex);
        for (int jj = 0; jj < 16 && ii < 8; ++jj) {
            g_mat_staging0[jj * 8 + ii] = float16_t(0.0);
            if (ii < OUT_CHANNELS) {
                g_mat_staging0[jj * 8 + ii] = g_biases[c + i * 8 + ii];
            }
            // zero out shared memory to avoid NANs later
            g_mat_staging1[jj * 8 + ii] = g_mat_staging2[jj * 8 + ii] = g_mat_staging3[jj * 8 + ii] = float16_t(0.0);
        }
        groupMemoryBarrier(); barrier();

        for (int j = 0; j < C_ROWS; ++j) {
            coopMatLoadNV(C0[j][i], g_mat_staging0, 0u, 8u, false);
            coopMatLoadNV(C1[j][i], g_mat_staging0, 0u, 8u, false);
        }
    }

#if IMG_INPUT1
    for (int j = 0; j < 3 * IN_CHANNELS1; j += 8) {
        fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            for (int jj = 0; jj < 8 && li.x < 16; ++jj) {
                const int x_off = (j + jj) / IN_CHANNELS1, ch = (j + jj) % IN_CHANNELS1;
                if (x_off < 3) {
                    const vec2 tex_coord = (vec2(x + x_off + li.x + i * 16, y) + vec2(0.5)) * g_params.inv_img_size;
                    if (ch < 3) {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch]));
                        g_mat_staging1[li.x * 8 + jj] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch]));
                        g_mat_staging2[li.x * 8 + jj] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch]));
                        g_mat_staging3[li.x * 8 + jj] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch]));
                    }
            #if IMG_INPUT2
                    else if (ch < 6) {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 3]);
                        g_mat_staging1[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 3]);
                        g_mat_staging2[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 3]);
                        g_mat_staging3[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 3]);
                    }
                #if IMG_INPUT3
                    else {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 6]));
                        g_mat_staging1[li.x * 8 + jj] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 6]));
                        g_mat_staging2[li.x * 8 + jj] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 6]));
                        g_mat_staging3[li.x * 8 + jj] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 6]));
                    }
                #endif // IMG_INPUT3
            #endif // IMG_INPUT2
                }
            }

            groupMemoryBarrier(); barrier();

            coopMatLoadNV(A0[i], g_mat_staging0, 0u, 8u, false);
            coopMatLoadNV(A1[i], g_mat_staging1, 0u, 8u, false);
            coopMatLoadNV(A2[i], g_mat_staging2, 0u, 8u, false);
            coopMatLoadNV(A3[i], g_mat_staging3, 0u, 8u, false);
        }

        for (int i = 0; i < C_COLS; ++i) {
            fcoopmatNV<16, gl_ScopeSubgroup, 8, 8> B0, B1, B2;

            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8); // stride and offset must be aligned

            coopMatLoadNV(B0, g_weights, (c + i * 8) * 3 * rounded_triple1 + 0 * rounded_triple1 + j, 3 * rounded_triple1, true);
            coopMatLoadNV(B1, g_weights, (c + i * 8) * 3 * rounded_triple1 + 1 * rounded_triple1 + j, 3 * rounded_triple1, true);
            coopMatLoadNV(B2, g_weights, (c + i * 8) * 3 * rounded_triple1 + 2 * rounded_triple1 + j, 3 * rounded_triple1, true);

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = coopMatMulAddNV(A0[k], B0, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A1[k], B1, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A2[k], B2, C0[k][i]);

                C1[k][i] = coopMatMulAddNV(A1[k], B0, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A2[k], B1, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // IMG_INPUT1

    const int rows_count = min(C_ROWS, (int(g_params.in_dims[0]) - x + 15) / 16);
    const int cols_count = min(C_COLS, (OUT_CHANNELS - c + 7) / 8);

#if BUF_INPUT1
    for (int j = 0; j < 3 * IN_CHANNELS1; j += 8) {
        fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
#if UPSCALE1
            for (int jj = 0; jj < 8 && li.x < 16; ++jj) {
                const int x_off = ((j + jj) / IN_CHANNELS1), ch = ((j + jj) % IN_CHANNELS1);
                const int x_final = (x + x_off + li.x + i * 16 - 1 + 2) / 2 - 1;

                const int i0 = IN_CHANNELS1 * (((y == 0 ? -1 : (y - 1) / 2) + 1) * g_params.input_stride1 + x_final + 1) + ch;
                const int i1 = IN_CHANNELS1 * (((y + 0) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch;
                const int i2 = IN_CHANNELS1 * (((y + 1) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch;
                const int i3 = IN_CHANNELS1 * (((y + 2) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch;

                g_mat_staging0[li.x * 8 + jj] = g_input1[i0];
                g_mat_staging1[li.x * 8 + jj] = g_input1[i1];
                g_mat_staging2[li.x * 8 + jj] = g_input1[i2];
                g_mat_staging3[li.x * 8 + jj] = g_input1[i3];
            }

            groupMemoryBarrier(); barrier();

            coopMatLoadNV(A0[i], g_mat_staging0, 0u, 8u, false);
            coopMatLoadNV(A1[i], g_mat_staging1, 0u, 8u, false);
            coopMatLoadNV(A2[i], g_mat_staging2, 0u, 8u, false);
            coopMatLoadNV(A3[i], g_mat_staging3, 0u, 8u, false);
#else // UPSCALE1
            const int i0 = IN_CHANNELS1 * ((y - 1 + 1) * g_params.input_stride1 + x + i * 16 - 1 + 1) + j;
            const int i1 = IN_CHANNELS1 * ((y + 0 + 1) * g_params.input_stride1 + x + i * 16 - 1 + 1) + j;
            const int i2 = IN_CHANNELS1 * ((y + 1 + 1) * g_params.input_stride1 + x + i * 16 - 1 + 1) + j;
            const int i3 = IN_CHANNELS1 * ((y + 2 + 1) * g_params.input_stride1 + x + i * 16 - 1 + 1) + j;

            coopMatLoadNV(A0[i], g_input1, i0, IN_CHANNELS1, false);
            coopMatLoadNV(A1[i], g_input1, i1, IN_CHANNELS1, false);
            coopMatLoadNV(A2[i], g_input1, i2, IN_CHANNELS1, false);
            coopMatLoadNV(A3[i], g_input1, i3, IN_CHANNELS1, false);
#endif // UPSCALE1
        }

        for (int i = 0; i < cols_count; ++i) {
            fcoopmatNV<16, gl_ScopeSubgroup, 8, 8> B0, B1, B2;

            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8),
                      rounded_triple2 = 8 * ((3 * IN_CHANNELS2 + 7) / 8); // stride and offset must be aligned

            coopMatLoadNV(B0, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 0 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B1, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 1 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B2, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 2 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = coopMatMulAddNV(A0[k], B0, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A1[k], B1, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A2[k], B2, C0[k][i]);

                C1[k][i] = coopMatMulAddNV(A1[k], B0, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A2[k], B1, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // BUF_INPUT1

#if BUF_INPUT2
    for (int j = 0; j < 3 * IN_CHANNELS2; j += 8) {
        fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            const int i0 = IN_CHANNELS2 * ((y - 1 + 1) * g_params.input_stride2 + x + i * 16 - 1 + 1) + j;
            const int i1 = IN_CHANNELS2 * ((y + 0 + 1) * g_params.input_stride2 + x + i * 16 - 1 + 1) + j;
            const int i2 = IN_CHANNELS2 * ((y + 1 + 1) * g_params.input_stride2 + x + i * 16 - 1 + 1) + j;
            const int i3 = IN_CHANNELS2 * ((y + 2 + 1) * g_params.input_stride2 + x + i * 16 - 1 + 1) + j;

            coopMatLoadNV(A0[i], g_input2, i0, IN_CHANNELS2, false);
            coopMatLoadNV(A1[i], g_input2, i1, IN_CHANNELS2, false);
            coopMatLoadNV(A2[i], g_input2, i2, IN_CHANNELS2, false);
            coopMatLoadNV(A3[i], g_input2, i3, IN_CHANNELS2, false);
        }

        for (int i = 0; i < C_COLS; ++i) {
            fcoopmatNV<16, gl_ScopeSubgroup, 8, 8> B0, B1, B2;

            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8),
                      rounded_triple2 = 8 * ((3 * IN_CHANNELS2 + 7) / 8); // stride and offset must be aligned

            coopMatLoadNV(B0, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 0 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B1, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 1 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B2, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 2 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = coopMatMulAddNV(A0[k], B0, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A1[k], B1, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A2[k], B2, C0[k][i]);

                C1[k][i] = coopMatMulAddNV(A1[k], B0, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A2[k], B1, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // BUF_INPUT1

#if IMG_INPUT2
    for (int j = 0; j < 3 * IN_CHANNELS2; j += 8) {
        fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            for (int jj = 0; jj < 8 && li.x < 16; ++jj) {
                const int x_off = (j + jj) / IN_CHANNELS2, ch = (j + jj) % IN_CHANNELS2;
                if (x_off < 3) {
                    const vec2 tex_coord = (vec2(x + x_off + li.x + i * 16, y) + vec2(0.5)) * g_params.inv_img_size;
                    if (ch < 3) {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch]));
                        g_mat_staging1[li.x * 8 + jj] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch]));
                        g_mat_staging2[li.x * 8 + jj] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch]));
                        g_mat_staging3[li.x * 8 + jj] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch]));
                    }
            #if IMG_INPUT3
                    else if (ch < 6) {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 3]);
                        g_mat_staging1[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 3]);
                        g_mat_staging2[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 3]);
                        g_mat_staging3[li.x * 8 + jj] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 3]);
                    }
                #if IMG_INPUT4
                    else {
                        g_mat_staging0[li.x * 8 + jj] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 6]));
                        g_mat_staging1[li.x * 8 + jj] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 6]));
                        g_mat_staging2[li.x * 8 + jj] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 6]));
                        g_mat_staging3[li.x * 8 + jj] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 6]));
                    }
                #endif // IMG_INPUT4
            #endif // IMG_INPUT3
                }
            }

            groupMemoryBarrier(); barrier();

            coopMatLoadNV(A0[i], g_mat_staging0, 0u, 8u, false);
            coopMatLoadNV(A1[i], g_mat_staging1, 0u, 8u, false);
            coopMatLoadNV(A2[i], g_mat_staging2, 0u, 8u, false);
            coopMatLoadNV(A3[i], g_mat_staging3, 0u, 8u, false);
        }

        for (int i = 0; i < C_COLS; ++i) {
            fcoopmatNV<16, gl_ScopeSubgroup, 8, 8> B0, B1, B2;

            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8),
                      rounded_triple2 = 8 * ((3 * IN_CHANNELS2 + 7) / 8); // stride and offset must be aligned

            coopMatLoadNV(B0, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 0 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B1, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 1 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);
            coopMatLoadNV(B2, g_weights, (c + i * 8) * 3 * (rounded_triple1 + rounded_triple2) + 2 * rounded_triple2 + 3 * rounded_triple1 + j, 3 * (rounded_triple1 + rounded_triple2), true);

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = coopMatMulAddNV(A0[k], B0, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A1[k], B1, C0[k][i]);
                C0[k][i] = coopMatMulAddNV(A2[k], B2, C0[k][i]);

                C1[k][i] = coopMatMulAddNV(A1[k], B0, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A2[k], B1, C1[k][i]);
                C1[k][i] = coopMatMulAddNV(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // IMG_INPUT2

#if DOWNSAMPLE
    for (int j = 0; j < C_ROWS; ++j) {
        for (int i = 0; i < cols_count; ++i) {
            for (int k = 0; k < C0[j][i].length(); ++k) {
                C0[j][i][k] = max(max(C0[j][i][k], C1[j][i][k]), float16_t(0.0));
            }

            coopMatStoreNV(C0[j][i], g_mat_staging0, 0u, 8u, false);
            groupMemoryBarrier(); barrier();

            for (int jj = 0; jj < 16; jj += 2) {
                for (int ii = 0; ii < 8; ++ii) {
                    float16_t out_val = max(g_mat_staging0[(jj + 0) * 8 + ii], g_mat_staging0[(jj + 1) * 8 + ii]);
                    g_out_buf[OUT_CHANNELS * ((y / 2 + 1) * g_params.output_stride + x / 2 + j * 16 / 2 + jj / 2 + 1) + c + i * 8 + ii] = max(out_val, float16_t(0.0));
                }
            }
        }
    }
#elif OUT_IMG
    for (int j = 0; j < rows_count && c == 0; ++j) {
        for (int k = 0; k < C0[j][0].length(); ++k) {
            C0[j][0][k] = max(C0[j][0][k], float16_t(0.0));
            C1[j][0][k] = max(C1[j][0][k], float16_t(0.0));
        }

        coopMatStoreNV(C0[j][0], g_mat_staging0, 0u, 8u, false);
        coopMatStoreNV(C1[j][0], g_mat_staging1, 0u, 8u, false);
        groupMemoryBarrier(); barrier();

        for (int jj = 0; jj < 16; ++jj) {
            vec4 val0 = vec4(g_mat_staging0[jj * 8 + 0], g_mat_staging0[jj * 8 + 1], g_mat_staging0[jj * 8 + 2], 1.0),
                 val1 = vec4(g_mat_staging1[jj * 8 + 0], g_mat_staging1[jj * 8 + 1], g_mat_staging1[jj * 8 + 2], 1.0);
            val0.xyz = transfer_output(val0.xyz);
            val1.xyz = transfer_output(val1.xyz);
            imageStore(g_out_img, ivec2(x + j * 16 + jj, y), val0);
            if (y + 1 < int(g_params.out_dims[1])) {
                imageStore(g_out_img, ivec2(x + j * 16 + jj, y + 1), val1);
            }
        #if TONEMAP
            [[dont_flatten]] if (g_params.tonemap_mode == 0) {
                val0 = TonemapStandard(g_params.inv_gamma, val0);
                val1 = TonemapStandard(g_params.inv_gamma, val1);
            } else {
                val0 = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, val0);
                val1 = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, val1);
            }
        #endif
            imageStore(g_out_tonemapped_img, ivec2(x + j * 16 + jj, y), val0);
            if (y + 1 < int(g_params.out_dims[1])) {
                imageStore(g_out_tonemapped_img, ivec2(x + j * 16 + jj, y + 1), val1);
            }
        }
    }
#else // OUT_IMG
    for (int j = 0; j < rows_count; ++j) {
        for (int i = 0; i < cols_count; ++i) {
            for (int k = 0; k < C0[j][i].length(); ++k) {
                C0[j][i][k] = max(C0[j][i][k], float16_t(0.0));
                C1[j][i][k] = max(C1[j][i][k], float16_t(0.0));
            }

            coopMatStoreNV(C0[j][i], g_out_buf, OUT_CHANNELS * ((y + 0 + 1) * g_params.output_stride + x + j * 16 + 1) + c + i * 8, OUT_CHANNELS, false);
            if (y + 1 < int(g_params.out_dims[1])) {
                coopMatStoreNV(C1[j][i], g_out_buf, OUT_CHANNELS * ((y + 1 + 1) * g_params.output_stride + x + j * 16 + 1) + c + i * 8, OUT_CHANNELS, false);
            }
        }
    }
#endif // OUT_IMG

    ///

#if !OUT_IMG
    for (int yy = y; yy < min(y + 2, g_params.out_dims.y); ++yy) {
        for (int xx = x + li.x; xx < min(x + TILE_M, g_params.out_dims.x); xx += 32) {
            for (int cc = c; cc < min(c + TILE_N, OUT_CHANNELS); ++cc) {
                const ivec2 gi = ivec2(xx, yy);

                if (gi.x == 0) {
                    g_out_buf[out_offset(gi.x + 0, gi.y + 1, cc)] = float16_t(0.0);
                    if (gi.y == 0) {
                        g_out_buf[out_offset(gi.x + 0, gi.y + 0, cc)] = float16_t(0.0);
                    }
                    if (gi.y == g_params.out_dims.y - 1) {
                        g_out_buf[out_offset(gi.x + 0, gi.y + 2, cc)] = float16_t(0.0);
                    }
                }
                if (gi.x == g_params.out_dims.x - 1) {
                    g_out_buf[out_offset(gi.x + 2, gi.y + 1, cc)] = float16_t(0.0);
                    if (gi.y == 0) {
                        g_out_buf[out_offset(gi.x + 2, gi.y + 0, cc)] = float16_t(0.0);
                    }
                    if (gi.y == g_params.out_dims.y - 1) {
                        g_out_buf[out_offset(gi.x + 2, gi.y + 2, cc)] = float16_t(0.0);
                    }
                }
                if (gi.y == 0) {
                    g_out_buf[out_offset(gi.x + 1, gi.y + 0, cc)] = float16_t(0.0);
                }
                if (gi.y == g_params.out_dims.y - 1) {
                    g_out_buf[out_offset(gi.x + 1, gi.y + 2, cc)] = float16_t(0.0);
                }
            }
        }
    }
#endif

#else // USE_NV_COOP_MATRIX

    int x = TILE_M * tile_id.x + C_ROWS * li.x;
    int y = tile_id.y * 2;
    int c = tile_id.z * TILE_N + C_COLS * li.z;

    if (x >= int(g_params.in_dims[0]) || y >= int(g_params.in_dims[1]) || c >= OUT_CHANNELS) {
        return;
    }

    float16_t C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
    for (int j = 0; j < C_ROWS; ++j) {
        for (int i = 0; i < C_COLS; ++i) {
            C0[j][i] = C1[j][i] = g_biases[c + i];
        }
    }

#if IMG_INPUT1
    for (int j = 0; j < 3 * IN_CHANNELS1; ++j) {
        float16_t A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            const int x_off = (j / IN_CHANNELS1), ch = (j % IN_CHANNELS1);

            vec2 tex_coord = (vec2(x + i + x_off, y) + vec2(0.5)) * g_params.inv_img_size;
            if (ch < 3) {
                A0[i] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch]));
                A1[i] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch]));
                A2[i] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch]));
                A3[i] = float16_t(transfer_input1(textureLodOffset(sampler2D(g_in_img1, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch]));
            }
    #if IMG_INPUT2
            else if (ch < 6) {
                A0[i] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 3]);
                A1[i] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 3]);
                A2[i] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 3]);
                A3[i] = float16_t(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 3]);
            }
        #if IMG_INPUT3
            else {
                A0[i] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 6]));
                A1[i] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 6]));
                A2[i] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 6]));
                A3[i] = float16_t(transfer_input3(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 6]));
            }
        #endif // IMG_INPUT3
    #endif // IMG_INPUT2
        }

        for (int i = 0; i < C_COLS; ++i) {
            const int rounded_triple = 8 * ((3 * IN_CHANNELS1 + 7) / 8); // stride and offset must be aligned

            float16_t B0 = g_weights[(c + i) * 3 * rounded_triple + 0 * rounded_triple + j];
            float16_t B1 = g_weights[(c + i) * 3 * rounded_triple + 1 * rounded_triple + j];
            float16_t B2 = g_weights[(c + i) * 3 * rounded_triple + 2 * rounded_triple + j];

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = fma(A0[k], B0, C0[k][i]);
                C0[k][i] = fma(A1[k], B1, C0[k][i]);
                C0[k][i] = fma(A2[k], B2, C0[k][i]);

                C1[k][i] = fma(A1[k], B0, C1[k][i]);
                C1[k][i] = fma(A2[k], B1, C1[k][i]);
                C1[k][i] = fma(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // IMG_INPUT1

#if BUF_INPUT1
    for (int j = 0; j < 3 * IN_CHANNELS1; ++j) {
        float16_t A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
#if UPSCALE1
        const int x_off = (j / IN_CHANNELS1), ch = (j % IN_CHANNELS1);
        for (int i = 0; i < C_ROWS; ++i) {
            const int x_final = (x + x_off + i - 1 + 2) / 2 - 1;
            A0[i] = g_input1[IN_CHANNELS1 * (((y == 0 ? -1 : (y - 1) / 2) + 1) * g_params.input_stride1 + x_final + 1) + ch];
            A1[i] = g_input1[IN_CHANNELS1 * (((y + 0) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch];
            A2[i] = g_input1[IN_CHANNELS1 * (((y + 1) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch];
            A3[i] = g_input1[IN_CHANNELS1 * (((y + 2) / 2 + 1) * g_params.input_stride1 + x_final + 1) + ch];
        }
#else // UPSCALE1
        for (int i = 0; i < C_ROWS; ++i) {
            A0[i] = g_input1[IN_CHANNELS1 * ((y - 1 + 1) * g_params.input_stride1 + x + i - 1 + 1) + j];
            A1[i] = g_input1[IN_CHANNELS1 * ((y + 0 + 1) * g_params.input_stride1 + x + i - 1 + 1) + j];
            A2[i] = g_input1[IN_CHANNELS1 * ((y + 1 + 1) * g_params.input_stride1 + x + i - 1 + 1) + j];
            A3[i] = g_input1[IN_CHANNELS1 * ((y + 2 + 1) * g_params.input_stride1 + x + i - 1 + 1) + j];
        }
#endif // UPSCALE1

        for (int i = 0; i < C_COLS; ++i) {
            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8),
                      rounded_triple2 = 8 * ((3 * IN_CHANNELS2 + 7) / 8); // stride and offset must be aligned

            float16_t B0 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 0 * rounded_triple1 + j];
            float16_t B1 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 1 * rounded_triple1 + j];
            float16_t B2 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 2 * rounded_triple1 + j];

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = fma(A0[k], B0, C0[k][i]);
                C0[k][i] = fma(A1[k], B1, C0[k][i]);
                C0[k][i] = fma(A2[k], B2, C0[k][i]);

                C1[k][i] = fma(A1[k], B0, C1[k][i]);
                C1[k][i] = fma(A2[k], B1, C1[k][i]);
                C1[k][i] = fma(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // BUF_INPUT1

#if BUF_INPUT2
    for (int j = 0; j < 3 * IN_CHANNELS2; ++j) {
        float16_t A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            A0[i] = g_input2[IN_CHANNELS2 * ((y - 1 + 1) * g_params.input_stride2 + x + i - 1 + 1) + j];
            A1[i] = g_input2[IN_CHANNELS2 * ((y + 0 + 1) * g_params.input_stride2 + x + i - 1 + 1) + j];
            A2[i] = g_input2[IN_CHANNELS2 * ((y + 1 + 1) * g_params.input_stride2 + x + i - 1 + 1) + j];
            A3[i] = g_input2[IN_CHANNELS2 * ((y + 2 + 1) * g_params.input_stride2 + x + i - 1 + 1) + j];
        }

        for (int i = 0; i < C_COLS; ++i) {
            float16_t B0 = g_weights[(c + i) * (IN_CHANNELS1 + IN_CHANNELS2) * 9 + 0 * IN_CHANNELS2 + 9 * IN_CHANNELS1 + j];
            float16_t B1 = g_weights[(c + i) * (IN_CHANNELS1 + IN_CHANNELS2) * 9 + 3 * IN_CHANNELS2 + 9 * IN_CHANNELS1 + j];
            float16_t B2 = g_weights[(c + i) * (IN_CHANNELS1 + IN_CHANNELS2) * 9 + 6 * IN_CHANNELS2 + 9 * IN_CHANNELS1 + j];

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = fma(A0[k], B0, C0[k][i]);
                C0[k][i] = fma(A1[k], B1, C0[k][i]);
                C0[k][i] = fma(A2[k], B2, C0[k][i]);

                C1[k][i] = fma(A1[k], B0, C1[k][i]);
                C1[k][i] = fma(A2[k], B1, C1[k][i]);
                C1[k][i] = fma(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // BUF_INPUT2

#if BUF_INPUT1 && !BUF_INPUT2 && IMG_INPUT2
    for (int j = 0; j < 3 * IN_CHANNELS2; ++j) {
        float16_t A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
        for (int i = 0; i < C_ROWS; ++i) {
            const int x_off = (j / IN_CHANNELS2), ch = (j % IN_CHANNELS2);

            vec2 tex_coord = (vec2(x + i + x_off, y) + vec2(0.5)) * g_params.inv_img_size;
            if (ch < 3) {
                A0[i] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch]));
                A1[i] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch]));
                A2[i] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch]));
                A3[i] = float16_t(transfer_input2(textureLodOffset(sampler2D(g_in_img2, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch]));
            }
    #if IMG_INPUT3
            else if (ch < 6) {
                A0[i] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 3]);
                A1[i] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 3]);
                A2[i] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 3]);
                A3[i] = float16_t(textureLodOffset(sampler2D(g_in_img3, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 3]);
            }
        #if IMG_INPUT4
            else {
                A0[i] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, -1))[ch - 6]));
                A1[i] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +0))[ch - 6]));
                A2[i] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +1))[ch - 6]));
                A3[i] = float16_t(transfer_input4(textureLodOffset(sampler2D(g_in_img4, g_sampler), tex_coord, 0.0, ivec2(-1, +2))[ch - 6]));
            }
        #endif // IMG_INPUT4
    #endif // IMG_INPUT3
        }

        for (int i = 0; i < C_COLS; ++i) {
            const int rounded_triple1 = 8 * ((3 * IN_CHANNELS1 + 7) / 8),
                      rounded_triple2 = 8 * ((3 * IN_CHANNELS2 + 7) / 8); // stride and offset must be aligned

            float16_t B0 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 0 * rounded_triple2 + 3 * rounded_triple1 + j];
            float16_t B1 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 1 * rounded_triple2 + 3 * rounded_triple1 + j];
            float16_t B2 = g_weights[(c + i) * 3 * (rounded_triple1 + rounded_triple2) + 2 * rounded_triple2 + 3 * rounded_triple1 + j];

            for (int k = 0; k < C_ROWS; ++k) {
                C0[k][i] = fma(A0[k], B0, C0[k][i]);
                C0[k][i] = fma(A1[k], B1, C0[k][i]);
                C0[k][i] = fma(A2[k], B2, C0[k][i]);

                C1[k][i] = fma(A1[k], B0, C1[k][i]);
                C1[k][i] = fma(A2[k], B1, C1[k][i]);
                C1[k][i] = fma(A3[k], B2, C1[k][i]);
            }
        }
    }
#endif // BUF_INPUT1 && IMG_INPUT2

#if DOWNSAMPLE
    for (int j = 0; j < C_ROWS; j += 2) {
        for (int i = 0; i < C_COLS; ++i) {
            C0[j][i] = max(max(C0[j][i], C0[j + 1][i]), max(C1[j][i], C1[j + 1][i]));
            C0[j][i] = max(C0[j][i], float16_t(0.0));
            g_out_buf[OUT_CHANNELS * ((y / 2 + 1) * g_params.output_stride + x / 2 + j / 2 + 1) + c + i] = C0[j][i];
        }
    }

#if DEBUG_IMG
    if (c == 0) {
        for (int j = 0; j < C_ROWS; j += 2) {
            float16_t val0 = C0[j][0];
            imageStore(g_out_debug_img, ivec2(x / 2 + j / 2, y / 2), vec4(val0, val0, val0, 1.0));
        }
    }
#endif // DEBUG_IMG

#elif OUT_IMG

    for (int j = 0; j < C_ROWS && c == 0; ++j) {
        for (int i = 0; i < C_COLS; ++i) {
            C0[j][i] = max(C0[j][i], float16_t(0.0));
            C1[j][i] = max(C1[j][i], float16_t(0.0));
        }

        vec4 val0 = vec4(C0[j][0], C0[j][1], C0[j][2], 1.0),
             val1 = vec4(C1[j][0], C1[j][1], C1[j][2], 1.0);
        val0.xyz = transfer_output(val0.xyz);
        val1.xyz = transfer_output(val1.xyz);
        imageStore(g_out_img, ivec2(x + j, y), val0);
        if (y + 1 < int(g_params.out_dims[1])) {
            imageStore(g_out_img, ivec2(x + j, y + 1), val1);
        }
    #if TONEMAP
        [[dont_flatten]] if (g_params.tonemap_mode == 0) {
            val0 = TonemapStandard(g_params.inv_gamma, val0);
            val1 = TonemapStandard(g_params.inv_gamma, val1);
        } else {
            val0 = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, val0);
            val1 = TonemapLUT_manual(g_tonemap_lut, g_params.inv_gamma, val1);
        }
    #endif
        imageStore(g_out_tonemapped_img, ivec2(x + j, y), val0);
        if (y + 1 < int(g_params.out_dims[1])) {
            imageStore(g_out_tonemapped_img, ivec2(x + j, y + 1), val1);
        }
    }

#else

    for (int j = 0; j < C_ROWS; ++j) {
        for (int i = 0; i < C_COLS; ++i) {
            C0[j][i] = max(C0[j][i], float16_t(0.0));
            C1[j][i] = max(C1[j][i], float16_t(0.0));

            g_out_buf[OUT_CHANNELS * ((y + 1) * g_params.output_stride + x + j + 1) + c + i] = C0[j][i];
            if (y + 1 < int(g_params.out_dims[1])) {
                g_out_buf[OUT_CHANNELS * ((y + 2) * g_params.output_stride + x + j + 1) + c + i] = C1[j][i];
            }
        }
    }

#if DEBUG_IMG
    if (c == 0) {
        for (int j = 0; j < C_ROWS; ++j) {
            float16_t val0 = C0[j][0], val1 = C1[j][0];
            imageStore(g_out_debug_img, ivec2(x + j, y + 0), vec4(val0, val0, val0, 1.0));
            imageStore(g_out_debug_img, ivec2(x + j, y + 1), vec4(val1, val1, val1, 1.0));
        }
    }
#endif // DEBUG_IMG

#endif

    ///

#if !OUT_IMG
    for (int cc = c; cc < min(c + C_COLS, OUT_CHANNELS); ++cc) {
        for (int yy = y; yy < min(y + 2, g_params.out_dims.y); ++yy) {
            for (int xx = x; xx < min(x + C_ROWS, g_params.out_dims.x); ++xx) {
                const ivec2 gi = ivec2(xx, yy);

                if (gi.x == 0) {
                    g_out_buf[out_offset(gi.x + 0, gi.y + 1, cc)] = float16_t(0.0);
                    if (gi.y == 0) {
                        g_out_buf[out_offset(gi.x + 0, gi.y + 0, cc)] = float16_t(0.0);
                    }
                    if (gi.y == g_params.out_dims.y - 1) {
                        g_out_buf[out_offset(gi.x + 0, gi.y + 2, cc)] = float16_t(0.0);
                    }
                }
                if (gi.x == g_params.out_dims.x - 1) {
                    g_out_buf[out_offset(gi.x + 2, gi.y + 1, cc)] = float16_t(0.0);
                    if (gi.y == 0) {
                        g_out_buf[out_offset(gi.x + 2, gi.y + 0, cc)] = float16_t(0.0);
                    }
                    if (gi.y == g_params.out_dims.y - 1) {
                        g_out_buf[out_offset(gi.x + 2, gi.y + 2, cc)] = float16_t(0.0);
                    }
                }
                if (gi.y == 0) {
                    g_out_buf[out_offset(gi.x + 1, gi.y + 0, cc)] = float16_t(0.0);
                }
                if (gi.y == g_params.out_dims.y - 1) {
                    g_out_buf[out_offset(gi.x + 1, gi.y + 2, cc)] = float16_t(0.0);
                }
            }
        }
    }
#endif // !OUT_IMG

#endif // USE_NV_COOP_MATRIX
}
