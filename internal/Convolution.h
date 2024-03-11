#pragma once

#include "../Types.h"

namespace Ray {
namespace NS {
void ClearBorders(const rect_t &rect, int w, int h, bool downscaled, int out_channels, float output[]) {
    if (!downscaled) {
        for (int y = rect.y; y < rect.y + rect.h; ++y) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.x == 0) {
                    output[out_channels * ((y + 1) * (w + 2) + 0) + i] = 0.0f;
                }
                if (rect.x + rect.w == w) {
                    output[out_channels * ((y + 1) * (w + 2) + w + 1) + i] = 0.0f;
                }
            }
        }
        const int rect_x = (rect.x == 0) ? rect.x : rect.x + 1;
        int rect_w = (rect.x == 0) ? rect.w + 1 : rect.w;
        if (rect.x + rect.w == w) {
            ++rect_w;
        }
        for (int x = rect_x; x < rect_x + rect_w; ++x) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.y == 0) {
                    output[out_channels * (x + 0) + i] = 0.0f;
                }
                if (rect.y + rect.h == h) {
                    output[out_channels * ((h + 1) * (w + 2) + x + 0) + i] = 0.0f;
                }
            }
        }
    } else {
        for (int y = (rect.y / 2); y < (rect.y + rect.h + 1) / 2; ++y) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.x == 0) {
                    output[out_channels * ((y + 1) * ((w + 1) / 2 + 2) + 0) + i] = 0.0f;
                }
                if (rect.x + rect.w == w) {
                    output[out_channels * ((y + 1) * ((w + 1) / 2 + 2) + (w + 1) / 2 + 1) + i] = 0.0f;
                }
            }
        }
        const int rect_x = (rect.x == 0) ? (rect.x / 2) : (rect.x / 2) + 1;
        int rect_w = (rect.x == 0) ? ((rect.w + 1) / 2) + 1 : (rect.w + 1) / 2;
        if (rect.x + rect.w == w) {
            ++rect_w;
        }
        for (int x = rect_x; x < rect_x + rect_w; ++x) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.y == 0) {
                    output[out_channels * (x + 0) + i] = 0.0f;
                }
                if (rect.y + rect.h == h) {
                    output[out_channels * (((h + 1) / 2 + 1) * ((w + 1) / 2 + 2) + x) + i] = 0.0f;
                }
            }
        }
    }
}

namespace transfer {
static const float a = 1.41283765e+03f;
static const float b = 1.64593172e+00f;
static const float c = 4.31384981e-01f;
static const float d = -2.94139609e-03f;
static const float e = 1.92653254e-01f;
static const float f = 6.26026094e-03f;
static const float g = 9.98620152e-01f;
static const float y0 = 1.57945760e-06f;
static const float y1 = 3.22087631e-02f;
static const float x0 = 2.23151711e-03f;
static const float x1 = 3.70974749e-01f;

template <ePreOp PreOp> force_inline float input(float val) {
    static const float norm_scale = 0.318967164f;

    if (PreOp == ePreOp::HDRTransfer) {
        if (val <= y0) {
            return a * val * norm_scale;
        } else if (val <= y1) {
            return (b * powf(val, c) + d) * norm_scale;
        } else {
            return (e * logf(val + f) + g) * norm_scale;
        }
    } else if (PreOp == ePreOp::PositiveNormalize) {
        return 0.5f * val + 0.5f;
    } else {
        return val;
    }
}

template <ePostOp PostOp> force_inline float output(float val) {
    static const float norm_scale = 3.13511896f;

    if (PostOp == ePostOp::HDRTransfer) {
        val *= norm_scale;
        if (val <= x0) {
            return val / a;
        } else if (val <= x1) {
            return powf((val - d) / b, 1.0f / c);
        } else {
            return expf((val - g) / e) - f;
        }
    } else if (PostOp == ePostOp::PositiveNormalize) {
        return 2.0f * val - 1.0f;
    } else {
        return val;
    }
}
} // namespace transfer

template <int RowsPortion, int S, int InChannels, int OutChannels, int OutPxPitch, ePostOp PostOp,
          eActivation Activation>
void Convolution3x3_Direct_ProcessRows(int y, const float *__restrict data, const rect_t &rect, int w, int h,
                                       int stride, const float *__restrict weights, const float *__restrict biases,
                                       float *__restrict output, const int output_stride) {
    static_assert((InChannels % S) == 0, "!");
    static_assert(RowsPortion <= 8, "!");

#define index(y, x) InChannels *((y)*stride + (x))

    for (int x = rect.x; x < rect.x + rect.w; ++x) {
        const int ii[] = {index(y - 1, x - 1), //
                          index(y, x - 1),     //
                          index(y + 1, x - 1), //
                          index(y + 2, x - 1), //
                          index(y + 3, x - 1), //
                          index(y + 4, x - 1), //
                          index(y + 5, x - 1), //
                          index(y + 6, x - 1), //
                          index(y + 7, x - 1), //
                          index(y + 8, x - 1)};

        for (int i = 0; i < OutChannels; ++i) {
            fvec<S> val[RowsPortion] = {};
            for (int j = 0; j < 3 * InChannels; j += S) {
                if (RowsPortion == 8) {
                    UNROLLED_FOR(k, 8, {
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 0 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 0] + j]}, val[k % RowsPortion]);
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 3 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 1] + j]}, val[k % RowsPortion]);
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 6 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 2] + j]}, val[k % RowsPortion]);
                    })
                } else if (RowsPortion == 4) {
                    UNROLLED_FOR(k, 4, {
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 0 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 0] + j]}, val[k % RowsPortion]);
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 3 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 1] + j]}, val[k % RowsPortion]);
                        val[k % RowsPortion] =
                            fmadd(fvec<S>{&weights[i * InChannels * 9 + 6 * InChannels + j], vector_aligned},
                                  fvec<S>{&data[ii[k + 2] + j]}, val[k % RowsPortion]);
                    })
                } else {
                    for (int k = 0; k < RowsPortion; ++k) {
                        val[k] = fmadd(fvec<S>{&weights[i * InChannels * 9 + 0 * InChannels + j], vector_aligned},
                                       fvec<S>{&data[ii[k + 0] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&weights[i * InChannels * 9 + 3 * InChannels + j], vector_aligned},
                                       fvec<S>{&data[ii[k + 1] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&weights[i * InChannels * 9 + 6 * InChannels + j], vector_aligned},
                                       fvec<S>{&data[ii[k + 2] + j]}, val[k]);
                    }
                }
            }

            for (int k = 0; k < RowsPortion; ++k) {
                float final_val = biases[i] + hsum(val[k]);
                if (Activation == eActivation::ReLU) {
                    final_val = fmaxf(0.0f, final_val);
                }

                if (PostOp == ePostOp::Downscale) {
                    float &out = output[OutPxPitch * (((y + k) / 2) * output_stride + (x / 2)) + i];
                    out = fmaxf(out, final_val);
                } else {
                    output[OutPxPitch * ((y + k) * output_stride + x) + i] = transfer::output<PostOp>(final_val);
                }
            }
        }
    }

#undef index
}

template <int RowsPortion, int S, int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1, ePostOp PostOp,
          eActivation Activation>
void ConvolutionConcat3x3_Direct_ProcessRows(int y, const float *__restrict data1, const float *__restrict data2,
                                             const rect_t &rect, int w, int h, int stride1, int stride2,
                                             const float *__restrict weights, const float *__restrict biases,
                                             float *__restrict output, int output_stride) {
    static_assert((InChannels1 % S) == 0 && (InChannels2 % S) == 0, "!");
    static_assert(RowsPortion <= 8, "!");

    const int div1 = (PreOp1 == ePreOp::Upscale) ? 2 : 1;

#define index1(y, x) InChannels1 *((y)*stride1 + (x))
#define index2(y, x) InChannels2 *((y)*stride2 + (x))

    for (int x = rect.x; x < rect.x + rect.w; ++x) {
        const int add = ((x + 1) % div1);

        const int ii1[] = {
            index1(y == 0 ? -1 : (y - 1) / div1, x == 0 ? -1 : (x - 1) / div1), //
            index1(y / div1, x == 0 ? -1 : (x - 1) / div1),                     //
            index1((y + 1) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 2) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 3) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 4) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 5) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 6) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 7) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 8) / div1, x == 0 ? -1 : (x - 1) / div1)                //
        };

        const int ii2[] = {
            index2(y - 1, x - 1), //
            index2(y, x - 1),     //
            index2(y + 1, x - 1), //
            index2(y + 2, x - 1), //
            index2(y + 3, x - 1), //
            index2(y + 4, x - 1), //
            index2(y + 5, x - 1), //
            index2(y + 6, x - 1), //
            index2(y + 7, x - 1), //
            index2(y + 8, x - 1)  //
        };

        for (int i = 0; i < OutChannels; ++i) {
            fvec<S> val[8] = {};

            const float *p_weights = &weights[i * (InChannels1 + InChannels2) * 9];
            for (int j = 0; j < InChannels1; j += S) {
                UNROLLED_FOR(k, 8, {
                    if (k < RowsPortion) {
                        val[k] = fmadd(fvec<S>{&p_weights[0 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[1 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[2 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);

                        val[k] = fmadd(fvec<S>{&p_weights[3 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[4 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[5 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);

                        val[k] = fmadd(fvec<S>{&p_weights[6 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[7 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[8 * InChannels1 + j], vector_aligned},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);
                    }
                })
            }
            p_weights += 9 * InChannels1;
            for (int j = 0; j < 3 * InChannels2; j += S) {
                if (RowsPortion == 8) {
                    UNROLLED_FOR(k, 8, {
                        val[k] = fmadd(fvec<S>{&p_weights[0 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 0] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[3 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 1] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[6 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 2] + j]}, val[k]);
                    })
                } else if (RowsPortion == 4) {
                    UNROLLED_FOR(k, 4, {
                        val[k] = fmadd(fvec<S>{&p_weights[0 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 0] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[3 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 1] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[6 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 2] + j]}, val[k]);
                    })
                } else {
                    for (int k = 0; k < RowsPortion; ++k) {
                        val[k] = fmadd(fvec<S>{&p_weights[0 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 0] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[3 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 1] + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[6 * InChannels2 + j], vector_aligned},
                                       fvec<S>{&data2[ii2[k + 2] + j]}, val[k]);
                    }
                }
            }

            for (int k = 0; k < RowsPortion; ++k) {
                float final_val = biases[i] + hsum(val[k]);
                if (Activation == eActivation::ReLU) {
                    final_val = fmaxf(0.0f, final_val);
                }
                output[OutChannels * ((y + k) * output_stride + x) + i] = final_val;
            }
        }
    }

#undef index1
#undef index2
}

template <int S, int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels, ePreOp PreOp1,
          ePreOp PreOp2, ePreOp PreOp3, ePostOp PostOp, eActivation Activation>
void Convolution3x3_GEMM(const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w,
                         int in_h, int w, int h, int stride, const float weights[], const float biases[],
                         float output[], int output_stride) {
    static_assert(S == 4 || S == 8 || S == 16, "!");
    if (!output_stride) {
        if (PostOp == ePostOp::Downscale) {
            output_stride = (w + 1) / 2;
        } else {
            output_stride = w;
        }
    }

    if (PostOp == ePostOp::Downscale) {
        for (int y = (rect.y / 2); y < (rect.y + rect.h + 1) / 2; ++y) {
            float *ptr = &output[OutChannels * (y * output_stride + (rect.x / 2))];
            std::fill(ptr, ptr + ((rect.w + 1) / 2) * OutChannels, 0.0f);
        }
    }

#define fetch1(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < in_w && (y) >= 0 && (y) < in_h)                                                                 \
        ? transfer::input<PreOp1>(data1[PxPitch * ((y)*stride + (x)) + (c)])                                           \
        : 0.0f
#define fetch2(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < in_w && (y) >= 0 && (y) < in_h)                                                                 \
        ? transfer::input<PreOp2>(data2[PxPitch * ((y)*stride + (x)) + (c)])                                           \
        : 0.0f
#define fetch3(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < in_w && (y) >= 0 && (y) < in_h)                                                                 \
        ? transfer::input<PreOp3>(data3[PxPitch * ((y)*stride + (x)) + (c)])                                           \
        : 0.0f

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        alignas(4 * S) float input[(InChannels1 + InChannels2 + InChannels3) * 9];
        for (int c = 0; c < InChannels1; ++c) {
            input[c * 9 + 0] = fetch1(y - 1, rect.x - 1, c);
            input[c * 9 + 1] = fetch1(y - 1, rect.x, c);

            input[c * 9 + 3] = fetch1(y, rect.x - 1, c);
            input[c * 9 + 4] = fetch1(y, rect.x, c);

            input[c * 9 + 6] = fetch1(y + 1, rect.x - 1, c);
            input[c * 9 + 7] = fetch1(y + 1, rect.x, c);
        }
        for (int c = 0; c < InChannels2; ++c) {
            input[(InChannels1 + c) * 9 + 0] = fetch2(y - 1, rect.x - 1, c);
            input[(InChannels1 + c) * 9 + 1] = fetch2(y - 1, rect.x, c);

            input[(InChannels1 + c) * 9 + 3] = fetch2(y, rect.x - 1, c);
            input[(InChannels1 + c) * 9 + 4] = fetch2(y, rect.x, c);

            input[(InChannels1 + c) * 9 + 6] = fetch2(y + 1, rect.x - 1, c);
            input[(InChannels1 + c) * 9 + 7] = fetch2(y + 1, rect.x, c);
        }
        for (int c = 0; c < InChannels3; ++c) {
            input[(InChannels1 + InChannels2 + c) * 9 + 0] = fetch3(y - 1, rect.x - 1, c);
            input[(InChannels1 + InChannels2 + c) * 9 + 1] = fetch3(y - 1, rect.x, c);

            input[(InChannels1 + InChannels2 + c) * 9 + 3] = fetch3(y, rect.x - 1, c);
            input[(InChannels1 + InChannels2 + c) * 9 + 4] = fetch3(y, rect.x, c);

            input[(InChannels1 + InChannels2 + c) * 9 + 6] = fetch3(y + 1, rect.x - 1, c);
            input[(InChannels1 + InChannels2 + c) * 9 + 7] = fetch3(y + 1, rect.x, c);
        }

        const int InChannels = (InChannels1 + InChannels2 + InChannels3);

        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            for (int c = 0; c < InChannels1; ++c) {
                input[c * 9 + 2] = fetch1(y - 1, x + 1, c);
                input[c * 9 + 5] = fetch1(y, x + 1, c);
                input[c * 9 + 8] = fetch1(y + 1, x + 1, c);
            }
            for (int c = 0; c < InChannels2; ++c) {
                input[(InChannels1 + c) * 9 + 2] = fetch2(y - 1, x + 1, c);
                input[(InChannels1 + c) * 9 + 5] = fetch2(y, x + 1, c);
                input[(InChannels1 + c) * 9 + 8] = fetch2(y + 1, x + 1, c);
            }
            for (int c = 0; c < InChannels3; ++c) {
                input[(InChannels1 + InChannels2 + c) * 9 + 2] = fetch3(y - 1, x + 1, c);
                input[(InChannels1 + InChannels2 + c) * 9 + 5] = fetch3(y, x + 1, c);
                input[(InChannels1 + InChannels2 + c) * 9 + 8] = fetch3(y + 1, x + 1, c);
            }

            for (int i = 0; i < OutChannels; ++i) {
                fvec<S> val = 0.0f;

                int j = 0;
                for (; j < InChannels * 9 - S + 1; j += S) {
                    val = fmadd(fvec<S>{&weights[i * InChannels * 9 + j]}, fvec<S>{&input[j], vector_aligned}, val);
                }

                float final_val = biases[i];
                final_val += hsum(val);

                for (; j < InChannels * 9; ++j) {
                    final_val += weights[i * InChannels * 9 + j] * input[j];
                }

                if (Activation == eActivation::ReLU) {
                    final_val = fmaxf(0.0f, final_val);
                }

                if (PostOp == ePostOp::Downscale) {
                    float &out = output[OutChannels * ((y / 2) * ((w + 1) / 2) + (x / 2)) + i];
                    out = fmaxf(out, final_val);
                } else {
                    output[OutChannels * (y * output_stride + x) + i] = final_val;
                }
            }

            for (int c = 0; c < InChannels; ++c) {
                input[c * 9 + 0] = input[c * 9 + 1];
                input[c * 9 + 1] = input[c * 9 + 2];

                input[c * 9 + 3] = input[c * 9 + 4];
                input[c * 9 + 4] = input[c * 9 + 5];

                input[c * 9 + 6] = input[c * 9 + 7];
                input[c * 9 + 7] = input[c * 9 + 8];
            }
        }
    }

#undef fetch1
#undef fetch2
#undef fetch3
}

template <int S, int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1, eActivation Activation>
void ConvolutionConcat3x3_GEMM(const float *__restrict data1, const float *__restrict data2, const rect_t &rect, int w,
                               int h, const float *__restrict weights, const float *__restrict biases,
                               float *__restrict output) {
    const int div1 = (PreOp1 == ePreOp::Upscale) ? 2 : 1;

    const int w1 = (w + div1 - 1) / div1, h1 = (h + div1 - 1) / div1;

#define fetch1(y, x, c) data1[InChannels1 * ((y)*w1 + (x)) + (c)]
#define fetch2(y, x, c) data2[InChannels2 * ((y)*w + (x)) + (c)]

    for (int y = rect.y; y < rect.y + rect.h; ++y) {
        alignas(S * 4) float input1[InChannels1 * 9], input2[InChannels2 * 9];
        for (int c = 0; c < InChannels1; ++c) {
            input1[c * 9 + 0] = fetch1(std::max((y - 1) / div1, 0), std::max((rect.x - 1) / div1, 0), c);
            input1[c * 9 + 1] = fetch1(std::max((y - 1) / div1, 0), rect.x / div1, c);

            input1[c * 9 + 3] = fetch1(y / div1, std::max((rect.x - 1) / div1, 0), c);
            input1[c * 9 + 4] = fetch1(y / div1, rect.x / div1, c);

            input1[c * 9 + 6] = fetch1(std::min((y + 1) / div1, h1 - 1), std::max((rect.x - 1) / div1, 0), c);
            input1[c * 9 + 7] = fetch1(std::min((y + 1) / div1, h1 - 1), rect.x / div1, c);
        }
        for (int c = 0; c < InChannels2; ++c) {
            input2[c * 9 + 0] = fetch2(std::max(y - 1, 0), std::max(rect.x - 1, 0), c);
            input2[c * 9 + 1] = fetch2(std::max(y - 1, 0), rect.x, c);

            input2[c * 9 + 3] = fetch2(y, std::max(rect.x - 1, 0), c);
            input2[c * 9 + 4] = fetch2(y, rect.x, c);

            input2[c * 9 + 6] = fetch2(std::min(y + 1, h - 1), std::max(rect.x - 1, 0), c);
            input2[c * 9 + 7] = fetch2(std::min(y + 1, h - 1), rect.x, c);
        }

        for (int x = rect.x; x < rect.x + rect.w; ++x) {
            for (int c = 0; c < InChannels1; ++c) {
                input1[c * 9 + 2] = fetch1(std::max((y - 1) / div1, 0), std::min((x + 1) / div1, w1 - 1), c);
                input1[c * 9 + 5] = fetch1(y / div1, std::min((x + 1) / div1, w1 - 1), c);
                input1[c * 9 + 8] = fetch1(std::min((y + 1) / div1, h1 - 1), std::min((x + 1) / div1, w1 - 1), c);
            }
            for (int c = 0; c < InChannels2; ++c) {
                input2[c * 9 + 2] = fetch2(std::max(y - 1, 0), std::min(x + 1, w - 1), c);
                input2[c * 9 + 5] = fetch2(y, std::min(x + 1, w - 1), c);
                input2[c * 9 + 8] = fetch2(std::min(y + 1, h - 1), std::min(x + 1, w - 1), c);
            }

            if ((InChannels1 % S) == 0 && InChannels1 >= S && (InChannels2 % S) == 0 && InChannels2 >= S) {
                for (int i = 0; i < OutChannels; ++i) {
                    fvec<S> val[3] = {0.0f};
                    for (int j = 0; j < InChannels1 * 9; j += S * 9) {
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 0 * S], vector_aligned},
                                  fvec<S>{&input1[j + 0 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 1 * S], vector_aligned},
                                  fvec<S>{&input1[j + 1 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 2 * S], vector_aligned},
                                  fvec<S>{&input1[j + 2 * S], vector_aligned}, val[2]);
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 3 * S], vector_aligned},
                                  fvec<S>{&input1[j + 3 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 4 * S], vector_aligned},
                                  fvec<S>{&input1[j + 4 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 5 * S], vector_aligned},
                                  fvec<S>{&input1[j + 5 * S], vector_aligned}, val[2]);
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 6 * S], vector_aligned},
                                  fvec<S>{&input1[j + 6 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 7 * S], vector_aligned},
                                  fvec<S>{&input1[j + 7 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 8 * S], vector_aligned},
                                  fvec<S>{&input1[j + 8 * S], vector_aligned}, val[2]);
                    }
                    for (int j = 0; j < InChannels2 * 9; j += S * 9) {
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 0 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 0 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 1 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 1 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 2 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 2 * S], vector_aligned}, val[2]);
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 3 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 3 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 4 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 4 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 5 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 5 * S], vector_aligned}, val[2]);
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 6 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 6 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 7 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 7 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 8 * S],
                                          vector_aligned},
                                  fvec<S>{&input2[j + 8 * S], vector_aligned}, val[2]);
                    }

                    val[0] += val[1];
                    val[0] += val[2];

                    float final_val = biases[i];
                    final_val += hsum(val[0]);
                    if (Activation == eActivation::ReLU) {
                        final_val = fmaxf(0.0f, final_val);
                    }

                    output[OutChannels * (y * w + x) + i] = final_val;
                }
            } else if ((InChannels1 % S) == 0 && InChannels1 >= S && InChannels2 == 3 && S <= 8) {
                for (int i = 0; i < OutChannels; ++i) {
                    fvec<S> val[3] = {0.0f, 0.0f, 0.0f};
                    for (int j = 0; j < InChannels1 * 9; j += S * 9) {
                        val[0] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 0 * S]},
                                       fvec<S>{&input1[j + 0 * S], vector_aligned}, val[0]);
                        val[1] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 1 * S]},
                                       fvec<S>{&input1[j + 1 * S], vector_aligned}, val[1]);
                        val[2] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 2 * S]},
                                       fvec<S>{&input1[j + 2 * S], vector_aligned}, val[2]);
                        val[0] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 3 * S]},
                                       fvec<S>{&input1[j + 3 * S], vector_aligned}, val[0]);
                        val[1] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 4 * S]},
                                       fvec<S>{&input1[j + 4 * S], vector_aligned}, val[1]);
                        val[2] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 5 * S]},
                                       fvec<S>{&input1[j + 5 * S], vector_aligned}, val[2]);
                        val[0] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 6 * S]},
                                       fvec<S>{&input1[j + 6 * S], vector_aligned}, val[0]);
                        val[1] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 7 * S]},
                                       fvec<S>{&input1[j + 7 * S], vector_aligned}, val[1]);
                        val[2] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + j + 8 * S]},
                                       fvec<S>{&input1[j + 8 * S], vector_aligned}, val[2]);
                    }

                    int j = 0;
                    for (; j < InChannels2 * 9 - S; j += S * 3) {
                        val[0] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 0 * S]},
                                  fvec<S>{&input2[j + 0 * S], vector_aligned}, val[0]);
                        val[1] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 1 * S]},
                                  fvec<S>{&input2[j + 1 * S], vector_aligned}, val[1]);
                        val[2] =
                            fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j + 2 * S]},
                                  fvec<S>{&input2[j + 2 * S], vector_aligned}, val[2]);
                    }
                    fvec<S> last_input = 0.0f;
                    last_input.template set<0>(input2[j + 0]);
                    last_input.template set<1>(input2[j + 1]);
                    last_input.template set<2>(input2[j + 2]);

                    val[0] = fmadd(fvec<S>{&weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j]},
                                   last_input, val[0]);

                    val[0] += val[1];
                    val[0] += val[2];

                    float final_val = biases[i];
                    final_val += hsum(val[0]);
                    if (Activation == eActivation::ReLU) {
                        final_val = fmaxf(0.0f, final_val);
                    }

                    output[OutChannels * (y * w + x) + i] = final_val;
                }
            } else {
                for (int i = 0; i < OutChannels; ++i) {
                    float val = biases[i];
                    for (int j = 0; j < InChannels1 * 9; ++j) {
                        val += weights[i * (InChannels1 + InChannels2) * 9 + j] * input1[j];
                    }
                    for (int j = 0; j < InChannels2 * 9; ++j) {
                        val += weights[i * (InChannels1 + InChannels2) * 9 + InChannels1 * 9 + j] * input2[j];
                    }
                    if (Activation == eActivation::ReLU) {
                        val = fmaxf(0.0f, val);
                    }

                    output[OutChannels * (y * w + x) + i] = val;
                }
            }

            for (int c = 0; c < InChannels1; ++c) {
                input1[c * 9 + 0] = input1[c * 9 + 1];
                input1[c * 9 + 1] = input1[c * 9 + 2];

                input1[c * 9 + 3] = input1[c * 9 + 4];
                input1[c * 9 + 4] = input1[c * 9 + 5];

                input1[c * 9 + 6] = input1[c * 9 + 7];
                input1[c * 9 + 7] = input1[c * 9 + 8];
            }

            for (int c = 0; c < InChannels2; ++c) {
                input2[c * 9 + 0] = input2[c * 9 + 1];
                input2[c * 9 + 1] = input2[c * 9 + 2];

                input2[c * 9 + 3] = input2[c * 9 + 4];
                input2[c * 9 + 4] = input2[c * 9 + 5];

                input2[c * 9 + 6] = input2[c * 9 + 7];
                input2[c * 9 + 7] = input2[c * 9 + 8];
            }
        }
    }

#undef fetch1
#undef fetch2
}

template <int RowsPortion, int S, int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch234,
          int OutChannels, ePreOp PreOp1, ePreOp PreOp2, ePreOp PreOp3, ePreOp PreOp4, ePostOp PostOp,
          eActivation Activation>
void ConvolutionConcat3x3_1Direct_2GEMM_ProcessRows(int y, const float data1[], const float data2[],
                                                    const float data3[], const float data4[], const rect_t &rect, int w,
                                                    int h, int w234, int h234, int stride1, int stride234,
                                                    const float *__restrict weights, const float biases[],
                                                    float *__restrict output, int output_stride) {
    const int div1 = (PreOp1 == ePreOp::Upscale) ? 2 : 1;

#define index1(y, x) InChannels1 *((y)*stride1 + (x))
#define fetch2(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < w234 && (y) >= 0 && (y) < h234)                                                                 \
        ? transfer::input<PreOp2>(data2[PxPitch234 * ((y)*stride234 + (x)) + (c)])                                     \
        : 0.0f
#define fetch3(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < w234 && (y) >= 0 && (y) < h234)                                                                 \
        ? transfer::input<PreOp3>(data3[PxPitch234 * ((y)*stride234 + (x)) + (c)])                                     \
        : 0.0f
#define fetch4(y, x, c)                                                                                                \
    ((x) >= 0 && (x) < w234 && (y) >= 0 && (y) < h234)                                                                 \
        ? transfer::input<PreOp4>(data4[PxPitch234 * ((y)*stride234 + (x)) + (c)])                                     \
        : 0.0f

    alignas(S * 4) float input234[8][(InChannels2 + InChannels3 + InChannels4) * 9];
    for (int k = 0; k < RowsPortion; ++k) {
        for (int c = 0; c < InChannels2; ++c) {
            input234[k][c * 9 + 0] = fetch2(y - 1 + k, rect.x - 1, c);
            input234[k][c * 9 + 1] = fetch2(y - 1 + k, rect.x, c);

            input234[k][c * 9 + 3] = fetch2(y + k, rect.x - 1, c);
            input234[k][c * 9 + 4] = fetch2(y + k, rect.x, c);

            input234[k][c * 9 + 6] = fetch2(y + 1 + k, rect.x - 1, c);
            input234[k][c * 9 + 7] = fetch2(y + 1 + k, rect.x, c);
        }
        for (int c = 0; c < InChannels3; ++c) {
            input234[k][(InChannels2 + c) * 9 + 0] = fetch3(y - 1 + k, rect.x - 1, c);
            input234[k][(InChannels2 + c) * 9 + 1] = fetch3(y - 1 + k, rect.x, c);

            input234[k][(InChannels2 + c) * 9 + 3] = fetch3(y + k, rect.x - 1, c);
            input234[k][(InChannels2 + c) * 9 + 4] = fetch3(y + k, rect.x, c);

            input234[k][(InChannels2 + c) * 9 + 6] = fetch3(y + 1 + k, rect.x - 1, c);
            input234[k][(InChannels2 + c) * 9 + 7] = fetch3(y + 1 + k, rect.x, c);
        }
        for (int c = 0; c < InChannels4; ++c) {
            input234[k][(InChannels2 + InChannels3 + c) * 9 + 0] = fetch4(y - 1 + k, rect.x - 1, c);
            input234[k][(InChannels2 + InChannels3 + c) * 9 + 1] = fetch4(y - 1 + k, rect.x, c);

            input234[k][(InChannels2 + InChannels3 + c) * 9 + 3] = fetch4(y + k, rect.x - 1, c);
            input234[k][(InChannels2 + InChannels3 + c) * 9 + 4] = fetch4(y + k, rect.x, c);

            input234[k][(InChannels2 + InChannels3 + c) * 9 + 6] = fetch4(y + 1 + k, rect.x - 1, c);
            input234[k][(InChannels2 + InChannels3 + c) * 9 + 7] = fetch4(y + 1 + k, rect.x, c);
        }
    }

    for (int x = rect.x; x < rect.x + rect.w; ++x) {
        const int add = ((x + 1) % div1);

        const int ii1[] = {
            index1(y == 0 ? -1 : (y - 1) / div1, x == 0 ? -1 : (x - 1) / div1), //
            index1(y / div1, x == 0 ? -1 : (x - 1) / div1),                     //
            index1((y + 1) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 2) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 3) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 4) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 5) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 6) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 7) / div1, x == 0 ? -1 : (x - 1) / div1),               //
            index1((y + 8) / div1, x == 0 ? -1 : (x - 1) / div1)                //
        };

        for (int c = 0; c < InChannels2; ++c) {
            UNROLLED_FOR(k, 8, {
                if (k < RowsPortion) {
                    input234[k][c * 9 + 2] = fetch2(y - 1 + k, x + 1, c);
                    input234[k][c * 9 + 5] = fetch2(y + k, x + 1, c);
                    input234[k][c * 9 + 8] = fetch2(y + 1 + k, x + 1, c);
                }
            })
        }
        for (int c = 0; c < InChannels3; ++c) {
            UNROLLED_FOR(k, 8, {
                if (k < RowsPortion) {
                    input234[k][(InChannels2 + c) * 9 + 2] = fetch3(y - 1 + k, x + 1, c);
                    input234[k][(InChannels2 + c) * 9 + 5] = fetch3(y + k, x + 1, c);
                    input234[k][(InChannels2 + c) * 9 + 8] = fetch3(y + 1 + k, x + 1, c);
                }
            })
        }
        for (int c = 0; c < InChannels4; ++c) {
            UNROLLED_FOR(k, 8, {
                if (k < RowsPortion) {
                    input234[k][(InChannels2 + InChannels3 + c) * 9 + 2] = fetch4(y - 1 + k, x + 1, c);
                    input234[k][(InChannels2 + InChannels3 + c) * 9 + 5] = fetch4(y + k, x + 1, c);
                    input234[k][(InChannels2 + InChannels3 + c) * 9 + 8] = fetch4(y + 1 + k, x + 1, c);
                }
            })
        }

        const int InChannels234 = (InChannels2 + InChannels3 + InChannels4);

        for (int i = 0; i < OutChannels; ++i) {
            fvec<S> val[8] = {};

            const float *p_weights = &weights[i * (InChannels1 + InChannels234) * 9];
            for (int j = 0; j < InChannels1; j += S) {
                UNROLLED_FOR(k, 8, {
                    if (k < RowsPortion) {
                        val[k] = fmadd(fvec<S>{&p_weights[0 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[1 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[2 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 0] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);

                        val[k] = fmadd(fvec<S>{&p_weights[3 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[4 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[5 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 1] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);

                        val[k] = fmadd(fvec<S>{&p_weights[6 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 0) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[7 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 1) / div1) * InChannels1 + j]}, val[k]);
                        val[k] = fmadd(fvec<S>{&p_weights[8 * InChannels1 + j]},
                                       fvec<S>{&data1[ii1[k + 2] + ((add + 2) / div1) * InChannels1 + j]}, val[k]);
                    }
                })
            }

            p_weights += InChannels1 * 9;

            int j = 0;
            for (; j < InChannels234 * 9 - S + 1; j += S) {
                UNROLLED_FOR(k, 8, {
                    if (k < RowsPortion) {
                        val[k] = fmadd(fvec<S>{&p_weights[j]}, fvec<S>{&input234[k][j]}, val[k]);
                    }
                })
            }

            for (int k = 0; k < RowsPortion; ++k) {
                fvec<S> last_input = 0.0f;
                UNROLLED_FOR(l, 16, {
                    if (l < ((InChannels234 * 9) % S)) {
                        last_input.template set<l>(input234[k][j + l]);
                    }
                })
                val[k] = fmadd(fvec<S>{&p_weights[j]}, last_input, val[k]);

                float final_val = biases[i] + hsum(val[k]);
                if (Activation == eActivation::ReLU) {
                    final_val = std::max(0.0f, final_val);
                }
                output[OutChannels * ((y + k) * output_stride + x) + i] = final_val;
            }
        }

        for (int c = 0; c < InChannels234; ++c) {
            UNROLLED_FOR(k, 8, {
                if (k < RowsPortion) {
                    input234[k][c * 9 + 0] = input234[k][c * 9 + 1];
                    input234[k][c * 9 + 1] = input234[k][c * 9 + 2];

                    input234[k][c * 9 + 3] = input234[k][c * 9 + 4];
                    input234[k][c * 9 + 4] = input234[k][c * 9 + 5];

                    input234[k][c * 9 + 6] = input234[k][c * 9 + 7];
                    input234[k][c * 9 + 7] = input234[k][c * 9 + 8];
                }
            })
        }
    }

#undef fetch3
#undef fetch2
#undef index1
}
} // namespace NS
} // namespace Ray