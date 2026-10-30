#pragma once

#include "../Types.h"

#include <cmath>

#pragma warning(push)
#pragma warning(disable : 6294) // Ill-defined for-loop

namespace Ray::NS {
using std::exp;
using std::log;
using std::pow;

// fp16 UNet tensors (uint16_t elements) on archs where the asm f16 kernels bind them (tensor_t in
// CoreSIMD.h); with RAY_NEON_FP16_CONV these C++ kernels run GPU-parity f16 numerics on hw f16vec8.
template <class TO>
void ClearBorders(const rect_t &rect, const int w, const int h, const bool downscaled, const int out_channels,
                  TO output[]) {
    if (!downscaled) {
        for (int y = rect.y; y < rect.y + rect.h; ++y) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.x == 0) {
                    output[out_channels * ((y + 1) * (w + 2) + 0) + i] = 0;
                }
                if (rect.x + rect.w == w) {
                    output[out_channels * ((y + 1) * (w + 2) + w + 1) + i] = 0;
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
                    output[out_channels * (x + 0) + i] = 0;
                }
                if (rect.y + rect.h == h) {
                    output[out_channels * ((h + 1) * (w + 2) + x + 0) + i] = 0;
                }
            }
        }
    } else {
        for (int y = (rect.y / 2); y < (rect.y + rect.h + 1) / 2; ++y) {
            for (int i = 0; i < out_channels; ++i) {
                if (rect.x == 0) {
                    output[out_channels * ((y + 1) * ((w + 1) / 2 + 2) + 0) + i] = 0;
                }
                if (rect.x + rect.w == w) {
                    output[out_channels * ((y + 1) * ((w + 1) / 2 + 2) + (w + 1) / 2 + 1) + i] = 0;
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
                    output[out_channels * (x + 0) + i] = 0;
                }
                if (rect.y + rect.h == h) {
                    output[out_channels * (((h + 1) / 2 + 1) * ((w + 1) / 2 + 2) + x) + i] = 0;
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

template <ePreOp PreOp> force_inline float input(const float val) {
    static const float norm_scale = 0.318967164f;

    if (PreOp == ePreOp::HDRTransfer) {
        if (val <= y0) {
            return a * val * norm_scale;
        } else if (val <= y1) {
            return (b * pow(val, c) + d) * norm_scale;
        } else {
            return (e * log(val + f) + g) * norm_scale;
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
            return pow((val - d) / b, 1.0f / c);
        } else {
            return exp((val - g) / e) - f;
        }
    } else if (PostOp == ePostOp::PositiveNormalize) {
        return 2.0f * val - 1.0f;
    } else {
        return val;
    }
}
} // namespace transfer

// uint16_t is the f16 tensor storage element type (see CoreSIMD.h tensor_t). f16 vector paths
// require the hardware f16vec8 - a soft fallback here would silently change UNet numerics.
// Float vectors must be S-wide to match step: REF/SSE41 conv run at S=4, AVX512 fallback at S=16.
template <bool Half, int S> using conv_vec = std::conditional_t<Half, f16vec<8>, fvec<S>>;

template <bool Half, int S> struct conv_traits {
    using V = conv_vec<Half, S>;
    static constexpr int step = Half ? 8 : S; // f16 vectors are 8 lanes; float vectors follow S
    static constexpr int bias_type = Half ? 1 : 0;
    static_assert(sizeof(f16vec8) == 16, "f16 conv step assumes 8 f16 lanes per vector");
    // NB: f16vec8::is_native() is checked with a runtime assert in the kernels (not constexpr-safe)
};

// broadcast scalar for the accumulator init: float for fp32 vectors; f16 biases are raw bit
// patterns in uint16_t storage, so they must be bit-reinterpreted, not numerically converted
template <int BiasIsF16, class TW> force_inline auto conv_bias(const TW b) {
    if constexpr (BiasIsF16) {
        float16 h;
        memcpy(&h, &b, sizeof(h));
        return h;
    } else
        return float(b);
}

// store an epilogue float into the output tensor (f16 storage rounds once, as the GPU/asm paths)
template <class TO> force_inline std::enable_if_t<!std::is_same_v<TO, uint16_t>> conv_store(TO *p, const float v) {
    *p = v;
}
template <class TO> force_inline std::enable_if_t<std::is_same_v<TO, uint16_t>> conv_store(TO *p, const float v) {
    *p = float16::f32_to_f16(v);
}

// staged tensor element from a float pre-op result: f16 storage packs bit patterns, never ints
template <class TT> force_inline TT conv_stage(const float v) {
    if constexpr (std::is_same_v<TT, uint16_t>)
        return float16::f32_to_f16(v);
    else
        return TT(v);
}

template <int S, int InChannels, int OutChannels, int OutPxPitch, ePostOp PostOp, eActivation Activation, class TD,
          class TW = float, class TO = float>
void Convolution3x3(const TD *restrict data, const rect_t &rect, const int w, const int h, const int stride,
                    const TW *restrict weights, const TW *restrict biases, TO *restrict output,
                    const int output_stride) {
    static_assert((InChannels % S) == 0, "!");

    constexpr bool half_w = std::is_same_v<TW, uint16_t>;
    static_assert(!std::is_same_v<TD, uint16_t> || half_w, "f16 conv tensors need f16 weights");
    using V = typename conv_traits<half_w, S>::V;
    constexpr int step = conv_traits<half_w, S>::step;
    assert(!half_w || V::is_native()); // only the f16 path forbids soft vectors; REF/SSE41 float is soft by design

    const int RoundedTriple = 8 * ((3 * InChannels + 7) / 8); // stride and offset must be aligned

#define C_ROWS 4
#define C_COLS 16

    for (int y = rect.y; y < rect.y + rect.h; y += 2) {
        for (int x = rect.x; x < rect.x + rect.w; x += C_ROWS) {
            for (int i = 0; i < OutChannels; i += C_COLS) {
                V C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
                for (int di = 0; di < C_COLS; ++di) {
                    C0[0][di] = {};
                    C0[0][di].template set<0>(conv_bias<conv_traits<half_w, S>::bias_type>(biases[i + di]));
                    UNROLLED_FOR(k, C_ROWS, { C1[k][di] = C0[k][di] = C0[0][di]; })
                }
                for (int j = 0; j < 3 * InChannels; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    UNROLLED_FOR(k, C_ROWS, {
                        A0[k] = V{&data[InChannels * ((y - 1) * stride + x - 1 + k) + j]};
                        A1[k] = V{&data[InChannels * ((y + 0) * stride + x - 1 + k) + j]};
                        A2[k] = V{&data[InChannels * ((y + 1) * stride + x - 1 + k) + j]};
                        A3[k] = V{&data[InChannels * ((y + 2) * stride + x - 1 + k) + j]};
                    })
                    const TW *p_weights = &weights[i * RoundedTriple * 3 + j];
                    for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                        const V B0 = V{&p_weights[0 * RoundedTriple], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple], vector_aligned};
                        p_weights += RoundedTriple * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][ii - i] = fmadd(A0[k], B0, C0[k][ii - i]);
                            C0[k][ii - i] = fmadd(A1[k], B1, C0[k][ii - i]);
                            C0[k][ii - i] = fmadd(A2[k], B2, C0[k][ii - i]);

                            C1[k][ii - i] = fmadd(A1[k], B0, C1[k][ii - i]);
                            C1[k][ii - i] = fmadd(A2[k], B1, C1[k][ii - i]);
                            C1[k][ii - i] = fmadd(A3[k], B2, C1[k][ii - i]);
                        })
                    }
                }
                float _C0[C_ROWS][C_COLS], _C1[C_ROWS][C_COLS];
                for (int k = 0; k < C_ROWS; ++k) {
                    for (int di = 0; di < C_COLS; ++di) {
                        _C0[k][di] = float(hsum(C0[k][di]));
                        _C1[k][di] = float(hsum(C1[k][di]));
                        if (Activation == eActivation::ReLU) {
                            _C0[k][di] = std::max(0.0f, _C0[k][di]);
                            _C1[k][di] = std::max(0.0f, _C1[k][di]);
                        }
                    }
                }
                if (PostOp == ePostOp::Downsample) {
                    for (int k = 0; k < C_ROWS; k += 2) {
                        for (int di = 0; di < C_COLS; ++di) {
                            _C0[k][di] =
                                std::max(std::max(_C0[k][di], _C0[k + 1][di]), std::max(_C1[k][di], _C1[k + 1][di]));
                            _C0[k][di] = std::max(0.0f, _C0[k][di]);
                        }
                    }
                    for (int xx = x; xx < std::min(x + C_ROWS, rect.x + rect.w); xx += 2) {
                        for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                            conv_store(&output[OutPxPitch * ((y / 2) * output_stride + (xx / 2)) + ii],
                                       transfer::output<PostOp>(_C0[xx - x][ii - i]));
                        }
                    }
                } else {
                    for (int xx = x; xx < std::min(x + C_ROWS, rect.x + rect.w); ++xx) {
                        for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                            conv_store(&output[OutPxPitch * ((y + 0) * output_stride + xx) + ii],
                                       transfer::output<PostOp>(_C0[xx - x][ii - i]));
                            if (y + 1 < rect.y + rect.h) {
                                conv_store(&output[OutPxPitch * ((y + 1) * output_stride + xx) + ii],
                                           transfer::output<PostOp>(_C1[xx - x][ii - i]));
                            }
                        }
                    }
                }
            }
        }
    }

#undef C_ROWS
#undef C_COLS
}

// Gathers the 3 inputs' rect+1px halo into temp_data as one InChannels123 tensor (pre-ops applied,
// OOB zero, RoundedTriple tail pad); shared with the asm pass-0 walk; TT follows the conv mode.
template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, ePreOp PreOp1, ePreOp PreOp2, ePreOp PreOp3,
          class TT>
void StageConv3x3(const float *restrict data1, const float *restrict data2, const float *restrict data3,
                  const rect_t &rect, const int in_w, const int in_h, const int stride,
                  aligned_vector<TT, 64> &temp_data) {
    const int InChannels123 = (InChannels1 + InChannels2 + InChannels3);
    const int RoundedTriple = 8 * ((3 * InChannels123 + 7) / 8); // stride and offset must be aligned
    const int stride123 = rect.w + 2, req_size = InChannels123 * stride123 * (rect.h + 2);
    temp_data.resize(req_size + RoundedTriple);
    TT *data123 = temp_data.data();
    int index = 0;
    for (int y = rect.y - 1; y < rect.y + rect.h + 1; ++y) {
        for (int x = rect.x - 1; x < rect.x + rect.w + 1; ++x) {
            if (x >= 0 && x < in_w && y >= 0 && y < in_h) {
                for (int j = 0; j < InChannels1; ++j) {
                    data123[index++] = conv_stage<TT>(transfer::input<PreOp1>(data1[PxPitch * (y * stride + x) + j]));
                }
                for (int j = 0; j < InChannels2; ++j) {
                    data123[index++] = conv_stage<TT>(transfer::input<PreOp2>(data2[PxPitch * (y * stride + x) + j]));
                }
                for (int j = 0; j < InChannels3; ++j) {
                    data123[index++] = conv_stage<TT>(transfer::input<PreOp3>(data3[PxPitch * (y * stride + x) + j]));
                }
            } else {
                for (int j = 0; j < InChannels123; ++j) {
                    data123[index++] = TT(0);
                }
            }
        }
    }
    while (index < temp_data.size()) {
        data123[index++] = TT(0);
    }
}

template <int S, int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels, ePreOp PreOp1,
          ePreOp PreOp2, ePreOp PreOp3, ePostOp PostOp, eActivation Activation, class TW = float, class TO = float,
          class TT = TW>
void Convolution3x3(const float *restrict data1, const float *restrict data2, const float *restrict data3,
                    const rect_t &rect, const int in_w, const int in_h, const int w, const int h, const int stride,
                    const TW *restrict weights, const TW *restrict biases, TO *restrict output, const int output_stride,
                    aligned_vector<TT, 64> &temp_data) {
    const int InChannels123 = (InChannels1 + InChannels2 + InChannels3);

    constexpr bool half_w = std::is_same_v<TW, uint16_t>;
    static_assert(std::is_same_v<TT, TW>, "staged tensor must match the conv vector type");
    using V = typename conv_traits<half_w, S>::V;
    constexpr int step = conv_traits<half_w, S>::step;
    assert(!half_w || V::is_native()); // only the f16 path forbids soft vectors; REF/SSE41 float is soft by design

    const int RoundedTriple = 8 * ((3 * InChannels123 + 7) / 8); // stride and offset must be aligned
    const int stride123 = rect.w + 2;

    StageConv3x3<InChannels1, InChannels2, InChannels3, PxPitch, PreOp1, PreOp2, PreOp3>(data1, data2, data3, rect,
                                                                                         in_w, in_h, stride, temp_data);
    TT *data123 = temp_data.data();

#define C_ROWS 4
#define C_COLS 16

    for (int y = rect.y; y < rect.y + rect.h; y += 2) {
        for (int x = rect.x; x < rect.x + rect.w; x += C_ROWS) {
            for (int i = 0; i < OutChannels; i += C_COLS) {
                V C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
                for (int di = 0; di < C_COLS; ++di) {
                    C0[0][di] = {};
                    C0[0][di].template set<0>(conv_bias<conv_traits<half_w, S>::bias_type>(biases[i + di]));
                    UNROLLED_FOR(k, C_ROWS, { C1[k][di] = C0[k][di] = C0[0][di]; })
                }
                for (int j = 0; j < 3 * InChannels123; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    const TT *p_data123 = &data123[InChannels123 * ((y - rect.y) * stride123 + x - rect.x) + j];
                    UNROLLED_FOR(k, C_ROWS, {
                        A0[k] = V{&p_data123[InChannels123 * (0 * stride123 + k)]};
                        A1[k] = V{&p_data123[InChannels123 * (1 * stride123 + k)]};
                        A2[k] = V{&p_data123[InChannels123 * (2 * stride123 + k)]};
                        A3[k] = V{&p_data123[InChannels123 * (3 * stride123 + k)]};
                    })
                    const TW *p_weights = &weights[i * RoundedTriple * 3 + j];
                    static_assert((OutChannels % C_COLS) == 0, "!");
                    for (int di = 0; di < C_COLS; ++di) {
                        const V B0 = V{&p_weights[0 * RoundedTriple], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple], vector_aligned};
                        p_weights += RoundedTriple * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][di] = fmadd(A0[k], B0, C0[k][di]);
                            C0[k][di] = fmadd(A1[k], B1, C0[k][di]);
                            C0[k][di] = fmadd(A2[k], B2, C0[k][di]);

                            C1[k][di] = fmadd(A1[k], B0, C1[k][di]);
                            C1[k][di] = fmadd(A2[k], B1, C1[k][di]);
                            C1[k][di] = fmadd(A3[k], B2, C1[k][di]);
                        })
                    }
                }
                float _C0[C_ROWS][C_COLS], _C1[C_ROWS][C_COLS];
                for (int k = 0; k < C_ROWS; ++k) {
                    for (int di = 0; di < C_COLS; ++di) {
                        _C0[k][di] = float(hsum(C0[k][di]));
                        _C1[k][di] = float(hsum(C1[k][di]));
                        if (Activation == eActivation::ReLU) {
                            _C0[k][di] = std::max(0.0f, _C0[k][di]);
                            _C1[k][di] = std::max(0.0f, _C1[k][di]);
                        }
                    }
                }
                for (int xx = x; xx < std::min(x + C_ROWS, rect.x + rect.w); ++xx) {
                    for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                        conv_store(&output[OutChannels * ((y + 0) * output_stride + xx) + ii],
                                   transfer::output<PostOp>(_C0[xx - x][ii - i]));
                        if (y + 1 < rect.y + rect.h) {
                            conv_store(&output[OutChannels * ((y + 1) * output_stride + xx) + ii],
                                       transfer::output<PostOp>(_C1[xx - x][ii - i]));
                        }
                    }
                }
            }
        }
    }

#undef C_ROWS
#undef C_COLS
}

template <int S, int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1, ePostOp PostOp,
          eActivation Activation, class TD1 = float, class TD2 = float, class TW = float, class TO = float>
void ConvolutionConcat3x3(const TD1 *restrict data1, const TD2 *restrict data2, const rect_t &rect, const int w,
                          const int h, const int stride1, const int stride2, const TW *restrict weights,
                          const TW *restrict biases, TO *restrict output, const int output_stride) {
    static_assert((InChannels1 % S) == 0 && (InChannels2 % S) == 0, "!");

    constexpr bool half_w = std::is_same_v<TW, uint16_t>;
    static_assert(!std::is_same_v<TD1, uint16_t> || half_w, "f16 conv tensors need f16 weights");
    static_assert(!std::is_same_v<TD2, uint16_t> || half_w, "f16 conv tensors need f16 weights");
    using V = typename conv_traits<half_w, S>::V;
    constexpr int step = conv_traits<half_w, S>::step;
    assert(!half_w || V::is_native()); // only the f16 path forbids soft vectors; REF/SSE41 float is soft by design

    const int RoundedTriple1 = 8 * ((3 * InChannels1 + 7) / 8),
              RoundedTriple2 = 8 * ((3 * InChannels2 + 7) / 8); // stride and offset must be aligned

#define index1(y, x) InChannels1 *((y) * stride1 + (x))
#define index2(y, x) InChannels2 *((y) * stride2 + (x))

#define C_ROWS 4
#define C_COLS 16

    for (int y = rect.y; y < rect.y + rect.h; y += 2) {
        for (int x = rect.x; x < rect.x + rect.w; x += C_ROWS) {
            for (int i = 0; i < OutChannels; i += C_COLS) {
                V C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
                for (int di = 0; di < C_COLS; ++di) {
                    C0[0][di] = {};
                    C0[0][di].template set<0>(conv_bias<conv_traits<half_w, S>::bias_type>(biases[i + di]));
                    UNROLLED_FOR(k, C_ROWS, { C1[k][di] = C0[k][di] = C0[0][di]; })
                }
                for (int j = 0; j < 3 * InChannels1; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    if (PreOp1 == ePreOp::Upsample) {
                        const int x_off = (j / InChannels1), ch = (j % InChannels1);
                        UNROLLED_FOR(k, C_ROWS, {
                            const int x_final = (x + x_off + k - 1 + 2) / 2 - 1;
                            A0[k] = V{&data1[index1((y == 0 ? -1 : (y - 1) / 2), x_final) + ch]};
                            A1[k] = V{&data1[index1((y + 0) / 2, x_final) + ch]};
                            A2[k] = V{&data1[index1((y + 1) / 2, x_final) + ch]};
                            A3[k] = V{&data1[index1((y + 2) / 2, x_final) + ch]};
                        })
                    } else {
                        UNROLLED_FOR(k, C_ROWS, {
                            A0[k] = V{&data1[index1(y - 1, x - 1 + k) + j]};
                            A1[k] = V{&data1[index1(y + 0, x - 1 + k) + j]};
                            A2[k] = V{&data1[index1(y + 1, x - 1 + k) + j]};
                            A3[k] = V{&data1[index1(y + 2, x - 1 + k) + j]};
                        })
                    }
                    const TW *p_weights = &weights[i * (RoundedTriple1 + RoundedTriple2) * 3 + j];
                    static_assert((OutChannels % C_COLS) == 0, "!");
                    for (int di = 0; di < C_COLS; ++di) {
                        const V B0 = V{&p_weights[0 * RoundedTriple1], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple1], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple1], vector_aligned};
                        p_weights += (RoundedTriple1 + RoundedTriple2) * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][di] = fmadd(A0[k], B0, C0[k][di]);
                            C0[k][di] = fmadd(A1[k], B1, C0[k][di]);
                            C0[k][di] = fmadd(A2[k], B2, C0[k][di]);

                            C1[k][di] = fmadd(A1[k], B0, C1[k][di]);
                            C1[k][di] = fmadd(A2[k], B1, C1[k][di]);
                            C1[k][di] = fmadd(A3[k], B2, C1[k][di]);
                        })
                    }
                }
                for (int j = 0; j < 3 * InChannels2; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    UNROLLED_FOR(k, C_ROWS, {
                        A0[k] = V{&data2[index2(y - 1, x - 1 + k) + j]};
                        A1[k] = V{&data2[index2(y + 0, x - 1 + k) + j]};
                        A2[k] = V{&data2[index2(y + 1, x - 1 + k) + j]};
                        A3[k] = V{&data2[index2(y + 2, x - 1 + k) + j]};
                    })
                    const TW *p_weights = &weights[i * (RoundedTriple1 + RoundedTriple2) * 3 + 3 * RoundedTriple1 + j];
                    static_assert((OutChannels % C_COLS) == 0, "!");
                    for (int di = 0; di < C_COLS; ++di) {
                        const V B0 = V{&p_weights[0 * RoundedTriple2], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple2], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple2], vector_aligned};
                        p_weights += (RoundedTriple1 + RoundedTriple2) * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][di] = fmadd(A0[k], B0, C0[k][di]);
                            C0[k][di] = fmadd(A1[k], B1, C0[k][di]);
                            C0[k][di] = fmadd(A2[k], B2, C0[k][di]);

                            C1[k][di] = fmadd(A1[k], B0, C1[k][di]);
                            C1[k][di] = fmadd(A2[k], B1, C1[k][di]);
                            C1[k][di] = fmadd(A3[k], B2, C1[k][di]);
                        })
                    }
                }
                float _C0[C_ROWS][C_COLS], _C1[C_ROWS][C_COLS];
                for (int k = 0; k < C_ROWS; ++k) {
                    for (int di = 0; di < C_COLS; ++di) {
                        _C0[k][di] = float(hsum(C0[k][di]));
                        _C1[k][di] = float(hsum(C1[k][di]));
                        if (Activation == eActivation::ReLU) {
                            _C0[k][di] = std::max(0.0f, _C0[k][di]);
                            _C1[k][di] = std::max(0.0f, _C1[k][di]);
                        }
                    }
                }
                for (int xx = x; xx < std::min(x + C_ROWS, rect.x + rect.w); ++xx) {
                    for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                        conv_store(&output[OutChannels * ((y + 0) * output_stride + xx) + ii],
                                   transfer::output<PostOp>(_C0[xx - x][ii - i]));
                        if (y + 1 < rect.y + rect.h) {
                            conv_store(&output[OutChannels * ((y + 1) * output_stride + xx) + ii],
                                       transfer::output<PostOp>(_C1[xx - x][ii - i]));
                        }
                    }
                }
            }
        }
    }

#undef C_ROWS
#undef C_COLS

#undef index1
#undef index2
}

// Gathers the 3 aux inputs' rect+1px halo into temp_data as one InChannels234 tensor (as above);
// shared with the asm pass-13 walk; TT follows the conv mode.
template <int InChannels2, int InChannels3, int InChannels4, int PxPitch234, ePreOp PreOp2, ePreOp PreOp3,
          ePreOp PreOp4, class TT>
void StageConvConcat3x3(const float *restrict data2, const float *restrict data3, const float *restrict data4,
                        const rect_t &rect, const int w234, const int h234, const int stride234,
                        aligned_vector<TT, 64> &temp_data) {
    const int InChannels234 = (InChannels2 + InChannels3 + InChannels4);
    const int RoundedTriple2 = 8 * ((3 * InChannels234 + 7) / 8); // stride and offset must be aligned
    const int _stride234 = rect.w + 2, req_size = InChannels234 * _stride234 * (rect.h + 2);
    temp_data.resize(req_size + RoundedTriple2);
    TT *data234 = temp_data.data();
    int index = 0;
    for (int y = rect.y - 1; y < rect.y + rect.h + 1; ++y) {
        for (int x = rect.x - 1; x < rect.x + rect.w + 1; ++x) {
            if (x >= 0 && x < w234 && y >= 0 && y < h234) {
                for (int j = 0; j < InChannels2; ++j) {
                    data234[index++] =
                        conv_stage<TT>(transfer::input<PreOp2>(data2[PxPitch234 * (y * stride234 + x) + j]));
                }
                for (int j = 0; j < InChannels3; ++j) {
                    data234[index++] =
                        conv_stage<TT>(transfer::input<PreOp3>(data3[PxPitch234 * (y * stride234 + x) + j]));
                }
                for (int j = 0; j < InChannels4; ++j) {
                    data234[index++] =
                        conv_stage<TT>(transfer::input<PreOp4>(data4[PxPitch234 * (y * stride234 + x) + j]));
                }
            } else {
                for (int j = 0; j < InChannels234; ++j) {
                    data234[index++] = TT(0);
                }
            }
        }
    }
    while (index < temp_data.size()) {
        data234[index++] = TT(0);
    }
}

template <int S, int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch234, int OutChannels,
          ePreOp PreOp1, ePreOp PreOp2, ePreOp PreOp3, ePreOp PreOp4, ePostOp PostOp, eActivation Activation,
          class TD = float, class TW = float, class TO = float, class TT = TW>
void ConvolutionConcat3x3(const TD *restrict data1, const float *restrict data2, const float *restrict data3,
                          const float *restrict data4, const rect_t &rect, const int w, const int h, const int w234,
                          const int h234, const int stride1, const int stride234, const TW *restrict weights,
                          const TW *restrict biases, TO *restrict output, int output_stride,
                          aligned_vector<TT, 64> &temp_data) {
    const int InChannels234 = (InChannels2 + InChannels3 + InChannels4);

    constexpr bool half_w = std::is_same_v<TW, uint16_t>;
    static_assert(!std::is_same_v<TD, uint16_t> || half_w, "f16 conv tensors need f16 weights");
    static_assert(std::is_same_v<TT, TW>, "staged tensor must match the conv vector type");
    using V = typename conv_traits<half_w, S>::V;
    constexpr int step = conv_traits<half_w, S>::step;
    assert(!half_w || V::is_native()); // only the f16 path forbids soft vectors; REF/SSE41 float is soft by design

    const int RoundedTriple1 = 8 * ((3 * InChannels1 + 7) / 8),
              RoundedTriple2 = 8 * ((3 * InChannels234 + 7) / 8); // stride and offset must be aligned

#define index1(y, x) InChannels1 *((y) * stride1 + (x))

    const int _stride234 = rect.w + 2;
    StageConvConcat3x3<InChannels2, InChannels3, InChannels4, PxPitch234, PreOp2, PreOp3, PreOp4>(
        data2, data3, data4, rect, w234, h234, stride234, temp_data);
    TT *data234 = temp_data.data();

#define C_ROWS 4
#define C_COLS 16

    for (int y = rect.y; y < rect.y + rect.h; y += 2) {
        for (int x = rect.x; x < rect.x + rect.w; x += C_ROWS) {
            for (int i = 0; i < OutChannels; i += C_COLS) {
                V C0[C_ROWS][C_COLS], C1[C_ROWS][C_COLS];
                for (int di = 0; di < C_COLS; ++di) {
                    C0[0][di] = {};
                    C0[0][di].template set<0>(conv_bias<conv_traits<half_w, S>::bias_type>(biases[i + di]));
                    UNROLLED_FOR(k, C_ROWS, { C1[k][di] = C0[k][di] = C0[0][di]; })
                }
                for (int j = 0; j < 3 * InChannels1; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    if (PreOp1 == ePreOp::Upsample) {
                        const int x_off = (j / InChannels1), ch = (j % InChannels1);
                        UNROLLED_FOR(k, C_ROWS, {
                            const int x_final = (x + x_off + k - 1 + 2) / 2 - 1;
                            A0[k] = V{&data1[index1((y == 0 ? -1 : (y - 1) / 2), x_final) + ch]};
                            A1[k] = V{&data1[index1((y + 0) / 2, x_final) + ch]};
                            A2[k] = V{&data1[index1((y + 1) / 2, x_final) + ch]};
                            A3[k] = V{&data1[index1((y + 2) / 2, x_final) + ch]};
                        })
                    } else {
                        UNROLLED_FOR(k, C_ROWS, {
                            A0[k] = V{&data1[index1(y - 1, x - 1 + k) + j]};
                            A1[k] = V{&data1[index1(y + 0, x - 1 + k) + j]};
                            A2[k] = V{&data1[index1(y + 1, x - 1 + k) + j]};
                            A3[k] = V{&data1[index1(y + 2, x - 1 + k) + j]};
                        })
                    }
                    const TW *p_weights = &weights[i * (RoundedTriple1 + RoundedTriple2) * 3 + j];
                    static_assert((OutChannels % C_COLS) == 0, "!");
                    for (int di = 0; di < C_COLS; ++di) {
                        const V B0 = V{&p_weights[0 * RoundedTriple1], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple1], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple1], vector_aligned};
                        p_weights += (RoundedTriple1 + RoundedTriple2) * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][di] = fmadd(A0[k], B0, C0[k][di]);
                            C0[k][di] = fmadd(A1[k], B1, C0[k][di]);
                            C0[k][di] = fmadd(A2[k], B2, C0[k][di]);

                            C1[k][di] = fmadd(A1[k], B0, C1[k][di]);
                            C1[k][di] = fmadd(A2[k], B1, C1[k][di]);
                            C1[k][di] = fmadd(A3[k], B2, C1[k][di]);
                        })
                    }
                }
                for (int j = 0; j < 3 * InChannels234; j += step) {
                    V A0[C_ROWS], A1[C_ROWS], A2[C_ROWS], A3[C_ROWS];
                    const TT *p_data234 = &data234[InChannels234 * ((y - rect.y) * _stride234 + x - rect.x) + j];
                    UNROLLED_FOR(k, C_ROWS, {
                        A0[k] = V{&p_data234[InChannels234 * (0 * _stride234 + k)]};
                        A1[k] = V{&p_data234[InChannels234 * (1 * _stride234 + k)]};
                        A2[k] = V{&p_data234[InChannels234 * (2 * _stride234 + k)]};
                        A3[k] = V{&p_data234[InChannels234 * (3 * _stride234 + k)]};
                    })
                    const TW *p_weights = &weights[i * (RoundedTriple1 + RoundedTriple2) * 3 + 3 * RoundedTriple1 + j];
                    static_assert((OutChannels % C_COLS) == 0, "!");
                    for (int di = 0; di < C_COLS; ++di) {
                        const V B0 = V{&p_weights[0 * RoundedTriple2], vector_aligned};
                        const V B1 = V{&p_weights[1 * RoundedTriple2], vector_aligned};
                        const V B2 = V{&p_weights[2 * RoundedTriple2], vector_aligned};
                        p_weights += (RoundedTriple1 + RoundedTriple2) * 3;

                        UNROLLED_FOR(k, C_ROWS, {
                            C0[k][di] = fmadd(A0[k], B0, C0[k][di]);
                            C0[k][di] = fmadd(A1[k], B1, C0[k][di]);
                            C0[k][di] = fmadd(A2[k], B2, C0[k][di]);

                            C1[k][di] = fmadd(A1[k], B0, C1[k][di]);
                            C1[k][di] = fmadd(A2[k], B1, C1[k][di]);
                            C1[k][di] = fmadd(A3[k], B2, C1[k][di]);
                        })
                    }
                }
                float _C0[C_ROWS][C_COLS], _C1[C_ROWS][C_COLS];
                for (int k = 0; k < C_ROWS; ++k) {
                    for (int di = 0; di < C_COLS; ++di) {
                        _C0[k][di] = float(hsum(C0[k][di]));
                        _C1[k][di] = float(hsum(C1[k][di]));
                        if (Activation == eActivation::ReLU) {
                            _C0[k][di] = std::max(0.0f, _C0[k][di]);
                            _C1[k][di] = std::max(0.0f, _C1[k][di]);
                        }
                    }
                }
                for (int xx = x; xx < std::min(x + C_ROWS, rect.x + rect.w); ++xx) {
                    for (int ii = i; ii < std::min(i + C_COLS, OutChannels); ++ii) {
                        conv_store(&output[OutChannels * ((y + 0) * output_stride + xx) + ii],
                                   transfer::output<PostOp>(_C0[xx - x][ii - i]));
                        if (y + 1 < rect.y + rect.h) {
                            conv_store(&output[OutChannels * ((y + 1) * output_stride + xx) + ii],
                                       transfer::output<PostOp>(_C1[xx - x][ii - i]));
                        }
                    }
                }
            }
        }
    }

#undef C_ROWS
#undef C_COLS

#undef index1
}
} // namespace Ray::NS

#pragma warning(pop)
