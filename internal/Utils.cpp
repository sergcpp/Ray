#include "Utils.h"

#include <deque>
#include <limits>

#include "CoreRef.h"

#include "simd/detect.h"

#define _MIN(x, y) ((x) < (y) ? (x) : (y))
#define _MAX(x, y) ((x) < (y) ? (y) : (x))
#define _ABS(x) ((x) < 0 ? -(x) : (x))
#define _CLAMP(x, lo, hi) (_MIN(_MAX((x), (lo)), (hi)))

#define _MIN3(x, y, z) _MIN((x), _MIN((y), (z)))
#define _MAX3(x, y, z) _MAX((x), _MAX((y), (z)))

#define _MIN4(x, y, z, w) _MIN(_MIN((x), (y)), _MIN((z), (w)))
#define _MAX4(x, y, z, w) _MAX(_MAX((x), (y)), _MAX((z), (w)))

namespace Ray {
uint16_t f32_to_f16(const float value) {
    int32_t i;
    memcpy(&i, &value, sizeof(float));

    int32_t s = (i >> 16) & 0x00008000;
    int32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
    int32_t m = i & 0x007fffff;
    if (e <= 0) {
        if (e < -10) {
            uint16_t ret;
            memcpy(&ret, &s, sizeof(uint16_t));
            return ret;
        }

        m = (m | 0x00800000) >> (1 - e);

        if (m & 0x00001000)
            m += 0x00002000;

        s = s | (m >> 13);
        uint16_t ret;
        memcpy(&ret, &s, sizeof(uint16_t));
        return ret;
    } else if (e == 0xff - (127 - 15)) {
        if (m == 0) {
            s = s | 0x7c00;
            uint16_t ret;
            memcpy(&ret, &s, sizeof(uint16_t));
            return ret;
        } else {
            m >>= 13;

            s = s | 0x7c00 | m | (m == 0);
            uint16_t ret;
            memcpy(&ret, &s, sizeof(uint16_t));
            return ret;
        }
    } else {
        if (m & 0x00001000) {
            m += 0x00002000;

            if (m & 0x00800000) {
                m = 0;  // overflow in significand,
                e += 1; // adjust exponent
            }
        }

        if (e > 30) {
            s = s | 0x7c00;
            uint16_t ret;
            memcpy(&ret, &s, sizeof(uint16_t));
            return ret;
        }

        s = s | (e << 10) | (m >> 13);
        uint16_t ret;
        memcpy(&ret, &s, sizeof(uint16_t));
        return ret;
    }
}

force_inline int16_t f32_to_s16(const float value) { return int16_t(value * 32767); }

force_inline uint16_t f32_to_u16(const float value) { return uint16_t(value * 65535); }

/*
    RGB <-> YCoCg

    Y  = [ 1/4  1/2   1/4] [R]
    Co = [ 1/2    0  -1/2] [G]
    CG = [-1/4  1/2  -1/4] [B]

    R  = [   1    1    -1] [Y]
    G  = [   1    0     1] [Co]
    B  = [   1   -1    -1] [Cg]
*/

force_inline uint8_t to_clamped_uint8(const int x) { return ((x) < 0 ? (0) : ((x) > 255 ? 255 : (x))); }

//
// Perfectly reversible RGB <-> YCoCg conversion (relies on integer wrap around)
//

force_inline void RGB_to_YCoCg_reversible(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) {
    out_YCoCg[1] = in_RGB[0] - in_RGB[2];
    const uint8_t t = in_RGB[2] + (out_YCoCg[1] >> 1);
    out_YCoCg[2] = in_RGB[1] - t;
    out_YCoCg[0] = t + (out_YCoCg[2] >> 1);
}

force_inline void YCoCg_to_RGB_reversible(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) {
    const uint8_t t = in_YCoCg[0] - (in_YCoCg[2] >> 1);
    out_RGB[1] = in_YCoCg[2] + t;
    out_RGB[2] = t - (in_YCoCg[1] >> 1);
    out_RGB[0] = in_YCoCg[1] + out_RGB[2];
}

//
// Not-so-perfectly reversible RGB <-> YCoCg conversion (to use in shaders)
//

force_inline void RGB_to_YCoCg(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) {
    const int R = int(in_RGB[0]);
    const int G = int(in_RGB[1]);
    const int B = int(in_RGB[2]);

    out_YCoCg[0] = (R + 2 * G + B) / 4;
    out_YCoCg[1] = to_clamped_uint8(128 + (R - B) / 2);
    out_YCoCg[2] = to_clamped_uint8(128 + (-R + 2 * G - B) / 4);
}

force_inline void YCoCg_to_RGB(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) {
    const int Y = int(in_YCoCg[0]);
    const int Co = int(in_YCoCg[1]) - 128;
    const int Cg = int(in_YCoCg[2]) - 128;

    out_RGB[0] = to_clamped_uint8(Y + Co - Cg);
    out_RGB[1] = to_clamped_uint8(Y + Cg);
    out_RGB[2] = to_clamped_uint8(Y - Co - Cg);
}

const uint8_t _blank_BC3_block_4x4[] = {0x00, 0x00, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24,
                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const int _blank_BC3_block_4x4_len = sizeof(_blank_BC3_block_4x4);

const uint8_t _blank_ASTC_block_4x4[] = {0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const int _blank_ASTC_block_4x4_len = sizeof(_blank_ASTC_block_4x4);
} // namespace Ray

void Ray::RGBMDecode(const uint8_t rgbm[4], float out_rgb[3]) {
    out_rgb[0] = 4.0f * (rgbm[0] / 255.0f) * (rgbm[3] / 255.0f);
    out_rgb[1] = 4.0f * (rgbm[1] / 255.0f) * (rgbm[3] / 255.0f);
    out_rgb[2] = 4.0f * (rgbm[2] / 255.0f) * (rgbm[3] / 255.0f);
}

void Ray::RGBMEncode(const float rgb[3], uint8_t out_rgbm[4]) {
    float fr = rgb[0] / 4.0f;
    float fg = rgb[1] / 4.0f;
    float fb = rgb[2] / 4.0f;
    float fa = std::max(std::max(fr, fg), std::max(fb, 1e-6f));
    if (fa > 1.0f)
        fa = 1.0f;

    fa = std::ceil(fa * 255.0f) / 255.0f;
    fr /= fa;
    fg /= fa;
    fb /= fa;

    out_rgbm[0] = (uint8_t)_CLAMP(int(fr * 255), 0, 255);
    out_rgbm[1] = (uint8_t)_CLAMP(int(fg * 255), 0, 255);
    out_rgbm[2] = (uint8_t)_CLAMP(int(fb * 255), 0, 255);
    out_rgbm[3] = (uint8_t)_CLAMP(int(fa * 255), 0, 255);
}

std::unique_ptr<float[]> Ray::ConvertRGBE_to_RGB32F(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<float[]> fp_data(new float[w * h * 3]);

    for (int i = 0; i < w * h; i++) {
        const uint8_t r = image_data[4 * i + 0], g = image_data[4 * i + 1], b = image_data[4 * i + 2],
                      a = image_data[4 * i + 3];

        const float f = std::exp2(float(a) - 128.0f);
        const float k = 1.0f / 255;

        fp_data[3 * i + 0] = k * float(r) * f;
        fp_data[3 * i + 1] = k * float(g) * f;
        fp_data[3 * i + 2] = k * float(b) * f;
    }

    return fp_data;
}

std::unique_ptr<uint16_t[]> Ray::ConvertRGBE_to_RGB16F(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint16_t[]> fp16_data(new uint16_t[w * h * 3]);
    ConvertRGBE_to_RGB16F(image_data, w, h, fp16_data.get());
    return fp16_data;
}

void Ray::ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h, uint16_t *out_data) {
    for (int i = 0; i < w * h; i++) {
        const uint8_t r = image_data[4 * i + 0], g = image_data[4 * i + 1], b = image_data[4 * i + 2],
                      a = image_data[4 * i + 3];

        const float f = std::exp2(float(a) - 128.0f);
        const float k = 1.0f / 255;

        out_data[3 * i + 0] = f32_to_f16(k * float(r) * f);
        out_data[3 * i + 1] = f32_to_f16(k * float(g) * f);
        out_data[3 * i + 2] = f32_to_f16(k * float(b) * f);
    }
}

std::unique_ptr<uint8_t[]> Ray::ConvertRGB32F_to_RGBE(const float image_data[], const int w, const int h,
                                                      const int channels) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Ref::simd_fvec4 val;

            if (channels == 3) {
                val = Ref::simd_fvec4{image_data[3 * (y * w + x) + 0], image_data[3 * (y * w + x) + 1],
                                      image_data[3 * (y * w + x) + 2], 0.0f};
            } else if (channels == 4) {
                val = Ref::simd_fvec4{image_data[4 * (y * w + x) + 0], image_data[4 * (y * w + x) + 1],
                                      image_data[4 * (y * w + x) + 2], 0.0f};
            }

            auto exp = Ref::simd_fvec4{std::log2(val[0]), std::log2(val[1]), std::log2(val[2]), 0.0f};
            for (int i = 0; i < 3; i++) {
                exp.set(i, std::ceil(exp[i]));
                if (exp[i] < -128.0f) {
                    exp.set(i, -128.0f);
                } else if (exp[i] > 127.0f) {
                    exp.set(i, 127.0f);
                }
            }

            const float common_exp = std::max(exp[0], std::max(exp[1], exp[2]));
            const float range = std::exp2(common_exp);

            Ref::simd_fvec4 mantissa = val / range;
            for (int i = 0; i < 3; i++) {
                if (mantissa[i] < 0.0f) {
                    mantissa.set(i, 0.0f);
                } else if (mantissa[i] > 1.0f) {
                    mantissa.set(i, 1.0f);
                }
            }

            const auto res = Ref::simd_fvec4{mantissa[0], mantissa[1], mantissa[2], common_exp + 128.0f};

            u8_data[(y * w + x) * 4 + 0] = (uint8_t)_CLAMP(int(res[0] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 1] = (uint8_t)_CLAMP(int(res[1] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 2] = (uint8_t)_CLAMP(int(res[2] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 3] = (uint8_t)_CLAMP(int(res[3]), 0, 255);
        }
    }

    return u8_data;
}

std::unique_ptr<uint8_t[]> Ray::ConvertRGB32F_to_RGBM(const float image_data[], const int w, const int h,
                                                      const int channels) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            RGBMEncode(&image_data[channels * (y * w + x)], &u8_data[(y * w + x) * 4]);
        }
    }

    return u8_data;
}

void Ray::ConvertRGB_to_YCoCg_rev(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) {
    RGB_to_YCoCg_reversible(in_RGB, out_YCoCg);
}

void Ray::ConvertYCoCg_to_RGB_rev(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) {
    YCoCg_to_RGB_reversible(in_YCoCg, out_RGB);
}

std::unique_ptr<uint8_t[]> Ray::ConvertRGB_to_CoCgxY_rev(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t YCoCg[3];
            RGB_to_YCoCg_reversible(&image_data[(y * w + x) * 3], YCoCg);

            u8_data[(y * w + x) * 4 + 0] = YCoCg[1];
            u8_data[(y * w + x) * 4 + 1] = YCoCg[2];
            u8_data[(y * w + x) * 4 + 2] = 0;
            u8_data[(y * w + x) * 4 + 3] = YCoCg[0];
        }
    }

    return u8_data;
}

std::unique_ptr<uint8_t[]> Ray::ConvertCoCgxY_to_RGB_rev(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 3]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t YCoCg[] = {image_data[(y * w + x) * 4 + 3], image_data[(y * w + x) * 4 + 0],
                                     image_data[(y * w + x) * 4 + 1]};
            YCoCg_to_RGB_reversible(YCoCg, &u8_data[(y * w + x) * 3]);
        }
    }

    return u8_data;
}

void Ray::ConvertRGB_to_YCoCg(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) { RGB_to_YCoCg(in_RGB, out_YCoCg); }
void Ray::ConvertYCoCg_to_RGB(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) { YCoCg_to_RGB(in_YCoCg, out_RGB); }

std::unique_ptr<uint8_t[]> Ray::ConvertRGB_to_CoCgxY(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t YCoCg[3];
            RGB_to_YCoCg(&image_data[(y * w + x) * 3], YCoCg);

            u8_data[(y * w + x) * 4 + 0] = YCoCg[1];
            u8_data[(y * w + x) * 4 + 1] = YCoCg[2];
            u8_data[(y * w + x) * 4 + 2] = 0;
            u8_data[(y * w + x) * 4 + 3] = YCoCg[0];
        }
    }

    return u8_data;
}

std::unique_ptr<uint8_t[]> Ray::ConvertCoCgxY_to_RGB(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 3]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t YCoCg[] = {image_data[(y * w + x) * 4 + 3], image_data[(y * w + x) * 4 + 0],
                                     image_data[(y * w + x) * 4 + 1]};
            YCoCg_to_RGB(YCoCg, &u8_data[(y * w + x) * 3]);
        }
    }

    return u8_data;
}

int Ray::InitMipMaps(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16], const int channels,
                     const eMipOp op[4]) {
    int mip_count = 1;

    int _w = widths[0], _h = heights[0];
    while (_w > 1 || _h > 1) {
        int _prev_w = _w, _prev_h = _h;
        _w = std::max(_w / 2, 1);
        _h = std::max(_h / 2, 1);
        if (!mipmaps[mip_count]) {
            mipmaps[mip_count].reset(new uint8_t[_w * _h * channels]);
        }
        widths[mip_count] = _w;
        heights[mip_count] = _h;
        const uint8_t *tex = mipmaps[mip_count - 1].get();

        int count = 0;

        for (int j = 0; j < _prev_h; j += 2) {
            for (int i = 0; i < _prev_w; i += 2) {
                for (int k = 0; k < channels; k++) {
                    if (op[k] == eMipOp::Skip) {
                        continue;
                    } else if (op[k] == eMipOp::Zero) {
                        mipmaps[mip_count][count * channels + k] = 0;
                    }

                    // 4x4 pixel neighbourhood
                    int c[4][4];

                    // fetch inner quad
                    c[1][1] = tex[((j + 0) * _prev_w + i + 0) * channels + k];
                    c[1][2] = tex[((j + 0) * _prev_w + i + 1) * channels + k];
                    c[2][1] = tex[((j + 1) * _prev_w + i + 0) * channels + k];
                    c[2][2] = tex[((j + 1) * _prev_w + i + 1) * channels + k];

                    if (op[k] == eMipOp::Avg) {
                        mipmaps[mip_count][count * channels + k] = uint8_t((c[1][1] + c[1][2] + c[2][1] + c[2][2]) / 4);
                    } else if (op[k] == eMipOp::Min) {
                        mipmaps[mip_count][count * channels + k] = uint8_t(_MIN4(c[1][1], c[1][2], c[2][1], c[2][2]));
                    } else if (op[k] == eMipOp::Max) {
                        mipmaps[mip_count][count * channels + k] = uint8_t(_MAX4(c[1][1], c[1][2], c[2][1], c[2][2]));
                    } else if (op[k] == eMipOp::MinBilinear || op[k] == eMipOp::MaxBilinear) {

                        // fetch outer quad
                        for (int dy = -1; dy < 3; dy++) {
                            for (int dx = -1; dx < 3; dx++) {
                                if ((dx == 0 || dx == 1) && (dy == 0 || dy == 1)) {
                                    continue;
                                }

                                const int i0 = (i + dx + _prev_w) % _prev_w;
                                const int j0 = (j + dy + _prev_h) % _prev_h;

                                c[dy + 1][dx + 1] = tex[(j0 * _prev_w + i0) * channels + k];
                            }
                        }

                        static const int quadrants[2][2][2] = {{{-1, -1}, {+1, -1}}, {{-1, +1}, {+1, +1}}};

                        int test_val = c[1][1];

                        for (int dj = 1; dj < 3; dj++) {
                            for (int di = 1; di < 3; di++) {
                                const int i0 = di + quadrants[dj - 1][di - 1][0];
                                const int j0 = dj + quadrants[dj - 1][di - 1][1];

                                if (op[k] == eMipOp::MinBilinear) {
                                    test_val = _MIN(test_val, (c[dj][di] + c[dj][i0]) / 2);
                                    test_val = _MIN(test_val, (c[dj][di] + c[j0][di]) / 2);
                                } else if (op[k] == eMipOp::MaxBilinear) {
                                    test_val = _MAX(test_val, (c[dj][di] + c[dj][i0]) / 2);
                                    test_val = _MAX(test_val, (c[dj][di] + c[j0][di]) / 2);
                                }
                            }
                        }

                        for (int dj = 0; dj < 3; dj++) {
                            for (int di = 0; di < 3; di++) {
                                if (di == 1 && dj == 1) {
                                    continue;
                                }

                                if (op[k] == eMipOp::MinBilinear) {
                                    test_val = _MIN(test_val, (c[dj + 0][di + 0] + c[dj + 0][di + 1] +
                                                               c[dj + 1][di + 0] + c[dj + 1][di + 1]) /
                                                                  4);
                                } else if (op[k] == eMipOp::MaxBilinear) {
                                    test_val = _MAX(test_val, (c[dj + 0][di + 0] + c[dj + 0][di + 1] +
                                                               c[dj + 1][di + 0] + c[dj + 1][di + 1]) /
                                                                  4);
                                }
                            }
                        }

                        c[1][1] = test_val;

                        if (op[k] == eMipOp::MinBilinear) {
                            mipmaps[mip_count][count * channels + k] =
                                uint8_t(_MIN4(c[1][1], c[1][2], c[2][1], c[2][2]));
                        } else if (op[k] == eMipOp::MaxBilinear) {
                            mipmaps[mip_count][count * channels + k] =
                                uint8_t(_MAX4(c[1][1], c[1][2], c[2][1], c[2][2]));
                        }
                    }
                }

                count++;
            }
        }

        mip_count++;
    }

    return mip_count;
}

int Ray::InitMipMapsRGBM(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16]) {
    int mip_count = 1;

    int _w = widths[0], _h = heights[0];
    while (_w > 1 || _h > 1) {
        int _prev_w = _w, _prev_h = _h;
        _w = std::max(_w / 2, 1);
        _h = std::max(_h / 2, 1);
        mipmaps[mip_count].reset(new uint8_t[_w * _h * 4]);
        widths[mip_count] = _w;
        heights[mip_count] = _h;
        const uint8_t *tex = mipmaps[mip_count - 1].get();

        int count = 0;

        for (int j = 0; j < _prev_h; j += 2) {
            for (int i = 0; i < _prev_w; i += 2) {
                float rgb_sum[3];
                RGBMDecode(&tex[((j + 0) * _prev_w + i) * 4], rgb_sum);

                float temp[3];
                RGBMDecode(&tex[((j + 0) * _prev_w + i + 1) * 4], temp);
                rgb_sum[0] += temp[0];
                rgb_sum[1] += temp[1];
                rgb_sum[2] += temp[2];

                RGBMDecode(&tex[((j + 1) * _prev_w + i) * 4], temp);
                rgb_sum[0] += temp[0];
                rgb_sum[1] += temp[1];
                rgb_sum[2] += temp[2];

                RGBMDecode(&tex[((j + 1) * _prev_w + i + 1) * 4], temp);
                rgb_sum[0] += temp[0];
                rgb_sum[1] += temp[1];
                rgb_sum[2] += temp[2];

                rgb_sum[0] /= 4.0f;
                rgb_sum[1] /= 4.0f;
                rgb_sum[2] /= 4.0f;

                RGBMEncode(rgb_sum, &mipmaps[mip_count][count * 4]);
                count++;
            }
        }

        mip_count++;
    }

    return mip_count;
}

void Ray::ReorderTriangleIndices(const uint32_t *indices, const uint32_t indices_count, const uint32_t vtx_count,
                                 uint32_t *out_indices) {
    // From https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html

    uint32_t prim_count = indices_count / 3;

    struct vtx_data_t {
        int32_t cache_pos = -1;
        float score = 0.0f;
        uint32_t ref_count = 0;
        uint32_t active_tris_count = 0;
        std::unique_ptr<int32_t[]> tris;
    };

    static const int MaxSizeVertexCache = 32;

    auto get_vertex_score = [](int32_t cache_pos, uint32_t active_tris_count) -> float {
        const float CacheDecayPower = 1.5f;
        const float LastTriScore = 0.75f;
        const float ValenceBoostScale = 2.0f;
        const float ValenceBoostPower = 0.5f;

        if (active_tris_count == 0) {
            // No tri needs this vertex!
            return -1.0f;
        }

        float score = 0.0f;

        if (cache_pos < 0) {
            // Vertex is not in FIFO cache - no score.
        } else if (cache_pos < 3) {
            // This vertex was used in the last triangle,
            // so it has a fixed score, whichever of the three
            // it's in. Otherwise, you can get very different
            // answers depending on whether you add
            // the triangle 1,2,3 or 3,1,2 - which is silly.
            score = LastTriScore;
        } else {
            assert(cache_pos < MaxSizeVertexCache);
            // Points for being high in the cache.
            const float scaler = 1.0f / (MaxSizeVertexCache - 3);
            score = 1.0f - float(cache_pos - 3) * scaler;
            score = std::pow(score, CacheDecayPower);
        }

        // Bonus points for having a low number of tris still to
        // use the vert, so we get rid of lone verts quickly.

        const float valence_boost = std::pow((float)active_tris_count, -ValenceBoostPower);
        score += ValenceBoostScale * valence_boost;
        return score;
    };

    struct tri_data_t {
        bool is_in_list = false;
        float score = 0.0f;
        uint32_t indices[3] = {};
    };

    std::unique_ptr<vtx_data_t[]> _vertices(new vtx_data_t[vtx_count]);
    std::unique_ptr<tri_data_t[]> _triangles(new tri_data_t[prim_count]);

    // avoid operator[] call overhead in debug
    vtx_data_t *vertices = _vertices.get();
    tri_data_t *triangles = _triangles.get();

    for (uint32_t i = 0; i < indices_count; i += 3) {
        tri_data_t &tri = triangles[i / 3];

        tri.indices[0] = indices[i + 0];
        tri.indices[1] = indices[i + 1];
        tri.indices[2] = indices[i + 2];

        vertices[indices[i + 0]].active_tris_count++;
        vertices[indices[i + 1]].active_tris_count++;
        vertices[indices[i + 2]].active_tris_count++;
    }

    for (uint32_t i = 0; i < vtx_count; i++) {
        vtx_data_t &v = vertices[i];
        v.tris.reset(new int32_t[v.active_tris_count]);
        v.score = get_vertex_score(v.cache_pos, v.active_tris_count);
    }

    int32_t next_best_index = -1, next_next_best_index = -1;
    float next_best_score = -1.0f, next_next_best_score = -1.0f;

    for (uint32_t i = 0; i < indices_count; i += 3) {
        tri_data_t &tri = triangles[i / 3];

        vtx_data_t &v0 = vertices[indices[i + 0]];
        vtx_data_t &v1 = vertices[indices[i + 1]];
        vtx_data_t &v2 = vertices[indices[i + 2]];

        v0.tris[v0.ref_count++] = i / 3;
        v1.tris[v1.ref_count++] = i / 3;
        v2.tris[v2.ref_count++] = i / 3;

        tri.score = v0.score + v1.score + v2.score;

        if (tri.score > next_best_score) {
            if (next_best_score > next_next_best_score) {
                next_next_best_index = next_best_index;
                next_next_best_score = next_best_score;
            }
            next_best_index = i / 3;
            next_best_score = tri.score;
        }

        if (tri.score > next_next_best_score) {
            next_next_best_index = i / 3;
            next_next_best_score = tri.score;
        }
    }

    std::deque<uint32_t> lru_cache;

    auto use_vertex = [](std::deque<uint32_t> &lru_cache, uint32_t vtx_index) {
        auto it = find(begin(lru_cache), end(lru_cache), vtx_index);

        if (it == end(lru_cache)) {
            lru_cache.push_back(vtx_index);
            it = begin(lru_cache);
        }

        if (it != begin(lru_cache)) {
            lru_cache.erase(it);
            lru_cache.push_front(vtx_index);
        }
    };

    auto enforce_size = [&get_vertex_score](std::deque<uint32_t> &lru_cache, vtx_data_t *vertices, uint32_t max_size,
                                            std::vector<uint32_t> &out_tris_to_update) {
        out_tris_to_update.clear();

        if (lru_cache.size() > max_size) {
            lru_cache.resize(max_size);
        }

        for (size_t i = 0; i < lru_cache.size(); i++) {
            vtx_data_t &v = vertices[lru_cache[i]];

            v.cache_pos = (int32_t)i;
            v.score = get_vertex_score(v.cache_pos, v.active_tris_count);

            for (uint32_t j = 0; j < v.ref_count; j++) {
                int tri_index = v.tris[j];
                if (tri_index != -1) {
                    auto it = find(begin(out_tris_to_update), end(out_tris_to_update), tri_index);
                    if (it == end(out_tris_to_update)) {
                        out_tris_to_update.push_back(tri_index);
                    }
                }
            }
        }
    };

    for (int32_t out_index = 0; out_index < (int32_t)indices_count;) {
        if (next_best_index < 0) {
            next_best_score = next_next_best_score = -1.0f;
            next_best_index = next_next_best_index = -1;

            for (int32_t i = 0; i < (int32_t)prim_count; i++) {
                const tri_data_t &tri = triangles[i];
                if (!tri.is_in_list) {
                    if (tri.score > next_best_score) {
                        if (next_best_score > next_next_best_score) {
                            next_next_best_index = next_best_index;
                            next_next_best_score = next_best_score;
                        }
                        next_best_index = i;
                        next_best_score = tri.score;
                    }

                    if (tri.score > next_next_best_score) {
                        next_next_best_index = i;
                        next_next_best_score = tri.score;
                    }
                }
            }
        }

        tri_data_t &next_best_tri = triangles[next_best_index];

        for (unsigned int indice : next_best_tri.indices) {
            out_indices[out_index++] = indice;

            vtx_data_t &v = vertices[indice];
            v.active_tris_count--;
            for (uint32_t k = 0; k < v.ref_count; k++) {
                if (v.tris[k] == next_best_index) {
                    v.tris[k] = -1;
                    break;
                }
            }

            use_vertex(lru_cache, indice);
        }

        next_best_tri.is_in_list = true;

        std::vector<uint32_t> tris_to_update;
        enforce_size(lru_cache, &vertices[0], MaxSizeVertexCache, tris_to_update);

        next_best_score = -1.0f;
        next_best_index = -1;

        for (const uint32_t ti : tris_to_update) {
            tri_data_t &tri = triangles[ti];

            if (!tri.is_in_list) {
                tri.score =
                    vertices[tri.indices[0]].score + vertices[tri.indices[1]].score + vertices[tri.indices[2]].score;

                if (tri.score > next_best_score) {
                    if (next_best_score > next_next_best_score) {
                        next_next_best_index = next_best_index;
                        next_next_best_score = next_best_score;
                    }
                    next_best_index = ti;
                    next_best_score = tri.score;
                }

                if (tri.score > next_next_best_score) {
                    next_next_best_index = ti;
                    next_next_best_score = tri.score;
                }
            }
        }

        if (next_best_index == -1 && next_next_best_index != -1) {
            if (!triangles[next_next_best_index].is_in_list) {
                next_best_index = next_next_best_index;
                next_best_score = next_next_best_score;
            }

            next_next_best_index = -1;
            next_next_best_score = -1.0f;
        }
    }
}

namespace Ray {
//
// https://software.intel.com/sites/default/files/23/1d/324337_324337.pdf
//

template <int SrcChannels, int DstChannels = 4>
void Extract4x4Block_Ref(const uint8_t src[], const int stride, uint8_t dst[16 * DstChannels]) {
    if (SrcChannels == 4 && DstChannels == 4) {
        for (int j = 0; j < 4; j++) {
            memcpy(&dst[j * 4 * 4], src, 4 * 4);
            src += stride;
        }
    } else {
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                memcpy(&dst[i * DstChannels], &src[i * SrcChannels],
                       SrcChannels < DstChannels ? SrcChannels : DstChannels);
            }
            dst += 4 * DstChannels;
            src += stride;
        }
    }
}

template <int SrcChannels, int DstChannels = 4>
void ExtractIncomplete4x4Block_Ref(const uint8_t src[], const int stride, const int blck_w, const int blck_h,
                                   uint8_t dst[16 * DstChannels]) {
    if (SrcChannels == 4 && DstChannels == 4) {
        for (int j = 0; j < blck_h; j++) {
            assert(blck_w <= 4);
            memcpy(&dst[0], src, 4 * blck_w);
            for (int i = blck_w; i < 4; i++) {
                memcpy(&dst[i * 4], &dst[(blck_w - 1) * 4], 4);
            }
            dst += 4 * 4;
            src += stride;
        }
    } else {
        for (int j = 0; j < blck_h; j++) {
            for (int i = 0; i < blck_w; i++) {
                memcpy(&dst[i * DstChannels], &src[i * SrcChannels],
                       SrcChannels < DstChannels ? SrcChannels : DstChannels);
            }
            for (int i = blck_w; i < 4; i++) {
                memcpy(&dst[i * DstChannels], &dst[(blck_w - 1) * DstChannels], DstChannels);
            }
            dst += 4 * DstChannels;
            src += stride;
        }
    }
    uint8_t *dst2 = dst - 4 * DstChannels;
    for (int j = blck_h; j < 4; j++) {
        memcpy(dst, dst2, 4 * DstChannels);
        dst += 4 * DstChannels;
    }
}

// WARNING: Reads 4 bytes outside of block!
template <int Channels> void Extract4x4Block_SSSE3(const uint8_t src[], int stride, uint8_t dst[64]);

force_inline int ColorDistance(const uint8_t c1[3], const uint8_t c2[3]) {
    // euclidean distance
    return (c1[0] - c2[0]) * (c1[0] - c2[0]) + (c1[1] - c2[1]) * (c1[1] - c2[1]) + (c1[2] - c2[2]) * (c1[2] - c2[2]);
}

force_inline int ColorLumaApprox(const uint8_t color[3]) { return int(color[0] + color[1] * 2 + color[2]); }

force_inline uint16_t rgb888_to_rgb565(const uint8_t color[3]) {
    return ((color[0] >> 3) << 11) | ((color[1] >> 2) << 5) | (color[2] >> 3);
}

force_inline void swap_rgb(uint8_t c1[3], uint8_t c2[3]) {
    uint8_t tm[3];
    memcpy(tm, c1, 3);
    memcpy(c1, c2, 3);
    memcpy(c2, tm, 3);
}

void GetMinMaxColorByDistance(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]) {
    int max_dist = -1;

    for (int i = 0; i < 64 - 4; i += 4) {
        for (int j = i + 4; j < 64; j += 4) {
            const int dist = ColorDistance(&block[i], &block[j]);
            if (dist > max_dist) {
                max_dist = dist;
                memcpy(min_color, &block[i], 3);
                memcpy(max_color, &block[j], 3);
            }
        }
    }

    if (rgb888_to_rgb565(max_color) < rgb888_to_rgb565(min_color)) {
        swap_rgb(min_color, max_color);
    }
}

void GetMinMaxColorByLuma(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]) {
    int max_luma = -1, min_luma = std::numeric_limits<int>::max();

    for (int i = 0; i < 16; i++) {
        const int luma = ColorLumaApprox(&block[i * 4]);
        if (luma > max_luma) {
            memcpy(max_color, &block[i * 4], 3);
            max_luma = luma;
        }
        if (luma < min_luma) {
            memcpy(min_color, &block[i * 4], 3);
            min_luma = luma;
        }
    }

    if (rgb888_to_rgb565(max_color) < rgb888_to_rgb565(min_color)) {
        swap_rgb(min_color, max_color);
    }
}

template <bool UseAlpha = false, bool Is_YCoCg = false>
void GetMinMaxColorByBBox_Ref(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]) {
    min_color[0] = min_color[1] = min_color[2] = min_color[3] = 255;
    max_color[0] = max_color[1] = max_color[2] = max_color[3] = 0;

    // clang-format off
    for (int i = 0; i < 16; i++) {
        if (block[i * 4 + 0] < min_color[0]) min_color[0] = block[i * 4 + 0];
        if (block[i * 4 + 1] < min_color[1]) min_color[1] = block[i * 4 + 1];
        if (block[i * 4 + 2] < min_color[2]) min_color[2] = block[i * 4 + 2];
        if (UseAlpha && block[i * 4 + 3] < min_color[3]) min_color[3] = block[i * 4 + 3];
        if (block[i * 4 + 0] > max_color[0]) max_color[0] = block[i * 4 + 0];
        if (block[i * 4 + 1] > max_color[1]) max_color[1] = block[i * 4 + 1];
        if (block[i * 4 + 2] > max_color[2]) max_color[2] = block[i * 4 + 2];
        if (UseAlpha && block[i * 4 + 3] > max_color[3]) max_color[3] = block[i * 4 + 3];
    }
    // clang-format on

    if (!Is_YCoCg) {
        // offset bbox inside by 1/16 of it's dimentions, this improves MSR (???)
        const uint8_t inset[] = {
            uint8_t((max_color[0] - min_color[0]) / 16), uint8_t((max_color[1] - min_color[1]) / 16),
            uint8_t((max_color[2] - min_color[2]) / 16), uint8_t((max_color[3] - min_color[3]) / 32)};

        min_color[0] = (min_color[0] + inset[0] <= 255) ? min_color[0] + inset[0] : 255;
        min_color[1] = (min_color[1] + inset[1] <= 255) ? min_color[1] + inset[1] : 255;
        min_color[2] = (min_color[2] + inset[2] <= 255) ? min_color[2] + inset[2] : 255;
        if (UseAlpha) {
            min_color[3] = (min_color[3] + inset[3] <= 255) ? min_color[3] + inset[3] : 255;
        }

        max_color[0] = (max_color[0] >= inset[0]) ? max_color[0] - inset[0] : 0;
        max_color[1] = (max_color[1] >= inset[1]) ? max_color[1] - inset[1] : 0;
        max_color[2] = (max_color[2] >= inset[2]) ? max_color[2] - inset[2] : 0;
        if (UseAlpha) {
            max_color[3] = (max_color[3] >= inset[3]) ? max_color[3] - inset[3] : 0;
        }
    }
}

void GetMinMaxAlphaByBBox_Ref(const uint8_t block[64], uint8_t &min_alpha, uint8_t &max_alpha) {
    min_alpha = 255;
    max_alpha = 0;

    // clang-format off
    for (int i = 0; i < 16; i++) {
        if (block[i] < min_alpha) min_alpha = block[i];
        if (block[i] > max_alpha) max_alpha = block[i];
    }
    // clang-format on
}

template <bool UseAlpha = false, bool Is_YCoCg = false>
void GetMinMaxColorByBBox_SSE2(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]);

void InsetYCoCgBBox_Ref(uint8_t min_color[4], uint8_t max_color[4]) {
    const int inset[] = {(max_color[0] - min_color[0]) - ((1 << (4 - 1)) - 1),
                         (max_color[1] - min_color[1]) - ((1 << (4 - 1)) - 1), 0,
                         (max_color[3] - min_color[3]) - ((1 << (5 - 1)) - 1)};

    int mini[4], maxi[4];

    mini[0] = ((min_color[0] * 16) + inset[0]) / 16;
    mini[1] = ((min_color[1] * 16) + inset[1]) / 16;
    mini[3] = ((min_color[3] * 32) + inset[3]) / 32;

    maxi[0] = ((max_color[0] * 16) - inset[0]) / 16;
    maxi[1] = ((max_color[1] * 16) - inset[1]) / 16;
    maxi[3] = ((max_color[3] * 32) - inset[3]) / 32;

    mini[0] = (mini[0] >= 0) ? mini[0] : 0;
    mini[1] = (mini[1] >= 0) ? mini[1] : 0;
    mini[3] = (mini[3] >= 0) ? mini[3] : 0;

    maxi[0] = (maxi[0] <= 255) ? maxi[0] : 255;
    maxi[1] = (maxi[1] <= 255) ? maxi[1] : 255;
    maxi[3] = (maxi[3] <= 255) ? maxi[3] : 255;

    min_color[0] = (mini[0] & 0b11111000) | (mini[0] >> 5u);
    min_color[1] = (mini[1] & 0b11111100) | (mini[1] >> 6u);
    min_color[3] = mini[3];

    max_color[0] = (maxi[0] & 0b11111000) | (maxi[0] >> 5u);
    max_color[1] = (maxi[1] & 0b11111100) | (maxi[1] >> 6u);
    max_color[3] = maxi[3];
}

void InsetYCoCgBBox_SSE2(uint8_t min_color[4], uint8_t max_color[4]);

void SelectYCoCgDiagonal_Ref(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
    const uint8_t mid0 = (int(min_color[0]) + max_color[0] + 1) / 2;
    const uint8_t mid1 = (int(min_color[1]) + max_color[1] + 1) / 2;

#if 0 // use covariance
    int covariance = 0;
    for (int i = 0; i < 16; i++) {
        const int b0 = block[i * 4 + 0] - mid0;
        const int b1 = block[i * 4 + 1] - mid1;
        covariance += (b0 * b1);
    }

    // flip diagonal
    if (covariance) {
        const uint8_t t = min_color[1];
        min_color[1] = max_color[1];
        max_color[1] = t;
    }
#else // use sign only
    uint8_t side = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t b0 = block[i * 4 + 0] >= mid0;
        const uint8_t b1 = block[i * 4 + 1] >= mid1;
        side += (b0 ^ b1);
    }

    uint8_t mask = -(side > 8);

    uint8_t c0 = min_color[1];
    uint8_t c1 = max_color[1];

    //c0 ^= c1 ^= mask &= c0 ^= c1;
    c0 ^= c1;
    mask &= c0;
    c1 ^= mask;
    c0 ^= c1;

    min_color[1] = c0;
    max_color[1] = c1;
#endif
}

void SelectYCoCgDiagonal_SSE2(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]);

void ScaleYCoCg_Ref(uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]) {
    int m0 = _ABS(min_color[0] - 128);
    int m1 = _ABS(min_color[1] - 128);
    int m2 = _ABS(max_color[0] - 128);
    int m3 = _ABS(max_color[1] - 128);

    // clang-format off
    if (m1 > m0) m0 = m1;
    if (m3 > m2) m2 = m3;
    if (m2 > m0) m0 = m2;
    // clang-format on

    const int s0 = 128 / 2 - 1;
    const int s1 = 128 / 4 - 1;

    const int mask0 = -(m0 <= s0);
    const int mask1 = -(m0 <= s1);
    const int scale = 1 + (1 & mask0) + (2 & mask1);

    min_color[0] = (min_color[0] - 128) * scale + 128;
    min_color[1] = (min_color[1] - 128) * scale + 128;
    min_color[2] = (scale - 1) * 8;

    max_color[0] = (max_color[0] - 128) * scale + 128;
    max_color[1] = (max_color[1] - 128) * scale + 128;
    max_color[2] = (scale - 1) * 8;

    for (int i = 0; i < 16; i++) {
        block[i * 4 + 0] = (block[i * 4 + 0] - 128) * scale + 128;
        block[i * 4 + 1] = (block[i * 4 + 1] - 128) * scale + 128;
    }
}

void ScaleYCoCg_SSE2(uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]);

force_inline void push_u8(const uint8_t v, uint8_t *&out_data) { (*out_data++) = v; }

force_inline void push_u16(const uint16_t v, uint8_t *&out_data) {
    (*out_data++) = (v >> 0) & 0xFF;
    (*out_data++) = (v >> 8) & 0xFF;
}

force_inline void push_u32(const uint32_t v, uint8_t *&out_data) {
    (*out_data++) = (v >> 0) & 0xFF;
    (*out_data++) = (v >> 8) & 0xFF;
    (*out_data++) = (v >> 16) & 0xFF;
    (*out_data++) = (v >> 24) & 0xFF;
}

void EmitColorIndices_Ref(const uint8_t block[64], const uint8_t min_color[3], const uint8_t max_color[3],
                          uint8_t *&out_data) {
    uint8_t colors[4][4];

    // get two initial colors (as if they were converted to rgb565 and back
    // note: the last 3 bits are replicated from the first 3 bits (???)
    colors[0][0] = (max_color[0] & 0b11111000) | (max_color[0] >> 5u);
    colors[0][1] = (max_color[1] & 0b11111100) | (max_color[1] >> 6u);
    colors[0][2] = (max_color[2] & 0b11111000) | (max_color[2] >> 5u);
    colors[1][0] = (min_color[0] & 0b11111000) | (min_color[0] >> 5u);
    colors[1][1] = (min_color[1] & 0b11111100) | (min_color[1] >> 6u);
    colors[1][2] = (min_color[2] & 0b11111000) | (min_color[2] >> 5u);
    // get two interpolated colors
    colors[2][0] = (2 * colors[0][0] + 1 * colors[1][0]) / 3;
    colors[2][1] = (2 * colors[0][1] + 1 * colors[1][1]) / 3;
    colors[2][2] = (2 * colors[0][2] + 1 * colors[1][2]) / 3;
    colors[3][0] = (1 * colors[0][0] + 2 * colors[1][0]) / 3;
    colors[3][1] = (1 * colors[0][1] + 2 * colors[1][1]) / 3;
    colors[3][2] = (1 * colors[0][2] + 2 * colors[1][2]) / 3;

    // division by 3 can be 'emulated' with:
    // y = (1 << 16) / 3 + 1
    // x = (x * y) >> 16          -->      pmulhw x, y

    // find best ind for each pixel in a block
    uint32_t result_indices = 0;

#if 0   // use euclidian distance (slower)
        uint32_t palette_indices[16];
        for (int i = 0; i < 16; i++) {
            uint32_t min_dist = std::numeric_limits<uint32_t>::max();
            for (int j = 0; j < 4; j++) {
                const uint32_t dist = ColorDistance(&block[i * 4], &colors[j][0]);
                if (dist < min_dist) {
                    palette_indices[i] = j;
                    min_dist = dist;
                }
            }
        }

        // pack ind in 2 bits each
        for (int i = 0; i < 16; i++) {
            result_indices |= (palette_indices[i] << uint32_t(i * 2));
        }
#elif 1 // use absolute differences (faster)
    for (int i = 15; i >= 0; i--) {
        const int c0 = block[i * 4 + 0];
        const int c1 = block[i * 4 + 1];
        const int c2 = block[i * 4 + 2];

        const int d0 = _ABS(colors[0][0] - c0) + _ABS(colors[0][1] - c1) + _ABS(colors[0][2] - c2);
        const int d1 = _ABS(colors[1][0] - c0) + _ABS(colors[1][1] - c1) + _ABS(colors[1][2] - c2);
        const int d2 = _ABS(colors[2][0] - c0) + _ABS(colors[2][1] - c1) + _ABS(colors[2][2] - c2);
        const int d3 = _ABS(colors[3][0] - c0) + _ABS(colors[3][1] - c1) + _ABS(colors[3][2] - c2);

        const int b0 = d0 > d3;
        const int b1 = d1 > d2;
        const int b2 = d0 > d2;
        const int b3 = d1 > d3;
        const int b4 = d2 > d3;

        const int x0 = b1 & b2;
        const int x1 = b0 & b3;
        const int x2 = b0 & b4;

        result_indices |= (x2 | ((x0 | x1) << 1)) << (i * 2);
    }
#endif

    push_u32(result_indices, out_data);
}

void EmitColorIndices_SSE2(const uint8_t block[64], const uint8_t min_color[4], const uint8_t max_color[4],
                           uint8_t *&out_data);

void EmitAlphaIndices_Ref(const uint8_t block[64], const uint8_t min_alpha, const uint8_t max_alpha,
                          uint8_t *&out_data) {
    uint8_t ind[16];

#if 0 // simple version
    const uint8_t alphas[8] = {max_alpha,
                               min_alpha,
                               uint8_t((6 * max_alpha + 1 * min_alpha) / 7),
                               uint8_t((5 * max_alpha + 2 * min_alpha) / 7),
                               uint8_t((4 * max_alpha + 3 * min_alpha) / 7),
                               uint8_t((3 * max_alpha + 4 * min_alpha) / 7),
                               uint8_t((2 * max_alpha + 5 * min_alpha) / 7),
                               uint8_t((1 * max_alpha + 6 * min_alpha) / 7)};
    for (int i = 0; i < 16; i++) {
        int min_dist = std::numeric_limits<int>::max();
        const uint8_t a = block[i * 4 + 3];
        for (int j = 0; j < 8; j++) {
            const int dist = _ABS(a - alphas[j]);
            if (dist < min_dist) {
                ind[i] = j;
                min_dist = dist;
            }
        }
    }
#else // parallel-friendly version
    const uint8_t half_step = (max_alpha - min_alpha) / (2 * 7);

    // division by 14 and 7 can be 'emulated' with:
    // y = (1 << 16) / 14 + 1
    // x = (x * y) >> 16          -->      pmulhw x, y

    const uint8_t ab1 = min_alpha + half_step;
    const uint8_t ab2 = (6 * max_alpha + 1 * min_alpha) / 7 + half_step;
    const uint8_t ab3 = (5 * max_alpha + 2 * min_alpha) / 7 + half_step;
    const uint8_t ab4 = (4 * max_alpha + 3 * min_alpha) / 7 + half_step;
    const uint8_t ab5 = (3 * max_alpha + 4 * min_alpha) / 7 + half_step;
    const uint8_t ab6 = (2 * max_alpha + 5 * min_alpha) / 7 + half_step;
    const uint8_t ab7 = (1 * max_alpha + 6 * min_alpha) / 7 + half_step;

    for (int i = 0; i < 16; i++) {
        const uint8_t a = block[i * 4 + 3];

        const int b1 = (a <= ab1);
        const int b2 = (a <= ab2);
        const int b3 = (a <= ab3);
        const int b4 = (a <= ab4);
        const int b5 = (a <= ab5);
        const int b6 = (a <= ab6);
        const int b7 = (a <= ab7);

        // x <= y can be emulated with min(x, y) == x

        const int ndx = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + 1) & 0b00000111;
        ind[i] = ndx ^ (2 > ndx);
    }
#endif

    // Write indices 3 bit each (48 = 4x8 in total)
    // [ 2][ 2][ 1][ 1][ 1][ 0][ 0][ 0]
    push_u8((ind[0] >> 0) | (ind[1] << 3) | (ind[2] << 6), out_data);
    // [ 5][ 4][ 4][ 4][ 3][ 3][ 3][ 2]
    push_u8((ind[2] >> 2) | (ind[3] << 1) | (ind[4] << 4) | (ind[5] << 7), out_data);
    // [ 7][ 7][ 7][ 6][ 6][ 6][ 5][ 5]
    push_u8((ind[5] >> 1) | (ind[6] << 2) | (ind[7] << 5), out_data);
    // [10][10][ 9][ 9][ 9][ 8][ 8][ 8]
    push_u8((ind[8] >> 0) | (ind[9] << 3) | (ind[10] << 6), out_data);
    // [13][12][12][12][11][11][11][10]
    push_u8((ind[10] >> 2) | (ind[11] << 1) | (ind[12] << 4) | (ind[13] << 7), out_data);
    // [15][15][15][14][14][14][13][13]
    push_u8((ind[13] >> 1) | (ind[14] << 2) | (ind[15] << 5), out_data);
}

void EmitAlphaOnlyIndices_Ref(const uint8_t block[16], const uint8_t min_alpha, const uint8_t max_alpha,
                              uint8_t *&out_data) {
    uint8_t ind[16];

    const uint8_t half_step = (max_alpha - min_alpha) / (2 * 7);

    // division by 14 and 7 can be 'emulated' with:
    // y = (1 << 16) / 14 + 1
    // x = (x * y) >> 16          -->      pmulhw x, y

    const uint8_t ab1 = min_alpha + half_step;
    const uint8_t ab2 = (6 * max_alpha + 1 * min_alpha) / 7 + half_step;
    const uint8_t ab3 = (5 * max_alpha + 2 * min_alpha) / 7 + half_step;
    const uint8_t ab4 = (4 * max_alpha + 3 * min_alpha) / 7 + half_step;
    const uint8_t ab5 = (3 * max_alpha + 4 * min_alpha) / 7 + half_step;
    const uint8_t ab6 = (2 * max_alpha + 5 * min_alpha) / 7 + half_step;
    const uint8_t ab7 = (1 * max_alpha + 6 * min_alpha) / 7 + half_step;

    for (int i = 0; i < 16; i++) {
        const uint8_t a = block[i];

        const int b1 = (a <= ab1);
        const int b2 = (a <= ab2);
        const int b3 = (a <= ab3);
        const int b4 = (a <= ab4);
        const int b5 = (a <= ab5);
        const int b6 = (a <= ab6);
        const int b7 = (a <= ab7);

        // x <= y can be emulated with min(x, y) == x

        const int ndx = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + 1) & 0b00000111;
        ind[i] = ndx ^ (2 > ndx);
    }

    // Write indices 3 bit each (48 = 4x8 in total)
    // [ 2][ 2][ 1][ 1][ 1][ 0][ 0][ 0]
    push_u8((ind[0] >> 0) | (ind[1] << 3) | (ind[2] << 6), out_data);
    // [ 5][ 4][ 4][ 4][ 3][ 3][ 3][ 2]
    push_u8((ind[2] >> 2) | (ind[3] << 1) | (ind[4] << 4) | (ind[5] << 7), out_data);
    // [ 7][ 7][ 7][ 6][ 6][ 6][ 5][ 5]
    push_u8((ind[5] >> 1) | (ind[6] << 2) | (ind[7] << 5), out_data);
    // [10][10][ 9][ 9][ 9][ 8][ 8][ 8]
    push_u8((ind[8] >> 0) | (ind[9] << 3) | (ind[10] << 6), out_data);
    // [13][12][12][12][11][11][11][10]
    push_u8((ind[10] >> 2) | (ind[11] << 1) | (ind[12] << 4) | (ind[13] << 7), out_data);
    // [15][15][15][14][14][14][13][13]
    push_u8((ind[13] >> 1) | (ind[14] << 2) | (ind[15] << 5), out_data);
}

void EmitAlphaIndices_SSE2(const uint8_t block[64], uint8_t min_alpha, uint8_t max_alpha, uint8_t *&out_data);

void Emit_BC1_Block_Ref(const uint8_t block[64], uint8_t *&out_data) {
    uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_Ref(block, min_color, max_color);

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_Ref(block, min_color, max_color, out_data);
}

template <bool Is_YCoCg> void Emit_BC3_Block_Ref(uint8_t block[64], uint8_t *&out_data) {
    uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_Ref<true /* UseAlpha */, Is_YCoCg>(block, min_color, max_color);
    if (Is_YCoCg) {
        ScaleYCoCg_Ref(block, min_color, max_color);
        InsetYCoCgBBox_Ref(min_color, max_color);
        SelectYCoCgDiagonal_Ref(block, min_color, max_color);
    }

    //
    // Write alpha block
    //

    push_u8(max_color[3], out_data);
    push_u8(min_color[3], out_data);

    EmitAlphaIndices_Ref(block, min_color[3], max_color[3], out_data);

    //
    // Write color block
    //

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_Ref(block, min_color, max_color, out_data);
}

void Emit_BC4_Block_Ref(uint8_t block[16], uint8_t *&out_data) {
    uint8_t min_alpha, max_alpha;
    GetMinMaxAlphaByBBox_Ref(block, min_alpha, max_alpha);

    //
    // Write alpha block
    //

    push_u8(max_alpha, out_data);
    push_u8(min_alpha, out_data);

    EmitAlphaOnlyIndices_Ref(block, min_alpha, max_alpha, out_data);
}

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
void Emit_BC1_Block_SSE2(const uint8_t block[64], uint8_t *&out_data) {
    alignas(16) uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_SSE2(block, min_color, max_color);

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_SSE2(block, min_color, max_color, out_data);
}

template <bool Is_YCoCg> void Emit_BC3_Block_SSE2(uint8_t block[64], uint8_t *&out_data) {
    alignas(16) uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_SSE2<true /* UseAlpha */, Is_YCoCg>(block, min_color, max_color);
    if (Is_YCoCg) {
        ScaleYCoCg_SSE2(block, min_color, max_color);
        InsetYCoCgBBox_SSE2(min_color, max_color);
        SelectYCoCgDiagonal_SSE2(block, min_color, max_color);
    }

    //
    // Write alpha block
    //

    push_u8(max_color[3], out_data);
    push_u8(min_color[3], out_data);

    EmitAlphaIndices_SSE2(block, min_color[3], max_color[3], out_data);

    //
    // Write color block
    //

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_SSE2(block, min_color, max_color, out_data);
}
#endif

// clang-format off

const int BlockSize_BC1 = 2 * sizeof(uint16_t) + sizeof(uint32_t);
//                        \_ low/high colors_/   \_ 16 x 2-bit _/
const int BlockSize_BC4 = 2 * sizeof(uint8_t) + 6 * sizeof(uint8_t);
//                        \_ low/high alpha_/     \_ 16 x 3-bit _/
const int BlockSize_BC3 = BlockSize_BC1 + BlockSize_BC4;
const int BlockSize_BC5 = BlockSize_BC4 + BlockSize_BC4;

// clang-format on

} // namespace Ray

int Ray::GetRequiredMemory_BC1(const int w, const int h) { return BlockSize_BC1 * ((w + 3) / 4) * ((h + 3) / 4); }
int Ray::GetRequiredMemory_BC3(const int w, const int h) { return BlockSize_BC3 * ((w + 3) / 4) * ((h + 3) / 4); }
int Ray::GetRequiredMemory_BC4(const int w, const int h) { return BlockSize_BC4 * ((w + 3) / 4) * ((h + 3) / 4); }
int Ray::GetRequiredMemory_BC5(const int w, const int h) { return BlockSize_BC5 * ((w + 3) / 4) * ((h + 3) / 4); }

template <int SrcChannels>
void Ray::CompressImage_BC1(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[]) {
    alignas(16) uint8_t block[64] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
    const CpuFeatures cpu = GetCpuFeatures();
    if (cpu.sse2_supported && cpu.ssse3_supported) {
        for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
            const int w_limited =
                (SrcChannels == 3 && j == h_aligned - 4 && h_aligned == h) ? w_aligned - 4 : w_aligned;
            for (int i = 0; i < w_limited; i += 4) {
                Extract4x4Block_SSSE3<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
                Emit_BC1_Block_SSE2(block, p_out);
            }
            if (w_limited != w_aligned && w_aligned >= 4) {
                // process last block (avoid reading 4 bytes outside of range)
                Extract4x4Block_Ref<SrcChannels>(&img_src[(w_aligned - 4) * SrcChannels], w * SrcChannels, block);
                Emit_BC1_Block_SSE2(block, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4,
                                                           block);
                Emit_BC1_Block_SSE2(block, p_out);
            }
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                       h % 4, block);
            Emit_BC1_Block_SSE2(block, p_out);
        }
    } else
#endif
    {
        for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
                Emit_BC1_Block_Ref(block, p_out);
            }
            // process last column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4,
                                                           block);
                Emit_BC1_Block_Ref(block, p_out);
            }
        }
        // process last row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                       h % 4, block);
            Emit_BC1_Block_Ref(block, p_out);
        }
    }
}

template void Ray::CompressImage_BC1<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC1<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);

template <bool Is_YCoCg>
void Ray::CompressImage_BC3(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[]) {
    alignas(16) uint8_t block[64] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
    const CpuFeatures cpu = GetCpuFeatures();
    if (cpu.sse2_supported && cpu.ssse3_supported) {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * 4) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_SSSE3<4 /* SrcChannels */>(&img_src[i * 4], w * 4, block);
                Emit_BC3_Block_SSE2<Is_YCoCg>(block, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<4 /* SrcChannels */>(&img_src[w_aligned * 4], w * 4, w % 4, 4, block);
                Emit_BC3_Block_SSE2<Is_YCoCg>(block, p_out);
            }
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<4 /* SrcChannels */>(&img_src[i * 4], w * 4, _MIN(4, w - i), h % 4, block);
            Emit_BC3_Block_SSE2<Is_YCoCg>(block, p_out);
        }
    } else
#endif
    {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * 4) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<4>(&img_src[i * 4], w * 4, block);
                Emit_BC3_Block_Ref<Is_YCoCg>(block, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<4>(&img_src[w_aligned * 4], w * 4, w % 4, 4, block);
                Emit_BC3_Block_Ref<Is_YCoCg>(block, p_out);
            }
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<4>(&img_src[i * 4], w * 4, _MIN(4, w - i), h % 4, block);
            Emit_BC3_Block_Ref<Is_YCoCg>(block, p_out);
        }
    }
}

template void Ray::CompressImage_BC3<false /* Is_YCoCg */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC3<true /* Is_YCoCg */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);

template <int SrcChannels> void Ray::CompressImage_BC4(const uint8_t img_src[], int w, int h, uint8_t img_dst[]) {
    alignas(16) uint8_t block[16] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

#if 0
    // TODO: SIMD implementation
#endif
    {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, block);
                Emit_BC4_Block_Ref(block, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4,
                                                              4, block);
                Emit_BC4_Block_Ref(block, p_out);
            }
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                          h % 4, block);
            Emit_BC4_Block_Ref(block, p_out);
        }
    }
}

template void Ray::CompressImage_BC4<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC4<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC4<2 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC4<1 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);

template <int SrcChannels>
void Ray::CompressImage_BC5(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[]) {
    alignas(16) uint8_t block1[16] = {}, block2[16] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

#if 0
    // TODO: SIMD implementation
#endif
    {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels, block1);
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels, block2);
                Emit_BC4_Block_Ref(block1, p_out);
                Emit_BC4_Block_Ref(block2, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 0], w * SrcChannels,
                                                              w % 4, 4, block1);
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 1], w * SrcChannels,
                                                              w % 4, 4, block2);
                Emit_BC4_Block_Ref(block1, p_out);
                Emit_BC4_Block_Ref(block2, p_out);
            }
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels,
                                                          _MIN(4, w - i), h % 4, block1);
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels,
                                                          _MIN(4, w - i), h % 4, block2);
            Emit_BC4_Block_Ref(block1, p_out);
            Emit_BC4_Block_Ref(block2, p_out);
        }
    }
}

template void Ray::CompressImage_BC5<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC5<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template void Ray::CompressImage_BC5<2 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);

#undef _MIN
#undef _MAX
#undef _ABS
#undef _CLAMP

#undef _MIN3
#undef _MAX3

#undef _MIN4
#undef _MAX4
