#include "TextureUtils.h"

#include <climits>

#include <array>
#include <fstream>

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

float f16_to_f32(const uint16_t h) {
    static const uint32_t magic = {113 << 23};
    static const uint32_t shifted_exp = 0x7c00 << 13; // exponent mask after shift
    uint32_t o;

    o = (h & 0x7fff) << 13;         // exponent/mantissa bits
    uint32_t exp = shifted_exp & o; // just the exponent
    o += (127 - 15) << 23;          // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) {  // Inf/NaN?
        o += (128 - 16) << 23; // extra exp adjust
    } else if (exp == 0) {     // Zero/Denormal?
        o += 1 << 23;          // extra exp adjust

        float f;
        memcpy(&f, &o, sizeof(float));
        f -= reinterpret_cast<const float &>(magic); // renormalize
        memcpy(&o, &f, sizeof(float));
    }

    o |= (h & 0x8000) << 16; // sign bit

    float ret;
    memcpy(&ret, &o, sizeof(float));
    return ret;
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

int round_up(int v, int align) { return align * ((v + align - 1) / align); }

uint32_t next_power_of_two(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
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
            mipmaps[mip_count] = std::make_unique<uint8_t[]>(_w * _h * channels);
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
        mipmaps[mip_count] = std::make_unique<uint8_t[]>(_w * _h * 4);
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
template <int Channels> void Extract4x4Block_NEON(const uint8_t src[], int stride, uint8_t dst[64]);

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
    int max_luma = -1, min_luma = INT_MAX;

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

void GetMinMaxAlphaByBBox_Ref(const uint8_t block[16], uint8_t &min_alpha, uint8_t &max_alpha) {
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
template <bool UseAlpha = false, bool Is_YCoCg = false>
void GetMinMaxColorByBBox_NEON(const uint8_t block[64], uint8_t min_color[4], uint8_t max_color[4]);

void GetMinMaxAlphaByBBox_SSE2(const uint8_t block[16], uint8_t &min_alpha, uint8_t &max_alpha);
void GetMinMaxAlphaByBBox_NEON(const uint8_t block[16], uint8_t &min_alpha, uint8_t &max_alpha);

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
void InsetYCoCgBBox_NEON(uint8_t min_color[4], uint8_t max_color[4]);

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

    // c0 ^= c1 ^= mask &= c0 ^= c1;
    c0 ^= c1;
    mask &= c0;
    c1 ^= mask;
    c0 ^= c1;

    min_color[1] = c0;
    max_color[1] = c1;
#endif
}

void SelectYCoCgDiagonal_SSE2(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]);
void SelectYCoCgDiagonal_NEON(const uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]);

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
void ScaleYCoCg_NEON(uint8_t block[64], uint8_t min_color[3], uint8_t max_color[3]);

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
void EmitColorIndices_NEON(const uint8_t block[64], const uint8_t min_color[4], const uint8_t max_color[4],
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
void EmitAlphaIndices_NEON(const uint8_t block[64], uint8_t min_alpha, uint8_t max_alpha, uint8_t *&out_data);

void EmitAlphaOnlyIndices_SSE2(const uint8_t block[16], uint8_t min_alpha, uint8_t max_alpha, uint8_t *&out_data);
void EmitAlphaOnlyIndices_NEON(const uint8_t block[16], uint8_t min_alpha, uint8_t max_alpha, uint8_t *&out_data);

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

#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
void Emit_BC1_Block_NEON(const uint8_t block[64], uint8_t *&out_data) {
    alignas(16) uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_NEON(block, min_color, max_color);

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_NEON(block, min_color, max_color, out_data);
}

template <bool Is_YCoCg> void Emit_BC3_Block_NEON(uint8_t block[64], uint8_t *&out_data) {
    uint8_t min_color[4], max_color[4];
    GetMinMaxColorByBBox_NEON<true /* UseAlpha */, Is_YCoCg>(block, min_color, max_color);
    if (Is_YCoCg) {
        ScaleYCoCg_NEON(block, min_color, max_color);
        InsetYCoCgBBox_NEON(min_color, max_color);
        SelectYCoCgDiagonal_NEON(block, min_color, max_color);
    }

    //
    // Write alpha block
    //

    push_u8(max_color[3], out_data);
    push_u8(min_color[3], out_data);

    EmitAlphaIndices_NEON(block, min_color[3], max_color[3], out_data);

    //
    // Write color block
    //

    push_u16(rgb888_to_rgb565(max_color), out_data);
    push_u16(rgb888_to_rgb565(min_color), out_data);

    EmitColorIndices_NEON(block, min_color, max_color, out_data);
}

void Emit_BC4_Block_NEON(uint8_t block[16], uint8_t *&out_data) {
    uint8_t min_alpha, max_alpha;
    GetMinMaxAlphaByBBox_NEON(block, min_alpha, max_alpha);

    //
    // Write alpha block
    //

    push_u8(max_alpha, out_data);
    push_u8(min_alpha, out_data);

    EmitAlphaOnlyIndices_NEON(block, min_alpha, max_alpha, out_data);
}
#else
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

void Emit_BC4_Block_SSE2(uint8_t block[16], uint8_t *&out_data) {
    uint8_t min_alpha, max_alpha;
    GetMinMaxAlphaByBBox_SSE2(block, min_alpha, max_alpha);

    //
    // Write alpha block
    //

    push_u8(max_alpha, out_data);
    push_u8(min_alpha, out_data);

    EmitAlphaOnlyIndices_SSE2(block, min_alpha, max_alpha, out_data);
}
#endif
} // namespace Ray

int Ray::GetRequiredMemory_BC1(const int w, const int h, const int pitch_align) {
    return round_up(BlockSize_BC1 * ((w + 3) / 4), pitch_align) * ((h + 3) / 4);
}
int Ray::GetRequiredMemory_BC3(const int w, const int h, const int pitch_align) {
    return round_up(BlockSize_BC3 * ((w + 3) / 4), pitch_align) * ((h + 3) / 4);
}
int Ray::GetRequiredMemory_BC4(const int w, const int h, const int pitch_align) {
    return round_up(BlockSize_BC4 * ((w + 3) / 4), pitch_align) * ((h + 3) / 4);
}
int Ray::GetRequiredMemory_BC5(const int w, const int h, const int pitch_align) {
    return round_up(BlockSize_BC5 * ((w + 3) / 4), pitch_align) * ((h + 3) / 4);
}

template <int SrcChannels>
void Ray::CompressImage_BC1(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[], int dst_pitch) {
    alignas(16) uint8_t block[64] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - BlockSize_BC1 * ((w + 3) / 4);

#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    for (int j = 0; j < h_aligned; j += 4, img_src += 4 * w * SrcChannels) {
        const int w_limited = (SrcChannels == 3 && j == h_aligned - 4 && h_aligned == h) ? w_aligned - 4 : w_aligned;
        for (int i = 0; i < w_limited; i += 4) {
            Extract4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, block);
            Emit_BC1_Block_NEON(block, p_out);
        }
        if (w_limited != w_aligned && w_aligned >= 4) {
            // process last block (avoid reading 4 bytes outside of range)
            Extract4x4Block_Ref<SrcChannels>(&img_src[(w_aligned - 4) * SrcChannels], w * SrcChannels, block);
            Emit_BC1_Block_NEON(block, p_out);
        }
        // process last (incomplete) column
        if (w_aligned != w) {
            ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4,
                                                       block);
            Emit_BC1_Block_NEON(block, p_out);
        }
        p_out += pitch_pad;
    }
    // process last (incomplete) row
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i), h % 4,
                                                   block);
        Emit_BC1_Block_NEON(block, p_out);
    }
#else
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
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                       h % 4, block);
            Emit_BC1_Block_SSE2(block, p_out);
        }
    } else {
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
            p_out += pitch_pad;
        }
        // process last row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                       h % 4, block);
            Emit_BC1_Block_Ref(block, p_out);
        }
    }
#endif
}

template void Ray::CompressImage_BC1<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC1<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);

template <bool Is_YCoCg>
void Ray::CompressImage_BC3(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[], int dst_pitch) {
    alignas(16) uint8_t block[64] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - BlockSize_BC3 * ((w + 3) / 4);

#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * 4) {
        for (int i = 0; i < w_aligned; i += 4) {
            Extract4x4Block_NEON<4>(&img_src[i * 4], w * 4, block);
            Emit_BC3_Block_NEON<Is_YCoCg>(block, p_out);
        }
        // process last (incomplete) column
        if (w_aligned != w) {
            ExtractIncomplete4x4Block_Ref<4>(&img_src[w_aligned * 4], w * 4, w % 4, 4, block);
            Emit_BC3_Block_NEON<Is_YCoCg>(block, p_out);
        }
        p_out += pitch_pad;
    }
    // process last (incomplete) row
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        ExtractIncomplete4x4Block_Ref<4>(&img_src[i * 4], w * 4, _MIN(4, w - i), h % 4, block);
        Emit_BC3_Block_NEON<Is_YCoCg>(block, p_out);
    }
#else
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
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<4 /* SrcChannels */>(&img_src[i * 4], w * 4, _MIN(4, w - i), h % 4, block);
            Emit_BC3_Block_SSE2<Is_YCoCg>(block, p_out);
        }
    } else {
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
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<4>(&img_src[i * 4], w * 4, _MIN(4, w - i), h % 4, block);
            Emit_BC3_Block_Ref<Is_YCoCg>(block, p_out);
        }
    }
#endif
}

template void Ray::CompressImage_BC3<false /* Is_YCoCg */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                           int dst_pitch);
template void Ray::CompressImage_BC3<true /* Is_YCoCg */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);

template <int SrcChannels>
void Ray::CompressImage_BC4(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[], int dst_pitch) {
    alignas(16) uint8_t block[16] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - BlockSize_BC4 * ((w + 3) / 4);

#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
        for (int i = 0; i < w_aligned; i += 4) {
            Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, block);
            Emit_BC4_Block_NEON(block, p_out);
        }
        // process last (incomplete) column
        if (w_aligned != w) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4, 4,
                                                          block);
            Emit_BC4_Block_NEON(block, p_out);
        }
        p_out += pitch_pad;
    }
    // process last (incomplete) row
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i), h % 4,
                                                      block);
        Emit_BC4_Block_NEON(block, p_out);
    }
#else
    const CpuFeatures cpu = GetCpuFeatures();
    if (cpu.sse2_supported) {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, block);
                Emit_BC4_Block_SSE2(block, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels], w * SrcChannels, w % 4,
                                                              4, block);
                Emit_BC4_Block_SSE2(block, p_out);
            }
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                          h % 4, block);
            Emit_BC4_Block_SSE2(block, p_out);
        }
    } else {
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
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels], w * SrcChannels, _MIN(4, w - i),
                                                          h % 4, block);
            Emit_BC4_Block_Ref(block, p_out);
        }
    }
#endif
}

template void Ray::CompressImage_BC4<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC4<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC4<2 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC4<1 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);

#include "TextureParams.h"

template <int SrcChannels>
void Ray::CompressImage_BC5(const uint8_t img_src[], const int w, const int h, uint8_t img_dst[], int dst_pitch) {
    alignas(16) uint8_t block1[16] = {}, block2[16] = {};
    uint8_t *p_out = img_dst;

    const int w_aligned = w - (w % 4);
    const int h_aligned = h - (h % 4);

    const int pitch_pad = dst_pitch == 0 ? 0 : dst_pitch - BlockSize_BC5 * ((w + 3) / 4);

#if defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
        for (int i = 0; i < w_aligned; i += 4) {
            Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels, block1);
            Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels, block2);
            Emit_BC4_Block_NEON(block1, p_out);
            Emit_BC4_Block_NEON(block2, p_out);
        }
        // process last (incomplete) column
        if (w_aligned != w) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 0], w * SrcChannels, w % 4,
                                                          4, block1);
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 1], w * SrcChannels, w % 4,
                                                          4, block2);
            Emit_BC4_Block_NEON(block1, p_out);
            Emit_BC4_Block_NEON(block2, p_out);
        }
        p_out += pitch_pad;
    }
    // process last (incomplete) row
    for (int i = 0; i < w && h_aligned != h; i += 4) {
        ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels, _MIN(4, w - i),
                                                      h % 4, block1);
        ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels, _MIN(4, w - i),
                                                      h % 4, block2);
        Emit_BC4_Block_NEON(block1, p_out);
        Emit_BC4_Block_NEON(block2, p_out);
    }
#else
    const CpuFeatures cpu = GetCpuFeatures();
    if (cpu.sse2_supported) {
        for (int j = 0; j < h_aligned; j += 4, img_src += w * 4 * SrcChannels) {
            for (int i = 0; i < w_aligned; i += 4) {
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels, block1);
                Extract4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels, block2);
                Emit_BC4_Block_SSE2(block1, p_out);
                Emit_BC4_Block_SSE2(block2, p_out);
            }
            // process last (incomplete) column
            if (w_aligned != w) {
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 0], w * SrcChannels,
                                                              w % 4, 4, block1);
                ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[w_aligned * SrcChannels + 1], w * SrcChannels,
                                                              w % 4, 4, block2);
                Emit_BC4_Block_SSE2(block1, p_out);
                Emit_BC4_Block_SSE2(block2, p_out);
            }
            p_out += pitch_pad;
        }
        // process last (incomplete) row
        for (int i = 0; i < w && h_aligned != h; i += 4) {
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 0], w * SrcChannels,
                                                          _MIN(4, w - i), h % 4, block1);
            ExtractIncomplete4x4Block_Ref<SrcChannels, 1>(&img_src[i * SrcChannels + 1], w * SrcChannels,
                                                          _MIN(4, w - i), h % 4, block2);
            Emit_BC4_Block_SSE2(block1, p_out);
            Emit_BC4_Block_SSE2(block2, p_out);
        }
    } else {
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
            p_out += pitch_pad;
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
#endif
}

template void Ray::CompressImage_BC5<4 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC5<3 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);
template void Ray::CompressImage_BC5<2 /* SrcChannels */>(const uint8_t img_src[], int w, int h, uint8_t img_dst[],
                                                          int dst_pitch);

template <int N>
int Ray::Preprocess_BCn(const uint8_t in_data[], const int tiles_w, const int tiles_h, const bool flip_vertical,
                        const bool invert_green, uint8_t out_data[], int out_pitch) {
    int read_bytes = tiles_w * tiles_h * GetBlockSize_BCn<N>();
    if (flip_vertical || invert_green) {
        struct bc1_block_t {
            uint16_t color[2];
            uint8_t cl[4];
        };
        static_assert(sizeof(bc1_block_t) == 8, "!");

        struct bc4_block_t {
            uint8_t alpha[2];
            union {
                struct {
                    // ushort 0
                    uint16_t l0 : 12;
                    uint16_t l1_0 : 4;
                    // ushort 1
                    uint16_t l1_1 : 4;
                    uint16_t l1_2 : 4;
                    uint16_t l2_0 : 4;
                    uint16_t l2_1 : 4;
                    // ushort 2
                    uint16_t l2_2 : 4;
                    uint16_t l3 : 12;
                };
                uint8_t ndx[6];
            };
        };
        static_assert(sizeof(bc4_block_t) == 8, "!");

        struct bc5_block_t {
            bc4_block_t r, g;
        };
        static_assert(sizeof(bc5_block_t) == 16, "!");

        struct bc3_block_t {
            bc4_block_t a;
            bc1_block_t rgb;
        };
        static_assert(sizeof(bc3_block_t) == 16, "!");

        if (out_pitch == 0) {
            out_pitch = tiles_w * GetBlockSize_BCn<N>();
        }

        for (int y = 0; y < tiles_h; ++y) {
            if (N == 4) {
                const bc3_block_t *src_blocks = reinterpret_cast<const bc3_block_t *>(in_data);
                src_blocks += (tiles_h - y - 1) * tiles_w;
                bc3_block_t *dst_blocks = reinterpret_cast<bc3_block_t *>(out_data + y * out_pitch);

                for (int i = 0; i < tiles_w; ++i) {
                    const bc3_block_t &src = src_blocks[i];
                    bc3_block_t &dst = dst_blocks[i];

                    dst.rgb.color[0] = src.rgb.color[0];
                    dst.rgb.color[1] = src.rgb.color[1];
                    // flip lines
                    dst.rgb.cl[0] = src.rgb.cl[3];
                    dst.rgb.cl[1] = src.rgb.cl[2];
                    dst.rgb.cl[2] = src.rgb.cl[1];
                    dst.rgb.cl[3] = src.rgb.cl[0];

                    dst.a.alpha[0] = src.a.alpha[0];
                    dst.a.alpha[1] = src.a.alpha[1];
                    // flip lines
                    dst.a.l0 = src.a.l3;
                    dst.a.l1_0 = src.a.l2_0;
                    dst.a.l1_1 = src.a.l2_1;
                    dst.a.l1_2 = src.a.l2_2;
                    dst.a.l2_0 = src.a.l1_0;
                    dst.a.l2_1 = src.a.l1_1;
                    dst.a.l2_2 = src.a.l1_2;
                    dst.a.l3 = src.a.l0;
                }
            } else if (N == 3) {
                const bc1_block_t *src_blocks = reinterpret_cast<const bc1_block_t *>(in_data);
                src_blocks += (tiles_h - y - 1) * tiles_w;
                bc1_block_t *dst_blocks = reinterpret_cast<bc1_block_t *>(out_data + y * out_pitch);

                for (int i = 0; i < tiles_w; ++i) {
                    const bc1_block_t &src = src_blocks[i];
                    bc1_block_t &dst = dst_blocks[i];

                    dst.color[0] = src.color[0];
                    dst.color[1] = src.color[1];
                    // flip lines
                    dst.cl[0] = src.cl[3];
                    dst.cl[1] = src.cl[2];
                    dst.cl[2] = src.cl[1];
                    dst.cl[3] = src.cl[0];
                }
            } else {
                const bc4_block_t *src_blocks = reinterpret_cast<const bc4_block_t *>(in_data);
                src_blocks += N * (tiles_h - y - 1) * tiles_w;
                bc4_block_t *dst_blocks = reinterpret_cast<bc4_block_t *>(out_data + y * out_pitch);

                for (int i = 0; i < tiles_w; ++i) {
                    for (int j = 0; j < N; ++j) {
                        const bc4_block_t &src = src_blocks[i * N + j];
                        bc4_block_t dst;

                        if (invert_green && j == 1) {
                            const bool is_6step = (src.alpha[0] > src.alpha[1]);

                            // flip lo/hi values
                            dst.alpha[1] = 255 - src.alpha[0];
                            dst.alpha[0] = 255 - src.alpha[1];
                            // flip lines
                            dst.l0 = src.l3;
                            dst.l1_0 = src.l2_0;
                            dst.l1_1 = src.l2_1;
                            dst.l1_2 = src.l2_2;
                            dst.l2_0 = src.l1_0;
                            dst.l2_1 = src.l1_1;
                            dst.l2_2 = src.l1_2;
                            dst.l3 = src.l0;

                            // remap indices
                            static const int _6step_mapping[] = {1, 0, 7, 6, 5, 4, 3, 2};
                            static const int _4step_mapping[] = {1, 0, 5, 4, 3, 2, 6, 7};

                            for (int i = 0; i < 16; ++i) {
                                int next_bit = i * 3;

                                int idx = 0, bit;
                                bit = (dst.ndx[next_bit >> 3] >> (next_bit & 7)) & 1;
                                idx += bit << 0;
                                ++next_bit;
                                bit = (dst.ndx[next_bit >> 3] >> (next_bit & 7)) & 1;
                                idx += bit << 1;
                                ++next_bit;
                                bit = (dst.ndx[next_bit >> 3] >> (next_bit & 7)) & 1;
                                idx += bit << 2;
                                ++next_bit;

                                idx = is_6step ? _6step_mapping[idx] : _4step_mapping[idx];

                                --next_bit;
                                bit = (idx >> 2) & 1;
                                dst.ndx[next_bit >> 3] &= ~(1 << (next_bit & 7));
                                dst.ndx[next_bit >> 3] |= (bit << (next_bit & 7));
                                --next_bit;
                                bit = (idx >> 1) & 1;
                                dst.ndx[next_bit >> 3] &= ~(1 << (next_bit & 7));
                                dst.ndx[next_bit >> 3] |= (bit << (next_bit & 7));
                                --next_bit;
                                bit = (idx >> 0) & 1;
                                dst.ndx[next_bit >> 3] &= ~(1 << (next_bit & 7));
                                dst.ndx[next_bit >> 3] |= (bit << (next_bit & 7));
                            }
                        } else {
                            dst.alpha[0] = src.alpha[0];
                            dst.alpha[1] = src.alpha[1];
                            // flip lines
                            dst.l0 = src.l3;
                            dst.l1_0 = src.l2_0;
                            dst.l1_1 = src.l2_1;
                            dst.l1_2 = src.l2_2;
                            dst.l2_0 = src.l1_0;
                            dst.l2_1 = src.l1_1;
                            dst.l2_2 = src.l1_2;
                            dst.l3 = src.l0;
                        }

                        dst_blocks[i * N + j] = dst;
                    }
                }
            }
        }
    } else {
        // no preprocessing needed, just copy
        if (out_pitch == 0) {
            memcpy(out_data, in_data, read_bytes);
        } else {
            int in_offset = 0, out_offset = 0;
            for (int ty = 0; ty < tiles_h; ++ty) {
                const int line_len = tiles_w * GetBlockSize_BCn<N>();
                memcpy(&out_data[out_offset], &in_data[in_offset], line_len);
                in_offset += line_len;
                out_offset += out_pitch;
            }
        }
    }
    return read_bytes;
}

template int Ray::Preprocess_BCn<1>(const uint8_t in_data[], const int tiles_w, const int tiles_h,
                                    const bool flip_vertical, const bool invert_green, uint8_t out_data[],
                                    int out_pitch);
template int Ray::Preprocess_BCn<2>(const uint8_t in_data[], const int tiles_w, const int tiles_h,
                                    const bool flip_vertical, const bool invert_green, uint8_t out_data[],
                                    int out_pitch);
template int Ray::Preprocess_BCn<3>(const uint8_t in_data[], const int tiles_w, const int tiles_h,
                                    const bool flip_vertical, const bool invert_green, uint8_t out_data[],
                                    int out_pitch);
template int Ray::Preprocess_BCn<4>(const uint8_t in_data[], const int tiles_w, const int tiles_h,
                                    const bool flip_vertical, const bool invert_green, uint8_t out_data[],
                                    int out_pitch);

void Ray::ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                              Span<uint32_t> new_vtx_indices, Span<const uint32_t> indices) {
    auto cross = [](const Ref::simd_fvec3 &v1, const Ref::simd_fvec3 &v2) -> Ref::simd_fvec3 {
        return Ref::simd_fvec3{v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
                               v1[0] * v2[1] - v1[1] * v2[0]};
    };

    std::vector<std::array<uint32_t, 3>> twin_verts(vertices.size(), {0, 0, 0});
    aligned_vector<Ref::simd_fvec3> binormals(vertices.size());
    for (int i = 0; i < indices.size(); i += 3) {
        vertex_t *v0 = &vertices[indices[i + 0]];
        vertex_t *v1 = &vertices[indices[i + 1]];
        vertex_t *v2 = &vertices[indices[i + 2]];

        Ref::simd_fvec3 &b0 = binormals[indices[i + 0]];
        Ref::simd_fvec3 &b1 = binormals[indices[i + 1]];
        Ref::simd_fvec3 &b2 = binormals[indices[i + 2]];

        const Ref::simd_fvec3 dp1 = Ref::simd_fvec3(v1->p) - Ref::simd_fvec3(v0->p);
        const Ref::simd_fvec3 dp2 = Ref::simd_fvec3(v2->p) - Ref::simd_fvec3(v0->p);

        const Ref::simd_fvec2 dt1 = Ref::simd_fvec2(v1->t) - Ref::simd_fvec2(v0->t);
        const Ref::simd_fvec2 dt2 = Ref::simd_fvec2(v2->t) - Ref::simd_fvec2(v0->t);

        Ref::simd_fvec3 tangent, binormal;

        const float det = std::abs(dt1[0] * dt2[1] - dt1[1] * dt2[0]);
        if (det > FLT_EPS) {
            const float inv_det = 1.0f / det;
            tangent = (dp1 * dt2[1] - dp2 * dt1[1]) * inv_det;
            binormal = (dp2 * dt1[0] - dp1 * dt2[0]) * inv_det;
        } else {
            Ref::simd_fvec3 plane_N = cross(dp1, dp2);

            int w = 2;
            tangent = Ref::simd_fvec3{0.0f, 1.0f, 0.0f};
            if (std::abs(plane_N[0]) <= std::abs(plane_N[1]) && std::abs(plane_N[0]) <= std::abs(plane_N[2])) {
                tangent = Ref::simd_fvec3{1.0f, 0.0f, 0.0f};
                w = 1;
            } else if (std::abs(plane_N[2]) <= std::abs(plane_N[0]) && std::abs(plane_N[2]) <= std::abs(plane_N[1])) {
                tangent = Ref::simd_fvec3{0.0f, 0.0f, 1.0f};
                w = 0;
            }

            if (std::abs(plane_N[w]) > FLT_EPS) {
                binormal = normalize(cross(Ref::simd_fvec3(plane_N), tangent));
                tangent = normalize(cross(Ref::simd_fvec3(plane_N), binormal));
            } else {
                binormal = {0.0f};
                tangent = {0.0f};
            }
        }

        int i1 = (v0->b[0] * tangent[0] + v0->b[1] * tangent[1] + v0->b[2] * tangent[2]) < 0;
        int i2 = 2 * (b0[0] * binormal[0] + b0[1] * binormal[1] + b0[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 0]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v0);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 0]][i1 + i2 - 1] = index;

                v1 = &vertices[indices[i + 1]];
                v2 = &vertices[indices[i + 2]];
            }
            new_vtx_indices[i] = index;
            v0 = &vertices[index - vtx_offset];
        } else {
            b0 = binormal;
        }

        v0->b[0] += tangent[0];
        v0->b[1] += tangent[1];
        v0->b[2] += tangent[2];

        i1 = v1->b[0] * tangent[0] + v1->b[1] * tangent[1] + v1->b[2] * tangent[2] < 0;
        i2 = 2 * (b1[0] * binormal[0] + b1[1] * binormal[1] + b1[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 1]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v1);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 1]][i1 + i2 - 1] = index;

                v0 = &vertices[indices[i + 0]];
                v2 = &vertices[indices[i + 2]];
            }
            new_vtx_indices[i + 1] = index;
            v1 = &vertices[index - vtx_offset];
        } else {
            b1 = binormal;
        }

        v1->b[0] += tangent[0];
        v1->b[1] += tangent[1];
        v1->b[2] += tangent[2];

        i1 = v2->b[0] * tangent[0] + v2->b[1] * tangent[1] + v2->b[2] * tangent[2] < 0;
        i2 = 2 * (b2[0] * binormal[0] + b2[1] * binormal[1] + b2[2] * binormal[2] < 0);

        if (i1 || i2) {
            uint32_t index = twin_verts[indices[i + 2]][i1 + i2 - 1];
            if (index == 0) {
                index = uint32_t(vtx_offset + vertices.size());
                vertices.push_back(*v2);
                memset(&vertices.back().b[0], 0, 3 * sizeof(float));
                twin_verts[indices[i + 2]][i1 + i2 - 1] = index;

                v0 = &vertices[indices[i + 0]];
                v1 = &vertices[indices[i + 1]];
            }
            new_vtx_indices[i + 2] = index;
            v2 = &vertices[index - vtx_offset];
        } else {
            b2 = binormal;
        }

        v2->b[0] += tangent[0];
        v2->b[1] += tangent[1];
        v2->b[2] += tangent[2];
    }

    for (size_t i = vtx_start; i < vertices.size(); i++) {
        vertex_t &v = vertices[i];

        if (std::abs(v.b[0]) > FLT_EPS || std::abs(v.b[1]) > FLT_EPS || std::abs(v.b[2]) > FLT_EPS) {
            const auto tangent = Ref::simd_fvec3{v.b};
            Ref::simd_fvec3 binormal = cross(Ref::simd_fvec3(v.n), tangent);
            const float l = length(binormal);
            if (l > FLT_EPS) {
                binormal /= l;
                memcpy(&v.b[0], value_ptr(binormal), 3 * sizeof(float));
            }
        }
    }
}

std::unique_ptr<uint8_t[]> Ray::ReadTGAFile(const void *data, const int data_len, int &w, int &h, eTexFormat &format) {
    uint32_t img_size;
    ReadTGAFile(data, data_len, w, h, format, nullptr, img_size);

    std::unique_ptr<uint8_t[]> image_ret(new uint8_t[img_size]);
    ReadTGAFile(data, data_len, w, h, format, image_ret.get(), img_size);

    return image_ret;
}

bool Ray::ReadTGAFile(const void *data, const int data_len, int &w, int &h, eTexFormat &format, uint8_t *out_data,
                      uint32_t &out_size) {
    const uint8_t tga_header[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto *tga_compare = (const uint8_t *)data;
    const uint8_t *img_header = (const uint8_t *)data + sizeof(tga_header);
    bool compressed = false;

    if (data_len && data_len < sizeof(tga_header)) {
        return {};
    }

    if (memcmp(tga_header, tga_compare, sizeof(tga_header)) != 0) {
        if (tga_compare[2] == 1) {
            // fprintf(stderr, "Image cannot be indexed color.");
            return false;
        }
        if (tga_compare[2] == 3) {
            // fprintf(stderr, "Image cannot be greyscale color.");
            return false;
        }
        if (tga_compare[2] == 9 || tga_compare[2] == 10) {
            compressed = true;
        }
    }

    w = int(img_header[1] * 256u + img_header[0]);
    h = int(img_header[3] * 256u + img_header[2]);

    if (w <= 0 || h <= 0 || (img_header[4] != 24 && img_header[4] != 32)) {
        if (w <= 0 || h <= 0) {
            // fprintf(stderr, "Image must have a width and height greater than 0");
        }
        if (img_header[4] != 24 && img_header[4] != 32) {
            // fprintf(stderr, "Image must be 24 or 32 bit");
        }
        return false;
    }

    const uint32_t bpp = img_header[4];
    const uint32_t bytes_per_pixel = bpp / 8;
    if (bpp == 32) {
        format = eTexFormat::RawRGBA8888;
    } else if (bpp == 24) {
        format = eTexFormat::RawRGB888;
    }

    if (out_data && out_size < w * h * bytes_per_pixel) {
        return false;
    }

    out_size = w * h * bytes_per_pixel;
    if (out_data) {
        const bool flip_y = (img_header[5] & (1 << 5)) == 0;
        const auto *image_data = reinterpret_cast<const uint8_t *>(data) + 18;

        if (!compressed) {
            if (flip_y) {
                out_data += w * (h - 1) * bytes_per_pixel;
            }
            for (int y = 0; y < h; ++y) {
                for (uint32_t i = 0; i < w * bytes_per_pixel; i += bytes_per_pixel) {
                    out_data[i + 0] = image_data[i + 2];
                    out_data[i + 1] = image_data[i + 1];
                    out_data[i + 2] = image_data[i + 0];
                    if (bytes_per_pixel == 4) {
                        out_data[i + 3] = image_data[i + 3];
                    }
                }
                image_data += w * bytes_per_pixel;
                out_data += flip_y ? -int(w * bytes_per_pixel) : int(w * bytes_per_pixel);
            }
        } else {
            for (size_t num = 0; num < out_size;) {
                uint8_t packet_header = *image_data++;
                if (packet_header & (1u << 7u)) {
                    uint8_t color[4];
                    unsigned size = (packet_header & ~(1u << 7u)) + 1;
                    size *= bytes_per_pixel;
                    for (unsigned i = 0; i < bytes_per_pixel; i++) {
                        color[i] = *image_data++;
                    }
                    for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                        out_data[num] = color[2];
                        out_data[num + 1] = color[1];
                        out_data[num + 2] = color[0];
                        if (bytes_per_pixel == 4) {
                            out_data[num + 3] = color[3];
                        }
                    }
                } else {
                    unsigned size = (packet_header & ~(1u << 7u)) + 1;
                    size *= bytes_per_pixel;
                    for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                        out_data[num] = image_data[i + 2];
                        out_data[num + 1] = image_data[i + 1];
                        out_data[num + 2] = image_data[i];
                        if (bytes_per_pixel == 4) {
                            out_data[num + 3] = image_data[i + 3];
                        }
                    }
                    image_data += size;
                }
            }

            for (int y = 0; y < (h / 2) && flip_y; ++y) {
                if (bytes_per_pixel == 4) {
                    for (int i = 0; i < w * 4; ++i) {
                        const uint8_t t = out_data[y * w * 4 + i];
                        out_data[y * w * 4 + i] = out_data[(h - y - 1) * w * 4 + i];
                        out_data[(h - y - 1) * w * 4 + i] = t;
                    }
                } else {
                    for (int i = 0; i < w * 3; ++i) {
                        const uint8_t t = out_data[y * w * 3 + i];
                        out_data[y * w * 3 + i] = out_data[(h - y - 1) * w * 3 + i];
                        out_data[(h - y - 1) * w * 3 + i] = t;
                    }
                }
            }
        }
    }

    return true;
}

void Ray::WriteTGA(const uint8_t *data, const int w, const int h, const int bpp, const char *name) {
    std::ofstream file(name, std::ios::binary);

    unsigned char header[18] = {0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    header[12] = w & 0xFF;
    header[13] = (w >> 8) & 0xFF;
    header[14] = (h)&0xFF;
    header[15] = (h >> 8) & 0xFF;
    header[16] = bpp * 8;
    header[17] |= (1 << 5); // set origin to upper left corner

    file.write((char *)&header[0], sizeof(unsigned char) * 18);

    uint8_t out_data[512];
    int out_data_count = 0;

    out_data[0] = data[2];
    out_data[1] = data[1];
    out_data[2] = data[0];
    if (bpp == 4) {
        out_data[3] = data[3];
    }
    out_data_count = bpp;

    int ref_point = 0, is_rle = -1;

    for (int i = 1; i < w * h; ++i) {
        uint8_t temp[4] = {};
        temp[0] = data[i * bpp + 2];
        temp[1] = data[i * bpp + 1];
        temp[2] = data[i * bpp + 0];
        if (bpp == 4) {
            temp[3] = data[i * bpp + 3];
        }

        const int prev_rle = is_rle;
        is_rle = (memcmp(temp, out_data, bpp) == 0);

        if ((is_rle != prev_rle && prev_rle != -1) || (i - ref_point) >= 128 || i == w * h - 1) {
            { // write data
                uint8_t packet_header = uint8_t(i - ref_point - 1);
                if (prev_rle == 1) {
                    packet_header |= (1u << 7u);
                }

                file.write((const char *)&packet_header, 1);
                file.write((const char *)out_data, out_data_count);

                out_data_count = 0;
            }

            ref_point = i;
            is_rle = -1;
        }

        if (is_rle != 1) {
            memcpy(out_data + out_data_count, temp, bpp);
            out_data_count += bpp;
        }
    }

    static const char footer[26] = "\0\0\0\0"         // no extension area
                                   "\0\0\0\0"         // no developer directory
                                   "TRUEVISION-XFILE" // yep, this is a TGA file
                                   ".";
    file.write((const char *)&footer, sizeof(footer));
}

void Ray::WritePFM(const char *base_name, const float values[], int w, int h, int channels) {
    for (int c = 0; c < channels; ++c) {
        // Open the file
        const std::string filename = base_name + std::to_string(c) + ".pfm";
        std::ofstream file(filename, std::ios::binary);
        if (file.fail()) {
            throw std::runtime_error("cannot open image file: " + std::string(filename));
        }

        // Write the header
        file << "Pf" << std::endl;
        file << w << " " << h << std::endl;
        file << "-1.0" << std::endl;

        // Write the pixels
        for (int y = h - 1; y >= 0; --y) {
            for (int x = 0; x < w; ++x) {
                const float v = values[channels * (y * w + x) + c];
                if (std::isnan(v)) {
                    printf("NAN at (%i %i)!!\n", x, y);
                }
                file.write((char *)&v, sizeof(float));
            }
        }
    }
}

#undef _MIN
#undef _MAX
#undef _ABS
#undef _CLAMP

#undef _MIN3
#undef _MAX3

#undef _MIN4
#undef _MAX4
