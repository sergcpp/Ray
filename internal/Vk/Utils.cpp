#include "Utils.h"

#include <cmath>

#include <deque>

#include "../CoreVK.h"
#include "Texture.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#define _MIN(x, y) ((x) < (y) ? (x) : (y))
#define _MAX(x, y) ((x) < (y) ? (y) : (x))
#define _ABS(x) ((x) < 0 ? -(x) : (x))
#define _CLAMP(x, lo, hi) (_MIN(_MAX((x), (lo)), (hi)))

#define _MIN3(x, y, z) _MIN((x), _MIN((y), (z)))
#define _MAX3(x, y, z) _MAX((x), _MAX((y), (z)))

#define _MIN4(x, y, z, w) _MIN(_MIN((x), (y)), _MIN((z), (w)))
#define _MAX4(x, y, z, w) _MAX(_MAX((x), (y)), _MAX((z), (w)))

namespace Ray {
namespace Vk {
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

const uint8_t _blank_DXT5_block_4x4[] = {0x00, 0x00, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const int _blank_DXT5_block_4x4_len = sizeof(_blank_DXT5_block_4x4);

const uint8_t _blank_ASTC_block_4x4[] = {0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const int _blank_ASTC_block_4x4_len = sizeof(_blank_ASTC_block_4x4);

} // namespace Vk
} // namespace Ray

std::unique_ptr<uint8_t[]> Ray::Vk::ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format) {
    uint32_t img_size;
    ReadTGAFile(data, w, h, format, nullptr, img_size);

    std::unique_ptr<uint8_t[]> image_ret(new uint8_t[img_size]);
    ReadTGAFile(data, w, h, format, image_ret.get(), img_size);

    return image_ret;
}

bool Ray::Vk::ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format, uint8_t *out_data, uint32_t &out_size) {
    const uint8_t tga_header[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto *tga_compare = (const uint8_t *)data;
    const uint8_t *img_header = (const uint8_t *)data + sizeof(tga_header);
    bool compressed = false;

    if (memcmp(tga_header, tga_compare, sizeof(tga_header)) != 0) {
        if (tga_compare[2] == 1) {
            fprintf(stderr, "Image cannot be indexed color.");
            return false;
        }
        if (tga_compare[2] == 3) {
            fprintf(stderr, "Image cannot be greyscale color.");
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
            fprintf(stderr, "Image must have a width and height greater than 0");
        }
        if (img_header[4] != 24 && img_header[4] != 32) {
            fprintf(stderr, "Image must be 24 or 32 bit");
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
        const uint8_t *image_data = (const uint8_t *)data + 18;

        if (!compressed) {
            for (size_t i = 0; i < out_size; i += bytes_per_pixel) {
                out_data[i] = image_data[i + 2];
                out_data[i + 1] = image_data[i + 1];
                out_data[i + 2] = image_data[i];
                if (bytes_per_pixel == 4) {
                    out_data[i + 3] = image_data[i + 3];
                }
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
        }
    }

    return true;
}

void Ray::Vk::RGBMDecode(const uint8_t rgbm[4], float out_rgb[3]) {
    out_rgb[0] = 4.0f * (rgbm[0] / 255.0f) * (rgbm[3] / 255.0f);
    out_rgb[1] = 4.0f * (rgbm[1] / 255.0f) * (rgbm[3] / 255.0f);
    out_rgb[2] = 4.0f * (rgbm[2] / 255.0f) * (rgbm[3] / 255.0f);
}

void Ray::Vk::RGBMEncode(const float rgb[3], uint8_t out_rgbm[4]) {
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

std::unique_ptr<float[]> Ray::Vk::ConvertRGBE_to_RGB32F(const uint8_t image_data[], const int w, const int h) {
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

std::unique_ptr<uint16_t[]> Ray::Vk::ConvertRGBE_to_RGB16F(const uint8_t image_data[], const int w, const int h) {
    std::unique_ptr<uint16_t[]> fp16_data(new uint16_t[w * h * 3]);
    ConvertRGBE_to_RGB16F(image_data, w, h, fp16_data.get());
    return fp16_data;
}

void Ray::Vk::ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h, uint16_t *out_data) {
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

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertRGB32F_to_RGBE(const float image_data[], const int w, const int h,
                                                          const int channels) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            simd_fvec4 val;

            if (channels == 3) {
                val[0] = image_data[3 * (y * w + x) + 0];
                val[1] = image_data[3 * (y * w + x) + 1];
                val[2] = image_data[3 * (y * w + x) + 2];
            } else if (channels == 4) {
                val[0] = image_data[4 * (y * w + x) + 0];
                val[1] = image_data[4 * (y * w + x) + 1];
                val[2] = image_data[4 * (y * w + x) + 2];
            }

            auto exp = simd_fvec4{std::log2(val[0]), std::log2(val[1]), std::log2(val[2]), 0.0f};
            for (int i = 0; i < 3; i++) {
                exp[i] = std::ceil(exp[i]);
                if (exp[i] < -128.0f) {
                    exp[i] = -128.0f;
                } else if (exp[i] > 127.0f) {
                    exp[i] = 127.0f;
                }
            }

            const float common_exp = std::max(exp[0], std::max(exp[1], exp[2]));
            const float range = std::exp2(common_exp);

            simd_fvec4 mantissa = val / range;
            for (int i = 0; i < 3; i++) {
                if (mantissa[i] < 0.0f)
                    mantissa[i] = 0.0f;
                else if (mantissa[i] > 1.0f)
                    mantissa[i] = 1.0f;
            }

            const auto res = simd_fvec4{mantissa[0], mantissa[1], mantissa[2], common_exp + 128.0f};

            u8_data[(y * w + x) * 4 + 0] = (uint8_t)_CLAMP(int(res[0] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 1] = (uint8_t)_CLAMP(int(res[1] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 2] = (uint8_t)_CLAMP(int(res[2] * 255), 0, 255);
            u8_data[(y * w + x) * 4 + 3] = (uint8_t)_CLAMP(int(res[3]), 0, 255);
        }
    }

    return u8_data;
}

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertRGB32F_to_RGBM(const float image_data[], const int w, const int h,
                                                          const int channels) {
    std::unique_ptr<uint8_t[]> u8_data(new uint8_t[w * h * 4]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            RGBMEncode(&image_data[channels * (y * w + x)], &u8_data[(y * w + x) * 4]);
        }
    }

    return u8_data;
}

void Ray::Vk::ConvertRGB_to_YCoCg_rev(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) {
    RGB_to_YCoCg_reversible(in_RGB, out_YCoCg);
}

void Ray::Vk::ConvertYCoCg_to_RGB_rev(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) {
    YCoCg_to_RGB_reversible(in_YCoCg, out_RGB);
}

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertRGB_to_CoCgxY_rev(const uint8_t image_data[], const int w, const int h) {
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

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertCoCgxY_to_RGB_rev(const uint8_t image_data[], const int w, const int h) {
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

void Ray::Vk::ConvertRGB_to_YCoCg(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]) { RGB_to_YCoCg(in_RGB, out_YCoCg); }
void Ray::Vk::ConvertYCoCg_to_RGB(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]) { YCoCg_to_RGB(in_YCoCg, out_RGB); }

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertRGB_to_CoCgxY(const uint8_t image_data[], const int w, const int h) {
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

std::unique_ptr<uint8_t[]> Ray::Vk::ConvertCoCgxY_to_RGB(const uint8_t image_data[], const int w, const int h) {
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

int Ray::Vk::InitMipMaps(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16], const int channels,
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

int Ray::Vk::InitMipMapsRGBM(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16]) {
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

void Ray::Vk::ReorderTriangleIndices(const uint32_t *indices, const uint32_t indices_count, const uint32_t vtx_count,
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

    const int MaxSizeVertexCache = 32;

    auto get_vertex_score = [MaxSizeVertexCache](int32_t cache_pos, uint32_t active_tris_count) -> float {
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
        auto it = std::find(std::begin(lru_cache), std::end(lru_cache), vtx_index);

        if (it == std::end(lru_cache)) {
            lru_cache.push_back(vtx_index);
            it = std::begin(lru_cache);
        }

        if (it != std::begin(lru_cache)) {
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
                    auto it = std::find(std::begin(out_tris_to_update), std::end(out_tris_to_update), tri_index);
                    if (it == std::end(out_tris_to_update)) {
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

#undef _MIN
#undef _MAX

#undef force_inline
