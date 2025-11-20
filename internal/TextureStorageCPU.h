#pragma once

#include <cmath>
#include <memory>

#include "Core.h"
#include "ImageSplitter.h"
#include "TextureUtils.h"

#pragma warning(push)
#pragma warning(disable : 6294) // Ill-defined for-loop
#pragma warning(disable : 6201) // Index is out of valid index range

namespace Ray {
namespace Cpu {
class TexStorageBase {
  public:
    virtual ~TexStorageBase() = default;

    virtual void GetIRes(int index, int lod, int res[2]) const = 0;
    virtual void GetFRes(int index, int lod, float res[2]) const = 0;

    virtual color_rgba_t Fetch(int index, int x, int y, int lod) const = 0;
    virtual color_rgba_t Fetch(int index, float x, float y, int lod) const = 0;

    virtual bool Free(int index) = 0;
};

template <typename T, int N> class TexStorageLinear : public TexStorageBase {
    using ColorType = color_t<T, N>;
    struct ImgData {
        int res[NUM_MIP_LEVELS][2];
        int lod_offsets[NUM_MIP_LEVELS];
        std::unique_ptr<ColorType[]> pixels;
    };

    std::vector<ImgData> images_;
    std::vector<int> free_slots_;

    void WriteImageData(int index, int lod, const ColorType data[]);

  public:
    force_inline int img_count() const { return int(images_.size() - free_slots_.size()); }

    force_inline ColorType Get(const int index, int x, int y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];

        x %= p.res[lod][0];
        y %= p.res[lod][1];

        return p.pixels[p.lod_offsets[lod] + w * y + x];
    }

    force_inline ColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];
        const int h = p.res[lod][1];

        x -= std::floor(x);
        y -= std::floor(y);

        return p.pixels[p.lod_offsets[lod] + w * int(y * h - 0.5f) + int(x * w - 0.5f)];
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0];
        res[1] = p.res[lod][1];
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0]);
        res[1] = float(p.res[lod][1]);
    }

    color_rgba_t Fetch(const int index, const int x, const int y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    color_rgba_t Fetch(const int index, const float x, const float y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    int Allocate(Span<const ColorType> data, const int res[2], bool mips);
    bool Free(int index) override final;
};

extern template class Ray::Cpu::TexStorageLinear<uint8_t, 4>;
extern template class Ray::Cpu::TexStorageLinear<uint8_t, 3>;
extern template class Ray::Cpu::TexStorageLinear<uint8_t, 2>;
extern template class Ray::Cpu::TexStorageLinear<uint8_t, 1>;

template <typename T, int N> class TexStorageTiled : public TexStorageBase {
    static const int TileSize = 4;

    using ColorType = color_t<T, N>;
    struct ImgData {
        int res[NUM_MIP_LEVELS][2], res_in_tiles[NUM_MIP_LEVELS][2];
        int lod_offsets[NUM_MIP_LEVELS];
        std::unique_ptr<ColorType[]> pixels;
    };

    std::vector<ImgData> images_;
    std::vector<int> free_slots_;

  public:
    force_inline int img_count() const { return int(images_.size() - free_slots_.size()); }

    force_inline ColorType Get(const int index, int x, int y, const int lod) const {
        const ImgData &p = images_[index];

        x %= p.res[lod][0];
        y %= p.res[lod][1];

        const int tilex = x / TileSize, tiley = y / TileSize;
        const int in_tilex = x % TileSize, in_tiley = y % TileSize;

        const int w_in_tiles = p.res_in_tiles[lod][0];
        return p.pixels[p.lod_offsets[lod] + (tiley * w_in_tiles + tilex) * TileSize * TileSize + in_tiley * TileSize +
                        in_tilex];
    }

    force_inline ColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];
        const int h = p.res[lod][1];

        x -= std::floor(x);
        y -= std::floor(y);

        return Get(index, int(x * w - 0.5f), int(y * h - 0.5f), lod);
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0];
        res[1] = p.res[lod][1];
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0]);
        res[1] = float(p.res[lod][1]);
    }

    color_rgba_t Fetch(const int index, const int x, const int y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    color_rgba_t Fetch(const int index, const float x, const float y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    int Allocate(Span<const ColorType> data, const int res[2], bool mips);
    bool Free(int index) override;
};

extern template class TexStorageTiled<uint8_t, 4>;
extern template class TexStorageTiled<uint8_t, 3>;
extern template class TexStorageTiled<uint8_t, 2>;
extern template class TexStorageTiled<uint8_t, 1>;

template <typename T, int N> class TexStorageSwizzled : public TexStorageBase {
    using ColorType = color_t<T, N>;
    struct ImgData {
        int res[NUM_MIP_LEVELS][2], tile_y_stride[NUM_MIP_LEVELS];
        int lod_offsets[NUM_MIP_LEVELS];
        std::unique_ptr<ColorType[]> pixels;
    };

    std::vector<ImgData> images_;
    std::vector<int> free_slots_;

    static const uint32_t OuterTileW = 64;
    static const uint32_t OuterTileH = 64;

    force_inline static uint32_t swizzle_x_tile(uint32_t x) {
        return ((x & 0x03) << 0) | ((x & 0x04) << 2) | ((x & 0x38) << 4) | ((x & ~0x3f) << 6);
    }
    force_inline static uint32_t swizzle_y(uint32_t y) {
        return ((y & 0x03) << 2) | ((y & 0x0c) << 3) | ((y & 0x30) << 6);
    }

    force_inline uint32_t EncodeSwizzle(const uint32_t x, const uint32_t y, const uint32_t tile_y_stride) const {
        const uint32_t y_off = (y / OuterTileH) * tile_y_stride + swizzle_y(y);
        const uint32_t x_off = swizzle_x_tile(x);
        return y_off + x_off;
    }

  public:
    force_inline int img_count() const { return int(images_.size() - free_slots_.size()); }

    force_inline ColorType Get(const int index, int x, int y, const int lod) const {
        const ImgData &p = images_[index];

        x %= p.res[lod][0];
        y %= p.res[lod][1];

        return p.pixels[p.lod_offsets[lod] + EncodeSwizzle(x, y, p.tile_y_stride[lod])];
    }

    force_inline ColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];
        const int h = p.res[lod][1];

        x -= std::floor(x);
        y -= std::floor(y);

        return Get(index, int(x * w - 0.5f), int(y * h - 0.5f), lod);
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0];
        res[1] = p.res[lod][1];
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0]);
        res[1] = float(p.res[lod][1]);
    }

    color_rgba_t Fetch(const int index, const int x, const int y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    color_rgba_t Fetch(const int index, const float x, const float y, const int lod) const override {
        const ColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    int Allocate(Span<const ColorType> data, const int res[2], bool mips);
    bool Free(int index) override;
};

extern template class TexStorageSwizzled<uint8_t, 4>;
extern template class TexStorageSwizzled<uint8_t, 3>;
extern template class TexStorageSwizzled<uint8_t, 2>;
extern template class TexStorageSwizzled<uint8_t, 1>;

template <int N> struct BCCache {
    int decoded_index = -1, decoded_block_offset = -1;
    color_t<uint8_t, N> decoded_block[16];

    void Invalidate() {
        decoded_index = -1;
        decoded_block_offset = -1;
    }
};

template <int N> BCCache<N> &get_per_thread_BCCache() {
    static thread_local BCCache<N> g_block_cache;
    return g_block_cache;
}

force_inline int convert_bit_range(int c, int from_bits, int to_bits) {
    int b = (1 << (from_bits - 1)) + c * ((1 << to_bits) - 1);
    return (b + (b >> from_bits)) >> from_bits;
}
force_inline void rgb_888_from_565(unsigned int c, int *r, int *g, int *b) {
    *r = convert_bit_range((c >> 11) & 31, 5, 8);
    *g = convert_bit_range((c >> 05) & 63, 6, 8);
    *b = convert_bit_range((c >> 00) & 31, 5, 8);
}

template <int N> class TexStorageBCn : public TexStorageBase {
    static_assert(N <= 4, "!");
    const int BlockSizes[4] = {BlockSize_BC4, BlockSize_BC5, BlockSize_BC1, BlockSize_BC3};
    static const int InChannels = (N == 4) ? 3 : N;
    static const int TileSize = 4;

    using InColorType = color_t<uint8_t, InChannels>;
    using OutColorType = color_t<uint8_t, N>;
    struct ImgData {
        int res[NUM_MIP_LEVELS][2], res_in_tiles[NUM_MIP_LEVELS][2];
        int lod_offsets[NUM_MIP_LEVELS];
        std::unique_ptr<uint8_t[]> pixels;
    };

    std::vector<ImgData> images_;
    std::vector<int> free_slots_;

  public:
    force_inline int img_count() const { return int(images_.size() - free_slots_.size()); }

    force_inline OutColorType Get(const int index, int x, int y, const int lod) const {
        const ImgData &p = images_[index];

        x %= p.res[lod][0];
        y %= p.res[lod][1];

        const int tilex = x / TileSize, tiley = y / TileSize;
        const int in_tilex = x % TileSize, in_tiley = y % TileSize;

        const int w_in_tiles = p.res_in_tiles[lod][0];
        const int block_offset = p.lod_offsets[lod] + (tiley * w_in_tiles + tilex) * BlockSizes[N - 1];

        BCCache<N> &cache = get_per_thread_BCCache<N>();

        if (index != cache.decoded_index || block_offset != cache.decoded_block_offset) {
            const uint8_t *compressed_block = &p.pixels[block_offset];

            if (N == 4) {
                { // decode alpha
                    uint8_t decode_data[8];
                    decode_data[0] = compressed_block[0];
                    decode_data[1] = compressed_block[1];

                    // 6-step intermediate values
                    decode_data[2] = (6 * decode_data[0] + 1 * decode_data[1]) / 7;
                    decode_data[3] = (5 * decode_data[0] + 2 * decode_data[1]) / 7;
                    decode_data[4] = (4 * decode_data[0] + 3 * decode_data[1]) / 7;
                    decode_data[5] = (3 * decode_data[0] + 4 * decode_data[1]) / 7;
                    decode_data[6] = (2 * decode_data[0] + 5 * decode_data[1]) / 7;
                    decode_data[7] = (1 * decode_data[0] + 6 * decode_data[1]) / 7;

                    int next_bit = 8 * 2;
                    for (OutColorType &c : cache.decoded_block) {
                        int idx = 0, bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 0;
                        ++next_bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 1;
                        ++next_bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 2;
                        ++next_bit;

                        c.v[3 % N] = decode_data[idx & 7];
                    }
                    compressed_block += BlockSize_BC4;
                }
                { // decode color
                    // find the 2 primary colors
                    const int c0 = compressed_block[0] + (compressed_block[1] << 8);
                    const int c1 = compressed_block[2] + (compressed_block[3] << 8);
                    int r, g, b;
                    rgb_888_from_565(c0, &r, &g, &b);
                    uint8_t decode_colors[4 * 3];
                    decode_colors[0] = r;
                    decode_colors[1] = g;
                    decode_colors[2] = b;
                    rgb_888_from_565(c1, &r, &g, &b);
                    decode_colors[3] = r;
                    decode_colors[4] = g;
                    decode_colors[5] = b;
                    //	Like DXT1, but no choicees:
                    //	no alpha, 2 interpolated colors
                    decode_colors[6] = (2 * decode_colors[0] + decode_colors[3]) / 3;
                    decode_colors[7] = (2 * decode_colors[1] + decode_colors[4]) / 3;
                    decode_colors[8] = (2 * decode_colors[2] + decode_colors[5]) / 3;
                    decode_colors[9] = (decode_colors[0] + 2 * decode_colors[3]) / 3;
                    decode_colors[10] = (decode_colors[1] + 2 * decode_colors[4]) / 3;
                    decode_colors[11] = (decode_colors[2] + 2 * decode_colors[5]) / 3;
                    //	decode the block
                    int next_bit = 4 * 8;
                    for (OutColorType &c : cache.decoded_block) {
                        const int idx = ((compressed_block[next_bit >> 3] >> (next_bit & 7)) & 3) * 3;
                        next_bit += 2;
                        c.v[0 % N] = decode_colors[idx + 0];
                        c.v[1 % N] = decode_colors[idx + 1];
                        c.v[2 % N] = decode_colors[idx + 2];
                    }
                }
            } else if (N == 3) {
                // find the 2 primary colors
                const int c0 = compressed_block[0] + (compressed_block[1] << 8);
                const int c1 = compressed_block[2] + (compressed_block[3] << 8);
                int r, g, b;
                rgb_888_from_565(c0, &r, &g, &b);
                uint8_t decode_colors[4 * 3];
                decode_colors[0] = r;
                decode_colors[1] = g;
                decode_colors[2] = b;
                rgb_888_from_565(c1, &r, &g, &b);
                decode_colors[3] = r;
                decode_colors[4] = g;
                decode_colors[5] = b;
                //	Like DXT1, but no choicees:
                //	no alpha, 2 interpolated colors
                decode_colors[6] = (2 * decode_colors[0] + decode_colors[3]) / 3;
                decode_colors[7] = (2 * decode_colors[1] + decode_colors[4]) / 3;
                decode_colors[8] = (2 * decode_colors[2] + decode_colors[5]) / 3;
                decode_colors[9] = (decode_colors[0] + 2 * decode_colors[3]) / 3;
                decode_colors[10] = (decode_colors[1] + 2 * decode_colors[4]) / 3;
                decode_colors[11] = (decode_colors[2] + 2 * decode_colors[5]) / 3;
                //	decode the block
                int next_bit = 4 * 8;
                for (OutColorType &c : cache.decoded_block) {
                    const int idx = ((compressed_block[next_bit >> 3] >> (next_bit & 7)) & 3) * 3;
                    next_bit += 2;
                    c.v[0] = decode_colors[idx + 0];
                    c.v[1] = decode_colors[idx + 1];
                    c.v[2] = decode_colors[idx + 2];
                }
            } else {
                for (int ch = 0; ch < N; ++ch) {
                    uint8_t decode_data[8];
                    decode_data[0] = compressed_block[0];
                    decode_data[1] = compressed_block[1];

                    if (decode_data[0] > decode_data[1]) {
                        // 6-step intermediate values
                        decode_data[2] = (6 * decode_data[0] + 1 * decode_data[1]) / 7;
                        decode_data[3] = (5 * decode_data[0] + 2 * decode_data[1]) / 7;
                        decode_data[4] = (4 * decode_data[0] + 3 * decode_data[1]) / 7;
                        decode_data[5] = (3 * decode_data[0] + 4 * decode_data[1]) / 7;
                        decode_data[6] = (2 * decode_data[0] + 5 * decode_data[1]) / 7;
                        decode_data[7] = (1 * decode_data[0] + 6 * decode_data[1]) / 7;
                    } else {
                        // 4-step intermediate values + full and none
                        decode_data[2] = (4 * decode_data[0] + 1 * decode_data[1]) / 5;
                        decode_data[3] = (3 * decode_data[0] + 2 * decode_data[1]) / 5;
                        decode_data[4] = (2 * decode_data[0] + 3 * decode_data[1]) / 5;
                        decode_data[5] = (1 * decode_data[0] + 4 * decode_data[1]) / 5;
                        decode_data[6] = 0;
                        decode_data[7] = 255;
                    }

                    int next_bit = 8 * 2;
                    for (OutColorType &c : cache.decoded_block) {
                        int idx = 0, bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 0;
                        ++next_bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 1;
                        ++next_bit;
                        bit = (compressed_block[next_bit >> 3] >> (next_bit & 7)) & 1;
                        idx += bit << 2;
                        ++next_bit;

                        c.v[ch] = decode_data[idx & 7];
                    }
                    compressed_block += BlockSize_BC4;
                }
            }

            cache.decoded_index = index;
            cache.decoded_block_offset = block_offset;
        }

        return cache.decoded_block[in_tiley * TileSize + in_tilex];
    }

    force_inline OutColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];
        const int h = p.res[lod][1];

        x -= std::floor(x);
        y -= std::floor(y);

        return Get(index, int(x * w - 0.5f), int(y * h - 0.5f), lod);
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0];
        res[1] = p.res[lod][1];
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0]);
        res[1] = float(p.res[lod][1]);
    }

    color_rgba_t Fetch(const int index, const int x, const int y, const int lod) const override {
        const OutColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    color_rgba_t Fetch(const int index, const float x, const float y, const int lod) const override {
        const OutColorType col = Get(index, x, y, lod);

        color_rgba_t ret;
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(col.v[i]);
        }
        for (int i = N; i < 4; ++i) {
            ret.v[i] = ret.v[N - 1];
        }

        ret.v[0] /= 255.0f;
        ret.v[1] /= 255.0f;
        ret.v[2] /= 255.0f;
        ret.v[3] /= 255.0f;

        return ret;
    }

    int Allocate(Span<const InColorType> data, const int res[2], bool mips);
    int AllocateRaw(Span<const uint8_t> data, const int res[2], int mips_count, bool flip_vertical, bool invert_green);
    bool Free(int index) override { return true; }
};

extern template class TexStorageBCn<1>;
extern template class TexStorageBCn<2>;
extern template class TexStorageBCn<3>;
extern template class TexStorageBCn<4>;

} // namespace Cpu
} // namespace Ray

#pragma warning(pop)