#pragma once

#include <cmath>
#include <memory>

#include "Core.h"
#include "TextureSplitter.h"

#define _MAX(x, y) ((x) < (y) ? (y) : (x))

namespace Ray {
namespace Ref {
class TexStorageBase {
  public:
    virtual ~TexStorageBase() {}

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

    force_inline ColorType Get(const int index, const int x, const int y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0];
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

        res[0] = p.res[lod][0] - 1;
        res[1] = p.res[lod][1] - 1;
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0] - 1);
        res[1] = float(p.res[lod][1] - 1);
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

    int Allocate(const ColorType *data, const int res[2], bool mips);
    bool Free(int index) override final;
};

extern template class Ray::Ref::TexStorageLinear<uint8_t, 4>;
extern template class Ray::Ref::TexStorageLinear<uint8_t, 3>;
extern template class Ray::Ref::TexStorageLinear<uint8_t, 2>;
extern template class Ray::Ref::TexStorageLinear<uint8_t, 1>;

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

    force_inline ColorType Get(const int index, const int x, const int y, const int lod) const {
        const ImgData &p = images_[index];

        const int tilex = x / TileSize, tiley = y / TileSize;
        const int in_tilex = x % TileSize, in_tiley = y % TileSize;

        const int w_in_tiles = p.res_in_tiles[lod][0];
        return p.pixels[p.lod_offsets[lod] + (tiley * w_in_tiles + tilex) * TileSize * TileSize + in_tiley * TileSize +
                        in_tilex];
    }

    force_inline ColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0] - 1;
        const int h = p.res[lod][1] - 1;

        x -= std::floor(x);
        y -= std::floor(y);

        return Get(index, int(x * w - 0.5f), int(y * h - 0.5f), lod);
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0] - 1;
        res[1] = p.res[lod][1] - 1;
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0] - 1);
        res[1] = float(p.res[lod][1] - 1);
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

    int Allocate(const ColorType *data, const int res[2], bool mips);
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

    force_inline ColorType Get(const int index, const int x, const int y, const int lod) const {
        const ImgData &p = images_[index];
        return p.pixels[p.lod_offsets[lod] + EncodeSwizzle(x, y, p.tile_y_stride[lod])];
    }

    force_inline ColorType Get(const int index, float x, float y, const int lod) const {
        const ImgData &p = images_[index];
        const int w = p.res[lod][0] - 1;
        const int h = p.res[lod][1] - 1;

        x -= std::floor(x);
        y -= std::floor(y);

        return Get(index, int(x * w - 0.5f), int(y * h - 0.5f), lod);
    }

    void GetIRes(const int index, const int lod, int res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = p.res[lod][0] - 1;
        res[1] = p.res[lod][1] - 1;
    }

    void GetFRes(const int index, const int lod, float res[2]) const override {
        const ImgData &p = images_[index];

        res[0] = float(p.res[lod][0] - 1);
        res[1] = float(p.res[lod][1] - 1);
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

    int Allocate(const ColorType *data, const int res[2], bool mips);
    bool Free(int index) override;
};

extern template class TexStorageSwizzled<uint8_t, 4>;
extern template class TexStorageSwizzled<uint8_t, 3>;
extern template class TexStorageSwizzled<uint8_t, 2>;
extern template class TexStorageSwizzled<uint8_t, 1>;

} // namespace Ref
} // namespace Ray

#undef _MAX
