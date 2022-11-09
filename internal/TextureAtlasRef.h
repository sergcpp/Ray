#pragma once

#include <memory>

#include "Core.h"
#include "TextureSplitter.h"

namespace Ray {
namespace Ref {
class TextureAtlasBase {
  protected:
    const int res_[2];
    const float res_f_[2];

  public:
    TextureAtlasBase(int resx, int resy) : res_{resx, resy}, res_f_{float(resx), float(resy)} {}
    virtual ~TextureAtlasBase() {}

    force_inline float size_x() const { return res_f_[0]; }
    force_inline float size_y() const { return res_f_[1]; }

    virtual color_rgba_t Fetch(int page, int x, int y) const = 0;
    virtual color_rgba_t Fetch(int page, float x, float y) const = 0;
};

template <typename T, int N> class TextureAtlasLinear : public TextureAtlasBase {
    using ColorType = color_t<T, N>;
    using PageData = std::unique_ptr<ColorType[]>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData> pages_;
    std::vector<ColorType> temp_storage_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const ColorType *data);

  public:
    TextureAtlasLinear(int resx, int resy, int initial_page_count = 0);

    force_inline int page_count() const { return int(pages_.size()); }

    force_inline ColorType Get(const int page, const int x, const int y) const { return pages_[page][res_[0] * y + x]; }

    force_inline ColorType Get(const int page, const float x, const float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    color_rgba_t Fetch(int page, int x, int y) const override {
        const ColorType col = Get(page, x, y);

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

    color_rgba_t Fetch(int page, float x, float y) const override {
        const ColorType col = Get(page, x, y);

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

    int Allocate(const ColorType *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);
};

extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 4>;
extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 3>;
extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 2>;
extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 1>;

template <typename T, int N> class TextureAtlasTiled : public TextureAtlasBase {
    static const int TileSize = 4;

    const int res_in_tiles_[2];

    using ColorType = color_t<T, N>;
    using PageData = std::unique_ptr<ColorType[]>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData> pages_;
    std::unique_ptr<ColorType[]> temp_storage_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const ColorType *data);

  public:
    TextureAtlasTiled(int resx, int resy, int initial_page_count = 0);

    force_inline int page_count() const { return int(pages_.size()); }

    force_inline ColorType Get(const int page, const int x, const int y) const {
        const int tilex = x / TileSize, tiley = y / TileSize;
        const int in_tilex = x % TileSize, in_tiley = y % TileSize;

        return pages_[page][(tiley * res_in_tiles_[0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex];
    }

    force_inline ColorType Get(const int page, const float x, const float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    color_rgba_t Fetch(const int page, const int x, const int y) const override {
        const ColorType col = Get(page, x, y);

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

    color_rgba_t Fetch(const int page, const float x, const float y) const override {
        const ColorType col = Get(page, x, y);

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

    int Allocate(const ColorType *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);

    int DownsampleRegion(int src_page, const int src_pos[2], const int src_res[2], int dst_pos[2]);
};

extern template class TextureAtlasTiled<uint8_t, 4>;
extern template class TextureAtlasTiled<uint8_t, 3>;
extern template class TextureAtlasTiled<uint8_t, 2>;
extern template class TextureAtlasTiled<uint8_t, 1>;

template <typename T, int N> class TextureAtlasSwizzled : public TextureAtlasBase {
    using ColorType = color_t<T, N>;
    using PageData = std::unique_ptr<ColorType[]>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData> pages_;
    std::unique_ptr<ColorType[]> temp_storage_;

    uint32_t tile_y_stride_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const ColorType *data);

    static const uint32_t OuterTileW = 64;
    static const uint32_t OuterTileH = 64;

    force_inline static uint32_t swizzle_x_tile(uint32_t x) {
        return ((x & 0x03) << 0) | ((x & 0x04) << 2) | ((x & 0x38) << 4) | ((x & ~0x3f) << 6);
    }
    force_inline static uint32_t swizzle_y(uint32_t y) {
        return ((y & 0x03) << 2) | ((y & 0x0c) << 3) | ((y & 0x30) << 6);
    }

    force_inline uint32_t EncodeSwizzle(const uint32_t x, const uint32_t y) const {
        const uint32_t y_off = (y / OuterTileH) * tile_y_stride_ + swizzle_y(y);
        const uint32_t x_off = swizzle_x_tile(x);
        return y_off + x_off;
    }

  public:
    TextureAtlasSwizzled(int resx, int resy, int initial_page_count = 0);

    force_inline int page_count() const { return int(pages_.size()); }

    force_inline ColorType Get(const int page, const int x, const int y) const {
        return pages_[page][EncodeSwizzle(x, y)];
    }

    force_inline ColorType Get(const int page, const float x, const float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    color_rgba_t Fetch(const int page, const int x, const int y) const override {
        const ColorType col = Get(page, x, y);

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

    color_rgba_t Fetch(const int page, const float x, const float y) const override {
        const ColorType col = Get(page, x, y);

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

    int Allocate(const ColorType *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);

    int DownsampleRegion(int src_page, const int src_pos[2], const int src_res[2], int dst_pos[2]);
};

extern template class TextureAtlasSwizzled<uint8_t, 4>;
extern template class TextureAtlasSwizzled<uint8_t, 3>;
extern template class TextureAtlasSwizzled<uint8_t, 2>;
extern template class TextureAtlasSwizzled<uint8_t, 1>;

} // namespace Ref
} // namespace Ray
