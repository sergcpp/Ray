#pragma once

#include "Core.h"
#include "TextureSplitter.h"

namespace Ray {
namespace Ref {
template <typename T, int N>
class TextureAtlasLinear {
    const int res_[2];
    const float res_f_[2];
    int page_count_;

    using ColorType = color_t<T, N>;
    using PageData = std::vector<ColorType>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData> pages_;
    std::vector<ColorType> temp_storage_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const ColorType *data);

  public:
    TextureAtlasLinear(int resx, int resy, int initial_page_count = 0);

    force_inline float size_x() const { return res_f_[0]; }
    force_inline float size_y() const { return res_f_[1]; }

    force_inline int page_count() const { return int(pages_.size()); }

    force_inline ColorType Get(const int page, const int x, const int y) const {
        return pages_[page][res_[0] * y + x];
    }

    force_inline ColorType Get(const int page, const float x, const float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    int Allocate(const ColorType *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);
};

extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 4>;
extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 3>;
extern template class Ray::Ref::TextureAtlasLinear<uint8_t, 1>;

template <typename T, int N> class TextureAtlasTiled {
    static const int TileSize = 4;

    const int res_[2], res_in_tiles_[2];
    const float res_f_[2];
    int page_count_;

    using ColorType = color_t<T, N>;
    using PageData = std::vector<ColorType>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData> pages_;
    std::vector<ColorType> temp_storage_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const ColorType *data);

  public:
    TextureAtlasTiled(int resx, int resy, int initial_page_count = 0);

    force_inline float size_x() const { return res_f_[0]; }
    force_inline float size_y() const { return res_f_[1]; }

    force_inline int page_count() const { return int(pages_.size()); }

    force_inline ColorType Get(const int page, const int x, const int y) const {
        const int tilex = x / TileSize, tiley = y / TileSize;
        const int in_tilex = x % TileSize, in_tiley = y % TileSize;

        return pages_[page][(tiley * res_in_tiles_[0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex];
    }

    force_inline ColorType Get(const int page, const float x, const float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    int Allocate(const ColorType *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);
};

extern template class Ray::Ref::TextureAtlasTiled<uint8_t, 4>;
extern template class Ray::Ref::TextureAtlasTiled<uint8_t, 3>;
extern template class Ray::Ref::TextureAtlasTiled<uint8_t, 1>;

} // namespace Ref
} // namespace Ray
