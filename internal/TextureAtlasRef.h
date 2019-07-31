#pragma once

#include "Core.h"
#include "TextureSplitter.h"

namespace Ray {
namespace Ref {
class TextureAtlas {
    const int   res_[2];
    const float res_f_[2];
    int         page_count_;

    using PageData = std::vector<pixel_color8_t>;

    std::vector<TextureSplitter> splitters_;
    std::vector<PageData>        pages_;
    std::vector<pixel_color8_t>  temp_storage_;
public:
    TextureAtlas(int resx, int resy, int initial_page_count = 4);

    force_inline float size_x() const { return res_f_[0]; }
    force_inline float size_y() const { return res_f_[1]; }

    force_inline pixel_color8_t Get(int page, int x, int y) const {
        return pages_[page][res_[0] * y + x];
    }

    force_inline pixel_color8_t Get(int page, float x, float y) const {
        return Get(page, int(x * res_[0] - 0.5f), int(y * res_[1] - 0.5f));
    }

    int Allocate(const pixel_color8_t *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int new_page_count);
};
}
}
