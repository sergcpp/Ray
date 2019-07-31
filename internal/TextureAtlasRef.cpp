#include "TextureAtlasRef.h"

#include <cstring>

Ray::Ref::TextureAtlas::TextureAtlas(int resx, int resy, int pages_count) : res_{ resx, resy }, res_f_{ (float)resx, (float)resy }, pages_count_(0) {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

int Ray::Ref::TextureAtlas::Allocate(const pixel_color8_t *data, const int _res[2], int pos[2]) {
    int res[2] = { _res[0] + 2, _res[1] + 2 };

    if (res[0] > res_[0] || res[1] > res_[1]) return -1;

    auto write_page = [_res, &res, this](std::vector<pixel_color8_t> &page, int posx, int posy, int sizex, int sizey, const pixel_color8_t *data) {
        for (int y = 0; y < sizey; y++) {
            memcpy(&page[(posy + y) * res_[0] + posx], &data[y * sizex], sizex * sizeof(pixel_color8_t));
        }
    };

    for (int page_index = 0; page_index < pages_count_; page_index++) {
        int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
        if (index != -1) {
            Page &page = pages_[page_index];

            write_page(page, pos[0] + 1, pos[1] + 1, _res[0], _res[1], &data[0]);

            // add 1px border
            write_page(page, pos[0] + 1, pos[1], _res[0], 1, &data[(_res[1] - 1) * _res[0]]);

            write_page(page, pos[0] + 1, pos[1] + res[1] - 1, _res[0], 1, &data[0]);

            std::vector<pixel_color8_t> vertical_border(res[1]);
            vertical_border[0] = data[(_res[1] - 1) * _res[0] + _res[0] - 1];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0] + _res[0] - 1];
            }
            vertical_border[res[1] - 1] = data[0 * _res[0] + _res[0] - 1];

            write_page(page, pos[0], pos[1], 1, res[1], &vertical_border[0]);

            vertical_border[0] = data[(_res[1] - 1) * _res[0]];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0]];
            }
            vertical_border[res[1] - 1] = data[0];

            write_page(page, pos[0] + res[0] - 1, pos[1], 1, res[1], &vertical_border[0]);

            return page_index;
        }
    }

    Resize(pages_count_ * 2);
    return Allocate(data, _res, pos);
}

bool Ray::Ref::TextureAtlas::Free(int page, const int pos[2]) {
    if (page < 0 || page > pages_count_) return false;
#ifndef NDEBUG
    int size[2];
    int index = splitters_[page].FindNode(&pos[0], &size[0]);
    if (index != -1) {
        for (int j = pos[1]; j < pos[1] + size[1]; j++) {
            memset(&pages_[page][j * res_[0] + pos[0]], 0, size[0] * sizeof(pixel_color8_t));
        }
        return splitters_[page].Free(index);
    } else {
        return false;
    }
#else
    return splitters_[page].Free(&pos[0]);
#endif
}

bool Ray::Ref::TextureAtlas::Resize(int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < pages_count_; i++) {
        if (!splitters_[i].empty()) return false;
    }

    pages_.resize(pages_count);
    for (Page &p : pages_) {
        p.resize(res_[0] * res_[1], { 0, 0, 0, 0 });
    }

    splitters_.resize(pages_count, TextureSplitter{ &res_[0] });

    pages_count_ = pages_count;

    return true;
}
