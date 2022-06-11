#include "TextureAtlasRef.h"

#include <cstring>

#include <algorithm> // for std::max

template <typename T, int N>
Ray::Ref::TextureAtlasLinear<T, N>::TextureAtlasLinear(const int resx, const int resy, const int initial_page_count)
    : TextureAtlasBase(resx, resy) {
    if (!Resize(initial_page_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }

    // Allocate once
    temp_storage_.reserve(std::max(resx, resy));
}

template <typename T, int N>
int Ray::Ref::TextureAtlasLinear<T, N>::Allocate(const ColorType *data, const int _res[2], int pos[2]) {
    int res[2] = {_res[0] + 2, _res[1] + 2};

    if (res[0] > res_[0] || res[1] > res_[1]) {
        return -1;
    }

    for (int page_index = 0; page_index < page_count_; page_index++) {
        int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
        if (index != -1) {
            WritePageData(page_index, pos[0] + 1, pos[1] + 1, _res[0], _res[1], &data[0]);

            // add 1px border
            WritePageData(page_index, pos[0] + 1, pos[1], _res[0], 1, &data[(_res[1] - 1) * _res[0]]);

            WritePageData(page_index, pos[0] + 1, pos[1] + res[1] - 1, _res[0], 1, &data[0]);

            temp_storage_.resize(res[1]);
            PageData &vertical_border = temp_storage_;
            vertical_border[0] = data[(_res[1] - 1) * _res[0] + _res[0] - 1];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0] + _res[0] - 1];
            }
            vertical_border[res[1] - 1] = data[0 * _res[0] + _res[0] - 1];

            WritePageData(page_index, pos[0], pos[1], 1, res[1], &vertical_border[0]);

            vertical_border[0] = data[(_res[1] - 1) * _res[0]];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0]];
            }
            vertical_border[res[1] - 1] = data[0];

            WritePageData(page_index, pos[0] + res[0] - 1, pos[1], 1, res[1], &vertical_border[0]);

            return page_index;
        }
    }

    Resize(page_count_ + 1);
    return Allocate(data, _res, pos);
}

template <typename T, int N> bool Ray::Ref::TextureAtlasLinear<T, N>::Free(const int page, const int pos[2]) {
    if (page < 0 || page > page_count_)
        return false;
#ifndef NDEBUG // Fill region with zeros in debug
    int size[2];
    int index = splitters_[page].FindNode(&pos[0], &size[0]);
    if (index != -1) {
        for (int j = pos[1]; j < pos[1] + size[1]; j++) {
            memset(&pages_[page][j * res_[0] + pos[0]], 0, size[0] * sizeof(ColorType));
        }
        return splitters_[page].Free(index);
    } else {
        return false;
    }
#else
    return splitters_[page].Free(&pos[0]);
#endif
}

template <typename T, int N> bool Ray::Ref::TextureAtlasLinear<T, N>::Resize(const int new_page_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = new_page_count; i < page_count_; i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    pages_.resize(new_page_count);
    for (PageData &p : pages_) {
        p.resize(res_[0] * res_[1], ColorType{});
    }

    splitters_.resize(new_page_count, TextureSplitter{&res_[0]});
    page_count_ = new_page_count;

    return true;
}

template <typename T, int N>
void Ray::Ref::TextureAtlasLinear<T, N>::WritePageData(int page, int posx, int posy, int sizex, int sizey,
                                                       const ColorType *data) {
    for (int y = 0; y < sizey; y++) {
        memcpy(&pages_[page][(posy + y) * res_[0] + posx], &data[y * sizex], sizex * sizeof(ColorType));
    }
}

template class Ray::Ref::TextureAtlasLinear<uint8_t, 4>;
template class Ray::Ref::TextureAtlasLinear<uint8_t, 3>;
template class Ray::Ref::TextureAtlasLinear<uint8_t, 2>;
template class Ray::Ref::TextureAtlasLinear<uint8_t, 1>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
Ray::Ref::TextureAtlasTiled<T, N>::TextureAtlasTiled(int resx, int resy, int initial_page_count)
    : TextureAtlasBase(resx, resy), res_in_tiles_{resx / TileSize, resy / TileSize} {
    if ((resx % TileSize) || (resy % TileSize)) {
        throw std::invalid_argument("TextureAtlas resolution should be multiple of tile size!");
    }

    if (!Resize(initial_page_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }

    // Allocate once
    temp_storage_.reserve(std::max(resx, resy));
}

template <typename T, int N>
int Ray::Ref::TextureAtlasTiled<T, N>::Allocate(const ColorType *data, const int _res[2], int pos[2]) {
    const int res[2] = {_res[0] + 2, _res[1] + 2};

    if (res[0] > res_[0] || res[1] > res_[1]) {
        return -1;
    }

    for (int page_index = 0; page_index < page_count_; page_index++) {
        const int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
        if (index != -1) {
            WritePageData(page_index, pos[0] + 1, pos[1] + 1, _res[0], _res[1], &data[0]);

            // add 1px border
            WritePageData(page_index, pos[0] + 1, pos[1], _res[0], 1, &data[(_res[1] - 1) * _res[0]]);

            WritePageData(page_index, pos[0] + 1, pos[1] + res[1] - 1, _res[0], 1, &data[0]);

            temp_storage_.resize(res[1]);
            PageData &vertical_border = temp_storage_;
            vertical_border[0] = data[(_res[1] - 1) * _res[0] + _res[0] - 1];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0] + _res[0] - 1];
            }
            vertical_border[res[1] - 1] = data[0 * _res[0] + _res[0] - 1];

            WritePageData(page_index, pos[0], pos[1], 1, res[1], &vertical_border[0]);

            vertical_border[0] = data[(_res[1] - 1) * _res[0]];
            for (int i = 0; i < _res[1]; i++) {
                vertical_border[i + 1] = data[i * _res[0]];
            }
            vertical_border[res[1] - 1] = data[0];

            WritePageData(page_index, pos[0] + res[0] - 1, pos[1], 1, res[1], &vertical_border[0]);

            return page_index;
        }
    }

    Resize(page_count_ + 1);
    return Allocate(data, _res, pos);
}

template <typename T, int N> bool Ray::Ref::TextureAtlasTiled<T, N>::Free(int page, const int pos[2]) {
    if (page < 0 || page > page_count_) {
        return false;
    }
#ifndef NDEBUG // Fill region with zeros in debug
    int size[2];
    const int index = splitters_[page].FindNode(&pos[0], &size[0]);
    if (index != -1) {
        for (int j = pos[1]; j < pos[1] + size[1]; j++) {
            memset(&pages_[page][j * res_[0] + pos[0]], 0, size[0] * sizeof(ColorType));
        }
        return splitters_[page].Free(index);
    } else {
        return false;
    }
#else
    return splitters_[page].Free(&pos[0]);
#endif
}

template <typename T, int N> bool Ray::Ref::TextureAtlasTiled<T, N>::Resize(int new_page_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = new_page_count; i < page_count_; i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    pages_.resize(new_page_count);
    for (PageData &p : pages_) {
        p.resize(res_[0] * res_[1], ColorType{});
    }

    splitters_.resize(new_page_count, TextureSplitter{&res_[0]});
    page_count_ = new_page_count;

    return true;
}

template <typename T, int N>
void Ray::Ref::TextureAtlasTiled<T, N>::WritePageData(const int page, const int posx, const int posy, const int sizex,
                                                      const int sizey, const ColorType *data) {
    for (int y = 0; y < sizey; y++) {
        const int tiley = (posy + y) / TileSize, in_tiley = (posy + y) % TileSize;

        for (int x = 0; x < sizex; x++) {
            const int tilex = (posx + x) / TileSize, in_tilex = (posx + x) % TileSize;

            pages_[page][(tiley * res_in_tiles_[0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                data[y * sizex + x];
        }
    }
}

template class Ray::Ref::TextureAtlasTiled<uint8_t, 4>;
template class Ray::Ref::TextureAtlasTiled<uint8_t, 3>;
template class Ray::Ref::TextureAtlasTiled<uint8_t, 2>;
template class Ray::Ref::TextureAtlasTiled<uint8_t, 1>;