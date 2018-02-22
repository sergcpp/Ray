#include "TextureAtlasRef.h"

ray::ref::TextureAtlas::TextureAtlas(const math::ivec2 &res, int pages_count) : res_(res), pages_count_(0) {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

int ray::ref::TextureAtlas::Allocate(const pixel_color8_t *data, const math::ivec2 &_res, math::ivec2 &pos) {
    // TODO: add 1px border
    math::ivec2 res = _res;// +math::ivec2{ 2, 2 };

    if (res.x > res_.x || res.y > res_.y) return -1;

    for (int page_index = 0; page_index < pages_count_; page_index++) {
        int index = splitters_[page_index].Allocate(res, pos);
        if (index != -1) {
            auto &page = pages_[page_index];

            for (int y = 0; y < res.y; y++) {
                memcpy(&page[(pos.y + y) * res_.x + pos.x], &data[y * res.x], res.x * sizeof(pixel_color8_t));
            }

            return page_index;
        }
    }

    Resize(pages_count_ * 2);
    return Allocate(data, _res, pos);
}

bool ray::ref::TextureAtlas::Free(int page, const math::ivec2 &pos) {
    if (page < 0 || page > pages_count_) return false;
    // TODO: fill with black in debug
    return splitters_[page].Free(pos);
}

bool ray::ref::TextureAtlas::Resize(int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < pages_count_; i++) {
        if (!splitters_[i].empty()) return false;
    }

    pages_.resize(pages_count);
    for (auto &p : pages_) {
        p.resize(res_.x * res_.y, { 0, 0, 0, 0 });
    }

    splitters_.resize(pages_count, TextureSplitter{ res_ });

    pages_count_ = pages_count;

    return true;
}