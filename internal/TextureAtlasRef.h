#pragma once

#include "CoreRef.h"
#include "TextureSplitter.h"

namespace ray {
namespace ref {
class Renderer;

class TextureAtlas {
    friend class ref::Renderer;

    const math::ivec2 res_;
    const math::vec2 res_f_;
    int pages_count_;

    using Page = std::vector<pixel_color8_t>;

    std::vector<TextureSplitter> splitters_;
    std::vector<Page> pages_;
public:
    TextureAtlas(const math::ivec2 &res, int pages_count = 4);

    int Allocate(const pixel_color8_t *data, const math::ivec2 &res, math::ivec2 &pos);
    bool Free(int page, const math::ivec2 &pos);

    bool Resize(int pages_count);

    pixel_color_t SampleNearest(const texture_t &t, const math::vec2 &uvs, float lod) const;
    pixel_color_t SampleBilinear(const texture_t &t, const math::vec2 &uvs, float lod) const;
};
}
}