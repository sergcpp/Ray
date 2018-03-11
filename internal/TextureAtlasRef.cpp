#include "TextureAtlasRef.h"

ray::ref::TextureAtlas::TextureAtlas(const math::ivec2 &res, int pages_count) : res_(res), res_f_{ (float)res.x, (float)res.y }, pages_count_(0) {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

int ray::ref::TextureAtlas::Allocate(const pixel_color8_t *data, const math::ivec2 &_res, math::ivec2 &pos) {
    using namespace math;

    ivec2 res = _res + ivec2{ 2, 2 };

    if (res.x > res_.x || res.y > res_.y) return -1;

    auto write_page = [_res, res, this](std::vector<pixel_color8_t> &page, const ivec2 &pos, const ivec2 &size, const pixel_color8_t *data) {
        for (int y = 0; y < size.y; y++) {
            memcpy(&page[(pos.y + y) * res_.x + pos.x], &data[y * size.x], size.x * sizeof(pixel_color8_t));
        }
    };

    for (int page_index = 0; page_index < pages_count_; page_index++) {
        int index = splitters_[page_index].Allocate(res, pos);
        if (index != -1) {
            auto &page = pages_[page_index];

            write_page(page, { pos.x + 1, pos.y + 1 }, _res, &data[0]);

            // add 1px border
            write_page(page, { pos.x + 1, pos.y }, { _res.x, 1 }, &data[(_res.y - 1) * _res.x]);

            write_page(page, { pos.x + 1, pos.y + res.y - 1 }, { _res.x, 1 }, &data[0]);
            
            std::vector<pixel_color8_t> vertical_border(res.y);
            vertical_border[0] = data[(_res.y - 1) * _res.x + _res.y - 1];
            for (int i = 0; i < _res.y; i++) {
                vertical_border[i + 1] = data[i * _res.x + _res.y - 1];
            }
            vertical_border[res.y - 1] = data[0 * _res.x + _res.y - 1];

            write_page(page, pos, { 1, res.y }, &vertical_border[0]);

            vertical_border[0] = data[(_res.y - 1) * _res.x];
            for (int i = 0; i < _res.y; i++) {
                vertical_border[i + 1] = data[i * _res.x];
            }
            vertical_border[res.y - 1] = data[0];

            write_page(page, { pos.x + res.x - 1, pos.y }, { 1, res.y }, &vertical_border[0]);
            
            return page_index;
        }
    }

    Resize(pages_count_ * 2);
    return Allocate(data, _res, pos);
}

bool ray::ref::TextureAtlas::Free(int page, const math::ivec2 &pos) {
    if (page < 0 || page > pages_count_) return false;
#ifndef NDEBUG
    math::ivec2 size;
    int index = splitters_[page].FindNode(pos, size);
    if (index != -1) {
        for (int j = pos.y; j < pos.y + size.y; j++) {
            memset(&pages_[page][j * res_.x + pos.x], 0, size.x * sizeof(pixel_color8_t));
        }
        return splitters_[page].Free(index);
    } else {
        return false;
    }
#else
    return splitters_[page].Free(pos);
#endif
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

math::vec4 ray::ref::TextureAtlas::SampleNearest(const texture_t &t, const math::vec2 &uvs, float lod) const {
    using namespace math;

    int _lod = (int)floor(lod);

    float _uvs[2];
    TransformUVs(value_ptr(uvs), value_ptr(res_f_), &t, _lod, _uvs);

    _lod = min(_lod, MAX_MIP_LEVEL);

    int page = t.page[_lod];

    const auto &pix = pages_[page][res_.x * int(_uvs[1] * res_.y - 0.5f) + int(_uvs[0] * res_.x - 0.5f)];

    const float k = 1.0f / 255.0f;

    return vec4{ pix.r * k, pix.g * k, pix.b * k, pix.a * k };
}

math::vec4 ray::ref::TextureAtlas::SampleBilinear(const texture_t &t, const math::vec2 &uvs, int lod) const {
    using namespace math;

    float _uvs[2];
    TransformUVs(value_ptr(uvs), value_ptr(res_f_), &t, lod, _uvs);

    int page = t.page[lod];

    _uvs[0] = _uvs[0] * res_.x - 0.5f;
    _uvs[1] = _uvs[1] * res_.y - 0.5f;

    const auto &p00 = pages_[page][res_.x * int(_uvs[1]) + int(_uvs[0])];
    const auto &p01 = pages_[page][res_.x * int(_uvs[1]) + int(_uvs[0]) + 1];
    const auto &p10 = pages_[page][res_.x * (int(_uvs[1]) + 1) + int(_uvs[0])];
    const auto &p11 = pages_[page][res_.x * (int(_uvs[1]) + 1) + int(_uvs[0]) + 1];

    float kx = fract(_uvs[0]), ky = fract(_uvs[1]);

    const auto p0 = pixel_color_t{ p01.r * kx + p00.r * (1 - kx),
                                   p01.g * kx + p00.g * (1 - kx),
                                   p01.b * kx + p00.b * (1 - kx),
                                   p01.a * kx + p00.a * (1 - kx) };

    const auto p1 = pixel_color_t{ p11.r * kx + p10.r * (1 - kx),
                                   p11.g * kx + p10.g * (1 - kx),
                                   p11.b * kx + p10.b * (1 - kx),
                                   p11.a * kx + p10.a * (1 - kx) };

    const float k = 1.0f / 255.0f;
    return vec4{ k * (p1.r * ky + p0.r * (1 - ky)),
                 k * (p1.g * ky + p0.g * (1 - ky)),
                 k * (p1.b * ky + p0.b * (1 - ky)),
                 k * (p1.a * ky + p0.a * (1 - ky)) };
}

math::vec4 ray::ref::TextureAtlas::SampleTrilinear(const texture_t &t, const math::vec2 &uvs, float lod) const {
    using namespace math;
    
    auto col1 = SampleBilinear(t, uvs, (int)floor(lod));
    auto col2 = SampleBilinear(t, uvs, (int)ceil(lod));

    return mix(col1, col2, lod - floor(lod));
}

math::vec4 ray::ref::TextureAtlas::SampleAnisotropic(const texture_t &t, const math::vec2 &uvs,
                                                     const math::vec2 &duv_dx, const math::vec2 &duv_dy) const {
    using namespace math;

    auto sz = vec2{ (float)t.size[0], (float)t.size[1] };

    float l1 = length(duv_dx * sz);
    float l2 = length(duv_dy * sz);

    float lod;
    float k;
    vec2 step{ Uninitialize };

    if (l1 <= l2) {
        lod = log2(l1);
        k = l1 / l2;
        step = duv_dy;
    } else {
        lod = log2(l2);
        k = l2 / l1;
        step = duv_dx;
    }

    lod = clamp(lod, 0.0f, (float)MAX_MIP_LEVEL);

    vec2 _uvs = uvs - step * 0.5f;

    int num = clamp((int)(2.0f / k), 1, 32);
    step = step / float(num);

    vec4 res;

    for (int i = 0; i < num; i++) {
        res += SampleTrilinear(t, _uvs, lod);
        _uvs += step;
    }

    return res / float(num);
}