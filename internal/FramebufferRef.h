#pragma once

#include <cstddef>

#include <vector>

#include "../Types.h"

namespace ray {
namespace ref {
class Framebuffer {
    int w_, h_;
    std::vector<pixel_color_t> pixels_;
public:
    Framebuffer(int w, int h);

    int w() const {
        return w_;
    }
    int h() const {
        return h_;
    }

    void SetPixel(int x, int y, const pixel_color_t &p) {
        int i = y * w_ + x;
        pixels_[i] = p;
    }

    void GetPixel(int x, int y, pixel_color_t &out_p) {
        int i = y * w_ + x;
        out_p = pixels_[i];
    }

    void Resize(int w, int h);
    void Clear(const pixel_color_t &p);

    template <typename F>
    void Apply(const rect_t &reg, F &&f) {
        for (int y = reg.y; y < reg.y + reg.h; y++) {
            for (int x = reg.x; x < reg.x + reg.w; x++) {
                f(pixels_[y * w_ + x]);
            }
        }
    }

    const pixel_color_t *get_pixels_ref() const {
        return &pixels_[0];
    }
};
}
}