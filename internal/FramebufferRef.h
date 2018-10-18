#pragma once

#include <cstddef>

#include <vector>

#include "Core.h"
#include "../Types.h"

namespace Ray {
namespace Ref {
class Framebuffer {
    int w_, h_;
    std::vector<pixel_color_t> pixels_;
    std::vector<SHL1_data> sh_data_;
public:
    Framebuffer(int w, int h);

    force_inline int w() const {
        return w_;
    }

    force_inline int h() const {
        return h_;
    }

    force_inline void SetPixel(int x, int y, const pixel_color_t &p) {
        int i = y * w_ + x;
        pixels_[i] = p;
    }

    force_inline pixel_color_t GetPixel(int x, int y) const {
        int i = y * w_ + x;
        return pixels_[i];
    }

    force_inline void AddPixel(int x, int y, const pixel_color_t &p) {
        int i = y * w_ + x;
        pixels_[i].r += p.r;
        pixels_[i].g += p.g;
        pixels_[i].b += p.b;
        pixels_[i].a += p.a;
    }

    force_inline void MixPixel(int x, int y, const pixel_color_t &p, float k) {
        int i = y * w_ + x;
        pixels_[i].r += (p.r - pixels_[i].r) * k;
        pixels_[i].g += (p.g - pixels_[i].g) * k;
        pixels_[i].b += (p.b - pixels_[i].b) * k;
        pixels_[i].a += (p.a - pixels_[i].a) * k;
    }

    force_inline void SetPixelDir(int x, int y, const float v[3]) {
        int i = y * w_ + x;

        const float SH_Y0 = 0.282094806f; // sqrt(1.0f / (4.0f * PI))
        const float SH_Y1 = 0.488602519f; // sqrt(3.0f / (4.0f * PI))

        // temporary store sh coefficients in place of red channel
        sh_data_[i].coeff_r[0] = SH_Y0;
        sh_data_[i].coeff_r[1] = SH_Y1 * v[1];
        sh_data_[i].coeff_r[2] = SH_Y1 * v[2];
        sh_data_[i].coeff_r[3] = SH_Y1 * v[0];
    }

    void Resize(int w, int h, bool alloc_sh);
    void Clear(const pixel_color_t &p);

    void ComputeSHData(const rect_t &rect);

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

    const SHL1_data *get_sh_data_ref() const {
        return &sh_data_[0];
    }

    void MixWith(const Framebuffer &f2, const rect_t &rect, float k);
    void MixWith_SH(const Framebuffer &f2, const rect_t &rect, float k);

    template <typename F>
    void CopyFrom(const Framebuffer &f2, const rect_t &rect, F &&filter) {
        for (int y = rect.y; y < rect.y + rect.h; y++) {
            for (int x = rect.x; x < rect.x + rect.w; x++) {
                this->SetPixel(x, y, filter(f2.GetPixel(x, y)));
            }
        }
    }
};
}
}