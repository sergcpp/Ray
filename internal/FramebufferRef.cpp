#include "FramebufferRef.h"

#include <cassert>
#include <cstring>

Ray::Ref::Framebuffer::Framebuffer(int w, int h) {
    Resize(w, h, false);
}

void Ray::Ref::Framebuffer::Resize(int w, int h, bool alloc_sh) {
    assert(w > 0 && h > 0);

    if (alloc_sh && sh_data_.empty() || w_ != w || h_ != h) {
        size_t buf_size = (size_t)w * h;
        sh_data_.resize(buf_size, { 0 });
    }

    if (w_ != w || h_ != h) {
        w_ = w;
        h_ = h;
        size_t buf_size = (size_t)w * h;
        pixels_.resize(buf_size, pixel_color_t{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
}

void Ray::Ref::Framebuffer::Clear(const pixel_color_t &p) {
    for (int i = 0; i < w_; i++) {
        pixels_[i] = p;
    }
    for (int i = 1; i < h_; i++) {
        memcpy(&pixels_[i * w_], &pixels_[0], w_ * sizeof(pixel_color_t));
    }
}

void Ray::Ref::Framebuffer::ComputeSHData(const rect_t &rect) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            int i = y * w_ + x;

            const auto &p = pixels_[i];
            auto &sh_data = sh_data_[i];
            const float *sh_coeff = sh_data.coeff_r;

            for (int j = 0; j < 4; j++) {
                sh_data.coeff_g[j] = sh_coeff[j] * p.g;
                sh_data.coeff_b[j] = sh_coeff[j] * p.b;
                sh_data.coeff_r[j] = sh_coeff[j] * p.r;
            }
        }
    }
}

void Ray::Ref::Framebuffer::MixWith(const Framebuffer &f2, const rect_t &rect, float k) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            this->MixPixel(x, y, f2.GetPixel(x, y), k);
        }
    }
}

void Ray::Ref::Framebuffer::MixWith_SH(const Framebuffer &f2, const rect_t &rect, float k) {
    const float weight = 4.0f * PI * k;

    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            int i = y * w_ + x;

            const auto &in_sh = f2.sh_data_[i];
            auto &out_sh = sh_data_[i];

            for (int j = 0; j < 4; j++) {
                out_sh.coeff_r[j] += (in_sh.coeff_r[j] * weight);
                out_sh.coeff_g[j] += (in_sh.coeff_g[j] * weight);
                out_sh.coeff_b[j] += (in_sh.coeff_b[j] * weight);
            }
        }
    }
}
