#include "FramebufferRef.h"

#include <cassert>
#include <cstring>

Ray::Ref::Framebuffer::Framebuffer(const int w, const int h) : w_(0), h_(0) { Resize(w, h, 0); }

void Ray::Ref::Framebuffer::Resize(const int w, const int h, const uint32_t aux_buffers) {
    assert(w > 0 && h > 0);

    if (aux_buffers & eAUXBuffer::SHL1) {
        sh_data_.resize(w * h, {});
        sh_data_.shrink_to_fit();
    } else {
        sh_data_ = {};
    }

    if (aux_buffers & eAUXBuffer::BaseColor) {
        base_color_.resize(w * h, {});
        base_color_.shrink_to_fit();
    } else {
        base_color_ = {};
    }

    if (aux_buffers & eAUXBuffer::DepthNormals) {
        depth_normals_.resize(w * h, {});
        depth_normals_.shrink_to_fit();
    } else {
        depth_normals_ = {};
    }

    w_ = w;
    h_ = h;

    pixels_.resize(w * h, {});
    pixels_.shrink_to_fit();
}

void Ray::Ref::Framebuffer::Clear(const color_rgba_t &p) {
    for (int i = 0; i < w_; i++) {
        pixels_[i] = p;
    }
    for (int i = 1; i < h_; i++) {
        memcpy(&pixels_[i * w_], &pixels_[0], w_ * sizeof(color_rgba_t));
    }

    if (!sh_data_.empty()) {
        for (int i = 0; i < w_; i++) {
            sh_data_[i] = {};
        }
        for (int i = 1; i < h_; i++) {
            memcpy(&sh_data_[i * w_], &sh_data_[0], w_ * sizeof(shl1_data_t));
        }
    }
}

void Ray::Ref::Framebuffer::ResetSampleData(const rect_t &rect) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            int i = y * w_ + x;
            sh_data_[i].coeff_g[0] = 0.0f;
        }
    }
}

void Ray::Ref::Framebuffer::ComputeSHData(const rect_t &rect) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            int i = y * w_ + x;

            shl1_data_t &sh_data = sh_data_[i];
            const float *sh_coeff = sh_data.coeff_r;
            const float inv_weight = sh_data.coeff_g[0] > FLT_EPS ? (2.0f * PI / sh_data.coeff_g[0]) : 0.0f;

            color_rgba_t p = pixels_[i];
            p.v[0] *= inv_weight;
            p.v[1] *= inv_weight;
            p.v[2] *= inv_weight;

            for (int j = 0; j < 4; j++) {
                sh_data.coeff_g[j] = sh_coeff[j] * p.v[1];
                sh_data.coeff_b[j] = sh_coeff[j] * p.v[2];
                sh_data.coeff_r[j] = sh_coeff[j] * p.v[0];
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
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            this->MixSHData(x, y, f2.GetSHData(x, y), k);
        }
    }
}

void Ray::Ref::Framebuffer::MixWith_BaseColor(const Framebuffer &f2, const rect_t &rect, float k) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            this->MixBaseColor(x, y, f2.GetBaseColor(x, y), k);
        }
    }
}

void Ray::Ref::Framebuffer::MixWith_DepthNormal(const Framebuffer& f2, const rect_t& rect, float k) {
    for (int y = rect.y; y < rect.y + rect.h; y++) {
        for (int x = rect.x; x < rect.x + rect.w; x++) {
            this->MixDepthNormal(x, y, f2.GetDepthNormal(x, y), k);
        }
    }
}
