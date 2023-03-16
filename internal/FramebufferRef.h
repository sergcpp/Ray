#pragma once

#include <cstddef>

#include <vector>

#include "../Types.h"
#include "CoreRef.h"

namespace Ray {
namespace Ref {
class Framebuffer {
    int w_, h_;
    std::vector<color_rgba_t> pixels_;
    std::vector<shl1_data_t> sh_data_;
    std::vector<color_rgba_t> base_color_;
    std::vector<color_rgba_t> depth_normals_;

  public:
    Framebuffer(int w, int h);

    force_inline int w() const { return w_; }
    force_inline int h() const { return h_; }

    force_inline void SetPixel(const int x, const int y, const color_rgba_t &p) {
        const int i = y * w_ + x;
        pixels_[i] = p;
    }

    force_inline color_rgba_t GetPixel(const int x, const int y) const {
        const int i = y * w_ + x;
        return pixels_[i];
    }

    force_inline void SetBaseColor(const int x, const int y, const color_rgba_t &c) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { base_color_[i].v[j] = c.v[j]; })
    }

    force_inline void SetDepthNormal(const int x, const int y, const color_rgba_t &n) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { depth_normals_[i].v[j] = n.v[j]; })
    }

    force_inline color_rgba_t GetBaseColor(const int x, const int y) const {
        const int i = y * w_ + x;
        return base_color_[i];
    }

    force_inline color_rgba_t GetDepthNormal(const int x, const int y) const {
        const int i = y * w_ + x;
        return depth_normals_[i];
    }

    force_inline const shl1_data_t &GetSHData(const int x, const int y) const {
        const int i = y * w_ + x;
        return sh_data_[i];
    }

    force_inline void AddPixel(const int x, const int y, const color_rgba_t &p) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { pixels_[i].v[j] += p.v[j]; })
    }

    force_inline void MixPixel(const int x, const int y, const color_rgba_t &p, const float k) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { pixels_[i].v[j] += (p.v[j] - pixels_[i].v[j]) * k; })
    }

    force_inline void MixBaseColor(const int x, const int y, const color_rgba_t &c, const float k) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { base_color_[i].v[j] += (c.v[j] - base_color_[i].v[j]) * k; })
    }

    force_inline void MixDepthNormal(const int x, const int y, const color_rgba_t &n, const float k) {
        const int i = y * w_ + x;
        UNROLLED_FOR(j, 4, { depth_normals_[i].v[j] += (n.v[j] - depth_normals_[i].v[j]) * k; })
    }

    force_inline void MixSHData(const int x, const int y, const shl1_data_t &sh, const float k) {
        const int i = y * w_ + x;
        shl1_data_t &out_sh = sh_data_[i];

        UNROLLED_FOR(j, 4, {
            out_sh.coeff_r[j] += (sh.coeff_r[j] - out_sh.coeff_r[j]) * k;
            out_sh.coeff_g[j] += (sh.coeff_g[j] - out_sh.coeff_g[j]) * k;
            out_sh.coeff_b[j] += (sh.coeff_b[j] - out_sh.coeff_b[j]) * k;
        })
    }

    force_inline void SetSampleDir(const int x, const int y, const float dir_x, const float dir_y, const float dir_z) {
        const int i = y * w_ + x;

        static const float SH_Y0 = 0.282094806f; // sqrt(1.0f / (4.0f * PI))
        static const float SH_Y1 = 0.488602519f; // sqrt(3.0f / (4.0f * PI))

        // temporary store sh coefficients in place of red channel
        sh_data_[i].coeff_r[0] = SH_Y0;
        sh_data_[i].coeff_r[1] = SH_Y1 * dir_y;
        sh_data_[i].coeff_r[2] = SH_Y1 * dir_z;
        sh_data_[i].coeff_r[3] = SH_Y1 * dir_x;
    }

    force_inline void SetSampleWeight(const int x, const int y, const float weight) {
        const int i = y * w_ + x;

        // temporary store sample weight in place of green channel
        sh_data_[i].coeff_g[0] = weight;
    }

    void Resize(int w, int h, uint32_t aux_buffers);
    void Clear(const color_rgba_t &p);

    void ResetSampleData(const rect_t &rect);
    void ComputeSHData(const rect_t &rect);

    template <typename F> void Apply(const rect_t &reg, F &&f) {
        for (int y = reg.y; y < reg.y + reg.h; y++) {
            for (int x = reg.x; x < reg.x + reg.w; x++) {
                f(pixels_[y * w_ + x]);
            }
        }
    }

    const color_rgba_t *get_pixels_ref() const { return pixels_.data(); }
    const shl1_data_t *get_sh_data_ref() const { return sh_data_.data(); }
    const color_rgba_t *get_base_color_ref() const { return base_color_.data(); }
    const color_rgba_t *get_depth_normals_ref() const { return depth_normals_.data(); }

    void MixWith(const Framebuffer &f2, const rect_t &rect, float k);
    void MixWith_SH(const Framebuffer &f2, const rect_t &rect, float k);
    void MixWith_BaseColor(const Framebuffer &f2, const rect_t &rect, float k);
    void MixWith_DepthNormal(const Framebuffer &f2, const rect_t &rect, float k);

    template <typename F> void CopyFrom(const Framebuffer &f2, const rect_t &rect, F &&filter) {
        for (int y = rect.y; y < rect.y + rect.h; y++) {
            for (int x = rect.x; x < rect.x + rect.w; x++) {
                this->SetPixel(x, y, filter(f2.GetPixel(x, y)));
            }
        }
    }
};
} // namespace Ref
} // namespace Ray