#include "FramebufferRef.h"

#include <cassert>
#include <cstring>

ray::ref::Framebuffer::Framebuffer(int w, int h) {
    Resize(w, h);
}

void ray::ref::Framebuffer::Resize(int w, int h) {
    assert(w > 0 && h > 0);
    w_ = w;
    h_ = h;
    size_t buf_size = w * h;
    pixels_.resize(buf_size, pixel_color_t{});
}

void ray::ref::Framebuffer::Clear(const pixel_color_t &p) {
    for (int i = 0; i < w_; i++) {
        pixels_[i] = p;
    }
    for (int i = 1; i < h_; i++) {
        memcpy(&pixels_[i * w_], &pixels_[0], w_ * sizeof(pixel_color_t));
    }
}
