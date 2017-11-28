#pragma once

#include "CoreRef.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace ray {
namespace ref {
class Renderer : public RendererBase {
    ray::ref::Framebuffer framebuf_;

    math::aligned_vector<ray_packet_t> primary_rays_;
    math::aligned_vector<hit_data_t> intersections_;

    std::vector<pixel_color_t> color_table_;
public:
    Renderer(int w, int h);

    std::pair<int, int> size() const override {
        return std::make_pair(framebuf_.w(), framebuf_.h());
    }

    const pixel_color_t *get_pixels_ref() const override {
        return framebuf_.get_pixels_ref();
    }

    void Resize(int w, int h) override {
        framebuf_.Resize(w, h);
    }
    void Clear(const pixel_color_t &c) override {
        framebuf_.Clear(c);
    }

    std::shared_ptr<SceneBase> CreateScene() override;
    void RenderScene(const std::shared_ptr<SceneBase> &s) override;

    virtual void GetStats(stats_t &st) override {
        st.iterations_count = 0;
    }
};
}
}