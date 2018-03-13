#pragma once

#include "CoreRef.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace ray {
namespace ref {
class Renderer : public RendererBase {
    ray::ref::Framebuffer framebuf_;

    std::vector<pixel_color_t> color_table_;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);
public:
    Renderer(int w, int h);

    eRendererType type() const override { return RendererRef; }

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
    void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) override;

    virtual void GetStats(stats_t &st) override {
        st.iterations_count = 0;
    }
};
}
}