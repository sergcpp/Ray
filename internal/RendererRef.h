#pragma once

#include <mutex>

#include "CoreRef.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace ray {
namespace ref {
struct PassData {
    aligned_vector<ray_packet_t> primary_rays;
    aligned_vector<ray_packet_t> secondary_rays;
    aligned_vector<hit_data_t> intersections;

    PassData() = default;

    PassData(const PassData &rhs) = delete;
    PassData(PassData &&rhs) { *this = std::move(rhs); }

    PassData &operator=(const PassData &rhs) = delete;
    PassData &operator=(PassData &&rhs) {
        primary_rays = std::move(rhs.primary_rays);
        secondary_rays = std::move(rhs.secondary_rays);
        intersections = std::move(rhs.intersections);
        return *this;
    }
};

class Renderer : public RendererBase {
    ray::ref::Framebuffer clean_buf_, final_buf_, temp_buf_;

    std::mutex pass_cache_mtx_;
    std::vector<PassData> pass_cache_;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);
public:
    Renderer(int w, int h);

    eRendererType type() const override { return RendererRef; }

    std::pair<int, int> size() const override {
        return std::make_pair(final_buf_.w(), final_buf_.h());
    }

    const pixel_color_t *get_pixels_ref() const override {
        return final_buf_.get_pixels_ref();
    }

    void Resize(int w, int h) override {
        clean_buf_.Resize(w, h);
        final_buf_.Resize(w, h);
        temp_buf_.Resize(w, h);
    }

    void Clear(const pixel_color_t &c) override {
        clean_buf_.Clear(c);
    }

    std::shared_ptr<SceneBase> CreateScene() override;
    void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) override;

    virtual void GetStats(stats_t &st) override {
        
    }
};
}
}