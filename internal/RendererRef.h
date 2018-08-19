#pragma once

#include <mutex>

#include "CoreRef.h"
#include "FramebufferRef.h"
#include "../RendererBase.h"

namespace Ray {
namespace Ref {

struct PassData {
    aligned_vector<ray_packet_t> primary_rays;
    aligned_vector<ray_packet_t> secondary_rays;
    aligned_vector<hit_data_t> intersections;

    std::vector<uint32_t> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;

    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;

    PassData() = default;

    PassData(const PassData &rhs) = delete;
    PassData(PassData &&rhs) noexcept { *this = std::move(rhs); }

    PassData &operator=(const PassData &rhs) = delete;
    PassData &operator=(PassData &&rhs) noexcept {
        primary_rays = std::move(rhs.primary_rays);
        secondary_rays = std::move(rhs.secondary_rays);
        intersections = std::move(rhs.intersections);
        head_flags = std::move(rhs.head_flags);
        return *this;
    }
};

class Renderer : public RendererBase {
    Ref::Framebuffer clean_buf_, final_buf_, temp_buf_;

    std::mutex pass_cache_mtx_;
    std::vector<PassData> pass_cache_;

    stats_t stats_ = { 0 };

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

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = { 0 }; }
};
}
}