#pragma once

#include <mutex>

#include "../RendererBase.h"
#include "CoreRef.h"
#include "FramebufferRef.h"

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
    ILog *log_;

    bool use_wide_bvh_;
    Ref::Framebuffer clean_buf_, final_buf_, temp_buf_;

    std::mutex pass_cache_mtx_;
    std::vector<PassData> pass_cache_;

    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    Renderer(const settings_t &s, ILog *log);

    eRendererType type() const override { return RendererRef; }

    std::pair<int, int> size() const override { return std::make_pair(final_buf_.w(), final_buf_.h()); }

    const pixel_color_t *get_pixels_ref() const override { return final_buf_.get_pixels_ref(); }

    const shl1_data_t *get_sh_data_ref() const override { return clean_buf_.get_sh_data_ref(); }

    void Resize(int w, int h) override {
        if (w_ != w || h_ != h) {
            clean_buf_.Resize(w, h, false);
            final_buf_.Resize(w, h, false);
            temp_buf_.Resize(w, h, false);

            w_ = w;
            h_ = h;
        }
    }

    void Clear(const pixel_color_t &c) override { clean_buf_.Clear(c); }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};
} // namespace Ref
} // namespace Ray