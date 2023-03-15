#pragma once

#include <mutex>

#include "../RendererBase.h"
#include "CoreRef.h"

namespace Ray {
class ILog;
namespace Ref {

struct PassData {
    aligned_vector<ray_data_t> primary_rays;
    aligned_vector<ray_data_t> secondary_rays;
    aligned_vector<shadow_ray_t> shadow_rays;
    aligned_vector<hit_data_t> intersections;

    aligned_vector<color_rgba_t, 16> temp_final_buf;
    aligned_vector<color_rgba_t, 16> variance_buf;
    aligned_vector<color_rgba_t, 16> filtered_variance_buf;

    std::vector<uint32_t> hash_values;
    std::vector<int> head_flags;
    std::vector<uint32_t> scan_values;

    std::vector<ray_chunk_t> chunks, chunks_temp;
    std::vector<uint32_t> skeleton;
};

class Renderer : public RendererBase {
    ILog *log_;

    bool use_wide_bvh_;
    aligned_vector<color_rgba_t, 16> dual_buf_[2], base_color_buf_, depth_normals_buf_, temp_buf_, final_buf_,
        raw_final_buf_, filtered_final_buf_;

    std::mutex mtx_;

    stats_t stats_ = {0};
    int w_ = 0, h_ = 0;

    std::vector<uint16_t> permutations_;
    void UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq);

  public:
    Renderer(const settings_t &s, ILog *log);

    eRendererType type() const override { return RendererRef; }

    ILog *log() const override { return log_; }

    const char *device_name() const override { return "CPU"; }

    std::pair<int, int> size() const override { return std::make_pair(w_, h_); }

    const color_rgba_t *get_pixels_ref() const override { return final_buf_.data(); }
    const color_rgba_t *get_raw_pixels_ref() const override { return raw_final_buf_.data(); }
    const color_rgba_t *get_aux_pixels_ref(const eAUXBuffer buf) const override {
        if (buf == eAUXBuffer::BaseColor) {
            return base_color_buf_.data();
        } else if (buf == eAUXBuffer::DepthNormals) {
            return depth_normals_buf_.data();
        }
        return nullptr;
    }
    const color_rgba_t *get_denoised_pixels_ref() const override { return filtered_final_buf_.data(); }

    const shl1_data_t *get_sh_data_ref() const override { return nullptr; }

    void Resize(const int w, const int h) override {
        if (w_ != w || h_ != h) {
            for (auto &buf : dual_buf_) {
                buf.assign(w * h, {});
                buf.shrink_to_fit();
            }
            temp_buf_.assign(w * h, {});
            temp_buf_.shrink_to_fit();
            final_buf_.assign(w * h, {});
            final_buf_.shrink_to_fit();
            raw_final_buf_.assign(w * h, {});
            raw_final_buf_.shrink_to_fit();
            filtered_final_buf_.assign(w * h, {});
            filtered_final_buf_.shrink_to_fit();

            w_ = w;
            h_ = h;
        }
    }

    void Clear(const color_rgba_t &c) override {
        for (auto &buf : dual_buf_) {
            buf.assign(w_ * h_, c);
        }
    }

    SceneBase *CreateScene() override;
    void RenderScene(const SceneBase *scene, RegionContext &region) override;
    void DenoiseImage(const RegionContext &region) override;

    void GetStats(stats_t &st) override { st = stats_; }
    void ResetStats() override { stats_ = {0}; }
};
} // namespace Ref
} // namespace Ray