#pragma once

#include <memory>

#include "SceneBase.h"
#include "Types.h"

namespace ray {
enum eRendererType {
    RendererRef = 1,
    RendererSSE = 2,
    RendererAVX = 4,
    RendererOCL = 8,
};

class RegionContext {
    const rect_t rect_;
public:
    int iteration = 0;
    std::unique_ptr<float[]> halton_seq;

    explicit RegionContext(const rect_t &rect) : rect_(rect) {}

    rect_t rect() const { return rect_; }

    void Clear() {
        iteration = 0;
        halton_seq = nullptr;
    }
};

class RendererBase {
public:
    virtual ~RendererBase() = default;

    virtual eRendererType type() const = 0;

    virtual std::pair<int, int> size() const = 0;

    virtual const pixel_color_t *get_pixels_ref() const = 0;

    virtual void Resize(int w, int h) = 0;
    virtual void Clear(const pixel_color_t &c = { 0, 0, 0, 0 }) = 0;

    virtual std::shared_ptr<SceneBase> CreateScene() = 0;
    virtual void RenderScene(const std::shared_ptr<SceneBase> &s, RegionContext &region) = 0;

    struct stats_t {
        int iterations_count;
    };
    virtual void GetStats(stats_t &st) = 0;
};
}