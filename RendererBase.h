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

class RendererBase {
public:
    virtual ~RendererBase() = default;

    virtual eRendererType type() const = 0;

    virtual std::pair<int, int> size() const = 0;

    virtual const pixel_color_t *get_pixels_ref() const = 0;

    virtual void Resize(int w, int h) = 0;
    virtual void Clear(const pixel_color_t &c = { 0, 0, 0, 0 }) = 0;

    virtual std::shared_ptr<SceneBase> CreateScene() = 0;
    virtual void RenderScene(const std::shared_ptr<SceneBase> &s, region_t region = { 0, 0, 0, 0 }) = 0;

    struct stats_t {
        int iterations_count;
    };
    virtual void GetStats(stats_t &st) = 0;
};
}