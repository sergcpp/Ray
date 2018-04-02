#pragma once

#define NS sse
#define USE_SSE
#include "RendererSIMD.h"
#undef USE_SSE
#undef NS

namespace ray {
namespace sse {
class Renderer : public RendererSIMD<4, 4> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererSSE; }
};
}
}