#pragma once

#define NS avx2
#define USE_AVX
#include "RendererSIMD.h"
#undef USE_AVX
#undef NS

namespace ray {
namespace avx2 {
class Renderer : public RendererSIMD<4, 4> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererAVX; }
};
}
}