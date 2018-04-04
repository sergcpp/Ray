#pragma once

#define NS ref2
#include "RendererSIMD.h"
#undef NS

namespace ray {
namespace ref2 {
class Renderer : public RendererSIMD<4, 4> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererRef; }
};
}
}