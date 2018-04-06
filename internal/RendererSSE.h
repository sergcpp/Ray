#pragma once

#define NS sse
#define USE_SSE
#include "RendererSIMD.h"
#undef USE_SSE
#undef NS

namespace ray {
namespace sse {
const int RayPacketDimX = 2;
const int RayPacketDimY = 2;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererSSE; }
};
}
}