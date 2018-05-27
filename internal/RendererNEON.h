#pragma once

#define NS neon
#define USE_NEON
#include "RendererSIMD.h"
#undef USE_NEON
#undef NS

namespace ray {
namespace neon {
const int RayPacketDimX = 2;
const int RayPacketDimY = 2;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererNEON; }
};
}
}