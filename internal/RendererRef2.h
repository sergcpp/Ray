#pragma once

#define NS ref2
#include "RendererSIMD.h"
#undef NS

namespace ray {
namespace ref2 {
const int RayPacketDimX = 4;
const int RayPacketDimY = 4;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererRef; }
};
}
}