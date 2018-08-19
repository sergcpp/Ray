#pragma once

#define NS ref2
#include "RendererSIMD.h"
#undef NS

namespace Ray {
namespace ref2 {
const int RayPacketDimX = 1;
const int RayPacketDimY = 1;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererRef; }
};
}
}