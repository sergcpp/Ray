#pragma once

#define NS avx2
#define USE_AVX
#include "RendererSIMD.h"
#undef USE_AVX
#undef NS

namespace ray {
namespace avx2 {
const int RayPacketDimX = 4;
const int RayPacketDimY = 2;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
public:
    Renderer(int w, int h) : RendererSIMD(w, h) {}

    eRendererType type() const override { return RendererAVX; }
};
}
}