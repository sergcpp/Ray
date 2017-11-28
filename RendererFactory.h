#pragma once

#include <memory>

#include "RendererBase.h"

namespace ray {
enum ePreferFlags {
    PreferRef = 1,
    PreferSSE = 2,
    PreferAVX = 4,
    PreferOCL = 8,
};
const uint32_t default_prefer_flags = PreferRef | PreferSSE | PreferAVX | PreferOCL;

std::shared_ptr<RendererBase> CreateRenderer(int w, int h, uint32_t flags = default_prefer_flags);
}