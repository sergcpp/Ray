#pragma once

#include <memory>

#include "RendererBase.h"

namespace ray {
const uint32_t default_renderer_flags = RendererRef | RendererSSE | RendererAVX | RendererOCL;

std::shared_ptr<RendererBase> CreateRenderer(int w, int h, uint32_t flags = default_renderer_flags);
}