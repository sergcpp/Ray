#pragma once

#include <iostream>
#include <memory>

#include "RendererBase.h"

/**
  @file RendererFactory.h
*/

namespace ray {
/// Default renderer flags used to choose backend, by default tries to create gpu opencl renderer first
const uint32_t default_renderer_flags = RendererRef | RendererSSE | RendererAVX | RendererNEON | RendererOCL;

/** @brief Creates renderer
    @param w initial image width
    @param h initial image height
    @return shared pointer to created renderer
*/
std::shared_ptr<RendererBase> CreateRenderer(int w, int h, uint32_t flags = default_renderer_flags, std::ostream &log_stream = std::cout);
}