#pragma once

#include <iostream>
#include <memory>

#include "RendererBase.h"

/**
  @file RendererFactory.h
*/

namespace Ray {
/// Default renderer flags used to choose backend, by default tries to create gpu opencl renderer first
const uint32_t default_renderer_flags = RendererRef | RendererSSE2 | RendererAVX | RendererAVX2 | RendererNEON | RendererOCL;

struct settings_t {
    int w, h;
#if !defined(DISABLE_OCL)
    int platform_index = -1, device_index = -1;
#endif
};

/** @brief Creates renderer
    @return shared pointer to created renderer
*/
std::shared_ptr<RendererBase> CreateRenderer(const settings_t &s, uint32_t flags = default_renderer_flags, std::ostream &log_stream = std::cout);

#if !defined(DISABLE_OCL)
namespace Ocl {
    std::vector<Platform> QueryPlatforms();
}
#endif
}