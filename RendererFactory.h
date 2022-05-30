#pragma once

#include <iostream>
#include <memory>

#include "RendererBase.h"

/**
  @file RendererFactory.h
*/

namespace Ray {
/// Default renderer flags used to choose backend, by default tries to create gpu opencl renderer first
const uint32_t DefaultEnabledRenderTypes =
    RendererRef /*| RendererSSE2 | RendererAVX | RendererAVX2 | RendererNEON | RendererOCL*/;

/** @brief Creates renderer
    @return shared pointer to created renderer
*/
RendererBase *CreateRenderer(const settings_t &s, uint32_t enabled_types = DefaultEnabledRenderTypes,
                             std::ostream &log_stream = std::cout);

#if !defined(DISABLE_OCL)
/*namespace Ocl {
std::vector<Platform> QueryPlatforms();
}*/
#endif
} // namespace Ray