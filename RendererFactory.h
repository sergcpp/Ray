#pragma once

#include "Config.h"
#include "Log.h"
#include "RendererBase.h"

/**
  @file RendererFactory.h
*/

namespace Ray {
extern LogNull g_null_log;

/// Default renderer flags used to choose backend, by default tries to create gpu opencl renderer first
const uint32_t DefaultEnabledRenderTypes =
    RendererRef | RendererSSE2 | RendererAVX | RendererAVX2 | RendererNEON | RendererVK;

/** @brief Creates renderer
    @return pointer to created renderer
*/
RendererBase *CreateRenderer(const settings_t &s, ILog *log = &g_null_log,
                             uint32_t enabled_types = DefaultEnabledRenderTypes);
} // namespace Ray