#pragma once

#include "../RendererBase.h"
#include "Vk/ContextVK.h"

namespace Ray {
class ILog;
namespace Vk {
RendererBase *CreateRenderer(const settings_t &s, ILog *log,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
} // namespace Vk
} // namespace Ray
