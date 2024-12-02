#pragma once

#include "../RendererBase.h"
#include "Dx/ContextDX.h"

namespace Ray {
namespace Dx {
RendererBase *CreateRenderer(const settings_t &s, ILog *log,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
} // namespace Dx
} // namespace Ray
