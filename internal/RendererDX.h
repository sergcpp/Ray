#pragma once

#include "../RendererBase.h"
#include "Dx/ContextDX.h"

namespace Ray::Dx {
RendererBase *CreateRenderer(const settings_t &s, ILog *log,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
} // namespace Ray::Dx
