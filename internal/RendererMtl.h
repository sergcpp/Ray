#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;
namespace Mtl {
RendererBase *CreateRenderer(const settings_t &s, ILog *log,
                             const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
} // namespace Mtl
} // namespace Ray
