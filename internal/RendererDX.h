#pragma once

#include "../RendererBase.h"
#include "Dx/ContextDX.h"

namespace Ray {
namespace Dx {
RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Dx
} // namespace Ray
