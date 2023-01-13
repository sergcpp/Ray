#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Neon {
const int RPDimX = 2;
const int RPDimY = 2;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Neon
} // namespace Ray
