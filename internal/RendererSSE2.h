#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Sse2 {
const int RPDimX = 2, RPDimY = 2;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Sse2
} // namespace Ray