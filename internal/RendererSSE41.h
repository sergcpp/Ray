#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Sse41 {
const int RPDimX = 2;
const int RPDimY = 2;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Sse41
} // namespace Ray