#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Avx {
const int RPDimX = 4;
const int RPDimY = 2;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Avx
} // namespace Ray