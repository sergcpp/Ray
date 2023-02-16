#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Avx2 {
const int RPDimX = 4;
const int RPDimY = 2;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Avx2
} // namespace Ray