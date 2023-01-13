#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;

namespace Avx512 {
const int RPDimX = 4;
const int RPDimY = 4;
const int RPSize = RPDimX * RPDimY;

RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Avx512
} // namespace Ray