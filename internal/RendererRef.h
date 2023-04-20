#pragma once

#include "../RendererBase.h"

namespace Ray {
class ILog;
namespace Ref {
RendererBase *CreateRenderer(const settings_t &s, ILog *log);
} // namespace Ref
} // namespace Ray
