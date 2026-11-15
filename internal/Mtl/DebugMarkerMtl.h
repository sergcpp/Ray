#pragma once

#include <string_view>

#include "ApiMtl.h"

namespace Ray::Mtl {
class Context;

// Metal Frame Capture signal markers not implemented; RAII surface kept for the shared call sites
struct DebugMarker {
    DebugMarker(Context *, CommandBuffer, std::string_view) {}
    ~DebugMarker() { End(); }
    void End() {}
};
} // namespace Ray::Mtl
