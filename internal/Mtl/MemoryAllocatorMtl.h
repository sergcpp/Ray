#pragma once

#include <cstdint>
#include <string_view>

namespace Ray::Mtl {
// Metal allocates per-resource (newBuffer/newTexture manage their own storage): the
// MemAllocators surface from VK/DX is an empty shell kept for Context API parity.
class MemAllocators {
  public:
    MemAllocators(std::string_view, uint32_t, float, uint32_t) {}
};
} // namespace Ray::Mtl
