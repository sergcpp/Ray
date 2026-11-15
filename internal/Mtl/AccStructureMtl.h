#pragma once

#include <cstdint>
#include <utility>

namespace Ray {
namespace Mtl {
class Context;

// Placeholder: SceneGPU.h carries HWRT members that need the type; nothing GPU-side
// touches it while Context::ray_query_supported() is false (HWRT unimplemented).
class AccStructure {
    uint32_t buf_index_ = 0xffffffff;
    uint32_t buf_offset_ = 0;

  public:
    AccStructure() = default;
    explicit AccStructure(Context *ctx) {}
    AccStructure(const AccStructure &rhs) = delete;
    AccStructure(AccStructure &&rhs) noexcept { (*this) = std::move(rhs); }

    AccStructure &operator=(AccStructure &&rhs) noexcept {
        buf_index_ = rhs.buf_index_;
        buf_offset_ = rhs.buf_offset_;
        return *this;
    }

    explicit operator bool() const { return false; }
    bool ready() const { return false; }
};

} // namespace Mtl
} // namespace Ray
