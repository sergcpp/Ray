#pragma once

#include <utility>
#include <vector>

#include "ApiMtl.h"

namespace Ray {
namespace Mtl {
class Context;

// Metal AS objects own their memory (no placement API), so unlike VK there is no
// buffer+offset split: the handle is the whole resource. Release is immediate - ObjC
// refcounting keeps in-flight command buffers alive (same rule as Image/Buffer).
class AccStructure {
    MTL::AccelerationStructure *as_ = nullptr;
    // TLAS only: BLAS handles referenced by as_; Metal requires useResources() on them
    // in every compute pass that traces through the TLAS, else the driver evicts them.
    // Losing this list (move/Free without handoff) silently breaks HWRT rendering.
    std::vector<MTL::AccelerationStructure *> blas_refs_;

  public:
    AccStructure() = default;
    ~AccStructure() { Free(); }

    AccStructure(const AccStructure &rhs) = delete;
    AccStructure(AccStructure &&rhs) noexcept
        : as_(std::exchange(rhs.as_, nullptr)), blas_refs_(std::move(rhs.blas_refs_)) {}

    AccStructure &operator=(const AccStructure &rhs) = delete;
    AccStructure &operator=(AccStructure &&rhs) noexcept {
        if (this != &rhs) {
            Free();
            as_ = std::exchange(rhs.as_, nullptr);
            blas_refs_ = std::move(rhs.blas_refs_);
        }
        return *this;
    }

    MTL::AccelerationStructure *mtl_handle() const { return as_; }
    const std::vector<MTL::AccelerationStructure *> &blas_refs() const { return blas_refs_; }
    void set_blas_refs(std::vector<MTL::AccelerationStructure *> refs) { blas_refs_ = std::move(refs); }

    void Init(Context *ctx, MTL::AccelerationStructure *as) {
        Free();
        as_ = as;
    }

    void Free() {
        if (as_) {
            as_->release();
            as_ = nullptr;
        }
        blas_refs_.clear();
    }

    explicit operator bool() const { return as_ != nullptr; }
    bool ready() const { return as_ != nullptr; }
};

} // namespace Mtl
} // namespace Ray
