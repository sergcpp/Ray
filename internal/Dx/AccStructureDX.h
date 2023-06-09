#pragma once

#include "DescriptorPoolDX.h"
#include "ResourceDX.h"

namespace Ray {
namespace Dx {
class Context;

class AccStructure {
    Context *ctx_ = nullptr;
    uint64_t gpu_virtual_address_ = 0;
    PoolRef view_ref_;

  public:
    AccStructure() = default;
    ~AccStructure() { Free(); }

    AccStructure(const AccStructure &rhs) = delete;
    AccStructure(AccStructure &&rhs) noexcept;

    AccStructure &operator=(const AccStructure &rhs) = delete;
    AccStructure &operator=(AccStructure &&rhs) noexcept;

    uint64_t gpu_virtual_address() const { return gpu_virtual_address_; }
    const PoolRef view_ref() const { return view_ref_; }

    bool Init(Context *ctx, uint64_t gpu_virtual_address);

    void Free() { FreeImmediate(); }
    void FreeImmediate();

    eResState resource_state = eResState::Undefined;
};
} // namespace Dx
} // namespace Ray