#pragma once

#include "ResourceVK.h"

namespace Ray {
namespace Vk {
class Context;

class AccStructure {
    Context *ctx_ = nullptr;
    VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;

  public:
    AccStructure() = default;
    ~AccStructure() { Free(); }

    AccStructure(const AccStructure &rhs) = delete;
    AccStructure(AccStructure &&rhs) noexcept;

    AccStructure &operator=(const AccStructure &rhs) = delete;
    AccStructure &operator=(AccStructure &&rhs) noexcept;

    const VkAccelerationStructureKHR &vk_handle() const {
        return handle_;
    } // needs to reference as we take it's address later
    VkDeviceAddress vk_device_address() const;

    bool Init(Context *ctx, VkAccelerationStructureKHR handle);

    void Free();
    void FreeImmediate();

    eResState resource_state = eResState::Undefined;
};
} // namespace Vk
} // namespace Ray