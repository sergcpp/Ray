#pragma once

#include "Resource.h"
#include "VK.h"

namespace Ray {
namespace Vk {
class Context;

class AccStructure {
    Context *ctx_ = nullptr;
    VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;

    void Destroy();

  public:
    uint32_t geo_index = 0, geo_count = 0;

    AccStructure() = default;
    ~AccStructure() { Destroy(); }

    AccStructure(const AccStructure &rhs) = delete;
    AccStructure(AccStructure &&rhs) = delete;

    AccStructure &operator=(const AccStructure &rhs) = delete;
    AccStructure &operator=(AccStructure &&rhs) = delete;

    const VkAccelerationStructureKHR &vk_handle() const {
        return handle_;
    } // needs to reference as we take it's address later
    VkDeviceAddress vk_device_address() const;

    bool Init(Context *ctx, VkAccelerationStructureKHR handle);

    eResState resource_state = eResState::Undefined;
};
} // namespace Vk
} // namespace Ray