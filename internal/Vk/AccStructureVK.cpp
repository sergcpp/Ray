#include "AccStructureVK.h"

#include "ContextVK.h"

Ray::Vk::AccStructure::AccStructure(AccStructure &&rhs) noexcept
    : ctx_(std::exchange(rhs.ctx_, nullptr)), handle_(std::exchange(rhs.handle_, {})) {}

Ray::Vk::AccStructure &Ray::Vk::AccStructure::operator=(AccStructure &&rhs) noexcept {
    Free();

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    resource_state = std::exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Vk::AccStructure::Free() {
    if (handle_) {
        ctx_->acc_structs_to_destroy[ctx_->backend_frame].push_back(handle_);
        handle_ = {};
    }
}

void Ray::Vk::AccStructure::FreeImmediate() {
    if (handle_) {
        ctx_->api().vkDestroyAccelerationStructureKHR(ctx_->device(), handle_, nullptr);
        handle_ = {};
    }
}

VkDeviceAddress Ray::Vk::AccStructure::vk_device_address() const {
    VkAccelerationStructureDeviceAddressInfoKHR info = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    info.accelerationStructure = handle_;
    return ctx_->api().vkGetAccelerationStructureDeviceAddressKHR(ctx_->device(), &info);
}

bool Ray::Vk::AccStructure::Init(Context *ctx, VkAccelerationStructureKHR handle) {
    Free();

    ctx_ = ctx;
    handle_ = handle;
    return true;
}