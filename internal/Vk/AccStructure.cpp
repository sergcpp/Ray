#include "AccStructure.h"

#include "Context.h"

void Ray::Vk::AccStructure::Destroy() {
    if (handle_) {
        ctx_->acc_structs_to_destroy[ctx_->backend_frame].push_back(handle_);
        handle_ = {};
    }
}

VkDeviceAddress Ray::Vk::AccStructure::vk_device_address() const {
    VkAccelerationStructureDeviceAddressInfoKHR info = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    info.accelerationStructure = handle_;
    return vkGetAccelerationStructureDeviceAddressKHR(ctx_->device(), &info);
}

bool Ray::Vk::AccStructure::Init(Context *ctx, VkAccelerationStructureKHR handle) {
    Destroy();

    ctx_ = ctx;
    handle_ = handle;
    return true;
}