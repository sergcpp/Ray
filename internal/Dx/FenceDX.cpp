#include "FenceDX.h"

#include <cassert>

#include <utility>

Ray::Dx::SyncFence::~SyncFence() {
    //if (fence_) {
    //    vkDestroyFence(device_, fence_, nullptr);
    //}
}

Ray::Dx::SyncFence::SyncFence(SyncFence &&rhs) noexcept {
    //device_ = std::exchange(rhs.device_, VkDevice{VK_NULL_HANDLE});
    //fence_ = std::exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
}

Ray::Dx::SyncFence &Ray::Dx::SyncFence::operator=(SyncFence &&rhs) noexcept {
    //if (fence_) {
    //    vkDestroyFence(device_, fence_, nullptr);
    //}
    //device_ = std::exchange(rhs.device_, VkDevice{VK_NULL_HANDLE});
    //fence_ = std::exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
    return (*this);
}

bool Ray::Dx::SyncFence::Reset() {
    //const VkResult res = vkResetFences(device_, 1, &fence_);
    //return res == VK_SUCCESS;
    return false;
}

Ray::Dx::WaitResult Ray::Dx::SyncFence::ClientWaitSync(const uint64_t timeout_us) {
    //assert(fence_ != VK_NULL_HANDLE);
    //const VkResult res = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_us * 1000);
    //
    WaitResult ret = WaitResult::Fail;
    //if (res == VK_TIMEOUT) {
    //    ret = WaitResult::Timeout;
    //} else if (res == VK_SUCCESS) {
    //    ret = WaitResult::Success;
    //}

    return ret;
}

Ray::Dx::SyncFence Ray::Dx::MakeFence() { return SyncFence{/*VK_NULL_HANDLE, VK_NULL_HANDLE*/}; }
