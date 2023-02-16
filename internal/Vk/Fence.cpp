#include "Fence.h"

#include <cassert>

#include <utility>

namespace Ray {
#ifndef RAY_EXCHANGE_DEFINED
template <class T, class U = T> T exchange(T &obj, U &&new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}
#define RAY_EXCHANGE_DEFINED
#endif
}

Ray::Vk::SyncFence::~SyncFence() {
    if (fence_) {
        vkDestroyFence(device_, fence_, nullptr);
    }
}

Ray::Vk::SyncFence::SyncFence(SyncFence &&rhs) {
    device_ = exchange(rhs.device_, VkDevice{VK_NULL_HANDLE});
    fence_ = exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
}

Ray::Vk::SyncFence &Ray::Vk::SyncFence::operator=(SyncFence &&rhs) {
    if (fence_) {
        vkDestroyFence(device_, fence_, nullptr);
    }
    device_ = exchange(rhs.device_, VkDevice{VK_NULL_HANDLE});
    fence_ = exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
    return (*this);
}

bool Ray::Vk::SyncFence::Reset() {
    const VkResult res = vkResetFences(device_, 1, &fence_);
    return res == VK_SUCCESS;
}

Ray::Vk::WaitResult Ray::Vk::SyncFence::ClientWaitSync(const uint64_t timeout_us) {
    assert(fence_ != VK_NULL_HANDLE);
    const VkResult res = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_us * 1000);

    WaitResult ret = WaitResult::Fail;
    if (res == VK_TIMEOUT) {
        ret = WaitResult::Timeout;
    } else if (res == VK_SUCCESS) {
        ret = WaitResult::Success;
    }

    return ret;
}

Ray::Vk::SyncFence Ray::Vk::MakeFence() { return SyncFence{VK_NULL_HANDLE, VK_NULL_HANDLE}; }
