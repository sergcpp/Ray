#pragma once

#include <cstdint>

#include "VK.h"

namespace Ray {
namespace Vk {
enum class WaitResult { Success, Timeout, Fail };

class SyncFence {
    VkDevice device_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

  public:
    SyncFence() = default;
    SyncFence(VkDevice device, VkFence fence) : device_(device), fence_(fence) {}
    ~SyncFence();

    SyncFence(const SyncFence &rhs) = delete;
    SyncFence(SyncFence &&rhs);
    SyncFence &operator=(const SyncFence &rhs) = delete;
    SyncFence &operator=(SyncFence &&rhs);

    operator bool() const { return fence_ != VK_NULL_HANDLE; }
    VkFence fence() { return fence_; }

    bool signaled() const { return vkGetFenceStatus(device_, fence_) == VK_SUCCESS; }

    bool Reset();

    WaitResult ClientWaitSync(uint64_t timeout_us = 1000000000);
};

SyncFence MakeFence();
} // namespace Vk
}