#pragma once

#include <cstdint>

#include "Api.h"

namespace Ray {
namespace Vk {
class Context;

enum class WaitResult { Success, Timeout, Fail };

class SyncFence {
    Context *ctx_ = nullptr;
    VkFence fence_ = VK_NULL_HANDLE;

  public:
    SyncFence() = default;
    SyncFence(Context *ctx, VkFence fence) : ctx_(ctx), fence_(fence) {}
    ~SyncFence();

    SyncFence(const SyncFence &rhs) = delete;
    SyncFence(SyncFence &&rhs) noexcept;
    SyncFence &operator=(const SyncFence &rhs) = delete;
    SyncFence &operator=(SyncFence &&rhs) noexcept;

    explicit operator bool() const { return fence_ != VK_NULL_HANDLE; }
    VkFence fence() { return fence_; }

    bool signaled() const;

    bool Reset();

    WaitResult ClientWaitSync(uint64_t timeout_us = 1000000000);
};

SyncFence MakeFence();
} // namespace Vk
}