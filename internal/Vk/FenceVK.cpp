#include "FenceVK.h"

#include <cassert>

#include <utility>

#include "ContextVK.h"

Ray::Vk::SyncFence::~SyncFence() {
    if (fence_) {
        ctx_->api().vkDestroyFence(ctx_->device(), fence_, nullptr);
    }
}

Ray::Vk::SyncFence::SyncFence(SyncFence &&rhs) noexcept {
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    fence_ = std::exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
}

Ray::Vk::SyncFence &Ray::Vk::SyncFence::operator=(SyncFence &&rhs) noexcept {
    if (fence_) {
        ctx_->api().vkDestroyFence(ctx_->device(), fence_, nullptr);
    }
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    fence_ = std::exchange(rhs.fence_, VkFence{VK_NULL_HANDLE});
    return (*this);
}

bool Ray::Vk::SyncFence::signaled() const { return ctx_->api().vkGetFenceStatus(ctx_->device(), fence_) == VK_SUCCESS; }

bool Ray::Vk::SyncFence::Reset() {
    const VkResult res = ctx_->api().vkResetFences(ctx_->device(), 1, &fence_);
    return res == VK_SUCCESS;
}

Ray::Vk::WaitResult Ray::Vk::SyncFence::ClientWaitSync(const uint64_t timeout_us) {
    assert(fence_ != VK_NULL_HANDLE);
    const VkResult res = ctx_->api().vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, timeout_us * 1000);

    WaitResult ret = WaitResult::Fail;
    if (res == VK_TIMEOUT) {
        ret = WaitResult::Timeout;
    } else if (res == VK_SUCCESS) {
        ret = WaitResult::Success;
    }

    return ret;
}

Ray::Vk::SyncFence Ray::Vk::MakeFence() { return SyncFence{VK_NULL_HANDLE, VK_NULL_HANDLE}; }
