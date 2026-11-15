#include "FenceMtl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <utility>

#include "ContextMtl.h"

namespace Ray::Mtl {
SyncFence::~SyncFence() {
    if (event_) {
        event_->release();
    }
    event_ = nullptr;
}

SyncFence::SyncFence(SyncFence &&rhs) noexcept {
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    event_ = std::exchange(rhs.event_, nullptr);
    wait_value_ = std::exchange(rhs.wait_value_, 0);
}

SyncFence &SyncFence::operator=(SyncFence &&rhs) noexcept {
    if (event_) {
        event_->release();
    }
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    event_ = std::exchange(rhs.event_, nullptr);
    wait_value_ = std::exchange(rhs.wait_value_, 0);
    return (*this);
}

bool SyncFence::signaled() const { return event_ != nullptr && event_->signaledValue() >= wait_value_; }

bool SyncFence::Reset() {
    // shared-event fences have no reset; re-arming means waiting on a newer value
    return event_ != nullptr;
}

WaitResult SyncFence::ClientWaitSync(const uint64_t timeout_us) {
    assert(event_ != nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    while (event_->signaledValue() < wait_value_) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return WaitResult::Timeout;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return WaitResult::Success;
}

SyncFence MakeFence() { return SyncFence{}; }
} // namespace Ray::Mtl
