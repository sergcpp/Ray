#pragma once

#include <cstdint>

#include "ApiMtl.h"

namespace Ray::Mtl {
class Context;

enum class WaitResult { Success, Timeout, Fail };

class SyncFence {
    Context *ctx_ = nullptr;
    MTL::SharedEvent *event_ = nullptr;
    uint64_t wait_value_ = 0;

  public:
    SyncFence() = default;
    SyncFence(Context *ctx, MTL::SharedEvent *event, uint64_t wait_value)
        : ctx_(ctx), event_(event), wait_value_(wait_value) {}
    ~SyncFence();

    SyncFence(const SyncFence &rhs) = delete;
    SyncFence(SyncFence &&rhs) noexcept;
    SyncFence &operator=(const SyncFence &rhs) = delete;
    SyncFence &operator=(SyncFence &&rhs) noexcept;

    explicit operator bool() const { return event_ != nullptr; }

    bool signaled() const;
    bool Reset();
    WaitResult ClientWaitSync(uint64_t timeout_us = 1000000000);
};

SyncFence MakeFence();
} // namespace Ray::Mtl
