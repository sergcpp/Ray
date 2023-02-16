#pragma once

#include <utility>

#include "SamplingParams.h"
#include "VK.h"

namespace Ray {
namespace Vk {
class Context;
class Sampler {
    Context *ctx_ = nullptr;
    VkSampler handle_ = VK_NULL_HANDLE;
    SamplingParams params_;

  public:
    Sampler() = default;
    Sampler(Context *ctx, SamplingParams params) { Init(ctx, params); }
    Sampler(const Sampler &rhs) = delete;
    Sampler(Sampler &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Sampler() { Free(); }

    VkSampler vk_handle() const { return handle_; }
    SamplingParams params() const { return params_; }

    operator bool() const { return handle_ != VK_NULL_HANDLE; }

    Sampler &operator=(const Sampler &rhs) = delete;
    Sampler &operator=(Sampler &&rhs) noexcept;

    void Init(Context *ctx, SamplingParams params);
    void Free();
    void FreeImmediate();
};
} // namespace Vk
} // namespace Ray