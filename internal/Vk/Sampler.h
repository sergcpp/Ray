#pragma once

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
    Sampler(const Sampler &rhs) = delete;
    Sampler(Sampler &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Sampler() { Destroy(); }

    VkSampler vk_handle() const { return handle_; }
    SamplingParams params() const { return params_; }

    operator bool() const { return handle_ != VK_NULL_HANDLE; }

    Sampler &operator=(const Sampler &rhs) = delete;
    Sampler &operator=(Sampler &&rhs) noexcept;

    void Init(Context *ctx, SamplingParams params);
    void Destroy();
};
} // namespace Vk
} // namespace Ray