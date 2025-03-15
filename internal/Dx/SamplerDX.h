#pragma once

#include <utility>

#include "../SamplingParams.h"
#include "DescriptorPoolDX.h"

namespace Ray {
namespace Dx {
class Context;
class Sampler {
    Context *ctx_ = nullptr;
    PoolRef ref_;
    SamplingParamsPacked params_;

  public:
    Sampler() = default;
    Sampler(Context *ctx, SamplingParams params) { Init(ctx, params); }
    Sampler(const Sampler &rhs) = delete;
    Sampler(Sampler &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Sampler() { Free(); }

    PoolRef ref() const { return ref_; }
    SamplingParams params() const { return params_; }

    explicit operator bool() const { return ref_; }

    Sampler &operator=(const Sampler &rhs) = delete;
    Sampler &operator=(Sampler &&rhs) noexcept;

    void Init(Context *ctx, SamplingParams params);
    void Free();
    void FreeImmediate();
};
} // namespace Dx
} // namespace Ray