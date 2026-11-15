#pragma once

#include <utility>

#include "../SamplingParams.h"
#include "ApiMtl.h"

namespace Ray::Mtl {
class Context;
class Sampler {
    Context *ctx_ = nullptr;
    MTL::SamplerState *handle_ = nullptr;
    SamplingParamsPacked params_;

  public:
    Sampler() = default;
    Sampler(Context *ctx, SamplingParams params) { Init(ctx, params); }
    Sampler(const Sampler &rhs) = delete;
    Sampler(Sampler &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Sampler() { Free(); }

    MTL::SamplerState *mtl() const { return handle_; }
    SamplingParams params() const { return params_; }

    explicit operator bool() const { return handle_ != nullptr; }

    Sampler &operator=(const Sampler &rhs) = delete;
    Sampler &operator=(Sampler &&rhs) noexcept;

    void Init(Context *ctx, SamplingParams params);
    void Free();
    void FreeImmediate();
};
} // namespace Ray::Mtl
