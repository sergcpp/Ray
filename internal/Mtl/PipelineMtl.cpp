#include "PipelineMtl.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>

#include "../../Log.h"
#include "ContextMtl.h"

namespace Ray::Mtl {
Pipeline::~Pipeline() {
    if (pso_) {
        pso_->release();
        pso_ = nullptr;
    }
}

Pipeline &Pipeline::operator=(Pipeline &&rhs) noexcept {
    if (pso_) {
        pso_->release();
    }
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    type_ = std::exchange(rhs.type_, ePipelineType::Undefined);
    prog_ = std::exchange(rhs.prog_, nullptr);
    pso_ = std::exchange(rhs.pso_, nullptr);
    memcpy(workgroup_, rhs.workgroup_, sizeof(workgroup_));
    refl_ = std::move(rhs.refl_);
    loc_index_ = std::move(rhs.loc_index_);
    sampler_index_ = std::move(rhs.sampler_index_);
    pushconst_index_ = std::exchange(rhs.pushconst_index_, -1);
    descr_array_index_ = std::exchange(rhs.descr_array_index_, -1);
    return (*this);
}

bool Pipeline::Init(Context *ctx, Program *prog, ILog *log) {
    assert(prog && prog->ready());
    ctx_ = ctx;
    prog_ = prog;

    const Shader *cs = prog->shader(eShaderType::Comp);
    refl_ = cs->refl();
    memcpy(workgroup_, cs->workgroup_size(), sizeof(workgroup_));

    int32_t max_loc = -1;
    for (const MslBindingDesc &d : refl_) {
        if (d.kind == msl_k_push) {
            pushconst_index_ = int32_t(&d - refl_.data());
        } else if (d.kind == msl_k_descr_arr)
            descr_array_index_ = int32_t(&d - refl_.data());
        else if (d.set == 0)
            max_loc = std::max(max_loc, int32_t(d.binding));
    }

    loc_index_.assign(size_t(max_loc + 1), -1);
    sampler_index_.assign(size_t(max_loc + 1), -1);
    for (const MslBindingDesc &d : refl_) {
        if (d.set == 0 && d.kind != msl_k_push) {
            // combined GLSL samplers emit a second entry with the same (set, binding) - keep
            // the non-sampler one in loc_index_ and remember the companion for setSamplerState
            if (d.kind == msl_k_sampler) {
                sampler_index_[size_t(d.binding)] = int32_t(&d - refl_.data());
            } else {
                loc_index_[size_t(d.binding)] = int32_t(&d - refl_.data());
            }
        }
    }

    NS::Error *err = nullptr;
    pso_ = ctx->device()->newComputePipelineState(cs->function(), &err);
    if (pso_ == nullptr) {
        log->Error("Failed to create Metal compute PSO: %s", err ? err->localizedDescription()->utf8String() : "?");
        return false; // out-param error is +0 autoreleased - do NOT release
    }

    type_ = ePipelineType::Compute;
    return true;
}

const MslBindingDesc *Pipeline::find(const uint16_t loc) const {
    if (loc >= loc_index_.size()) {
        return nullptr;
    }
    const int32_t idx = loc_index_[size_t(loc)];
    return idx >= 0 ? &refl_[size_t(idx)] : nullptr;
}

const MslBindingDesc *Pipeline::find_sampler(const uint16_t loc) const {
    if (loc >= sampler_index_.size()) {
        return nullptr;
    }
    const int32_t idx = sampler_index_[size_t(loc)];
    return idx >= 0 ? &refl_[size_t(idx)] : nullptr;
}

const MslBindingDesc *Pipeline::pushconst() const {
    return pushconst_index_ >= 0 ? &refl_[size_t(pushconst_index_)] : nullptr;
}

const MslBindingDesc *Pipeline::descr_array() const {
    return descr_array_index_ >= 0 ? &refl_[size_t(descr_array_index_)] : nullptr;
}
} // namespace Ray::Mtl
