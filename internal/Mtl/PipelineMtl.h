#pragma once

#include <cstdint>
#include <vector>

#include "../../Span.h"
#include "ProgramMtl.h"

namespace Ray {
class ILog;
namespace Mtl {
class Context;

enum class ePipelineType : uint8_t { Undefined, Graphics, Compute, Raytracing };

// byte-identical to MTLDispatchThreadgroupsIndirectArguments and VkDispatchIndirectCommand
struct DispatchIndirectCommand {
    uint32_t num_groups_x;
    uint32_t num_groups_y;
    uint32_t num_groups_z;
};

struct TraceRaysIndirectCommand {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

class Pipeline {
    Context *ctx_ = nullptr;
    ePipelineType type_ = ePipelineType::Undefined;
    Program *prog_ = nullptr;
    MTL::ComputePipelineState *pso_ = nullptr;

    uint32_t workgroup_[3] = {};
    std::vector<MslBindingDesc> refl_;
    std::vector<int32_t> loc_index_;     // SPIR-V binding (set 0) -> refl_ entry
    std::vector<int32_t> sampler_index_; // SPIR-V binding (set 0) -> companion msl_k_sampler entry
    int32_t pushconst_index_ = -1;       // refl_ entry of kind msl_k_push
    int32_t descr_array_index_ = -1;     // refl_ entry of kind msl_k_descr_arr (bindless set 1)

  public:
    Pipeline() = default;
    Pipeline(const Pipeline &rhs) = delete;
    Pipeline(Pipeline &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Pipeline();

    Pipeline &operator=(const Pipeline &rhs) = delete;
    Pipeline &operator=(Pipeline &&rhs) noexcept;

    explicit operator bool() const { return pso_ != nullptr; }

    ePipelineType type() const { return type_; }
    const Program *prog() const { return prog_; }
    Context *ctx() const { return ctx_; }

    MTL::ComputePipelineState *handle() const { return pso_; }
    const uint32_t *workgroup_size() const { return workgroup_; }

    const MslBindingDesc *find(uint16_t loc) const;
    const MslBindingDesc *find_sampler(uint16_t loc) const;
    const MslBindingDesc *pushconst() const;
    const MslBindingDesc *descr_array() const;

    bool Init(Context *ctx, Program *prog, ILog *log);
};
} // namespace Mtl
} // namespace Ray
