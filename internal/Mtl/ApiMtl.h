#pragma once

// metal-cpp umbrella: the PRIVATE_IMPLEMENTATION blocks land in ShaderMtl.cpp only,
// every other TU includes these headers without the defines.
#include <dispatch/dispatch.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdint>

namespace Ray::Mtl {
using CommandBuffer = MTL::CommandBuffer *;

// Kind order must match msl_k_* in scripts/compile_shaders.py (MLR1 refl payload)
enum MslBindingKind : uint32_t {
    msl_k_sbuf_ro = 0,
    msl_k_sbuf_rw,
    msl_k_tex,
    msl_k_sampler,
    msl_k_image_ro,
    msl_k_image_wo,
    msl_k_image_rw,
    msl_k_push,
    msl_k_acc,
    msl_k_descr_arr,
};

struct MslBindingDesc {
    uint32_t set, binding, msl_index, arg_buffer_id_slot, count, kind;
};

} // namespace Ray::Mtl
