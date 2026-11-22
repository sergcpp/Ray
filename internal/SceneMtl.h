#pragma once

#include "Mtl/AccStructureMtl.h"
#include "Mtl/BufferMtl.h"
#include "Mtl/DrawCallMtl.h"
#include "Mtl/ImageAtlasMtl.h"
#include "Mtl/ImageMtl.h"
#include "Mtl/PipelineMtl.h"
#include "Mtl/ProgramMtl.h"
#include "Mtl/SamplerMtl.h"
#include "Mtl/ShaderMtl.h"
#include "Mtl/SparseStorageMtl.h"
#include "Mtl/VectorMtl.h"

namespace Ray::Mtl {
class Context;
class Renderer;

struct BindlessTexData {
    Sampler shared_sampler;
    Buffer tex_sizes;

    // Bindless array (spvDescriptorSetBuffer1 at buffer(1)): AGX encoder stores driver-private indirect
    // indices, so slots are only writable via setTextures; bound per-dispatch with setBuffer.
    MTL::ArgumentEncoder *encoder = nullptr;
    MTL::Buffer *arg_buf = nullptr;
    // Same slots as arg_buf, for the per-dispatch useResource(Sample) registration
    std::vector<MTL::Texture *> mtl_tex_ptrs;
    uint32_t tex_count = 0; // filled prefix of mtl_tex_ptrs, not the whole 64K table

    explicit BindlessTexData(Context *) {}
};
} // namespace Ray::Mtl

#define NS Mtl
#include "SceneGPU.h"
#undef NS
