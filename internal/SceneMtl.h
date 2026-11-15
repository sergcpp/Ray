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

    // Tier-2 argument buffer for the MSL bindless array (spvDescriptorSetBuffer1 at buffer(1)),
    // laid out from refl kind msl_k_descr_arr; bound per-dispatch by RendererMtl
    MTL::ArgumentEncoder *encoder = nullptr;
    MTL::Buffer *arg_buf = nullptr;

    explicit BindlessTexData(Context *) {}
};
} // namespace Ray::Mtl

#define NS Mtl
#include "SceneGPU.h"
#undef NS
