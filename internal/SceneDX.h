#pragma once

#include "Dx/AccStructureDX.h"
#include "Dx/DescriptorPoolDX.h"
#include "Dx/DrawCallDX.h"
#include "Dx/ImageAtlasDX.h"
#include "Dx/ImageDX.h"
#include "Dx/PipelineDX.h"
#include "Dx/ProgramDX.h"
#include "Dx/ShaderDX.h"
#include "Dx/SparseStorageDX.h"
#include "Dx/VectorDX.h"

namespace Ray::Dx {
class Context;
class Renderer;

struct BindlessTexData {
    DescrPool<BumpAlloc> srv_descr_pool;
    DescrTable srv_descr_table;
    Sampler shared_sampler;
    Buffer tex_sizes;

    explicit BindlessTexData(Context *ctx) : srv_descr_pool(ctx, eDescrType::CBV_SRV_UAV) {}
};
} // namespace Ray::Dx

#define NS Dx
#include "SceneGPU.h"
#undef NS
