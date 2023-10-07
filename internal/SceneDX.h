#pragma once

#include "Dx/AccStructureDX.h"
#include "Dx/DescriptorPoolDX.h"
#include "Dx/SparseStorageDX.h"
#include "Dx/TextureAtlasDX.h"
#include "Dx/TextureDX.h"
#include "Dx/VectorDX.h"

namespace Ray {
namespace Dx {
class Context;
class Renderer;

struct BindlessTexData {
    DescrPool<BumpAlloc> srv_descr_pool;
    DescrTable srv_descr_table;
    Sampler shared_sampler;
    Buffer tex_sizes;

    explicit BindlessTexData(Context *ctx) : srv_descr_pool(ctx, eDescrType::CBV_SRV_UAV) {}
};

} // namespace Dx
} // namespace Ray

#define NS Dx
#include "SceneGPU.h"
#undef NS
