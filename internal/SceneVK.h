#pragma once

#include "Vk/AccStructureVK.h"
#include "Vk/DescriptorPoolVK.h"
#include "Vk/DrawCallVK.h"
#include "Vk/PipelineVK.h"
#include "Vk/ProgramVK.h"
#include "Vk/ShaderVK.h"
#include "Vk/SparseStorageVK.h"
#include "Vk/TextureAtlasVK.h"
#include "Vk/TextureVK.h"
#include "Vk/VectorVK.h"

namespace Ray::Vk {
class Context;
class Renderer;

struct BindlessTexData {
    DescrPool descr_pool;
    VkDescriptorSetLayout descr_layout = {}, rt_descr_layout = {};
    VkDescriptorSet descr_set = {}, rt_descr_set = {};
    Sampler shared_sampler;
    Buffer tex_sizes;

    explicit BindlessTexData(Context *ctx) : descr_pool(ctx) {}
};
} // namespace Ray::Vk

#define NS Vk
#include "SceneGPU.h"
#undef NS
