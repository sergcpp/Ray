#include "SceneMtl.h"

#include <cassert>
#include <cstring>
#include <vector>

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif

#include "BVHSplit.h"
#include "ImageParams.h"
#include "Mtl/ContextMtl.h"
#include "TextureUtils.h"
#include "inflate/Inflate.h"

namespace Ray {
uint32_t next_power_of_two(uint32_t v);

namespace Mtl {
#include "shaders/output/bake_sky.comp.metallib.inl"
} // namespace Mtl
} // namespace Ray

Ray::Mtl::Scene::~Scene() {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    for (auto it = mesh_instances_.begin(); it != mesh_instances_.end();) {
        MeshInstanceHandle to_delete = {it.index(), it.block()};
        ++it;
        Scene::RemoveMeshInstance_nolock(to_delete);
    }
    for (auto it = meshes_.begin(); it != meshes_.end();) {
        MeshHandle to_delete = {it.index(), it.block()};
        ++it;
        Scene::RemoveMesh_nolock(to_delete);
    }

    if (tlas_root_ != 0xffffffff) {
        nodes_.Erase(tlas_block_);
        tlas_root_ = tlas_block_ = 0xffffffff;
    }

    bindless_textures_.clear();
    if (bindless_tex_data_.encoder) {
        bindless_tex_data_.encoder->release();
        bindless_tex_data_.encoder = nullptr;
    }
    if (bindless_tex_data_.arg_buf) {
        bindless_tex_data_.arg_buf->release();
        bindless_tex_data_.arg_buf = nullptr;
    }
}

bool Ray::Mtl::Scene::InitPipelines() {
    sh_bake_sky_ =
        Shader{"Bake Sky", ctx_, Inflate(internal_shaders_output_bake_sky_comp_metallib), eShaderType::Comp, log_};
    prog_bake_sky_ = Program{"Bake Sky", ctx_, &sh_bake_sky_, log_};
    return pi_bake_sky_.Init(ctx_, &prog_bake_sky_, log_);
}

// atlas mips are baked on CPU by ImageAtlas::AllocateMips: image_blit_supported() is false,
// so the shared AddAtlasTexture path never asks the GPU for them (DX precedent)
void Ray::Mtl::Scene::GenerateTextureMips_nolock() {}

void Ray::Mtl::Scene::PrepareBindlessTextures_nolock() {
    if (!use_bindless_) { // the shared caller is unconditional; skip the 64K-slot arg table
        return;
    }
    const uint32_t tex_cap = ctx_->max_sampled_images();
    assert(bindless_textures_.capacity() <= tex_cap);

    { // Init shared sampler
        SamplingParams params;
        params.filter = eFilter::Nearest;
        params.wrap = eWrap::Repeat;

        bindless_tex_data_.shared_sampler.Init(ctx_, params);
    }

    { // Single texture-array argument (index 0) matching spvDescriptorSetBuffer1's
      // spvDescriptor<texture2d<float>> member emitted by spirv-cross
        MTL::ArgumentDescriptor *arg_desc = MTL::ArgumentDescriptor::alloc()->init();
        arg_desc->setIndex(0);
        arg_desc->setDataType(MTL::DataType::DataTypeTexture);
        arg_desc->setTextureType(MTL::TextureType::TextureType2D);
        arg_desc->setArrayLength(tex_cap);
        arg_desc->setAccess(MTL::BindingAccess::BindingAccessReadOnly);

        NS::Object *objs[] = {arg_desc};
        NS::Array *args = NS::Array::alloc()->init(objs, 1);
        bindless_tex_data_.encoder = ctx_->device()->newArgumentEncoder(args);
        args->release();
        arg_desc->release();

        if (!bindless_tex_data_.encoder) {
            log_->Error("Failed to create Metal bindless argument encoder!");
            return;
        }

        bindless_tex_data_.arg_buf =
            ctx_->device()->newBuffer(bindless_tex_data_.encoder->encodedLength(), MTL::ResourceStorageModeShared);
        if (!bindless_tex_data_.arg_buf) {
            log_->Error("Failed to allocate Metal bindless argument buffer!");
            return;
        }
        bindless_tex_data_.encoder->setArgumentBuffer(bindless_tex_data_.arg_buf, 0);
    }

    bindless_tex_data_.tex_sizes = Buffer{"Texture sizes", ctx_, eBufType::Storage,
                                          uint32_t(std::max(1u, bindless_textures_.capacity()) * sizeof(uint32_t))};
    Buffer tex_sizes_stage = Buffer{"Texture sizes Stage", ctx_, eBufType::Upload,
                                    uint32_t(std::max(1u, bindless_textures_.capacity()) * sizeof(uint32_t))};

    std::vector<MTL::Texture *> tex_ptrs(size_t(tex_cap), nullptr);
    for (auto it = bindless_textures_.begin(); it != bindless_textures_.end(); ++it) {
        const Image &tex = bindless_textures_[it.index()];
        tex_ptrs[size_t(it.index())] = tex.mtl();
    }
    bindless_tex_data_.encoder->setTextures(tex_ptrs.data(), NS::Range::Make(0, tex_cap));

    uint32_t *p_tex_sizes = reinterpret_cast<uint32_t *>(tex_sizes_stage.Map());
    memset(p_tex_sizes, 0, bindless_tex_data_.tex_sizes.size());

    for (auto it = bindless_textures_.begin(); it != bindless_textures_.end(); ++it) {
        const Image &tex = bindless_textures_[it.index()];
        p_tex_sizes[it.index()] = (uint32_t(tex.params.w) << 16) | tex.params.h;
    }

    tex_sizes_stage.Unmap();

    CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
    CopyBufferToBuffer(tex_sizes_stage, 0, bindless_tex_data_.tex_sizes, 0, bindless_tex_data_.tex_sizes.size(),
                       cmd_buf);
    ctx_->EndSingleTimeCommands(cmd_buf);

    tex_sizes_stage.FreeImmediate();
}

std::pair<uint32_t, uint32_t> Ray::Mtl::Scene::Build_HWRT_BLAS_nolock(uint32_t vert_index, uint32_t vert_count,
                                                                      eBVHBuildQuality quality) {
    assert(false && "HWRT requires Metal ray queries");
    return {};
}

void Ray::Mtl::Scene::Rebuild_HWRT_TLAS_nolock() { assert(false && "HWRT requires Metal ray queries"); }

// Metal encoders resolve read-after-write hazards natively (DX precedent)
void Ray::Mtl::Scene::_insert_mem_barrier(void *cmd_buf) {}
