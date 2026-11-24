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
    bindless_tex_data_.mtl_tex_ptrs.clear();
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

bool Ray::Mtl::Scene::InitBindlessArray_nolock() {
    if (!use_bindless_) { // the shared caller is unconditional; skip the 64K-slot arg table
        return false;
    }
    const uint32_t tex_cap = ctx_->max_sampled_images();
    assert(bindless_textures_.capacity() <= tex_cap);

    if (bindless_tex_data_.arg_buf) { // re-run (SetEnvironment) reuses the arg table: rebuild = leak + dangling rebind
        return true;
    }

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
        return false;
    }

    bindless_tex_data_.arg_buf =
        ctx_->device()->newBuffer(bindless_tex_data_.encoder->encodedLength(), MTL::ResourceStorageModeShared);
    if (!bindless_tex_data_.arg_buf) {
        log_->Error("Failed to allocate Metal bindless argument buffer!");
        return false;
    }
    bindless_tex_data_.encoder->setArgumentBuffer(bindless_tex_data_.arg_buf, 0);
    bindless_tex_data_.mtl_tex_ptrs.assign(tex_cap, nullptr);
    return true;
}

void Ray::Mtl::Scene::SetBindlessTextureSlot_nolock(const uint32_t index, const Image &tex) {
    MTL::Texture *tex_mtl = tex.mtl();
    bindless_tex_data_.encoder->setTextures(&tex_mtl, NS::Range::Make(index, 1));
    bindless_tex_data_.mtl_tex_ptrs[index] = tex_mtl;
    bindless_tex_data_.tex_count = std::max(bindless_tex_data_.tex_count, index + 1);
}

void Ray::Mtl::Scene::ClearBindlessTextureSlot_nolock(const uint32_t index) {
    if (!bindless_tex_data_.encoder || index >= bindless_tex_data_.mtl_tex_ptrs.size()) {
        return;
    }
    MTL::Texture *tex_mtl = nullptr;
    bindless_tex_data_.encoder->setTextures(&tex_mtl, NS::Range::Make(index, 1));
    bindless_tex_data_.mtl_tex_ptrs[index] = nullptr;
}

void Ray::Mtl::Scene::TransitionBindlessTextures_nolock() {}

void Ray::Mtl::Scene::FinalizeBindlessTextures_nolock(Buffer &tex_sizes_stage) {
    CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
    CopyBufferToBuffer(tex_sizes_stage, 0, bindless_tex_data_.tex_sizes, 0, bindless_tex_data_.tex_sizes.size(),
                       cmd_buf);
    ctx_->EndSingleTimeCommands(cmd_buf);

    tex_sizes_stage.FreeImmediate();
}

std::pair<uint32_t, uint32_t> Ray::Mtl::Scene::Build_HWRT_BLAS_nolock(uint32_t vert_index, uint32_t vert_count,
                                                                      eBVHBuildQuality quality) {
    // Metal has no build-speed or compaction knobs (the driver picks the layout and the
    // final size up front), so `quality` and the VK rt_blas_mem_alloc_ pool are unused here.
    MTL::AccelerationStructureTriangleGeometryDescriptor *tri =
        MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    // vtx_indices_ holds GLOBAL vertex indices (offset at AddMesh), so the vertex
    // buffer is referenced at 0 - mirrors VK's base-address + maxVertex form
    tri->setVertexBuffer(vertices_.gpu_buf().mtl());
    tri->setVertexBufferOffset(0);
    tri->setVertexStride(sizeof(vertex_t));
    tri->setVertexFormat(MTL::AttributeFormatFloat3);
    tri->setIndexBuffer(vtx_indices_.gpu_buf().mtl());
    tri->setIndexBufferOffset(vert_index * sizeof(uint32_t));
    tri->setIndexType(MTL::IndexTypeUInt32);
    tri->setTriangleCount(vert_count / 3);

    NS::Object *geoms[] = {tri};
    NS::Array *geom_arr = NS::Array::alloc()->init(geoms, 1);
    MTL::PrimitiveAccelerationStructureDescriptor *blas_desc =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    blas_desc->setGeometryDescriptors(geom_arr);

    // query sizes before Allocate: a failed build must not leak a SparseStorage slot,
    // its sentinel node_index would poison RemoveMesh_nolock's Erase later
    const MTL::AccelerationStructureSizes sizes = ctx_->device()->accelerationStructureSizes(blas_desc);
    if (sizes.accelerationStructureSize == 0) {
        log_->Error("Failed to query Metal BLAS size!");
        geom_arr->release();
        blas_desc->release();
        tri->release();
        return {0xffffffff, 0xffffffff};
    }

    MTL::AccelerationStructure *blas = ctx_->device()->newAccelerationStructure(sizes.accelerationStructureSize);
    MTL::Buffer *scratch = sizes.buildScratchBufferSize > 0
                               ? ctx_->device()->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModeShared)
                               : nullptr;
    if (blas == nullptr || (sizes.buildScratchBufferSize > 0 && scratch == nullptr)) {
        log_->Error("Failed to allocate Metal BLAS!");
        if (blas) {
            blas->release();
        }
        if (scratch) {
            scratch->release();
        }
        geom_arr->release();
        blas_desc->release();
        tri->release();
        return {0xffffffff, 0xffffffff};
    }

    const std::pair<uint32_t, uint32_t> blas_index = rt_mesh_blases_.Allocate(1);

    CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
    MTL::AccelerationStructureCommandEncoder *as_enc = cmd_buf->accelerationStructureCommandEncoder();
    as_enc->buildAccelerationStructure(blas, blas_desc, scratch, 0);
    as_enc->endEncoding();
    ctx_->EndSingleTimeCommands(cmd_buf);

    rt_mesh_blases_[blas_index.first].acc.Init(ctx_, blas);

    if (scratch) {
        scratch->release();
    }
    geom_arr->release();
    blas_desc->release();
    tri->release();

    return blas_index;
}

void Ray::Mtl::Scene::Rebuild_HWRT_TLAS_nolock() {
    struct RTGeoInstance {
        uint32_t indices_start;
        uint32_t vertices_start;
        uint32_t material_index;
        uint32_t flags;
    };
    static_assert(sizeof(RTGeoInstance) == 16, "!");

    std::vector<RTGeoInstance> geo_instances;
    std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> instances;
    SmallVector<NS::Object *, 16> blas_refs;

    for (auto it = mesh_instances_.cbegin(); it != mesh_instances_.cend(); ++it) {
        const mesh_instance_t &instance = *it;
        const mesh_t &m = meshes_[instance.mesh_index];

        // AddMesh stores the BLAS sentinel node_index unconditionally when the build failed
        if (m.node_index == 0xffffffff || rt_mesh_blases_[m.node_index].acc.mtl_handle() == nullptr) {
            log_->Error("Skipping TLAS instance with failed BLAS build");
            continue;
        }
        auto &blas = rt_mesh_blases_[m.node_index];
        blas.geo_index = uint32_t(geo_instances.size());
        blas.geo_count = 0;

        geo_instances.emplace_back(RTGeoInstance{}); // VK keeps this per-instance table zero-filled: no shader binds it

        instances.emplace_back();
        auto &new_instance = instances.back();
        std::memset(&new_instance, 0, sizeof(new_instance));
        // PackedFloat4x3 columns hold the row-major 3x4 affine rows (ColumnMajor layout)
        const float *x = instance.xform;
        new_instance.transformationMatrix =
            MTL::PackedFloat4x3{MTL::PackedFloat3{x[0], x[1], x[2]}, MTL::PackedFloat3{x[4], x[5], x[6]},
                                MTL::PackedFloat3{x[8], x[9], x[10]}, MTL::PackedFloat3{x[12], x[13], x[14]}};
        new_instance.options = MTL::AccelerationStructureInstanceOptionOpaque;
        new_instance.mask = instance.ray_visibility & 0xff;
        new_instance.intersectionFunctionTableOffset = 0;
        new_instance.accelerationStructureIndex = uint32_t(blas_refs.size());
        new_instance.userID = m.vert_index / 3; // shader reads it as instance-custom-index

        blas_refs.push_back(blas.acc.mtl_handle());
    }

    // instanceCount 0 is legal: the TLAS builds empty and queries miss cleanly (verified on M4),
    // so an instance-less scene still binds a valid AS - no VK-style dummy instance needed.
    if (!instances.empty()) {
        Buffer geo_data_stage_buf{"RT GeoData Stage Buf", ctx_, eBufType::Upload,
                                  uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
        memcpy(geo_data_stage_buf.Map(), geo_instances.data(), geo_instances.size() * sizeof(RTGeoInstance));
        geo_data_stage_buf.Unmap();

        Buffer instance_stage_buf{
            "RT Instance Stage Buf", ctx_, eBufType::Upload,
            uint32_t(instances.size() * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor))};
        memcpy(instance_stage_buf.Map(), instances.data(),
               instances.size() * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor));
        instance_stage_buf.Unmap();

        rt_geo_data_buf_ =
            Buffer{"RT GeoData Buf", ctx_, eBufType::Storage, uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
        rt_instance_buf_ =
            Buffer{"RT Instance Buf", ctx_, eBufType::Storage,
                   uint32_t(instances.size() * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor))};

        CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
        CopyBufferToBuffer(geo_data_stage_buf, 0, rt_geo_data_buf_, 0, geo_data_stage_buf.size(), cmd_buf);
        CopyBufferToBuffer(instance_stage_buf, 0, rt_instance_buf_, 0, instance_stage_buf.size(), cmd_buf);
        ctx_->EndSingleTimeCommands(cmd_buf);

        geo_data_stage_buf.FreeImmediate();
        instance_stage_buf.FreeImmediate();
    }

    NS::Array *blas_arr = NS::Array::alloc()->init(blas_refs.data(), blas_refs.size());
    MTL::InstanceAccelerationStructureDescriptor *tlas_desc =
        MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    tlas_desc->setInstanceDescriptorBuffer(rt_instance_buf_.mtl());
    tlas_desc->setInstanceDescriptorBufferOffset(0);
    tlas_desc->setInstanceDescriptorStride(sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor));
    tlas_desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
    tlas_desc->setInstancedAccelerationStructures(blas_arr);
    tlas_desc->setInstanceCount(uint32_t(instances.size()));

    const MTL::AccelerationStructureSizes sizes = ctx_->device()->accelerationStructureSizes(tlas_desc);
    if (sizes.accelerationStructureSize == 0) {
        log_->Error("Failed to query Metal TLAS size!");
        // drop the old TLAS with it: its blas_refs_ may point at BLASes RemoveMesh already released
        rt_tlas_.Free();
        blas_arr->release();
        tlas_desc->release();
        return;
    }

    MTL::AccelerationStructure *tlas = ctx_->device()->newAccelerationStructure(sizes.accelerationStructureSize);
    MTL::Buffer *scratch = sizes.buildScratchBufferSize > 0
                               ? ctx_->device()->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModeShared)
                               : nullptr;
    if (tlas == nullptr || (sizes.buildScratchBufferSize > 0 && scratch == nullptr)) {
        log_->Error("Failed to allocate Metal TLAS!");
        if (tlas) {
            tlas->release();
        }
        if (scratch) {
            scratch->release();
        }
        rt_tlas_.Free();
        blas_arr->release();
        tlas_desc->release();
        return;
    }

    CommandBuffer tlas_cmd_buf = ctx_->BeginSingleTimeCommands();
    MTL::AccelerationStructureCommandEncoder *as_enc = tlas_cmd_buf->accelerationStructureCommandEncoder();
    as_enc->buildAccelerationStructure(tlas, tlas_desc, scratch, 0);
    as_enc->endEncoding();
    ctx_->EndSingleTimeCommands(tlas_cmd_buf);

    std::vector<MTL::AccelerationStructure *> blas_handles;
    for (NS::Object *o : blas_refs) {
        blas_handles.push_back((MTL::AccelerationStructure *)o);
    }
    rt_tlas_.Init(ctx_, tlas);
    rt_tlas_.set_blas_refs(std::move(blas_handles));

    if (scratch) {
        scratch->release();
    }
    blas_arr->release();
    tlas_desc->release();
}

// Metal encoders resolve read-after-write hazards natively (DX precedent)
void Ray::Mtl::Scene::_insert_mem_barrier(void *cmd_buf) {}
