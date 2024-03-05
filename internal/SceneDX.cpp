#include "SceneDX.h"

#include <cassert>

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "BVHSplit.h"
#include "Dx/ContextDX.h"
#include "TextureParams.h"
#include "TextureUtils.h"

namespace Ray {
uint32_t next_power_of_two(uint32_t v);
int round_up(int v, int align);

void to_dxr_xform(const float xform[16], float matrix[3][4]) {
    // transpose
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix[i][j] = xform[4 * j + i];
        }
    }
}

const uint32_t AccStructAlignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
} // namespace Ray

Ray::Dx::Scene::~Scene() {
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

    if (macro_nodes_root_ != 0xffffffff) {
        nodes_.Erase(macro_nodes_block_);
        macro_nodes_root_ = macro_nodes_block_ = 0xffffffff;
    }

    bindless_textures_.clear();
}

void Ray::Dx::Scene::GenerateTextureMips_nolock() {}

void Ray::Dx::Scene::PrepareBindlessTextures_nolock() {
    assert(bindless_textures_.capacity() <= ctx_->max_combined_image_samplers());

    { // Init shared sampler
        SamplingParams params;
        params.filter = eTexFilter::Nearest;
        params.wrap = eTexWrap::Repeat;

        bindless_tex_data_.shared_sampler.Init(ctx_, params);
    }

    const bool bres = bindless_tex_data_.srv_descr_pool.Init(ctx_->max_combined_image_samplers(), false);
    if (!bres) {
        log_->Error("Failed to init descriptor pool!");
    }

    const uint32_t off = bindless_tex_data_.srv_descr_pool.Alloc(bindless_textures_.capacity()).first;
    assert(off == 0);
    (void)off;

    ID3D12Device *device = ctx_->device();
    ID3D12DescriptorHeap *srv_descr_heap = bindless_tex_data_.srv_descr_pool.heap();

    const UINT CBV_SRV_UAV_INCR = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu_handle = srv_descr_heap->GetCPUDescriptorHandleForHeapStart();

    bindless_tex_data_.tex_sizes = Buffer{"Texture sizes", ctx_, eBufType::Storage,
                                          uint32_t(std::max(1u, bindless_textures_.capacity()) * sizeof(uint32_t))};
    Buffer tex_sizes_stage = Buffer{"Texture sizes Stage", ctx_, eBufType::Upload,
                                    uint32_t(std::max(1u, bindless_textures_.capacity()) * sizeof(uint32_t))};

    uint32_t *p_tex_sizes = reinterpret_cast<uint32_t *>(tex_sizes_stage.Map());
    memset(p_tex_sizes, 0, bindless_tex_data_.tex_sizes.size());

    for (auto it = bindless_textures_.begin(); it != bindless_textures_.end(); ++it) {
        const Texture2D &tex = bindless_textures_[it.index()];

        { // copy srv
            D3D12_CPU_DESCRIPTOR_HANDLE src_handle = tex.handle().views_ref.heap->GetCPUDescriptorHandleForHeapStart();
            src_handle.ptr += CBV_SRV_UAV_INCR * tex.handle().views_ref.offset;

            D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = srv_cpu_handle;
            dest_handle.ptr += CBV_SRV_UAV_INCR * it.index();

            device->CopyDescriptorsSimple(1, dest_handle, src_handle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        p_tex_sizes[it.index()] = (uint32_t(tex.params.w) << 16) | tex.params.h;
    }

    tex_sizes_stage.Unmap();

    D3D12_CPU_DESCRIPTOR_HANDLE srv_gpu_handle = srv_descr_heap->GetCPUDescriptorHandleForHeapStart();
    bindless_tex_data_.srv_descr_table.type = eDescrType::CBV_SRV_UAV;
    bindless_tex_data_.srv_descr_table.cpu_heap = srv_descr_heap;
    bindless_tex_data_.srv_descr_table.cpu_ptr = srv_gpu_handle.ptr;
    bindless_tex_data_.srv_descr_table.count = int(bindless_textures_.capacity());

    CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    { // Transition resources
        std::vector<TransitionInfo> img_transitions;
        img_transitions.reserve(bindless_textures_.size());

        for (const auto &tex : bindless_textures_) {
            img_transitions.emplace_back(&tex, eResState::ShaderResource);
        }

        TransitionResourceStates(cmd_buf, AllStages, AllStages, img_transitions);
    }

    CopyBufferToBuffer(tex_sizes_stage, 0, bindless_tex_data_.tex_sizes, 0, bindless_tex_data_.tex_sizes.size(),
                       cmd_buf);

    TransitionInfo trans(&bindless_tex_data_.tex_sizes, eResState::ShaderResource);
    TransitionResourceStates(cmd_buf, AllStages, AllStages, {&trans, 1});

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
}

std::pair<uint32_t, uint32_t> Ray::Dx::Scene::Build_HWRT_BLAS_nolock(const uint32_t vert_index,
                                                                     const uint32_t vert_count) {
    //
    // Gather geometries
    //
    SmallVector<D3D12_RAYTRACING_GEOMETRY_DESC, 16> geometries;
    SmallVector<uint32_t, 16> prim_counts;
    {
        auto &new_geo = geometries.emplace_back();
        new_geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        new_geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

        new_geo.Triangles.Transform3x4 = 0;
        new_geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        new_geo.Triangles.VertexBuffer.StartAddress = vertices_.gpu_buf().dx_resource()->GetGPUVirtualAddress();
        new_geo.Triangles.VertexBuffer.StrideInBytes = sizeof(vertex_t);
        new_geo.Triangles.VertexCount = vertices_.gpu_buf().size() / sizeof(vertex_t);
        new_geo.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        new_geo.Triangles.IndexBuffer =
            vtx_indices_.gpu_buf().dx_resource()->GetGPUVirtualAddress() + vert_index * sizeof(uint32_t);
        new_geo.Triangles.IndexCount = vert_count;

        // auto &new_range = new_blas.build_ranges.emplace_back();
        // new_range.firstVertex = 0; // mesh.vert_index;
        // new_range.primitiveCount = mesh.vert_count / 3;
        // new_range.primitiveOffset = mesh.vert_index * sizeof(uint32_t);
        // new_range.transformOffset = 0;

        prim_counts.push_back(vert_count / 3);
    }

    //
    // Query needed memory
    //
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC as_desc = {};
    as_desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    as_desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    as_desc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                           D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    as_desc.Inputs.NumDescs = UINT(geometries.size());
    as_desc.Inputs.pGeometryDescs = geometries.data();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_prebuild_info = {};
    ctx_->device5()->GetRaytracingAccelerationStructurePrebuildInfo(&as_desc.Inputs, &as_prebuild_info);

    // make sure we will not use this potentially stale pointer
    as_desc.Inputs.pGeometryDescs = nullptr;

    const uint32_t needed_build_scratch_size = uint32_t(as_prebuild_info.ScratchDataSizeInBytes);
    const uint32_t needed_acc_struct_size =
        uint32_t(round_up(int(as_prebuild_info.ResultDataMaxSizeInBytes), AccStructAlignment));

    const std::pair<uint32_t, uint32_t> blas_index = rt_mesh_blases_.Allocate(1);

    //
    // Allocate memory
    //
    Buffer scratch_buf("BLAS Scratch Buf", ctx_, eBufType::Storage, next_power_of_two(needed_build_scratch_size));
    const uint64_t scratch_addr = scratch_buf.dx_resource()->GetGPUVirtualAddress();

    Buffer acc_structs_buf("BLAS Before-Compaction Buf", ctx_, eBufType::AccStructure, needed_acc_struct_size);
    const uint64_t acc_structs_addr = acc_structs_buf.dx_resource()->GetGPUVirtualAddress();

    Buffer compacted_sizes_buf(
        "BLAS Compacted Sizes Buf", ctx_, eBufType::Storage,
        uint32_t(sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC)));
    const uint64_t compacted_sizes_addr = compacted_sizes_buf.dx_resource()->GetGPUVirtualAddress();

    Buffer compacted_sizes_readback_buf("BLAS Compacted Sizes Readback Buf", ctx_, eBufType::Readback,
                                        compacted_sizes_buf.size());

    { // Submit build commands
        auto *cmd_buf = reinterpret_cast<ID3D12GraphicsCommandList4 *>(
            BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool()));

        { // transition buffers to required states
            const TransitionInfo transitions[] = {{&scratch_buf, eResState::UnorderedAccess},
                                                  {&compacted_sizes_buf, eResState::UnorderedAccess}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, transitions);
        }

        as_desc.Inputs.pGeometryDescs = geometries.data();
        as_desc.ScratchAccelerationStructureData = scratch_addr;
        as_desc.DestAccelerationStructureData = acc_structs_addr;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuild_info = {};
        postbuild_info.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        postbuild_info.DestBuffer = compacted_sizes_addr;
        cmd_buf->BuildRaytracingAccelerationStructure(&as_desc, 1, &postbuild_info);

        {
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Transition.pResource = scratch_buf.dx_resource();
            barrier.Transition.StateBefore = DXResourceState(eResState::UnorderedAccess);
            barrier.Transition.StateAfter = DXResourceState(eResState::UnorderedAccess);
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            cmd_buf->ResourceBarrier(1, &barrier);
        }

        // copy compacted sizes for readback
        CopyBufferToBuffer(compacted_sizes_buf, 0, compacted_sizes_readback_buf, 0, compacted_sizes_buf.size(),
                           cmd_buf);

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    const uint32_t compact_size =
        uint32_t(reinterpret_cast<const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC *>(
                     compacted_sizes_readback_buf.Map())
                     ->CompactedSizeInBytes);
    compacted_sizes_readback_buf.Unmap();

    FreelistAlloc::Allocation mem_alloc = rt_blas_mem_alloc_.Alloc(AccStructAlignment, compact_size);
    if (mem_alloc.offset == 0xffffffff) {
        // allocate one more buffer
        const uint32_t buf_size = std::max(next_power_of_two(compact_size), RtBLASChunkSize);
        rt_blas_buffers_.emplace_back("RT BLAS Buffer", ctx_, eBufType::AccStructure, buf_size);
        const uint16_t pool_index = rt_blas_mem_alloc_.AddPool(buf_size);
        assert(pool_index == rt_blas_buffers_.size() - 1);
        (void)pool_index;
        // try to allocate again
        mem_alloc = rt_blas_mem_alloc_.Alloc(AccStructAlignment, compact_size);
        assert(mem_alloc.offset != 0xffffffff);
    }

    rt_mesh_blases_[blas_index.first].mem_alloc = mem_alloc;

    { // Submit compaction commands
        auto *cmd_buf = reinterpret_cast<ID3D12GraphicsCommandList4 *>(
            BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool()));

        const uint64_t old_address = as_desc.DestAccelerationStructureData;
        const uint64_t new_address =
            rt_blas_buffers_[mem_alloc.pool].dx_resource()->GetGPUVirtualAddress() + mem_alloc.offset;

        cmd_buf->CopyRaytracingAccelerationStructure(new_address, old_address,
                                                     D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        auto &vk_blas = rt_mesh_blases_[blas_index.first].acc;
        if (!vk_blas.Init(ctx_, new_address)) {
            log_->Error("Blas compaction failed!");
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        acc_structs_buf.FreeImmediate();
        scratch_buf.FreeImmediate();
    }

    return blas_index;
}

void Ray::Dx::Scene::Rebuild_HWRT_TLAS_nolock() {
    struct RTGeoInstance {
        uint32_t indices_start;
        uint32_t vertices_start;
        uint32_t material_index;
        uint32_t flags;
    };
    static_assert(sizeof(RTGeoInstance) == 16, "!");

    std::vector<RTGeoInstance> geo_instances;
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> tlas_instances;

    for (auto it = mesh_instances_.cbegin(); it != mesh_instances_.cend(); ++it) {
        const mesh_instance_t &instance = *it;
        const mesh_t &m = meshes_[instance.mesh_index];

        auto &blas = rt_mesh_blases_[m.node_index];
        blas.geo_index = uint32_t(geo_instances.size());
        blas.geo_count = 0;

        auto &dx_blas = blas.acc;

        tlas_instances.emplace_back();
        auto &new_instance = tlas_instances.back();
        to_dxr_xform(instance.xform, new_instance.Transform);
        new_instance.InstanceID = meshes_[instance.mesh_index].vert_index / 3;
        new_instance.InstanceMask = (instance.ray_visibility & 0xff);
        new_instance.InstanceContributionToHitGroupIndex = 0;
        new_instance.Flags = 0;
        new_instance.AccelerationStructure = dx_blas.gpu_virtual_address();

        // const mesh_t &mesh = meshes_[instance.mesh_index];
        {
            ++blas.geo_count;

            geo_instances.emplace_back();
            auto &geo = geo_instances.back();
            geo.indices_start = 0;  // mesh.
            geo.vertices_start = 0; // acc.mesh->attribs_buf1().offset / 16;
            geo.material_index = 0; // grp.mat.index();
            geo.flags = 0;
        }
    }

    if (geo_instances.empty()) {
        geo_instances.emplace_back();
        auto &dummy_geo = geo_instances.back();
        dummy_geo = {};

        tlas_instances.emplace_back();
        auto &dummy_instance = tlas_instances.back();
        dummy_instance = {};
    }

    rt_geo_data_buf_ =
        Buffer{"RT Geo Data Buf", ctx_, eBufType::Storage, uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
    Buffer geo_data_stage_buf{"RT Geo Data Stage Buf", ctx_, eBufType::Upload,
                              uint32_t(geo_instances.size() * sizeof(RTGeoInstance))};
    {
        uint8_t *geo_data_stage = geo_data_stage_buf.Map();
        memcpy(geo_data_stage, geo_instances.data(), geo_instances.size() * sizeof(RTGeoInstance));
        geo_data_stage_buf.Unmap();
    }

    rt_instance_buf_ = Buffer{"RT Instance Buf", ctx_, eBufType::Storage,
                              uint32_t(tlas_instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC))};
    Buffer instance_stage_buf{"RT Instance Stage Buf", ctx_, eBufType::Upload,
                              uint32_t(tlas_instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC))};
    {
        uint8_t *instance_stage = instance_stage_buf.Map();
        memcpy(instance_stage, tlas_instances.data(), tlas_instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
        instance_stage_buf.Unmap();
    }

    uint64_t instance_buf_addr = rt_instance_buf_.dx_resource()->GetGPUVirtualAddress();

    auto *cmd_buf = reinterpret_cast<ID3D12GraphicsCommandList4 *>(
        BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool()));

    CopyBufferToBuffer(geo_data_stage_buf, 0, rt_geo_data_buf_, 0, geo_data_stage_buf.size(), cmd_buf);
    CopyBufferToBuffer(instance_stage_buf, 0, rt_instance_buf_, 0, instance_stage_buf.size(), cmd_buf);

    Buffer tlas_scratch_buf;

    { //
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_build_info = {};
        tlas_build_info.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlas_build_info.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlas_build_info.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        tlas_build_info.Inputs.NumDescs = uint32_t(tlas_instances.size());
        tlas_build_info.Inputs.InstanceDescs = instance_buf_addr;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {};
        ctx_->device5()->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_build_info.Inputs, &prebuild_info);

        rt_tlas_buf_ =
            Buffer{"TLAS Buf", ctx_, eBufType::AccStructure, uint32_t(prebuild_info.ResultDataMaxSizeInBytes)};
        tlas_scratch_buf =
            Buffer{"TLAS Scratch Buf", ctx_, eBufType::Storage, uint32_t(prebuild_info.ScratchDataSizeInBytes)};

        { // transition buffers to required states
            const TransitionInfo transitions[] = {{&rt_instance_buf_, eResState::ShaderResource},
                                                  {&tlas_scratch_buf, eResState::UnorderedAccess}};
            TransitionResourceStates(cmd_buf, AllStages, AllStages, transitions);
        }

        tlas_build_info.DestAccelerationStructureData = rt_tlas_buf_.dx_resource()->GetGPUVirtualAddress();
        tlas_build_info.ScratchAccelerationStructureData = tlas_scratch_buf.dx_resource()->GetGPUVirtualAddress();

        cmd_buf->BuildRaytracingAccelerationStructure(&tlas_build_info, 0, nullptr);

        if (!rt_tlas_.Init(ctx_, tlas_build_info.DestAccelerationStructureData)) {
            log_->Error("[SceneManager::InitHWAccStructures]: Failed to init TLAS!");
        }
    }

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    tlas_scratch_buf.FreeImmediate();
    instance_stage_buf.FreeImmediate();
    geo_data_stage_buf.FreeImmediate();
}

void Ray::Dx::Scene::_insert_mem_barrier(void *cmd_buf) {}