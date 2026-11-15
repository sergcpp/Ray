#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../SmallVector.h"
#include "ApiMtl.h"
#include "MemoryAllocatorMtl.h"

namespace Ray {
class ILog;
struct gpu_device_t;
namespace Mtl {
static const int MaxFramesInFlight = 6;
static const int MaxTimestampQueries = 1024;

// Metal binds ids directly - the type exists only to keep the Dispatch*() signatures
// shared with the VK/DX kernel bodies.
class DescrMultiPoolAlloc;

class Context {
    ILog *log_ = nullptr;
    MTL::Device *device_ = nullptr;
    MTL::CommandQueue *cmd_queue_ = nullptr;
    std::unique_ptr<MemAllocators> default_mem_allocs_;

    bool astc_supported_ = false;
    bool bc_supported_ = false;
    std::string device_name_;

    // One lazily opened encoder per command buffer: the compute/blit switch replaces VK's
    // barrier + pipeline-bind points (Metal encoders resolve hazards natively).
    MTL::ComputeCommandEncoder *compute_encoder_ = nullptr;
    MTL::BlitCommandEncoder *blit_encoder_ = nullptr;
    MTL::RenderCommandEncoder *render_encoder_ = nullptr;
    CommandBuffer encoded_cmd_buf_ = nullptr;

    // uniform params staging (VK push constants / DX root params analog)
    static const uint32_t UniformRingSize = 1u << 20;
    MTL::Buffer *uniform_ring_[MaxFramesInFlight] = {};
    uint32_t uniform_ring_pos_[MaxFramesInFlight] = {};
    MTL::SamplerState *dummy_sampler_ = nullptr;

  public:
    ~Context();

    bool Init(ILog *log, std::string_view preferred_device, int validation_level);
    void Destroy();

    ILog *log() const { return log_; }
    MTL::Device *device() const { return device_; }
    MTL::CommandQueue *command_queue() const { return cmd_queue_; }

    // per-Context so a device-A sampler can never reach a device-B encoder (multi-renderer safe)
    MTL::SamplerState *dummy_sampler();

    // SceneGPU.h call-shape shims: Metal has no api table / command pools, encoders
    // resolve hazards natively, so these exist only to keep the shared code verbatim.
    const Context &api() const { return *this; }
    MTL::CommandQueue *graphics_queue() const { return cmd_queue_; }
    void *temp_command_pool() const { return nullptr; }

    std::string_view device_name() const { return device_name_; }

    MemAllocators *default_mem_allocs() { return default_mem_allocs_.get(); }
    DescrMultiPoolAlloc *default_descr_alloc() const { return nullptr; }

    bool raytracing_supported() const { return false; } // HWRT not implemented in this backend
    bool ray_query_supported() const { return false; }
    bool rgb8_unorm_is_supported() const { return false; } // Metal has no RGB8 pixel format
    bool rgb8_promoted_internally() const { return true; } // ImageMtl/ImageAtlasMtl store RGB8 as RGBA8 at upload
    bool fp16_supported() const { return true; }
    bool int64_supported() const { return true; }
    bool int64_atomics_supported() const {
        return false; // gates spatial_cache_update_compat
    }
    bool subgroup_supported() const { return true; }
    bool subgroup_size_control_supported() const { return false; }

    // ASTC LDR is mandatory on Apple7+ (macOS 11+); BC stays a runtime gate.
    bool astc_supported() const { return astc_supported_; }
    bool bc_supported() const { return bc_supported_; }

    // Metal blits are unfiltered, and atlas mips are separate regions anyway: mips come from
    // ImageAtlas::AllocateMips on CPU - the same path DX takes.
    bool image_blit_supported() const { return false; }

    // Bindless capacity: argument-buffer texture handles are only bound-address limited,
    // so we comfortably clear the renderer's >=16384 bindless gate
    uint32_t max_sampled_images() const { return 65536u; }

    MTL::ComputeCommandEncoder *ComputeEncoder(CommandBuffer cmd_buf);
    MTL::BlitCommandEncoder *BlitEncoder(CommandBuffer cmd_buf);
    MTL::RenderCommandEncoder *RenderEncoder(CommandBuffer cmd_buf, MTL::RenderPassDescriptor *rpd);
    void EndEncoders();

    // VK pipeline-barrier semantics for TransitionResourceStates: Metal B2+ may overlap
    // adjacent dispatches in one encoder, so barriers encode onto the currently open encoder
    MTL::ComputeCommandEncoder *openComputeEncoder() const { return compute_encoder_; }
    static Context *LookupForCommandBuffer(MTL::CommandBuffer *cb);

    // retained until EndSingleTimeCommands; matches the VK/DX helpers the shared scene
    // code calls around one-shot uploads
    CommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(CommandBuffer cmd_buf);

    MTL::Buffer *UniformSlice(uint32_t size, uint32_t &out_offset);
    void ResetUniformRing();

    int WriteTimestamp(CommandBuffer cmd_buf, bool start);
    uint64_t GetTimestampIntervalDurationUs(int query_start, int query_end) const;

    bool ReadbackTimestampQueries(int i);
    void DestroyDeferredResources(int i);

    int backend_frame = 0;
    bool render_finished_semaphore_is_set[MaxFramesInFlight] = {};

    bool frame_cpu_synced[MaxFramesInFlight] = {};

    // coarse timestamps (exact ones would need MTLCounterSampleBuffer): WriteTimestamp records CPU times,
    // completed CBs add their GPUStartTime..GPUEndTime span; interval duration = the GPU busy time
    // of the segment (period since previous Readback) scaled over the interval's CPU recording span
    struct TsSeg {
        int lo_id, hi_id;
        uint64_t span_us, gpu_busy_us;
    };
    std::vector<uint64_t> ts_cpu_ = {};
    std::vector<TsSeg> ts_segs_ = {};
    TsSeg open_seg_ = {-1, -1, 0, 0};
    uint64_t gpu_busy_us_ = 0;

    // VK-mirrored call-site drains only: Metal frees happen immediately (ObjC refcounting
    // keeps in-flight CBs' resources alive), these vectors stay empty
    std::vector<MTL::Buffer *> bufs_to_destroy[MaxFramesInFlight];
    std::vector<MTL::Texture *> images_to_destroy[MaxFramesInFlight];
    std::vector<MTL::SamplerState *> samplers_to_destroy[MaxFramesInFlight];

    static int QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], int capacity);
};

// SceneGPU.h call-shape shims (same names as the VK/DX helpers, Metal params elided);
// ContextMtl.h is the common include of all shared GPU storage headers, mirroring ContextVK.h
inline CommandBuffer BegSingleTimeCommands(const Context &ctx, MTL::Device *, void *) {
    return const_cast<Context &>(ctx).BeginSingleTimeCommands();
}
inline void EndSingleTimeCommands(const Context &ctx, MTL::Device *, MTL::CommandQueue *, CommandBuffer cmd_buf,
                                  void *) {
    const_cast<Context &>(ctx).EndSingleTimeCommands(cmd_buf);
}
inline void InsertReadbackMemoryBarrier(const Context &, CommandBuffer) {} // encoders order hazards

} // namespace Mtl
} // namespace Ray
