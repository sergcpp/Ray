#pragma once

#include <memory>

#include "../SmallVector.h"
#include "MemoryAllocatorDX.h"

#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

struct ID3D12Device;
struct ID3D12Device5;
struct ID3D12CommandQueue;
struct ID3D12CommandAllocator;
struct ID3D12GraphicsCommandList;
struct ID3D12PipelineState;
struct ID3D12Resource;
struct ID3D12DescriptorHeap;
struct ID3D12Fence;
struct ID3D12QueryHeap;
struct ID3D12CommandSignature;
struct ID3D12InfoQueue1;
struct IUnknown;

typedef void *HANDLE;

namespace Ray {
class ILog;
struct gpu_device_t;
namespace Dx {
static const int MaxFramesInFlight = 6;
static const int MaxTimestampQueries = 1024;

class Buffer;
class BumpAlloc;
class FreelistAllocAdapted;
template <class Allocator> class DescrMultiPoolAlloc;
class MemAllocators;

using CommandBuffer = ID3D12GraphicsCommandList *;

class Context {
    ILog *log_ = nullptr;
    unsigned long debug_callback_cookie_ = {};
    int validation_level_ = 0;
    ID3D12Device *device_ = {};
    ID3D12Device5 *device5_ = {};
    std::string device_name_;

    bool raytracing_supported_ = false, ray_query_supported_ = false;
    // VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props_ = {
    //     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    // bool dynamic_rendering_supported_ = false;

    bool rgb8_unorm_is_supported_ = false;

    bool subgroup_supported_ = false;

    bool fp16_supported_ = false;

    bool int64_supported_ = false;

    bool int64_atomics_supported_ = false;

    ID3D12CommandQueue *command_queue_ = {};

    ID3D12CommandAllocator *command_allocators_[MaxFramesInFlight] = {}, *temp_command_allocator_ = {};

    ID3D12GraphicsCommandList *command_list_ = {};

    ID3D12CommandSignature *indirect_dispatch_cmd_signature_ = {};

    ID3D12Fence *in_flight_fences_[MaxFramesInFlight] = {};
    HANDLE fence_event_ = {};

    ID3D12QueryHeap *query_heaps_[MaxFramesInFlight] = {};
    uint32_t query_counts_[MaxFramesInFlight] = {};
    uint64_t query_results_[MaxFramesInFlight][MaxTimestampQueries] = {};
    std::unique_ptr<Buffer> query_readback_buf_[MaxFramesInFlight];
    double timestamp_period_us_ = 0.0;

    uint32_t max_combined_image_samplers_ = 0, max_sampled_images_ = 0, max_samplers_ = 0;

    std::unique_ptr<MemAllocators> default_mem_allocs_;
    std::unique_ptr<DescrMultiPoolAlloc<BumpAlloc>> default_descr_alloc_[MaxFramesInFlight];
    std::unique_ptr<DescrMultiPoolAlloc<FreelistAllocAdapted>> staging_descr_alloc_;

  public:
    Context();
    ~Context();

    bool Init(ILog *log, const char *preferred_device, int validation_level);
    void Destroy();

    ID3D12Device *device() const { return device_; }
    ID3D12Device5 *device5() const { return device5_; }
    const std::string &device_name() const { return device_name_; }

    ILog *log() const { return log_; }
    void *api() const { return nullptr; }

    uint32_t max_combined_image_samplers() const { return max_combined_image_samplers_; }
    uint32_t max_sampled_images() const { return max_sampled_images_; }
    uint32_t max_samplers() const { return max_samplers_; }

    bool raytracing_supported() const { return raytracing_supported_; }
    bool ray_query_supported() const { return ray_query_supported_; }
    bool subgroup_supported() const { return subgroup_supported_; }
    bool fp16_supported() const { return fp16_supported_; }
    bool int64_supported() const { return int64_supported_; }
    bool int64_atomics_supported() const { return int64_atomics_supported_; }

    bool rgb8_unorm_is_supported() const { return rgb8_unorm_is_supported_; }

    bool image_blit_supported() const { return false; }

    // const VkPhysicalDeviceLimits &phys_device_limits() const { return phys_device_limits_; }
    // const VkPhysicalDeviceProperties &device_properties() const { return device_properties_; }
    // const VkPhysicalDeviceMemoryProperties &mem_properties() const { return mem_properties_; }

    ID3D12CommandQueue *graphics_queue() const { return command_queue_; }

    // VkCommandPool command_pool() const { return command_pool_; }
    ID3D12CommandAllocator *temp_command_pool() const { return temp_command_allocator_; }

    ID3D12CommandSignature *indirect_dispatch_cmd_signature() const { return indirect_dispatch_cmd_signature_; }

    CommandBuffer draw_cmd_buf() const { return command_list_; }
    ID3D12CommandAllocator *draw_cmd_alloc(const int i) const { return command_allocators_[i]; }
    // const VkSemaphore &render_finished_semaphore(const int i) const { return render_finished_semaphores_[i]; }
    ID3D12Fence *in_flight_fence(const int i) const { return in_flight_fences_[i]; }
    HANDLE fence_event() const { return fence_event_; }
    ID3D12QueryHeap *query_heap(const int i) const { return query_heaps_[i]; }

    MemAllocators *default_mem_allocs() { return default_mem_allocs_.get(); }
    DescrMultiPoolAlloc<BumpAlloc> *default_descr_alloc() { return default_descr_alloc_[backend_frame].get(); }
    DescrMultiPoolAlloc<FreelistAllocAdapted> *staging_descr_alloc() { return staging_descr_alloc_.get(); }

    int WriteTimestamp(CommandBuffer cmd_buf, bool start);
    uint64_t GetTimestampIntervalDurationUs(int query_start, int query_end) const;

    void ResolveTimestampQueries(int i);
    bool ReadbackTimestampQueries(int i);
    void DestroyDeferredResources(int i);

    int backend_frame = 0;
    bool render_finished_semaphore_is_set[MaxFramesInFlight] = {};
    uint64_t fence_values[MaxFramesInFlight] = {};

    bool frame_cpu_synced[MaxFramesInFlight] = {};

    Buffer uniform_data_bufs[MaxFramesInFlight];
    uint32_t uniform_data_buf_offs[MaxFramesInFlight];

    // resources scheduled for deferred destruction
    SmallVector<MemAllocation, 128> allocs_to_free[MaxFramesInFlight];
    SmallVector<ID3D12Resource *, 128> resources_to_destroy[MaxFramesInFlight];
    SmallVector<ID3D12PipelineState *, 128> pipelines_to_destroy[MaxFramesInFlight];
    SmallVector<ID3D12DescriptorHeap *, 128> descriptor_heaps_to_release[MaxFramesInFlight];
    SmallVector<IUnknown *, 128> opaques_to_release[MaxFramesInFlight];

    static int QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], int capacity);
};

CommandBuffer BegSingleTimeCommands(void *api, ID3D12Device *device, ID3D12CommandAllocator *temp_command_allocator);
void EndSingleTimeCommands(void *api, ID3D12Device *device, ID3D12CommandQueue *cmd_queue, CommandBuffer command_list,
                           ID3D12CommandAllocator *temp_command_allocator);

void InsertReadbackMemoryBarrier(void *api, CommandBuffer command_list);

} // namespace Dx
} // namespace Ray