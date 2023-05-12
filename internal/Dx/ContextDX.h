#pragma once

#include <memory>

#include "../SmallVector.h"
#include "MemoryAllocatorDX.h"

#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12CommandAllocator;
struct ID3D12GraphicsCommandList;
struct ID3D12PipelineState;
struct ID3D12Resource;
struct ID3D12DescriptorHeap;
struct ID3D12Fence;
struct ID3D12QueryHeap;
struct IUnknown;

typedef void *HANDLE;

namespace Ray {
class ILog;
struct gpu_device_t;
namespace Dx {
static const int MaxFramesInFlight = 6;
static const int MaxTimestampQueries = 256;

class Buffer;
class DescrMultiPoolAlloc;
class MemoryAllocators;

using CommandBuffer = ID3D12GraphicsCommandList *;

class Context {
    ILog *log_ = nullptr;
#ifndef NDEBUG
    unsigned long debug_callback_cookie_ = {};
#endif
    ID3D12Device *device_ = {};
    std::string device_name_;

    bool raytracing_supported_ = false, ray_query_supported_ = false;
    // VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props_ = {
    //     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    bool dynamic_rendering_supported_ = false;

    bool rgb8_unorm_is_supported_ = false;

    ID3D12CommandQueue *command_queue_ = {};

    ID3D12CommandAllocator *command_allocators_[MaxFramesInFlight] = {}, *temp_command_allocator_ = {};

    ID3D12GraphicsCommandList *command_list_ = {};

    // VkCommandPool command_pool_ = {}, temp_command_pool_ = {};
    // VkCommandBuffer setup_cmd_buf_, draw_cmd_bufs_[MaxFramesInFlight];

    // VkSemaphore image_avail_semaphores_[MaxFramesInFlight] = {};
    // VkSemaphore render_finished_semaphores_[MaxFramesInFlight] = {};
    ID3D12Fence *in_flight_fences_[MaxFramesInFlight] = {};
    HANDLE fence_event_ = {};

    ID3D12QueryHeap *query_heaps_[MaxFramesInFlight] = {};
    uint32_t query_counts_[MaxFramesInFlight] = {};
    uint64_t query_results_[MaxFramesInFlight][MaxTimestampQueries] = {};
    std::unique_ptr<Buffer> query_readback_buf_[MaxFramesInFlight];
    double timestamp_period_us_ = 0.0;

    uint32_t max_combined_image_samplers_ = 0;

    std::unique_ptr<MemoryAllocators> default_memory_allocs_;
    std::unique_ptr<DescrMultiPoolAlloc> default_descr_alloc_[MaxFramesInFlight];

  public:
    Context();
    ~Context();

    bool Init(ILog *log, const char *preferred_device);
    void Destroy();

    ID3D12Device *device() const { return device_; }
    const std::string &device_name() const { return device_name_; }

    ILog *log() const { return log_; }

    uint32_t max_combined_image_samplers() const { return max_combined_image_samplers_; }

    bool raytracing_supported() const { return raytracing_supported_; }
    bool ray_query_supported() const { return ray_query_supported_; }

    bool rgb8_unorm_is_supported() const { return rgb8_unorm_is_supported_; }

    // const VkPhysicalDeviceLimits &phys_device_limits() const { return phys_device_limits_; }
    // const VkPhysicalDeviceProperties &device_properties() const { return device_properties_; }
    // const VkPhysicalDeviceMemoryProperties &mem_properties() const { return mem_properties_; }

    ID3D12CommandQueue *graphics_queue() const { return command_queue_; }

    // VkCommandPool command_pool() const { return command_pool_; }
    ID3D12CommandAllocator *temp_command_pool() const { return temp_command_allocator_; }

    CommandBuffer draw_cmd_buf() const { return command_list_; }
    ID3D12CommandAllocator *draw_cmd_alloc(const int i) const { return command_allocators_[i]; }
    // const VkSemaphore &render_finished_semaphore(const int i) const { return render_finished_semaphores_[i]; }
    ID3D12Fence *in_flight_fence(const int i) const { return in_flight_fences_[i]; }
    HANDLE fence_event() const { return fence_event_; }
    ID3D12QueryHeap *query_heap(const int i) const { return query_heaps_[i]; }

    MemoryAllocators *default_memory_allocs() { return default_memory_allocs_.get(); }
    DescrMultiPoolAlloc *default_descr_alloc() { return default_descr_alloc_[backend_frame].get(); }

    int WriteTimestamp(CommandBuffer cmd_buf, bool start);
    uint64_t GetTimestampIntervalDurationUs(int query_start, int query_end) const;

    void ResolveTimestampQueries(int i);
    bool ReadbackTimestampQueries(int i);
    void DestroyDeferredResources(int i);

    int backend_frame = 0;
    bool render_finished_semaphore_is_set[MaxFramesInFlight] = {};
    uint64_t fence_values[MaxFramesInFlight] = {};

    // resources scheduled for deferred destruction
    /*SmallVector<VkImage, 128> images_to_destroy[MaxFramesInFlight];
    SmallVector<VkImageView, 128> image_views_to_destroy[MaxFramesInFlight];
    SmallVector<VkSampler, 128> samplers_to_destroy[MaxFramesInFlight];*/
    SmallVector<MemAllocation, 128> allocs_to_free[MaxFramesInFlight];
    /*SmallVector<VkBufferView, 128> buf_views_to_destroy[MaxFramesInFlight];
    SmallVector<VkDeviceMemory, 128> mem_to_free[MaxFramesInFlight];
    SmallVector<VkRenderPass, 128> render_passes_to_destroy[MaxFramesInFlight];
    SmallVector<VkFramebuffer, 128> framebuffers_to_destroy[MaxFramesInFlight];
    SmallVector<VkDescriptorPool, 16> descriptor_pools_to_destroy[MaxFramesInFlight];
    SmallVector<VkPipelineLayout, 128> pipeline_layouts_to_destroy[MaxFramesInFlight];*/
    SmallVector<ID3D12Resource *, 128> resources_to_destroy[MaxFramesInFlight];
    SmallVector<ID3D12PipelineState *, 128> pipelines_to_destroy[MaxFramesInFlight];
    SmallVector<ID3D12DescriptorHeap *, 128> descriptor_heaps_to_destroy[MaxFramesInFlight];
    SmallVector<IUnknown *, 128> opaques_to_release[MaxFramesInFlight];
    /*SmallVector<VkAccelerationStructureKHR, 128> acc_structs_to_destroy[MaxFramesInFlight];*/

    static int QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], int capacity);

  private:
    /*static bool InitVkInstance(VkInstance &instance, const char *enabled_layers[], int enabled_layers_count,
                                    ILog *log);
    static bool ChooseVkPhysicalDevice(VkPhysicalDevice &physical_device, VkPhysicalDeviceProperties &device_properties,
                                       VkPhysicalDeviceMemoryProperties &mem_properties,
                                       uint32_t &graphics_family_index, bool &out_raytracing_supported,
                                       bool &out_ray_query_supported, bool &out_dynamic_rendering_supported,
                                       const char *preferred_device, VkInstance instance, ILog *log);
    static bool InitVkDevice(VkDevice &device, VkPhysicalDevice physical_device, uint32_t graphics_family_index,
                             bool enable_raytracing, bool enable_ray_query, bool enable_dynamic_rendering,
                             const char *enabled_layers[], int enabled_layers_count, ILog *log);
    static bool InitCommandBuffers(VkCommandPool &command_pool, VkCommandPool &temp_command_pool,
                                   VkCommandBuffer &setup_cmd_buf, VkCommandBuffer draw_cmd_bufs[MaxFramesInFlight],
                                   VkSemaphore image_avail_semaphores[MaxFramesInFlight],
                                   VkSemaphore render_finished_semaphores[MaxFramesInFlight],
                                   VkFence in_flight_fences[MaxFramesInFlight],
                                   VkQueryPool query_pools[MaxFramesInFlight], VkQueue &graphics_queue, VkDevice device,
                                   uint32_t graphics_family_index, ILog *log);*/
};

CommandBuffer BegSingleTimeCommands(ID3D12Device *device, ID3D12CommandAllocator *temp_command_allocator);
void EndSingleTimeCommands(ID3D12Device *device, ID3D12CommandQueue *cmd_queue, CommandBuffer command_list,
                           ID3D12CommandAllocator *temp_command_allocator);

} // namespace Dx
} // namespace Ray