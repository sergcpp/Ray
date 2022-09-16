#pragma once

#include <memory>

#include "../SmallVector.h"
#include "MemoryAllocator.h"
#include "VK.h"

#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

namespace Ray {
class ILog;
namespace Vk {
static const int MaxFramesInFlight = 3;
static const int MaxTimestampQueries = 256;

class DescrMultiPoolAlloc;

class Context {
    ILog *log_ = nullptr;
    VkInstance instance_ = {};
#ifndef NDEBUG
    VkDebugReportCallbackEXT debug_callback_ = {};
#endif
    VkPhysicalDevice physical_device_ = {};
    VkPhysicalDeviceLimits phys_device_limits_ = {};
    VkPhysicalDeviceProperties device_properties_ = {};
    VkPhysicalDeviceMemoryProperties mem_properties_ = {};
    uint32_t graphics_family_index_ = 0;

    VkDevice device_ = {};

    bool raytracing_supported_ = false, ray_query_supported_ = false;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props_ = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    bool dynamic_rendering_supported_ = false;

    VkQueue graphics_queue_ = {};

    VkCommandPool command_pool_ = {}, temp_command_pool_ = {};
    VkCommandBuffer setup_cmd_buf_, draw_cmd_bufs_[MaxFramesInFlight];

    VkSemaphore image_avail_semaphores_[MaxFramesInFlight] = {};
    VkSemaphore render_finished_semaphores_[MaxFramesInFlight] = {};
    VkFence in_flight_fences_[MaxFramesInFlight] = {};

    VkQueryPool query_pools_[MaxFramesInFlight] = {};
    uint32_t query_counts_[MaxFramesInFlight] = {};
    uint64_t query_results_[MaxFramesInFlight][MaxTimestampQueries] = {};

    uint32_t max_combined_image_samplers_ = 0;

    std::unique_ptr<MemoryAllocators> default_memory_allocs_;
    std::unique_ptr<DescrMultiPoolAlloc> default_descr_alloc_[MaxFramesInFlight];

  public:
    ~Context();

    bool Init(ILog *log, const char *preferred_device);
    void Destroy();

    VkDevice device() const { return device_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }

    ILog *log() const { return log_; }

    uint32_t max_combined_image_samplers() const { return max_combined_image_samplers_; }

    bool raytracing_supported() const { return raytracing_supported_; }
    bool ray_query_supported() const { return ray_query_supported_; }

    const VkPhysicalDeviceLimits &phys_device_limits() const { return phys_device_limits_; }
    const VkPhysicalDeviceProperties &device_properties() const { return device_properties_; }
    const VkPhysicalDeviceMemoryProperties &mem_properties() const { return mem_properties_; }

    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &rt_props() const { return rt_props_; }

    VkQueue graphics_queue() const { return graphics_queue_; }

    VkCommandPool command_pool() const { return command_pool_; }
    VkCommandPool temp_command_pool() const { return temp_command_pool_; }

    const VkCommandBuffer &draw_cmd_buf(const int i) const { return draw_cmd_bufs_[i]; }
    const VkSemaphore &render_finished_semaphore(const int i) const { return render_finished_semaphores_[i]; }
    const VkFence &in_flight_fence(const int i) const { return in_flight_fences_[i]; }

    MemoryAllocators *default_memory_allocs() { return default_memory_allocs_.get(); }
    DescrMultiPoolAlloc *default_descr_alloc() const { return default_descr_alloc_[backend_frame].get(); }

    void DestroyDeferredResources(const int i);

    int backend_frame = 0;

    // resources scheduled for deferred destruction
    SmallVector<VkImage, 128> images_to_destroy[MaxFramesInFlight];
    SmallVector<VkImageView, 128> image_views_to_destroy[MaxFramesInFlight];
    SmallVector<VkSampler, 128> samplers_to_destroy[MaxFramesInFlight];
    SmallVector<MemAllocation, 128> allocs_to_free[MaxFramesInFlight];
    SmallVector<VkBuffer, 128> bufs_to_destroy[MaxFramesInFlight];
    SmallVector<VkBufferView, 128> buf_views_to_destroy[MaxFramesInFlight];
    SmallVector<VkDeviceMemory, 128> mem_to_free[MaxFramesInFlight];
    SmallVector<VkRenderPass, 128> render_passes_to_destroy[MaxFramesInFlight];
    SmallVector<VkFramebuffer, 128> framebuffers_to_destroy[MaxFramesInFlight];
    SmallVector<VkDescriptorPool, 16> descriptor_pools_to_destroy[MaxFramesInFlight];
    SmallVector<VkPipelineLayout, 128> pipeline_layouts_to_destroy[MaxFramesInFlight];
    SmallVector<VkPipeline, 128> pipelines_to_destroy[MaxFramesInFlight];
    SmallVector<VkAccelerationStructureKHR, 128> acc_structs_to_destroy[MaxFramesInFlight];

  private:
    static bool InitVkInstance(VkInstance &instance, const char *enabled_layers[], int enabled_layers_count, ILog *log);
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
                                   uint32_t graphics_family_index, ILog *log);
};

VkCommandBuffer BegSingleTimeCommands(VkDevice device, VkCommandPool temp_command_pool);
void EndSingleTimeCommands(VkDevice device, VkQueue cmd_queue, VkCommandBuffer command_buf,
                           VkCommandPool temp_command_pool);

} // namespace Vk
} // namespace Ray