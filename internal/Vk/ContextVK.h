#pragma once

#include <memory>

#include "../SmallVector.h"
#include "Api.h"
#include "MemoryAllocatorVK.h"

namespace Ray {
class ILog;
struct gpu_device_t;
namespace Vk {
static const int MaxFramesInFlight = 6;
static const int MaxTimestampQueries = 1024;

class DescrMultiPoolAlloc;

using CommandBuffer = VkCommandBuffer;

class Context {
    ILog *log_ = nullptr;
    Api api_;
    bool external_ = false;
    VkInstance instance_ = {};
    VkDebugReportCallbackEXT debug_callback_ = {};
    VkPhysicalDevice physical_device_ = {};
    VkPhysicalDeviceLimits phys_device_limits_ = {};
    VkPhysicalDeviceProperties device_properties_ = {};
    VkPhysicalDeviceMemoryProperties mem_properties_ = {};
    uint32_t graphics_family_index_ = 0;

    VkDevice device_ = {};
    VkPipelineCache pipeline_cache_ = {};

    bool raytracing_supported_ = false, ray_query_supported_ = false;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props_ = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    bool rgb8_unorm_is_supported_ = false;

    bool fp16_supported_ = false;

    bool int64_supported_ = false;

    bool int64_atomics_supported_ = false;

    bool subgroup_supported_ = false;

    int coop_matrix_size_[3] = {-1, -1, -1};

    bool pageable_memory_supported_ = false;

    uint32_t supported_stages_mask_ = 0xffffffff;

    VkQueue graphics_queue_ = {};

    VkCommandPool command_pool_ = {}, temp_command_pool_ = {};
    VkCommandBuffer draw_cmd_bufs_[MaxFramesInFlight];

    VkSemaphore render_finished_semaphores_[MaxFramesInFlight] = {};
    VkFence in_flight_fences_[MaxFramesInFlight] = {};

    VkQueryPool query_pools_[MaxFramesInFlight] = {};
    uint32_t query_counts_[MaxFramesInFlight] = {};
    uint64_t query_results_[MaxFramesInFlight][MaxTimestampQueries] = {};

    uint32_t max_combined_image_samplers_ = 0, max_sampled_images_ = 0, max_samplers_ = 0;

    std::unique_ptr<MemAllocators> default_mem_allocs_;
    std::unique_ptr<DescrMultiPoolAlloc> default_descr_alloc_[MaxFramesInFlight];

  public:
    ~Context();

    bool Init(ILog *log, const VulkanDevice &vk_device, const VulkanFunctions &vk_functions,
              std::string_view preferred_device, int validation_level);
    void Destroy();

    VkDevice device() const { return device_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkPipelineCache pipeline_cache() const { return pipeline_cache_; }

    ILog *log() const { return log_; }
    const Api &api() const { return api_; }

    uint32_t max_combined_image_samplers() const { return max_combined_image_samplers_; }
    uint32_t max_sampled_images() const { return max_sampled_images_; }
    uint32_t max_samplers() const { return max_samplers_; }

    bool raytracing_supported() const { return raytracing_supported_; }
    bool ray_query_supported() const { return ray_query_supported_; }
    bool rgb8_unorm_is_supported() const { return rgb8_unorm_is_supported_; }
    bool fp16_supported() const { return fp16_supported_; }
    bool int64_supported() const { return int64_supported_; }
    bool int64_atomics_supported() const { return int64_atomics_supported_; }
    bool subgroup_supported() const { return subgroup_supported_; }
    const int *coop_matrix_size() const { return coop_matrix_size_; }

    uint32_t supported_stages_mask() const { return supported_stages_mask_; };
    bool image_blit_supported() const { return true; }

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
    VkQueryPool query_pool(const int i) const { return query_pools_[i]; }

    MemAllocators *default_mem_allocs() { return default_mem_allocs_.get(); }
    DescrMultiPoolAlloc *default_descr_alloc() const { return default_descr_alloc_[backend_frame].get(); }

    int WriteTimestamp(VkCommandBuffer cmd_buf, bool start);
    uint64_t GetTimestampIntervalDurationUs(int query_start, int query_end) const;

    bool ReadbackTimestampQueries(int i);
    void DestroyDeferredResources(int i);

    int backend_frame = 0;
    bool render_finished_semaphore_is_set[MaxFramesInFlight] = {};

    bool frame_cpu_synced[MaxFramesInFlight] = {};

    // resources scheduled for deferred destruction
    std::vector<VkImage> images_to_destroy[MaxFramesInFlight];
    std::vector<VkImageView> image_views_to_destroy[MaxFramesInFlight];
    std::vector<VkSampler> samplers_to_destroy[MaxFramesInFlight];
    std::vector<MemAllocation> allocs_to_free[MaxFramesInFlight];
    std::vector<VkBuffer> bufs_to_destroy[MaxFramesInFlight];
    std::vector<VkBufferView> buf_views_to_destroy[MaxFramesInFlight];
    std::vector<VkDeviceMemory> mem_to_free[MaxFramesInFlight];
    std::vector<VkRenderPass> render_passes_to_destroy[MaxFramesInFlight];
    std::vector<VkDescriptorPool> descriptor_pools_to_destroy[MaxFramesInFlight];
    std::vector<VkPipelineLayout> pipeline_layouts_to_destroy[MaxFramesInFlight];
    std::vector<VkPipeline> pipelines_to_destroy[MaxFramesInFlight];
    std::vector<VkAccelerationStructureKHR> acc_structs_to_destroy[MaxFramesInFlight];

    static int QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], int capacity);

  private:
    static bool InitVkInstance(const Api &api, VkInstance &instance, const char *enabled_layers[],
                               int enabled_layers_count, int validation_level, ILog *log);
    static bool ChooseVkPhysicalDevice(const Api &api, VkPhysicalDevice &physical_device,
                                       std::string_view preferred_device, VkInstance instance, ILog *log);
    static void CheckVkPhysicalDeviceFeatures(const Api &api, VkPhysicalDevice &physical_device,
                                              VkPhysicalDeviceProperties &device_properties,
                                              VkPhysicalDeviceMemoryProperties &mem_properties,
                                              uint32_t &graphics_family_index, bool &out_raytracing_supported,
                                              bool &out_ray_query_supported, bool &out_shader_fp16_supported,
                                              bool &out_shader_int64_supported, bool &out_int64_atomics_supported,
                                              int out_coop_matrix_size[3], bool &out_pageable_memory_supported);
    static bool InitVkDevice(const Api &api, VkDevice &device, VkPhysicalDevice physical_device,
                             uint32_t graphics_family_index, bool enable_raytracing, bool enable_ray_query,
                             bool enable_fp16, bool enable_int64, bool enable_int64_atomics, bool enable_coop_matrix,
                             bool enable_pageable_memory, ILog *log);
    static bool InitCommandBuffers(const Api &api, VkCommandPool &command_pool, VkCommandPool &temp_command_pool,
                                   VkCommandBuffer draw_cmd_bufs[MaxFramesInFlight],
                                   VkSemaphore render_finished_semaphores[MaxFramesInFlight],
                                   VkFence in_flight_fences[MaxFramesInFlight],
                                   VkQueryPool query_pools[MaxFramesInFlight], VkQueue &graphics_queue, VkDevice device,
                                   uint32_t graphics_family_index, ILog *log);
};

VkCommandBuffer BegSingleTimeCommands(const Api &api, VkDevice device, VkCommandPool temp_command_pool);
void EndSingleTimeCommands(const Api &api, VkDevice device, VkQueue cmd_queue, VkCommandBuffer command_buf,
                           VkCommandPool temp_command_pool);

void InsertReadbackMemoryBarrier(const Api &api, VkCommandBuffer cmd_buf);

} // namespace Vk
} // namespace Ray