#pragma once

/**
  @file VulkanFunctions.h
*/

// Vulkan Forward Declarations
#define DEFINE_VK_HANDLE(object) typedef struct object##_T *object;
DEFINE_VK_HANDLE(VkInstance);
DEFINE_VK_HANDLE(VkPhysicalDevice);
DEFINE_VK_HANDLE(VkDevice);
DEFINE_VK_HANDLE(VkQueue);
DEFINE_VK_HANDLE(VkCommandPool);
DEFINE_VK_HANDLE(VkCommandBuffer);
DEFINE_VK_HANDLE(VkFence);
DEFINE_VK_HANDLE(VkSemaphore);
DEFINE_VK_HANDLE(VkQueryPool);
DEFINE_VK_HANDLE(VkShaderModule);
DEFINE_VK_HANDLE(VkDescriptorSetLayout);
DEFINE_VK_HANDLE(VkPipelineLayout);
DEFINE_VK_HANDLE(VkPipelineCache);
DEFINE_VK_HANDLE(VkPipeline);
DEFINE_VK_HANDLE(VkBuffer);
DEFINE_VK_HANDLE(VkBufferView);
DEFINE_VK_HANDLE(VkDeviceMemory);
DEFINE_VK_HANDLE(VkDescriptorSet);
DEFINE_VK_HANDLE(VkImage);
DEFINE_VK_HANDLE(VkImageView);
DEFINE_VK_HANDLE(VkSampler);
DEFINE_VK_HANDLE(VkDescriptorPool);

struct VkPhysicalDeviceFeatures;
struct VkPhysicalDeviceProperties;
struct VkPhysicalDeviceMemoryProperties;
struct VkFormatProperties;
struct VkImageFormatProperties;
struct VkQueueFamilyProperties;
struct VkExtensionProperties;
struct VkCommandPoolCreateInfo;
struct VkAllocationCallbacks;
struct VkCommandBufferAllocateInfo;
struct VkFenceCreateInfo;
struct VkSemaphoreCreateInfo;
struct VkQueryPoolCreateInfo;
struct VkShaderModuleCreateInfo;
struct VkDescriptorSetLayoutCreateInfo;
struct VkPipelineLayoutCreateInfo;
struct VkGraphicsPipelineCreateInfo;
struct VkComputePipelineCreateInfo;
struct VkBufferCreateInfo;
struct VkBufferViewCreateInfo;
struct VkMemoryRequirements;
struct VkMemoryAllocateInfo;
struct VkCommandBufferBeginInfo;
struct VkMemoryBarrier;
struct VkBufferMemoryBarrier;
struct VkImageMemoryBarrier;
struct VkBufferImageCopy;
struct VkBufferCopy;
struct VkImageBlit;
union VkClearColorValue;
struct VkImageSubresourceRange;
struct VkImageCopy;
struct VkSubmitInfo;
struct VkImageCreateInfo;
struct VkImageViewCreateInfo;
struct VkSamplerCreateInfo;
struct VkDescriptorPoolCreateInfo;
struct VkDescriptorSetAllocateInfo;
struct VkWriteDescriptorSet;
struct VkCopyDescriptorSet;

typedef uint32_t VkFlags;
typedef VkFlags VkImageUsageFlags;
typedef VkFlags VkImageCreateFlags;
typedef VkFlags VkQueryResultFlags;
typedef VkFlags VkMemoryMapFlags;
typedef VkFlags VkCommandBufferResetFlags;
typedef VkFlags VkPipelineStageFlags;
typedef VkFlags VkDependencyFlags;
typedef VkFlags VkShaderStageFlags;
typedef VkFlags VkDescriptorPoolResetFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;

#ifndef VKAPI_PTR
#if defined(_WIN32)
#define VKAPI_CALL __stdcall
#define VKAPI_PTR VKAPI_CALL
#else
#define VKAPI_PTR
#endif
#endif

namespace Ray {
typedef int32_t VkResult;
typedef void(VKAPI_PTR *PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction(VKAPI_PTR *PFN_vkGetInstanceProcAddr)(VkInstance instance, const char *pName);
typedef PFN_vkVoidFunction(VKAPI_PTR *PFN_vkGetDeviceProcAddr)(VkDevice device, const char *pName);
typedef void(VKAPI_PTR *PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice physicalDevice,
                                                           VkPhysicalDeviceProperties *pProperties);
typedef void(VKAPI_PTR *PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice physicalDevice,
                                                                 VkPhysicalDeviceMemoryProperties *pMemoryProperties);
typedef void(VKAPI_PTR *PFN_vkGetPhysicalDeviceFormatProperties)(VkPhysicalDevice physicalDevice, int32_t format,
                                                                 VkFormatProperties *pFormatProperties);
typedef VkResult(VKAPI_PTR *PFN_vkGetPhysicalDeviceImageFormatProperties)(
    VkPhysicalDevice physicalDevice, int32_t format, int32_t type, int32_t tiling, VkImageUsageFlags usage,
    VkImageCreateFlags flags, VkImageFormatProperties *pImageFormatProperties);
typedef void(VKAPI_PTR *PFN_vkGetPhysicalDeviceFeatures)(VkPhysicalDevice physicalDevice,
                                                         VkPhysicalDeviceFeatures *pFeatures);
typedef void(VKAPI_PTR *PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice physicalDevice,
                                                                      uint32_t *pQueueFamilyPropertyCount,
                                                                      VkQueueFamilyProperties *pQueueFamilyProperties);
typedef VkResult(VKAPI_PTR *PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice physicalDevice,
                                                                      const char *pLayerName, uint32_t *pPropertyCount,
                                                                      VkExtensionProperties *pProperties);
typedef void(VKAPI_PTR *PFN_vkGetDeviceQueue)(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex,
                                              VkQueue *pQueue);
typedef VkResult(VKAPI_PTR *PFN_vkCreateCommandPool)(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                                                     const VkAllocationCallbacks *pAllocator,
                                                     VkCommandPool *pCommandPool);
typedef void(VKAPI_PTR *PFN_vkDestroyCommandPool)(VkDevice device, VkCommandPool commandPool,
                                                  const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkAllocateCommandBuffers)(VkDevice device,
                                                          const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                          VkCommandBuffer *pCommandBuffers);
typedef void(VKAPI_PTR *PFN_vkFreeCommandBuffers)(VkDevice device, VkCommandPool commandPool,
                                                  uint32_t commandBufferCount, const VkCommandBuffer *pCommandBuffers);
typedef VkResult(VKAPI_PTR *PFN_vkCreateFence)(VkDevice device, const VkFenceCreateInfo *pCreateInfo,
                                               const VkAllocationCallbacks *pAllocator, VkFence *pFence);
typedef void(VKAPI_PTR *PFN_vkDestroyFence)(VkDevice device, VkFence fence, const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkResetFences)(VkDevice device, uint32_t fenceCount, const VkFence *pFences);
typedef VkResult(VKAPI_PTR *PFN_vkGetFenceStatus)(VkDevice device, VkFence fence);
typedef VkResult(VKAPI_PTR *PFN_vkWaitForFences)(VkDevice device, uint32_t fenceCount, const VkFence *pFences,
                                                 VkBool32 waitAll, uint64_t timeout);
typedef VkResult(VKAPI_PTR *PFN_vkCreateSemaphore)(VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore);
typedef void(VKAPI_PTR *PFN_vkDestroySemaphore)(VkDevice device, VkSemaphore semaphore,
                                                const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateQueryPool)(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool);
typedef void(VKAPI_PTR *PFN_vkDestroyQueryPool)(VkDevice device, VkQueryPool queryPool,
                                                const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkGetQueryPoolResults)(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                                                       uint32_t queryCount, size_t dataSize, void *pData,
                                                       VkDeviceSize stride, VkQueryResultFlags flags);
typedef VkResult(VKAPI_PTR *PFN_vkCreateShaderModule)(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                      const VkAllocationCallbacks *pAllocator,
                                                      VkShaderModule *pShaderModule);
typedef void(VKAPI_PTR *PFN_vkDestroyShaderModule)(VkDevice device, VkShaderModule shaderModule,
                                                   const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateDescriptorSetLayout)(VkDevice device,
                                                             const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                             const VkAllocationCallbacks *pAllocator,
                                                             VkDescriptorSetLayout *pSetLayout);
typedef void(VKAPI_PTR *PFN_vkDestroyDescriptorSetLayout)(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                          const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreatePipelineLayout)(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                        const VkAllocationCallbacks *pAllocator,
                                                        VkPipelineLayout *pPipelineLayout);
typedef void(VKAPI_PTR *PFN_vkDestroyPipelineLayout)(VkDevice device, VkPipelineLayout pipelineLayout,
                                                     const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateGraphicsPipelines)(VkDevice device, VkPipelineCache pipelineCache,
                                                           uint32_t createInfoCount,
                                                           const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                           const VkAllocationCallbacks *pAllocator,
                                                           VkPipeline *pPipelines);
typedef VkResult(VKAPI_PTR *PFN_vkCreateComputePipelines)(VkDevice device, VkPipelineCache pipelineCache,
                                                          uint32_t createInfoCount,
                                                          const VkComputePipelineCreateInfo *pCreateInfos,
                                                          const VkAllocationCallbacks *pAllocator,
                                                          VkPipeline *pPipelines);
typedef void(VKAPI_PTR *PFN_vkDestroyPipeline)(VkDevice device, VkPipeline pipeline,
                                               const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkAllocateMemory)(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory);
typedef void(VKAPI_PTR *PFN_vkFreeMemory)(VkDevice device, VkDeviceMemory memory,
                                          const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateBuffer)(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);
typedef void(VKAPI_PTR *PFN_vkDestroyBuffer)(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateBufferView)(VkDevice device, const VkBufferViewCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkBufferView *pView);
typedef void(VKAPI_PTR *PFN_vkDestroyBufferView)(VkDevice device, VkBufferView bufferView,
                                                 const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkBindBufferMemory)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                    VkDeviceSize memoryOffset);
typedef void(VKAPI_PTR *PFN_vkGetBufferMemoryRequirements)(VkDevice device, VkBuffer buffer,
                                                           VkMemoryRequirements *pMemoryRequirements);
typedef VkResult(VKAPI_PTR *PFN_vkMapMemory)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                             VkDeviceSize size, VkMemoryMapFlags flags, void **ppData);
typedef void(VKAPI_PTR *PFN_vkUnmapMemory)(VkDevice device, VkDeviceMemory memory);
typedef VkResult(VKAPI_PTR *PFN_vkBeginCommandBuffer)(VkCommandBuffer commandBuffer,
                                                      const VkCommandBufferBeginInfo *pBeginInfo);
typedef VkResult(VKAPI_PTR *PFN_vkEndCommandBuffer)(VkCommandBuffer commandBuffer);
typedef VkResult(VKAPI_PTR *PFN_vkResetCommandBuffer)(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);
typedef VkResult(VKAPI_PTR *PFN_vkQueueSubmit)(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits,
                                               VkFence fence);
typedef VkResult(VKAPI_PTR *PFN_vkQueueWaitIdle)(VkQueue queue);
typedef VkResult(VKAPI_PTR *PFN_vkCreateImage)(VkDevice device, const VkImageCreateInfo *pCreateInfo,
                                               const VkAllocationCallbacks *pAllocator, VkImage *pImage);
typedef void(VKAPI_PTR *PFN_vkDestroyImage)(VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator);
typedef void(VKAPI_PTR *PFN_vkGetImageMemoryRequirements)(VkDevice device, VkImage image,
                                                          VkMemoryRequirements *pMemoryRequirements);
typedef VkResult(VKAPI_PTR *PFN_vkBindImageMemory)(VkDevice device, VkImage image, VkDeviceMemory memory,
                                                   VkDeviceSize memoryOffset);
typedef VkResult(VKAPI_PTR *PFN_vkCreateImageView)(VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkImageView *pView);
typedef void(VKAPI_PTR *PFN_vkDestroyImageView)(VkDevice device, VkImageView imageView,
                                                const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateSampler)(VkDevice device, const VkSamplerCreateInfo *pCreateInfo,
                                                 const VkAllocationCallbacks *pAllocator, VkSampler *pSampler);
typedef void(VKAPI_PTR *PFN_vkDestroySampler)(VkDevice device, VkSampler sampler,
                                              const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkCreateDescriptorPool)(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                                        const VkAllocationCallbacks *pAllocator,
                                                        VkDescriptorPool *pDescriptorPool);
typedef void(VKAPI_PTR *PFN_vkDestroyDescriptorPool)(VkDevice device, VkDescriptorPool descriptorPool,
                                                     const VkAllocationCallbacks *pAllocator);
typedef VkResult(VKAPI_PTR *PFN_vkResetDescriptorPool)(VkDevice device, VkDescriptorPool descriptorPool,
                                                       VkDescriptorPoolResetFlags flags);
typedef VkResult(VKAPI_PTR *PFN_vkAllocateDescriptorSets)(VkDevice device,
                                                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                          VkDescriptorSet *pDescriptorSets);
typedef VkResult(VKAPI_PTR *PFN_vkFreeDescriptorSets)(VkDevice device, VkDescriptorPool descriptorPool,
                                                      uint32_t descriptorSetCount,
                                                      const VkDescriptorSet *pDescriptorSets);
typedef void(VKAPI_PTR *PFN_vkUpdateDescriptorSets)(VkDevice device, uint32_t descriptorWriteCount,
                                                    const VkWriteDescriptorSet *pDescriptorWrites,
                                                    uint32_t descriptorCopyCount,
                                                    const VkCopyDescriptorSet *pDescriptorCopies);

typedef void(VKAPI_PTR *PFN_vkCmdPipelineBarrier)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
                                                  VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
                                                  uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                                  uint32_t bufferMemoryBarrierCount,
                                                  const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                                  uint32_t imageMemoryBarrierCount,
                                                  const VkImageMemoryBarrier *pImageMemoryBarriers);
typedef void(VKAPI_PTR *PFN_vkCmdBindPipeline)(VkCommandBuffer commandBuffer, int32_t pipelineBindPoint,
                                               VkPipeline pipeline);
typedef void(VKAPI_PTR *PFN_vkCmdBindDescriptorSets)(VkCommandBuffer commandBuffer, int32_t pipelineBindPoint,
                                                     VkPipelineLayout layout, uint32_t firstSet,
                                                     uint32_t descriptorSetCount,
                                                     const VkDescriptorSet *pDescriptorSets,
                                                     uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets);
typedef void(VKAPI_PTR *PFN_vkCmdBindIndexBuffer)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                  int32_t indexType);
typedef void(VKAPI_PTR *PFN_vkCmdBindVertexBuffers)(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                                                    uint32_t bindingCount, const VkBuffer *pBuffers,
                                                    const VkDeviceSize *pOffsets);
typedef void(VKAPI_PTR *PFN_vkCmdCopyBufferToImage)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
                                                    int32_t dstImageLayout, uint32_t regionCount,
                                                    const VkBufferImageCopy *pRegions);
typedef void(VKAPI_PTR *PFN_vkCmdCopyImageToBuffer)(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                    int32_t srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount,
                                                    const VkBufferImageCopy *pRegions);
typedef void(VKAPI_PTR *PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                                             uint32_t regionCount, const VkBufferCopy *pRegions);
typedef void(VKAPI_PTR *PFN_vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                             VkDeviceSize size, uint32_t data);
typedef void(VKAPI_PTR *PFN_vkCmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                                               VkDeviceSize dstOffset, VkDeviceSize dataSize, const void *pData);
typedef void(VKAPI_PTR *PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                                VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size,
                                                const void *pValues);
typedef void(VKAPI_PTR *PFN_vkCmdBlitImage)(VkCommandBuffer commandBuffer, VkImage srcImage, int32_t srcImageLayout,
                                            VkImage dstImage, int32_t dstImageLayout, uint32_t regionCount,
                                            const VkImageBlit *pRegions, int32_t filter);
typedef void(VKAPI_PTR *PFN_vkCmdClearColorImage)(VkCommandBuffer commandBuffer, VkImage image, int32_t imageLayout,
                                                  const VkClearColorValue *pColor, uint32_t rangeCount,
                                                  const VkImageSubresourceRange *pRanges);
typedef void(VKAPI_PTR *PFN_vkCmdCopyImage)(VkCommandBuffer commandBuffer, VkImage srcImage, int32_t srcImageLayout,
                                            VkImage dstImage, int32_t dstImageLayout, uint32_t regionCount,
                                            const VkImageCopy *pRegions);
typedef void(VKAPI_PTR *PFN_vkCmdDispatch)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY,
                                           uint32_t groupCountZ);
typedef void(VKAPI_PTR *PFN_vkCmdDispatchIndirect)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
typedef void(VKAPI_PTR *PFN_vkCmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                                                 uint32_t firstQuery, uint32_t queryCount);
typedef void(VKAPI_PTR *PFN_vkCmdWriteTimestamp)(VkCommandBuffer commandBuffer, int32_t pipelineStage,
                                                 VkQueryPool queryPool, uint32_t query);

struct VulkanDevice {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
};
struct VulkanFunctions {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties;
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
    PFN_vkGetDeviceQueue vkGetDeviceQueue;

    PFN_vkCreateCommandPool vkCreateCommandPool;
    PFN_vkDestroyCommandPool vkDestroyCommandPool;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers;

    PFN_vkCreateFence vkCreateFence;
    PFN_vkResetFences vkResetFences;
    PFN_vkDestroyFence vkDestroyFence;
    PFN_vkGetFenceStatus vkGetFenceStatus;
    PFN_vkWaitForFences vkWaitForFences;

    PFN_vkCreateSemaphore vkCreateSemaphore;
    PFN_vkDestroySemaphore vkDestroySemaphore;

    PFN_vkCreateQueryPool vkCreateQueryPool;
    PFN_vkDestroyQueryPool vkDestroyQueryPool;
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults;

    PFN_vkCreateShaderModule vkCreateShaderModule;
    PFN_vkDestroyShaderModule vkDestroyShaderModule;

    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;

    PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;

    PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines;
    PFN_vkCreateComputePipelines vkCreateComputePipelines;
    PFN_vkDestroyPipeline vkDestroyPipeline;

    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkFreeMemory vkFreeMemory;

    PFN_vkCreateBuffer vkCreateBuffer;
    PFN_vkDestroyBuffer vkDestroyBuffer;
    PFN_vkBindBufferMemory vkBindBufferMemory;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;

    PFN_vkCreateBufferView vkCreateBufferView;
    PFN_vkDestroyBufferView vkDestroyBufferView;

    PFN_vkMapMemory vkMapMemory;
    PFN_vkUnmapMemory vkUnmapMemory;

    PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
    PFN_vkResetCommandBuffer vkResetCommandBuffer;

    PFN_vkQueueSubmit vkQueueSubmit;
    PFN_vkQueueWaitIdle vkQueueWaitIdle;

    PFN_vkCreateImage vkCreateImage;
    PFN_vkDestroyImage vkDestroyImage;
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    PFN_vkBindImageMemory vkBindImageMemory;

    PFN_vkCreateImageView vkCreateImageView;
    PFN_vkDestroyImageView vkDestroyImageView;

    PFN_vkCreateSampler vkCreateSampler;
    PFN_vkDestroySampler vkDestroySampler;

    PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
    PFN_vkResetDescriptorPool vkResetDescriptorPool;

    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;

    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
    PFN_vkCmdBindPipeline vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
    PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers;
    PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer;
    PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage;
    PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    PFN_vkCmdFillBuffer vkCmdFillBuffer;
    PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer;
    PFN_vkCmdPushConstants vkCmdPushConstants;
    PFN_vkCmdBlitImage vkCmdBlitImage;
    PFN_vkCmdClearColorImage vkCmdClearColorImage;
    PFN_vkCmdCopyImage vkCmdCopyImage;
    PFN_vkCmdDispatch vkCmdDispatch;
    PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect;
    PFN_vkCmdResetQueryPool vkCmdResetQueryPool;
    PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;
};

#undef DEFINE_VK_HANDLE
} // namespace Ray