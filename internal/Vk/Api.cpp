#include "Api.h"

#include "../../Log.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

bool Ray::Vk::Api::Load(ILog *log) {
#if defined(_WIN32)
    vulkan_module = LoadLibrary("vulkan-1.dll");
    if (!vulkan_module) {
        log->Error("Failed to load vulkan-1.dll");
        return false;
    }

#define LOAD_VK_FUN(x)                                                                                                 \
    x = (decltype(x))GetProcAddress(vulkan_module, #x);                                                                \
    if (!(x)) {                                                                                                        \
        log->Error("Failed to load %s", #x);                                                                           \
        return false;                                                                                                  \
    }
#elif defined(__linux__)
    vulkan_module = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (!vulkan_module) {
        log->Error("Failed to load libvulkan.so");
        return false;
    }

#define LOAD_VK_FUN(x)                                                                                                 \
    x = (decltype(x))dlsym(vulkan_module, #x);                                                                             \
    if (!(x)) {                                                                                                        \
        log->Error("Failed to load %s", #x);                                                                           \
        return false;                                                                                                  \
    }
#else

#if defined(VK_USE_PLATFORM_IOS_MVK)
    vulkan_module = dlopen("libMoltenVK.dylib", RTLD_LAZY);
    if (!vulkan_module) {
        log->Error("Failed to load libMoltenVK.dylib");
        return false;
    }
#else
    vulkan_module = dlopen("libvulkan.dylib", RTLD_LAZY);
    if (!vulkan_module) {
        log->Error("Failed to load libvulkan.dylib");
        return false;
    }
#endif

#define LOAD_VK_FUN(x)                                                                                                 \
    x = (decltype(x))dlsym(vulkan_module, #x);                                                                             \
    if (!(x)) {                                                                                                        \
        log->Error("Failed to load %s", #x);                                                                           \
        return false;                                                                                                  \
    }
#endif

    LOAD_VK_FUN(vkCreateInstance)
    LOAD_VK_FUN(vkDestroyInstance)

    LOAD_VK_FUN(vkEnumerateInstanceLayerProperties)
    LOAD_VK_FUN(vkEnumerateInstanceExtensionProperties)
    LOAD_VK_FUN(vkGetInstanceProcAddr)
    LOAD_VK_FUN(vkGetDeviceProcAddr)

    LOAD_VK_FUN(vkEnumeratePhysicalDevices)
    LOAD_VK_FUN(vkGetPhysicalDeviceProperties)
    LOAD_VK_FUN(vkGetPhysicalDeviceFeatures)
    LOAD_VK_FUN(vkGetPhysicalDeviceQueueFamilyProperties)

    LOAD_VK_FUN(vkCreateDevice)
    LOAD_VK_FUN(vkDestroyDevice)

    LOAD_VK_FUN(vkEnumerateDeviceExtensionProperties)

    LOAD_VK_FUN(vkGetDeviceQueue)
    LOAD_VK_FUN(vkCreateCommandPool)
    LOAD_VK_FUN(vkDestroyCommandPool)

    LOAD_VK_FUN(vkAllocateCommandBuffers)
    LOAD_VK_FUN(vkFreeCommandBuffers)

    LOAD_VK_FUN(vkCreateFence)
    LOAD_VK_FUN(vkWaitForFences)
    LOAD_VK_FUN(vkResetFences)
    LOAD_VK_FUN(vkDestroyFence)
    LOAD_VK_FUN(vkGetFenceStatus)

    LOAD_VK_FUN(vkBeginCommandBuffer)
    LOAD_VK_FUN(vkEndCommandBuffer)
    LOAD_VK_FUN(vkResetCommandBuffer)
    LOAD_VK_FUN(vkCmdPipelineBarrier)

    LOAD_VK_FUN(vkQueueSubmit)
    LOAD_VK_FUN(vkQueueWaitIdle)

    LOAD_VK_FUN(vkCreateImageView)
    LOAD_VK_FUN(vkDestroyImageView)

    LOAD_VK_FUN(vkGetPhysicalDeviceMemoryProperties)
    LOAD_VK_FUN(vkGetPhysicalDeviceFormatProperties)
    LOAD_VK_FUN(vkGetPhysicalDeviceImageFormatProperties)

    LOAD_VK_FUN(vkCreateImage)
    LOAD_VK_FUN(vkDestroyImage)

    LOAD_VK_FUN(vkGetImageMemoryRequirements)
    LOAD_VK_FUN(vkAllocateMemory)
    LOAD_VK_FUN(vkFreeMemory)
    LOAD_VK_FUN(vkBindImageMemory)

    LOAD_VK_FUN(vkCreateRenderPass)
    LOAD_VK_FUN(vkDestroyRenderPass)

    LOAD_VK_FUN(vkCreateBuffer)
    LOAD_VK_FUN(vkGetBufferMemoryRequirements)
    LOAD_VK_FUN(vkBindBufferMemory)
    LOAD_VK_FUN(vkDestroyBuffer)

    LOAD_VK_FUN(vkCreateBufferView)
    LOAD_VK_FUN(vkDestroyBufferView)

    LOAD_VK_FUN(vkMapMemory)
    LOAD_VK_FUN(vkUnmapMemory)

    LOAD_VK_FUN(vkCreateShaderModule)
    LOAD_VK_FUN(vkDestroyShaderModule)
    LOAD_VK_FUN(vkCreateDescriptorSetLayout)
    LOAD_VK_FUN(vkDestroyDescriptorSetLayout)
    LOAD_VK_FUN(vkCreatePipelineLayout)
    LOAD_VK_FUN(vkDestroyPipelineLayout)

    LOAD_VK_FUN(vkCreateGraphicsPipelines)
    LOAD_VK_FUN(vkCreateComputePipelines)
    LOAD_VK_FUN(vkDestroyPipeline)

    LOAD_VK_FUN(vkCreateSemaphore)
    LOAD_VK_FUN(vkDestroySemaphore)
    LOAD_VK_FUN(vkCreateSampler)
    LOAD_VK_FUN(vkDestroySampler)

    LOAD_VK_FUN(vkCreateDescriptorPool)
    LOAD_VK_FUN(vkDestroyDescriptorPool)
    LOAD_VK_FUN(vkResetDescriptorPool)

    LOAD_VK_FUN(vkAllocateDescriptorSets)
    LOAD_VK_FUN(vkFreeDescriptorSets)
    LOAD_VK_FUN(vkUpdateDescriptorSets)

    LOAD_VK_FUN(vkCreateQueryPool)
    LOAD_VK_FUN(vkDestroyQueryPool)
    LOAD_VK_FUN(vkGetQueryPoolResults)

    LOAD_VK_FUN(vkCmdBindPipeline)
    LOAD_VK_FUN(vkCmdBindDescriptorSets)
    LOAD_VK_FUN(vkCmdBindVertexBuffers)
    LOAD_VK_FUN(vkCmdBindIndexBuffer)
    LOAD_VK_FUN(vkCmdCopyBufferToImage)
    LOAD_VK_FUN(vkCmdCopyImageToBuffer)
    LOAD_VK_FUN(vkCmdCopyBuffer)
    LOAD_VK_FUN(vkCmdFillBuffer)
    LOAD_VK_FUN(vkCmdUpdateBuffer)
    LOAD_VK_FUN(vkCmdPushConstants)
    LOAD_VK_FUN(vkCmdBlitImage)
    LOAD_VK_FUN(vkCmdClearColorImage)
    LOAD_VK_FUN(vkCmdCopyImage)
    LOAD_VK_FUN(vkCmdDispatch)
    LOAD_VK_FUN(vkCmdDispatchIndirect)
    LOAD_VK_FUN(vkCmdResetQueryPool)
    LOAD_VK_FUN(vkCmdWriteTimestamp)

#undef LOAD_VK_FUN

    return true;
}

bool Ray::Vk::Api::LoadExtensions(VkInstance instance, ILog *log) {
#define LOAD_INSTANCE_FUN(x)                                                                                           \
    x = (PFN_##x)vkGetInstanceProcAddr(instance, #x);                                                                 \
    if (!(x)) {                                                                                                        \
        log->Error("Failed to load %s", #x);                                                                           \
        return false;                                                                                                  \
    }
#define LOAD_OPTIONAL_INSTANCE_FUN(x) x = (PFN_##x)vkGetInstanceProcAddr(instance, #x);

    LOAD_INSTANCE_FUN(vkCreateDebugReportCallbackEXT)
    LOAD_INSTANCE_FUN(vkDestroyDebugReportCallbackEXT)
    LOAD_INSTANCE_FUN(vkDebugReportMessageEXT)

    LOAD_INSTANCE_FUN(vkCreateAccelerationStructureKHR)
    LOAD_INSTANCE_FUN(vkDestroyAccelerationStructureKHR)

    LOAD_INSTANCE_FUN(vkCmdBeginDebugUtilsLabelEXT)
    LOAD_INSTANCE_FUN(vkCmdEndDebugUtilsLabelEXT)
    LOAD_INSTANCE_FUN(vkSetDebugUtilsObjectNameEXT)

    LOAD_INSTANCE_FUN(vkCmdBuildAccelerationStructuresKHR)
    LOAD_INSTANCE_FUN(vkCmdWriteAccelerationStructuresPropertiesKHR)
    LOAD_INSTANCE_FUN(vkCmdCopyAccelerationStructureKHR)
    LOAD_INSTANCE_FUN(vkCmdTraceRaysKHR)
    LOAD_INSTANCE_FUN(vkCmdTraceRaysIndirectKHR)

    LOAD_INSTANCE_FUN(vkDeviceWaitIdle)

    LOAD_INSTANCE_FUN(vkGetPhysicalDeviceProperties2KHR)
    LOAD_INSTANCE_FUN(vkGetPhysicalDeviceFeatures2KHR);
    LOAD_INSTANCE_FUN(vkGetBufferDeviceAddressKHR)
    LOAD_INSTANCE_FUN(vkGetAccelerationStructureBuildSizesKHR)
    LOAD_INSTANCE_FUN(vkGetAccelerationStructureDeviceAddressKHR)
    LOAD_INSTANCE_FUN(vkGetRayTracingShaderGroupHandlesKHR)

    // allowed to fail
    LOAD_OPTIONAL_INSTANCE_FUN(vkCreateRayTracingPipelinesKHR)
    LOAD_OPTIONAL_INSTANCE_FUN(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

#undef LOAD_INSTANCE_FUN
#undef LOAD_OPTIONAL_INSTANCE_FUN

    return true;
}

Ray::Vk::Api::~Api() {
    if (vulkan_module) {
#if defined(_WIN32)
        FreeLibrary(vulkan_module);
#else
        dlclose(vulkan_module);
#endif
        vulkan_module = {};
    }
}