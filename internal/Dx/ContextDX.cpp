#include "ContextDX.h"

#include <codecvt>

#include "../../Log.h"
#include "../../Types.h"
#include "../ScopeExit.h"
#include "../SmallVector.h"
#include "BufferDX.h"
#include "DescriptorPoolDX.h"
#include "MemoryAllocatorDX.h"

#include "../../third-party/renderdoc/renderdoc_app.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>

namespace Ray {
bool MatchDeviceNames(const char *name, const char *pattern);

extern const std::pair<uint32_t, const char *> KnownGPUVendors[];
extern const int KnownGPUVendorsCount;

extern RENDERDOC_DevicePointer g_rdoc_device;

namespace Dx {
void DebugReportCallback(D3D12_MESSAGE_CATEGORY Category, D3D12_MESSAGE_SEVERITY Severity, D3D12_MESSAGE_ID ID,
                         LPCSTR pDescription, void *pContext) {
    if (ID == D3D12_MESSAGE_ID_CORRUPTED_PARAMETER2) {
        return;
    }
    auto *ctx = reinterpret_cast<const Context *>(pContext);
    ctx->log()->Error("%s\n", pDescription);
}
} // namespace Dx
} // namespace Ray

Ray::Dx::Context::Context() = default;

Ray::Dx::Context::~Context() { Destroy(); }

void Ray::Dx::Context::Destroy() {

#define SAFE_RELEASE(p)                                                                                                \
    if ((p)) {                                                                                                         \
        (p)->Release();                                                                                                \
        (p) = 0;                                                                                                       \
    }

#ifndef NDEBUG
    if (debug_callback_cookie_) {
        ID3D12InfoQueue1 *info_queue;
        HRESULT hr = device_->QueryInterface(IID_PPV_ARGS(&info_queue));
        if (SUCCEEDED(hr)) {
            info_queue->UnregisterMessageCallback(debug_callback_cookie_);
            debug_callback_cookie_ = {};
        }
    }
#endif

    SAFE_RELEASE(device_);
    SAFE_RELEASE(command_queue_);
    SAFE_RELEASE(command_list_);

    for (int i = 0; i < MaxFramesInFlight; ++i) {
        SAFE_RELEASE(command_allocators_[i]);
        SAFE_RELEASE(in_flight_fences_[i]);
        SAFE_RELEASE(query_heaps_[i]);

        backend_frame = i; // default_descr_alloc_'s destructors rely on this
        default_descr_alloc_[i].reset();
        DestroyDeferredResources(i);
    }
    SAFE_RELEASE(temp_command_allocator_);

    CloseHandle(fence_event_);

#undef SAFE_RELEASE

    /*if (device_) {
        vkDeviceWaitIdle(device_);

        for (int i = 0; i < MaxFramesInFlight; ++i) {
            backend_frame = i; // default_descr_alloc_'s destructors rely on this
            default_descr_alloc_[i].reset();
            DestroyDeferredResources(i);

            vkDestroyFence(device_, in_flight_fences_[i], nullptr);
            vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
            vkDestroySemaphore(device_, image_avail_semaphores_[i], nullptr);

            vkDestroyQueryPool(device_, query_pools_[i], nullptr);
        }

        default_memory_allocs_.reset();

        vkFreeCommandBuffers(device_, command_pool_, 1, &setup_cmd_buf_);
        vkFreeCommandBuffers(device_, command_pool_, MaxFramesInFlight, draw_cmd_bufs_);

        // for (int i = 0; i < StageBufferCount; ++i) {
        //     default_stage_bufs_.fences[i].ClientWaitSync();
        //     default_stage_bufs_.fences[i] = {};
        //     default_stage_bufs_.bufs[i] = {};
        // }

        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyCommandPool(device_, temp_command_pool_, nullptr);

        // for (size_t i = 0; i < api_ctx_->present_image_views.size(); ++i) {
        //     vkDestroyImageView(api_ctx_->device, api_ctx_->present_image_views[i], nullptr);
        // }

        // vkDestroySwapchainKHR(device_, api_ctx_->swapchain, nullptr);

        vkDestroyDevice(device_, nullptr);
        // vkDestroySurfaceKHR(instance_, surface_, nullptr);
#ifndef NDEBUG
        vkDestroyDebugReportCallbackEXT(instance_, debug_callback_, nullptr);
#endif

        vkDestroyInstance(instance_, nullptr);
    }*/
}

bool Ray::Dx::Context::Init(ILog *log, const char *preferred_device) {
    log_ = log;

#ifndef NDEBUG
    { // Enable debug layer
        ID3D12Debug *debug_controller = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
            debug_controller->EnableDebugLayer();
        }
    }
#endif

    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> str_converter;

    IDXGIFactory4 *dxgi_factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
    if (FAILED(hr)) {
        return false;
    }

    IDXGIAdapter1 *adapter = nullptr, *best_adapter = nullptr;
    int adapter_index = 0;
    int best_adapter_score = 0;

    while (dxgi_factory->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        int adapter_score = 0;
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            adapter_score += 1000;
        }

        if (preferred_device) {
            if (MatchDeviceNames(str_converter.to_bytes(desc.Description).c_str(), preferred_device)) {
                adapter_score += 100000;
            }
        }

        if (adapter_score > best_adapter_score) {
            best_adapter = adapter;
            best_adapter_score = adapter_score;
        }

        ++adapter_index;
    }

    hr = D3D12CreateDevice(best_adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
    if (FAILED(hr)) {
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC cq_desc = {};
    cq_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    cq_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; // direct means the gpu can directly execute this command queue

    hr = device_->CreateCommandQueue(&cq_desc, IID_PPV_ARGS(&command_queue_)); // create the command queue
    if (FAILED(hr)) {
        return false;
    }

    for (int i = 0; i < MaxFramesInFlight; ++i) {
        hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocators_[i]));
        if (FAILED(hr)) {
            return false;
        }

        hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&in_flight_fences_[i]));
        if (FAILED(hr)) {
            return false;
        }

        D3D12_QUERY_HEAP_DESC heap_desc = {};
        heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap_desc.Count = MaxTimestampQueries;

        hr = device_->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&query_heaps_[i]));
        if (FAILED(hr)) {
            return false;
        }
    }

    fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) {
        return false;
    }

    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&temp_command_allocator_));
    if (FAILED(hr)) {
        return false;
    }

    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocators_[0], NULL,
                                    IID_PPV_ARGS(&command_list_));
    if (FAILED(hr)) {
        return false;
    }

    command_list_->Close();

#ifndef NDEBUG
    ID3D12InfoQueue1 *info_queue;
    hr = device_->QueryInterface(IID_PPV_ARGS(&info_queue));
    if (SUCCEEDED(hr)) {
        hr = info_queue->RegisterMessageCallback(DebugReportCallback,
                                                 D3D12_MESSAGE_CALLBACK_FLAGS::D3D12_MESSAGE_CALLBACK_FLAG_NONE, this,
                                                 &debug_callback_cookie_);
        if (FAILED(hr)) {
            log->Error("Failed to register message callback!");
        }
    }
#endif

    UINT64 timestamp_frequency = 0;
    hr = command_queue_->GetTimestampFrequency(&timestamp_frequency);
    if (FAILED(hr)) {
        return false;
    }
    timestamp_period_us_ = 1000000.0 / double(timestamp_frequency);

    // if (!ChooseVkPhysicalDevice(physical_device_, device_properties_, mem_properties_, graphics_family_index_,
    //                             raytracing_supported_, ray_query_supported_, dynamic_rendering_supported_,
    //                             preferred_device, instance_, log)) {
    //     return false;
    // }

    // Disable as it is not needed for now
    // dynamic_rendering_supported_ = false;

    // if (!InitVkDevice(device_, physical_device_, graphics_family_index_, raytracing_supported_, ray_query_supported_,
    //                   dynamic_rendering_supported_, g_enabled_layers, g_enabled_layers_count, log)) {
    //     return false;
    // }

    // Workaround for a buggy linux AMD driver, make sure vkGetBufferDeviceAddressKHR is not NULL
    // auto dev_vkGetBufferDeviceAddressKHR =
    //    (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device_, "vkGetBufferDeviceAddressKHR");
    // if (!dev_vkGetBufferDeviceAddressKHR) {
    //    raytracing_supported_ = ray_query_supported_ = false;
    //}

    // if (!InitCommandBuffers(command_pool_, temp_command_pool_, setup_cmd_buf_, draw_cmd_bufs_,
    // image_avail_semaphores_,
    //                         render_finished_semaphores_, in_flight_fences_, query_pools_, graphics_queue_, device_,
    //                         graphics_family_index_, log)) {
    //     return false;
    // }

    DXGI_ADAPTER_DESC1 desc;
    best_adapter->GetDesc1(&desc);

    device_name_ = str_converter.to_bytes(desc.Description);

    log_->Info("===========================================");
    log_->Info("Device info:");

    auto it = find_if(KnownGPUVendors, KnownGPUVendors + KnownGPUVendorsCount,
                      [&](std::pair<uint32_t, const char *> v) { return desc.VendorId == v.first; });
    if (it != KnownGPUVendors + KnownGPUVendorsCount) {
        log_->Info("\tVendor\t\t: %s", it->second);
    }

    log_->Info("\tName\t\t: %s", device_name_.c_str());

    log_->Info("===========================================");

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    hr = device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    if (SUCCEEDED(hr)) {
        if (options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1) {
            max_combined_image_samplers_ = 128;
        } else {
            max_combined_image_samplers_ = 16384;
        }
    }

    default_memory_allocs_ = std::make_unique<MemoryAllocators>(
        "Default Allocs", this, 32 * 1024 * 1024 /* initial_block_size */, 1.5f /* growth_factor */);

    for (int i = 0; i < MaxFramesInFlight; ++i) {
        query_readback_buf_[i] = std::make_unique<Buffer>("Query Readback Buf", this, eBufType::Readback,
                                                          uint32_t(sizeof(uint64_t) * MaxTimestampQueries));
        default_descr_alloc_[i] = std::make_unique<DescrMultiPoolAlloc>(this, 16 * 1024);
    }

    g_rdoc_device = device_;

    return true;
}

ID3D12GraphicsCommandList *Ray::Dx::BegSingleTimeCommands(ID3D12Device *device,
                                                          ID3D12CommandAllocator *temp_command_allocator) {
    HRESULT hr = temp_command_allocator->Reset();
    if (FAILED(hr)) {
        return nullptr;
    }

    ID3D12GraphicsCommandList *temp_command_list = nullptr;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, temp_command_allocator, NULL,
                                   IID_PPV_ARGS(&temp_command_list));
    if (FAILED(hr)) {
        return nullptr;
    }

    return temp_command_list;
}

void Ray::Dx::EndSingleTimeCommands(ID3D12Device *device, ID3D12CommandQueue *cmd_queue,
                                    ID3D12GraphicsCommandList *command_list,
                                    ID3D12CommandAllocator *temp_command_allocator) {
    HRESULT hr = command_list->Close();
    if (FAILED(hr)) {
        return;
    }

    ID3D12Fence *temp_fence = nullptr;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&temp_fence));
    if (FAILED(hr)) {
        return;
    }
    SCOPE_EXIT(temp_fence->Release());

    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event) {
        return;
    }
    SCOPE_EXIT(CloseHandle(fence_event));

    ID3D12CommandList *pp_command_lists[] = {command_list};
    cmd_queue->ExecuteCommandLists(1, pp_command_lists);

    UINT64 fence_value = 1;
    hr = cmd_queue->Signal(temp_fence, fence_value);
    if (FAILED(hr)) {
        return;
    }

    temp_fence->SetEventOnCompletion(fence_value, fence_event);

    WaitForSingleObject(fence_event, INFINITE);

    command_list->Release();
}

int Ray::Dx::Context::WriteTimestamp(ID3D12GraphicsCommandList *cmd_buf, const bool start) {
    const uint32_t query_index = query_counts_[backend_frame]++;
    assert(query_index < MaxTimestampQueries);
    cmd_buf->EndQuery(query_heaps_[backend_frame], D3D12_QUERY_TYPE_TIMESTAMP, query_index);
    return int(query_index);
}

uint64_t Ray::Dx::Context::GetTimestampIntervalDurationUs(const int query_beg, const int query_end) const {
    return uint64_t(double(query_results_[backend_frame][query_end] - query_results_[backend_frame][query_beg]) *
                    timestamp_period_us_);
}

void Ray::Dx::Context::ResolveTimestampQueries(const int i) {
    command_list_->ResolveQueryData(query_heaps_[i], D3D12_QUERY_TYPE_TIMESTAMP, 0, query_counts_[i],
                                    query_readback_buf_[i]->dx_resource(), 0);
}

bool Ray::Dx::Context::ReadbackTimestampQueries(const int i) {
    if (!query_counts_[i]) {
        return true;
    }

    uint8_t *mapped_ptr = query_readback_buf_[i]->Map();
    if (!mapped_ptr) {
        return false;
    }
    memcpy(&query_results_[i][0], mapped_ptr, query_counts_[i] * sizeof(uint64_t));
    query_readback_buf_[i]->Unmap();

    query_counts_[i] = 0;
    return true;
}

void Ray::Dx::Context::DestroyDeferredResources(const int i) {
    /*for (VkImageView view : image_views_to_destroy[i]) {
        vkDestroyImageView(device_, view, nullptr);
    }
    image_views_to_destroy[i].clear();
    for (VkImage img : images_to_destroy[i]) {
        vkDestroyImage(device_, img, nullptr);
    }
    images_to_destroy[i].clear();
    for (VkSampler sampler : samplers_to_destroy[i]) {
        vkDestroySampler(device_, sampler, nullptr);
    }
    samplers_to_destroy[i].clear();*/

    allocs_to_free[i].clear();

    /*for (VkBufferView view : buf_views_to_destroy[i]) {
        vkDestroyBufferView(device_, view, nullptr);
    }
    buf_views_to_destroy[i].clear();

    for (VkDeviceMemory mem : mem_to_free[i]) {
        vkFreeMemory(device_, mem, nullptr);
    }
    mem_to_free[i].clear();

    for (VkRenderPass rp : render_passes_to_destroy[i]) {
        vkDestroyRenderPass(device_, rp, nullptr);
    }
    render_passes_to_destroy[i].clear();

    for (VkFramebuffer fb : framebuffers_to_destroy[i]) {
        vkDestroyFramebuffer(device_, fb, nullptr);
    }
    framebuffers_to_destroy[i].clear();

    for (VkDescriptorPool pool : descriptor_pools_to_destroy[i]) {
        vkDestroyDescriptorPool(device_, pool, nullptr);
    }
    descriptor_pools_to_destroy[i].clear();

    for (VkPipelineLayout pipe_layout : pipeline_layouts_to_destroy[i]) {
        vkDestroyPipelineLayout(device_, pipe_layout, nullptr);
    }
    pipeline_layouts_to_destroy[i].clear();*/

    for (ID3D12Resource *res : resources_to_destroy[i]) {
        res->Release();
    }
    resources_to_destroy[i].clear();
    for (ID3D12PipelineState *pipe : pipelines_to_destroy[i]) {
        pipe->Release();
    }
    pipelines_to_destroy[i].clear();
    for (ID3D12DescriptorHeap *heap : descriptor_heaps_to_destroy[i]) {
        heap->Release();
    }
    descriptor_heaps_to_destroy[i].clear();
    for (IUnknown *unknown : opaques_to_release[i]) {
        unknown->Release();
    }
    opaques_to_release[i].clear();

    // for (VkAccelerationStructureKHR acc_struct : acc_structs_to_destroy[i]) {
    //     vkDestroyAccelerationStructureKHR(device_, acc_struct, nullptr);
    // }
    // acc_structs_to_destroy[i].clear();
}

#if 0
int Ray::Vk::Context::QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], const int capacity) {
    if (!LoadVulkan(log)) {
        log->Error("Failed to initialize vulkan!");
        return 0;
    }

    VkInstance instance;
    if (!InitVkInstance(instance, g_enabled_layers, g_enabled_layers_count, log)) {
        log->Error("Failed to initialize VkInstance!");
        return 0;
    }

    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);

    SmallVector<VkPhysicalDevice, 4> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, &physical_devices[0]);

    if (out_devices) {
        if (int(physical_device_count) > capacity) {
            log->Warning("Insufficiend devices copacity");
            physical_device_count = capacity;
        }

        for (int i = 0; i < int(physical_device_count); ++i) {
            VkPhysicalDeviceProperties device_properties = {};
            vkGetPhysicalDeviceProperties(physical_devices[i], &device_properties);

#pragma warning(suppress : 4996)
            strncpy(out_devices[i].name, device_properties.deviceName, sizeof(out_devices[i].name));
        }
    }
    vkDestroyInstance(instance, nullptr);

    return int(physical_device_count);
}
#endif