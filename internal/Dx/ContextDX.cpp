#include "ContextDX.h"

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
bool MatchDeviceNames(std::string_view name, std::string_view pattern);

extern const std::pair<uint32_t, std::string_view> KnownGPUVendors[];
extern const int KnownGPUVendorsCount;

extern RENDERDOC_DevicePointer g_rdoc_device;

namespace Dx {
void DebugReportCallback(D3D12_MESSAGE_CATEGORY Category, D3D12_MESSAGE_SEVERITY Severity, D3D12_MESSAGE_ID ID,
                         LPCSTR pDescription, void *pContext) {
    if (ID == D3D12_MESSAGE_ID_CORRUPTED_PARAMETER2) {
        return;
    }
    if (ID == D3D12_MESSAGE_ID_LIVE_DEVICE && strstr(pDescription, "Refcount: 2")) {
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

    for (int i = 0; i < MaxFramesInFlight; ++i) {
        backend_frame = (backend_frame + 1) % MaxFramesInFlight;

        SAFE_RELEASE(command_allocators_[backend_frame]);
        SAFE_RELEASE(in_flight_fences_[backend_frame]);
        SAFE_RELEASE(query_heaps_[backend_frame]);

        if (uniform_data_bufs[backend_frame]) {
            uniform_data_bufs[backend_frame].Unmap();
            uniform_data_bufs[backend_frame].Free();
        }

        default_descr_alloc_[backend_frame] = {};
        query_readback_buf_[backend_frame] = {};
        DestroyDeferredResources(backend_frame);
    }

    default_mem_allocs_ = {};
    staging_descr_alloc_ = {};

    SAFE_RELEASE(command_queue_);
    SAFE_RELEASE(command_list_);
    SAFE_RELEASE(temp_command_allocator_);
    SAFE_RELEASE(indirect_dispatch_cmd_signature_);

    SAFE_RELEASE(device5_);

    if (validation_level_) {
        ID3D12DebugDevice *debug_device = nullptr;
        if (device_ && SUCCEEDED(device_->QueryInterface(IID_PPV_ARGS(&debug_device)))) {
            debug_device->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL | D3D12_RLDO_IGNORE_INTERNAL);
            debug_device->Release();
        }
    }

    if (debug_callback_cookie_) {
        ID3D12InfoQueue1 *info_queue = {};
        HRESULT hr = device_->QueryInterface(IID_PPV_ARGS(&info_queue));
        if (SUCCEEDED(hr)) {
            info_queue->UnregisterMessageCallback(debug_callback_cookie_);
            debug_callback_cookie_ = {};
            info_queue->Release();
        }
    }

    SAFE_RELEASE(device_);

    CloseHandle(fence_event_);

#undef SAFE_RELEASE
}

bool Ray::Dx::Context::Init(ILog *log, std::string_view preferred_device, const int validation_level) {
    log_ = log;
    validation_level_ = validation_level;

    if (validation_level) { // Enable debug layer
        ID3D12Debug *debug_controller = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
            debug_controller->EnableDebugLayer();
        }
    }

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

        const int utf8_len = WideCharToMultiByte(CP_UTF8,          // Code page
                                                 0,                // Flags
                                                 desc.Description, // Wide char string to convert
                                                 -1,               // Null-terminated input
                                                 nullptr,          // Output buffer
                                                 0,                // Output buffer size (0 means we want the size)
                                                 nullptr, nullptr);

        std::string device_name(utf8_len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, device_name.data(), utf8_len, nullptr, nullptr);
        device_name.pop_back();

        if (!preferred_device.empty() && MatchDeviceNames(device_name, preferred_device)) {
            adapter_score += 100000;
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

    if (validation_level) {
        ID3D12InfoQueue1 *info_queue = {};
        hr = device_->QueryInterface(IID_PPV_ARGS(&info_queue));
        if (SUCCEEDED(hr)) {
            hr = info_queue->RegisterMessageCallback(DebugReportCallback,
                                                     D3D12_MESSAGE_CALLBACK_FLAGS::D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                                     this, &debug_callback_cookie_);
            if (FAILED(hr)) {
                log->Error("Failed to register message callback!");
            }
            info_queue->Release();
        }
    }

    UINT64 timestamp_frequency = 0;
    hr = command_queue_->GetTimestampFrequency(&timestamp_frequency);
    if (FAILED(hr)) {
        return false;
    }
    timestamp_period_us_ = 1000000.0 / double(timestamp_frequency);

    // Destroy();

    DXGI_ADAPTER_DESC1 desc;
    best_adapter->GetDesc1(&desc);

    { // Get device name
        const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);

        device_name_.resize(utf8_len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, device_name_.data(), utf8_len, nullptr, nullptr);
        device_name_.pop_back();
    }

    log_->Info("============================================================================");
    log_->Info("Device info:");

    auto it = find_if(KnownGPUVendors, KnownGPUVendors + KnownGPUVendorsCount,
                      [&](std::pair<uint32_t, std::string_view> v) { return desc.VendorId == v.first; });
    if (it != KnownGPUVendors + KnownGPUVendorsCount) {
        log_->Info("\tVendor\t\t: %s", it->second.data());
    }

    log_->Info("\tName\t\t: %s", device_name_.c_str());
    log_->Info("============================================================================");

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    hr = device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    if (SUCCEEDED(hr)) {
        if (options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1) {
            max_sampled_images_ = 128;
            max_samplers_ = 16;
        } else {
            max_sampled_images_ = 16384; // made-up limitation
            max_samplers_ = 2048;
        }
    }
    max_combined_image_samplers_ = std::min(max_sampled_images_, max_samplers_);

    ID3D12Device1 *device1 = {};
    hr = device_->QueryInterface(IID_PPV_ARGS(&device1));
    if (SUCCEEDED(hr)) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
        hr = device1->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
        if (SUCCEEDED(hr)) {
            if (options1.Int64ShaderOps == TRUE) {
                int64_supported_ = true;
            }
        }
        device1->Release();
    }

    ID3D12Device4 *device4 = {};
    hr = device_->QueryInterface(IID_PPV_ARGS(&device4));
    if (SUCCEEDED(hr)) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {};
        hr = device4->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4));
        if (SUCCEEDED(hr)) {
            if (options4.Native16BitShaderOpsSupported == TRUE) {
                fp16_supported_ = true;
            }
        }
        device4->Release();
    }

    hr = device_->QueryInterface(IID_PPV_ARGS(&device5_));
    if (SUCCEEDED(hr)) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
        hr = device5_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
        if (SUCCEEDED(hr)) {
            if (options5.RaytracingTier > D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
                raytracing_supported_ = ray_query_supported_ = true;
            }
        }
    }

    ID3D12Device9 *device9 = {};
    hr = device_->QueryInterface(IID_PPV_ARGS(&device9));
    if (SUCCEEDED(hr)) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS9 feature_support = {};
        hr = device9->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &feature_support, sizeof(feature_support));
        if (SUCCEEDED(hr)) {
            if (feature_support.AtomicInt64OnTypedResourceSupported == TRUE) {
                int64_atomics_supported_ = true;
            }
        }
        device9->Release();
    }

    subgroup_supported_ = true;

    { // check shader model support
        D3D12_FEATURE_DATA_SHADER_MODEL supported_shader_models = {};

        supported_shader_models.HighestShaderModel = D3D_SHADER_MODEL_6_0;
        hr = device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &supported_shader_models,
                                          sizeof(supported_shader_models));
        if (FAILED(hr) || supported_shader_models.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
            subgroup_supported_ = false;
            int64_supported_ = false;
            return false;
        }

        supported_shader_models.HighestShaderModel = D3D_SHADER_MODEL_6_2;
        hr = device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &supported_shader_models,
                                          sizeof(supported_shader_models));
        if (FAILED(hr) || supported_shader_models.HighestShaderModel < D3D_SHADER_MODEL_6_2) {
            fp16_supported_ = false;
        }

        supported_shader_models.HighestShaderModel = D3D_SHADER_MODEL_6_5;
        hr = device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &supported_shader_models,
                                          sizeof(supported_shader_models));
        if (FAILED(hr) || supported_shader_models.HighestShaderModel < D3D_SHADER_MODEL_6_5) {
            raytracing_supported_ = ray_query_supported_ = false;
        }

        supported_shader_models.HighestShaderModel = D3D_SHADER_MODEL_6_6;
        hr = device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &supported_shader_models,
                                          sizeof(supported_shader_models));
        if (FAILED(hr) || supported_shader_models.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
            int64_atomics_supported_ = false;
        }
    }

    { // check wave ops support
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
        hr = device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1,
                                          sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS1));
        if (FAILED(hr)) {
            subgroup_supported_ = false;
        }

        subgroup_supported_ = (options1.WaveOps == TRUE);
    }

    default_mem_allocs_ =
        std::make_unique<MemAllocators>("Default Allocs", this, 32 * 1024 * 1024 /* initial_pool_size */,
                                        1.5f /* growth_factor */, 128 * 1024 * 1024 /* max_pool_size */);
    staging_descr_alloc_ = std::make_unique<DescrMultiPoolAlloc<FreelistAllocAdapted>>(this, false, 16 * 1024);

    for (int i = 0; i < MaxFramesInFlight; ++i) {
        uniform_data_bufs[i] = Buffer{"Uniform data buf", this, eBufType::Upload, 1 * 1024 * 1024};
        uniform_data_bufs[i].Map(true /* persistent */);
        uniform_data_buf_offs[i] = 0;

        query_readback_buf_[i] = std::make_unique<Buffer>("Query Readback Buf", this, eBufType::Readback,
                                                          uint32_t(sizeof(uint64_t) * MaxTimestampQueries));
        default_descr_alloc_[i] = std::make_unique<DescrMultiPoolAlloc<BumpAlloc>>(this, true, 16 * 1024);
    }

    { // create indirect dispatch signature
        D3D12_INDIRECT_ARGUMENT_DESC indir_arg_desc = {};
        indir_arg_desc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

        D3D12_COMMAND_SIGNATURE_DESC cmd_signature_desc = {};
        cmd_signature_desc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        cmd_signature_desc.NumArgumentDescs = 1;
        cmd_signature_desc.pArgumentDescs = &indir_arg_desc;

        hr = device_->CreateCommandSignature(&cmd_signature_desc, nullptr,
                                             IID_PPV_ARGS(&indirect_dispatch_cmd_signature_));
        if (FAILED(hr)) {
            log_->Error("Failed to create command signature!");
            return false;
        }
    }

    g_rdoc_device = device_;

    return true;
}

ID3D12GraphicsCommandList *Ray::Dx::BegSingleTimeCommands(void *api, ID3D12Device *device,
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

void Ray::Dx::EndSingleTimeCommands(void *api, ID3D12Device *device, ID3D12CommandQueue *cmd_queue,
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

void Ray::Dx::InsertReadbackMemoryBarrier(void *api, CommandBuffer command_list) {}

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
    allocs_to_free[i].clear();

    for (ID3D12Resource *res : resources_to_destroy[i]) {
        res->Release();
    }
    resources_to_destroy[i].clear();
    for (ID3D12PipelineState *pipe : pipelines_to_destroy[i]) {
        pipe->Release();
    }
    pipelines_to_destroy[i].clear();
    for (ID3D12DescriptorHeap *heap : descriptor_heaps_to_release[i]) {
        heap->Release();
    }
    descriptor_heaps_to_release[i].clear();
    for (IUnknown *unknown : opaques_to_release[i]) {
        unknown->Release();
    }
    opaques_to_release[i].clear();
}

int Ray::Dx::Context::QueryAvailableDevices(ILog *log, gpu_device_t out_devices[], const int capacity) {
    IDXGIFactory4 *dxgi_factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
    if (FAILED(hr)) {
        return 0;
    }

    IDXGIAdapter1 *adapter = nullptr;
    int adapter_index = 0;

    int count = 0;
    while (dxgi_factory->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            ++adapter_index;
            continue;
        }

        const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);

        std::string device_name(utf8_len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, device_name.data(), utf8_len, nullptr, nullptr);
        device_name.pop_back();

        strncpy_s(out_devices[count].name, sizeof(out_devices[count].name), device_name.c_str(),
                  sizeof(out_devices[count].name) - 1);
        ++count;

        ++adapter_index;
    }

    return count;
}
