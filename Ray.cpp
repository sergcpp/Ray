#include "Ray.h"

#include <regex>

#include "internal/Core.h"

#ifdef ENABLE_REF_IMPL
#include "internal/RendererRef.h"
#else // ENABLE_REF_IMPL
#pragma message("Compiling without reference backend")
#endif // ENABLE_REF_IMPL

#ifdef ENABLE_SIMD_IMPL
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#include "internal/RendererAVX.h"
#include "internal/RendererAVX2.h"
#include "internal/RendererAVX512.h"
#include "internal/RendererSSE2.h"
#include "internal/RendererSSE41.h"
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include "internal/RendererNEON.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE2.h"
#endif
#else // ENABLE_SIMD_IMPL
#pragma message("Compiling without SIMD support")
#endif // #ifdef ENABLE_SIMD_IMPL

#if defined(ENABLE_DX_IMPL) && defined(_WIN32)
#include "internal/RendererDX.h"
#endif // defined(ENABLE_DX_IMPL) && defined(_WIN32)
#ifdef ENABLE_VK_IMPL
#include "internal/RendererVK.h"
#endif // ENABLE_VK_IMPL

#if !(defined(ENABLE_DX_IMPL) && defined(_WIN32)) && !defined(ENABLE_VK_IMPL)
#pragma message("Ray: Compiling without GPU support")
#endif

#include "internal/simd/detect.h"

#include "third-party/renderdoc/renderdoc_app.h"

namespace Ray {
LogNull g_null_log;
LogStdout g_stdout_log;
RENDERDOC_DevicePointer g_rdoc_device = {};

extern const std::pair<uint32_t, const char *> KnownGPUVendors[] = {
    {0x1002, "AMD"}, {0x10DE, "NVIDIA"}, {0x8086, "INTEL"}, {0x13B5, "ARM"}};
extern const int KnownGPUVendorsCount = 4;
} // namespace Ray

Ray::RendererBase *Ray::CreateRenderer(const settings_t &s, ILog *log, const Bitmask<eRendererType> enabled_types) {
#if defined(ENABLE_VK_IMPL)
    if (enabled_types & eRendererType::Vulkan) {
        log->Info("Ray: Creating Vulkan renderer %ix%i", s.w, s.h);
        try {
            return Vk::CreateRenderer(s, log);
        } catch (std::exception &e) {
            log->Info("Ray: Failed to create Vulkan renderer, %s", e.what());
        }
    }
#endif // ENABLE_VK_IMPL
#if defined(ENABLE_DX_IMPL) && defined(_WIN32)
    if (enabled_types & eRendererType::DirectX12) {
        log->Info("Ray: Creating DirectX12 renderer %ix%i", s.w, s.h);
        try {
            return Dx::CreateRenderer(s, log);
        } catch (std::exception &e) {
            log->Info("Ray: Failed to create DirectX12 renderer, %s", e.what());
        }
    }
#endif // defined(ENABLE_DX_IMPL) && defined(_WIN32)
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#ifdef ENABLE_SIMD_IMPL
    const CpuFeatures features = GetCpuFeatures();
    if ((enabled_types & eRendererType::SIMD_AVX512) && features.avx512_supported) {
        log->Info("Ray: Creating AVX512 renderer %ix%i", s.w, s.h);
        return Avx512::CreateRenderer(s, log);
    }
    if ((enabled_types & eRendererType::SIMD_AVX2) && features.avx2_supported) {
        log->Info("Ray: Creating AVX2 renderer %ix%i", s.w, s.h);
        return Avx2::CreateRenderer(s, log);
    }
    if ((enabled_types & eRendererType::SIMD_AVX) && features.avx_supported) {
        log->Info("Ray: Creating AVX renderer %ix%i", s.w, s.h);
        return Avx::CreateRenderer(s, log);
    }
    if ((enabled_types & eRendererType::SIMD_SSE41) && features.sse41_supported) {
        log->Info("Ray: Creating SSE41 renderer %ix%i", s.w, s.h);
        return Sse41::CreateRenderer(s, log);
    }
    if ((enabled_types & eRendererType::SIMD_SSE2) && features.sse2_supported) {
        log->Info("Ray: Creating SSE2 renderer %ix%i", s.w, s.h);
        return Sse2::CreateRenderer(s, log);
    }
#endif
#ifdef ENABLE_REF_IMPL
    if (enabled_types & eRendererType::Reference) {
        log->Info("Ray: Creating Reference renderer %ix%i", s.w, s.h);
        return Ref::CreateRenderer(s, log);
    }
#endif
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#ifdef ENABLE_SIMD_IMPL
    if (enabled_types & eRendererType::SIMD_NEON) {
        log->Info("Ray: Creating NEON renderer %ix%i", s.w, s.h);
        return Neon::CreateRenderer(s, log);
    }
#endif
#ifdef ENABLE_REF_IMPL
    if (enabled_types & eRendererType::Reference) {
        log->Info("Ray: Creating Reference renderer %ix%i", s.w, s.h);
        return Ref::CreateRenderer(s, log);
    }
#endif
#endif
#ifdef ENABLE_REF_IMPL
    log->Info("Ray: Creating Reference renderer %ix%i", s.w, s.h);
    return Ref::CreateRenderer(s, log);
#else
    return nullptr;
#endif
}

int Ray::QueryAvailableGPUDevices(ILog *log, gpu_device_t out_devices[], const int capacity) {
    int count = 0;
#ifdef ENABLE_VK_IMPL
    count = Vk::Context::QueryAvailableDevices(log, out_devices, capacity);
#endif
#if defined(ENABLE_DX_IMPL) && defined(_WIN32)
    count = Dx::Context::QueryAvailableDevices(log, out_devices, capacity);
#endif
    return count;
}

bool Ray::MatchDeviceNames(const char *name, const char *pattern) {
    std::regex match_name(pattern);
    return std::regex_search(name, match_name) || strcmp(name, pattern) == 0;
}

const char *Ray::Version() { return "v0.3.0-unknown-commit"; }

// Workaround for a strange clang behavior
template class std::vector<float, Ray::aligned_allocator<float, 64>>;
