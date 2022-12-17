#include "RendererFactory.h"

#ifdef ENABLE_REF_IMPL
#include "internal/RendererRef.h"
#else // ENABLE_REF_IMPL
#pragma message("Compiling without reference backend")
#endif // ENABLE_REF_IMPL

#ifdef ENABLE_SIMD_IMPL
#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#include "internal/RendererAVX.h"
#include "internal/RendererAVX2.h"
#include "internal/RendererAVX512.h"
#include "internal/RendererSSE2.h"
#include "internal/RendererSSE41.h"
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include "internal/RendererNEON.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE2.h"
#endif
#else // ENABLE_SIMD_IMPL
#pragma message("Compiling without SIMD support")
#endif // #ifdef ENABLE_SIMD_IMPL

#ifdef ENABLE_GPU_IMPL
#include "internal/RendererVK.h"
#else // ENABLE_GPU_IMPL
#pragma message("Compiling without GPU support")
#endif // ENABLE_GPU_IMPL

#include "internal/simd/detect.h"

namespace Ray {
LogNull g_null_log;
} // namespace Ray

Ray::RendererBase *Ray::CreateRenderer(const settings_t &s, ILog *log, const uint32_t enabled_types) {
    CpuFeatures features = GetCpuFeatures();

#ifdef ENABLE_GPU_IMPL
    if (enabled_types & RendererVK) {
        log->Info("Ray: Creating Vulkan renderer %ix%i", s.w, s.h);
        try {
            return new Vk::Renderer(s, log);
        } catch (std::exception &e) {
            log->Info("Ray: Creating Vulkan renderer failed, %s", e.what());
        }
    }
#endif // ENABLE_GPU_IMPL

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#ifdef ENABLE_SIMD_IMPL
    if ((enabled_types & RendererAVX512) && features.avx512_supported) {
        log->Info("Ray: Creating AVX512 renderer %ix%i", s.w, s.h);
        return new Avx512::Renderer(s, log);
    }
    if ((enabled_types & RendererAVX2) && features.avx2_supported) {
        log->Info("Ray: Creating AVX2 renderer %ix%i", s.w, s.h);
        return new Avx2::Renderer(s, log);
    }
    if ((enabled_types & RendererAVX) && features.avx_supported) {
        log->Info("Ray: Creating AVX renderer %ix%i", s.w, s.h);
        return new Avx::Renderer(s, log);
    }
    if ((enabled_types & RendererSSE41) && features.sse41_supported) {
        log->Info("Ray: Creating SSE41 renderer %ix%i", s.w, s.h);
        return new Sse41::Renderer(s, log);
    }
    if ((enabled_types & RendererSSE2) && features.sse2_supported) {
        log->Info("Ray: Creating SSE2 renderer %ix%i", s.w, s.h);
        return new Sse2::Renderer(s, log);
    }
#endif
#ifdef ENABLE_REF_IMPL
    if (enabled_types & RendererRef) {
        log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
        return new Ref::Renderer(s, log);
    }
#endif
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#ifdef ENABLE_SIMD_IMPL
    if (enabled_types & RendererNEON) {
        log->Info("Ray: Creating NEON renderer %ix%i", s.w, s.h);
        return new Neon::Renderer(s, log);
    }
#endif
#ifdef ENABLE_REF_IMPL
    if (enabled_types & RendererRef) {
        log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
        return new Ref::Renderer(s, log);
    }
#endif
#endif
#ifdef ENABLE_REF_IMPL
    log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
    return new Ref::Renderer(s, log);
#else
    return nullptr;
#endif
}
