#include "RendererFactory.h"

#include "internal/RendererRef.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#include "internal/RendererAVX.h"
#include "internal/RendererAVX2.h"
#include "internal/RendererSSE2.h"
#include "internal/RendererSSE41.h"
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include "internal/RendererNEON.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE2.h"
#endif

#if !defined(DISABLE_GPU)
#include "internal/RendererVK.h"
#else
#pragma message("Compiling without GPU support")
#endif

#include "internal/simd/detect.h"

namespace Ray {
LogNull g_null_log;
} // namespace Ray

Ray::RendererBase *Ray::CreateRenderer(const settings_t &s, ILog *log, const uint32_t enabled_types) {
    CpuFeatures features = GetCpuFeatures();

#if !defined(DISABLE_GPU)
    if (enabled_types & RendererVK) {
        log->Info("Ray: Creating Vulkan renderer %ix%i", s.w, s.h);
        try {
            return new Vk::Renderer(s, log);
        } catch (std::exception &e) {
            log->Info("Ray: Creating Vulkan renderer failed, %s", e.what());
        }
    }
#endif

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
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
    if (enabled_types & RendererRef) {
        log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
        return new Ref::Renderer(s, log);
    }
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    if (enabled_types & RendererNEON) {
        log->Info("Ray: Creating NEON renderer %ix%i", s.w, s.h);
        return new Neon::Renderer(s, log);
    }
    if (enabled_types & RendererRef) {
        log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
        return new Ref::Renderer(s, log);
    }
#endif
    log->Info("Ray: Creating Ref renderer %ix%i", s.w, s.h);
    return new Ref::Renderer(s, log);
}

#if !defined(DISABLE_GPU)
//std::vector<Ray::Ocl::Platform> Ray::Ocl::QueryPlatforms() { return Renderer::QueryPlatforms(); }
#endif
