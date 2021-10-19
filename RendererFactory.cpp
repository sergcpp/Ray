#include "RendererFactory.h"

#include "internal/RendererRef.h"

#if !defined(__ANDROID__)
#include "internal/RendererAVX.h"
#include "internal/RendererAVX2.h"
#include "internal/RendererSSE2.h"
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#include "internal/RendererNEON.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE2.h"
#endif

#if !defined(DISABLE_OCL)
#include "internal/RendererOCL.h"
#else
#pragma message("Compiling without OpenCL support")
#endif

#include "internal/simd/detect.h"

std::shared_ptr<Ray::RendererBase> Ray::CreateRenderer(const settings_t &s, const uint32_t enabled_types,
                                                       std::ostream &log_stream) {
    CpuFeatures features = GetCpuFeatures();

#if !defined(DISABLE_OCL)
    if (enabled_types & RendererOCL) {
        log_stream << "Ray: Creating OpenCL renderer " << s.w << "x" << s.h << std::endl;
        try {
            return std::make_shared<Ocl::Renderer>(s.w, s.h, s.platform_index, s.device_index);
        } catch (std::exception &e) {
            log_stream << "Ray: Creating OpenCL renderer failed, " << e.what() << std::endl;
        }
    }
#endif

#if !defined(__ANDROID__)
    if ((enabled_types & RendererAVX2) && features.avx2_supported) {
        log_stream << "Ray: Creating AVX2 renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Avx2::Renderer>(s);
    }
    if ((enabled_types & RendererAVX) && features.avx_supported) {
        log_stream << "Ray: Creating AVX renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Avx::Renderer>(s);
    }
    if ((enabled_types & RendererSSE2) && features.sse2_supported) {
        log_stream << "Ray: Creating SSE2 renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Sse2::Renderer>(s);
    }
    if (enabled_types & RendererRef) {
        log_stream << "Ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Ref::Renderer>(s);
    }
#elif defined(__ARM_NEON__) || defined(__aarch64__)
    if (enabled_types & RendererNEON) {
        log_stream << "Ray: Creating NEON renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Neon::Renderer>(s);
    }
    if (enabled_types & RendererRef) {
        log_stream << "Ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Ref::Renderer>(s);
    }
#elif defined(__i386__) || defined(__x86_64__)
    if ((enabled_types & RendererSSE2) && features.sse2_supported) {
        log_stream << "Ray: Creating SSE2 renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Sse2::Renderer>(s);
    }
    if (enabled_types & RendererRef) {
        log_stream << "Ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<Ref::Renderer>(s);
    }
#endif
    log_stream << "Ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
    return std::make_shared<Ref::Renderer>(s);
}

#if !defined(DISABLE_OCL)
std::vector<Ray::Ocl::Platform> Ray::Ocl::QueryPlatforms() { return Renderer::QueryPlatforms(); }
#endif