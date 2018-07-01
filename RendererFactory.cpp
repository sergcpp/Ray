#include "RendererFactory.h"

#include "internal/RendererRef.h"

#if !defined(__ANDROID__)
#include "internal/RendererRef2.h"
#include "internal/RendererSSE.h"
#include "internal/RendererAVX.h"
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#include "internal/RendererNEON.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE.h"
#endif

#if !defined(DISABLE_OCL)
#include "internal/RendererOCL.h"
#else
#pragma message("Compiling without OpenCL support")
#endif

#include "internal/simd/detect.h"

std::shared_ptr<ray::RendererBase> ray::CreateRenderer(const settings_t &s, uint32_t flags, std::ostream &log_stream) {
    auto features = GetCpuFeatures();

#if !defined(DISABLE_OCL)
    if (flags & RendererOCL) {
        log_stream << "ray: Creating OpenCL renderer " << s.w << "x" << s.h << std::endl;
        try {
            return std::make_shared<ocl::Renderer>(s.w, s.h, s.platform_index, s.device_index);
        } catch (std::exception &e) {
            log_stream << "ray: Creating OpenCL renderer failed, " << e.what() << std::endl;
        }
    }
#endif

#if !defined(__ANDROID__)
    if ((flags & RendererAVX) && features.avx_supported) {
        log_stream << "ray: Creating AVX renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<avx::Renderer>(s.w, s.h);
    }
    if ((flags & RendererSSE) && features.sse2_supported) {
        log_stream << "ray: Creating SSE renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<sse::Renderer>(s.w, s.h);
    }
    if (flags & RendererRef) {
        log_stream << "ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<ref::Renderer>(s.w, s.h);
    }
#elif defined(__ARM_NEON__) || defined(__aarch64__)
    if (flags & RendererNEON) {
        log_stream << "ray: Creating NEON renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<neon::Renderer>(s.w, s.h);
    }
    if (flags & RendererRef) {
        log_stream << "ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<ref::Renderer>(s.w, s.h);
    }
#elif defined(__i386__) || defined(__x86_64__)
    if ((flags & RendererSSE) && features.sse2_supported) {
        log_stream << "ray: Creating SSE renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<sse::Renderer>(s.w, s.h);
    }
    if (flags & RendererRef) {
        log_stream << "ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
        return std::make_shared<ref::Renderer>(s.w, s.h);
    }
#endif
    log_stream << "ray: Creating Ref renderer " << s.w << "x" << s.h << std::endl;
    return std::make_shared<ref::Renderer>(s.w, s.h);
}

#if !defined(DISABLE_OCL)
std::vector<ray::ocl::Platform> ray::ocl::QueryPlatforms() {
    return Renderer::QueryPlatforms();
}
#endif