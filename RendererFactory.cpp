#include "RendererFactory.h"

#include <iostream>

#include "internal/RendererRef.h"
#if !defined(__ANDROID__)
#include "internal/RendererRef2.h"
#include "internal/RendererSSE.h"
#include "internal/RendererAVX.h"
#include "internal/RendererOCL.h"
#endif

#include "internal/simd/detect.h"

std::shared_ptr<ray::RendererBase> ray::CreateRenderer(int w, int h, uint32_t flags) {
    auto features = GetCpuFeatures();
#if !defined(__ANDROID__)
    if (flags & RendererOCL) {
        std::cout << "ray: Creating OpenCL renderer " << w << "x" << h << std::endl;
        try {
            return std::make_shared<ocl::Renderer>(w, h);
        } catch (...) {
            std::cout << "ray: Creating OpenCL renderer failed" << std::endl;
        }
    }
    if ((flags & RendererAVX) && features.avx_supported) {
        std::cout << "ray: Creating AVX renderer " << w << "x" << h << std::endl;
        return std::make_shared<avx::Renderer>(w, h);
    }
    if ((flags & RendererSSE) && features.sse2_supported) {
        std::cout << "ray: Creating SSE renderer " << w << "x" << h << std::endl;
        return std::make_shared<sse::Renderer>(w, h);
    }
    if (flags & RendererRef) {
        std::cout << "ray: Creating Ref renderer " << w << "x" << h << std::endl;
        return std::make_shared<ref::Renderer>(w, h);
    }
#endif
    std::cout << "ray: Creating Ref renderer " << w << "x" << h << std::endl;
    return std::make_shared<ref::Renderer>(w, h);
}