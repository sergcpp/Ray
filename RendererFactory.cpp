#include "RendererFactory.h"

#include <iostream>

#include <math/math.hpp>

#include "internal/RendererRef.h"
#include "internal/RendererSSE.h"
#include "internal/RendererAVX.h"
#include "internal/RendererOCL.h"

std::shared_ptr<ray::RendererBase> ray::CreateRenderer(int w, int h, uint32_t flags) {
    math::init();

    if (flags & PreferOCL) {
        std::cout << "ray: Creating OpenCL renderer " << w << "x" << h << std::endl;
        try {
            return std::make_shared<ocl::Renderer>(w, h);
        } catch (...) {
            std::cout << "ray: Creating OpenCL renderer failed" << std::endl;
        }
    }
    if ((flags & PreferAVX) && math::supported(math::AVX)) {
        std::cout << "ray: Creating AVX renderer " << w << "x" << h << std::endl;
        return std::make_shared<avx::Renderer>(w, h);
    }
    if ((flags & PreferSSE) && math::supported(math::SSE2)) {
        std::cout << "ray: Creating SSE renderer " << w << "x" << h << std::endl;
        return std::make_shared<sse::Renderer>(w, h);
    }
    if (flags & PreferRef) {
        std::cout << "ray: Creating Ref renderer " << w << "x" << h << std::endl;
        return std::make_shared<ref::Renderer>(w, h);
    }
    std::cout << "ray: Creating Ref renderer " << w << "x" << h << std::endl;
    return std::make_shared<ref::Renderer>(w, h);
}