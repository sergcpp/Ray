#include "RendererBase.h"

#include <cstring>

namespace Ray {
const char *RendererTypeName(const eRendererType rt) {
    switch (rt) {
    case eRendererType::Reference:
        return "REF";
    case eRendererType::SIMD_SSE2:
        return "SSE2";
    case eRendererType::SIMD_SSE41:
        return "SSE41";
    case eRendererType::SIMD_AVX:
        return "AVX";
    case eRendererType::SIMD_AVX2:
        return "AVX2";
    case eRendererType::SIMD_AVX512:
        return "AVX512";
    case eRendererType::SIMD_NEON:
        return "NEON";
    case eRendererType::Vulkan:
        return "VK";
    case eRendererType::DirectX12:
        return "DX";
    default:
        return "";
    }
}

eRendererType RendererTypeFromName(const char *name) {
    if (strcmp(name, "REF") == 0) {
        return eRendererType::Reference;
    } else if (strcmp(name, "SSE2") == 0) {
        return eRendererType::SIMD_SSE2;
    } else if (strcmp(name, "SSE41") == 0) {
        return eRendererType::SIMD_SSE41;
    } else if (strcmp(name, "AVX") == 0) {
        return eRendererType::SIMD_AVX;
    } else if (strcmp(name, "AVX2") == 0) {
        return eRendererType::SIMD_AVX2;
    } else if (strcmp(name, "AVX512") == 0) {
        return eRendererType::SIMD_AVX512;
    } else if (strcmp(name, "NEON") == 0) {
        return eRendererType::SIMD_NEON;
    } else if (strcmp(name, "VK") == 0) {
        return eRendererType::Vulkan;
    } else if (strcmp(name, "DX") == 0) {
        return eRendererType::DirectX12;
    }
    return eRendererType::Reference;
}

bool RendererSupportsMultithreading(const eRendererType rt) {
    switch (rt) {
    case eRendererType::Reference:
    case eRendererType::SIMD_SSE2:
    case eRendererType::SIMD_SSE41:
    case eRendererType::SIMD_AVX:
    case eRendererType::SIMD_AVX2:
    case eRendererType::SIMD_AVX512:
    case eRendererType::SIMD_NEON:
        return true;
    default:
        return false;
    }
}

bool RendererSupportsHWRT(eRendererType rt) { return rt == eRendererType::Vulkan || rt == eRendererType::DirectX12; }
} // namespace Ray