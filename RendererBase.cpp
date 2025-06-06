#include "RendererBase.h"

#include <cstring>

namespace Ray {
std::string_view RendererTypeName(const eRendererType rt) {
    switch (rt) {
    case eRendererType::Reference:
        return "REF";
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

eRendererType RendererTypeFromName(std::string_view name) {
    if (name == "REF") {
        return eRendererType::Reference;
    } else if (name == "SSE41") {
        return eRendererType::SIMD_SSE41;
    } else if (name == "AVX") {
        return eRendererType::SIMD_AVX;
    } else if (name == "AVX2") {
        return eRendererType::SIMD_AVX2;
    } else if (name == "AVX512") {
        return eRendererType::SIMD_AVX512;
    } else if (name == "NEON") {
        return eRendererType::SIMD_NEON;
    } else if (name == "VK") {
        return eRendererType::Vulkan;
    } else if (name == "DX") {
        return eRendererType::DirectX12;
    }
    return eRendererType::Reference;
}

bool RendererSupportsMultithreading(const eRendererType rt) {
    switch (rt) {
    case eRendererType::Reference:
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