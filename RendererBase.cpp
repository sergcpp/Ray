#include "RendererBase.h"

#include <cstring>

namespace Ray {
const char *RendererTypeName(const eRendererType rt) {
    switch (rt) {
    case RendererRef:
        return "REF";
    case RendererSSE2:
        return "SSE2";
    case RendererSSE41:
        return "SSE41";
    case RendererAVX:
        return "AVX";
    case RendererAVX2:
        return "AVX2";
    case RendererAVX512:
        return "AVX512";
    case RendererNEON:
        return "NEON";
    case RendererVK:
        return "VK";
    default:
        return "";
    }
}

eRendererType RendererTypeFromName(const char *name) {
    if (strcmp(name, "REF") == 0) {
        return RendererRef;
    } else if (strcmp(name, "SSE2") == 0) {
        return RendererSSE2;
    } else if (strcmp(name, "SSE41") == 0) {
        return RendererSSE41;
    } else if (strcmp(name, "AVX") == 0) {
        return RendererAVX;
    } else if (strcmp(name, "AVX2") == 0) {
        return RendererAVX2;
    } else if (strcmp(name, "AVX512") == 0) {
        return RendererAVX512;
    } else if (strcmp(name, "NEON") == 0) {
        return RendererNEON;
    } else if (strcmp(name, "VK") == 0) {
        return RendererVK;
    }
    return RendererRef;
}

bool RendererSupportsMultithreading(const eRendererType rt) {
    switch (rt) {
    case RendererRef:
    case RendererSSE2:
    case RendererSSE41:
    case RendererAVX:
    case RendererAVX2:
    case RendererAVX512:
    case RendererNEON:
        return true;
    default:
        return false;
    }
}
} // namespace Ray