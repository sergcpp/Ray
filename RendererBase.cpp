#include "RendererBase.h"

#include <cstring>

namespace Ray {
const char *RendererTypeName(const eRendererType rt) {
    if (rt == RendererRef) {
        return "ref";
    } else if (rt == RendererSSE2) {
        return "sse2";
    } else if (rt == RendererSSE41) {
        return "sse41";
    } else if (rt == RendererAVX) {
        return "avx";
    } else if (rt == RendererAVX2) {
        return "avx2";
    } else if (rt == RendererNEON) {
        return "neon";
    } else if (rt == RendererVK) {
        return "vk";
    }
    return "";
}

eRendererType RendererTypeFromName(const char *name) {
    if (strcmp(name, "ref") == 0) {
        return RendererRef;
    } else if (strcmp(name, "sse2") == 0) {
        return RendererSSE2;
    } else if (strcmp(name, "sse41") == 0) {
        return RendererSSE41;
    } else if (strcmp(name, "avx") == 0) {
        return RendererAVX;
    } else if (strcmp(name, "avx2") == 0) {
        return RendererAVX2;
    } else if (strcmp(name, "neon") == 0) {
        return RendererNEON;
    } else if (strcmp(name, "vk") == 0) {
        return RendererVK;
    }
    return RendererRef;
}
}
