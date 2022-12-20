#include "RendererBase.h"

#include <cstring>

namespace Ray {
const char *RendererTypeName(const eRendererType rt) {
    if (rt == RendererRef) {
        return "REF";
    } else if (rt == RendererSSE2) {
        return "SSE2";
    } else if (rt == RendererSSE41) {
        return "SSE41";
    } else if (rt == RendererAVX) {
        return "AVX";
    } else if (rt == RendererAVX2) {
        return "AVX2";
    } else if (rt == RendererAVX512) {
        return "AVX512";
    } else if (rt == RendererNEON) {
        return "NEON";
    } else if (rt == RendererVK) {
        return "VK";
    }
    return "";
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
}
