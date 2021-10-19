#include "RendererBase.h"

namespace Ray {
const char *RendererTypeName(const eRendererType rt) {
    if (rt == RendererRef) {
        return "ref";
    } else if (rt == RendererSSE2) {
        return "sse2";
    } else if (rt == RendererAVX) {
        return "avx";
    } else if (rt == RendererAVX2) {
        return "avx2";
    } else if (rt == RendererNEON) {
        return "neon";
    } else if (rt == RendererOCL) {
        return "ocl";
    }
    return "";
}
}
