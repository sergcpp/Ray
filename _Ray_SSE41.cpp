
// MSVC allows setting /arch option only for separate translation unit, so put it here.
// /arch is most likely already set to SSE2 by default, but it is allowed to override it to IA32

#include "Config.h"

// This is needed only with clang on windows
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("sse4.1"))), apply_to = function)
#endif

#if defined(ENABLE_SIMD_IMPL) && (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__))
#include "internal/RendererSSE41.cpp"
#endif

#ifdef __clang__
#pragma clang attribute pop
#endif