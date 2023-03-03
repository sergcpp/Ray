
// MSVC allows setting /arch option only for separate translation unit, so put it here.
// /arch is most likely already set to SSE2 by default, but it is allowed to override it to IA32

#include "Config.h"

#if defined(ENABLE_SIMD_IMPL) && (defined(__ARM_NEON__) || defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64))
#include "internal/RendererNEON.cpp"
#endif
