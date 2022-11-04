
// MSVC allows setting /arch option only for separate translation unit, so put it here.
// /arch is most likely already set to SSE2 by default, but it is allowed to override it to IA32

#if defined(_M_X86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE2.cpp"
#include "internal/UtilsSSE2.cpp"
#endif
