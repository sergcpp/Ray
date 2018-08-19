
// MSVC allows setting /arch option only for separate translation unit, so put it here.
// /arch is most likely already set to SSE2 by default, but it is allowed to override it to IA32

#if !defined(__ANDROID__) || defined(__i386__) || defined(__x86_64__)
#include "internal/RendererSSE.cpp"
#endif
