
// MSVC allows setting /arch:AVX only for separate translation unit, so put it here.

#if !defined(__ANDROID__)
#include "internal/RendererAVX.cpp"
#endif