
// MSVC allows setting /arch option only for separate translation units, so put it here.
// (simd_vec is vectorized manually with intrinsics, but compiling whole core functions with /arch:AVX2 allows to avoid SSE/AVX switch overhead)

#if !defined(__ANDROID__)
#include "internal/RendererAVX2.cpp"
#endif