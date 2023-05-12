#pragma once

#include "Core.h"

#pragma push_macro("NS")
#undef NS

#define NS Dx
#if defined(_M_AMD64) || defined(_M_X64) || (!defined(__ANDROID__) && defined(__x86_64__)) ||                          \
    (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#define USE_SSE2
//#pragma message("Ray::Ref::simd_vec will use SSE2")
#elif defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#define USE_NEON
//#pragma message("Ray::Ref::simd_vec will use NEON")
#elif defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define USE_SSE2
//#pragma message("Ray::Ref::simd_vec will use SSE2")
#else
#pragma message("Ray::Ref::simd_vec will not use SIMD")
#endif

#include "simd/simd_vec.h"

#undef USE_SSE2
#undef USE_NEON
#undef NS