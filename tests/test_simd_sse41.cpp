#include "test_common.h"

#include <iostream>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("sse4.1")
#endif
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("sse4.1"))), apply_to=function)
#endif

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Sse41
#define USE_SSE41
#include "../internal/simd/simd_vec.h"

void test_simd_sse41() {
#include "test_simd.ipp"
}
#undef USE_SSE41
#undef NS

#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif
#ifdef __clang__
#pragma clang attribute pop
#endif
