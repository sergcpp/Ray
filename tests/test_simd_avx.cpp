#include "test_common.h"

#include <iostream>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx")
#endif
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("avx"))), apply_to=function)
#endif

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Avx
#define USE_AVX
#include "../internal/simd/simd_vec.h"

void test_simd_avx() {
#include "test_simd.ipp"
}
#undef USE_AVX
#undef NS

#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif
#ifdef __clang__
#pragma clang attribute pop
#endif
