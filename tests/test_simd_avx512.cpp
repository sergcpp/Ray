#include "test_common.h"

#include <iostream>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512dq")
#pragma clang attribute push (__attribute__((target("avx512f,avx512bw,avx512dq"))), apply_to=function)
#endif

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Avx512
#define USE_AVX512
#include "../internal/simd/simd_vec.h"

void test_simd_avx512() {
#include "test_simd.ipp"
}
#undef USE_AVX512
#undef NS

#endif

#ifdef __GNUC__
#pragma GCC pop_options
#pragma clang attribute pop
#endif
