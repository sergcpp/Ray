#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"

#if !defined(__ANDROID__)

#define NS Ref2
#include "../internal/simd/simd_vec.h"

void test_simd_ref() {
#include "test_simd.ipp"
}
#undef NS

#define NS Sse
#define USE_SSE
#include "../internal/simd/simd_vec.h"

void test_simd_sse() {
#include "test_simd.ipp"
}
#undef USE_SSE
#undef NS

#define NS Avx
#define USE_AVX
#include "../internal/simd/simd_vec.h"

void test_simd_avx() {
#include "test_simd.ipp"
}
#undef USE_AVX
#undef NS

#endif

void test_simd() {
#if !defined(__ANDROID__)
    test_simd_ref();
    test_simd_sse();
    test_simd_avx();
#endif
}