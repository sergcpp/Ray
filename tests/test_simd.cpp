#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#define NS Ref2
#include "../internal/simd/simd_vec.h"

void test_simd_ref() {
#include "test_simd.ipp"
}
#undef NS

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#define NS Sse2
#define USE_SSE2
#include "../internal/simd/simd_vec.h"

void test_simd_sse() {
#include "test_simd.ipp"
}
#undef USE_SSE2
#undef NS

#define NS Avx
#define USE_AVX
#include "../internal/simd/simd_vec.h"

void test_simd_avx() {
#include "test_simd.ipp"
}
#undef USE_AVX
#undef NS

#define NS Avx2
#define USE_AVX2
#include "../internal/simd/simd_vec.h"

void test_simd_avx2() {
#include "test_simd.ipp"
}
#undef USE_AVX2
#undef NS

void test_simd_avx512();

#else // !defined(__aarch64__)

#define NS Neon
#define USE_NEON
#include "../internal/simd/simd_vec.h"

void test_simd_neon() {
#include "test_simd.ipp"
}

#endif

void test_simd() {
    puts(" SIMD NONE:");
    test_simd_ref();

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
    auto features = Ray::GetCpuFeatures();

    puts(" SIMD SSE2:");
    if (features.sse2_supported) {
        test_simd_sse();
    } else {
        puts("Skipping... (not supported)");
    }

    puts(" SIMD AVX:");
    if (features.avx_supported) {
        test_simd_avx();
    } else {
        puts("Skipping... (not supported)");
    }

    puts(" SIMD AVX2:");
    if (features.avx2_supported) {
        test_simd_avx2();
    } else {
        puts("Skipping... (not supported)");
    }

    puts(" SIMD AVX512:");
    if (features.avx512_supported) {
        test_simd_avx512();
    } else {
        puts("Skipping... (not supported)");
    }
#else
    puts(" SIMD NEON:");
    test_simd_neon();
#endif
}
