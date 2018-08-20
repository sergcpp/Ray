#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

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
    auto features = Ray::GetCpuFeatures();

    test_simd_ref();

    if (features.sse2_supported) {
        test_simd_sse();
    } else {
        puts("Skipping sse2 test!");
    }

    if (features.avx_supported) {
        test_simd_avx();
    } else {
        puts("Skipping avx test!");
    }
#endif
}