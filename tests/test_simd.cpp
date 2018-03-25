#include "test_common.h"

#include <iostream>

#define INSTANTIATION_ID 0
#include "../internal/simd/simd_vec.h"

void test_simd_ref() {
#include "test_simd.ipp"
}
#undef INSTANTIATION_ID

#define USE_SSE
#define INSTANTIATION_ID 1
#include "../internal/simd/simd_vec.h"

void test_simd_sse() {
#include "test_simd.ipp"
}
#undef USE_SSE
#undef INSTANTIATION_ID

#define USE_AVX
#define INSTANTIATION_ID 2
#include "../internal/simd/simd_vec.h"

void test_simd_avx() {
#include "test_simd.ipp"
}
#undef USE_AVX
#undef INSTANTIATION_ID

void test_simd() {
    test_simd_ref();
    test_simd_sse();
    test_simd_avx();
}