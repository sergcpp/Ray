#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Avx512
#define USE_AVX512
#include "../internal/simd/simd.h"

void test_simd_avx512() {
#include "test_simd.ipp"
}
#undef USE_AVX512
#undef NS

#endif
