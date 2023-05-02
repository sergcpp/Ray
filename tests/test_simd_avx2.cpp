#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Avx2
#define USE_AVX2
#include "../internal/simd/simd_vec.h"

void test_simd_avx2() {
#include "test_simd.ipp"
}
#undef USE_AVX2
#undef NS

#endif
