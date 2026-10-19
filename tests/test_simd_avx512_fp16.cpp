#include "test_common.h"

#include <iostream>

#include "../internal/Core.h"
#include "../internal/simd/detect.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)

#define NS Avx512_fp16
#define USE_AVX512_FP16
#include "../internal/simd/simd.h"

void test_simd_avx512_fp16() {
#include "test_simd.ipp"
}
#undef USE_AVX512_FP16
#undef NS

#endif
