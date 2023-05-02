#include "test_common.h"

#include <iostream>

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
