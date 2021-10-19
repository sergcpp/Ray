#include "detect.h"

#if defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
//  Windows
#include <intrin.h>
#ifdef __GNUC__
#include <cpuid.h>
inline void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#if defined(__GNUC__) && (__GNUC__ < 9)
inline unsigned long long _xgetbv(unsigned int index) {
    unsigned int eax, edx;
    __asm__ __volatile__(
        "xgetbv;"
        : "=a" (eax), "=d"(edx)
        : "c" (index)
    );
    return ((unsigned long long)edx << 32) | eax;
}
#endif
#else
#define cpuid(info, x)    __cpuidex(info, x, 0)
#endif

#else

#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64) && !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
//  GCC Intrinsics
#include <cpuid.h>
#include <immintrin.h>
inline void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#if defined(__GNUC__) && (__GNUC__ < 9) && !defined(__APPLE__)
inline unsigned long long _xgetbv(unsigned int index) {
    unsigned int eax, edx;
    __asm__ __volatile__(
        "xgetbv;"
        : "=a" (eax), "=d"(edx)
        : "c" (index)
    );
    return ((unsigned long long)edx << 32) | eax;
}
#endif
#endif

#endif

Ray::CpuFeatures Ray::GetCpuFeatures() {
    CpuFeatures ret = {};
#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
    int info[4];
    cpuid(info, 0);
    int ids_count = info[0];

    cpuid(info, 0x80000000);
    //unsigned ex_ids_count = info[0];

    //  Detect Features
    if (ids_count >= 0x00000001) {
        cpuid(info, 0x00000001);
        ret.sse2_supported = (info[3] & ((int)1 << 26)) != 0;
        ret.sse3_supported = (info[2] & ((int)1 << 0)) != 0;
        ret.sse41_supported = (info[2] & ((int)1 << 19)) != 0;

        bool os_uses_XSAVE_XRSTORE = (info[2] & (1 << 27)) != 0;
        bool os_saves_YMM = false;
        if (os_uses_XSAVE_XRSTORE) {
            // Check if the OS will save the YMM registers
            // _XCR_XFEATURE_ENABLED_MASK = 0
            unsigned long long xcr_feature_mask = _xgetbv(0);
            os_saves_YMM = (xcr_feature_mask & 0x6) != 0;
        }

        bool cpu_FMA_support = (info[2] & ((int)1 << 12)) != 0;

        bool cpu_AVX_support = (info[2] & (1 << 28)) != 0;
        ret.avx_supported = os_saves_YMM && cpu_AVX_support;

        if (ids_count >= 0x00000007) {
            cpuid(info, 0x00000007);

            bool cpu_AVX2_support = (info[1] & (1 << 5)) != 0;
            // use fma in conjunction with avx2 support (like microsoft compiler does)
            ret.avx2_supported = os_saves_YMM && cpu_AVX2_support && cpu_FMA_support;

            ret.avx512_supported = (info[1] & (1 << 16)) != 0;      // HW_AVX512F
            //ret.avx512_supported &= (info[1] & (1 << 28)) != 0;   // HW_AVX512CD
            //ret.avx512_supported &= (info[1] & (1 << 26)) != 0;   // HW_AVX512PF
            //ret.avx512_supported &= (info[1] & (1 << 27)) != 0;   // HW_AVX512ER
            //ret.avx512_supported &= (info[1] & (1 << 31)) != 0;   // HW_AVX512VL
            ret.avx512_supported &= (info[1] & (1 << 30)) != 0;     // HW_AVX512BW
            ret.avx512_supported &= (info[1] & (1 << 17)) != 0;     // HW_AVX512DQ
            //ret.avx512_supported &= (info[1] & (1 << 21)) != 0;   // HW_AVX512IFMA
            //ret.avx512_supported &= (info[2] & (1 << 1)) != 0;    // HW_AVX512VBMI
        }
    }
#elif defined(__i386__) || defined(__x86_64__)
    ret.sse2_supported = true;
#endif

    return ret;
}

#undef cpuid
