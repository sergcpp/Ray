#pragma once

#ifdef _WIN32

//  Windows
#include <intrin.h>
#ifdef __GNUC__
#include <cpuid.h>
inline void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#else
#define cpuid(info, x)    __cpuidex(info, x, 0)
#endif

#else

#if !defined(__arm__) && !defined(__aarch64__) && !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
//  GCC Intrinsics
#include <cpuid.h>
inline void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

#endif

namespace Ray {
    struct CpuFeatures {
        bool sse2_supported, avx_supported;
    };

    inline CpuFeatures GetCpuFeatures() {
        CpuFeatures ret;

        ret.sse2_supported = false;
        ret.avx_supported = false;
#if !defined(__ANDROID__)
        int info[4];
        cpuid(info, 0);
        int nIds = info[0];

        cpuid(info, 0x80000000);
        unsigned nExIds = info[0];

        //  Detect Features
        if (nIds >= 0x00000001) {
            cpuid(info, 0x00000001);
            ret.sse2_supported = (info[3] & ((int)1 << 26)) != 0;
            ret.avx_supported = (info[2] & ((int)1 << 28)) != 0;
        }
#elif defined(__i386__) || defined(__x86_64__)
        ret.sse2_supported = true;
#endif

        return ret;
    }
}

#undef cpuid