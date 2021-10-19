#pragma once

namespace Ray {
    struct CpuFeatures {
        unsigned sse2_supported : 1;
        unsigned sse3_supported : 1;
        unsigned sse41_supported : 1;
        unsigned avx_supported : 1;
        unsigned avx2_supported : 1;
        unsigned avx512_supported : 1;
    };

    CpuFeatures GetCpuFeatures();
}
