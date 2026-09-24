#include "test_common.h"

#include "../internal/CoreRef.h"

#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
#define NS Sse2
#define USE_SSE2
// clang-format off
// simd.h must precede Trig.h (Trig.h needs the fvec aliases)
#include "../internal/simd/simd.h"
#include "../internal/Trig.h"
// clang-format on
#undef USE_SSE2
#undef NS
#else
#define NS Neon
#define USE_NEON
// clang-format off
// simd.h must precede Trig.h (Trig.h needs the fvec aliases)
#include "../internal/simd/simd.h"
#include "../internal/Trig.h"
// clang-format on
#undef USE_NEON
#undef NS
#endif

void test_trig() {
    using namespace Ray;

    printf("Test trig               | ");

    { // Reference version
        double max_error = 0.0;
        for (float x = -2.0f * PI; x <= 2.0f * PI; x += 0.0001f) {
            const double ref_cos = std::cos(double(x));
            const double ref_sin = std::sin(double(x));

            const float test_cos = Ref::portable_cos(x);
            const float test_sin = Ref::portable_sin(x);

            max_error = std::max(max_error, std::abs(ref_cos - test_cos));
            max_error = std::max(max_error, std::abs(ref_sin - test_sin));

            const Ref::fvec2 test_sincos = Ref::portable_sincos(x);

            max_error = std::max(max_error, std::abs(ref_cos - test_sincos.get<1>()));
            max_error = std::max(max_error, std::abs(ref_sin - test_sincos.get<0>()));
        }
        require(max_error < 6.25e-07);
    }
    { // Reference version portable_atan2f (arbitrary y)
        double max_error = 0.0;
        for (float y = -4.0f; y <= 4.0f; y += 0.01f) {
            for (float x = -4.0f; x <= 4.0f; x += 0.01f) {
                const double ref_atan2 = std::atan2(double(y), double(x));
                max_error = std::max(max_error, std::abs(ref_atan2 - double(Ref::portable_atan2f(y, x))));
            }
        }
        require(max_error < 6.417e-07);
    }
#if !defined(__aarch64__) && !defined(_M_ARM) && !defined(_M_ARM64)
    { // SIMD version
        double max_error = 0.0;
        for (float x = -2.0f * PI; x <= 2.0f * PI; x += 0.0004f) {
            const double xx[4] = {x, x + 0.0001f, x + 0.0002f, x + 0.0003f};

            const double ref_cos[4] = {std::cos(xx[0]), std::cos(xx[1]), std::cos(xx[2]), std::cos(xx[3])};
            const double ref_sin[4] = {std::sin(xx[0]), std::sin(xx[1]), std::sin(xx[2]), std::sin(xx[3])};

            const float fxx[4] = {float(xx[0]), float(xx[1]), float(xx[2]), float(xx[3])};
            const Sse2::fvec4 test_cos = Sse2::portable_cos(Sse2::fvec4{fxx});
            const Sse2::fvec4 test_sin = Sse2::portable_sin(Sse2::fvec4{fxx});

            max_error = std::max(max_error, std::abs(ref_cos[0] - test_cos.get<0>()));
            max_error = std::max(max_error, std::abs(ref_cos[1] - test_cos.get<1>()));
            max_error = std::max(max_error, std::abs(ref_cos[2] - test_cos.get<2>()));
            max_error = std::max(max_error, std::abs(ref_cos[3] - test_cos.get<3>()));
            max_error = std::max(max_error, std::abs(ref_sin[0] - test_sin.get<0>()));
            max_error = std::max(max_error, std::abs(ref_sin[1] - test_sin.get<1>()));
            max_error = std::max(max_error, std::abs(ref_sin[2] - test_sin.get<2>()));
            max_error = std::max(max_error, std::abs(ref_sin[3] - test_sin.get<3>()));

            Sse2::fvec4 test_sincos[2];
            Sse2::portable_sincos(Sse2::fvec4{fxx}, test_sincos[0], test_sincos[1]);

            max_error = std::max(max_error, std::abs(ref_cos[0] - test_sincos[1].get<0>()));
            max_error = std::max(max_error, std::abs(ref_cos[1] - test_sincos[1].get<1>()));
            max_error = std::max(max_error, std::abs(ref_cos[2] - test_sincos[1].get<2>()));
            max_error = std::max(max_error, std::abs(ref_cos[3] - test_sincos[1].get<3>()));
            max_error = std::max(max_error, std::abs(ref_sin[0] - test_sincos[0].get<0>()));
            max_error = std::max(max_error, std::abs(ref_sin[1] - test_sincos[0].get<1>()));
            max_error = std::max(max_error, std::abs(ref_sin[2] - test_sincos[0].get<2>()));
            max_error = std::max(max_error, std::abs(ref_sin[3] - test_sincos[0].get<3>()));
        }
        require(max_error < 6.417e-07);
    }
    { // Sse2 portable_atan2f (arbitrary y)
        double max_error = 0.0;
        for (float y = -4.0f; y <= 4.0f; y += 0.04f) {
            for (float x = -4.0f; x <= 4.0f; x += 0.004f) {
                const float fyy[4] = {y, y, y, y};
                const float fxx[4] = {x, x + 0.001f, x + 0.002f, x + 0.003f};
                const Sse2::fvec4 res = Sse2::portable_atan2f(Sse2::fvec4{fyy}, Sse2::fvec4{fxx});
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[0]), double(fxx[0])) - double(res.get<0>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[1]), double(fxx[1])) - double(res.get<1>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[2]), double(fxx[2])) - double(res.get<2>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[3]), double(fxx[3])) - double(res.get<3>())));
            }
        }
        require(max_error < 6.417e-07);
    }
#else
    { // SIMD version
        double max_error = 0.0;
        for (float x = -2.0f * PI; x <= 2.0f * PI; x += 0.0004f) {
            const double xx[4] = {x, x + 0.0001f, x + 0.0002f, x + 0.0003f};

            const double ref_cos[4] = {std::cos(xx[0]), std::cos(xx[1]), std::cos(xx[2]), std::cos(xx[3])};
            const double ref_sin[4] = {std::sin(xx[0]), std::sin(xx[1]), std::sin(xx[2]), std::sin(xx[3])};

            const float fxx[4] = {float(xx[0]), float(xx[1]), float(xx[2]), float(xx[3])};
            const Neon::fvec4 test_cos = Neon::portable_cos(Neon::fvec4{fxx});
            const Neon::fvec4 test_sin = Neon::portable_sin(Neon::fvec4{fxx});

            max_error = std::max(max_error, std::abs(ref_cos[0] - test_cos.get<0>()));
            max_error = std::max(max_error, std::abs(ref_cos[1] - test_cos.get<1>()));
            max_error = std::max(max_error, std::abs(ref_cos[2] - test_cos.get<2>()));
            max_error = std::max(max_error, std::abs(ref_cos[3] - test_cos.get<3>()));
            max_error = std::max(max_error, std::abs(ref_sin[0] - test_sin.get<0>()));
            max_error = std::max(max_error, std::abs(ref_sin[1] - test_sin.get<1>()));
            max_error = std::max(max_error, std::abs(ref_sin[2] - test_sin.get<2>()));
            max_error = std::max(max_error, std::abs(ref_sin[3] - test_sin.get<3>()));

            Neon::fvec4 test_sincos[2];
            Neon::portable_sincos(Neon::fvec4{fxx}, test_sincos[0], test_sincos[1]);

            max_error = std::max(max_error, std::abs(ref_cos[0] - test_sincos[1].get<0>()));
            max_error = std::max(max_error, std::abs(ref_cos[1] - test_sincos[1].get<1>()));
            max_error = std::max(max_error, std::abs(ref_cos[2] - test_sincos[1].get<2>()));
            max_error = std::max(max_error, std::abs(ref_cos[3] - test_sincos[1].get<3>()));
            max_error = std::max(max_error, std::abs(ref_sin[0] - test_sincos[0].get<0>()));
            max_error = std::max(max_error, std::abs(ref_sin[1] - test_sincos[0].get<1>()));
            max_error = std::max(max_error, std::abs(ref_sin[2] - test_sincos[0].get<2>()));
            max_error = std::max(max_error, std::abs(ref_sin[3] - test_sincos[0].get<3>()));
        }
        require(max_error < 6.417e-07);
    }
    { // Neon portable_atan2f (arbitrary y)
        double max_error = 0.0;
        for (float y = -4.0f; y <= 4.0f; y += 0.04f) {
            for (float x = -4.0f; x <= 4.0f; x += 0.004f) {
                const float fyy[4] = {y, y, y, y};
                const float fxx[4] = {x, x + 0.001f, x + 0.002f, x + 0.003f};
                const Neon::fvec4 res = Neon::portable_atan2f(Neon::fvec4{fyy}, Neon::fvec4{fxx});
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[0]), double(fxx[0])) - double(res.get<0>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[1]), double(fxx[1])) - double(res.get<1>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[2]), double(fxx[2])) - double(res.get<2>())));
                max_error =
                    std::max(max_error, std::abs(std::atan2(double(fyy[3]), double(fxx[3])) - double(res.get<3>())));
            }
        }
        require(max_error < 6.417e-07);
    }
#endif
    printf("OK\n");
}
