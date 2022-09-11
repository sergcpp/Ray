#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

extern bool g_stop_on_fail;
extern bool g_tests_success;

static void handle_assert(bool passed, const char* assert, const char* file, long line, bool fatal) {
    if (!passed) {
        printf("Assertion failed %s in %s at line %d\n", assert, file, int(line));
        g_tests_success = false;
        if (fatal) {
            exit(-1);
        }
    }
}

#define require(x) handle_assert(x, #x , __FILE__, __LINE__, g_stop_on_fail)
#define require_fatal(x) handle_assert(x, #x , __FILE__, __LINE__, true)
#define require_skip(x) handle_assert(x, #x , __FILE__, __LINE__, false); if (!(x)) return

#define require_throws(expr) {          \
            bool _ = false;             \
            try {                       \
                expr;                   \
            } catch (...) {             \
                _ = true;               \
            }                           \
            assert(_);                  \
        }

#define require_nothrow(expr) {         \
            bool _ = false;             \
            try {                       \
                expr;                   \
            } catch (...) {             \
                _ = true;               \
            }                           \
            assert(!_);                 \
        }

class Approx {
public:
    Approx(double val, double eps = 0.001) : val(val), eps(eps) {
        require(eps > 0);
    }

    double val, eps;
};

inline bool operator==(double val, const Approx &app) {
    return std::abs(val - app.val) < app.eps;
}

inline bool operator!=(double val, const Approx &app) { return !operator==(val, app); }

inline bool operator==(float val, const Approx &app) {
    return std::abs(val - app.val) < app.eps;
}

inline bool operator!=(float val, const Approx &app) { return !operator==(val, app); }
