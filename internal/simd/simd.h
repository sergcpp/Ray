
#include <cmath>
#include <cstring>

#include <algorithm> // for std::max

#pragma warning(push)
#pragma warning(disable : 4789) // buffer overrun
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 6326) // potential comparison of a constant with another constant

// Used to force loop unroll and make index compile-time constant
// clang-format off
#define UNROLLED_FOR_0(I, N, C)
#define UNROLLED_FOR_1(I, N, C) { const int I =  0 % (N); C }
#define UNROLLED_FOR_2(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C }
#define UNROLLED_FOR_3(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C }
#define UNROLLED_FOR_4(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C }
#define UNROLLED_FOR_5(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C }
#define UNROLLED_FOR_6(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C }
#define UNROLLED_FOR_7(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C }
#define UNROLLED_FOR_8(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C }
#define UNROLLED_FOR_9(I, N, C) { const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C }
#define UNROLLED_FOR_10(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C }
#define UNROLLED_FOR_11(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C }
#define UNROLLED_FOR_12(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C } { const int I = 11 % (N); C }
#define UNROLLED_FOR_13(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C } { const int I = 11 % (N); C } \
                                { const int I = 12 % (N); C }
#define UNROLLED_FOR_14(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C } { const int I = 11 % (N); C } \
                                { const int I = 12 % (N); C } { const int I = 13 % (N); C }
#define UNROLLED_FOR_15(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C } { const int I = 11 % (N); C } \
                                { const int I = 12 % (N); C } { const int I = 13 % (N); C } { const int I = 14 % (N); C }
#define UNROLLED_FOR_16(I, N, C){ const int I =  0 % (N); C } { const int I =  1 % (N); C } { const int I =  2 % (N); C } { const int I =  3 % (N); C } \
                                { const int I =  4 % (N); C } { const int I =  5 % (N); C } { const int I =  6 % (N); C } { const int I =  7 % (N); C } \
                                { const int I =  8 % (N); C } { const int I =  9 % (N); C } { const int I = 10 % (N); C } { const int I = 11 % (N); C } \
                                { const int I = 12 % (N); C } { const int I = 13 % (N); C } { const int I = 14 % (N); C } { const int I = 15 % (N); C }

#define UNROLLED_FOR_WRAPPER(INDEX, COUNT, CODE) UNROLLED_FOR_##COUNT(INDEX, COUNT, CODE)
#define UNROLLED_FOR(INDEX, COUNT, CODE) UNROLLED_FOR_WRAPPER(INDEX, COUNT, CODE)

// Needed for cases when count is not a literal constant (e.g. template parameter)
#define UNROLLED_FOR_S(INDEX, COUNT, CODE)                      \
    static_assert(COUNT == 1 || COUNT == 2 || COUNT == 3 ||     \
                  COUNT == 4 || COUNT == 7 || COUNT == 8 ||     \
                  COUNT == 15 || COUNT == 16, "!");             \
    switch(COUNT) {                                             \
        /*case  0: UNROLLED_FOR_0(INDEX, COUNT, CODE); break;*/ \
        case  1: UNROLLED_FOR_1(INDEX, COUNT, CODE); break;     \
        case  2: UNROLLED_FOR_2(INDEX, COUNT, CODE); break;     \
        case  3: UNROLLED_FOR_3(INDEX, COUNT, CODE); break;     \
        case  4: UNROLLED_FOR_4(INDEX, COUNT, CODE); break;     \
        /*case  5: UNROLLED_FOR_5(INDEX, COUNT, CODE); break;*/ \
        /*case  6: UNROLLED_FOR_6(INDEX, COUNT, CODE); break;*/ \
        case  7: UNROLLED_FOR_7(INDEX, COUNT, CODE); break;     \
        case  8: UNROLLED_FOR_8(INDEX, COUNT, CODE); break;     \
        /*case  9: UNROLLED_FOR_9(INDEX, COUNT, CODE); break;*/ \
        /*case 10: UNROLLED_FOR_10(INDEX, COUNT, CODE); break;*/\
        /*case 11: UNROLLED_FOR_11(INDEX, COUNT, CODE); break;*/\
        /*case 12: UNROLLED_FOR_12(INDEX, COUNT, CODE); break;*/\
        /*case 13: UNROLLED_FOR_13(INDEX, COUNT, CODE); break;*/\
        /*case 14: UNROLLED_FOR_14(INDEX, COUNT, CODE); break;*/\
        case 15: UNROLLED_FOR_15(INDEX, COUNT, CODE); break;    \
        case 16: UNROLLED_FOR_16(INDEX, COUNT, CODE); break;    \
        default:                                      break;    \
    }

#define UNROLLED_FOR_R(INDEX, COUNT, CODE)              \
    switch(COUNT) {                                     \
        case 16: { const int INDEX = 15 % COUNT; CODE } \
        case 15: { const int INDEX = 14 % COUNT; CODE } \
        case 14: { const int INDEX = 13 % COUNT; CODE } \
        case 13: { const int INDEX = 12 % COUNT; CODE } \
        case 12: { const int INDEX = 11 % COUNT; CODE } \
        case 11: { const int INDEX = 10 % COUNT; CODE } \
        case 10: { const int INDEX =  9 % COUNT; CODE } \
        case  9: { const int INDEX =  8 % COUNT; CODE } \
        case  8: { const int INDEX =  7 % COUNT; CODE } \
        case  7: { const int INDEX =  6 % COUNT; CODE } \
        case  6: { const int INDEX =  5 % COUNT; CODE } \
        case  5: { const int INDEX =  4 % COUNT; CODE } \
        case  4: { const int INDEX =  3 % COUNT; CODE } \
        case  3: { const int INDEX =  2 % COUNT; CODE } \
        case  2: { const int INDEX =  1 % COUNT; CODE } \
        case  1: { const int INDEX =  0 % COUNT; CODE } \
    }
// clang-format on

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace Ray {
namespace NS {

enum vector_aligned_tag { vector_aligned };

template <typename T, int S> class fixed_size_simd {
    T comp_[S];

    friend class fixed_size_simd<int, S>;
    friend class fixed_size_simd<unsigned, S>;
    friend class fixed_size_simd<float, S>;

  public:
    fixed_size_simd() = default;
    fixed_size_simd(T f) {
        UNROLLED_FOR_S(i, S, { comp_[i] = f; })
    }
    template <typename... Tail>
    force_inline fixed_size_simd(typename std::enable_if<sizeof...(Tail) + 1 == S, T>::type head, Tail... tail)
        : comp_{head, T(tail)...} {}
    force_inline explicit fixed_size_simd(const T *f) { memcpy(&comp_, f, S * sizeof(T)); }
    force_inline fixed_size_simd(const T *_f, vector_aligned_tag) {
        const auto *f = (const T *)assume_aligned(_f, sizeof(T));
        memcpy(&comp_[0], f, S * sizeof(T));
    }

    force_inline T operator[](const int i) const { return comp_[i]; }
    force_inline T operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline T get() const { return comp_[i]; }
    template <int i> force_inline void set(const T f) { comp_[i] = f; }
    force_inline void set(const int i, const T f) { comp_[i] = f; }

    fixed_size_simd<T, S> &operator+=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] += rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<T, S> &operator-=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] -= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<T, S> &operator*=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<T, S> &operator/=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<T, S> &operator|=(const fixed_size_simd<T, S> &rhs) {
        const auto *src2 = reinterpret_cast<const uint8_t *>(&rhs.comp_[0]);

        auto *dst = reinterpret_cast<uint8_t *>(&comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] |= src2[i];
        }

        return *this;
    }

    fixed_size_simd<T, S> &operator^=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S, { comp_[i] ^= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<T, S> operator-() const {
        fixed_size_simd<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = -comp_[i]; })
        return temp;
    }

    fixed_size_simd<T, S> operator==(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] == rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> operator!=(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] != rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> operator<(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] < rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> operator<=(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] <= rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> operator>(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] > rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> operator>=(const fixed_size_simd<T, S> &rhs) const {
        fixed_size_simd<int, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = comp_[i] >= rhs.comp_[i] ? -1 : 0; })

        static_assert(sizeof(fixed_size_simd<T, S>) == sizeof(fixed_size_simd<int, S>), "!");

        fixed_size_simd<T, S> ret;
        memcpy(&ret, &temp, sizeof(fixed_size_simd<T, S>));

        return ret;
    }

    fixed_size_simd<T, S> &operator&=(const fixed_size_simd<T, S> &rhs) {
        UNROLLED_FOR_S(i, S,
                       { reinterpret_cast<uint32_t &>(comp_[i]) &= reinterpret_cast<const uint32_t &>(rhs.comp_[i]); })
        return *this;
    }

    fixed_size_simd<T, S> operator~() const {
        fixed_size_simd<T, S> ret;
        UNROLLED_FOR_S(i, S, {
            const uint32_t temp = ~reinterpret_cast<const uint32_t &>(comp_[i]);
            ret.comp_[i] = reinterpret_cast<const T &>(temp);
        })
        return ret;
    }

    explicit operator fixed_size_simd<int, S>() const {
        fixed_size_simd<int, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = int(comp_[i]); })
        return ret;
    }

    explicit operator fixed_size_simd<unsigned, S>() const {
        fixed_size_simd<unsigned, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = unsigned(comp_[i]); })
        return ret;
    }

    explicit operator fixed_size_simd<float, S>() const {
        fixed_size_simd<float, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = float(comp_[i]); })
        return ret;
    }

    fixed_size_simd<T, S> sqrt() const {
        fixed_size_simd<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.set<i>(std::sqrt(comp_[i])); })
        return temp;
    }

    fixed_size_simd<T, S> log() const {
        fixed_size_simd<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.set<i>(std::log(comp_[i])); })
        return temp;
    }

    T length() const {
        T temp = {0};
        UNROLLED_FOR_S(i, S, { temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    T length2() const {
        T temp = {0};
        UNROLLED_FOR_S(i, S, { temp += comp_[i] * comp_[i]; })
        return temp;
    }

    T hsum() const {
        T temp = {0};
        UNROLLED_FOR_S(i, S, { temp += comp_[i]; })
        return temp;
    }

    force_inline void store_to(T *f) const { memcpy(f, &comp_[0], S * sizeof(T)); }

    force_inline void store_to(T *_f, vector_aligned_tag) const {
        auto *f = (T *)assume_aligned(_f, sizeof(T));
        memcpy(f, &comp_[0], S * sizeof(T));
    }

    bool all_zeros() const {
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != 0) {
                return false;
            }
        })
        return true;
    }

    bool all_zeros(const fixed_size_simd<int, S> &mask) const {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&mask.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            if ((src1[i] & src2[i]) != 0) {
                return false;
            }
        }

        return true;
    }

    bool not_all_zeros() const {
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != 0) {
                return true;
            }
        })
        return false;
    }

    // clang-format off
    void blend_to(const fixed_size_simd<T, S> &mask, const fixed_size_simd<T, S> &v1) {
        UNROLLED_FOR_S(i, S, {
            if (mask.comp_[i] != T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    }

    void blend_inv_to(const fixed_size_simd<T, S> &mask, const fixed_size_simd<T, S> &v1) {
        UNROLLED_FOR_S(i, S, {
            if (mask.comp_[i] == T(0)) {
                comp_[i] = v1.comp_[i];
            }
        })
    } // clang-format on

    int movemask() const {
        int res = 0;
        UNROLLED_FOR_S(i, S, {
            if (comp_[i] != T(0)) {
                res |= (1 << i);
            }
        })
        return res;
    }

    friend fixed_size_simd<T, S> min(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        fixed_size_simd<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    friend fixed_size_simd<T, S> max(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        fixed_size_simd<T, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
        return temp;
    }

    static fixed_size_simd<T, S> and_not(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);

        fixed_size_simd<T, S> ret;

        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);

        for (int i = 0; i < S * sizeof(T); i++) {
            dst[i] = (~src1[i]) & src2[i];
        }

        return ret;
    }

    static fixed_size_simd<float, S> floor(const fixed_size_simd<float, S> &v1) {
        fixed_size_simd<float, S> temp;
        UNROLLED_FOR_S(i, S, { temp.comp_[i] = float(int(v1.comp_[i]) - (v1.comp_[i] < 0.0f)); })
        return temp;
    }

    static fixed_size_simd<float, S> ceil(const fixed_size_simd<float, S> &v1) {
        fixed_size_simd<float, S> temp;
        UNROLLED_FOR_S(i, S, {
            int _v = int(v1.comp_[i]);
            temp.comp_[i] = float(_v + (v1.comp_[i] != _v));
        })
        return temp;
    }

#define DEFINE_BITS_OPERATOR(OP)                                                                                       \
    friend fixed_size_simd<T, S> operator OP(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {       \
        const auto *src1 = reinterpret_cast<const uint8_t *>(&v1.comp_[0]);                                            \
        const auto *src2 = reinterpret_cast<const uint8_t *>(&v2.comp_[0]);                                            \
        fixed_size_simd<T, S> ret;                                                                                     \
        auto *dst = reinterpret_cast<uint8_t *>(&ret.comp_[0]);                                                        \
        for (int i = 0; i < S * sizeof(T); i++) {                                                                      \
            dst[i] = src1[i] OP src2[i];                                                                               \
        }                                                                                                              \
        return ret;                                                                                                    \
    }

    DEFINE_BITS_OPERATOR(&)
    DEFINE_BITS_OPERATOR(|)
    DEFINE_BITS_OPERATOR(^)

#undef DEFINE_BITS_OPERATOR

#define DEFINE_ARITHMETIC_OPERATOR(OP)                                                                                 \
    friend fixed_size_simd<T, S> operator OP(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {       \
        fixed_size_simd<T, S> ret;                                                                                     \
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] OP v2.comp_[i]; })                                           \
        return ret;                                                                                                    \
    }

    DEFINE_ARITHMETIC_OPERATOR(+)
    DEFINE_ARITHMETIC_OPERATOR(-)
    DEFINE_ARITHMETIC_OPERATOR(*)
    DEFINE_ARITHMETIC_OPERATOR(/)
    DEFINE_ARITHMETIC_OPERATOR(>>)
    DEFINE_ARITHMETIC_OPERATOR(<<)

#undef DEFINE_ARITHMETIC_OPERATOR

    friend fixed_size_simd<T, S> srai(const fixed_size_simd<T, S> &v1, int v2) {
        fixed_size_simd<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend fixed_size_simd<T, S> srli(const fixed_size_simd<T, S> &v1, int v2) {
        fixed_size_simd<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = unsigned(v1.comp_[i]) >> v2; })
        return ret;
    }

    friend T dot(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        T ret = {0};
        UNROLLED_FOR_S(i, S, { ret += v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<T, S> clamp(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &_min,
                                                    const fixed_size_simd<T, S> &_max) {
        return min(max(v1, _min), _max);
    }

    friend force_inline fixed_size_simd<T, S> saturate(const fixed_size_simd<T, S> &v1) {
        return clamp(v1, T(0), T(1));
    }

    friend fixed_size_simd<T, S> pow(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        fixed_size_simd<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend fixed_size_simd<T, S> exp(const fixed_size_simd<T, S> &v1) {
        fixed_size_simd<T, S> ret;
        UNROLLED_FOR_S(i, S, { ret.comp_[i] = std::exp(v1.comp_[i]); })
        return ret;
    }

    friend force_inline fixed_size_simd<T, S> normalize(const fixed_size_simd<T, S> &v1) { return v1 / v1.length(); }

    friend force_inline fixed_size_simd<T, S> normalize_len(const fixed_size_simd<T, S> &v1, T &out_len) {
        return v1 / (out_len = v1.length());
    }

    friend bool is_equal(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
        bool res = true;
        UNROLLED_FOR_S(i, S, { res = res && (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    friend force_inline const T *value_ptr(const fixed_size_simd<T, S> &v1) { return &v1.comp_[0]; }
    friend force_inline T *value_ptr(fixed_size_simd<T, S> &v1) { return &v1.comp_[0]; }

    static int size() { return S; }
    static bool is_native() { return false; }
};

template <typename T, int S>
force_inline fixed_size_simd<T, S> and_not(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
    return fixed_size_simd<T, S>::and_not(v1, v2);
}

template <typename T, int S> force_inline fixed_size_simd<T, S> floor(const fixed_size_simd<T, S> &v1) {
    return fixed_size_simd<T, S>::floor(v1);
}

template <typename T, int S> force_inline fixed_size_simd<T, S> ceil(const fixed_size_simd<T, S> &v1) {
    return fixed_size_simd<T, S>::ceil(v1);
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> mod(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
    return v1 - v2 * floor(v1 / v2);
}

template <typename T, int S> force_inline fixed_size_simd<T, S> sqrt(const fixed_size_simd<T, S> &v1) {
    return v1.sqrt();
}
template <typename T, int S> force_inline fixed_size_simd<T, S> log(const fixed_size_simd<T, S> &v1) {
    return v1.log();
}

template <typename T, int S> force_inline T length(const fixed_size_simd<T, S> &v1) { return v1.length(); }

template <typename T, int S> force_inline T length2(const fixed_size_simd<T, S> &v1) { return v1.length2(); }

template <typename T, int S> force_inline T hsum(const fixed_size_simd<T, S> &v1) { return v1.hsum(); }

template <typename T, int S> force_inline fixed_size_simd<T, S> fract(const fixed_size_simd<T, S> &v1) {
    return v1 - floor(v1);
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> max(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2) {
    return fixed_size_simd<T, S>::max(v1, v2);
}

template <typename T, int S> force_inline fixed_size_simd<T, S> abs(const fixed_size_simd<T, S> &v) {
    // TODO: find faster implementation
    return max(v, -v);
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmadd(const fixed_size_simd<T, S> &a, const fixed_size_simd<T, S> &b,
                                         const fixed_size_simd<T, S> &c) {
    return a * b + c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmadd(const fixed_size_simd<T, S> &a, const float b,
                                         const fixed_size_simd<T, S> &c) {
    return a * b + c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmadd(const float a, const fixed_size_simd<T, S> &b, const float c) {
    return a * b + c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmsub(const fixed_size_simd<T, S> &a, const fixed_size_simd<T, S> &b,
                                         const fixed_size_simd<T, S> &c) {
    return a * b - c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmsub(const fixed_size_simd<T, S> &a, const float b,
                                         const fixed_size_simd<T, S> &c) {
    return a * b - c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> fmsub(const float a, const fixed_size_simd<T, S> &b, const float c) {
    return a * b - c;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> mix(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2, T k) {
    return (1.0f - k) * v1 + k * v2;
}

template <typename T, int S>
force_inline fixed_size_simd<T, S> mix(const fixed_size_simd<T, S> &v1, const fixed_size_simd<T, S> &v2,
                                       fixed_size_simd<T, S> k) {
    return (fixed_size_simd<T, S>{1} - k) * v1 + k * v2;
}

template <typename T, int S> fixed_size_simd<T, S> gather(const T *base_addr, const fixed_size_simd<int, S> &vindex) {
    fixed_size_simd<T, S> res;
    UNROLLED_FOR_S(i, S, { res.template set<i>(base_addr[vindex.template get<i>()]); });
    return res;
}

template <typename T, int S>
fixed_size_simd<T, S> gather(const fixed_size_simd<T, S> &src, const T *base_addr, const fixed_size_simd<int, S> &mask,
                             const fixed_size_simd<int, S> &vindex) {
    fixed_size_simd<T, S> res = src;
    UNROLLED_FOR_S(i, S, {
        if (mask.template get<i>()) {
            res.template set<i>(base_addr[vindex.template get<i>()]);
        }
    });
    return res;
}

template <typename T, int S>
void scatter(T *base_addr, const fixed_size_simd<int, S> &vindex, const fixed_size_simd<T, S> &v) {
    UNROLLED_FOR_S(i, S, { base_addr[vindex.template get<i>()] = v.template get<i>(); });
}

template <typename T, int S> void scatter(T *base_addr, const fixed_size_simd<int, S> &vindex, const T v) {
    UNROLLED_FOR_S(i, S, { base_addr[vindex.template get<i>()] = v; });
}

template <typename T, int S>
void scatter(T *base_addr, const fixed_size_simd<int, S> &mask, const fixed_size_simd<int, S> &vindex,
             const fixed_size_simd<T, S> &v) {
    UNROLLED_FOR_S(i, S, {
        if (mask.template get<i>()) {
            base_addr[vindex.template get<i>()] = v.template get<i>();
        }
    });
}

template <typename T, int S>
void scatter(T *base_addr, const fixed_size_simd<int, S> &mask, const fixed_size_simd<int, S> &vindex, const T v) {
    UNROLLED_FOR_S(i, S, {
        if (mask.template get<i>()) {
            base_addr[vindex.template get<i>()] = v;
        }
    });
}

template <typename T, int S> fixed_size_simd<T, S> inclusive_scan(const fixed_size_simd<T, S> &vec) {
    fixed_size_simd<T, S> res = vec;
    UNROLLED_FOR_S(i, S - 1, { res.template set<i + 1>(res.template get<i + 1>() + res.template get<i>()); });
    return res;
}

template <typename T, int S>
fixed_size_simd<T, S> copysign(const fixed_size_simd<T, S> &val, const fixed_size_simd<T, S> &sign) {
    fixed_size_simd<T, S> res;
    UNROLLED_FOR_S(i, S, { res.template set<i>(std::copysign(val.template get<i>(), sign.template get<i>())); });
    return res;
}

template <typename T, typename U, int S> class simd_where_expression {
    const fixed_size_simd<T, S> &mask_;
    fixed_size_simd<T, S> &comp_;

  public:
    force_inline simd_where_expression(const fixed_size_simd<U, S> &mask, fixed_size_simd<T, S> &vec)
        : mask_(reinterpret_cast<const fixed_size_simd<T, S> &>(mask)), comp_(vec) {}

    force_inline void operator=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, vec); }
    force_inline void operator+=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ + vec); }
    force_inline void operator-=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ - vec); }
    force_inline void operator*=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ * vec); }
    force_inline void operator/=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ / vec); }
    force_inline void operator|=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ | vec); }
    force_inline void operator&=(const fixed_size_simd<T, S> &vec) && { comp_.blend_to(mask_, comp_ & vec); }
};

template <typename T, typename U, int S> class simd_where_inv_expression {
    const fixed_size_simd<T, S> &mask_;
    fixed_size_simd<T, S> &comp_;

  public:
    force_inline simd_where_inv_expression(const fixed_size_simd<U, S> &mask, fixed_size_simd<T, S> &vec)
        : mask_(reinterpret_cast<const fixed_size_simd<T, S> &>(mask)), comp_(vec) {}

    force_inline void operator=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, vec); }
    force_inline void operator+=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ + vec); }
    force_inline void operator-=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ - vec); }
    force_inline void operator*=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ * vec); }
    force_inline void operator/=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ / vec); }
    force_inline void operator|=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ | vec); }
    force_inline void operator&=(const fixed_size_simd<T, S> &vec) && { comp_.blend_inv_to(mask_, comp_ & vec); }
};

template <typename T, typename U, int S>
force_inline simd_where_expression<T, U, S> where(const fixed_size_simd<U, S> &mask, fixed_size_simd<T, S> &vec) {
    return {mask, vec};
}

template <typename T, typename U, int S>
force_inline simd_where_inv_expression<T, U, S> where_not(const fixed_size_simd<U, S> &mask,
                                                          fixed_size_simd<T, S> &vec) {
    return {mask, vec};
}

template <typename T, typename U, int S>
inline fixed_size_simd<T, S> select(const fixed_size_simd<U, S> &mask, const fixed_size_simd<T, S> &vec1,
                                    const fixed_size_simd<T, S> &vec2) {
    fixed_size_simd<T, S> ret;
    UNROLLED_FOR_S(i, S,
                   { ret.template set<i>(mask.template get<i>() ? vec1.template get<i>() : vec2.template get<i>()); });
    return ret;
}

template <int S> force_inline fixed_size_simd<int, S> simd_cast(const fixed_size_simd<float, S> &vec) {
    fixed_size_simd<int, S> ret;
    memcpy(&ret, &vec, sizeof(fixed_size_simd<int, S>));
    return ret;
}

template <int S> force_inline const fixed_size_simd<float, S> simd_cast(const fixed_size_simd<int, S> &vec) {
    fixed_size_simd<float, S> ret;
    memcpy(&ret, &vec, sizeof(fixed_size_simd<float, S>));
    return ret;
}

} // namespace NS
} // namespace Ray

#if defined(USE_SSE2) || defined(USE_SSE41)
#include "simd_sse.h"
#elif defined(USE_AVX) || defined(USE_AVX2)
#include "simd_avx.h"
#elif defined(USE_AVX512)
#include "simd_avx512.h"
#elif defined(USE_NEON)
#include "simd_neon.h"
#endif

namespace Ray {
namespace NS {
template <int S> using fvec = fixed_size_simd<float, S>;
using fvec2 = fvec<2>;
using fvec3 = fvec<3>;
using fvec4 = fvec<4>;
using fvec8 = fvec<8>;
using fvec16 = fvec<16>;

template <int S> using ivec = fixed_size_simd<int, S>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;
using ivec8 = ivec<8>;
using ivec16 = ivec<16>;

template <int S> using uvec = fixed_size_simd<unsigned, S>;
using uvec2 = uvec<2>;
using uvec3 = uvec<3>;
using uvec4 = uvec<4>;
using uvec8 = uvec<8>;
using uvec16 = uvec<16>;

template <int S> using dvec = fixed_size_simd<double, S>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;
using dvec8 = dvec<8>;
using dvec16 = dvec<16>;
} // namespace NS
} // namespace Ray

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#pragma warning(pop)
