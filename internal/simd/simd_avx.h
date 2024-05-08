// #pragma once

#include "simd_sse.h"

#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define _mm256_test_all_zeros(mask, val) _mm256_testz_si256((mask), (val))
#endif

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
#define USE_FMA
#endif

#pragma warning(push)
#pragma warning(disable : 4752)

#if defined(USE_AVX2) || defined(USE_AVX512)
#define avx2_inline force_inline
#else
#define avx2_inline inline
#endif

namespace Ray {
namespace NS {

template <> force_inline __m256 _mm_cast(__m256i x) { return _mm256_castsi256_ps(x); }
template <> force_inline __m256i _mm_cast(__m256 x) { return _mm256_castps_si256(x); }

template <> class fixed_size_simd<int, 8>;
template <> class fixed_size_simd<unsigned, 8>;

template <> class fixed_size_simd<float, 8> {
    union {
        __m256 vec_;
        float comp_[8];
    };

    friend class fixed_size_simd<int, 8>;
    friend class fixed_size_simd<unsigned, 8>;

    force_inline fixed_size_simd(const __m256 vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) { vec_ = _mm256_set1_ps(f); }
    force_inline fixed_size_simd(const float f1, const float f2, const float f3, const float f4, const float f5,
                                 const float f6, const float f7, const float f8) {
        vec_ = _mm256_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8);
    }
    force_inline explicit fixed_size_simd(const float *f) { vec_ = _mm256_loadu_ps(f); }
    force_inline fixed_size_simd(const float *f, vector_aligned_tag) { vec_ = _mm256_load_ps(f); }

    force_inline float operator[](const int i) const { return comp_[i]; }
    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float get() const { return comp_[i & 7]; }
    template <int i> force_inline void set(const float v) { comp_[i & 7] = v; }
    force_inline void set(const int i, const float v) { comp_[i] = v; }

    force_inline fixed_size_simd<float, 8> &vectorcall operator+=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> &vectorcall operator-=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> &vectorcall operator*=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> &vectorcall operator/=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> &vectorcall operator|=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> &vectorcall operator&=(const fixed_size_simd<float, 8> rhs) {
        vec_ = _mm256_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 8> operator~() const;
    force_inline fixed_size_simd<float, 8> operator-() const;
    force_inline explicit vectorcall operator fixed_size_simd<int, 8>() const;
    force_inline explicit vectorcall operator fixed_size_simd<unsigned, 8>() const;

    force_inline fixed_size_simd<float, 8> sqrt() const;
    force_inline fixed_size_simd<float, 8> log() const;

    force_inline float length() const { return sqrtf(length2()); }

    float length2() const {
        float ret = 0;
        UNROLLED_FOR(i, 8, { ret += comp_[i] * comp_[i]; })
        return ret;
    }

    force_inline float hsum() const {
#if 1
        __m256 temp = _mm256_hadd_ps(vec_, vec_);
        temp = _mm256_hadd_ps(temp, temp);

        __m256 ret = _mm256_permute2f128_ps(temp, temp, 1);
        ret = _mm256_add_ps(ret, temp);

        return _mm256_cvtss_f32(ret);
#else
        // ( x3+x7, x2+x6, x1+x5, x0+x4 )
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(vec_, 1), _mm256_castps256_ps128(vec_));
        // ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 )
        const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        // ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 )
        const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        // Conversion to float is a no-op on x86-64
        return _mm_cvtss_f32(x32);
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline fixed_size_simd<float, 8> vectorcall inclusive_scan(fixed_size_simd<float, 8> v1) {
        v1.vec_ = _mm256_add_ps(v1.vec_, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v1.vec_), 4)));
        v1.vec_ = _mm256_add_ps(v1.vec_, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v1.vec_), 8)));

        __m256 temp = _mm256_shuffle_ps(v1.vec_, v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2f128_ps(_mm256_setzero_ps(), temp, 0x20);

        v1.vec_ = _mm256_add_ps(v1.vec_, temp);

        return v1;
    }
#endif

    friend force_inline fixed_size_simd<float, 8> vectorcall copysign(const fixed_size_simd<float, 8> val,
                                                                      const fixed_size_simd<float, 8> sign) {
        const __m256 sign_mask = _mm256_and_ps(sign.vec_, _mm256_set1_ps(-0.0f));
        const __m256 abs_val = _mm256_andnot_ps(sign_mask, val.vec_);
        return _mm256_or_ps(abs_val, sign_mask);
    }

    force_inline void store_to(float *f) const { _mm256_storeu_ps(f, vec_); }
    force_inline void store_to(float *f, vector_aligned_tag) const { _mm256_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float, 8> mask, const fixed_size_simd<float, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_blendv_ps(vec_, v1.vec_, mask.vec_);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float, 8> mask,
                                              const fixed_size_simd<float, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_blendv_ps(v1.vec_, vec_, mask.vec_);
    }

    friend force_inline fixed_size_simd<float, 8> vectorcall min(fixed_size_simd<float, 8> v1,
                                                                 fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall max(fixed_size_simd<float, 8> v1,
                                                                 fixed_size_simd<float, 8> v2);

    friend force_inline fixed_size_simd<float, 8> vectorcall and_not(fixed_size_simd<float, 8> v1,
                                                                     fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall floor(fixed_size_simd<float, 8> v1);
    friend force_inline fixed_size_simd<float, 8> vectorcall ceil(fixed_size_simd<float, 8> v1);

    friend force_inline fixed_size_simd<float, 8> vectorcall operator&(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator|(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator^(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator+(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator-(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator*(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator/(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);

    friend force_inline fixed_size_simd<float, 8> vectorcall operator<(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator<=(fixed_size_simd<float, 8> v1,
                                                                        fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator>(fixed_size_simd<float, 8> v1,
                                                                       fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator>=(fixed_size_simd<float, 8> v1,
                                                                        fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator==(fixed_size_simd<float, 8> v1,
                                                                        fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall operator!=(fixed_size_simd<float, 8> v1,
                                                                        fixed_size_simd<float, 8> v2);

    friend force_inline fixed_size_simd<float, 8>
        vectorcall clamp(fixed_size_simd<float, 8> v1, fixed_size_simd<float, 8> min, fixed_size_simd<float, 8> max);
    // friend force_inline fixed_size_simd<float, 8> vectorcall clamp(fixed_size_simd<float, 8> v1, float min, float
    // max);
    friend force_inline fixed_size_simd<float, 8> vectorcall saturate(fixed_size_simd<float, 8> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }
    friend force_inline fixed_size_simd<float, 8> vectorcall pow(fixed_size_simd<float, 8> v1,
                                                                 fixed_size_simd<float, 8> v2);
    friend force_inline fixed_size_simd<float, 8> vectorcall exp(fixed_size_simd<float, 8> v1);

    friend force_inline fixed_size_simd<float, 8> vectorcall normalize(fixed_size_simd<float, 8> v1);
    friend force_inline fixed_size_simd<float, 8> vectorcall normalize_len(fixed_size_simd<float, 8> v1,
                                                                           float &out_len);

#ifdef USE_FMA
    friend force_inline fixed_size_simd<float, 8>
        vectorcall fmadd(fixed_size_simd<float, 8> a, fixed_size_simd<float, 8> b, fixed_size_simd<float, 8> c);
    friend force_inline fixed_size_simd<float, 8>
        vectorcall fmsub(fixed_size_simd<float, 8> a, fixed_size_simd<float, 8> b, fixed_size_simd<float, 8> c);
#endif // USE_FMA

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline fixed_size_simd<float, 8> vectorcall gather(const float *base_addr,
                                                                    fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<float, 8> vectorcall gather(fixed_size_simd<float, 8> src,
                                                                    const float *base_addr,
                                                                    fixed_size_simd<int, 8> mask,
                                                                    fixed_size_simd<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline fixed_size_simd<float, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                    const fixed_size_simd<float, 8> vec1,
                                                                    const fixed_size_simd<float, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                  const fixed_size_simd<int, 8> vec1,
                                                                  const fixed_size_simd<int, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                       const fixed_size_simd<unsigned, 8> vec1,
                                                                       const fixed_size_simd<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *value_ptr(const fixed_size_simd<float, 8> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(fixed_size_simd<float, 8> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 8; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<int, 8> {
    union {
        __m256i vec_;
        int comp_[8];
    };

    friend class fixed_size_simd<float, 8>;
    friend class fixed_size_simd<unsigned, 8>;

    force_inline fixed_size_simd(const __m256i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const int f) { vec_ = _mm256_set1_epi32(f); }
    force_inline fixed_size_simd(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6,
                                 const int i7, const int i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline explicit fixed_size_simd(const int *f) { vec_ = _mm256_loadu_si256((const __m256i *)f); }
    force_inline fixed_size_simd(const int *f, vector_aligned_tag) { vec_ = _mm256_load_si256((const __m256i *)f); }

    force_inline int operator[](const int i) const { return comp_[i]; }
    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const { return _mm256_extract_epi32(vec_, i & 7); }
    template <int i> force_inline void set(const int v) { vec_ = _mm256_insert_epi32(vec_, v, i & 7); }
    force_inline void set(const int i, const int v) { comp_[i] = v; }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator+=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] += rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline fixed_size_simd<int, 8> &vectorcall operator+=(const int rhs) {
        return operator+=(fixed_size_simd<int, 8>{rhs});
    }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator-=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] -= rhs.comp_[i]; })
#endif
        return *this;
    }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator*=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_mullo_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] *= rhs.comp_[i]; })
#endif
        return *this;
    }

    fixed_size_simd<int, 8> &vectorcall operator/=(const fixed_size_simd<int, 8> rhs) {
        UNROLLED_FOR(i, 8, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator|=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] |= rhs.comp_[i]; })
#endif
        return *this;
    }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator^=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_xor_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] ^= rhs.comp_[i]; })
#endif
        return *this;
    }

    avx2_inline fixed_size_simd<int, 8> vectorcall operator-() const {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(_mm256_setzero_si256(), vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = -comp_[i]; })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<int, 8> vectorcall operator==(const fixed_size_simd<int, 8> rhs) const {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (comp_[i] == rhs.comp_[i]) ? -1 : 0; })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<int, 8> vectorcall operator!=(const fixed_size_simd<int, 8> rhs) const {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (comp_[i] != rhs.comp_[i]) ? -1 : 0; })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<int, 8> &vectorcall operator&=(const fixed_size_simd<int, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_and_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] &= rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline explicit vectorcall operator fixed_size_simd<float, 8>() const {
        fixed_size_simd<float, 8> ret;
        ret.vec_ = _mm256_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline explicit vectorcall operator fixed_size_simd<unsigned, 8>() const;

    avx2_inline int hsum() const {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i temp = _mm256_hadd_epi32(vec_, vec_);
        temp = _mm256_hadd_epi32(temp, temp);

        __m256i ret = _mm256_permute2f128_si256(temp, temp, 1);
        ret = _mm256_add_epi32(ret, temp);

        return _mm256_cvtsi256_si32(ret);
#else
        int ret = comp_[0];
        UNROLLED_FOR(i, 7, { ret += comp_[i + 1]; })
        return ret;
#endif
    }

    force_inline void store_to(int *f) const { _mm256_storeu_si256((__m256i *)f, vec_); }
    force_inline void store_to(int *f, vector_aligned_tag) const { _mm256_store_si256((__m256i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<int, 8> mask, const fixed_size_simd<int, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<int, 8> mask, const fixed_size_simd<int, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline int movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(vec_)); }

    force_inline bool vectorcall all_zeros() const { return _mm256_test_all_zeros(vec_, vec_) != 0; }
    force_inline bool vectorcall all_zeros(const fixed_size_simd<int, 8> mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool vectorcall not_all_zeros() const {
        int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall min(const fixed_size_simd<int, 8> v1,
                                                              const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_min_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
#endif
        return ret;
    }

    avx2_inline static fixed_size_simd<int, 8> vectorcall max(const fixed_size_simd<int, 8> v1,
                                                              const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_max_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
#endif
        return ret;
    }

    friend force_inline fixed_size_simd<int, 8> vectorcall clamp(const fixed_size_simd<int, 8> v1,
                                                                 const fixed_size_simd<int, 8> _min,
                                                                 const fixed_size_simd<int, 8> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<int, 8> vectorcall and_not(const fixed_size_simd<int, 8> v1,
                                                                   const fixed_size_simd<int, 8> v2) {
        return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<int, 8> vectorcall operator&(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<int, 8> vectorcall operator|(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<int, 8> vectorcall operator^(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator+(const fixed_size_simd<int, 8> v1,
                                                                    const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator-(const fixed_size_simd<int, 8> v1,
                                                                    const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator*(const fixed_size_simd<int, 8> v1,
                                                                    const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_mullo_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
#endif
        return ret;
    }

    friend fixed_size_simd<int, 8> vectorcall operator/(const fixed_size_simd<int, 8> v1,
                                                        const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] / v2.comp_[i]); })
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator<(const fixed_size_simd<int, 8> v1,
                                                                    const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v2.vec_, v1.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] < v2.comp_[i]) ? -1 : 0; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator>(const fixed_size_simd<int, 8> v1,
                                                                    const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpgt_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] > v2.comp_[i]) ? -1 : 0; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator>=(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpgt_epi32(v2.vec_, v1.vec_), _mm256_set1_epi32(-1));
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] >= v2.comp_[i]) ? -1 : 0; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator>>(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = int(unsigned(v1.comp_[i]) >> unsigned(v2.comp_[i])); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator>>(const fixed_size_simd<int, 8> v1, const int v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = int(unsigned(v1.comp_[i]) >> v2); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator<<(const fixed_size_simd<int, 8> v1,
                                                                     const fixed_size_simd<int, 8> v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = int(unsigned(v1.comp_[i]) << unsigned(v2.comp_[i])); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall operator<<(const fixed_size_simd<int, 8> v1, const int v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = int(unsigned(v1.comp_[i]) << v2); })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<int, 8> operator~() const {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(vec_, _mm256_set1_epi32(~0));
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = ~comp_[i]; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<int, 8> vectorcall srai(const fixed_size_simd<int, 8> v1, const int v2) {
        fixed_size_simd<int, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srai_epi32(v1.vec_, v2);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] >> v2); })
#endif
        return ret;
    }

    friend avx2_inline bool vectorcall is_equal(const fixed_size_simd<int, 8> v1, const fixed_size_simd<int, 8> v2) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i vcmp = _mm256_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm256_movemask_epi8(vcmp) == 0xffffffff);
#else
        bool ret = true;
        UNROLLED_FOR(i, 8, { ret &= (v1.comp_[i] == v2.comp_[i]); })
        return ret;
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline fixed_size_simd<int, 8> vectorcall inclusive_scan(fixed_size_simd<int, 8> v1) {
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 4));
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 8));

        __m256i temp = _mm256_shuffle_epi32(v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2x128_si256(_mm256_setzero_si256(), temp, 0x20);

        v1.vec_ = _mm256_add_epi32(v1.vec_, temp);

        return v1;
    }
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline fixed_size_simd<float, 8> vectorcall gather(const float *base_addr,
                                                                    fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<float, 8> vectorcall gather(fixed_size_simd<float, 8> src,
                                                                    const float *base_addr,
                                                                    fixed_size_simd<int, 8> mask,
                                                                    fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<int, 8> vectorcall gather(const int *base_addr, fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<int, 8> vectorcall gather(fixed_size_simd<int, 8> src, const int *base_addr,
                                                                  fixed_size_simd<int, 8> mask,
                                                                  fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall gather(const unsigned *base_addr,
                                                                       fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall gather(fixed_size_simd<unsigned, 8> src,
                                                                       const unsigned *base_addr,
                                                                       fixed_size_simd<unsigned, 8> mask,
                                                                       fixed_size_simd<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline fixed_size_simd<float, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                    const fixed_size_simd<float, 8> vec1,
                                                                    const fixed_size_simd<float, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                  const fixed_size_simd<int, 8> vec1,
                                                                  const fixed_size_simd<int, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                       const fixed_size_simd<unsigned, 8> vec1,
                                                                       const fixed_size_simd<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<int, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const fixed_size_simd<int, 8> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(fixed_size_simd<int, 8> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 8; }
    static bool is_native() {
#if defined(USE_AVX2) || defined(USE_AVX512)
        return true;
#else
        // mostly not native, so return false here
        return false;
#endif
    }
};

template <> class fixed_size_simd<unsigned, 8> {
    union {
        __m256i vec_;
        unsigned comp_[8];
    };

    friend class fixed_size_simd<float, 8>;
    friend class fixed_size_simd<int, 8>;

    force_inline fixed_size_simd(const __m256i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const unsigned f) { vec_ = _mm256_set1_epi32(f); }
    force_inline fixed_size_simd(const unsigned i1, const unsigned i2, const unsigned i3, const unsigned i4,
                                 const unsigned i5, const unsigned i6, const unsigned i7, const unsigned i8) {
        vec_ = _mm256_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8);
    }
    force_inline explicit fixed_size_simd(const unsigned *f) { vec_ = _mm256_loadu_si256((const __m256i *)f); }
    force_inline fixed_size_simd(const unsigned *f, vector_aligned_tag) {
        vec_ = _mm256_load_si256((const __m256i *)f);
    }

    force_inline unsigned operator[](const int i) const { return comp_[i]; }
    force_inline unsigned operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline unsigned get() const { return _mm256_extract_epi32(vec_, i & 7); }
    template <int i> force_inline void set(const unsigned v) { vec_ = _mm256_insert_epi32(vec_, v, i & 7); }
    force_inline void set(const int i, const unsigned v) { comp_[i] = v; }

    avx2_inline fixed_size_simd<unsigned, 8> &vectorcall operator+=(const fixed_size_simd<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_add_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] += rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 8> &vectorcall operator+=(const unsigned rhs) {
        return operator+=(fixed_size_simd<unsigned, 8>{rhs});
    }

    avx2_inline fixed_size_simd<unsigned, 8> &vectorcall operator-=(const fixed_size_simd<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_sub_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] -= rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 8> &vectorcall operator-=(const unsigned rhs) {
        return operator-=(fixed_size_simd<unsigned, 8>{rhs});
    }

    fixed_size_simd<unsigned, 8> &vectorcall operator*=(const fixed_size_simd<unsigned, 8> rhs) {
        UNROLLED_FOR(i, 8, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 8> &vectorcall operator*=(const unsigned rhs) {
        return operator*=(fixed_size_simd<unsigned, 8>{rhs});
    }

    fixed_size_simd<unsigned, 8> &vectorcall operator/=(const fixed_size_simd<unsigned, 8> rhs) {
        UNROLLED_FOR(i, 8, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 8> &vectorcall operator/=(const unsigned rhs) {
        return operator/=(fixed_size_simd<unsigned, 8>{rhs});
    }

    avx2_inline fixed_size_simd<unsigned, 8> &vectorcall operator|=(const fixed_size_simd<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_or_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] |= rhs.comp_[i]; })
#endif
        return *this;
    }

    avx2_inline fixed_size_simd<unsigned, 8> &vectorcall operator^=(const fixed_size_simd<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_xor_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] ^= rhs.comp_[i]; })
#endif
        return *this;
    }

    avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator==(const fixed_size_simd<unsigned, 8> rhs) const {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_cmpeq_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (comp_[i] == rhs.comp_[i]) ? 0xffffffff : 0; })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator!=(const fixed_size_simd<unsigned, 8> rhs) const {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_, rhs.vec_), _mm256_set1_epi32(~0));
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (comp_[i] != rhs.comp_[i]) ? 0xffffffff : 0; })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<unsigned, 8> &vectorcall operator&=(const fixed_size_simd<unsigned, 8> rhs) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        vec_ = _mm256_and_si256(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 8, { comp_[i] &= rhs.comp_[i]; })
#endif
        return *this;
    }

    force_inline explicit vectorcall operator fixed_size_simd<float, 8>() const { return _mm256_cvtepi32_ps(vec_); }

    force_inline explicit vectorcall operator fixed_size_simd<int, 8>() const {
        fixed_size_simd<int, 8> ret;
        ret.vec_ = vec_;
        return ret;
    }

    avx2_inline unsigned hsum() const {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i temp = _mm256_hadd_epi32(vec_, vec_);
        temp = _mm256_hadd_epi32(temp, temp);

        __m256i ret = _mm256_permute2f128_si256(temp, temp, 1);
        ret = _mm256_add_epi32(ret, temp);

        return _mm256_cvtsi256_si32(ret);
#else
        unsigned ret = comp_[0];
        UNROLLED_FOR(i, 7, { ret += comp_[i + 1]; })
        return ret;
#endif
    }

    force_inline void store_to(unsigned *f) const { _mm256_storeu_si256((__m256i *)f, vec_); }
    force_inline void store_to(unsigned *f, vector_aligned_tag) const { _mm256_store_si256((__m256i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<unsigned, 8> mask,
                                          const fixed_size_simd<unsigned, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vec_), _mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<unsigned, 8> mask,
                                              const fixed_size_simd<unsigned, 8> v1) {
        validate_mask(mask);
        vec_ = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(vec_), _mm256_castsi256_ps(mask.vec_)));
    }

    force_inline int movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(vec_)); }

    force_inline bool vectorcall all_zeros() const { return _mm256_test_all_zeros(vec_, vec_) != 0; }
    force_inline bool vectorcall all_zeros(const fixed_size_simd<unsigned, 8> mask) const {
        return _mm256_test_all_zeros(vec_, mask.vec_) != 0;
    }

    force_inline bool vectorcall not_all_zeros() const {
        int res = _mm256_test_all_zeros(vec_, vec_);
        return res == 0;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall min(const fixed_size_simd<unsigned, 8> v1,
                                                                   const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_min_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = std::min(v1.comp_[i], v2.comp_[i]); })
#endif
        return ret;
    }

    avx2_inline static fixed_size_simd<unsigned, 8> vectorcall max(const fixed_size_simd<unsigned, 8> v1,
                                                                   const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_max_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = std::max(v1.comp_[i], v2.comp_[i]); })
#endif
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 8> vectorcall clamp(const fixed_size_simd<unsigned, 8> v1,
                                                                      const fixed_size_simd<unsigned, 8> _min,
                                                                      const fixed_size_simd<unsigned, 8> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<unsigned, 8> vectorcall and_not(const fixed_size_simd<unsigned, 8> v1,
                                                                        const fixed_size_simd<unsigned, 8> v2) {
        return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<unsigned, 8> vectorcall operator&(const fixed_size_simd<unsigned, 8> v1,
                                                                          const fixed_size_simd<unsigned, 8> v2) {
        return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<unsigned, 8> vectorcall operator|(const fixed_size_simd<unsigned, 8> v1,
                                                                          const fixed_size_simd<unsigned, 8> v2) {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend force_inline fixed_size_simd<unsigned, 8> vectorcall operator^(const fixed_size_simd<unsigned, 8> v1,
                                                                          const fixed_size_simd<unsigned, 8> v2) {
        return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v1.vec_), _mm256_castsi256_ps(v2.vec_)));
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator+(const fixed_size_simd<unsigned, 8> v1,
                                                                         const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_add_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] + v2.comp_[i]; })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator-(const fixed_size_simd<unsigned, 8> v1,
                                                                         const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sub_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] - v2.comp_[i]; })
#endif
        return ret;
    }

    friend fixed_size_simd<unsigned, 8> vectorcall operator*(const fixed_size_simd<unsigned, 8> v1,
                                                             const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
        UNROLLED_FOR(i, 8, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 8> vectorcall operator/(const fixed_size_simd<unsigned, 8> v1,
                                                             const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] / v2.comp_[i]); })
        return ret;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator>>(const fixed_size_simd<unsigned, 8> v1,
                                                                          const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srlv_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] >> v2.comp_[i]); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator>>(const fixed_size_simd<unsigned, 8> v1,
                                                                          const unsigned v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_srli_epi32(v1.vec_, v2);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] >> v2); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator<<(const fixed_size_simd<unsigned, 8> v1,
                                                                          const fixed_size_simd<unsigned, 8> v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_sllv_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] << v2.comp_[i]); })
#endif
        return ret;
    }

    friend avx2_inline fixed_size_simd<unsigned, 8> vectorcall operator<<(const fixed_size_simd<unsigned, 8> v1,
                                                                          const unsigned v2) {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_slli_epi32(v1.vec_, v2);
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = (v1.comp_[i] << v2); })
#endif
        return ret;
    }

    avx2_inline fixed_size_simd<unsigned, 8> operator~() const {
        fixed_size_simd<unsigned, 8> ret;
#if defined(USE_AVX2) || defined(USE_AVX512)
        ret.vec_ = _mm256_andnot_si256(vec_, _mm256_set1_epi32(~0));
#else
        UNROLLED_FOR(i, 8, { ret.comp_[i] = ~comp_[i]; })
#endif
        return ret;
    }

    friend avx2_inline bool vectorcall is_equal(const fixed_size_simd<unsigned, 8> v1,
                                                const fixed_size_simd<unsigned, 8> v2) {
#if defined(USE_AVX2) || defined(USE_AVX512)
        __m256i vcmp = _mm256_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm256_movemask_epi8(vcmp) == 0xffffffff);
#else
        bool ret = true;
        UNROLLED_FOR(i, 8, { ret &= (v1.comp_[i] == v2.comp_[i]); })
        return ret;
#endif
    }

#if defined(USE_AVX2) || defined(USE_AVX512)
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall inclusive_scan(fixed_size_simd<unsigned, 8> v1) {
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 4));
        v1.vec_ = _mm256_add_epi32(v1.vec_, _mm256_slli_si256(v1.vec_, 8));

        __m256i temp = _mm256_shuffle_epi32(v1.vec_, _MM_SHUFFLE(3, 3, 3, 3));
        temp = _mm256_permute2x128_si256(_mm256_setzero_si256(), temp, 0x20);

        v1.vec_ = _mm256_add_epi32(v1.vec_, temp);

        return v1;
    }
#endif

#if defined(USE_AVX2) || defined(USE_AVX512)
    // friend force_inline fixed_size_simd<float, 8> vectorcall gather(const float *base_addr, fixed_size_simd<int, 8>
    // vindex); friend force_inline fixed_size_simd<float, 8> vectorcall gather(fixed_size_simd<float, 8> src, const
    // float *base_addr,
    //                                                          fixed_size_simd<int, 8> mask, fixed_size_simd<int, 8>
    //                                                          vindex);
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall gather(const unsigned *base_addr,
                                                                       fixed_size_simd<int, 8> vindex);
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall gather(fixed_size_simd<unsigned, 8> src,
                                                                       const unsigned *base_addr,
                                                                       fixed_size_simd<unsigned, 8> mask,
                                                                       fixed_size_simd<int, 8> vindex);
#endif

    template <typename U>
    friend force_inline fixed_size_simd<float, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                    const fixed_size_simd<float, 8> vec1,
                                                                    const fixed_size_simd<float, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                  const fixed_size_simd<int, 8> vec1,
                                                                  const fixed_size_simd<int, 8> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                                       const fixed_size_simd<unsigned, 8> vec1,
                                                                       const fixed_size_simd<unsigned, 8> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<unsigned, 8> mask) {
        UNROLLED_FOR(i, 8, {
            const int val = mask.get<i>();
            assert(val == 0 || val == 0xffffffff);
        })
    }
#endif

    friend force_inline const unsigned *value_ptr(const fixed_size_simd<unsigned, 8> &v1) {
        return reinterpret_cast<const unsigned *>(&v1.vec_);
    }
    friend force_inline unsigned *value_ptr(fixed_size_simd<unsigned, 8> &v1) {
        return reinterpret_cast<unsigned *>(&v1.vec_);
    }

    static int size() { return 8; }
    static bool is_native() {
#if defined(USE_AVX2) || defined(USE_AVX512)
        return true;
#else
        // mostly not native, so return false here
        return false;
#endif
    }
};

avx2_inline fixed_size_simd<float, 8> fixed_size_simd<float, 8>::operator~() const {
#if defined(USE_AVX2) || defined(USE_AVX512)
    fixed_size_simd<float, 8> ret;
    ret.vec_ = _mm256_castsi256_ps(_mm256_andnot_si256(_mm256_castps_si256(vec_), _mm256_set1_epi32(~0)));
    return ret;
#else
    alignas(32) uint32_t temp[8];
    _mm256_store_ps((float *)temp, vec_);
    UNROLLED_FOR(i, 8, { temp[i] = ~temp[i]; })
    return fixed_size_simd<float, 8>{(const float *)temp, vector_aligned};
#endif
}

force_inline fixed_size_simd<float, 8> fixed_size_simd<float, 8>::operator-() const {
    __m256 m = _mm256_set1_ps(-0.0f);
    return _mm256_xor_ps(vec_, m);
}

force_inline fixed_size_simd<float, 8>::operator fixed_size_simd<int, 8>() const { return _mm256_cvttps_epi32(vec_); }

force_inline fixed_size_simd<float, 8>::operator fixed_size_simd<unsigned, 8>() const {
    return _mm256_cvttps_epi32(vec_);
}

force_inline fixed_size_simd<int, 8>::operator fixed_size_simd<unsigned, 8>() const {
    fixed_size_simd<unsigned, 8> ret;
    ret.vec_ = vec_;
    return ret;
}

force_inline fixed_size_simd<float, 8> fixed_size_simd<float, 8>::sqrt() const { return _mm256_sqrt_ps(vec_); }

avx2_inline fixed_size_simd<float, 8> fixed_size_simd<float, 8>::log() const {
    fixed_size_simd<float, 8> ret;
    UNROLLED_FOR(i, 8, { ret.comp_[i] = logf(comp_[i]); })
    return ret;
}

force_inline fixed_size_simd<float, 8> vectorcall min(const fixed_size_simd<float, 8> v1,
                                                      const fixed_size_simd<float, 8> v2) {
    return _mm256_min_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall max(const fixed_size_simd<float, 8> v1,
                                                      const fixed_size_simd<float, 8> v2) {
    return _mm256_max_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall clamp(const fixed_size_simd<float, 8> v1,
                                                        const fixed_size_simd<float, 8> min,
                                                        const fixed_size_simd<float, 8> max) {
    return _mm256_max_ps(min.vec_, _mm256_min_ps(v1.vec_, max.vec_));
}

force_inline fixed_size_simd<float, 8> vectorcall and_not(const fixed_size_simd<float, 8> v1,
                                                          const fixed_size_simd<float, 8> v2) {
    return _mm256_andnot_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall floor(const fixed_size_simd<float, 8> v1) {
    return _mm256_floor_ps(v1.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall ceil(const fixed_size_simd<float, 8> v1) {
    return _mm256_ceil_ps(v1.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator&(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_and_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator|(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_or_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator^(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_xor_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator+(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_add_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator-(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_sub_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator*(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_mul_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator/(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_div_ps(v1.vec_, v2.vec_);
}

force_inline fixed_size_simd<float, 8> vectorcall operator<(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LT_OS);
}

force_inline fixed_size_simd<float, 8> vectorcall operator<=(const fixed_size_simd<float, 8> v1,
                                                             const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_LE_OS);
}

force_inline fixed_size_simd<float, 8> vectorcall operator>(const fixed_size_simd<float, 8> v1,
                                                            const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GT_OS);
}

force_inline fixed_size_simd<float, 8> vectorcall operator>=(const fixed_size_simd<float, 8> v1,
                                                             const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_GE_OS);
}

force_inline fixed_size_simd<float, 8> vectorcall operator==(const fixed_size_simd<float, 8> v1,
                                                             const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_EQ_OS);
}

force_inline fixed_size_simd<float, 8> vectorcall operator!=(const fixed_size_simd<float, 8> v1,
                                                             const fixed_size_simd<float, 8> v2) {
    return _mm256_cmp_ps(v1.vec_, v2.vec_, _CMP_NEQ_OS);
}

inline fixed_size_simd<float, 8> vectorcall pow(const fixed_size_simd<float, 8> v1,
                                                const fixed_size_simd<float, 8> v2) {
    alignas(32) float comp1[8], comp2[8];
    _mm256_store_ps(comp1, v1.vec_);
    _mm256_store_ps(comp2, v2.vec_);
    UNROLLED_FOR(i, 8, { comp1[i] = powf(comp1[i], comp2[i]); })
    return fixed_size_simd<float, 8>{comp1, vector_aligned};
}

inline fixed_size_simd<float, 8> vectorcall exp(const fixed_size_simd<float, 8> v1) {
    alignas(32) float comp1[8];
    _mm256_store_ps(comp1, v1.vec_);
    UNROLLED_FOR(i, 8, { comp1[i] = expf(comp1[i]); })
    return fixed_size_simd<float, 8>{comp1, vector_aligned};
}

force_inline fixed_size_simd<float, 8> vectorcall normalize(const fixed_size_simd<float, 8> v1) {
    return v1 / v1.length();
}

force_inline fixed_size_simd<float, 8> vectorcall normalize_len(const fixed_size_simd<float, 8> v1, float &out_len) {
    return v1 / (out_len = v1.length());
}

#ifdef USE_FMA
force_inline fixed_size_simd<float, 8> vectorcall fmadd(const fixed_size_simd<float, 8> a,
                                                        const fixed_size_simd<float, 8> b,
                                                        const fixed_size_simd<float, 8> c) {
    fixed_size_simd<float, 8> ret;
    ret.vec_ = _mm256_fmadd_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}

force_inline fixed_size_simd<float, 8> vectorcall fmsub(const fixed_size_simd<float, 8> a,
                                                        const fixed_size_simd<float, 8> b,
                                                        const fixed_size_simd<float, 8> c) {
    fixed_size_simd<float, 8> ret;
    ret.vec_ = _mm256_fmsub_ps(a.vec_, b.vec_, c.vec_);
    return ret;
}
#endif // USE_FMA

#if defined(USE_AVX2) || defined(USE_AVX512)
force_inline fixed_size_simd<float, 8> vectorcall gather(const float *base_addr, const fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<float, 8> ret;
    ret.vec_ = _mm256_i32gather_ps(base_addr, vindex.vec_, sizeof(float));
    return ret;
}

force_inline fixed_size_simd<float, 8> vectorcall gather(fixed_size_simd<float, 8> src, const float *base_addr,
                                                         fixed_size_simd<int, 8> mask, fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<float, 8> ret;
    ret.vec_ =
        _mm256_mask_i32gather_ps(src.vec_, base_addr, vindex.vec_, _mm256_castsi256_ps(mask.vec_), sizeof(float));
    return ret;
}

force_inline fixed_size_simd<int, 8> vectorcall gather(const int *base_addr, const fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<int, 8> ret;
    ret.vec_ = _mm256_i32gather_epi32(base_addr, vindex.vec_, sizeof(int));
    return ret;
}

force_inline fixed_size_simd<int, 8> vectorcall gather(fixed_size_simd<int, 8> src, const int *base_addr,
                                                       fixed_size_simd<int, 8> mask, fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<int, 8> ret;
    ret.vec_ = _mm256_mask_i32gather_epi32(src.vec_, base_addr, vindex.vec_, mask.vec_, sizeof(int));
    return ret;
}

force_inline fixed_size_simd<unsigned, 8> vectorcall gather(const unsigned *base_addr,
                                                            const fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<unsigned, 8> ret;
    ret.vec_ = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base_addr), vindex.vec_, sizeof(int));
    return ret;
}

force_inline fixed_size_simd<unsigned, 8> vectorcall gather(fixed_size_simd<unsigned, 8> src, const unsigned *base_addr,
                                                            fixed_size_simd<unsigned, 8> mask,
                                                            fixed_size_simd<int, 8> vindex) {
    fixed_size_simd<unsigned, 8> ret;
    ret.vec_ = _mm256_mask_i32gather_epi32(src.vec_, reinterpret_cast<const int *>(base_addr), vindex.vec_, mask.vec_,
                                           sizeof(unsigned));
    return ret;
}
#endif

template <typename U>
force_inline fixed_size_simd<float, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                         const fixed_size_simd<float, 8> vec1,
                                                         const fixed_size_simd<float, 8> vec2) {
    validate_mask(mask);
    return _mm256_blendv_ps(vec2.vec_, vec1.vec_, _mm_cast<__m256>(mask.vec_));
}

template <typename U>
force_inline fixed_size_simd<int, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                       const fixed_size_simd<int, 8> vec1,
                                                       const fixed_size_simd<int, 8> vec2) {
    validate_mask(mask);
    return _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(vec2.vec_), _mm256_castsi256_ps(vec1.vec_), _mm_cast<__m256>(mask.vec_)));
}

template <typename U>
force_inline fixed_size_simd<unsigned, 8> vectorcall select(const fixed_size_simd<U, 8> mask,
                                                            const fixed_size_simd<unsigned, 8> vec1,
                                                            const fixed_size_simd<unsigned, 8> vec2) {
    validate_mask(mask);
    return _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(vec2.vec_), _mm256_castsi256_ps(vec1.vec_), _mm_cast<__m256>(mask.vec_)));
}

} // namespace NS
} // namespace Ray

#pragma warning(pop)

#undef avx2_inline

#undef validate_mask
