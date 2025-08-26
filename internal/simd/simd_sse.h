// #pragma once

#include <type_traits>

#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

namespace Ray {
namespace NS {

template <typename To, typename From> To _mm_cast(const From x) { return x; }
template <> force_inline __m128 _mm_cast(const __m128i x) { return _mm_castsi128_ps(x); }
template <> force_inline __m128i _mm_cast(const __m128 x) { return _mm_castps_si128(x); }

template <> class fixed_size_simd<int, 4>;
template <> class fixed_size_simd<unsigned, 4>;

template <> class fixed_size_simd<float, 4> {
    union {
        __m128 vec_;
        float comp_[4];
    };

    friend class fixed_size_simd<int, 4>;
    friend class fixed_size_simd<unsigned, 4>;

    force_inline fixed_size_simd(const __m128 vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) { vec_ = _mm_set1_ps(f); }
    force_inline fixed_size_simd(const float f1, const float f2, const float f3, const float f4) {
        vec_ = _mm_setr_ps(f1, f2, f3, f4);
    }
    force_inline explicit fixed_size_simd(const float *f) { vec_ = _mm_loadu_ps(f); }
    force_inline fixed_size_simd(const float *f, vector_aligned_tag) { vec_ = _mm_load_ps(f); }

    force_inline float operator[](const int i) const { return comp_[i]; }

    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float get() const { return comp_[i]; }
    template <int i> force_inline void set(const float v) { comp_[i] = v; }
    force_inline void set(const int i, const float v) { comp_[i] = v; }

    force_inline fixed_size_simd<float, 4> &vectorcall operator+=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator-=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator*=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator/=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator|=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator-() const {
        const __m128 m = _mm_set1_ps(-0.0f);
        return _mm_xor_ps(vec_, m);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator<(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmplt_ps(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator<=(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmple_ps(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator>(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmpgt_ps(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator>=(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmpge_ps(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator~() const {
        return _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(vec_), _mm_set1_epi32(~0)));
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator&=(const fixed_size_simd<float, 4> rhs) {
        vec_ = _mm_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit vectorcall operator fixed_size_simd<int, 4>() const;
    force_inline explicit vectorcall operator fixed_size_simd<unsigned, 4>() const;

    force_inline fixed_size_simd<float, 4> vectorcall sqrt() const { return _mm_sqrt_ps(vec_); }

    fixed_size_simd<float, 4> vectorcall log() const {
        fixed_size_simd<float, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = logf(comp_[i]); })
        return ret;
    }

    float vectorcall length() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(_mm_sqrt_ss(r1));
    }

    float vectorcall length2() const {
        __m128 r1, r2;
        r1 = _mm_mul_ps(vec_, vec_);

        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);

        return _mm_cvtss_f32(r1);
    }

    force_inline float hsum() const {
#if defined(USE_SSE41)
        __m128 temp = _mm_hadd_ps(vec_, vec_);
        temp = _mm_hadd_ps(temp, temp);
        return _mm_cvtss_f32(temp);
#else
        return comp_[0] + comp_[1] + comp_[2] + comp_[3];
#endif
    }

    force_inline void vectorcall store_to(float *f) const { _mm_storeu_ps(f, vec_); }
    force_inline void vectorcall store_to(float *f, vector_aligned_tag) const { _mm_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<float, 4> mask, const fixed_size_simd<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(vec_, v1.vec_, mask.vec_);
#else
        __m128 temp1 = _mm_and_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_andnot_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float, 4> mask,
                                              const fixed_size_simd<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(v1.vec_, vec_, mask.vec_);
#else
        __m128 temp1 = _mm_andnot_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_and_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall min(const fixed_size_simd<float, 4> v1,
                                                                 const fixed_size_simd<float, 4> v2) {
        return _mm_min_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall max(const fixed_size_simd<float, 4> v1,
                                                                 const fixed_size_simd<float, 4> v2) {
        return _mm_max_ps(v1.vec_, v2.vec_);
    }

    force_inline static fixed_size_simd<float, 4> vectorcall and_not(const fixed_size_simd<float, 4> v1,
                                                                     const fixed_size_simd<float, 4> v2) {
        return _mm_andnot_ps(v1.vec_, v2.vec_);
    }

    force_inline static fixed_size_simd<float, 4> vectorcall floor(const fixed_size_simd<float, 4> v1) {
#if defined(USE_SSE41)
        return _mm_floor_ps(v1.vec_);
#else
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        return _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
#endif
    }

    force_inline static fixed_size_simd<float, 4> vectorcall ceil(const fixed_size_simd<float, 4> v1) {
#if defined(USE_SSE41)
        return _mm_ceil_ps(v1.vec_);
#else
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        return _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
#endif
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator&(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_and_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator|(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_or_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator^(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_xor_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator+(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_add_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator-(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_sub_ps(v1.vec_, v2.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator==(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmpeq_ps(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator!=(const fixed_size_simd<float, 4> rhs) const {
        return _mm_cmpneq_ps(vec_, rhs.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator*(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_mul_ps(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator/(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return _mm_div_ps(v1.vec_, v2.vec_);
    }

    friend force_inline float vectorcall dot(const fixed_size_simd<float, 4> v1, const fixed_size_simd<float, 4> v2) {
#if defined(USE_SSE41)
        return _mm_cvtss_f32(_mm_dp_ps(v1.vec_, v2.vec_, 0xff));
#else
        __m128 r1, r2;
        r1 = _mm_mul_ps(v1.vec_, v2.vec_);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);
        return _mm_cvtss_f32(r1);
#endif
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall clamp(const fixed_size_simd<float, 4> v1,
                                                                   const fixed_size_simd<float, 4> min,
                                                                   const fixed_size_simd<float, 4> max) {
        return _mm_max_ps(min.vec_, _mm_min_ps(v1.vec_, max.vec_));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall saturate(const fixed_size_simd<float, 4> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    friend fixed_size_simd<float, 4> vectorcall pow(const fixed_size_simd<float, 4> v1,
                                                    const fixed_size_simd<float, 4> v2) {
        alignas(16) float comp1[4], comp2[4];
        _mm_store_ps(comp1, v1.vec_);
        _mm_store_ps(comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = powf(comp1[i], comp2[i]); })
        return fixed_size_simd<float, 4>{comp1, vector_aligned};
    }

    friend fixed_size_simd<float, 4> vectorcall exp(const fixed_size_simd<float, 4> v1) {
        alignas(16) float comp1[4];
        _mm_store_ps(comp1, v1.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = expf(comp1[i]); })
        return fixed_size_simd<float, 4>{comp1, vector_aligned};
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall normalize(const fixed_size_simd<float, 4> v1) {
        return v1 / v1.length();
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall normalize_len(const fixed_size_simd<float, 4> v1,
                                                                           float &out_len) {
        return v1 / (out_len = v1.length());
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall inclusive_scan(fixed_size_simd<float, 4> v1) {
        v1.vec_ = _mm_add_ps(v1.vec_, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v1.vec_), 4)));
        v1.vec_ = _mm_add_ps(v1.vec_, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v1.vec_), 8)));
        return v1;
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall copysign(const fixed_size_simd<float, 4> val,
                                                                      const fixed_size_simd<float, 4> sign) {
        const __m128 sign_bit_mask = _mm_set1_ps(-0.0f);
        const __m128 val_abs = _mm_andnot_ps(sign_bit_mask, val.vec_);   // abs(val)
        const __m128 sign_bits = _mm_and_ps(sign_bit_mask, sign.vec_); // sign of 'sign'
        return _mm_or_ps(val_abs, sign_bits);
    }

    template <typename U>
    friend force_inline fixed_size_simd<float, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                    const fixed_size_simd<float, 4> vec1,
                                                                    const fixed_size_simd<float, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                  const fixed_size_simd<int, 4> vec1,
                                                                  const fixed_size_simd<int, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                       const fixed_size_simd<unsigned, 4> vec1,
                                                                       const fixed_size_simd<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<float, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *vectorcall value_ptr(const fixed_size_simd<float, 4> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *vectorcall value_ptr(fixed_size_simd<float, 4> &v1) {
        return reinterpret_cast<float *>(&v1.vec_);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<int, 4> {
    union {
        __m128i vec_;
        int comp_[4];
    };

    friend class fixed_size_simd<float, 4>;
    friend class fixed_size_simd<unsigned, 4>;

    force_inline fixed_size_simd(const __m128i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const int v) { vec_ = _mm_set1_epi32(v); }
    force_inline fixed_size_simd(const int i1, const int i2, const int i3, const int i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline explicit fixed_size_simd(const int *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline fixed_size_simd(const int *f, vector_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

    force_inline int operator[](const int i) const { return comp_[i]; }
    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const { return comp_[i]; }
    template <int i> force_inline void set(const int v) { comp_[i & 3] = v; }
    force_inline void set(const int i, const int v) { comp_[i] = v; }

    force_inline fixed_size_simd<int, 4> &vectorcall operator+=(const fixed_size_simd<int, 4> rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator-=(const fixed_size_simd<int, 4> rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    fixed_size_simd<int, 4> &vectorcall operator*=(const fixed_size_simd<int, 4> rhs) {
#if defined(USE_SSE41)
        vec_ = _mm_mullo_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 4, { comp_[i] *= rhs.comp_[i]; })
#endif
        return *this;
    }

    fixed_size_simd<int, 4> &vectorcall operator/=(const fixed_size_simd<int, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator|=(const fixed_size_simd<int, 4> rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator^=(const fixed_size_simd<int, 4> rhs) {
        vec_ = _mm_xor_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator-() const {
        fixed_size_simd<int, 4> temp;
        temp.vec_ = _mm_sub_epi32(_mm_setzero_si128(), vec_);
        return temp;
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator==(const fixed_size_simd<int, 4> rhs) const {
        return _mm_cmpeq_epi32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator!=(const fixed_size_simd<int, 4> rhs) const {
        return _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator<(const fixed_size_simd<int, 4> rhs) const {
        return _mm_cmplt_epi32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator<=(const fixed_size_simd<int, 4> rhs) const {
        return _mm_andnot_si128(_mm_cmpgt_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator>(const fixed_size_simd<int, 4> rhs) const {
        return _mm_cmpgt_epi32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator>=(const fixed_size_simd<int, 4> rhs) const {
        return _mm_andnot_si128(_mm_cmplt_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator&=(const fixed_size_simd<int, 4> rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator~() const {
        return _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
    }

    force_inline explicit vectorcall operator fixed_size_simd<float, 4>() const { return _mm_cvtepi32_ps(vec_); }

    force_inline explicit vectorcall operator fixed_size_simd<unsigned, 4>() const;

    force_inline int hsum() const {
#if defined(USE_SSE41)
        __m128i temp = _mm_hadd_epi32(vec_, vec_);
        temp = _mm_hadd_epi32(temp, temp);
        return _mm_cvtsi128_si32(temp);
#else
        return comp_[0] + comp_[1] + comp_[2] + comp_[3];
#endif
    }

    force_inline void store_to(int *f) const { _mm_storeu_si128((__m128i *)f, vec_); }
    force_inline void store_to(int *f, vector_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<int, 4> mask, const fixed_size_simd<int, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<int, 4> mask, const fixed_size_simd<int, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(v1.vec_, vec_, mask.vec_);
#else
        __m128i temp1 = _mm_andnot_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_and_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline int movemask() const { return _mm_movemask_ps(_mm_castsi128_ps(vec_)); }

    force_inline bool all_zeros() const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(vec_, _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<int, 4> mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend fixed_size_simd<int, 4> vectorcall min(const fixed_size_simd<int, 4> v1, const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] < v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    static fixed_size_simd<int, 4> vectorcall max(const fixed_size_simd<int, 4> v1, const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_max_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] > v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall clamp(const fixed_size_simd<int, 4> v1,
                                                                 const fixed_size_simd<int, 4> _min,
                                                                 const fixed_size_simd<int, 4> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<int, 4> vectorcall and_not(const fixed_size_simd<int, 4> v1,
                                                                   const fixed_size_simd<int, 4> v2) {
        return _mm_andnot_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator&(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return _mm_and_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator|(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return _mm_or_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator^(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return _mm_xor_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator+(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return _mm_add_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator-(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return _mm_sub_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<int, 4> vectorcall operator*(const fixed_size_simd<int, 4> v1,
                                                        const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mullo_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
#endif
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator/(const fixed_size_simd<int, 4> v1,
                                                        const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator>>(const fixed_size_simd<int, 4> v1,
                                                         const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = int(unsigned(v1.comp_[i]) >> unsigned(v2.comp_[i])); })
        return ret;
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator>>(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator<<(const fixed_size_simd<int, 4> v1,
                                                         const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator<<(const fixed_size_simd<int, 4> v1, const int v2) {
        return _mm_slli_epi32(v1.vec_, v2);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall srai(const fixed_size_simd<int, 4> v1, const int v2) {
        return _mm_srai_epi32(v1.vec_, v2);
    }

    friend force_inline bool vectorcall is_equal(const fixed_size_simd<int, 4> v1, const fixed_size_simd<int, 4> v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall inclusive_scan(fixed_size_simd<int, 4> v1) {
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 4));
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 8));
        return v1;
    }

    template <typename U>
    friend force_inline fixed_size_simd<float, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                    const fixed_size_simd<float, 4> vec1,
                                                                    const fixed_size_simd<float, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                  const fixed_size_simd<int, 4> vec1,
                                                                  const fixed_size_simd<int, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                       const fixed_size_simd<unsigned, 4> vec1,
                                                                       const fixed_size_simd<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<int, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const fixed_size_simd<int, 4> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(fixed_size_simd<int, 4> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<unsigned, 4> {
    union {
        __m128i vec_;
        unsigned comp_[4];
    };

    friend class fixed_size_simd<float, 4>;
    friend class fixed_size_simd<int, 4>;

    force_inline fixed_size_simd(const __m128i vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const unsigned v) { vec_ = _mm_set1_epi32(v); }
    force_inline fixed_size_simd(const unsigned i1, const unsigned i2, const unsigned i3, const unsigned i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline explicit fixed_size_simd(const unsigned *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline fixed_size_simd(const unsigned *f, vector_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

    force_inline unsigned operator[](const int i) const { return comp_[i]; }
    force_inline unsigned operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline unsigned get() const {
#if defined(USE_SSE41)
        return _mm_extract_epi32(vec_, i & 3);
#else
        return comp_[i];
#endif
    }
    template <int i> force_inline void set(const unsigned v) {
#if defined(USE_SSE41)
        vec_ = _mm_insert_epi32(vec_, v, i & 3);
#else
        comp_[i] = v;
#endif
    }
    force_inline void set(const int i, const unsigned v) { comp_[i] = v; }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator+=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator-=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    fixed_size_simd<unsigned, 4> &vectorcall operator*=(const fixed_size_simd<unsigned, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<unsigned, 4> &vectorcall operator/=(const fixed_size_simd<unsigned, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator|=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator^=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = _mm_xor_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator==(const fixed_size_simd<unsigned, 4> rhs) const {
        return _mm_cmpeq_epi32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator!=(const fixed_size_simd<unsigned, 4> rhs) const {
        return _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator&=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator~() const {
        return _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
    }

    force_inline explicit vectorcall operator fixed_size_simd<float, 4>() const {
        __m128i v_hi = _mm_srli_epi32(vec_, 16);
        __m128i v_lo = _mm_and_si128(vec_, _mm_set1_epi32(0xffff));
        __m128 v_hi_f = _mm_cvtepi32_ps(v_hi);
        __m128 v_lo_f = _mm_cvtepi32_ps(v_lo);
        return _mm_add_ps(_mm_mul_ps(v_hi_f, _mm_set1_ps(65536.0f)), v_lo_f);
    }

    force_inline explicit vectorcall operator fixed_size_simd<int, 4>() const {
        fixed_size_simd<int, 4> ret;
        ret.vec_ = vec_;
        return ret;
    }

    force_inline unsigned hsum() const {
#if defined(USE_SSE41)
        __m128i temp = _mm_hadd_epi32(vec_, vec_);
        temp = _mm_hadd_epi32(temp, temp);
        return _mm_cvtsi128_si32(temp);
#else
        return comp_[0] + comp_[1] + comp_[2] + comp_[3];
#endif
    }

    force_inline void store_to(unsigned *f) const { _mm_storeu_si128((__m128i *)f, vec_); }
    force_inline void store_to(unsigned *f, vector_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void vectorcall blend_to(const fixed_size_simd<unsigned, 4> mask,
                                          const fixed_size_simd<unsigned, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<unsigned, 4> mask,
                                              const fixed_size_simd<unsigned, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(v1.vec_, vec_, mask.vec_);
#else
        __m128i temp1 = _mm_andnot_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_and_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline int movemask() const { return _mm_movemask_ps(_mm_castsi128_ps(vec_)); }

    force_inline bool all_zeros() const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(vec_, _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<unsigned, 4> mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    static fixed_size_simd<unsigned, 4> vectorcall min(const fixed_size_simd<unsigned, 4> v1,
                                                       const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] < v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    static fixed_size_simd<unsigned, 4> vectorcall max(const fixed_size_simd<unsigned, 4> v1,
                                                       const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_max_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] > v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    force_inline static fixed_size_simd<unsigned, 4> vectorcall and_not(const fixed_size_simd<unsigned, 4> v1,
                                                                        const fixed_size_simd<unsigned, 4> v2) {
        return _mm_andnot_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator&(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return _mm_and_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator|(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return _mm_or_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator^(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return _mm_xor_si128(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator+(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return _mm_add_epi32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator-(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return _mm_sub_epi32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator*(const fixed_size_simd<unsigned, 4> v1,
                                                             const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator/(const fixed_size_simd<unsigned, 4> v1,
                                                             const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator>>(const fixed_size_simd<unsigned, 4> v1,
                                                              const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator>>(const fixed_size_simd<unsigned, 4> v1,
                                                                           const unsigned v2) {
        fixed_size_simd<unsigned, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator<<(const fixed_size_simd<unsigned, 4> v1,
                                                              const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator<<(const fixed_size_simd<unsigned, 4> v1,
                                                                           const unsigned v2) {
        return _mm_slli_epi32(v1.vec_, v2);
    }

    friend force_inline bool vectorcall is_equal(const fixed_size_simd<unsigned, 4> v1,
                                                 const fixed_size_simd<unsigned, 4> v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall inclusive_scan(fixed_size_simd<unsigned, 4> v1) {
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 4));
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 8));
        return v1;
    }

    template <typename U>
    friend force_inline fixed_size_simd<float, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                    const fixed_size_simd<float, 4> vec1,
                                                                    const fixed_size_simd<float, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<int, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                  const fixed_size_simd<int, 4> vec1,
                                                                  const fixed_size_simd<int, 4> vec2);
    template <typename U>
    friend force_inline fixed_size_simd<unsigned, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                                       const fixed_size_simd<unsigned, 4> vec1,
                                                                       const fixed_size_simd<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const fixed_size_simd<unsigned, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const unsigned val = mask.get<i>();
            assert(val == 0 || val == 0xffffffff);
        })
    }
#endif

    friend force_inline const unsigned *value_ptr(const fixed_size_simd<unsigned, 4> &v1) {
        return reinterpret_cast<const unsigned *>(&v1.vec_);
    }
    friend force_inline unsigned *value_ptr(fixed_size_simd<unsigned, 4> &v1) {
        return reinterpret_cast<unsigned *>(&v1.vec_);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline vectorcall fixed_size_simd<float, 4>::operator fixed_size_simd<int, 4>() const {
    return _mm_cvttps_epi32(vec_);
}

force_inline vectorcall fixed_size_simd<float, 4>::operator fixed_size_simd<unsigned, 4>() const {
    return _mm_cvttps_epi32(vec_);
}

force_inline vectorcall fixed_size_simd<int, 4>::operator fixed_size_simd<unsigned, 4>() const {
    fixed_size_simd<unsigned, 4> ret;
    ret.vec_ = vec_;
    return ret;
}

template <typename U>
force_inline fixed_size_simd<float, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                         const fixed_size_simd<float, 4> vec1,
                                                         const fixed_size_simd<float, 4> vec2) {
    validate_mask(mask);
    fixed_size_simd<float, 4> ret;
#if defined(USE_SSE41)
    ret.vec_ = _mm_blendv_ps(vec2.vec_, vec1.vec_, _mm_cast<__m128>(mask.vec_));
#else
    const __m128 temp1 = _mm_and_ps(_mm_cast<__m128>(mask.vec_), vec1.vec_);
    const __m128 temp2 = _mm_andnot_ps(_mm_cast<__m128>(mask.vec_), vec2.vec_);
    ret.vec_ = _mm_or_ps(temp1, temp2);
#endif
    return ret;
}

template <typename U>
force_inline fixed_size_simd<int, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                       const fixed_size_simd<int, 4> vec1,
                                                       const fixed_size_simd<int, 4> vec2) {
    validate_mask(mask);
    fixed_size_simd<int, 4> ret;
#if defined(USE_SSE41)
    ret.vec_ = _mm_blendv_epi8(vec2.vec_, vec1.vec_, _mm_cast<__m128i>(mask.vec_));
#else
    const __m128i temp1 = _mm_and_si128(_mm_cast<__m128i>(mask.vec_), vec1.vec_);
    const __m128i temp2 = _mm_andnot_si128(_mm_cast<__m128i>(mask.vec_), vec2.vec_);
    ret.vec_ = _mm_or_si128(temp1, temp2);
#endif
    return ret;
}

template <typename U>
force_inline fixed_size_simd<unsigned, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                            const fixed_size_simd<unsigned, 4> vec1,
                                                            const fixed_size_simd<unsigned, 4> vec2) {
    validate_mask(mask);
    fixed_size_simd<unsigned, 4> ret;
#if defined(USE_SSE41)
    ret.vec_ = _mm_blendv_epi8(vec2.vec_, vec1.vec_, _mm_cast<__m128i>(mask.vec_));
#else
    const __m128i temp1 = _mm_and_si128(_mm_cast<__m128i>(mask.vec_), vec1.vec_);
    const __m128i temp2 = _mm_andnot_si128(_mm_cast<__m128i>(mask.vec_), vec2.vec_);
    ret.vec_ = _mm_or_si128(temp1, temp2);
#endif
    return ret;
}

} // namespace NS
} // namespace Ray

#undef validate_mask
