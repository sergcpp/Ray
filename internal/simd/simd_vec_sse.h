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

template <typename To, typename From> To _mm_cast(From x) { return x; }
template <> force_inline __m128 _mm_cast(__m128i x) { return _mm_castsi128_ps(x); }
template <> force_inline __m128i _mm_cast(__m128 x) { return _mm_castps_si128(x); }

template <> class simd_vec<int, 4>;
template <> class simd_vec<unsigned, 4>;

template <> class simd_vec<float, 4> {
    union {
        __m128 vec_;
        float comp_[4];
    };

    friend class simd_vec<int, 4>;
    friend class simd_vec<unsigned, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = _mm_set1_ps(f); }
    template <typename... Tail> force_inline simd_vec(const float f1, const float f2, const float f3, const float f4) {
        vec_ = _mm_setr_ps(f1, f2, f3, f4);
    }
    force_inline explicit simd_vec(const float *f) { vec_ = _mm_loadu_ps(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) { vec_ = _mm_load_ps(f); }

    force_inline float operator[](const int i) const { return comp_[i]; }

    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float get() const { return comp_[i]; }
    template <int i> force_inline void set(const float v) { comp_[i] = v; }
    force_inline void set(const int i, const float v) { comp_[i] = v; }

    force_inline simd_vec<float, 4> &vectorcall operator+=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator-=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator*=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator/=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &vectorcall operator|=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_or_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> vectorcall operator-() const {
        simd_vec<float, 4> temp;
        __m128 m = _mm_set1_ps(-0.0f);
        temp.vec_ = _mm_xor_ps(vec_, m);
        return temp;
    }

    force_inline simd_vec<float, 4> vectorcall operator<(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator<=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmple_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator>=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator~() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(vec_), _mm_set1_epi32(~0)));
        return ret;
    }

    force_inline simd_vec<float, 4> &vectorcall operator&=(const simd_vec<float, 4> rhs) {
        vec_ = _mm_and_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline explicit vectorcall operator simd_vec<int, 4>() const;
    force_inline explicit vectorcall operator simd_vec<unsigned, 4>() const;

    force_inline simd_vec<float, 4> vectorcall sqrt() const {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_sqrt_ps(vec_);
        return temp;
    }

    simd_vec<float, 4> vectorcall log() const {
        simd_vec<float, 4> ret;
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
    force_inline void vectorcall store_to(float *f, simd_mem_aligned_tag) const { _mm_store_ps(f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<float, 4> mask, const simd_vec<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(vec_, v1.vec_, mask.vec_);
#else
        __m128 temp1 = _mm_and_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_andnot_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<float, 4> mask, const simd_vec<float, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_ps(v1.vec_, vec_, mask.vec_);
#else
        __m128 temp1 = _mm_andnot_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_and_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    friend force_inline simd_vec<float, 4> vectorcall min(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_min_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall max(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_max_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall and_not(const simd_vec<float, 4> v1,
                                                              const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_andnot_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall floor(const simd_vec<float, 4> v1) {
        simd_vec<float, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_floor_ps(v1.vec_);
#else
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        temp.vec_ = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
#endif
        return temp;
    }

    force_inline static simd_vec<float, 4> vectorcall ceil(const simd_vec<float, 4> v1) {
        simd_vec<float, 4> temp;
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator&(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_and_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator|(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_or_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator^(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = _mm_xor_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator+(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator-(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, v2.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator==(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpeq_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> vectorcall operator!=(const simd_vec<float, 4> rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cmpneq_ps(vec_, rhs.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator*(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall operator/(const simd_vec<float, 4> v1,
                                                                const simd_vec<float, 4> v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline float vectorcall dot(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        __m128 r1, r2;
        r1 = _mm_mul_ps(v1.vec_, v2.vec_);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);
        return _mm_cvtss_f32(r1);
    }

    friend force_inline simd_vec<float, 4> vectorcall clamp(const simd_vec<float, 4> v1, const simd_vec<float, 4> min,
                                                            const simd_vec<float, 4> max) {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_max_ps(min.vec_, _mm_min_ps(v1.vec_, max.vec_));
        return ret;
    }

    friend force_inline simd_vec<float, 4> vectorcall saturate(const simd_vec<float, 4> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    friend simd_vec<float, 4> vectorcall pow(const simd_vec<float, 4> v1, const simd_vec<float, 4> v2) {
        alignas(16) float comp1[4], comp2[4];
        _mm_store_ps(comp1, v1.vec_);
        _mm_store_ps(comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = powf(comp1[i], comp2[i]); })
        return simd_vec<float, 4>{comp1, simd_mem_aligned};
    }

    friend force_inline simd_vec<float, 4> vectorcall normalize(const simd_vec<float, 4> v1) {
        return v1 / v1.length();
    }

    friend force_inline simd_vec<float, 4> vectorcall normalize_len(const simd_vec<float, 4> v1, float &out_len) {
        return v1 / (out_len = v1.length());
    }

    friend force_inline simd_vec<float, 4> vectorcall inclusive_scan(simd_vec<float, 4> v1) {
        v1.vec_ = _mm_add_ps(v1.vec_, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v1.vec_), 4)));
        v1.vec_ = _mm_add_ps(v1.vec_, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v1.vec_), 8)));
        return v1;
    }

    template <typename U>
    friend force_inline simd_vec<float, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<float, 4> vec1,
                                                             const simd_vec<float, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<int, 4> vec1,
                                                           const simd_vec<int, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 4> vectorcall select(const simd_vec<U, 4> mask,
                                                                const simd_vec<unsigned, 4> vec1,
                                                                const simd_vec<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<float, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const float val = mask.get<i>();
            assert(reinterpret_cast<const uint32_t &>(val) == 0 ||
                   reinterpret_cast<const uint32_t &>(val) == 0xffffffff);
        })
    }
#endif

    friend force_inline const float *vectorcall value_ptr(const simd_vec<float, 4> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *vectorcall value_ptr(simd_vec<float, 4> &v1) {
        return reinterpret_cast<float *>(&v1.vec_);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 4> {
    union {
        __m128i vec_;
        int comp_[4];
    };

    friend class simd_vec<float, 4>;
    friend class simd_vec<unsigned, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const int v) { vec_ = _mm_set1_epi32(v); }
    force_inline simd_vec(const int i1, const int i2, const int i3, const int i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline explicit simd_vec(const int *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

    force_inline int operator[](const int i) const { return comp_[i]; }
    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const { return comp_[i]; }
    template <int i> force_inline void set(const int v) { comp_[i & 3] = v; }
    force_inline void set(const int i, const int v) { comp_[i] = v; }

    force_inline simd_vec<int, 4> &vectorcall operator+=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator-=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    simd_vec<int, 4> &vectorcall operator*=(const simd_vec<int, 4> rhs) {
#if defined(USE_SSE41)
        vec_ = _mm_mullo_epi32(vec_, rhs.vec_);
#else
        UNROLLED_FOR(i, 4, { comp_[i] *= rhs.comp_[i]; })
#endif
        return *this;
    }

    simd_vec<int, 4> &vectorcall operator/=(const simd_vec<int, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator|=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &vectorcall operator^=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_xor_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> vectorcall operator-() const {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_sub_epi32(_mm_setzero_si128(), vec_);
        return temp;
    }

    force_inline simd_vec<int, 4> vectorcall operator==(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator!=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator<(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator<=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> vectorcall operator>=(const simd_vec<int, 4> rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, 4> &vectorcall operator&=(const simd_vec<int, 4> rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> vectorcall operator~() const {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<unsigned, 4>() const;

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
    force_inline void store_to(int *f, simd_mem_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<int, 4> mask, const simd_vec<int, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<int, 4> mask, const simd_vec<int, 4> v1) {
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

    force_inline bool vectorcall all_zeros(const simd_vec<int, 4> mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend simd_vec<int, 4> vectorcall min(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] < v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    static simd_vec<int, 4> vectorcall max(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_max_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] > v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall clamp(const simd_vec<int, 4> v1, const simd_vec<int, 4> _min,
                                                          const simd_vec<int, 4> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static simd_vec<int, 4> vectorcall and_not(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_andnot_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator&(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_and_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator|(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_or_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator^(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = _mm_xor_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator+(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator-(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend simd_vec<int, 4> vectorcall operator*(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
#if defined(USE_SSE41)
        ret.vec_ = _mm_mullo_epi32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
#endif
        return ret;
    }

    friend simd_vec<int, 4> vectorcall operator/(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend simd_vec<int, 4> vectorcall operator>>(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = int(unsigned(v1.comp_[i]) >> unsigned(v2.comp_[i])); })
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator>>(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend simd_vec<int, 4> vectorcall operator<<(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        simd_vec<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall operator<<(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_slli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, 4> vectorcall srai(const simd_vec<int, 4> v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = _mm_srai_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<int, 4> v1, const simd_vec<int, 4> v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

    friend force_inline simd_vec<int, 4> vectorcall inclusive_scan(simd_vec<int, 4> v1) {
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 4));
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 8));
        return v1;
    }

    template <typename U>
    friend force_inline simd_vec<float, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<float, 4> vec1,
                                                             const simd_vec<float, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<int, 4> vec1,
                                                           const simd_vec<int, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 4> vectorcall select(const simd_vec<U, 4> mask,
                                                                const simd_vec<unsigned, 4> vec1,
                                                                const simd_vec<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<int, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const int val = mask.get<i>();
            assert(val == 0 || val == -1);
        })
    }
#endif

    friend force_inline const int *value_ptr(const simd_vec<int, 4> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(simd_vec<int, 4> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class simd_vec<unsigned, 4> {
    union {
        __m128i vec_;
        unsigned comp_[4];
    };

    friend class simd_vec<float, 4>;
    friend class simd_vec<int, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const unsigned v) { vec_ = _mm_set1_epi32(v); }
    force_inline simd_vec(const unsigned i1, const unsigned i2, const unsigned i3, const unsigned i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline explicit simd_vec(const unsigned *f) { vec_ = _mm_loadu_si128((const __m128i *)f); }
    force_inline simd_vec(const unsigned *f, simd_mem_aligned_tag) { vec_ = _mm_load_si128((const __m128i *)f); }

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
        const int ndx = (i & 3);
#if defined(USE_SSE41)
        vec_ = _mm_insert_epi32(vec_, v, ndx);
#else
        comp_[i] = v;
#endif
    }
    force_inline void set(const int i, const unsigned v) { comp_[i] = v; }

    force_inline simd_vec<unsigned, 4> &vectorcall operator+=(const simd_vec<unsigned, 4> rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<unsigned, 4> &vectorcall operator-=(const simd_vec<unsigned, 4> rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    simd_vec<unsigned, 4> &vectorcall operator*=(const simd_vec<unsigned, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] *= rhs.comp_[i]; })
        return *this;
    }

    simd_vec<unsigned, 4> &vectorcall operator/=(const simd_vec<unsigned, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<unsigned, 4> &vectorcall operator|=(const simd_vec<unsigned, 4> rhs) {
        vec_ = _mm_or_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<unsigned, 4> &vectorcall operator^=(const simd_vec<unsigned, 4> rhs) {
        vec_ = _mm_xor_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<unsigned, 4> vectorcall operator==(const simd_vec<unsigned, 4> rhs) const {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<unsigned, 4> vectorcall operator!=(const simd_vec<unsigned, 4> rhs) const {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<unsigned, 4> &vectorcall operator&=(const simd_vec<unsigned, 4> rhs) {
        vec_ = _mm_and_si128(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<unsigned, 4> vectorcall operator~() const {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_andnot_si128(vec_, _mm_set1_epi32(~0));
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = _mm_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline explicit vectorcall operator simd_vec<int, 4>() const {
        simd_vec<int, 4> ret;
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
    force_inline void store_to(unsigned *f, simd_mem_aligned_tag) const { _mm_store_si128((__m128i *)f, vec_); }

    force_inline void vectorcall blend_to(const simd_vec<unsigned, 4> mask, const simd_vec<unsigned, 4> v1) {
        validate_mask(mask);
#if defined(USE_SSE41)
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline void vectorcall blend_inv_to(const simd_vec<unsigned, 4> mask, const simd_vec<unsigned, 4> v1) {
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

    force_inline bool vectorcall all_zeros(const simd_vec<unsigned, 4> mask) const {
#if defined(USE_SSE41)
        return _mm_test_all_zeros(vec_, mask.vec_);
#else
        return _mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) == 0xFFFF;
#endif
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    static simd_vec<unsigned, 4> vectorcall min(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_min_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] < v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    static simd_vec<unsigned, 4> vectorcall max(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
#if defined(USE_SSE41)
        temp.vec_ = _mm_max_epu32(v1.vec_, v2.vec_);
#else
        UNROLLED_FOR(i, 4, { temp.comp_[i] = (v1.comp_[i] > v2.comp_[i]) ? v1.comp_[i] : v2.comp_[i]; })
#endif
        return temp;
    }

    force_inline static simd_vec<unsigned, 4> vectorcall and_not(const simd_vec<unsigned, 4> v1,
                                                                 const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
        temp.vec_ = _mm_andnot_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator&(const simd_vec<unsigned, 4> v1,
                                                                   const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
        temp.vec_ = _mm_and_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator|(const simd_vec<unsigned, 4> v1,
                                                                   const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
        temp.vec_ = _mm_or_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator^(const simd_vec<unsigned, 4> v1,
                                                                   const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> temp;
        temp.vec_ = _mm_xor_si128(v1.vec_, v2.vec_);
        ;
        return temp;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator+(const simd_vec<unsigned, 4> v1,
                                                                   const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator-(const simd_vec<unsigned, 4> v1,
                                                                   const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend simd_vec<unsigned, 4> vectorcall operator*(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend simd_vec<unsigned, 4> vectorcall operator/(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend simd_vec<unsigned, 4> vectorcall operator>>(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator>>(const simd_vec<unsigned, 4> v1, const unsigned v2) {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend simd_vec<unsigned, 4> vectorcall operator<<(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        simd_vec<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall operator<<(const simd_vec<unsigned, 4> v1, const unsigned v2) {
        simd_vec<unsigned, 4> ret;
        ret.vec_ = _mm_slli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline bool vectorcall is_equal(const simd_vec<unsigned, 4> v1, const simd_vec<unsigned, 4> v2) {
        __m128i vcmp = _mm_cmpeq_epi32(v1.vec_, v2.vec_);
        return (_mm_movemask_epi8(vcmp) == 0xffff);
    }

    friend force_inline simd_vec<unsigned, 4> vectorcall inclusive_scan(simd_vec<unsigned, 4> v1) {
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 4));
        v1.vec_ = _mm_add_epi32(v1.vec_, _mm_slli_si128(v1.vec_, 8));
        return v1;
    }

    template <typename U>
    friend force_inline simd_vec<float, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<float, 4> vec1,
                                                             const simd_vec<float, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<int, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<int, 4> vec1,
                                                           const simd_vec<int, 4> vec2);
    template <typename U>
    friend force_inline simd_vec<unsigned, 4> vectorcall select(const simd_vec<U, 4> mask,
                                                                const simd_vec<unsigned, 4> vec1,
                                                                const simd_vec<unsigned, 4> vec2);

#ifndef NDEBUG
    friend void vectorcall __assert_valid_mask(const simd_vec<unsigned, 4> mask) {
        UNROLLED_FOR(i, 4, {
            const unsigned val = mask.get<i>();
            assert(val == 0 || val == 0xffffffff);
        })
    }
#endif

    friend force_inline const unsigned *value_ptr(const simd_vec<unsigned, 4> &v1) {
        return reinterpret_cast<const unsigned *>(&v1.vec_);
    }
    friend force_inline unsigned *value_ptr(simd_vec<unsigned, 4> &v1) {
        return reinterpret_cast<unsigned *>(&v1.vec_);
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline vectorcall simd_vec<float, 4>::operator simd_vec<int, 4>() const {
    simd_vec<int, 4> ret;
    ret.vec_ = _mm_cvttps_epi32(vec_);
    return ret;
}

force_inline vectorcall simd_vec<float, 4>::operator simd_vec<unsigned, 4>() const {
    simd_vec<unsigned, 4> ret;
    ret.vec_ = _mm_cvttps_epi32(vec_);
    return ret;
}

force_inline vectorcall simd_vec<int, 4>::operator simd_vec<unsigned, 4>() const {
    simd_vec<unsigned, 4> ret;
    ret.vec_ = vec_;
    return ret;
}

template <typename U>
force_inline simd_vec<float, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<float, 4> vec1,
                                                  const simd_vec<float, 4> vec2) {
    validate_mask(mask);
    simd_vec<float, 4> ret;
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
force_inline simd_vec<int, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<int, 4> vec1,
                                                const simd_vec<int, 4> vec2) {
    validate_mask(mask);
    simd_vec<int, 4> ret;
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
force_inline simd_vec<unsigned, 4> vectorcall select(const simd_vec<U, 4> mask, const simd_vec<unsigned, 4> vec1,
                                                     const simd_vec<unsigned, 4> vec2) {
    validate_mask(mask);
    simd_vec<unsigned, 4> ret;
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
