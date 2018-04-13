//#pragma once

#include <type_traits>

#include <immintrin.h>
#include <xmmintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("sse2")
#endif

namespace ray {
namespace NS {

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0
#if defined(USE_AVX)
    && S % 8 != 0
#endif
    , float>::type, S> {
    union {
        __m128 vec_[S/4];
        float comp_[S];
    };

    friend class simd_vec<int, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        ITERATE(S/4, { vec_[i] = _mm_set1_ps(f); })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, float>::type head, Tail... tail) {
        const float _tail[] = { tail... };
        vec_[0] = _mm_setr_ps(head, _tail[0], _tail[1], _tail[2]);
        if (S > 4) {
            vec_[1] = _mm_setr_ps(_tail[3], _tail[4], _tail[5], _tail[6]);
            if (S > 8) {
                vec_[2] = _mm_setr_ps(_tail[7], _tail[8], _tail[9], _tail[10]);
                if (S > 12) {
                    vec_[3] = _mm_setr_ps(_tail[11], _tail[12], _tail[13], _tail[14]);
                }
            }
        }

        for (int i = 15; i < S - 1; i += 4) {
            vec_[(i+1)/4] = _mm_setr_ps(_tail[i], _tail[i+1], _tail[i+2], _tail[i+3]);
        }
    }
    force_inline simd_vec(const float *f) {
        ITERATE(S/4, {
            vec_[i] = _mm_loadu_ps(f);
            f += 4;
        })
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        ITERATE(S/4, {
            vec_[i] = _mm_load_ps(f);
            f += 4;
        })
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_add_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator+=(float rhs) {
        __m128 _rhs = _mm_set1_ps(rhs);
        ITERATE(S/4, { vec_[i] = _mm_add_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_sub_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(float rhs) {
        __m128 _rhs = _mm_set1_ps(rhs);
        ITERATE(S/4, { vec_[i] = _mm_sub_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_mul_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(float rhs) {
        __m128 _rhs = _mm_set1_ps(rhs);
        ITERATE(S/4, { vec_[i] = _mm_mul_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_div_ps(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(float rhs) {
        __m128 _rhs = _mm_set1_ps(rhs);
        ITERATE(S/4, { vec_[i] = _mm_div_ps(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> operator-() const {
        simd_vec<float, S> temp;
        __m128 m = _mm_set1_ps(-0.0f);
        ITERATE(S/4, { temp.vec_[i] = _mm_xor_ps(vec_[i], m); })
        return temp;
    }

    force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmplt_ps(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmple_ps(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpgt_ps(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpge_ps(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> operator<(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmplt_ps(vec_[i], _mm_set1_ps(rhs)); })
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmple_ps(vec_[i], _mm_set1_ps(rhs)); })
        return ret;
    }

    force_inline simd_vec<float, S> operator>(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpgt_ps(vec_[i], _mm_set1_ps(rhs)); })
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpge_ps(vec_[i], _mm_set1_ps(rhs)); })
        return ret;
    }

    force_inline operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cvtps_epi32(vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> sqrt() const {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_sqrt_ps(vec_[i]); })
        return temp;
    }

    force_inline void copy_to(float *f) const {
        ITERATE(S/4, { _mm_storeu_ps(f, vec_[i]); f += 4; })
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        ITERATE(S/4, { _mm_store_ps(f, vec_[i]); f += 4; })
    }

    force_inline void blend_to(const simd_vec<float, S> &mask, const simd_vec<float, S> &v1) {
        ITERATE(S/4, { vec_[i] = _mm_blendv_ps(vec_[i], v1.vec_[i], mask.vec_[i]); })
    }

    force_inline static simd_vec<float, S> min(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_min_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> max(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_max_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> and_not(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_andnot_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/4, {
            __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_[i]));
            __m128 r = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_[i], t), _mm_set1_ps(1.0f)));
            temp.vec_[i] = r;
        })
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/4, {
            __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_[i]));
            __m128 r = _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_[i], t), _mm_set1_ps(1.0f)));
            temp.vec_[i] = r;
        })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator&(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_and_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator|(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_or_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator^(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_xor_ps(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_add_ps(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_ps(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_mul_ps(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_div_ps(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_add_ps(v1.vec_[i], _mm_set1_ps(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_ps(v1.vec_[i], _mm_set1_ps(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_mul_ps(v1.vec_[i], _mm_set1_ps(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_div_ps(v1.vec_[i], _mm_set1_ps(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(float v1, const simd_vec<float, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, S> operator-(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_ps(_mm_set1_ps(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_mul_ps(_mm_set1_ps(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_div_ps(_mm_set1_ps(v1), v2.vec_[i]); })
        return ret;
    }

    static const size_t alignment = alignof(__m128);

    static int size() { return S; }
    static int native_count() { return S/4; }
    static bool is_native() { return native_count() == 1; }
};

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0
#if defined(USE_AVX)
    && S % 8 != 0
#endif
    , int>::type, S> {
    union {
        __m128i vec_[S/4];
        int comp_[S];
    };

    friend class simd_vec<float, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        ITERATE(S/4, { vec_[i] = _mm_set1_epi32(f); })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, int>::type head, Tail... tail) {
        const int _tail[] = { tail... };
        vec_[0] = _mm_setr_ps(head, _tail[0], _tail[1], _tail[2]);
        if (S > 4) {
            vec_[1] = _mm_setr_ps(_tail[3], _tail[4], _tail[5], _tail[6]);
            if (S > 8) {
                vec_[2] = _mm_setr_ps(_tail[7], _tail[8], _tail[9], _tail[10]);
                if (S > 12) {
                    vec_[3] = _mm_setr_ps(_tail[11], _tail[12], _tail[13], _tail[14]);
                }
            }
        }

        for (int i = 15; i < S - 1; i += 4) {
            vec_[(i + 1)/4] = _mm_setr_ps(_tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3]);
        }
    }
    force_inline simd_vec(const int *f) {
        ITERATE(S/4, { vec_[i] = _mm_loadu_si128((const __m128i *)f); f += 4; })
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        ITERATE(S/4, { vec_[i] = _mm_load_si128((const __m128i *)f); f += 4; })
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_add_epi32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator+=(int rhs) {
        ITERATE(S/4, { vec_[i] = _mm_add_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        ITERATE(S/4, { vec_[i] = _mm_sub_epi32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(int rhs) {
        ITERATE(S/4, { vec_[i] = _mm_sub_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        ITERATE(S, { comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(int rhs) {
        ITERATE(S, { comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        ITERATE(S, { comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(int rhs) {
        ITERATE(S, { comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> operator==(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpeq_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmplt_epi32(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmple_epi32(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpgt_epi32(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpge_epi32(vec_[i], rhs.vec_[i]); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmplt_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmple_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpgt_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cmpge_epi32(vec_[i], _mm_set1_epi32(rhs)); })
        return ret;
    }

    force_inline operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_cvtepi32_ps(vec_[i]); })
        return ret;
    }

    force_inline void copy_to(int *f) const {
        ITERATE(S/4, { _mm_storeu_si128((__m128i *)f, vec_[i]); f += 4; })
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        ITERATE(S/4, { _mm_store_si128((__m128i *)f, vec_[i]); f += 4; })
    }

    force_inline void blend_to(const simd_vec<int, S> &mask, const simd_vec<int, S> &v1) {
        ITERATE(S/4, { vec_[i] = _mm_blendv_epi8(vec_[i], v1.vec_[i], mask.vec_[i]); })
    }

    force_inline bool all_zeros() const {
        ITERATE(S/4, { if (!_mm_test_all_zeros(vec_[i], vec_[i])) return false; });
#if 0
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(vec_[i], _mm_setzero_si128())) != 0xFFFF) return false;         
#endif
        return true;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        ITERATE(S/4, { if (!_mm_test_all_zeros(vec_[i], mask.vec_[i])) return false; })
#if 1
            
#else
#error "!!!"
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(vec_[i], _mm_setzero_si128())) != 0xFFFF) return false;
#endif
        return true;
    }

    force_inline bool not_all_zeros() const {
        ITERATE(S/4, { if (!_mm_test_all_zeros(vec_[i], vec_[i])) return true; })
#if 0
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(vec_[i], _mm_setzero_si128())) == 0xFFFF) return true;       
#endif
        return false;
    }

    force_inline static simd_vec<int, S> min(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_min_si128(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> max(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_max_epi32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> and_not(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_andnot_si128(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator&(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_and_si128(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator|(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_or_si128(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator^(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = _mm_xor_si128(v1.vec_[i], v2.vec_[i]); });
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_add_epi32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_epi32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.vec_[i] / v2.vec_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_add_epi32(v1.vec_[i], _mm_set1_epi32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_epi32(v1.vec_[i], _mm_set1_epi32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.vec_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator+(int v1, const simd_vec<int, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<int, S> operator-(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_sub_epi32(_mm_set1_epi32(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = _mm_srli_epi32(v1.vec_[i], v2); })
        return ret;
    }

    static const size_t alignment = alignof(__m128i);

    static int size() { return S; }
    static int native_count() { return S/4; }
    static bool is_native() { return native_count() == 1; }
};

#if defined(USE_SSE)
using native_simd_fvec = simd_fvec<4>;
using native_simd_ivec = simd_ivec<4>;
#endif

}
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif