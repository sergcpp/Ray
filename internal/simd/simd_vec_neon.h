//#pragma once

#include <type_traits>

#include <arm_neon.h>

namespace ray {
namespace NS {

force_inline float32x4_t vdivq_f32(float32x4_t num, float32x4_t den) {
    const float32x4_t q_inv0 = vrecpeq_f32(den);
    const float32x4_t q_step0 = vrecpsq_f32(q_inv0, den);

    const float32x4_t q_inv1 = vmulq_f32(q_step0, q_inv0);
    return vmulq_f32(num, q_inv1);
}

force_inline int32x4_t neon_cvt_f32_to_s32(float32x4_t a) {
#if defined(__aarch64__)
    return vcvtnq_s32_f32(a);
#else
    uint32x4_t signmask = vdupq_n_u32(0x80000000);
    float32x4_t half = vbslq_f32(signmask, a, vdupq_n_f32(0.5f)); /* +/- 0.5 */
    int32x4_t r_normal = vcvtq_s32_f32(vaddq_f32(a, half)); /* round to integer: [a + 0.5]*/
    int32x4_t r_trunc = vcvtq_s32_f32(a); /* truncate to integer: [a] */
    int32x4_t plusone = vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(vnegq_s32(r_trunc)), 31)); /* 1 or 0 */
    int32x4_t r_even = vbicq_s32(vaddq_s32(r_trunc, plusone), vdupq_n_s32(1)); /* ([a] + {0,1}) & ~1 */
    float32x4_t delta = vsubq_f32(a, vcvtq_f32_s32(r_trunc)); /* compute delta: delta = (a - [a]) */
    uint32x4_t is_delta_half = vceqq_f32(delta, half); /* delta == +/- 0.5 */
    return vbslq_s32(is_delta_half, r_even, r_normal);
#endif
}

force_inline float32x4_t neon_cvt_s32_to_f32(int32x4_t a) {
    return vcvtq_f32_s32(a);
}

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0, float>::type, S> {
    union {
        float32x4_t vec_[S/4];
        float comp_[S];
    };

    friend class simd_vec<int, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        ITERATE(S/4, { vec_[i] = vdupq_n_f32(f); })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, float>::type head, Tail... tail) {
        const float _tail[] = { tail... };
        vec_[0] = float32x4_t{ head, _tail[0], _tail[1], _tail[2] };
        if (S > 4) {
            vec_[1] = float32x4_t{ _tail[3], _tail[4], _tail[5], _tail[6] };
            if (S > 8) {
                vec_[2] = float32x4_t{ _tail[7], _tail[8], _tail[9], _tail[10] };
                if (S > 12) {
                    vec_[3] = float32x4_t{ _tail[11], _tail[12], _tail[13], _tail[14] };
                }
            }
        }

        for (int i = 15; i < S - 1; i += 4) {
            vec_[(i+1)/4] = float32x4_t{ _tail[i], _tail[i+1], _tail[i+2], _tail[i+3] };
        }
    }
    force_inline simd_vec(const float *f) {
        ITERATE(S/4, {
            vec_[i] = vld1q_f32(f);
            f += 4;
        })
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        const float *_f = (const float *)__builtin_assume_aligned(f, 16);
        ITERATE(S/4, {
            vec_[i] = vld1q_f32(_f);
            _f += 4;
        })
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vaddq_f32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator+=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        ITERATE(S/4, { vec_[i] = vaddq_f32(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vsubq_f32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        ITERATE(S/4, { vec_[i] = vsubq_f32(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vmulq_f32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        ITERATE(S/4, { vec_[i] = vmulq_f32(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vdivq_f32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        ITERATE(S/4, { vec_[i] = vdivq_f32(vec_[i], _rhs); })
        return *this;
    }

    force_inline simd_vec<float, S> operator-() const {
        simd_vec<float, S> temp;
        float32x4_t m = vdupq_n_f32(-0.0f);
        ITERATE(S/4, {
            int32x4_t res = veorq_s32(vreinterpretq_s32_f32(vec_[i]), vreinterpretq_s32_f32(m));
            temp.vec_[i] = vreinterpretq_f32_s32(res);
        })
        return temp;
    }

    force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcltq_f32(vec_[i], rhs.vec_[i]);
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcleq_f32(vec_[i], rhs.vec_[i]);
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcgtq_f32(vec_[i], rhs.vec_[i]);
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcgeq_f32(vec_[i], rhs.vec_[i]);
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator<(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcltq_f32(vec_[i], vdupq_n_f32(rhs));
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcleq_f32(vec_[i], vdupq_n_f32(rhs));
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator>(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcgtq_f32(vec_[i], vdupq_n_f32(rhs));
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(float rhs) const {
        simd_vec<float, S> ret;
        ITERATE(S/4, {
            uint32x4_t res = vcgeq_f32(vec_[i], vdupq_n_f32(rhs));
            ret.vec_[i] = vreinterpretq_f32_u32(res);
        })
        return ret;
    }

    force_inline operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = neon_cvt_f32_to_s32(vec_[i]); })
        return ret;
    }

    force_inline simd_vec<float, S> sqrt() const {
        simd_vec<float, S> temp;
        ITERATE(S/4, {
            float32x4_t recipsq = vrsqrteq_f32(vec_[i]);
            temp.vec_[i] = vrecpeq_f32(recipsq);
        })
        return temp;
    }

    force_inline void copy_to(float *f) const {
        ITERATE(S/4, { vst1q_f32(f, vec_[i]); f += 4; })
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        float *_f = (float *)__builtin_assume_aligned(f, 16);
        ITERATE(S/4, { vst1q_f32(_f, vec_[i]); _f += 4; })
    }

    force_inline void blend_to(const simd_vec<float, S> &mask, const simd_vec<float, S> &v1) {
        ITERATE(S/4, {
            int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(mask.vec_[i]), vreinterpretq_s32_f32(v1.vec_[i]));
            int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(vec_[i]), vreinterpretq_s32_f32(mask.vec_[i]));
            vec_[i] = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
        })
    }

    force_inline static simd_vec<float, S> min(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vminq_f32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> max(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vmaxq_f32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<float, S> and_not(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(v2.vec_[i]), vreinterpretq_s32_f32(v1.vec_[i]))); })
        return temp;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/4, {
            float32x4_t t = neon_cvt_s32_to_f32(neon_cvt_f32_to_s32(v1.vec_[i]));
            float32x4_t r = vsubq_f32(t, vandq_s32(vcltq_f32(v1.vec_[i], t), vdupq_n_f32(1.0f)));
            temp.vec_[i] = r;
        })
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        ITERATE(S/4, {
            float32x4_t t = neon_cvt_s32_to_f32(neon_cvt_f32_to_s32(v1.vec_[i]));
            float32x4_t r = vaddq_f32(t, vandq_s32(vcgtq_f32(v1.vec_[i], t), vdupq_n_f32(1.0f)));
            temp.vec_[i] = r;
        })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator&(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(v1.vec_[i]), vreinterpretq_s32_f32(v2.vec_[i]))); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator|(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(v1.vec_[i]), vreinterpretq_s32_f32(v2.vec_[i]))); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator^(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(v1.vec_[i]), vreinterpretq_s32_f32(v2.vec_[i]))); })
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vaddq_f32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vsubq_f32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vmulq_f32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vdivq_f32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vaddq_f32(v1.vec_[i], vdupq_n_f32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vsubq_f32(v1.vec_[i], vdupq_n_f32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vmulq_f32(v1.vec_[i], vdupq_n_f32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vdivq_f32(v1.vec_[i], vdupq_n_f32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(float v1, const simd_vec<float, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, S> operator-(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vsubq_f32(vdupq_n_f32(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vmulq_f32(vdupq_n_f32(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vdivq_f32(vdupq_n_f32(v1), v2.vec_[i]); })
        return ret;
    }

    friend force_inline float dot(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        float ret = { 0 };
        
        ITERATE(S/4, ({
            float32x4_t r1 = vmulq_f32(v1.vec_[i], v2.vec_[i]);
            float32x2_t r2 = vadd_f32(vget_high_f32(r1), vget_low_f32(r1));
            ret += vget_lane_f32(vpadd_f32(r2, r2), 0);
        });)
            
        return ret;
    }

    friend force_inline simd_vec<float, S> clamp(const simd_vec<float, S> &v1, float min, float max) {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vmaxq_f32(vdupq_n_f32(min), vminq_f32(v1.vec_[i], vdupq_n_f32(max))); })
        return ret;
    }

    friend force_inline simd_vec<float, S> pow(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE(S, { ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> normalize(const simd_vec<float, S> &v1) {
        return v1 / v1.length();
    }

    friend force_inline const float *value_ptr(const simd_vec<float, S> &v1) {
        return &v1.comp_[0];
    }

    static const size_t alignment = alignof(float32x4_t);

    static int size() { return S; }
    static int native_count() { return S/4; }
    static bool is_native() { return native_count() == 1; }
};

template <int S>
class simd_vec<typename std::enable_if<S % 4 == 0, int>::type, S> {
    union {
        int32x4_t vec_[S/4];
        int comp_[S];
    };

    friend class simd_vec<float, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        ITERATE(S/4, { vec_[i] = vdupq_n_s32(f); })
    }
    template <typename... Tail>
    force_inline simd_vec(typename std::enable_if<sizeof...(Tail)+1 == S, int>::type head, Tail... tail) {
        const int _tail[] = { tail... };
        vec_[0] = int32x4_t{ head, _tail[0], _tail[1], _tail[2] };
        if (S > 4) {
            vec_[1] = int32x4_t{ _tail[3], _tail[4], _tail[5], _tail[6] };
            if (S > 8) {
                vec_[2] = int32x4_t{ _tail[7], _tail[8], _tail[9], _tail[10] };
                if (S > 12) {
                    vec_[3] = int32x4_t{ _tail[11], _tail[12], _tail[13], _tail[14] };
                }
            }
        }

        for (int i = 15; i < S - 1; i += 4) {
            vec_[(i + 1)/4] = int32x4_t{ _tail[i], _tail[i + 1], _tail[i + 2], _tail[i + 3] };
        }
    }
    force_inline simd_vec(const int *f) {
        ITERATE(S/4, { vec_[i] = vld1q_s32((const int32_t *)f); f += 4; })
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        ITERATE(S/4, { vec_[i] = vld1q_s32((const int32_t *)_f); _f += 4; })
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vaddq_s32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator+=(int rhs) {
        ITERATE(S/4, { vec_[i] = vaddq_s32(vec_[i], vdupq_n_s32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vsubq_s32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(int rhs) {
        ITERATE(S/4, { vec_[i] = vsubq_s32(vec_[i], vdupq_n_s32(rhs)); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        ITERATE(S/4, { vec_[i] = vmulq_s32(vec_[i], rhs.vec_[i]); })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(int rhs) {
        ITERATE(S/4, { vec_[i] = vmulq_s32(vec_[i], vdupq_n_s32(rhs)); })
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
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vceqq_s32(vec_[i], vdupq_n_s32(rhs))); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcltq_s32(vec_[i], rhs.vec_[i])); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcleq_s32(vec_[i], rhs.vec_[i])); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcgtq_s32(vec_[i], rhs.vec_[i])); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcgeq_s32(vec_[i], rhs.vec_[i])); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcltq_s32(vec_[i], vdupq_n_s32(rhs))); })
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcleq_s32(vec_[i], vdupq_n_s32(rhs))); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcgtq_s32(vec_[i], vdupq_n_s32(rhs))); })
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(int rhs) const {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vreinterpretq_s32_u32(vcgeq_s32(vec_[i], vdupq_n_s32(rhs))); })
        return ret;
    }

    force_inline operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vcvtq_f32_s32(vec_[i]); })
        return ret;
    }

    force_inline void copy_to(int *f) const {
        ITERATE(S/4, { vst1q_s32((int32_t *)f, vec_[i]); f += 4; })
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        ITERATE(S/4, { vst1q_s32((int32_t *)_f, vec_[i]); _f += 4; })
    }

    force_inline void blend_to(const simd_vec<int, S> &mask, const simd_vec<int, S> &v1) {
        ITERATE(S/4, {
            int32x4_t temp1 = vandq_s32(mask.vec_[i], v1.vec_[i]);
            int32x4_t temp2 = vbicq_s32(vec_[i], mask.vec_[i]);
            vec_[i] = vorrq_s32(temp1, temp2);
        })
    }

    force_inline bool all_zeros() const {
        int32_t res = 0;
#if defined(__aarch64__)
        ITERATE(S/4, { res |= vaddvq_s32(vec_[i]); })
#else
        ITERATE(S, { res |= comp_[i] != 0; })
#endif
        return res == 0;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const {
        int32_t res = 0;
#if defined(__aarch64__)
        ITERATE(S/4, { res |= vaddvq_s32(vandq_s32(vec_[i], mask.vec_[i])); })
#else
        ITERATE(S, { res |= (comp_[i] & mask.comp_[i]) != 0; })
#endif
        return res == 0;
    }

    force_inline bool not_all_zeros() const {
        return !all_zeros();
    }

    force_inline static simd_vec<int, S> min(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vminq_s32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> max(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vmaxq_s32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    force_inline static simd_vec<int, S> and_not(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vbicq_s32(v2.vec_[i], v1.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator&(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vandq_s32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator|(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = vorrq_s32(v1.vec_[i], v2.vec_[i]); })
        return temp;
    }

    friend force_inline simd_vec<int, S> operator^(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        ITERATE(S/4, { temp.vec_[i] = veorq_s32(v1.vec_[i], v2.vec_[i]); });
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vaddq_s32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vsubq_s32(v1.vec_[i], v2.vec_[i]); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vaddq_s32(v1.vec_[i], vdupq_n_s32(v2)); })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S/4, { ret.vec_[i] = vsubq_s32(v1.vec_[i], vdupq_n_s32(v2)); })
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
        ITERATE(S/4, { ret.vec_[i] = vsubq_s32(vdupq_n_s32(v1), v2.vec_[i]); })
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

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE(S, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    static const size_t alignment = alignof(int32x4_t);

    static int size() { return S; }
    static int native_count() { return S/4; }
    static bool is_native() { return native_count() == 1; }
};

#if defined(USE_NEON)
using native_simd_fvec = simd_fvec<4>;
using native_simd_ivec = simd_ivec<4>;
#endif

}
}
