//#pragma once

#include <type_traits>

#include <arm_neon.h>

namespace Ray {
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

template <>
class simd_vec<int, 4>;

template <>
class simd_vec<float, 4> {
    union {
        float32x4_t vec_;
        float comp_[4];
    };

    friend class simd_vec<int, 4>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        vec_ = vdupq_n_f32(f);
    }
    force_inline simd_vec(float f1, float f2, float f3, float f4) {
        vec_ = float32x4_t{ f1, f2, f3, f4 };
    }
    force_inline simd_vec(const float *f) {
        vec_ = vld1q_f32(f);
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        const float *_f = (const float *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_f32(_f);
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, 4> &operator+=(const simd_vec<float, 4> &rhs) {
        vec_ = vaddq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator+=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vaddq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(const simd_vec<float, 4> &rhs) {
        vec_ = vsubq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vsubq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(const simd_vec<float, 4> &rhs) {
        vec_ = vmulq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vmulq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(const simd_vec<float, 4> &rhs) {
        vec_ = vdivq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vdivq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> operator-() const {
        simd_vec<float, 4> temp;
        float32x4_t m = vdupq_n_f32(-0.0f);
        int32x4_t res = veorq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(m));
        temp.vec_ = vreinterpretq_f32_s32(res);
        return temp;
    }

    force_inline simd_vec<float, 4> operator<(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcltq_f32(vec_, rhs.vec_);
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator<=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcleq_f32(vec_, rhs.vec_);
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgtq_f32(vec_, rhs.vec_);
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgeq_f32(vec_, rhs.vec_);
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator<(float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcltq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator<=(float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcleq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>(float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgtq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>=(float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgeq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline operator simd_vec<int, 4>() const;

    force_inline simd_vec<float, 4> sqrt() const {
        simd_vec<float, 4> temp;
        float32x4_t recipsq = vrsqrteq_f32(vec_);
        temp.vec_ = vrecpeq_f32(recipsq);
        return temp;
    }

    force_inline float length() const {
        float temp = 0.0f;
        ITERATE(4, { temp += comp_[i] * comp_[i]; })
        return std::sqrt(temp);
    }

    force_inline void copy_to(float *f) const {
        vst1q_f32(f, vec_); f += 4;
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        float *_f = (float *)__builtin_assume_aligned(f, 16);
        vst1q_f32(_f, vec_); _f += 4;
    }

    force_inline void blend_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
        int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(mask.vec_));
        int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(mask.vec_));
        vec_ = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
    }

    force_inline void blend_inv_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
        int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(mask.vec_));
        int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(mask.vec_));
        vec_ = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
    }

    force_inline static simd_vec<float, 4> min(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vminq_f32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> max(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vmaxq_f32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, 4> and_not(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(v2.vec_), vreinterpretq_s32_f32(v1.vec_)));
        return temp;
    }

    force_inline static simd_vec<float, 4> floor(const simd_vec<float, 4> &v1) {
        simd_vec<float, 4> temp;
        float32x4_t t = neon_cvt_s32_to_f32(neon_cvt_f32_to_s32(v1.vec_));
        float32x4_t r = vsubq_f32(t, vandq_s32(vcltq_f32(v1.vec_, t), vdupq_n_f32(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    force_inline static simd_vec<float, 4> ceil(const simd_vec<float, 4> &v1) {
        simd_vec<float, 4> temp;
        float32x4_t t = neon_cvt_s32_to_f32(neon_cvt_f32_to_s32(v1.vec_));
        float32x4_t r = vaddq_f32(t, vandq_s32(vcgtq_f32(v1.vec_, t), vdupq_n_f32(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator&(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator|(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator^(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> temp;
        temp.vec_ = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
        return temp;
    }

    friend force_inline simd_vec<float, 4> operator+(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vaddq_f32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator-(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vsubq_f32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmulq_f32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vdivq_f32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator+(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vaddq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator-(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vsubq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmulq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const simd_vec<float, 4> &v1, float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vdivq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator+(float v1, const simd_vec<float, 4> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, 4> operator-(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vsubq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmulq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vdivq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline float dot(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        float32x4_t r1 = vmulq_f32(v1.vec_, v2.vec_);
        float32x2_t r2 = vadd_f32(vget_high_f32(r1), vget_low_f32(r1));
        return vget_lane_f32(vpadd_f32(r2, r2), 0);

    }

    friend force_inline simd_vec<float, 4> clamp(const simd_vec<float, 4> &v1, float min, float max) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmaxq_f32(vdupq_n_f32(min), vminq_f32(v1.vec_, vdupq_n_f32(max)));
        return ret;
    }

    friend force_inline simd_vec<float, 4> pow(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ITERATE_4({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, 4> normalize(const simd_vec<float, 4> &v1) {
        return v1 / v1.length();
    }

    friend force_inline const float *value_ptr(const simd_vec<float, 4> &v1) {
        return &v1.comp_[0];
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <>
class simd_vec<int, 4> {
    union {
        int32x4_t vec_;
        int comp_[4];
    };

    friend class simd_vec<float, 4>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        vec_ = vdupq_n_s32(f);
    }
    force_inline simd_vec(int i1, int i2, int i3, int i4) {
        vec_ = int32x4_t{ i1, i2, i3, i4 };
    }
    force_inline simd_vec(const int *f) {
        vec_ = vld1q_s32((const int32_t *)f);
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_s32((const int32_t *)_f);
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, 4> &operator+=(const simd_vec<int, 4> &rhs) {
        vec_ = vaddq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator+=(int rhs) {
        vec_ = vaddq_s32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &operator-=(const simd_vec<int, 4> &rhs) {
        vec_ = vsubq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator-=(int rhs) {
        vec_ = vsubq_s32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &operator*=(const simd_vec<int, 4> &rhs) {
        vec_ = vmulq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator*=(int rhs) {
        vec_ = vmulq_s32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> &operator/=(const simd_vec<int, 4> &rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, 4> &operator/=(int rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, 4> operator==(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vceqq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator==(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vceqq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator!=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(vec_, vdupq_n_s32(rhs))));
        return ret;
    }

    force_inline simd_vec<int, 4> operator!=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(vec_, rhs.vec_)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcltq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcleq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgtq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>=(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgeq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcltq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcleq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgtq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>=(int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgeq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = vcvtq_f32_s32(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const {
        vst1q_s32((int32_t *)f, vec_);
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vst1q_s32((int32_t *)_f, vec_);
    }

    force_inline void blend_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
        int32x4_t temp1 = vandq_s32(v1.vec_, mask.vec_);
        int32x4_t temp2 = vbicq_s32(vec_, mask.vec_);
        vec_ = vorrq_s32(temp1, temp2);
    }

    force_inline void blend_inv_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
        int32x4_t temp1 = vandq_s32(vec_, mask.vec_);
        int32x4_t temp2 = vbicq_s32(v1.vec_, mask.vec_);
        vec_ = vorrq_s32(temp1, temp2);
    }

    force_inline bool all_zeros() const {
        int32_t res = 0;
#if defined(__aarch64__)
        res |= vaddvq_s32(vec_);
#else
        ITERATE_4({ res |= comp_[i] != 0; })
#endif
        return res == 0;
    }

    force_inline bool all_zeros(const simd_vec<int, 4> &mask) const {
        int32_t res = 0;
#if defined(__aarch64__)
        res |= vaddvq_s32(vandq_s32(vec_, mask.vec_));
#else
        ITERATE_4({ res |= (comp_[i] & mask.comp_[i]) != 0; })
#endif
        return res == 0;
    }

    force_inline bool not_all_zeros() const {
        return !all_zeros();
    }

    force_inline static simd_vec<int, 4> min(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = vminq_s32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 4> max(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = vmaxq_s32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, 4> and_not(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = vbicq_s32(v2.vec_, v1.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator&(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = vandq_s32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator|(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = vorrq_s32(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator^(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> temp;
        temp.vec_ = veorq_s32(v1.vec_, v2.vec_);;
        return temp;
    }

    friend force_inline simd_vec<int, 4> operator+(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vaddq_s32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator-(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vsubq_s32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vaddq_s32(v1.vec_, vdupq_n_s32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator-(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vsubq_s32(v1.vec_, vdupq_n_s32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(int v1, const simd_vec<int, 4> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<int, 4> operator-(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vsubq_s32(vdupq_n_s32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = reinterpret_cast<const unsigned &>(v1.comp_[i]) >> v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = reinterpret_cast<const unsigned &>(v1.comp_[i]) >> v2; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, int v2) {
        simd_vec<int, 4> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] << v2; })
        return ret;
    }

    friend force_inline bool is_equal(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        bool res = true;
        ITERATE_4({ res &= (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline simd_vec<float, 4>::operator simd_vec<int, 4>() const {
    simd_vec<int, 4> ret;
    ret.vec_ = neon_cvt_f32_to_s32(vec_);
    return ret;
}

#if defined(USE_NEON)
using native_simd_fvec = simd_vec<float, 4>;
using native_simd_ivec = simd_vec<int, 4>;
#endif

}
}
