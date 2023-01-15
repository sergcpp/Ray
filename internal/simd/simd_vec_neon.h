// #pragma once

#include <type_traits>

#include <arm_neon.h>

#ifndef NDEBUG
#define VALIDATE_MASKS 1
#endif

namespace Ray {
namespace NS {
/*#ifndef vdivq_f32 force_inline float32x4_t vdivq_f32(float32x4_t num, float32x4_t den) {
    const float32x4_t q_inv0 = vrecpeq_f32(den);
    const float32x4_t q_step0 = vrecpsq_f32(q_inv0, den);

    const float32x4_t q_inv1 = vmulq_f32(q_step0, q_inv0);
    return vmulq_f32(num, q_inv1);
}
#endif*/

template <> class simd_vec<int, 4>;

template <> class simd_vec<float, 4> {
    float32x4_t vec_;

    friend class simd_vec<int, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const float f) { vec_ = vdupq_n_f32(f); }
    force_inline simd_vec(const float f1, const float f2, const float f3, const float f4) {
        alignas(16) const float init[4] = {f1, f2, f3, f4};
        vec_ = vld1q_f32(init);
    }
    force_inline simd_vec(const float *f) { vec_ = vld1q_f32(f); }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        const float *_f = (const float *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_f32(_f);
    }

    force_inline float operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.n128_f32[i];
#else
        alignas(16) float temp[4];
        vst1q_f32(temp, vec_);
        return temp[i];
#endif
    }

    template <int i> force_inline float get() const { return vgetq_lane_f32(vec_, i & 3); }
    template <int i> force_inline void set(const float f) { vec_ = vsetq_lane_f32(f, vec_, i & 3); }
    force_inline void set(const int i, const float v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.n128_f32[i] = v;
#else
        alignas(16) float temp[4];
        vst1q_f32(temp, vec_);
        temp[i] = v;
        vec_ = vld1q_f32(temp);
#endif
    }

    force_inline simd_vec<float, 4> &operator+=(const simd_vec<float, 4> &rhs) {
        vec_ = vaddq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator+=(const float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vaddq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(const simd_vec<float, 4> &rhs) {
        vec_ = vsubq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator-=(const float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vsubq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(const simd_vec<float, 4> &rhs) {
        vec_ = vmulq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator*=(const float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vmulq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(const simd_vec<float, 4> &rhs) {
        vec_ = vdivq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator/=(const float rhs) {
        float32x4_t _rhs = vdupq_n_f32(rhs);
        vec_ = vdivq_f32(vec_, _rhs);
        return *this;
    }

    force_inline simd_vec<float, 4> &operator|=(const simd_vec<float, 4> &rhs) {
        vec_ = vorrq_u32(vreinterpretq_u32_f32(vec_), vreinterpretq_u32_f32(rhs.vec_));
        return *this;
    }

    force_inline simd_vec<float, 4> &operator|=(const float rhs) {
        vec_ = vorrq_u32(vreinterpretq_u32_f32(vec_), vreinterpretq_u32_f32(vdupq_n_f32(rhs)));
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

    force_inline simd_vec<float, 4> operator&=(const simd_vec<float, 4> &rhs) {
        vec_ = vandq_u32(vreinterpretq_u32_f32(vec_), vreinterpretq_u32_f32(rhs.vec_));
        return *this;
    }

    force_inline simd_vec<float, 4> operator~() const {
        simd_vec<float, 4> ret;
        ret.vec_ = vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(vec_)));
        return ret;
    }

    force_inline simd_vec<float, 4> operator<(const float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcltq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator<=(const float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcleq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>(const float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgtq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline simd_vec<float, 4> operator>=(const float rhs) const {
        simd_vec<float, 4> ret;
        uint32x4_t res = vcgeq_f32(vec_, vdupq_n_f32(rhs));
        ret.vec_ = vreinterpretq_f32_u32(res);
        return ret;
    }

    force_inline operator simd_vec<int, 4>() const;

    force_inline simd_vec<float, 4> sqrt() const {
        // This is not precise enough :(
        // float32x4_t recipsq = vrsqrteq_f32(vec_);
        // temp.vec_ = vrecpeq_f32(recipsq);

        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        ITERATE_4({ comp[i] = std::sqrt(comp[i]); })
        return simd_vec<float, 4>{comp, simd_mem_aligned};
    }

    force_inline simd_vec<float, 4> log() const {
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        ITERATE_4({ comp[i] = std::log(comp[i]); })
        return simd_vec<float, 4>{comp, simd_mem_aligned};
    }

    force_inline float length() const { return std::sqrt(length2()); }

    force_inline float length2() const {
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);

        float temp = 0.0f;
        ITERATE_4({ temp += comp[i] * comp[i]; })
        return temp;
    }

    force_inline void copy_to(float *f) const { vst1q_f32(f, vec_); }
    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        float *_f = (float *)__builtin_assume_aligned(f, 16);
        vst1q_f32(_f, vec_);
    }

    force_inline void blend_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
#if VALIDATE_MASKS
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        ITERATE_4({
            assert(reinterpret_cast<const uint32_t &>(comp[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(comp[i]) == 0xffffffff);
        })
#endif
        int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(mask.vec_));
        int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(mask.vec_));
        vec_ = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
    }

    force_inline void blend_inv_to(const simd_vec<float, 4> &mask, const simd_vec<float, 4> &v1) {
#if VALIDATE_MASKS
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        ITERATE_4({
            assert(reinterpret_cast<const uint32_t &>(comp[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(comp[i]) == 0xffffffff);
        })
#endif
        int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(mask.vec_));
        int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(mask.vec_));
        vec_ = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
    }

    force_inline int movemask() const {
        // Taken from sse2neon
        uint32x4_t input = vreinterpretq_u32_f32(vec_);
        // Shift out everything but the sign bits with a 32-bit unsigned shift right.
        uint64x2_t high_bits = vreinterpretq_u64_u32(vshrq_n_u32(input, 31));
        // Merge the two pairs together with a 64-bit unsigned shift right + add.
        uint8x16_t paired = vreinterpretq_u8_u64(vsraq_n_u64(high_bits, high_bits, 31));
        // Extract the result.
        return vgetq_lane_u8(paired, 0) | (vgetq_lane_u8(paired, 8) << 2);
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
        float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(v1.vec_));
        float32x4_t r = vsubq_f32(t, vandq_s32(vcltq_f32(v1.vec_, t), vdupq_n_f32(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    force_inline static simd_vec<float, 4> ceil(const simd_vec<float, 4> &v1) {
        simd_vec<float, 4> temp;
        float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(v1.vec_));
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

    force_inline simd_vec<float, 4> operator==(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = vceqq_f32(vec_, vdupq_n_f32(rhs));
        return ret;
    }

    force_inline simd_vec<float, 4> operator==(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = vceqq_f32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, 4> operator!=(const float rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(vec_, vdupq_n_f32(rhs))));
        return ret;
    }

    force_inline simd_vec<float, 4> operator!=(const simd_vec<float, 4> &rhs) const {
        simd_vec<float, 4> ret;
        ret.vec_ = vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(vec_, rhs.vec_)));
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

    friend force_inline simd_vec<float, 4> operator+(const simd_vec<float, 4> &v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vaddq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator-(const simd_vec<float, 4> &v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vsubq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const simd_vec<float, 4> &v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmulq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const simd_vec<float, 4> &v1, const float v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vdivq_f32(v1.vec_, vdupq_n_f32(v2));
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator+(const float v1, const simd_vec<float, 4> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, 4> operator-(const float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vsubq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator*(const float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmulq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, 4> operator/(const float v1, const simd_vec<float, 4> &v2) {
        simd_vec<float, 4> ret;
        ret.vec_ = vdivq_f32(vdupq_n_f32(v1), v2.vec_);
        return ret;
    }

    friend force_inline float dot(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        float32x4_t r1 = vmulq_f32(v1.vec_, v2.vec_);
        float32x2_t r2 = vadd_f32(vget_high_f32(r1), vget_low_f32(r1));
        return vget_lane_f32(vpadd_f32(r2, r2), 0);
    }

    friend force_inline simd_vec<float, 4> clamp(const simd_vec<float, 4> &v1, const float min, const float max) {
        simd_vec<float, 4> ret;
        ret.vec_ = vmaxq_f32(vdupq_n_f32(min), vminq_f32(v1.vec_, vdupq_n_f32(max)));
        return ret;
    }

    friend force_inline simd_vec<float, 4> pow(const simd_vec<float, 4> &v1, const simd_vec<float, 4> &v2) {
        alignas(16) float comp1[4], comp2[4];
        vst1q_f32(comp1, v1.vec_);
        vst1q_f32(comp2, v2.vec_);
        ITERATE_4({ comp1[i] = std::pow(comp1[i], comp2[i]); })
        return simd_vec<float, 4>{comp1, simd_mem_aligned};
    }

    friend force_inline simd_vec<float, 4> normalize(const simd_vec<float, 4> &v1) { return v1 / v1.length(); }

    friend force_inline const float *value_ptr(const simd_vec<float, 4> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(simd_vec<float, 4> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class simd_vec<int, 4> {
    int32x4_t vec_;

    friend class simd_vec<float, 4>;

  public:
    force_inline simd_vec() = default;
    force_inline simd_vec(const int f) { vec_ = vdupq_n_s32(f); }
    force_inline simd_vec(const int i1, const int i2, const int i3, const int i4) {
        alignas(16) const int init[4] = {i1, i2, i3, i4};
        vec_ = vld1q_s32(init);
    }
    force_inline simd_vec(const int *f) { vec_ = vld1q_s32((const int32_t *)f); }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_s32((const int32_t *)_f);
    }

    force_inline int operator[](const int i) const {
#if defined(_MSC_VER) && !defined(__clang__)
        return vec_.n128_i32[i];
#else
        alignas(16) int temp[4];
        vst1q_s32(temp, vec_);
        return temp[i];
#endif
    }

    template <int i> force_inline int get() const { return vgetq_lane_s32(vec_, i & 3); }
    template <int i> force_inline void set(const int f) { vec_ = vsetq_lane_s32(f, vec_, i & 3); }
    force_inline void set(const int i, const int v) {
#if defined(_MSC_VER) && !defined(__clang__)
        vec_.n128_i32[i] = v;
#else
        alignas(16) int temp[4];
        vst1q_s32(temp, vec_);
        temp[i] = v;
        vec_ = vld1q_s32(temp);
#endif
    }

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
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ vec_.n128_i32[i] /= rhs.vec_.n128_i32[i]; })
#else
        alignas(16) int comp[4], rhs_comp[4];
        vst1q_s32(comp, vec_);
        vst1q_s32(rhs_comp, rhs.vec_);
        ITERATE_4({ comp[i] = comp[i] / rhs_comp[i]; })
        vec_ = vld1q_s32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &operator/=(const int rhs) {
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ vec_.n128_i32[i] /= rhs; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, vec_);
        ITERATE_4({ comp[i] = comp[i] / rhs; })
        vec_ = vld1q_s32(comp);
#endif
        return *this;
    }

    force_inline simd_vec<int, 4> &operator|=(const simd_vec<int, 4> &rhs) {
        vec_ = vorrq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> &operator|=(const int rhs) {
        vec_ = vorrq_s32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    force_inline simd_vec<int, 4> operator-() const {
        simd_vec<int, 4> temp;
        temp.vec_ = vsubq_s32(vdupq_n_s32(0), vec_);
        return temp;
    }

    force_inline simd_vec<int, 4> operator==(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vceqq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator==(const simd_vec<int, 4> &rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vceqq_s32(vec_, rhs.vec_));
        return ret;
    }

    force_inline simd_vec<int, 4> operator!=(const int rhs) const {
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

    force_inline simd_vec<int, 4> operator&=(const simd_vec<int, 4> &rhs) {
        vec_ = vandq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, 4> operator~() const {
        simd_vec<int, 4> ret;
        ret.vec_ = vmvnq_u32(vec_);
        return ret;
    }

    force_inline simd_vec<int, 4> operator<(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcltq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator<=(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcleq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgtq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> operator>=(const int rhs) const {
        simd_vec<int, 4> ret;
        ret.vec_ = vreinterpretq_s32_u32(vcgeq_s32(vec_, vdupq_n_s32(rhs)));
        return ret;
    }

    force_inline simd_vec<int, 4> &operator&=(const int rhs) {
        vec_ = vandq_u32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    force_inline operator simd_vec<float, 4>() const {
        simd_vec<float, 4> ret;
        ret.vec_ = vcvtq_f32_s32(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const { vst1q_s32((int32_t *)f, vec_); }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vst1q_s32((int32_t *)_f, vec_);
    }

    force_inline void blend_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
#if VALIDATE_MASKS
        alignas(16) int comp[4];
        vst1q_s32(comp, mask.vec_);
        ITERATE_4({
            assert(reinterpret_cast<const uint32_t &>(comp[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(comp[i]) == 0xffffffff);
        })
#endif
        int32x4_t temp1 = vandq_s32(v1.vec_, mask.vec_);
        int32x4_t temp2 = vbicq_s32(vec_, mask.vec_);
        vec_ = vorrq_s32(temp1, temp2);
    }

    force_inline void blend_inv_to(const simd_vec<int, 4> &mask, const simd_vec<int, 4> &v1) {
#if VALIDATE_MASKS
        alignas(16) int comp[4];
        vst1q_s32(comp, mask.vec_);
        ITERATE_4({
            assert(reinterpret_cast<const uint32_t &>(comp[i]) == 0 ||
                   reinterpret_cast<const uint32_t &>(comp[i]) == 0xffffffff);
        })
#endif
        int32x4_t temp1 = vandq_s32(vec_, mask.vec_);
        int32x4_t temp2 = vbicq_s32(v1.vec_, mask.vec_);
        vec_ = vorrq_s32(temp1, temp2);
    }

    force_inline int movemask() const {
        // Taken from sse2neon
        uint32x4_t input = vreinterpretq_u32_s32(vec_);
        // Shift out everything but the sign bits with a 32-bit unsigned shift right.
        uint64x2_t high_bits = vreinterpretq_u64_u32(vshrq_n_u32(input, 31));
        // Merge the two pairs together with a 64-bit unsigned shift right + add.
        uint8x16_t paired = vreinterpretq_u8_u64(vsraq_n_u64(high_bits, high_bits, 31));
        // Extract the result.
        return vgetq_lane_u8(paired, 0) | (vgetq_lane_u8(paired, 8) << 2);
    }

    force_inline bool all_zeros() const {
        int32_t res = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
        res |= vaddvq_s32(vec_);
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, vec_);
        ITERATE_4({ res |= comp[i] != 0; })
#endif
        return res == 0;
    }

    force_inline bool all_zeros(const simd_vec<int, 4> &mask) const {
        int32_t res = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
        res |= vaddvq_s32(vandq_s32(vec_, mask.vec_));
#else
        alignas(16) int comp[4], mask_comp[4];
        vst1q_s32(comp, vec_);
        vst1q_s32(mask_comp, mask.vec_);
        ITERATE_4({ res |= (comp[i] & mask_comp[i]) != 0; })
#endif
        return res == 0;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

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
        temp.vec_ = veorq_s32(v1.vec_, v2.vec_);
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
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1.vec_.n128_i32[i] * v2.vec_.n128_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        vst1q_s32(comp1, v1.vec_);
        vst1q_s32(comp2, v2.vec_);
        ITERATE_4({ comp1[i] = comp1[i] * comp2[i]; })
        ret.vec_ = vld1q_s32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1.vec_.n128_i32[i] / v2.vec_.n128_i32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        vst1q_s32(comp1, v1.vec_);
        vst1q_s32(comp2, v2.vec_);
        ITERATE_4({ comp1[i] = comp1[i] / comp2[i]; })
        ret.vec_ = vld1q_s32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vaddq_s32(v1.vec_, vdupq_n_s32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator-(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vsubq_s32(v1.vec_, vdupq_n_s32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1.vec_.n128_i32[i] * v2; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v1.vec_);
        ITERATE_4({ comp[i] *= v2; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1.vec_.n128_i32[i] / v2; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v1.vec_);
        ITERATE_4({ comp[i] /= v2; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator+(const int v1, const simd_vec<int, 4> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<int, 4> operator-(const int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vsubq_s32(vdupq_n_s32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator*(const int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1 * v2.vec_.n128_i32[i]; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v2.vec_);
        ITERATE_4({ comp[i] *= v1; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator/(const int v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_i32[i] = v1 / v2.vec_.n128_i32[i]; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v2.vec_);
        ITERATE_4({ comp[i] = v1 / comp[i]; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_u32[i] = v1.vec_.n128_u32[i] >> v2.vec_.n128_u32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        vst1q_s32(comp1, v1.vec_);
        vst1q_s32(comp2, v2.vec_);
        ITERATE_4({ comp1[i] = reinterpret_cast<const unsigned &>(comp1[i]) >> comp2[i]; })
        ret.vec_ = vld1q_s32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator>>(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_u32[i] = v1.vec_.n128_u32[i] >> v2; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v1.vec_);
        ITERATE_4({ comp[i] = reinterpret_cast<const unsigned &>(comp[i]) >> v2; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_u32[i] = v1.vec_.n128_u32[i] << v2.vec_.n128_u32[i]; })
#else
        alignas(16) int comp1[4], comp2[4];
        vst1q_s32(comp1, v1.vec_);
        vst1q_s32(comp2, v2.vec_);
        ITERATE_4({ comp1[i] = comp1[i] << comp2[i]; })
        ret.vec_ = vld1q_s32(comp1);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> operator<<(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
#if defined(_MSC_VER) && !defined(__clang__)
        ITERATE_4({ ret.vec_.n128_u32[i] = v1.vec_.n128_u32[i] << v2; })
#else
        alignas(16) int comp[4];
        vst1q_s32(comp, v1.vec_);
        ITERATE_4({ comp[i] = comp[i] << v2; })
        ret.vec_ = vld1q_s32(comp);
#endif
        return ret;
    }

    friend force_inline simd_vec<int, 4> srai(const simd_vec<int, 4> &v1, const int v2) {
        simd_vec<int, 4> ret;
        ret.vec_ = vshlq_s32(v1.vec_, vdupq_n_s32(-v2));
        return ret;
    }

    friend force_inline bool is_equal(const simd_vec<int, 4> &v1, const simd_vec<int, 4> &v2) {
#if defined(_MSC_VER) && !defined(__clang__)
        bool res = true;
        ITERATE_4({ res &= (v1.vec_.n128_i32[i] == v2.vec_.n128_i32[i]); })
        return res;
#else
        alignas(16) int comp1[4], comp2[4];
        vst1q_s32(comp1, v1.vec_);
        vst1q_s32(comp2, v2.vec_);

        bool res = true;
        ITERATE_4({ res &= (comp1[i] == comp2[i]); })
        return res;
#endif
    }

    friend force_inline const int *value_ptr(const simd_vec<int, 4> &v1) {
        return reinterpret_cast<const int *>(&v1.vec_);
    }
    friend force_inline int *value_ptr(simd_vec<int, 4> &v1) { return reinterpret_cast<int *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

force_inline simd_vec<float, 4>::operator simd_vec<int, 4>() const {
    simd_vec<int, 4> ret;
    ret.vec_ = vcvtq_s32_f32(vec_);
    return ret;
}

} // namespace NS
} // namespace Ray

#undef VALIDATE_MASKS
