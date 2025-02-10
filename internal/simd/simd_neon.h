// #pragma once

#include <type_traits>

#include <arm_neon.h>

#ifndef NDEBUG
#define validate_mask(m) __assert_valid_mask(m)
#else
#define validate_mask(m) ((void)m)
#endif

namespace Ray {
namespace NS {
#if !defined(__aarch64__) && !defined(_M_ARM64)
force_inline float32x4_t vdivq_f32(float32x4_t num, float32x4_t den) {
    const float32x4_t q_inv0 = vrecpeq_f32(den);
    const float32x4_t q_step0 = vrecpsq_f32(q_inv0, den);

    const float32x4_t q_inv1 = vmulq_f32(q_step0, q_inv0);
    return vmulq_f32(num, q_inv1);
}
#define vfmaq_f32 vmlaq_f32
#endif

template <int imm> force_inline int32x4_t slli(int32x4_t a) {
    if (imm == 0) {
        return a;
    } else if (imm & ~15) {
        return vdupq_n_s32(0);
    }
    return vextq_s8(vdupq_n_s8(0), vreinterpretq_s32_s64(a), ((imm <= 0 || imm > 15) ? 0 : (16 - imm)));
}

template <typename To, typename From> To _vcast(From x) { return x; }
#if !defined(_MSC_VER) || defined(__clang__)
template <> force_inline float32x4_t _vcast(int32x4_t x) { return vreinterpretq_f32_s32(x); }
template <> force_inline float32x4_t _vcast(uint32x4_t x) { return vreinterpretq_f32_u32(x); }
template <> force_inline int32x4_t _vcast(float32x4_t x) { return vreinterpretq_s32_f32(x); }
template <> force_inline uint32x4_t _vcast(float32x4_t x) { return vreinterpretq_u32_f32(x); }
#endif

template <> class fixed_size_simd<int, 4>;
template <> class fixed_size_simd<unsigned, 4>;

template <> class fixed_size_simd<float, 4> {
    union {
        float32x4_t vec_;
        float comp_[4];
    };

    friend class fixed_size_simd<int, 4>;
    friend class fixed_size_simd<unsigned, 4>;

    force_inline fixed_size_simd(const float32x4_t vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const float f) { vec_ = vdupq_n_f32(f); }
    force_inline fixed_size_simd(const float f1, const float f2, const float f3, const float f4) {
        alignas(16) const float init[4] = {f1, f2, f3, f4};
        vec_ = vld1q_f32(init);
    }
    force_inline fixed_size_simd(const float *f) { vec_ = vld1q_f32(f); }
    force_inline fixed_size_simd(const float *f, vector_aligned_tag) {
        const float *_f = (const float *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_f32(_f);
    }

    force_inline float operator[](const int i) const { return comp_[i]; }
    force_inline float operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline float get() const { return vgetq_lane_f32(vec_, i & 3); }
    template <int i> force_inline void set(const float f) { vec_ = vsetq_lane_f32(f, vec_, i & 3); }
    force_inline void set(const int i, const float v) { comp_[i] = v; }

    force_inline fixed_size_simd<float, 4> &vectorcall operator+=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vaddq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator-=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vsubq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator*=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vmulq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator/=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vdivq_f32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<float, 4> &vectorcall operator|=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vorrq_u32(vreinterpretq_u32_f32(vec_), vreinterpretq_u32_f32(rhs.vec_));
        return *this;
    }

    force_inline fixed_size_simd<float, 4> operator-() const {
        float32x4_t m = vdupq_n_f32(-0.0f);
        int32x4_t res = veorq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(m));
        return vreinterpretq_f32_s32(res);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator<(const fixed_size_simd<float, 4> rhs) const {
        const uint32x4_t res = vcltq_f32(vec_, rhs.vec_);
        return vreinterpretq_f32_u32(res);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator<=(const fixed_size_simd<float, 4> rhs) const {
        const uint32x4_t res = vcleq_f32(vec_, rhs.vec_);
        return vreinterpretq_f32_u32(res);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator>(const fixed_size_simd<float, 4> rhs) const {
        const uint32x4_t res = vcgtq_f32(vec_, rhs.vec_);
        return vreinterpretq_f32_u32(res);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator>=(const fixed_size_simd<float, 4> rhs) const {
        const uint32x4_t res = vcgeq_f32(vec_, rhs.vec_);
        return vreinterpretq_f32_u32(res);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator&=(const fixed_size_simd<float, 4> rhs) {
        vec_ = vandq_u32(vreinterpretq_u32_f32(vec_), vreinterpretq_u32_f32(rhs.vec_));
        return *this;
    }

    force_inline fixed_size_simd<float, 4> operator~() const {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(vec_)));
    }

    force_inline operator fixed_size_simd<int, 4>() const;
    force_inline operator fixed_size_simd<unsigned, 4>() const;

    fixed_size_simd<float, 4> sqrt() const {
        // This is not precise enough :(
        // float32x4_t recipsq = vrsqrteq_f32(vec_);
        // temp.vec_ = vrecpeq_f32(recipsq);

        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        UNROLLED_FOR(i, 4, { comp[i] = sqrtf(comp[i]); })
        return fixed_size_simd<float, 4>{comp, vector_aligned};
    }

    fixed_size_simd<float, 4> log() const {
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        UNROLLED_FOR(i, 4, { comp[i] = logf(comp[i]); })
        return fixed_size_simd<float, 4>{comp, vector_aligned};
    }

    force_inline float length() const { return sqrtf(length2()); }

    float length2() const {
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);

        float temp = 0.0f;
        UNROLLED_FOR(i, 4, { temp += comp[i] * comp[i]; })
        return temp;
    }

    force_inline float hsum() const {
        alignas(16) float comp[4];
        vst1q_f32(comp, vec_);
        return comp[0] + comp[1] + comp[2] + comp[3];
    }

    force_inline void store_to(float *f) const { vst1q_f32(f, vec_); }
    force_inline void store_to(float *f, vector_aligned_tag) const {
        float *_f = (float *)__builtin_assume_aligned(f, 16);
        vst1q_f32(_f, vec_);
    }

    force_inline void vectorcall blend_to(const fixed_size_simd<float, 4> mask, const fixed_size_simd<float, 4> v1) {
        validate_mask(mask);
        int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(mask.vec_));
        int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(vec_), vreinterpretq_s32_f32(mask.vec_));
        vec_ = vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<float, 4> mask,
                                              const fixed_size_simd<float, 4> v1) {
        validate_mask(mask);
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

    friend force_inline fixed_size_simd<float, 4> vectorcall min(const fixed_size_simd<float, 4> v1,
                                                                 const fixed_size_simd<float, 4> v2) {
        return vminq_f32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall max(const fixed_size_simd<float, 4> v1,
                                                                 const fixed_size_simd<float, 4> v2) {
        return vmaxq_f32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall clamp(const fixed_size_simd<float, 4> v1,
                                                                   const fixed_size_simd<float, 4> _min,
                                                                   const fixed_size_simd<float, 4> _max) {
        return max(_min, min(v1, _max));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall saturate(const fixed_size_simd<float, 4> v1) {
        return clamp(v1, 0.0f, 1.0f);
    }

    force_inline static fixed_size_simd<float, 4> vectorcall and_not(const fixed_size_simd<float, 4> v1,
                                                                     const fixed_size_simd<float, 4> v2) {
        return vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(v2.vec_), vreinterpretq_s32_f32(v1.vec_)));
    }

    force_inline static fixed_size_simd<float, 4> vectorcall floor(const fixed_size_simd<float, 4> v1) {
        const float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(v1.vec_));
        return vsubq_f32(t, vandq_s32(vcltq_f32(v1.vec_, t), vdupq_n_f32(1.0f)));
    }

    force_inline static fixed_size_simd<float, 4> vectorcall ceil(const fixed_size_simd<float, 4> v1) {
        const float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(v1.vec_));
        return vaddq_f32(t, vandq_s32(vcgtq_f32(v1.vec_, t), vdupq_n_f32(1.0f)));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator&(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator|(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator^(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(v1.vec_), vreinterpretq_s32_f32(v2.vec_)));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator+(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vaddq_f32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator-(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vsubq_f32(v1.vec_, v2.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator==(const fixed_size_simd<float, 4> rhs) const {
        return vceqq_f32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<float, 4> vectorcall operator!=(const fixed_size_simd<float, 4> rhs) const {
        return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(vec_, rhs.vec_)));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator*(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vmulq_f32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator/(const fixed_size_simd<float, 4> v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vdivq_f32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator*(const fixed_size_simd<float, 4> v1,
                                                                       const float v2) {
        return vmulq_f32(v1.vec_, vdupq_n_f32(v2));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator/(const fixed_size_simd<float, 4> v1,
                                                                       const float v2) {
        return vdivq_f32(v1.vec_, vdupq_n_f32(v2));
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator*(const float v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vmulq_f32(vdupq_n_f32(v1), v2.vec_);
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall operator/(const float v1,
                                                                       const fixed_size_simd<float, 4> v2) {
        return vdivq_f32(vdupq_n_f32(v1), v2.vec_);
    }

    friend force_inline float vectorcall dot(const fixed_size_simd<float, 4> v1, const fixed_size_simd<float, 4> v2) {
        const float32x4_t r1 = vmulq_f32(v1.vec_, v2.vec_);
        const float32x2_t r2 = vadd_f32(vget_high_f32(r1), vget_low_f32(r1));
        return vget_lane_f32(vpadd_f32(r2, r2), 0);
    }

    friend fixed_size_simd<float, 4> vectorcall pow(const fixed_size_simd<float, 4> v1,
                                                    const fixed_size_simd<float, 4> v2) {
        alignas(16) float comp1[4], comp2[4];
        vst1q_f32(comp1, v1.vec_);
        vst1q_f32(comp2, v2.vec_);
        UNROLLED_FOR(i, 4, { comp1[i] = powf(comp1[i], comp2[i]); })
        return fixed_size_simd<float, 4>{comp1, vector_aligned};
    }

    friend fixed_size_simd<float, 4> vectorcall exp(const fixed_size_simd<float, 4> v1) {
        alignas(16) float comp1[4];
        vst1q_f32(comp1, v1.vec_);
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

    friend force_inline fixed_size_simd<float, 4>
        vectorcall fmadd(fixed_size_simd<float, 4> a, fixed_size_simd<float, 4> b, fixed_size_simd<float, 4> c) {
        fixed_size_simd<float, 4> ret;
        ret.vec_ = vfmaq_f32(c.vec_, b.vec_, a.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<float, 4>
        vectorcall fmsub(fixed_size_simd<float, 4> a, fixed_size_simd<float, 4> b, fixed_size_simd<float, 4> c) {
        fixed_size_simd<float, 4> ret;
        ret.vec_ = vfmaq_f32(vnegq_f32(c.vec_), b.vec_, a.vec_);
        return ret;
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall inclusive_scan(fixed_size_simd<float, 4> v1) {
        v1.vec_ = vaddq_f32(v1.vec_, vreinterpretq_f32_s32(slli<4>(vreinterpretq_s32_f32(v1.vec_))));
        v1.vec_ = vaddq_f32(v1.vec_, vreinterpretq_f32_s32(slli<8>(vreinterpretq_s32_f32(v1.vec_))));
        return v1;
    }

    friend force_inline fixed_size_simd<float, 4> vectorcall copysign(const fixed_size_simd<float, 4> val,
                                                                      const fixed_size_simd<float, 4> sign) {
        const uint32x4_t sign_mask =
            vandq_u32(vreinterpretq_u32_f32(sign.vec_), vreinterpretq_u32_f32(vdupq_n_f32(-0.0f)));
        const uint32x4_t abs_val = vbicq_u32(vreinterpretq_u32_f32(val.vec_), sign_mask);
        return vreinterpretq_f32_u32(vorrq_u32(abs_val, sign_mask));
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

    friend force_inline const float *value_ptr(const fixed_size_simd<float, 4> &v1) {
        return reinterpret_cast<const float *>(&v1.vec_);
    }
    friend force_inline float *value_ptr(fixed_size_simd<float, 4> &v1) { return reinterpret_cast<float *>(&v1.vec_); }

    static int size() { return 4; }
    static bool is_native() { return true; }
};

template <> class fixed_size_simd<int, 4> {
    union {
        int32x4_t vec_;
        int comp_[4];
        unsigned ucomp_[4];
    };

    friend class fixed_size_simd<float, 4>;
    friend class fixed_size_simd<unsigned, 4>;

    force_inline fixed_size_simd(const int32x4_t vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const int f) { vec_ = vdupq_n_s32(f); }
    force_inline fixed_size_simd(const int i1, const int i2, const int i3, const int i4) {
        alignas(16) const int init[4] = {i1, i2, i3, i4};
        vec_ = vld1q_s32(init);
    }
    force_inline fixed_size_simd(const int *f) { vec_ = vld1q_s32((const int32_t *)f); }
    force_inline fixed_size_simd(const int *f, vector_aligned_tag) {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_s32((const int32_t *)_f);
    }

    force_inline int operator[](const int i) const { return comp_[i]; }

    force_inline int operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline int get() const { return vgetq_lane_s32(vec_, i & 3); }
    template <int i> force_inline void set(const int f) { vec_ = vsetq_lane_s32(f, vec_, i & 3); }
    force_inline void set(const int i, const int v) { comp_[i] = v; }

    force_inline fixed_size_simd<int, 4> &vectorcall operator+=(const fixed_size_simd<int, 4> rhs) {
        vec_ = vaddq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator-=(const fixed_size_simd<int, 4> rhs) {
        vec_ = vsubq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator*=(const fixed_size_simd<int, 4> rhs) {
        vec_ = vmulq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator*=(const int rhs) {
        vec_ = vmulq_s32(vec_, vdupq_n_s32(rhs));
        return *this;
    }

    fixed_size_simd<int, 4> &vectorcall operator/=(const fixed_size_simd<int, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<int, 4> &vectorcall operator/=(const int rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs; })
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator|=(const fixed_size_simd<int, 4> rhs) {
        vec_ = vorrq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> &vectorcall operator^=(const fixed_size_simd<int, 4> rhs) {
        vec_ = veorq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> operator-() const { return vsubq_s32(vdupq_n_s32(0), vec_); }

    force_inline fixed_size_simd<int, 4> vectorcall operator==(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vceqq_s32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator!=(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(vec_, rhs.vec_)));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator<(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vcltq_s32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator<=(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vcleq_s32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator>(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vcgtq_s32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator>=(const fixed_size_simd<int, 4> rhs) const {
        return vreinterpretq_s32_u32(vcgeq_s32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<int, 4> vectorcall operator&=(const fixed_size_simd<int, 4> rhs) {
        vec_ = vandq_s32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<int, 4> operator~() const { return vmvnq_u32(vec_); }

    force_inline operator fixed_size_simd<float, 4>() const { return vcvtq_f32_s32(vec_); }

    force_inline operator fixed_size_simd<unsigned, 4>() const;

    force_inline int hsum() const {
        alignas(16) int comp[4];
        vst1q_s32(comp, vec_);
        return comp[0] + comp[1] + comp[2] + comp[3];
    }

    force_inline void store_to(int *f) const { vst1q_s32((int32_t *)f, vec_); }
    force_inline void store_to(int *f, vector_aligned_tag) const {
        const int *_f = (const int *)__builtin_assume_aligned(f, 16);
        vst1q_s32((int32_t *)_f, vec_);
    }

    force_inline void vectorcall blend_to(const fixed_size_simd<int, 4> mask, const fixed_size_simd<int, 4> v1) {
        validate_mask(mask);
        int32x4_t temp1 = vandq_s32(v1.vec_, mask.vec_);
        int32x4_t temp2 = vbicq_s32(vec_, mask.vec_);
        vec_ = vorrq_s32(temp1, temp2);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<int, 4> mask, const fixed_size_simd<int, 4> v1) {
        validate_mask(mask);
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
        UNROLLED_FOR(i, 4, { res |= comp[i] != 0; })
#endif
        return res == 0;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<int, 4> mask) const {
        int32_t res = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
        res |= vaddvq_s32(vandq_s32(vec_, mask.vec_));
#else
        alignas(16) int comp[4], mask_comp[4];
        vst1q_s32(comp, vec_);
        vst1q_s32(mask_comp, mask.vec_);
        UNROLLED_FOR(i, 4, { res |= (comp[i] & mask_comp[i]) != 0; })
#endif
        return res == 0;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<int, 4> vectorcall min(const fixed_size_simd<int, 4> v1,
                                                               const fixed_size_simd<int, 4> v2) {
        return vminq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall max(const fixed_size_simd<int, 4> v1,
                                                               const fixed_size_simd<int, 4> v2) {
        return vmaxq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall clamp(const fixed_size_simd<int, 4> v1,
                                                                 const fixed_size_simd<int, 4> _min,
                                                                 const fixed_size_simd<int, 4> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<int, 4> vectorcall and_not(const fixed_size_simd<int, 4> v1,
                                                                   const fixed_size_simd<int, 4> v2) {
        return vbicq_s32(v2.vec_, v1.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator&(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return vandq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator|(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return vorrq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator^(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return veorq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator+(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return vaddq_s32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall operator-(const fixed_size_simd<int, 4> v1,
                                                                     const fixed_size_simd<int, 4> v2) {
        return vsubq_s32(v1.vec_, v2.vec_);
    }

    friend fixed_size_simd<int, 4> vectorcall operator*(const fixed_size_simd<int, 4> v1,
                                                        const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator/(const fixed_size_simd<int, 4> v1,
                                                        const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator*(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator/(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator*(const int v1, const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator/(const int v1, const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator>>(const fixed_size_simd<int, 4> v1,
                                                         const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.ucomp_[i] = v1.ucomp_[i] >> v2.ucomp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator>>(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.ucomp_[i] = v1.ucomp_[i] >> v2; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator<<(const fixed_size_simd<int, 4> v1,
                                                         const fixed_size_simd<int, 4> v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.ucomp_[i] = v1.ucomp_[i] << v2.ucomp_[i]; })
        return ret;
    }

    friend fixed_size_simd<int, 4> vectorcall operator<<(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        UNROLLED_FOR(i, 4, { ret.ucomp_[i] = v1.ucomp_[i] << v2; })
        return ret;
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall srai(const fixed_size_simd<int, 4> v1, const int v2) {
        fixed_size_simd<int, 4> ret;
        ret.vec_ = vshlq_s32(v1.vec_, vdupq_n_s32(-v2));
        return ret;
    }

    friend bool vectorcall is_equal(const fixed_size_simd<int, 4> v1, const fixed_size_simd<int, 4> v2) {
        bool res = true;
        UNROLLED_FOR(i, 4, { res &= (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    friend force_inline fixed_size_simd<int, 4> vectorcall inclusive_scan(fixed_size_simd<int, 4> v1) {
        v1.vec_ = vaddq_s32(v1.vec_, slli<4>(v1.vec_));
        v1.vec_ = vaddq_s32(v1.vec_, slli<8>(v1.vec_));
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
        uint32x4_t vec_;
        unsigned comp_[4];
    };

    friend class fixed_size_simd<float, 4>;
    friend class fixed_size_simd<int, 4>;

    force_inline fixed_size_simd(const uint32x4_t vec) : vec_(vec) {}

  public:
    force_inline fixed_size_simd() = default;
    force_inline fixed_size_simd(const unsigned f) { vec_ = vdupq_n_u32(f); }
    force_inline fixed_size_simd(const unsigned i1, const unsigned i2, const unsigned i3, const unsigned i4) {
        alignas(16) const unsigned init[4] = {i1, i2, i3, i4};
        vec_ = vld1q_u32(init);
    }
    force_inline fixed_size_simd(const unsigned *f) { vec_ = vld1q_u32(f); }
    force_inline fixed_size_simd(const unsigned *f, vector_aligned_tag) {
        const unsigned *_f = (const unsigned *)__builtin_assume_aligned(f, 16);
        vec_ = vld1q_u32(_f);
    }

    force_inline unsigned operator[](const int i) const { return comp_[i]; }
    force_inline unsigned operator[](const long i) const { return operator[](int(i)); }

    template <int i> force_inline unsigned get() const { return vgetq_lane_u32(vec_, i & 3); }
    template <int i> force_inline void set(const unsigned f) { vec_ = vsetq_lane_u32(f, vec_, i & 3); }
    force_inline void set(const int i, const unsigned v) { comp_[i] = v; }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator+=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = vaddq_u32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator-=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = vsubq_u32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator*=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = vmulq_u32(vec_, rhs.vec_);
        return *this;
    }

    fixed_size_simd<unsigned, 4> &vectorcall operator/=(const fixed_size_simd<unsigned, 4> rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs.comp_[i]; })
        return *this;
    }

    fixed_size_simd<unsigned, 4> &vectorcall operator/=(const unsigned rhs) {
        UNROLLED_FOR(i, 4, { comp_[i] /= rhs; })
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator|=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = vorrq_u32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> &vectorcall operator^=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = veorq_u32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator==(const fixed_size_simd<unsigned, 4> rhs) const {
        return vceqq_u32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator!=(const fixed_size_simd<unsigned, 4> rhs) const {
        return vmvnq_u32(vceqq_u32(vec_, rhs.vec_));
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator<(const fixed_size_simd<unsigned, 4> rhs) const {
        return vcltq_u32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator<=(const fixed_size_simd<unsigned, 4> rhs) const {
        return vcleq_u32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator>(const fixed_size_simd<unsigned, 4> rhs) const {
        return vcgtq_u32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator>=(const fixed_size_simd<unsigned, 4> rhs) const {
        return vcgeq_u32(vec_, rhs.vec_);
    }

    force_inline fixed_size_simd<unsigned, 4> vectorcall operator&=(const fixed_size_simd<unsigned, 4> rhs) {
        vec_ = vandq_u32(vec_, rhs.vec_);
        return *this;
    }

    force_inline fixed_size_simd<unsigned, 4> operator~() const { return vmvnq_u32(vec_); }

    force_inline operator fixed_size_simd<float, 4>() const { return vcvtq_f32_u32(vec_); }

    force_inline operator fixed_size_simd<int, 4>() const { return vreinterpretq_s32_u32(vec_); }

    force_inline unsigned hsum() const {
        alignas(16) unsigned comp[4];
        vst1q_u32(comp, vec_);
        return comp[0] + comp[1] + comp[2] + comp[3];
    }

    force_inline void store_to(unsigned *f) const { vst1q_u32(f, vec_); }
    force_inline void store_to(unsigned *f, vector_aligned_tag) const {
        unsigned *_f = (unsigned *)__builtin_assume_aligned(f, 16);
        vst1q_u32(_f, vec_);
    }

    force_inline void vectorcall blend_to(const fixed_size_simd<unsigned, 4> mask,
                                          const fixed_size_simd<unsigned, 4> v1) {
        validate_mask(mask);
        uint32x4_t temp1 = vandq_u32(v1.vec_, mask.vec_);
        uint32x4_t temp2 = vbicq_u32(vec_, mask.vec_);
        vec_ = vorrq_u32(temp1, temp2);
    }

    force_inline void vectorcall blend_inv_to(const fixed_size_simd<unsigned, 4> mask,
                                              const fixed_size_simd<unsigned, 4> v1) {
        validate_mask(mask);
        uint32x4_t temp1 = vandq_u32(vec_, mask.vec_);
        uint32x4_t temp2 = vbicq_u32(v1.vec_, mask.vec_);
        vec_ = vorrq_u32(temp1, temp2);
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
        UNROLLED_FOR(i, 4, { res |= comp[i] != 0; })
#endif
        return res == 0;
    }

    force_inline bool vectorcall all_zeros(const fixed_size_simd<unsigned, 4> mask) const {
        int32_t res = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
        res |= vaddvq_u32(vandq_u32(vec_, mask.vec_));
#else
        alignas(16) unsigned comp[4], mask_comp[4];
        vst1q_u32(comp, vec_);
        vst1q_u32(mask_comp, mask.vec_);
        UNROLLED_FOR(i, 4, { res |= (comp[i] & mask_comp[i]) != 0; })
#endif
        return res == 0;
    }

    force_inline bool not_all_zeros() const { return !all_zeros(); }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall min(const fixed_size_simd<unsigned, 4> v1,
                                                                    const fixed_size_simd<unsigned, 4> v2) {
        return vminq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall max(const fixed_size_simd<unsigned, 4> v1,
                                                                    const fixed_size_simd<unsigned, 4> v2) {
        return vmaxq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall clamp(const fixed_size_simd<unsigned, 4> v1,
                                                                      const fixed_size_simd<unsigned, 4> _min,
                                                                      const fixed_size_simd<unsigned, 4> _max) {
        return max(_min, min(v1, _max));
    }

    force_inline static fixed_size_simd<unsigned, 4> vectorcall and_not(const fixed_size_simd<unsigned, 4> v1,
                                                                        const fixed_size_simd<unsigned, 4> v2) {
        return vbicq_u32(v2.vec_, v1.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator&(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return vandq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator|(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return vorrq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator^(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return veorq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator+(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return vaddq_u32(v1.vec_, v2.vec_);
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall operator-(const fixed_size_simd<unsigned, 4> v1,
                                                                          const fixed_size_simd<unsigned, 4> v2) {
        return vsubq_u32(v1.vec_, v2.vec_);
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

    friend fixed_size_simd<unsigned, 4> vectorcall operator*(const fixed_size_simd<unsigned, 4> v1, const unsigned v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator/(const fixed_size_simd<unsigned, 4> v1, const unsigned v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] / v2; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator*(const unsigned v1, const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator/(const unsigned v1, const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator>>(const fixed_size_simd<unsigned, 4> v1,
                                                              const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator>>(const fixed_size_simd<unsigned, 4> v1,
                                                              const unsigned v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] >> v2; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator<<(const fixed_size_simd<unsigned, 4> v1,
                                                              const fixed_size_simd<unsigned, 4> v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
        return ret;
    }

    friend fixed_size_simd<unsigned, 4> vectorcall operator<<(const fixed_size_simd<unsigned, 4> v1,
                                                              const unsigned v2) {
        fixed_size_simd<unsigned, 4> ret;
        UNROLLED_FOR(i, 4, { ret.comp_[i] = v1.comp_[i] << v2; })
        return ret;
    }

    friend bool vectorcall is_equal(const fixed_size_simd<unsigned, 4> v1, const fixed_size_simd<unsigned, 4> v2) {
        bool res = true;
        UNROLLED_FOR(i, 4, { res &= (v1.comp_[i] == v2.comp_[i]); })
        return res;
    }

    friend force_inline fixed_size_simd<unsigned, 4> vectorcall inclusive_scan(fixed_size_simd<unsigned, 4> v1) {
        v1.vec_ = vaddq_s32(v1.vec_, slli<4>(v1.vec_));
        v1.vec_ = vaddq_s32(v1.vec_, slli<8>(v1.vec_));
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
            const int val = mask.get<i>();
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

force_inline fixed_size_simd<float, 4>::operator fixed_size_simd<int, 4>() const { return vcvtq_s32_f32(vec_); }

force_inline fixed_size_simd<float, 4>::operator fixed_size_simd<unsigned, 4>() const { return vcvtq_u32_f32(vec_); }

force_inline fixed_size_simd<int, 4>::operator fixed_size_simd<unsigned, 4>() const {
    return vreinterpretq_u32_s32(vec_);
}

template <typename U>
force_inline fixed_size_simd<float, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                         const fixed_size_simd<float, 4> vec1,
                                                         const fixed_size_simd<float, 4> vec2) {
    validate_mask(mask);
    const int32x4_t temp1 = vandq_s32(vreinterpretq_s32_f32(vec1.vec_), _vcast<int32x4_t>(mask.vec_));
    const int32x4_t temp2 = vbicq_s32(vreinterpretq_s32_f32(vec2.vec_), _vcast<int32x4_t>(mask.vec_));
    return vreinterpretq_f32_s32(vorrq_s32(temp1, temp2));
}

template <typename U>
force_inline fixed_size_simd<int, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                       const fixed_size_simd<int, 4> vec1,
                                                       const fixed_size_simd<int, 4> vec2) {
    validate_mask(mask);
    const int32x4_t temp1 = vandq_s32(vec1.vec_, _vcast<int32x4_t>(mask.vec_));
    const int32x4_t temp2 = vbicq_s32(vec2.vec_, _vcast<int32x4_t>(mask.vec_));
    return vorrq_s32(temp1, temp2);
}

template <typename U>
force_inline fixed_size_simd<unsigned, 4> vectorcall select(const fixed_size_simd<U, 4> mask,
                                                            const fixed_size_simd<unsigned, 4> vec1,
                                                            const fixed_size_simd<unsigned, 4> vec2) {
    validate_mask(mask);
    const uint32x4_t temp1 = vandq_u32(vec1.vec_, _vcast<uint32x4_t>(mask.vec_));
    const uint32x4_t temp2 = vbicq_u32(vec2.vec_, _vcast<uint32x4_t>(mask.vec_));
    return vorrq_u32(temp1, temp2);
}

} // namespace NS
} // namespace Ray

#undef validate_mask
