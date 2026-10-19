#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace Ray {

class float16 {
    uint16_t bits_;

  public:
    static uint16_t f32_to_f16(const float f) {
        uint32_t x;
        memcpy(&x, &f, sizeof(x));

        const uint32_t sign = (x >> 16) & 0x8000u;
        const uint32_t exp32 = (x >> 23) & 0xffu;
        const uint32_t mant32 = x & 0x7fffffu;

        if (exp32 == 0xffu) {
            // Inf → fp16 Inf, any NaN → fp16 quiet NaN
            return uint16_t(sign | 0x7c00u | (mant32 ? 0x0200u : 0u));
        }

        const int e = int(exp32) - 112; // (exp32 - 127) + 15, re-biased for fp16

        if (e >= 31) {
            return uint16_t(sign | 0x7c00u); // overflow → ±Inf
        }

        if (e <= 0) {
            if (e < -10) {
                return uint16_t(sign); // underflow → ±0
            }
            // fp16 denormal: restore the implicit leading 1 and shift into 10-bit mantissa
            const uint32_t full = mant32 | 0x800000u;
            const uint32_t shift = uint32_t(14 - e);
            const uint32_t mant = full >> shift;
            const uint32_t half = 1u << (shift - 1);
            const uint32_t lost = full & ((1u << shift) - 1u);
            const uint16_t h = uint16_t(sign | mant);
            // round-to-nearest-even
            return (lost > half || (lost == half && (mant & 1u))) ? uint16_t(h + 1u) : h;
        }

        // Normal fp16
        const uint32_t mant = mant32 >> 13;
        const uint32_t guard = (mant32 >> 12) & 1u;
        const uint32_t sticky = mant32 & 0xfffu;
        uint16_t h = uint16_t(sign | (uint32_t(e) << 10) | mant);
        // round-to-nearest-even; increment carries naturally into the exponent field
        if (guard && (sticky || (mant & 1u))) {
            ++h;
        }
        return h;
    }

    static float f16_to_f32(const uint16_t h) {
        const uint32_t sign = uint32_t(h & 0x8000u) << 16;
        const uint32_t exp16 = (h >> 10) & 0x1fu;
        const uint32_t mant16 = h & 0x3ffu;

        uint32_t x;
        if (exp16 == 0x1fu) {
            // Inf or NaN
            x = sign | 0x7f800000u | (mant16 << 13);
        } else if (exp16 == 0u) {
            if (mant16 == 0u) {
                x = sign; // ±zero
            } else {
                // fp16 denormal → fp32 normal: find leading 1, then normalize
                uint32_t m = mant16;
                int e = -14;
                while (!(m & 0x400u)) {
                    m <<= 1;
                    --e;
                }
                x = sign | (uint32_t(e + 127) << 23) | ((m & 0x3ffu) << 13);
            }
        } else {
            // Normal fp16 → normal fp32: adjust bias (127 - 15 = 112)
            x = sign | ((exp16 + 112u) << 23) | (mant16 << 13);
        }

        float result;
        memcpy(&result, &x, sizeof(result));
        return result;
    }

  public:
    float16() = default;
    float16(const float f) : bits_(f32_to_f16(f)) {}

    explicit operator float() const { return f16_to_f32(bits_); }
    explicit operator bool() const { return bits_ != 0; }

    float16 operator-() const {
        float16 ret;
        ret.bits_ = bits_ ^ uint16_t(0x8000u);
        return ret;
    }

    float16 &operator+=(const float16 rhs) {
        bits_ = f32_to_f16(f16_to_f32(bits_) + f16_to_f32(rhs.bits_));
        return *this;
    }
    float16 &operator-=(const float16 rhs) {
        bits_ = f32_to_f16(f16_to_f32(bits_) - f16_to_f32(rhs.bits_));
        return *this;
    }
    float16 &operator*=(const float16 rhs) {
        bits_ = f32_to_f16(f16_to_f32(bits_) * f16_to_f32(rhs.bits_));
        return *this;
    }
    float16 &operator/=(const float16 rhs) {
        bits_ = f32_to_f16(f16_to_f32(bits_) / f16_to_f32(rhs.bits_));
        return *this;
    }

    friend float16 operator+(const float16 a, const float16 b) { return f16_to_f32(a.bits_) + f16_to_f32(b.bits_); }
    friend float16 operator-(const float16 a, const float16 b) { return f16_to_f32(a.bits_) - f16_to_f32(b.bits_); }
    friend float16 operator*(const float16 a, const float16 b) { return f16_to_f32(a.bits_) * f16_to_f32(b.bits_); }
    friend float16 operator/(const float16 a, const float16 b) { return f16_to_f32(a.bits_) / f16_to_f32(b.bits_); }

    friend bool operator<(const float16 a, const float16 b) { return f16_to_f32(a.bits_) < f16_to_f32(b.bits_); }
    friend bool operator<=(const float16 a, const float16 b) { return f16_to_f32(a.bits_) <= f16_to_f32(b.bits_); }
    friend bool operator>(const float16 a, const float16 b) { return f16_to_f32(a.bits_) > f16_to_f32(b.bits_); }
    friend bool operator>=(const float16 a, const float16 b) { return f16_to_f32(a.bits_) >= f16_to_f32(b.bits_); }
    friend bool operator==(const float16 a, const float16 b) { return f16_to_f32(a.bits_) == f16_to_f32(b.bits_); }
    friend bool operator!=(const float16 a, const float16 b) { return f16_to_f32(a.bits_) != f16_to_f32(b.bits_); }

    friend float16 fmadd(const float16 a, const float16 b, const float16 c) {
        return f16_to_f32(a.bits_) * f16_to_f32(b.bits_) + f16_to_f32(c.bits_);
    }
    friend float16 fmsub(const float16 a, const float16 b, const float16 c) {
        return f16_to_f32(a.bits_) * f16_to_f32(b.bits_) - f16_to_f32(c.bits_);
    }

    friend float16 sqrt(const float16 v) { return sqrtf(f16_to_f32(v.bits_)); }

    friend float16 abs(const float16 v) {
        float16 ret;
        ret.bits_ = v.bits_ & uint16_t(0x7fffu);
        return ret;
    }

    friend float16 min(const float16 a, const float16 b) { return f16_to_f32(a.bits_) < f16_to_f32(b.bits_) ? a : b; }
    friend float16 max(const float16 a, const float16 b) { return f16_to_f32(a.bits_) > f16_to_f32(b.bits_) ? a : b; }
    friend float16 clamp(const float16 v, const float16 lo, const float16 hi) { return min(max(v, lo), hi); }

    friend float16 copysign(const float16 val, const float16 sign) {
        float16 ret;
        ret.bits_ = (val.bits_ & uint16_t(0x7fffu)) | (sign.bits_ & uint16_t(0x8000u));
        return ret;
    }
};

} // namespace Ray
