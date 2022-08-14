#pragma once

#include <limits>
#undef min
#undef max

namespace Ray {
namespace Vk {
template <typename T, int FpBits> class Fixed {
    T value_;

  public:
    Fixed() : value_(0) {}
    explicit Fixed(const T v) : value_(v) {}
    explicit Fixed(const float v) : value_(T(v * One)) {}

    T value() const { return value_; }
    void set_value(T v) { value_ = v; }

    float to_float() const { return float(value_) / One; }
    void from_float(const float v) { value_ = T(v * One); }

    friend bool operator==(const Fixed<T, FpBits> &lhs, const Fixed<T, FpBits> &rhs) {
        return lhs.value_ == rhs.value_;
    }

    static Fixed lowest() { return Fixed<T, FpBits>{std::numeric_limits<T>::lowest()}; }

    static Fixed max() { return Fixed<T, FpBits>{std::numeric_limits<T>::max()}; }

    static const T One = T(1) << FpBits;
};
} // namespace Vk
} // namespace Ray