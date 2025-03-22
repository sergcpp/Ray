#pragma once

#include <cassert>
#include <limits>
#include <type_traits>

namespace Ray {
template <class enum_type, typename = typename std::enable_if<std::is_enum<enum_type>::value>::type> class Bitmask {
    using underlying_type = typename std::underlying_type<enum_type>::type;

    static constexpr underlying_type to_mask(const enum_type e) {
      assert(1ull << static_cast<underlying_type>(e) <= std::numeric_limits<underlying_type>::max());
      return 1 << static_cast<underlying_type>(e);
    }

  public:
    constexpr Bitmask() : mask_(0) {}
    constexpr Bitmask(const enum_type e) : mask_(to_mask(e)) {}
    explicit constexpr Bitmask(const underlying_type mask) : mask_(mask) {}

    Bitmask(const Bitmask &rhs) = default;
    Bitmask(Bitmask &&rhs) = default;

    Bitmask &operator=(const Bitmask &rhs) = default;
    Bitmask &operator=(Bitmask &&rhs) = default;

    Bitmask operator|(const Bitmask rhs) const { return Bitmask(mask_ | rhs.mask_); }
    Bitmask operator|=(const Bitmask rhs) { return (*this) = Bitmask(mask_ | rhs.mask_); }

    Bitmask operator&(const Bitmask rhs) const { return Bitmask(mask_ & rhs.mask_); }
    Bitmask operator&=(const Bitmask rhs) { return (*this) = Bitmask(mask_ & rhs.mask_); }

    Bitmask operator~() const { return Bitmask(~mask_); }

    bool operator==(const enum_type rhs) const { return mask_ == to_mask(rhs); }
    bool operator==(const Bitmask rhs) const { return mask_ == rhs.mask_; }

    bool operator!=(const enum_type rhs) const { return mask_ != to_mask(rhs); }
    bool operator!=(const Bitmask rhs) const { return mask_ != rhs.mask_; }

    operator bool() const { return mask_ != 0; }
    explicit constexpr operator underlying_type() const { return mask_; }

  private:
    underlying_type mask_;
};

} // namespace Ray