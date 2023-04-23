#pragma once

#include <cstddef>
#include <cstdint>

#include <type_traits>
#include <vector>

#include "simd/aligned_allocator.h"

namespace Ray {
template <typename T> class Span {
    T *p_data_ = nullptr;
    ptrdiff_t size_ = 0;

  public:
    Span() = default;
    Span(T *p_data, const ptrdiff_t size) : p_data_(p_data), size_(size) {}
    Span(T *p_data, const size_t size) : p_data_(p_data), size_(size) {}
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_WIN64)
    Span(T *p_data, const int size) : p_data_(p_data), size_(size) {}
#endif
    Span(T *p_begin, T *p_end) : p_data_(p_begin), size_(p_end - p_begin) {}
    template <typename Alloc>
    Span(const std::vector<typename std::remove_const<T>::type, Alloc> &v) : Span(v.data(), size_t(v.size())) {}
    template <typename Alloc>
    Span(std::vector<typename std::remove_const<T>::type, Alloc> &v) : Span(v.data(), size_t(v.size())) {}

    template <size_t N> Span(T (&arr)[N]) : p_data_(arr), size_(N) {}

    template <typename U> Span(const Span<U> &rhs) : Span(rhs.data(), rhs.size()) {}

    Span(const Span &rhs) = default;
    Span &operator=(const Span &rhs) = default;

    T *data() const { return p_data_; }
    ptrdiff_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T &operator[](const ptrdiff_t i) const { return p_data_[i]; }
    T &operator()(const ptrdiff_t i) const { return p_data_[i]; }

    using iterator = T *;
    using const_iterator = const T *;

    iterator begin() const { return p_data_; }
    iterator end() const { return p_data_ + size_; }
    const_iterator cbegin() const { return p_data_; }
    const_iterator cend() const { return p_data_ + size_; }
};
} // namespace Ray
