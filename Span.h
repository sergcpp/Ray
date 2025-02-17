#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <array>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

namespace Ray {
template <typename T> struct remove_all_const : std::remove_const<T> {};

template <typename T> struct remove_all_const<T *> {
    typedef typename remove_all_const<T>::type *type;
};

template <typename T> struct remove_all_const<T *const> {
    typedef typename remove_all_const<T>::type *type;
};

template <typename T> class Span {
    T *p_data_ = nullptr;
    ptrdiff_t size_ = 0;

  public:
    Span() = default;
    Span(T *p_data, const ptrdiff_t size) : p_data_(p_data), size_(size) {}
    Span(T *p_data, const size_t size) : p_data_(p_data), size_(size) {}
#if INTPTR_MAX == INT64_MAX
    Span(T *p_data, const int size) : p_data_(p_data), size_(size) {}
    Span(T *p_data, const uint32_t size) : p_data_(p_data), size_(size) {}
#endif
    Span(T *p_begin, T *p_end) : p_data_(p_begin), size_(p_end - p_begin) {}

    template <size_t N>
    Span(const std::array<typename remove_all_const<T>::type, N> &arr) : Span(arr.data(), arr.size()) {}
    template <size_t N>
    Span(const std::array<const typename remove_all_const<T>::type, N> &arr) : Span(arr.data(), arr.size()) {}
    template <size_t N> Span(std::array<typename std::remove_cv<T>::type, N> &arr) : Span(arr.data(), arr.size()) {}

    template <typename Alloc>
    Span(const std::vector<typename remove_all_const<T>::type, Alloc> &v) : Span(v.data(), v.size()) {}
    template <typename Alloc>
    Span(const std::vector<const typename remove_all_const<T>::type, Alloc> &v) : Span(v.data(), v.size()) {}
    template <typename Alloc>
    Span(std::vector<typename std::remove_cv<T>::type, Alloc> &v) : Span(v.data(), v.size()) {}

    template <size_t N> Span(T (&arr)[N]) : Span(arr, N) {}

    template <typename U = typename remove_all_const<T>::type,
              typename = typename std::enable_if<!std::is_same<T, U>::value>::type>
    Span(const Span<U> &rhs) : Span(rhs.data(), rhs.size()) {}

    Span(const Span &rhs) = default;
    Span &operator=(const Span &rhs) = default;

    force_inline T *data() const { return p_data_; }
    force_inline ptrdiff_t size() const { return size_; }
    force_inline ptrdiff_t size_bytes() const { return size_ * sizeof(T); }
    force_inline bool empty() const { return size_ == 0; }

    force_inline T &front() const { return p_data_[0]; }
    force_inline T &back() const { return p_data_[size_ - 1]; }

    force_inline T &operator[](const ptrdiff_t i) const {
        assert(i >= 0 && i < size_);
        return p_data_[i];
    }
    force_inline T &operator()(const ptrdiff_t i) const {
        assert(i >= 0 && i < size_);
        return p_data_[i];
    }

    using iterator = T *;
    using const_iterator = const T *;

    force_inline iterator begin() const { return p_data_; }
    force_inline iterator end() const { return p_data_ + size_; }
    force_inline const_iterator cbegin() const { return p_data_; }
    force_inline const_iterator cend() const { return p_data_ + size_; }

    template <typename IterType> class reverse_iterator_t {
        IterType iter_;

      public:
        explicit reverse_iterator_t(IterType iter) : iter_(iter) {}

        template <typename U> reverse_iterator_t(reverse_iterator_t<U> &rhs) : iter_(rhs.base()) {}

        template <typename U> operator reverse_iterator_t<U>() { return reverse_iterator_t<U>(iter_); }

        IterType base() const { return iter_; }

        const T &operator*() const { return *(iter_ - 1); }
        const T *operator->() const { return &*(iter_ - 1); }

        reverse_iterator_t &operator++() {
            --iter_;
            return *this;
        }

        reverse_iterator_t operator++(int) {
            reverse_iterator_t ret(*this);
            --iter_;
            return ret;
        }

        reverse_iterator_t &operator--() {
            ++iter_;
            return *this;
        }

        reverse_iterator_t operator--(int) {
            reverse_iterator_t ret(*this);
            ++iter_;
            return ret;
        }

        reverse_iterator_t operator+(ptrdiff_t n) const { return reverse_iterator_t(iter_ - n); }
        reverse_iterator_t &operator+=(ptrdiff_t n) const {
            iter_ -= n;
            return *this;
        }

        reverse_iterator_t operator-(ptrdiff_t n) const { return reverse_iterator_t(iter_ + n); }
        reverse_iterator_t &operator-=(ptrdiff_t n) const {
            iter_ += n;
            return *this;
        }

        T &operator[](ptrdiff_t n) const { return iter_[-n - 1]; }
    };

    using reverse_iterator = reverse_iterator_t<T *>;
    using const_reverse_iterator = reverse_iterator_t<const T *>;

    force_inline reverse_iterator rbegin() const { return reverse_iterator(p_data_ + size_); }
    force_inline reverse_iterator rend() const { return reverse_iterator(p_data_); }
    force_inline const_reverse_iterator crbegin() const { return const_reverse_iterator(p_data_ + size_); }
    force_inline const_reverse_iterator crend() const { return const_reverse_iterator(p_data_); }

    Span<T> first(const size_t n) const { return Span<T>{p_data_, n}; }
    Span<T> last(const size_t n) const { return Span<T>{p_data_ + size_ - n, n}; }
    Span<T> subspan(const size_t offset, const size_t count = SIZE_MAX) const {
        return Span<T>{p_data_ + offset, count != SIZE_MAX ? count : (size_ - offset)};
    }
};

template <typename Iter1, typename Iter2>
bool lexicographical_compare(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2) {
    for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
        if (*first1 < *first2) {
            return true;
        }
        if (*first2 < *first1) {
            return false;
        }
    }
    return first1 == last1 && first2 != last2;
}

template <typename T, typename U> bool operator==(Span<T> lhs, Span<U> rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}
template <typename T, typename U> bool operator!=(Span<T> lhs, Span<U> rhs) {
    return lhs.size() != rhs.size() || !std::equal(lhs.begin(), lhs.end(), rhs.begin());
}
template <typename T, typename U> bool operator<(Span<T> lhs, Span<U> rhs) {
    return lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}
template <typename T, typename U> bool operator<=(Span<T> lhs, Span<U> rhs) { return !operator<(rhs, lhs); }
template <typename T, typename U> bool operator>(Span<T> lhs, Span<U> rhs) { return operator<(rhs, lhs); }
template <typename T, typename U> bool operator>=(Span<T> lhs, Span<U> rhs) { return !operator<(lhs, rhs); }
} // namespace Ray

#undef force_inline