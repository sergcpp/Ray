#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <initializer_list>
#include <new>
#include <utility>

#include "../Span.h"

namespace Ray {
#ifndef RAY_ALIGNED_MALLOC_DEFINED
inline void *aligned_malloc(size_t size, size_t alignment) {
    while (alignment < sizeof(void *)) {
        alignment *= 2;
    }
    size_t space = size + (alignment - 1);

    void *ptr = malloc(space + sizeof(void *));
    void *original_ptr = ptr;

    char *ptr_bytes = static_cast<char *>(ptr);
    ptr_bytes += sizeof(void *);

    auto off = static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr_bytes) % alignment);
    if (off) {
        off = alignment - off;
    }
    ptr_bytes += off;

    ptr = static_cast<void *>(ptr_bytes);
    ptr_bytes -= sizeof(void *);

    memcpy(ptr_bytes, &original_ptr, sizeof(void *));

    return ptr;
}

inline void aligned_free(void *p) {
    if (p) {
        free(static_cast<void **>(p)[-1]);
    }
}
#define RAY_ALIGNED_MALLOC_DEFINED
#endif

template <typename T, int AlignmentOfT = alignof(T)> class SmallVectorImpl {
    T *begin_, *end_;
    size_t capacity_;

    // occupy one last bit of capacity to identify that we own the buffer
    static const size_t OwnerBit = (1ull << (8u * sizeof(size_t) - 1u));
    static const size_t CapacityMask = ~OwnerBit;

  protected:
    SmallVectorImpl(T *begin, T *end, const size_t capacity) : begin_(begin), end_(end), capacity_(capacity) {}

    ~SmallVectorImpl() {
        while (end_ != begin_) {
            (--end_)->~T();
        }

        if (capacity_ & OwnerBit) {
            aligned_free(begin_);
        }
    }

    void ensure_reserved(const size_t req_capacity) {
        const size_t cur_capacity = (capacity_ & CapacityMask);
        if (req_capacity <= cur_capacity) {
            return;
        }

        size_t new_capacity = cur_capacity;
        while (new_capacity < req_capacity) {
            new_capacity *= 2;
        }
        reserve(new_capacity);
    }

  public:
    using iterator = T *;
    using const_iterator = const T *;

    SmallVectorImpl(const SmallVectorImpl &rhs) = delete;
    SmallVectorImpl(SmallVectorImpl &&rhs) = delete;

    SmallVectorImpl &operator=(const SmallVectorImpl &rhs) {
        if (&rhs == this) {
            return (*this);
        }

        while (end_ != begin_) {
            (--end_)->~T();
        }

        if (capacity_ & OwnerBit) {
            aligned_free(begin_);
            capacity_ = 0;
        }

        reserve(rhs.capacity_ & CapacityMask);

        end_ = begin_ + (rhs.end_ - rhs.begin_);

        if (rhs.end_ != rhs.begin_) {
            T *src = rhs.end_ - 1;
            T *dst = end_ - 1;
            do {
                new (dst--) T(*src--);
            } while (src >= rhs.begin_);
        }

        return (*this);
    }

    SmallVectorImpl &operator=(SmallVectorImpl &&rhs) noexcept {
        if (this == &rhs) {
            return (*this);
        }

        while (end_ != begin_) {
            (--end_)->~T();
        }

        if (capacity_ & OwnerBit) {
            aligned_free(begin_);
            capacity_ = 0;
        }

        if (rhs.capacity_ & OwnerBit) {
            begin_ = std::exchange(rhs.begin_, (T *)nullptr);
            end_ = std::exchange(rhs.end_, (T *)nullptr);
            capacity_ = std::exchange(rhs.capacity_, 0);
        } else {
            reserve(rhs.capacity_ & CapacityMask);

            end_ = begin_ + (rhs.end_ - rhs.begin_);

            T *dst = end_ - 1;
            while (rhs.end_ != rhs.begin_) {
                new (dst--) T(std::move(*--rhs.end_));
                rhs.end_->~T();
            }
        }

        return (*this);
    }

    const T *cdata() const noexcept { return begin_; }
    const T *data() const noexcept { return begin_; }
    const T *begin() const noexcept { return begin_; }
    const T *end() const noexcept { return end_; }

    T *data() noexcept { return begin_; }
    iterator begin() noexcept { return begin_; }
    iterator end() noexcept { return end_; }

    const T &front() const {
        assert(begin_ != end_);
        return *begin_;
    }
    const T &back() const {
        assert(begin_ != end_);
        return *(end_ - 1);
    }

    T &front() {
        assert(begin_ != end_);
        return *begin_;
    }
    T &back() {
        assert(begin_ != end_);
        return *(end_ - 1);
    }

    bool empty() const noexcept { return end_ == begin_; }
    size_t size() const noexcept { return end_ - begin_; }
    size_t capacity() const noexcept { return (capacity_ & CapacityMask); }

    operator Span<const T>() const { return Span<const T>(data(), size()); }

    template <typename IntType> const T &operator[](const IntType i) const {
        assert(i >= 0 && begin_ + i < end_);
        return begin_[i];
    }

    template <typename IntType> T &operator[](const IntType i) {
        assert(i >= 0 && begin_ + i < end_);
        return begin_[i];
    }

    void push_back(const T &el) {
        ensure_reserved(size_t(end_ - begin_) + 1);
        new (end_++) T(el);
    }

    void push_back(T &&el) {
        ensure_reserved(size_t(end_ - begin_) + 1);
        new (end_++) T(std::move(el));
    }

    template <class... Args> T &emplace_back(Args &&...args) {
        ensure_reserved(size_t(end_ - begin_) + 1);
        new (end_++) T(std::forward<Args>(args)...);
        return *(end_ - 1);
    }

    void pop_back() {
        assert(begin_ != end_);
        (--end_)->~T();
    }

    void reserve(const size_t req_capacity) {
        const size_t cur_capacity = (capacity_ & CapacityMask);
        if (req_capacity <= cur_capacity) {
            return;
        }

        T *new_begin = (T *)aligned_malloc(req_capacity * sizeof(T), AlignmentOfT);
        T *new_end = new_begin + (end_ - begin_);

        if (end_ != begin_) {
            T *src = end_ - 1;
            T *dst = new_end - 1;
            do {
                new (dst--) T(std::move(*src));
                (src--)->~T();
            } while (src >= begin_);
        }

        if (capacity_ & OwnerBit) {
            aligned_free(begin_);
        }

        begin_ = new_begin;
        end_ = new_end;
        capacity_ = (req_capacity | OwnerBit);
    }

    void resize(const size_t req_size) {
        reserve(req_size);

        while (end_ > begin_ + req_size) {
            (--end_)->~T();
        }

        while (end_ < begin_ + req_size) {
            new (end_++) T();
        }
    }

    void resize(const size_t req_size, const T &val) {
        if (req_size == size()) {
            return;
        }
        reserve(req_size);

        while (end_ > begin_ + req_size) {
            (--end_)->~T();
        }

        while (end_ < begin_ + req_size) {
            new (end_++) T(val);
        }
    }

    void clear() {
        for (T *el = end_; el > begin_;) {
            (--el)->~T();
        }
        end_ = begin_;
    }

    iterator erase(iterator pos) {
        assert(pos >= begin_ && pos < end_);

        iterator move_dst = pos;
        iterator move_src = pos + 1;
        while (move_src != end_) {
            (*move_dst) = std::move(*move_src);

            ++move_dst;
            ++move_src;
        }
        (--end_)->~T();

        return pos;
    }

    template <class InputIt> void assign(const InputIt first, const InputIt last) {
        clear();
        for (InputIt it = first; it != last; ++it) {
            push_back(*it);
        }
    }
};

template <typename T, int AlignmentOfT = alignof(T)>
bool operator==(const SmallVectorImpl<T, AlignmentOfT> &lhs, const SmallVectorImpl<T, AlignmentOfT> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const T *lhs_it = lhs.begin(), *rhs_it = rhs.begin(); lhs_it != lhs.end(); ++lhs_it, ++rhs_it) {
        if (*lhs_it != *rhs_it) {
            return false;
        }
    }
    return true;
}

template <typename T, int AlignmentOfT = alignof(T)>
bool operator!=(const SmallVectorImpl<T, AlignmentOfT> &lhs, const SmallVectorImpl<T, AlignmentOfT> &rhs) {
    return operator==(lhs, rhs);
}

template <typename T, int N, int AlignmentOfT = alignof(T)>
class SmallVector : public SmallVectorImpl<T, AlignmentOfT> {
    alignas(AlignmentOfT) char buffer_[sizeof(T) * N];

  public:
    SmallVector() // NOLINT
        : SmallVectorImpl<T, AlignmentOfT>((T *)buffer_, (T *)buffer_, N) {}
    SmallVector(size_t initial_size, const T &val = T()) // NOLINT
        : SmallVectorImpl<T, AlignmentOfT>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::resize(initial_size, val);
    }
    SmallVector(const SmallVector<T, N, AlignmentOfT> &rhs) // NOLINT
        : SmallVectorImpl<T, AlignmentOfT>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(rhs);
    }
    SmallVector(const SmallVectorImpl<T, AlignmentOfT> &rhs) // NOLINT
        : SmallVectorImpl<T, AlignmentOfT>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(rhs);
    }
    SmallVector(SmallVector<T, N, AlignmentOfT> &&rhs) noexcept // NOLINT
        : SmallVectorImpl<T>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(std::move(rhs));
    }
    SmallVector(SmallVectorImpl<T, AlignmentOfT> &&rhs) noexcept // NOLINT
        : SmallVectorImpl<T>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(std::move(rhs));
    }

    SmallVector(std::initializer_list<T> l) : SmallVectorImpl<T, AlignmentOfT>((T *)buffer_, (T *)buffer_, N) {
        SmallVectorImpl<T, AlignmentOfT>::reserve(l.size());
        SmallVectorImpl<T, AlignmentOfT>::assign(l.begin(), l.end());
    }

    SmallVector &operator=(const SmallVectorImpl<T, AlignmentOfT> &rhs) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(rhs);
        return (*this);
    }
    SmallVector &operator=(const SmallVector<T, N, AlignmentOfT> &rhs) {
        SmallVectorImpl<T, AlignmentOfT>::operator=(rhs);
        return (*this);
    }
    SmallVector &operator=(SmallVector &&rhs) noexcept {
        SmallVectorImpl<T, AlignmentOfT>::operator=(std::move(rhs));
        return (*this);
    }

    bool is_on_heap() const { return uintptr_t(this->begin()) != uintptr_t(&buffer_[0]); }
};
} // namespace Ray