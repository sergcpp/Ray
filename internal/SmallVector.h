#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <initializer_list>
#include <new>
#include <utility>

#include "../Span.h"
#include "simd/aligned_allocator.h"

namespace Ray {
template <typename T, typename Allocator = aligned_allocator<T, alignof(T)>> class SmallVectorImpl : Allocator {
    T *begin_;
    uint32_t size_, capacity_;

    // occupy one last bit of capacity to identify that we own the buffer
    static const uint32_t OwnerBit = (1u << (8u * sizeof(uint32_t) - 1u));
    static const uint32_t CapacityMask = ~OwnerBit;

  protected:
    SmallVectorImpl(T *begin, T *end, const uint32_t capacity, const Allocator &alloc)
        : Allocator(alloc), begin_(begin), size_(uint32_t(end - begin)), capacity_(capacity) {}

    ~SmallVectorImpl() {
        while (size_) {
            (begin_ + --size_)->~T();
        }
        if (capacity_ & OwnerBit) {
            this->deallocate(begin_, (capacity_ & CapacityMask));
        }
    }

    void ensure_reserved(const uint32_t req_capacity) {
        const uint32_t cur_capacity = (capacity_ & CapacityMask);
        if (req_capacity <= cur_capacity) {
            return;
        }

        uint32_t new_capacity = cur_capacity;
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

        while (size_) {
            (begin_ + --size_)->~T();
        }

        if (capacity_ & OwnerBit) {
            this->deallocate(begin_, (capacity_ & CapacityMask));
            capacity_ = 0;
        }

        reserve(rhs.size_);

        while (size_ < rhs.size_) {
            new (begin_ + size_) T(*(rhs.begin_ + size_));
            ++size_;
        }

        return (*this);
    }

    SmallVectorImpl &operator=(SmallVectorImpl &&rhs) noexcept {
        if (this == &rhs) {
            return (*this);
        }

        while (size_) {
            (begin_ + --size_)->~T();
        }

        if (capacity_ & OwnerBit) {
            this->deallocate(begin_, (capacity_ & CapacityMask));
            capacity_ = 0;
        }

        if (rhs.capacity_ & OwnerBit) {
            begin_ = std::exchange(rhs.begin_, (T *)nullptr);
            size_ = std::exchange(rhs.size_, 0);
            capacity_ = std::exchange(rhs.capacity_, 0);
        } else {
            reserve(rhs.size_);

            while (rhs.size_) {
                new (begin_ + size_) T(std::move(*(rhs.begin_ + size_)));
                ++size_;
                --rhs.size_;
            }
        }

        return (*this);
    }

    operator Span<const T>() const { return Span<const T>(data(), size()); }

    bool operator==(const SmallVectorImpl &rhs) const {
        if (size_ != rhs.size_) {
            return false;
        }
        bool eq = true;
        for (uint32_t i = 0; i < size_ && eq; ++i) {
            eq &= begin_[i] == rhs.begin_[i];
        }
        return eq;
    }
    bool operator!=(const SmallVectorImpl &rhs) const {
        if (size_ != rhs.size_) {
            return true;
        }
        bool neq = false;
        for (uint32_t i = 0; i < size_ && !neq; ++i) {
            neq |= begin_[i] != rhs.begin_[i];
        }
        return neq;
    }
    bool operator<(const SmallVectorImpl &rhs) const {
        return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
    }
    bool operator<=(const SmallVectorImpl &rhs) const {
        return !std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
    }
    bool operator>(const SmallVectorImpl &rhs) const {
        return std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
    }
    bool operator>=(const SmallVectorImpl &rhs) const {
        return !std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
    }

    const T *cdata() const noexcept { return begin_; }
    const T *data() const noexcept { return begin_; }
    const_iterator begin() const noexcept { return begin_; }
    const_iterator end() const noexcept { return begin_ + size_; }
    const_iterator cbegin() const noexcept { return begin_; }
    const_iterator cend() const noexcept { return begin_ + size_; }

    T *data() noexcept { return begin_; }
    iterator begin() noexcept { return begin_; }
    iterator end() noexcept { return begin_ + size_; }

    const Allocator &alloc() const { return (*this); }

    const T &front() const {
        assert(size_ != 0);
        return *begin_;
    }
    const T &back() const {
        assert(size_ != 0);
        return *(begin_ + size_ - 1);
    }

    T &front() {
        assert(size_ != 0);
        return *begin_;
    }
    T &back() {
        assert(size_ != 0);
        return *(begin_ + size_ - 1);
    }

    bool empty() const noexcept { return size_ == 0; }
    uint32_t size() const noexcept { return size_; }
    uint32_t capacity() const noexcept { return (capacity_ & CapacityMask); }

    template <typename IntType> const T &operator[](const IntType i) const {
        assert(i >= 0 && i < IntType(size_));
        return begin_[i];
    }

    template <typename IntType> T &operator[](const IntType i) {
        assert(i >= 0 && i < IntType(size_));
        return begin_[i];
    }

    void push_back(const T &el) {
        ensure_reserved(size_ + 1);
        new (begin_ + size_++) T(el);
    }

    void push_back(T &&el) {
        ensure_reserved(size_ + 1);
        new (begin_ + size_++) T(std::move(el));
    }

    template <class... Args> T &emplace_back(Args &&...args) {
        ensure_reserved(size_ + 1);
        new (begin_ + size_++) T(std::forward<Args>(args)...);
        return *(begin_ + size_ - 1);
    }

    void pop_back() {
        assert(size_ != 0);
        (begin_ + --size_)->~T();
    }

    void reserve(const uint32_t req_capacity) {
        const uint32_t cur_capacity = (capacity_ & CapacityMask);
        if (req_capacity <= cur_capacity) {
            return;
        }

        T *new_begin = this->allocate(req_capacity);
        T *new_end = new_begin + size_;

        if (size_) {
            T *src = begin_ + size_ - 1;
            T *dst = new_end - 1;
            do {
                new (dst--) T(std::move(*src));
                (src--)->~T();
            } while (src >= begin_);
        }

        if (capacity_ & OwnerBit) {
            this->deallocate(begin_, (capacity_ & CapacityMask));
        }

        begin_ = new_begin;
        capacity_ = (req_capacity | OwnerBit);
    }

    void resize(const uint32_t req_size) {
        reserve(req_size);

        while (size_ > req_size) {
            (begin_ + --size_)->~T();
        }

        while (size_ < req_size) {
            new (begin_ + size_++) T();
        }
    }

    void resize(const uint32_t req_size, const T &val) {
        if (req_size == size()) {
            return;
        }
        reserve(req_size);

        while (size_ > req_size) {
            (begin_ + --size_)->~T();
        }

        while (size_ < req_size) {
            new (begin_ + size_++) T(val);
        }
    }

    void clear() {
        for (T *el = begin_ + size_; el > begin_;) {
            (--el)->~T();
        }
        size_ = 0;
    }

    iterator insert(iterator pos, const T &value) {
        assert(pos >= begin_ && pos <= begin_ + size_);

        const uint32_t off = uint32_t(pos - begin_);
        ensure_reserved(size_ + 1);
        pos = begin_ + off;

        iterator move_src = begin_ + size_ - 1, move_dst = move_src + 1;
        while (move_src != pos - 1) {
            new (move_dst) T(std::move(*move_src));
            move_src->~T();

            --move_dst;
            --move_src;
        }

        new (pos) T(value);
        ++size_;

        return pos;
    }

    iterator insert(iterator pos, iterator beg, iterator end) {
        assert(pos >= begin_ && pos <= begin_ + size_);

        const uint32_t count = uint32_t(end - beg);
        const uint32_t off = uint32_t(pos - begin_);
        ensure_reserved(size_ + count);
        pos = begin_ + off;

        iterator move_src = begin_ + size_ - 1, move_dst = move_src + count;
        while (move_src != pos - 1) {
            new (move_dst) T(std::move(*move_src));
            move_src->~T();

            --move_dst;
            --move_src;
        }

        move_dst = pos;
        while (move_dst != pos + count) {
            new (move_dst++) T(*beg++);
        }
        size_ += count;

        return pos;
    }

    iterator erase(iterator pos) {
        assert(pos >= begin_ && pos < begin_ + size_);

        iterator move_dst = pos, move_src = pos + 1;
        while (move_src != begin_ + size_) {
            (*move_dst) = std::move(*move_src);

            ++move_dst;
            ++move_src;
        }
        (begin_ + --size_)->~T();

        return pos;
    }

    iterator erase(iterator first, iterator last) {
        assert(first >= begin_ && first <= begin_ + size_);

        iterator move_dst = first, move_src = last;
        while (move_src != begin_ + size_) {
            (*move_dst) = std::move(*move_src);

            ++move_dst;
            ++move_src;
        }
        while (begin_ + size_ != move_dst) {
            (begin_ + --size_)->~T();
        }

        return move_dst;
    }

    void assign(const uint32_t count, const T &val) {
        clear();
        reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            push_back(val);
        }
    }

    template <class InputIt> void assign(const InputIt first, const InputIt last) {
        clear();
        reserve(uint32_t(last - first));
        for (InputIt it = first; it != last; ++it) {
            push_back(*it);
        }
    }
};

template <typename T, int N, int AlignmentOfT = alignof(T), typename Allocator = aligned_allocator<T, AlignmentOfT>>
class SmallVector : public SmallVectorImpl<T, Allocator> {
    alignas(AlignmentOfT) char buffer_[sizeof(T) * N];

  public:
    SmallVector(const Allocator &alloc = Allocator()) // NOLINT
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, alloc) {}
    explicit SmallVector(const uint32_t size, const T &val = T(), const Allocator &alloc = Allocator()) // NOLINT
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, alloc) {
        SmallVectorImpl<T, Allocator>::resize(size, val);
    }
    SmallVector(const SmallVector<T, N, AlignmentOfT, Allocator> &rhs) // NOLINT
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, rhs.alloc()) {
        SmallVectorImpl<T, Allocator>::operator=(rhs);
    }
    SmallVector(const SmallVectorImpl<T, Allocator> &rhs) // NOLINT
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, rhs.alloc()) {
        SmallVectorImpl<T, Allocator>::operator=(rhs);
    }
    SmallVector(SmallVector<T, N, AlignmentOfT> &&rhs) noexcept // NOLINT
        : SmallVectorImpl<T>((T *)buffer_, (T *)buffer_, N, rhs.alloc()) {
        SmallVectorImpl<T, Allocator>::operator=(std::move(rhs));
    }
    SmallVector(SmallVectorImpl<T, Allocator> &&rhs) noexcept // NOLINT
        : SmallVectorImpl<T>((T *)buffer_, (T *)buffer_, N, rhs.alloc()) {
        SmallVectorImpl<T, Allocator>::operator=(std::move(rhs));
    }

    template <class InputIt>
    SmallVector(InputIt beg, InputIt end, const Allocator &alloc = Allocator())
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, alloc) {
        SmallVectorImpl<T, Allocator>::assign(beg, end);
    }

    SmallVector(std::initializer_list<T> l, const Allocator &alloc = Allocator())
        : SmallVectorImpl<T, Allocator>((T *)buffer_, (T *)buffer_, N, alloc) {
        SmallVectorImpl<T, Allocator>::assign(l.begin(), l.end());
    }

    SmallVector &operator=(const SmallVectorImpl<T, Allocator> &rhs) {
        SmallVectorImpl<T, Allocator>::operator=(rhs);
        return (*this);
    }
    SmallVector &operator=(const SmallVector<T, N, AlignmentOfT, Allocator> &rhs) {
        SmallVectorImpl<T, Allocator>::operator=(rhs);
        return (*this);
    }
    SmallVector &operator=(SmallVector &&rhs) noexcept {
        SmallVectorImpl<T, Allocator>::operator=(std::move(rhs));
        return (*this);
    }

    bool is_on_heap() const { return uintptr_t(this->begin()) != uintptr_t(&buffer_[0]); }
};
} // namespace Ray