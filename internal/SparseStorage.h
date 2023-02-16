#pragma once

#include <cassert>
#include <climits>

#include "Bitmap.h"
#include "simd/aligned_allocator.h"

namespace Ray {
namespace Ref {
template <typename T> class SparseStorage {
    Bitmap bits_;
    T *data_ = nullptr;
    uint32_t capacity_ = 0, size_ = 0;
    // TODO: add next_free_ to speedup allocation

    static const uint32_t InitialNonZeroCapacity = 8;

  public:
    explicit SparseStorage(uint32_t initial_capacity = 0) : bits_(int(initial_capacity)) {
        if (initial_capacity) {
            reserve(initial_capacity);
        }
    }

    ~SparseStorage() { aligned_free(data_); }

    force_inline uint32_t size() const { return size_; }
    force_inline uint32_t capacity() const { return capacity_; }

    force_inline bool empty() const { return size_ == 0; }

    force_inline T *data() { return data_; }
    force_inline const T *data() const { return data_; }

    force_inline bool exists(const uint32_t index) const { return bits_.IsSet(index); }

    void reserve(uint32_t new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

        bits_.Resize(int(new_capacity));

        T *new_data = (T *)aligned_malloc(new_capacity * sizeof(T), alignof(T));

        // move old data
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (bits_.IsSet(i)) {
                new (&new_data[i]) T(std::move(data_[i]));
                data_[i].~T();
            }
        }

        aligned_free(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

    template <class... Args> uint32_t emplace(Args &&...args) {
        if (size_ + 1 > capacity_) {
            reserve(std::max(capacity_ * 2, InitialNonZeroCapacity));
        }

        const uint32_t index = bits_.Alloc_FirstFit(1);

        T *el = data_ + index;
        new (el) T(std::forward<Args>(args)...);

        ++size_;
        return index;
    }

    uint32_t push(const T &el) {
        if (size_ + 1 > capacity_) {
            reserve(std::max(capacity_ * 2, InitialNonZeroCapacity));
        }

        const uint32_t index = bits_.Alloc_FirstFit(1);
        new (&data_[index]) T(el);

        ++size_;
        return index;
    }

    void clear() {
        for (uint32_t i = 0; i < capacity_ && size_; ++i) {
            if (bits_.IsSet(i)) {
                erase(i);
            }
        }
    }

    void erase(const uint32_t index) {
        assert(bits_.IsSet(index) && "Invalid index!");

        data_[index].~T();
        bits_.Free(int(index), 1);

        --size_;
    }

    force_inline T &at(const uint32_t index) {
        assert(bits_.IsSet(index) && "Invalid index!");
        return data_[index];
    }

    force_inline const T &at(const uint32_t index) const {
        assert(bits_.IsSet(index) && "Invalid index!");
        return data_[index];
    }

    force_inline T &operator[](const uint32_t index) {
        assert(bits_.IsSet(index) && "Invalid index!");
        return data_[index];
    }

    force_inline const T &operator[](const uint32_t index) const {
        assert(bits_.IsSet(index) && "Invalid index!");
        return data_[index];
    }

    class SparseStorageIterator : public std::iterator<std::forward_iterator_tag, T> {
        friend class SparseStorage<T>;

        SparseStorage<T> *container_;
        uint32_t index_;

        SparseStorageIterator(SparseStorage<T> *container, uint32_t index) : container_(container), index_(index) {}

      public:
        T &operator*() { return container_->at(index_); }
        T *operator->() { return &container_->at(index_); }
        SparseStorageIterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }
        SparseStorageIterator operator++(int) {
            SparseStorageIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const SparseStorageIterator &rhs) const { return index_ < rhs.index_; }
        bool operator<=(const SparseStorageIterator &rhs) const { return index_ <= rhs.index_; }
        bool operator>(const SparseStorageIterator &rhs) const { return index_ > rhs.index_; }
        bool operator>=(const SparseStorageIterator &rhs) const { return index_ >= rhs.index_; }
        bool operator==(const SparseStorageIterator &rhs) const { return index_ == rhs.index_; }
        bool operator!=(const SparseStorageIterator &rhs) const { return index_ != rhs.index_; }
    };

    class SparseStorageConstIterator : public std::iterator<std::forward_iterator_tag, T> {
        friend class SparseStorage<T>;

        const SparseStorage<T> *container_;
        uint32_t index_;

        SparseStorageConstIterator(const SparseStorage<T> *container, uint32_t index)
            : container_(container), index_(index) {}

      public:
        const T &operator*() const { return container_->at(index_); }
        const T *operator->() const { return &container_->at(index_); }
        SparseStorageConstIterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }

        SparseStorageConstIterator operator++(int) {
            SparseStorageConstIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const SparseStorageConstIterator &rhs) const { return index_ < rhs.index_; }
        bool operator<=(const SparseStorageConstIterator &rhs) const { return index_ <= rhs.index_; }
        bool operator>(const SparseStorageConstIterator &rhs) const { return index_ > rhs.index_; }
        bool operator>=(const SparseStorageConstIterator &rhs) const { return index_ >= rhs.index_; }
        bool operator==(const SparseStorageConstIterator &rhs) const { return index_ == rhs.index_; }
        bool operator!=(const SparseStorageConstIterator &rhs) const { return index_ != rhs.index_; }
    };

    using iterator = SparseStorageIterator;
    using const_iterator = SparseStorageConstIterator;

    iterator begin() {
        for (uint32_t i = 0; i < capacity_; i++) {
            if (bits_.IsSet(i)) {
                return iterator(this, i);
            }
        }
        return end();
    }

    const_iterator cbegin() const {
        for (uint32_t i = 0; i < capacity_; i++) {
            if (bits_.IsSet(i)) {
                return const_iterator(this, i);
            }
        }
        return cend();
    }

    iterator end() { return iterator(this, capacity_); }
    const_iterator cend() const { return const_iterator(this, capacity_); }

    iterator iter_at(uint32_t i) { return iterator(this, i); }
    const_iterator citer_at(uint32_t i) const { return const_iterator(this, i); }

    iterator erase(iterator it) {
        const uint32_t next_index = NextOccupied(it.index());
        erase(it.index());
        return iterator(this, next_index);
    }

  private:
    uint32_t NextOccupied(uint32_t index) const {
        for (uint32_t i = index + 1; i < capacity_; ++i) {
            if (bits_.IsSet(i)) {
                return i;
            }
        }
        return capacity_;
    }
};

template <typename T> const uint32_t SparseStorage<T>::InitialNonZeroCapacity;
} // namespace Ref
} // namespace Ray
