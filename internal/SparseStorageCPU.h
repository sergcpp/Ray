#pragma once

#include <cassert>
#include <climits>
#include <cstdint>

#include <memory>

#include "Core.h"
#include "FreelistAlloc.h"
#include "simd/aligned_allocator.h"

namespace Ray::Cpu {
template <typename T> class SparseStorage {
    std::unique_ptr<FreelistAlloc> alloc_; // TODO: merge with data allocation
    T *data_ = nullptr;
    uint32_t capacity_ = 0, size_ = 0;

    static const uint32_t InitialNonZeroCapacity = 8;

  public:
    explicit SparseStorage(const uint32_t initial_capacity = 0) {
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

    void reserve(uint32_t new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

        if (!alloc_) {
            alloc_ = std::make_unique<FreelistAlloc>(new_capacity);
        } else {
            alloc_->ResizePool(0, new_capacity);
        }

        T *new_data = (T *)aligned_malloc(new_capacity * sizeof(T), alignof(T));

        // move old data
        FreelistAlloc::Range r = alloc_->GetFirstOccupiedBlock(0);
        while (r.size) {
            for (uint32_t i = r.offset; i < r.offset + r.size; ++i) {
                new (&new_data[i]) T(std::move(data_[i]));
                data_[i].~T();
            }
            r = alloc_->GetNextOccupiedBlock(r.block);
        }

        aligned_free(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

    template <class... Args> std::pair<uint32_t, uint32_t> emplace(Args &&...args) {
        if (size_ + 1 > capacity_) {
            reserve(std::max(capacity_ * 2, InitialNonZeroCapacity));
        }

        const FreelistAlloc::Allocation al = alloc_->Alloc(1);
        new (&data_[al.offset]) T(std::forward<Args>(args)...);

        ++size_;
        return std::pair{al.offset, al.block};
    }

    std::pair<uint32_t, uint32_t> push(const T &el) {
        if (size_ + 1 > capacity_) {
            reserve(std::max(capacity_ * 2, InitialNonZeroCapacity));
        }

        const FreelistAlloc::Allocation al = alloc_->Alloc(1);
        new (&data_[al.offset]) T(el);

        ++size_;
        return std::pair{al.offset, al.block};
    }

    void clear() {
        if (!alloc_) {
            return;
        }
        FreelistAlloc::Range r = alloc_->GetFirstOccupiedBlock(0);
        while (r.size) {
            for (uint32_t i = r.offset; i < r.offset + r.size; ++i) {
                data_[i].~T();
                assert(size_ > 0);
                --size_;
            }
            const uint32_t to_release = r.block;
            r = alloc_->GetNextOccupiedBlock(r.block);
            alloc_->Free(to_release);
        }
        assert(size_ == 0);
    }

    uint32_t GetCount(const uint32_t block_index) { return alloc_->GetBlockRange(block_index).size; }

    void Erase(const uint32_t block_index) {
        const FreelistAlloc::Range r = alloc_->GetBlockRange(block_index);
        for (uint32_t i = r.offset; i < r.offset + r.size; ++i) {
            data_[i].~T();
            --size_;
        }
        alloc_->Free(block_index);
    }

    template <class... Args> std::pair<uint32_t, uint32_t> Allocate(const uint32_t count, Args &&...args) {
        if (size_ + count > capacity_) {
            uint32_t new_capacity = std::max(capacity_, InitialNonZeroCapacity);
            while (new_capacity < size_ + count) {
                new_capacity *= 2;
            }
            reserve(new_capacity);
        }

        FreelistAlloc::Allocation al = alloc_->Alloc(count);
        while (al.offset == 0xffffffff) {
            reserve(std::max(capacity_ * 2, InitialNonZeroCapacity));
            al = alloc_->Alloc(count);
        }

        for (uint32_t i = al.offset; i < al.offset + count; ++i) {
            new (&data_[i]) T(std::forward<Args>(args)...);
        }

        size_ += count;

        return std::pair{al.offset, al.block};
    }

    force_inline T &at(const uint32_t index) { return data_[index]; }
    force_inline const T &at(const uint32_t index) const { return data_[index]; }

    force_inline T &operator[](const uint32_t index) { return data_[index]; }
    force_inline const T &operator[](const uint32_t index) const { return data_[index]; }

    class SparseStorageIterator {
        friend class SparseStorage<T>;

        SparseStorage<T> *container_;
        FreelistAlloc::Range range_;

        SparseStorageIterator(SparseStorage<T> *container, FreelistAlloc::Range range)
            : container_(container), range_(range) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        T &operator*() { return container_->at(range_.offset); }
        T *operator->() { return &container_->at(range_.offset); }
        SparseStorageIterator &operator++() {
            if (range_.size > 1) {
                ++range_.offset;
                --range_.size;
            } else {
                range_ = container_->alloc_->GetNextOccupiedBlock(range_.block);
            }
            return *this;
        }
        SparseStorageIterator operator++(int) {
            SparseStorageIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return range_.offset; }
        uint32_t block() const { return range_.block; }

        bool operator<(const SparseStorageIterator &rhs) const { return range_.offset < rhs.range_.offset; }
        bool operator<=(const SparseStorageIterator &rhs) const { return range_.offset <= rhs.range_.offset; }
        bool operator>(const SparseStorageIterator &rhs) const { return range_.offset > rhs.range_.offset; }
        bool operator>=(const SparseStorageIterator &rhs) const { return range_.offset >= rhs.range_.offset; }
        bool operator==(const SparseStorageIterator &rhs) const { return range_.offset == rhs.range_.offset; }
        bool operator!=(const SparseStorageIterator &rhs) const { return range_.offset != rhs.range_.offset; }
    };

    class SparseStorageConstIterator {
        friend class SparseStorage<T>;

        const SparseStorage<T> *container_;
        FreelistAlloc::Range range_;

        SparseStorageConstIterator(const SparseStorage<T> *container, FreelistAlloc::Range range)
            : container_(container), range_(range) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        const T &operator*() { return container_->at(range_.offset); }
        const T *operator->() { return &container_->at(range_.offset); }
        SparseStorageConstIterator &operator++() {
            if (range_.size > 1) {
                ++range_.offset;
                --range_.size;
            } else {
                range_ = container_->alloc_->GetNextOccupiedBlock(range_.block);
            }
            return *this;
        }
        SparseStorageConstIterator operator++(int) {
            SparseStorageIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return range_.offset; }
        uint32_t block() const { return range_.block; }

        bool operator<(const SparseStorageConstIterator &rhs) const { return range_.offset < rhs.range_.offset; }
        bool operator<=(const SparseStorageConstIterator &rhs) const { return range_.offset <= rhs.range_.offset; }
        bool operator>(const SparseStorageConstIterator &rhs) const { return range_.offset > rhs.range_.offset; }
        bool operator>=(const SparseStorageConstIterator &rhs) const { return range_.offset >= rhs.range_.offset; }
        bool operator==(const SparseStorageConstIterator &rhs) const { return range_.offset == rhs.range_.offset; }
        bool operator!=(const SparseStorageConstIterator &rhs) const { return range_.offset != rhs.range_.offset; }
    };

    using iterator = SparseStorageIterator;
    using const_iterator = SparseStorageConstIterator;

    iterator begin() {
        if (alloc_) {
            return iterator(this, alloc_->GetFirstOccupiedBlock(0));
        }
        return end();
    }

    const_iterator cbegin() const {
        if (alloc_) {
            return const_iterator(this, alloc_->GetFirstOccupiedBlock(0));
        }
        return cend();
    }

    iterator end() { return iterator(this, FreelistAlloc::Range{0xffffffff, capacity_, 0}); }
    const_iterator cend() const { return const_iterator(this, FreelistAlloc::Range{0xffffffff, capacity_, 0}); }

    iterator erase(iterator it) {
        iterator ret = it;
        Erase(it.block());
        return ++ret;
    }

    bool IntegrityCheck() const { return alloc_->IntegrityCheck(); }
};

template <typename T> const uint32_t SparseStorage<T>::InitialNonZeroCapacity;
} // namespace Ray::Cpu
