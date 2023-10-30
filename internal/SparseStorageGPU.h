#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "FreelistAlloc.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

namespace Ray {
namespace NS {
class Context;

template <typename T, bool Replicate = true> class SparseStorage {
    Context *ctx_ = nullptr;
    std::string name_;
    std::unique_ptr<FreelistAlloc> alloc_;
    mutable Buffer cpu_buf_;
    mutable Buffer gpu_buf_;
    uint32_t size_ = 0;

    static const uint32_t InitialNonZeroCapacity = 8;

    static_assert(std::is_trivially_copyable<T>::value, "!");

  public:
    SparseStorage(Context *ctx, const char *name, const uint32_t initial_capacity = 8) : ctx_(ctx), name_(name) {
        if (initial_capacity) {
            reserve(initial_capacity);
        }
    }

    ~SparseStorage() {
        if (cpu_buf_.is_mapped()) {
            cpu_buf_.Unmap();
        }
    }

    force_inline uint32_t size() const { return size_; }
    force_inline uint32_t capacity() const { return uint32_t(gpu_buf_.size() / sizeof(T)); }

    force_inline bool empty() const { return size_ == 0; }

    // force_inline T *::type data() { return cpu_buf_.mapped_ptr<T>(); }
    force_inline const T *data() const { return cpu_buf_.mapped_ptr<T>(); }

    Buffer &gpu_buf() const { return gpu_buf_; }
    Buffer &cpu_buf() const { return cpu_buf_; }

    void reserve(const uint32_t new_capacity) {
        if (new_capacity <= capacity()) {
            return;
        }

        if (!alloc_) {
            alloc_ = std::make_unique<FreelistAlloc>(new_capacity);
        } else {
            alloc_->ResizePool(0, new_capacity);
        }

        if (!gpu_buf_.ctx()) {
            if (Replicate) {
                cpu_buf_ = Buffer{name_.c_str(), ctx_, eBufType::Upload, uint32_t(new_capacity * sizeof(T))};
            }
            gpu_buf_ = Buffer{name_.c_str(), ctx_, eBufType::Storage, uint32_t(new_capacity * sizeof(T))};
        } else {
            if (Replicate) {
                cpu_buf_.Unmap();
                cpu_buf_.Resize(new_capacity * sizeof(T));
            }
            gpu_buf_.Resize(new_capacity * sizeof(T));
        }

        assert(new_capacity == capacity());

        if (Replicate) {
            cpu_buf_.Map(true /* persistent */);
        }
    }

    template <class... Args> std::pair<uint32_t, uint32_t> emplace(Args &&...args) {
        if (size_ + 1 > capacity()) {
            reserve(std::max(capacity() * 2, InitialNonZeroCapacity));
        }

        const FreelistAlloc::Allocation al = alloc_->Alloc(1);

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        Buffer temp_buf;

        if (Replicate) {
            T *el = cpu_buf_.mapped_ptr<T>() + al.offset;
            new (el) T(std::forward<Args>(args)...);
            cpu_buf_.FlushMappedRange(al.offset * sizeof(T), sizeof(T), true /* align size */);

            gpu_buf_.UpdateSubRegion(al.offset * sizeof(T), sizeof(T), cpu_buf_, al.offset * sizeof(T), cmd_buf);
        } else {
            temp_buf = Buffer("Temp staging buf", ctx_, eBufType::Upload, sizeof(T));

            T *el = reinterpret_cast<T *>(temp_buf.Map());
            new (el) T(std::forward<Args>(args)...);
            temp_buf.FlushMappedRange(0, sizeof(T), true /* align size */);
            temp_buf.Unmap();

            gpu_buf_.UpdateSubRegion(al.offset * sizeof(T), sizeof(T), temp_buf, 0, cmd_buf);
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
        temp_buf.FreeImmediate();

        ++size_;
        return std::make_pair(al.offset, al.block);
    }

    std::pair<uint32_t, uint32_t> push(const T &el) { return emplace(el); }

    std::pair<uint32_t, uint32_t> Allocate(const T *beg, const uint32_t count) {
        if (size_ + count > capacity()) {
            uint32_t new_capacity = std::max(capacity(), InitialNonZeroCapacity);
            while (new_capacity < size_ + count) {
                new_capacity *= 2;
            }
            reserve(new_capacity);
        }

        FreelistAlloc::Allocation al = alloc_->Alloc(count);
        while (al.offset == 0xffffffff) {
            reserve(std::max(capacity() * 2, InitialNonZeroCapacity));
            al = alloc_->Alloc(count);
        }

        if (beg && count) {
            CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
            Buffer temp_buf;

            const T *it = beg;
            if (Replicate) {
                for (uint32_t i = al.offset; i < al.offset + count; ++i) {
                    new (cpu_buf_.mapped_ptr<T>() + i) T(*it++);
                }
                cpu_buf_.FlushMappedRange(al.offset * sizeof(T), count * sizeof(T), true /* align size */);

                gpu_buf_.UpdateSubRegion(al.offset * sizeof(T), count * sizeof(T), cpu_buf_, al.offset * sizeof(T),
                                         cmd_buf);
            } else {
                temp_buf = Buffer("Temp staging buf", ctx_, eBufType::Upload, count * sizeof(T));
                T *el = reinterpret_cast<T *>(temp_buf.Map());
                for (uint32_t i = 0; i < count; ++i) {
                    new (el + i) T(*it++);
                }
                temp_buf.FlushMappedRange(0, count * sizeof(T), true /* align size */);
                temp_buf.Unmap();

                gpu_buf_.UpdateSubRegion(al.offset * sizeof(T), count * sizeof(T), temp_buf, 0, cmd_buf);
            }

            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());
            temp_buf.FreeImmediate();
        }

        size_ += count;

        return std::make_pair(al.offset, al.block);
    }

    void clear() {
        if (!alloc_) {
            return;
        }

        Ray::FreelistAlloc::Range r = alloc_->GetFirstOccupiedBlock(0);
        while (r.size) {
            for (uint32_t i = r.offset; i < r.offset + r.size; ++i) {
                if (Replicate) {
                    (cpu_buf_.mapped_ptr<T>() + i)->~T();
                }
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
            if (Replicate) {
                (cpu_buf_.mapped_ptr<T>() + i)->~T();
            }
            --size_;
        }
        alloc_->Free(block_index);
    }

    void Set(const uint32_t index, const T &el) { Set(index, 1, &el); }

    void Set(const uint32_t start, const uint32_t count, const T els[]) {
        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        Buffer temp_buf;

        if (Replicate) {
            for (uint32_t i = 0; i < count; ++i) {
                new (cpu_buf_.mapped_ptr<T>() + start + i) T(els[i]);
            }
            cpu_buf_.FlushMappedRange(start * sizeof(T), count * sizeof(T), true /* align size */);

            gpu_buf_.UpdateSubRegion(start * sizeof(T), count * sizeof(T), cpu_buf_, start * sizeof(T), cmd_buf);
        } else {
            temp_buf = Buffer("Temp staging buf", ctx_, eBufType::Upload, count * sizeof(T));
            T *el = reinterpret_cast<T *>(temp_buf.Map());
            for (uint32_t i = 0; i < count; ++i) {
                new (el + i) T(els[i]);
            }
            temp_buf.FlushMappedRange(0, count * sizeof(T), true /* align size */);
            temp_buf.Unmap();

            gpu_buf_.UpdateSubRegion(start * sizeof(T), count * sizeof(T), temp_buf, 0, cmd_buf);
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
        temp_buf.FreeImmediate();
    }

    force_inline const T &at(const uint32_t index) const { return *(cpu_buf_.mapped_ptr<T>() + index); }
    force_inline T &at(const uint32_t index) { return *(cpu_buf_.mapped_ptr<T>() + index); }

    force_inline const T &operator[](const uint32_t index) const { return *(cpu_buf_.mapped_ptr<T>() + index); }
    force_inline T &operator[](const uint32_t index) { return *(cpu_buf_.mapped_ptr<T>() + index); }

    class SparseStorageIterator : public std::iterator<std::forward_iterator_tag, T> {
        friend class SparseStorage<T>;

        SparseStorage<T> *container_;
        FreelistAlloc::Range range_;

        SparseStorageIterator(SparseStorage<T> *container, FreelistAlloc::Range range)
            : container_(container), range_(range) {}

      public:
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

    class SparseStorageConstIterator : public std::iterator<std::forward_iterator_tag, T> {
        friend class SparseStorage<T>;

        const SparseStorage<T> *container_;
        FreelistAlloc::Range range_;

        SparseStorageConstIterator(const SparseStorage<T> *container, FreelistAlloc::Range range)
            : container_(container), range_(range) {}

      public:
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
            SparseStorageConstIterator tmp(*this);
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

    iterator end() { return iterator(this, FreelistAlloc::Range{0xffffffff, capacity(), 0}); }
    const_iterator cend() const { return const_iterator(this, FreelistAlloc::Range{0xffffffff, capacity(), 0}); }

    iterator erase(iterator it) {
        iterator ret = it;
        Erase(it.block());
        return ++ret;
    }

    bool IntegrityCheck() const { return alloc_->IntegrityCheck(); }
};

template <typename T, bool Replicate> const uint32_t SparseStorage<T, Replicate>::InitialNonZeroCapacity;
} // namespace NS
} // namespace Ray