#pragma once

#include "Bitmap.h"
#include "CoreVK.h"
#include "Vk/Buffer.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

namespace Ray {
namespace Vk {
class Context;

template <typename T> class SparseStorage {
    Context *ctx_ = nullptr;
    std::string name_;
    mutable Buffer cpu_buf_;
    // TODO: remove this pointer
    T *cpu_data_ = nullptr; // persistently mapped pointer
    mutable Buffer gpu_buf_;
    uint32_t size_ = 0;

    static const uint32_t InitialNonZeroCapacity = 8;

    static_assert(std::is_trivially_copyable<T>::value, "!");

  public:
    SparseStorage(Context *ctx, const char *name, const uint32_t initial_capacity = 0) : ctx_(ctx), name_(name) {
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
    force_inline uint32_t capacity() const { return uint32_t(cpu_buf_.size() / sizeof(T)); }

    force_inline bool empty() const { return size_ == 0; }

    force_inline T *data() { return cpu_data_; }
    force_inline const T *data() const { return cpu_data_; }

    force_inline bool exists(const uint32_t index) const { return cpu_buf_.IsSet(index); }

    Buffer &gpu_buf() const { return gpu_buf_; }
    Buffer &cpu_buf() const { return cpu_buf_; }

    void reserve(const uint32_t new_capacity) {
        if (new_capacity <= capacity()) {
            return;
        }

        if (!cpu_buf_.ctx()) {
            cpu_buf_ =
                Buffer{name_.c_str(), ctx_, eBufType::Stage, uint32_t(new_capacity * sizeof(T)), uint32_t(sizeof(T))};
            gpu_buf_ =
                Buffer{name_.c_str(), ctx_, eBufType::Storage, uint32_t(new_capacity * sizeof(T)), uint32_t(sizeof(T))};
        } else {
            cpu_buf_.Unmap();
            cpu_data_ = nullptr;
            cpu_buf_.Resize(new_capacity * sizeof(T));
            gpu_buf_.Resize(new_capacity * sizeof(T));
        }

        cpu_data_ = (T *)cpu_buf_.Map(BufMapRead | BufMapWrite, true /* persistent */);
    }

    template <class... Args> uint32_t emplace(Args &&...args) {
        if (size_ + 1 > capacity()) {
            reserve(std::max(capacity() * 2, InitialNonZeroCapacity));
        }

        const auto cpu_index = uint32_t(cpu_buf_.AllocSubRegion(sizeof(T), nullptr) / sizeof(T));
        const auto gpu_index = uint32_t(gpu_buf_.AllocSubRegion(sizeof(T), nullptr) / sizeof(T));
        assert(cpu_index == gpu_index);

        T *el = cpu_data_ + cpu_index;
        new (el) T(std::forward<Args>(args)...);

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());
        gpu_buf_.UpdateSubRegion(gpu_index * sizeof(T), sizeof(T), cpu_buf_, cpu_index * sizeof(T), cmd_buf);
        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        ++size_;
        return cpu_index;
    }

    uint32_t push(const T &el) {
        if (size_ + 1 > capacity()) {
            reserve(std::max(capacity() * 2, InitialNonZeroCapacity));
        }

        const uint32_t cpu_index = cpu_buf_.AllocSubRegion(sizeof(T), nullptr) / sizeof(T);
        const uint32_t gpu_index = gpu_buf_.AllocSubRegion(sizeof(T), nullptr) / sizeof(T);
        assert(cpu_index == gpu_index);

        new (&cpu_data_[cpu_index]) T(el);
        cpu_buf_.FlushMappedRange(cpu_index * sizeof(T), sizeof(T), true /* align size */);

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());
        gpu_buf_.UpdateSubRegion(gpu_index * sizeof(T), sizeof(T), cpu_buf_, cpu_index * sizeof(T), cmd_buf);
        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        ++size_;
        return cpu_index;
    }

    void clear() {
        for (uint32_t i = 0; i < capacity() && size_; ++i) {
            if (cpu_buf_.IsSet(i)) {
                erase(i);
            }
        }
    }

    void erase(const uint32_t index) {
        cpu_buf_.FreeSubRegion(index * sizeof(T), sizeof(T));
        gpu_buf_.FreeSubRegion(index * sizeof(T), sizeof(T));

        --size_;
    }

    void Set(const uint32_t index, const T &el) {
        new (&cpu_data_[index]) T(el);
        cpu_buf_.FlushMappedRange(index * sizeof(T), sizeof(T), true /* align size */);

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->device(), ctx_->temp_command_pool());
        gpu_buf_.UpdateSubRegion(index * sizeof(T), sizeof(T), cpu_buf_, index * sizeof(T), cmd_buf);
        EndSingleTimeCommands(ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        ++size_;
    }

    // force_inline T &at(const uint32_t index) {
    //     assert(bits_.IsSet(index) && "Invalid index!");
    //     return cpu_data_[index];
    // }

    force_inline const T &at(const uint32_t index) const {
        assert(cpu_buf_.IsSet(index) && "Invalid index!");
        return cpu_data_[index];
    }

    // force_inline T &operator[](const uint32_t index) {
    //     assert(bits_.IsSet(index) && "Invalid index!");
    //     return cpu_data_[index];
    // }

    force_inline const T &operator[](const uint32_t index) const {
        assert(cpu_buf_.IsSet(index) && "Invalid index!");
        return cpu_data_[index];
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

    // using iterator = SparseStorageIterator;
    using const_iterator = SparseStorageConstIterator;

    const_iterator begin() const {
        for (uint32_t i = 0; i < capacity(); i++) {
            if (cpu_buf_.IsSet(i)) {
                return const_iterator(this, i);
            }
        }
        return end();
    }

    const_iterator cbegin() const {
        for (uint32_t i = 0; i < capacity(); i++) {
            if (cpu_buf_.IsSet(i)) {
                return const_iterator(this, i);
            }
        }
        return cend();
    }

    const_iterator end() const { return const_iterator(this, capacity()); }
    const_iterator cend() const { return const_iterator(this, capacity()); }

    const_iterator iter_at(const uint32_t i) const { return const_iterator(this, i); }
    const_iterator citer_at(const uint32_t i) const { return const_iterator(this, i); }

    /*iterator erase(iterator it) {
        const uint32_t next_index = NextOccupied(it.index());
        erase(it.index());
        return iterator(this, next_index);
    }*/

  private:
    uint32_t NextOccupied(uint32_t index) const {
        for (uint32_t i = index + 1; i < capacity(); ++i) {
            if (cpu_buf_.IsSet(i)) {
                return i;
            }
        }
        return capacity();
    }
};

template <typename T> const uint32_t SparseStorage<T>::InitialNonZeroCapacity;
} // namespace Vk
} // namespace Ray