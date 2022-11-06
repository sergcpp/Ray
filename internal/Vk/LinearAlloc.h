#pragma once

#include <cassert>
#include <climits>
#include <cstring>

#include <string>

namespace Ray {
#ifndef RAY_EXCHANGE_DEFINED
template <class T, class U = T> T exchange(T &obj, U &&new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}
#define RAY_EXCHANGE_DEFINED
#endif

class ILog;

namespace Vk {
class LinearAlloc {
  protected:
    uint32_t block_size_ = 0;
    uint32_t block_count_ = 0;

    uint64_t *bitmap_ = nullptr;

    static const int BitmapGranularity = sizeof(uint64_t) * CHAR_BIT;

  public:
    LinearAlloc() = default;
    LinearAlloc(const uint32_t block_size, const uint32_t total_size) {
        block_size_ = block_size;
        block_count_ = (total_size + block_size - 1) / block_size;
        block_count_ = BitmapGranularity * ((block_count_ + BitmapGranularity - 1) / BitmapGranularity);

        bitmap_ = new uint64_t[block_count_ / BitmapGranularity];
        memset(bitmap_, 0xff, sizeof(uint64_t) * (block_count_ / BitmapGranularity));
    }
    ~LinearAlloc() { delete[] bitmap_; }

    LinearAlloc(const LinearAlloc &rhs) = delete;
    LinearAlloc(LinearAlloc &&rhs) noexcept
        : block_size_(rhs.block_size_), block_count_(rhs.block_count_), bitmap_(exchange(rhs.bitmap_, nullptr)) {}

    LinearAlloc &operator=(const LinearAlloc &rhs) = delete;
    LinearAlloc &operator=(LinearAlloc &&rhs) noexcept {
        if (&rhs == this) {
            return (*this);
        }

        delete[] bitmap_;

        block_size_ = exchange(rhs.block_size_, 0);
        block_count_ = exchange(rhs.block_count_, 0);
        bitmap_ = exchange(rhs.bitmap_, nullptr);

        return (*this);
    }

    void Resize(const uint32_t total_size) {
        uint32_t new_block_count = (total_size + block_size_ - 1) / block_size_;
        new_block_count = BitmapGranularity * ((new_block_count + BitmapGranularity - 1) / BitmapGranularity);

        auto *new_bitmap = new uint64_t[new_block_count / BitmapGranularity];

        const uint32_t blocks_to_copy = std::min(block_count_, new_block_count);
        memcpy(new_bitmap, bitmap_, sizeof(uint64_t) * (blocks_to_copy / BitmapGranularity));

        if (blocks_to_copy < new_block_count) {
            memset(new_bitmap + (blocks_to_copy / BitmapGranularity), 0xff,
                   sizeof(uint64_t) * ((new_block_count - blocks_to_copy) / BitmapGranularity));
        }

        delete[] bitmap_;

        block_count_ = new_block_count;
        bitmap_ = new_bitmap;
    }

    bool IsSet(const uint32_t block_index) const {
        assert(int(block_index) < block_count_);
        const int xword_index = block_index / BitmapGranularity;
        const int bit_index = block_index % BitmapGranularity;
        return (bitmap_[xword_index] & (1ull << bit_index)) == 0;
    }

    uint32_t size() const { return block_size_ * block_count_; }

    uint32_t Alloc(uint32_t req_size, const char *tag);
    void Free(uint32_t offset, uint32_t size);

    void Clear();
};
} // namespace Vk
} // namespace Ray