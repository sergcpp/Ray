#pragma once

#include <cassert>
#include <climits>

#include "Core.h"
#include "simd/aligned_allocator.h"

namespace Ray {
class Bitmap {
    int block_count_ = 0;
    uint64_t *bitmap_ = nullptr;

    static const int BitmapGranularity = sizeof(uint64_t) * CHAR_BIT;

  public:
    Bitmap(const int block_count) { Resize(block_count); }

    ~Bitmap() {
        for (int i = 0; i < block_count_ / BitmapGranularity; ++i) {
            assert(bitmap_[i] == 0xffffffffffffffff && "Not all allocations freed!");
        }
        delete[] bitmap_;
    }

    Bitmap(const Bitmap &rhs) = delete;
    Bitmap(Bitmap &&rhs) = delete;

    Bitmap &operator=(const Bitmap &rhs) = delete;
    Bitmap &operator=(Bitmap &&rhs) = delete;

    void Resize(int new_block_count) {
        uint64_t *new_bitmap = nullptr;
        // round up
        new_block_count = BitmapGranularity * ((new_block_count + BitmapGranularity - 1) / BitmapGranularity);
        if (new_block_count) {
            new_bitmap = new uint64_t[new_block_count / BitmapGranularity];
            // mark blocks as free
            memset(new_bitmap, 0xff, sizeof(uint64_t) * (new_block_count / BitmapGranularity));
            if (bitmap_) {
                // copy old data
                memcpy(new_bitmap, bitmap_,
                       sizeof(uint64_t) * (std::min(block_count_, new_block_count) / BitmapGranularity));
            }
        }
        delete[] bitmap_;
        bitmap_ = new_bitmap;
        block_count_ = new_block_count;
    }

    bool IsSet(const uint32_t block_index) const {
        assert(int(block_index) < block_count_);
        const int xword_index = block_index / BitmapGranularity;
        const int bit_index = block_index % BitmapGranularity;
        return (bitmap_[xword_index] & (1ull << bit_index)) == 0;
    }

    int Alloc_FirstFit(const int blocks_required) {
        int loc_beg = 0;
        // Skip initial occupied blocks
        while (!bitmap_[loc_beg]) {
            ++loc_beg;
        }
#if 1
        const int loc_lim = (block_count_ - blocks_required + BitmapGranularity - 1) / BitmapGranularity;
        unsigned long bit_beg = 0;
        while (loc_beg < loc_lim) {
            if (GetFirstBit(bitmap_[loc_beg] & ~((1ull << bit_beg) - 1), &bit_beg)) {
                int bit_end = CountTrailingZeroes(~(bitmap_[loc_beg] | ((1ull << bit_beg) - 1)));
                int loc_end = loc_beg;
                if (bit_end == BitmapGranularity) {
                    ++loc_end;
                    bit_end = 0;
                    while (loc_end < (block_count_ / BitmapGranularity) &&
                           (loc_end - loc_beg) * BitmapGranularity - int(bit_beg) + bit_end < blocks_required) {
                        bit_end = CountTrailingZeroes(~bitmap_[loc_end]);
                        if (bit_end != BitmapGranularity) {
                            break;
                        }
                        ++loc_end;
                        bit_end = 0;
                    }
                }

                const int blocks_found = (loc_end - loc_beg) * BitmapGranularity - bit_beg + bit_end;
                if (blocks_found >= blocks_required) {
                    // Mark blocks as occupied
                    const int block_beg = loc_beg * BitmapGranularity + bit_beg;
                    for (int i = block_beg; i < block_beg + blocks_required; ++i) {
                        const int xword_index = i / BitmapGranularity;
                        const int bit_index = i % BitmapGranularity;
                        bitmap_[xword_index] &= ~(1ull << bit_index);
                    }
                    return loc_beg * BitmapGranularity + bit_beg;
                }
                bit_beg = bit_end;
                loc_beg = loc_end;
            } else {
                ++loc_beg;
            }
        }
#else
        loc_beg *= BitmapGranularity;
        for (; loc_beg <= block_count_ - blocks_required;) {
            if (!bitmap_[loc_beg / BitmapGranularity]) {
                const int count = (BitmapGranularity - (loc_beg % BitmapGranularity));
                loc_beg += count;
                continue;
            }

            // Count the number of available blocks
            int loc_end = loc_beg;
            while (loc_end < loc_beg + blocks_required) {
                const int xword_index = loc_end / BitmapGranularity;
                const int bit_index = loc_end % BitmapGranularity;
                if ((bitmap_[xword_index] & (1ull << bit_index)) == 0) {
                    break;
                }
                ++loc_end;
            }

            if ((loc_end - loc_beg) >= blocks_required) {
                // Mark blocks as occupied
                for (int i = loc_beg; i < loc_beg + blocks_required; ++i) {
                    const int xword_index = i / BitmapGranularity;
                    const int bit_index = i % BitmapGranularity;
                    bitmap_[xword_index] &= ~(1ull << bit_index);
                }
                return loc_beg;
            } else {
                loc_beg = loc_end + 1;
            }
        }
#endif
        return -1;
    }

    int Alloc_BestFit(const int blocks_required) {
        int best_blocks_available = block_count_ + 1;
        int best_loc = -1;

        int loc_beg = 0;
        // Skip initial occupied blocks
        while (!bitmap_[loc_beg]) {
            ++loc_beg;
        }

#if 1
        const int loc_lim = (block_count_ - blocks_required + BitmapGranularity - 1) / BitmapGranularity;
        unsigned long bit_beg = 0;
        while (loc_beg < loc_lim) {
            if (GetFirstBit(bitmap_[loc_beg] & ~((1ull << bit_beg) - 1), &bit_beg)) {
                int bit_end = CountTrailingZeroes(~(bitmap_[loc_beg] | ((1ull << bit_beg) - 1)));
                int loc_end = loc_beg;
                if (bit_end == BitmapGranularity) {
                    ++loc_end;
                    bit_end = 0;
                    while (loc_end < loc_lim) {
                        bit_end = CountTrailingZeroes(~bitmap_[loc_end]);
                        if (bit_end != BitmapGranularity) {
                            break;
                        }
                        ++loc_end;
                        bit_end = 0;
                    }
                }

                const int blocks_found = (loc_end - loc_beg) * BitmapGranularity - bit_beg + bit_end;
                if (blocks_found >= blocks_required && blocks_found < best_blocks_available) {
                    best_blocks_available = blocks_found;
                    best_loc = loc_beg * BitmapGranularity + bit_beg;
                    if (blocks_found == blocks_required) {
                        // Perfect fit was found, can stop here
                        break;
                    }
                }
                bit_beg = bit_end;
                loc_beg = loc_end;
            } else {
                ++loc_beg;
            }
        }
#else
        loc_beg *= BitmapGranularity;
        for (; loc_beg <= block_count_ - blocks_required;) {
            if (!bitmap_[loc_beg / BitmapGranularity]) {
                const int count = (BitmapGranularity - (loc_beg % BitmapGranularity));
                loc_beg += count;
                continue;
            }

            // Count the number of available blocks
            int loc_end = loc_beg;
            while (loc_end < block_count_) {
                const int xword_index = loc_end / BitmapGranularity;
                const int bit_index = loc_end % BitmapGranularity;
                if ((bitmap_[xword_index] & (1ull << bit_index)) == 0) {
                    break;
                }
                ++loc_end;
            }

            if ((loc_end - loc_beg) >= blocks_required && (loc_end - loc_beg) < best_blocks_available) {
                best_blocks_available = (loc_end - loc_beg);
                best_loc = loc_beg;
                if ((loc_end - loc_beg) == blocks_required) {
                    // Perfect fit was found, can stop here
                    break;
                }
            }
            loc_beg = loc_end + 1;
        }
#endif

        if (best_loc != -1) {
            // Mark blocks as occupied
            for (int i = best_loc; i < best_loc + blocks_required; ++i) {
                const int xword_index = i / BitmapGranularity;
                const int bit_index = i % BitmapGranularity;
                bitmap_[xword_index] &= ~(1ull << bit_index);
            }
            return best_loc;
        }

        return -1;
    }

    void Free(const int block_index, const int block_count) {
        assert(block_index < block_count_);
        // Mark blocks as free
        for (int i = block_index; i < block_index + block_count; ++i) {
            const int xword_index = i / BitmapGranularity;
            const int bit_index = i % BitmapGranularity;
            assert((bitmap_[xword_index] & (1ull << bit_index)) == 0);
            bitmap_[xword_index] |= (1ull << bit_index);
        }
    }
};

template <typename T> class SparseStorage {
    Bitmap bits_;
    T *data_ = nullptr;
    uint32_t capacity_ = 0, size_ = 0;
    // TODO: add next_free_ to speedup allocation

    static const uint32_t InitialNonZeroCapacity = 8;
  public:
    explicit SparseStorage(uint32_t initial_capacity = 0) : bits_(initial_capacity) {
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

    force_inline bool exists(const uint32_t index) const {
        return bits_.IsSet(index);
    }

    void reserve(uint32_t new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

        bits_.Resize(new_capacity);

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

    template <class... Args> uint32_t emplace(Args &&... args) {
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
        bits_.Free(index, 1);

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
};

template <typename T>
const uint32_t SparseStorage<T>::InitialNonZeroCapacity;

} // namespace Ray
