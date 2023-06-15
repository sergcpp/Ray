#pragma once

#include <climits>

#include "Core.h"

namespace Ray {
class Bitmap {
    int block_count_ = 0;
    uint64_t *bitmap_ = nullptr;

    static const int BitmapGranularity = sizeof(uint64_t) * CHAR_BIT;

  public:
    explicit Bitmap(const int block_count) { Resize(block_count); }

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
        int blocks_found = blocks_required;
        const int pos = FindEmpty<true>(0, blocks_found);
        if (pos != -1) {
            // Mark blocks as occupied
            for (int i = pos; i < pos + blocks_required; ++i) {
                const int xword_index = i / BitmapGranularity;
                const int bit_index = i % BitmapGranularity;
                bitmap_[xword_index] &= ~(1ull << bit_index);
            }
        }
        return pos;
    }

    int Alloc_BestFit(const int blocks_required) {
        int best_blocks_available = block_count_ + 1;
        int best_loc = -1;

        int blocks_found = blocks_required;
        int pos = FindEmpty<false>(0, blocks_found);
        while (pos != -1) {
            if (blocks_found < best_blocks_available) {
                best_blocks_available = blocks_found;
                best_loc = pos;
                if (blocks_found == blocks_required) {
                    break;
                }
            }
            pos = FindEmpty<false>(pos, blocks_found);
        }

        if (best_loc != -1) {
            // Mark blocks as occupied
            for (int i = best_loc; i < best_loc + blocks_required; ++i) {
                const int xword_index = i / BitmapGranularity;
                const int bit_index = i % BitmapGranularity;
                bitmap_[xword_index] &= ~(1ull << bit_index);
            }
        }

        return best_loc;
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

    void Occupy(const int block_index, const int block_count) {
        assert(block_index < block_count_);
        // Mark blocks as occupied
        for (int i = block_index; i < block_index + block_count; ++i) {
            const int xword_index = i / BitmapGranularity;
            const int bit_index = i % BitmapGranularity;
            assert((bitmap_[xword_index] & (1ull << bit_index)) != 0);
            bitmap_[xword_index] &= ~(1ull << bit_index);
        }
    }

    template <bool EarlyExit> int FindEmpty(int start_index, int &blocks_required) const {
#if 1
        if (start_index >= block_count_) {
            return -1;
        }
        int xword_beg = start_index / BitmapGranularity, bit_beg = start_index % BitmapGranularity;
        int xword_end, bit_end;
        for (xword_end = xword_beg, bit_end = bit_beg + 1; xword_end < (block_count_ / BitmapGranularity);) {
            if (!bitmap_[xword_end]) {
                ++xword_end;
                bit_end = 0;
                continue;
            }
            if ((bitmap_[xword_end] & (1ull << bit_end)) == 0) {
                if ((bitmap_[xword_beg] & (1ull << bit_beg)) != 0) {
                    const int free_count = (xword_end - xword_beg) * BitmapGranularity + (bit_end - bit_beg);
                    if (free_count >= blocks_required) {
                        blocks_required = free_count;
                        return xword_beg * BitmapGranularity + bit_beg;
                    }
                }
                xword_beg = xword_end;
                bit_beg = bit_end;

                bit_end = CountTrailingZeroes((bitmap_[xword_end] & ~((1ull << bit_end) - 1)));
            } else {
                if ((bitmap_[xword_beg] & (1ull << bit_beg)) == 0) {
                    xword_beg = xword_end;
                    bit_beg = bit_end;
                } else if (EarlyExit) {
                    const int free_count = (xword_end - xword_beg) * BitmapGranularity + (bit_end - bit_beg);
                    if (free_count >= blocks_required) {
                        blocks_required = free_count;
                        return xword_beg * BitmapGranularity + bit_beg;
                    }
                }
                bit_end = CountTrailingZeroes(~(bitmap_[xword_end] | ((1ull << bit_end) - 1)));
            }

            if (bit_end == BitmapGranularity) {
                ++xword_end;
                bit_end = 0;
            }
        }

        if ((bitmap_[xword_beg] & (1ull << bit_beg)) != 0) {
            const int free_count = (xword_end - xword_beg) * BitmapGranularity + (bit_end - bit_beg);
            if (free_count >= blocks_required) {
                blocks_required = free_count;
                return xword_beg * BitmapGranularity + bit_beg;
            }
        }

#else
        int xword_beg = start_index / BitmapGranularity, bit_beg = start_index % BitmapGranularity;
        for (int end = start_index + 1; end < block_count_;) {
            const int xword_end = end / BitmapGranularity, bit_end = end % BitmapGranularity;
            if (!bitmap_[xword_end]) {
                end += (BitmapGranularity - bit_end);
                continue;
            }
            if ((bitmap_[xword_end] & (1ull << bit_end)) == 0) {
                if ((bitmap_[xword_beg] & (1ull << bit_beg)) != 0) {
                    const int free_count = (xword_end - xword_beg) * BitmapGranularity + (bit_end - bit_beg);
                    if (free_count >= blocks_required) {
                        blocks_required = free_count;
                        return xword_beg * BitmapGranularity + bit_beg;
                    }
                }
                xword_beg = xword_end;
                bit_beg = bit_end;
            }
            ++end;
        }
#endif
        return -1;
    }
};
} // namespace Ray