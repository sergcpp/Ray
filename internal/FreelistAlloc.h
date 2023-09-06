#pragma once

#include <cstdint>

#include <utility>
#include <vector>

// Based on : https://github.com/mattconte/tlsf

namespace Ray {
template <typename OffsetType, bool InPlace> struct tlsf_index_t {
    static const int ALIGN_SIZE_LOG2 = InPlace ? (sizeof(OffsetType) == 8 ? 3 : 2) : 0;
    static const int ALIGN_SIZE = (1 << ALIGN_SIZE_LOG2);
    static const int SL_INDEX_COUNT_LOG2 = 5;

    static const int FL_INDEX_MAX = sizeof(OffsetType) == 8 ? 32 : 30;
    static const int SL_INDEX_COUNT = (1 << SL_INDEX_COUNT_LOG2);
    static const int FL_INDEX_SHIFT = (SL_INDEX_COUNT_LOG2 + ALIGN_SIZE_LOG2);
    static const int FL_INDEX_COUNT = (FL_INDEX_MAX - FL_INDEX_SHIFT + 1);
    static const int SMALL_BLOCK_SIZE = (1 << FL_INDEX_SHIFT);

    static_assert(InPlace || (SMALL_BLOCK_SIZE / SL_INDEX_COUNT) == 1, "!");

    // First and second level bitmap
    uint32_t fl_bitmap = 0; // zero means 'no free blocks'
    uint32_t sl_bitmap[FL_INDEX_COUNT] = {};
    // Free blocks arranged by size
    OffsetType free_heads[FL_INDEX_COUNT][SL_INDEX_COUNT] = {}; // block 0 is a fake 'null block'

    static std::pair<int, int> mapping_insert(OffsetType size);
    static std::pair<int, int> mapping_search(OffsetType size);

    OffsetType search_suitable_block(std::pair<int, int> &inout_index);
};

// Non-intrusive allocator (blocks are stored separately)
class FreelistAlloc {
    tlsf_index_t<uint32_t, false> index_;

    struct block_t {
        uint32_t prev_phys = 0xffffffff, next_phys = 0xffffffff;
        uint16_t pool = 0xffff;
        union {
            struct {
                uint16_t is_free : 1;
                uint16_t is_prev_free : 1;
                uint16_t _unused : 14;
            };
            uint16_t flags = 0;
        };
        uint32_t offset = 0, size = 0;
        uint32_t prev_free = 0xffffffff, next_free = 0xffffffff;
    };
    static_assert(sizeof(block_t) == 28, "!");

    std::vector<block_t> all_blocks_;
    std::vector<uint32_t> unused_blocks_;

    struct pool_t {
        uint32_t head, tail;
    };
    std::vector<pool_t> pools_;
    std::vector<uint16_t> unused_pools_;

    uint16_t create_pool() {
        uint16_t ret;
        if (!unused_pools_.empty()) {
            ret = unused_pools_.back();
            unused_pools_.pop_back();
        } else {
            ret = uint16_t(pools_.size());
            pools_.push_back({0xffffffff, 0xffffffff});
        }
        return ret;
    }

    void remove_pool(const uint16_t pool) {
        pools_[pool] = {0xffffffff, 0xffffffff};
        if (pool == uint16_t(pools_.size() - 1)) {
            pools_.pop_back();
        } else {
            unused_pools_.emplace_back(pool);
        }
    }

    uint32_t create_block() {
        uint32_t ret;
        if (!unused_blocks_.empty()) {
            ret = unused_blocks_.back();
            unused_blocks_.pop_back();
        } else {
            ret = uint32_t(all_blocks_.size());
            all_blocks_.emplace_back();
        }
        return ret;
    }

    void remove_block(const uint32_t block_index) {
#ifndef NDEBUG
        all_blocks_[block_index] = {};
#endif
        if (block_index == uint32_t(all_blocks_.size() - 1)) {
            all_blocks_.pop_back();
        } else {
            unused_blocks_.push_back(block_index);
        }
    }

    void insert_free_block(uint32_t block_index, std::pair<int, int> index);
    uint32_t block_locate_free(uint32_t size);
    void disconnect_free_block(uint32_t block_index, std::pair<int, int> index);
    void block_trim_free(uint32_t block_index, uint32_t size);
    uint32_t block_trim_free_leading(uint32_t block_index, uint32_t size);
    uint32_t block_prepare_used(uint32_t block_index, uint32_t size);
    void block_mark_as_free(uint32_t block_index);
    void block_absorb(uint32_t prev_index, uint32_t block_index);
    uint32_t block_merge_prev(uint32_t block_index);
    void block_merge_next(uint32_t block_index);

  public:
    FreelistAlloc() {
        // Create null block
        all_blocks_.emplace_back();
    }
    FreelistAlloc(const uint32_t size) : FreelistAlloc() { AddPool(size); }

    int pools_count() const { return int(pools_.size() - unused_pools_.size()); }

    struct Allocation {
        uint32_t offset = 0xffffffff;
        uint32_t block = 0xffffffff;
        uint16_t pool = 0xffff;
    };

    uint16_t AddPool(uint32_t size);
    void RemovePool(uint16_t pool);
    void ResizePool(uint16_t pool, uint32_t size);

    Allocation Alloc(uint32_t size);
    Allocation Alloc(uint32_t align, uint32_t size);
    void Free(uint32_t block);

    struct Range {
        uint32_t block = 0xffffffff;
        uint32_t offset = 0xffffffff;
        uint32_t size = 0;
    };

    Range GetFirstOccupiedBlock(uint16_t pool) const;
    Range GetNextOccupiedBlock(uint32_t block) const;
    Range GetBlockRange(const uint32_t block) const { return {block, all_blocks_[block].offset, all_blocks_[block].size}; }

    bool IntegrityCheck() const;
};
} // namespace Ray
