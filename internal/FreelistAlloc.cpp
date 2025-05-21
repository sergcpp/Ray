#include "FreelistAlloc.h"

#include <cassert>

#include "Core.h"

namespace Ray {
force_inline int fls(const uint32_t word) {
#ifdef _MSC_VER
    unsigned long index;
    return _BitScanReverse(&index, word) ? index : -1;
#else
    const int bit = word ? 32 - __builtin_clz(word) : 0;
    return bit - 1;
#endif
}

force_inline int fls(const uint64_t size) {
    int high = int(size >> 32);
    int bits = 0;
    if (high) {
        bits = 32 + fls(uint32_t(high));
    } else {
        bits = fls(uint32_t(size & 0xffffffff));
    }
    return bits;
}

force_inline int ffs(uint32_t word) {
#ifdef _MSC_VER
    unsigned long index;
    return _BitScanForward(&index, word) ? index : -1;
#else
    return __builtin_ffs(word) - 1;
#endif
}
} // namespace Ray

template <typename OffsetType, bool InPlace>
OffsetType Ray::tlsf_index_t<OffsetType, InPlace>::rounded_size(OffsetType size) {
    if (size >= SMALL_BLOCK_SIZE) {
        const uint32_t round = (1 << (fls(size) - SL_INDEX_COUNT_LOG2)) - 1;
        size += round;
    }
    return size;
}

template <typename OffsetType, bool InPlace>
std::pair<int, int> Ray::tlsf_index_t<OffsetType, InPlace>::mapping_insert(const OffsetType size) {
    if (size < SMALL_BLOCK_SIZE) {
        return std::pair{0, int(size / (SMALL_BLOCK_SIZE / SL_INDEX_COUNT))};
    } else {
        const int fl = fls(size);
        const int sl = int(size >> (fl - SL_INDEX_COUNT_LOG2)) ^ (1 << SL_INDEX_COUNT_LOG2);
        return std::pair{fl - (FL_INDEX_SHIFT - 1), sl};
    }
}

template <typename OffsetType, bool InPlace>
std::pair<int, int> Ray::tlsf_index_t<OffsetType, InPlace>::mapping_search(OffsetType size) {
    return mapping_insert(rounded_size(size));
}

template <typename OffsetType, bool InPlace>
OffsetType Ray::tlsf_index_t<OffsetType, InPlace>::search_suitable_block(std::pair<int, int> &inout_index) {
    std::pair<int, int> index = inout_index;

    // First, search for a block in the list associated with the given index
    uint32_t sl_map = sl_bitmap[index.first] & (~0u << index.second);
    if (sl_map == 0) {
        // No block exists, search in the next largest first-level list
        const uint32_t fl_map = fl_bitmap & (~0u << (index.first + 1));
        if (fl_map == 0) {
            // No free blocks available, memory has been exhausted
            return 0xffffffff;
        }

        index.first = ffs(fl_map);
        inout_index.first = index.first;
        sl_map = sl_bitmap[index.first];
    }
    assert(sl_map && "internal error - second level bitmap is null");
    index.second = ffs(sl_map);
    inout_index.second = index.second;

    // Return the first block in the free list
    return free_heads[index.first][index.second];
}

template struct Ray::tlsf_index_t<uint32_t, false>;
template struct Ray::tlsf_index_t<uint32_t, true>;
template struct Ray::tlsf_index_t<uint64_t, false>;
template struct Ray::tlsf_index_t<uint64_t, true>;

/////////////

uint16_t Ray::FreelistAlloc::AddPool(const uint32_t size) {
    uint16_t pool_index = create_pool();

    const uint32_t main_block_index = create_block();
    // zero-size sentinel block
    const uint32_t zero_block_index = create_block();

    pools_[pool_index] = {main_block_index, zero_block_index};

    all_blocks_[main_block_index].prev_phys = 0xffffffff;
    all_blocks_[main_block_index].next_phys = zero_block_index;
    all_blocks_[main_block_index].pool = pool_index;
    all_blocks_[main_block_index].offset = 0;
    all_blocks_[main_block_index].size = size;
    all_blocks_[main_block_index].prev_free = 0xffffffff;
    all_blocks_[main_block_index].next_free = 0xffffffff;
    all_blocks_[main_block_index].is_free = 1;
    all_blocks_[main_block_index].is_prev_free = 0;
    insert_free_block(main_block_index, tlsf_index_t<uint32_t, false>::mapping_insert(size));

    all_blocks_[zero_block_index].prev_phys = main_block_index;
    all_blocks_[zero_block_index].next_phys = 0xffffffff;
    all_blocks_[zero_block_index].pool = pool_index;
    all_blocks_[zero_block_index].offset = size;
    all_blocks_[zero_block_index].size = 0;
    all_blocks_[zero_block_index].prev_free = 0xffffffff;
    all_blocks_[zero_block_index].next_free = 0xffffffff;
    all_blocks_[zero_block_index].is_free = 0;
    all_blocks_[zero_block_index].is_prev_free = 1;

    return pool_index;
}

void Ray::FreelistAlloc::RemovePool(const uint16_t pool) {
    const uint32_t block_index = pools_[pool].head;

    assert(all_blocks_[block_index].is_free && "block must be free");
    assert(!all_blocks_[all_blocks_[block_index].next_phys].is_free && "next block must not be free");
    assert(all_blocks_[all_blocks_[block_index].next_phys].size == 0 && "next block size must be zero");

    disconnect_free_block(block_index, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[block_index].size));

    remove_block(all_blocks_[block_index].next_phys);
    remove_block(block_index);

    remove_pool(pool);
}

void Ray::FreelistAlloc::ResizePool(uint16_t pool, uint32_t size) {
    const uint32_t tail_index = pools_[pool].tail;
    assert(all_blocks_[tail_index].size == 0);
    const uint32_t prev_index = all_blocks_[tail_index].prev_phys;

    if (all_blocks_[prev_index].offset + all_blocks_[prev_index].size >= size) {
        // no need to resize
        return;
    }

    if (all_blocks_[prev_index].is_free) {
        disconnect_free_block(prev_index, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[prev_index].size));
        // expand existing free block
        all_blocks_[prev_index].size = size - all_blocks_[prev_index].offset;
        all_blocks_[tail_index].offset = size;
        insert_free_block(prev_index, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[prev_index].size));
    } else {
        // create new emply block
        const uint32_t new_block_index = create_block();

        all_blocks_[new_block_index].prev_phys = prev_index;
        all_blocks_[new_block_index].next_phys = tail_index;
        all_blocks_[new_block_index].pool = all_blocks_[prev_index].pool;
        all_blocks_[new_block_index].offset = all_blocks_[prev_index].offset + all_blocks_[prev_index].size;
        all_blocks_[new_block_index].size = size - all_blocks_[new_block_index].offset;
        all_blocks_[new_block_index].prev_free = 0xffffffff;
        all_blocks_[new_block_index].next_free = 0xffffffff;
        all_blocks_[new_block_index].is_free = 1;
        all_blocks_[new_block_index].is_prev_free = 0;

        all_blocks_[prev_index].next_phys = new_block_index;

        all_blocks_[tail_index].offset = size;
        all_blocks_[tail_index].prev_phys = new_block_index;
        all_blocks_[tail_index].is_prev_free = 1;

        insert_free_block(new_block_index,
                          tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[new_block_index].size));
    }
}

Ray::FreelistAlloc::Allocation Ray::FreelistAlloc::Alloc(const uint32_t size) {
    const uint32_t block = block_locate_free(size);
    const uint32_t offset = block_prepare_used(block, size);
    uint16_t pool = 0xffff;
    if (block != 0xffffffff) {
        pool = all_blocks_[block].pool;
    }
    return {offset, block, pool};
}

Ray::FreelistAlloc::Allocation Ray::FreelistAlloc::Alloc(uint32_t align, uint32_t size) {
    uint32_t block = block_locate_free(size + align);
    uint32_t offset = 0xffffffff;
    uint16_t pool = 0xffff;
    if (block != 0xffffffff) {
        offset = all_blocks_[block].offset;
        pool = all_blocks_[block].pool;
        const uint32_t aligned = align * ((offset + align - 1) / align);
        if (aligned - offset) {
            block = block_trim_free_leading(block, aligned - offset);
        }
        offset = block_prepare_used(block, size);
    }
    return {offset, block, pool};
}

void Ray::FreelistAlloc::Free(uint32_t block) {
    assert(!all_blocks_[block].is_free);
    block_mark_as_free(block);
    block = block_merge_prev(block);
    block_merge_next(block);

    insert_free_block(block, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[block].size));
}

Ray::FreelistAlloc::Range Ray::FreelistAlloc::GetFirstOccupiedBlock(const uint16_t pool) const {
    uint32_t block = pools_[pool].head;
    if (block == 0xffffffff) {
        return {block, 0xffffffff, 0};
    }
    while (all_blocks_[block].is_free) {
        block = all_blocks_[block].next_phys;
    }
    return {block, all_blocks_[block].offset, all_blocks_[block].size};
}

Ray::FreelistAlloc::Range Ray::FreelistAlloc::GetNextOccupiedBlock(uint32_t block) const {
    block = all_blocks_[block].next_phys;
    if (block == 0xffffffff) {
        return {block, 0xffffffff, 0};
    }
    while (all_blocks_[block].is_free) {
        block = all_blocks_[block].next_phys;
    }
    return {block, all_blocks_[block].offset, all_blocks_[block].size};
}

bool Ray::FreelistAlloc::IntegrityCheck() const {
    int errors = 0;

#define insist(cond)                                                                                                   \
    assert(cond);                                                                                                      \
    if (!(cond)) {                                                                                                     \
        ++errors;                                                                                                      \
    }

    for (int i = 0; i < tlsf_index_t<uint32_t, false>::FL_INDEX_COUNT; ++i) {
        for (int j = 0; j < tlsf_index_t<uint32_t, false>::SL_INDEX_COUNT; ++j) {
            const uint32_t fl_map = index_.fl_bitmap & (1u << i);
            const uint32_t sl_list = index_.sl_bitmap[i];
            const uint32_t sl_map = sl_list & (1u << j);

            uint32_t block_index = index_.free_heads[i][j];

            if (fl_map == 0) {
                insist(sl_map == 0 && "second-level map must be null");
            }

            if (sl_map == 0) {
                insist(block_index == 0 && "block list must be null");
                continue;
            }

            insist(sl_list != 0 && "no free blocks in second-level map");
            insist(block_index != 0 && "block should not be null");

            while (block_index != 0) {
                insist(all_blocks_[block_index].is_free && "block should be free");
                insist(!all_blocks_[block_index].is_prev_free && "blocks should have coalesced");
                insist(!all_blocks_[all_blocks_[block_index].next_phys].is_free && "blocks should have coalesced");
                insist(all_blocks_[all_blocks_[block_index].next_phys].is_prev_free && "block should be free");

                const std::pair<int, int> index =
                    tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[block_index].size);
                insist(index.first == i && index.second == j && "block size indexed in wrong list");
                assert(block_index != all_blocks_[block_index].next_free && "cycle detected");
                block_index = all_blocks_[block_index].next_free;
            }
        }
    }

#undef insist

    return (errors == 0);
}

void Ray::FreelistAlloc::insert_free_block(const uint32_t block_index, const std::pair<int, int> index) {
    const uint32_t current_index = index_.free_heads[index.first][index.second];
    all_blocks_[block_index].next_free = current_index;
    all_blocks_[block_index].prev_free = 0; // point to the null block

    all_blocks_[current_index].prev_free = block_index;

    // Insert the new block at the head of the list, and mark bitmaps appropriately
    index_.free_heads[index.first][index.second] = block_index;
    index_.fl_bitmap |= (1 << index.first);
    index_.sl_bitmap[index.first] |= (1 << index.second);
}

uint32_t Ray::FreelistAlloc::block_locate_free(uint32_t size) {
    std::pair<int, int> index;
    uint32_t block = 0xffffffff;

    if (size) {
        index = tlsf_index_t<uint32_t, false>::mapping_search(size);
        if (index.first < tlsf_index_t<uint32_t, false>::FL_INDEX_COUNT) {
            block = index_.search_suitable_block(index);
        }
    }

    if (block != 0xffffffff) {
        assert(all_blocks_[block].size >= size);
        disconnect_free_block(block, index);
    }

    return block;
}

void Ray::FreelistAlloc::disconnect_free_block(const uint32_t block_index, const std::pair<int, int> index) {
    uint32_t prev_index = all_blocks_[block_index].prev_free;
    uint32_t next_index = all_blocks_[block_index].next_free;
    assert(prev_index != 0xffffffff && "prev_free field can not be null");
    assert(next_index != 0xffffffff && "next_free field can not be null");
    all_blocks_[next_index].prev_free = prev_index;
    all_blocks_[prev_index].next_free = next_index;

    // If this block is the head of the free list, set new head
    if (index_.free_heads[index.first][index.second] == block_index) {
        index_.free_heads[index.first][index.second] = next_index;

        // If the new head if null, clear the bitmap
        if (next_index == 0) {
            index_.sl_bitmap[index.first] &= ~(1u << index.second);

            // If the second bitmap is now empty, clear the fl bitmap
            if (!index_.sl_bitmap[index.first]) {
                index_.fl_bitmap &= ~(1u << index.first);
            }
        }
    }
}

void Ray::FreelistAlloc::block_trim_free(uint32_t block_index, uint32_t size) {
    assert(all_blocks_[block_index].is_free);
    if (all_blocks_[block_index].size > size) {
        const uint32_t remaining_block = create_block();
        all_blocks_[remaining_block].prev_phys = block_index;
        all_blocks_[remaining_block].next_phys = all_blocks_[block_index].next_phys;
        all_blocks_[remaining_block].pool = all_blocks_[block_index].pool;
        all_blocks_[remaining_block].offset = all_blocks_[block_index].offset + size;
        all_blocks_[remaining_block].size = all_blocks_[block_index].size - size;
        all_blocks_[remaining_block].prev_free = 0xffffffff;
        all_blocks_[remaining_block].next_free = 0xffffffff;
        all_blocks_[remaining_block].is_free = 1;
        all_blocks_[remaining_block].is_prev_free = 1;

        all_blocks_[block_index].next_phys = remaining_block;
        all_blocks_[block_index].size = size;

        all_blocks_[all_blocks_[remaining_block].next_phys].prev_phys = remaining_block;

        insert_free_block(remaining_block,
                          tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[remaining_block].size));
    }
}

uint32_t Ray::FreelistAlloc::block_trim_free_leading(uint32_t block_index, uint32_t size) {
    uint32_t remaining_block = block_index;
    if (true /* split? */) {
        remaining_block = create_block();
        all_blocks_[remaining_block].prev_phys = block_index;
        all_blocks_[remaining_block].next_phys = all_blocks_[block_index].next_phys;
        all_blocks_[remaining_block].pool = all_blocks_[block_index].pool;
        all_blocks_[remaining_block].offset = all_blocks_[block_index].offset + size;
        all_blocks_[remaining_block].size = all_blocks_[block_index].size - size;
        all_blocks_[remaining_block].prev_free = 0xffffffff;
        all_blocks_[remaining_block].next_free = 0xffffffff;
        all_blocks_[remaining_block].is_free = 1;
        all_blocks_[remaining_block].is_prev_free = 1;

        all_blocks_[block_index].next_phys = remaining_block;
        all_blocks_[block_index].size = size;

        all_blocks_[all_blocks_[remaining_block].next_phys].prev_phys = remaining_block;

        insert_free_block(block_index, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[block_index].size));
    }
    return remaining_block;
}

uint32_t Ray::FreelistAlloc::block_prepare_used(uint32_t block_index, uint32_t size) {
    uint32_t ret = 0xffffffff;
    if (block_index != 0xffffffff) {
        assert(size && "size must be non-zero");
        block_trim_free(block_index, size);
        all_blocks_[block_index].is_free = 0;
        all_blocks_[all_blocks_[block_index].next_phys].is_prev_free = 0;
        ret = all_blocks_[block_index].offset;
    }
    return ret;
}

void Ray::FreelistAlloc::block_mark_as_free(const uint32_t block_index) {
    const uint32_t next = all_blocks_[block_index].next_phys;
    all_blocks_[next].is_prev_free = 1;
    all_blocks_[block_index].is_free = 1;
}

void Ray::FreelistAlloc::block_absorb(const uint32_t prev_index, const uint32_t block_index) {
    all_blocks_[prev_index].size += all_blocks_[block_index].size;

    all_blocks_[all_blocks_[block_index].next_phys].prev_phys = prev_index;
    all_blocks_[prev_index].next_phys = all_blocks_[block_index].next_phys;

    remove_block(block_index);
}

uint32_t Ray::FreelistAlloc::block_merge_prev(uint32_t block_index) {
    if (!all_blocks_[block_index].is_prev_free) {
        return block_index;
    }

    const uint32_t prev_index = all_blocks_[block_index].prev_phys;
    assert(prev_index != 0xffffffff && "invalid prev physical block");
    assert(all_blocks_[prev_index].is_free);

    disconnect_free_block(prev_index, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[prev_index].size));

    block_absorb(prev_index, block_index);
    return prev_index;
}

void Ray::FreelistAlloc::block_merge_next(uint32_t block_index) {
    const uint32_t next = all_blocks_[block_index].next_phys;
    assert(next != 0xffffffff);

    if (all_blocks_[next].is_free) {
        disconnect_free_block(next, tlsf_index_t<uint32_t, false>::mapping_insert(all_blocks_[next].size));
        block_absorb(block_index, next);
    }
}
