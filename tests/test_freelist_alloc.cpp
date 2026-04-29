#include "test_common.h"

#include <memory>

#include "../internal/FreelistAlloc.h"

void test_freelist_alloc() {
    using namespace Ray;

    printf("Test freelist_alloc     | ");

    { // basic usage
        auto alloc = std::make_unique<FreelistAlloc>();
        require(alloc->IntegrityCheck());

        const uint16_t pool = alloc->AddPool(2048);
        require(pool == 0);
        require(alloc->IntegrityCheck());

        const FreelistAlloc::Allocation a = alloc->Alloc(128);
        require(a.offset == 0);
        require(a.pool == 0);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());
    }
    { // block merging 1
        auto alloc = std::make_unique<FreelistAlloc>(2048);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(1);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(123);
        require(b.offset == 1);
        require(alloc->IntegrityCheck());

        const auto c = alloc->Alloc(1234);
        require(c.offset == 124);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());
        alloc->Free(c.block);
        require(alloc->IntegrityCheck());

        // the whole space must be available
        const auto d = alloc->Alloc(2048);
        require(d.offset == 0);
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }
    { // block merging 2
        auto alloc = std::make_unique<FreelistAlloc>(2048);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(123);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());
        alloc->Free(a.block);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(123);
        require(b.offset == 0);
        require(alloc->IntegrityCheck());
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());

        // the whole space must be available
        const auto d = alloc->Alloc(2048);
        require(d.offset == 0);
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }
    { // reuse 1
        auto alloc = std::make_unique<FreelistAlloc>(8192);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(1024);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(3456);
        require(b.offset == 1024);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());

        const auto c = alloc->Alloc(1024);
        require(c.offset == 0);
        require(alloc->IntegrityCheck());

        alloc->Free(c.block);
        require(alloc->IntegrityCheck());
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());

        // the whole space must be available
        const auto d = alloc->Alloc(8192);
        require(d.offset == 0);
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }
    { // reuse 2
        auto alloc = std::make_unique<FreelistAlloc>(8192);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(1024);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(3456);
        require(b.offset == 1024);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);

        const auto c = alloc->Alloc(2345);
        require(c.offset == 1024 + 3456);
        require(alloc->IntegrityCheck());

        const auto d = alloc->Alloc(456);
        require(d.offset == 0);
        require(alloc->IntegrityCheck());

        const auto e = alloc->Alloc(512);
        require(e.offset == 456);
        require(alloc->IntegrityCheck());

        alloc->Free(c.block);
        require(alloc->IntegrityCheck());
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());
        alloc->Free(e.block);
        require(alloc->IntegrityCheck());

        // the whole space must be available
        const auto f = alloc->Alloc(8192);
        require(f.offset == 0);
        alloc->Free(f.block);
    }
    { // multiple pools
        auto alloc = std::make_unique<FreelistAlloc>();

        const uint16_t pool1 = alloc->AddPool(1024);
        require(pool1 == 0);
        require(alloc->IntegrityCheck());
        const uint16_t pool2 = alloc->AddPool(2048);
        require(pool2 == 1);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(512);
        require(a.offset == 0 && a.pool == 0);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(512);
        require(b.offset == 512 && b.pool == 0);
        require(alloc->IntegrityCheck());

        const auto c = alloc->Alloc(512);
        require(c.offset == 0 && c.pool == 1);
        require(alloc->IntegrityCheck());

        alloc->Free(b.block);
        require(alloc->IntegrityCheck());

        const auto d = alloc->Alloc(512);
        require(d.offset == 512 && d.pool == 0);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());
        alloc->Free(c.block);
        require(alloc->IntegrityCheck());
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }
    { // fragmentation
        auto alloc = std::make_unique<FreelistAlloc>(256 * 1024 * 1024);

        std::vector<FreelistAlloc::Allocation> allocations(256);
        for (int i = 0; i < 256; ++i) {
            allocations[i] = alloc->Alloc(1 * 1024 * 1024);
            require(allocations[i].offset == i * 1024 * 1024);
            require(alloc->IntegrityCheck());
        }

        // free 4 random allocations
        alloc->Free(allocations[243].block);
        alloc->Free(allocations[5].block);
        alloc->Free(allocations[123].block);
        alloc->Free(allocations[95].block);
        require(alloc->IntegrityCheck());

        // free 4 consequtive allocations
        alloc->Free(allocations[151].block);
        alloc->Free(allocations[152].block);
        alloc->Free(allocations[153].block);
        alloc->Free(allocations[154].block);
        require(alloc->IntegrityCheck());

        allocations[243] = alloc->Alloc(1 * 1024 * 1024);
        allocations[5] = alloc->Alloc(1 * 1024 * 1024);
        allocations[123] = alloc->Alloc(1 * 1024 * 1024);
        allocations[95] = alloc->Alloc(1 * 1024 * 1024);
        require(alloc->IntegrityCheck());

        require(allocations[243].offset != 0xffffffff);
        require(allocations[5].offset != 0xffffffff);
        require(allocations[123].offset != 0xffffffff);
        require(allocations[95].offset != 0xffffffff);

        for (int i = 0; i < 256; ++i) {
            if (i < 151 || i > 154) {
                alloc->Free(allocations[i].block);
                require(alloc->IntegrityCheck());
            }
        }

        const auto a = alloc->Alloc(256 * 1024 * 1024);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());
    }
    { // resize pool
        auto alloc = std::make_unique<FreelistAlloc>(256 * 1024);

        std::vector<FreelistAlloc::Allocation> allocations(512);
        for (int i = 0; i < 256; ++i) {
            allocations[i] = alloc->Alloc(1 * 1024);
            require(allocations[i].offset == i * 1024);
            require(alloc->IntegrityCheck());
        }

        const auto a = alloc->Alloc(1 * 1024);
        require(a.offset == 0xffffffff);

        alloc->ResizePool(0, 512 * 1024);
        require(alloc->IntegrityCheck());

        for (int i = 0; i < 256; ++i) {
            allocations[256 + i] = alloc->Alloc(1 * 1024);
            require(allocations[256 + i].offset == (256 + i) * 1024);
            require(alloc->IntegrityCheck());
        }
    }
    { // block iteration
        auto alloc = std::make_unique<FreelistAlloc>(256 * 1024);

        std::vector<FreelistAlloc::Allocation> allocations(256);
        for (int i = 0; i < 256; ++i) {
            allocations[i] = alloc->Alloc(1 * 1024);
            require(allocations[i].offset == i * 1024);
            require(alloc->IntegrityCheck());
        }

        for (int i = 0; i < 256; i += 2) {
            alloc->Free(allocations[i].block);
            require(alloc->IntegrityCheck());
        }

        FreelistAlloc::Range r = alloc->GetFirstOccupiedBlock(0);
        int i = 1;
        require(r.offset == allocations[i].offset);
        while (r.size) {
            require(r.offset == allocations[i].offset);
            i += 2;
            r = alloc->GetNextOccupiedBlock(r.block);
        }
    }

    { // stress: shuffled free order fully coalesces
        auto alloc = std::make_unique<FreelistAlloc>(4 * 1024 * 1024);

        const int count = 256;
        std::vector<FreelistAlloc::Allocation> allocs(count);
        for (int i = 0; i < count; ++i) {
            allocs[i] = alloc->Alloc(16 * 1024);
            require(allocs[i].offset != 0xffffffff);
        }
        require(alloc->IntegrityCheck());

        // Fisher-Yates shuffle with a fixed-seed LCG for reproducibility
        std::vector<int> order(count);
        for (int i = 0; i < count; ++i) { order[i] = i; }
        uint32_t rng = 0xdeadbeef;
        for (int i = count - 1; i > 0; --i) {
            rng = rng * 1664525u + 1013904223u;
            const int j = int(rng % uint32_t(i + 1));
            std::swap(order[i], order[j]);
        }
        for (int i = 0; i < count; ++i) {
            alloc->Free(allocs[order[i]].block);
        }
        require(alloc->IntegrityCheck());

        // every block must have coalesced back into a single free span
        const auto a = alloc->Alloc(4 * 1024 * 1024);
        require(a.offset == 0);
        alloc->Free(a.block);
        require(alloc->IntegrityCheck());
    }
    { // block iteration stays within each pool's boundary
        auto alloc = std::make_unique<FreelistAlloc>();

        // Fill pool0 completely so subsequent allocs go to pool1
        const uint16_t pool0 = alloc->AddPool(4 * 1024);
        std::vector<FreelistAlloc::Allocation> p0(4);
        for (int i = 0; i < 4; ++i) {
            p0[i] = alloc->Alloc(1024);
            require(p0[i].pool == pool0);
        }

        const uint16_t pool1 = alloc->AddPool(4 * 1024);
        std::vector<FreelistAlloc::Allocation> p1(4);
        for (int i = 0; i < 4; ++i) {
            p1[i] = alloc->Alloc(1024);
            require(p1[i].pool == pool1);
        }
        require(alloc->IntegrityCheck());

        // Free every other block in both pools to create gaps
        alloc->Free(p0[1].block);
        alloc->Free(p0[3].block);
        alloc->Free(p1[1].block);
        alloc->Free(p1[3].block);
        require(alloc->IntegrityCheck());

        // pool0 iteration: must see p0[0] then p0[2] then end, not spill into pool1
        FreelistAlloc::Range r = alloc->GetFirstOccupiedBlock(pool0);
        require(r.offset == p0[0].offset && r.size == 1024);
        r = alloc->GetNextOccupiedBlock(r.block);
        require(r.offset == p0[2].offset && r.size == 1024);
        r = alloc->GetNextOccupiedBlock(r.block);
        require(r.size == 0);

        // pool1 iteration: must see p1[0] then p1[2] then end
        r = alloc->GetFirstOccupiedBlock(pool1);
        require(r.offset == p1[0].offset && r.size == 1024);
        r = alloc->GetNextOccupiedBlock(r.block);
        require(r.offset == p1[2].offset && r.size == 1024);
        r = alloc->GetNextOccupiedBlock(r.block);
        require(r.size == 0);
    }
    { // GetBlockRange returns correct offset and size for live blocks
        auto alloc = std::make_unique<FreelistAlloc>(1024);

        const auto a = alloc->Alloc(128);
        const auto b = alloc->Alloc(256);
        require(alloc->IntegrityCheck());

        const auto ra = alloc->GetBlockRange(a.block);
        require(ra.block == a.block);
        require(ra.offset == a.offset);
        require(ra.size == 128);

        const auto rb = alloc->GetBlockRange(b.block);
        require(rb.block == b.block);
        require(rb.offset == b.offset);
        require(rb.size == 256);

        alloc->Free(a.block);
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());
    }
    { // size boundaries around SMALL_BLOCK_SIZE (32)
        // sizes < 32 use the small-block path in mapping_insert; >= 32 use the general path
        auto alloc = std::make_unique<FreelistAlloc>(4096);

        for (const uint32_t sz : {1u, 31u, 32u, 33u}) {
            const auto a = alloc->Alloc(sz);
            require(a.offset == 0);
            require(a.pool == 0);
            require(alloc->IntegrityCheck());
            alloc->Free(a.block);
            require(alloc->IntegrityCheck());

            // pool must be fully available after every alloc/free cycle
            const auto b = alloc->Alloc(4096);
            require(b.offset == 0);
            alloc->Free(b.block);
            require(alloc->IntegrityCheck());
        }
    }
    { // OOM: exhausted pool returns failure sentinels
        auto alloc = std::make_unique<FreelistAlloc>(1024);

        const auto a = alloc->Alloc(512);
        require(a.offset == 0);
        const auto b = alloc->Alloc(512);
        require(b.offset == 512);
        require(alloc->IntegrityCheck());

        const auto c = alloc->Alloc(1);
        require(c.offset == 0xffffffff);
        require(c.block == 0xffffffff);
        require(c.pool == 0xffff);

        alloc->Free(a.block);
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());

        // after freeing, full pool must be available again
        const auto d = alloc->Alloc(1024);
        require(d.offset == 0);
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }
    { // resize pool - expand existing free trailing block
        auto alloc = std::make_unique<FreelistAlloc>(256 * 1024);

        std::vector<FreelistAlloc::Allocation> allocs(128);
        for (int i = 0; i < 128; ++i) {
            allocs[i] = alloc->Alloc(1024);
            require(allocs[i].offset == i * 1024);
        }
        // 128 KB remains free at the tail
        require(alloc->IntegrityCheck());

        // ResizePool must expand the existing trailing free block rather than create a new one
        alloc->ResizePool(0, 512 * 1024);
        require(alloc->IntegrityCheck());

        // 384 KB of new space (128 KB original free + 256 KB growth) must now be usable
        std::vector<FreelistAlloc::Allocation> more(384);
        for (int i = 0; i < 384; ++i) {
            more[i] = alloc->Alloc(1024);
            require(more[i].offset == (128 + i) * 1024);
        }
        require(alloc->IntegrityCheck());
    }
    { // remove pool and index reuse
        auto alloc = std::make_unique<FreelistAlloc>();
        require(alloc->pools_count() == 0);

        const uint16_t pool0 = alloc->AddPool(1024);
        require(pool0 == 0);
        require(alloc->pools_count() == 1);
        require(alloc->IntegrityCheck());

        const auto a = alloc->Alloc(512);
        require(a.pool == pool0);

        alloc->Free(a.block);
        require(alloc->IntegrityCheck());

        alloc->RemovePool(pool0);
        require(alloc->pools_count() == 0);
        require(alloc->IntegrityCheck());

        // after removal the index must be reused
        const uint16_t pool1 = alloc->AddPool(2048);
        require(pool1 == 0);
        require(alloc->pools_count() == 1);
        require(alloc->IntegrityCheck());

        const auto b = alloc->Alloc(2048);
        require(b.offset == 0);
        require(b.pool == pool1);
        alloc->Free(b.block);
        require(alloc->IntegrityCheck());

        alloc->RemovePool(pool1);
        require(alloc->pools_count() == 0);
        require(alloc->IntegrityCheck());
    }
    { // aligned allocation
        auto alloc = std::make_unique<FreelistAlloc>(4096);

        // Alloc(1, size) is a fast-path that delegates to plain Alloc
        const auto a = alloc->Alloc(1, 100);
        require(a.offset == 0);
        require(alloc->IntegrityCheck());

        // Offset is now 100 (not 256-aligned); allocator must trim the 156-byte lead
        const auto b = alloc->Alloc(256, 128);
        require(b.offset == 256);
        require(alloc->IntegrityCheck());

        // Offset is now 384 (not 512-aligned); allocator must trim the 128-byte lead
        const auto c = alloc->Alloc(512, 64);
        require(c.offset == 512);
        require(alloc->IntegrityCheck());

        alloc->Free(a.block);
        alloc->Free(b.block);
        alloc->Free(c.block);
        require(alloc->IntegrityCheck());

        // all trimmed leading fragments must coalesce back to the full pool
        const auto d = alloc->Alloc(4096);
        require(d.offset == 0);
        alloc->Free(d.block);
        require(alloc->IntegrityCheck());
    }

    printf("OK\n");
}
