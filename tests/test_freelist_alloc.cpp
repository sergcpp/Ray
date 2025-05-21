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

        FreelistAlloc::Allocation allocations[256];
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

        FreelistAlloc::Allocation allocations[512];
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

        FreelistAlloc::Allocation allocations[256];
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

    printf("OK\n");
}
