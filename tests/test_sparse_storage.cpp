#include "test_common.h"

#include <algorithm>
#include <random>
#include <vector>

#include "../internal/SparseStorageCPU.h"

namespace {
std::vector<int> GenTestData(int size) {
    std::vector<int> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = i;
    }
    return vec;
}
} // namespace

void test_sparse_storage() {
    printf("Test sparse_storage     | ");

    { // reserve method
        Ray::Cpu::SparseStorage<int> s1;
        require(s1.size() == 0);
        s1.reserve(128);
        require(s1.size() == 0);
        require(s1.capacity() == 128);
        require(s1.IntegrityCheck());
    }

    { // pushing elements
        Ray::Cpu::SparseStorage<int> s1;
        const std::pair<uint32_t, uint32_t> i1 = s1.emplace(1);
        require(s1.IntegrityCheck());
        const std::pair<uint32_t, uint32_t> i2 = s1.push(12);
        require(s1.IntegrityCheck());
        const std::pair<uint32_t, uint32_t> i3 = s1.push(45);
        require(s1.IntegrityCheck());

        require(i1.first == 0);
        require(i2.first == 1);
        require(i3.first == 2);

        require(s1.at(0) == 1);
        require(s1.at(1) == 12);
        require(s1[2] == 45);

        s1.Erase(i2.second);
        require(s1.IntegrityCheck());

        require(s1.at(0) == 1);
        require(s1[2] == 45);

        const std::pair<uint32_t, uint32_t> i4 = s1.push(32);
        const std::pair<uint32_t, uint32_t> i5 = s1.push(78);

        require(i4.first == 1);
        require(i5.first == 3);

        require(s1.at(0) == 1);
        require(s1.at(1) == 32);
        require(s1[3] == 78);

        auto it = s1.begin();
        require(*it == 1);
        ++it;
        require(*it == 32);

        s1.Erase(i1.second);
        require(s1.IntegrityCheck());
        s1.Erase(i3.second);
        require(s1.IntegrityCheck());
        s1.Erase(i4.second);
        require(s1.IntegrityCheck());
        s1.Erase(i5.second);
        require(s1.IntegrityCheck());
    }

    { // range allocations
        Ray::Cpu::SparseStorage<int> s1;

        const std::pair<uint32_t, uint32_t> i1 = s1.Allocate(100, 42);
        require(i1.first == 0);
        require(s1.IntegrityCheck());
        for (uint32_t i = i1.first; i < i1.first + 100; ++i) {
            require(s1[i] == 42);
        }

        const std::pair<uint32_t, uint32_t> i2 = s1.Allocate(100, 24);
        require(i2.first == 100);
        require(s1.IntegrityCheck());
        for (uint32_t i = i2.first; i < i2.first + 100; ++i) {
            require(s1[i] == 24);
        }

        s1.Erase(i1.second);
        require(s1.IntegrityCheck());

        const std::pair<uint32_t, uint32_t> i3 = s1.Allocate(100, 24);
        require(i3.first == 0);
        require(s1.IntegrityCheck());

        s1.Erase(i2.second);
        require(s1.IntegrityCheck());
        s1.Erase(i3.second);
        require(s1.IntegrityCheck());
    }

    { // iteration
        std::vector<int> data = GenTestData(1000);
        std::vector<std::pair<uint32_t, uint32_t>> allocs;

        Ray::Cpu::SparseStorage<int> s1;
        for (int v : data) {
            allocs.push_back(s1.push(v));
            require(s1.IntegrityCheck());
        }

        auto it = s1.begin();
        for (int i = 0; i < 1000; i++) {
            require(*it == data[i]);
            ++it;
        }

        auto it2 = s1.cbegin();
        for (int i = 0; i < 1000; i++) {
            require(*it2 == data[i]);
            ++it2;
        }

        std::vector<std::pair<uint32_t, uint32_t>> to_delete;
        for (uint32_t i = 0; i < 1000; i += 2) {
            to_delete.push_back(allocs[i]);
        }

        // make deletion happen in random order
        std::shuffle(to_delete.begin(), to_delete.end(), std::default_random_engine(0));

        for (std::pair<uint32_t, uint32_t> i : to_delete) {
            s1.Erase(i.second);
            require(s1.IntegrityCheck());
        }

        it = s1.begin();
        for (int i = 1; i < 1000; i += 2) {
            require(*it == data[i]);
            ++it;
        }

        // fill the gaps and make it reallocate
        for (int v : data) {
            for (int i = 0; i < 100; i++) {
                s1.push(v);
                require(s1.IntegrityCheck());
            }
        }

        // check again
        for (int i = 1; i < 1000; i += 2) {
            require(s1[i] == data[i]);
        }

        s1.clear();
    }

    printf("OK\n");
}
