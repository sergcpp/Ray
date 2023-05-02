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
    { // reserve method
        Ray::Cpu::SparseStorage<int> s1;
        require(s1.size() == 0);
        s1.reserve(128);
        require(s1.size() == 0);
        require(s1.capacity() == 128);
    }

    { // pushing elements
        Ray::Cpu::SparseStorage<int> s1;
        uint32_t i1 = s1.emplace(1);
        uint32_t i2 = s1.push(12);
        uint32_t i3 = s1.push(45);

        require(i1 == 0);
        require(i2 == 1);
        require(i3 == 2);

        require(s1.at(0) == 1);
        require(s1.at(1) == 12);
        require(s1[2] == 45);

        s1.erase(1);

        require(s1.at(0) == 1);
        require(s1[2] == 45);

        uint32_t i4 = s1.push(32);
        uint32_t i5 = s1.push(78);

        require(i4 == 1);
        require(i5 == 3);

        require(s1.at(0) == 1);
        require(s1.at(1) == 32);
        require(s1[3] == 78);

        auto it = s1.begin();
        require(*it == 1);
        ++it;
        require(*it == 32);

        s1.erase(i1);
        s1.erase(i3);
        s1.erase(i4);
        s1.erase(i5);
    }

    { // iteration test
        std::vector<int> data = GenTestData(1000);
        Ray::Cpu::SparseStorage<int> s1;
        for (int v : data) {
            s1.push(v);
        }

        auto it = s1.begin();
        for (int i = 0; i < 1000; i++) {
            require(*it == data[i]);
            ++it;
        }

        std::vector<uint32_t> to_delete;
        for (uint32_t i = 0; i < 1000; i += 2) {
            to_delete.push_back(i);
        }

        // make deletion to happen in random order
        std::shuffle(to_delete.begin(), to_delete.end(), std::default_random_engine(0));

        for (uint32_t i : to_delete) {
            s1.erase(i);
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
            }
        }

        // check again
        for (int i = 1; i < 1000; i += 2) {
            require(s1[i] == data[i]);
        }

        s1.clear();
    }
}
