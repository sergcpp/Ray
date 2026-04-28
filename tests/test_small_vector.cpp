#include "test_common.h"

#include <string>

#include "../internal/SmallVector.h"

void test_small_vector() {
    using namespace Ray;

    printf("Test small_vector       | ");

    static_assert(sizeof(SmallVectorImpl<int>) <= 16, "!");

    { // basic usage with trivial type
        SmallVector<int, 16> vec;

        for (int i = 0; i < 8; i++) {
            vec.push_back(i);
        }
        for (int i = 8; i < 16; i++) {
            vec.emplace_back(i);
        }

        require(vec.empty() == false);
        require(vec.size() == 16);
        require(vec.capacity() == 16);
        require(vec.is_on_heap() == false);

        for (int i = 0; i < 16; i++) {
            require(vec[i] == i);
        }
        require(vec.back() == 15);

        vec.push_back(42);

        require(vec.empty() == false);
        require(vec.size() == 17);
        require(vec.is_on_heap() == true);

        for (int i = 0; i < 16; i++) {
            require(vec[i] == i);
        }
        require(vec.back() == 42);

        vec.insert(vec.begin(), -42);

        require(vec.empty() == false);
        require(vec.size() == 18);
        require(vec.is_on_heap() == true);

        require(vec[0] == -42);
        require(vec[1] == 0);
        require(vec[2] == 1);
        require(vec[3] == 2);
        require(vec[4] == 3);
        require(vec[5] == 4);
        require(vec[6] == 5);
        require(vec[7] == 6);
        require(vec[8] == 7);
        require(vec[9] == 8);
        require(vec[10] == 9);
        require(vec[11] == 10);
        require(vec[12] == 11);
        require(vec[13] == 12);
        require(vec[14] == 13);
        require(vec[15] == 14);
        require(vec[16] == 15);
        require(vec[17] == 42);
    }

    { // basic usage with complicated type
        SmallVector<std::string, 16> vec;

        for (int i = 0; i < 8; i++) {
            vec.push_back(std::to_string(i));
        }
        for (int i = 8; i < 16; i++) {
            vec.emplace_back(std::to_string(i));
        }

        require(vec.empty() == false);
        require(vec.size() == 16);
        require(vec.capacity() == 16);
        require(vec.is_on_heap() == false);

        for (int i = 0; i < 16; i++) {
            require(vec[i] == std::to_string(i));
        }
        require(vec.back() == "15");

        vec.push_back("42");

        require(vec.empty() == false);
        require(vec.size() == 17);
        require(vec.is_on_heap() == true);

        for (int i = 0; i < 16; i++) {
            require(vec[i] == std::to_string(i));
        }
        require(vec.back() == "42");

        vec.insert(vec.begin(), "-42");

        require(vec.empty() == false);
        require(vec.size() == 18);
        require(vec.is_on_heap() == true);

        require(vec[0] == "-42");
        require(vec[1] == "0");
        require(vec[2] == "1");
        require(vec[3] == "2");
        require(vec[4] == "3");
        require(vec[5] == "4");
        require(vec[6] == "5");
        require(vec[7] == "6");
        require(vec[8] == "7");
        require(vec[9] == "8");
        require(vec[10] == "9");
        require(vec[11] == "10");
        require(vec[12] == "11");
        require(vec[13] == "12");
        require(vec[14] == "13");
        require(vec[15] == "14");
        require(vec[16] == "15");
        require(vec[17] == "42");
    }

    { // usage with custom type
        struct AAA {
            char more_data[16] = {};
            int data;

            explicit AAA(int _data) : data(_data) {}

            AAA(const AAA &rhs) = delete;
            AAA(AAA &&rhs) = default;
            AAA &operator=(const AAA &rhs) = delete;
            AAA &operator=(AAA &&rhs) = default;
        };

        SmallVector<AAA, 16> vec;

        for (int i = 0; i < 8; i++) {
            vec.push_back(AAA{2 * i});
            vec.emplace_back(2 * i + 1);
        }
        require(vec.is_on_heap() == false);
        require(vec.back().data == 15);

        require(vec.empty() == false);
        require(vec.size() == 16);
        require(vec.capacity() == 16);

        vec.push_back(AAA{42});

        require(vec.is_on_heap() == true);
        require(vec.back().data == 42);

        require(vec.empty() == false);
        require(vec.size() == 17);
    }

    { // erase
        SmallVector<int, 16> vec;
        for (int i = 0; i < 8; i++) {
            vec.push_back(i);
        }
        for (int i = 8; i < 16; i++) {
            vec.emplace_back(i);
        }

        vec.erase(vec.begin() + 8, vec.begin() + 12);

        require(vec.empty() == false);
        require(vec.size() == 12);
        require(vec.capacity() == 16);

        require(vec[0] == 0);
        require(vec[1] == 1);
        require(vec[2] == 2);
        require(vec[3] == 3);
        require(vec[4] == 4);
        require(vec[5] == 5);
        require(vec[6] == 6);
        require(vec[7] == 7);
        require(vec[8] == 12);
        require(vec[9] == 13);
        require(vec[10] == 14);
        require(vec[11] == 15);
    }

    { // destructor tracking
        static int live_count;
        struct Tracked {
            int val;
            explicit Tracked(int v) : val(v) { ++live_count; }
            Tracked(Tracked &&o) noexcept : val(o.val) { ++live_count; }
            Tracked &operator=(Tracked &&o) noexcept { val = o.val; return *this; }
            ~Tracked() { --live_count; }
        };

        { // inline path: destructor called on scope exit
            SmallVector<Tracked, 4> v;
            v.emplace_back(1);
            v.emplace_back(2);
            require(live_count == 2);
        }
        require(live_count == 0);

        { // heap path
            SmallVector<Tracked, 2> v;
            for (int i = 0; i < 8; i++) {
                v.emplace_back(i);
            }
            require(live_count == 8);
        }
        require(live_count == 0);

        { // erase calls destructors
            SmallVector<Tracked, 4> v;
            for (int i = 0; i < 4; i++) {
                v.emplace_back(i);
            }
            v.erase(v.begin() + 1, v.begin() + 3);
            require(live_count == 2);
        }
        require(live_count == 0);

        { // clear calls destructors
            SmallVector<Tracked, 4> v;
            for (int i = 0; i < 4; i++) {
                v.emplace_back(i);
            }
            v.clear();
            require(live_count == 0);
        }
    }

    { // erase return value
        SmallVector<int, 8> vec = {0, 1, 2, 3, 4};

        auto it = vec.erase(vec.begin() + 1, vec.begin() + 3);
        require(it == vec.begin() + 1);
        require(*it == 3);

        it = vec.erase(vec.begin() + 1);
        require(it == vec.begin() + 1);
        require(*it == 4);
    }

    { // comparison operators
        const SmallVector<int, 8> v1 = {1, 2, 3, 4, 5};
        const SmallVector<int, 8> v2 = {1, 2, 3, 4, 6};
        const SmallVector<int, 8> v3 = {1, 2, 3, 4, 5};
        const SmallVector<int, 8> v4 = {1, 2, 3, 4, 6};

        require(v1 < v2);
        require(v1 <= v3);
        require(v2 > v1);
        require(v2 >= v4);
        require(v1 == v3);
        require(v2 == v4);
        require(v1 != v2);
        require(v3 != v4);
    }

    { // resize, pop_back, clear, assign
        SmallVector<int, 8> vec;

        vec.resize(5, 99);
        require(vec.size() == 5);
        require(vec[0] == 99 && vec[4] == 99);
        require(vec.is_on_heap() == false);

        vec.resize(3);
        require(vec.size() == 3);
        require(vec[2] == 99);

        vec.pop_back();
        require(vec.size() == 2);
        require(vec.back() == 99);

        vec.clear();
        require(vec.empty());
        require(vec.capacity() == 8); // capacity unchanged

        vec.assign(4, 7);
        require(vec.size() == 4);
        require(vec[0] == 7 && vec[3] == 7);

        // resize beyond inline capacity spills to heap
        vec.resize(16, 3);
        require(vec.size() == 16);
        require(vec.is_on_heap() == true);
        require(vec[4] == 3 && vec[15] == 3);

        vec.resize(2);
        require(vec.size() == 2);
        require(vec[0] == 7 && vec[1] == 7);
    }

    { // copy and move semantics
        // copy construction from inline buffer
        SmallVector<std::string, 4> src = {"a", "b", "c"};
        SmallVector<std::string, 4> copy1(src);
        require(copy1 == src);
        require(copy1.is_on_heap() == false);

        // copy assignment
        SmallVector<std::string, 4> copy2;
        copy2 = src;
        require(copy2 == src);

        // move construction from inline buffer
        SmallVector<std::string, 4> moved1(std::move(src));
        require(moved1.size() == 3);
        require(moved1[0] == "a" && moved1[2] == "c");
        require(src.empty());

        // copy construction from heap
        SmallVector<std::string, 2> heap_src = {"a", "b", "c", "d"};
        require(heap_src.is_on_heap() == true);
        SmallVector<std::string, 2> copy3(heap_src);
        require(copy3 == heap_src);
        require(copy3.is_on_heap() == true);

        // move construction from heap (steals the buffer)
        SmallVector<std::string, 2> moved2(std::move(heap_src));
        require(moved2.size() == 4);
        require(moved2.is_on_heap() == true);
        require(heap_src.empty());

        // move assignment
        SmallVector<std::string, 4> move_dst;
        move_dst = std::move(moved1);
        require(move_dst.size() == 3);
        require(move_dst[0] == "a");
        require(moved1.empty());
    }

    { // erase and insert edge cases
        // erase first element
        SmallVector<int, 8> vec = {1, 2, 3, 4, 5};
        vec.erase(vec.begin());
        require(vec.size() == 4);
        require(vec[0] == 2);

        // erase last element
        vec.erase(vec.end() - 1);
        require(vec.size() == 3);
        require(vec.back() == 4);

        // erase all elements one by one
        while (!vec.empty()) {
            vec.erase(vec.begin());
        }
        require(vec.empty());

        // insert range into middle
        SmallVector<int, 8> dst = {1, 2, 5, 6};
        SmallVector<int, 8> mid = {3, 4};
        dst.insert(dst.begin() + 2, mid.begin(), mid.end());
        require(dst.size() == 6);
        require(dst[0] == 1 && dst[1] == 2 && dst[2] == 3);
        require(dst[3] == 4 && dst[4] == 5 && dst[5] == 6);

        // insert at end
        dst.insert(dst.end(), mid.begin(), mid.end());
        require(dst.size() == 8);
        require(dst[6] == 3 && dst[7] == 4);
    }

    { // comparison with different sizes
        const SmallVector<int, 4> shorter = {1, 2, 3};
        const SmallVector<int, 4> longer = {1, 2, 3, 4};
        const SmallVector<int, 4> same = {1, 2, 3};

        require(shorter != longer);
        require(shorter == same);
        require(shorter < longer);
        require(longer > shorter);
        require(shorter <= same);
        require(shorter >= same);
    }

    printf("OK\n");
}
