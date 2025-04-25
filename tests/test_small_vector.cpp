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

    printf("OK\n");
}
