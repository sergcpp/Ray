#include "test_common.h"

#include <vector>

#include "../internal/HashSet32.h"

struct alignas(64) TestHashSetStruct {
    uint32_t val;

    TestHashSetStruct(const uint32_t _val) : val(_val) { require((uintptr_t(this) % 64) == 0); }
    TestHashSetStruct(const TestHashSetStruct &rhs) : val(rhs.val) { require((uintptr_t(this) % 64) == 0); }
};

template <> class Ray::Hash<TestHashSetStruct> {
  public:
    uint32_t operator()(const TestHashSetStruct &s) const { return s.val; }
};

template <> class Ray::Equal<TestHashSetStruct> {
  public:
    bool operator()(const TestHashSetStruct &k1, const TestHashSetStruct &k2) const { return k1.val == k2.val; }
};

void test_hashset() {
    using namespace Ray;

    printf("Test hashset            | ");

    { // Basic test
        HashSet32<int> cont;

        for (int i = 0; i < 100; i++) {
            require(cont.Insert(12));
            require(cont.Insert(42));
            require(cont.Insert(15));
            require(cont.Insert(10));
            require(cont.Insert(-42));
            require(cont.Insert(144));

            require(!cont.Insert(12));
            require(!cont.Insert(42));
            require(!cont.Insert(15));
            require(!cont.Insert(10));
            require(!cont.Insert(-42));
            require(!cont.Insert(144));

            int *p_val = nullptr;

            p_val = cont.Find(12);
            require(p_val && *p_val == 12);

            p_val = cont.Find(12);
            require(p_val && *p_val == 12);

            p_val = cont.Find(42);
            require(p_val && *p_val == 42);

            p_val = cont.Find(15);
            require(p_val && *p_val == 15);

            p_val = cont.Find(10);
            require(p_val && *p_val == 10);

            p_val = cont.Find(-42);
            require(p_val && *p_val == -42);

            p_val = cont.Find(144);
            require(p_val && *p_val == 144);

            require(cont.Erase(12));
            require(cont.Erase(42));
            require(cont.Erase(15));
            require(cont.Erase(10));
            require(cont.Erase(-42));
            require(cont.Erase(144));
        }
    }

    { // Initializer list
        HashSet32<int> cont{{12, 42, 15, 10, -42, 144}};

        require(!cont.Insert(12));
        require(!cont.Insert(42));
        require(!cont.Insert(15));
        require(!cont.Insert(10));
        require(!cont.Insert(-42));
        require(!cont.Insert(144));

        int *p_val = nullptr;

        p_val = cont.Find(12);
        require(p_val && *p_val == 12);

        p_val = cont.Find(12);
        require(p_val && *p_val == 12);

        p_val = cont.Find(42);
        require(p_val && *p_val == 42);

        p_val = cont.Find(15);
        require(p_val && *p_val == 15);

        p_val = cont.Find(10);
        require(p_val && *p_val == 10);

        p_val = cont.Find(-42);
        require(p_val && *p_val == -42);

        p_val = cont.Find(144);
        require(p_val && *p_val == 144);

        require(cont.Erase(12));
        require(cont.Erase(42));
        require(cont.Erase(15));
        require(cont.Erase(10));
        require(cont.Erase(-42));
        require(cont.Erase(144));
    }

    { // Alignemnt test
        HashSet32<TestHashSetStruct> cont;

        for (int i = 0; i < 100; i++) {
            require(cont.Insert(12));
            require(cont.Insert(42));
            require(cont.Insert(15));
            require(cont.Insert(10));
            require(cont.Insert(-42));
            require(cont.Insert(144));

            TestHashSetStruct *p_val = nullptr;

            p_val = cont.Find(12);
            require(p_val && p_val->val == 12);

            p_val = cont.Find(12);
            require(p_val && p_val->val == 12);

            p_val = cont.Find(42);
            require(p_val && p_val->val == 42);

            p_val = cont.Find(15);
            require(p_val && p_val->val == 15);

            p_val = cont.Find(10);
            require(p_val && p_val->val == 10);

            p_val = cont.Find(-42);
            require(p_val && p_val->val == -42);

            p_val = cont.Find(144);
            require(p_val && p_val->val == 144);

            require(cont.Erase(12));
            require(cont.Erase(42));
            require(cont.Erase(15));
            require(cont.Erase(10));
            require(cont.Erase(-42));
            require(cont.Erase(144));
        }
    }

    { // Test with reallocation
        HashSet32<std::string> cont(16);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            cont.Insert(std::move(key));
        }

        require(cont.size() == 100000);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            std::string *_key = cont.Find(key);
            require(_key && *_key == key);
            require(cont.Erase(key));
        }

        require(cont.size() == 0);
    }

    { // Test iteration
        HashSet32<std::string> cont(16);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            cont.Insert(std::move(key));
        }

        require(cont.size() == 100000);

        { // const iterator
            int values_count = 0;
            for (auto it = cont.cbegin(); it != cont.cend(); ++it) {
                // require(it->key == std::to_string(it->val));
                values_count++;
            }

            require(values_count == 100000);
        }

        { // normal iterator
            int values_count = 0;
            for (auto it = cont.begin(); it != cont.end(); ++it) {
                // require(it->key == std::to_string(it->val));
                values_count++;
            }

            require(values_count == 100000);
        }
    }

    { // Initializer list
    }

    printf("OK\n");
}