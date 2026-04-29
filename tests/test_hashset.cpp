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

    { // GetOrNull
        HashSet32<int> cont;

        require(cont.Insert(42));
        require(cont.Insert(100));

        // occupied slots are visible via index
        for (auto it = cont.begin(); it != cont.end(); ++it) {
            const uint32_t idx = it.index();
            const HashSet32<int>::Node *node = cont.GetOrNull(idx);
            require(node != nullptr);
            require(node->key == *it);
        }

        // out-of-range returns nullptr
        require(cont.GetOrNull(cont.capacity()) == nullptr);
        require(cont.GetOrNull(0xffffffff) == nullptr);

        // erased slot returns nullptr
        const uint32_t first_idx = cont.begin().index();
        const int first_key = *cont.begin();
        require(cont.Erase(first_key));
        require(cont.GetOrNull(first_idx) == nullptr);
    }

    { // Hash collisions / linear probing
        struct ZeroHash {
            uint32_t operator()(int) const { return 0; }
        };

        HashSet32<int, ZeroHash> cont;
        const int N = 200;
        for (int i = 0; i < N; i++) {
            require(cont.Insert(i));
        }
        require(cont.size() == uint32_t(N));

        // all values findable despite every key mapping to the same bucket
        for (int i = 0; i < N; i++) {
            require(cont.Find(i) != nullptr);
        }

        // erase every other entry; surviving entries still findable
        for (int i = 0; i < N; i += 2) {
            require(cont.Erase(i));
        }
        require(cont.size() == uint32_t(N / 2));

        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                require(cont.Find(i) == nullptr);
            } else {
                require(cont.Find(i) != nullptr);
            }
        }
    }

    { // Erase invariants
        HashSet32<int> cont;

        // erase reduces size and makes key unfindable
        require(cont.Insert(7));
        require(cont.size() == 1u);
        require(cont.Erase(7));
        require(cont.size() == 0u);
        require(cont.Find(7) == nullptr);

        // erase on missing key returns false
        require(!cont.Erase(7));
        require(!cont.Erase(999));

        // iterator erase: remove all even keys
        for (int i = 0; i < 10; i++) {
            cont.Insert(i);
        }
        for (auto it = cont.begin(); it != cont.end();) {
            it = (*it % 2 == 0) ? cont.erase(it) : ++it;
        }
        require(cont.size() == 5u);
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                require(cont.Find(i) == nullptr);
            } else {
                require(cont.Find(i) != nullptr);
            }
        }
    }

    { // clear, reserve
        HashSet32<int> cont;

        cont.reserve(64);
        require(cont.capacity() >= 64u);
        require(cont.size() == 0u);
        require(cont.empty());

        for (int i = 0; i < 50; i++) {
            cont.Insert(i);
        }
        require(cont.size() == 50u);
        require(!cont.empty());

        cont.clear();
        require(cont.size() == 0u);
        require(cont.empty());
        require(cont.Find(1) == nullptr);

        // set is usable after clear
        require(cont.Insert(99));
        require(cont.size() == 1u);
        require(cont.Find(99) != nullptr);
    }

    { // Move semantics
        HashSet32<int> a;
        for (int i = 0; i < 10; i++) {
            a.Insert(i);
        }

        // move constructor transfers all entries, leaves source empty
        HashSet32<int> b(std::move(a));
        require(a.size() == 0u);
        require(b.size() == 10u);
        for (int i = 0; i < 10; i++) {
            require(b.Find(i) != nullptr);
        }

        // move assignment replaces target, leaves source empty
        HashSet32<int> c;
        c.Insert(100);
        c = std::move(b);
        require(b.size() == 0u);
        require(c.size() == 10u);
        for (int i = 0; i < 10; i++) {
            require(c.Find(i) != nullptr);
        }
        require(c.Find(100) == nullptr);

        // Insert(K&&) moves the key
        HashSet32<std::string> d;
        std::string key = "hello";
        require(d.Insert(std::move(key)));
        require(key.empty());
        require(d.Find(std::string("hello")) != nullptr);
    }

    printf("OK\n");
}