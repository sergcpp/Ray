#include "test_common.h"

#include <vector>

#include "../internal/HashMap32.h"

struct alignas(64) TestHashMapStruct {
    uint32_t val;

    TestHashMapStruct(const uint32_t _val) : val(_val) { require((uintptr_t(this) % 64) == 0); }
    TestHashMapStruct(const TestHashMapStruct &rhs) : val(rhs.val) { require((uintptr_t(this) % 64) == 0); }
};

template <> class Ray::Hash<TestHashMapStruct> {
  public:
    uint32_t operator()(const TestHashMapStruct &s) const { return s.val; }
};

template <> class Ray::Equal<TestHashMapStruct> {
  public:
    bool operator()(const TestHashMapStruct &k1, const TestHashMapStruct &k2) const { return k1.val == k2.val; }
};

void test_hashmap() {
    using namespace Ray;

    printf("Test hashmap            | ");

    { // Basic test
        HashMap32<int, double> cont;

        for (int i = 0; i < 100; i++) {
            require(cont.Insert(12, 12.0));
            require(cont.Insert(42, 16.5));
            require(cont.Insert(15, 11.15));
            require(cont.Insert(10, 18.53));
            require(cont.Insert(-42, -16.5));
            require(cont.Insert(144, 916.0));

            require(!cont.Insert(12, 0.0));
            require(!cont.Insert(42, 0.0));
            require(!cont.Insert(15, 0.0));
            require(!cont.Insert(10, 0.0));
            require(!cont.Insert(-42, 0.0));
            require(!cont.Insert(144, 0.0));

            cont[15] = 17.894;
            cont[27] = -13.0;

            double *p_val = nullptr;

            p_val = cont.Find(12);
            require(p_val && *p_val == 12.0);

            p_val = cont.Find(12);
            require(p_val && *p_val == 12.0);

            p_val = cont.Find(42);
            require(p_val && *p_val == 16.5);

            p_val = cont.Find(15);
            require(p_val && *p_val == 17.894);

            p_val = cont.Find(10);
            require(p_val && *p_val == 18.53);

            p_val = cont.Find(-42);
            require(p_val && *p_val == -16.5);

            p_val = cont.Find(144);
            require(p_val && *p_val == 916.0);

            p_val = cont.Find(27);
            require(p_val && *p_val == -13.0);

            require(cont.Erase(12));
            require(cont.Erase(42));
            require(cont.Erase(15));
            require(cont.Erase(10));
            require(cont.Erase(-42));
            require(cont.Erase(144));
        }
    }

    { // Initializer list
        HashMap32<int, double> cont{{12, 12.0}, {42, 16.5}, {15, 11.15}, {10, 18.53}, {-42, -16.5}, {144, 916.0}};

        require(!cont.Insert(12, 0.0));
        require(!cont.Insert(42, 0.0));
        require(!cont.Insert(15, 0.0));
        require(!cont.Insert(10, 0.0));
        require(!cont.Insert(-42, 0.0));
        require(!cont.Insert(144, 0.0));

        cont[15] = 17.894;
        cont[27] = -13.0;

        double *p_val = nullptr;

        p_val = cont.Find(12);
        require(p_val && *p_val == 12.0);

        p_val = cont.Find(12);
        require(p_val && *p_val == 12.0);

        p_val = cont.Find(42);
        require(p_val && *p_val == 16.5);

        p_val = cont.Find(15);
        require(p_val && *p_val == 17.894);

        p_val = cont.Find(10);
        require(p_val && *p_val == 18.53);

        p_val = cont.Find(-42);
        require(p_val && *p_val == -16.5);

        p_val = cont.Find(144);
        require(p_val && *p_val == 916.0);

        p_val = cont.Find(27);
        require(p_val && *p_val == -13.0);

        require(cont.Erase(12));
        require(cont.Erase(42));
        require(cont.Erase(15));
        require(cont.Erase(10));
        require(cont.Erase(-42));
        require(cont.Erase(144));
    }

    { // Alignment test
        HashMap32<TestHashMapStruct, double> cont;

        for (int i = 0; i < 100; i++) {
            require(cont.Insert(12, 12.0));
            require(cont.Insert(42, 16.5));
            require(cont.Insert(15, 11.15));
            require(cont.Insert(10, 18.53));
            require(cont.Insert(-42, -16.5));
            require(cont.Insert(144, 916.0));

            require(!cont.Insert(12, 0.0));
            require(!cont.Insert(42, 0.0));
            require(!cont.Insert(15, 0.0));
            require(!cont.Insert(10, 0.0));
            require(!cont.Insert(-42, 0.0));
            require(!cont.Insert(144, 0.0));

            cont[15] = 17.894;
            cont[27] = -13.0;

            double *p_val = nullptr;

            p_val = cont.Find(12);
            require(p_val && *p_val == 12.0);

            p_val = cont.Find(12);
            require(p_val && *p_val == 12.0);

            p_val = cont.Find(42);
            require(p_val && *p_val == 16.5);

            p_val = cont.Find(15);
            require(p_val && *p_val == 17.894);

            p_val = cont.Find(10);
            require(p_val && *p_val == 18.53);

            p_val = cont.Find(-42);
            require(p_val && *p_val == -16.5);

            p_val = cont.Find(144);
            require(p_val && *p_val == 916.0);

            p_val = cont.Find(27);
            require(p_val && *p_val == -13.0);

            require(cont.Erase(12));
            require(cont.Erase(42));
            require(cont.Erase(15));
            require(cont.Erase(10));
            require(cont.Erase(-42));
            require(cont.Erase(144));
        }
    }

    { // Test with reallocation
        HashMap32<std::string, int> cont(16);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            cont[key] = i;
        }

        require(cont.size() == 100000);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            require(cont[key] == i);
            require(cont.Erase(key));
        }

        require(cont.size() == 0);
    }

    { // Test iteration
        HashMap32<std::string, int> cont(16);

        for (int i = 0; i < 100000; i++) {
            std::string key = std::to_string(i);
            cont[key] = i;
        }

        require(cont.size() == 100000);

        { // const iterator
            int values_count = 0;
            for (auto it = cont.cbegin(); it != cont.cend(); ++it) {
                require(it->key == std::to_string(it->val));
                values_count++;
            }

            require(values_count == 100000);
        }

        { // normal iterator
            int values_count = 0;
            for (auto it = cont.begin(); it != cont.end(); ++it) {
                require(it->key == std::to_string(it->val));
                values_count++;
            }

            require(values_count == 100000);
        }
    }

    { // GetOrNull
        HashMap32<int, double> cont;

        require(cont.Insert(42, 16.5));
        require(cont.Insert(100, 1.0));

        // occupied slots are visible via index
        for (auto it = cont.begin(); it != cont.end(); ++it) {
            const uint32_t idx = it.index();
            const HashMap32<int, double>::Node *node = cont.GetOrNull(idx);
            require(node != nullptr);
            require(node->key == it->key);
            require(node->val == it->val);
        }

        // out-of-range returns nullptr
        require(cont.GetOrNull(cont.capacity()) == nullptr);
        require(cont.GetOrNull(0xffffffff) == nullptr);

        // erased slot returns nullptr
        const uint32_t first_idx = cont.begin().index();
        const int first_key = cont.begin()->key;
        require(cont.Erase(first_key));
        require(cont.GetOrNull(first_idx) == nullptr);
    }

    { // Hash collisions / linear probing
        struct ZeroHash {
            uint32_t operator()(int) const { return 0; }
        };

        HashMap32<int, int, ZeroHash> cont;
        const int N = 200;
        for (int i = 0; i < N; i++) {
            require(cont.Insert(i, i * 10));
        }
        require(cont.size() == uint32_t(N));

        // all values findable despite every key mapping to the same bucket
        for (int i = 0; i < N; i++) {
            int *v = cont.Find(i);
            require(v && *v == i * 10);
        }

        // erase every other entry; surviving entries still findable
        for (int i = 0; i < N; i += 2) {
            require(cont.Erase(i));
        }
        require(cont.size() == uint32_t(N / 2));

        for (int i = 0; i < N; i++) {
            int *v = cont.Find(i);
            if (i % 2 == 0) {
                require(v == nullptr);
            } else {
                require(v && *v == i * 10);
            }
        }
    }

    { // Erase invariants
        HashMap32<int, int> cont;

        // erase reduces size and makes key unfindable
        require(cont.Insert(7, 77));
        require(cont.size() == 1u);
        require(cont.Erase(7));
        require(cont.size() == 0u);
        require(cont.Find(7) == nullptr);

        // erase on missing key returns false
        require(!cont.Erase(7));
        require(!cont.Erase(999));

        // iterator erase: remove all even keys
        for (int i = 0; i < 10; i++) {
            cont.Insert(i, i);
        }
        for (auto it = cont.begin(); it != cont.end();) {
            it = (it->key % 2 == 0) ? cont.erase(it) : ++it;
        }
        require(cont.size() == 5u);
        for (int i = 0; i < 10; i++) {
            int *v = cont.Find(i);
            if (i % 2 == 0) {
                require(v == nullptr);
            } else {
                require(v && *v == i);
            }
        }
    }

    { // Set, clear, reserve
        HashMap32<int, int> cont;

        cont.reserve(64);
        require(cont.capacity() >= 64u);
        require(cont.size() == 0u);

        // Set inserts when key is absent
        cont.Set(1, 10);
        require(cont.size() == 1u);
        int *v = cont.Find(1);
        require(v && *v == 10);

        // Set updates when key is present
        cont.Set(1, 20);
        require(cont.size() == 1u);
        v = cont.Find(1);
        require(v && *v == 20);

        for (int i = 2; i <= 50; i++) {
            cont.Set(i, i * 3);
        }
        require(cont.size() == 50u);

        cont.clear();
        require(cont.size() == 0u);
        require(cont.Find(1) == nullptr);

        // map is usable after clear
        require(cont.Insert(99, 999));
        require(cont.size() == 1u);
        v = cont.Find(99);
        require(v && *v == 999);
    }

    { // Move semantics
        HashMap32<int, int> a;
        for (int i = 0; i < 10; i++) {
            a.Insert(i, i * 2);
        }

        // move constructor transfers all entries, leaves source empty
        HashMap32<int, int> b(std::move(a));
        require(a.size() == 0u);
        require(b.size() == 10u);
        for (int i = 0; i < 10; i++) {
            int *v = b.Find(i);
            require(v && *v == i * 2);
        }

        // move assignment replaces target, leaves source empty
        HashMap32<int, int> c;
        c.Insert(100, 200);
        c = std::move(b);
        require(b.size() == 0u);
        require(c.size() == 10u);
        for (int i = 0; i < 10; i++) {
            int *v = c.Find(i);
            require(v && *v == i * 2);
        }
        require(c.Find(100) == nullptr);

        // Insert(K&&, V&&) moves the key and value
        HashMap32<std::string, std::string> d;
        std::string key = "hello", val = "world";
        require(d.Insert(std::move(key), std::move(val)));
        require(key.empty());
        require(val.empty());
        auto *sv = d.Find(std::string("hello"));
        require(sv && *sv == "world");
    }

    printf("OK\n");
}