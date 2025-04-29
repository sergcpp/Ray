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

    printf("OK\n");
}