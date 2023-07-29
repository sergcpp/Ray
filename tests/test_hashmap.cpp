#include "test_common.h"

#include <vector>

#include "../internal/HashMap32.h"

void test_hashmap() {
    printf("Test hashmap            | ");

    { // Basic test
        Ray::HashMap32<int, double> cont;

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
        Ray::HashMap32<std::string, int> cont(16);

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
        Ray::HashMap32<std::string, int> cont(16);

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