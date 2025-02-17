#include "test_common.h"

#include <array>
#include <vector>

#include "../Span.h"

void test_span() {
    using namespace Ray;

    printf("Test span               | ");

    { // constructor
        {
            Span<const int> s;
            require(s.empty());
            require(s.size() == 0);
            require(s.data() == nullptr);
        }
        {
            const int arr[5] = {0, 1, 2, 3, 4};
            Span<const int> s(std::begin(arr), 5);
            require(s.data() == std::begin(arr));
            require(s.size() == 5);
            require(!s.empty());
        }
        {
            const int arr[5] = {0, 1, 2, 3, 4};
            Span<const int> s(std::begin(arr), std::end(arr));
            require(s.data() == std::begin(arr));
            require(s.size() == 5);
            require(!s.empty());
        }
        {
            const int arr[5] = {0, 1, 2, 3, 4};
            Span<const int> s(arr);
            require(s.data() == std::begin(arr));
            require(s.size() == 5);
            require(s.data()[2] == arr[2]);
            require(!s.empty());
        }
        {
            std::array<int, 5> arr = {0, 1, 2, 3, 4};
            Span<int> s(arr);
            require(s.data() == &*std::begin(arr));
            require(s.size() == 5);
            require(s.data()[2] == arr.data()[2]);
            require(!s.empty());
        }
        {
            const std::array<int, 5> arr = {0, 1, 2, 3, 4};
            Span<const int> s(arr);
            require(s.data() == &*std::begin(arr));
            require(s.size() == 5);
            require(s.data()[2] == arr.data()[2]);
            require(!s.empty());
        }
        {
            const std::array<const int, 5> arr = {0, 1, 2, 3, 4};
            const Span<const int> s(arr);
            require(s.data() == &*std::begin(arr));
            require(s.size() == 5);
            require(s.data()[2] == arr.data()[2]);
        }
        { // const removal
            int val = 1;
            std::array<int *, 5> arr_values = {&val, &val, &val, &val, &val};
            Span<const int *const> test0(arr_values);
            int sum = 0;
            for (const int *i : test0) {
                sum += *i;
            }
            require(sum == 5);
        }
        { // const removal
            int val = 1;
            std::vector<int *> arr_values = {&val, &val, &val, &val, &val};
            Span<const int *const> test0(arr_values);
            int sum = 0;
            for (const int *i : test0) {
                sum += *i;
            }
            require(sum == 5);
        }
        {
            struct foo {};
            auto f1 = [](Span<const foo *>) {};
            auto f2 = [](Span<const foo *const>) {};

            foo *pFoo = nullptr;
            std::array<const foo *, 1> foos = {pFoo};

            f1(foos);
            f2(foos);
        }
        {
            struct foo {};
            auto f1 = [](Span<const foo *>) {};
            auto f2 = [](Span<const foo *const>) {};

            foo *pFoo = nullptr;
            std::vector<const foo *> foos = {pFoo};

            f1(foos);
            f2(foos);
        }
    }
    { // size bytes
        {
            int arr[5] = {0, 1, 2, 3, 4};
            Span<int> s(arr);
            require(s.size_bytes() == sizeof(arr));
            require(s.size_bytes() == (5 * sizeof(int)));
        }
        {
            float arr[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
            Span<float> s(arr);
            require(s.size_bytes() == sizeof(arr));
            require(s.size_bytes() == (8 * sizeof(float)));
        }
        {
            int64_t arr[5] = {0, 1, 2, 3, 4};
            Span<int64_t> s(arr);
            require(s.size_bytes() == sizeof(arr));
            require(s.size_bytes() == (5 * sizeof(int64_t)));
        }
    }
    { // element access
        int arr[5] = {0, 1, 2, 3, 4};
        Span<int> s(arr);

        require(s.front() == 0);
        require(s.back() == 4);

        require(s[0] == 0);
        require(s[1] == 1);
        require(s[2] == 2);
        require(s[3] == 3);
        require(s[4] == 4);

        require(s(0) == 0);
        require(s(1) == 1);
        require(s(2) == 2);
        require(s(3) == 3);
        require(s(4) == 4);
    }
    { // iterators
        int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        Span<int> s(arr);

        { // ranged-for
            int *p_begin = arr;
            for (int &e : arr) {
                require(e == *p_begin++);
            }
        }
        {
            auto test_iterator_begin = [&](Span<int>::const_iterator p) {
                require(*p++ == 0);
                require(*p++ == 1);
                require(*p++ == 2);
                require(*p++ == 3);
                require(*p++ == 4);
                require(*p++ == 5);
                require(*p++ == 6);
                require(*p++ == 7);
                require(*p++ == 8);
                require(*p++ == 9);
            };

            auto test_iterator_end = [&](Span<int>::const_iterator p) {
                p--;
                require(*p-- == 9);
                require(*p-- == 8);
                require(*p-- == 7);
                require(*p-- == 6);
                require(*p-- == 5);
                require(*p-- == 4);
                require(*p-- == 3);
                require(*p-- == 2);
                require(*p-- == 1);
                require(*p-- == 0);
            };

            test_iterator_begin(s.begin());
            test_iterator_begin(s.cbegin());
            test_iterator_end(s.end());
            test_iterator_end(s.cend());
        }
        {
            auto test_reverse_iterator_begin = [&](Span<int>::const_reverse_iterator p) {
                require(*p++ == 9);
                require(*p++ == 8);
                require(*p++ == 7);
                require(*p++ == 6);
                require(*p++ == 5);
                require(*p++ == 4);
                require(*p++ == 3);
                require(*p++ == 2);
                require(*p++ == 1);
                require(*p++ == 0);
            };

            auto test_reverse_iterator_end = [&](Span<int>::const_reverse_iterator p) {
                p--;
                require(*p-- == 0);
                require(*p-- == 1);
                require(*p-- == 2);
                require(*p-- == 3);
                require(*p-- == 4);
                require(*p-- == 5);
                require(*p-- == 6);
                require(*p-- == 7);
                require(*p-- == 8);
                require(*p-- == 9);
            };

            test_reverse_iterator_begin(s.rbegin());
            test_reverse_iterator_begin(s.crbegin());
            test_reverse_iterator_end(s.rend());
            test_reverse_iterator_end(s.crend());
        }
    }
    { // copy assignment
        int arr[5] = {0, 1, 2, 3, 4};
        Span<int> s(arr);
        Span<int> sc = s;

        require(s[0] == sc[0]);
        require(s[1] == sc[1]);
        require(s[2] == sc[2]);
        require(s[3] == sc[3]);
        require(s[4] == sc[4]);

        require(s(0) == sc(0));
        require(s(1) == sc(1));
        require(s(2) == sc(2));
        require(s(3) == sc(3));
        require(s(4) == sc(4));
    }
    { // container conversion
        {
            std::vector<int> v = {0, 1, 2, 3, 4, 5};
            Span<const int> s(v);

            require(s.size() == v.size());
            require(s.data() == v.data());

            require(s[0] == v[0]);
            require(s[1] == v[1]);
            require(s[2] == v[2]);
            require(s[3] == v[3]);
            require(s[4] == v[4]);
            require(s[5] == v[5]);
        }
        {
            const std::vector<int> v = {0, 1, 2, 3, 4, 5};
            Span<const int> s(v);

            require(s.size() == v.size());
            require(s.data() == v.data());

            require(s[0] == v[0]);
            require(s[1] == v[1]);
            require(s[2] == v[2]);
            require(s[3] == v[3]);
            require(s[4] == v[4]);
            require(s[5] == v[5]);
        }
        {
            auto f1 = [](Span<int> s) { return s.size(); };
            auto f2 = [](Span<const int> s) { return s.size(); };

            {
                std::vector<int> v = {0, 1, 2, 3, 4, 5};

                require(f1(v) == v.size());
                require(f2(v) == v.size());
            }
            {
                int a[] = {0, 1, 2, 3, 4, 5};

                require(f1(a) == 6);
                require(f2(a) == 6);
            }
        }
    }
    { // comparison
        int arr1[5] = {0, 1, 2, 3, 4};
        int arr2[8] = {0, 1, 2, 3, 4, 5, 6, 7};

        Span<int> s1 = arr1;
        Span<const int> s2 = arr2;
        Span<int> s3 = arr2;
        require(s2 == s3);
        require(s1 != s2);
        require(s1 < s2);
        require(s1 <= s2);
        require(s2 > s1);
        require(s2 >= s1);
    }
    { // subviews
        int arr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        {
            Span<int> s = arr1;
            Span<int> first_span = s.first(4);
            require(first_span.size() == 4);
            require(first_span[0] == 0);
            require(first_span[1] == 1);
            require(first_span[2] == 2);
            require(first_span[3] == 3);
        }
        {
            Span<int> s = arr1;
            Span<int> last_span = s.last(4);
            require(last_span.size() == 4);
            require(last_span[0] == 6);
            require(last_span[1] == 7);
            require(last_span[2] == 8);
            require(last_span[3] == 9);
        }
        { // empty range
            Span<int> s{};

            Span<int> dynamic_span;
            require(dynamic_span.empty());
            dynamic_span = s.first(0);
            require(dynamic_span.empty());
            dynamic_span = s.last(0);
            require(dynamic_span.empty());
        }
        { // subspan: full range
            Span<int> s = arr1;

            Span<int> dynamic_span = s.subspan(0, s.size());
            require(dynamic_span.size() == 10);
            require(dynamic_span[0] == 0);
            require(dynamic_span[1] == 1);
            require(dynamic_span[8] == 8);
            require(dynamic_span[9] == 9);
        }
        { // subspan: subrange
            Span<int> s = arr1;

            Span<int> dynamic_span = s.subspan(3, 4);
            require(dynamic_span.size() == 4);
            require(dynamic_span[0] == 3);
            require(dynamic_span[1] == 4);
            require(dynamic_span[2] == 5);
            require(dynamic_span[3] == 6);
        }
        { // subspan: default count
            Span<int> s = arr1;

            Span<int> dynamic_span = s.subspan(3);
            require(dynamic_span.size() == 7);
            require(dynamic_span[0] == 3);
            require(dynamic_span[1] == 4);
            require(dynamic_span[5] == 8);
            require(dynamic_span[6] == 9);
        }
    }

    printf("OK\n");
}
