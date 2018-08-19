#pragma once

#include <vector>

namespace Ray {
template <typename RandFunc>
void Shuffle(uint16_t *arr, int count, RandFunc &&rand_func) {
    for (int i = 0; i < count; i++) {
        int other;
        uint32_t thres = (~((uint32_t)count) + 1u) % count;
        while (true) {
            uint32_t r = rand_func();
            if (r >= thres) {
                other = r % count;
                break;
            }
        }
        std::swap(arr[i], arr[other]);
    }
}

template <typename RandFunc>
std::vector<uint16_t> ComputeRadicalInversePermutations(const int *primes, int primes_count, RandFunc &&rand_func) {
    std::vector<uint16_t> perms;

    int perm_array_size = 0;
    for (int i = 0; i < primes_count; i++) perm_array_size += primes[i];

    perms.resize(perm_array_size);

    uint16_t *p = &perms[0];
    for (int i = 0; i < primes_count; i++) {
        for (int j = 0; j < primes[i]; j++) {
            p[j] = j;
        }

        Shuffle(p, primes[i], std::move(rand_func));

        p += primes[i];
    }

    return perms;
}

template <int base, typename Real = float>
Real RadicalInverse(uint64_t a) {
    const Real inv_base = Real(1) / base;
    uint64_t reversed_digits = 0;
    Real inv_base_n = 1;
    while (a) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_n *= inv_base;
        a = next;
    }
    return std::min(reversed_digits * inv_base_n, Real(1) - std::numeric_limits<Real>::epsilon());
}

template <int base, typename Real = float>
Real ScrambledRadicalInverse(const uint16_t *perm, uint64_t a) {
    const Real inv_base = Real(1) / base;
    uint64_t reversed_digits = 0;
    Real inv_base_n = 1;
    while (a) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversed_digits = reversed_digits * base + perm[digit];
        inv_base_n *= inv_base;
        a = next;
    }
    return std::min(inv_base_n * (reversed_digits + inv_base * perm[0] / (1 - inv_base)),
                    Real(1) - std::numeric_limits<Real>::epsilon());
}
}