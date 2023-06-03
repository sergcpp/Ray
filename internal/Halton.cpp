#include "Halton.h"

#include <functional>
#include <random>

#include "UniformIntDistribution.h"

std::vector<uint16_t> Ray::ComputeRadicalInversePermutations(const int* primes, const int primes_count) {
    auto rand_func = std::bind(UniformIntDistribution<uint32_t>(), std::mt19937(0));
    return ComputeRadicalInversePermutations(primes, primes_count, rand_func);
}
