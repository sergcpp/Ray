#pragma once

#include <limits>
#include <utility>

// Portable uniform int distribution. Taken from:
// https://stackoverflow.com/questions/26538627/c11-cross-compiler-standard-library-random-distribution-reproducibility
template <class IntType = int> class UniformIntDistribution {
  public:
    // types
    typedef IntType result_type;
    typedef std::pair<int, int> param_type;

    // constructors and reset functions
    explicit UniformIntDistribution(IntType a = 0, IntType b = std::numeric_limits<IntType>::max());
    explicit UniformIntDistribution(const param_type &parm);
    void reset();

    // generating functions
    template <class URNG> result_type operator()(URNG &g);
    template <class URNG> result_type operator()(URNG &g, const param_type &parm);

    // property functions
    result_type a() const;
    result_type b() const;
    param_type param() const;
    void param(const param_type &parm);
    result_type min() const;
    result_type max() const;

  private:
    typedef typename std::make_unsigned<IntType>::type diff_type;

    IntType lower;
    IntType upper;
};

template <class IntType> UniformIntDistribution<IntType>::UniformIntDistribution(IntType a, IntType b) {
    param({a, b});
}

template <class IntType> UniformIntDistribution<IntType>::UniformIntDistribution(const param_type &parm) {
    param(parm);
}

template <class IntType> void UniformIntDistribution<IntType>::reset() {}

template <class IntType>
template <class URNG>
auto UniformIntDistribution<IntType>::operator()(URNG &g) -> result_type {
    return operator()(g, param());
}

template <class IntType>
template <class URNG>
auto UniformIntDistribution<IntType>::operator()(URNG &g, const param_type &parm) -> result_type {
    diff_type diff = (diff_type)parm.second - (diff_type)parm.first + 1;
    if (diff == 0) {
        // If the +1 overflows we are using the full range, just return g()
        return g();
    }

    diff_type badDistLimit = std::numeric_limits<diff_type>::max() / diff;
    do {
        diff_type generatedRand = g();

        if (generatedRand / diff < badDistLimit) {
            return (IntType)((generatedRand % diff) + (diff_type)parm.first);
        }
    } while (true);
}

template <class IntType> auto UniformIntDistribution<IntType>::a() const -> result_type { return lower; }

template <class IntType> auto UniformIntDistribution<IntType>::b() const -> result_type { return upper; }

template <class IntType> auto UniformIntDistribution<IntType>::param() const -> param_type { return {lower, upper}; }

template <class IntType> void UniformIntDistribution<IntType>::param(const param_type &parm) {
    std::tie(lower, upper) = parm;
    if (upper < lower) {
        throw std::exception();
    }
}

template <class IntType> auto UniformIntDistribution<IntType>::min() const -> result_type { return lower; }

template <class IntType> auto UniformIntDistribution<IntType>::max() const -> result_type { return upper; }