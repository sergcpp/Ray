/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * These functions generate PMJ(0,2) sequences from
 * "Progressive Multi-Jittered Sample Sequences", Christensen et al. 2018, using
 * the algorithm from "Efficient Generation of Points that Satisfy
 * Two-Dimensional Elementary Intervals", Matt Pharr, 2019.
 *
 * Thanks to Matt's paper, the non-best-candidate sampling is quite fast. On my
 * 2017 Macbook Pro, 65536 samples takes <50ms, i.e. it generates 1.43 million
 * samples/sec, with -O3 compilation. Best candidate sampling is slower at
 * ~500,000 samples/sec with 10 candidates. If you want to use the
 * Best-Candidate samples in a production raytracer, for example, probably
 * better to precompute a bunch of tables and do lookups into them. Also worth
 * noting that the pmjbn algorithm (in pmj.cc) has much better blue-noise
 * characteristics than pmj02bn.
 */
#ifndef SAMPLE_GENERATION_PMJ02_H_
#define SAMPLE_GENERATION_PMJ02_H_

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "sample_generation/util.h"

namespace pmj {

// Generates progressive multi-jittered (0,2) samples WITHOUT blue noise
// properties. Takes in a number of samples.
std::unique_ptr<Point[]> GetPMJ02Samples(const int num_samples);

// Generates progressive multi-jittered (0,2) samples with blue noise
// properties.
std::unique_ptr<Point[]> GetPMJ02SamplesWithBlueNoise(
    const int num_samples);

/*
 * -----------------------------------------------------------------------
 * These functions are just for experimentation, but likely not useful for
 * real purposes, since they perform worse than the ones above.
 * -----------------------------------------------------------------------
 */

// Generates progressive multi-jittered (0,2) samples, but instead of
// ensuring (0,2) subsequences, it chooses subquadrants randomly between
// even and odd powers of two.
std::unique_ptr<Point[]> GetPMJ02SamplesNoBalance(const int num_samples);

// Generates progressive multi-jittered (0,2) samples, but instead of
// ensuring (0,2) subsequences, it chooses subquadrants using the ox-plowing
// technique in Christensen et al.
std::unique_ptr<Point[]> GetPMJ02SamplesOxPlowing(const int num_samples);

}  // namespace pmj

#endif  // SAMPLE_GENERATION_PMJ02_H_
