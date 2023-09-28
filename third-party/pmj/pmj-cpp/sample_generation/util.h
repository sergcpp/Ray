/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * Implements a few utility functions useful across the code base, especially
 * random number generation.
 */
#ifndef SAMPLE_GENERATION_UTIL_H_
#define SAMPLE_GENERATION_UTIL_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace pmj {

typedef struct {
  double x;
  double y;
} Point;

// Gets a random double between any two numbers. Thread-safe.
double UniformRand(double min = 0.0, double max = 1.0);
// Generates a random int in the given range. Thread-safe.
int UniformInt(int min, int max);

// Given a set of samples, a grid that points to existing samples, and the
// number of cells in one dimension of that grid, returns the candidate which
// is the furthest from all existing points.
Point GetBestCandidateOfSamples(const std::vector<Point>& candidates,
                                const Point* sample_grid[],
                                const int dim);

// Given a sequence of PMJ02 points, this will shuffle them, while the resulting
// shuffle will still be a progressive (0,2) sequence. We don't actually use it
// anywhere, this is just to show how easy it is.
std::vector<const Point*> ShufflePMJ02Sequence(const pmj::Point points[],
                                               const int n);

// This performs a shuffle similar to the one above, but it's easier and doesn't
// require storing the shuffle, only a single random int. It doesn't shuffle
// quite as well though.
std::vector<const Point*> ShufflePMJ02SequenceXor(const pmj::Point points[],
                                                  const int n);

// Just for comparison with performance testing and error analysis.
std::unique_ptr<Point[]> GetUniformRandomSamples(
    const int num_samples);

}  // namespace pmj

#endif  // SAMPLE_GENERATION_UTIL_H_
