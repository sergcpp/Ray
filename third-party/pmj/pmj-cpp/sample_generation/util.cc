/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * This file implements a few functions that are useful for multiple sampling
 * sequences, especially random number generation and nearest neighbor search
 * on a grid.
 */
#include "sample_generation/util.h"

#include <cassert>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace pmj {

thread_local static std::random_device r;
thread_local static std::default_random_engine gen(r());

double UniformRand(double min, double max) {
  thread_local static std::uniform_real_distribution<double> uniform;

  std::uniform_real_distribution<double>::param_type param(min, max);

  return uniform(gen, param);
}

int UniformInt(int min, int max) {
  thread_local static std::uniform_int_distribution<int> uniform;

  std::uniform_int_distribution<int>::param_type param(min, max);

  return uniform(gen, param);
}

namespace {
inline double GetToroidalDistSq(double x1, double y1, double x2, double y2) {
  double x_diff = abs(x2-x1);
  if (x_diff > 0.5) x_diff = 1.0 - x_diff;
  double y_diff = abs(y2-y1);
  if (y_diff > 0.5) y_diff = 1.0 - y_diff;

  return (x_diff*x_diff)+(y_diff*y_diff);
}

inline int WrapIndex(const int index,
                     const int limit) {
  if (index < 0) return index+limit;
  if (index >= limit) return index-limit;
  return index;
}

inline void UpdateMinDistSq(
    const Point& candidate, const Point& point, double* min_dist_sq) {
  double dist_sq =
      GetToroidalDistSq(point.x, point.y, candidate.x, candidate.y);
  if (dist_sq < *min_dist_sq) {
    *min_dist_sq = dist_sq;
  }
}

inline void UpdateMinDistSqWithPointInCell(const Point& sample,
                                           const Point* sample_grid[],
                                           const int x,
                                           const int y,
                                           const int dim,
                                           double* min_dist_sq) {
  const int wrapped_x = WrapIndex(x, dim);
  const int wrapped_y = WrapIndex(y, dim);
  const Point* pt = sample_grid[wrapped_y*dim + wrapped_x];
  if (pt != nullptr) {
    UpdateMinDistSq(sample, *pt, min_dist_sq);
  }
}

double GetNearestNeighborDistSq(const Point& sample,
                                const Point* sample_grid[],
                                const int dim,
                                const double max_min_dist_sq) {
  // This function works by using the sample grid, since we know that the points
  // are well-distributed with at most one point in each cell.
  //
  // Start with the cells that are adjacent to our current cell and each
  // loop iteration we move outwards. We keep a track of the "grid radius",
  // which is the radius of the circle contained within our squares. If the
  // nearest point falls within this radius, we know that the next outward shift
  // can't find any nearer points.
  //
  // We do wrap around cells, and compute toroidal distances.
  const int x_pos = sample.x * dim;
  const int y_pos = sample.y * dim;

  // The minimum distance we've found between points, so far. 2.0 is the highest
  // possible value in the [0, 1) interval in 2 dimensions.
  double min_dist_sq = 2.0;
  const double grid_size = 1.0 / dim;
  for (int i = 1; i <= dim/2; i++) {
    const int x_min = x_pos - i;
    const int x_max = x_pos + i;
    const int y_min = y_pos - i;
    const int y_max = y_pos + i;

    int x = x_min;
    int y = y_min;
    for (; x < x_max; x++)  // Scan right over bottom row, ending at corner.
      UpdateMinDistSqWithPointInCell(
          sample, sample_grid, x, y, dim, &min_dist_sq);
    for (; y < y_max; y++)  // Scan up over right column, ending at corner.
      UpdateMinDistSqWithPointInCell(
          sample, sample_grid, x, y, dim, &min_dist_sq);
    for (; x > x_min; x--)  // Scan left over top row.
      UpdateMinDistSqWithPointInCell(
          sample, sample_grid, x, y, dim, &min_dist_sq);
    for (; y > y_min; y--)  // Scan down over left column.
      UpdateMinDistSqWithPointInCell(
          sample, sample_grid, x, y, dim, &min_dist_sq);

    // sqrt(0.5)*grid_size is the furthest a point can be from the center of its
    // square, so we add that.
    const double grid_radius = grid_size * (i + 0.7072);
    const double grid_radius_sq = grid_radius * grid_radius;
    if (min_dist_sq < grid_radius_sq ||
        min_dist_sq < max_min_dist_sq) {
      break;
    }
  }

  return min_dist_sq;
}
}  // namespace

Point GetBestCandidateOfSamples(const std::vector<Point>& candidates,
                                const Point* sample_grid[],
                                const int dim) {
  // Hypothetically, it could be faster to search all the points in parallel,
  // culling points as we go, but a naive implementation of this was only a tiny
  // bit faster, and the code was uglier, so we'll leave it for now.
  Point best_candidate;
  double max_min_dist_sq = 0.0;

  for (int i = 0; i < candidates.size(); i++) {
    Point cand_sample = candidates[i];
    double dist_sq =
        GetNearestNeighborDistSq(cand_sample,
                                 sample_grid,
                                 dim,
                                 max_min_dist_sq);
    if (dist_sq > max_min_dist_sq) {
      best_candidate = cand_sample;
      max_min_dist_sq = dist_sq;
    }
  }

  return best_candidate;
}

/*
 * This is kind of like a binary tree shuffle. Easy to think about for 4 points.
 * We can swap points 1 and 2, and we can swap 3 and 4, and we can swap the
 * pairs (1,2) and (3,4), but we can't swap 1 with 3 or 4, or 2 with 3 or 4. So
 * 2143 is a possible sequence, but 1324 is not. This works because in a
 * PMJ(0,2) sequence, every multiple of a power of two is also a valid PMJ(0,2)
 * sequence, at least if it's constructed using the ShuffleSwap subquadrant
 * selection.
 */
std::vector<const Point*> ShufflePMJ02Sequence(const pmj::Point points[],
                                               const int n) {
  assert((n & (n - 1)) == 0);  // This function only works for powers of two.
  std::vector<const Point*> shuffled_points(n);
  for (int i = 0; i < n; i++) {
    shuffled_points[i] = &points[i];
  }

  for (int stride = 2; stride < n; stride <<= 1) {
    for (int i = 0; i < n; i += stride) {
      if (UniformRand() < 0.5) {
        for (int j = 0; j < stride/2; j++) {
          const Point* pt = shuffled_points[i+j];
          shuffled_points[i+j] = shuffled_points[i+j+stride/2];
          shuffled_points[i+j+stride/2] = pt;
        }
      }
    }
  }

  return shuffled_points;
}

/*
 * This is equivalent to the previous function, ShufflePMJ02Sequence, but if you
 * moved the if (UniformRand() < 0.5) outside the (int i = 0) loop. It doesn't
 * shuffle as well, but the advantage is that you only need to have a single int
 * to encode the whole shuffle.
 */
std::vector<const Point*> ShufflePMJ02SequenceXor(const pmj::Point points[],
                                                  const int n) {
  assert((n & (n - 1)) == 0);  // This function only works for powers of two.
  std::vector<const Point*> shuffled_points(n);
  int random_encode = UniformInt(0, n-1);
  for (int i = 0; i < n; i++) {
    shuffled_points[i] = &points[i^random_encode];
  }

  return shuffled_points;
}

std::unique_ptr<Point[]> GetUniformRandomSamples(
    const int num_samples) {
  auto samples = std::make_unique<Point[]>(num_samples);

  for (int i = 0; i < num_samples; i++) {
    samples[i] = {UniformRand(), UniformRand()};
  }

  return samples;
}

}  // namespace pmj

