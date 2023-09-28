/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * Implementation of PMJ(0,2) sequences.
 *
 * This incorporates Christensen et al.'s balancing characteristic:
 * sub-sequences between odd and even powers of two are themselves (0,2)
 * sequences. This significantly reduces integration errors.
 *
 * If you're reading this code for the first time and want to understand the
 * algorithm, start with the function "GenerateSamples".
 */
#include "sample_generation/pmj02.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stack>
#include <utility>
#include <vector>

#include "sample_generation/pmj02_util.h"
#include "sample_generation/select_subquad.h"
#include "sample_generation/util.h"

namespace pmj {
namespace {

using std::vector;

static constexpr int kBestCandidateSamples = 100;

/*
 * The SampleSet is a class that contains the generated samples, as well as the
 * currently populated strata. It's used to generate new samples within the
 * unpopulated strata.
 */
class SampleSet {
 public:
  explicit SampleSet(const int num_samples,
                     const int num_candidates)
                     : num_candidates_(num_candidates) {
    samples_ = std::make_unique<Point[]>(num_samples);
    std::fill_n(samples_.get(), num_samples, Point({0.0, 0.0}));

    int grid_memory_size = 1;
    while (grid_memory_size < num_samples)
      grid_memory_size <<= 2;
    sample_grid_ = std::make_unique<const Point*[]>(grid_memory_size);
    std::fill_n(sample_grid_.get(), grid_memory_size, nullptr);
  }

  void GenerateFirstSample();

  // This generates a new sample at the given index, given the X position and Y
  // position of the subquadrant. It won't generate a new sample in an existing
  // strata.
  void GenerateNewSample(const int sample_index,
                         const int x_pos,
                         const int y_pos);

  // This function should be called after every power of 2 samples. It divides
  // the strata up into the next elementary (0,2) intervals, and marks the
  // occupied strata..
  void SubdivideStrata();

  std::unique_ptr<Point[]> ReleaseSamples() {
    return std::move(samples_);
  }

  const Point& sample(const int sample_index) const {
    return samples_[sample_index];
  }
  const Point* samples() const {
    return samples_.get();
  }
  const int dim() const { return dim_; }

 private:
  // Adds a new point at index i. Updates the necessary data structures.
  void AddSample(const int i, const Point& sample);

  // Given a sample, sets all the correct strata to true.
  void UpdateStrata(const int sample_index);

  Point GetCandidateSample(const vector<int>& valid_x_strata,
                           const vector<int>& valid_y_strata) const;

  std::unique_ptr<Point[]> samples_;

  // Contains all strata of elementary (0,2) intervals. Each value is true/false
  // representing if a sample point resides there.
  vector<vector<bool>> strata_ {{false}};

  // The sample grid is used for nearest neighbor lookups.
  std::unique_ptr<const Point*[]> sample_grid_;

  int n_ = 1;  // Number of samples in the next pass.
  bool is_power_of_4_ = true;
  int dim_ = 1;  // Number of cells in one dimension in next pass, i.e. sqrt(n).

  // Number of candidates to use for best-candidate sampling.
  const int num_candidates_;
};

void SampleSet::SubdivideStrata() {
  const int old_n = n_;

  n_ *= 2;
  is_power_of_4_ = !is_power_of_4_;
  if (!is_power_of_4_) {
    dim_ *= 2;
  }

  // For the first sample this is 1x1. For sample 2 it's 1x2 and 2x1. For
  // samples 3-4 it's 4x1, 2x2, and 1x4. So every time it goes up by one.
  strata_.resize(strata_.size()+1);

  // Clear all the strata and mark the occupied ones again.
  std::fill(strata_.begin(), strata_.end(), vector<bool>(n_, false));
  std::fill_n(sample_grid_.get(), n_, nullptr);
  for (int i = 0; i < old_n; i++) {
    UpdateStrata(i);
  }
}

// This generates a sample within the grid position, verifying that it doesn't
// overlap strata with any other sample.
Point SampleSet::GetCandidateSample(const vector<int>& valid_x_strata,
                                    const vector<int>& valid_y_strata) const {
  Point sample;

  int x_strata_index = valid_x_strata[UniformInt(0, valid_x_strata.size()-1)];
  int y_strata_index = valid_y_strata[UniformInt(0, valid_y_strata.size()-1)];

  double strata_width = 1.0 / n_;
  sample.x = UniformRand(strata_width*x_strata_index,
                         strata_width*(x_strata_index+1));
  sample.y = UniformRand(strata_width*y_strata_index,
                         strata_width*(y_strata_index+1));

  assert(sample.x >= 0.0 && sample.x < 1.0 && sample.y >= 0 && sample.y < 1.0);

  return sample;
}

void SampleSet::GenerateFirstSample() {
  Point sample = {UniformRand(), UniformRand()};
  AddSample(0, sample);
}

void SampleSet::GenerateNewSample(const int sample_index,
                                  const int x_pos,
                                  const int y_pos) {
  Point best_candidate;

  const std::pair<vector<int>, vector<int>>& valid_strata =
      GetValidStrata(x_pos, y_pos, strata_);

  if (num_candidates_ <= 1) {
    best_candidate =
        GetCandidateSample(valid_strata.first, valid_strata.second);
  } else {
    vector<Point> candidate_samples(num_candidates_);
    for (int i = 0; i < num_candidates_; i++) {
      candidate_samples[i] =
          GetCandidateSample(valid_strata.first, valid_strata.second);
    }

    best_candidate = GetBestCandidateOfSamples(
        candidate_samples, sample_grid_.get(), dim_);
  }
  AddSample(sample_index, best_candidate);
}

void SampleSet::UpdateStrata(const int sample_index) {
  const Point& sample = samples_[sample_index];

  for (int i = 0, strata_n_cols = n_, strata_n_rows = 1;
       strata_n_cols >= 1;
       strata_n_cols /= 2, strata_n_rows *= 2, i++) {
    int x_pos = sample.x * strata_n_cols;
    int y_pos = sample.y * strata_n_rows;
    strata_[i][y_pos*strata_n_cols + x_pos] = true;
  }

  const int x_pos = sample.x * dim_, y_pos = sample.y * dim_;
  sample_grid_[y_pos*dim_ + x_pos] = &sample;
}

void SampleSet::AddSample(const int i,
                          const Point& sample) {
  samples_[i] = sample;
  UpdateStrata(i);
}

/*
 * The core of Christensen et al.'s algorithm.
 */
std::unique_ptr<Point[]> GenerateSamples(
    const int num_samples,
    const int num_candidates,
    const subquad_fn subquad_func = &GetSubQuadrantsSwapXOrY) {
  SampleSet sample_set(num_samples, num_candidates);

  sample_set.GenerateFirstSample();

  // Number of samples from the previous iteration. Always a power of 4.
  int n = 1;
  while (n < num_samples) {
    // Subdivide the strata. On the first call, this takes the strata from 1x1
    // to 2x1 and 1x2.
    sample_set.SubdivideStrata();

    // For every sample, we first generate the diagonally opposite one at the
    // current grid level.
    for (int i = 0; i < n && n+i < num_samples; i++) {
      const Point& sample = sample_set.sample(i);

      int x_pos = sample.x * sample_set.dim();
      int y_pos = sample.y * sample_set.dim();

      sample_set.GenerateNewSample(/*sample_index=*/n+i, x_pos ^ 1, y_pos ^ 1);
    }

    if (2*n >= num_samples) break;

    // Subdivide the strata, for instance strata of 2x1 and 1x2 will become
    // strata of 4x1, 2x2, and 1x4.
    sample_set.SubdivideStrata();

    // For the remaining subquadrants, we need to pick what order to sample from
    // them. This will get us the set of subquadrants for the next n samples.
    const std::vector<std::pair<int, int>> sub_quad_choices =
        (*subquad_func)(sample_set.samples(), sample_set.dim());
    for (int i = 0; i < n && 2*n+i < num_samples; i++) {
      sample_set.GenerateNewSample(/*sample_index=*/2*n+i,
                                   sub_quad_choices[i].first,
                                   sub_quad_choices[i].second);
    }

    // Finally we sample from the subquadrants diagonally opposite to the ones
    // we just did.
    for (int i = 0; i < n && 3*n+i < num_samples; i++) {
      sample_set.GenerateNewSample(/*sample_index=*/3*n+i,
                                   sub_quad_choices[i].first ^ 1,
                                   sub_quad_choices[i].second ^ 1);
    }

    n *= 4;
  }

  return sample_set.ReleaseSamples();
}

}  // namespace

std::unique_ptr<Point[]> GetPMJ02Samples(
    const int num_samples) {
  return GenerateSamples(num_samples, /*num_candidates=*/1);
}
std::unique_ptr<Point[]> GetPMJ02SamplesWithBlueNoise(
    const int num_samples) {
  return GenerateSamples(num_samples, kBestCandidateSamples);
}

// These functions are just for experimentation, no reason to actually use them.
std::unique_ptr<Point[]> GetPMJ02SamplesNoBalance(
    const int num_samples) {
  return GenerateSamples(num_samples,
                         kBestCandidateSamples,
                         /*subquad_func=*/&GetSubQuadrantsRandomly);
}
std::unique_ptr<Point[]> GetPMJ02SamplesOxPlowing(
    const int num_samples) {
  return GenerateSamples(num_samples,
                         kBestCandidateSamples,
                         /*subquad_func=*/&GetSubQuadrantsOxPlowing);
}
}  // namespace pmj
