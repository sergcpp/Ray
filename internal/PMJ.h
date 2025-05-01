#pragma once

#include <vector>

#include "CoreRef.h"

// https://jcgt.org/published/0008/01/04/
// Based on: https://github.com/Andrew-Helmer/pmj-cpp

namespace Ray {
void UpdateStrata(const Ref::dvec2 &p, int next_sample_count, int dim, std::vector<std::vector<bool>> &strata,
                  std::vector<const Ref::dvec2 *> &sample_grid);

void GetValidXOffsets(int x_pos, int y_pos, int strata_index, const std::vector<std::vector<bool>> &strata,
                      std::vector<int> &x_offsets);
void GetValidYOffsets(int x_pos, int y_pos, int strata_index, const std::vector<std::vector<bool>> &strata,
                      std::vector<int> &y_offsets);
void GetValidOffsets(int x_pos, int y_pos, const std::vector<std::vector<bool>> &strata,
                     std::vector<int> &x_offsets, std::vector<int> &y_offsets);

aligned_vector<Ref::dvec2> GeneratePMJSamples(unsigned int seed, int sample_count, int candidates_count);
} // namespace Ray