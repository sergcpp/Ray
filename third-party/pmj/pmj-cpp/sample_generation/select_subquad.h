/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * This file implements different methods of selecting the subquadrants in
 * between odd and even powers of 4 for the PMJ and PMJ02 algorithms. Compared
 * to random, they make a big difference for the overall error!
 */
#ifndef SAMPLE_GENERATION_SELECT_SUBQUAD_H_
#define SAMPLE_GENERATION_SELECT_SUBQUAD_H_

#include <utility>
#include <vector>

#include "sample_generation/util.h"

namespace pmj {
  typedef std::vector<std::pair<int, int>> (*subquad_fn)(
      const Point samples[], const int dim);

  /*
   * This will randomly choose once to swap X or swap Y, and will always swap
   * X or Y for all subquadrants. For PMJ02, this ensures that the next set of
   * samples are themselves a (0,2) sequence. Based off some basic analysis,
   * it seems like this is the only way to maintain this property.
   *
   * Credit goes to Simon Brown for discovering this method with his Rust
   * implementation: https://github.com/sjb3d/pmj
   */
  std::vector<std::pair<int, int>> GetSubQuadrantsSwapXOrY(
    const Point samples[],
    const int dim);

  /*
   * Pick which subquadrants to use, using the ox-plowing technique from
   * Christensen et al.
   */
  std::vector<std::pair<int, int>> GetSubQuadrantsOxPlowing(
      const Point samples[],
      const int dim);

  /*
   * Pick which subquadrants to use randomly. No reason to actually use this:
   * OxPlowing is better for pmj and ShuffleSwap is better for pmj(0,2).
   */
  std::vector<std::pair<int, int>> GetSubQuadrantsRandomly(
      const Point samples[],
      const int dim);
}  // namespace pmj

#endif  // SAMPLE_GENERATION_SELECT_SUBQUAD_H_
