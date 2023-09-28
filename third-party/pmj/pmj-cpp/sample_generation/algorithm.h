/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * A function to convert a string to the appropriate sampling algorithm.
 */
#ifndef SAMPLE_GENERATION_ALGORITHM_H_
#define SAMPLE_GENERATION_ALGORITHM_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "sample_generation/pj.h"
#include "sample_generation/pmj.h"
#include "sample_generation/pmj02.h"

namespace pmj {

// Given a string like "pmj" or "pmj02bn", returns the function to generate
// samples from that algorithm.
typedef std::unique_ptr<pmj::Point[]> (*sample_fn)(int);
sample_fn GetSamplingFunction(const std::string& algorithm) {
  static const std::unordered_map<std::string, sample_fn> kAlgorithmMap = {
    {"uniform", &GetUniformRandomSamples},
    {"pj", &GetProgJitteredSamples},
    {"pmj", &GetProgMultiJitteredSamples},
    {"pmjbn", &GetProgMultiJitteredSamplesWithBlueNoise},
    {"pmj02", &GetPMJ02Samples},
    {"pmj02bn", &GetPMJ02SamplesWithBlueNoise},
    /* Experimental/Explicit Algorithms */
    {"pmj-random", &GetProgMultiJitteredSamplesRandom},
    {"pmj-oxplowing", &GetProgMultiJitteredSamplesOxPlowing},
    {"pmj02-oxplowing", &GetProgMultiJitteredSamplesOxPlowing},
    {"pmj02-no-balance", &GetPMJ02SamplesNoBalance},
  };

  auto find_iterator = kAlgorithmMap.find(algorithm);
  if (find_iterator == kAlgorithmMap.end())
    throw std::invalid_argument(algorithm + " is not a valid algorithm.");

  return find_iterator->second;
}

}  // namespace pmj

#endif  // SAMPLE_GENERATION_ALGORITHM_H_
