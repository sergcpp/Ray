/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * This tool takes three command-line arguments:
 * --algorithm
 *    Can be one of "pj", "pmj", "pmjbn", "pmj02", or "pmj02bn". By
 *    default it is pmj02.
 * --n
 *    The number of samples, and it must be an integer greater than 0.
 * --out
 *    The file path for a text file that the sample sequence will be
 *    written to.
 *
 * Example usage with make:
 * $  make release
 * $  ./generate_samples --algorithm=pmj02bn --n=4096 --out=$PWD/samples.txt
 *
 * Example usage with bazel:
 * $  bazel run -c opt :generate_samples -- --algorithm=pmj02bn --n=4096 --out=$PWD/samples.txt
 */
#include <array>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <utility>

#include "sample_generation/algorithm.h"

using std::string;

namespace {
  void GetArguments(const int argc,
                    char* argv[],
                    int* n_samples,
                    std::string* algorithm,
                    std::string* outfile) {
    string samples_str = "256";
    string algorithm_str = "pmj02";
    for (int i = 1; i < argc; ++i) {
      string arg = argv[i];
      if (arg == "--n") {
          if (i + 1 < argc) samples_str = argv[i+1];
      } else if (arg.rfind("--n=", 0) == 0) {
        samples_str = arg.substr(4);
      }

      if (arg == "--algorithm") {
          if (i + 1 < argc) algorithm_str = argv[i+1];
      } else if (arg.rfind("--algorithm=", 0) == 0) {
        algorithm_str = arg.substr(12);
      }

      if (arg == "--out") {
          if (i + 1 < argc) *outfile = argv[i+1];
      } else if (arg.rfind("--out=", 0) == 0) {
        *outfile = arg.substr(6);
      }
    }
    *n_samples = std::stoi(samples_str);
    if (*n_samples <= 0) {
      throw std::invalid_argument("--n must be positive.");
    }
    *algorithm = algorithm_str;
  }
}  // namespace

int main(int argc, char* argv[]) {
  int n_samples;
  string algorithm;
  string outfile_path;
  GetArguments(argc, argv, &n_samples, &algorithm, &outfile_path);

  auto* sample_func = pmj::GetSamplingFunction(algorithm);

  std::unique_ptr<pmj::Point[]> samples = (*sample_func)(n_samples);

  std::unique_ptr<std::ofstream> outfile;
  if (!outfile_path.empty()) {
    outfile = std::make_unique<std::ofstream>(outfile_path, std::ofstream::out);
    // Full double precision.
    outfile->precision(17);
  } else {
    std::cout.precision(17);
  }

  for (int i = 0; i < n_samples; i++) {
    const auto& sample = samples[i];
    if (outfile)
      *outfile << "(" << sample.x << ", " << sample.y << "),\n";
    else
      std::cout << "(" << sample.x << ", " << sample.y << "),\n";
  }

  if (outfile) {
    std::cout << n_samples << " " << algorithm << " samples written to "
              << outfile_path << "\n";
  }
}
