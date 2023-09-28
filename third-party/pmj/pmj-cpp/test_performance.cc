/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * Utility to test the performance of various sampling algorithms. You must use
 * bazel to build this, because it uses the ABSL library, and the Bazel build
 * downloads this from Github.
 *
 * Example command:
 *   bazel run -c opt test_performance -- --n=65536 --runs=64 --algorithms=pmj,pmj02
 */
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "sample_generation/algorithm.h"

ABSL_FLAG(int, n, 65536,
    "The number of samples to generate in each run.");
ABSL_FLAG(int, runs, 16, "The number of runs to make for each algorithm");
ABSL_FLAG(std::string, algorithms, "all",
    "Comma-separated list of algorithms to run. Use 'all' for all. Options are"
    "'uniform', 'pj', 'pmj', 'pmjbn', 'pmj02', 'pmj02bn'");

namespace chrono = std::chrono;

using std::string;
using std::vector;

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  vector<string> algorithms_list = {"pmj02"};
  if (absl::GetFlag(FLAGS_algorithms) == "all") {
    algorithms_list = {"uniform", "pj", "pmj", "pmjbn", "pmj02", "pmj02bn"};
  } else {
    algorithms_list = absl::StrSplit(absl::GetFlag(FLAGS_algorithms), ',');
  }

  for (const string& algorithm : algorithms_list) {
    pmj::sample_fn sample_func = pmj::GetSamplingFunction(algorithm);
    chrono::duration<double> total_time(0);
    for (int i = 0; i < absl::GetFlag(FLAGS_runs); i++) {
      auto start = chrono::high_resolution_clock::now();
      auto samples = (*sample_func)(absl::GetFlag(FLAGS_n));
      auto end = chrono::high_resolution_clock::now();
      total_time += (end - start);
    }

    const double total_microseconds =
        chrono::duration_cast<chrono::microseconds>(total_time).count();
    double samples_per_second = 1000000.0
        * (absl::GetFlag(FLAGS_n) * absl::GetFlag(FLAGS_runs))
        / total_microseconds;

    std::cout << algorithm << ": " << samples_per_second << " samples/s\n";
    std::cout << "\t"
              << "avg " << total_microseconds/1000.0/absl::GetFlag(FLAGS_runs)
              << "ms for " << absl::GetFlag(FLAGS_n)  << " samples" << "\n";
  }

  return 0;
}
