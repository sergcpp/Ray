/*
 * Copyright (C) Andrew Helmer 2020.
 * Licensed under MIT Open-Source License: see LICENSE.
 *
 * Utility to analyze how well each algorithm performs for estimating different
 * distributions.
 *
 * Example command:
 * $  bazel run -c opt analyze_error -- --algorithms=pmj,pmj02,pmj02bn
 *
 * If you want to output a python file for analysis, use the pyfile flag, but
 * you need to build the executable rather than using bazel run. E.g.
 * $  bazel build -c opt analyze_error
 * $  ./bazel-bin/analyze_error --pyfile=analysis.py
 */
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "sample_generation/algorithm.h"

ABSL_FLAG(std::string, distr, "disc,gaussian,bilinear",
    "Which distributions to use for the error analysis.");
ABSL_FLAG(std::string, algorithms, "all",
    "Comma-separated list of algorithms to run. Use 'all' for all. Options are"
    "'uniform', 'pj', 'pmj', 'pmjbn', 'pmj02', 'pmj02bn'");
ABSL_FLAG(std::string, pyfile, "",
    "If set, this will write all the analysis data into a valid python file, "
    "rather than outputting it. This python file can be executed to create "
    "a dictionary representing all the results.");
ABSL_FLAG(int, max_n, 1024,
    "The number of samples to generate in each run.");
ABSL_FLAG(int, runs, 1024, "The number of runs to make for each algorithm");

namespace {

using std::string;

/*
 * These distributions were taken from here:
 * https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/pmj_slides.pdf
 */
double disc_distr(const pmj::Point& point) {
  constexpr double radius_sq = 2.0 / 3.14159265;
  if (point.x*point.x + point.y*point.y <= radius_sq) return 1.0;
  else
    return 0.0;
}
double gaussian_distr(const pmj::Point& point) {
  // This is really a scaled gaussian, where the mean (0,0) has a value of 1.
  double dist_sq = point.x*point.x + point.y*point.y;

  return exp(-dist_sq);
}
double bilinear_distr(const pmj::Point& point) {
  return point.x * point.y;
}

typedef double (*dist_fn)(const pmj::Point& point);
  dist_fn GetDistribution(const string& distribution) {
  return distribution == "disc" ? &disc_distr :
         distribution == "bilinear" ? &bilinear_distr :
         distribution == "gaussian" ? &gaussian_distr :
      throw std::invalid_argument(distribution
                                  + " is not a valid distribution.");
}

std::vector<double> GetErrors(const dist_fn dist_func,
                              const pmj::sample_fn sample_func,
                              const int num_samples,
                              const int num_runs,
                              const double ground_truth) {
  std::vector<double> avg_error(num_samples);
  for (int i = 0; i < num_runs; i++) {
    std::unique_ptr<pmj::Point[]> points = (*sample_func)(num_samples);
    double avg = 0.0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      double val = (*dist_func)(points[sample_idx]);
      avg = val/(sample_idx+1.0)
          + avg*static_cast<double>(sample_idx)/(sample_idx+1.0);
      double error = avg - ground_truth;
      avg_error[sample_idx] +=
          (error*error) / (static_cast<double>(num_runs));
    }
  }

  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    avg_error[sample_idx] = sqrt(avg_error[sample_idx]);
  }

  return avg_error;
}

class AnalysisOutput {
 public:
  explicit AnalysisOutput(const string& pyfile = "") {
    if (!pyfile.empty()) {
      python_file_.reset(new std::ofstream(pyfile, std::ofstream::out));
    }
  }

  void SetDistribution(const string& distribution,
                       const double true_avg) {
    if (python_file_) {
      errors_.push_back({distribution, {}});
    } else {
      std::cout << distribution << "\n";
      std::cout << "\tTrue Average: " << true_avg << "\n";
    }
  }
  void SetSamplingResults(const string& method,
                          const std::vector<double>& errors) {
    if (python_file_) {
      // This copies the whole errors array, but do we really care?
      errors_.back().second.push_back({method, errors});
    } else {
      std::cout << "\t" << method << "\n";
      std::cout << "\t\tFinal Error (" << errors.size() << " Samples): "
                << errors.back() << "\n";
    }
  }
  void Finish() {
    if (python_file_) {
      (*python_file_) << "analyses = {\n";
      for (const auto& dist_errors : errors_) {
        const string& distribution = dist_errors.first;
        (*python_file_) << "\t'" << distribution << "': {\n";
        for (const auto& algorithm_errors : dist_errors.second) {
          const string& algorithm = algorithm_errors.first;
          (*python_file_) << "\t\t'" << algorithm << "': [\n";
          for (const double avg_error : algorithm_errors.second) {
            (*python_file_) << "\t\t\t" << avg_error << ",\n";
          }
          (*python_file_) << "\t\t],\n";
        }
        (*python_file_) << "\t},\n";
      }
      (*python_file_) << "}\n";
      python_file_->close();
    }
  }

 private:
  std::vector<
      std::pair<string,
                std::vector<std::pair<string,
                                      std::vector<double>>>>> errors_;
  std::unique_ptr<std::ofstream> python_file_;
};
}  // namespace

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::vector<std::string> distributions =
      absl::StrSplit(absl::GetFlag(FLAGS_distr), ',');

  std::vector<std::string> algorithms;
  if (absl::GetFlag(FLAGS_algorithms) == "all") {
    algorithms = {"uniform", "pj", "pmj", "pmjbn", "pmj02", "pmj02bn"};
  } else {
    algorithms = absl::StrSplit(absl::GetFlag(FLAGS_algorithms), ',');
  }

  const int num_samples = absl::GetFlag(FLAGS_max_n);
  const int runs = absl::GetFlag(FLAGS_runs);

  // Full double precision.
  std::cout.precision(17);

  AnalysisOutput output(absl::GetFlag(FLAGS_pyfile));

  for (const std::string& distribution : distributions) {
    auto dist_func = GetDistribution(distribution);

    // For some of the distributions, they have analytical results, but why not
    // just get a good numerical one. Makes it easier to add more distributions
    // later. We just use stratified sampling with a lot of samples.
    constexpr int kStratificationDim = 2048;
    constexpr int kTotalSamples = kStratificationDim*kStratificationDim;
    double avg = 0.0;
    for (int y = 0; y < kStratificationDim; y++) {
      for (int x = 0; x < kStratificationDim; x++) {
        pmj::Point point;
        point.x = pmj::UniformRand(static_cast<double>(x)/kStratificationDim,
                                   static_cast<double>(x+1)/kStratificationDim);
        point.y = pmj::UniformRand(static_cast<double>(y)/kStratificationDim,
                                   static_cast<double>(y+1)/kStratificationDim);
        double val = (*dist_func)(point);
        avg += val/kTotalSamples;
      }
    }

    output.SetDistribution(distribution, avg);

    for (const std::string& algorithm : algorithms) {
      pmj::sample_fn sample_func = pmj::GetSamplingFunction(algorithm);
      std::vector<double> error_per_sample =
          GetErrors(dist_func, sample_func, num_samples, runs, avg);

      output.SetSamplingResults(algorithm, error_per_sample);
    }
  }
  output.Finish();
}
