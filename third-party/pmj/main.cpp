
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <vector>

#include "pmj-cpp/sample_generation/pmj02.h"

int main(int argc, char *argv[]) {
    const int SAMPLE_COUNT = 4096;
    const int DIMS_COUNT = 32;

    std::ofstream out_file("../../internal/pmj/__pmj02_samples.inl", std::ios::binary);

    out_file << "extern const int __pmj02_sample_count = " << SAMPLE_COUNT << ";\n";
    out_file << "extern const int __pmj02_dims_count = " << DIMS_COUNT << ";\n";
    out_file << "extern const uint32_t __pmj02_samples[" << 2 * SAMPLE_COUNT * DIMS_COUNT << "] = {\n    ";

    for (int dim = 0; dim < DIMS_COUNT; ++dim) {
        std::unique_ptr<pmj::Point[]> samples = pmj::GetPMJ02Samples(SAMPLE_COUNT);

        for (int i = 0; i < SAMPLE_COUNT; ++i) {
            pmj::Point p = samples[i];
            out_file << (uint32_t(p.x * 16777216.0) << 8) << ", " << (uint32_t(p.y * 16777216.0) << 8) << ", ";
        }
    }
    out_file << "\n};\n";

    return 0;
}
