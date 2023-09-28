
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <vector>

#include "pmj-cpp/sample_generation/pmj02.h"

int main(int argc, char *argv[]) {
    const int SAMPLE_COUNT = 4096;

    std::unique_ptr<pmj::Point[]> samples = pmj::GetPMJ02Samples(SAMPLE_COUNT);

    std::ofstream out_file("../../internal/pmj/__pmj02_samples.inl", std::ios::binary);

    out_file << "extern const int __pmj02_sample_count = " << SAMPLE_COUNT << ";\n";
    out_file << "extern const float __pmj02_samples[" << 2 * SAMPLE_COUNT << "] = {\n    ";

    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        pmj::Point p = samples[i];
        out_file << float(p.x) << "f, " << float(p.y) << "f, ";
    }

    out_file << "\n};\n";

    return 0;
}
