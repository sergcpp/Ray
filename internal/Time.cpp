#include "Time_.h"

#include <chrono>

namespace Ray {
std::chrono::steady_clock::time_point init_time = std::chrono::steady_clock::now();
}

uint64_t Ray::GetTimeMs() {
    auto t = (std::chrono::steady_clock::now() - init_time);
    auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(t);
    return uint64_t(tt.count());
}

uint64_t Ray::GetTimeUs() {
    auto t = (std::chrono::steady_clock::now() - init_time);
    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(t);
    return uint64_t(tt.count());
}

uint64_t Ray::GetTimeNs() {
    auto t = (std::chrono::steady_clock::now() - init_time);
    auto tt = std::chrono::duration_cast<std::chrono::nanoseconds>(t);
    return uint64_t(tt.count());
}

double Ray::GetTimeS() {
    std::chrono::duration<double> t = (std::chrono::steady_clock::now() - init_time);
    return double(t.count());
}