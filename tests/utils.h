#pragma once

#include <cstdint>

#include <tuple>
#include <vector>

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>> LoadBIN(const char file_name[]);

std::vector<uint8_t> LoadTGA(const char file_name[], bool flip_y, int &w, int &h);
inline std::vector<uint8_t> LoadTGA(const char file_name[], int &w, int &h) { return LoadTGA(file_name, false, w, h); }
std::vector<uint8_t> LoadDDS(const char file_name[], int &w, int &h, int &mips, int &channels);
std::vector<uint8_t> LoadHDR(const char file_name[], int &w, int &h);
