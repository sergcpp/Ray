#pragma once

#include <cstdint>

#include <tuple>
#include <vector>

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>> LoadBIN(const char file_name[]);

std::vector<uint8_t> LoadTGA(const char file_name[], int &w, int &h);
void WriteTGA(const uint8_t *data, int w, int h, int bpp, const char *name);

