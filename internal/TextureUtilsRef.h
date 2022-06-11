#pragma once

#include <vector>

#include "../Types.h"
#include "Core.h"

namespace Ray {
namespace Ref {
template<int N>
std::vector<color_t<uint8_t, N>> DownsampleTexture(const color_t<uint8_t, N> tex[], const int res[2]);

extern template std::vector<color_t<uint8_t, 4>> DownsampleTexture<4>(const color_t<uint8_t, 4> tex[], const int res[2]);
extern template std::vector<color_t<uint8_t, 3>> DownsampleTexture<3>(const color_t<uint8_t, 3> tex[], const int res[2]);
extern template std::vector<color_t<uint8_t, 1>> DownsampleTexture<1>(const color_t<uint8_t, 1> tex[], const int res[2]);

void ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                         std::vector<uint32_t> &new_vtx_indices, const uint32_t *indices, size_t indices_count);
} // namespace Ref
} // namespace Ray
