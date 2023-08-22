#pragma once

#include <vector>

#include "../Span.h"
#include "../Types.h"
#include "Core.h"

namespace Ray {
void ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                         std::vector<uint32_t> &new_vtx_indices, Span<const uint32_t> indices);
} // namespace Ray
