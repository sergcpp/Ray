#pragma once

#include <vector>

#include "../Types.h"
#include "Core.h"

namespace Ray {
namespace Ref {
void ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                         std::vector<uint32_t> &new_vtx_indices, const uint32_t *indices, size_t indices_count);
} // namespace Ref
} // namespace Ray
