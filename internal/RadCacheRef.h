#pragma once

#include "CoreRef.h"

//
// Based on https://github.com/NVIDIAGameWorks/SHARC
//

namespace Ray {
namespace Ref {
// http://burtleburtle.net/bob/hash/integer.html
force_inline constexpr uint32_t hash_jenkins32(uint32_t a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

force_inline uint32_t hash64(const uint64_t hash_key) {
    return hash_jenkins32(uint32_t((hash_key >> 0) & 0xffffffff)) ^
           hash_jenkins32(uint32_t((hash_key >> 32) & 0xffffffff));
}

uint32_t calc_grid_level(const fvec4 &p, const cache_grid_params_t &params);

uint32_t insert_entry(Span<uint64_t> entries, const fvec4 &p, const fvec4 &n, const cache_grid_params_t &params);
uint32_t find_entry(Span<const uint64_t> entries, const fvec4 &p, const fvec4 &n, const cache_grid_params_t &params);

void SpatialCacheUpdate(const cache_grid_params_t &params, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                        Span<cache_data_t> cache_data, const color_rgba_t radiance[],
                        const color_rgba_t depth_normals[], int img_w, Span<uint64_t> entries,
                        Span<packed_cache_voxel_t> voxels_curr);
void SpatialCacheResolve(const cache_grid_params_t &params, Span<uint64_t> entries,
                         Span<packed_cache_voxel_t> voxels_curr, Span<const packed_cache_voxel_t> voxels_prev,
                         uint32_t start, uint32_t count);
} // namespace Ref
} // namespace Ray