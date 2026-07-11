#include "RadCacheRef.h"

namespace Ray {
namespace Ref {
// Based on logarithmic caching by Johannes Jendersie
ivec4 calc_grid_position_log(fvec4 p, const cache_grid_params_t &params) {
    p += HASH_GRID_POSITION_BIAS;

    const uint32_t grid_level = calc_grid_level(p, params);
    const float voxel_size = calc_voxel_size(grid_level, params);
    ivec4 grid_position = ivec4(floor(p / voxel_size));
    grid_position.set<3>(grid_level);
    return grid_position;
}

force_inline uint32_t hash_map_base_slot(const uint64_t hash_key) {
    const uint32_t hash = hash64(hash_key);
    const uint32_t slot = hash % HASH_GRID_CACHE_ENTRIES_COUNT;

    return std::min(slot, HASH_GRID_CACHE_ENTRIES_COUNT - HASH_GRID_HASH_MAP_BUCKET_SIZE);
}

uint64_t compute_hash(const fvec4 &p, const fvec4 &n, const cache_grid_params_t &params) {
    const uvec4 grid_pos = uvec4(calc_grid_position_log(p, params));

    uint64_t hash_key =
        ((uint64_t(grid_pos.get<0>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 0)) |
        ((uint64_t(grid_pos.get<1>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 1)) |
        ((uint64_t(grid_pos.get<2>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 2)) |
        ((uint64_t(grid_pos.get<3>()) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 3));

    if (HASH_GRID_USE_NORMALS) {
        const uint32_t normal_bits = (n.get<0>() + HASH_GRID_NORMAL_BIAS >= 0 ? 0 : 1) +
                                     (n.get<1>() + HASH_GRID_NORMAL_BIAS >= 0 ? 0 : 2) +
                                     (n.get<2>() + HASH_GRID_NORMAL_BIAS >= 0 ? 0 : 4);
        hash_key |= (uint64_t(normal_bits) << (HASH_GRID_POSITION_BIT_COUNT * 3 + HASH_GRID_LEVEL_BIT_COUNT));
    }

    return hash_key;
}

force_inline int grid_dist2(const ivec4 &pos) {
    return pos.get<0>() * pos.get<0>() + pos.get<1>() * pos.get<1>() + pos.get<2>() * pos.get<2>();
}

uint64_t get_adjacent_level_hash(const uint64_t hash_key, const cache_grid_params_t &params) {
    static const int32_t SignBit = int32_t(1u << (HASH_GRID_POSITION_BIT_COUNT - 1u));
    static const int32_t SignMask = int32_t(~((1u << HASH_GRID_POSITION_BIT_COUNT) - 1u));

    ivec4 grid_pos = 0;
    grid_pos.set<0>(int((hash_key >> (HASH_GRID_POSITION_BIT_COUNT * 0)) & HASH_GRID_POSITION_BIT_MASK));
    grid_pos.set<1>(int((hash_key >> (HASH_GRID_POSITION_BIT_COUNT * 1)) & HASH_GRID_POSITION_BIT_MASK));
    grid_pos.set<2>(int((hash_key >> (HASH_GRID_POSITION_BIT_COUNT * 2)) & HASH_GRID_POSITION_BIT_MASK));

    // Fix negative coordinates
    grid_pos.set<0>((grid_pos.get<0>() & SignBit) ? grid_pos.get<0>() | SignMask : grid_pos.get<0>());
    grid_pos.set<1>((grid_pos.get<1>() & SignBit) ? grid_pos.get<1>() | SignMask : grid_pos.get<1>());
    grid_pos.set<2>((grid_pos.get<2>() & SignBit) ? grid_pos.get<2>() | SignMask : grid_pos.get<2>());

    int level = uint32_t((hash_key >> (HASH_GRID_POSITION_BIT_COUNT * 3)) & HASH_GRID_LEVEL_BIT_MASK);

    const float voxel_size = calc_voxel_size(level, params);
    const ivec4 camera_grid_pos_curr = ivec4(floor(make_fvec3(params.cam_pos_curr) / voxel_size));
    const ivec4 camera_vector_curr = camera_grid_pos_curr - grid_pos;
    const int camera_distance_curr = grid_dist2(camera_vector_curr);

    const ivec4 camera_grid_pos_prev = ivec4(floor(make_fvec3(params.cam_pos_prev) / voxel_size));
    const ivec4 camera_vector_prev = camera_grid_pos_prev - grid_pos;
    const int camera_distance_prev = grid_dist2(camera_vector_prev);

    if (camera_distance_curr < camera_distance_prev) {
        grid_pos = ivec4(floor(fvec4(grid_pos) / params.log_base));
        level = std::min(level + 1, int(HASH_GRID_LEVEL_BIT_MASK));
    } else {
        grid_pos = ivec4(floor(fvec4(grid_pos) * params.log_base));
        level = std::max(level - 1, 1);
    }

    uint64_t modified_hash_key =
        ((uint64_t(grid_pos.get<0>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 0)) |
        ((uint64_t(grid_pos.get<1>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 1)) |
        ((uint64_t(grid_pos.get<2>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 2)) |
        ((uint64_t(level) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_COUNT * 3));

    if (HASH_GRID_USE_NORMALS) {
        modified_hash_key |= hash_key & (uint64_t(HASH_GRID_NORMAL_BIT_MASK)
                                         << (HASH_GRID_POSITION_BIT_COUNT * 3 + HASH_GRID_LEVEL_BIT_COUNT));
    }

    return modified_hash_key;
}

bool hash_map_insert(Span<uint64_t> entries, const uint64_t hash_key, uint32_t &cache_entry) {
    const uint32_t base_slot = hash_map_base_slot(hash_key);
    for (uint32_t bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t prev_hash_key =
            Ray_InterlockedCompareExchange64(&entries[base_slot + bucket_offset], hash_key, HASH_GRID_INVALID_HASH_KEY);
        if (prev_hash_key == HASH_GRID_INVALID_HASH_KEY || prev_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        }
    }
    return false;
}

bool hash_map_find(Span<const uint64_t> entries, const uint64_t hash_key, uint32_t &cache_entry,
                   uint32_t &bucket_offset) {
    const uint32_t base_slot = hash_map_base_slot(hash_key);
    for (bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t stored_hash_key = entries[base_slot + bucket_offset];
        if (stored_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        }
    }
    return false;
}

fvec4 GetColorFromHash32(const uint32_t hash) {
    fvec4 color = 0.0f;
    color.set<0>(((hash >> 0) & 0x3ff) / 1023.0f);
    color.set<1>(((hash >> 11) & 0x7ff) / 2047.0f);
    color.set<2>(((hash >> 22) & 0x7ff) / 2047.0f);
    color.set<3>(1.0f);
    return color;
}

fvec4 hash_grid_debug(const fvec4 &p, const fvec4 &n, const cache_grid_params_t &params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    return GetColorFromHash32(hash64(hash_key));
}

void accumulate_cache_voxel(packed_cache_voxel_t &voxel, const fvec4 &r, const uint32_t sample_data) {
    const uvec4 data = uvec4(r * RAD_CACHE_RADIANCE_SCALE);

    if (data.get<0>()) {
        Ray_InterlockedExchangeAdd(&voxel.v[0], data.get<0>());
    }
    if (data.get<1>()) {
        Ray_InterlockedExchangeAdd(&voxel.v[1], data.get<1>());
    }
    if (data.get<2>()) {
        Ray_InterlockedExchangeAdd(&voxel.v[2], data.get<2>());
    }
    if (sample_data) {
        Ray_InterlockedExchangeAdd(&voxel.v[3], sample_data);
    }
}
} // namespace Ref
} // namespace Ray

uint32_t Ray::Ref::calc_grid_level(const fvec4 &p, const cache_grid_params_t &params) {
    const float distance2 = length2(make_fvec3(params.cam_pos_curr) - p);
    const float ret = Ray::clamp(floorf(0.5f * log_base(distance2, params.log_base) + HASH_GRID_LEVEL_BIAS), 1.0f,
                                 HASH_GRID_LEVEL_BIT_MASK);
    return uint32_t(ret);
}

uint32_t Ray::Ref::insert_entry(Span<uint64_t> entries, const fvec4 &p, const fvec4 &n,
                                const cache_grid_params_t &params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    uint32_t cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
    hash_map_insert(entries, hash_key, cache_entry);
    return cache_entry;
}

uint32_t Ray::Ref::find_entry(Span<const uint64_t> entries, const fvec4 &p, const fvec4 &n,
                              const cache_grid_params_t &params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    uint32_t cache_entry = HASH_GRID_INVALID_CACHE_ENTRY, collisions_count;
    hash_map_find(entries, hash_key, cache_entry, collisions_count);
    return cache_entry;
}

void Ray::Ref::SpatialCacheUpdate(const cache_grid_params_t &params, Span<const hit_data_t> inters,
                                  Span<const ray_data_t> rays, Span<cache_data_t> cache_data,
                                  const color_rgba_t radiance[], const color_rgba_t depth_normals[], const int img_w,
                                  Span<uint64_t> entries, Span<packed_cache_voxel_t> voxels_curr) {
    for (int i = 0; i < int(inters.size()); ++i) {
        const ray_data_t &r = rays[i];
        const hit_data_t &inter = inters[i];

        const uint32_t x = (r.xy >> 16) & 0x0000ffff;
        const uint32_t y = r.xy & 0x0000ffff;

        const fvec4 I = make_fvec3(r.d);
        const fvec4 ro = make_fvec3(r.o);

        const fvec4 P = ro + inter.t * I;
        const fvec4 N = fvec4{depth_normals[y * img_w + x].v};
        fvec4 rad = fvec4{radiance[y * img_w + x].v} * params.exposure;

        cache_data_t &cache = cache_data[y * (img_w / RAD_CACHE_DOWNSAMPLING_FACTOR) + x];
        cache.sample_weight[0][0] *= r.c[0];
        cache.sample_weight[0][1] *= r.c[1];
        cache.sample_weight[0][2] *= r.c[2];
        if (inter.v < 0.0f || inter.obj_index < 0 || cache.path_len == RAD_CACHE_PROPAGATION_DEPTH) {
            for (int j = 0; j < cache.path_len; ++j) {
                rad *= make_fvec3(cache.sample_weight[j]);
                if (cache.cache_entries[j] != HASH_GRID_INVALID_CACHE_ENTRY) {
                    accumulate_cache_voxel(voxels_curr[cache.cache_entries[j]], rad, 0);
                }
            }
        } else {
            for (int j = cache.path_len; j > 0; --j) {
                cache.cache_entries[j] = cache.cache_entries[j - 1];
                memcpy(cache.sample_weight[j], cache.sample_weight[j - 1], 3 * sizeof(float));
            }

            cache.sample_weight[0][0] = cache.sample_weight[0][1] = cache.sample_weight[0][2] = 1.0f;
            cache.cache_entries[0] = insert_entry(entries, P, N, params);
            if (cache.cache_entries[0] != HASH_GRID_INVALID_CACHE_ENTRY) {
                accumulate_cache_voxel(voxels_curr[cache.cache_entries[0]], rad, 1);
            }
            ++cache.path_len;

            for (int j = 1; j < cache.path_len; ++j) {
                rad *= make_fvec3(cache.sample_weight[j]);
                if (cache.cache_entries[j] != HASH_GRID_INVALID_CACHE_ENTRY) {
                    accumulate_cache_voxel(voxels_curr[cache.cache_entries[j]], rad, 0);
                }
            }
        }
    }
}

void Ray::Ref::SpatialCacheResolve(const cache_grid_params_t &params, Span<uint64_t> entries,
                                   Span<packed_cache_voxel_t> voxels_curr, Span<const packed_cache_voxel_t> voxels_prev,
                                   const uint32_t start, const uint32_t count) {
    assert((start % HASH_GRID_HASH_MAP_BUCKET_SIZE) == 0);
    assert((count % HASH_GRID_HASH_MAP_BUCKET_SIZE) == 0);
    const bool cam_moved = length2(make_fvec3(params.cam_pos_curr) - make_fvec3(params.cam_pos_prev)) > FLT_EPS;
    for (uint32_t i = start; i < start + count; ++i) {
        const uint64_t hash_key = entries[i];
        if (hash_key == HASH_GRID_INVALID_HASH_KEY) {
            continue;
        }

        const packed_cache_voxel_t voxel_curr = voxels_curr[i];
        packed_cache_voxel_t voxel_prev_packed = voxels_prev[i];
        cache_voxel_t voxel_prev = unpack_voxel_data(voxel_prev_packed);

        float sample_count_curr = float(voxel_curr.v[3]);
        float sample_count_prev = voxel_prev.sample_count;
        uint32_t accumulated_frame_count = voxel_prev.frame_count + 1;
        uint32_t accumulated_stale_count = voxel_prev.stale_count;
        accumulated_stale_count = (sample_count_curr != 0) ? 0 : (accumulated_stale_count + 1);

        if (accumulated_stale_count >= RAD_CACHE_STALE_FRAME_COUNT) {
            voxels_curr[i] = {};
            entries[i] = HASH_GRID_INVALID_HASH_KEY;
            continue;
        } else if (sample_count_curr == 0) {
            voxel_prev_packed.v[2] +=
                (1u << RAD_CACHE_FRAME_COUNTER_BIT_OFFSET) | (1u << RAD_CACHE_STALE_COUNTER_BIT_OFFSET);
            voxels_curr[i] = voxel_prev_packed;
            continue;
        }

        if (sample_count_prev == 0.0f) {
            for (uint32_t j = i + 1; j < i + 1 + RAD_CACHE_LINEAR_PROBE_WINDOW; ++j) {
                const uint32_t slot_index = j % HASH_GRID_CACHE_ENTRIES_COUNT;
                const uint64_t hash_key_old = entries[slot_index];
                if (hash_key_old == hash_key) {
                    voxel_prev = unpack_voxel_data(voxels_prev[slot_index]);
                    sample_count_prev = voxel_prev.sample_count;
                    accumulated_frame_count = voxel_prev.frame_count + 1;
                    accumulated_stale_count = 0;
                    break;
                }
            }
        }

        fvec4 accumulated_radiance =
            fvec4(uvec4{voxel_curr.v[0], voxel_curr.v[1], voxel_curr.v[2], 0u}) / RAD_CACHE_RADIANCE_SCALE;
        fvec4 accumulated_radiance_prev = make_fvec3(voxel_prev.radiance);

        if (accumulated_frame_count > RAD_CACHE_ACCUMULATION_FRAME_COUNT) {
            const float k = float(RAD_CACHE_ACCUMULATION_FRAME_COUNT) / float(accumulated_frame_count);
            accumulated_frame_count = RAD_CACHE_ACCUMULATION_FRAME_COUNT;
            sample_count_prev *= k;
        }

        float accumulated_sample_count = sample_count_prev + sample_count_curr;
        const float accumulated_sample_count_inv = 1.0f / accumulated_sample_count;

        accumulated_radiance =
            (sample_count_prev * accumulated_radiance_prev + accumulated_radiance) * accumulated_sample_count_inv;

        if (RAD_CACHE_FILTER_ADJACENT_LEVELS && cam_moved &&
            accumulated_frame_count <= RAD_CACHE_SAMPLE_COUNT_THRESHOLD) {
            const uint64_t adjacent_level_hash = get_adjacent_level_hash(hash_key, params);

            uint32_t cache_entry = HASH_GRID_INVALID_CACHE_ENTRY, collisions_count;
            if (hash_map_find(entries, adjacent_level_hash, cache_entry, collisions_count)) {
                const cache_voxel_t adjacent_voxel_prev = unpack_voxel_data(voxels_prev[cache_entry]);
                const float adjacent_sample_count = adjacent_voxel_prev.sample_count;
                if (adjacent_sample_count > RAD_CACHE_SAMPLE_COUNT_THRESHOLD) {
                    const float k = 1.0f / (adjacent_sample_count + accumulated_sample_count);

                    accumulated_radiance = (adjacent_sample_count * make_fvec3(adjacent_voxel_prev.radiance) +
                                            accumulated_sample_count * accumulated_radiance) *
                                           k;
                    accumulated_sample_count += adjacent_sample_count;
                }
            }
        }

        accumulated_radiance = min(accumulated_radiance, 65504.0f);

        packed_cache_voxel_t packed_data = {};
        packed_data.v[0] = packHalf2x16(accumulated_radiance.get<0>(), accumulated_radiance.get<1>());
        packed_data.v[1] = packHalf2x16(accumulated_radiance.get<2>(), accumulated_sample_count);
        packed_data.v[2] = accumulated_frame_count | (accumulated_stale_count << RAD_CACHE_STALE_COUNTER_BIT_OFFSET);

        voxels_curr[i] = packed_data;
    }
}
