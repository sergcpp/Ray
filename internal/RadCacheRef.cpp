#include "RadCacheRef.h"

namespace Ray {
namespace Ref {
// Based on logarithmic caching by Johannes Jendersie
ivec4 calc_grid_position_log(const fvec4 &p, const cache_grid_params_t &params) {
    const uint32_t grid_level = calc_grid_level(p, params);
    const float voxel_size = calc_voxel_size(grid_level, params);
    ivec4 grid_position = ivec4(floor(p / voxel_size));
    grid_position.set<3>(grid_level);
    return grid_position;
}

force_inline uint32_t hash_map_base_slot(const uint32_t slot) {
    if (HASH_GRID_ALLOW_COMPACTION) {
        return (slot / HASH_GRID_HASH_MAP_BUCKET_SIZE) * HASH_GRID_HASH_MAP_BUCKET_SIZE;
    } else {
        return slot;
    }
}

uint64_t compute_hash(const fvec4 &p, const fvec4 &n, const cache_grid_params_t &params) {
    const uvec4 grid_pos = uvec4(calc_grid_position_log(p, params));

    uint64_t hash_key =
        ((uint64_t(grid_pos.get<0>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0)) |
        ((uint64_t(grid_pos.get<1>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1)) |
        ((uint64_t(grid_pos.get<2>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2)) |
        ((uint64_t(grid_pos.get<3>()) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

    if (HASH_GRID_USE_NORMALS) {
        const uint32_t normal_bits = (n.get<0>() >= 0 ? 1 : 0) + (n.get<1>() >= 0 ? 2 : 0) + (n.get<2>() >= 0 ? 4 : 0);
        hash_key |= (uint64_t(normal_bits) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
    }

    return hash_key;
}

force_inline int grid_dist2(const ivec4 &pos) {
    return pos.get<0>() * pos.get<0>() + pos.get<1>() * pos.get<1>() + pos.get<2>() * pos.get<2>();
}

uint64_t get_adjacent_level_hash(const uint64_t hash_key, const cache_grid_params_t &params) {
    static const uint32_t NegativeBit = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    static const uint32_t NegativeMask = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    ivec4 grid_pos = 0;
    grid_pos.set<0>(int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 0) & HASH_GRID_POSITION_BIT_MASK));
    grid_pos.set<1>(int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 1) & HASH_GRID_POSITION_BIT_MASK));
    grid_pos.set<2>(int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 2) & HASH_GRID_POSITION_BIT_MASK));

    // Fix negative coordinates
    grid_pos.set<0>((grid_pos.get<0>() & NegativeBit) ? grid_pos.get<0>() | NegativeMask : grid_pos.get<0>());
    grid_pos.set<1>((grid_pos.get<1>() & NegativeBit) ? grid_pos.get<1>() | NegativeMask : grid_pos.get<1>());
    grid_pos.set<2>((grid_pos.get<2>() & NegativeBit) ? grid_pos.get<2>() | NegativeMask : grid_pos.get<2>());

    int level = uint32_t((hash_key >> (HASH_GRID_POSITION_BIT_NUM * 3)) & HASH_GRID_LEVEL_BIT_MASK);

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
        ((uint64_t(grid_pos.get<0>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0)) |
        ((uint64_t(grid_pos.get<1>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1)) |
        ((uint64_t(grid_pos.get<2>()) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2)) |
        ((uint64_t(level) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

    if (HASH_GRID_USE_NORMALS) {
        modified_hash_key |= hash_key & (uint64_t(HASH_GRID_NORMAL_BIT_MASK)
                                         << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
    }

    return modified_hash_key;
}

bool hash_map_insert(Span<uint64_t> entries, const uint64_t hash_key, uint32_t &cache_entry) {
    const uint32_t hash = hash64(hash_key);
    const uint32_t slot = hash % entries.size();
    const uint32_t base_slot = hash_map_base_slot(slot);
    for (uint32_t bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE && base_slot < entries.size();
         ++bucket_offset) {
        const uint64_t prev_hash_key = InterlockedCompareExchange64((long long *)&entries[base_slot + bucket_offset],
                                                                    hash_key, HASH_GRID_INVALID_HASH_KEY);
        if (prev_hash_key == HASH_GRID_INVALID_HASH_KEY || prev_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        }
    }
    cache_entry = 0;
    return false;
}

bool hash_map_find(Span<const uint64_t> entries, const uint64_t hash_key, uint32_t &cache_entry) {
    const uint32_t hash = hash64(hash_key);
    const uint32_t slot = hash % entries.size();
    const uint32_t base_slot = hash_map_base_slot(slot);
    for (uint32_t bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t stored_hash_key = entries[base_slot + bucket_offset];
        if (stored_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        } else if (HASH_GRID_ALLOW_COMPACTION && stored_hash_key == HASH_GRID_INVALID_HASH_KEY) {
            return false;
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

    InterlockedExchangeAdd((long *)&voxel.v[0], data.get<0>());
    InterlockedExchangeAdd((long *)&voxel.v[1], data.get<1>());
    InterlockedExchangeAdd((long *)&voxel.v[2], data.get<2>());
    InterlockedExchangeAdd((long *)&voxel.v[3], sample_data);
}
} // namespace Ref
} // namespace Ray

uint32_t Ray::Ref::calc_grid_level(const fvec4 &p, const cache_grid_params_t &params) {
    const float distance = length(make_fvec3(params.cam_pos_curr) - p);
    const float ret =
        Ray::clamp(floorf(log_base(distance, params.log_base) + HASH_GRID_LEVEL_BIAS), 1.0f, HASH_GRID_LEVEL_BIT_MASK);
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
    uint32_t cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
    hash_map_find(entries, hash_key, cache_entry);
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
        memcpy(cache.sample_weight[0], r.c, 3 * sizeof(float));
        if (inter.v < 0.0f || inter.obj_index < 0) {
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

            cache.cache_entries[0] = insert_entry(entries, P, N, params);
            if (cache.cache_entries[0] != HASH_GRID_INVALID_CACHE_ENTRY) {
                accumulate_cache_voxel(voxels_curr[cache.cache_entries[0]], rad, 1);
            }

            cache.path_len = std::min(cache.path_len + 1, RAD_CACHE_PROPAGATION_DEPTH - 1);

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
    for (uint32_t i = start; i < start + count; i += HASH_GRID_HASH_MAP_BUCKET_SIZE) {
        uint32_t ndx = i; // compact index
        for (uint32_t j = 0; j < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++j) {
            const uint64_t hash_key = entries[i + j];
            if (hash_key == HASH_GRID_INVALID_HASH_KEY) {
                continue;
            }

            const packed_cache_voxel_t voxel_prev = voxels_prev[i + j];
            const packed_cache_voxel_t voxel_curr = voxels_curr[i + j];
            packed_cache_voxel_t packed_data{voxel_prev.v[0] + voxel_curr.v[0], voxel_prev.v[1] + voxel_curr.v[1],
                                             voxel_prev.v[2] + voxel_curr.v[2], voxel_prev.v[3] + voxel_curr.v[3]};
            uint32_t sample_count = packed_data.v[3] & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;

            if (RAD_CACHE_FILTER_ADJACENT_LEVELS && cam_moved && sample_count < RAD_CACHE_SAMPLE_COUNT_MIN &&
                voxel_curr.v[3]) {
                const uint64_t adjacent_level_hash = get_adjacent_level_hash(hash_key, params);

                uint32_t cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
                if (hash_map_find(entries, adjacent_level_hash, cache_entry)) {
                    const packed_cache_voxel_t adjacent_voxel_prev = voxels_prev[cache_entry];
                    const uint32_t adjacent_sample_count = adjacent_voxel_prev.v[3] & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;
                    if (adjacent_sample_count > RAD_CACHE_SAMPLE_COUNT_MIN) {
                        /*packed_data.v[0] += adjacent_voxel_prev.v[0];
                        packed_data.v[1] += adjacent_voxel_prev.v[1];
                        packed_data.v[2] += adjacent_voxel_prev.v[2];
                        sample_count += adjacent_sample_count;*/

                        // less 'sticky' version
                        const float k = float(RAD_CACHE_SAMPLE_COUNT_MIN) / float(adjacent_sample_count);
                        packed_data.v[0] += uint32_t(float(adjacent_voxel_prev.v[0]) * k);
                        packed_data.v[1] += uint32_t(float(adjacent_voxel_prev.v[1]) * k);
                        packed_data.v[2] += uint32_t(float(adjacent_voxel_prev.v[2]) * k);
                        sample_count += RAD_CACHE_SAMPLE_COUNT_MIN;
                    }
                }
            }

            if (sample_count > RAD_CACHE_SAMPLE_COUNT_MAX) {
                const float k = float(RAD_CACHE_SAMPLE_COUNT_MAX) / float(sample_count);
                packed_data.v[0] = uint32_t(float(packed_data.v[0]) * k);
                packed_data.v[1] = uint32_t(float(packed_data.v[1]) * k);
                packed_data.v[2] = uint32_t(float(packed_data.v[2]) * k);
                sample_count = RAD_CACHE_SAMPLE_COUNT_MAX;
            }

            uint32_t frame_count =
                (voxel_prev.v[3] >> RAD_CACHE_SAMPLE_COUNTER_BIT_NUM) & RAD_CACHE_FRAME_COUNTER_BIT_MASK;
            packed_data.v[3] = sample_count;

            if ((voxel_curr.v[3] & RAD_CACHE_FRAME_COUNTER_BIT_MASK) == 0) {
                ++frame_count;
                packed_data.v[3] |= (frame_count & RAD_CACHE_FRAME_COUNTER_BIT_MASK)
                                    << RAD_CACHE_SAMPLE_COUNTER_BIT_NUM;
            }

            if (frame_count > RAD_CACHE_STALE_FRAME_NUM_MAX) {
                packed_data = {};
                if (!RAD_CACHE_ENABLE_COMPACTION) {
                    entries[i + j] = HASH_GRID_INVALID_HASH_KEY;
                }
            }

            if (RAD_CACHE_ENABLE_COMPACTION) {
                entries[i + j] = HASH_GRID_INVALID_HASH_KEY;
                voxels_curr[i + j] = {};
                if (packed_data.v[3]) {
                    entries[ndx] = hash_key;
                    voxels_curr[ndx++] = packed_data;
                }
            } else {
                voxels_curr[i + j] = packed_data;
            }
        }
    }
}