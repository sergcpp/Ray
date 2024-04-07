#ifndef RAD_CACHE_GLSL
#define RAD_CACHE_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "types.h"

cache_voxel_t unpack_voxel_data(const uvec4 v) {
    cache_voxel_t ret;
    ret.radiance = vec3(v.xyz) / RAD_CACHE_RADIANCE_SCALE;
    ret.sample_count = (v.w >> 0) & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;
    ret.frame_count = (v.w >> RAD_CACHE_SAMPLE_COUNTER_BIT_NUM) & RAD_CACHE_FRAME_COUNTER_BIT_MASK;
    return ret;
}

uint hash_jenkins32(uint a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

uint hash64(const uint64_t hash_key) {
    return hash_jenkins32(uint((hash_key >> 0) & 0xffffffff)) ^
           hash_jenkins32(uint((hash_key >> 32) & 0xffffffff));
}

float log_base(const float x, const float base) { return log(x) / log(base); }

uint calc_grid_level(const vec3 p, const cache_grid_params_t params) {
    const float distance = length(params.cam_pos_curr - p);
    const float ret =
        clamp(floor(log_base(distance, params.log_base) + HASH_GRID_LEVEL_BIAS), 1.0, HASH_GRID_LEVEL_BIT_MASK);
    return uint(ret);
}

float calc_voxel_size(uint grid_level, const cache_grid_params_t params) {
    return pow(params.log_base, grid_level) / (params.scale * pow(params.log_base, HASH_GRID_LEVEL_BIAS));
}

ivec4 calc_grid_position_log(const vec3 p, const cache_grid_params_t params) {
    const uint grid_level = calc_grid_level(p, params);
    const float voxel_size = calc_voxel_size(grid_level, params);
    ivec4 grid_position;
    grid_position.xyz = ivec3(floor(p / voxel_size));
    grid_position.w = int(grid_level);
    return grid_position;
}

uint hash_map_base_slot(const uint slot) {
    if (HASH_GRID_ALLOW_COMPACTION) {
        return (slot / HASH_GRID_HASH_MAP_BUCKET_SIZE) * HASH_GRID_HASH_MAP_BUCKET_SIZE;
    } else {
        return slot;
    }
}

uint64_t compute_hash(const vec3 p, const vec3 n, const cache_grid_params_t params) {
    const uvec4 grid_pos = uvec4(calc_grid_position_log(p, params));

    uint64_t hash_key =
        ((uint64_t(grid_pos.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0)) |
        ((uint64_t(grid_pos.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1)) |
        ((uint64_t(grid_pos.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2)) |
        ((uint64_t(grid_pos.w) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

    if (HASH_GRID_USE_NORMALS) {
        const uint normal_bits = (n.x >= 0 ? 1 : 0) + (n.y >= 0 ? 2 : 0) + (n.z >= 0 ? 4 : 0);
        hash_key |= (uint64_t(normal_bits) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
    }

    return hash_key;
}

int grid_dist2(const ivec3 pos) {
    return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
}

vec3 GetColorFromHash32(const uint hash) {
    vec3 color;
    color.x = float((hash >> 0) & 0x3ff) / 1023.0;
    color.y = float((hash >> 11) & 0x7ff) / 2047.0;
    color.z = float((hash >> 22) & 0x7ff) / 2047.0;
    return color;
}

vec3 hash_grid_debug(const vec3 p, const vec3 n, const cache_grid_params_t params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    return GetColorFromHash32(hash64(hash_key));
}

uint64_t get_adjacent_level_hash(const uint64_t hash_key, const cache_grid_params_t params) {
    const uint NegativeBit = 1u << (HASH_GRID_POSITION_BIT_NUM - 1);
    const uint NegativeMask = ~((1u << HASH_GRID_POSITION_BIT_NUM) - 1);

    ivec3 grid_pos;
    grid_pos.x = int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 0) & HASH_GRID_POSITION_BIT_MASK);
    grid_pos.y = int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 1) & HASH_GRID_POSITION_BIT_MASK);
    grid_pos.z = int((hash_key >> HASH_GRID_POSITION_BIT_NUM * 2) & HASH_GRID_POSITION_BIT_MASK);

    // Fix negative coordinates
    grid_pos.x = int((grid_pos.x & NegativeBit) != 0 ? grid_pos.x | NegativeMask : grid_pos.x);
    grid_pos.y = int((grid_pos.y & NegativeBit) != 0 ? grid_pos.y | NegativeMask : grid_pos.y);
    grid_pos.z = int((grid_pos.z & NegativeBit) != 0 ? grid_pos.z | NegativeMask : grid_pos.z);

    int level = int((hash_key >> (HASH_GRID_POSITION_BIT_NUM * 3)) & HASH_GRID_LEVEL_BIT_MASK);

    const float voxel_size = calc_voxel_size(level, params);
    const ivec3 camera_grid_pos_curr = ivec3(floor(params.cam_pos_curr / voxel_size));
    const ivec3 camera_vector_curr = camera_grid_pos_curr - grid_pos;
    const int camera_distance_curr = grid_dist2(camera_vector_curr);

    const ivec3 camera_grid_pos_prev = ivec3(floor(params.cam_pos_prev / voxel_size));
    const ivec3 camera_vector_prev = camera_grid_pos_prev - grid_pos;
    const int camera_distance_prev = grid_dist2(camera_vector_prev);

    if (camera_distance_curr < camera_distance_prev) {
        grid_pos = ivec3(floor(vec3(grid_pos) / params.log_base));
        level = min(level + 1, int(HASH_GRID_LEVEL_BIT_MASK));
    } else {
        grid_pos = ivec3(floor(vec3(grid_pos) * params.log_base));
        level = max(level - 1, 1);
    }

    uint64_t modified_hash_key =
        ((uint64_t(grid_pos.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0)) |
        ((uint64_t(grid_pos.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1)) |
        ((uint64_t(grid_pos.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2)) |
        ((uint64_t(level) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

    if (HASH_GRID_USE_NORMALS) {
        modified_hash_key |= hash_key & (uint64_t(HASH_GRID_NORMAL_BIT_MASK)
                                         << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
    }

    return modified_hash_key;
}

#endif // RAD_CACHE_GLSL