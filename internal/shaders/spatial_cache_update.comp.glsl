#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_atomic_int64 : require

#include "spatial_cache_update_interface.h"
#include "common.glsl"
#include "spatial_radiance_cache.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HITS_BUF_SLOT) readonly buffer Hits {
    hit_data_t g_hits[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(binding = RADIANCE_TEX_SLOT) uniform texture2D g_radiance_tex;
layout(binding = DEPTH_NORMAL_TEX_SLOT) uniform texture2D g_depth_normal_tex;

layout(std430, binding = INOUT_CACHE_DATA_BUF_SLOT) buffer InOutCacheData {
    cache_data_t g_inout_cache_data[];
};

layout(std430, binding = INOUT_CACHE_ENTRIES_BUF_SLOT) buffer InOutCacheEntries {
    uint64_t g_inout_cache_entries[];
};

layout(std430, binding = INOUT_CACHE_VOXELS_BUF_SLOT) buffer InOutCacheVoxels {
    uint g_inout_cache_voxels[];
};

#if NO_64BIT_ATOMICS
layout(std430, binding = INOUT_CACHE_LOCK_BUF_SLOT) buffer InOutLockBuffer {
    uint g_inout_lock_buffer[];
};
#endif

//
// Based on https://github.com/NVIDIAGameWorks/SHARC
//

uint64_t AtomicCompSwapEntry(const uint i, const uint64_t compare, const uint64_t data) {
#if !NO_64BIT_ATOMICS
    uint64_t ret = atomicCompSwap(g_inout_cache_entries[i], compare, data);
#else
    const uint Retries = 8;
    uint fuse = 0;
    bool busy = true;
    uint64_t ret = 0xffffffffu;
    while (busy && fuse < Retries) {
        const uint state = atomicExchange(g_inout_lock_buffer[i], 0xAAAAAAAAu);
        busy = (state != 0);
        if (state != 0xAAAAAAAAu) {
            ret = g_inout_cache_entries[i];
            if (ret == compare) {
                g_inout_cache_entries[i] = data;
            }
            atomicExchange(g_inout_lock_buffer[i], state);
            fuse = Retries;
        }
        ++fuse;
    }
#endif
    return ret;
}

bool hash_map_insert(const uint64_t hash_key, out uint cache_entry) {
    const uint hash = hash64(hash_key);
    const uint slot = hash % g_params.entries_count;
    const uint base_slot = hash_map_base_slot(slot);
    for (uint bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE && base_slot < g_params.entries_count;
         ++bucket_offset) {
        const uint64_t prev_hash_key = AtomicCompSwapEntry(base_slot + bucket_offset, HASH_GRID_INVALID_HASH_KEY, hash_key);
        if (prev_hash_key == HASH_GRID_INVALID_HASH_KEY || prev_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        }
    }
    cache_entry = 0;
    return false;
}

uint insert_entry(const vec3 p, const vec3 n, const cache_grid_params_t params) {
    const uint64_t hash_key = compute_hash(p, n, params);
    uint cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
    hash_map_insert(hash_key, cache_entry);
    return cache_entry;
}

void accumulate_cache_voxel(const uint i, const vec3 r, const uint sample_data) {
    const uvec3 data = uvec3(r * RAD_CACHE_RADIANCE_SCALE);
    if (data.x > 0) {
        atomicAdd(g_inout_cache_voxels[4 * i + 0], data.x);
    }
    if (data.y > 0) {
        atomicAdd(g_inout_cache_voxels[4 * i + 1], data.y);
    }
    if (data.z > 0) {
        atomicAdd(g_inout_cache_voxels[4 * i + 2], data.z);
    }
    if (sample_data > 0) {
        atomicAdd(g_inout_cache_voxels[4 * i + 3], sample_data);
    }
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

void main() {
    const int index = int(gl_WorkGroupID.x * 64 + gl_LocalInvocationIndex);
    if (index >= g_counters[6]) {
        return;
    }

    cache_grid_params_t grid_params;
    grid_params.cam_pos_curr = g_params.cam_pos_curr.xyz;
    grid_params.log_base = RAD_CACHE_GRID_LOGARITHM_BASE;
    grid_params.scale = RAD_CACHE_GRID_SCALE;
    grid_params.exposure = g_params.exposure;

    const int x = int((g_rays[index].xy >> 16) & 0xffff);
    const int y = int(g_rays[index].xy & 0xffff);

    const hit_data_t inter = g_hits[index];
    const ray_data_t ray = g_rays[index];

    const vec3 ro = vec3(ray.o[0], ray.o[1], ray.o[2]);
    const vec3 I = vec3(ray.d[0], ray.d[1], ray.d[2]);

    const vec3 P = ro + inter.t * I;
    const vec3 N = texelFetch(g_depth_normal_tex, ivec2(x, y), 0).xyz;
    vec3 rad = texelFetch(g_radiance_tex, ivec2(x, y), 0).rgb * grid_params.exposure;

    cache_data_t cache = g_inout_cache_data[y * g_params.cache_w + x];
    cache.sample_weight[0][0] = ray.c[0];
    cache.sample_weight[0][1] = ray.c[1];
    cache.sample_weight[0][2] = ray.c[2];
    if (inter.v < 0.0 || inter.obj_index < 0 || cache.path_len == RAD_CACHE_PROPAGATION_DEPTH) {
        for (int j = 0; j < cache.path_len; ++j) {
            rad *= vec3(cache.sample_weight[j][0], cache.sample_weight[j][1], cache.sample_weight[j][2]);
            if (cache.cache_entries[j] != HASH_GRID_INVALID_CACHE_ENTRY) {
                accumulate_cache_voxel(cache.cache_entries[j], rad, 0);
            }
        }
    } else {
        for (int j = cache.path_len; j > 0; --j) {
            cache.cache_entries[j] = cache.cache_entries[j - 1];
            for (int k = 0; k < 3; ++k) {
                cache.sample_weight[j][k] = cache.sample_weight[j - 1][k];
            }
        }

        cache.sample_weight[0][0] = cache.sample_weight[0][1] = cache.sample_weight[0][2] = 1.0;
        cache.cache_entries[0] = insert_entry(P, N, grid_params);
        if (cache.cache_entries[0] != HASH_GRID_INVALID_CACHE_ENTRY) {
            accumulate_cache_voxel(cache.cache_entries[0], rad, 1);
        }
        ++cache.path_len;

        for (int j = 1; j < cache.path_len; ++j) {
            rad *= vec3(cache.sample_weight[j][0], cache.sample_weight[j][1], cache.sample_weight[j][2]);
            if (cache.cache_entries[j] != HASH_GRID_INVALID_CACHE_ENTRY) {
                accumulate_cache_voxel(cache.cache_entries[j], rad, 0);
            }
        }
    }
    g_inout_cache_data[y * g_params.cache_w + x] = cache;
}
