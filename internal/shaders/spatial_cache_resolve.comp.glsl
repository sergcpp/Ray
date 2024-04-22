#version 450
#extension GL_GOOGLE_include_directive : require

#include "spatial_cache_resolve_interface.h"
#include "common.glsl"
#include "spatial_radiance_cache.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = CACHE_VOXELS_PREV_BUF_SLOT) buffer InOutCacheVoxelsPrev {
    uvec4 g_cache_voxels_prev[];
};

layout(std430, binding = INOUT_CACHE_ENTRIES_BUF_SLOT) buffer InOutCacheEntries {
    uint64_t g_inout_cache_entries[];
};

layout(std430, binding = INOUT_CACHE_VOXELS_CURR_BUF_SLOT) buffer InOutCacheVoxelsCurr {
    uvec4 g_inout_cache_voxels_curr[];
};

//
// Based on https://github.com/NVIDIAGameWorks/SHARC
//

bool hash_map_find(const uint64_t hash_key, inout uint cache_entry) {
    const uint hash = hash64(hash_key);
    const uint slot = hash % g_params.entries_count;
    const uint base_slot = hash_map_base_slot(slot);
    for (uint bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t stored_hash_key = g_inout_cache_entries[base_slot + bucket_offset];
        if (stored_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        } else if (HASH_GRID_ALLOW_COMPACTION && stored_hash_key == HASH_GRID_INVALID_HASH_KEY) {
            return false;
        }
    }
    return false;
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

shared uint g_temp[4];

void main() {
    const int index = int(gl_GlobalInvocationID.x);

    if ((gl_LocalInvocationIndex % 32) == 0) {
        g_temp[gl_LocalInvocationIndex / 32] = 0;
    }

    cache_grid_params_t grid_params;
    grid_params.cam_pos_curr = g_params.cam_pos_curr.xyz;
    grid_params.cam_pos_prev = g_params.cam_pos_prev.xyz;
    grid_params.log_base = RAD_CACHE_GRID_LOGARITHM_BASE;
    grid_params.scale = RAD_CACHE_GRID_SCALE;

    uvec4 packed_data = uvec4(0);

    const uint64_t hash_key = g_inout_cache_entries[index];
    if (hash_key != HASH_GRID_INVALID_HASH_KEY) {
        const uvec4 voxel_prev = g_cache_voxels_prev[index];
        const uvec4 voxel_curr = g_inout_cache_voxels_curr[index];
        packed_data = voxel_prev + voxel_curr;
        uint sample_count = packed_data.w & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;

        if (RAD_CACHE_FILTER_ADJACENT_LEVELS && g_params.cam_moved > 0.5 && sample_count < RAD_CACHE_SAMPLE_COUNT_MIN && voxel_curr.w != 0) {
            const uint64_t adjacent_level_hash = get_adjacent_level_hash(hash_key, grid_params);

            uint cache_entry = HASH_GRID_INVALID_CACHE_ENTRY;
            if (hash_map_find(adjacent_level_hash, cache_entry)) {
                const uvec4 adjacent_voxel_prev = g_cache_voxels_prev[cache_entry];
                const uint adjacent_sample_count = adjacent_voxel_prev.w & RAD_CACHE_SAMPLE_COUNTER_BIT_MASK;
                if (adjacent_sample_count > RAD_CACHE_SAMPLE_COUNT_MIN) {
                    /*packed_data.xyz += adjacent_voxel_prev.xyz;
                    sample_count += adjacent_sample_count;*/

                    // less 'sticky' version
                    const float k = float(RAD_CACHE_SAMPLE_COUNT_MIN) / float(adjacent_sample_count);
                    packed_data.xyz += uvec3(vec3(adjacent_voxel_prev.xyz) * k);
                    sample_count += RAD_CACHE_SAMPLE_COUNT_MIN;
                }
            }
        }

        if (sample_count > RAD_CACHE_SAMPLE_COUNT_MAX) {
            const float k = float(RAD_CACHE_SAMPLE_COUNT_MAX) / float(sample_count);
            packed_data.xyz = uvec3(vec3(packed_data.xyz) * k);
            sample_count = RAD_CACHE_SAMPLE_COUNT_MAX;
        }

        uint frame_count = (voxel_prev.w >> RAD_CACHE_SAMPLE_COUNTER_BIT_NUM) & RAD_CACHE_FRAME_COUNTER_BIT_MASK;
        packed_data.w = sample_count;

        if ((voxel_curr.w & RAD_CACHE_FRAME_COUNTER_BIT_MASK) == 0) {
            ++frame_count;
            packed_data.w |= (frame_count & RAD_CACHE_FRAME_COUNTER_BIT_MASK) << RAD_CACHE_SAMPLE_COUNTER_BIT_NUM;
        }

        if (frame_count > RAD_CACHE_STALE_FRAME_NUM_MAX) {
            packed_data = uvec4(0);
            if (!RAD_CACHE_ENABLE_COMPACTION) {
                g_inout_cache_entries[index] = uint64_t(HASH_GRID_INVALID_HASH_KEY);
            }
        }
    }

    if (RAD_CACHE_ENABLE_COMPACTION) {
        g_inout_cache_entries[index] = uint64_t(HASH_GRID_INVALID_HASH_KEY);
        g_inout_cache_voxels_curr[index] = uvec4(0);

        groupMemoryBarrier(); barrier();
        const uint temp = (packed_data.w != 0) ? 1u : 0u;
        atomicOr(g_temp[gl_LocalInvocationIndex / 32], (temp << (gl_LocalInvocationIndex % 32)));
        groupMemoryBarrier(); barrier();

        if (packed_data.w != 0) {
            const uint val = g_temp[gl_LocalInvocationIndex / 32] & ((1u << (gl_LocalInvocationIndex % 32)) - 1);
            const uint ndx = hash_map_base_slot(index) + bitCount(val);

            g_inout_cache_entries[ndx] = hash_key;
            g_inout_cache_voxels_curr[ndx] = packed_data;
        }
    } else {
        g_inout_cache_voxels_curr[index] = packed_data;
    }
}