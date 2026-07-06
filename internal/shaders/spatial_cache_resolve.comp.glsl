#version 450
#extension GL_GOOGLE_include_directive : require

#include "spatial_cache_resolve_interface.h"
#include "common.glsl"
#include "spatial_radiance_cache.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = IN_CACHE_VOXELS_PREV_BUF_SLOT) buffer InCacheVoxelsPrev {
    uvec4 g_in_cache_voxels_prev[];
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

bool hash_map_find(const uint64_t hash_key, inout uint cache_entry, out uint bucket_offset) {
    const uint base_slot = hash_map_base_slot(hash_key);
    for (bucket_offset = 0; bucket_offset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucket_offset) {
        const uint64_t stored_hash_key = g_inout_cache_entries[base_slot + bucket_offset];
        if (stored_hash_key == hash_key) {
            cache_entry = base_slot + bucket_offset;
            return true;
        }
    }
    return false;
}

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int index = int(gl_GlobalInvocationID.x);

    cache_grid_params_t grid_params;
    grid_params.cam_pos_curr = g_params.cam_pos_curr.xyz;
    grid_params.cam_pos_prev = g_params.cam_pos_prev.xyz;
    grid_params.log_base = RAD_CACHE_GRID_LOGARITHM_BASE;
    grid_params.scale = RAD_CACHE_GRID_SCALE;

    const uint64_t hash_key = g_inout_cache_entries[index];
    if (hash_key != HASH_GRID_INVALID_HASH_KEY) {
        uvec4 voxel_prev_packed = g_in_cache_voxels_prev[index];
        cache_voxel_t voxel_prev = unpack_voxel_data(voxel_prev_packed);
        const uvec4 voxel_curr = g_inout_cache_voxels_curr[index];

        float sample_count_curr = float(voxel_curr.w);
        float sample_count_prev = voxel_prev.sample_count;
        uint accumulated_frame_count = voxel_prev.frame_count + 1;
        uint accumulated_stale_count = voxel_prev.stale_count;
        accumulated_stale_count = (sample_count_curr != 0) ? 0 : (accumulated_stale_count + 1);

        if (accumulated_stale_count >= RAD_CACHE_STALE_FRAME_COUNT) {
            g_inout_cache_voxels_curr[index] = uvec4(0);
            g_inout_cache_entries[index] = uint64_t(HASH_GRID_INVALID_HASH_KEY);
            return;
        } else if (sample_count_curr == 0) {
            voxel_prev_packed.z += (1u << RAD_CACHE_FRAME_COUNTER_BIT_OFFSET) | (1u << RAD_CACHE_STALE_COUNTER_BIT_OFFSET);
            g_inout_cache_voxels_curr[index] = voxel_prev_packed;
            return;
        }

        if (sample_count_prev == 0.0) {
            for (uint j = index + 1; j < index + 1 + RAD_CACHE_LINEAR_PROBE_WINDOW; ++j) {
                const uint slot_index = j % HASH_GRID_CACHE_ENTRIES_COUNT;
                const uint64_t hash_key_old = g_inout_cache_entries[slot_index];
                if (hash_key_old == hash_key) {
                    voxel_prev = unpack_voxel_data(g_in_cache_voxels_prev[slot_index]);
                    sample_count_prev = voxel_prev.sample_count;
                    accumulated_frame_count = voxel_prev.frame_count + 1;
                    accumulated_stale_count = 0;
                    break;
                }
            }
        }

        vec3 accumulated_radiance = vec3(voxel_curr.xyz) / RAD_CACHE_RADIANCE_SCALE;
        vec3 accumulated_radiance_prev = voxel_prev.radiance;

        if (accumulated_frame_count > RAD_CACHE_ACCUMULATION_FRAME_COUNT) {
            const float k = float(RAD_CACHE_ACCUMULATION_FRAME_COUNT) / float(accumulated_frame_count);
            accumulated_frame_count = RAD_CACHE_ACCUMULATION_FRAME_COUNT;
            sample_count_prev *= k;
        }

        float accumulated_sample_count = sample_count_prev + sample_count_curr;
        const float accumulated_sample_count_inv = 1.0 / accumulated_sample_count;

        accumulated_radiance = (sample_count_prev * accumulated_radiance_prev + accumulated_radiance) * accumulated_sample_count_inv;

        if (RAD_CACHE_FILTER_ADJACENT_LEVELS && g_params.cam_moved > 0.5 && accumulated_frame_count <= 2) {
            const uint64_t adjacent_level_hash = get_adjacent_level_hash(hash_key, grid_params);

            uint cache_entry = HASH_GRID_INVALID_CACHE_ENTRY, collisions_count;
            if (hash_map_find(adjacent_level_hash, cache_entry, collisions_count)) {
                const cache_voxel_t adjacent_voxel_prev = unpack_voxel_data(g_in_cache_voxels_prev[cache_entry]);
                const float adjacent_sample_count = adjacent_voxel_prev.sample_count;
                if (adjacent_sample_count > RAD_CACHE_SAMPLE_COUNT_THRESHOLD) {
                    const float k = 1.0 / float(adjacent_sample_count + accumulated_sample_count);

                    accumulated_radiance = (adjacent_sample_count * adjacent_voxel_prev.radiance + accumulated_sample_count * accumulated_radiance) * k;
                    accumulated_sample_count += adjacent_sample_count;
                }
            }
        }

        accumulated_radiance = min(accumulated_radiance, 65504.0);

        uvec4 packed_data = uvec4(0);
        packed_data.x = packHalf2x16(accumulated_radiance.xy);
        packed_data.y = packHalf2x16(vec2(accumulated_radiance.z, accumulated_sample_count));
        packed_data.z = accumulated_frame_count | (accumulated_stale_count << RAD_CACHE_STALE_COUNTER_BIT_OFFSET);

        g_inout_cache_voxels_curr[index] = packed_data;
    }
}