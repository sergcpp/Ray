#version 450
#extension GL_GOOGLE_include_directive : require

#include "sort_init_count_table_interface.h"
#include "common.glsl"
#include "sort_common.h"
#include "sort_hash.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HASHES_BUF_SLOT) readonly buffer RayChunks {
    ray_hash_t g_hashes[];
};

layout(std430, binding = COUNTERS_BUF_SLOT) readonly buffer Counters {
    uint g_counters[];
};

layout(std430, binding = OUT_COUNT_TABLE_BUF_SLOT) writeonly buffer OutBuf {
    uint g_count_table[];
};

layout(std430, binding = RAYS_BUF_SLOT) readonly buffer Rays {
    ray_data_t g_rays[];
};

layout(std430, binding = OUT_HASHES_BUF_SLOT) writeonly buffer OutHashes {
    ray_hash_t g_out_hashes[];
};

shared uint g_shared_counters[SORT_HIST_ROWS * SORT_BINS_COUNT];

layout (local_size_x = SORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

void main() {
    const int li = int(gl_LocalInvocationID.x), wi = int(gl_WorkGroupID.x);
    for (int i1 = li; i1 < SORT_HIST_ROWS * SORT_BINS_COUNT; i1 += SORT_THREADGROUP_SIZE) {
        g_shared_counters[i1] = 0;
    }
    groupMemoryBarrier(); barrier();

    const uint data_count = g_counters[g_params.counter];

    const int BlockSize = SORT_ELEMENTS_PER_THREAD * SORT_THREADGROUP_SIZE;
    const int block_count = int(data_count + BlockSize - 1) / BlockSize;
    const int block_start = BlockSize * wi;

    int data_index = block_start + li;

    uint hashes[SORT_ELEMENTS_PER_THREAD];
    [[unroll]] for (int i2 = 0; i2 < SORT_ELEMENTS_PER_THREAD; ++i2) {
        const int idx = data_index + i2 * SORT_THREADGROUP_SIZE;
        if (idx < data_count) {
            if (g_params.hash_input != 0) {
                // Pass 0: hash the rays inline (the old sort_hash_rays dispatch is fused here)
                const uint h = get_ray_hash(g_rays[idx], g_params.root_min.xyz, g_params.cell_size.xyz);
                g_out_hashes[idx] = ray_hash_t(h, uint(idx));
                hashes[i2] = h;
            } else {
                hashes[i2] = g_hashes[idx].hash;
            }
        }
    }

    for (int i3 = 0; i3 < SORT_ELEMENTS_PER_THREAD; ++i3) {
        if (data_index < data_count) {
            uint local_key = (hashes[i3] >> g_params.shift) & (SORT_BINS_COUNT - 1);
            atomicAdd(g_shared_counters[(li >> 5) * SORT_BINS_COUNT + local_key], 1u);
            data_index += SORT_THREADGROUP_SIZE;
        }
    }

    groupMemoryBarrier(); barrier();

    for (int bin = li; bin < SORT_BINS_COUNT; bin += SORT_THREADGROUP_SIZE) {
        uint sum = 0;
        for (int row = 0; row < SORT_HIST_ROWS; ++row) {
            sum += g_shared_counters[row * SORT_BINS_COUNT + bin];
        }
        g_count_table[bin * block_count + wi] = sum;
    }
}
