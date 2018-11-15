R"(

__constant
int g_morton_table_16[] = { 0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85 };

__constant
int g_morton_table_256[] = {
    0,          1,          8,          9,          64,         65,         72,         73,         512,        513,        520,        521,        576,        577,        584,        585,
    4096,       4097,       4104,       4105,       4160,       4161,       4168,       4169,       4608,       4609,       4616,       4617,       4672,       4673,       4680,       4681,
    32768,      32769,      32776,      32777,      32832,      32833,      32840,      32841,      33280,      33281,      33288,      33289,      33344,      33345,      33352,      33353,
    36864,      36865,      36872,      36873,      36928,      36929,      36936,      36937,      37376,      37377,      37384,      37385,      37440,      37441,      37448,      37449,
    262144,     262145,     262152,     262153,     262208,     262209,     262216,     262217,     262656,     262657,     262664,     262665,     262720,     262721,     262728,     262729,
    266240,     266241,     266248,     266249,     266304,     266305,     266312,     266313,     266752,     266753,     266760,     266761,     266816,     266817,     266824,     266825,
    294912,     294913,     294920,     294921,     294976,     294977,     294984,     294985,     295424,     295425,     295432,     295433,     295488,     295489,     295496,     295497,
    299008,     299009,     299016,     299017,     299072,     299073,     299080,     299081,     299520,     299521,     299528,     299529,     299584,     299585,     299592,     299593,
    2097152,    2097153,    2097160,    2097161,    2097216,    2097217,    2097224,    2097225,    2097664,    2097665,    2097672,    2097673,    2097728,    2097729,    2097736,    2097737,
    2101248,    2101249,    2101256,    2101257,    2101312,    2101313,    2101320,    2101321,    2101760,    2101761,    2101768,    2101769,    2101824,    2101825,    2101832,    2101833,
    2129920,    2129921,    2129928,    2129929,    2129984,    2129985,    2129992,    2129993,    2130432,    2130433,    2130440,    2130441,    2130496,    2130497,    2130504,    2130505,
    2134016,    2134017,    2134024,    2134025,    2134080,    2134081,    2134088,    2134089,    2134528,    2134529,    2134536,    2134537,    2134592,    2134593,    2134600,    2134601,
    2359296,    2359297,    2359304,    2359305,    2359360,    2359361,    2359368,    2359369,    2359808,    2359809,    2359816,    2359817,    2359872,    2359873,    2359880,    2359881,
    2363392,    2363393,    2363400,    2363401,    2363456,    2363457,    2363464,    2363465,    2363904,    2363905,    2363912,    2363913,    2363968,    2363969,    2363976,    2363977,
    2392064,    2392065,    2392072,    2392073,    2392128,    2392129,    2392136,    2392137,    2392576,    2392577,    2392584,    2392585,    2392640,    2392641,    2392648,    2392649,
    2396160,    2396161,    2396168,    2396169,    2396224,    2396225,    2396232,    2396233,    2396672,    2396673,    2396680,    2396681,    2396736,    2396737,    2396744,    2396745
};

__constant
float g_omega_step = 0.0625f;

__constant
int g_omega_table[] = { 15, 14, 13, 12, 12, 11, 11, 11, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2, 1, 0 };

__constant
float g_phi_step = 0.125f;

__constant
int g_phi_table[][17] = { { 2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6  },
                          { 1,  2,  2,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  5,  6,  6  },
                          { 1,  1,  2,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6  },
                          { 1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  6,  6,  6,  6  },
                          { 1,  1,  1,  1,  2,  2,  2,  3,  4,  4,  5,  5,  6,  6,  6,  6,  6  },
                          { 0,  1,  1,  1,  1,  2,  2,  3,  4,  4,  5,  6,  6,  6,  6,  6,  7  },
                          { 0,  0,  0,  0,  1,  1,  2,  2,  4,  5,  6,  6,  6,  7,  7,  7,  7  },
                          { 0,  0,  0,  0,  0,  0,  1,  2,  4,  6,  6,  7,  7,  7,  7,  7,  7  },
                          { 15, 15, 15, 15, 15, 15, 15, 15, 8,  8,  8,  8,  8,  8,  8,  8,  8  },
                          { 15, 15, 15, 15, 15, 15, 14, 14, 12, 10, 9,  8,  8,  8,  8,  8,  8  },
                          { 15, 15, 15, 15, 14, 14, 14, 13, 12, 10, 10, 9,  9,  8,  8,  8,  8  },
                          { 15, 14, 14, 14, 14, 14, 13, 12, 12, 11, 10, 10, 9,  9,  9,  9,  8  },
                          { 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 10, 10, 9,  9,  9,  9  },
                          { 14, 14, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 9,  9,  9  },
                          { 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 10, 9,  9  },
                          { 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 10, 10, 9  },
                          { 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10 } };

uint get_ray_hash(__global ray_packet_t *r, const float3 root_min, const float3 cell_size) {
    int x = clamp((int)((r->o.x - root_min.x) / cell_size.x), 0, 255),
        y = clamp((int)((r->o.y - root_min.y) / cell_size.y), 0, 255),
        z = clamp((int)((r->o.z - root_min.z) / cell_size.z), 0, 255);

    x = g_morton_table_256[x];
    y = g_morton_table_256[y];
    z = g_morton_table_256[z];

    int oi = clamp((int)((1.0f + r->d.z) / g_omega_step), 0, 32);
    int pi = clamp((int)((1.0f + r->d.y) / g_phi_step), 0, 16),
        pj = clamp((int)((1.0f + r->d.x) / g_phi_step), 0, 16);

    int o = g_morton_table_16[g_omega_table[oi]];
    int p = g_morton_table_16[g_phi_table[pi][pj]];

    return (o << 25) | (p << 24) | (y << 2) | (z << 1) | (x << 0);
}

__kernel
void ComputeRayHashes(__global ray_packet_t *rays, float3 root_min, float3 cell_size, __global uint *out_hashes) {
    const int i = get_global_id(0);
    out_hashes[i] = get_ray_hash(&rays[i], root_min, cell_size);
}

__kernel
void SetHeadFlags(__global uint *hashes, __global uint *out_head_flags) {
    const int i = get_global_id(0);
    if (i == 0) {
        out_head_flags[i] = 1;
    } else {
        out_head_flags[i] = hashes[i] != hashes[i - 1];
    }
}

__kernel
void ExclusiveScan(__global uint *in_values, int bytes_offset, int bytes_stride, __global uint *out_scan_values, __global uint *partial_sums) {
    const int gi = get_global_id(0);
    const int li = get_local_id(0);

    __local uint temp[2][SCAN_PORTION];

    int pout = 0, pin = 1;

    temp[pout][li] = (li == 0) ? 0 : in_values[(bytes_stride/4) * (gi - 1) + (bytes_offset/4)];
    temp[pin][li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = 1; offset < SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            temp[pout][li] = temp[pin][li] + temp[pin][li - offset];
        } else {
            temp[pout][li] = temp[pin][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out_scan_values[gi] = temp[pout][li];
    if (li == get_local_size(0) - 1) {
        partial_sums[get_group_id(0)] = temp[pout][li] + in_values[(bytes_stride/4) * gi + (bytes_offset/4)];
    }
}

__kernel
void InclusiveScan(__global uint *in_values, __global uint *out_scan_values, __global uint *partial_sums) {
    const int gi = get_global_id(0);
    const int li = get_local_id(0);

    __local uint temp[2][SCAN_PORTION];

    int pout = 0, pin = 1;

    temp[pout][li] = in_values[gi];
    temp[pin][li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = 1; offset < SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            temp[pout][li] = temp[pin][li] + temp[pin][li - offset];
        } else {
            temp[pout][li] = temp[pin][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out_scan_values[gi] = temp[pout][li];
    if (li == get_local_size(0) - 1) {
        partial_sums[get_group_id(0)] = temp[pout][li];
    }
}

__kernel
void AddPartialSums(__global uint *in_values, __global uint *partial_sums) {
    const int global_index = get_global_id(0);
    const int group_index = get_group_id(0);
    if (group_index != 0) {
        in_values[global_index] += partial_sums[group_index - 1];
    }
}

__kernel
void InitChunkHashAndBase(__global ray_chunk_t *chunks, __global uint *hash_values, __global uint *head_flags, __global uint *scan_values) {
    const int i = get_global_id(0);

    if (head_flags[i]) {
        chunks[scan_values[i]].hash = hash_values[i];
        chunks[scan_values[i]].base = (uint)i;
    }
}

__kernel
void InitChunkSize(__global ray_chunk_t *chunks, int ray_count) {
    const int i = get_global_id(0);
    
    if (i == get_global_size(0) - 1) {
        chunks[i].size = (uint)ray_count - chunks[i].base;
    } else {
        chunks[i].size = chunks[i + 1].base - chunks[i].base;
    }
}

__kernel
void InitSkeletonAndHeadFlags(__global uint *scan_values, __global ray_chunk_t *chunks, __global uint *skeleton, __global uint *head_flags) {
    const int i = get_global_id(0);

    skeleton[scan_values[i]] = chunks[i].base;
    head_flags[scan_values[i]] = 1;
}

__kernel
void InitCountTable(__global ray_chunk_t *chunks, int shift, __global uint *counters) {
    const int gi = get_global_id(0);
    const int li = get_local_id(0);

    __local uint local_counters[0x10];
    
    for (int i = li; i < 0x10; i += get_local_size(0)) {
        local_counters[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_counters[(chunks[gi].hash >> shift) & 0xF]);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = li; i < 0x10; i += get_local_size(0)) {
        counters[i * get_num_groups(0) + get_group_id(0)] = local_counters[i];
    }
}

__kernel
void WriteSortedChunks(__global ray_chunk_t *chunks, __global uint *offsets, int shift, int group_size, int global_size, __global ray_chunk_t *chunks_out) {
    const int gi = get_global_id(0);

    uint local_offsets[0x10];
    for (int i = 0; i < 0x10; i++) {
        local_offsets[i] = offsets[i * global_size + gi];
    }

    for (int i = 0; i < group_size; i++) {
        uint index = (chunks[gi * group_size + i].hash >> shift) & 0xF;
        uint local_index = local_offsets[index]++;

        chunks_out[local_index] = chunks[gi * group_size + i];
    }
}

__kernel
void ExclusiveSegScan(__global uint *in_values, __global uint *in_flags, __global uint *out_scan_values, __global uint *partial_sums) {
    const int gi = get_global_id(0);
    const int li = get_local_id(0);

    __local uint temp[2][SEG_SCAN_PORTION];
    __local uint temp_flags[2][SEG_SCAN_PORTION];

    int pout = 0, pin = 1;

    temp[pout][li] = (li == 0) ? 0 : in_values[gi - 1];
    temp_flags[pout][li] = in_flags[gi];
    temp[pin][li] = 0;
    temp_flags[pin][li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = 1; offset < SEG_SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            temp_flags[pout][li] = temp_flags[pin][li] | temp_flags[pin][li - offset];
            temp[pout][li] = temp_flags[pin][li] ? temp[pin][li] : (temp[pin][li] + temp[pin][li - offset]);
        } else {
            temp[pout][li] = temp[pin][li];
            temp_flags[pout][li] = temp_flags[pin][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out_scan_values[gi] = temp[pout][li];
    if (li == get_local_size(0) - 1) {
        partial_sums[get_group_id(0)] = temp[pout][li] + in_values[gi];
    }
}

__kernel
void InclusiveSegScan(__global uint *in_values, __global uint *in_flags, __global uint *out_scan_values, __global uint *partial_sums, __global uint *partial_flags) {
    const int gi = get_global_id(0);
    const int li = get_local_id(0);

    __local uint temp[2][SEG_SCAN_PORTION];
    __local uint temp_flags[2][SEG_SCAN_PORTION];

    int pout = 0, pin = 1;

    temp[pout][li] = in_values[gi];
    temp_flags[pout][li] = in_flags[gi];
    temp[pin][li] = 0;
    temp_flags[pin][li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = 1; offset < SEG_SCAN_PORTION; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (li >= offset) {
            temp_flags[pout][li] = temp_flags[pin][li] | temp_flags[pin][li - offset];
            temp[pout][li] = temp_flags[pin][li] ? temp[pin][li] : (temp[pin][li] + temp[pin][li - offset]);
        } else {
            temp_flags[pout][li] = temp_flags[pin][li];
            temp[pout][li] = temp[pin][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out_scan_values[gi] = temp[pout][li];
    if (li == get_local_size(0) - 1) {
        partial_sums[get_group_id(0)] = temp[pout][li];
        partial_flags[get_group_id(0)] = temp_flags[pout][li];
    }
}

__kernel
void AddSegPartialSums(__global uint *flags, __global uint *in_values, __global uint *partial_sums, int group_size) {
    const int gi = get_global_id(0);    

    uint flag = 0;
    for (int i = 0; i < group_size && gi; i++) {
        flag = flag | flags[gi * group_size + i];
        in_values[gi * group_size + i] = flag ? in_values[gi * group_size + i] : (in_values[gi * group_size + i] + partial_sums[gi - 1]);
    }
}

__kernel
void ReorderRays(__global ray_packet_t *in_rays, __global uint *in_indices, __global ray_packet_t *out_rays) {
    const int gi = get_global_id(0);
    out_rays[gi] = in_rays[in_indices[gi]];
}

)"