#include "Core.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <deque>
#include <limits>
#include <vector>

#include "BVHSplit.h"

namespace Ray {
const int g_primes[] = {
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,   41,   43,   47,   53,   59,   61,   67,
    71,   73,   79,   83,   89,   97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,  157,  163,
    167,  173,  179,  181,  191,  193,  197,  199,  211,  223,  227,  229,  233,  239,  241,  251,  257,  263,  269,
    271,  277,  281,  283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,  367,  373,  379,  383,
    389,  397,  401,  409,  419,  421,  431,  433,  439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,
    503,  509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,  599,  601,  607,  613,  617,  619,
    631,  641,  643,  647,  653,  659,  661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,  751,
    757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,  829,  839,  853,  857,  859,  863,  877,  881,
    883,  887,  907,  911,  919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,  1009, 1013, 1019,
    1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
    1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289,
    1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567,
    1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
    1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867,
    1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003,
    2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141,
    2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297,
    2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437,
    2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617,
    2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731,
    2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887,
    2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049,
    3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221,
    3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371,
    3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533,
    3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673,
    3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833,
    3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001,
    4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139,
    4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289,
    4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481,
    4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643,
    4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799,
    4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969,
    4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113,
    5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303,
    5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471,
    5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641,
    5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791,
    5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927,
    5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121,
    6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277,
    6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427,
    6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619,
    6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791,
    6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959,
    6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121,
    7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307,
    7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499,
    7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639,
    7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817,
    7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919};

force_inline Ref::simd_fvec3 cross(const Ref::simd_fvec3 &v1, const Ref::simd_fvec3 &v2) {
    return {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
}

// "Insert" two 0 bits after each of the 10 low bits of x
force_inline uint32_t Part1By2(uint32_t x) {
    x = x & 0b00000000000000000000001111111111;               // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0b00000011000000000000000011111111; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0b00000011000000001111000000001111;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0b00000011000011000011000011000011;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0b00001001001001001001001001001001;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

force_inline uint32_t EncodeMorton3(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

struct prim_chunk_t {
    uint32_t code, base, size;
};

const int BitsPerDim = 10;
const int BitsTotal = 3 * BitsPerDim;

void radix_sort_prim_chunks(prim_chunk_t *begin, prim_chunk_t *end, prim_chunk_t *begin1) {
    prim_chunk_t *end1 = begin1 + (end - begin);

    const int bits_per_pass = 6;
    const int bucket_size = 1 << bits_per_pass;
    const int bit_mask = bucket_size - 1;

    for (int shift = 0; shift < BitsTotal; shift += bits_per_pass) {
        size_t count[bucket_size] = {};
        for (prim_chunk_t *p = begin; p != end; p++) {
            count[(p->code >> shift) & bit_mask]++;
        }
        prim_chunk_t *bucket[bucket_size], *q = begin1;
        for (int i = 0; i < bucket_size; q += count[i++]) {
            bucket[i] = q;
        }
        for (prim_chunk_t *p = begin; p != end; p++) {
            *bucket[(p->code >> shift) & bit_mask]++ = *p;
        }
        std::swap(begin, begin1);
        std::swap(end, end1);
    }
}

void sort_mort_codes(uint32_t *morton_codes, size_t prims_count, uint32_t *out_indices) {
    std::vector<prim_chunk_t> run_chunks;
    run_chunks.reserve(prims_count);

    for (uint32_t start = 0, end = 1; end <= (uint32_t)prims_count; end++) {
        if (end == (uint32_t)prims_count || (morton_codes[start] != morton_codes[end])) {

            run_chunks.push_back({morton_codes[start], start, end - start});

            start = end;
        }
    }

    std::vector<prim_chunk_t> run_chunks2(run_chunks.size());

    radix_sort_prim_chunks(&run_chunks[0], &run_chunks[0] + run_chunks.size(), &run_chunks2[0]);
    std::swap(run_chunks, run_chunks2);

    size_t counter = 0;
    for (const prim_chunk_t &ch : run_chunks) {
        for (uint32_t j = 0; j < ch.size; j++) {
            morton_codes[counter] = ch.code;
            out_indices[counter++] = ch.base + j;
        }
    }
}
} // namespace Ray

void Ray::CanonicalToDir(const float p[2], const float y_rotation, float out_d[3]) {
    const float cos_theta = 2 * p[0] - 1;
    float phi = 2 * PI * p[1] + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float sin_theta = std::sqrt(1 - cos_theta * cos_theta);

    const float sin_phi = std::sin(phi);
    const float cos_phi = std::cos(phi);

    out_d[0] = sin_theta * cos_phi;
    out_d[1] = cos_theta;
    out_d[2] = -sin_theta * sin_phi;
}

void Ray::DirToCanonical(const float d[3], const float y_rotation, float out_p[2]) {
    const float cos_theta = std::min(std::max(d[1], -1.0f), 1.0f);

    float phi = -std::atan2(d[2], d[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    out_p[0] = (cos_theta + 1.0f) / 2.0f;
    out_p[1] = phi / (2.0f * PI);
}

// Used to convert 16x16 sphere sector coordinates to single value
const uint8_t Ray::morton_table_16[] = {0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85};

// Used to convert 256x256x256 grid coordinates to single value, i think it leads to more uniform distribution than (z
// << 16) | (y << 8) | (x << 0)
const int Ray::morton_table_256[] = {
    0,       1,       8,       9,       64,      65,      72,      73,      512,     513,     520,     521,     576,
    577,     584,     585,     4096,    4097,    4104,    4105,    4160,    4161,    4168,    4169,    4608,    4609,
    4616,    4617,    4672,    4673,    4680,    4681,    32768,   32769,   32776,   32777,   32832,   32833,   32840,
    32841,   33280,   33281,   33288,   33289,   33344,   33345,   33352,   33353,   36864,   36865,   36872,   36873,
    36928,   36929,   36936,   36937,   37376,   37377,   37384,   37385,   37440,   37441,   37448,   37449,   262144,
    262145,  262152,  262153,  262208,  262209,  262216,  262217,  262656,  262657,  262664,  262665,  262720,  262721,
    262728,  262729,  266240,  266241,  266248,  266249,  266304,  266305,  266312,  266313,  266752,  266753,  266760,
    266761,  266816,  266817,  266824,  266825,  294912,  294913,  294920,  294921,  294976,  294977,  294984,  294985,
    295424,  295425,  295432,  295433,  295488,  295489,  295496,  295497,  299008,  299009,  299016,  299017,  299072,
    299073,  299080,  299081,  299520,  299521,  299528,  299529,  299584,  299585,  299592,  299593,  2097152, 2097153,
    2097160, 2097161, 2097216, 2097217, 2097224, 2097225, 2097664, 2097665, 2097672, 2097673, 2097728, 2097729, 2097736,
    2097737, 2101248, 2101249, 2101256, 2101257, 2101312, 2101313, 2101320, 2101321, 2101760, 2101761, 2101768, 2101769,
    2101824, 2101825, 2101832, 2101833, 2129920, 2129921, 2129928, 2129929, 2129984, 2129985, 2129992, 2129993, 2130432,
    2130433, 2130440, 2130441, 2130496, 2130497, 2130504, 2130505, 2134016, 2134017, 2134024, 2134025, 2134080, 2134081,
    2134088, 2134089, 2134528, 2134529, 2134536, 2134537, 2134592, 2134593, 2134600, 2134601, 2359296, 2359297, 2359304,
    2359305, 2359360, 2359361, 2359368, 2359369, 2359808, 2359809, 2359816, 2359817, 2359872, 2359873, 2359880, 2359881,
    2363392, 2363393, 2363400, 2363401, 2363456, 2363457, 2363464, 2363465, 2363904, 2363905, 2363912, 2363913, 2363968,
    2363969, 2363976, 2363977, 2392064, 2392065, 2392072, 2392073, 2392128, 2392129, 2392136, 2392137, 2392576, 2392577,
    2392584, 2392585, 2392640, 2392641, 2392648, 2392649, 2396160, 2396161, 2396168, 2396169, 2396224, 2396225, 2396232,
    2396233, 2396672, 2396673, 2396680, 2396681, 2396736, 2396737, 2396744, 2396745};

// Used to bind horizontal vector angle to sector on sphere
const float Ray::omega_step = 0.0625f;
const char Ray::omega_table[] = {15, 14, 13, 12, 12, 11, 11, 11, 10, 10, 9, 9, 9, 8, 8, 8, 8,
                                 7,  7,  7,  6,  6,  6,  5,  5,  4,  4,  4, 3, 3, 2, 1, 0};

// Used to bind vectical vector angle to sector on sphere
const float Ray::phi_step = 0.125f;
// clang-format off
const char Ray::phi_table[][17] = {{ 2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6  },
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
                                   { 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10 }};

const int Ray::ray_packet_pixel_layout[] = {
    0, 1, 2, 3,   1, 2, 3, 0,
    2, 3, 0, 1,   3, 0, 1, 2,
    1, 2, 3, 0,   2, 1, 0, 3,
    3, 0, 1, 2,   0, 3, 2, 1,

    2, 3, 0, 1,   1, 0, 3, 2,
    0, 1, 2, 3,   0, 3, 2, 1,
    3, 2, 1, 0,   3, 2, 1, 0,
    1, 0, 3, 2,   2, 1, 0, 3
};
// clang-format on

bool Ray::PreprocessTri(const float *p, int stride, tri_accel_t *out_acc) {
    if (!stride) {
        stride = 3;
    }

    // edges
    const float e0[3] = {p[stride] - p[0], p[stride + 1] - p[1], p[stride + 2] - p[2]},
                e1[3] = {p[stride * 2] - p[0], p[stride * 2 + 1] - p[1], p[stride * 2 + 2] - p[2]};

    float n[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};

    const float n_len_sqr = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (n_len_sqr == 0.0f) {
        // degenerate triangle
        return false;
    }

    const float u[3] = {(e1[1] * n[2] - e1[2] * n[1]) / n_len_sqr, (e1[2] * n[0] - e1[0] * n[2]) / n_len_sqr,
                        (e1[0] * n[1] - e1[1] * n[0]) / n_len_sqr};

    // edge planes
    out_acc->u_plane[0] = u[0];
    out_acc->u_plane[1] = u[1];
    out_acc->u_plane[2] = u[2];
    out_acc->u_plane[3] = -(u[0] * p[0] + u[1] * p[1] + u[2] * p[2]);

    const float v[3] = {(n[1] * e0[2] - n[2] * e0[1]) / n_len_sqr, (n[2] * e0[0] - n[0] * e0[2]) / n_len_sqr,
                        (n[0] * e0[1] - n[1] * e0[0]) / n_len_sqr};

    out_acc->v_plane[0] = v[0];
    out_acc->v_plane[1] = v[1];
    out_acc->v_plane[2] = v[2];
    out_acc->v_plane[3] = -(v[0] * p[0] + v[1] * p[1] + v[2] * p[2]);

    // normal plane
    const float l = std::sqrt(n_len_sqr);
    n[0] /= l;
    n[1] /= l;
    n[2] /= l;

    out_acc->n_plane[0] = n[0];
    out_acc->n_plane[1] = n[1];
    out_acc->n_plane[2] = n[2];
    out_acc->n_plane[3] = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];

    return true;
}

uint32_t Ray::PreprocessMesh(const float *attrs, Span<const uint32_t> vtx_indices, const eVertexLayout layout,
                             const int base_vertex, const uint32_t tris_start, const bvh_settings_t &s,
                             std::vector<bvh_node_t> &out_nodes, std::vector<tri_accel_t> &out_tris,
                             std::vector<uint32_t> &out_tri_indices, aligned_vector<mtri_accel_t> &out_tris2) {
    assert(!vtx_indices.empty() && vtx_indices.size() % 3 == 0);

    std::vector<prim_t> primitives;
    std::vector<tri_accel_t> triangles;
    std::vector<uint32_t> real_indices;

    primitives.reserve(vtx_indices.size() / 3);
    triangles.reserve(vtx_indices.size() / 3);
    real_indices.reserve(vtx_indices.size() / 3);

    const float *positions = attrs;
    const size_t attr_stride = AttrStrides[layout];

    for (int j = 0; j < int(vtx_indices.size()); j += 3) {
        Ref::simd_fvec4 p[3] = {{0.0f}, {0.0f}, {0.0f}};

        const uint32_t i0 = vtx_indices[j + 0] + base_vertex, i1 = vtx_indices[j + 1] + base_vertex,
                       i2 = vtx_indices[j + 2] + base_vertex;

        memcpy(&p[0][0], &positions[i0 * attr_stride], 3 * sizeof(float));
        memcpy(&p[1][0], &positions[i1 * attr_stride], 3 * sizeof(float));
        memcpy(&p[2][0], &positions[i2 * attr_stride], 3 * sizeof(float));

        tri_accel_t tri = {};
        if (PreprocessTri(&p[0][0], 4, &tri)) {
            real_indices.push_back(uint32_t(j / 3));
            triangles.push_back(tri);
        } else {
            continue;
        }

        const Ref::simd_fvec4 _min = min(p[0], min(p[1], p[2])), _max = max(p[0], max(p[1], p[2]));

        primitives.push_back({i0, i1, i2, _min, _max});
    }

    const size_t indices_start = out_tri_indices.size();
    uint32_t num_out_nodes;
    if (!s.use_fast_bvh_build) {
        num_out_nodes = PreprocessPrims_SAH(primitives, positions, attr_stride, s, out_nodes, out_tri_indices);
    } else {
        num_out_nodes = PreprocessPrims_HLBVH(primitives, out_nodes, out_tri_indices);
    }

    // make sure mesh occupies the whole 8-triangle slot
    while (out_tri_indices.size() % 8) {
        out_tri_indices.push_back(0);
    }

    // const uint32_t tris_start = out_tris2.size(), tris_count = out_tri_indices.size() - indices_start;
    out_tris.resize(out_tri_indices.size());
    out_tris2.resize(out_tri_indices.size() / 8);

    for (size_t i = indices_start; i < out_tri_indices.size(); i++) {
        const uint32_t j = out_tri_indices[i];

        out_tris[i] = triangles[j];

        for (int k = 0; k < 4; ++k) {
            out_tris2[i / 8].n_plane[k][i % 8] = triangles[j].n_plane[k];
            out_tris2[i / 8].u_plane[k][i % 8] = triangles[j].u_plane[k];
            out_tris2[i / 8].v_plane[k][i % 8] = triangles[j].v_plane[k];
        }

        out_tri_indices[i] = uint32_t(real_indices[j] + tris_start);
    }

    return num_out_nodes;
}

uint32_t Ray::EmitLBVH_Recursive(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes,
                                 uint32_t prim_index, uint32_t prim_count, uint32_t index_offset, int bit_index,
                                 std::vector<bvh_node_t> &out_nodes) {
    if (bit_index == -1 || prim_count < 8) {
        Ref::simd_fvec4 bbox_min = {std::numeric_limits<float>::max()},
                        bbox_max = {std::numeric_limits<float>::lowest()};

        for (uint32_t i = prim_index; i < prim_index + prim_count; i++) {
            bbox_min = min(bbox_min, prims[indices[i]].bbox_min);
            bbox_max = max(bbox_max, prims[indices[i]].bbox_max);
        }

        const auto node_index = uint32_t(out_nodes.size());

        out_nodes.emplace_back();
        bvh_node_t &node = out_nodes.back();

        node.prim_index = LEAF_NODE_BIT + prim_index + index_offset;
        node.prim_count = prim_count;

        memcpy(&node.bbox_min[0], &bbox_min[0], 3 * sizeof(float));
        memcpy(&node.bbox_max[0], &bbox_max[0], 3 * sizeof(float));

        return node_index;
    } else {
        const uint32_t mask = (1u << bit_index);

        if ((morton_codes[prim_index] & mask) == (morton_codes[prim_index + prim_count - 1] & mask)) {
            return EmitLBVH_Recursive(prims, indices, morton_codes, prim_index, prim_count, index_offset, bit_index - 1,
                                      out_nodes);
        }

        uint32_t search_start = prim_index, search_end = search_start + prim_count - 1;
        while (search_start + 1 != search_end) {
            const uint32_t mid = (search_start + search_end) / 2;
            if ((morton_codes[search_start] & mask) == (morton_codes[mid] & mask)) {
                search_start = mid;
            } else {
                search_end = mid;
            }
        }

        const uint32_t split_offset = search_end - prim_index;

        const auto node_index = uint32_t(out_nodes.size());
        out_nodes.emplace_back();

        uint32_t child0 = EmitLBVH_Recursive(prims, indices, morton_codes, prim_index, split_offset, index_offset,
                                             bit_index - 1, out_nodes);
        uint32_t child1 = EmitLBVH_Recursive(prims, indices, morton_codes, prim_index + split_offset,
                                             prim_count - split_offset, index_offset, bit_index - 1, out_nodes);

        uint32_t space_axis = bit_index % 3;
        if (out_nodes[child0].bbox_min[space_axis] > out_nodes[child1].bbox_min[space_axis]) {
            std::swap(child0, child1);
        }

        bvh_node_t &par_node = out_nodes[node_index];
        par_node.left_child = child0;
        par_node.right_child = (space_axis << 30) + child1;

        for (int i = 0; i < 3; i++) {
            par_node.bbox_min[i] = std::min(out_nodes[child0].bbox_min[i], out_nodes[child1].bbox_min[i]);
            par_node.bbox_max[i] = std::max(out_nodes[child0].bbox_max[i], out_nodes[child1].bbox_max[i]);
        }

        return node_index;
    }
}

uint32_t Ray::EmitLBVH_NonRecursive(const prim_t *prims, const uint32_t *indices, const uint32_t *morton_codes,
                                    uint32_t prim_index, uint32_t prim_count, uint32_t index_offset, int bit_index,
                                    std::vector<bvh_node_t> &out_nodes) {
    struct proc_item_t {
        int bit_index;
        uint32_t prim_index, prim_count;
        uint32_t split_offset, node_index;
    };

    proc_item_t proc_stack[256];
    uint32_t stack_size = 0;

    const auto root_node_index = uint32_t(out_nodes.size());
    out_nodes.emplace_back();
    proc_stack[stack_size++] = {bit_index, prim_index, prim_count, 0xffffffff, root_node_index};

    while (stack_size) {
        proc_item_t &cur = proc_stack[stack_size - 1];

        if (cur.bit_index == -1 || cur.prim_count < 8) {
            Ref::simd_fvec4 bbox_min = {std::numeric_limits<float>::max()},
                            bbox_max = {std::numeric_limits<float>::lowest()};

            for (uint32_t i = cur.prim_index; i < cur.prim_index + cur.prim_count; i++) {
                bbox_min = min(bbox_min, prims[indices[i]].bbox_min);
                bbox_max = max(bbox_max, prims[indices[i]].bbox_max);
            }

            bvh_node_t &node = out_nodes[cur.node_index];

            node.prim_index = LEAF_NODE_BIT + cur.prim_index + index_offset;
            node.prim_count = cur.prim_count;

            memcpy(&node.bbox_min[0], &bbox_min[0], 3 * sizeof(float));
            memcpy(&node.bbox_max[0], &bbox_max[0], 3 * sizeof(float));
        } else {
            if (cur.split_offset == 0xffffffff) {
                const uint32_t mask = (1u << cur.bit_index);

                uint32_t search_start = cur.prim_index, search_end = search_start + cur.prim_count - 1;

                if ((morton_codes[search_start] & mask) == (morton_codes[search_end] & mask)) {
                    cur.bit_index--;
                    continue;
                }

                while (search_start + 1 != search_end) {
                    const uint32_t mid = (search_start + search_end) / 2;
                    if ((morton_codes[search_start] & mask) == (morton_codes[mid] & mask)) {
                        search_start = mid;
                    } else {
                        search_end = mid;
                    }
                }

                cur.split_offset = search_end - cur.prim_index;

                const auto child0 = uint32_t(out_nodes.size());
                out_nodes.emplace_back();

                const auto child1 = uint32_t(out_nodes.size());
                out_nodes.emplace_back();

                out_nodes[cur.node_index].left_child = child0;
                out_nodes[cur.node_index].right_child = child1;

                proc_stack[stack_size++] = {cur.bit_index - 1, cur.prim_index + cur.split_offset,
                                            cur.prim_count - cur.split_offset, 0xffffffff, child1};
                proc_stack[stack_size++] = {cur.bit_index - 1, cur.prim_index, cur.split_offset, 0xffffffff, child0};
                continue;
            } else {
                bvh_node_t &node = out_nodes[cur.node_index];

                for (int i = 0; i < 3; i++) {
                    node.bbox_min[i] =
                        std::min(out_nodes[node.left_child].bbox_min[i], out_nodes[node.right_child].bbox_min[i]);
                    node.bbox_max[i] =
                        std::max(out_nodes[node.left_child].bbox_max[i], out_nodes[node.right_child].bbox_max[i]);
                }

                const uint32_t space_axis = (cur.bit_index % 3);
                if (out_nodes[node.left_child].bbox_min[space_axis] >
                    out_nodes[node.right_child].bbox_min[space_axis]) {
                    std::swap(node.left_child, node.right_child);
                }
                node.right_child += (space_axis << 30);
            }
        }

        stack_size--;
    }

    return root_node_index;
}

uint32_t Ray::PreprocessPrims_SAH(Span<const prim_t> prims, const float *positions, const size_t stride,
                                  const bvh_settings_t &s, std::vector<bvh_node_t> &out_nodes,
                                  std::vector<uint32_t> &out_indices) {
    struct prims_coll_t {
        std::vector<uint32_t> indices;
        Ref::simd_fvec4 min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(), 0.0f},
                        max = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest(), 0.0f};
        prims_coll_t() = default;
        prims_coll_t(std::vector<uint32_t> &&_indices, const Ref::simd_fvec4 &_min, const Ref::simd_fvec4 &_max)
            : indices(std::move(_indices)), min(_min), max(_max) {}
    };

    std::deque<prims_coll_t> prim_lists;
    prim_lists.emplace_back();

    size_t num_nodes = out_nodes.size();
    const auto root_node_index = uint32_t(num_nodes);

    for (uint32_t j = 0; j < uint32_t(prims.size()); j++) {
        prim_lists.back().indices.push_back(j);
        prim_lists.back().min = min(prim_lists.back().min, prims[j].bbox_min);
        prim_lists.back().max = max(prim_lists.back().max, prims[j].bbox_max);
    }

    Ref::simd_fvec4 root_min = prim_lists.back().min, root_max = prim_lists.back().max;

    while (!prim_lists.empty()) {
        split_data_t split_data =
            SplitPrimitives_SAH(prims.data(), prim_lists.back().indices, positions, stride, prim_lists.back().min,
                                prim_lists.back().max, root_min, root_max, s);
        prim_lists.pop_back();

        if (split_data.right_indices.empty()) {
            Ref::simd_fvec4 bbox_min = split_data.left_bounds[0], bbox_max = split_data.left_bounds[1];

            out_nodes.emplace_back();
            bvh_node_t &n = out_nodes.back();

            n.prim_index = LEAF_NODE_BIT + uint32_t(out_indices.size());
            n.prim_count = uint32_t(split_data.left_indices.size());
            memcpy(&n.bbox_min[0], &bbox_min[0], 3 * sizeof(float));
            memcpy(&n.bbox_max[0], &bbox_max[0], 3 * sizeof(float));
            out_indices.insert(out_indices.end(), split_data.left_indices.begin(), split_data.left_indices.end());
        } else {
            const auto index = uint32_t(num_nodes);

            uint32_t space_axis = 0;
            const Ref::simd_fvec4 c_left = (split_data.left_bounds[0] + split_data.left_bounds[1]) / 2.0f,
                                  c_right = (split_data.right_bounds[0] + split_data.right_bounds[1]) / 2.0f;

            const Ref::simd_fvec4 dist = abs(c_left - c_right);

            if (dist[0] > dist[1] && dist[0] > dist[2]) {
                space_axis = 0;
            } else if (dist[1] > dist[0] && dist[1] > dist[2]) {
                space_axis = 1;
            } else {
                space_axis = 2;
            }

            const Ref::simd_fvec4 bbox_min = min(split_data.left_bounds[0], split_data.right_bounds[0]),
                                  bbox_max = max(split_data.left_bounds[1], split_data.right_bounds[1]);

            out_nodes.emplace_back();
            bvh_node_t &n = out_nodes.back();
            n.left_child = index + 1;
            n.right_child = (space_axis << 30) + index + 2;
            memcpy(&n.bbox_min[0], value_ptr(bbox_min), 3 * sizeof(float));
            memcpy(&n.bbox_max[0], value_ptr(bbox_max), 3 * sizeof(float));
            prim_lists.emplace_front(std::move(split_data.left_indices), split_data.left_bounds[0],
                                     split_data.left_bounds[1]);
            prim_lists.emplace_front(std::move(split_data.right_indices), split_data.right_bounds[0],
                                     split_data.right_bounds[1]);

            num_nodes += 2;
        }
    }

    return uint32_t(out_nodes.size() - root_node_index);
}

uint32_t Ray::PreprocessPrims_HLBVH(Span<const prim_t> prims, std::vector<bvh_node_t> &out_nodes,
                                    std::vector<uint32_t> &out_indices) {
    std::vector<uint32_t> morton_codes(prims.size());

    Ref::simd_fvec4 whole_min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(), 0.0f},
                    whole_max = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::lowest(), 0.0f};

    const auto indices_start = uint32_t(out_indices.size());
    out_indices.reserve(out_indices.size() + prims.size());

    for (uint32_t j = 0; j < prims.size(); j++) {
        whole_min = min(whole_min, prims[j].bbox_min);
        whole_max = max(whole_max, prims[j].bbox_max);

        out_indices.push_back(j);
    }

    uint32_t *indices = &out_indices[indices_start];

    const Ref::simd_fvec4 scale =
        float(1 << BitsPerDim) / (whole_max - whole_min + std::numeric_limits<float>::epsilon());

    // compute morton codes
    for (int i = 0; i < int(prims.size()); i++) {
        const Ref::simd_fvec4 center = 0.5f * (prims[i].bbox_min + prims[i].bbox_max);
        const Ref::simd_fvec4 code = (center - whole_min) * scale;

        const auto x = uint32_t(code[0]), y = uint32_t(code[1]), z = uint32_t(code[2]);

        morton_codes[i] = EncodeMorton3(x, y, z);
    }

    sort_mort_codes(&morton_codes[0], morton_codes.size(), indices);

    struct treelet_t {
        uint32_t index, count;
        uint32_t node_index;
    };

    std::vector<treelet_t> treelets;
    treelets.reserve(1 << 12); // Top-level bvh can have up to 4096 items

    // Use upper 12 bits to extract treelets
    for (uint32_t start = 0, end = 1; end <= (uint32_t)morton_codes.size(); end++) {
        uint32_t mask = 0b00111111111111000000000000000000;
        if (end == (uint32_t)morton_codes.size() || ((morton_codes[start] & mask) != (morton_codes[end] & mask))) {

            treelets.push_back({start, end - start});

            start = end;
        }
    }

    std::vector<bvh_node_t> bottom_nodes;

    // Build bottom-level hierarchy from each treelet using LBVH
    const int start_bit = 29 - 12;
    for (treelet_t &tr : treelets) {
        tr.node_index = EmitLBVH_NonRecursive(prims.data(), indices, &morton_codes[0], tr.index, tr.count,
                                              indices_start, start_bit, bottom_nodes);
    }

    std::vector<prim_t> top_prims;
    for (const treelet_t &tr : treelets) {
        const bvh_node_t &node = bottom_nodes[tr.node_index];

        top_prims.emplace_back();
        prim_t &p = top_prims.back();
        memcpy(&p.bbox_min[0], node.bbox_min, 3 * sizeof(float));
        memcpy(&p.bbox_max[0], node.bbox_max, 3 * sizeof(float));
    }

    const auto top_nodes_start = uint32_t(out_nodes.size());

    std::vector<uint32_t> top_indices;

    // Force spliting until each primitive will be in separate leaf node
    bvh_settings_t s;
    s.oversplit_threshold = -1.0f;
    s.allow_spatial_splits = false;
    s.min_primitives_in_leaf = 1;

    // Build top level hierarchy using SAH
    const uint32_t new_nodes_count =
        PreprocessPrims_SAH({&top_prims[0], top_prims.size()}, nullptr, 0, s, out_nodes, top_indices);
    unused(new_nodes_count);

    auto bottom_nodes_start = uint32_t(out_nodes.size());

    // Replace leaf nodes of top-level bvh with bottom level nodes
    for (uint32_t i = top_nodes_start; i < (uint32_t)out_nodes.size(); i++) {
        bvh_node_t &n = out_nodes[i];
        if (!(n.prim_index & LEAF_NODE_BIT)) {
            bvh_node_t &left = out_nodes[n.left_child], &right = out_nodes[n.right_child & RIGHT_CHILD_BITS];

            if (left.prim_index & LEAF_NODE_BIT) {
                assert(left.prim_count == 1);
                const uint32_t index = (left.prim_index & PRIM_INDEX_BITS);

                const treelet_t &tr = treelets[top_indices[index]];
                n.left_child = bottom_nodes_start + tr.node_index;
            }

            if (right.prim_index & LEAF_NODE_BIT) {
                assert(right.prim_count == 1);
                const uint32_t index = (right.prim_index & PRIM_INDEX_BITS);

                const treelet_t &tr = treelets[top_indices[index]];
                n.right_child = (n.right_child & SEP_AXIS_BITS) + bottom_nodes_start + tr.node_index;
            }
        }
    }

    // Remove top-level leaf nodes
    for (auto it = out_nodes.begin() + top_nodes_start; it != out_nodes.end();) {
        if (it->prim_index & LEAF_NODE_BIT) {
            const auto index = uint32_t(std::distance(out_nodes.begin(), it));

            it = out_nodes.erase(it);
            bottom_nodes_start--;

            for (auto next_it = out_nodes.begin() + top_nodes_start; next_it != out_nodes.end(); ++next_it) {
                if (!(next_it->prim_index & LEAF_NODE_BIT)) {
                    if (next_it->left_child > index) {
                        next_it->left_child--;
                    }
                    if ((next_it->right_child & RIGHT_CHILD_BITS) > index) {
                        next_it->right_child--;
                    }
                }
            }
        } else {
            ++it;
        }
    }

    const uint32_t bottom_nodes_offset = bottom_nodes_start;

    // Offset nodes in bottom-level bvh
    for (bvh_node_t &n : bottom_nodes) {
        if (!(n.prim_index & LEAF_NODE_BIT)) {
            n.left_child += bottom_nodes_offset;
            n.right_child += bottom_nodes_offset;
        }
    }

    out_nodes.insert(out_nodes.end(), bottom_nodes.begin(), bottom_nodes.end());

    return uint32_t(out_nodes.size() - top_nodes_start);
}

uint32_t Ray::FlattenBVH_Recursive(const bvh_node_t *nodes, const uint32_t node_index, const uint32_t parent_index,
                                   aligned_vector<mbvh_node_t> &out_nodes) {
    const bvh_node_t &cur_node = nodes[node_index];

    // allocate new node
    const auto new_node_index = uint32_t(out_nodes.size());
    out_nodes.emplace_back();

    if (cur_node.prim_index & LEAF_NODE_BIT) {
        mbvh_node_t &new_node = out_nodes[new_node_index];

        new_node.bbox_min[0][0] = cur_node.bbox_min[0];
        new_node.bbox_min[1][0] = cur_node.bbox_min[1];
        new_node.bbox_min[2][0] = cur_node.bbox_min[2];

        new_node.bbox_max[0][0] = cur_node.bbox_max[0];
        new_node.bbox_max[1][0] = cur_node.bbox_max[1];
        new_node.bbox_max[2][0] = cur_node.bbox_max[2];

        new_node.child[0] = cur_node.prim_index;
        new_node.child[1] = cur_node.prim_count;

        return new_node_index;
    }

    // Gather children 2 levels deep

    uint32_t children[8];
    int children_count = 0;

    const bvh_node_t &child0 = nodes[cur_node.left_child];

    if (child0.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.left_child & LEFT_CHILD_BITS;
    } else {
        const bvh_node_t &child00 = nodes[child0.left_child];
        const bvh_node_t &child01 = nodes[child0.right_child & RIGHT_CHILD_BITS];

        if (child00.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child00.left_child;
            children[children_count++] = child00.right_child & RIGHT_CHILD_BITS;
        }

        if (child01.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child0.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child01.left_child;
            children[children_count++] = child01.right_child & RIGHT_CHILD_BITS;
        }
    }

    const bvh_node_t &child1 = nodes[cur_node.right_child & RIGHT_CHILD_BITS];

    if (child1.prim_index & LEAF_NODE_BIT) {
        children[children_count++] = cur_node.right_child & RIGHT_CHILD_BITS;
    } else {
        const bvh_node_t &child10 = nodes[child1.left_child];
        const bvh_node_t &child11 = nodes[child1.right_child & RIGHT_CHILD_BITS];

        if (child10.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.left_child & LEFT_CHILD_BITS;
        } else {
            children[children_count++] = child10.left_child;
            children[children_count++] = child10.right_child & RIGHT_CHILD_BITS;
        }

        if (child11.prim_index & LEAF_NODE_BIT) {
            children[children_count++] = child1.right_child & RIGHT_CHILD_BITS;
        } else {
            children[children_count++] = child11.left_child;
            children[children_count++] = child11.right_child & RIGHT_CHILD_BITS;
        }
    }

    // Sort children in morton order
    Ref::simd_fvec3 children_centers[8], whole_box_min = {std::numeric_limits<float>::max()},
                                         whole_box_max = {std::numeric_limits<float>::lowest()};
    for (int i = 0; i < children_count; i++) {
        children_centers[i] =
            0.5f * (Ref::simd_fvec3{nodes[children[i]].bbox_min} + Ref::simd_fvec3{nodes[children[i]].bbox_max});
        whole_box_min = min(whole_box_min, children_centers[i]);
        whole_box_max = max(whole_box_max, children_centers[i]);
    }

    whole_box_max += Ref::simd_fvec3{0.001f};

    const Ref::simd_fvec3 scale = 2.0f / (whole_box_max - whole_box_min);

    uint32_t sorted_children[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                   0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    for (int i = 0; i < children_count; i++) {
        Ref::simd_fvec3 code = (children_centers[i] - whole_box_min) * scale;

        const auto x = uint32_t(code[0]), y = uint32_t(code[1]), z = uint32_t(code[2]);

        uint32_t mort = (z << 2) | (y << 1) | (x << 0);

        while (sorted_children[mort] != 0xffffffff) {
            mort = (mort + 1) % 8;
        }

        sorted_children[mort] = children[i];
    }

    uint32_t new_children[8];

    for (int i = 0; i < 8; i++) {
        if (sorted_children[i] != 0xffffffff) {
            new_children[i] = FlattenBVH_Recursive(nodes, sorted_children[i], node_index, out_nodes);
        } else {
            new_children[i] = 0x7fffffff;
        }
    }

    mbvh_node_t &new_node = out_nodes[new_node_index];
    memcpy(new_node.child, new_children, sizeof(new_children));

    for (int i = 0; i < 8; i++) {
        if (new_children[i] != 0x7fffffff) {
            new_node.bbox_min[0][i] = nodes[sorted_children[i]].bbox_min[0];
            new_node.bbox_min[1][i] = nodes[sorted_children[i]].bbox_min[1];
            new_node.bbox_min[2][i] = nodes[sorted_children[i]].bbox_min[2];

            new_node.bbox_max[0][i] = nodes[sorted_children[i]].bbox_max[0];
            new_node.bbox_max[1][i] = nodes[sorted_children[i]].bbox_max[1];
            new_node.bbox_max[2][i] = nodes[sorted_children[i]].bbox_max[2];
        } else {
            // Init as invalid bounding box
            new_node.bbox_min[0][i] = new_node.bbox_min[1][i] = new_node.bbox_min[2][i] = 0.0f;
            new_node.bbox_max[0][i] = new_node.bbox_max[1][i] = new_node.bbox_max[2][i] = 0.0f;
        }
    }

    return new_node_index;
}

bool Ray::NaiivePluckerTest(const float p[9], const float o[3], const float d[3]) {
    // plucker coordinates for edges
    const float e0[6] = {p[6] - p[0],
                         p[7] - p[1],
                         p[8] - p[2],
                         p[7] * p[2] - p[8] * p[1],
                         p[8] * p[0] - p[6] * p[2],
                         p[6] * p[1] - p[7] * p[0]},
                e1[6] = {p[3] - p[6],
                         p[4] - p[7],
                         p[5] - p[8],
                         p[4] * p[8] - p[5] * p[7],
                         p[5] * p[6] - p[3] * p[8],
                         p[3] * p[7] - p[4] * p[6]},
                e2[6] = {p[0] - p[3],
                         p[1] - p[4],
                         p[2] - p[5],
                         p[1] * p[5] - p[2] * p[4],
                         p[2] * p[3] - p[0] * p[5],
                         p[0] * p[4] - p[1] * p[3]};

    // plucker coordinates for Ray
    const float R[6] = {
        d[1] * o[2] - d[2] * o[1], d[2] * o[0] - d[0] * o[2], d[0] * o[1] - d[1] * o[0], d[0], d[1], d[2]};

    float t0 = 0, t1 = 0, t2 = 0;
    for (int w = 0; w < 6; w++) {
        t0 += e0[w] * R[w];
        t1 += e1[w] * R[w];
        t2 += e2[w] * R[w];
    }

    return (t0 <= 0 && t1 <= 0 && t2 <= 0) || (t0 >= 0 && t1 >= 0 && t2 >= 0);
}

void Ray::ConstructCamera(const eCamType type, const eFilterType filter, eDeviceType dtype, const float origin[3],
                          const float fwd[3], const float up[3], const float shift[2], const float fov,
                          const float sensor_height, const float gamma, const float focus_distance, const float fstop,
                          const float lens_rotation, const float lens_ratio, const int lens_blades,
                          const float clip_start, const float clip_end, camera_t *cam) {
    if (type == Persp) {
        auto o = Ref::simd_fvec3{origin}, f = Ref::simd_fvec3{fwd}, u = Ref::simd_fvec3{up};

        if (u.length2() < FLT_EPS) {
            if (std::abs(f[1]) >= 0.999f) {
                u = {1.0f, 0.0f, 0.0f};
            } else {
                u = {0.0f, 1.0f, 0.0f};
            }
        }

        const Ref::simd_fvec3 s = normalize(cross(f, u));
        u = cross(s, f);

        cam->type = type;
        cam->filter = filter;
        cam->dtype = dtype;
        cam->ltype = eLensUnits::FOV;
        cam->fov = fov;
        cam->gamma = gamma;
        cam->sensor_height = sensor_height;
        cam->focus_distance = std::max(focus_distance, 0.0f);
        cam->focal_length = 0.5f * sensor_height / std::tan(0.5f * fov * PI / 180.0f);
        cam->fstop = fstop;
        cam->lens_rotation = lens_rotation;
        cam->lens_ratio = lens_ratio;
        cam->lens_blades = lens_blades;
        cam->clip_start = clip_start;
        cam->clip_end = clip_end;
        memcpy(&cam->origin[0], &o[0], 3 * sizeof(float));
        memcpy(&cam->fwd[0], &f[0], 3 * sizeof(float));
        memcpy(&cam->side[0], value_ptr(s), 3 * sizeof(float));
        memcpy(&cam->up[0], &u[0], 3 * sizeof(float));
        memcpy(&cam->shift[0], shift, 2 * sizeof(float));
    } else if (type == Ortho) {
        // TODO!
    }
}

void Ray::TransformBoundingBox(const float bbox_min[3], const float bbox_max[3], const float *xform,
                               float out_bbox_min[3], float out_bbox_max[3]) {
    out_bbox_min[0] = out_bbox_max[0] = xform[12];
    out_bbox_min[1] = out_bbox_max[1] = xform[13];
    out_bbox_min[2] = out_bbox_max[2] = xform[14];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            const float a = xform[i * 4 + j] * bbox_min[i];
            const float b = xform[i * 4 + j] * bbox_max[i];

            if (a < b) {
                out_bbox_min[j] += a;
                out_bbox_max[j] += b;
            } else {
                out_bbox_min[j] += b;
                out_bbox_max[j] += a;
            }
        }
    }
}

void Ray::InverseMatrix(const float mat[16], float out_mat[16]) {
    const float A2323 = mat[10] * mat[15] - mat[11] * mat[14];
    const float A1323 = mat[9] * mat[15] - mat[11] * mat[13];
    const float A1223 = mat[9] * mat[14] - mat[10] * mat[13];
    const float A0323 = mat[8] * mat[15] - mat[11] * mat[12];
    const float A0223 = mat[8] * mat[14] - mat[10] * mat[12];
    const float A0123 = mat[8] * mat[13] - mat[9] * mat[12];
    const float A2313 = mat[6] * mat[15] - mat[7] * mat[14];
    const float A1313 = mat[5] * mat[15] - mat[7] * mat[13];
    const float A1213 = mat[5] * mat[14] - mat[6] * mat[13];
    const float A2312 = mat[6] * mat[11] - mat[7] * mat[10];
    const float A1312 = mat[5] * mat[11] - mat[7] * mat[9];
    const float A1212 = mat[5] * mat[10] - mat[6] * mat[9];
    const float A0313 = mat[4] * mat[15] - mat[7] * mat[12];
    const float A0213 = mat[4] * mat[14] - mat[6] * mat[12];
    const float A0312 = mat[4] * mat[11] - mat[7] * mat[8];
    const float A0212 = mat[4] * mat[10] - mat[6] * mat[8];
    const float A0113 = mat[4] * mat[13] - mat[5] * mat[12];
    const float A0112 = mat[4] * mat[9] - mat[5] * mat[8];

    const float inv_det = 1.0f / (mat[0] * (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223) -
                                  mat[1] * (mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223) +
                                  mat[2] * (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123) -
                                  mat[3] * (mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123));

    out_mat[0] = inv_det * (mat[5] * A2323 - mat[6] * A1323 + mat[7] * A1223);
    out_mat[1] = inv_det * -(mat[1] * A2323 - mat[2] * A1323 + mat[3] * A1223);
    out_mat[2] = inv_det * (mat[1] * A2313 - mat[2] * A1313 + mat[3] * A1213);
    out_mat[3] = inv_det * -(mat[1] * A2312 - mat[2] * A1312 + mat[3] * A1212);
    out_mat[4] = inv_det * -(mat[4] * A2323 - mat[6] * A0323 + mat[7] * A0223);
    out_mat[5] = inv_det * (mat[0] * A2323 - mat[2] * A0323 + mat[3] * A0223);
    out_mat[6] = inv_det * -(mat[0] * A2313 - mat[2] * A0313 + mat[3] * A0213);
    out_mat[7] = inv_det * (mat[0] * A2312 - mat[2] * A0312 + mat[3] * A0212);
    out_mat[8] = inv_det * (mat[4] * A1323 - mat[5] * A0323 + mat[7] * A0123);
    out_mat[9] = inv_det * -(mat[0] * A1323 - mat[1] * A0323 + mat[3] * A0123);
    out_mat[10] = inv_det * (mat[0] * A1313 - mat[1] * A0313 + mat[3] * A0113);
    out_mat[11] = inv_det * -(mat[0] * A1312 - mat[1] * A0312 + mat[3] * A0112);
    out_mat[12] = inv_det * -(mat[4] * A1223 - mat[5] * A0223 + mat[6] * A0123);
    out_mat[13] = inv_det * (mat[0] * A1223 - mat[1] * A0223 + mat[2] * A0123);
    out_mat[14] = inv_det * -(mat[0] * A1213 - mat[1] * A0213 + mat[2] * A0113);
    out_mat[15] = inv_det * (mat[0] * A1212 - mat[1] * A0212 + mat[2] * A0112);
}
