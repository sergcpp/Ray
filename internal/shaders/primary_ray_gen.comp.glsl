#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#include "primary_ray_gen_interface.h"
#include "common.glsl"

layout(push_constant) uniform UniformParams {
    Params g_params;
};

layout(std430, binding = RANDOM_SEQ_BUF_SLOT) readonly buffer Random {
    uint g_random_seq[];
};

layout(std430, binding = FILTER_TABLE_BUF_SLOT) readonly buffer FilterTable {
    float g_filter_table[];
};

layout(binding = REQUIRED_SAMPLES_IMG_SLOT) uniform utexture2D g_required_samples_img;

layout(std430, binding = INOUT_COUNTERS_BUF_SLOT) buffer InoutCounters {
    uint g_inout_counters[];
};

layout(std430, binding = OUT_RAYS_BUF_SLOT) writeonly buffer OutRays {
    ray_data_t g_out_rays[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

vec3 get_pix_dir(float x, float y, vec3 _origin, float prop) {
    float k = g_params.cam_origin.w * g_params.cam_side.w;
    vec3 p = vec3(2 * k * (x / float(g_params.img_size[0]) + g_params.shift_x / prop) - k,
                  2 * k * (-y / float(g_params.img_size[1]) + g_params.shift_y) + k, g_params.cam_side.w);
    p = g_params.cam_origin.xyz + prop * p.x * g_params.cam_side.xyz + p.y * g_params.cam_up.xyz + p.z * g_params.cam_fwd.xyz;
    return normalize(p - _origin);
}

float lookup_filter_table(float x) {
    x *= (FILTER_TABLE_SIZE - 1);

    const int index = min(int(x), FILTER_TABLE_SIZE - 1);
    const int nindex = min(index + 1, FILTER_TABLE_SIZE - 1);
    const float t = x - float(index);

    const float data0 = g_filter_table[index];
    if (t == 0.0) {
        return data0;
    }

    const float data1 = g_filter_table[nindex];
    return (1.0 - t) * data0 + t * data1;
}

float ngon_rad(const float theta, const float n) {
    return cos(PI / n) / cos(theta - (2.0 * PI / n) * floor((n * theta + PI) / (2.0 * PI)));
}

vec2 get_scrambled_2d_rand(const uint dim, const uint seed, const int _sample) {
    const uint i_seed = hash_combine(seed, dim),
               x_seed = hash_combine(seed, 2 * dim + 0),
               y_seed = hash_combine(seed, 2 * dim + 1);

    const uint shuffled_dim = uint(nested_uniform_scramble_base2(dim, seed) & (RAND_DIMS_COUNT - 1));
    const uint shuffled_i = uint(nested_uniform_scramble_base2(_sample, i_seed) & (RAND_SAMPLES_COUNT - 1));
    return vec2(scramble_unorm(x_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 0]),
                scramble_unorm(y_seed, g_random_seq[shuffled_dim * 2 * RAND_SAMPLES_COUNT + 2 * shuffled_i + 1]));
}

void main() {
    if (gl_GlobalInvocationID.x >= g_params.rect.z || gl_GlobalInvocationID.y >= g_params.rect.w) {
        return;
    }

    int x = int(g_params.rect.x + gl_GlobalInvocationID.x);
    int y = int(g_params.rect.y + gl_GlobalInvocationID.y);

#if ADAPTIVE
    if (texelFetch(g_required_samples_img, ivec2(x, y), 0).r < g_params.iteration) {
        return;
    }
    uint index = atomicAdd(g_inout_counters[0], 1);
#else
    atomicAdd(g_inout_counters[0], 1);
    int index = int(gl_GlobalInvocationID.y * g_params.rect.z + gl_GlobalInvocationID.x);
#endif

    float k = float(g_params.img_size[0]) / float(g_params.img_size[1]);

    const uint px_hash = hash((x << 16) | y);
    const uint rand_hash = hash_combine(px_hash, g_params.rand_seed);

    float fx = float(x);
    float fy = float(y);

    vec2 filter_rand = get_scrambled_2d_rand(RAND_DIM_FILTER, rand_hash, g_params.iteration - 1);

    if ((g_params.cam_filter_and_lens_blades >> 8) != FILTER_BOX) {
        filter_rand.x = lookup_filter_table(filter_rand.x);
        filter_rand.y = lookup_filter_table(filter_rand.y);
    }

    fx += filter_rand.x;
    fy += filter_rand.y;

    vec2 offset = vec2(0.0);

    if (g_params.cam_fstop > 0) {
        const vec2 lens_rand = get_scrambled_2d_rand(RAND_DIM_LENS, rand_hash, g_params.iteration - 1);

        offset = 2.0 * lens_rand - vec2(1.0);
        if (offset.x != 0.0 && offset.y != 0.0) {
            float theta, r;
            if (abs(offset[0]) > abs(offset[1])) {
                r = offset[0];
                theta = 0.25 * PI * (offset[1] / offset[0]);
            } else {
                r = offset[1];
                theta = 0.5 * PI - 0.25 * PI * (offset[0] / offset[1]);
            }

            if ((g_params.cam_filter_and_lens_blades & 0xff) > 0) {
                r *= ngon_rad(theta, float(g_params.cam_filter_and_lens_blades & 0xff));
            }

            theta += g_params.cam_lens_rotation;

            offset.x = 0.5 * r * cos(theta) / g_params.cam_lens_ratio;
            offset.y = 0.5 * r * sin(theta);
        }

        const float coc = 0.5 * (g_params.cam_focal_length / g_params.cam_fstop);
        offset *= coc * g_params.cam_up[3];
    }

    vec3 _origin = g_params.cam_origin.xyz + g_params.cam_side.xyz * offset.x + g_params.cam_up.xyz * offset.y;
    vec3 _d = get_pix_dir(fx, fy, _origin, k);

    _origin += _d * (g_params.cam_fwd.w / dot(_d, g_params.cam_fwd.xyz));

    ray_data_t new_ray;
    new_ray.o[0] = _origin[0];
    new_ray.o[1] = _origin[1];
    new_ray.o[2] = _origin[2];
    new_ray.d[0] = _d[0];
    new_ray.d[1] = _d[1];
    new_ray.d[2] = _d[2];
    new_ray.c[0] = new_ray.c[1] = new_ray.c[2] = 1.0;

    // air ior is implicit
    new_ray.ior[0] = new_ray.ior[1] = new_ray.ior[2] = new_ray.ior[3] = -1.0;

    new_ray.cone_width = 0.0;
    new_ray.cone_spread = g_params.spread_angle;

    new_ray.pdf = 1e6;
    new_ray.xy = uint((x << 16) | y);
    new_ray.depth = 0;

    g_out_rays[index] = new_ray;
}
