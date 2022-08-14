#version 450
#extension GL_GOOGLE_include_directive : require

#include "primary_ray_gen_interface.glsl"
#include "types.glsl"
#include "common.glsl"

LAYOUT_PARAMS uniform UniformParams {
    Params g_params;
};

layout(std430, binding = HALTON_SEQ_BUF_SLOT) readonly buffer Halton {
    float g_halton[];
};

layout(std430, binding = OUT_RAYS_BUF_SLOT) writeonly buffer OutRays {
    ray_data_t g_out_rays[];
};

layout (local_size_x = LOCAL_GROUP_SIZE_X, local_size_y = LOCAL_GROUP_SIZE_Y, local_size_z = 1) in;

vec3 get_pix_dir(float x, float y, vec3 _origin, float prop) {
    float k = tan(radians(0.5 * g_params.cam_origin.w)) * g_params.cam_side.w;
    vec3 p = vec3(2 * k * x / float(g_params.img_size.x) - k, 2 * k * -y / float(g_params.img_size.y) + k, g_params.cam_side.w);
    p = g_params.cam_origin.xyz + prop * p.x * g_params.cam_side.xyz + p.y * g_params.cam_up.xyz + p.z * g_params.cam_fwd.xyz;
    return normalize(p - _origin);
}

void main() {
    if (gl_GlobalInvocationID.x >= g_params.img_size.x || gl_GlobalInvocationID.y >= g_params.img_size.y) {
        return;
    }

    float k = float(g_params.img_size.x) / float(g_params.img_size.y);

    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);

    int index = y * int(g_params.img_size.x) + x;
    int hash_val = hash(index);

    float _x = float(x);
    float _y = float(y);

    vec2 sample_off = vec2(construct_float(hash_val), construct_float(hash(hash_val)));

    if (false) {

    } else {
        _x += fract(g_halton[g_params.hi + RAND_DIM_FILTER_U] + sample_off.x);
        _y += fract(g_halton[g_params.hi + RAND_DIM_FILTER_V] + sample_off.y);
    }

    float ff1 = g_params.cam_up.w * (-0.5 + fract(g_halton[g_params.hi + RAND_DIM_LENS_U] + sample_off.x));
    float ff2 = g_params.cam_up.w * (-0.5 + fract(g_halton[g_params.hi + RAND_DIM_LENS_V] + sample_off.y));

    vec3 _origin = g_params.cam_origin.xyz + g_params.cam_side.xyz * ff1 + g_params.cam_up.xyz * ff2;
    vec3 _d = get_pix_dir(_x, _y, _origin, k);

    ray_data_t new_ray;
    new_ray.o[0] = _origin[0];
    new_ray.o[1] = _origin[1];
    new_ray.o[2] = _origin[2];
    new_ray.d[0] = _d[0];
    new_ray.d[1] = _d[1];
    new_ray.d[2] = _d[2];
    new_ray.c[0] = new_ray.c[1] = new_ray.c[2] = 1.0;

#ifdef USE_RAY_DIFFERENTIALS
#else
    new_ray.cone_width = 0.0;
    new_ray.cone_spread = g_params.spread_angle;
#endif

    new_ray.pdf = 1e6;
    new_ray.xy = (x << 16) | y;
    new_ray.ray_depth = 0;

    g_out_rays[index] = new_ray;
}
