#ifndef SHADE_SKY_INTERFACE_H
#define SHADE_SKY_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(ShadeSky)

struct Params {
    vec4 light_dir;
    vec4 light_col;
    vec4 light_col_point;
    vec4 env_col;
    vec4 back_col;
    //
    uint rand_seed;
    int env_qtree_levels;
    float env_rotation;
    float back_rotation;
    //
    int env_light_index;
    float limit;
    int max_total_depth;
    int li_count;
    //
    ivec2 res;
    ivec2 _pad;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int ATMOSPHERE_PARAMS_BUF_SLOT = 3;
const int RAY_INDICES_BUF_SLOT = 4;
const int COUNTERS_BUF_SLOT = 5;
const int HITS_BUF_SLOT = 6;
const int RAYS_BUF_SLOT = 7;
const int ENV_QTREE_TEX_SLOT = 8;
const int TRANSMITTANCE_LUT_SLOT = 9;
const int MULTISCATTER_LUT_SLOT = 10;
const int MOON_TEX_SLOT = 11;
const int WEATHER_TEX_SLOT = 12;
const int CIRRUS_TEX_SLOT = 13;
const int NOISE3D_TEX_SLOT = 14;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // SHADE_SKY_INTERFACE_H
