#ifndef MIX_INCREMENTAL_INTERFACE_H
#define MIX_INCREMENTAL_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(MixIncremental)

struct Params {
    uvec4 rect;
    float mix_factor;
    float half_mix_factor;
    int iteration;
    float accumulate_half_img;
    float exposure;
    float _pad0;
    float _pad1;
    float _pad2;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_TEMP_IMG_SLOT = 4;
const int IN_TEMP_BASE_COLOR_SLOT = 5;
const int IN_TEMP_DEPTH_NORMALS_SLOT = 6;
const int IN_REQ_SAMPLES_SLOT = 7;

const int OUT_FULL_IMG_SLOT = 0;
const int OUT_HALF_IMG_SLOT = 1;
const int OUT_BASE_COLOR_IMG_SLOT = 2;
const int OUT_DEPTH_NORMALS_IMG_SLOT = 3;

INTERFACE_END

#endif // MIX_INCREMENTAL_INTERFACE_H