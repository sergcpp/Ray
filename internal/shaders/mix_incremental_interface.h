#ifndef MIX_INCREMENTAL_INTERFACE_H
#define MIX_INCREMENTAL_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(MixIncremental)

struct Params {
    UVEC2_TYPE img_size;
    float main_mix_factor;
    float aux_mix_factor;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG1_SLOT = 3;
const int IN_IMG2_SLOT = 4;
const int IN_TEMP_BASE_COLOR_SLOT = 5;
const int IN_TEMP_DEPTH_NORMALS_SLOT = 6;

const int OUT_IMG_SLOT = 0;
const int OUT_BASE_COLOR_IMG_SLOT = 1;
const int OUT_DEPTH_NORMALS_IMG_SLOT = 2;

INTERFACE_END

#endif // MIX_INCREMENTAL_INTERFACE_H