#ifndef MIX_INCREMENTAL_INTERFACE_H
#define MIX_INCREMENTAL_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(MixIncremental)

struct Params {
    UVEC2_TYPE img_size;
    float k;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG1_SLOT = 1;
const int IN_IMG2_SLOT = 2;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // MIX_INCREMENTAL_INTERFACE_H