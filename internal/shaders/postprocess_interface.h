#ifndef POSTPROCESS_INTERFACE_H
#define POSTPROCESS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Postprocess)

struct Params {
    uvec4 rect;
    float inv_gamma;
    int tonemap_mode;
    float variance_threshold;
    int iteration;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_FULL_IMG_SLOT = 4;
const int IN_HALF_IMG_SLOT = 5;
const int TONEMAP_LUT_SLOT = 6;

const int OUT_IMG_SLOT = 0;
const int OUT_VARIANCE_IMG_SLOT = 1;
const int OUT_REQ_SAMPLES_IMG_SLOT = 2;

INTERFACE_END

#endif // POSTPROCESS_INTERFACE_H