#ifndef POSTPROCESS_INTERFACE_H
#define POSTPROCESS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Postprocess)

struct Params {
    UVEC4_TYPE rect;
    int srgb, _clamp;
    float exposure;
    float inv_gamma;
    float img0_weight, img1_weight;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG0_SLOT = 3;
const int IN_IMG1_SLOT = 4;

const int OUT_IMG_SLOT = 0;
const int OUT_RAW_IMG_SLOT = 1;
const int OUT_VARIANCE_IMG_SLOT = 2;

INTERFACE_END

#endif // POSTPROCESS_INTERFACE_H