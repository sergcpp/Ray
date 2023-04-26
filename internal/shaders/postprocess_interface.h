#ifndef POSTPROCESS_INTERFACE_H
#define POSTPROCESS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Postprocess)

struct Params {
    UVEC4_TYPE rect;
    float exposure;
    float inv_gamma;
    float img0_weight;
    float img1_weight;
    int tonemap_mode;
    float variance_threshold;
    int iteration;
    float _pad2;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG0_SLOT = 4;
const int IN_IMG1_SLOT = 5;
const int TONEMAP_LUT_SLOT = 6;

const int OUT_IMG_SLOT = 0;
const int OUT_RAW_IMG_SLOT = 1;
const int OUT_VARIANCE_IMG_SLOT = 2;
const int OUT_REQ_SAMPLES_IMG_SLOT = 3;

INTERFACE_END

#endif // POSTPROCESS_INTERFACE_H