#ifndef FILTER_VARIANCE_INTERFACE_H
#define FILTER_VARIANCE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(FilterVariance)

struct Params {
    uvec4 rect;
    vec2 inv_img_size;
    float variance_threshold;
    int iteration;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 2;

const int OUT_IMG_SLOT = 0;
const int OUT_REQ_SAMPLES_IMG_SLOT = 1;

INTERFACE_END

#endif // FILTER_VARIANCE_INTERFACE_H