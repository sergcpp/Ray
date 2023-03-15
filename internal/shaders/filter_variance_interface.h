#ifndef FILTER_VARIANCE_INTERFACE_H
#define FILTER_VARIANCE_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(FilterVariance)

struct Params {
    UVEC2_TYPE img_size;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 1;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // FILTER_VARIANCE_INTERFACE_H