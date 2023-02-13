#ifndef POSTPROCESS_INTERFACE_H
#define POSTPROCESS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Postprocess)

struct Params {
    UVEC2_TYPE img_size;
    int srgb, _clamp;
    float exposure;
    float _pad0, _pad1, _pad2;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 1;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // POSTPROCESS_INTERFACE_H