#ifndef NLM_FILTER_INTERFACE_H
#define NLM_FILTER_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(NLMFilter)

struct Params {
    UVEC2_TYPE img_size;
    float alpha;
    float damping;
    int srgb, _clamp;
    float exposure;
    float inv_gamma;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 2;
const int VARIANCE_IMG_SLOT = 3;

const int OUT_IMG_SLOT = 0;
const int OUT_RAW_IMG_SLOT = 1;

INTERFACE_END

#endif // NLM_FILTER_INTERFACE_H