#ifndef NLM_FILTER_INTERFACE_H
#define NLM_FILTER_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(NLMFilter)

struct Params {
    uvec4 rect;
    vec2 inv_img_size;
    float alpha;
    float damping;
    float inv_gamma;
    int tonemap_mode;
    float base_color_weight;
    float depth_normal_weight;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 2;
const int VARIANCE_IMG_SLOT = 3;
const int TONEMAP_LUT_SLOT = 4;
const int BASE_COLOR_IMG_SLOT = 5;
const int DEPTH_NORMAL_IMG_SLOT = 6;

const int OUT_IMG_SLOT = 0;
const int OUT_RAW_IMG_SLOT = 1;

INTERFACE_END

#endif // NLM_FILTER_INTERFACE_H