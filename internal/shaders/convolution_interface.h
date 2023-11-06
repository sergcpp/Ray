#ifndef CONVOLUTION_INTERFACE_H
#define CONVOLUTION_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(Convolution)

struct Params {
    uvec4 rect;
    vec2 inv_img_size;
    int input_stride1;
    int input_stride2;
    int output_stride;
    int tonemap_mode;
    float inv_gamma;
    int _pad2;
    uvec2 out_dims;
    uvec2 in_dims;
};

const int LOCAL_GROUP_SIZE_X = 2;
const int LOCAL_GROUP_SIZE_Y = 2;
const int LOCAL_GROUP_SIZE_Z = 16;

const int TILE_M = 32;
const int TILE_N = 32;

const int IN_IMG1_SLOT = 2;
const int IN_SAMPLER_SLOT = 3;

const int IN_BUF1_SLOT = 2;
const int IN_BUF2_SLOT = 4;

const int IN_IMG2_SLOT = 4;
const int IN_IMG3_SLOT = 5;
const int IN_IMG4_SLOT = 6;

const int WEIGHTS_BUF_SLOT = 7;
const int BIASES_BUF_SLOT = 8;
const int TONEMAP_LUT_SLOT = 9;

const int OUT_BUF_SLOT = 0;
const int OUT_IMG_SLOT = 0;
const int OUT_TONEMAPPED_IMG_SLOT = 1;
const int OUT_DEBUG_IMG_SLOT = 1;

INTERFACE_END

#endif // CONVOLUTION_INTERFACE_H