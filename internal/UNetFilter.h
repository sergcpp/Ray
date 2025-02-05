#pragma once

#include "SmallVector.h"

namespace Ray {
const int UNetFilterPasses = 16;

struct unet_filter_tensors_t {
    int enc_conv0_offset, enc_conv0_size;
    int pool1_offset, pool1_size;
    int pool2_offset, pool2_size;
    int pool3_offset, pool3_size;
    int pool4_offset, pool4_size;
    int enc_conv5a_offset, enc_conv5a_size;
    int upsample4_offset, upsample4_size;
    int dec_conv4a_offset, dec_conv4a_size;
    int upsample3_offset, upsample3_size;
    int dec_conv3a_offset, dec_conv3a_size;
    int upsample2_offset, upsample2_size;
    int dec_conv2a_offset, dec_conv2a_size;
    int upsample1_offset, upsample1_size;
    int dec_conv1a_offset, dec_conv1a_size;
    int dec_conv1b_offset, dec_conv1b_size;
};

struct unet_weight_offsets_t {
    int enc_conv0_weight, enc_conv0_bias;
    int enc_conv1_weight, enc_conv1_bias;
    int enc_conv2_weight, enc_conv2_bias;
    int enc_conv3_weight, enc_conv3_bias;
    int enc_conv4_weight, enc_conv4_bias;
    int enc_conv5a_weight, enc_conv5a_bias;
    int enc_conv5b_weight, enc_conv5b_bias;
    int dec_conv4a_weight, dec_conv4a_bias;
    int dec_conv4b_weight, dec_conv4b_bias;
    int dec_conv3a_weight, dec_conv3a_bias;
    int dec_conv3b_weight, dec_conv3b_bias;
    int dec_conv2a_weight, dec_conv2a_bias;
    int dec_conv2b_weight, dec_conv2b_bias;
    int dec_conv1a_weight, dec_conv1a_bias;
    int dec_conv1b_weight, dec_conv1b_bias;
    int dec_conv0_weight, dec_conv0_bias;
};

int SetupUNetFilter(int w, int h, bool alias_memory, bool round_w, unet_filter_tensors_t &out_tensors,
                    SmallVector<int, 2> alias_dependencies[]);

template <typename T> int SetupUNetWeights(int alignment, unet_weight_offsets_t *out_offsets, T out_weights[]);
} // namespace Ray