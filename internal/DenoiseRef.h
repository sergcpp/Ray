#pragma once

#include "CoreRef.h"

namespace Ray {
namespace Ref {
template <int WINDOW_SIZE = 7, int NEIGHBORHOOD_SIZE = 3>
void JointNLMFilter(const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
                    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight,
                    const color_rgba_t feature1[], float feature1_weight, const rect_t &output_rect, int output_stride,
                    color_rgba_t output[]);

template <int InChannels, int OutChannels, int OutPxPitch, ePostOp PostOp = ePostOp::None,
          eActivation Activation = eActivation::ReLU>
void Convolution3x3(const float data[], const rect_t &rect, int w, int h, int stride, const float weights[],
                    const float biases[], float output[], int output_stride);
template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None, ePostOp PostOp = ePostOp::None,
          eActivation Activation = eActivation::ReLU>
void Convolution3x3(const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w,
                    int in_h, int w, int h, int stride, const float weights[], const float biases[], float output[],
                    int output_stride, aligned_vector<float, 64> &temp_data);

template <int InChannels1, int InChannels2, int OutChannels, ePreOp PreOp1 = ePreOp::None,
          ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3(const float data1[], const float data2[], const rect_t &rect, int w, int h, int stride1,
                          int stride2, const float weights[], const float biases[], float output[], int output_stride);
template <int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
          ePreOp PreOp1 = ePreOp::None, ePreOp PreOp2 = ePreOp::None, ePreOp PreOp3 = ePreOp::None,
          ePreOp PreOp4 = ePreOp::None, ePostOp PostOp = ePostOp::None, eActivation Activation = eActivation::ReLU>
void ConvolutionConcat3x3(const float data1[], const float data2[], const float data3[], const float data4[],
                          const rect_t &rect, int w, int h, int w2, int h2, int stride1, int stride2,
                          const float weights[], const float biases[], float output[], int output_stride,
                          aligned_vector<float, 64> &temp_data);
void ClearBorders(const rect_t &rect, int w, int h, bool downscaled, int out_channels, float output[]);
} // namespace Ref
} // namespace Ray