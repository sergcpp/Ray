#include "DenoiseRef.h"

#define NS Ref
#include "Convolution.h"
#undef NS

namespace Ray {
namespace Ref {
template <int WINDOW_SIZE, int NEIGHBORHOOD_SIZE, bool FEATURE0, bool FEATURE1>
void JointNLMFilter(const color_rgba_t *restrict input, const rect_t &rect, const int input_stride, const float alpha,
                    const float damping, const color_rgba_t variance[], const color_rgba_t *restrict feature0,
                    const float feature0_weight, const color_rgba_t *restrict feature1, const float feature1_weight,
                    const rect_t &output_rect, const int output_stride, color_rgba_t *restrict output) {
    const int WindowRadius = (WINDOW_SIZE - 1) / 2;
    const float PatchDistanceNormFactor = NEIGHBORHOOD_SIZE * NEIGHBORHOOD_SIZE;
    const int NeighborRadius = (NEIGHBORHOOD_SIZE - 1) / 2;

    assert(rect.w == output_rect.w);
    assert(rect.h == output_rect.h);

    for (int iy = rect.y; iy < rect.y + rect.h; ++iy) {
        for (int ix = rect.x; ix < rect.x + rect.w; ++ix) {
            fvec4 sum_output = {};
            float sum_weight = 0.0f;

            for (int k = -WindowRadius; k <= WindowRadius; ++k) {
                const int jy = iy + k;

                for (int l = -WindowRadius; l <= WindowRadius; ++l) {
                    const int jx = ix + l;

                    fvec4 color_distance = {};

                    for (int q = -NeighborRadius; q <= NeighborRadius; ++q) {
                        for (int p = -NeighborRadius; p <= NeighborRadius; ++p) {
                            const fvec4 ipx = {input[(iy + q) * input_stride + (ix + p)].v, vector_aligned};
                            const fvec4 jpx = {input[(jy + q) * input_stride + (jx + p)].v, vector_aligned};

                            const fvec4 ivar = {variance[(iy + q) * input_stride + (ix + p)].v, vector_aligned};
                            const fvec4 jvar = {variance[(jy + q) * input_stride + (jx + p)].v, vector_aligned};
                            const fvec4 min_var = min(ivar, jvar);

                            color_distance += ((ipx - jpx) * (ipx - jpx) - alpha * (ivar + min_var)) /
                                              (0.0001f + damping * damping * (ivar + jvar));
                        }
                    }

                    const float patch_distance = 0.25f * PatchDistanceNormFactor *
                                                 (color_distance.get<0>() + color_distance.get<1>() +
                                                  color_distance.get<2>() + color_distance.get<3>());
                    float weight = expf(-fmaxf(0.0f, patch_distance));

                    if (FEATURE0 || FEATURE1) {
                        fvec4 feature_distance = {};
                        if (FEATURE0) {
                            const fvec4 ipx = {feature0[iy * input_stride + ix].v, vector_aligned};
                            const fvec4 jpx = {feature0[jy * input_stride + jx].v, vector_aligned};

                            feature_distance = feature0_weight * (ipx - jpx) * (ipx - jpx);
                        }
                        if (FEATURE1) {
                            const fvec4 ipx = {feature1[iy * input_stride + ix].v, vector_aligned};
                            const fvec4 jpx = {feature1[jy * input_stride + jx].v, vector_aligned};

                            feature_distance = max(feature_distance, feature1_weight * (ipx - jpx) * (ipx - jpx));
                        }

                        const float feature_patch_distance =
                            0.25f * (feature_distance.get<0>() + feature_distance.get<1>() + feature_distance.get<2>() +
                                     feature_distance.get<3>());
                        const float feature_weight = expf(-fmaxf(0.0f, fminf(10000.0f, feature_patch_distance)));

                        weight = fminf(weight, feature_weight);
                    }

                    sum_output += fvec4{input[jy * input_stride + jx].v, vector_aligned} * weight;
                    sum_weight += weight;
                }
            }

            if (sum_weight != 0.0f) {
                sum_output /= sum_weight;
            }

            sum_output.store_to(output[(output_rect.y + iy - rect.y) * output_stride + (output_rect.x + ix - rect.x)].v,
                                vector_aligned);
        }
    }
}
} // namespace Ref
} // namespace Ray

template <int WINDOW_SIZE, int NEIGHBORHOOD_SIZE>
void Ray::Ref::JointNLMFilter(const color_rgba_t input[], const rect_t &rect, const int input_stride, const float alpha,
                              const float damping, const color_rgba_t variance[], const color_rgba_t feature1[],
                              const float feature1_weight, const color_rgba_t feature2[], const float feature2_weight,
                              const rect_t &output_rect, const int output_stride, color_rgba_t output[]) {
    if (feature1 && feature2) {
        JointNLMFilter<WINDOW_SIZE, NEIGHBORHOOD_SIZE, true, true>(input, rect, input_stride, alpha, damping, variance,
                                                                   feature1, feature1_weight, feature2, feature2_weight,
                                                                   output_rect, output_stride, output);
    } else if (feature1) {
        JointNLMFilter<WINDOW_SIZE, NEIGHBORHOOD_SIZE, true, false>(input, rect, input_stride, alpha, damping, variance,
                                                                    feature1, feature1_weight, nullptr, 0.0f,
                                                                    output_rect, output_stride, output);
    } else if (feature2) {
        JointNLMFilter<WINDOW_SIZE, NEIGHBORHOOD_SIZE, true, false>(input, rect, input_stride, alpha, damping, variance,
                                                                    feature2, feature2_weight, nullptr, 0.0f,
                                                                    output_rect, output_stride, output);
    } else {
        JointNLMFilter<WINDOW_SIZE, NEIGHBORHOOD_SIZE, false, false>(input, rect, input_stride, alpha, damping,
                                                                     variance, nullptr, 0.0f, nullptr, 0.0f,
                                                                     output_rect, output_stride, output);
    }
}

template void Ray::Ref::JointNLMFilter<21 /* WINDOW_SIZE */, 5 /* NEIGHBORHOOD_SIZE */>(
    const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight, const color_rgba_t feature1[],
    float feature1_weight, const rect_t &output_rect, int output_stride, color_rgba_t output[]);
template void Ray::Ref::JointNLMFilter<21 /* WINDOW_SIZE */, 3 /* NEIGHBORHOOD_SIZE */>(
    const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight, const color_rgba_t feature1[],
    float feature1_weight, const rect_t &output_rect, int output_stride, color_rgba_t output[]);
template void Ray::Ref::JointNLMFilter<7 /* WINDOW_SIZE */, 3 /* NEIGHBORHOOD_SIZE */>(
    const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight, const color_rgba_t feature1[],
    float feature1_weight, const rect_t &output_rect, int output_stride, color_rgba_t output[]);
template void Ray::Ref::JointNLMFilter<3 /* WINDOW_SIZE */, 1 /* NEIGHBORHOOD_SIZE */>(
    const color_rgba_t input[], const rect_t &rect, int input_stride, float alpha, float damping,
    const color_rgba_t variance[], const color_rgba_t feature0[], float feature0_weight, const color_rgba_t feature1[],
    float feature1_weight, const rect_t &output_rect, int output_stride, color_rgba_t output[]);

template <int InChannels, int OutChannels, int OutPxPitch, Ray::ePostOp PostOp, Ray::eActivation Activation>
void Ray::Ref::Convolution3x3(const float data[], const rect_t &rect, const int w, const int h, const int stride,
                              const float weights[], const float biases[], float output[], const int output_stride) {
    Convolution3x3<4, InChannels, OutChannels, OutPxPitch, PostOp, Activation>(data, rect, w, h, stride, weights,
                                                                               biases, output, output_stride);
}

template <int InChannels1, int InChannels2, int InChannels3, int PxPitch, int OutChannels, Ray::ePreOp PreOp1,
          Ray::ePreOp PreOp2, Ray::ePreOp PreOp3, Ray::ePostOp PostOp, Ray::eActivation Activation>
void Ray::Ref::Convolution3x3(const float data1[], const float data2[], const float data3[], const rect_t &rect,
                              const int in_w, const int in_h, const int w, const int h, const int stride,
                              const float weights[], const float biases[], float output[], const int output_stride,
                              aligned_vector<float, 64> &temp_data) {
    Convolution3x3<4, InChannels1, InChannels2, InChannels3, PxPitch, OutChannels, PreOp1, PreOp2, PreOp3, PostOp,
                   Activation>(data1, data2, data3, rect, in_w, in_h, w, h, stride, weights, biases, output,
                               output_stride, temp_data);
}

template void Ray::Ref::Convolution3x3<32, 3, 4, Ray::ePostOp::HDRTransfer, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<32, 32, 32, Ray::ePostOp::Downsample, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<32, 48, 48, Ray::ePostOp::Downsample, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<48, 64, 64, Ray::ePostOp::Downsample, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<64, 32, 32, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<64, 64, 64, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<64, 80, 80, Ray::ePostOp::Downsample, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<80, 96, 96, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<96, 96, 96, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);
template void Ray::Ref::Convolution3x3<112, 112, 112, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data[], const rect_t &rect, int w, int h, int stride, const float weights[], const float biases[],
    float output[], int output_stride);

template void Ray::Ref::Convolution3x3<3, 0, 0, 4, 32, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None, Ray::ePreOp::None,
                                       Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w, int in_h, int w, int h,
    int stride, const float weights[], const float biases[], float output[], int output_stride,
    aligned_vector<float, 64> &temp_data);
template void Ray::Ref::Convolution3x3<3, 3, 0, 4, 32, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None, Ray::ePreOp::None,
                                       Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w, int in_h, int w, int h,
    int stride, const float weights[], const float biases[], float output[], int output_stride,
    aligned_vector<float, 64> &temp_data);
template void Ray::Ref::Convolution3x3<3, 3, 3, 4, 32, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None,
                                       Ray::ePreOp::PositiveNormalize, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const rect_t &rect, int in_w, int in_h, int w, int h,
    int stride, const float weights[], const float biases[], float output[], int output_stride,
    aligned_vector<float, 64> &temp_data);

template <int InChannels1, int InChannels2, int OutChannels, Ray::ePreOp PreOp1, Ray::ePostOp PostOp,
          Ray::eActivation Activation>
void Ray::Ref::ConvolutionConcat3x3(const float data1[], const float data2[], const rect_t &rect, const int w,
                                    const int h, const int stride1, const int stride2, const float weights[],
                                    const float biases[], float output[], const int output_stride) {
    ConvolutionConcat3x3<4, InChannels1, InChannels2, OutChannels, PreOp1, PostOp, Activation>(
        data1, data2, rect, w, h, stride1, stride2, weights, biases, output, output_stride);
}

template <int InChannels1, int InChannels2, int InChannels3, int InChannels4, int PxPitch2, int OutChannels,
          Ray::ePreOp PreOp1, Ray::ePreOp PreOp2, Ray::ePreOp PreOp3, Ray::ePreOp PreOp4, Ray::ePostOp PostOp,
          Ray::eActivation Activation>
void Ray::Ref::ConvolutionConcat3x3(const float data1[], const float data2[], const float data3[], const float data4[],
                                    const rect_t &rect, const int w, const int h, const int w2, const int h2,
                                    const int stride1, const int stride2, const float weights[], const float biases[],
                                    float output[], const int output_stride, aligned_vector<float, 64> &temp_data) {
    ConvolutionConcat3x3<4, InChannels1, InChannels2, InChannels3, InChannels4, PxPitch2, OutChannels, PreOp1, PreOp2,
                         PreOp3, PreOp4, PostOp, Activation>(data1, data2, data3, data4, rect, w, h, w2, h2, stride1,
                                                             stride2, weights, biases, output, output_stride,
                                                             temp_data);
}

template void
Ray::Ref::ConvolutionConcat3x3<96, 64, 112, Ray::ePreOp::Upsample, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const rect_t &rect, int w, int h, int stride1, int stride2,
    const float weights[], const float biases[], float output[], int output_stride);
template void
Ray::Ref::ConvolutionConcat3x3<112, 48, 96, Ray::ePreOp::Upsample, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const rect_t &rect, int w, int h, int stride1, int stride2,
    const float weights[], const float biases[], float output[], int output_stride);
template void
Ray::Ref::ConvolutionConcat3x3<96, 32, 64, Ray::ePreOp::Upsample, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const rect_t &rect, int w, int h, int stride1, int stride2,
    const float weights[], const float biases[], float output[], int output_stride);

template void
Ray::Ref::ConvolutionConcat3x3<64, 3, 0, 0, 4, 64, Ray::ePreOp::Upsample, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None,
                               Ray::ePreOp::None, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const float data4[], const rect_t &rect, int w,
    int h, int w2, int h2, int stride1, int stride2, const float weights[], const float biases[], float output[],
    int output_stride, aligned_vector<float, 64> &temp_data);
template void
Ray::Ref::ConvolutionConcat3x3<64, 3, 3, 0, 4, 64, Ray::ePreOp::Upsample, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None,
                               Ray::ePreOp::None, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const float data4[], const rect_t &rect, int w,
    int h, int w2, int h2, int stride1, int stride2, const float weights[], const float biases[], float output[],
    int output_stride, aligned_vector<float, 64> &temp_data);
template void
Ray::Ref::ConvolutionConcat3x3<64, 3, 3, 3, 4, 64, Ray::ePreOp::Upsample, Ray::ePreOp::HDRTransfer, Ray::ePreOp::None,
                               Ray::ePreOp::PositiveNormalize, Ray::ePostOp::None, Ray::eActivation::ReLU>(
    const float data1[], const float data2[], const float data3[], const float data4[], const rect_t &rect, int w,
    int h, int w2, int h2, int stride1, int stride2, const float weights[], const float biases[], float output[],
    int output_stride, aligned_vector<float, 64> &temp_data);