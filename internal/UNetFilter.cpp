#include "UNetFilter.h"

#include <cstring>

#include <algorithm>

namespace Ray {
float f16_to_f32(uint16_t h);
int round_up(int v, int align);
namespace unet_weights_hdr {
#include "precomputed/__oidn_weights_hdr.inl"
}
namespace unet_weights_hdr_alb {
#include "precomputed/__oidn_weights_hdr_alb.inl"
}
namespace unet_weights_hdr_alb_nrm {
#include "precomputed/__oidn_weights_hdr_alb_nrm.inl"
}

template <typename T> T convert_weight(const uint16_t val);

template <> uint16_t convert_weight<uint16_t>(const uint16_t val) { return val; }
template <> float convert_weight<float>(const uint16_t val) { return f16_to_f32(val); }

// Reorder weights for direct 3x3 convolution
template <typename T>
void ReorderWeights_Conv3x3_Direct(const T in_weights[], const int in_channels, const int out_channels,
                                   const int alignment, T out_weights[]) {
    const int rounded_triple = round_up(3 * in_channels, alignment);

    int out_index = 0;
    for (int j = 0; j < out_channels; ++j) {
        assert(out_index == j * 3 * rounded_triple);
        // line 0
        for (int i = 0; i < 3 * in_channels; ++i) {
            out_weights[out_index++] = in_weights[j * in_channels * 9 + (i % in_channels) * 9 + (i / in_channels)];
        }
        for (int i = 3 * in_channels; i < rounded_triple; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 1
        for (int i = 3 * in_channels; i < 6 * in_channels; ++i) {
            out_weights[out_index++] = in_weights[j * in_channels * 9 + (i % in_channels) * 9 + (i / in_channels)];
        }
        for (int i = rounded_triple + 3 * in_channels; i < 2 * rounded_triple; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 2
        for (int i = 6 * in_channels; i < 9 * in_channels; ++i) {
            out_weights[out_index++] = in_weights[j * in_channels * 9 + (i % in_channels) * 9 + (i / in_channels)];
        }
        for (int i = 2 * rounded_triple + 3 * in_channels; i < 3 * rounded_triple; ++i) {
            out_weights[out_index++] = T(0);
        }
    }
}

template <typename T>
void ReorderWeights_Conv3x3_Direct(const T in_weights[], const int in_channels1, const int in_channels2,
                                   const int out_channels, const int alignment, T out_weights[]) {
    const int rounded_triple1 = round_up(3 * in_channels1, alignment),
              rounded_triple2 = round_up(3 * in_channels2, alignment);

    int out_index = 0;
    for (int j = 0; j < out_channels; ++j) {
        assert(out_index == j * 3 * (rounded_triple1 + rounded_triple2));
        // line 0
        for (int i = 0; i < 3 * in_channels1; ++i) {
            out_weights[out_index++] =
                in_weights[j * (in_channels1 + in_channels2) * 9 + (i % in_channels1) * 9 + (i / in_channels1)];
        }
        for (int i = 3 * in_channels1; i < rounded_triple1; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 1
        for (int i = 3 * in_channels1; i < 6 * in_channels1; ++i) {
            out_weights[out_index++] =
                in_weights[j * (in_channels1 + in_channels2) * 9 + (i % in_channels1) * 9 + (i / in_channels1)];
        }
        for (int i = rounded_triple1 + 3 * in_channels1; i < 2 * rounded_triple1; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 2
        for (int i = 6 * in_channels1; i < 9 * in_channels1; ++i) {
            out_weights[out_index++] =
                in_weights[j * (in_channels1 + in_channels2) * 9 + (i % in_channels1) * 9 + (i / in_channels1)];
        }
        for (int i = 2 * rounded_triple1 + 3 * in_channels1; i < 3 * rounded_triple1; ++i) {
            out_weights[out_index++] = T(0);
        }
        assert(out_index == j * 3 * (rounded_triple1 + rounded_triple2) + 3 * rounded_triple1);
        // line 0
        for (int i = 0; i < 3 * in_channels2; ++i) {
            out_weights[out_index++] = in_weights[j * (in_channels1 + in_channels2) * 9 +
                                                  (in_channels1 + (i % in_channels2)) * 9 + (i / in_channels2)];
        }
        for (int i = 3 * in_channels2; i < rounded_triple2; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 1
        for (int i = 3 * in_channels2; i < 6 * in_channels2; ++i) {
            out_weights[out_index++] = in_weights[j * (in_channels1 + in_channels2) * 9 +
                                                  (in_channels1 + (i % in_channels2)) * 9 + (i / in_channels2)];
        }
        for (int i = rounded_triple2 + 3 * in_channels2; i < 2 * rounded_triple2; ++i) {
            out_weights[out_index++] = T(0);
        }
        // line 2
        for (int i = 6 * in_channels2; i < 9 * in_channels2; ++i) {
            out_weights[out_index++] = in_weights[j * (in_channels1 + in_channels2) * 9 +
                                                  (in_channels1 + (i % in_channels2)) * 9 + (i / in_channels2)];
        }
        for (int i = 2 * rounded_triple2 + 3 * in_channels2; i < 3 * rounded_triple2; ++i) {
            out_weights[out_index++] = T(0);
        }
    }
}

template <typename T>
void ReorderWeights_Conv3x3_1Direct_2GEMM(const T in_weights[], const int in_channels1, const int in_channels2,
                                          const int out_channels, T out_weights[]) {
    for (int j = 0; j < out_channels; ++j) {
        for (int c = 0; c < 9; ++c) {
            for (int i = 0; i < in_channels1; ++i) {
                out_weights[j * (in_channels1 + in_channels2) * 9 + c * in_channels1 + i] =
                    in_weights[j * (in_channels1 + in_channels2) * 9 + i * 9 + c];
            }
        }
        for (int c = 0; c < 9; ++c) {
            for (int i = 0; i < in_channels2; ++i) {
                out_weights[j * (in_channels1 + in_channels2) * 9 + 9 * in_channels1 + i * 9 + c] =
                    in_weights[j * (in_channels1 + in_channels2) * 9 + (in_channels1 + i) * 9 + c];
            }
        }
    }
}
} // namespace Ray

int Ray::SetupUNetFilter(int w, int h, bool alias_memory, bool round_w, unet_filter_tensors_t &out_tensors,
                         SmallVector<int, 2> alias_dependencies[]) {
    struct resource_t {
        const char *name;
        int offset;
        int resolution_div;
        int depth;
        int size;
        int lifetime[2] = {99999, -1};
    };

    resource_t resources[] = {
        {"encConv0", 0, 1, 32},   //
        {"pool1", 0, 2, 32},      //
        {"pool2", 0, 4, 48},      //
        {"pool3", 0, 8, 64},      //
        {"pool4", 0, 16, 80},     //
        {"encConv5a", 0, 16, 96}, //
        {"upsample4", 0, 16, 96}, //
        {"decConv4a", 0, 8, 112}, //
        {"upsample3", 0, 8, 112}, //
        {"decConv3a", 0, 4, 96},  //
        {"upsample2", 0, 4, 96},  //
        {"decConv2a", 0, 2, 64},  //
        {"upsample1", 0, 2, 64},  //
        {"decConv1a", 0, 1, 64},  //
        {"decConv1b", 0, 1, 32}   //
    };
    const int resource_count = sizeof(resources) / sizeof(resource_t);

    const int w_rounded = 16 * ((w + 15) / 16);
    const int h_rounded = 16 * ((h + 15) / 16);

    for (resource_t &r : resources) {
        assert((w_rounded % r.resolution_div) == 0);
        r.size = (w_rounded / r.resolution_div) + 1;
        if (round_w) {
            r.size = round_up(r.size, 16);
        }
        r.size += 1;
        assert((h_rounded % r.resolution_div) == 0);
        r.size *= (h_rounded / r.resolution_div) + 2;
        r.size *= r.depth;
        r.size = round_up(r.size, 128);
    }

    struct pass_t {
        const char *used_resources[3];
        int used_resource_indices[3] = {-1, -1, -1};
    };
    pass_t passes[UNetFilterPasses] = {
        {{"encConv0"}},                        // enc_conv0
        {{"encConv0", "pool1"}},               // enc_conv1
        {{"pool1", "pool2"}},                  // enc_conv2
        {{"pool2", "pool3"}},                  // enc_conv3
        {{"pool3", "pool4"}},                  // enc_conv4
        {{"pool4", "encConv5a"}},              // enc_conv5a
        {{"encConv5a", "upsample4"}},          // enc_conv5b
        {{"upsample4", "pool3", "decConv4a"}}, // dec_conv4a
        {{"decConv4a", "upsample3"}},          // dec_conv4b
        {{"upsample3", "pool2", "decConv3a"}}, // dec_conv3a
        {{"decConv3a", "upsample2"}},          // dec_conv3b
        {{"upsample2", "pool1", "decConv2a"}}, // dec_conv2a
        {{"decConv2a", "upsample1"}},          // dec_conv2b
        {{"upsample1", "decConv1a"}},          // dec_conv1a
        {{"decConv1a", "decConv1b"}},          // dec_conv1b
        {{"decConv1b"}},                       // dec_conv0
    };

    for (int i = 0; i < UNetFilterPasses; ++i) {
        pass_t &pass = passes[i];
        for (int r = 0; r < 3; ++r) {
            const char *resource_name = pass.used_resources[r];
            if (!resource_name) {
                continue;
            }

            for (int j = 0; j < resource_count; ++j) {
                if (strcmp(resources[j].name, resource_name) == 0) {
                    pass.used_resource_indices[r] = j;
                    resources[j].lifetime[0] = std::min(resources[j].lifetime[0], i);
                    resources[j].lifetime[1] = std::max(resources[j].lifetime[1], i);
                    break;
                }
            }
        }
    }

    int required_memory = 0;
    if (alias_memory) {
        std::vector<int> placement_order(resource_count);
        for (int i = 0; i < resource_count; ++i) {
            placement_order[i] = i;
        }

        std::sort(std::begin(placement_order), std::end(placement_order), [&](const int lhs, const int rhs) {
            return resources[lhs].size * (resources[lhs].lifetime[1] - resources[lhs].lifetime[0]) >
                   resources[rhs].size * (resources[rhs].lifetime[1] - resources[rhs].lifetime[0]);
        });

        std::vector<int> heap_tops(UNetFilterPasses, 0);

        for (int i = 0; i < resource_count; ++i) {
            resource_t &r = resources[placement_order[i]];

            int heap_top = 0;
            for (int j = r.lifetime[0]; j <= r.lifetime[1]; ++j) {
                heap_top = std::max(heap_top, heap_tops[j]);
            }

            r.offset = heap_top;
            heap_top += r.size;

            for (int j = r.lifetime[0]; j <= r.lifetime[1]; ++j) {
                heap_tops[j] = heap_top;
            }
        }

        for (int i = 0; i < UNetFilterPasses; ++i) {
            required_memory = std::max(required_memory, heap_tops[i]);
        }

        for (int i = UNetFilterPasses - 1; i >= 0; --i) {
            const pass_t &pass = passes[i];
            for (const int res_index : pass.used_resource_indices) {
                if (res_index == -1) {
                    continue;
                }

                bool exit = false;
                for (int j = i - 1; j >= 0 && !exit; --j) {
                    const pass_t &pass2 = passes[j];
                    for (const int res2_index : pass2.used_resource_indices) {
                        if (res2_index == -1 || res2_index == res_index) {
                            continue;
                        }

                        if (resources[res_index].offset == resources[res2_index].offset &&
                            resources[res_index].size == resources[res2_index].size) {
                            // Skip exact alias
                            continue;
                        }

                        if (resources[res_index].offset < resources[res2_index].offset + resources[res2_index].size &&
                            resources[res_index].offset + resources[res_index].size > resources[res2_index].offset) {
                            auto it = std::find(std::begin(alias_dependencies[i]), std::end(alias_dependencies[i]), j);
                            if (it == std::end(alias_dependencies[i])) {
                                alias_dependencies[i].push_back(j);
                            }
                            exit = true;
                        }
                    }
                }
            }
        }
    } else {
        for (resource_t &res : resources) {
            res.offset = required_memory;
            required_memory += res.size;
        }
    }

    out_tensors.enc_conv0_offset = resources[0].offset;
    out_tensors.enc_conv0_size = resources[0].size;
    out_tensors.pool1_offset = resources[1].offset;
    out_tensors.pool1_size = resources[1].size;
    out_tensors.pool2_offset = resources[2].offset;
    out_tensors.pool2_size = resources[2].size;
    out_tensors.pool3_offset = resources[3].offset;
    out_tensors.pool3_size = resources[3].size;
    out_tensors.pool4_offset = resources[4].offset;
    out_tensors.pool4_size = resources[4].size;
    out_tensors.enc_conv5a_offset = resources[5].offset;
    out_tensors.enc_conv5a_size = resources[5].size;
    out_tensors.upsample4_offset = resources[6].offset;
    out_tensors.upsample4_size = resources[6].size;
    out_tensors.dec_conv4a_offset = resources[7].offset;
    out_tensors.dec_conv4a_size = resources[7].size;
    out_tensors.upsample3_offset = resources[8].offset;
    out_tensors.upsample3_size = resources[8].size;
    out_tensors.dec_conv3a_offset = resources[9].offset;
    out_tensors.dec_conv3a_size = resources[9].size;
    out_tensors.upsample2_offset = resources[10].offset;
    out_tensors.upsample2_size = resources[10].size;
    out_tensors.dec_conv2a_offset = resources[11].offset;
    out_tensors.dec_conv2a_size = resources[11].size;
    out_tensors.upsample1_offset = resources[12].offset;
    out_tensors.upsample1_size = resources[12].size;
    out_tensors.dec_conv1a_offset = resources[13].offset;
    out_tensors.dec_conv1a_size = resources[13].size;
    out_tensors.dec_conv1b_offset = resources[14].offset;
    out_tensors.dec_conv1b_size = resources[14].size;

    return required_memory;
}

template <typename T>
int Ray::SetupUNetWeights(const bool albedo, const bool normals, const bool gemm, const int alignment,
                          unet_weight_offsets_t *out_offsets, T out_weights[]) {
    int enc_conv0_weight_count, enc_conv0_bias_count, enc_conv1_weight_count, enc_conv1_bias_count,
        enc_conv2_weight_count, enc_conv2_bias_count, enc_conv3_weight_count, enc_conv3_bias_count,
        enc_conv4_weight_count, enc_conv4_bias_count, enc_conv5a_weight_count, enc_conv5a_bias_count,
        enc_conv5b_weight_count, enc_conv5b_bias_count, dec_conv4a_weight_count, dec_conv4a_bias_count,
        dec_conv4b_weight_count, dec_conv4b_bias_count, dec_conv3a_weight_count, dec_conv3a_bias_count,
        dec_conv3b_weight_count, dec_conv3b_bias_count, dec_conv2a_weight_count, dec_conv2a_bias_count,
        dec_conv2b_weight_count, dec_conv2b_bias_count, dec_conv1a_weight_count, dec_conv1a_bias_count,
        dec_conv1b_weight_count, dec_conv1b_bias_count, dec_conv0_weight_count, dec_conv0_bias_count;

    const uint16_t *enc_conv0_weight, *enc_conv0_bias, *enc_conv1_weight, *enc_conv1_bias, *enc_conv2_weight,
        *enc_conv2_bias, *enc_conv3_weight, *enc_conv3_bias, *enc_conv4_weight, *enc_conv4_bias, *enc_conv5a_weight,
        *enc_conv5a_bias, *enc_conv5b_weight, *enc_conv5b_bias, *dec_conv4a_weight, *dec_conv4a_bias,
        *dec_conv4b_weight, *dec_conv4b_bias, *dec_conv3a_weight, *dec_conv3a_bias, *dec_conv3b_weight,
        *dec_conv3b_bias, *dec_conv2a_weight, *dec_conv2a_bias, *dec_conv2b_weight, *dec_conv2b_bias,
        *dec_conv1a_weight, *dec_conv1a_bias, *dec_conv1b_weight, *dec_conv1b_bias, *dec_conv0_weight, *dec_conv0_bias;

    if (albedo && normals) {
        enc_conv0_weight = unet_weights_hdr_alb_nrm::enc_conv0_weight;
        enc_conv0_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv0_weight) / sizeof(uint16_t);
        enc_conv0_bias = unet_weights_hdr_alb_nrm::enc_conv0_bias;
        enc_conv0_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv0_bias) / sizeof(uint16_t);
        enc_conv1_weight = unet_weights_hdr_alb_nrm::enc_conv1_weight;
        enc_conv1_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv1_weight) / sizeof(uint16_t);
        enc_conv1_bias = unet_weights_hdr_alb_nrm::enc_conv1_bias;
        enc_conv1_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv1_bias) / sizeof(uint16_t);
        enc_conv2_weight = unet_weights_hdr_alb_nrm::enc_conv2_weight;
        enc_conv2_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv2_weight) / sizeof(uint16_t);
        enc_conv2_bias = unet_weights_hdr_alb_nrm::enc_conv2_bias;
        enc_conv2_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv2_bias) / sizeof(uint16_t);
        enc_conv3_weight = unet_weights_hdr_alb_nrm::enc_conv3_weight;
        enc_conv3_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv3_weight) / sizeof(uint16_t);
        enc_conv3_bias = unet_weights_hdr_alb_nrm::enc_conv3_bias;
        enc_conv3_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv3_bias) / sizeof(uint16_t);
        enc_conv4_weight = unet_weights_hdr_alb_nrm::enc_conv4_weight;
        enc_conv4_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv4_weight) / sizeof(uint16_t);
        enc_conv4_bias = unet_weights_hdr_alb_nrm::enc_conv4_bias;
        enc_conv4_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv4_bias) / sizeof(uint16_t);
        enc_conv5a_weight = unet_weights_hdr_alb_nrm::enc_conv5a_weight;
        enc_conv5a_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv5a_weight) / sizeof(uint16_t);
        enc_conv5a_bias = unet_weights_hdr_alb_nrm::enc_conv5a_bias;
        enc_conv5a_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv5a_bias) / sizeof(uint16_t);
        enc_conv5b_weight = unet_weights_hdr_alb_nrm::enc_conv5b_weight;
        enc_conv5b_weight_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv5b_weight) / sizeof(uint16_t);
        enc_conv5b_bias = unet_weights_hdr_alb_nrm::enc_conv5b_bias;
        enc_conv5b_bias_count = sizeof(unet_weights_hdr_alb_nrm::enc_conv5b_bias) / sizeof(uint16_t);
        dec_conv4a_weight = unet_weights_hdr_alb_nrm::dec_conv4a_weight;
        dec_conv4a_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv4a_weight) / sizeof(uint16_t);
        dec_conv4a_bias = unet_weights_hdr_alb_nrm::dec_conv4a_bias;
        dec_conv4a_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv4a_bias) / sizeof(uint16_t);
        dec_conv4b_weight = unet_weights_hdr_alb_nrm::dec_conv4b_weight;
        dec_conv4b_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv4b_weight) / sizeof(uint16_t);
        dec_conv4b_bias = unet_weights_hdr_alb_nrm::dec_conv4b_bias;
        dec_conv4b_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv4b_bias) / sizeof(uint16_t);
        dec_conv3a_weight = unet_weights_hdr_alb_nrm::dec_conv3a_weight;
        dec_conv3a_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv3a_weight) / sizeof(uint16_t);
        dec_conv3a_bias = unet_weights_hdr_alb_nrm::dec_conv3a_bias;
        dec_conv3a_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv3a_bias) / sizeof(uint16_t);
        dec_conv3b_weight = unet_weights_hdr_alb_nrm::dec_conv3b_weight;
        dec_conv3b_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv3b_weight) / sizeof(uint16_t);
        dec_conv3b_bias = unet_weights_hdr_alb_nrm::dec_conv3b_bias;
        dec_conv3b_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv3b_bias) / sizeof(uint16_t);
        dec_conv2a_weight = unet_weights_hdr_alb_nrm::dec_conv2a_weight;
        dec_conv2a_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv2a_weight) / sizeof(uint16_t);
        dec_conv2a_bias = unet_weights_hdr_alb_nrm::dec_conv2a_bias;
        dec_conv2a_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv2a_bias) / sizeof(uint16_t);
        dec_conv2b_weight = unet_weights_hdr_alb_nrm::dec_conv2b_weight;
        dec_conv2b_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv2b_weight) / sizeof(uint16_t);
        dec_conv2b_bias = unet_weights_hdr_alb_nrm::dec_conv2b_bias;
        dec_conv2b_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv2b_bias) / sizeof(uint16_t);
        dec_conv1a_weight = unet_weights_hdr_alb_nrm::dec_conv1a_weight;
        dec_conv1a_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv1a_weight) / sizeof(uint16_t);
        dec_conv1a_bias = unet_weights_hdr_alb_nrm::dec_conv1a_bias;
        dec_conv1a_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv1a_bias) / sizeof(uint16_t);
        dec_conv1b_weight = unet_weights_hdr_alb_nrm::dec_conv1b_weight;
        dec_conv1b_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv1b_weight) / sizeof(uint16_t);
        dec_conv1b_bias = unet_weights_hdr_alb_nrm::dec_conv1b_bias;
        dec_conv1b_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv1b_bias) / sizeof(uint16_t);
        dec_conv0_weight = unet_weights_hdr_alb_nrm::dec_conv0_weight;
        dec_conv0_weight_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv0_weight) / sizeof(uint16_t);
        dec_conv0_bias = unet_weights_hdr_alb_nrm::dec_conv0_bias;
        dec_conv0_bias_count = sizeof(unet_weights_hdr_alb_nrm::dec_conv0_bias) / sizeof(uint16_t);
    } else if (albedo) {
        enc_conv0_weight = unet_weights_hdr_alb::enc_conv0_weight;
        enc_conv0_weight_count = sizeof(unet_weights_hdr_alb::enc_conv0_weight) / sizeof(uint16_t);
        enc_conv0_bias = unet_weights_hdr_alb::enc_conv0_bias;
        enc_conv0_bias_count = sizeof(unet_weights_hdr_alb::enc_conv0_bias) / sizeof(uint16_t);
        enc_conv1_weight = unet_weights_hdr_alb::enc_conv1_weight;
        enc_conv1_weight_count = sizeof(unet_weights_hdr_alb::enc_conv1_weight) / sizeof(uint16_t);
        enc_conv1_bias = unet_weights_hdr_alb::enc_conv1_bias;
        enc_conv1_bias_count = sizeof(unet_weights_hdr_alb::enc_conv1_bias) / sizeof(uint16_t);
        enc_conv2_weight = unet_weights_hdr_alb::enc_conv2_weight;
        enc_conv2_weight_count = sizeof(unet_weights_hdr_alb::enc_conv2_weight) / sizeof(uint16_t);
        enc_conv2_bias = unet_weights_hdr_alb::enc_conv2_bias;
        enc_conv2_bias_count = sizeof(unet_weights_hdr_alb::enc_conv2_bias) / sizeof(uint16_t);
        enc_conv3_weight = unet_weights_hdr_alb::enc_conv3_weight;
        enc_conv3_weight_count = sizeof(unet_weights_hdr_alb::enc_conv3_weight) / sizeof(uint16_t);
        enc_conv3_bias = unet_weights_hdr_alb::enc_conv3_bias;
        enc_conv3_bias_count = sizeof(unet_weights_hdr_alb::enc_conv3_bias) / sizeof(uint16_t);
        enc_conv4_weight = unet_weights_hdr_alb::enc_conv4_weight;
        enc_conv4_weight_count = sizeof(unet_weights_hdr_alb::enc_conv4_weight) / sizeof(uint16_t);
        enc_conv4_bias = unet_weights_hdr_alb::enc_conv4_bias;
        enc_conv4_bias_count = sizeof(unet_weights_hdr_alb::enc_conv4_bias) / sizeof(uint16_t);
        enc_conv5a_weight = unet_weights_hdr_alb::enc_conv5a_weight;
        enc_conv5a_weight_count = sizeof(unet_weights_hdr_alb::enc_conv5a_weight) / sizeof(uint16_t);
        enc_conv5a_bias = unet_weights_hdr_alb::enc_conv5a_bias;
        enc_conv5a_bias_count = sizeof(unet_weights_hdr_alb::enc_conv5a_bias) / sizeof(uint16_t);
        enc_conv5b_weight = unet_weights_hdr_alb::enc_conv5b_weight;
        enc_conv5b_weight_count = sizeof(unet_weights_hdr_alb::enc_conv5b_weight) / sizeof(uint16_t);
        enc_conv5b_bias = unet_weights_hdr_alb::enc_conv5b_bias;
        enc_conv5b_bias_count = sizeof(unet_weights_hdr_alb::enc_conv5b_bias) / sizeof(uint16_t);
        dec_conv4a_weight = unet_weights_hdr_alb::dec_conv4a_weight;
        dec_conv4a_weight_count = sizeof(unet_weights_hdr_alb::dec_conv4a_weight) / sizeof(uint16_t);
        dec_conv4a_bias = unet_weights_hdr_alb::dec_conv4a_bias;
        dec_conv4a_bias_count = sizeof(unet_weights_hdr_alb::dec_conv4a_bias) / sizeof(uint16_t);
        dec_conv4b_weight = unet_weights_hdr_alb::dec_conv4b_weight;
        dec_conv4b_weight_count = sizeof(unet_weights_hdr_alb::dec_conv4b_weight) / sizeof(uint16_t);
        dec_conv4b_bias = unet_weights_hdr_alb::dec_conv4b_bias;
        dec_conv4b_bias_count = sizeof(unet_weights_hdr_alb::dec_conv4b_bias) / sizeof(uint16_t);
        dec_conv3a_weight = unet_weights_hdr_alb::dec_conv3a_weight;
        dec_conv3a_weight_count = sizeof(unet_weights_hdr_alb::dec_conv3a_weight) / sizeof(uint16_t);
        dec_conv3a_bias = unet_weights_hdr_alb::dec_conv3a_bias;
        dec_conv3a_bias_count = sizeof(unet_weights_hdr_alb::dec_conv3a_bias) / sizeof(uint16_t);
        dec_conv3b_weight = unet_weights_hdr_alb::dec_conv3b_weight;
        dec_conv3b_weight_count = sizeof(unet_weights_hdr_alb::dec_conv3b_weight) / sizeof(uint16_t);
        dec_conv3b_bias = unet_weights_hdr_alb::dec_conv3b_bias;
        dec_conv3b_bias_count = sizeof(unet_weights_hdr_alb::dec_conv3b_bias) / sizeof(uint16_t);
        dec_conv2a_weight = unet_weights_hdr_alb::dec_conv2a_weight;
        dec_conv2a_weight_count = sizeof(unet_weights_hdr_alb::dec_conv2a_weight) / sizeof(uint16_t);
        dec_conv2a_bias = unet_weights_hdr_alb::dec_conv2a_bias;
        dec_conv2a_bias_count = sizeof(unet_weights_hdr_alb::dec_conv2a_bias) / sizeof(uint16_t);
        dec_conv2b_weight = unet_weights_hdr_alb::dec_conv2b_weight;
        dec_conv2b_weight_count = sizeof(unet_weights_hdr_alb::dec_conv2b_weight) / sizeof(uint16_t);
        dec_conv2b_bias = unet_weights_hdr_alb::dec_conv2b_bias;
        dec_conv2b_bias_count = sizeof(unet_weights_hdr_alb::dec_conv2b_bias) / sizeof(uint16_t);
        dec_conv1a_weight = unet_weights_hdr_alb::dec_conv1a_weight;
        dec_conv1a_weight_count = sizeof(unet_weights_hdr_alb::dec_conv1a_weight) / sizeof(uint16_t);
        dec_conv1a_bias = unet_weights_hdr_alb::dec_conv1a_bias;
        dec_conv1a_bias_count = sizeof(unet_weights_hdr_alb::dec_conv1a_bias) / sizeof(uint16_t);
        dec_conv1b_weight = unet_weights_hdr_alb::dec_conv1b_weight;
        dec_conv1b_weight_count = sizeof(unet_weights_hdr_alb::dec_conv1b_weight) / sizeof(uint16_t);
        dec_conv1b_bias = unet_weights_hdr_alb::dec_conv1b_bias;
        dec_conv1b_bias_count = sizeof(unet_weights_hdr_alb::dec_conv1b_bias) / sizeof(uint16_t);
        dec_conv0_weight = unet_weights_hdr_alb::dec_conv0_weight;
        dec_conv0_weight_count = sizeof(unet_weights_hdr_alb::dec_conv0_weight) / sizeof(uint16_t);
        dec_conv0_bias = unet_weights_hdr_alb::dec_conv0_bias;
        dec_conv0_bias_count = sizeof(unet_weights_hdr_alb::dec_conv0_bias) / sizeof(uint16_t);
    } else {
        enc_conv0_weight = unet_weights_hdr::enc_conv0_weight;
        enc_conv0_weight_count = sizeof(unet_weights_hdr::enc_conv0_weight) / sizeof(uint16_t);
        enc_conv0_bias = unet_weights_hdr::enc_conv0_bias;
        enc_conv0_bias_count = sizeof(unet_weights_hdr::enc_conv0_bias) / sizeof(uint16_t);
        enc_conv1_weight = unet_weights_hdr::enc_conv1_weight;
        enc_conv1_weight_count = sizeof(unet_weights_hdr::enc_conv1_weight) / sizeof(uint16_t);
        enc_conv1_bias = unet_weights_hdr::enc_conv1_bias;
        enc_conv1_bias_count = sizeof(unet_weights_hdr::enc_conv1_bias) / sizeof(uint16_t);
        enc_conv2_weight = unet_weights_hdr::enc_conv2_weight;
        enc_conv2_weight_count = sizeof(unet_weights_hdr::enc_conv2_weight) / sizeof(uint16_t);
        enc_conv2_bias = unet_weights_hdr::enc_conv2_bias;
        enc_conv2_bias_count = sizeof(unet_weights_hdr::enc_conv2_bias) / sizeof(uint16_t);
        enc_conv3_weight = unet_weights_hdr::enc_conv3_weight;
        enc_conv3_weight_count = sizeof(unet_weights_hdr::enc_conv3_weight) / sizeof(uint16_t);
        enc_conv3_bias = unet_weights_hdr::enc_conv3_bias;
        enc_conv3_bias_count = sizeof(unet_weights_hdr::enc_conv3_bias) / sizeof(uint16_t);
        enc_conv4_weight = unet_weights_hdr::enc_conv4_weight;
        enc_conv4_weight_count = sizeof(unet_weights_hdr::enc_conv4_weight) / sizeof(uint16_t);
        enc_conv4_bias = unet_weights_hdr::enc_conv4_bias;
        enc_conv4_bias_count = sizeof(unet_weights_hdr::enc_conv4_bias) / sizeof(uint16_t);
        enc_conv5a_weight = unet_weights_hdr::enc_conv5a_weight;
        enc_conv5a_weight_count = sizeof(unet_weights_hdr::enc_conv5a_weight) / sizeof(uint16_t);
        enc_conv5a_bias = unet_weights_hdr::enc_conv5a_bias;
        enc_conv5a_bias_count = sizeof(unet_weights_hdr::enc_conv5a_bias) / sizeof(uint16_t);
        enc_conv5b_weight = unet_weights_hdr::enc_conv5b_weight;
        enc_conv5b_weight_count = sizeof(unet_weights_hdr::enc_conv5b_weight) / sizeof(uint16_t);
        enc_conv5b_bias = unet_weights_hdr::enc_conv5b_bias;
        enc_conv5b_bias_count = sizeof(unet_weights_hdr::enc_conv5b_bias) / sizeof(uint16_t);
        dec_conv4a_weight = unet_weights_hdr::dec_conv4a_weight;
        dec_conv4a_weight_count = sizeof(unet_weights_hdr::dec_conv4a_weight) / sizeof(uint16_t);
        dec_conv4a_bias = unet_weights_hdr::dec_conv4a_bias;
        dec_conv4a_bias_count = sizeof(unet_weights_hdr::dec_conv4a_bias) / sizeof(uint16_t);
        dec_conv4b_weight = unet_weights_hdr::dec_conv4b_weight;
        dec_conv4b_weight_count = sizeof(unet_weights_hdr::dec_conv4b_weight) / sizeof(uint16_t);
        dec_conv4b_bias = unet_weights_hdr::dec_conv4b_bias;
        dec_conv4b_bias_count = sizeof(unet_weights_hdr::dec_conv4b_bias) / sizeof(uint16_t);
        dec_conv3a_weight = unet_weights_hdr::dec_conv3a_weight;
        dec_conv3a_weight_count = sizeof(unet_weights_hdr::dec_conv3a_weight) / sizeof(uint16_t);
        dec_conv3a_bias = unet_weights_hdr::dec_conv3a_bias;
        dec_conv3a_bias_count = sizeof(unet_weights_hdr::dec_conv3a_bias) / sizeof(uint16_t);
        dec_conv3b_weight = unet_weights_hdr::dec_conv3b_weight;
        dec_conv3b_weight_count = sizeof(unet_weights_hdr::dec_conv3b_weight) / sizeof(uint16_t);
        dec_conv3b_bias = unet_weights_hdr::dec_conv3b_bias;
        dec_conv3b_bias_count = sizeof(unet_weights_hdr::dec_conv3b_bias) / sizeof(uint16_t);
        dec_conv2a_weight = unet_weights_hdr::dec_conv2a_weight;
        dec_conv2a_weight_count = sizeof(unet_weights_hdr::dec_conv2a_weight) / sizeof(uint16_t);
        dec_conv2a_bias = unet_weights_hdr::dec_conv2a_bias;
        dec_conv2a_bias_count = sizeof(unet_weights_hdr::dec_conv2a_bias) / sizeof(uint16_t);
        dec_conv2b_weight = unet_weights_hdr::dec_conv2b_weight;
        dec_conv2b_weight_count = sizeof(unet_weights_hdr::dec_conv2b_weight) / sizeof(uint16_t);
        dec_conv2b_bias = unet_weights_hdr::dec_conv2b_bias;
        dec_conv2b_bias_count = sizeof(unet_weights_hdr::dec_conv2b_bias) / sizeof(uint16_t);
        dec_conv1a_weight = unet_weights_hdr::dec_conv1a_weight;
        dec_conv1a_weight_count = sizeof(unet_weights_hdr::dec_conv1a_weight) / sizeof(uint16_t);
        dec_conv1a_bias = unet_weights_hdr::dec_conv1a_bias;
        dec_conv1a_bias_count = sizeof(unet_weights_hdr::dec_conv1a_bias) / sizeof(uint16_t);
        dec_conv1b_weight = unet_weights_hdr::dec_conv1b_weight;
        dec_conv1b_weight_count = sizeof(unet_weights_hdr::dec_conv1b_weight) / sizeof(uint16_t);
        dec_conv1b_bias = unet_weights_hdr::dec_conv1b_bias;
        dec_conv1b_bias_count = sizeof(unet_weights_hdr::dec_conv1b_bias) / sizeof(uint16_t);
        dec_conv0_weight = unet_weights_hdr::dec_conv0_weight;
        dec_conv0_weight_count = sizeof(unet_weights_hdr::dec_conv0_weight) / sizeof(uint16_t);
        dec_conv0_bias = unet_weights_hdr::dec_conv0_bias;
        dec_conv0_bias_count = sizeof(unet_weights_hdr::dec_conv0_bias) / sizeof(uint16_t);
    }

    auto extend = [&](int val, const int in_channels) {
        val /= 3 * in_channels;
        val *= alignment * ((3 * in_channels + alignment - 1) / alignment);
        return val;
    };

    auto count2 = [&](const int in_channels1, const int in_channels2, const int out_channels) {
        const int per_output = 3 * round_up(3 * in_channels1, alignment) * 3 * round_up(3 * in_channels2, alignment);
        return per_output * out_channels;
    };

    int input_channels = 3;
    if (albedo && normals) {
        input_channels = 9;
    } else if (albedo) {
        input_channels = 6;
    }

    const int el_align = (256 / sizeof(T));

    const int total_count =
        round_up(extend(enc_conv0_weight_count, input_channels), el_align) + round_up(enc_conv0_bias_count, el_align) +
        round_up(enc_conv1_weight_count, el_align) + round_up(enc_conv1_bias_count, el_align) +
        round_up(enc_conv2_weight_count, el_align) + round_up(enc_conv2_bias_count, el_align) +
        round_up(enc_conv3_weight_count, el_align) + round_up(enc_conv3_bias_count, el_align) +
        round_up(enc_conv4_weight_count, el_align) + round_up(enc_conv4_bias_count, el_align) +
        round_up(enc_conv5a_weight_count, el_align) + round_up(enc_conv5a_bias_count, el_align) +
        round_up(enc_conv5b_weight_count, el_align) + round_up(enc_conv5b_bias_count, el_align) +
        round_up(dec_conv4a_weight_count, el_align) + round_up(dec_conv4a_bias_count, el_align) +
        round_up(dec_conv4b_weight_count, el_align) + round_up(dec_conv4b_bias_count, el_align) +
        round_up(dec_conv3a_weight_count, el_align) + round_up(dec_conv3a_bias_count, el_align) +
        round_up(dec_conv3b_weight_count, el_align) + round_up(dec_conv3b_bias_count, el_align) +
        round_up(dec_conv2a_weight_count, el_align) + round_up(dec_conv2a_bias_count, el_align) +
        round_up(dec_conv2b_weight_count, el_align) + round_up(dec_conv2b_bias_count, el_align) +
        round_up(count2(64, input_channels, 64), el_align) + round_up(dec_conv1a_bias_count, el_align) +
        round_up(dec_conv1b_weight_count, el_align) + round_up(dec_conv1b_bias_count, el_align) +
        round_up(dec_conv0_weight_count, el_align) + dec_conv0_bias_count;

    if (out_offsets) {
        out_offsets->enc_conv0_weight = 0;
        out_offsets->enc_conv0_bias =
            out_offsets->enc_conv0_weight + round_up(extend(enc_conv0_weight_count, input_channels), el_align);
        out_offsets->enc_conv1_weight = out_offsets->enc_conv0_bias + round_up(enc_conv0_bias_count, el_align);
        out_offsets->enc_conv1_bias = out_offsets->enc_conv1_weight + round_up(enc_conv1_weight_count, el_align);
        out_offsets->enc_conv2_weight = out_offsets->enc_conv1_bias + round_up(enc_conv1_bias_count, el_align);
        out_offsets->enc_conv2_bias = out_offsets->enc_conv2_weight + round_up(enc_conv2_weight_count, el_align);
        out_offsets->enc_conv3_weight = out_offsets->enc_conv2_bias + round_up(enc_conv2_bias_count, el_align);
        out_offsets->enc_conv3_bias = out_offsets->enc_conv3_weight + round_up(enc_conv3_weight_count, el_align);
        out_offsets->enc_conv4_weight = out_offsets->enc_conv3_bias + round_up(enc_conv3_bias_count, el_align);
        out_offsets->enc_conv4_bias = out_offsets->enc_conv4_weight + round_up(enc_conv4_weight_count, el_align);
        out_offsets->enc_conv5a_weight = out_offsets->enc_conv4_bias + round_up(enc_conv4_bias_count, el_align);
        out_offsets->enc_conv5a_bias = out_offsets->enc_conv5a_weight + round_up(enc_conv5a_weight_count, el_align);
        out_offsets->enc_conv5b_weight = out_offsets->enc_conv5a_bias + round_up(enc_conv5a_bias_count, el_align);
        out_offsets->enc_conv5b_bias = out_offsets->enc_conv5b_weight + round_up(enc_conv5b_weight_count, el_align);
        out_offsets->dec_conv4a_weight = out_offsets->enc_conv5b_bias + round_up(enc_conv5b_bias_count, el_align);
        out_offsets->dec_conv4a_bias = out_offsets->dec_conv4a_weight + round_up(dec_conv4a_weight_count, el_align);
        out_offsets->dec_conv4b_weight = out_offsets->dec_conv4a_bias + round_up(dec_conv4a_bias_count, el_align);
        out_offsets->dec_conv4b_bias = out_offsets->dec_conv4b_weight + round_up(dec_conv4b_weight_count, el_align);
        out_offsets->dec_conv3a_weight = out_offsets->dec_conv4b_bias + round_up(dec_conv4b_bias_count, el_align);
        out_offsets->dec_conv3a_bias = out_offsets->dec_conv3a_weight + round_up(dec_conv3a_weight_count, el_align);
        out_offsets->dec_conv3b_weight = out_offsets->dec_conv3a_bias + round_up(dec_conv3a_bias_count, el_align);
        out_offsets->dec_conv3b_bias = out_offsets->dec_conv3b_weight + round_up(dec_conv3b_weight_count, el_align);
        out_offsets->dec_conv2a_weight = out_offsets->dec_conv3b_bias + round_up(dec_conv3b_bias_count, el_align);
        out_offsets->dec_conv2a_bias = out_offsets->dec_conv2a_weight + round_up(dec_conv2a_weight_count, el_align);
        out_offsets->dec_conv2b_weight = out_offsets->dec_conv2a_bias + round_up(dec_conv2a_bias_count, el_align);
        out_offsets->dec_conv2b_bias = out_offsets->dec_conv2b_weight + round_up(dec_conv2b_weight_count, el_align);
        out_offsets->dec_conv1a_weight = out_offsets->dec_conv2b_bias + round_up(dec_conv2b_bias_count, el_align);
        out_offsets->dec_conv1a_bias =
            out_offsets->dec_conv1a_weight + round_up(count2(64, input_channels, 64), el_align);
        out_offsets->dec_conv1b_weight = out_offsets->dec_conv1a_bias + round_up(dec_conv1a_bias_count, el_align);
        out_offsets->dec_conv1b_bias = out_offsets->dec_conv1b_weight + round_up(dec_conv1b_weight_count, el_align);
        out_offsets->dec_conv0_weight = out_offsets->dec_conv1b_bias + round_up(dec_conv1b_bias_count, el_align);
        out_offsets->dec_conv0_bias = out_offsets->dec_conv0_weight + round_up(dec_conv0_weight_count, el_align);

        assert(out_offsets->dec_conv0_bias + dec_conv0_bias_count == total_count);
        assert((out_offsets->enc_conv0_weight % el_align) == 0 && (out_offsets->enc_conv0_bias % el_align) == 0);
        assert((out_offsets->enc_conv1_weight % el_align) == 0 && (out_offsets->enc_conv1_bias % el_align) == 0);
        assert((out_offsets->enc_conv2_weight % el_align) == 0 && (out_offsets->enc_conv2_bias % el_align) == 0);
        assert((out_offsets->enc_conv3_weight % el_align) == 0 && (out_offsets->enc_conv3_bias % el_align) == 0);
        assert((out_offsets->enc_conv4_weight % el_align) == 0 && (out_offsets->enc_conv4_bias % el_align) == 0);
        assert((out_offsets->enc_conv5a_weight % el_align) == 0 && (out_offsets->enc_conv5a_bias % el_align) == 0);
        assert((out_offsets->enc_conv5b_weight % el_align) == 0 && (out_offsets->enc_conv5b_bias % el_align) == 0);
        assert((out_offsets->dec_conv4a_weight % el_align) == 0 && (out_offsets->dec_conv4a_bias % el_align) == 0);
        assert((out_offsets->dec_conv4b_weight % el_align) == 0 && (out_offsets->dec_conv4b_bias % el_align) == 0);
        assert((out_offsets->dec_conv3a_weight % el_align) == 0 && (out_offsets->dec_conv3a_bias % el_align) == 0);
        assert((out_offsets->dec_conv3b_weight % el_align) == 0 && (out_offsets->dec_conv3b_bias % el_align) == 0);
        assert((out_offsets->dec_conv2a_weight % el_align) == 0 && (out_offsets->dec_conv2a_bias % el_align) == 0);
        assert((out_offsets->dec_conv2b_weight % el_align) == 0 && (out_offsets->dec_conv2b_bias % el_align) == 0);
        assert((out_offsets->dec_conv1a_weight % el_align) == 0 && (out_offsets->dec_conv1a_bias % el_align) == 0);
        assert((out_offsets->dec_conv1b_weight % el_align) == 0 && (out_offsets->dec_conv1b_bias % el_align) == 0);
        assert((out_offsets->dec_conv0_weight % el_align) == 0 && (out_offsets->dec_conv0_bias % el_align) == 0);
    }

    if (out_weights) {
        std::vector<T> temp;

        if (gemm) {
            for (int i = 0; i < enc_conv0_weight_count; ++i) {
                out_weights[out_offsets->enc_conv0_weight + i] = convert_weight<T>(enc_conv0_weight[i]);
            }
        } else {
            temp.resize(enc_conv0_weight_count);
            for (int i = 0; i < enc_conv0_weight_count; ++i) {
                temp[i] = convert_weight<T>(enc_conv0_weight[i]);
            }
            ReorderWeights_Conv3x3_Direct(temp.data(), input_channels, 32, alignment,
                                          &out_weights[out_offsets->enc_conv0_weight]);
        }
        for (int i = 0; i < enc_conv0_bias_count; ++i) {
            out_weights[out_offsets->enc_conv0_bias + i] = convert_weight<T>(enc_conv0_bias[i]);
        }

        temp.resize(enc_conv1_weight_count);
        for (int i = 0; i < enc_conv1_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv1_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 32, 32, alignment, &out_weights[out_offsets->enc_conv1_weight]);
        for (int i = 0; i < enc_conv1_bias_count; ++i) {
            out_weights[out_offsets->enc_conv1_bias + i] = convert_weight<T>(enc_conv1_bias[i]);
        }

        temp.resize(enc_conv2_weight_count);
        for (int i = 0; i < enc_conv2_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv2_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 32, 48, alignment, &out_weights[out_offsets->enc_conv2_weight]);
        for (int i = 0; i < enc_conv2_bias_count; ++i) {
            out_weights[out_offsets->enc_conv2_bias + i] = convert_weight<T>(enc_conv2_bias[i]);
        }

        temp.resize(enc_conv3_weight_count);
        for (int i = 0; i < enc_conv3_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv3_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 48, 64, alignment, &out_weights[out_offsets->enc_conv3_weight]);
        for (int i = 0; i < enc_conv3_bias_count; ++i) {
            out_weights[out_offsets->enc_conv3_bias + i] = convert_weight<T>(enc_conv3_bias[i]);
        }

        temp.resize(enc_conv4_weight_count);
        for (int i = 0; i < enc_conv4_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv4_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 64, 80, alignment, &out_weights[out_offsets->enc_conv4_weight]);
        for (int i = 0; i < enc_conv4_bias_count; ++i) {
            out_weights[out_offsets->enc_conv4_bias + i] = convert_weight<T>(enc_conv4_bias[i]);
        }

        temp.resize(enc_conv5a_weight_count);
        for (int i = 0; i < enc_conv5a_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv5a_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 80, 96, alignment, &out_weights[out_offsets->enc_conv5a_weight]);
        for (int i = 0; i < enc_conv5a_bias_count; ++i) {
            out_weights[out_offsets->enc_conv5a_bias + i] = convert_weight<T>(enc_conv5a_bias[i]);
        }

        temp.resize(enc_conv5b_weight_count);
        for (int i = 0; i < enc_conv5b_weight_count; ++i) {
            temp[i] = convert_weight<T>(enc_conv5b_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 96, 96, alignment, &out_weights[out_offsets->enc_conv5b_weight]);
        for (int i = 0; i < enc_conv5b_bias_count; ++i) {
            out_weights[out_offsets->enc_conv5b_bias + i] = convert_weight<T>(enc_conv5b_bias[i]);
        }

        temp.resize(dec_conv4a_weight_count);
        for (int i = 0; i < dec_conv4a_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv4a_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 96, 64, 112, alignment,
                                      &out_weights[out_offsets->dec_conv4a_weight]);
        for (int i = 0; i < dec_conv4a_bias_count; ++i) {
            out_weights[out_offsets->dec_conv4a_bias + i] = convert_weight<T>(dec_conv4a_bias[i]);
        }

        temp.resize(dec_conv4b_weight_count);
        for (int i = 0; i < dec_conv4b_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv4b_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 112, 112, alignment, &out_weights[out_offsets->dec_conv4b_weight]);
        for (int i = 0; i < dec_conv4b_bias_count; ++i) {
            out_weights[out_offsets->dec_conv4b_bias + i] = convert_weight<T>(dec_conv4b_bias[i]);
        }

        temp.resize(dec_conv3a_weight_count);
        for (int i = 0; i < dec_conv3a_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv3a_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 112, 48, 96, alignment,
                                      &out_weights[out_offsets->dec_conv3a_weight]);
        for (int i = 0; i < dec_conv3a_bias_count; ++i) {
            out_weights[out_offsets->dec_conv3a_bias + i] = convert_weight<T>(dec_conv3a_bias[i]);
        }

        temp.resize(dec_conv3b_weight_count);
        for (int i = 0; i < dec_conv3b_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv3b_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 96, 96, alignment, &out_weights[out_offsets->dec_conv3b_weight]);
        for (int i = 0; i < dec_conv3b_bias_count; ++i) {
            out_weights[out_offsets->dec_conv3b_bias + i] = convert_weight<T>(dec_conv3b_bias[i]);
        }

        temp.resize(dec_conv2a_weight_count);
        for (int i = 0; i < dec_conv2a_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv2a_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 96, 32, 64, alignment, &out_weights[out_offsets->dec_conv2a_weight]);
        for (int i = 0; i < dec_conv2a_bias_count; ++i) {
            out_weights[out_offsets->dec_conv2a_bias + i] = convert_weight<T>(dec_conv2a_bias[i]);
        }

        temp.resize(dec_conv2b_weight_count);
        for (int i = 0; i < dec_conv2b_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv2b_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 64, 64, alignment, &out_weights[out_offsets->dec_conv2b_weight]);
        for (int i = 0; i < dec_conv2b_bias_count; ++i) {
            out_weights[out_offsets->dec_conv2b_bias + i] = convert_weight<T>(dec_conv2b_bias[i]);
        }

        temp.resize(dec_conv1a_weight_count);
        for (int i = 0; i < dec_conv1a_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv1a_weight[i]);
        }
        if (gemm) {
            ReorderWeights_Conv3x3_1Direct_2GEMM(temp.data(), 64, input_channels, 64,
                                                 &out_weights[out_offsets->dec_conv1a_weight]);
        } else {
            // ReorderWeights_Conv3x3_Direct(temp.data(), 64 + input_channels, 64, 1,
            //                               &out_weights[out_offsets->dec_conv1a_weight]);

            ReorderWeights_Conv3x3_Direct(temp.data(), 64, input_channels, 64, alignment,
                                          &out_weights[out_offsets->dec_conv1a_weight]);
        }
        for (int i = 0; i < dec_conv1a_bias_count; ++i) {
            out_weights[out_offsets->dec_conv1a_bias + i] = convert_weight<T>(dec_conv1a_bias[i]);
        }

        temp.resize(dec_conv1b_weight_count);
        for (int i = 0; i < dec_conv1b_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv1b_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 64, 32, alignment, &out_weights[out_offsets->dec_conv1b_weight]);
        for (int i = 0; i < dec_conv1b_bias_count; ++i) {
            out_weights[out_offsets->dec_conv1b_bias + i] = convert_weight<T>(dec_conv1b_bias[i]);
        }

        temp.resize(dec_conv0_weight_count);
        for (int i = 0; i < dec_conv0_weight_count; ++i) {
            temp[i] = convert_weight<T>(dec_conv0_weight[i]);
        }
        ReorderWeights_Conv3x3_Direct(temp.data(), 32, 3, alignment, &out_weights[out_offsets->dec_conv0_weight]);
        for (int i = 0; i < dec_conv0_bias_count; ++i) {
            out_weights[out_offsets->dec_conv0_bias + i] = convert_weight<T>(dec_conv0_bias[i]);
        }
    }

    return total_count;
}

template int Ray::SetupUNetWeights<float>(bool albedo, bool normals, bool gemm, int alignment,
                                          unet_weight_offsets_t *out_offsets, float out_weights[]);
template int Ray::SetupUNetWeights<uint16_t>(bool albedo, bool normals, bool gemm, int alignment,
                                             unet_weight_offsets_t *out_offsets, uint16_t out_weights[]);
