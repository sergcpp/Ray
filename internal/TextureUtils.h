#pragma once

#include <cstdint>

#include <memory>

#include "Core.h"
#include "TextureParams.h"

namespace Ray {
std::unique_ptr<float[]> ConvertRGBE_to_RGB32F(const uint8_t image_data[], int w, int h);
std::unique_ptr<uint16_t[]> ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h);
void ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h, uint16_t *out_data);

std::unique_ptr<uint8_t[]> ConvertRGB32F_to_RGBE(const float image_data[], int w, int h, int channels);

// Perfectly reversible conversion between RGB and YCoCg (breaks bilinear filtering)
void ConvertRGB_to_YCoCg_rev(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]);
void ConvertYCoCg_to_RGB_rev(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]);

std::unique_ptr<uint8_t[]> ConvertRGB_to_CoCgxY_rev(const uint8_t image_data[], int w, int h);
std::unique_ptr<uint8_t[]> ConvertCoCgxY_to_RGB_rev(const uint8_t image_data[], int w, int h);

// Not-so-perfectly reversible conversion between RGB and YCoCg
void ConvertRGB_to_YCoCg(const uint8_t in_RGB[3], uint8_t out_YCoCg[3]);
void ConvertYCoCg_to_RGB(const uint8_t in_YCoCg[3], uint8_t out_RGB[3]);

std::unique_ptr<uint8_t[]> ConvertRGB_to_CoCgxY(const uint8_t image_data[], int w, int h);
std::unique_ptr<uint8_t[]> ConvertCoCgxY_to_RGB(const uint8_t image_data[], int w, int h);

uint16_t f32_to_f16(float value);
float f16_to_f32(const uint16_t h);

extern const uint8_t _blank_BC3_block_4x4[];
extern const int _blank_BC3_block_4x4_len;

extern const uint8_t _blank_ASTC_block_4x4[];
extern const int _blank_ASTC_block_4x4_len;

//
// BCn compression
//

// clang-format off

const int BlockSize_BC1 = 2 * sizeof(uint16_t) + sizeof(uint32_t);
//                        \_ low/high colors_/   \_ 16 x 2-bit _/
const int BlockSize_BC4 = 2 * sizeof(uint8_t) + 6 * sizeof(uint8_t);
//                        \_ low/high alpha_/     \_ 16 x 3-bit _/
const int BlockSize_BC3 = BlockSize_BC1 + BlockSize_BC4;
const int BlockSize_BC5 = BlockSize_BC4 + BlockSize_BC4;

// clang-format on

int GetRequiredMemory_BC1(int w, int h, int pitch_align);
int GetRequiredMemory_BC3(int w, int h, int pitch_align);
int GetRequiredMemory_BC4(int w, int h, int pitch_align);
int GetRequiredMemory_BC5(int w, int h, int pitch_align);

template <int N> int GetBlockSize_BCn() {
    if (N == 4) {
        return BlockSize_BC3;
    } else if (N == 3) {
        return BlockSize_BC1;
    } else if (N == 2) {
        return BlockSize_BC5;
    } else /*if (N == 1)*/ {
        return BlockSize_BC4;
    }
}

template <int N> int GetRequiredMemory_BCn(const int w, const int h, const int pitch_align) {
    if (N == 4) {
        return GetRequiredMemory_BC3(w, h, pitch_align);
    } else if (N == 3) {
        return GetRequiredMemory_BC1(w, h, pitch_align);
    } else if (N == 2) {
        return GetRequiredMemory_BC5(w, h, pitch_align);
    } else /*if (N == 1)*/ {
        return GetRequiredMemory_BC4(w, h, pitch_align);
    }
}

// NOTE: intended for realtime compression, quality may be not the best
template <int SrcChannels>
void CompressImage_BC1(const uint8_t img_src[], int w, int h, uint8_t img_dst[], int dst_pitch = 0);
template <bool Is_YCoCg = false>
void CompressImage_BC3(const uint8_t img_src[], int w, int h, uint8_t img_dst[], int dst_pitch = 0);
template <int SrcChannels = 1>
void CompressImage_BC4(const uint8_t img_src[], int w, int h, uint8_t img_dst[], int dst_pitch = 0);
template <int SrcChannels = 2>
void CompressImage_BC5(const uint8_t img_src[], int w, int h, uint8_t img_dst[], int dst_pitch = 0);

template <int N>
int Preprocess_BCn(const uint8_t in_data[], int tiles_w, int tiles_h, bool flip_vertical, bool invert_green,
                   uint8_t out_data[], int out_pitch = 0);

void ComputeTangentBasis(size_t vtx_offset, size_t vtx_start, std::vector<vertex_t> &vertices,
                         Span<uint32_t> new_vtx_indices, Span<const uint32_t> indices);

std::unique_ptr<uint8_t[]> ReadTGAFile(const void *data, int data_len, int &w, int &h, eTexFormat &format);
bool ReadTGAFile(const void *data, int data_len, int &w, int &h, eTexFormat &format, uint8_t *out_data,
                 uint32_t &out_size);

void WriteTGA(const uint8_t *data, int w, int h, int bpp, std::string_view name);
void WritePFM(std::string_view base_name, const float values[], int w, int h, int channels);
} // namespace Ray
