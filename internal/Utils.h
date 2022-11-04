#pragma once

#include <cstdint>

#include <memory>

namespace Ray {
void RGBMDecode(const uint8_t rgbm[4], float out_rgb[3]);
void RGBMEncode(const float rgb[3], uint8_t out_rgbm[4]);

std::unique_ptr<float[]> ConvertRGBE_to_RGB32F(const uint8_t image_data[], int w, int h);
std::unique_ptr<uint16_t[]> ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h);
void ConvertRGBE_to_RGB16F(const uint8_t image_data[], int w, int h, uint16_t *out_data);

std::unique_ptr<uint8_t[]> ConvertRGB32F_to_RGBE(const float image_data[], int w, int h, int channels);
std::unique_ptr<uint8_t[]> ConvertRGB32F_to_RGBM(const float image_data[], int w, int h, int channels);

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

enum class eMipOp {
    Skip = 0,
    Zero,        // fill with zeroes
    Avg,         // average value of 4 pixels
    Min,         // min value of 4 pixels
    Max,         // max value of 4 pixels
    MinBilinear, // min value of 4 pixels and result of bilinear interpolation with
                 // neighbours
    MaxBilinear  // max value of 4 pixels and result of bilinear interpolation with
                 // neighbours
};
int InitMipMaps(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16], int channels,
                const eMipOp op[4]);
int InitMipMapsRGBM(std::unique_ptr<uint8_t[]> mipmaps[16], int widths[16], int heights[16]);

void ReorderTriangleIndices(const uint32_t *indices, uint32_t indices_count, uint32_t vtx_count, uint32_t *out_indices);

uint16_t f32_to_f16(float value);

struct KTXHeader { // NOLINT
    char identifier[12] = {'\xAB', 'K', 'T', 'X', ' ', '1', '1', '\xBB', '\r', '\n', '\x1A', '\n'};
    uint32_t endianness = 0x04030201;
    uint32_t gl_type;
    uint32_t gl_type_size;
    uint32_t gl_format;
    uint32_t gl_internal_format;
    uint32_t gl_base_internal_format;
    uint32_t pixel_width;
    uint32_t pixel_height;
    uint32_t pixel_depth;
    uint32_t array_elements_count;
    uint32_t faces_count;
    uint32_t mipmap_levels_count;
    uint32_t key_value_data_size;
};
static_assert(sizeof(KTXHeader) == 64, "!");

/*	the following constants were copied directly off the MSDN website	*/

/*	The dwFlags member of the original DDSURFACEDESC2 structure
        can be set to one or more of the following values.	*/
#define DDSD_CAPS 0x00000001
#define DDSD_HEIGHT 0x00000002
#define DDSD_WIDTH 0x00000004
#define DDSD_PITCH 0x00000008
#define DDSD_PIXELFORMAT 0x00001000
#define DDSD_MIPMAPCOUNT 0x00020000
#define DDSD_LINEARSIZE 0x00080000
#define DDSD_DEPTH 0x00800000

/*	DirectDraw Pixel Format	*/
#define DDPF_ALPHAPIXELS 0x00000001
#define DDPF_FOURCC 0x00000004
#define DDPF_RGB 0x00000040

/*	The dwCaps1 member of the DDSCAPS2 structure can be
        set to one or more of the following values.	*/
#define DDSCAPS_COMPLEX 0x00000008
#define DDSCAPS_TEXTURE 0x00001000
#define DDSCAPS_MIPMAP 0x00400000

struct DDSHeader {
    uint32_t dwMagic;
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];

    /*  DDPIXELFORMAT	*/
    struct {
        uint32_t dwSize;
        uint32_t dwFlags;
        uint32_t dwFourCC;
        uint32_t dwRGBBitCount;
        uint32_t dwRBitMask;
        uint32_t dwGBitMask;
        uint32_t dwBBitMask;
        uint32_t dwAlphaBitMask;
    } sPixelFormat;

    /*  DDCAPS2	*/
    struct {
        uint32_t dwCaps1;
        uint32_t dwCaps2;
        uint32_t dwDDSX;
        uint32_t dwReserved;
    } sCaps;
    uint32_t dwReserved2;
};
static_assert(sizeof(DDSHeader) == 128, "!");

extern const uint8_t _blank_BC3_block_4x4[];
extern const int _blank_BC3_block_4x4_len;

extern const uint8_t _blank_ASTC_block_4x4[];
extern const int _blank_ASTC_block_4x4_len;

//
// BCn compression
//

int GetRequiredMemory_BC1(int w, int h);
int GetRequiredMemory_BC3(int w, int h);
int GetRequiredMemory_BC4(int w, int h);

// NOTE: intended for realtime compression, quality may be not the best
template <int SrcChannels> void CompressImage_BC1(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template <bool Is_YCoCg = false> void CompressImage_BC3(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
template <int SrcChannels = 1> void CompressImage_BC4(const uint8_t img_src[], int w, int h, uint8_t img_dst[]);
}
