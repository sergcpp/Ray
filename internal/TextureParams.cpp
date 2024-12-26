#include "TextureParams.h"

#include <cassert>

#include <algorithm>

namespace Ray {
#define DECORATE(X, Y, Z, W, XX) {Y, Z},
struct {
    int channel_count;
    int pp_data_len;
} g_tex_format_info[] = {
#include "TextureFormat.inl"
};
#undef DECORATE

extern const int g_block_res[][2] = {
    {4, 4},   // _4x4
    {5, 4},   // _5x4
    {5, 5},   // _5x5
    {6, 5},   // _6x5
    {6, 6},   // _6x6
    {8, 5},   // _8x5
    {8, 6},   // _8x6
    {8, 8},   // _8x8
    {10, 5},  // _10x5
    {10, 6},  // _10x6
    {10, 8},  // _10x8
    {10, 10}, // _10x10
    {12, 10}, // _12x10
    {12, 12}  // _12x12
};
static_assert(sizeof(g_block_res) / sizeof(g_block_res[0]) == int(eTexBlock::_None), "!");
} // namespace Ray

int Ray::GetChannelCount(const eTexFormat format) {
    return g_tex_format_info[int(format)].channel_count;
}

int Ray::GetPerPixelDataLen(const eTexFormat format) { return g_tex_format_info[int(format)].pp_data_len; }

int Ray::GetBlockLenBytes(const eTexFormat format, const eTexBlock block) {
    static_assert(int(eTexFormat::_Count) == 31, "Update the list below!");
    switch (format) {
    case eTexFormat::BC1:
        assert(block == eTexBlock::_4x4);
        return 8;
    case eTexFormat::BC2:
    case eTexFormat::BC3:
    case eTexFormat::BC5:
        assert(block == eTexBlock::_4x4);
        return 16;
    case eTexFormat::BC4:
        assert(block == eTexBlock::_4x4);
        return 8;
    default:
        return -1;
    }
    return -1;
}

int Ray::GetBlockCount(const int w, const int h, const eTexBlock block) {
    const int i = int(block);
    return ((w + g_block_res[i][0] - 1) / g_block_res[i][0]) * ((h + g_block_res[i][1] - 1) / g_block_res[i][1]);
}

uint32_t Ray::EstimateMemory(const Tex2DParams &params) {
    uint32_t total_len = 0;
    for (int i = 0; i < params.mip_count; i++) {
        const int w = std::max(params.w >> i, 1);
        const int h = std::max(params.h >> i, 1);

        if (IsCompressedFormat(params.format)) {
            const int block_len = GetBlockLenBytes(params.format, params.block);
            const int block_cnt = GetBlockCount(w, h, params.block);

            total_len += uint32_t(block_len) * block_cnt;
        } else {
            assert(g_tex_format_info[int(params.format)].pp_data_len != 0);
            total_len += w * h * g_tex_format_info[int(params.format)].pp_data_len;
        }
    }
    return total_len;
}

//
// All this is needed for reading KTX files
//

#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 33776
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 33777
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 33778
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 33779

#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT 35917
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT 35918
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT 35919

#define GL_COMPRESSED_RGBA_ASTC_4x4_KHR 0x93B0
#define GL_COMPRESSED_RGBA_ASTC_5x4_KHR 0x93B1
#define GL_COMPRESSED_RGBA_ASTC_5x5_KHR 0x93B2
#define GL_COMPRESSED_RGBA_ASTC_6x5_KHR 0x93B3
#define GL_COMPRESSED_RGBA_ASTC_6x6_KHR 0x93B4
#define GL_COMPRESSED_RGBA_ASTC_8x5_KHR 0x93B5
#define GL_COMPRESSED_RGBA_ASTC_8x6_KHR 0x93B6
#define GL_COMPRESSED_RGBA_ASTC_8x8_KHR 0x93B7
#define GL_COMPRESSED_RGBA_ASTC_10x5_KHR 0x93B8
#define GL_COMPRESSED_RGBA_ASTC_10x6_KHR 0x93B9
#define GL_COMPRESSED_RGBA_ASTC_10x8_KHR 0x93BA
#define GL_COMPRESSED_RGBA_ASTC_10x10_KHR 0x93BB
#define GL_COMPRESSED_RGBA_ASTC_12x10_KHR 0x93BC
#define GL_COMPRESSED_RGBA_ASTC_12x12_KHR 0x93BD

#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR 0x93D0
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR 0x93D1
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR 0x93D2
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR 0x93D3
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR 0x93D4
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR 0x93D5
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR 0x93D6
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR 0x93D7
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR 0x93D8
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR 0x93D9
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR 0x93DA
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR 0x93DB
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR 0x93DC
#define GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR 0x93DD

Ray::eTexFormat Ray::FormatFromGLInternalFormat(const uint32_t gl_internal_format, eTexBlock *block, bool *is_srgb) {
    (*is_srgb) = false;

    switch (gl_internal_format) {
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        (*block) = eTexBlock::_4x4;
        return eTexFormat::BC1;
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        (*block) = eTexBlock::_4x4;
        return eTexFormat::BC2;
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        (*block) = eTexBlock::_4x4;
        return eTexFormat::BC3;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
        (*is_srgb) = true;
    /*case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
        (*block) = eTexBlock::_4x4;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:
        (*block) = eTexBlock::_5x4;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:
        (*block) = eTexBlock::_5x5;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:
        (*block) = eTexBlock::_6x5;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:
        (*block) = eTexBlock::_6x6;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:
        (*block) = eTexBlock::_8x5;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:
        (*block) = eTexBlock::_8x6;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
        (*block) = eTexBlock::_8x8;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:
        (*block) = eTexBlock::_10x5;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:
        (*block) = eTexBlock::_10x6;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:
        (*block) = eTexBlock::_10x8;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:
        (*block) = eTexBlock::_10x10;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:
        (*block) = eTexBlock::_12x10;
        return eTexFormat::ASTC;
    case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
        (*is_srgb) = true;
    case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:
        (*block) = eTexBlock::_12x12;
        return eTexFormat::ASTC;*/
    default:
        assert(false && "Unsupported format!");
    }

    return eTexFormat::Undefined;
}

int Ray::BlockLenFromGLInternalFormat(const uint32_t gl_internal_format) {
    switch (gl_internal_format) {
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        return 8;
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        return 16;
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        return 16;
    default:
        assert(false);
    }
    return -1;
}