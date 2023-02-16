#include "Texture.h"

#include <memory>

#include "../../Log.h"
#include "../Utils.h"
#include "Context.h"
#include "Utils.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef NDEBUG
// #define TEX_VERBOSE_LOGGING
#endif

namespace Ray {
namespace Vk {
extern const VkFilter g_vk_min_mag_filter[];
extern const VkSamplerAddressMode g_vk_wrap_mode[];
extern const VkSamplerMipmapMode g_vk_mipmap_mode[];
extern const VkCompareOp g_vk_compare_ops[];

extern const float AnisotropyLevel;

extern const VkFormat g_vk_formats[] = {
    VK_FORMAT_UNDEFINED,                // Undefined
    VK_FORMAT_R8G8B8_UNORM,             // RawRGB888
    VK_FORMAT_R8G8B8A8_UNORM,           // RawRGBA8888
    VK_FORMAT_R8G8B8A8_SNORM,           // RawRGBA8888Signed
    VK_FORMAT_B8G8R8A8_UNORM,           // RawBGRA8888
    VK_FORMAT_R32_SFLOAT,               // RawR32F
    VK_FORMAT_R16_SFLOAT,               // RawR16F
    VK_FORMAT_R8_UNORM,                 // RawR8
    VK_FORMAT_R32_UINT,                 // RawR32UI
    VK_FORMAT_R8G8_UNORM,               // RawRG88
    VK_FORMAT_R32G32B32_SFLOAT,         // RawRGB32F
    VK_FORMAT_R32G32B32A32_SFLOAT,      // RawRGBA32F
    VK_FORMAT_UNDEFINED,                // RawRGBE8888
    VK_FORMAT_R16G16B16_SFLOAT,         // RawRGB16F
    VK_FORMAT_R16G16B16A16_SFLOAT,      // RawRGBA16F
    VK_FORMAT_R16G16_SNORM,             // RawRG16Snorm
    VK_FORMAT_R16G16_UNORM,             // RawRG16
    VK_FORMAT_R16G16_SFLOAT,            // RawRG16F
    VK_FORMAT_R32G32_SFLOAT,            // RawRG32F
    VK_FORMAT_R32G32_UINT,              // RawRG32U
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, // RawRGB10_A2
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,  // RawRG11F_B10F
    VK_FORMAT_D16_UNORM,                // Depth16
    VK_FORMAT_D24_UNORM_S8_UINT,        // Depth24Stencil8
    VK_FORMAT_D32_SFLOAT_S8_UINT,       // Depth32Stencil8
#ifndef __ANDROID__
    VK_FORMAT_D32_SFLOAT, // Depth32
#endif
    VK_FORMAT_BC1_RGBA_UNORM_BLOCK, // BC1
    VK_FORMAT_BC2_UNORM_BLOCK,      // BC2
    VK_FORMAT_BC3_UNORM_BLOCK,      // BC3
    VK_FORMAT_BC4_UNORM_BLOCK,      // BC4
    VK_FORMAT_BC5_UNORM_BLOCK,      // BC5
    VK_FORMAT_UNDEFINED,            // ASTC
    VK_FORMAT_UNDEFINED,            // None
};
static_assert(sizeof(g_vk_formats) / sizeof(g_vk_formats[0]) == size_t(eTexFormat::_Count), "!");

uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties *mem_properties, uint32_t mem_type_bits,
                        VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size);

uint32_t TextureHandleCounter = 0;

// make sure we can simply cast these
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT == 1, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_2_BIT == 2, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_4_BIT == 4, "!");
static_assert(VkSampleCountFlagBits::VK_SAMPLE_COUNT_8_BIT == 8, "!");

VkFormat ToSRGBFormat(const VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return VK_FORMAT_R8_SRGB;
    case VK_FORMAT_R8G8_UNORM:
        return VK_FORMAT_R8G8_SRGB;
    case VK_FORMAT_R8G8B8_UNORM:
        return VK_FORMAT_R8G8B8_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case VK_FORMAT_BC2_UNORM_BLOCK:
        return VK_FORMAT_BC2_SRGB_BLOCK;
    case VK_FORMAT_BC3_UNORM_BLOCK:
        return VK_FORMAT_BC3_SRGB_BLOCK;
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
        return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    default:
        return format;
    }
    return VK_FORMAT_UNDEFINED;
}

VkImageUsageFlags to_vk_image_usage(const eTexUsage usage, const eTexFormat format) {
    VkImageUsageFlags ret = 0;
    if (uint8_t(usage & eTexUsage::Transfer)) {
        ret |= (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    if (uint8_t(usage & eTexUsage::Sampled)) {
        ret |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (uint8_t(usage & eTexUsage::Storage)) {
        assert(!IsCompressedFormat(format));
        ret |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (uint8_t(usage & eTexUsage::RenderTarget)) {
        assert(!IsCompressedFormat(format));
        if (IsDepthFormat(format)) {
            ret |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        } else {
            ret |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
    }
    return ret;
}

static const int g_block_res[][2] = {
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

const eTexUsage g_tex_usage_per_state[] = {
    {},                      // Undefined
    {},                      // VertexBuffer
    {},                      // UniformBuffer
    {},                      // IndexBuffer
    eTexUsage::RenderTarget, // RenderTarget
    eTexUsage::Storage,      // UnorderedAccess
    eTexUsage::RenderTarget, // DepthRead
    eTexUsage::RenderTarget, // DepthWrite
    eTexUsage::RenderTarget, // StencilTestDepthFetch
    eTexUsage::Sampled,      // ShaderResource
    {},                      // IndirectArgument
    eTexUsage::Transfer,     // CopyDst
    eTexUsage::Transfer,     // CopySrc
    {},                      // BuildASRead
    {},                      // BuildASWrite
    {}                       // RayTracing
};
static_assert(sizeof(g_tex_usage_per_state) / sizeof(g_tex_usage_per_state[0]) == int(eResState::_Count), "!");

const int g_per_pixel_data_len[] = {
    -1, // Undefined
    3,  // RawRGB888
    4,  // RawRGBA8888
    4,  // RawRGBA8888Snorm
    4,  // RawBGRA8888
    4,  // RawR32F
    2,  // RawR16F
    1,  // RawR8
    4,  // RawR32UI
    2,  // RawRG88
    12, // RawRGB32F
    16, // RawRGBA32F
    4,  // RawRGBE8888
    6,  // RawRGB16F
    8,  // RawRGBA16F
    4,  // RawRG16Snorm
    4,  // RawRG16
    4,  // RawRG16F
    8,  // RawRG32F
    8,  // RawRG32UI
    4,  // RawRGB10_A2
    4,  // RawRG11F_B10F
    2,  // Depth16
    4,  // Depth24Stencil8
    5,  // Depth32Stencil8
#ifndef __ANDROID__
    4, // Depth32
#endif
    -1, // BC1
    -1, // BC2
    -1, // BC3
    -1, // BC4
    -1, // BC5
    -1, // ASTC
    -1  // None
};
static_assert(sizeof(g_per_pixel_data_len) / sizeof(g_per_pixel_data_len[0]) == int(eTexFormat::_Count), "!");

} // namespace Vk

bool EndsWith(const std::string &str1, const char *str2) {
    size_t len = strlen(str2);
    for (size_t i = 0; i < len; i++) {
        if (str1[str1.length() - i] != str2[len - i]) {
            return false;
        }
    }
    return true;
}
} // namespace Ray

int Ray::Vk::CalcMipCount(const int w, const int h, const int min_res, const eTexFilter filter) {
    int mip_count = 0;
    if (filter == eTexFilter::Trilinear || filter == eTexFilter::Bilinear) {
        int max_dim = std::max(w, h);
        do {
            mip_count++;
        } while ((max_dim /= 2) >= min_res);
    } else {
        mip_count = 1;
    }
    return mip_count;
}

int Ray::Vk::GetColorChannelCount(const eTexFormat format) {
    static_assert(int(eTexFormat::_Count) == 33, "Update the list below!");
    switch (format) {
    case eTexFormat::RawRGBA8888:
    case eTexFormat::RawRGBA8888Snorm:
    case eTexFormat::RawBGRA8888:
    case eTexFormat::RawRGBA32F:
    case eTexFormat::RawRGBE8888:
    case eTexFormat::RawRGBA16F:
    case eTexFormat::RawRGB10_A2:
    case eTexFormat::BC2:
    case eTexFormat::BC3:
        return 4;
    case eTexFormat::RawRGB888:
    case eTexFormat::RawRGB32F:
    case eTexFormat::RawRGB16F:
    case eTexFormat::RawRG11F_B10F:
    case eTexFormat::BC1:
        return 3;
    case eTexFormat::RawRG88:
    case eTexFormat::RawRG16:
    case eTexFormat::RawRG16Snorm:
    case eTexFormat::RawRG16F:
    case eTexFormat::RawRG32F:
    case eTexFormat::RawRG32UI:
    case eTexFormat::BC5:
        return 2;
    case eTexFormat::RawR32F:
    case eTexFormat::RawR16F:
    case eTexFormat::RawR8:
    case eTexFormat::RawR32UI:
    case eTexFormat::BC4:
        return 1;
    case eTexFormat::Depth16:
    case eTexFormat::Depth24Stencil8:
    case eTexFormat::Depth32Stencil8:
#ifndef __ANDROID__
    case eTexFormat::Depth32:
#endif
    case eTexFormat::Undefined:
    default:
        return 0;
    }
}

int Ray::Vk::GetPerPixelDataLen(const eTexFormat format) { return g_per_pixel_data_len[int(format)]; }

int Ray::Vk::GetBlockLenBytes(const eTexFormat format, const eTexBlock block) {
    static_assert(int(eTexFormat::_Count) == 33, "Update the list below!");
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
    case eTexFormat::ASTC:
        assert(false);
    default:
        return -1;
    }
    return -1;
}

int Ray::Vk::GetBlockCount(const int w, const int h, const eTexBlock block) {
    const int i = int(block);
    return ((w + g_block_res[i][0] - 1) / g_block_res[i][0]) * ((h + g_block_res[i][1] - 1) / g_block_res[i][1]);
}

uint32_t Ray::Vk::EstimateMemory(const Tex2DParams &params) {
    uint32_t total_len = 0;
    for (int i = 0; i < params.mip_count; i++) {
        const int w = std::max(params.w >> i, 1);
        const int h = std::max(params.h >> i, 1);

        if (IsCompressedFormat(params.format)) {
            const int block_len = GetBlockLenBytes(params.format, params.block);
            const int block_cnt = GetBlockCount(w, h, params.block);

            total_len += uint32_t(block_len) * block_cnt;
        } else {
            assert(g_per_pixel_data_len[int(params.format)] != -1);
            total_len += w * h * g_per_pixel_data_len[int(params.format)];
        }
    }
    return params.cube ? 6 * total_len : total_len;
}

//
// All this is needed when reading KTX files
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

Ray::Vk::eTexFormat Ray::Vk::FormatFromGLInternalFormat(const uint32_t gl_internal_format, eTexBlock *block,
                                                        bool *is_srgb) {
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
    case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
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
        return eTexFormat::ASTC;
    default:
        assert(false && "Unsupported format!");
    }

    return eTexFormat::Undefined;
}

int Ray::Vk::BlockLenFromGLInternalFormat(const uint32_t gl_internal_format) {
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

Ray::Vk::eTexUsage Ray::Vk::TexUsageFromState(const eResState state) { return g_tex_usage_per_state[int(state)]; }

Ray::Vk::Texture2D::Texture2D(const char *name, Context *ctx, const Tex2DParams &p, MemoryAllocators *mem_allocs,
                              ILog *log)
    : ctx_(ctx), name_(name) {
    Init(p, mem_allocs, log);
}

Ray::Vk::Texture2D::Texture2D(const char *name, Context *ctx, const void *data, const uint32_t size,
                              const Tex2DParams &p, Buffer &stage_buf, void *_cmd_buf, MemoryAllocators *mem_allocs,
                              eTexLoadStatus *load_status, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, _cmd_buf, mem_allocs, load_status, log);
}

Ray::Vk::Texture2D::Texture2D(const char *name, Context *ctx, const void *data[6], const int size[6],
                              const Tex2DParams &p, Buffer &stage_buf, void *_cmd_buf, MemoryAllocators *mem_allocs,
                              eTexLoadStatus *load_status, ILog *log)
    : ctx_(ctx), name_(name) {
    Init(data, size, p, stage_buf, _cmd_buf, mem_allocs, load_status, log);
}

Ray::Vk::Texture2D::~Texture2D() { Free(); }

Ray::Vk::Texture2D &Ray::Vk::Texture2D::operator=(Texture2D &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    ctx_ = exchange(rhs.ctx_, nullptr);
    handle_ = exchange(rhs.handle_, {});
    alloc_ = exchange(rhs.alloc_, {});
    params = exchange(rhs.params, {});
    ready_ = exchange(rhs.ready_, false);
    cubemap_ready_ = exchange(rhs.cubemap_ready_, 0);
    name_ = std::move(rhs.name_);

    resource_state = exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Vk::Texture2D::Init(const Tex2DParams &p, MemoryAllocators *mem_allocs, ILog *log) {
    InitFromRAWData(nullptr, 0, nullptr, mem_allocs, p, log);
    ready_ = true;
}

void Ray::Vk::Texture2D::Init(const void *data, const uint32_t size, const Tex2DParams &p, Buffer &sbuf, void *_cmd_buf,
                              MemoryAllocators *mem_allocs, eTexLoadStatus *load_status, ILog *log) {
    if (!data) {
        uint8_t *stage_data = sbuf.Map(BufMapWrite);
        memcpy(stage_data, p.fallback_color, 4);
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
        sbuf.Unmap();

        Tex2DParams _p = p;
        _p.w = _p.h = 1;
        _p.mip_count = 1;
        _p.format = eTexFormat::RawRGBA8888;
        _p.usage = eTexUsage::Sampled | eTexUsage::Transfer;

        InitFromRAWData(&sbuf, 0, _cmd_buf, mem_allocs, _p, log);
        // mark it as not ready
        ready_ = false;
        (*load_status) = eTexLoadStatus::CreatedDefault;
    } else {
        if (EndsWith(name_, ".tga_rgbe") != 0 || EndsWith(name_, ".TGA_RGBE") != 0) {
            InitFromTGA_RGBEFile(data, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".tga") != 0 || EndsWith(name_, ".TGA") != 0) {
            InitFromTGAFile(data, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".dds") != 0 || EndsWith(name_, ".DDS") != 0) {
            InitFromDDSFile(data, size, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".ktx") != 0 || EndsWith(name_, ".KTX") != 0) {
            InitFromKTXFile(data, size, sbuf, _cmd_buf, mem_allocs, p, log);
        } else {
            uint8_t *stage_data = sbuf.Map(BufMapWrite);
            memcpy(stage_data, data, size);
            sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(size));
            sbuf.Unmap();

            InitFromRAWData(&sbuf, 0, _cmd_buf, mem_allocs, p, log);
        }
        ready_ = true;
        (*load_status) = eTexLoadStatus::CreatedFromData;
    }
}

void Ray::Vk::Texture2D::Init(const void *data[6], const int size[6], const Tex2DParams &p, Buffer &sbuf,
                              void *_cmd_buf, MemoryAllocators *mem_allocs, eTexLoadStatus *load_status, ILog *log) {
    if (!data) {
        uint8_t *stage_data = sbuf.Map(BufMapWrite);
        memcpy(stage_data, p.fallback_color, 4);
        sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
        sbuf.Unmap();

        int data_off[6] = {};

        Tex2DParams _p = p;
        _p.w = _p.h = 1;
        _p.format = eTexFormat::RawRGBA8888;
        _p.usage = eTexUsage::Sampled | eTexUsage::Transfer;

        InitFromRAWData(sbuf, data_off, _cmd_buf, mem_allocs, _p, log);
        // mark it as not ready
        ready_ = false;
        cubemap_ready_ = 0;
        (*load_status) = eTexLoadStatus::CreatedDefault;
    } else {
        if (EndsWith(name_, ".tga_rgbe") != 0 || EndsWith(name_, ".TGA_RGBE") != 0) {
            InitFromTGA_RGBEFile(data, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".tga") != 0 || EndsWith(name_, ".TGA") != 0) {
            InitFromTGAFile(data, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".ktx") != 0 || EndsWith(name_, ".KTX") != 0) {
            InitFromKTXFile(data, size, sbuf, _cmd_buf, mem_allocs, p, log);
        } else if (EndsWith(name_, ".dds") != 0 || EndsWith(name_, ".DDS") != 0) {
            InitFromDDSFile(data, size, sbuf, _cmd_buf, mem_allocs, p, log);
        } else {
            uint8_t *stage_data = sbuf.Map(BufMapWrite);
            uint32_t stage_off = 0;

            int data_off[6];
            for (int i = 0; i < 6; i++) {
                if (data[i]) {
                    memcpy(&stage_data[stage_off], data[i], size[i]);
                    data_off[i] = int(stage_off);
                    stage_off += size[i];
                } else {
                    data_off[i] = -1;
                }
            }
            sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(4));
            sbuf.Unmap();

            InitFromRAWData(sbuf, data_off, _cmd_buf, mem_allocs, p, log);
        }

        ready_ = (cubemap_ready_ & (1u << 0u)) == 1;
        for (unsigned i = 1; i < 6; i++) {
            ready_ = ready_ && ((cubemap_ready_ & (1u << i)) == 1);
        }
        (*load_status) = eTexLoadStatus::CreatedFromData;
    }
}

void Ray::Vk::Texture2D::Free() {
    if (params.format != eTexFormat::Undefined && !bool(params.flags & eTexFlagBits::NoOwnership)) {
        for (VkImageView view : handle_.views) {
            if (view) {
                ctx_->image_views_to_destroy[ctx_->backend_frame].push_back(view);
            }
        }
        ctx_->images_to_destroy[ctx_->backend_frame].push_back(handle_.img);
        ctx_->samplers_to_destroy[ctx_->backend_frame].push_back(handle_.sampler);
        ctx_->allocs_to_free[ctx_->backend_frame].emplace_back(std::move(alloc_));

        handle_ = {};
        params.format = eTexFormat::Undefined;
    }
}

bool Ray::Vk::Texture2D::Realloc(const int w, const int h, int mip_count, const int samples, const eTexFormat format,
                                 const eTexBlock block, const bool is_srgb, void *_cmd_buf,
                                 MemoryAllocators *mem_allocs, ILog *log) {
    VkImage new_image = VK_NULL_HANDLE;
    VkImageView new_image_view = VK_NULL_HANDLE;
    MemAllocation new_alloc = {};
    eResState new_resource_state = eResState::Undefined;

    mip_count = std::min(mip_count, CalcMipCount(w, h, 1, eTexFilter::Trilinear));

    { // create new image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(w);
        img_info.extent.height = uint32_t(h);
        img_info.extent.depth = 1;
        img_info.mipLevels = mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_vk_formats[size_t(format)];
        if (is_srgb) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(params.usage) != 0);
        img_info.usage = to_vk_image_usage(params.usage, format);

        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(samples);
        img_info.flags = 0;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &new_image);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return false;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(new_image);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), new_image, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        new_alloc = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                         FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                        img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                         name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        const VkDeviceSize aligned_offset = AlignTo(VkDeviceSize(new_alloc.alloc_off), tex_mem_req.alignment);

        res = vkBindImageMemory(ctx_->device(), new_image, new_alloc.owner->mem(new_alloc.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return false;
        }
    }

    { // create new image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = new_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = g_vk_formats[size_t(format)];
        if (is_srgb) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetColorChannelCount(format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &new_image_view);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return false;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(new_image_view);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

#ifdef TEX_VERBOSE_LOGGING
    if (params_.format != eTexFormat::Undefined) {
        log->Info("Realloc %s, %ix%i (%i mips) -> %ix%i (%i mips)", name_.c_str(), int(params_.w), int(params_.h),
                  int(params_.mip_count), w, h, mip_count);
    } else {
        log->Info("Alloc %s %ix%i (%i mips)", name_.c_str(), w, h, mip_count);
    }
#endif

    const TexHandle new_handle = {new_image, new_image_view, VK_NULL_HANDLE, exchange(handle_.sampler, {}),
                                  TextureHandleCounter++};
    uint16_t new_initialized_mips = 0;

    // copy data from old texture
    if (params.format == format) {
        int src_mip = 0, dst_mip = 0;
        while (std::max(params.w >> src_mip, 1) != std::max(w >> dst_mip, 1) ||
               std::max(params.h >> src_mip, 1) != std::max(h >> dst_mip, 1)) {
            if (std::max(params.w >> src_mip, 1) > std::max(w >> dst_mip, 1) ||
                std::max(params.h >> src_mip, 1) > std::max(h >> dst_mip, 1)) {
                ++src_mip;
            } else {
                ++dst_mip;
            }
        }

        VkImageCopy copy_regions[16];
        uint32_t copy_regions_count = 0;

        for (; src_mip < int(params.mip_count) && dst_mip < mip_count; ++src_mip, ++dst_mip) {
            if (initialized_mips_ & (1u << src_mip)) {
                VkImageCopy &reg = copy_regions[copy_regions_count++];

                reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                reg.srcSubresource.baseArrayLayer = 0;
                reg.srcSubresource.layerCount = 1;
                reg.srcSubresource.mipLevel = src_mip;
                reg.srcOffset = {0, 0, 0};
                reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                reg.dstSubresource.baseArrayLayer = 0;
                reg.dstSubresource.layerCount = 1;
                reg.dstSubresource.mipLevel = dst_mip;
                reg.dstOffset = {0, 0, 0};
                reg.extent = {uint32_t(std::max(w >> dst_mip, 1)), uint32_t(std::max(h >> dst_mip, 1)), 1};

#ifdef TEX_VERBOSE_LOGGING
                log->Info("Copying data mip %i [old] -> mip %i [new]", src_mip, dst_mip);
#endif

                new_initialized_mips |= (1u << dst_mip);
            }
        }

        if (copy_regions_count) {
            VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

            VkPipelineStageFlags src_stages = 0, dst_stages = 0;
            SmallVector<VkImageMemoryBarrier, 2> barriers;

            // src image barrier
            if (this->resource_state != eResState::CopySrc) {
                auto &new_barrier = barriers.emplace_back();
                new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
                new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
                new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
                new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
                new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.image = handle_.img;
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                new_barrier.subresourceRange.baseMipLevel = 0;
                new_barrier.subresourceRange.levelCount = params.mip_count; // transit the whole image
                new_barrier.subresourceRange.baseArrayLayer = 0;
                new_barrier.subresourceRange.layerCount = 1;

                src_stages |= VKPipelineStagesForState(this->resource_state);
                dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
            }

            // dst image barrier
            if (new_resource_state != eResState::CopyDst) {
                auto &new_barrier = barriers.emplace_back();
                new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                new_barrier.srcAccessMask = VKAccessFlagsForState(new_resource_state);
                new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
                new_barrier.oldLayout = VKImageLayoutForState(new_resource_state);
                new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
                new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                new_barrier.image = new_image;
                new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                new_barrier.subresourceRange.baseMipLevel = 0;
                new_barrier.subresourceRange.levelCount = mip_count; // transit the whole image
                new_barrier.subresourceRange.baseArrayLayer = 0;
                new_barrier.subresourceRange.layerCount = 1;

                src_stages |= VKPipelineStagesForState(new_resource_state);
                dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
            }

            if (!barriers.empty()) {
                vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages,
                                     0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.cdata());
            }

            this->resource_state = eResState::CopySrc;
            new_resource_state = eResState::CopyDst;

            vkCmdCopyImage(cmd_buf, handle_.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, new_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy_regions_count, copy_regions);
        }
    }
    Free();

    handle_ = new_handle;
    alloc_ = std::move(new_alloc);
    params.w = w;
    params.h = h;
    if (is_srgb) {
        params.flags |= eTexFlagBits::SRGB;
    } else {
        params.flags &= ~eTexFlagBits::SRGB;
    }
    params.mip_count = mip_count;
    params.samples = samples;
    params.format = format;
    params.block = block;
    initialized_mips_ = new_initialized_mips;

    this->resource_state = new_resource_state;

    return true;
}

void Ray::Vk::Texture2D::InitFromRAWData(Buffer *sbuf, int data_off, void *_cmd_buf, MemoryAllocators *mem_allocs,
                                         const Tex2DParams &p, ILog *log) {
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;
    initialized_mips_ = 0;

    int mip_count = params.mip_count;
    if (!mip_count) {
        mip_count = CalcMipCount(p.w, p.h, 1, p.sampling.filter);
    }

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(p.w);
        img_info.extent.height = uint32_t(p.h);
        img_info.extent.depth = 1;
        img_info.mipLevels = mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(p.usage) != 0);
        img_info.usage = to_vk_image_usage(p.usage, p.format);

        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(p.samples);
        img_info.flags = 0;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                      FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                     img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                      name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        if (!alloc_) {
            log->Error("Failed to allocate memory!");
            return;
        }

        const VkDeviceSize aligned_offset = AlignTo(VkDeviceSize(alloc_.alloc_off), tex_mem_req.alignment);

        res = vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view(s)
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        if (IsDepthStencilFormat(p.format)) {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        } else if (IsDepthFormat(p.format)) {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetColorChannelCount(p.format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

        if (IsDepthStencilFormat(p.format)) {
            // create additional depth-only image view
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            VkImageView depth_only_view;
            const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &depth_only_view);
            if (res != VK_SUCCESS) {
                log->Error("Failed to create image view!");
                return;
            }
            handle_.views.push_back(depth_only_view);
        }

#ifdef ENABLE_OBJ_LABELS
        for (VkImageView view : handle_.views) {
            VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
            name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
            name_info.objectHandle = uint64_t(view);
            name_info.pObjectName = name_.c_str();
            vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
        }
#endif
    }

    this->resource_state = eResState::Undefined;

    if (sbuf) {
        assert(p.samples == 1);
        assert(sbuf && sbuf->type() == eBufType::Stage);
        VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

        VkPipelineStageFlags src_stages = 0, dst_stages = 0;
        SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
        SmallVector<VkImageMemoryBarrier, 1> img_barriers;

        if (sbuf->resource_state != eResState::Undefined && sbuf->resource_state != eResState::CopySrc) {
            auto &new_barrier = buf_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf->resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.buffer = sbuf->vk_handle();
            new_barrier.offset = VkDeviceSize(data_off);
            new_barrier.size = VkDeviceSize(sbuf->size() - data_off);

            src_stages |= VKPipelineStagesForState(sbuf->resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
        }

        if (this->resource_state != eResState::CopyDst) {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
            new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
            new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = handle_.img;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = mip_count; // transit whole image
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = 1;

            src_stages |= VKPipelineStagesForState(this->resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
        }

        if (!buf_barriers.empty() || !img_barriers.empty()) {
            vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                                 nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                                 uint32_t(img_barriers.size()), img_barriers.cdata());
        }

        sbuf->resource_state = eResState::CopySrc;
        this->resource_state = eResState::CopyDst;

        VkBufferImageCopy region = {};
        region.bufferOffset = VkDeviceSize(data_off);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {uint32_t(p.w), uint32_t(p.h), 1};

        vkCmdCopyBufferToImage(cmd_buf, sbuf->vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        initialized_mips_ |= (1u << 0);
    }

    { // create new sampler
        VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = g_vk_min_mag_filter[size_t(p.sampling.filter)];
        sampler_info.minFilter = g_vk_min_mag_filter[size_t(p.sampling.filter)];
        sampler_info.addressModeU = g_vk_wrap_mode[size_t(p.sampling.wrap)];
        sampler_info.addressModeV = g_vk_wrap_mode[size_t(p.sampling.wrap)];
        sampler_info.addressModeW = g_vk_wrap_mode[size_t(p.sampling.wrap)];
        sampler_info.anisotropyEnable = VK_TRUE;
        sampler_info.maxAnisotropy = AnisotropyLevel;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = p.sampling.compare != eTexCompare::None ? VK_TRUE : VK_FALSE;
        sampler_info.compareOp = g_vk_compare_ops[size_t(p.sampling.compare)];
        sampler_info.mipmapMode = g_vk_mipmap_mode[size_t(p.sampling.filter)];
        sampler_info.mipLodBias = p.sampling.lod_bias.to_float();
        sampler_info.minLod = p.sampling.min_lod.to_float();
        sampler_info.maxLod = p.sampling.max_lod.to_float();

        const VkResult res = vkCreateSampler(ctx_->device(), &sampler_info, nullptr, &handle_.sampler);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create sampler!");
        }
    }
}

void Ray::Vk::Texture2D::InitFromTGAFile(const void *data, Buffer &sbuf, void *_cmd_buf, MemoryAllocators *mem_allocs,
                                         const Tex2DParams &p, ILog *log) {
    int w = 0, h = 0;
    eTexFormat format = eTexFormat::Undefined;
    uint32_t img_size = 0;
    const bool res1 = ReadTGAFile(data, w, h, format, nullptr, img_size);
    if (!res1 || img_size <= sbuf.size()) {
        ctx_->log()->Error("Failed to read tga data!");
        return;
    }

    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    if (!stage_data) {
        ctx_->log()->Error("Failed to map stage buffer!");
        return;
    }

    const bool res2 = ReadTGAFile(data, w, h, format, stage_data, img_size);
    if (!res2) {
        ctx_->log()->Error("Failed to read tga data!");
    }
    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(img_size));
    sbuf.Unmap();

    Tex2DParams _p = p;
    _p.w = w;
    _p.h = h;
    _p.format = format;

    InitFromRAWData(&sbuf, 0, _cmd_buf, mem_allocs, _p, log);
}

void Ray::Vk::Texture2D::InitFromTGA_RGBEFile(const void *data, Buffer &sbuf, void *_cmd_buf,
                                              MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    int w = 0, h = 0;
    eTexFormat format = eTexFormat::Undefined;
    std::unique_ptr<uint8_t[]> image_data = ReadTGAFile(data, w, h, format);
    assert(format == eTexFormat::RawRGBA8888);

    uint16_t *stage_data = reinterpret_cast<uint16_t *>(sbuf.Map(BufMapWrite));
    ConvertRGBE_to_RGB16F(image_data.get(), w, h, stage_data);
    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(3 * w * h * sizeof(uint16_t)));
    sbuf.Unmap();

    Tex2DParams _p = p;
    _p.w = w;
    _p.h = h;
    _p.format = eTexFormat::RawRGB16F;

    InitFromRAWData(&sbuf, 0, _cmd_buf, mem_allocs, _p, log);
}

void Ray::Vk::Texture2D::InitFromDDSFile(const void *data, const int size, Buffer &sbuf, void *_cmd_buf,
                                         MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    DDSHeader header;
    memcpy(&header, data, sizeof(DDSHeader));

    eTexFormat format;
    eTexBlock block;
    int block_size_bytes;

    const int px_format = int(header.sPixelFormat.dwFourCC >> 24u) - '0';
    switch (px_format) {
    case 1:
        format = eTexFormat::BC1;
        block = eTexBlock::_4x4;
        block_size_bytes = 8;
        break;
    case 3:
        format = eTexFormat::BC2;
        block = eTexBlock::_4x4;
        block_size_bytes = 16;
        break;
    case 5:
        format = eTexFormat::BC3;
        block = eTexBlock::_4x4;
        block_size_bytes = 16;
        break;
    default:
        log->Error("Unknow DDS pixel format %i", px_format);
        return;
    }

    Free();
    Realloc(int(header.dwWidth), int(header.dwHeight), int(header.dwMipMapCount), 1, format, block,
            bool(p.flags & eTexFlagBits::SRGB), _cmd_buf, mem_allocs, log);

    params.flags = p.flags;
    params.block = block;
    params.sampling = p.sampling;

    int w = params.w, h = params.h;
    uint32_t bytes_left = uint32_t(size) - sizeof(DDSHeader);
    const uint8_t *p_data = (uint8_t *)data + sizeof(DDSHeader);

    assert(bytes_left <= sbuf.size());
    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    memcpy(stage_data, p_data, bytes_left);
    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(bytes_left));
    sbuf.Unmap();

    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = 0;
        new_barrier.size = VkDeviceSize(bytes_left);

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = params.mip_count; // transit the whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[16] = {};
    int regions_count = 0;

    uintptr_t data_off = 0;
    for (uint32_t i = 0; i < header.dwMipMapCount; i++) {
        const uint32_t len = ((w + 3) / 4) * ((h + 3) / 4) * block_size_bytes;
        if (len > bytes_left) {
            log->Error("Insufficient data length, bytes left %i, expected %i", bytes_left, len);
            return;
        }

        VkBufferImageCopy &reg = regions[regions_count++];

        reg.bufferOffset = VkDeviceSize(data_off);
        reg.bufferRowLength = 0;
        reg.bufferImageHeight = 0;

        reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.mipLevel = i;
        reg.imageSubresource.baseArrayLayer = 0;
        reg.imageSubresource.layerCount = 1;

        reg.imageOffset = {0, 0, 0};
        reg.imageExtent = {uint32_t(w), uint32_t(h), 1};

        initialized_mips_ |= (1u << i);

        data_off += len;
        bytes_left -= len;
        w = std::max(w / 2, 1);
        h = std::max(h / 2, 1);
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions_count,
                           regions);

    ApplySampling(p.sampling, log);
}

void Ray::Vk::Texture2D::InitFromKTXFile(const void *data, const int size, Buffer &sbuf, void *_cmd_buf,
                                         MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    KTXHeader header;
    memcpy(&header, data, sizeof(KTXHeader));

    eTexBlock block;
    bool is_srgb_format;
    eTexFormat format = FormatFromGLInternalFormat(header.gl_internal_format, &block, &is_srgb_format);

    if (is_srgb_format && !bool(params.flags & eTexFlagBits::SRGB)) {
        log->Warning("Loading SRGB texture as non-SRGB!");
    }

    Free();
    Realloc(int(header.pixel_width), int(header.pixel_height), int(header.mipmap_levels_count), 1, format, block,
            bool(p.flags & eTexFlagBits::SRGB), nullptr, nullptr, log);

    params.flags = p.flags;
    params.block = block;
    params.sampling = p.sampling;

    int w = int(params.w);
    int h = int(params.h);

    params.w = w;
    params.h = h;

    const auto *_data = (const uint8_t *)data;
    int data_offset = sizeof(KTXHeader);

    assert(uint32_t(size - data_offset) <= sbuf.size());
    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    memcpy(stage_data, _data, size);
    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(size));
    sbuf.Unmap();

    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = 0;
        new_barrier.size = VkDeviceSize(size);

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = params.mip_count; // transit the whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[16] = {};
    int regions_count = 0;

    for (int i = 0; i < int(header.mipmap_levels_count); i++) {
        if (data_offset + int(sizeof(uint32_t)) > size) {
            log->Error("Insufficient data length, bytes left %i, expected %i", size - data_offset, sizeof(uint32_t));
            break;
        }

        uint32_t img_size;
        memcpy(&img_size, &_data[data_offset], sizeof(uint32_t));
        if (data_offset + int(img_size) > size) {
            log->Error("Insufficient data length, bytes left %i, expected %i", size - data_offset, img_size);
            break;
        }

        data_offset += sizeof(uint32_t);

        VkBufferImageCopy &reg = regions[regions_count++];

        reg.bufferOffset = VkDeviceSize(data_offset);
        reg.bufferRowLength = 0;
        reg.bufferImageHeight = 0;

        reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.mipLevel = i;
        reg.imageSubresource.baseArrayLayer = 0;
        reg.imageSubresource.layerCount = 1;

        reg.imageOffset = {0, 0, 0};
        reg.imageExtent = {uint32_t(w), uint32_t(h), 1};

        initialized_mips_ |= (1u << i);
        data_offset += img_size;

        w = std::max(w / 2, 1);
        h = std::max(h / 2, 1);

        const int pad = (data_offset % 4) ? (4 - (data_offset % 4)) : 0;
        data_offset += pad;
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions_count,
                           regions);

    ApplySampling(p.sampling, log);
}

void Ray::Vk::Texture2D::InitFromRAWData(Buffer &sbuf, int data_off[6], void *_cmd_buf, MemoryAllocators *mem_allocs,
                                         const Tex2DParams &p, ILog *log) {
    assert(p.w > 0 && p.h > 0);
    Free();

    handle_.generation = TextureHandleCounter++;
    params = p;
    initialized_mips_ = 0;

    const int mip_count = CalcMipCount(p.w, p.h, 1, p.sampling.filter);

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(p.w);
        img_info.extent.height = uint32_t(p.h);
        img_info.extent.depth = 1;
        img_info.mipLevels = mip_count;
        img_info.arrayLayers = 1;
        img_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(p.usage) != 0);
        img_info.usage = to_vk_image_usage(p.usage, p.format);
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VkSampleCountFlagBits(p.samples);
        img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                      FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                     img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                      name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        const VkDeviceSize aligned_offset = AlignTo(VkDeviceSize(alloc_.alloc_off), tex_mem_req.alignment);

        res = vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (GetColorChannelCount(p.format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(handle_.views[0]);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

    assert(p.samples == 1);
    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(sbuf.size());

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[6] = {};
    for (int i = 0; i < 6; i++) {
        regions[i].bufferOffset = VkDeviceSize(data_off[i]);
        regions[i].bufferRowLength = 0;
        regions[i].bufferImageHeight = 0;

        regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[i].imageSubresource.mipLevel = 0;
        regions[i].imageSubresource.baseArrayLayer = i;
        regions[i].imageSubresource.layerCount = 1;

        regions[i].imageOffset = {0, 0, 0};
        regions[i].imageExtent = {uint32_t(p.w), uint32_t(p.h), 1};
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);

    initialized_mips_ |= (1u << 0);

    ApplySampling(p.sampling, log);
}

void Ray::Vk::Texture2D::InitFromTGAFile(const void *data[6], Buffer &sbuf, void *_cmd_buf,
                                         MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    int w = 0, h = 0;
    eTexFormat format = eTexFormat::Undefined;

    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    uint32_t stage_off = 0;

    int data_off[6] = {-1, -1, -1, -1, -1, -1};

    for (int i = 0; i < 6; i++) {
        if (data[i]) {
            uint32_t data_size;
            const bool res1 = ReadTGAFile(data[i], w, h, format, nullptr, data_size);
            if (!res1) {
                ctx_->log()->Error("Failed to read tga data!");
                break;
            }

            assert(stage_off + data_size < sbuf.size());
            const bool res2 = ReadTGAFile(data[i], w, h, format, &stage_data[stage_off], data_size);
            if (!res2) {
                ctx_->log()->Error("Failed to read tga data!");
                break;
            }

            data_off[i] = int(stage_off);
            stage_off += data_size;
        }
    }

    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(stage_off));
    sbuf.Unmap();

    Tex2DParams _p = p;
    _p.w = w;
    _p.h = h;
    _p.format = format;

    InitFromRAWData(sbuf, data_off, _cmd_buf, mem_allocs, _p, log);
}

void Ray::Vk::Texture2D::InitFromTGA_RGBEFile(const void *data[6], Buffer &sbuf, void *_cmd_buf,
                                              MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    int w = p.w, h = p.h;

    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    uint32_t stage_off = 0;

    int data_off[6];

    for (int i = 0; i < 6; i++) {
        if (data[i]) {
            const uint32_t img_size = 3 * w * h * sizeof(uint16_t);
            assert(stage_off + img_size <= sbuf.size());
            ConvertRGBE_to_RGB16F((const uint8_t *)data[i], w, h, (uint16_t *)&stage_data[stage_off]);
            data_off[i] = int(stage_off);
            stage_off += img_size;
        } else {
            data_off[i] = -1;
        }
    }

    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(stage_off));
    sbuf.Unmap();

    Tex2DParams _p = p;
    _p.w = w;
    _p.h = h;
    _p.format = eTexFormat::RawRGB16F;

    InitFromRAWData(sbuf, data_off, _cmd_buf, mem_allocs, _p, log);
}

void Ray::Vk::Texture2D::InitFromDDSFile(const void *data[6], const int size[6], Buffer &sbuf, void *_cmd_buf,
                                         MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    assert(p.w > 0 && p.h > 0);
    Free();

    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    uint32_t data_off[6] = {};
    uint32_t stage_len = 0;

    eTexFormat first_format = eTexFormat::None;
    eTexBlock first_block = eTexBlock::_None;
    uint32_t first_mip_count = 0;
    int first_block_size_bytes = 0;

    for (int i = 0; i < 6; ++i) {
        const DDSHeader *header = reinterpret_cast<const DDSHeader *>(data[i]);

        eTexFormat format;
        eTexBlock block;
        int block_size_bytes;
        const int px_format = int(header->sPixelFormat.dwFourCC >> 24u) - '0';
        switch (px_format) {
        case 1:
            format = eTexFormat::BC1;
            block = eTexBlock::_4x4;
            block_size_bytes = 8;
            break;
        case 3:
            format = eTexFormat::BC2;
            block = eTexBlock::_4x4;
            block_size_bytes = 16;
            break;
        case 5:
            format = eTexFormat::BC3;
            block = eTexBlock::_4x4;
            block_size_bytes = 16;
            break;
        default:
            log->Error("Unknow DDS pixel format %i", px_format);
            return;
        }

        if (i == 0) {
            first_format = format;
            first_block = block;
            first_mip_count = header->dwMipMapCount;
            first_block_size_bytes = block_size_bytes;
        } else {
            assert(format == first_format);
            assert(block == first_block);
            assert(first_mip_count == header->dwMipMapCount);
            assert(block_size_bytes == first_block_size_bytes);
        }

        memcpy(stage_data + stage_len, data[i], size[i]);

        data_off[i] = stage_len;
        stage_len += size[i];
    }

    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(stage_len));
    sbuf.Unmap();

    handle_.generation = TextureHandleCounter++;
    params = p;
    params.block = first_block;
    params.cube = 1;
    initialized_mips_ = 0;

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(p.w);
        img_info.extent.height = uint32_t(p.h);
        img_info.extent.depth = 1;
        img_info.mipLevels = first_mip_count;
        img_info.arrayLayers = 6;
        img_info.format = g_vk_formats[size_t(first_format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(p.usage) != 0);
        img_info.usage = to_vk_image_usage(p.usage, first_format);
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                      FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                     img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                      name_.c_str());
        if (!alloc_) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        const VkDeviceSize aligned_offset = AlignTo(VkDeviceSize(alloc_.alloc_off), tex_mem_req.alignment);

        res = vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = first_mip_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 6;

        if (GetColorChannelCount(p.format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(handle_.views[0]);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

    assert(p.samples == 1);
    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(sbuf.size());

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = first_mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 6;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[6 * 16] = {};
    int regions_count = 0;

    for (int i = 0; i < 6; i++) {
        const auto *header = reinterpret_cast<const DDSHeader *>(data[i]);

        int offset = sizeof(DDSHeader);
        int data_len = size[i] - int(sizeof(DDSHeader));

        for (uint32_t j = 0; j < header->dwMipMapCount; j++) {
            const int width = std::max(int(header->dwWidth >> j), 1), height = std::max(int(header->dwHeight >> j), 1);

            const int image_len = ((width + 3) / 4) * ((height + 3) / 4) * first_block_size_bytes;
            if (image_len > data_len) {
                log->Error("Insufficient data length, bytes left %i, expected %i", data_len, image_len);
                break;
            }

            auto &reg = regions[regions_count++];

            reg.bufferOffset = VkDeviceSize(data_off[i] + offset);
            reg.bufferRowLength = 0;
            reg.bufferImageHeight = 0;

            reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            reg.imageSubresource.mipLevel = uint32_t(j);
            reg.imageSubresource.baseArrayLayer = i;
            reg.imageSubresource.layerCount = 1;

            reg.imageOffset = {0, 0, 0};
            reg.imageExtent = {uint32_t(width), uint32_t(height), 1};

            offset += image_len;
            data_len -= image_len;
        }
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions_count,
                           regions);

    ApplySampling(p.sampling, log);
}

void Ray::Vk::Texture2D::InitFromKTXFile(const void *data[6], const int size[6], Buffer &sbuf, void *_cmd_buf,
                                         MemoryAllocators *mem_allocs, const Tex2DParams &p, ILog *log) {
    Free();

    const auto *first_header = reinterpret_cast<const KTXHeader *>(data[0]);

    uint8_t *stage_data = sbuf.Map(BufMapWrite);
    uint32_t data_off[6] = {};
    uint32_t stage_len = 0;

    for (int i = 0; i < 6; ++i) {
        const auto *_data = (const uint8_t *)data[i];
        const auto *this_header = reinterpret_cast<const KTXHeader *>(_data);

        // make sure all images have same properties
        if (this_header->pixel_width != first_header->pixel_width) {
            log->Error("Image width mismatch %i, expected %i", int(this_header->pixel_width),
                       int(first_header->pixel_width));
            continue;
        }
        if (this_header->pixel_height != first_header->pixel_height) {
            log->Error("Image height mismatch %i, expected %i", int(this_header->pixel_height),
                       int(first_header->pixel_height));
            continue;
        }
        if (this_header->gl_internal_format != first_header->gl_internal_format) {
            log->Error("Internal format mismatch %i, expected %i", int(this_header->gl_internal_format),
                       int(first_header->gl_internal_format));
            continue;
        }

        memcpy(stage_data + stage_len, _data, size[i]);

        data_off[i] = stage_len;
        stage_len += size[i];
    }

    sbuf.FlushMappedRange(0, sbuf.AlignMapOffset(stage_len));
    sbuf.Unmap();

    handle_.generation = TextureHandleCounter++;
    params = p;
    params.cube = 1;
    initialized_mips_ = 0;

    bool is_srgb_format;
    params.format = FormatFromGLInternalFormat(first_header->gl_internal_format, &params.block, &is_srgb_format);

    if (is_srgb_format && !bool(params.flags & eTexFlagBits::SRGB)) {
        log->Warning("Loading SRGB texture as non-SRGB!");
    }

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = uint32_t(p.w);
        img_info.extent.height = uint32_t(p.h);
        img_info.extent.depth = 1;
        img_info.mipLevels = first_header->mipmap_levels_count;
        img_info.arrayLayers = 6;
        img_info.format = g_vk_formats[size_t(params.format)];
        if (bool(params.flags & eTexFlagBits::SRGB)) {
            img_info.format = ToSRGBFormat(img_info.format);
        }
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(uint8_t(p.usage) != 0);
        img_info.usage = to_vk_image_usage(p.usage, params.format);
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VkResult res = vkCreateImage(ctx_->device(), &img_info, nullptr, &handle_.img);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE;
        name_info.objectHandle = uint64_t(handle_.img);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

        VkMemoryRequirements tex_mem_req;
        vkGetImageMemoryRequirements(ctx_->device(), handle_.img, &tex_mem_req);

        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                      FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                     img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                      name_.c_str());
        if (!alloc_ && (img_tex_desired_mem_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                          FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                         img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                          name_.c_str());
            if (!alloc_) {
                img_tex_desired_mem_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                alloc_ = mem_allocs->Allocate(uint32_t(tex_mem_req.size), uint32_t(tex_mem_req.alignment),
                                              FindMemoryType(&ctx_->mem_properties(), tex_mem_req.memoryTypeBits,
                                                             img_tex_desired_mem_flags, uint32_t(tex_mem_req.size)),
                                              name_.c_str());
            }
        }

        const VkDeviceSize aligned_offset = AlignTo(VkDeviceSize(alloc_.alloc_off), tex_mem_req.alignment);

        res = vkBindImageMemory(ctx_->device(), handle_.img, alloc_.owner->mem(alloc_.block_ndx), aligned_offset);
        if (res != VK_SUCCESS) {
            log->Error("Failed to bind memory!");
            return;
        }
    }

    { // create default image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = handle_.img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format = g_vk_formats[size_t(p.format)];
        if (bool(p.flags & eTexFlagBits::SRGB)) {
            view_info.format = ToSRGBFormat(view_info.format);
        }
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = first_header->mipmap_levels_count;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 6;

        if (GetColorChannelCount(p.format) == 1) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = vkCreateImageView(ctx_->device(), &view_info, nullptr, &handle_.views[0]);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create image view!");
            return;
        }

#ifdef ENABLE_OBJ_LABELS
        VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        name_info.objectHandle = uint64_t(handle_.views[0]);
        name_info.pObjectName = name_.c_str();
        vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif
    }

    assert(p.samples == 1);
    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(sbuf.size());

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = first_header->mipmap_levels_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 6;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy regions[6 * 16] = {};
    int regions_count = 0;

    for (int i = 0; i < 6; ++i) {
        const auto *_data = (const uint8_t *)data[i];

#ifndef NDEBUG
        const auto *this_header = reinterpret_cast<const KTXHeader *>(data[i]);

        // make sure all images have same properties
        if (this_header->pixel_width != first_header->pixel_width) {
            log->Error("Image width mismatch %i, expected %i", int(this_header->pixel_width),
                       int(first_header->pixel_width));
            continue;
        }
        if (this_header->pixel_height != first_header->pixel_height) {
            log->Error("Image height mismatch %i, expected %i", int(this_header->pixel_height),
                       int(first_header->pixel_height));
            continue;
        }
        if (this_header->gl_internal_format != first_header->gl_internal_format) {
            log->Error("Internal format mismatch %i, expected %i", int(this_header->gl_internal_format),
                       int(first_header->gl_internal_format));
            continue;
        }
#endif
        int data_offset = sizeof(KTXHeader);
        int _w = params.w, _h = params.h;

        for (int j = 0; j < int(first_header->mipmap_levels_count); j++) {
            uint32_t img_size;
            memcpy(&img_size, &_data[data_offset], sizeof(uint32_t));
            data_offset += sizeof(uint32_t);

            auto &reg = regions[regions_count++];

            reg.bufferOffset = VkDeviceSize(data_off[i] + data_offset);
            reg.bufferRowLength = 0;
            reg.bufferImageHeight = 0;

            reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            reg.imageSubresource.mipLevel = uint32_t(j);
            reg.imageSubresource.baseArrayLayer = i;
            reg.imageSubresource.layerCount = 1;

            reg.imageOffset = {0, 0, 0};
            reg.imageExtent = {uint32_t(_w), uint32_t(_h), 1};

            data_offset += img_size;

            _w = std::max(_w / 2, 1);
            _h = std::max(_h / 2, 1);

            const int pad = (data_offset % 4) ? (4 - (data_offset % 4)) : 0;
            data_offset += pad;
        }
    }

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions_count,
                           regions);

    ApplySampling(p.sampling, log);
}

void Ray::Vk::Texture2D::SetSubImage(const int level, const int offsetx, const int offsety, const int sizex,
                                     const int sizey, const eTexFormat format, const Buffer &sbuf, void *_cmd_buf,
                                     const int data_off, const int data_len) {
    assert(format == params.format);
    assert(params.samples == 1);
    assert(offsetx >= 0 && offsetx + sizex <= std::max(params.w >> level, 1));
    assert(offsety >= 0 && offsety + sizey <= std::max(params.h >> level, 1));

    assert(sbuf.type() == eBufType::Stage);
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (sbuf.resource_state != eResState::Undefined && sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(sbuf.size());

        src_stages |= VKPipelineStagesForState(sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (this->resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(this->resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(this->resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = handle_.img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = params.mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(this->resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};

    region.bufferOffset = VkDeviceSize(data_off);
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = uint32_t(level);
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(offsetx), int32_t(offsety), 0};
    region.imageExtent = {uint32_t(sizex), uint32_t(sizey), 1};

    vkCmdCopyBufferToImage(cmd_buf, sbuf.vk_handle(), handle_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (offsetx == 0 && offsety == 0 && sizex == std::max(params.w >> level, 1) &&
        sizey == std::max(params.h >> level, 1)) {
        // consider this level initialized
        initialized_mips_ |= (1u << level);
    }
}

void Ray::Vk::Texture2D::SetSampling(const SamplingParams s) {
    if (handle_.sampler) {
        ctx_->samplers_to_destroy[ctx_->backend_frame].emplace_back(handle_.sampler);
    }

    VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = g_vk_min_mag_filter[size_t(s.filter)];
    sampler_info.minFilter = g_vk_min_mag_filter[size_t(s.filter)];
    sampler_info.addressModeU = g_vk_wrap_mode[size_t(s.wrap)];
    sampler_info.addressModeV = g_vk_wrap_mode[size_t(s.wrap)];
    sampler_info.addressModeW = g_vk_wrap_mode[size_t(s.wrap)];
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = AnisotropyLevel;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = s.compare != eTexCompare::None ? VK_TRUE : VK_FALSE;
    sampler_info.compareOp = g_vk_compare_ops[size_t(s.compare)];
    sampler_info.mipmapMode = g_vk_mipmap_mode[size_t(s.filter)];
    sampler_info.mipLodBias = s.lod_bias.to_float();
    sampler_info.minLod = s.min_lod.to_float();
    sampler_info.maxLod = s.max_lod.to_float();

    const VkResult res = vkCreateSampler(ctx_->device(), &sampler_info, nullptr, &handle_.sampler);
    if (res != VK_SUCCESS) {
        ctx_->log()->Error("Failed to create sampler!");
    }

    params.sampling = s;
}

void Ray::Vk::CopyImageToImage(void *_cmd_buf, Texture2D &src_tex, const uint32_t src_level, const uint32_t src_x,
                               const uint32_t src_y, Texture2D &dst_tex, const uint32_t dst_level, const uint32_t dst_x,
                               const uint32_t dst_y, const uint32_t width, const uint32_t height) {
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    assert(src_tex.resource_state == eResState::CopySrc);
    assert(dst_tex.resource_state == eResState::CopyDst);

    VkImageCopy reg;
    if (IsDepthFormat(src_tex.params.format)) {
        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    reg.srcSubresource.baseArrayLayer = 0;
    reg.srcSubresource.layerCount = 1;
    reg.srcSubresource.mipLevel = src_level;
    reg.srcOffset = {int32_t(src_x), int32_t(src_y), 0};
    if (IsDepthFormat(dst_tex.params.format)) {
        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    reg.dstSubresource.baseArrayLayer = 0;
    reg.dstSubresource.layerCount = 1;
    reg.dstSubresource.mipLevel = dst_level;
    reg.dstOffset = {int32_t(dst_x), int32_t(dst_y), 0};
    reg.extent = {width, height, 1};

    vkCmdCopyImage(cmd_buf, src_tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_tex.handle().img,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
}

void Ray::Vk::CopyImageToBuffer(const Texture2D &src_tex, const int level, const int x, const int y, const int w,
                                const int h, const Buffer &dst_buf, void *_cmd_buf, const int data_off) {
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (src_tex.resource_state != eResState::CopySrc) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(src_tex.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.oldLayout = VKImageLayoutForState(src_tex.resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = src_tex.handle().img;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = src_tex.params.mip_count; // transit whole image
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(src_tex.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    if (dst_buf.resource_state != eResState::Undefined && dst_buf.resource_state != eResState::CopyDst) {
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(dst_buf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = dst_buf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(dst_buf.size());

        src_stages |= VKPipelineStagesForState(dst_buf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    if (!buf_barriers.empty() || !img_barriers.empty()) {
        vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages, 0, 0,
                             nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                             uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    src_tex.resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};

    region.bufferOffset = VkDeviceSize(data_off);
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = uint32_t(level);
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(x), int32_t(y), 0};
    region.imageExtent = {uint32_t(w), uint32_t(h), 1};

    vkCmdCopyImageToBuffer(cmd_buf, src_tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_buf.vk_handle(), 1,
                           &region);
}

void Ray::Vk::ClearColorImage(Texture2D &tex, const float rgba[4], void *_cmd_buf) {
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);
    assert(tex.resource_state == eResState::CopyDst);

    VkClearColorValue clear_val = {};
    memcpy(clear_val.float32, rgba, 4 * sizeof(float));

    VkImageSubresourceRange clear_range = {};
    clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_range.layerCount = 1;
    clear_range.levelCount = 1;

    vkCmdClearColorImage(cmd_buf, tex.handle().img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_val, 1, &clear_range);
}

////////////////////////////////////////////////////////////////////////////////////////

Ray::Vk::Texture1D::Texture1D(const char *name, Buffer *buf, const eTexFormat format, const uint32_t offset,
                              const uint32_t size, ILog *log)
    : name_(name) {
    Init(std::move(buf), format, offset, size, log);
}

Ray::Vk::Texture1D::~Texture1D() { Free(); }

Ray::Vk::Texture1D &Ray::Vk::Texture1D::operator=(Texture1D &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Free();

    buf_ = std::move(rhs.buf_);
    params_ = exchange(rhs.params_, {});
    name_ = std::move(rhs.name_);
    buf_view_ = exchange(rhs.buf_view_, {});

    return (*this);
}

void Ray::Vk::Texture1D::Init(Buffer *buf, const eTexFormat format, const uint32_t offset, const uint32_t size,
                              ILog *log) {
    Free();

    VkBufferViewCreateInfo view_info = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
    view_info.buffer = buf->vk_handle();
    view_info.format = g_vk_formats[size_t(format)];
    view_info.offset = VkDeviceSize(offset);
    view_info.range = VkDeviceSize(size);

    const VkResult res = vkCreateBufferView(buf->ctx()->device(), &view_info, nullptr, &buf_view_);
    if (res != VK_SUCCESS) {
        buf_->ctx()->log()->Error("Failed to create buffer view!");
    }

    buf_ = std::move(buf);
    params_.offset = offset;
    params_.size = size;
    params_.format = format;
}

void Ray::Vk::Texture1D::Free() {
    if (buf_) {
        buf_->ctx()->buf_views_to_destroy[buf_->ctx()->backend_frame].push_back(buf_view_);
        buf_view_ = {};
        buf_ = {};
    }
}

VkFormat Ray::Vk::VKFormatFromTexFormat(eTexFormat format) { return g_vk_formats[size_t(format)]; }

#ifdef _MSC_VER
#pragma warning(pop)
#endif
