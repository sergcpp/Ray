#include "TextureAtlasVK.h"

#include <cassert>

#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextVK.h"
#include "TextureVK.h"

namespace Ray {
namespace Vk {
template <typename T, int N> eTexFormat tex_format();

template <> eTexFormat tex_format<uint8_t, 4>() { return eTexFormat::RawRGBA8888; }
template <> eTexFormat tex_format<uint8_t, 3>() { return eTexFormat::RawRGB888; }
template <> eTexFormat tex_format<uint8_t, 2>() { return eTexFormat::RawRG88; }
template <> eTexFormat tex_format<uint8_t, 1>() { return eTexFormat::RawR8; }

uint32_t FindMemoryType(uint32_t start_from, const VkPhysicalDeviceMemoryProperties *mem_properties,
                        uint32_t mem_type_bits, VkMemoryPropertyFlags desired_mem_flags, VkDeviceSize desired_size);

extern const VkFormat g_vk_formats[];
} // namespace Vk
} // namespace Ray

#define _MIN(x, y) ((x) < (y) ? (x) : (y))

Ray::Vk::TextureAtlas::TextureAtlas(Context *ctx, const char *name, const eTexFormat format, const eTexFilter filter,
                                    const int resx, const int resy, const int pages_count)
    : ctx_(ctx), name_(name), format_(format), filter_(filter), res_{resx, resy} {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

Ray::Vk::TextureAtlas::~TextureAtlas() {
    if (img_view_) {
        ctx_->image_views_to_destroy[ctx_->backend_frame].push_back(img_view_);
    }
    ctx_->images_to_destroy[ctx_->backend_frame].push_back(img_);
    ctx_->mem_to_free[ctx_->backend_frame].emplace_back(mem_);
}

template <typename T, int N>
int Ray::Vk::TextureAtlas::Allocate(const color_t<T, N> *data, const int _res[2], int pos[2]) {
    int res[2] = {_res[0], _res[1]};
    if (res[0] > res_[0] || res[1] > res_[1]) {
        return -1;
    }

    if (!IsCompressedFormat(format_)) {
        std::vector<color_t<T, N>> temp_storage;
        for (int page_index = 0; page_index < int(splitters_.size()); page_index++) {
            const int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
            if (index != -1) {
                if (data) {
                    WritePageData(page_index, pos[0], pos[1], res[0], res[1], &data[0]);
                }
                return page_index;
            }
        }
    } else {
        // round resolution up to block size
        res[0] = 4 * ((res[0] + 3) / 4);
        res[1] = 4 * ((res[1] + 3) / 4);

        // TODO: Get rid of allocation
        std::vector<color_t<T, N>> temp_storage;
        for (int page_index = 0; page_index < int(splitters_.size()); page_index++) {
            const int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
            if (index != -1) {
                if (data) {
                    temp_storage.resize(res[0] * res[1], {});
                    for (int y = 0; y < _res[1]; ++y) {
                        memcpy(&temp_storage[y * res[0]], &data[y * _res[0]], _res[0] * sizeof(color_t<T, N>));
                    }

                    std::unique_ptr<uint8_t[]> compressed_data;
                    if (format_ == eTexFormat::BC3) {
                        // TODO: get rid of allocation
                        auto temp_YCoCg = ConvertRGB_to_CoCgxY(&temp_storage[0].v[0], res[0], res[1]);

                        const int req_size = GetRequiredMemory_BC3(res[0], res[1], 1);
                        compressed_data = std::make_unique<uint8_t[]>(req_size);
                        CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), res[0], res[1], compressed_data.get());
                    } else if (format_ == eTexFormat::BC4) {
                        const int req_size = GetRequiredMemory_BC4(res[0], res[1], 1);
                        // NOTE: 1 byte is added due to BC4 compression write outside of memory block
                        compressed_data = std::make_unique<uint8_t[]>(req_size + 1);
                        CompressImage_BC4<N>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                    } else if (format_ == eTexFormat::BC5) {
                        const int req_size = GetRequiredMemory_BC5(res[0], res[1], 1);
                        // NOTE: 1 byte is added due to BC5 compression write outside of memory block
                        compressed_data = std::make_unique<uint8_t[]>(req_size + 1);
                        CompressImage_BC5<2>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                    }

                    WritePageData(page_index, pos[0], pos[1], res[0], res[1], compressed_data.get());
                }
                return page_index;
            }
        }
    }

    Resize(int(splitters_.size()) + 1);
    return Allocate(data, _res, pos);
}

template int Ray::Vk::TextureAtlas::Allocate<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2], int pos[2]);
template int Ray::Vk::TextureAtlas::Allocate<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2], int pos[2]);
template int Ray::Vk::TextureAtlas::Allocate<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2], int pos[2]);
template int Ray::Vk::TextureAtlas::Allocate<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2], int pos[2]);

template <typename T, int N>
void Ray::Vk::TextureAtlas::AllocateMips(const color_t<T, N> *data, const int _res[2], const int mip_count,
                                         int page[16], int pos[16][2]) {
    int src_res[2] = {_res[0], _res[1]};

    // TODO: try to get rid of these allocations
    std::vector<color_t<T, N>> _src_data, dst_data;
    int i = 0;
    for (; i < mip_count; ++i) {
        const int dst_res[2] = {(src_res[0] + 1) / 2, (src_res[1] + 1) / 2};
        if (dst_res[0] < 4 || dst_res[1] < 4) {
            break;
        }

        dst_data.clear();
        dst_data.reserve(dst_res[0] * dst_res[1]);

        const color_t<T, N> *src_data = (i == 0) ? data : _src_data.data();

        for (int y = 0; y < src_res[1]; y += 2) {
            for (int x = 0; x < src_res[0]; x += 2) {
                const color_t<T, N> c00 = src_data[(y + 0) * src_res[0] + (x + 0)];
                const color_t<T, N> c10 = src_data[(y + 0) * src_res[0] + _MIN(x + 1, src_res[0] - 1)];
                const color_t<T, N> c11 =
                    src_data[_MIN(y + 1, src_res[1] - 1) * src_res[0] + _MIN(x + 1, src_res[0] - 1)];
                const color_t<T, N> c01 = src_data[_MIN(y + 1, src_res[1] - 1) * src_res[0] + (x + 0)];

                color_t<T, N> res;
                for (int i = 0; i < N; ++i) {
                    res.v[i] = (c00.v[i] + c10.v[i] + c11.v[i] + c01.v[i]) / 4;
                }

                dst_data.push_back(res);
            }
        }

        assert(dst_data.size() == (dst_res[0] * dst_res[1]));

        page[i] = Allocate(dst_data.data(), dst_res, pos[i]);

        src_res[0] = dst_res[0];
        src_res[1] = dst_res[1];
        std::swap(_src_data, dst_data);
    }
    for (; i < mip_count; ++i) {
        pos[i][0] = pos[i - 1][0];
        pos[i][1] = pos[i - 1][1];
        page[i] = page[i - 1];
    }
}

template void Ray::Vk::TextureAtlas::AllocateMips<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Vk::TextureAtlas::AllocateMips<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Vk::TextureAtlas::AllocateMips<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Vk::TextureAtlas::AllocateMips<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);

int Ray::Vk::TextureAtlas::AllocateRaw(void *data, const int res[2], int pos[2]) {
    for (int page_index = 0; page_index < int(splitters_.size()); page_index++) {
        const int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
        if (index != -1) {
            if (data) {
                WritePageData(page_index, pos[0], pos[1], res[0], res[1], data);
            }
            return page_index;
        }
    }
    Resize(int(splitters_.size()) + 1);
    return AllocateRaw(data, res, pos);
}

int Ray::Vk::TextureAtlas::Allocate(const int _res[2], int pos[2]) {
    // add 1px border
    const int res[2] = {_res[0] + 2, _res[1] + 2};

    if (res[0] > res_[0] || res[1] > res_[1]) {
        return -1;
    }

    for (int page_index = 0; page_index < int(splitters_.size()); page_index++) {
        const int index = splitters_[page_index].Allocate(&res[0], &pos[0]);
        if (index != -1) {
            return page_index;
        }
    }

    Resize(int(splitters_.size()) + 1);
    return Allocate(_res, pos);
}

bool Ray::Vk::TextureAtlas::Free(const int page, const int pos[2]) {
    if (page < 0 || page > int(splitters_.size())) {
        return false;
    }
    // TODO: fill with black in debug
    return splitters_[page].Free(pos);
}

bool Ray::Vk::TextureAtlas::Resize(const int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < int(splitters_.size()); i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    real_format_ = format_;

    VkImage new_img = {};
    VkImageView new_img_view = {};
    VkDeviceMemory new_mem = {};

    { // create image
        VkImageCreateInfo img_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = pages_count ? uint32_t(res_[0]) : 1;
        img_info.extent.height = pages_count ? uint32_t(res_[1]) : 1;
        img_info.extent.depth = 1;
        img_info.mipLevels = 1;
        img_info.arrayLayers = uint32_t(std::max(pages_count, 1));
        img_info.format = g_vk_formats[size_t(real_format_)];
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.flags = 0;

        if (format_ == eTexFormat::RawRGB888 && !ctx_->rgb8_unorm_is_supported()) {
            // Fallback to 4-component texture
            img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            real_format_ = eTexFormat::RawRGBA8888;
        }

        VkResult res = ctx_->api().vkCreateImage(ctx_->device(), &img_info, nullptr, &new_img);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image!");
        }

        VkMemoryRequirements img_tex_mem_req = {};
        ctx_->api().vkGetImageMemoryRequirements(ctx_->device(), new_img, &img_tex_mem_req);

        VkMemoryAllocateInfo img_alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        img_alloc_info.allocationSize = img_tex_mem_req.size;

        uint32_t img_tex_type_bits = img_tex_mem_req.memoryTypeBits;
        VkMemoryPropertyFlags img_tex_desired_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        img_alloc_info.memoryTypeIndex = FindMemoryType(0, &ctx_->mem_properties(), img_tex_type_bits,
                                                        img_tex_desired_mem_flags, uint32_t(img_tex_mem_req.size));
        res = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        while (img_alloc_info.memoryTypeIndex != 0xffffffff) {
            res = ctx_->api().vkAllocateMemory(ctx_->device(), &img_alloc_info, nullptr, &new_mem);
            if (res == VK_SUCCESS) {
                break;
            }
            img_alloc_info.memoryTypeIndex =
                FindMemoryType(img_alloc_info.memoryTypeIndex + 1, &ctx_->mem_properties(), img_tex_type_bits,
                               img_tex_desired_mem_flags, uint32_t(img_tex_mem_req.size));
        }
        if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");
            img_tex_desired_mem_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            while (img_alloc_info.memoryTypeIndex != 0xffffffff) {
                res = ctx_->api().vkAllocateMemory(ctx_->device(), &img_alloc_info, nullptr, &new_mem);
                if (res == VK_SUCCESS) {
                    break;
                }
                img_alloc_info.memoryTypeIndex =
                    FindMemoryType(img_alloc_info.memoryTypeIndex + 1, &ctx_->mem_properties(), img_tex_type_bits,
                                   img_tex_desired_mem_flags, uint32_t(img_tex_mem_req.size));
            }
        }
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate memory!");
        }

        res = ctx_->api().vkBindImageMemory(ctx_->device(), new_img, new_mem, 0);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to bind memory!");
        }
    }

#ifdef ENABLE_OBJ_LABELS
    VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectType = VK_OBJECT_TYPE_IMAGE;
    name_info.objectHandle = uint64_t(new_img);
    name_info.pObjectName = name_.c_str();
    ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

    { // create default image view
        VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = new_img;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view_info.format = g_vk_formats[size_t(real_format_)];
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = std::max(pages_count, 1);

        if (real_format_ == eTexFormat::RawR8 || real_format_ == eTexFormat::BC4) {
            view_info.components.r = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g = VK_COMPONENT_SWIZZLE_R;
            view_info.components.b = VK_COMPONENT_SWIZZLE_R;
            view_info.components.a = VK_COMPONENT_SWIZZLE_R;
        }

        const VkResult res = ctx_->api().vkCreateImageView(ctx_->device(), &view_info, nullptr, &new_img_view);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view!");
        }
    }

    SamplingParams params;
    params.filter = filter_;

    Sampler new_sampler(ctx_, params);

    auto new_resource_state = eResState::Undefined;

    if (!splitters_.empty()) {
        VkPipelineStageFlags src_stages = 0, dst_stages = 0;

        SmallVector<VkImageMemoryBarrier, 2> img_barriers;
        if (resource_state != eResState::CopySrc) {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
            new_barrier.oldLayout = VKImageLayoutForState(resource_state);
            new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = img_;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

            src_stages |= VKPipelineStagesForState(resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
        }

        { // destination image
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(new_resource_state);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
            new_barrier.oldLayout = VKImageLayoutForState(new_resource_state);
            new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = new_img;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

            src_stages |= VKPipelineStagesForState(resource_state);
            dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
        }

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, 0, nullptr, uint32_t(img_barriers.size()),
                                         img_barriers.cdata());

        resource_state = eResState::CopySrc;
        new_resource_state = eResState::CopyDst;

        VkImageCopy reg;
        if (IsDepthFormat(format_)) {
            reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        reg.srcSubresource.baseArrayLayer = 0;
        reg.srcSubresource.layerCount = uint32_t(splitters_.size());
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffset = {0, 0, 0};
        if (IsDepthFormat(format_)) {
            reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        reg.dstSubresource.baseArrayLayer = 0;
        reg.dstSubresource.layerCount = uint32_t(splitters_.size());
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffset = {0, 0, 0};
        reg.extent = {uint32_t(res_[0]), uint32_t(res_[1]), 1u};

        ctx_->api().vkCmdCopyImage(cmd_buf, img_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, new_img,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        // destroy old image
        ctx_->api().vkDestroyImageView(ctx_->device(), img_view_, nullptr);
        ctx_->api().vkDestroyImage(ctx_->device(), img_, nullptr);
        ctx_->api().vkFreeMemory(ctx_->device(), mem_, nullptr);
    } else if (img_view_) {
        // destroy temp dummy texture
        ctx_->api().vkDestroyImageView(ctx_->device(), img_view_, nullptr);
        ctx_->api().vkDestroyImage(ctx_->device(), img_, nullptr);
        ctx_->api().vkFreeMemory(ctx_->device(), mem_, nullptr);
    }

    if (new_resource_state == eResState::Undefined) {
        SmallVector<VkImageMemoryBarrier, 1> img_barriers;
        { // image
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(eResState::Undefined);
            new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::ShaderResource);
            new_barrier.oldLayout = VKImageLayoutForState(eResState::Undefined);
            new_barrier.newLayout = VKImageLayoutForState(eResState::ShaderResource);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = new_img;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image
        }

        new_resource_state = eResState::ShaderResource;

        VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        ctx_->api().vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         VKPipelineStagesForState(eResState::ShaderResource), 0, 0, nullptr, 0, nullptr,
                                         uint32_t(img_barriers.size()), img_barriers.cdata());

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    img_ = new_img;
    img_view_ = new_img_view;
    mem_ = new_mem;

    sampler_.FreeImmediate();
    sampler_ = std::move(new_sampler);

    splitters_.resize(pages_count, TextureSplitter{res_});

    resource_state = new_resource_state;

    return true;
}

int Ray::Vk::TextureAtlas::DownsampleRegion(const int src_page, const int src_pos[2], const int src_res[2],
                                            int dst_pos[2]) {
    const int dst_res[2] = {src_res[0] / 2, src_res[1] / 2};
    const int dst_page = Allocate(dst_res, dst_pos);
    if (dst_page == -1) {
        return -1;
    }

    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    {
        VkPipelineStageFlags src_stages = 0, dst_stages = 0;

        SmallVector<VkImageMemoryBarrier, 1> img_barriers;
        {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
            new_barrier.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            new_barrier.oldLayout = VKImageLayoutForState(resource_state);
            new_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = img_;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

            src_stages |= VKPipelineStagesForState(resource_state);
            dst_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, 0, nullptr, uint32_t(img_barriers.size()),
                                         img_barriers.cdata());
    }

    {
        VkImageBlit main_reg;

        main_reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        main_reg.srcSubresource.baseArrayLayer = uint32_t(src_page);
        main_reg.srcSubresource.layerCount = 1;
        main_reg.srcSubresource.mipLevel = 0;
        main_reg.srcOffsets[0] = {src_pos[0], src_pos[1], 0};
        main_reg.srcOffsets[1] = {src_pos[0] + src_res[0], src_pos[1] + src_res[1], 1};

        main_reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        main_reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        main_reg.dstSubresource.layerCount = 1;
        main_reg.dstSubresource.mipLevel = 0;
        main_reg.dstOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + 1, 0};
        main_reg.dstOffsets[1] = {dst_pos[0] + 1 + dst_res[0], dst_pos[1] + 1 + dst_res[1], 1};

        ctx_->api().vkCmdBlitImage(cmd_buf, img_, VK_IMAGE_LAYOUT_GENERAL, img_, VK_IMAGE_LAYOUT_GENERAL, 1, &main_reg,
                                   VK_FILTER_LINEAR);
    }

    // TODO: try to do the same without barrier (e.g. downsample borders from original image)
    {
        VkPipelineStageFlags src_stages = 0, dst_stages = 0;

        SmallVector<VkImageMemoryBarrier, 1> img_barriers;
        {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            new_barrier.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            new_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            new_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = img_;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

            src_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, 0, nullptr, uint32_t(img_barriers.size()),
                                         img_barriers.cdata());
    }

    SmallVector<VkImageBlit, 8> regs;

    { // 1px top border
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + dst_res[1], 0};
        reg.srcOffsets[1] = {dst_pos[0] + 1 + dst_res[0], dst_pos[1] + dst_res[1] + 1, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0] + 1, dst_pos[1], 0};
        reg.dstOffsets[1] = {dst_pos[0] + 1 + dst_res[0], dst_pos[1] + 1, 1};
    }
    { // 1px bottom border
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + 1, 0};
        reg.srcOffsets[1] = {dst_pos[0] + 1 + dst_res[0], dst_pos[1] + 2, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + dst_res[1] + 1, 0};
        reg.dstOffsets[1] = {dst_pos[0] + 1 + dst_res[0], dst_pos[1] + dst_res[1] + 2, 1};
    }
    { // 1px left border
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + dst_res[0], dst_pos[1] + 1, 0};
        reg.srcOffsets[1] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1] + 1 + dst_res[1], 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0], dst_pos[1] + 1, 0};
        reg.dstOffsets[1] = {dst_pos[0] + 1, dst_pos[1] + 1 + dst_res[1], 1};
    }
    { // 1px right border
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + 1, 0};
        reg.srcOffsets[1] = {dst_pos[0] + 2, dst_pos[1] + 1 + dst_res[1], 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1] + 1, 0};
        reg.dstOffsets[1] = {dst_pos[0] + dst_res[0] + 2, dst_pos[1] + 1 + dst_res[1], 1};
    }
    { // 1px LT corner
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + dst_res[0], dst_pos[1] + dst_res[1], 0};
        reg.srcOffsets[1] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1] + dst_res[1] + 1, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0], dst_pos[1], 0};
        reg.dstOffsets[1] = {dst_pos[0] + 1, dst_pos[1] + 1, 1};
    }
    { // 1px RT corner
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + dst_res[1], 0};
        reg.srcOffsets[1] = {dst_pos[0] + 2, dst_pos[1] + dst_res[1] + 1, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1], 0};
        reg.dstOffsets[1] = {dst_pos[0] + dst_res[0] + 2, dst_pos[1] + 1, 1};
    }
    { // 1px LB corner
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + dst_res[0], dst_pos[1] + 1, 0};
        reg.srcOffsets[1] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1] + 2, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0], dst_pos[1] + dst_res[1] + 1, 0};
        reg.dstOffsets[1] = {dst_pos[0] + 1, dst_pos[1] + dst_res[1] + 2, 1};
    }
    { // 1px RB corner
        VkImageBlit &reg = regs.emplace_back();

        reg.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.srcSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.srcSubresource.layerCount = 1;
        reg.srcSubresource.mipLevel = 0;
        reg.srcOffsets[0] = {dst_pos[0] + 1, dst_pos[1] + 1, 0};
        reg.srcOffsets[1] = {dst_pos[0] + 2, dst_pos[1] + 2, 1};

        reg.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.dstSubresource.baseArrayLayer = uint32_t(dst_page);
        reg.dstSubresource.layerCount = 1;
        reg.dstSubresource.mipLevel = 0;
        reg.dstOffsets[0] = {dst_pos[0] + dst_res[0] + 1, dst_pos[1] + dst_res[1] + 1, 0};
        reg.dstOffsets[1] = {dst_pos[0] + dst_res[0] + 2, dst_pos[1] + dst_res[1] + 2, 1};
    }

    ctx_->api().vkCmdBlitImage(cmd_buf, img_, VK_IMAGE_LAYOUT_GENERAL, img_, VK_IMAGE_LAYOUT_GENERAL,
                               uint32_t(regs.size()), regs.cdata(), VK_FILTER_LINEAR);

    {
        VkPipelineStageFlags src_stages = 0, dst_stages = 0;

        SmallVector<VkImageMemoryBarrier, 1> img_barriers;
        {
            auto &new_barrier = img_barriers.emplace_back();
            new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            new_barrier.srcAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            new_barrier.dstAccessMask = VKAccessFlagsForState(resource_state);
            new_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            new_barrier.newLayout = VKImageLayoutForState(resource_state);
            new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            new_barrier.image = img_;
            new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            new_barrier.subresourceRange.baseMipLevel = 0;
            new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            new_barrier.subresourceRange.baseArrayLayer = 0;
            new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

            src_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stages |= VKPipelineStagesForState(resource_state);
        }

        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, 0, nullptr, uint32_t(img_barriers.size()),
                                         img_barriers.cdata());
    }

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    return dst_page;
}

void Ray::Vk::TextureAtlas::WritePageData(const int page, const int posx, const int posy, const int sizex,
                                          const int sizey, const void *data) {
    uint32_t data_size;
    if (!IsCompressedFormat(format_)) {
        data_size = sizex * sizey * GetPerPixelDataLen(format_);
    } else {
        if (format_ == eTexFormat::BC1) {
            data_size = GetRequiredMemory_BC1(sizex, sizey, 1);
        } else if (format_ == eTexFormat::BC3) {
            data_size = GetRequiredMemory_BC3(sizex, sizey, 1);
        } else if (format_ == eTexFormat::BC4) {
            data_size = GetRequiredMemory_BC4(sizex, sizey, 1);
        } else if (format_ == eTexFormat::BC5) {
            data_size = GetRequiredMemory_BC5(sizex, sizey, 1);
        }
    }

    const bool rgb_as_rgba = (format_ == eTexFormat::RawRGB888 && real_format_ == eTexFormat::RawRGBA8888);
    const uint32_t buf_size = rgb_as_rgba ? sizex * sizey * sizeof(color_t<uint8_t, 4>) : data_size;

    Buffer temp_sbuf("Temp Stage", ctx_, eBufType::Upload, buf_size);

    uint8_t *ptr = temp_sbuf.Map();
    if (rgb_as_rgba) {
        const auto *src = reinterpret_cast<const color_t<uint8_t, 3> *>(data);
        auto *dst = reinterpret_cast<color_t<uint8_t, 4> *>(ptr);
        for (int i = 0; i < sizex * sizey; ++i) {
            dst[i].v[0] = src[i].v[0];
            dst[i].v[1] = src[i].v[1];
            dst[i].v[2] = src[i].v[2];
            dst[i].v[3] = 255;
        }
    } else {
        memcpy(ptr, data, data_size);
    }
    temp_sbuf.Unmap();

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;

    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    { // transition stage buffer
        auto &new_barrier = buf_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(temp_sbuf.resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.buffer = temp_sbuf.vk_handle();
        new_barrier.offset = VkDeviceSize(0);
        new_barrier.size = VkDeviceSize(data_size);

        src_stages |= VKPipelineStagesForState(temp_sbuf.resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopySrc);
    }

    SmallVector<VkImageMemoryBarrier, 1> img_barriers;
    if (resource_state != eResState::CopyDst) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopyDst);
        new_barrier.oldLayout = VKImageLayoutForState(resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopyDst);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = img_;
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS; // transit whole image

        src_stages |= VKPipelineStagesForState(resource_state);
        dst_stages |= VKPipelineStagesForState(eResState::CopyDst);
    }

    VkCommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stages,
                                     0, 0, nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                                     uint32_t(img_barriers.size()), img_barriers.cdata());

    temp_sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = uint32_t(page);
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(posx), int32_t(posy), 0};
    region.imageExtent = {uint32_t(sizex), uint32_t(sizey), 1};

    ctx_->api().vkCmdCopyBufferToImage(cmd_buf, temp_sbuf.vk_handle(), img_, VKImageLayoutForState(eResState::CopyDst),
                                       1, &region);

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    temp_sbuf.FreeImmediate();
}

void Ray::Vk::TextureAtlas::CopyRegionTo(const int page, const int x, const int y, const int w, const int h,
                                         const Buffer &dst_buf, void *_cmd_buf, const int data_off) const {
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);

    VkPipelineStageFlags src_stages = 0, dst_stages = 0;
    SmallVector<VkBufferMemoryBarrier, 1> buf_barriers;
    SmallVector<VkImageMemoryBarrier, 1> img_barriers;

    if (resource_state != eResState::CopySrc) {
        auto &new_barrier = img_barriers.emplace_back();
        new_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        new_barrier.srcAccessMask = VKAccessFlagsForState(resource_state);
        new_barrier.dstAccessMask = VKAccessFlagsForState(eResState::CopySrc);
        new_barrier.oldLayout = VKImageLayoutForState(resource_state);
        new_barrier.newLayout = VKImageLayoutForState(eResState::CopySrc);
        new_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        new_barrier.image = vk_image();
        new_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        new_barrier.subresourceRange.baseMipLevel = 0;
        new_barrier.subresourceRange.levelCount = 1;
        new_barrier.subresourceRange.baseArrayLayer = 0;
        new_barrier.subresourceRange.layerCount = 1;

        src_stages |= VKPipelineStagesForState(resource_state);
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
        ctx_->api().vkCmdPipelineBarrier(cmd_buf, src_stages ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         dst_stages, 0, 0, nullptr, uint32_t(buf_barriers.size()), buf_barriers.cdata(),
                                         uint32_t(img_barriers.size()), img_barriers.cdata());
    }

    resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;

    VkBufferImageCopy region = {};

    region.bufferOffset = VkDeviceSize(data_off);
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = page;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {int32_t(x), int32_t(y), 0};
    region.imageExtent = {uint32_t(w), uint32_t(h), 1};

    ctx_->api().vkCmdCopyImageToBuffer(cmd_buf, vk_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_buf.vk_handle(),
                                       1, &region);
}

#undef _MIN
