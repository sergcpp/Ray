#pragma once

#include <cstdint>
#include <cstring>

#include "../ImageParams.h"
#include "BufferVK.h"
#include "MemoryAllocatorVK.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace Ray {
class ILog;
namespace Vk {
eImgUsage ImgUsageFromState(eResState state);

class MemAllocators;

const int ImageDataPitchAlignment = 1;

struct ImgHandle {
    VkImage img = VK_NULL_HANDLE;
    SmallVector<VkImageView, 1> views;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t generation = 0; // used to identify unique texture (name can be reused)

    ImgHandle() { views.push_back(VK_NULL_HANDLE); }
    ImgHandle(VkImage _img, VkImageView _view0, VkImageView _view1, VkSampler _sampler, uint32_t _generation)
        : img(_img), sampler(_sampler), generation(_generation) {
        assert(_view0 != VK_NULL_HANDLE);
        views.push_back(_view0);
        views.push_back(_view1);
    }

    explicit operator bool() const { return img != VK_NULL_HANDLE; }
};
static_assert(sizeof(ImgHandle) == 48, "!");

inline bool operator==(const ImgHandle &lhs, const ImgHandle &rhs) {
    return lhs.img == rhs.img && lhs.views == rhs.views && lhs.sampler == rhs.sampler &&
           lhs.generation == rhs.generation;
}
inline bool operator!=(const ImgHandle &lhs, const ImgHandle &rhs) { return !operator==(lhs, rhs); }
inline bool operator<(const ImgHandle &lhs, const ImgHandle &rhs) {
    if (lhs.img < rhs.img) {
        return true;
    } else if (lhs.img == rhs.img) {
        if (lhs.views[0] < rhs.views[0]) { // we always compare only the first view
            return true;
        } else {
            return lhs.generation < rhs.generation;
        }
    }
    return false;
}

class Image {
    Context *ctx_ = nullptr;
    ImgHandle handle_;
    MemAllocation alloc_;
    std::string name_;

    void Free();

    void InitFromRAWData(Buffer *sbuf, int data_off, VkCommandBuffer cmd_buf, MemAllocators *mem_allocs,
                         const ImgParams &p, ILog *log);

  public:
    ImgParamsPacked params;

    mutable eResState resource_state = eResState::Undefined;

    Image() = default;
    Image(std::string_view name, Context *ctx, const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    Image(std::string_view name, Context *ctx, const VkImage img, const VkImageView view, const VkSampler sampler,
          const ImgParams &_params, ILog *log)
        : handle_{img, view, VK_NULL_HANDLE, sampler, 0}, name_(name), params(_params) {}
    Image(std::string_view name, Context *ctx, const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf,
          VkCommandBuffer cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);
    Image(const Image &rhs) = delete;
    Image(Image &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Image();

    Image &operator=(const Image &rhs) = delete;
    Image &operator=(Image &&rhs) noexcept;

    operator bool() const { return (handle_.img != VK_NULL_HANDLE); }

    void Init(const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    void Init(const VkImage img, const VkImageView view, const VkSampler sampler, const ImgParams &_params, ILog *log) {
        handle_ = {img, view, VK_NULL_HANDLE, sampler, 0};
        params = _params;
    }
    void Init(const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf, VkCommandBuffer cmd_buf,
              MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);
    void Init(const void *data[6], const int size[6], const ImgParams &p, Buffer &stage_buf, VkCommandBuffer cmd_buf,
              MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);

    bool Realloc(int w, int h, int mip_count, int samples, eFormat format, VkCommandBuffer cmd_buf,
                 MemAllocators *mem_allocs, ILog *log);

    Context *ctx() const { return ctx_; }
    const ImgHandle &handle() const { return handle_; }
    ImgHandle &handle() { return handle_; }
    VkSampler vk_sampler() const { return handle_.sampler; }

    VkDescriptorImageInfo
    vk_desc_image_info(const int view_index = 0,
                       const VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) const {
        VkDescriptorImageInfo ret;
        ret.sampler = handle_.sampler;
        ret.imageView = handle_.views[view_index];
        ret.imageLayout = layout;
        return ret;
    }

    const std::string &name() const { return name_; }

    void SetSampling(SamplingParams sampling);
    void ApplySampling(const SamplingParams sampling, ILog *log) { SetSampling(sampling); }

    void SetSubImage(int level, int offsetx, int offsety, int offsetz, int sizex, int sizey, int sizez, eFormat format,
                     const Buffer &sbuf, VkCommandBuffer cmd_buf, int data_off, int data_len);
};

void CopyImageToImage(VkCommandBuffer cmd_buf, Image &src_tex, uint32_t src_level, uint32_t src_x, uint32_t src_y,
                      Image &dst_tex, uint32_t dst_level, uint32_t dst_x, uint32_t dst_y, uint32_t width,
                      uint32_t height);

void CopyImageToBuffer(const Image &src_tex, int level, int x, int y, int w, int h, const Buffer &dst_buf,
                       VkCommandBuffer cmd_buf, int data_off);

void ClearColorImage(Image &tex, const float rgba[4], VkCommandBuffer cmd_buf);
void ClearColorImage(Image &tex, const uint32_t rgba[4], VkCommandBuffer cmd_buf);

VkFormat VKFormatFromFormat(eFormat format);

bool CanBeBlockCompressed(int w, int h, int mip_count);

} // namespace Vk
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif
