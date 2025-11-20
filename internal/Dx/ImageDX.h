#pragma once

#include <cstdint>
#include <cstring>

#include "../ImageParams.h"
#include "BufferDX.h"
#include "DescriptorPoolDX.h"
#include "MemoryAllocatorDX.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

enum DXGI_FORMAT;

namespace Ray {
class ILog;
namespace Dx {
eImgUsage ImgUsageFromState(eResState state);

class MemAllocators;

const int ImageDataPitchAlignment = 256;

struct ImgHandle {
    ID3D12Resource *img = nullptr;
    PoolRef views_ref, sampler_ref;
    uint32_t generation = 0; // used to identify unique texture (name can be reused)

    ImgHandle() {}
    ImgHandle(ID3D12Resource *_img,
              /*VkImageView _view0, VkImageView _view1, VkSampler _sampler,*/ uint32_t _generation)
        : // img(_img), sampler(_sampler),
          generation(_generation) {
        // assert(_view0 != VK_NULL_HANDLE);
        // views.push_back(_view0);
        // views.push_back(_view1);
    }

    explicit operator bool() const { return img != nullptr; }
};
// static_assert(sizeof(ImgHandle) == 56, "!");
inline bool operator==(const ImgHandle &lhs, const ImgHandle &rhs) {
    return lhs.img == rhs.img && // lhs.views == rhs.views && lhs.sampler == rhs.sampler &&
           lhs.generation == rhs.generation;
}
inline bool operator!=(const ImgHandle &lhs, const ImgHandle &rhs) { return !operator==(lhs, rhs); }
inline bool operator<(const ImgHandle &lhs, const ImgHandle &rhs) {
    if (lhs.img < rhs.img) {
        return true;
    } /*else if (lhs.img == rhs.img) {
        if (lhs.views[0] < rhs.views[0]) { // we always compare only the first view
            return true;
        } else {
            return lhs.generation < rhs.generation;
        }
    }*/
    return false;
}

class Image {
    Context *ctx_ = nullptr;
    ImgHandle handle_;
    MemAllocation alloc_;
    std::string name_;

    void Free();

    void InitFromRAWData(Buffer *sbuf, int data_off, ID3D12GraphicsCommandList *cmd_buf, MemAllocators *mem_allocs,
                         const ImgParams &p, ILog *log);

  public:
    ImgParamsPacked params;

    mutable eResState resource_state = eResState::Undefined;

    Image() = default;
    Image(std::string_view name, Context *ctx, const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    Image(std::string_view name, Context *ctx,
          ID3D12Resource *img, // const VkImageView view, const VkSampler sampler,
          const ImgParams &_params, ILog *log)
        : handle_{img, /*view, VK_NULL_HANDLE, sampler,*/ 0}, name_(name), params(_params) {}
    Image(std::string_view name, Context *ctx, const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf,
          ID3D12GraphicsCommandList *cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);
    Image(const Image &rhs) = delete;
    Image(Image &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Image();

    Image &operator=(const Image &rhs) = delete;
    Image &operator=(Image &&rhs) noexcept;

    operator bool() const { return (handle_.img != nullptr); }

    void Init(const ImgParams &params, MemAllocators *mem_allocs, ILog *log);
    void Init(ID3D12Resource *img, /*const VkImageView view, const VkSampler sampler,*/ const ImgParams &_params,
              ILog *log) {
        handle_ = {img, /*view, VK_NULL_HANDLE, sampler,*/ 0};
        params = _params;
    }
    void Init(const void *data, uint32_t size, const ImgParams &p, Buffer &stage_buf,
              ID3D12GraphicsCommandList *cmd_buf, MemAllocators *mem_allocs, eImgLoadStatus *load_status, ILog *log);

    bool Realloc(int w, int h, int mip_count, int samples, eFormat format, ID3D12GraphicsCommandList *cmd_buf,
                 MemAllocators *mem_allocs, ILog *log);

    Context *ctx() { return ctx_; }
    const ImgHandle &handle() const { return handle_; }
    ImgHandle &handle() { return handle_; }
    ID3D12Resource *dx_resource() const { return handle_.img; }
    PoolRef sampler_ref() const { return handle_.sampler_ref; }

    /*VkDescriptorImageInfo
    vk_desc_image_info(const int view_index = 0,
                       const VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) const {
        VkDescriptorImageInfo ret;
        ret.sampler = handle_.sampler;
        ret.imageView = handle_.views[view_index];
        ret.imageLayout = layout;
        return ret;
    }*/

    const std::string &name() const { return name_; }

    void SetSampling(SamplingParams sampling);
    void ApplySampling(const SamplingParams sampling, ILog *log) { SetSampling(sampling); }

    void SetSubImage(int level, int offsetx, int offsety, int offsetz, int sizex, int sizey, int sizez, eFormat format,
                     const Buffer &sbuf, ID3D12GraphicsCommandList *cmd_buf, int data_off, int data_len);
};

void CopyImageToImage(ID3D12GraphicsCommandList *cmd_buf, Image &src_tex, uint32_t src_level, uint32_t src_x,
                      uint32_t src_y, Image &dst_tex, uint32_t dst_level, uint32_t dst_x, uint32_t dst_y,
                      uint32_t width, uint32_t height);

void CopyImageToBuffer(const Image &src_tex, int level, int x, int y, int w, int h, const Buffer &dst_buf,
                       ID3D12GraphicsCommandList *cmd_buf, int data_off);

void _ClearColorImage(Image &tex, const void *rgba, ID3D12GraphicsCommandList *cmd_buf);
inline void ClearColorImage(Image &tex, const float rgba[4], ID3D12GraphicsCommandList *cmd_buf) {
    _ClearColorImage(tex, rgba, cmd_buf);
}
inline void ClearColorImage(Image &tex, const uint32_t rgba[4], ID3D12GraphicsCommandList *cmd_buf) {
    _ClearColorImage(tex, rgba, cmd_buf);
}

DXGI_FORMAT DXFormatFromTexFormat(eFormat format);

bool CanBeBlockCompressed(int w, int h, int mip_count);

} // namespace Dx
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif
