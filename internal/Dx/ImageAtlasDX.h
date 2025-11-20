#pragma once

#include "../CoreDX.h"
#include "../ImageSplitter.h"
#include "DescriptorPoolDX.h"
#include "ResourceDX.h"
#include "SamplerDX.h"

struct ID3D12Resource;

namespace Ray {
enum class eFormat : uint8_t;
namespace Dx {
class Context;
class ImageAtlas {
    Context *ctx_;
    std::string name_;
    eFormat format_, real_format_;
    eFilter filter_;
    const int res_[2];

    ID3D12Resource *img_ = nullptr;
    PoolRef srv_ref_;
    Sampler sampler_;

    std::vector<ImageSplitter> splitters_;

    void WritePageData(int page, int posx, int posy, int sizex, int sizey, const void *data);

  public:
    ImageAtlas(Context *ctx, std::string_view name, eFormat format, eFilter filter, int resx, int resy,
               int page_count = 0);
    ~ImageAtlas();

    eFormat format() const { return format_; }
    eFormat real_format() const { return real_format_; }
    ID3D12Resource *dx_resource() const { return img_; }
    PoolRef srv_ref() const { return srv_ref_; }
    PoolRef sampler_ref() const { return sampler_.ref(); }

    int res_x() const { return res_[0]; }
    int res_y() const { return res_[1]; }
    int page_count() const { return int(splitters_.size()); }

    template <typename T, int N> int Allocate(const color_t<T, N> *data, const int res[2], int pos[2]);
    template <typename T, int N>
    void AllocateMips(const color_t<T, N> *data, const int res[2], int mip_count, int page[16], int pos[16][2]);
    int AllocateRaw(void *data, const int res[2], int pos[2]);
    int Allocate(const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int pages_count);

    int DownsampleRegion(int src_page, const int src_pos[2], const int src_res[2], int dst_pos[2]);

    void CopyRegionTo(int page, int x, int y, int w, int h, const Buffer &dst_buf, ID3D12GraphicsCommandList *cmd_buf,
                      int data_off) const;

    mutable eResState resource_state = eResState::Undefined;
};
} // namespace Dx
} // namespace Ray