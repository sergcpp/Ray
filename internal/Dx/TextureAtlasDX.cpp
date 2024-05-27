#include "TextureAtlasDX.h"

#include <cassert>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextDX.h"
#include "TextureDX.h"

namespace Ray {
int round_up(int v, int align);

namespace Dx {
template <typename T, int N> eTexFormat tex_format();

template <> eTexFormat tex_format<uint8_t, 4>() { return eTexFormat::RawRGBA8888; }
template <> eTexFormat tex_format<uint8_t, 3>() { return eTexFormat::RawRGB888; }
template <> eTexFormat tex_format<uint8_t, 2>() { return eTexFormat::RawRG88; }
template <> eTexFormat tex_format<uint8_t, 1>() { return eTexFormat::RawR8; }

extern const DXGI_FORMAT g_dx_formats[];

uint32_t D3D12CalcSubresource(uint32_t MipSlice, uint32_t ArraySlice, uint32_t PlaneSlice, uint32_t MipLevels,
                              uint32_t ArraySize);
} // namespace Dx
} // namespace Ray

#define _MIN(x, y) ((x) < (y) ? (x) : (y))

Ray::Dx::TextureAtlas::TextureAtlas(Context *ctx, const char *name, const eTexFormat format, const eTexFilter filter,
                                    const int resx, const int resy, const int pages_count)
    : ctx_(ctx), name_(name), format_(format), filter_(filter), res_{resx, resy} {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

Ray::Dx::TextureAtlas::~TextureAtlas() {
    ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, srv_ref_);
    ctx_->resources_to_destroy[ctx_->backend_frame].push_back(img_);
}

template <typename T, int N>
int Ray::Dx::TextureAtlas::Allocate(const color_t<T, N> *data, const int _res[2], int pos[2]) {
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

template int Ray::Dx::TextureAtlas::Allocate<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2], int pos[2]);
template int Ray::Dx::TextureAtlas::Allocate<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2], int pos[2]);
template int Ray::Dx::TextureAtlas::Allocate<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2], int pos[2]);
template int Ray::Dx::TextureAtlas::Allocate<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2], int pos[2]);

template <typename T, int N>
void Ray::Dx::TextureAtlas::AllocateMips(const color_t<T, N> *data, const int _res[2], const int mip_count,
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

template void Ray::Dx::TextureAtlas::AllocateMips<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Dx::TextureAtlas::AllocateMips<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Dx::TextureAtlas::AllocateMips<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);
template void Ray::Dx::TextureAtlas::AllocateMips<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2],
                                                              int mip_count, int page[16], int pos[16][2]);

int Ray::Dx::TextureAtlas::AllocateRaw(void *data, const int res[2], int pos[2]) {
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

int Ray::Dx::TextureAtlas::Allocate(const int _res[2], int pos[2]) {
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

bool Ray::Dx::TextureAtlas::Free(const int page, const int pos[2]) {
    if (page < 0 || page > int(splitters_.size())) {
        return false;
    }
    // TODO: fill with black in debug
    return splitters_[page].Free(pos);
}

bool Ray::Dx::TextureAtlas::Resize(const int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < int(splitters_.size()); i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    real_format_ = format_;

    ID3D12Resource *new_img = nullptr;
    PoolRef new_srv_ref;

    { // create image
        D3D12_RESOURCE_DESC image_desc = {};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = pages_count ? uint32_t(res_[0]) : 4;
        image_desc.Height = pages_count ? uint32_t(res_[1]) : 4;
        image_desc.DepthOrArraySize = std::max(pages_count, 1);
        image_desc.MipLevels = 1;
        image_desc.Format = g_dx_formats[int(real_format_)];
        image_desc.SampleDesc.Count = 1;
        image_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        image_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        if (format_ == eTexFormat::RawRGB888 && !ctx_->rgb8_unorm_is_supported()) {
            // Fallback to 4-component texture
            image_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            real_format_ = eTexFormat::RawRGBA8888;
        }

        D3D12_HEAP_PROPERTIES heap_properties = {};
        heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_properties.CreationNodeMask = 1;
        heap_properties.VisibleNodeMask = 1;

        HRESULT hr =
            ctx_->device()->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &image_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&new_img));
        if (hr == E_OUTOFMEMORY) {
            ctx_->log()->Warning("Not enough device memory, falling back to CPU RAM!");

            heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
            hr = ctx_->device()->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &image_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&new_img));
        }

        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create resource");
        }

#ifdef ENABLE_OBJ_LABELS
        std::wstring temp_str(name_.begin(), name_.end());
        new_img->SetName(temp_str.c_str());
#endif
    }

    const UINT CBV_SRV_UAV_INCR =
        ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    { // create default SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        if (GetColorChannelCount(real_format_) == 1) {
            srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 0, 0, 0);
        } else {
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
        srv_desc.Format = g_dx_formats[int(real_format_)];
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv_desc.Texture2DArray.FirstArraySlice = 0;
        srv_desc.Texture2DArray.ArraySize = std::max(pages_count, 1);
        srv_desc.Texture2DArray.MipLevels = 1;
        srv_desc.Texture2DArray.MostDetailedMip = 0;
        srv_desc.Texture2DArray.PlaneSlice = 0;
        srv_desc.Texture2DArray.ResourceMinLODClamp = 0.0f;

        new_srv_ref = ctx_->staging_descr_alloc()->Alloc(eDescrType::CBV_SRV_UAV, 1);

        D3D12_CPU_DESCRIPTOR_HANDLE dest_handle = new_srv_ref.heap->GetCPUDescriptorHandleForHeapStart();
        dest_handle.ptr += CBV_SRV_UAV_INCR * new_srv_ref.offset;
        ctx_->device()->CreateShaderResourceView(new_img, &srv_desc, dest_handle);
    }

    SamplingParams params;
    params.filter = filter_;

    Sampler new_sampler(ctx_, params);

    auto new_resource_state = eResState::Undefined;

    if (!splitters_.empty()) {
        SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;
        if (resource_state != eResState::CopySrc) {
            auto &new_barrier = barriers.emplace_back();
            new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            new_barrier.Transition.pResource = img_;
            new_barrier.Transition.StateBefore = DXResourceState(resource_state);
            new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
            new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        { // destination image
            auto &new_barrier = barriers.emplace_back();
            new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            new_barrier.Transition.pResource = new_img;
            new_barrier.Transition.StateBefore = DXResourceState(new_resource_state);
            new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
            new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());

        resource_state = eResState::CopySrc;
        new_resource_state = eResState::CopyDst;

        for (int i = 0; i < int(splitters_.size()); ++i) {
            D3D12_TEXTURE_COPY_LOCATION src_location = {};
            src_location.pResource = img_;
            src_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src_location.SubresourceIndex = D3D12CalcSubresource(0, i, 0, 1, uint32_t(splitters_.size()));

            D3D12_TEXTURE_COPY_LOCATION dst_location = {};
            dst_location.pResource = new_img;
            dst_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst_location.SubresourceIndex = D3D12CalcSubresource(0, i, 0, 1, pages_count);

            cmd_buf->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, nullptr);
        }

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        // destroy old image
        img_->Release();
        ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, srv_ref_);
    } else if (img_) {
        // destroy temp dummy texture
        img_->Release();
        ctx_->staging_descr_alloc()->Free(eDescrType::CBV_SRV_UAV, srv_ref_);
    }

    if (new_resource_state == eResState::Undefined) {
        SmallVector<D3D12_RESOURCE_BARRIER, 1> barriers;
        { // image
            auto &new_barrier = barriers.emplace_back();
            new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            new_barrier.Transition.pResource = new_img;
            new_barrier.Transition.StateBefore = DXResourceState(eResState::Undefined);
            new_barrier.Transition.StateAfter = DXResourceState(eResState::ShaderResource);
            new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }

        new_resource_state = eResState::ShaderResource;

        CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());

        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    img_ = new_img;
    srv_ref_ = new_srv_ref;

    sampler_.FreeImmediate();
    sampler_ = std::move(new_sampler);

    splitters_.resize(pages_count, TextureSplitter{res_});

    resource_state = new_resource_state;

    return true;
}

int Ray::Dx::TextureAtlas::DownsampleRegion(const int src_page, const int src_pos[2], const int src_res[2],
                                            int dst_pos[2]) {
    assert(false && "Not implemented!");
    return -1;
}

void Ray::Dx::TextureAtlas::WritePageData(const int page, const int posx, const int posy, const int sizex,
                                          const int sizey, const void *_data) {
    const uint8_t *data = reinterpret_cast<const uint8_t *>(_data);

    int pitch = 0, lines;
    if (!IsCompressedFormat(format_)) {
        pitch = sizex * GetPerPixelDataLen(real_format_);
        lines = sizey;
    } else {
        lines = (sizey + 3) / 4;
        if (format_ == eTexFormat::BC1) {
            pitch = GetRequiredMemory_BC1(sizex, 1, 1);
        } else if (format_ == eTexFormat::BC3) {
            pitch = GetRequiredMemory_BC3(sizex, 1, 1);
        } else if (format_ == eTexFormat::BC4) {
            pitch = GetRequiredMemory_BC4(sizex, 1, 1);
        } else if (format_ == eTexFormat::BC5) {
            pitch = GetRequiredMemory_BC5(sizex, 1, 1);
        }
    }
    const uint32_t data_size = round_up(pitch, TextureDataPitchAlignment) * lines;
    const bool rgb_as_rgba = (format_ == eTexFormat::RawRGB888 && real_format_ == eTexFormat::RawRGBA8888);

    Buffer temp_sbuf("Temp Stage", ctx_, eBufType::Upload, data_size);

    uint8_t *ptr = temp_sbuf.Map();
    if (rgb_as_rgba) {
        const auto *src = reinterpret_cast<const color_t<uint8_t, 3> *>(data);
        auto *dst = reinterpret_cast<color_t<uint8_t, 4> *>(ptr);
        for (int y = 0; y < sizey; ++y) {
            for (int x = 0; x < sizex; ++x) {
                dst[x].v[0] = src[x].v[0];
                dst[x].v[1] = src[x].v[1];
                dst[x].v[2] = src[x].v[2];
                dst[x].v[3] = 255;
            }
            src += sizex;
            dst += round_up(sizex, TextureDataPitchAlignment / sizeof(color_t<uint8_t, 4>));
        }
    } else {
        int i = 0;
        for (int y = 0; y < lines; ++y) {
            memcpy(ptr + i, &data[y * pitch], pitch);
            i += round_up(pitch, TextureDataPitchAlignment);
        }
    }
    temp_sbuf.Unmap();

    SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;
    if (temp_sbuf.resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = temp_sbuf.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(temp_sbuf.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    if (resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = img_;
        new_barrier.Transition.StateBefore = DXResourceState(resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    CommandBuffer cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }

    temp_sbuf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = temp_sbuf.dx_resource();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Offset = 0;
    src_loc.PlacedFootprint.Footprint.Width = sizex;
    src_loc.PlacedFootprint.Footprint.Height = sizey;
    src_loc.PlacedFootprint.Footprint.Depth = 1;
    src_loc.PlacedFootprint.Footprint.Format = g_dx_formats[int(real_format_)];
    src_loc.PlacedFootprint.Footprint.RowPitch = round_up(pitch, TextureDataPitchAlignment);

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = img_;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.SubresourceIndex = D3D12CalcSubresource(0, page, 0, 1, uint32_t(splitters_.size()));

    cmd_buf->CopyTextureRegion(&dst_loc, posx, posy, 0, &src_loc, nullptr);

    EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

    temp_sbuf.FreeImmediate();
}

void Ray::Dx::TextureAtlas::CopyRegionTo(const int page, const int x, const int y, const int w, const int h,
                                         const Buffer &dst_buf, ID3D12GraphicsCommandList *cmd_buf, const int data_off) const {
    SmallVector<D3D12_RESOURCE_BARRIER, 2> barriers;

    if (resource_state != eResState::CopySrc) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = img_;
        new_barrier.Transition.StateBefore = DXResourceState(resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopySrc);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (/*dst_buf.resource_state != eResState::Undefined &&*/ dst_buf.resource_state != eResState::CopyDst) {
        auto &new_barrier = barriers.emplace_back();
        new_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        new_barrier.Transition.pResource = dst_buf.dx_resource();
        new_barrier.Transition.StateBefore = DXResourceState(dst_buf.resource_state);
        new_barrier.Transition.StateAfter = DXResourceState(eResState::CopyDst);
        new_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }

    if (!barriers.empty()) {
        cmd_buf->ResourceBarrier(UINT(barriers.size()), barriers.data());
    }

    resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;

    D3D12_TEXTURE_COPY_LOCATION src_loc = {};
    src_loc.pResource = img_;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = D3D12CalcSubresource(0, page, 0, 1, uint32_t(splitters_.size()));

    D3D12_TEXTURE_COPY_LOCATION dst_loc = {};
    dst_loc.pResource = dst_buf.dx_resource();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Offset = data_off;
    dst_loc.PlacedFootprint.Footprint.Width = w;
    dst_loc.PlacedFootprint.Footprint.Height = h;
    dst_loc.PlacedFootprint.Footprint.Depth = 1;
    dst_loc.PlacedFootprint.Footprint.Format = g_dx_formats[int(real_format_)];
    if (IsCompressedFormat(real_format_)) {
        dst_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(GetBlockCount(w, 1, eTexBlock::_4x4) * GetBlockLenBytes(real_format_, eTexBlock::_4x4),
                     TextureDataPitchAlignment);
    } else {
        dst_loc.PlacedFootprint.Footprint.RowPitch =
            round_up(w * GetPerPixelDataLen(real_format_), TextureDataPitchAlignment);
    }

    D3D12_BOX src_region = {};
    src_region.left = x;
    src_region.right = x + w;
    src_region.top = y;
    src_region.bottom = y + h;
    src_region.front = 0;
    src_region.back = 1;

    cmd_buf->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_region);
}

#undef _MIN
