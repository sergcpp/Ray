#include "ImageAtlasMtl.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../Config.h"
#include "../../Log.h"
#include "../TextureUtils.h"
#include "ContextMtl.h"

namespace Ray {
int round_up(int v, int align);

namespace Mtl {
template <typename T, int N> eFormat img_format();

template <> eFormat img_format<uint8_t, 4>() { return eFormat::RGBA8; }
template <> eFormat img_format<uint8_t, 3>() { return eFormat::RGB8; }
template <> eFormat img_format<uint8_t, 2>() { return eFormat::RG8; }
template <> eFormat img_format<uint8_t, 1>() { return eFormat::R8; }

extern const MTL::PixelFormat g_formats_mtl[];
} // namespace Mtl
} // namespace Ray

#define _MIN(x, y) ((x) < (y) ? (x) : (y))

Ray::Mtl::ImageAtlas::ImageAtlas(Context *ctx, std::string_view name, const eFormat format, const eFilter filter,
                                 const int resx, const int resy, const int pages_count)
    : ctx_(ctx), name_(name), format_(format), real_format_(format), filter_(filter), res_{resx, resy} {
    if (!Resize(pages_count)) {
        throw std::runtime_error("ImageAtlas cannot be resized!");
    }
}

Ray::Mtl::ImageAtlas::~ImageAtlas() {
    if (img_) {
        img_->release();
    }
}

template <typename T, int N>
int Ray::Mtl::ImageAtlas::Allocate(const color_t<T, N> *data, const int _res[2], int pos[2]) {
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
                    if (format_ == eFormat::BC3) {
                        // TODO: get rid of allocation
                        auto temp_YCoCg = ConvertRGB_to_CoCgxY(&temp_storage[0].v[0], res[0], res[1]);

                        const int req_size = GetRequiredMemory_BC3(res[0], res[1], 1);
                        compressed_data = std::make_unique<uint8_t[]>(req_size);
                        CompressImage_BC3<true /* Is_YCoCg */>(temp_YCoCg.get(), res[0], res[1], compressed_data.get());
                    } else if (format_ == eFormat::BC4) {
                        const int req_size = GetRequiredMemory_BC4(res[0], res[1], 1);
                        // NOTE: 1 byte is added due to BC4 compression write outside of memory block
                        compressed_data = std::make_unique<uint8_t[]>(req_size + 1);
                        CompressImage_BC4<N>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                    } else if (format_ == eFormat::BC5) {
                        const int req_size = GetRequiredMemory_BC5(res[0], res[1], 1);
                        // NOTE: 1 byte is added due to BC5 compression write outside of memory block
                        compressed_data = std::make_unique<uint8_t[]>(req_size + 1);
                        CompressImage_BC5<2>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                    } else if (format_ == eFormat::ASTC_4x4) {
                        const int req_size = GetRequiredMemory_ASTC4x4(res[0], res[1], 1);
                        compressed_data = std::make_unique<uint8_t[]>(req_size);
                        if (N == 1) {
                            CompressImage_ASTC4x4_R8<1>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                        } else if (N == 2) {
                            CompressImage_ASTC4x4_RG<2>(&temp_storage[0].v[0], res[0], res[1], compressed_data.get());
                        } else {
                            assert(N == 3); // RGB888 source, YCoCg pre-transform
                            auto temp_YCoCg = ConvertRGB_to_CoCgxY(&temp_storage[0].v[0], res[0], res[1]);
                            CompressImage_ASTC4x4_RGBA<4>(temp_YCoCg.get(), res[0], res[1], compressed_data.get(), 0,
                                                          true);
                        }
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

template int Ray::Mtl::ImageAtlas::Allocate<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2], int pos[2]);
template int Ray::Mtl::ImageAtlas::Allocate<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2], int pos[2]);
template int Ray::Mtl::ImageAtlas::Allocate<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2], int pos[2]);
template int Ray::Mtl::ImageAtlas::Allocate<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2], int pos[2]);

template <typename T, int N>
void Ray::Mtl::ImageAtlas::AllocateMips(const color_t<T, N> *data, const int _res[2], const int mip_count, int page[16],
                                        int pos[16][2]) {
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
                for (int j = 0; j < N; ++j) {
                    res.v[j] = (c00.v[j] + c10.v[j] + c11.v[j] + c01.v[j]) / 4;
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

template void Ray::Mtl::ImageAtlas::AllocateMips<uint8_t, 1>(const color_t<uint8_t, 1> *data, const int res[2],
                                                             int mip_count, int page[16], int pos[16][2]);
template void Ray::Mtl::ImageAtlas::AllocateMips<uint8_t, 2>(const color_t<uint8_t, 2> *data, const int res[2],
                                                             int mip_count, int page[16], int pos[16][2]);
template void Ray::Mtl::ImageAtlas::AllocateMips<uint8_t, 3>(const color_t<uint8_t, 3> *data, const int res[2],
                                                             int mip_count, int page[16], int pos[16][2]);
template void Ray::Mtl::ImageAtlas::AllocateMips<uint8_t, 4>(const color_t<uint8_t, 4> *data, const int res[2],
                                                             int mip_count, int page[16], int pos[16][2]);

int Ray::Mtl::ImageAtlas::AllocateRaw(void *data, const int res[2], int pos[2]) {
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

int Ray::Mtl::ImageAtlas::Allocate(const int _res[2], int pos[2]) {
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

bool Ray::Mtl::ImageAtlas::Free(const int page, const int pos[2]) {
    if (page < 0 || page > int(splitters_.size())) {
        return false;
    }
    // TODO: fill with black in debug
    return splitters_[page].Free(pos);
}

bool Ray::Mtl::ImageAtlas::Resize(const int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < int(splitters_.size()); i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    real_format_ = MtlRealFormat(format_, Bitmask<eImgUsage>(eImgUsage::Transfer) | eImgUsage::Sampled);

    MTL::Texture *new_img = nullptr;
    MemAllocation new_alloc;
    { // create the arrayed page texture (arrayLength > 1 requires TextureType2DArray)
        MTL::TextureDescriptor *desc = MTL::TextureDescriptor::alloc()->init();
        desc->setTextureType(MTL::TextureType2DArray);
        desc->setPixelFormat(g_formats_mtl[size_t(real_format_)]);
        desc->setWidth(NS::UInteger(pages_count ? res_[0] : 1));
        desc->setHeight(NS::UInteger(pages_count ? res_[1] : 1));
        desc->setDepth(1);
        desc->setArrayLength(NS::UInteger(std::max(pages_count, 1)));
        desc->setMipmapLevelCount(1);
        desc->setSampleCount(1);
        desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        desc->setStorageMode(MTL::StorageModeShared);

        NS::UInteger page_bytes;
        if (IsCompressedFormat(real_format_)) {
            const NS::UInteger bpr =
                NS::UInteger(GetBlockCount(res_[0], 1, real_format_)) * NS::UInteger(GetBlockLenBytes(real_format_));
            page_bytes = bpr * NS::UInteger(GetBlockCount(res_[1], 1, real_format_));
        } else {
            page_bytes = NS::UInteger(res_[0]) * NS::UInteger(res_[1]) * NS::UInteger(GetPerPixelDataLen(real_format_));
        }
        // conservative container size: page rounding per page + row-align slack + headroom
        const NS::UInteger bytes =
            (((page_bytes * NS::UInteger(std::max(pages_count, 1)) + 16383) / 16384) * 16384) + page_bytes / 8 + 65536;

        new_alloc = ctx_->default_mem_allocs()->Allocate(16384, uint32_t(bytes), MTL::StorageModeShared);
        if (new_alloc) {
            ZeroHeapRange(new_alloc, bytes);
            new_img = new_alloc.owner->heap(new_alloc.pool)->newTexture(desc, new_alloc.offset);
            if (!new_img) {
                new_alloc = MemAllocation{};
            }
        }
        if (!new_img) {
            new_img = ctx_->device()->newTexture(desc);
        }
        desc->release();
        if (!new_img) {
            ctx_->log()->Error("Failed to create Metal atlas texture (%d pages)!", pages_count);
            return false;
        }
    }

    // keep the existing page contents; compressed atlases go through the CPU because
    // Metal blits cannot move compressed data between textures
    if (!splitters_.empty() && pages_count > 0) {
        const int old_pages = int(splitters_.size());
        const int copy_pages = _MIN(old_pages, pages_count);
        if (IsCompressedFormat(real_format_)) {
            const int bpr = GetBlockCount(res_[0], 1, real_format_) * GetBlockLenBytes(real_format_);
            const int bpi = bpr * GetBlockCount(res_[1], 1, real_format_);
            std::vector<uint8_t> page_data(bpi);
            for (int i = 0; i < copy_pages; ++i) {
                img_->getBytes(page_data.data(), NS::UInteger(bpr), NS::UInteger(bpi),
                               MTL::Region::Make2D(0, 0, res_[0], res_[1]), 0, NS::UInteger(i));
                new_img->replaceRegion(MTL::Region::Make2D(0, 0, res_[0], res_[1]), 0, NS::UInteger(i),
                                       page_data.data(), NS::UInteger(bpr), NS::UInteger(bpi));
            }
        } else {
            CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
            MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
            for (int i = 0; i < copy_pages; ++i) {
                blit->copyFromTexture(img_, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(res_[0], res_[1], 1),
                                      new_img, 0, 0, MTL::Origin::Make(0, 0, NS::UInteger(i)));
            }
            ctx_->EndSingleTimeCommands(cmd_buf);
        }
    }

    if (img_) {
        img_->release();
    }
    img_ = new_img;
    allocation_ = std::move(new_alloc); // frees the old block after the (completed) page copies

    SamplingParams params;
    params.filter = filter_;

    sampler_.FreeImmediate();
    sampler_.Init(ctx_, params);

    splitters_.resize(pages_count, ImageSplitter{res_});

    resource_state = eResState::ShaderResource;

    return true;
}

int Ray::Mtl::ImageAtlas::DownsampleRegion(const int src_page, const int src_pos[2], const int src_res[2],
                                           int dst_pos[2]) {
    assert(false && "Not implemented!");
    return -1;
}

void Ray::Mtl::ImageAtlas::WritePageData(const int page, const int posx, const int posy, const int sizex,
                                         const int sizey, const void *_data) {
    const uint8_t *data = reinterpret_cast<const uint8_t *>(_data);

    int pitch = 0, lines;
    if (!IsCompressedFormat(format_)) {
        pitch = sizex * GetPerPixelDataLen(format_);
        lines = sizey;
    } else {
        lines = (sizey + 3) / 4;
        if (format_ == eFormat::BC1) {
            pitch = GetRequiredMemory_BC1(sizex, 1, 1);
        } else if (format_ == eFormat::BC3) {
            pitch = GetRequiredMemory_BC3(sizex, 1, 1);
        } else if (format_ == eFormat::BC4) {
            pitch = GetRequiredMemory_BC4(sizex, 1, 1);
        } else if (format_ == eFormat::BC5) {
            pitch = GetRequiredMemory_BC5(sizex, 1, 1);
        } else if (format_ == eFormat::ASTC_4x4) {
            pitch = GetRequiredMemory_ASTC4x4(sizex, 1, 1);
        }
    }
    const int src_row = round_up(pitch, ImageDataPitchAlignment);
    const int dst_row = IsCompressedFormat(real_format_)
                            ? GetBlockCount(sizex, 1, real_format_) * GetBlockLenBytes(real_format_)
                            : sizex * GetPerPixelDataLen(real_format_);

    Buffer temp_sbuf("Temp Stage", ctx_, eBufType::Upload, dst_row * lines);
    uint8_t *ptr = temp_sbuf.Map();
    if (real_format_ != format_) {
        assert(!IsCompressedFormat(format_)); // promoted formats are uncompressed by construction
        for (int y = 0; y < lines; ++y) {
            RepackToReal(data + size_t(y) * src_row, ptr + size_t(y) * dst_row, sizex, format_, real_format_);
        }
    } else {
        for (int y = 0; y < lines; ++y) {
            memcpy(ptr + size_t(y) * dst_row, data + size_t(y) * pitch, size_t(pitch));
        }
    }
    temp_sbuf.Unmap();

    int ox = posx, oy = posy;
    if (IsCompressedFormat(real_format_)) {
        ox &= ~3;
        oy &= ~3;
    }

    CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
    MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
    blit->copyFromBuffer(temp_sbuf.mtl(), 0, NS::UInteger(dst_row), NS::UInteger(dst_row * lines),
                         MTL::Size::Make(NS::UInteger(sizex), NS::UInteger(sizey), 1), img_, NS::UInteger(page), 0,
                         MTL::Origin::Make(NS::UInteger(ox), NS::UInteger(oy), 0));
    ctx_->EndSingleTimeCommands(cmd_buf);

    temp_sbuf.FreeImmediate();

    resource_state = eResState::CopyDst;
}

void Ray::Mtl::ImageAtlas::CopyRegionTo(const int page, const int x, const int y, const int w, const int h,
                                        const Buffer &dst_buf, CommandBuffer cmd_buf, const int data_off) const {
    assert(!IsCompressedFormat(real_format_)); // Metal cannot blit compressed textures out

    const int row = w * GetPerPixelDataLen(real_format_);

    MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
    blit->copyFromTexture(img_, 0, 0, MTL::Origin::Make(NS::UInteger(x), NS::UInteger(y), 0),
                          MTL::Size::Make(NS::UInteger(w), NS::UInteger(h), 1), dst_buf.mtl(), NS::UInteger(data_off),
                          NS::UInteger(row), NS::UInteger(row * h));

    resource_state = eResState::CopySrc;
    dst_buf.resource_state = eResState::CopyDst;
}

#undef _MIN
