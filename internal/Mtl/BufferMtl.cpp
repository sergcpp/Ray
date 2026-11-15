#include "BufferMtl.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "../../Config.h"
#include "../../Log.h"
#include "ContextMtl.h"

namespace Ray::Mtl {
static MTL::ResourceOptions MTLOptionsForType(const eBufType) {
    // Apple Silicon is unified memory: Shared storage keeps Map()/Unmap() zero-cost for all buffer classes
    return MTL::StorageModeShared;
}
} // namespace Ray::Mtl

int Ray::Mtl::Buffer::g_GenCounter = 0;

Ray::Mtl::Buffer::Buffer(std::string_view name, Context *ctx, const eBufType type, const uint32_t initial_size)
    : ctx_(ctx), name_(name), type_(type), size_(0) {
    Resize(initial_size);
}

Ray::Mtl::Buffer::~Buffer() { Free(); }

Ray::Mtl::Buffer &Ray::Mtl::Buffer::operator=(Buffer &&rhs) noexcept {
    Free();

    assert(!mapped_ptr_);
    assert(mapped_offset_ == 0xffffffff);

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    handle_ = std::exchange(rhs.handle_, {});
    name_ = std::move(rhs.name_);

    type_ = std::exchange(rhs.type_, eBufType::Undefined);

    size_ = std::exchange(rhs.size_, 0);
    mapped_ptr_ = std::exchange(rhs.mapped_ptr_, nullptr);
    mapped_offset_ = std::exchange(rhs.mapped_offset_, 0xffffffff);

    resource_state = std::exchange(rhs.resource_state, eResState::Undefined);

    return (*this);
}

void Ray::Mtl::Buffer::UpdateSubRegion(const uint32_t offset, const uint32_t size, const Buffer &init_buf,
                                       const uint32_t init_off, CommandBuffer cmd_buf) {
    assert(init_buf.type_ == eBufType::Upload || init_buf.type_ == eBufType::Readback);
    assert(cmd_buf);

    MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
    blit->copyFromBuffer(init_buf.mtl(), init_off, handle_.buf, offset, size);

    init_buf.resource_state = eResState::CopySrc;
    this->resource_state = eResState::CopyDst;
}

void Ray::Mtl::Buffer::Resize(const uint32_t new_size, const bool keep_content) {
    if (size_ >= new_size) {
        return;
    }

    const uint32_t old_size = size_;

    size_ = new_size;
    assert(size_ > 0);

    MTL::Buffer *new_buf = ctx_->device()->newBuffer(AlignMapOffsetUp(size_), MTLOptionsForType(type_));
    assert(new_buf != nullptr && "Failed to create buffer!");
#ifdef RAY_ENABLE_GPU_DEBUG
    new_buf->setLabel(MTLSTR(name_.c_str()));
#endif

    if (handle_.buf != nullptr) {
        if (keep_content) {
            CommandBuffer cmd_buf = ctx_->BeginSingleTimeCommands();
            MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
            blit->copyFromBuffer(handle_.buf, 0, new_buf, 0, old_size);
            ctx_->EndSingleTimeCommands(cmd_buf);
        }
        // Metal refcounts ObjC resources: command buffers retain what they bound, so the old
        // storage dies right here; VK-style frame deferral would just hoard GPU memory (sponza OOM)
        handle_.buf->release();
    }

    handle_.buf = new_buf;
    handle_.generation = g_GenCounter++;
}

void Ray::Mtl::Buffer::Free() {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    if (handle_.buf != nullptr) {
        handle_.buf->release();

        handle_ = {};
        size_ = 0;
    }
}

void Ray::Mtl::Buffer::FreeImmediate() {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    if (handle_.buf != nullptr) {
        handle_.buf->release(); // Metal keeps the storage alive while command buffers reference it
        handle_ = {};
        size_ = 0;
    }
}

uint32_t Ray::Mtl::Buffer::AlignMapOffset(const uint32_t offset) const {
    const auto align_to = uint32_t(256);
    return offset - (offset % align_to);
}

uint32_t Ray::Mtl::Buffer::AlignMapOffsetUp(const uint32_t offset) const {
    const auto align_to = uint32_t(256);
    return align_to * ((offset + align_to - 1) / align_to);
}

uint8_t *Ray::Mtl::Buffer::MapRange(const uint32_t offset, const uint32_t size, const bool) {
    assert(mapped_offset_ == 0xffffffff && !mapped_ptr_);
    assert(offset + size <= size_);
    assert(type_ == eBufType::Upload || type_ == eBufType::Readback);
    assert(offset == AlignMapOffset(offset));
    assert((offset + size) == size_ || (offset + size) == AlignMapOffset(offset + size));

    // Shared storage: contents() is always valid, mapping is a no-op bookkeeping shim
    uint8_t *mapped = static_cast<uint8_t *>(handle_.buf->contents()) + offset;

    mapped_ptr_ = mapped;
    mapped_offset_ = offset;
    return mapped;
}

void Ray::Mtl::Buffer::Unmap() {
    assert(mapped_offset_ != 0xffffffff && mapped_ptr_);
    mapped_ptr_ = nullptr;
    mapped_offset_ = 0xffffffff;
}

void Ray::Mtl::Buffer::Fill(const uint32_t dst_offset, const uint32_t size, const uint32_t data,
                            CommandBuffer cmd_buf) {
    assert(size % 4 == 0);

    const uint8_t b0 = data & 0xff, b1 = (data >> 8) & 0xff, b2 = (data >> 16) & 0xff, b3 = data >> 24;
    if (b0 == b1 && b1 == b2 && b2 == b3) {
        MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
        blit->fillBuffer(handle_.buf, NS::Range::Make(dst_offset, size), b0);
    } else {
        // non-byte patterns (debug NaN poison in RendererGPU::InitUNetFilter) land on just-created,
        // GPU-idle buffers only - CPU write is safe but must stay GPU-order-free
        for (uint32_t i = 0; i < size; i += 4) {
            memcpy(static_cast<uint8_t *>(handle_.buf->contents()) + dst_offset + i, &data, 4);
        }
    }

    resource_state = eResState::CopyDst;
}

void Ray::Mtl::Buffer::UpdateImmediate(const uint32_t dst_offset, const uint32_t size, const void *data,
                                       CommandBuffer cmd_buf) {
    // route through a temp buffer + blit so the write lands in command-buffer order (VK CmdUpdateBuffer parity)
    MTL::Buffer *tmp = ctx_->device()->newBuffer(data, size, MTLOptionsForType(type_));
    MTL::BlitCommandEncoder *blit = ctx_->BlitEncoder(cmd_buf);
    blit->copyFromBuffer(tmp, 0, handle_.buf, dst_offset, size);
    tmp->release();

    resource_state = eResState::CopyDst;
}

void Ray::Mtl::CopyBufferToBuffer(Buffer &src, const uint32_t src_offset, Buffer &dst, const uint32_t dst_offset,
                                  const uint32_t size, CommandBuffer cmd_buf) {
    MTL::BlitCommandEncoder *blit = src.ctx()->BlitEncoder(cmd_buf);
    blit->copyFromBuffer(src.mtl(), src_offset, dst.mtl(), dst_offset, size);

    src.resource_state = eResState::CopySrc;
    dst.resource_state = eResState::CopyDst;
}

bool Ray::Mtl::UpdateBuffer(Buffer &dst, const uint32_t dst_offset, const uint32_t data_size, const void *data,
                            Buffer &stage, const uint32_t map_offset, const uint32_t map_size, CommandBuffer cmd_buf) {
    const uint32_t aligned_offset = stage.AlignMapOffset(map_offset);
    const uint32_t aligned_size = stage.AlignMapOffsetUp(data_size);

    if (aligned_size > map_size) {
        return false;
    }

    memcpy(stage.MapRange(aligned_offset, aligned_size), data, data_size);
    stage.Unmap();

    CopyBufferToBuffer(stage, aligned_offset, dst, dst_offset, data_size, cmd_buf);
    return true;
}
