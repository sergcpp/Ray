#pragma once

#include <string>
#include <vector>

#include "../SmallVector.h"
#include "DescriptorPoolDX.h"
#include "FenceDX.h"
#include "ResourceDX.h"

struct ID3D12Resource;

namespace Ray {
class ILog;
namespace Dx {
class Context;

enum class eType : uint8_t {
    Undefined,
    Float16,
    Float32,
    Uint32,
    Uint16,
    Uint16UNorm,
    Int16SNorm,
    Uint8UNorm,
    Int32,
    _Count
};
enum class eBufType : uint8_t {
    Undefined,
    VertexAttribs,
    VertexIndices,
    Texture,
    Uniform,
    Storage,
    Upload,
    Readback,
    AccStructure,
    ShaderBinding,
    Indirect,
    _Count
};

struct BufHandle {
    ID3D12Resource *buf = nullptr;
    PoolRef cbv_srv_uav_ref;
    uint32_t generation = 0;

    explicit operator bool() const { return buf != nullptr; }
};
inline bool operator==(const BufHandle lhs, const BufHandle rhs) {
    return lhs.buf == rhs.buf && lhs.generation == rhs.generation;
}

struct RangeFence {
    std::pair<uint32_t, uint32_t> range;
    SyncFence fence;

    RangeFence(const std::pair<uint32_t, uint32_t> _range, SyncFence &&_fence)
        : range(_range), fence(std::move(_fence)) {}
};

class Buffer {
    Context *ctx_ = nullptr;
    BufHandle handle_;
    std::string name_;
    eBufType type_ = eBufType::Undefined;
    uint32_t size_ = 0;
    uint8_t *mapped_ptr_ = nullptr;
    uint32_t mapped_offset_ = 0xffffffff;
    uint32_t mapped_size_ = 0;
#ifndef NDEBUG
    SmallVector<RangeFence, 4> flushed_ranges_;
#endif

    static int g_GenCounter;

  public:
    Buffer() = default;
    explicit Buffer(std::string_view name, Context *ctx, eBufType type, uint32_t initial_size);
    Buffer(const Buffer &rhs) = delete;
    Buffer(Buffer &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Buffer();

    Buffer &operator=(const Buffer &rhs) = delete;
    Buffer &operator=(Buffer &&rhs) noexcept;

    const std::string &name() const { return name_; }
    eBufType type() const { return type_; }
    uint32_t size() const { return size_; }

    BufHandle handle() const { return handle_; }

    Context *ctx() const { return ctx_; }
    ID3D12Resource *dx_resource() const { return handle_.buf; }

    uint32_t generation() const { return handle_.generation; }

    explicit operator bool() const { return handle_.buf != nullptr; }

    bool is_mapped() const { return mapped_ptr_ != nullptr; }
    template <typename T = uint8_t> T *mapped_ptr() const { return reinterpret_cast<T *>(mapped_ptr_); }

    void UpdateSubRegion(uint32_t offset, uint32_t size, const Buffer &init_buf, uint32_t init_off = 0,
                         ID3D12GraphicsCommandList *cmd_buf = nullptr);

    void Resize(uint32_t new_size, bool keep_content = true);
    void Free();
    void FreeImmediate();

    uint32_t AlignMapOffset(const uint32_t offset) const { return offset - (offset % 256); }
    uint32_t AlignMapOffsetUp(const uint32_t offset) const { return 256 * ((offset + 256 - 1) / 256); }

    uint8_t *Map(const bool persistent = false) { return MapRange(0, size_, persistent); }
    uint8_t *MapRange(uint32_t offset, uint32_t size, bool persistent = false);
    void FlushMappedRange(uint32_t offset, uint32_t size, bool autoalign = false) const;
    void Unmap();

    void Fill(uint32_t dst_offset, uint32_t size, uint32_t data, ID3D12GraphicsCommandList *cmd_buf);
    void UpdateImmediate(uint32_t dst_offset, uint32_t size, const void *data, ID3D12GraphicsCommandList *cmd_buf);

    mutable eResState resource_state = eResState::Undefined;
};

void CopyBufferToBuffer(Buffer &src, uint32_t src_offset, Buffer &dst, uint32_t dst_offset, uint32_t size,
                        ID3D12GraphicsCommandList *cmd_buf);
// Update buffer using stage buffer
bool UpdateBuffer(Buffer &dst, uint32_t dst_offset, uint32_t data_size, const void *data, Buffer &stage,
                  uint32_t map_offset, uint32_t map_size, ID3D12GraphicsCommandList *cmd_buf);

} // namespace Dx
} // namespace Ray