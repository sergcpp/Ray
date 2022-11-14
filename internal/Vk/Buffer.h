#pragma once

#include <vector>

#include "../SmallVector.h"
#include "Fence.h"
#include "LinearAlloc.h"
#include "Resource.h"

#include "VK.h"

namespace Ray {
class ILog;
namespace Vk {
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
    Stage,
    AccStructure,
    ShaderBinding,
    Indirect,
    _Count
};

const uint8_t BufMapRead = (1u << 0u);
const uint8_t BufMapWrite = (1u << 1u);

struct BufHandle {
    VkBuffer buf = VK_NULL_HANDLE;
    uint32_t generation = 0;

    operator bool() const { return buf != VK_NULL_HANDLE; }
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

class Buffer : public LinearAlloc {
    Context *ctx_ = nullptr;
    BufHandle handle_;
    std::string name_;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    eBufType type_ = eBufType::Undefined;
    uint32_t size_ = 0;
    uint8_t *mapped_ptr_ = nullptr;
    uint32_t mapped_offset_ = 0xffffffff;
    uint8_t mapped_dir_ = 0;
#ifndef NDEBUG
    SmallVector<RangeFence, 4> flushed_ranges_;
#endif

    static int g_GenCounter;

  public:
    Buffer() = default;
    explicit Buffer(const char *name, Context *ctx, eBufType type, uint32_t initial_size, uint32_t suballoc_align = 1);
    Buffer(const Buffer &rhs) = delete;
    Buffer(Buffer &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Buffer();

    Buffer &operator=(const Buffer &rhs) = delete;
    Buffer &operator=(Buffer &&rhs) noexcept;

    const std::string &name() const { return name_; }
    eBufType type() const { return type_; }
    // uint32_t size() const { return size_; }

    BufHandle handle() const { return handle_; }

    Context *ctx() const { return ctx_; }
    VkBuffer vk_handle() const { return handle_.buf; }
    VkDeviceMemory mem() const { return mem_; }
    VkDeviceAddress vk_device_address() const;

    uint32_t generation() const { return handle_.generation; }

    operator bool() const { return handle_.buf != VK_NULL_HANDLE; }

    bool is_mapped() const { return mapped_ptr_ != nullptr; }
    template <typename T = uint8_t> T *mapped_ptr() const { return reinterpret_cast<T *>(mapped_ptr_); }

    uint32_t AllocSubRegion(uint32_t size, const char *tag, const Buffer *init_buf = nullptr, void *cmd_buf = nullptr,
                            uint32_t init_off = 0);
    void UpdateSubRegion(uint32_t offset, uint32_t size, const Buffer &init_buf, uint32_t init_off = 0,
                         void *cmd_buf = nullptr);
    bool FreeSubRegion(uint32_t offset, uint32_t size);

    void Resize(uint32_t new_size, bool keep_content = true);
    void Free();
    void FreeImmediate();

    uint32_t AlignMapOffset(uint32_t offset) const;
    uint32_t AlignMapOffsetUp(uint32_t offset) const;

    uint8_t *Map(const uint8_t dir, const bool persistent = false) { return MapRange(dir, 0, size_, persistent); }
    uint8_t *MapRange(uint8_t dir, uint32_t offset, uint32_t size, bool persistent = false);
    void FlushMappedRange(uint32_t offset, uint32_t size, bool autoalign = false) const;
    void Unmap();

    void Fill(uint32_t dst_offset, uint32_t size, uint32_t data, void *_cmd_buf);
    void UpdateImmediate(uint32_t dst_offset, uint32_t size, const void *data, void *_cmd_buf);

    void Print(ILog *log);

    mutable eResState resource_state = eResState::Undefined;
};

void CopyBufferToBuffer(Buffer &src, uint32_t src_offset, Buffer &dst, uint32_t dst_offset, uint32_t size,
                        void *_cmd_buf);
// Update buffer using stage buffer
bool UpdateBuffer(Buffer &dst, uint32_t dst_offset, uint32_t data_size, const void *data, Buffer &stage,
                  uint32_t map_offset, uint32_t map_size, void *_cmd_buf);

} // namespace Vk
} // namespace Ray