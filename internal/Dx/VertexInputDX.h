#pragma once

#include "../../Span.h"
#include "BufferDX.h"

namespace Ray {
namespace Dx {
struct VtxAttribDesc {
    // BufHandle buf;
    uint8_t loc;
    uint8_t size;
    // eType type;
    uint8_t stride;
    uint32_t offset;

    // VtxAttribDesc(const BufHandle &_buf, int _loc, uint8_t _size, eType _type, int _stride, uint32_t _offset)
    //     : buf(_buf), loc(_loc), size(_size), type(_type), stride(_stride), offset(_offset) {}
    // VtxAttribDesc(const Buffer *_buf, int _loc, uint8_t _size, eType _type, int _stride, uint32_t _offset)
    //     : buf(_buf->handle()), loc(_loc), size(_size), type(_type), stride(_stride), offset(_offset) {}
};
inline bool operator==(const VtxAttribDesc &lhs, const VtxAttribDesc &rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(VtxAttribDesc)) == 0;
}

class VertexInput {
  public:
    SmallVector<VtxAttribDesc, 8> attribs;
    BufHandle elem_buf;

    VertexInput();
    VertexInput(const VertexInput &rhs) = delete;
    VertexInput(VertexInput &&rhs) noexcept { (*this) = std::move(rhs); }
    ~VertexInput();

    VertexInput &operator=(const VertexInput &rhs) = delete;
    VertexInput &operator=(VertexInput &&rhs) noexcept;

    // void BindBuffers(VkCommandBuffer cmd_buf, uint32_t index_offset, VkIndexType index_type) const;
    // void FillVKDescriptions(SmallVectorImpl<VkVertexInputBindingDescription> &out_bindings,
    //                         SmallVectorImpl<VkVertexInputAttributeDescription> &out_attribs) const;

    // bool Setup(Span<const VtxAttribDesc> attribs, const BufHandle &elem_buf);
    // bool Setup(Span<const VtxAttribDesc> _attribs, const Buffer *_elem_buf) {
    //     return Setup(_attribs, _elem_buf->handle());
    // }
};
} // namespace Dx
} // namespace Ray