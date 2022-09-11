#include "VertexInput.h"

namespace Ray {
namespace Vk {
const VkFormat g_vk_attrib_formats[][4] = {
    {}, // Undefined
    {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT,
     VK_FORMAT_R16G16B16A16_SFLOAT}, // Float16
    {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
     VK_FORMAT_R32G32B32A32_SFLOAT},                                                                    // Float32
    {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT}, // Uint32
    {VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT}, // Uint16
    {VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16B16_UNORM,
     VK_FORMAT_R16G16B16A16_UNORM}, // Uint16UNorm
    {VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16B16_SNORM,
     VK_FORMAT_R16G16B16A16_SNORM},                                                                     // Int16SNorm
    {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},       // Uint8UNorm
    {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT}, // Int32
};
static_assert(COUNT_OF(g_vk_attrib_formats) == int(eType::_Count), "!");

const int g_type_sizes[] = {
    -1,               // Undefined
    sizeof(uint16_t), // Float16
    sizeof(float),    // Float32
    sizeof(uint32_t), // Uint32
    sizeof(uint16_t), // Uint16
    sizeof(uint16_t), // Uint16UNorm
    sizeof(int16_t),  // Int16SNorm
    sizeof(uint8_t),  // Uint8UNorm
    sizeof(int32_t),  // Int32
};
static_assert(COUNT_OF(g_type_sizes) == int(eType::_Count), "!");

const int MaxVertexInputAttributeOffset = 16; // 16 seems to be supported by all implementations
} // namespace Vk
} // namespace Ray

Ray::Vk::VertexInput::VertexInput() = default;

Ray::Vk::VertexInput::~VertexInput() = default;

Ray::Vk::VertexInput &Ray::Vk::VertexInput::operator=(VertexInput &&rhs) noexcept = default;

void Ray::Vk::VertexInput::BindBuffers(VkCommandBuffer cmd_buf, const uint32_t index_offset,
                                       const VkIndexType index_type) const {
    SmallVector<VkBuffer, 8> buffers_to_bind;
    SmallVector<VkDeviceSize, 8> buffer_offsets;
    for (const auto &attr_descr : attribs) {
        int bound_index = -1;
        for (int i = 0; i < int(buffers_to_bind.size()); ++i) {
            if (buffers_to_bind[i] == attr_descr.buf.buf &&
                (attr_descr.offset <= MaxVertexInputAttributeOffset || buffer_offsets[i] == attr_descr.offset)) {
                bound_index = i;
                break;
            }
        }
        if (bound_index == -1) {
            buffers_to_bind.push_back(attr_descr.buf.buf);
            if (attr_descr.offset > MaxVertexInputAttributeOffset) {
                buffer_offsets.push_back(attr_descr.offset);
            } else {
                // attribute offset will be used instead
                buffer_offsets.push_back(0);
            }
        }
    }

    vkCmdBindVertexBuffers(cmd_buf, 0, uint32_t(buffers_to_bind.size()), buffers_to_bind.cdata(),
                           buffer_offsets.cdata());
    if (elem_buf) {
        vkCmdBindIndexBuffer(cmd_buf, elem_buf.buf, VkDeviceSize(index_offset), index_type);
    }
}

void Ray::Vk::VertexInput::FillVKDescriptions(SmallVectorImpl<VkVertexInputBindingDescription> &out_bindings,
                                              SmallVectorImpl<VkVertexInputAttributeDescription> &out_attribs) const {
    SmallVector<std::pair<BufHandle, VkDeviceSize>, 8> bound_buffers;
    for (const auto &attr_descr : attribs) {
        auto &vk_attr = out_attribs.emplace_back();

        vk_attr.location = uint32_t(attr_descr.loc);
        vk_attr.format = g_vk_attrib_formats[int(attr_descr.type)][attr_descr.size - 1];
        if (attr_descr.offset > MaxVertexInputAttributeOffset) {
            // binding offset will be used instead
            vk_attr.offset = 0;
        } else {
            vk_attr.offset = attr_descr.offset;
        }
        vk_attr.binding = 0xffffffff;

        for (uint32_t i = 0; i < uint32_t(bound_buffers.size()); ++i) {
            if (bound_buffers[i].first == attr_descr.buf &&
                (attr_descr.offset <= MaxVertexInputAttributeOffset || bound_buffers[i].second == attr_descr.offset)) {
                vk_attr.binding = i;
                break;
            }
        }

        if (vk_attr.binding == 0xffffffff) {
            vk_attr.binding = uint32_t(bound_buffers.size());

            auto &vk_binding = out_bindings.emplace_back();

            vk_binding.binding = uint32_t(bound_buffers.size());
            if (attr_descr.stride) {
                vk_binding.stride = uint32_t(attr_descr.stride);
            } else {
                vk_binding.stride = uint32_t(g_type_sizes[int(attr_descr.type)]) * attr_descr.size;
            }
            vk_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            bound_buffers.emplace_back(attr_descr.buf, attr_descr.offset);
        }
    }
}

bool Ray::Vk::VertexInput::Setup(Span<const VtxAttribDesc> _attribs, const BufHandle &_elem_buf) {
    if (_attribs.size() == attribs.size() && std::equal(_attribs.begin(), _attribs.end(), attribs.data()) &&
        elem_buf == _elem_buf) {
        return true;
    }

    attribs.assign(_attribs.begin(), _attribs.end());
    elem_buf = _elem_buf;

    return true;
}