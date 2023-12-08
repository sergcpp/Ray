#include "SPIRV.h"

namespace Ray {
static const union {
    uint8_t bytes[4];
    uint32_t value;
} host_byte_order = {{0, 1, 2, 3}};

const uint32_t HOST_ENDIAN_LITTLE = 0x03020100u;
const uint32_t HOST_ENDIAN_BIG = 0x00010203u;

uint32_t swap_endianness(uint32_t v) {
    return (v & 0x000000ff) << 24 | (v & 0x0000ff00) << 8 | (v & 0x00ff0000) >> 8 | (v & 0xff000000) >> 24;
}

uint32_t fix_endianness(const uint32_t v, const eEndianness target) {
    if ((target == eEndianness::Little && host_byte_order.value == HOST_ENDIAN_BIG) ||
        (target == eEndianness::Big && host_byte_order.value == HOST_ENDIAN_LITTLE)) {
        return swap_endianness(v);
    }
    return v;
}
} // namespace Ray

Ray::eEndianness Ray::Vk::spirv_test_endianness(Span<const uint32_t> code) {
    if (code[0] == SPIRV_MAGIC) {
        return eEndianness::Little;
    }
    if (swap_endianness(code[0]) == SPIRV_MAGIC) {
        return eEndianness::Big;
    }

    return eEndianness::Invalid;
}

const char *Ray::Vk::parse_debug_name(spirv_parser_state_t &ps, const uint32_t id) {
    const uint32_t *p_offset = ps.offsets.Find(id);
    if (!p_offset) {
        return nullptr;
    }
    const uint32_t offset = *p_offset;

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    // const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    switch (opcode) {
    case eSPIRVOp::Name: {
        return reinterpret_cast<const char *>(&ps.header.instructions[offset + 2]);
    } break;
    default:
        break;
    }
    return nullptr;
}

Ray::Vk::eType Ray::Vk::parse_numeric_type(spirv_parser_state_t &ps, const uint32_t id) {
    const uint32_t offset = ps.offsets[id];

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    // const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    switch (opcode) {
    case eSPIRVOp::TypeInt: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        const uint32_t width = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        const uint32_t signess = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        eType type = eType::Undefined;
        if (width == 32) {
            if (signess) {
                type = eType::Int32;
            } else {
                type = eType::Uint32;
            }
        } else if (width == 16) {
            if (signess) {
                type = eType::Int16SNorm;
            } else {
                type = eType::Uint16;
            }
        }
        return type;
    } break;
    case eSPIRVOp::TypeFloat: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        const uint32_t width = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        eType type = eType::Undefined;
        if (width == 32) {
            type = eType::Float32;
        } else if (width == 16) {
            type = eType::Float16;
        }
        return type;
    } break;
    default:
        break;
    }

    return eType::Undefined;
}

Ray::Vk::spirv_constant_t Ray::Vk::parse_constant(spirv_parser_state_t &ps, uint32_t id) {
    const uint32_t *p_offset = ps.offsets.Find(id);
    if (!p_offset) {
        return {};
    }
    const uint32_t offset = *p_offset;

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    // const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    switch (opcode) {
    case eSPIRVOp::Constant: {
        const uint32_t value = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        return {value};
    } break;
    default:
        break;
    }

    return {};
}

uint32_t Ray::Vk::parse_type_size(spirv_parser_state_t &ps, const uint32_t id) {
    const uint32_t *p_offset = ps.offsets.Find(id);
    if (!p_offset) {
        return {};
    }
    const uint32_t offset = *p_offset;

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    uint32_t ret = 0;

    switch (opcode) {
    case eSPIRVOp::TypeInt:
    case eSPIRVOp::TypeFloat: {
        const uint32_t width = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        ret += width / 8;
    } break;
    case eSPIRVOp::TypeVector: {
        const uint32_t component_type = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        const uint32_t component_count = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret += parse_type_size(ps, component_type) * component_count;
    } break;
    case eSPIRVOp::TypePointer: {
        const uint32_t id = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret += parse_type_size(ps, id);
    } break;
    case eSPIRVOp::TypeStruct: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        for (uint32_t i = 2; i < length; ++i) {
            const uint32_t id = fix_endianness(ps.header.instructions[offset + i], ps.endianness);
            ret += parse_type_size(ps, id);
        }
    } break;
    default:
        break;
    }

    return ret;
}

Ray::Vk::spirv_buffer_props_t Ray::Vk::parse_buffer_props(spirv_parser_state_t &ps, const uint32_t id) {
    const uint32_t *p_offset = ps.offsets.Find(id);
    if (!p_offset) {
        return {};
    }
    const uint32_t offset = *p_offset;

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    spirv_buffer_props_t ret = {};

    switch (opcode) {
    case eSPIRVOp::TypePointer: {
        const uint32_t id = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret = parse_buffer_props(ps, id);
    } break;
    case eSPIRVOp::TypeArray: {
        const uint32_t len = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret.count = int(len);
    } break;
    case eSPIRVOp::TypeStruct: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        for (uint32_t i = 2; i < length; ++i) {
            const uint32_t id = fix_endianness(ps.header.instructions[offset + i], ps.endianness);
            ret.runtime_array |= parse_buffer_props(ps, id).runtime_array;
        }
    } break;
    default:
        break;
    }

    return ret;
}

Ray::Vk::spirv_uniform_props_t Ray::Vk::parse_uniform_props(spirv_parser_state_t &ps, const uint32_t id) {
    const uint32_t *p_offset = ps.offsets.Find(id);
    if (!p_offset) {
        return {};
    }
    const uint32_t offset = *p_offset;

    const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

    // const uint32_t length = instruction >> 16;
    const eSPIRVOp opcode = eSPIRVOp(instruction & 0x0ffffu);

    spirv_uniform_props_t ret = {};

    switch (opcode) {
    case eSPIRVOp::TypePointer: {
        const uint32_t id = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret = parse_uniform_props(ps, id);
    } break;
    case eSPIRVOp::TypeArray: {
        const uint32_t id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        const uint32_t len_id = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
        ret = parse_uniform_props(ps, id);
        ret.count = parse_constant(ps, len_id).u32;
    } break;
    case eSPIRVOp::TypeRuntimeArray: {
        const uint32_t id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        ret = parse_uniform_props(ps, id);
        ret.runtime_array = true;
    } break;
    case eSPIRVOp::TypeImage: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        const uint32_t type_id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        const uint32_t sampled = fix_endianness(ps.header.instructions[offset + 7], ps.endianness);
        ret.descr_type = sampled == 2 ? VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                      : VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        ret.dim = eSPIRVDim(fix_endianness(ps.header.instructions[offset + 3], ps.endianness));
        ret.format = eSPIRVImageFormat(fix_endianness(ps.header.instructions[offset + 7], ps.endianness));
        ret.type = parse_numeric_type(ps, type_id);
    } break;
    case eSPIRVOp::TypeSampledImage: {
        // const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
        const uint32_t type_id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
        ret = parse_uniform_props(ps, type_id);
        ret.descr_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    } break;
    case eSPIRVOp::TypeAccelerationStructureKHR: {
        ret.descr_type = VkDescriptorType::VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    } break;
    default:
        break;
    }

    return ret;
}