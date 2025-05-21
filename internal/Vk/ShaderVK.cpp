#include "ShaderVK.h"

#include <stdexcept>

#include "../../Config.h"
#include "../../Log.h"
#include "../ScopeExit.h"
#include "../TextureParams.h"
#include "ContextVK.h"
#include "SPIRV.h"

namespace Ray {
int round_up(int v, int align);
namespace Vk {
extern const VkShaderStageFlagBits g_shader_stages_vk[] = {
    VK_SHADER_STAGE_VERTEX_BIT,                  // Vert
    VK_SHADER_STAGE_FRAGMENT_BIT,                // Frag
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,    // Tesc
    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, // Tese
    VK_SHADER_STAGE_COMPUTE_BIT,                 // Comp
    VK_SHADER_STAGE_RAYGEN_BIT_KHR,              // RayGen
    VK_SHADER_STAGE_MISS_BIT_KHR,                // Miss
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,         // ClosestHit
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR,             // AnyHit
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR         // Intersection
};
static_assert(std::size(g_shader_stages_vk) == int(eShaderType::_Count), "!");

// TODO: not rely on this order somehow
static_assert(int(eShaderType::RayGen) < int(eShaderType::Miss), "!");
static_assert(int(eShaderType::Miss) < int(eShaderType::ClosestHit), "!");
static_assert(int(eShaderType::ClosestHit) < int(eShaderType::AnyHit), "!");
static_assert(int(eShaderType::AnyHit) < int(eShaderType::Intersection), "!");
} // namespace Vk
} // namespace Ray

Ray::Vk::Shader::Shader(const char *name, Context *ctx, Span<const uint8_t> shader_code, const eShaderType type,
                        ILog *log) {
    if (!Init(name, ctx, shader_code, type, log)) {
        throw std::runtime_error("Shader Init error!");
    }
}

Ray::Vk::Shader::~Shader() {
    if (module_) {
        ctx_->api().vkDestroyShaderModule(ctx_->device(), module_, nullptr);
    }
}

Ray::Vk::Shader &Ray::Vk::Shader::operator=(Shader &&rhs) noexcept {
    if (module_) {
        ctx_->api().vkDestroyShaderModule(ctx_->device(), module_, nullptr);
    }

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    module_ = std::exchange(rhs.module_, VkShaderModule(VK_NULL_HANDLE));
    type_ = rhs.type_;
    name_ = std::move(rhs.name_);

    attr_bindings = std::move(rhs.attr_bindings);
    unif_bindings = std::move(rhs.unif_bindings);
    pc_ranges = std::move(rhs.pc_ranges);

    return (*this);
}

bool Ray::Vk::Shader::Init(const char *name, Context *ctx, Span<const uint8_t> shader_code, const eShaderType type,
                           ILog *log) {
    name_ = name;
    ctx_ = ctx;

    if (shader_code.size() % sizeof(uint32_t) != 0) {
        return false;
    }
    if (!InitFromSPIRV(Span<const uint32_t>{reinterpret_cast<const uint32_t *>(shader_code.data()),
                                            shader_code.size() / sizeof(uint32_t)},
                       type, log)) {
        return false;
    }

#ifdef ENABLE_GPU_DEBUG
    VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
    name_info.objectHandle = uint64_t(module_);
    name_info.pObjectName = name_.c_str();
    ctx_->api().vkSetDebugUtilsObjectNameEXT(ctx_->device(), &name_info);
#endif

    return true;
}

bool Ray::Vk::Shader::InitFromSPIRV(Span<const uint32_t> shader_code, const eShaderType type, ILog *log) {
    if (shader_code.empty()) {
        return false;
    }

    type_ = type;

    { // init module
        VkShaderModuleCreateInfo create_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        create_info.codeSize = static_cast<size_t>(shader_code.size() * sizeof(uint32_t));
        create_info.pCode = reinterpret_cast<const uint32_t *>(shader_code.data());

        const VkResult res = ctx_->api().vkCreateShaderModule(ctx_->device(), &create_info, nullptr, &module_);
        if (res != VK_SUCCESS) {
            log->Error("Failed to create shader module!");
            return false;
        }
    }

    attr_bindings.clear();
    unif_bindings.clear();
    pc_ranges.clear();

    spirv_parser_state_t ps;
    ps.endianness = spirv_test_endianness(shader_code);
    if (ps.endianness == eEndianness::Invalid) {
        return false;
    }
    ps.header.magic = fix_endianness(shader_code[SPIRV_INDEX_MAGIC_NUMBER], ps.endianness);
    ps.header.version = fix_endianness(shader_code[SPIRV_INDEX_VERSION_NUMBER], ps.endianness);
    ps.header.generator = fix_endianness(shader_code[SPIRV_INDEX_GENERATOR_NUMBER], ps.endianness);
    ps.header.bound = fix_endianness(shader_code[SPIRV_INDEX_BOUND], ps.endianness);
    ps.header.schema = fix_endianness(shader_code[SPIRV_INDEX_SCHEMA], ps.endianness);
    ps.header.instructions = &shader_code[SPIRV_INDEX_INSTRUCTION];

    uint32_t offset = 0;
    while (offset < shader_code.size() - SPIRV_INDEX_INSTRUCTION) {
        const uint32_t instruction = fix_endianness(ps.header.instructions[offset], ps.endianness);

        const uint32_t length = instruction >> 16;
        const auto opcode = eSPIRVOp(instruction & 0x0ffffu);

        switch (opcode) {
        case eSPIRVOp::Name:
        case eSPIRVOp::TypeInt:
        case eSPIRVOp::TypeFloat:
        case eSPIRVOp::TypeVector:
        case eSPIRVOp::TypeStruct:
        case eSPIRVOp::TypeImage:
        case eSPIRVOp::TypeSampledImage:
        case eSPIRVOp::TypeRuntimeArray:
        case eSPIRVOp::TypePointer:
        case eSPIRVOp::TypeArray:
        case eSPIRVOp::TypeAccelerationStructureKHR: {
            const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
            ps.offsets[result_id] = offset;
        } break;
        case eSPIRVOp::Constant: {
            const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
            ps.offsets[result_id] = offset;
        } break;
        case eSPIRVOp::Decorate: {
            const uint32_t target = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
            const auto decoration = eSPIRVDecoration(fix_endianness(ps.header.instructions[offset + 2], ps.endianness));
            if (decoration == eSPIRVDecoration::DescriptorSet) {
                ps.decorations[target].descriptor_set =
                    fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
            } else if (decoration == eSPIRVDecoration::Binding) {
                ps.decorations[target].binding = fix_endianness(ps.header.instructions[offset + 3], ps.endianness);
            }
        } break;
        case eSPIRVOp::Variable: {
            const uint32_t result_type = fix_endianness(ps.header.instructions[offset + 1], ps.endianness);
            const uint32_t result_id = fix_endianness(ps.header.instructions[offset + 2], ps.endianness);
            const auto storage_class =
                eSPIRVStorageClass(fix_endianness(ps.header.instructions[offset + 3], ps.endianness));
            const char *debug_name = parse_debug_name(ps, result_id);
            if (storage_class == eSPIRVStorageClass::Input) {
            } else if (storage_class == eSPIRVStorageClass::PushConstant) {
                // TODO: Properly initialize offset
                const uint32_t _offset = 0;
                const uint32_t size = round_up(parse_type_size(ps, result_type), SPIRV_DATA_ALIGNMENT);
                pc_ranges.push_back({_offset, size});
            } else if (storage_class == eSPIRVStorageClass::Output) {
            } else if (storage_class == eSPIRVStorageClass::UniformConstant) {
                const spirv_uniform_props_t props = parse_uniform_props(ps, result_type);
                const spirv_decoration_t &decorations = ps.decorations[result_id];

                Descr &new_item = unif_bindings.emplace_back();
                if (debug_name) {
                    new_item.name = debug_name;
                }
                new_item.desc_type = props.descr_type;
                new_item.loc = decorations.binding;
                new_item.set = decorations.descriptor_set;
                new_item.count = props.runtime_array ? 0 : props.count;
            } else if (storage_class == eSPIRVStorageClass::StorageBuffer) {
                const spirv_buffer_props_t props = parse_buffer_props(ps, result_id);
                const spirv_decoration_t &decorations = ps.decorations[result_id];

                Descr &new_item = unif_bindings.emplace_back();
                if (debug_name) {
                    new_item.name = debug_name;
                }
                new_item.desc_type = VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                new_item.loc = decorations.binding;
                new_item.set = decorations.descriptor_set;
                new_item.count = props.runtime_array ? 0 : props.count;
            } else if (storage_class == eSPIRVStorageClass::Uniform) {
                const spirv_buffer_props_t props = parse_buffer_props(ps, result_id);
                const spirv_decoration_t &decorations = ps.decorations[result_id];

                Descr &new_item = unif_bindings.emplace_back();
                if (debug_name) {
                    new_item.name = debug_name;
                }
                new_item.desc_type = VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                new_item.loc = decorations.binding;
                new_item.set = decorations.descriptor_set;
                new_item.count = props.runtime_array ? 0 : props.count;
            }
        } break;
        }

        offset += length;
    }

    return true;
}
