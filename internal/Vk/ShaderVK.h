#pragma once

#include <cstdint>

#include <string>

#include "../../Span.h"
#include "../SmallVector.h"
#include "Api.h"

namespace Ray {
class ILog;
namespace Vk {
class Context;

struct Range {
    uint32_t offset;
    uint32_t size;
};

struct Descr {
    std::string name;
    int loc = -1;
    VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    int set = 0, count = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
inline bool operator==(const Descr &lhs, const Descr &rhs) { return lhs.loc == rhs.loc && lhs.name == rhs.name; }
typedef Descr Attribute;
typedef Descr Uniform;
typedef Descr UniformBlock;

enum class eShaderType : uint8_t {
    Vert,
    Frag,
    Tesc,
    Tese,
    Comp,
    RayGen,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
    _Count
};

class Shader {
    Context *ctx_ = nullptr;
    VkShaderModule module_ = VK_NULL_HANDLE;
    eShaderType type_ = eShaderType::_Count;
    std::string name_;

    bool InitFromSPIRV(Span<const uint32_t> shader_code, eShaderType type, ILog *log);

  public:
    SmallVector<Descr, 16> attr_bindings, unif_bindings;
    SmallVector<Range, 4> pc_ranges;

    Shader() = default;
    Shader(const char *name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log);
    Shader(const Shader &rhs) = delete;
    Shader(Shader &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Shader();

    Shader &operator=(const Shader &rhs) = delete;
    Shader &operator=(Shader &&rhs) noexcept;

    bool ready() const { return module_ != VK_NULL_HANDLE; }
    VkShaderModule module() const { return module_; }
    eShaderType type() const { return type_; }
    const std::string &name() const { return name_; }

    bool Init(const char *name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log);
};

} // namespace Vk
} // namespace Ray